// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ckernel_ops.h"
#include "constants.h"
#include "fw.h"
#include <bit>
#include <stdint.h>

#include "kernels_sfpu.inc"

static constexpr uint32_t squares_replay = sfpu_attention_accumulate_body_instruction_count;
static constexpr uint32_t scale_weight_replay = squares_replay + sfpu_squares_body_instruction_count;
static_assert(scale_weight_replay + sfpu_scale_weight_body_instruction_count <= 32u);

#define NOC_COUNTER_CLEAR 0x060u
#define NOC_COUNTER_BASE 0x200u
#define NOC_MST_REQS_OUTSTANDING_ID0 16u
#define NOC_CTRL_RD_NONPOSTED 0x10u
#define NOC_CTRL_WR_NONPOSTED 0x12u
#define NOC_CTRL_AT_POSTED 0x01u
#define NOC_AT_INCR_GET 0x107Cu // NOC_AT_WRAP(31) | NOC_AT_INS_INCR_GET

#define NOC_READ_CMD_BUF 0u
#define NOC_WRITE_CMD_BUF 1u
#define NOC_ATOMIC_CMD_BUF 2u

static_assert(TT_ROPE_COS_L1_ADDR % 64u == 0 && TT_ROPE_SIN_L1_ADDR % 64u == 0,
              "unpacker RoPE table bases must be 64-byte aligned");
// Private residual additions can pack a full trailing group; no peer writes these buffers.
static_assert(TT_MOE_OUT_L1_ADDR + (TT_GPT_OSS_DIM + 63u) * sizeof(float) <= TT_MOE_RESIDUAL_L1_ADDR);
static_assert(TT_MOE_RESIDUAL_L1_ADDR + (TT_GPT_OSS_DIM + 63u) * sizeof(float) <= TT_MOE_PROJECTED_L1_ADDR);
static_assert(TT_MOE_PROJECTED_L1_ADDR + (TT_GPT_OSS_DIM + 63u) * sizeof(float) <= TT_RMS_WEIGHT_L1_ADDR);
static_assert(TT_GPT_OSS_DIM % 64u == 0u && TT_GPT_OSS_QUERY_WIDTH % 64u == 0u);
static_assert(TT_SWIGLU_GATE_L1_ADDR + TT_SWIGLU_EXPERT_BYTES <= TT_SWIGLU_UP_L1_ADDR,
              "SWIGLU gate buffer overlaps up buffer");
static_assert(TT_SWIGLU_UP_L1_ADDR + TT_SWIGLU_EXPERT_BYTES <= TT_SWIGLU_OUT_L1_ADDR,
              "SWIGLU up buffer overlaps output buffer");
static_assert(TT_SWIGLU_OUT_L1_ADDR + TT_SWIGLU_OUTPUT_BYTES <= TT_BFP8_STAGE_L1_ADDR,
              "SWIGLU output buffer overlaps matvec stage");
// SrcB is contiguous BF16; O has the widest input vector.
static_assert(TT_MVMUL_B_L1_ADDR + TT_GPT_OSS_QUERY_WIDTH * sizeof(uint16_t) <= TT_MOE_HIDDEN_L1_ADDR,
              "matvec B operand overlaps gathered SwiGLU outputs");
static_assert(TT_MOE_HIDDEN_L1_ADDR + 4u * TT_SWIGLU_OUTPUT_BYTES <= TT_BF16_MVMUL_OUT_L1_ADDR,
              "gathered SwiGLU outputs overlap the matvec output buffer");
static_assert(TT_BFP8_STAGE_L1_ADDR + 3u * TT_BFP8_STRIP_BYTES <= TT_ATTENTION_QUERY_F32_L1_ADDR,
              "matvec stage overlaps attention scratch");
static_assert(TT_BF16_STAGE_L1_ADDR + 3u * TT_GPT_OSS_QUERY_WIDTH * 32u <= TT_ATTENTION_QUERY_F32_L1_ADDR,
              "BF16 matvec stage overlaps attention scratch");
static_assert(TT_ATTENTION_QUERY_F32_L1_ADDR + TT_GPT_OSS_QUERY_WIDTH * sizeof(float) <= TT_ATTENTION_QUERY_L1_ADDR,
              "FP32 query overlaps BF16 query");
static_assert(TT_MOE_DOWN4_L1_ADDR + 4u * TT_MOE_DOWN4_EXPERT_BYTES <= TT_ATTENTION_SINK_L1_ADDR,
              "MOE down4 buffer overlaps attention scratch");

static void cfg(uint32_t reg, uint32_t value) {
    PHYS_WR32(TENSIX_CFG_BASE + reg * sizeof(uint32_t), value);
}

static inline void pack_config(bool read_32b, bool fp32_out, bool l1_accumulate) {
    cfg(18, read_32b ? 1u : 0u);
    cfg(70, fp32_out ? 0x00000001u : 0x00000551u);
    cfg(71, l1_accumulate ? 0x00080000u : 0u);
}

static inline void pack_source_y_stride(uint32_t bytes) {
    // Address units follow the intermediate format, independently of 32-bit Dst reads. X stride stays zero.
    // Kernels leave source Y/carry zero; only packs using nonzero Y need to set this stride.
    cfg(12u, bytes << 16);
}

static inline void math_pack_config(DataFormat src_a, DataFormat src_b, DataFormat pack_intermediate, bool fpu_fp32) {
    // SFPU loads/stores specify their format explicitly; leave the default-format bit clear.
    cfg(1, (uint32_t(src_a) << 17) | (uint32_t(src_b) << 21) | (uint32_t(pack_intermediate) << 25) |
               (uint32_t(fpu_fp32) << 29));
}

static inline void unpack0_config(DataFormat format, uint32_t x_dim, uint32_t y_dim, uint32_t z_dim, bool transpose,
                                  bool unpack_to_dst) {
    cfg(64, (x_dim << 16) | 0x10u | uint32_t(format)); // uncompressed
    cfg(65, (z_dim << 16) | y_dim);
    cfg(72, uint32_t(format) | (uint32_t(transpose) << 8) | (uint32_t(unpack_to_dst) << 11));
    cfg(73, unpack_to_dst ? 0x11u : 0x03u);
    TTI_SETC16(5u, unpack_to_dst ? 0u : 4u); // SRCA_SET_SetOvrdWithAddr
}

static inline void unpack1_descriptor(DataFormat format, uint32_t x_dim) {
    cfg(112, (x_dim << 16) | 0x10u | uint32_t(format)); // uncompressed
}

static inline void unpack0_destination(uint32_t tile_x_dim) {
    cfg(86, (tile_x_dim << 16) | tile_x_dim);
}

struct DramRef {
    uint32_t coord;
    uint32_t addr;
};

static void noc_wr(uint32_t offset, uint32_t value) {
    PHYS_WR32(NOC0_BASE + offset, value);
}

static uint32_t noc_rd(uint32_t offset) {
    return PHYS_RD32(NOC0_BASE + offset);
}

static uint32_t noc_cmd_offset(uint32_t cmd_buf, uint32_t reg) {
    return cmd_buf * NOC_CMD_BUF_OFFSET + reg;
}

static void waypoint(uint32_t value) {
    PHYS_WR32(TT_L1_WAYPOINT_ADDR, value);
}

// Charge elapsed cycles on each tile's BRISC timeline, without draining asynchronous work.
static uint32_t wall_clock() {
    return PHYS_RD32(0xFFB121F0u); // WALL_CLOCK_0, AICLK cycles
}

static void profile_begin() {
    if (!(PHYS_RD32(TT_L1_TOKEN_FLAGS_ADDR) & TT_TOKEN_FLAG_PROFILE)) {
        return;
    }
    for (uint32_t stage = 0; stage < TT_PROFILE_STAGES; ++stage) {
        PHYS_WR32(TT_L1_PROFILE_STAGE_ADDR + stage * sizeof(uint32_t), 0u);
    }
    PHYS_WR32(TT_L1_PROFILE_MARK_ADDR, wall_clock());
}

static void profile_stage(uint32_t stage) {
    if (!(PHYS_RD32(TT_L1_TOKEN_FLAGS_ADDR) & TT_TOKEN_FLAG_PROFILE)) {
        return;
    }
    const uint32_t now = wall_clock();
    const uint32_t slot = TT_L1_PROFILE_STAGE_ADDR + stage * sizeof(uint32_t);
    PHYS_WR32(slot, PHYS_RD32(slot) + (now - PHYS_RD32(TT_L1_PROFILE_MARK_ADDR)));
    PHYS_WR32(TT_L1_PROFILE_MARK_ADDR, now);
}

static void set_dma_reg(uint32_t value) {
    tensix_instruction(TT_OP_SETDMAREG(0, value & 0xFFFF, 0, 0));
    tensix_instruction(TT_OP_SETDMAREG(0, value >> 16, 0, 1));
}

static void wait_tensix(uint32_t wait_mask) {
    // GPR63 is reserved for completion; address construction uses GPR0.
    volatile uint32_t *done = (volatile uint32_t *)(TENSIX_REGFILE_BASE + 63u * sizeof(uint32_t));
    *done = 0u;
    // Consume the readback before issuing SETDMAREG: FENCE alone does not drain a BRISC store.
    while (*done) {
    }
    TTI_STALLWAIT(STALL_THCON, wait_mask);
    TTI_SETDMAREG(0u, 1u, 0u, 126u);
    while (!*done) {
    }
}

static void noc_wait_cmd_accepted(uint32_t cmd_buf) {
    while (noc_rd(noc_cmd_offset(cmd_buf, NOC_CMD_CTRL))) {
    }
}

static void noc_wait_outstanding_id0() {
    while (noc_rd(NOC_COUNTER_BASE + NOC_MST_REQS_OUTSTANDING_ID0 * sizeof(uint32_t)) & 0xFFu) {
    }
}

static void noc_wait_outstanding_id(uint32_t id) {
    while (noc_rd(NOC_COUNTER_BASE + (NOC_MST_REQS_OUTSTANDING_ID0 + id) * sizeof(uint32_t)) & 0xFFu) {
    }
}

static float bf16_to_f32(uint16_t value) {
    return std::bit_cast<float>(uint32_t(value) << 16);
}

static void noc_read_to_l1_async_id(uint32_t id, uint32_t src_coord, uint32_t src_addr, uint32_t dst_addr,
                                    uint32_t size) {
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_TARG_ADDR_LO), src_addr);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_TARG_ADDR_HI), src_coord);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_RET_ADDR_LO), dst_addr);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_PACKET_TAG), id << 10);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_AT_LEN_BE), size);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_CMD_CTRL), 1u);
    noc_wait_cmd_accepted(NOC_READ_CMD_BUF);
}

static void noc_read_to_l1_async(uint32_t src_coord, uint32_t src_addr, uint32_t dst_addr, uint32_t size) {
    noc_read_to_l1_async_id(0u, src_coord, src_addr, dst_addr, size);
}

// Completion only; scalar consumers must flush the noncoherent D$ before reading L1.
static void noc_read_to_l1(uint32_t src_coord, uint32_t src_addr, uint32_t dst_addr, uint32_t size) {
    noc_read_to_l1_async(src_coord, src_addr, dst_addr, size);
    noc_wait_outstanding_id0();
}

template<bool Queued = false>
static void noc_write_from_l1_async(uint32_t dst_coord, uint32_t dst_addr, uint32_t src_addr, uint32_t size) {
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_TARG_ADDR_LO), src_addr);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_LO), dst_addr);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_HI), dst_coord);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_AT_LEN_BE), size);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_CMD_CTRL), 1u);
    if constexpr (!Queued) {
        noc_wait_cmd_accepted(NOC_WRITE_CMD_BUF);
    }
}

static void noc_write_from_l1(uint32_t dst_coord, uint32_t dst_addr, uint32_t src_addr, uint32_t size) {
    noc_write_from_l1_async(dst_coord, dst_addr, src_addr, size);
    noc_wait_outstanding_id0();
}

// Direct BH DMA addresses work for both pinned host memory and peer BAR mappings.
// Restore MID after command acceptance; subsequent local writes assume it is zero.
// Callers may issue a batch, then wait before reusing its source buffers or signalling completion.
static void noc_write_from_l1_wide_async(uint32_t dst_coord, uint64_t dst_addr, uint32_t src_addr, uint32_t size) {
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_MID), uint32_t(dst_addr >> 32));
    noc_write_from_l1_async(dst_coord, uint32_t(dst_addr), src_addr, size);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_MID), 0u);
}

static int32_t f32_to_i32_rtne(float value) {
    int32_t result;
    __asm__ volatile("fcvt.w.s %0, %1, dyn" : "=r"(result) : "f"(value));
    return result;
}

static inline void set_base_cfg(L1Base base, uint32_t l1_addr) {
    cfg(uint32_t(base), l1_addr / 16u - 1u);
}

static inline void set_base_cfg_pipelined(L1Base base, uint32_t l1_addr) {
    set_dma_reg(l1_addr / 16u - 1u);
    tensix_instruction(TT_OP_WRCFG(0, 0, uint32_t(base)));
}

static void set_pack_output_base_pipelined(uint32_t output_addr) {
    // Finish the previous pack before changing its configuration or reusing Dst.
    // Blocking ThCon holds the following SETDMAREG and all later instructions.
    TTI_STALLWAIT(STALL_THCON, WAIT_PACK);
    set_base_cfg_pipelined(L1Base::Pack, output_addr);
}

static void pack_output_4_rows() {
    // Callers order the producer, Dst/config reuse, and completion before BRISC/NoC reads.
    TTI_PACR(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u); // last=1, all 4 read interfaces
}

// Pack 1..4 consecutive rows within one four-row group, then reset source Y.
static inline void pack_output_rows(uint32_t rows) {
    const uint32_t mask = (1u << rows) - 1u;
    tensix_instruction(TT_OP_PACR(0u, 0u, 0u, 2u, 0u, 0u, mask, 0u, 0u, 0u, 0u, 1u));
}

template<uint32_t reg> static inline void load_bits32_to_lreg(uint32_t value) {
    static_assert(reg < 8u);
    tensix_instruction(TT_OP_SFPLOADI(reg, 8u, value >> 16));
    tensix_instruction(pack_u16(value, TT_OP_SFPLOADI(reg, 10u, 0u) >> 16));
}

template<uint32_t elements> static inline void unpack_bf16_to_srca_bounds() {
    static_assert(elements == 64u || elements == 128u || elements == 256u || elements == 512u);
    TTI_SETADCXX(1u, elements - 1u, 0u);
}

static inline void unpack_bf16_to_srca_body() {
    TTI_SETADC(1u, 1u, 1u, 0u); // context base supplies SrcA row 0; no added destination offset
    TTI_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u);
}

static inline void unpack_bf16_to_srca(uint32_t l1_addr) {
    set_base_cfg(L1Base::Unpack0, l1_addr);
    unpack_bf16_to_srca_body();
}

static inline void unpack_bf16_to_srca_block(uint32_t block) {
    static_assert(TT_SWIGLU_TILE_BYTES == 16u * 16u * sizeof(uint16_t));
    tensix_instruction(TT_OP_SETADC(1u, 0u, 2u, block));
    TTI_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u);
}

static void swiglu_setup() {
    math_pack_config(DataFormat::BF16, DataFormat::BF16, DataFormat::BF16, false);
    pack_config(false, false, false);

    unpack0_config(DataFormat::BF16, 16u, 16u, 0u, false, false);
}

static void setup_bf16_unpack0() {
    unpack0_config(DataFormat::BF16, 16u, 16u, 0u, false, false);
    unpack0_destination(16u);
}

// SwiGLU is elementwise over the intermediate dimension, so a shard runs it on its own slice
// with no gather. Boundary groups compute stale values outside the shard's slice;
// the packer writes only the rows owned by this shard.
static void swiglu_blocks(uint32_t output_base, uint32_t first_block, uint32_t block_count) {
    constexpr uint32_t up_z = (TT_SWIGLU_UP_L1_ADDR - TT_SWIGLU_GATE_L1_ADDR) / TT_SWIGLU_TILE_BYTES;
    static_assert((TT_SWIGLU_UP_L1_ADDR - TT_SWIGLU_GATE_L1_ADDR) % TT_SWIGLU_TILE_BYTES == 0u);
    static_assert(up_z + TT_MOE_BLOCKS <= 256u);
    const uint32_t end_block = first_block + block_count;

    swiglu_setup();
    pack_source_y_stride(32u);
    set_base_cfg(L1Base::Unpack0, TT_SWIGLU_GATE_L1_ADDR);
    unpack_bf16_to_srca_bounds<256u>();
    TTI_SETADC(1u, 1u, 1u, 0u); // both arrays unpack to SrcA row 0
    for (uint32_t group = first_block / 4u; group * 4u < end_block; ++group) {
        const uint32_t group_first = group * 4u;
        const uint32_t owned_first = group_first > first_block ? group_first : first_block;
        const uint32_t owned_end = group_first + 4u < end_block ? group_first + 4u : end_block;
        const uint32_t rows = owned_end - owned_first;

        set_pack_output_base_pipelined(output_base + owned_first * 32u);
        tensix_instruction(TT_OP_SETADC(4u, 0u, 1u, owned_first - group_first));
        TTI_ZEROACC(1u, 0u, 0u, 0u, 0u); // clear 16 Dst rows
        // Keep gate above up/output in Dst; the exponential needs all eight LRegs.
        unpack_bf16_to_srca_block(group);
        unpack_bf16_to_srca_block(up_z + group);
        TTI_MOVA2D(0u, 0u, 0u, 2u, 16u);
        TTI_CLEARDVALID(1u, 0u);

        TTI_MOVA2D(0u, 0u, 0u, 2u, 0u);
        TTI_CLEARDVALID(1u, 0u);
        TTI_STALLWAIT(STALL_SFPU, WAIT_FPU);
        sfpu_swiglu_body<0u>();
        sfpu_swiglu_body<2u>();
        TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
        pack_output_rows(rows);
    }
    TTI_SETADC(1u, 0u, 2u, 0u); // restore source Z before subsequent kernels
    wait_tensix(WAIT_PACK);
}

static void moe_reduce_expert(const uint32_t *scale_bits, uint32_t expert, uint32_t block) {
    if (expert + 1u < 4u) {
        unpack_bf16_to_srca_block((expert + 1u) * TT_MOE_BLOCKS + block);
    }
    TTI_MOVA2D(0u, 0u, 0u, 2u, 0u);
    TTI_CLEARDVALID(1u, 0u);
    TTI_STALLWAIT(STALL_SFPU, WAIT_FPU);
    if (expert == 0u) {
        sfpu_moe_reduce_first_body(scale_bits[expert]);
    } else {
        sfpu_moe_reduce_next_body(scale_bits[expert]);
    }
}

static void moe_reduce_blocks(const uint32_t *scale_bits, uint32_t first_block, uint32_t block_count) {
    static_assert(TT_MOE_DOWN4_EXPERT_BYTES == TT_MOE_BLOCKS * TT_SWIGLU_TILE_BYTES);
    static_assert(4u * TT_MOE_BLOCKS <= 256u);
    const uint32_t end = first_block + block_count;
    setup_bf16_unpack0();
    set_base_cfg(L1Base::Unpack0, TT_MOE_DOWN4_L1_ADDR);
    unpack_bf16_to_srca_bounds<256u>();
    TTI_SETADC(1u, 1u, 1u, 0u); // all four arrays unpack to SrcA row 0
    math_pack_config(DataFormat::BF16, DataFormat::BF16, DataFormat::FP32, true);
    pack_config(true, true, false);
    pack_source_y_stride(64u);
    for (uint32_t group = first_block / 4u; group * 4u < end; ++group) {
        const uint32_t first = group * 4u > first_block ? group * 4u : first_block;
        const uint32_t last = group * 4u + 4u < end ? group * 4u + 4u : end;
        const uint32_t rows = last - first;
        set_pack_output_base_pipelined(TT_MOE_OUT_L1_ADDR + first * TT_MVMUL_TILE_DIM * sizeof(float));
        tensix_instruction(TT_OP_SETADC(4u, 0u, 1u, first - group * 4u));
        TTI_ZEROACC(1u, 0u, 0u, 0u, 0u);
        unpack_bf16_to_srca_block(group); // prime expert 0; the loop unpacks one bank ahead
        for (uint32_t expert = 0; expert < 4u; ++expert) {
            moe_reduce_expert(scale_bits, expert, group);
        }
        sfpu_moe_reduce_store_body();
        TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
        pack_output_rows(rows);
    }
    TTI_SETADC(1u, 0u, 2u, 0u); // restore source Z before subsequent kernels
    wait_tensix(WAIT_PACK);
}

template<DataFormat output> static inline void setup_f32_sfpu() {
    static_assert(output == DataFormat::FP32 || output == DataFormat::BF16);
    math_pack_config(DataFormat::BF16, DataFormat::BF16, output, true);
    pack_config(true, output == DataFormat::FP32, false);

    unpack0_config(DataFormat::FP32, 64u, 1u, 0u, false, true);
    unpack0_destination(64u);
}

static inline void unpack_f32_to_dst_block(uint32_t block, uint32_t dst_row) {
    tensix_instruction(TT_OP_SETADC(1u, 0u, 2u, block));
    tensix_instruction(TT_OP_SETADC(1u, 1u, 1u, dst_row * 4u));
    TTI_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
}

static void add_f32_range(uint32_t dst_addr, uint32_t src_addr, uint32_t elements) {
    // Keep each unpack within the FP32 input address-generator limit.
    static_assert((TT_GPT_OSS_DIM + 63u) / 64u * 4u <= 256u);
    math_pack_config(DataFormat::BF16, DataFormat::BF16, DataFormat::FP32, true);
    pack_config(true, true, true);
    unpack0_config(DataFormat::FP32, 64u, 1u, 0u, false, true);
    unpack0_destination(64u);
    set_base_cfg(L1Base::Unpack0, src_addr);
    set_base_cfg(L1Base::Pack, dst_addr);
    pack_source_y_stride(64u);
    const uint32_t blocks = (elements + 63u) / 64u;
    // FP32 unpack input addresses are limited to 8192 bytes, including XEnd.
    const uint32_t first_blocks = blocks < 32u ? blocks : 32u;
    tensix_instruction(TT_OP_SETADC(1u, 1u, 0u, first_blocks * 64u - 1u));
    unpack_f32_to_dst_block(0u, 0u);
    if (blocks > 32u) {
        tensix_instruction(TT_OP_SETADC(1u, 1u, 0u, (blocks - 32u) * 64u - 1u));
        unpack_f32_to_dst_block(32u, 128u);
    }
    TTI_STALLWAIT(STALL_PACK, WAIT_UNPACK0);
    for (uint32_t block = 1u; block < blocks; ++block) {
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    // Another tile may already be gathering into the unowned portion of this buffer.
    pack_output_rows(((elements - 1u) % 64u) / 16u + 1u);
    wait_tensix(WAIT_PACK);
    TTI_SETADC(1u, 0u, 2u, 0u); // restore source Z before subsequent kernels
}

static uint32_t residual_first(uint32_t shard, uint32_t shard_count) {
    return tt_residual_block_bound(shard, shard_count) * TT_MVMUL_TILE_DIM;
}

static uint32_t residual_elements(uint32_t shard, uint32_t shard_count) {
    return (tt_residual_block_bound(shard + 1u, shard_count) - tt_residual_block_bound(shard, shard_count)) *
           TT_MVMUL_TILE_DIM;
}

static void copy_residual_slice(uint32_t dst_base, uint32_t src_base, uint32_t shard, uint32_t shard_count) {
    static_assert(TT_GPT_OSS_DIM * sizeof(float) <= TT_NOC_MAX_BULK_BYTES);
    const uint32_t offset = residual_first(shard, shard_count) * sizeof(float);
    const uint32_t elements = residual_elements(shard, shard_count);
    noc_write_from_l1(noc_rd(NOC_ID_LOGICAL) & 0xFFFu, dst_base + offset, src_base + offset, elements * sizeof(float));
}

static void add_residual_to_moe_output() {
    const uint32_t offset =
        residual_first(PHYS_RD32(TT_L1_SHARD_INDEX_ADDR), PHYS_RD32(TT_L1_SHARD_COUNT_ADDR)) * sizeof(float);

    add_f32_range(TT_MOE_OUT_L1_ADDR + offset, TT_MOE_RESIDUAL_L1_ADDR + offset,
                  residual_elements(PHYS_RD32(TT_L1_SHARD_INDEX_ADDR), PHYS_RD32(TT_L1_SHARD_COUNT_ADDR)));
}

static void add_projected_to_residual() {
    const uint32_t offset =
        residual_first(PHYS_RD32(TT_L1_SHARD_INDEX_ADDR), PHYS_RD32(TT_L1_SHARD_COUNT_ADDR)) * sizeof(float);

    add_f32_range(TT_MOE_RESIDUAL_L1_ADDR + offset, TT_MOE_PROJECTED_L1_ADDR + offset,
                  residual_elements(PHYS_RD32(TT_L1_SHARD_INDEX_ADDR), PHYS_RD32(TT_L1_SHARD_COUNT_ADDR)));
}

static void hw_init() {
    cfg(2, 0x00000000u);
    cfg(20, 0x00000000u);
    cfg(24, 0x0000FFFFu);
    cfg(28, 0x00000100u);
    cfg(50, 0x00000100u); // unpack0: add destination counters to the context base
    cfg(56, 0x00100000u);
    cfg(59, 0x00000100u); // unpack1: destination Z stride, in bytes
    cfg(84, 0x00400040u); // unpack0 contexts: compensate for the four-row destination offset
    unpack1_descriptor(DataFormat::BF16, 4u * TT_MVMUL_B_STRIDE_BYTES / sizeof(uint16_t));
    cfg(113, 0x00000008u);
    cfg(120, 0x00000005u);
    cfg(121, 0x00000003u);

    // Firmware-wide SFPU constants: LReg11/12 = +/-7, LReg13/14 = log2(e)/ln(2).
    TTI_SFPLOADI(0u, 0u, 0x40E0u);
    TTI_SFPCONFIG(0u, 11u, 0u);
    TTI_SFPLOADI(0u, 0u, 0xC0E0u);
    TTI_SFPCONFIG(0u, 12u, 0u);
    load_bits32_to_lreg<0u>(0x3FB8AA3Bu);
    TTI_SFPCONFIG(0u, 13u, 0u);
    load_bits32_to_lreg<0u>(0x3F317218u);
    TTI_SFPCONFIG(0u, 14u, 0u);

    const uint32_t my_coord = noc_rd(NOC_ID_LOGICAL) & 0xFFFu;

    noc_wr(0x100u, noc_rd(0x100u) | (1u << 16)); // NIU_CFG_0: enable command FIFOs.
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_TARG_ADDR_MID), 0u);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_RET_ADDR_MID), 0u);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_RET_ADDR_HI), my_coord);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_PACKET_TAG), 0u);
    noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_CTRL), NOC_CTRL_RD_NONPOSTED);

    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_TARG_ADDR_MID), 0u);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_TARG_ADDR_HI), my_coord);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_MID), 0u);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_PACKET_TAG), 0u);
    noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_CTRL), NOC_CTRL_WR_NONPOSTED);

    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_TARG_ADDR_MID), 0u);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_RET_ADDR_LO), TT_L1_ATOMIC_RESULT_ADDR);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_RET_ADDR_MID), 0u);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_RET_ADDR_HI), my_coord);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_PACKET_TAG), 0u);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_CTRL), NOC_CTRL_AT_POSTED);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_AT_DATA), 1);
    const uint32_t arrival = TT_L1_ALLGATHER_ARRIVE_ADDR + PHYS_RD32(TT_L1_PEER_INDEX_ADDR) * sizeof(uint32_t);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_TARG_ADDR_LO), arrival);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_AT_LEN_BE), NOC_AT_INCR_GET | ((arrival >> 2) & 3u));

    noc_wr(NOC_COUNTER_CLEAR, 0xFFFFu);

    TTI_ZEROSRC(0u, 0u, 1u, 3u);     // zero all banks of SrcA and SrcB
    TTI_ZEROACC(3u, 0u, 0u, 0u, 0u); // set zero flags for all Dst rows
    TTI_SETC16(12u, 0x0000u);        // AddrMod0 AB: nop
    TTI_SETC16(28u, 0x0000u);        // AddrMod0 Dst: nop
    TTI_SETC16(13u, 0x0000u);        // AddrMod1 AB: nop
    TTI_SETC16(29u, 0x0004u);        // AddrMod1 Dst: advance four rows
    TTI_SETC16(14u, 0x0110u);        // AddrMod2 AB: SrcA += 16, SrcB += 1
    TTI_SETC16(30u, 0x0000u);        // AddrMod2 Dst: nop
    TTI_SETC16(16u, 0x8080u);        // AddrMod4 AB: rewind SrcA/SrcB
    TTI_SETC16(32u, 0x2000u);        // AddrMod4 Dst: increment fidelity
    TTI_SETC16(19u, 0x8080u);        // AddrMod7 AB: clear SrcA/SrcB rows
    TTI_SETC16(35u, 0x8000u);        // AddrMod7 Dst: clear fidelity
    TTI_SETC16(37u, 0x0000u);        // pack AddrMod0: nop
    TTI_SETC16(38u, 0x0004u);        // pack AddrMod1: source Y += 4 rows
    TTI_SETC16(39u, 0x0020u);        // pack AddrMod2: clear source Y and its carry register
    TTI_SETADCXX(2u, 63u, 0u);       // all unpack1 operations use X bounds 0..63
    TTI_SETADCXX(4u, 15u, 0u);       // all packs use X bounds 0..15

    // Resident replay body at slot 0; record without executing or touching LRegs/Dst.
    TTI_REPLAY(0u, sfpu_attention_accumulate_body_instruction_count, 0u, 1u);
    sfpu_attention_accumulate_body();
    TTI_REPLAY(squares_replay, sfpu_squares_body_instruction_count, 0u, 1u);
    sfpu_squares_body();
    TTI_REPLAY(scale_weight_replay, sfpu_scale_weight_body_instruction_count, 0u, 1u);
    sfpu_scale_weight_body();
}

static void mvmul_bfp8_setup() {
    math_pack_config(DataFormat::BFP8, DataFormat::BF16, DataFormat::BF16, true);
    pack_config(true, false, true);

    unpack0_config(DataFormat::BFP8, 16u, 16u, 180u, true, false);
    unpack0_destination(16u);
}

template<DataFormat output> static inline void mvmul_bf16_setup(bool l1_accumulate) {
    static_assert(output == DataFormat::FP32 || output == DataFormat::BF16);
    math_pack_config(DataFormat::BF16, DataFormat::BF16, output, true);
    pack_config(true, output == DataFormat::FP32, l1_accumulate);

    unpack0_config(DataFormat::BF16, 16u, 16u, 180u, false, false);
    unpack0_destination(16u);
}

static void mvmul_read_strip(DramRef weight, uint32_t offset, uint32_t bytes, uint32_t stage, uint32_t id) {
    for (uint32_t done = 0; done < bytes; done += TT_NOC_MAX_BULK_BYTES) {
        const uint32_t remaining = bytes - done;
        const uint32_t count = remaining < TT_NOC_MAX_BULK_BYTES ? remaining : TT_NOC_MAX_BULK_BYTES;
        noc_read_to_l1_async_id(id, weight.coord, weight.addr + offset + done, stage + done, count);
    }
}

static void mvmul_unpack64() {
    TTI_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u);
    // UNPACR's source-Z increment is only two bits.
    TTI_INCADCZW(1u, 0u, 0u, 0u, 4u);
    TTI_UNPACR(1u, 4u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u);
}

template<DataFormat weights> [[gnu::always_inline]] static inline void mvmul_math64() {
    static_assert(weights == DataFormat::BF16 || weights == DataFormat::BFP8);
    // SrcA rows 0/16/32/48 and SrcB rows 0/1/2/3. Dst bases 0/8 have disjoint replicas.
    TTI_MVMUL(0u, 1u, 2u, 0u);
    TTI_MVMUL(0u, 1u, 2u, 8u);
    TTI_MVMUL(0u, 1u, 2u, 0u);
    if constexpr (weights == DataFormat::BFP8) {
        TTI_MVMUL(3u, 1u, 7u, 8u);
    } else {
        TTI_MVMUL(0u, 1u, 4u, 8u); // rewind A/B and advance to the low SrcA fidelity phase
        TTI_MVMUL(0u, 1u, 2u, 0u);
        TTI_MVMUL(0u, 1u, 2u, 8u);
        TTI_MVMUL(0u, 1u, 2u, 0u);
        TTI_MVMUL(3u, 1u, 7u, 8u);
    }
}

static void mvmul_setup_counters(uint32_t b_base) {
    set_base_cfg(L1Base::Unpack1, b_base);
    TTI_SETADCXX(1u, 1023u, 0u);
    TTI_SETRWC(0u, 0u, 0u, 0u, 0u, 15u);
}

static void mvmul_reset_state() {
    TTI_SETADCXY(7u, 0u, 0u, 0u, 0u, 10u); // reset Y on both unpackers and packer, preserving X bounds
    TTI_SETADCZW(7u, 0u, 0u, 0u, 0u, 15u);
    TTI_SETRWC(0u, 0u, 0u, 0u, 0u, 15u);
}

template<DataFormat weights> static void mvmul_block(uint32_t stage, uint32_t output, uint32_t k_tiles) {
    static_assert(weights == DataFormat::BF16 || weights == DataFormat::BFP8);
    set_base_cfg(L1Base::Pack, output);
    set_base_cfg(L1Base::Unpack0, stage);
    TTI_ZEROACC(1u, 0u, 0u, 0u, 0u);
    TTI_ZEROACC(1u, 0u, 0u, 0u, 1u);
    TTI_SETADCXY(3u, 0u, 0u, 0u, 0u, 10u);
    TTI_SETADCZW(1u, 0u, 0u, 0u, 0u, 5u);

    mvmul_unpack64();
    for (uint32_t k = 4u; k < k_tiles; k += 4u) {
        mvmul_unpack64();
        mvmul_math64<weights>();
    }
    mvmul_math64<weights>();

    TTI_STALLWAIT(STALL_SFPU, WAIT_FPU);
    sfpu_mvmul_fold_body();
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    TTI_PACR(0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 1u);
    // Complete consumption before the host RISC reuses a staging buffer or changes configuration.
    wait_tensix(WAIT_PACK);
}

template<DataFormat weights>
static void mvmul_blocks_dram(DramRef weight, uint32_t output_base, uint32_t columns, uint32_t first_block,
                              uint32_t block_count, uint32_t b_base, uint32_t output_block_bytes) {
    static_assert(weights == DataFormat::BF16 || weights == DataFormat::BFP8);
    const uint32_t bytes = weights == DataFormat::BFP8 ? TT_BFP8_STRIP_BYTES : columns * 32u;
    const uint32_t stage_base = weights == DataFormat::BFP8 ? TT_BFP8_STAGE_L1_ADDR : TT_BF16_STAGE_L1_ADDR;
    uint32_t buffer = 0u;
    mvmul_setup_counters(b_base);
    // Two strips can be outstanding; the third buffer belongs exclusively to compute.
    for (uint32_t block = 0; block < 2u && block < block_count; ++block) {
        mvmul_read_strip(weight, block * bytes, bytes, stage_base + block * bytes, block);
    }
    for (uint32_t block = 0; block < block_count; ++block) {
        const uint32_t id = block & 1u;
        noc_wait_outstanding_id(id); // First ID 0 also completes the bias prefetch.
        uint32_t spare = buffer + 2u;
        if (spare >= 3u) {
            spare -= 3u;
        }
        if (block + 2u < block_count) {
            mvmul_read_strip(weight, (block + 2u) * bytes, bytes, stage_base + spare * bytes, id);
        }
        const uint32_t row = first_block + block;
        const uint32_t output = weights == DataFormat::BFP8
                                    ? output_base + (row / 4u) * TT_SWIGLU_TILE_BYTES + (row & 3u) * 32u
                                    : output_base + block * output_block_bytes;
        mvmul_block<weights>(stage_base + buffer * bytes, output, columns / TT_MVMUL_TILE_DIM);
        if (++buffer == 3u) {
            buffer = 0u;
        }
    }
    mvmul_reset_state();
}

static void mvmul_bfp8_dram_range(DramRef weight, DramRef bias, uint32_t b_base, uint32_t output_base,
                                  uint32_t first_block, uint32_t block_count) {
    const uint32_t end_block = first_block + block_count;
    mvmul_bfp8_setup();
    // Bias and packed output have four contiguous blocks per output tile.
    for (uint32_t block = first_block; block < end_block;) {
        const uint32_t tile_end = (block / 4u + 1u) * 4u;
        const uint32_t run_end = tile_end < end_block ? tile_end : end_block;
        const uint32_t output = output_base + (block / 4u) * TT_SWIGLU_TILE_BYTES + (block & 3u) * 32u;
        noc_read_to_l1_async(bias.coord, bias.addr + block * 32u, output, (run_end - block) * 32u);
        block = run_end;
    }
    mvmul_blocks_dram<DataFormat::BFP8>(weight, output_base, TT_GPT_OSS_DIM, first_block, block_count, b_base, 32u);
}

static void mvmul_bf16_dram(DramRef weight, uint32_t columns, uint32_t row_blocks) {
    mvmul_bf16_setup<DataFormat::BF16>(false);
    mvmul_blocks_dram<DataFormat::BF16>(weight, TT_BF16_MVMUL_OUT_L1_ADDR, columns, 0u, row_blocks, TT_MVMUL_B_L1_ADDR,
                                        32u);
}

template<DataFormat output>
static void mvmul_bf16_bias_dram(DramRef weight, DramRef bias, uint32_t columns, uint32_t row_blocks,
                                 uint32_t output_base, uint32_t b_base) {
    static_assert(output == DataFormat::FP32 || output == DataFormat::BF16);
    const uint32_t output_block_bytes =
        TT_MVMUL_TILE_DIM * (output == DataFormat::FP32 ? sizeof(float) : sizeof(uint16_t));

    mvmul_bf16_setup<output>(true);
    // The first weight chunk waits on ID 0, also completing this bias read before packing.
    noc_read_to_l1_async(bias.coord, bias.addr, output_base, row_blocks * output_block_bytes);
    mvmul_blocks_dram<DataFormat::BF16>(weight, output_base, columns, 0u, row_blocks, b_base, output_block_bytes);
}

static inline void unpack_f32_to_dst_blocks(uint32_t first_block, uint32_t blocks, uint32_t dst_row) {
    // Each unpack stays within the 8 KiB FP32 input X range.
    const uint32_t first_blocks = blocks < 32u ? blocks : 32u;
    tensix_instruction(TT_OP_SETADC(1u, 1u, 0u, first_blocks * 64u - 1u));
    unpack_f32_to_dst_block(first_block, dst_row);
    if (blocks > 32u) {
        tensix_instruction(TT_OP_SETADC(1u, 1u, 0u, (blocks - 32u) * 64u - 1u));
        unpack_f32_to_dst_block(first_block + 32u, dst_row + 128u);
    }
}

static void sfpu_sum_of_squares(uint32_t input_addr, uint32_t elements) {
    static_assert(TT_RMS_PARTIALS == 64u);
    static_assert((TT_GPT_OSS_DIM + 63u) / 64u <= 64u);
    setup_f32_sfpu<DataFormat::FP32>();
    set_base_cfg(L1Base::Unpack0, input_addr);
    set_base_cfg(L1Base::Pack, TT_RMS_PARTIAL_L1_ADDR);
    const uint32_t whole_blocks = elements / TT_RMS_PARTIALS;
    const uint32_t tail = elements % TT_RMS_PARTIALS;
    unpack_f32_to_dst_blocks(0u, (elements + 63u) / 64u, 0u);
    TTI_STALLWAIT(STALL_SFPU, WAIT_UNPACK0);
    sfpu_squares_zero_body();
    for (uint32_t block = 0; block < whole_blocks; ++block) {
        TTI_REPLAY(squares_replay, sfpu_squares_body_instruction_count, 0u, 0u);
    }
    if (tail) {
        // LReg15 is the even-column element index; both columns share this even-sized tail.
        tensix_instruction(TT_OP_SFPIADD((-tail) & 0xfffu, 15u, 6u, 5u));
        sfpu_squares_tail_body();
    }
    sfpu_squares_reduce_body();
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    pack_output_4_rows();
    wait_tensix(WAIT_PACK);
    TTI_SETADCXX(1u, 63u, 0u);
    TTI_SETADC(1u, 0u, 2u, 0u);
}

static void sfpu_scale_to_bf16(uint32_t input_addr, uint32_t weight_addr, uint32_t output_base, uint32_t scale_bits,
                               uint32_t elements) {
    static_assert((TT_RMS_WEIGHT_L1_ADDR - TT_MOE_RESIDUAL_L1_ADDR) % (64u * sizeof(float)) == 0u);
    static_assert((TT_RMS_WEIGHT_L1_ADDR - TT_MOE_OUT_L1_ADDR) % (64u * sizeof(float)) == 0u);
    static_assert(TT_MOE_OUT_L1_ADDR <= TT_MOE_RESIDUAL_L1_ADDR);
    static_assert((TT_RMS_WEIGHT_L1_ADDR - TT_MOE_OUT_L1_ADDR) / (64u * sizeof(float)) + (TT_GPT_OSS_DIM + 63u) / 64u <=
                  256u);
    static_assert((TT_GPT_OSS_DIM + 63u) / 64u <= 64u);            // each array fits in half of FP32 Dst
    static_assert(TT_DIM_ROW_BLOCKS / TT_MAX_GLOBAL_SHARDS >= 4u); // every shard owns at least 64 values
    const uint32_t weight_z = (weight_addr - input_addr) / (64u * sizeof(float));
    const uint32_t blocks = (elements + 63u) / 64u;
    setup_f32_sfpu<DataFormat::BF16>();
    set_base_cfg(L1Base::Unpack0, input_addr);
    set_base_cfg(L1Base::Pack, output_base);
    unpack_f32_to_dst_blocks(0u, blocks, 0u);
    unpack_f32_to_dst_blocks(weight_z, blocks, 256u);
    sfpu_scale_weight_setup(scale_bits);
    TTI_STALLWAIT(STALL_SFPU, WAIT_UNPACK0);
    for (uint32_t block = 0u; block < blocks; ++block) {
        TTI_REPLAY(scale_weight_replay, sfpu_scale_weight_body_instruction_count, 0u, 0u);
    }
    pack_source_y_stride(32u);
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    for (uint32_t block = 1u; block < blocks; ++block) {
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    pack_output_rows(((elements - 1u) % 64u) / 16u + 1u);
    wait_tensix(WAIT_PACK);
    TTI_SETRWC(0u, 0u, 0u, 0u, 0u, 4u);
    TTI_SETADCXX(1u, 63u, 0u);
    TTI_SETADC(1u, 0u, 2u, 0u);
}

static float rv32f_exp_approx(float x) {
    const float inv_ln2 = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;

    if (x <= -20.0f) {
        return 0.0f;
    }
    if (x >= 0.0f) {
        x = 0.0f;
    }

    int n = f32_to_i32_rtne(x * inv_ln2);
    const float r = x - float(n) * ln2;
    const float r2 = r * r;
    const float r3 = r2 * r;
    const float r4 = r2 * r2;
    const float r5 = r4 * r;

    if (n < -126) {
        return 0.0f;
    }

    const uint32_t scale_bits = uint32_t(n + 127) << 23;
    const float scale = std::bit_cast<float>(scale_bits);
    return scale * (1.0f + r + 0.5f * r2 + 0.1666666716f * r3 + 0.0416666679f * r4 + 0.0083333338f * r5);
}

static float rv32f_recip_approx(float x) {
    float y = std::bit_cast<float>(0x7EF311C3u - std::bit_cast<uint32_t>(x));

    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

static float rv32f_rsqrt_approx(float x) {
    float y = std::bit_cast<float>(0x5F3759DFu - (std::bit_cast<uint32_t>(x) >> 1u));

    for (uint32_t i = 0; i < 3u; ++i) {
        const float half_x = -0.5f * x;
        const float yy = y * y;
        const float correction = __builtin_fmaf(half_x, yy, 1.5f);

        y = y * correction;
    }
    return y;
}

static void allgather_arrive(uint32_t dst_coord) {
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_TARG_ADDR_HI), dst_coord);
    noc_wr(noc_cmd_offset(NOC_ATOMIC_CMD_BUF, NOC_CMD_CTRL), 1u);
}

static void allgather_wait(uint32_t first = 0u, uint32_t count = PHYS_RD32(TT_L1_PEER_COUNT_ADDR)) {
    noc_wait_cmd_accepted(NOC_ATOMIC_CMD_BUF);
    const uint32_t shard = PHYS_RD32(TT_L1_PEER_INDEX_ADDR);

    // Track senders separately: an early next arrival cannot stand in for a late peer.
    // Targets advance independently of observed arrivals and persist across tokens.
    for (uint32_t i = first; i < first + count; ++i) {
        if (i != shard) {
            const uint32_t at = TT_L1_ALLGATHER_TARGET_ADDR + i * sizeof(uint32_t);
            const uint32_t target = PHYS_RD32(at) + 1u;

            PHYS_WR32(at, target);
            while (int32_t(PHYS_RD32(TT_L1_ALLGATHER_ARRIVE_ADDR + i * sizeof(uint32_t)) - target) < 0) {
                FENCE();
            }
        }
    }
}

static void allgather_barrier() {
    const uint32_t shard = PHYS_RD32(TT_L1_PEER_INDEX_ADDR);
    const uint32_t count = PHYS_RD32(TT_L1_PEER_COUNT_ADDR);

    for (uint32_t i = 0; i < count; ++i) {
        if (i != shard) {
            allgather_arrive(PHYS_RD32(TT_L1_SHARD_COORD_ADDR + i * sizeof(uint32_t)));
        }
    }
    allgather_wait();
}

static void allgather_range(uint32_t base, uint32_t offset, uint32_t bytes) {
    const uint32_t shard = PHYS_RD32(TT_L1_PEER_INDEX_ADDR);
    const uint32_t count = PHYS_RD32(TT_L1_PEER_COUNT_ADDR);

    for (uint32_t i = 0u; i < count; ++i) {
        if (i != shard) {
            const uint32_t peer = PHYS_RD32(TT_L1_SHARD_COORD_ADDR + i * sizeof(uint32_t));

            noc_write_from_l1_async<true>(peer, base + offset, base + offset, bytes);
        }
    }
    noc_wait_cmd_accepted(NOC_WRITE_CMD_BUF);
    noc_wait_outstanding_id0();
    for (uint32_t i = 0u; i < count; ++i) {
        if (i != shard) {
            allgather_arrive(PHYS_RD32(TT_L1_SHARD_COORD_ADDR + i * sizeof(uint32_t)));
        }
    }
    allgather_wait();
}

static void allgather_strided(uint32_t base, uint32_t stride, uint32_t pieces, uint32_t offset, uint32_t bytes) {
    const uint32_t shard = PHYS_RD32(TT_L1_PEER_INDEX_ADDR);
    const uint32_t count = PHYS_RD32(TT_L1_PEER_COUNT_ADDR);

    for (uint32_t i = 0u; i < count; ++i) {
        if (i != shard) {
            const uint32_t peer = PHYS_RD32(TT_L1_SHARD_COORD_ADDR + i * sizeof(uint32_t));

            for (uint32_t piece = 0; piece < pieces; ++piece) {
                const uint32_t at = base + piece * stride + offset;

                noc_write_from_l1_async(peer, at, at, bytes);
            }
        }
    }
    noc_wait_outstanding_id0();
    for (uint32_t i = 0u; i < count; ++i) {
        if (i != shard) {
            allgather_arrive(PHYS_RD32(TT_L1_SHARD_COORD_ADDR + i * sizeof(uint32_t)));
        }
    }
    allgather_wait();
}

// The four shards of a shared KV head exchange only within their local group.
static void allgather_group_range(uint32_t base, uint32_t offset, uint32_t bytes, uint32_t first, uint32_t count) {
    const uint32_t self = PHYS_RD32(TT_L1_PEER_INDEX_ADDR);
    if (bytes) {
        for (uint32_t peer = first; peer < first + count; ++peer) {
            if (peer != self) {
                noc_write_from_l1_async<true>(PHYS_RD32(TT_L1_SHARD_COORD_ADDR + peer * sizeof(uint32_t)),
                                              base + offset, base + offset, bytes);
            }
        }
        noc_wait_cmd_accepted(NOC_WRITE_CMD_BUF);
        noc_wait_outstanding_id0();
    }
    for (uint32_t peer = first; peer < first + count; ++peer) {
        if (peer != self) {
            allgather_arrive(PHYS_RD32(TT_L1_SHARD_COORD_ADDR + peer * sizeof(uint32_t)));
        }
    }
    allgather_wait(first, count);
}

// All local peers enter after gathering this chip's contribution. Each courier owns one
// incoming staging buffer. Ready releases that buffer, not the live computation buffer.
// Each chip's pieces * bytes must fit TT_PCIE_STAGE_BYTES; spans must be disjoint and
// identically described by every peer on a chip. All chips use the same stride and pieces.
// Peer BAR windows must use Default mode: the read-back fences payload completion before notification.
static void exchange_across_chips(uint32_t base, uint32_t offset, uint32_t bytes, uint32_t stride, uint32_t pieces) {
    const uint32_t peers = PHYS_RD32(TT_L1_PEER_COUNT_ADDR);
    const uint32_t chips = PHYS_RD32(TT_L1_SHARD_COUNT_ADDR) / peers;
    if (chips == 1u) {
        return;
    }
    auto &state = *(volatile TtPcieExchange *)TT_PCIE_EXCHANGE_ADDR;
    const uint32_t self = PHYS_RD32(TT_L1_PEER_INDEX_ADDR);
    const uint32_t seq = state.sequence + 1u;
    const bool courier = self + 1u < chips;
    state.sequence = seq;
    if (courier) {
        const uint64_t send = (uint64_t(state.send_hi) << 32) | state.send_lo;
        while (int32_t(state.ready - (seq - 1u)) < 0) {
            FENCE();
        }
        state.out_offset = offset;
        state.out_bytes = bytes;
        FENCE();
        // Keep MID across the batch, while allowing a DMA mapping to cross 4 GiB.
        uint32_t mid = uint32_t(send >> 32);
        noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_MID), mid);
        const auto write = [&](uint64_t address, uint32_t source, uint32_t size) {
            if (uint32_t(address >> 32) != mid) {
                mid = uint32_t(address >> 32);
                noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_MID), mid);
            }
            noc_write_from_l1_async(state.coord, uint32_t(address), source, size);
        };
        if (bytes) {
            for (uint32_t piece = 0; piece < pieces; ++piece) {
                write(send + TT_PCIE_STAGE_ADDR + piece * bytes, base + piece * stride + offset, bytes);
            }
        }
        write(send + TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, in_offset),
              TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, out_offset), 8u);
        // Drain payload and span before publishing the sequence to the same PCIe peer.
        noc_wait_outstanding_id0();
        const uint64_t fence = send + TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, in_offset);
        noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_TARG_ADDR_MID), uint32_t(fence >> 32));
        noc_read_to_l1(state.coord, uint32_t(fence), TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, out_offset),
                       sizeof(uint32_t));
        noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_TARG_ADDR_MID), 0u);
        state.out_sequence = seq;
        FENCE();
        write(send + TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, in_sequence),
              TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, out_sequence), 4u);
        noc_wr(noc_cmd_offset(NOC_WRITE_CMD_BUF, NOC_RET_ADDR_MID), 0u);
        while (int32_t(state.in_sequence - seq) < 0) {
            FENCE();
        }
        FENCE();
        const uint32_t in_offset = state.in_offset;
        const uint32_t in_bytes = state.in_bytes;
        if (in_bytes) {
            for (uint32_t peer = 0; peer < peers; ++peer) {
                const uint32_t coord = PHYS_RD32(TT_L1_SHARD_COORD_ADDR + peer * sizeof(uint32_t));
                for (uint32_t piece = 0; piece < pieces; ++piece) {
                    noc_write_from_l1_async(coord, base + piece * stride + in_offset,
                                            TT_PCIE_STAGE_ADDR + piece * in_bytes, in_bytes);
                }
            }
        }
        noc_wait_outstanding_id0();
        const uint64_t source = (uint64_t(state.source_hi) << 32) | state.source_lo;
        state.out_ready = seq;
        FENCE();
        noc_write_from_l1_wide_async(state.coord, source + TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, ready),
                                     TT_PCIE_EXCHANGE_ADDR + offsetof(TtPcieExchange, out_ready), 4u);
    }
    // All remote contributions are visible before any peer leaves. Ready delivery overlaps
    // this barrier, but its source word must be drained before the next exchange can reuse it.
    allgather_barrier();
    if (courier) {
        noc_wait_outstanding_id0();
    }
}

static void rms_norm_to_bf16_vector(volatile float *input, DramRef weight, uint32_t output_base) {
    volatile float *partials = (volatile float *)TT_RMS_PARTIAL_L1_ADDR;
    volatile float *scalars = (volatile float *)TT_RMS_SCALAR_L1_ADDR;
    const uint32_t shard = PHYS_RD32(TT_L1_SHARD_INDEX_ADDR);
    const uint32_t shard_count = PHYS_RD32(TT_L1_SHARD_COUNT_ADDR);
    const uint32_t peers = PHYS_RD32(TT_L1_PEER_COUNT_ADDR);
    const uint32_t local = PHYS_RD32(TT_L1_PEER_INDEX_ADDR);
    const bool multichip = shard_count > peers;
    // Across chips, gather input once and normalize redundantly to avoid a second exchange.
    // Within one chip, sharded normalization is cheaper than gathering the FP32 input.
    const uint32_t first = multichip ? 0u : residual_first(local, peers);
    const uint32_t elements = multichip ? TT_GPT_OSS_DIM : residual_elements(local, peers);
    static_assert(TT_GPT_OSS_DIM * sizeof(float) <= TT_PCIE_STAGE_BYTES);
    // Keep the weight read out of the gather's ID 0 completion waits.
    noc_read_to_l1_async_id(1u, weight.coord, weight.addr + first * sizeof(float),
                            TT_RMS_WEIGHT_L1_ADDR + first * sizeof(float), elements * sizeof(float));
    if (multichip) {
        const uint32_t owned_first = residual_first(shard, shard_count);
        const uint32_t owned_elements = residual_elements(shard, shard_count);
        const uint32_t chip_first = shard - local;
        const uint32_t chip_begin = residual_first(chip_first, shard_count);
        const uint32_t chip_end = residual_first(chip_first + peers, shard_count);
        allgather_range((uint32_t)input, owned_first * sizeof(float), owned_elements * sizeof(float));
        exchange_across_chips((uint32_t)input, chip_begin * sizeof(float), (chip_end - chip_begin) * sizeof(float), 0u,
                              1u);
    }
    sfpu_sum_of_squares((uint32_t)input + first * sizeof(float), elements);
    FENCE(); // BRISC reads the partial sum just written by the packer.
    float sum = partials[0];
    if (!multichip) {
        scalars[local] = sum;
        allgather_range(TT_RMS_SCALAR_L1_ADDR, local * sizeof(float), sizeof(float));
        FENCE();
        sum = 0.0f;
        for (uint32_t i = 0; i < peers; ++i) {
            sum += scalars[i];
        }
    }
    const float mean = sum * (1.0f / float(TT_GPT_OSS_DIM));
    const float scale = rv32f_rsqrt_approx(mean + 1.0e-5f);

    noc_wait_outstanding_id(1u);
    sfpu_scale_to_bf16((uint32_t)input + first * sizeof(float), TT_RMS_WEIGHT_L1_ADDR + first * sizeof(float),
                       output_base + first * sizeof(uint16_t), std::bit_cast<uint32_t>(scale), elements);
    if (!multichip) {
        allgather_range(output_base, first * sizeof(uint16_t), elements * sizeof(uint16_t));
    }
}

template<uint32_t a, uint32_t b, uint32_t mode> static inline void router_swap() {
    TTI_SFPSWAP(0u, a, b, mode);
    TTI_SFPNOP; // Two cycles with an explicit NOP, three with the automatic stall.
}

template<uint32_t mode> static inline void router_merge4() {
    router_swap<0u, 2u, mode>();
    router_swap<1u, 3u, mode>();
    router_swap<0u, 1u, mode>();
    router_swap<2u, 3u, mode>();
}

template<uint32_t experts> static void router_top4_sfpu(uint32_t *expert_index, float *selected) {
    static_assert(experts == 32u || experts == 128u);
    setup_f32_sfpu<DataFormat::FP32>();
    set_base_cfg(L1Base::Unpack0, TT_BF16_MVMUL_OUT_L1_ADDR);
    set_base_cfg(L1Base::Pack, TT_SWIGLU_GATE_L1_ADDR);
    TTI_SETADCXX(1u, experts - 1u, 0u);
    unpack_f32_to_dst_block(0u, 0u);
    TTI_STALLWAIT(STALL_SFPU, WAIT_UNPACK0);
    TTI_SFPLOAD(0u, 3u, 0u, 0u);
    TTI_SFPLOAD(1u, 3u, 0u, 2u);
    if constexpr (experts == 128u) {
        TTI_SFPLOAD(2u, 3u, 0u, 4u);
        TTI_SFPLOAD(3u, 3u, 0u, 6u);
    }
    TTI_SFPIADD(0u, 15u, 4u, 5u);
    TTI_SFPIADD(1u, 4u, 5u, 5u);
    if constexpr (experts == 128u) {
        TTI_SFPIADD(64u, 4u, 6u, 5u);
        TTI_SFPIADD(65u, 4u, 7u, 5u);
    }
    TTI_SFPCONFIG(4u, 15u, 1u); // Paired value/index swaps; no ordinary index writes until disabled.
    if constexpr (experts == 128u) {
        // Retain and sort four of each sixteen inputs per column; discarded values cannot be top four.
        router_swap<0u, 1u, 3u>();
        router_swap<2u, 3u, 3u>();
        router_swap<0u, 2u, 3u>();
        router_swap<1u, 3u, 3u>();
        router_swap<1u, 2u, 3u>();
        TTI_SFPTRANSP(0u, 0u, 0u, 0u);
        router_swap<0u, 1u, 1u>();
        router_swap<2u, 3u, 1u>();
        TTI_SFPTRANSP(0u, 0u, 0u, 0u);
        router_merge4<2u>();
        TTI_SFPTRANSP(0u, 0u, 0u, 0u);
        router_swap<0u, 2u, 1u>();
        TTI_SFPTRANSP(0u, 0u, 0u, 0u);
        router_merge4<1u>();
        TTI_SFPTRANSP(0u, 0u, 0u, 0u); // Consolidate 32 candidates into LReg0/4.
    } else {
        router_swap<0u, 1u, 3u>();
        TTI_SFPTRANSP(0u, 0u, 0u, 0u);
        router_swap<0u, 1u, 1u>();
        TTI_SFPTRANSP(0u, 0u, 0u, 0u);
        router_swap<0u, 1u, 1u>();
    }
    TTI_SFPCONFIG(0u, 15u, 1u);
    // INT32 stores preserve raw bits (including small integer indices); FP32 packing is an identity.
    TTI_SFPSTORE(0u, 4u, 0u, 0u);
    if constexpr (experts == 128u) {
        TTI_SFPSTORE(4u, 4u, 0u, 2u);
    } else {
        TTI_SFPSTORE(1u, 4u, 0u, 2u);
        TTI_SFPSTORE(4u, 4u, 0u, 4u);
        TTI_SFPSTORE(5u, 4u, 0u, 6u);
    }
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    if constexpr (experts == 128u) {
        TTI_PACR(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
    } else {
        // Pack only the valid halves, with values followed by indices.
        pack_source_y_stride(64u);
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 3u, 0u, 0u, 0u, 0u, 0u);
        TTI_PACR(0u, 0u, 0u, 2u, 0u, 0u, 3u, 0u, 0u, 0u, 0u, 1u);
    }
    wait_tensix(WAIT_PACK);
    TTI_SETADCXX(1u, 63u, 0u);
    TTI_SETADC(1u, 0u, 2u, 0u);
    FENCE();

    // Merge eight sorted four-element lists, replenishing only the winning head.
    volatile float *out = (volatile float *)TT_SWIGLU_GATE_L1_ADDR;
    float head[8];
    uint32_t position[8];
#pragma GCC unroll 8
    for (uint32_t c = 0; c < 8u; ++c) {
        position[c] = 2u * c;
        head[c] = out[position[c]];
    }
    for (uint32_t rank = 0; rank < 4u; ++rank) {
        uint32_t best = 0u;
        float value = head[0];
#pragma GCC unroll 8
        for (uint32_t c = 1u; c < 8u; ++c) {
            if (head[c] > value) {
                value = head[c];
                best = c;
            }
        }
        const uint32_t pos = position[best];
        selected[rank] = value;
        expert_index[rank] = std::bit_cast<uint32_t>(out[pos + (experts == 128u ? 1u : 32u)]);
        if (rank != 3u) {
            position[best] += experts == 128u ? 16u : (pos & 1u) ? 15u : 1u;
            head[best] = out[position[best]];
        }
    }
}

static void moe_top4(uint32_t experts, uint32_t *expert_index, uint32_t *scale_bits) {
    float selected[4];
    if (experts == 128u) {
        router_top4_sfpu<128u>(expert_index, selected);
    } else {
        router_top4_sfpu<32u>(expert_index, selected);
    }
    float weights[4];
    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < 4u; ++i) {
        weights[i] = rv32f_exp_approx(selected[i] - selected[0]);
        weight_sum += weights[i];
    }
    const float reciprocal = rv32f_recip_approx(weight_sum);
    for (uint32_t i = 0; i < 4u; ++i) {
        scale_bits[i] = std::bit_cast<uint32_t>(weights[i] * reciprocal);
    }
}

static DramRef desc_dram_ref(volatile uint32_t *desc, uint32_t ref_index) {
    DramRef ref;

    ref.coord = desc[ref_index * 2u];
    ref.addr = desc[ref_index * 2u + 1u];
    return ref;
}

static void moe_hidden(volatile uint32_t *desc, const uint32_t *expert_index, uint32_t shard, uint32_t shard_count) {
    DramRef gate_weight = desc_dram_ref(desc, TT_LAYER_REF_GATE_WEIGHT);
    DramRef gate_bias = desc_dram_ref(desc, TT_LAYER_REF_GATE_BIAS);
    DramRef up_weight = desc_dram_ref(desc, TT_LAYER_REF_UP_WEIGHT);
    DramRef up_bias = desc_dram_ref(desc, TT_LAYER_REF_UP_BIAS);
    const uint32_t first_block = tt_hidden_block_bound(shard, shard_count);
    const uint32_t block_count = tt_hidden_block_bound(shard + 1u, shard_count) - first_block;
    const uint32_t expert_stride = block_count * TT_BFP8_STRIP_BYTES;

    for (uint32_t selected = 0; selected < 4u; ++selected) {
        const uint32_t expert = expert_index[selected];
        const uint32_t weight_offset = expert * expert_stride;
        const uint32_t bias_offset = expert * TT_GPT_OSS_DIM * sizeof(uint16_t);
        DramRef weight = gate_weight;
        DramRef bias = gate_bias;

        weight.addr += weight_offset;
        bias.addr += bias_offset;
        mvmul_bfp8_dram_range(weight, bias, TT_MVMUL_B_L1_ADDR, TT_SWIGLU_GATE_L1_ADDR, first_block, block_count);
        weight = up_weight;
        bias = up_bias;
        weight.addr += weight_offset;
        bias.addr += bias_offset;
        mvmul_bfp8_dram_range(weight, bias, TT_MVMUL_B_L1_ADDR, TT_SWIGLU_UP_L1_ADDR, first_block, block_count);
        profile_stage(TT_PROFILE_EXPERT);
        swiglu_blocks(TT_MOE_HIDDEN_L1_ADDR + selected * TT_SWIGLU_OUTPUT_BYTES, first_block, block_count);
        profile_stage(TT_PROFILE_SWIGLU);
    }
    allgather_strided(TT_MOE_HIDDEN_L1_ADDR, TT_SWIGLU_OUTPUT_BYTES, 4u,
                      first_block * TT_MVMUL_TILE_DIM * sizeof(uint16_t),
                      block_count * TT_MVMUL_TILE_DIM * sizeof(uint16_t));
    const uint32_t peers = PHYS_RD32(TT_L1_PEER_COUNT_ADDR);
    const uint32_t chip_first = shard - shard % peers;
    const uint32_t begin = tt_hidden_block_bound(chip_first, shard_count);
    const uint32_t end = tt_hidden_block_bound(chip_first + peers, shard_count);
    exchange_across_chips(TT_MOE_HIDDEN_L1_ADDR, begin * TT_MVMUL_TILE_DIM * sizeof(uint16_t),
                          (end - begin) * TT_MVMUL_TILE_DIM * sizeof(uint16_t), TT_SWIGLU_OUTPUT_BYTES, 4u);
}

static void moe_down(volatile uint32_t *desc, const uint32_t *expert_index, const uint32_t *scale_bits, uint32_t shard,
                     uint32_t shard_count) {
    DramRef down_weight = desc_dram_ref(desc, TT_LAYER_REF_DOWN_WEIGHT);
    DramRef down_bias = desc_dram_ref(desc, TT_LAYER_REF_DOWN_BIAS);
    const uint32_t first_block = tt_residual_block_bound(shard, shard_count);
    const uint32_t block_count = tt_residual_block_bound(shard + 1u, shard_count) - first_block;
    const uint32_t expert_stride = block_count * TT_BFP8_STRIP_BYTES;

    for (uint32_t selected = 0; selected < 4u; ++selected) {
        const uint32_t expert = expert_index[selected];
        DramRef weight = down_weight;
        DramRef bias = down_bias;

        weight.addr += expert * expert_stride;
        bias.addr += expert * TT_GPT_OSS_DIM * sizeof(uint16_t);
        mvmul_bfp8_dram_range(weight, bias, TT_MOE_HIDDEN_L1_ADDR + selected * TT_SWIGLU_OUTPUT_BYTES,
                              TT_MOE_DOWN4_L1_ADDR + selected * TT_MOE_DOWN4_EXPERT_BYTES, first_block, block_count);
    }
    profile_stage(TT_PROFILE_EXPERT);
    moe_reduce_blocks(scale_bits, first_block, block_count);
    profile_stage(TT_PROFILE_RESIDUAL);
}

static void moe_router(DramRef weight, DramRef bias, uint32_t *expert_index, uint32_t *scale_bits) {
    const uint32_t experts = PHYS_RD32(TT_L1_EXPERT_COUNT_ADDR);
    const uint32_t parts = tt_router_shards(experts, PHYS_RD32(TT_L1_PEER_COUNT_ADDR));
    const uint32_t rows = experts / parts;
    const uint32_t offset = (PHYS_RD32(TT_L1_PEER_INDEX_ADDR) % parts) * rows * sizeof(float);

    bias.addr += offset;
    mvmul_bf16_bias_dram<DataFormat::FP32>(weight, bias, TT_GPT_OSS_DIM, rows / TT_MVMUL_TILE_DIM,
                                           TT_BF16_MVMUL_OUT_L1_ADDR + offset, TT_MVMUL_B_L1_ADDR);
    if (parts > 1u) {
        allgather_range(TT_BF16_MVMUL_OUT_L1_ADDR, offset, rows * sizeof(float));
        FENCE(); // Top-four selection reads logits written by chip-local peers.
    }
    moe_top4(experts, expert_index, scale_bits);
}

static void moe(DramRef post_norm, volatile uint32_t *desc) {
    const uint32_t shard = PHYS_RD32(TT_L1_SHARD_INDEX_ADDR);
    const uint32_t shard_count = PHYS_RD32(TT_L1_SHARD_COUNT_ADDR);
    uint32_t expert_index[4];
    uint32_t scale_bits[4];

    add_projected_to_residual();
    profile_stage(TT_PROFILE_RESIDUAL);
    rms_norm_to_bf16_vector((volatile float *)TT_MOE_RESIDUAL_L1_ADDR, post_norm, TT_MVMUL_B_L1_ADDR);
    profile_stage(TT_PROFILE_NORM);
    moe_router(desc_dram_ref(desc, TT_LAYER_REF_ROUTER_WEIGHT), desc_dram_ref(desc, TT_LAYER_REF_ROUTER_BIAS),
               expert_index, scale_bits);
    profile_stage(TT_PROFILE_ROUTER);
    moe_hidden(desc, expert_index, shard, shard_count);
    moe_down(desc, expert_index, scale_bits, shard, shard_count);
    add_residual_to_moe_output();
    profile_stage(TT_PROFILE_RESIDUAL);
}

static uint32_t attention_cache_offset(uint32_t token, uint32_t kv_head, uint32_t kv_heads) {
    return ((token * kv_heads + kv_head) * TT_GPT_OSS_HEAD_SIZE) * sizeof(uint16_t);
}

static void attention_load_cache_chunk(DramRef cache, uint32_t first_token, uint32_t tokens, uint32_t kv_head,
                                       uint32_t kv_heads, uint32_t dst_addr) {
    const uint32_t row_bytes = TT_ATTENTION_CACHE_ROW_BYTES;

    if (kv_heads != 1u) {
        for (uint32_t token = 0; token < tokens; ++token) {
            noc_read_to_l1_async(cache.coord,
                                 cache.addr + attention_cache_offset(first_token + token, kv_head, kv_heads),
                                 dst_addr + token * row_bytes, row_bytes);
        }
        noc_wait_outstanding_id0();
        return;
    }

    const uint32_t src_addr = cache.addr + first_token * row_bytes;
    const uint32_t bytes = tokens * row_bytes;
    uint32_t offset = 0;

    while (offset < bytes) {
        const uint32_t remaining = bytes - offset;
        const uint32_t current = remaining < TT_NOC_MAX_BULK_BYTES ? remaining : TT_NOC_MAX_BULK_BYTES;

        noc_read_to_l1_async(cache.coord, src_addr + offset, dst_addr + offset, current);
        offset += current;
    }
    noc_wait_outstanding_id0();
}

static void attention_score_sfpu_setup() {
    setup_bf16_unpack0();
    math_pack_config(DataFormat::BF16, DataFormat::BF16, DataFormat::FP32, true);
    pack_config(true, true, false);
    set_base_cfg(L1Base::Pack, TT_ATTENTION_DOT_L1_ADDR);
}

// Query heads are the same for every attended token, so they are staged once per pass.
// Keep their BF16 Dst layout separate from the FP32 key and score batches.
static void attention_stage_query_sfpu(const volatile uint16_t *query, uint32_t head_base, bool owned_pair) {
    const uint32_t query_base = head_base * TT_GPT_OSS_HEAD_SIZE;

    math_pack_config(DataFormat::BF16, DataFormat::BF16, DataFormat::FP32, false);
    if (owned_pair) {
        unpack_bf16_to_srca_bounds<128u>();
    } else {
        unpack_bf16_to_srca_bounds<512u>();
    }
    unpack_bf16_to_srca((uint32_t)(query + query_base));
    TTI_MOVA2D(0u, 0u, 0u, 2u, 16u);
    if (!owned_pair) {
        TTI_MOVA2D(0u, 8u, 0u, 2u, 32u);
        TTI_MOVA2D(0u, 16u, 0u, 2u, 48u);
        TTI_MOVA2D(0u, 24u, 0u, 2u, 64u);
    }
    TTI_CLEARDVALID(1u, 0u);
    TTI_STALLWAIT(STALL_THCON, WAIT_FPU);
}

static void attention_unpack_setup(uint32_t base) {
    static_assert(TT_ATTENTION_CACHE_ROW_BYTES == 4u * 16u * sizeof(uint16_t));
    static_assert(TT_ATTENTION_SCORE_CHUNK_TOKENS <= 256u);
    unpack0_config(DataFormat::BF16, 16u, 4u, 0u, false, false);
    set_base_cfg(L1Base::Unpack0, base);
    unpack_bf16_to_srca_bounds<64u>();
    TTI_SETADC(1u, 1u, 1u, 0u);
}

static inline void attention_unpack_row() {
    TTI_UNPACR(0u, 1u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u); // source Z += one cache row
}

// One SrcA bank holds sixteen keys. FP32 Dst rows 128..191 are disjoint from
// the BF16 queries and FP32 score scratch, including their paired physical rows.
static constexpr uint32_t attention_key_batch = 16u;
static constexpr uint32_t attention_key_row = 128u;
static constexpr uint32_t attention_score_row = attention_key_row + attention_key_batch * 4u;
static_assert(attention_score_row + attention_key_batch * 4u <= 256u);
static_assert(TT_ATTENTION_DOT_L1_ADDR + attention_key_batch * 64u * sizeof(float) <= TT_ATTENTION_SCORE_L1_ADDR);

static void attention_unpack_keys(uint32_t first_token) {
    static_assert(attention_key_batch * TT_GPT_OSS_HEAD_SIZE == 64u * 16u);
    static_assert(TT_ATTENTION_SCORE_CHUNK_TOKENS % attention_key_batch == 0u);
    tensix_instruction(TT_OP_SETADC(1u, 0u, 2u, first_token));
    TTI_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u);
}

static void attention_stage_keys_sfpu() {
#pragma GCC unroll 8
    for (uint32_t row = 0u; row < 64u; row += 8u) {
        tensix_instruction(TT_OP_MOVA2D(0u, row, 0u, 2u, attention_key_row + row));
    }
    TTI_CLEARDVALID(1u, 0u);
    TTI_STALLWAIT(STALL_SFPU, WAIT_FPU);
}

template<uint32_t q_row_base> [[gnu::always_inline]] static inline void attention_scores_sfpu_4_body(uint32_t key_row) {
    sfpu_attention_load_key_body(key_row);
    sfpu_attention_partial_sum_body<q_row_base>();
}

template<bool owned_pair> static void attention_scores_sfpu_tokens(uint32_t tokens) {
    for (uint32_t token = 0; token < tokens; ++token) {
        const uint32_t key_row = attention_key_row + token * 4u;
        const uint32_t score_row = attention_score_row + token * 4u;
        if constexpr (owned_pair) {
            sfpu_attention_load_key_body(key_row);
            sfpu_attention_partial_sum_pair_body<16u>();
            sfpu_attention_store_four_scores_body(score_row);
        } else {
            attention_scores_sfpu_4_body<16u>(key_row);
            // Preserve the first group's partial sums across the second transpose.
            tensix_instruction(TT_OP_SFPSTORE(4u, 3u, 0u, score_row));
            attention_scores_sfpu_4_body<48u>(key_row);
            tensix_instruction(TT_OP_SFPLOAD(5u, 3u, 0u, score_row));
            sfpu_reduce_subvec_sum_pair_body();
            sfpu_attention_store_scores_body(score_row);
        }
    }
}

// Four-head groups occupy even/odd columns of the same packed rows.
static void attention_scores_sfpu_batch(volatile float *scores, float *maximum, uint32_t tokens, bool owned_pair) {
    const uint32_t body_heads = TT_GPT_OSS_QUERY_GROUP / 2u;
    volatile float *partials = (volatile float *)TT_ATTENTION_DOT_L1_ADDR;

    if (owned_pair) {
        attention_scores_sfpu_tokens<true>(tokens);
    } else {
        attention_scores_sfpu_tokens<false>(tokens);
    }
    TTI_SETADC(4u, 0u, 1u, attention_score_row);
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    for (uint32_t token = 1u; token < tokens; ++token) {
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    TTI_PACR(0u, 0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
    wait_tensix(WAIT_PACK);
    FENCE(); // BRISC reads scores just written by the packer.
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < body_heads; ++head) {
            const float even = partials[token * 64u + head * 16u];
            // Keep unused softmax lanes initialized without changing the eight-head score stride.
            const float odd = owned_pair ? even : partials[token * 64u + head * 16u + 1u];
            scores[token * TT_GPT_OSS_QUERY_GROUP + head] = even;
            scores[token * TT_GPT_OSS_QUERY_GROUP + body_heads + head] = odd;
            maximum[head] = __builtin_fmaxf(maximum[head], even);
            maximum[body_heads + head] = __builtin_fmaxf(maximum[body_heads + head], odd);
        }
    }
}

static_assert(TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS == 8u,
              "attention value batching assumes eight query heads per KV head");

static void attention_accumulate_begin_sfpu() {
    setup_f32_sfpu<DataFormat::FP32>();
    sfpu_attention_zero_body();
}

[[gnu::always_inline]] static inline void attention_accumulate_head(uint32_t score) {
    sfpu_attention_scale_body(score);
    TTI_REPLAY(0u, sfpu_attention_accumulate_body_instruction_count, 0u, 0u);
}

template<uint32_t heads>
[[gnu::always_inline]] static inline void attention_accumulate_value_sfpu(const volatile float *token_scores,
                                                                          bool unpack_next) {
    static_assert(heads == 2u || heads == 8u);
    // Hide the initial score loads behind V staging, then load one head ahead.
    const uint32_t score0 = std::bit_cast<uint32_t>(token_scores[0]);
    const uint32_t score1 = std::bit_cast<uint32_t>(token_scores[1]);
    if (unpack_next) {
        attention_unpack_row();
    }
    TTI_MOVA2D(0u, 0u, 0u, 2u, 32u);
    TTI_CLEARDVALID(1u, 0u);
    TTI_STALLWAIT(STALL_SFPU, WAIT_FPU);
    sfpu_attention_load_value_body();

    attention_accumulate_head(score0);
    if constexpr (heads == 8u) {
        const uint32_t score2 = std::bit_cast<uint32_t>(token_scores[2]);
        attention_accumulate_head(score1);
        const uint32_t score3 = std::bit_cast<uint32_t>(token_scores[3]);
        attention_accumulate_head(score2);
        const uint32_t score4 = std::bit_cast<uint32_t>(token_scores[4]);
        attention_accumulate_head(score3);
        const uint32_t score5 = std::bit_cast<uint32_t>(token_scores[5]);
        attention_accumulate_head(score4);
        const uint32_t score6 = std::bit_cast<uint32_t>(token_scores[6]);
        attention_accumulate_head(score5);
        const uint32_t score7 = std::bit_cast<uint32_t>(token_scores[7]);
        attention_accumulate_head(score6);
        attention_accumulate_head(score7);
    } else {
        attention_accumulate_head(score1);
    }
    TTI_SETRWC(0u, 0u, 0u, 0u, 0u, 4u); // restore Dst RWC for the next V staging and final normalization
}

static uint32_t attention_score_l1_token_addr(uint32_t token) {
    return TT_ATTENTION_SCORE_L1_ADDR + token * TT_ATTENTION_SCORE_STRIDE_BYTES;
}

template<uint32_t heads> static void attention_accumulate_chunk(uint32_t tokens, uint32_t head_first) {
    attention_unpack_row(); // prime the first bank; the loop unpacks one bank ahead
    for (uint32_t token = 0; token < tokens; ++token) {
        const volatile float *scores = (const volatile float *)attention_score_l1_token_addr(token);
        attention_accumulate_value_sfpu<heads>(scores + head_first, token + 1u < tokens);
    }
}

template<uint32_t head> static inline void attention_normalize_head(const float *reciprocal) {
    sfpu_attention_normalize_body<head * 4u>(std::bit_cast<uint32_t>(reciprocal[head]));
}

static void attention_accumulate_end_sfpu(uint32_t output_base, const float *reciprocal, uint32_t heads) {
    static_assert(TT_GPT_OSS_HEAD_SIZE == 64u);
    setup_f32_sfpu<DataFormat::BF16>();
    // Normalize the completed FP32 sums, including the sink in the denominator.
    attention_normalize_head<0u>(reciprocal);
    attention_normalize_head<1u>(reciprocal);
    if (heads == 8u) {
        attention_normalize_head<2u>(reciprocal);
        attention_normalize_head<3u>(reciprocal);
        attention_normalize_head<4u>(reciprocal);
        attention_normalize_head<5u>(reciprocal);
        attention_normalize_head<6u>(reciprocal);
        attention_normalize_head<7u>(reciprocal);
    }
    set_base_cfg(L1Base::Pack, output_base);
    pack_source_y_stride(32u);
    // Owned heads accumulate contiguously from Dst row zero.
    TTI_SETADC(4u, 0u, 1u, 0u);
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    for (uint32_t head = 1u; head < heads; ++head) {
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    TTI_PACR(0u, 0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
    wait_tensix(WAIT_PACK);
}

static uint32_t attention_score_offset(uint32_t token) {
    return token * TT_ATTENTION_SCORE_STRIDE_BYTES;
}

static uint32_t attention_score_chunk_count(uint32_t start, uint32_t count) {
    const uint32_t remaining = count - start;

    if (remaining < TT_ATTENTION_SCORE_CHUNK_TOKENS) {
        return remaining;
    }
    return TT_ATTENTION_SCORE_CHUNK_TOKENS;
}

static void attention_fill_maximum(const float *maximum) {
    volatile float *pattern = (volatile float *)TT_ATTENTION_MAXIMUM_L1_ADDR;

    for (uint32_t i = 0; i < TT_ATTENTION_MAXIMUM_VALUES; ++i) {
        pattern[i] = maximum[i & (TT_GPT_OSS_QUERY_GROUP - 1u)];
    }
}

static void sfpu_attention_softmax(uint32_t score_addr, uint32_t maximum_addr, uint32_t tokens) {
    // Softmax borrows the SwiGLU clamp registers for exp polynomial coefficients.
    load_bits32_to_lreg<0u>(0x3E2AAAABu);
    TTI_SFPCONFIG(0u, 11u, 0u);
    load_bits32_to_lreg<0u>(0x3D2AAAABu);
    TTI_SFPCONFIG(0u, 12u, 0u);
    const uint32_t blocks = (tokens + 7u) / 8u;
    constexpr uint32_t maximum_row = TT_ATTENTION_SCORE_CHUNK_TOKENS * TT_GPT_OSS_QUERY_GROUP / 16u;
    static_assert(maximum_row % 4u == 0u);
    static_assert(maximum_row * 16u * sizeof(float) <= 8192u); // unpack input X byte limit
    static_assert(maximum_row + 4u <= 512u);                   // FP32 Dst rows
    setup_f32_sfpu<DataFormat::FP32>();
    set_base_cfg(L1Base::Unpack0, maximum_addr);
    TTI_SETADCXX(1u, 63u, 0u);
    unpack_f32_to_dst_block(0u, maximum_row);
    set_base_cfg(L1Base::Pack, score_addr);
    // Keep the base change ordered behind the maximum unpack without blocking BRISC.
    set_base_cfg_pipelined(L1Base::Unpack0, score_addr);
    TTI_SETADC(1u, 1u, 1u, 0u); // Dst row 0; also separates WRCFG from UNPACR
    TTI_SETADC(1u, 0u, 2u, 0u);
    tensix_instruction(TT_OP_SETADC(1u, 1u, 0u, blocks * 64u - 1u));
    TTI_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
    TTI_STALLWAIT(STALL_SFPU, WAIT_UNPACK0);
    for (uint32_t row = 0u; row < blocks * 4u; row += 4u) {
        sfpu_attention_exp_half<0u, 0u>(maximum_row, row);
        sfpu_attention_exp_half<2u, 1u>(maximum_row, row); // Dst RWC += four rows
    }
    pack_source_y_stride(64u);
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    for (uint32_t block = 1u; block < blocks; ++block) {
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    TTI_PACR(0u, 0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
    wait_tensix(WAIT_PACK);
    // Reduce complete eight-token blocks into the now-unused dot-product scratch.
    // Last=1 restarts the L1 destination; AddrMod1 advances only the Dst source.
    const uint32_t sum_blocks = tokens / 8u;
    if (sum_blocks > 1u) {
        set_base_cfg(L1Base::Pack, TT_ATTENTION_DOT_L1_ADDR);
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
        wait_tensix(WAIT_PACK); // first block initializes the partial sums before enabling accumulation
        pack_config(true, true, true);
        for (uint32_t block = 1u; block + 1u < sum_blocks; ++block) {
            TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
        }
        TTI_PACR(0u, 0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
        wait_tensix(WAIT_PACK);
    }
    TTI_SETRWC(0u, 0u, 0u, 0u, 0u, 4u); // reset Dst RWC after SFPU completion
    TTI_SFPLOADI(0u, 0u, 0x40E0u);
    TTI_SFPCONFIG(0u, 11u, 0u);
    TTI_SFPLOADI(0u, 0u, 0xC0E0u);
    TTI_SFPCONFIG(0u, 12u, 0u);
    TTI_SETADCXX(1u, 63u, 0u);
}

static void sfpu_rope(uint32_t src_base, uint32_t dst_base, uint32_t heads) {
    static_assert(TT_GPT_OSS_HEAD_SIZE == 64u);
    static_assert(TT_GPT_OSS_QUERY_HEADS <= 64u);
    static_assert(TT_GPT_OSS_KV_HEADS <= 64u);

    setup_f32_sfpu<DataFormat::BF16>();
    set_base_cfg(L1Base::Unpack0, TT_ROPE_COS_L1_ADDR);
    TTI_SETADCXX(1u, 63u, 0u);
    unpack_f32_to_dst_block(0u, 256u);
    set_base_cfg_pipelined(L1Base::Unpack0, TT_ROPE_SIN_L1_ADDR);
    unpack_f32_to_dst_block(0u, 260u);
    TTI_STALLWAIT(STALL_SFPU, WAIT_UNPACK0);
    sfpu_rope_coefficients_body();
    // Keep coefficients in LRegs before reusing their Dst rows for swapped inputs.
    TTI_STALLWAIT(STALL_THCON, WAIT_SFPU);
    set_base_cfg_pipelined(L1Base::Unpack0, src_base);
    set_base_cfg(L1Base::Pack, dst_base);
    unpack_f32_to_dst_blocks(0u, heads, 0u);
    // Originals occupy rows 0..255; the two half-unpacks construct rows 256..511.
    TTI_SETADCXX(1u, 63u, 32u);
    for (uint32_t head = 0; head < heads; ++head) {
        unpack_f32_to_dst_block(head, 256u + head * 4u);
    }
    TTI_SETADCXX(1u, 31u, 0u);
    for (uint32_t head = 0; head < heads; ++head) {
        unpack_f32_to_dst_block(head, 258u + head * 4u);
    }
    TTI_STALLWAIT(STALL_SFPU, WAIT_UNPACK0);
    for (uint32_t head = 0; head < heads; ++head) {
        sfpu_rope_body();
    }
    pack_source_y_stride(32u);
    TTI_STALLWAIT(STALL_PACK, WAIT_SFPU);
    for (uint32_t head = 1u; head < heads; ++head) {
        TTI_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    TTI_PACR(0u, 0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u);
    wait_tensix(WAIT_PACK);
    TTI_SETRWC(0u, 0u, 0u, 0u, 0u, 4u);
    TTI_SETADCXX(1u, 63u, 0u);
    TTI_SETADC(1u, 0u, 2u, 0u);
}

static void attention_rv32f(DramRef key_cache, DramRef value_cache, DramRef score_cache, uint32_t query_head_first,
                            uint32_t kv_heads, uint32_t first, uint32_t count, uint32_t output_base,
                            uint32_t head_first, uint32_t heads) {
    volatile uint16_t *query = (volatile uint16_t *)TT_ATTENTION_QUERY_L1_ADDR;
    volatile uint16_t *sinks = (volatile uint16_t *)TT_ATTENTION_SINK_L1_ADDR;
    volatile uint16_t *output = (volatile uint16_t *)output_base;
    float maximum[TT_GPT_OSS_QUERY_GROUP];
    float denominator[TT_GPT_OSS_QUERY_GROUP];
    float reciprocal[TT_GPT_OSS_QUERY_GROUP];
    const bool spill_scores = count > TT_ATTENTION_SCORE_CHUNK_TOKENS;

    for (uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const uint32_t sink_base = query_head_first + kv_head * TT_GPT_OSS_QUERY_GROUP;
        const uint32_t local_base = kv_head * TT_GPT_OSS_QUERY_GROUP;

        attention_score_sfpu_setup();
        for (uint32_t local_head = 0; local_head < TT_GPT_OSS_QUERY_GROUP; ++local_head) {
            maximum[local_head] = bf16_to_f32(sinks[sink_base + local_head]);
        }
        const bool owned_pair = heads == 2u;
        attention_stage_query_sfpu(query, local_base + head_first, owned_pair);
        attention_unpack_setup(TT_ATTENTION_KEY_CHUNK_L1_ADDR);
        TTI_SETADCXX(1u, 1023u, 0u);
        pack_source_y_stride(64u);
        for (uint32_t chunk = 0; chunk < count; chunk += TT_ATTENTION_SCORE_CHUNK_TOKENS) {
            const uint32_t chunk_count = attention_score_chunk_count(chunk, count);

            attention_load_cache_chunk(key_cache, first + chunk, chunk_count, kv_head, kv_heads,
                                       TT_ATTENTION_KEY_CHUNK_L1_ADDR);
            math_pack_config(DataFormat::BF16, DataFormat::BF16, DataFormat::FP32, true);
            attention_unpack_keys(0u);
            uint32_t batch = 0u;
            for (; batch + attention_key_batch < chunk_count; batch += attention_key_batch) {
                // Unpack into the opposite bank while consuming the preceding batch.
                attention_unpack_keys(batch + attention_key_batch);
                attention_stage_keys_sfpu();
                attention_scores_sfpu_batch((volatile float *)attention_score_l1_token_addr(batch), maximum,
                                            attention_key_batch, owned_pair);
            }
            // Consume the final bank without submitting a read beyond this L1 chunk.
            attention_stage_keys_sfpu();
            attention_scores_sfpu_batch((volatile float *)attention_score_l1_token_addr(batch), maximum,
                                        chunk_count - batch, owned_pair);
            TTI_SETADC(1u, 0u, 2u, 0u); // restart the next chunk and leave source Z clear
            if (spill_scores) {
                noc_write_from_l1(score_cache.coord, score_cache.addr + attention_score_offset(chunk),
                                  TT_ATTENTION_SCORE_L1_ADDR, chunk_count * TT_ATTENTION_SCORE_STRIDE_BYTES);
            }
        }

        for (uint32_t local_head = 0; local_head < TT_GPT_OSS_QUERY_GROUP; ++local_head) {
            denominator[local_head] =
                rv32f_exp_approx(bf16_to_f32(sinks[sink_base + local_head]) - maximum[local_head]);
        }
        attention_fill_maximum(maximum);
        for (uint32_t chunk = 0; chunk < count; chunk += TT_ATTENTION_SCORE_CHUNK_TOKENS) {
            const uint32_t chunk_count = attention_score_chunk_count(chunk, count);

            if (spill_scores) {
                noc_read_to_l1(score_cache.coord, score_cache.addr + attention_score_offset(chunk),
                               TT_ATTENTION_SCORE_L1_ADDR, chunk_count * TT_ATTENTION_SCORE_STRIDE_BYTES);
            }
            sfpu_attention_softmax(TT_ATTENTION_SCORE_L1_ADDR, TT_ATTENTION_MAXIMUM_L1_ADDR, chunk_count);
            FENCE(); // BRISC consumes the packer's scores and partial sums.
            const uint32_t reduced = chunk_count >= 16u ? chunk_count / 8u * 8u : 0u;
            if (reduced != 0u) {
                const volatile float *partials = (const volatile float *)TT_ATTENTION_DOT_L1_ADDR;
                for (uint32_t token = 0u; token < 8u; ++token) {
                    for (uint32_t head = 0u; head < TT_GPT_OSS_QUERY_GROUP; head += 2u) {
                        const float a = partials[token * TT_GPT_OSS_QUERY_GROUP + head];
                        const float b = partials[token * TT_GPT_OSS_QUERY_GROUP + head + 1u];
                        denominator[head] += a;
                        denominator[head + 1u] += b;
                    }
                }
            }
            for (uint32_t token = reduced; token < chunk_count; ++token) {
                volatile float *token_scores = (volatile float *)attention_score_l1_token_addr(token);

                for (uint32_t local_head = 0; local_head < TT_GPT_OSS_QUERY_GROUP; local_head += 2u) {
                    const float a = token_scores[local_head];
                    const float b = token_scores[local_head + 1u];
                    denominator[local_head] += a;
                    denominator[local_head + 1u] += b;
                }
            }
            if (spill_scores) {
                noc_write_from_l1(score_cache.coord, score_cache.addr + attention_score_offset(chunk),
                                  TT_ATTENTION_SCORE_L1_ADDR, chunk_count * TT_ATTENTION_SCORE_STRIDE_BYTES);
            }
        }

        for (uint32_t local_head = 0; local_head < TT_GPT_OSS_QUERY_GROUP; ++local_head) {
            reciprocal[local_head] = rv32f_recip_approx(denominator[local_head]);
        }
        attention_accumulate_begin_sfpu();
        setup_bf16_unpack0();
        math_pack_config(DataFormat::TF32, DataFormat::FP32, DataFormat::FP32, true);
        attention_unpack_setup(TT_ATTENTION_VALUE_CHUNK_L1_ADDR);
        for (uint32_t chunk = 0; chunk < count; chunk += TT_ATTENTION_SCORE_CHUNK_TOKENS) {
            const uint32_t chunk_count = attention_score_chunk_count(chunk, count);

            if (spill_scores) {
                noc_read_to_l1(score_cache.coord, score_cache.addr + attention_score_offset(chunk),
                               TT_ATTENTION_SCORE_L1_ADDR, chunk_count * TT_ATTENTION_SCORE_STRIDE_BYTES);
                FENCE(); // RV32F reads the weights in attention_accumulate_value_sfpu.
            }
            if (chunk != 0u) {
                // BRISC can outrun queued unpacks; finish reading the old value chunk
                // before NoC overwrites it. The final chunk is drained by the output pack.
                wait_tensix(WAIT_UNPACK0);
            }
            attention_load_cache_chunk(value_cache, first + chunk, chunk_count, kv_head, kv_heads,
                                       TT_ATTENTION_VALUE_CHUNK_L1_ADDR);
            if (heads == 2u) {
                attention_accumulate_chunk<2u>(chunk_count, head_first);
            } else {
                attention_accumulate_chunk<8u>(chunk_count, head_first);
            }
            TTI_SETADC(1u, 0u, 2u, 0u); // restart the next chunk and leave source Z clear
        }
        attention_accumulate_end_sfpu((uint32_t)(output + kv_head * heads * TT_GPT_OSS_HEAD_SIZE),
                                      reciprocal + head_first, heads);
    }
}

static DramRef make_dram_ref(uint32_t coord_addr, uint32_t src_addr) {
    DramRef ref;

    ref.coord = PHYS_RD32(coord_addr);
    ref.addr = PHYS_RD32(src_addr);
    return ref;
}

static void write_cache_vector(DramRef cache, uint32_t position, uint32_t kv_heads, uint32_t l1_base) {
    const uint32_t bytes = kv_heads * TT_GPT_OSS_HEAD_SIZE * sizeof(uint16_t);

    noc_write_from_l1(cache.coord, cache.addr + position * bytes, l1_base, bytes);
}

// Each sender always increments its own arrival slot on the destination tile.
// Batches contain at most seven commands, below the 16-entry FIFO capacity.
static_assert(TT_MAX_SHARDS - 1u <= 16u);
static void run_attention_output_projection(DramRef weight, DramRef bias, uint32_t shard, uint32_t shard_count) {
    const uint32_t first_block = tt_residual_block_bound(shard, shard_count);
    const uint32_t block_count = tt_residual_block_bound(shard + 1u, shard_count) - first_block;
    const uint32_t offset = first_block * TT_MVMUL_TILE_DIM * sizeof(float);

    bias.addr += offset;
    mvmul_bf16_bias_dram<DataFormat::FP32>(weight, bias, TT_GPT_OSS_QUERY_WIDTH, block_count,
                                           TT_MOE_PROJECTED_L1_ADDR + offset, TT_ATTENTION_OUT_L1_ADDR);
}

static void attention_phase(volatile uint32_t *desc, uint32_t position, uint32_t first, uint32_t count) {
    const uint32_t shard = PHYS_RD32(TT_L1_SHARD_INDEX_ADDR);
    const uint32_t shard_count = PHYS_RD32(TT_L1_SHARD_COUNT_ADDR);
    const uint32_t q_start = tt_attention_block_bound(TT_QUERY_ROW_BLOCKS, shard, shard_count);
    const uint32_t q_end = tt_attention_block_bound(TT_QUERY_ROW_BLOCKS, shard + 1u, shard_count);
    const uint32_t kv_start = tt_attention_block_bound(TT_KV_ROW_BLOCKS, shard, shard_count);
    const uint32_t kv_end = tt_attention_block_bound(TT_KV_ROW_BLOCKS, shard + 1u, shard_count);
    const uint32_t query_head_first = q_start * TT_MVMUL_TILE_DIM / TT_GPT_OSS_HEAD_SIZE;
    const uint32_t query_heads = (q_end - q_start) * TT_MVMUL_TILE_DIM / TT_GPT_OSS_HEAD_SIZE;
    const uint32_t kv_group = shard_count > TT_GPT_OSS_KV_HEADS ? shard_count / TT_GPT_OSS_KV_HEADS : 1u;
    const uint32_t kv_heads = shard_count == 1u ? TT_GPT_OSS_KV_HEADS : 1u;
    const uint32_t piece = shard % kv_group;
    const uint32_t group_first = (PHYS_RD32(TT_L1_PEER_INDEX_ADDR) / kv_group) * kv_group;
    const uint32_t query_slice = query_heads * TT_GPT_OSS_HEAD_SIZE * sizeof(uint16_t);
    const uint32_t bf16_block_bytes = TT_MVMUL_TILE_DIM * sizeof(uint16_t);
    const uint32_t f32_block_bytes = TT_MVMUL_TILE_DIM * sizeof(float);
    DramRef q_weight = desc_dram_ref(desc, TT_LAYER_REF_QUERY_WEIGHT);
    DramRef q_bias = desc_dram_ref(desc, TT_LAYER_REF_QUERY_BIAS);
    DramRef k_weight = desc_dram_ref(desc, TT_LAYER_REF_KEY_WEIGHT);
    DramRef k_bias = desc_dram_ref(desc, TT_LAYER_REF_KEY_BIAS);
    DramRef v_weight = desc_dram_ref(desc, TT_LAYER_REF_VALUE_WEIGHT);
    DramRef v_bias = desc_dram_ref(desc, TT_LAYER_REF_VALUE_BIAS);
    DramRef sinks = desc_dram_ref(desc, TT_LAYER_REF_ATTENTION_SINKS);
    DramRef key_cache = desc_dram_ref(desc, TT_LAYER_REF_ATTENTION_KEY_CACHE);
    DramRef value_cache = desc_dram_ref(desc, TT_LAYER_REF_ATTENTION_VALUE_CACHE);
    DramRef score_cache = make_dram_ref(TT_L1_ATTENTION_SCORE_COORD_ADDR, TT_L1_ATTENTION_SCORE_SRC_ADDR);

    q_bias.addr += q_start * f32_block_bytes;
    k_bias.addr += kv_start * f32_block_bytes;
    v_bias.addr += kv_start * bf16_block_bytes;

    mvmul_bf16_bias_dram<DataFormat::FP32>(q_weight, q_bias, TT_GPT_OSS_DIM, q_end - q_start,
                                           TT_ATTENTION_QUERY_F32_L1_ADDR, TT_MVMUL_B_L1_ADDR);
    profile_stage(TT_PROFILE_QKV);
    sfpu_rope(TT_ATTENTION_QUERY_F32_L1_ADDR, TT_ATTENTION_QUERY_L1_ADDR + piece * query_slice, query_heads);
    if (kv_group > 1u) {
        allgather_group_range(TT_ATTENTION_QUERY_L1_ADDR, piece * query_slice, query_slice, group_first, kv_group);
    }

    profile_stage(TT_PROFILE_ROPE);
    mvmul_bf16_bias_dram<DataFormat::FP32>(k_weight, k_bias, TT_GPT_OSS_DIM, kv_end - kv_start,
                                           TT_ATTENTION_KEY_F32_L1_ADDR + piece * TT_MVMUL_TILE_DIM * sizeof(float),
                                           TT_MVMUL_B_L1_ADDR);

    profile_stage(TT_PROFILE_QKV);
    mvmul_bf16_bias_dram<DataFormat::BF16>(v_weight, v_bias, TT_GPT_OSS_DIM, kv_end - kv_start,
                                           TT_ATTENTION_VALUE_L1_ADDR + piece * TT_MVMUL_TILE_DIM * sizeof(uint16_t),
                                           TT_MVMUL_B_L1_ADDR);

    profile_stage(TT_PROFILE_QKV);
    if (kv_group > 1u) {
        allgather_group_range(TT_ATTENTION_KEY_F32_L1_ADDR, piece * TT_MVMUL_TILE_DIM * sizeof(float),
                              TT_MVMUL_TILE_DIM * sizeof(float), group_first, kv_group);
        allgather_group_range(TT_ATTENTION_VALUE_L1_ADDR, piece * TT_MVMUL_TILE_DIM * sizeof(uint16_t),
                              TT_MVMUL_TILE_DIM * sizeof(uint16_t), group_first, kv_group);
    }
    sfpu_rope(TT_ATTENTION_KEY_F32_L1_ADDR, TT_ATTENTION_KEY_L1_ADDR, kv_heads);
    profile_stage(TT_PROFILE_ROPE);
    if (piece == 0u) {
        write_cache_vector(key_cache, position, kv_heads, TT_ATTENTION_KEY_L1_ADDR);
        write_cache_vector(value_cache, position, kv_heads, TT_ATTENTION_VALUE_L1_ADDR);
    }
    if (kv_group > 1u) {
        allgather_group_range(0u, 0u, 0u, group_first, kv_group);
    }
    noc_read_to_l1(sinks.coord, sinks.addr, TT_ATTENTION_SINK_L1_ADDR, TT_GPT_OSS_QUERY_HEADS * sizeof(uint16_t));
    FENCE(); // RV32F reads the sinks in attention_rv32f.
    profile_stage(TT_PROFILE_KVCACHE);

    const uint32_t attended_slice = query_heads * TT_GPT_OSS_HEAD_SIZE * sizeof(uint16_t);

    const uint32_t group_head_first = query_head_first / TT_GPT_OSS_QUERY_GROUP * TT_GPT_OSS_QUERY_GROUP;
    attention_rv32f(key_cache, value_cache, score_cache, group_head_first, kv_heads, first, count,
                    TT_ATTENTION_OUT_L1_ADDR + shard * attended_slice, query_head_first - group_head_first,
                    TT_GPT_OSS_QUERY_GROUP / kv_group);
    allgather_range(TT_ATTENTION_OUT_L1_ADDR, shard * attended_slice, attended_slice);
    const uint32_t peers = PHYS_RD32(TT_L1_PEER_COUNT_ADDR);
    const uint32_t chip_first = shard - shard % peers;
    exchange_across_chips(TT_ATTENTION_OUT_L1_ADDR, chip_first * attended_slice, peers * attended_slice, 0u, 1u);
    profile_stage(TT_PROFILE_ATTENTION);
}

static uint32_t attention_first_token(uint32_t layer_index, uint32_t position) {
    const uint32_t sliding_window = 128u;
    uint32_t first = 0u;

    if ((layer_index & 1u) == 0u && position + 1u > sliding_window) {
        first = position + 1u - sliding_window;
    }
    return first;
}

static void layer(volatile uint32_t *desc, uint32_t position, uint32_t attention_first, uint32_t attention_count) {
    DramRef attention_norm = desc_dram_ref(desc, TT_LAYER_REF_ATTENTION_NORM);
    DramRef post_norm = desc_dram_ref(desc, TT_LAYER_REF_POST_ATTENTION_NORM);
    DramRef o_weight = desc_dram_ref(desc, TT_LAYER_REF_ATTENTION_OUTPUT_WEIGHT);
    DramRef o_bias = desc_dram_ref(desc, TT_LAYER_REF_ATTENTION_OUTPUT_BIAS);
    const uint32_t shard = PHYS_RD32(TT_L1_SHARD_INDEX_ADDR);
    const uint32_t shard_count = PHYS_RD32(TT_L1_SHARD_COUNT_ADDR);

    rms_norm_to_bf16_vector((volatile float *)TT_MOE_RESIDUAL_L1_ADDR, attention_norm, TT_MVMUL_B_L1_ADDR);
    profile_stage(TT_PROFILE_NORM);
    attention_phase(desc, position, attention_first, attention_count);

    run_attention_output_projection(o_weight, o_bias, shard, shard_count);
    profile_stage(TT_PROFILE_PROJECT);
    moe(post_norm, desc);
}

static void layers(uint32_t position, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        volatile uint32_t *desc = (volatile uint32_t *)(TT_LAYER_DESC_L1_ADDR + i * TT_LAYER_DESC_BYTES);
        const uint32_t attention_first = attention_first_token(i, position);
        const uint32_t attention_count = position + 1u - attention_first;

        layer(desc, position, attention_first, attention_count);
        if (i + 1u < count) {
            copy_residual_slice(TT_MOE_RESIDUAL_L1_ADDR, TT_MOE_OUT_L1_ADDR, PHYS_RD32(TT_L1_SHARD_INDEX_ADDR),
                                PHYS_RD32(TT_L1_SHARD_COUNT_ADDR));
        }
    }
}

static uint32_t logits_block_first(uint32_t shard, uint32_t shard_count) {
    return tt_logits_block_bound(shard, shard_count);
}

static uint32_t logits_block_count(uint32_t shard, uint32_t shard_count) {
    return logits_block_first(shard + 1u, shard_count) - logits_block_first(shard, shard_count);
}

static void final_logits(uint32_t first_block, uint32_t block_count) {
    DramRef output_norm = make_dram_ref(TT_L1_TOKEN_OUTPUT_NORM_COORD_ADDR, TT_L1_TOKEN_OUTPUT_NORM_SRC_ADDR);
    DramRef output_weight = make_dram_ref(TT_L1_TOKEN_OUTPUT_WEIGHT_COORD_ADDR, TT_L1_TOKEN_OUTPUT_WEIGHT_SRC_ADDR);
    DramRef logits = make_dram_ref(TT_L1_TOKEN_LOGITS_COORD_ADDR, TT_L1_TOKEN_LOGITS_SRC_ADDR);
    const uint64_t logits_addr = (uint64_t(PHYS_RD32(TT_L1_TOKEN_LOGITS_MID_ADDR)) << 32) | logits.addr;
    const uint32_t chunk_rows = (TT_SWIGLU_GATE_L1_ADDR - TT_BF16_MVMUL_OUT_L1_ADDR) / sizeof(uint16_t);
    const uint32_t chunk_blocks = chunk_rows / TT_MVMUL_TILE_DIM;
    const uint32_t k_tiles = TT_GPT_OSS_DIM / TT_MVMUL_TILE_DIM;
    const uint32_t block_bytes = k_tiles * TT_MVMUL_BF16_A_TILE_BYTES;

    rms_norm_to_bf16_vector((volatile float *)TT_MOE_OUT_L1_ADDR, output_norm, TT_MVMUL_B_L1_ADDR);

    for (uint32_t block = first_block; block < first_block + block_count; block += chunk_blocks) {
        DramRef chunk_weight;
        const uint32_t remaining_blocks = first_block + block_count - block;
        const uint32_t current_blocks = remaining_blocks < chunk_blocks ? remaining_blocks : chunk_blocks;
        const uint32_t current_rows = current_blocks * TT_MVMUL_TILE_DIM;

        chunk_weight.coord = output_weight.coord;
        chunk_weight.addr = output_weight.addr + (block - first_block) * block_bytes;
        mvmul_bf16_dram(chunk_weight, TT_GPT_OSS_DIM, current_blocks);
        noc_write_from_l1_wide_async(logits.coord, logits_addr + block * TT_MVMUL_TILE_DIM * sizeof(uint16_t),
                                     TT_BF16_MVMUL_OUT_L1_ADDR, current_rows * sizeof(uint16_t));
        noc_wait_outstanding_id0();
    }
}

static void final_logits_shard() {
    const uint32_t shard = PHYS_RD32(TT_L1_SHARD_INDEX_ADDR);
    const uint32_t shard_count = PHYS_RD32(TT_L1_SHARD_COUNT_ADDR);
    const uint32_t first_block = logits_block_first(shard, shard_count);
    const uint32_t block_count = logits_block_count(shard, shard_count);

    final_logits(first_block, block_count);
}

static void load_token_embedding(uint32_t token) {
    volatile uint16_t *input = (volatile uint16_t *)TT_RMS_WEIGHT_L1_ADDR;
    volatile float *output = (volatile float *)TT_MOE_RESIDUAL_L1_ADDR;
    const uint32_t shard = PHYS_RD32(TT_L1_SHARD_INDEX_ADDR);
    const uint32_t shard_count = PHYS_RD32(TT_L1_SHARD_COUNT_ADDR);
    const uint32_t first = residual_first(shard, shard_count);
    const uint32_t elements = residual_elements(shard, shard_count);
    const uint32_t dst = TT_RMS_WEIGHT_L1_ADDR + first * sizeof(uint16_t);
    if (shard_count > TT_MAX_SHARDS) {
        const uint64_t base =
            uint64_t(PHYS_RD32(TT_L1_EMBEDDING_DMA_ADDR)) | (uint64_t(PHYS_RD32(TT_L1_EMBEDDING_DMA_ADDR + 4u)) << 32);
        const uint64_t address = base + (uint64_t(token) * TT_GPT_OSS_DIM + first) * sizeof(uint16_t);
        noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_TARG_ADDR_MID), uint32_t(address >> 32));
        noc_read_to_l1_async(19u | (24u << 6), uint32_t(address), dst, elements * sizeof(uint16_t));
        noc_wr(noc_cmd_offset(NOC_READ_CMD_BUF, NOC_TARG_ADDR_MID), 0u);
        noc_wait_outstanding_id0();
    } else {
        const uint32_t rows_per_stripe = TT_GPT_OSS_VOCAB / shard_count;
        const uint32_t stripe = token / rows_per_stripe;
        const uint32_t ref_addr = TT_L1_EMBEDDING_STRIPE_ADDR + stripe * 8u;
        const DramRef embeddings = make_dram_ref(ref_addr, ref_addr + 4u);
        const uint32_t offset = (token % rows_per_stripe) * TT_GPT_OSS_DIM * sizeof(uint16_t);
        noc_read_to_l1(embeddings.coord, embeddings.addr + offset + first * sizeof(uint16_t), dst,
                       elements * sizeof(uint16_t));
    }
    FENCE(); // RV32F widens the embedding below.
    for (uint32_t i = first; i < first + elements; ++i) {
        output[i] = bf16_to_f32(input[i]);
    }
}

static void token(uint32_t token_index, uint32_t position, uint32_t layer_count, uint32_t flags) {
    load_token_embedding(token_index);
    profile_stage(TT_PROFILE_EMBED);
    layers(position, layer_count);
    if (flags & TT_TOKEN_FLAG_LOGITS) {
        final_logits_shard();
        profile_stage(TT_PROFILE_LOGITS);
    }
}

extern "C" void brisc_main() {
    hw_init();
    waypoint(0); // unused, but easy to re-add if we need it
    for (;;) {
        while (!PHYS_RD32(TT_L1_GO_ADDR)) {
            FENCE();
        }

        uint32_t command = PHYS_RD32(TT_L1_COMMAND_ADDR);
        if (command == TT_COMMAND_TOKEN) {
            const uint32_t token_index = PHYS_RD32(TT_L1_TOKEN_INDEX_ADDR);
            const uint32_t position = PHYS_RD32(TT_L1_LAYER_POSITION_ADDR);
            const uint32_t layer_count = PHYS_RD32(TT_L1_LAYER_COUNT_ADDR);
            const uint32_t flags = PHYS_RD32(TT_L1_TOKEN_FLAGS_ADDR);

            profile_begin();
            token(token_index, position, layer_count, flags);
            // Clear before the barrier: the host can post the next token as soon as
            // shard 0 reports completion, while another shard is still leaving it.
            PHYS_WR32(TT_L1_GO_ADDR, 0);
            FENCE();
            allgather_barrier();
            exchange_across_chips(0u, 0u, 0u, 0u, 1u);
            profile_stage(TT_PROFILE_COMPLETION);
            if (PHYS_RD32(TT_L1_SHARD_INDEX_ADDR) == 0u) {
                PHYS_WR32(TT_L1_SENTINEL_ADDR, TT_SENTINEL_TOKEN);
            }
        } else {
            PHYS_WR32(TT_L1_GO_ADDR, 0);
            PHYS_WR32(TT_L1_SENTINEL_ADDR, 0xBAD00000u | command);
        }
    }
}
