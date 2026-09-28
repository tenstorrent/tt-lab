// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// General firmware helpers/constants that are minimally project-specific and likely to be useful for
// arbitrary device-side code (e.g. other workloads or probes)
#pragma once

#define PHYS_RD32(addr) (*(volatile uint32_t *)(addr))
#define PHYS_WR32(addr, data)                                                                                          \
    do {                                                                                                               \
        *(volatile uint32_t *)(addr) = data;                                                                           \
    } while (0)

#define FENCE() __asm__ volatile("fence rw, rw" ::: "memory")

#define NOC0_BASE 0xFFB20000u
#define NOC_OVERLAY_BASE 0xFFB40000u
#define TENSIX_REGFILE_BASE 0xFFE00000u
#define TENSIX_INST_BASE 0xFFE40000u
#define TENSIX_CFG_BASE 0xFFEF0000u

#define NOC_TARG_ADDR_LO 0x000u
#define NOC_TARG_ADDR_MID 0x004u
#define NOC_TARG_ADDR_HI 0x008u
#define NOC_RET_ADDR_LO 0x00Cu
#define NOC_RET_ADDR_MID 0x010u
#define NOC_RET_ADDR_HI 0x014u
#define NOC_PACKET_TAG 0x018u
#define NOC_CTRL 0x01Cu
#define NOC_AT_LEN_BE 0x020u
#define NOC_AT_DATA 0x028u
#define NOC_CMD_CTRL 0x040u
#define NOC_ID_LOGICAL 0x148u

#define NOC_CMD_BUF_OFFSET 0x800u

#define STALL_PACK (1 << 2)
#define STALL_THCON (1 << 5)
#define STALL_SFPU (1 << 8)
#define WAIT_UNPACK0 (1 << 1)
#define WAIT_PACK (1 << 3)
#define WAIT_FPU (1 << 4)
#define WAIT_SFPU (1 << 11)

enum class DataFormat : uint32_t {
    FP32 = 0u,
    TF32 = 4u,
    BF16 = 5u,
    BFP8 = 6u
};

enum class L1Base : uint32_t {
    Pack = 69,
    Unpack0 = 76,
    Unpack1 = 124
};

static inline void tensix_instruction(uint32_t instruction) {
    // Replace with __builtin_rvtt_submit(uint32_t opcode) when SFPI supports it:
    // https://github.com/tenstorrent/tt-metal/issues/56906
    // Select after inlining/constant propagation; runtime operands use the instruction port.
    if (__builtin_constant_p(instruction)) {
        INSTRUCTION_WORD(instruction);
    } else {
        PHYS_WR32(TENSIX_INST_BASE, instruction);
    }
}

static inline uint32_t pack_u16(uint32_t low, uint32_t high) {
    if (__builtin_constant_p(low) && __builtin_constant_p(high)) {
        return (low & 0xFFFFu) | (high << 16);
    }
    // BH supports PACK, but not the full Zbkb extension required by GCC's pattern.
    // SFPI support for the BH subset would let us replace this inline assembly:
    // https://github.com/tenstorrent/tt-metal/issues/56904
    uint32_t result;
    __asm__(".insn r 0x33, 4, 0x04, %0, %1, %2" : "=r"(result) : "r"(low), "r"(high));
    return result;
}
