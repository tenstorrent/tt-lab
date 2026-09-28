// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

// Firmware text occupies L1 below this; the control block and layer descriptors follow.
#define TT_L1_SENTINEL_ADDR 0x6000u
#define TT_L1_GO_ADDR 0x6004u
#define TT_L1_COMMAND_ADDR 0x6008u
#define TT_L1_TOKEN_FLAGS_ADDR 0x600Cu
#define TT_L1_TOKEN_INDEX_ADDR 0x6010u
#define TT_L1_LAYER_POSITION_ADDR 0x6014u
#define TT_L1_LAYER_COUNT_ADDR 0x6018u
#define TT_L1_PEER_INDEX_ADDR 0x6024u
#define TT_L1_PEER_COUNT_ADDR 0x6028u
#define TT_PCIE_STAGE_ADDR 0xF0000u
#define TT_PCIE_STAGE_BYTES 16384u
#define TT_PCIE_EXCHANGE_ADDR (TT_PCIE_STAGE_ADDR + TT_PCIE_STAGE_BYTES)

// Courier p sends to chip c+p+1 and receives from c-p-1, modulo chip count.
// Incoming payload is separate from live compute buffers until locally rebroadcast.
struct TtPcieExchange {
    uint32_t send_lo, send_hi;
    uint32_t source_lo, source_hi;
    uint32_t coord, sequence;
    uint32_t reserved[2];
    uint32_t in_offset, in_bytes, in_sequence, ready;
    uint32_t out_offset, out_bytes, out_sequence, out_ready;
};
static_assert(sizeof(TtPcieExchange) == 64u);
static_assert(offsetof(TtPcieExchange, out_offset) - offsetof(TtPcieExchange, in_offset) == 16u);
static_assert(offsetof(TtPcieExchange, out_sequence) - offsetof(TtPcieExchange, in_sequence) == 16u);
static_assert(offsetof(TtPcieExchange, out_ready) - offsetof(TtPcieExchange, ready) == 16u);
static_assert(TT_PCIE_EXCHANGE_ADDR + sizeof(TtPcieExchange) <= 1024u * 1024u);
#define TT_L1_SHARD_INDEX_ADDR 0x6044u
#define TT_L1_WAYPOINT_ADDR 0x6048u
#define TT_L1_SHARD_COUNT_ADDR 0x604Cu
#define TT_MAX_GLOBAL_SHARDS 32u
#define TT_MAX_SHARDS 8u                    // Chip-local peers; global shard identity is separate.
#define TT_L1_SHARD_COORD_ADDR 0x6050u      // TT_MAX_SHARDS words
#define TT_L1_EMBEDDING_STRIPE_ADDR 0x6080u // TT_MAX_SHARDS coordinate/address pairs
#define TT_L1_PROFILE_MARK_ADDR 0x60C8u
#define TT_L1_PROFILE_STAGE_ADDR 0x60CCu
#define TT_PROFILE_STAGES 13u
#define TT_PROFILE_EMBED 0u
#define TT_PROFILE_NORM 1u
#define TT_PROFILE_QKV 2u
#define TT_PROFILE_ROPE 3u
#define TT_PROFILE_KVCACHE 4u
#define TT_PROFILE_ATTENTION 5u
#define TT_PROFILE_PROJECT 6u
#define TT_PROFILE_ROUTER 7u
#define TT_PROFILE_EXPERT 8u
#define TT_PROFILE_SWIGLU 9u
#define TT_PROFILE_RESIDUAL 10u
#define TT_PROFILE_LOGITS 11u
#define TT_PROFILE_COMPLETION 12u
#define TT_L1_ATOMIC_RESULT_ADDR 0x614Cu
#define TT_L1_EXPERT_COUNT_ADDR 0x6150u
#define TT_L1_EMBEDDING_DMA_ADDR 0x61A0u    // 64-bit host address, used with multiple chips
#define TT_L1_ALLGATHER_ARRIVE_ADDR 0x6160u // TT_MAX_SHARDS words, indexed by sender
#define TT_L1_ALLGATHER_TARGET_ADDR 0x6180u // TT_MAX_SHARDS words, indexed by sender
#define TT_L1_ATTENTION_SCORE_COORD_ADDR 0x601Cu
#define TT_L1_ATTENTION_SCORE_SRC_ADDR 0x6020u

#define TT_COMMAND_TOKEN 19u

#define TT_SENTINEL_TOKEN 0x544F4B4Eu

#define TT_L1_TOKEN_OUTPUT_NORM_COORD_ADDR 0x602Cu
#define TT_L1_TOKEN_OUTPUT_NORM_SRC_ADDR 0x6030u
#define TT_L1_TOKEN_OUTPUT_WEIGHT_COORD_ADDR 0x6034u
#define TT_L1_TOKEN_OUTPUT_WEIGHT_SRC_ADDR 0x6038u
#define TT_L1_TOKEN_LOGITS_COORD_ADDR 0x603Cu
#define TT_L1_TOKEN_LOGITS_SRC_ADDR 0x6040u
#define TT_L1_TOKEN_LOGITS_MID_ADDR 0x6158u
#define TT_TOKEN_FLAG_LOGITS 1u
#define TT_TOKEN_FLAG_PROFILE 2u

#define TT_LAYER_REF_ATTENTION_NORM 0u
#define TT_LAYER_REF_POST_ATTENTION_NORM 1u
#define TT_LAYER_REF_QUERY_WEIGHT 2u
#define TT_LAYER_REF_QUERY_BIAS 3u
#define TT_LAYER_REF_KEY_WEIGHT 4u
#define TT_LAYER_REF_KEY_BIAS 5u
#define TT_LAYER_REF_VALUE_WEIGHT 6u
#define TT_LAYER_REF_VALUE_BIAS 7u
#define TT_LAYER_REF_ATTENTION_OUTPUT_WEIGHT 8u
#define TT_LAYER_REF_ATTENTION_OUTPUT_BIAS 9u
#define TT_LAYER_REF_ATTENTION_SINKS 10u
#define TT_LAYER_REF_ROUTER_WEIGHT 11u
#define TT_LAYER_REF_ROUTER_BIAS 12u
#define TT_LAYER_REF_GATE_WEIGHT 13u
#define TT_LAYER_REF_GATE_BIAS 14u
#define TT_LAYER_REF_UP_WEIGHT 15u
#define TT_LAYER_REF_UP_BIAS 16u
#define TT_LAYER_REF_DOWN_WEIGHT 17u
#define TT_LAYER_REF_DOWN_BIAS 18u
#define TT_LAYER_REF_ATTENTION_KEY_CACHE 19u
#define TT_LAYER_REF_ATTENTION_VALUE_CACHE 20u

#define TT_LAYER_DESC_L1_ADDR 0x6300u

// Scratch buffers, ordered by address. Some ranges are reused by disjoint stages.
#define TT_MVMUL_B_L1_ADDR 0xC000u
#define TT_MOE_HIDDEN_L1_ADDR 0x18000u
#define TT_BF16_MVMUL_OUT_L1_ADDR 0x1E000u
#define TT_SWIGLU_GATE_L1_ADDR 0x20000u
#define TT_SWIGLU_UP_L1_ADDR (TT_SWIGLU_GATE_L1_ADDR + TT_SWIGLU_EXPERT_BYTES)
#define TT_SWIGLU_OUT_L1_ADDR (TT_SWIGLU_UP_L1_ADDR + TT_SWIGLU_EXPERT_BYTES)
#define TT_BFP8_STAGE_L1_ADDR (TT_SWIGLU_OUT_L1_ADDR + TT_SWIGLU_OUTPUT_BYTES)
// BF16 matvecs run outside expert evaluation; their larger staging area reuses SwiGLU scratch.
#define TT_BF16_STAGE_L1_ADDR TT_SWIGLU_GATE_L1_ADDR
#define TT_ATTENTION_QUERY_F32_L1_ADDR 0x88000u
#define TT_ATTENTION_QUERY_L1_ADDR 0x8C000u
#define TT_ATTENTION_KEY_F32_L1_ADDR 0x8E000u
#define TT_ATTENTION_KEY_L1_ADDR 0x8E800u
#define TT_ATTENTION_VALUE_L1_ADDR 0x8F000u
#define TT_ATTENTION_OUT_L1_ADDR 0x90000u
#define TT_ROPE_TABLE_ENTRIES TT_GPT_OSS_HEAD_SIZE
#define TT_ROPE_COS_L1_ADDR 0x93000u
#define TT_ROPE_SIN_L1_ADDR (TT_ROPE_COS_L1_ADDR + TT_ROPE_TABLE_ENTRIES * sizeof(float))
#define TT_MOE_OUT_L1_ADDR 0xA0000u
#define TT_MOE_RESIDUAL_L1_ADDR 0xA3000u
#define TT_MOE_PROJECTED_L1_ADDR 0xA6000u
#define TT_RMS_WEIGHT_L1_ADDR 0xA9000u
#define TT_MOE_DOWN4_L1_ADDR 0xB5000u
#define TT_ATTENTION_SINK_L1_ADDR 0xCD000u
#define TT_ATTENTION_DOT_L1_ADDR 0xCE000u
#define TT_ATTENTION_SCORE_L1_ADDR 0xD0000u
#define TT_RMS_PARTIAL_L1_ADDR 0xD8000u
#define TT_RMS_SCALAR_L1_ADDR 0xD8100u
#define TT_ATTENTION_MAXIMUM_L1_ADDR 0xD8200u

#define TT_MVMUL_TILE_DIM 16u
#define TT_MVMUL_A_TILE_BYTES 144u
#define TT_MVMUL_BF16_A_TILE_BYTES 512u
#define TT_MVMUL_B_STRIDE_BYTES (TT_MVMUL_TILE_DIM * sizeof(uint16_t))
#define TT_NOC_MAX_BULK_BYTES 16384u
#define TT_SWIGLU_TILE_BYTES 512u
#define TT_ATTENTION_SCORE_STRIDE_BYTES 32u
#define TT_ATTENTION_SCORE_CHUNK_TOKENS 256u
#define TT_ATTENTION_MAXIMUM_VALUES 64u
#define TT_ATTENTION_CACHE_ROW_BYTES (TT_GPT_OSS_HEAD_SIZE * sizeof(uint16_t))
#define TT_ATTENTION_KEY_CHUNK_L1_ADDR 0xE0000u
#define TT_ATTENTION_VALUE_CHUNK_L1_ADDR 0xE8000u
#define TT_RMS_PARTIALS 64u
#define TT_MOE_BLOCK_ELEMENTS 64u
#define TT_MOE_BLOCKS (TT_GPT_OSS_DIM / TT_MOE_BLOCK_ELEMENTS)
#define TT_MOE_DOWN4_EXPERT_BYTES (TT_MOE_BLOCKS * TT_SWIGLU_TILE_BYTES)

#define TT_GPT_OSS_20B_LAYERS 24u
#define TT_GPT_OSS_20B_EXPERTS 32u
#define TT_GPT_OSS_120B_LAYERS 36u
#define TT_GPT_OSS_120B_EXPERTS 128u
#define TT_GPT_OSS_MAX_LAYERS TT_GPT_OSS_120B_LAYERS
#define TT_GPT_OSS_MAX_EXPERTS TT_GPT_OSS_120B_EXPERTS
#define TT_GPT_OSS_VOCAB 201088u
#define TT_GPT_OSS_DIM 2880u
#define TT_GPT_OSS_QUERY_HEADS 64u
#define TT_GPT_OSS_KV_HEADS 8u
#define TT_GPT_OSS_HEAD_SIZE 64u
#define TT_GPT_OSS_QUERY_GROUP (TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS)
#define TT_GPT_OSS_QUERY_WIDTH (TT_GPT_OSS_QUERY_HEADS * TT_GPT_OSS_HEAD_SIZE)
#define TT_GPT_OSS_KV_WIDTH (TT_GPT_OSS_KV_HEADS * TT_GPT_OSS_HEAD_SIZE)
#define TT_MVMUL_K_TILES (TT_GPT_OSS_DIM / TT_MVMUL_TILE_DIM)
#define TT_QUERY_ROW_BLOCKS (TT_GPT_OSS_QUERY_WIDTH / TT_MVMUL_TILE_DIM)
#define TT_KV_ROW_BLOCKS (TT_GPT_OSS_KV_WIDTH / TT_MVMUL_TILE_DIM)
#define TT_DIM_ROW_BLOCKS (TT_GPT_OSS_DIM / TT_MVMUL_TILE_DIM)
#define TT_LOGITS_ROW_BLOCKS (TT_GPT_OSS_VOCAB / TT_MVMUL_TILE_DIM)
#define TT_SWIGLU_EXPERT_BYTES (TT_MVMUL_K_TILES * TT_SWIGLU_TILE_BYTES)
#define TT_SWIGLU_OUTPUT_BYTES (TT_MOE_BLOCKS * 128u)
#define TT_MVMUL_BFP4_TILE_EXP_BYTES 16u
#define TT_MVMUL_BFP4_TILE_DATUM_BYTES (TT_MVMUL_A_TILE_BYTES - TT_MVMUL_BFP4_TILE_EXP_BYTES)
#define TT_BFP4_STRIP_EXP_BYTES (TT_MVMUL_K_TILES * TT_MVMUL_BFP4_TILE_EXP_BYTES)
#define TT_BFP4_STRIP_DATUM_BYTES (TT_MVMUL_K_TILES * TT_MVMUL_BFP4_TILE_DATUM_BYTES)
#define TT_BFP4_STRIP_BYTES (TT_BFP4_STRIP_EXP_BYTES + TT_BFP4_STRIP_DATUM_BYTES)
#define TT_MVMUL_BFP8_TILE_EXP_BYTES 16u
#define TT_MVMUL_BFP8_TILE_DATUM_BYTES (TT_MVMUL_TILE_DIM * TT_MVMUL_TILE_DIM)
#define TT_BFP8_STRIP_EXP_BYTES (TT_MVMUL_K_TILES * TT_MVMUL_BFP8_TILE_EXP_BYTES)
#define TT_BFP8_STRIP_DATUM_BYTES (TT_MVMUL_K_TILES * TT_MVMUL_BFP8_TILE_DATUM_BYTES)
#define TT_BFP8_STRIP_BYTES (TT_BFP8_STRIP_EXP_BYTES + TT_BFP8_STRIP_DATUM_BYTES)
#define TT_BFP8_STRIP_TAIL_BYTES (TT_BFP8_STRIP_BYTES - 2u * TT_NOC_MAX_BULK_BYTES)
#define TT_LAYER_DESC_WORDS 48u
#define TT_LAYER_DESC_BYTES (TT_LAYER_DESC_WORDS * 4u)

constexpr uint32_t tt_router_shards(uint32_t experts, uint32_t peers) {
    return experts == TT_GPT_OSS_120B_EXPERTS ? peers : 1u;
}

// Q and K/V row blocks divide evenly for every supported shard count.
constexpr uint32_t tt_attention_block_bound(uint32_t blocks, uint32_t shard, uint32_t shards) {
    return (blocks / shards) * shard;
}

constexpr uint32_t tt_logits_block_bound(uint32_t shard, uint32_t shards) {
    return TT_LOGITS_ROW_BLOCKS * shard / shards;
}
static_assert(TT_GPT_OSS_120B_EXPERTS % (TT_MAX_SHARDS * TT_MVMUL_TILE_DIM) == 0u);
static_assert(TT_L1_ATOMIC_RESULT_ADDR + sizeof(uint32_t) <= TT_L1_EXPERT_COUNT_ADDR);
static_assert(TT_L1_EXPERT_COUNT_ADDR + sizeof(uint32_t) <= TT_L1_TOKEN_LOGITS_MID_ADDR);

// Gate/up use single row blocks (22/23 per shard); SwiGLU handles partial groups.
constexpr uint32_t tt_hidden_block_bound(uint32_t index, uint32_t shard_count) {
    return TT_DIM_ROW_BLOCKS * index / shard_count;
}

// Keep whole SFPU groups on one chip; 32-way stripes need single row blocks for balance.
constexpr uint32_t tt_residual_block_bound(uint32_t index, uint32_t shard_count) {
    if (shard_count > TT_MAX_SHARDS) {
        return tt_hidden_block_bound(index, shard_count);
    }
    return ((TT_DIM_ROW_BLOCKS / 4u) * index / shard_count) * 4u;
}

static_assert(TT_QUERY_ROW_BLOCKS % TT_MAX_GLOBAL_SHARDS == 0u);
static_assert(TT_KV_ROW_BLOCKS % TT_MAX_GLOBAL_SHARDS == 0u);
static_assert(TT_GPT_OSS_VOCAB % TT_MAX_GLOBAL_SHARDS == 0u);

static_assert(TT_DIM_ROW_BLOCKS % 4u == 0u, "residual must contain whole SFPU groups");
static_assert((TT_GPT_OSS_DIM % TT_MAX_SHARDS) == 0u, "max sharding must evenly divide the model dimension");
static_assert((TT_QUERY_ROW_BLOCKS % TT_MAX_SHARDS) == 0u, "max sharding must evenly divide query row blocks");
static_assert((TT_KV_ROW_BLOCKS % TT_MAX_SHARDS) == 0u, "max sharding must evenly divide KV row blocks");
static_assert((TT_GPT_OSS_VOCAB % TT_MVMUL_TILE_DIM) == 0u, "vocab must evenly divide into matvec row blocks");
static_assert((TT_LOGITS_ROW_BLOCKS % TT_MAX_SHARDS) == 0u, "max sharding must evenly divide logits row blocks");
static_assert((TT_GPT_OSS_VOCAB % TT_MAX_SHARDS) == 0u, "max sharding must evenly divide embedding rows");
static_assert(TT_L1_TOKEN_LOGITS_MID_ADDR + sizeof(uint32_t) <= TT_L1_ALLGATHER_ARRIVE_ADDR);
static_assert(TT_L1_ALLGATHER_ARRIVE_ADDR + TT_MAX_SHARDS * sizeof(uint32_t) <= TT_L1_ALLGATHER_TARGET_ADDR);
static_assert(TT_L1_ALLGATHER_TARGET_ADDR + TT_MAX_SHARDS * sizeof(uint32_t) <= TT_LAYER_DESC_L1_ADDR);
static_assert(TT_L1_ALLGATHER_TARGET_ADDR + TT_MAX_SHARDS * sizeof(uint32_t) <= TT_L1_EMBEDDING_DMA_ADDR);
static_assert(TT_L1_EMBEDDING_DMA_ADDR + sizeof(uint64_t) <= TT_LAYER_DESC_L1_ADDR);
static_assert(TT_RMS_PARTIAL_L1_ADDR + TT_RMS_PARTIALS * sizeof(float) <= TT_RMS_SCALAR_L1_ADDR);
static_assert(TT_RMS_SCALAR_L1_ADDR + TT_MAX_SHARDS * sizeof(float) <= TT_ATTENTION_MAXIMUM_L1_ADDR);
static_assert(TT_LAYER_DESC_L1_ADDR + TT_GPT_OSS_MAX_LAYERS * TT_LAYER_DESC_BYTES <= TT_MVMUL_B_L1_ADDR,
              "layer descriptors overlap matvec input");
static_assert(TT_L1_SHARD_COORD_ADDR + TT_MAX_SHARDS * 4u <= TT_L1_EMBEDDING_STRIPE_ADDR,
              "shard coordinates overlap embedding references");
static_assert(TT_L1_EMBEDDING_STRIPE_ADDR + TT_MAX_SHARDS * 8u <= TT_L1_PROFILE_MARK_ADDR);
static_assert(TT_L1_PROFILE_MARK_ADDR + sizeof(uint32_t) <= TT_L1_PROFILE_STAGE_ADDR);
static_assert(TT_L1_PROFILE_STAGE_ADDR + TT_PROFILE_STAGES * sizeof(uint32_t) <= TT_L1_ATOMIC_RESULT_ADDR);
static_assert(TT_ROPE_SIN_L1_ADDR + TT_ROPE_TABLE_ENTRIES * sizeof(float) <= TT_MOE_OUT_L1_ADDR,
              "RoPE tables overlap attention scratch");
static_assert(TT_ATTENTION_VALUE_CHUNK_L1_ADDR + TT_ATTENTION_SCORE_CHUNK_TOKENS * TT_ATTENTION_CACHE_ROW_BYTES <=
                  TT_PCIE_STAGE_ADDR,
              "attention scratch overlaps PCIe staging");
