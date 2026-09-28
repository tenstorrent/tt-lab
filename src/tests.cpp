// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "constants.h"
#include "requant.h"
#include "tensor.h"
#include "tt_backend.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

static_assert(TtMeshConfig{1}.valid() && TtMeshConfig{1}.chip_count() == 1 && TtMeshConfig{1}.local_shard_count() == 1);
static_assert(TtMeshConfig{8}.valid() && TtMeshConfig{8}.chip_count() == 1 && TtMeshConfig{8}.local_shard_count() == 8);
static_assert(TtMeshConfig{32}.valid() && TtMeshConfig{32}.chip_count() == 4 &&
              TtMeshConfig{32}.local_shard_count() == 8);
static_assert(!TtMeshConfig{0}.valid() && !TtMeshConfig{2}.valid() && !TtMeshConfig{4}.valid() &&
              !TtMeshConfig{16}.valid() && !TtMeshConfig{64}.valid());

void check_close(float actual, float expected, float tolerance, const char *message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(message) + ": got " + std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
    }
}

GgufTensor tensor(std::string_view name, std::vector<uint64_t> shape, uint32_t type, const void *data) {
    return {name, std::move(shape), type, 0, (const std::byte *)data};
}

void test_float_formats() {
    check_close(fp16_to_fp32(0x3c00), 1.0f, 0.0f, "fp16 one");
    check_close(fp16_to_fp32(0xbc00), -1.0f, 0.0f, "fp16 minus one");
    check_close(fp16_to_fp32(0x0001), std::ldexp(1.0f, -24), 0.0f, "fp16 subnormal");
    check_close(mxfp4_scale(127), 0.5f, 0.0f, "E8M0 scale");
    if (fp32_to_bf16(1.0f) != 0x3f80) {
        throw std::runtime_error("BF16 one conversion failed");
    }
    if (fp32_to_bf16(1.00390625f) != 0x3f80) {
        throw std::runtime_error("BF16 ties-to-even conversion failed");
    }
    if (fp32_to_bf16(1.005f) != 0x3f81) {
        throw std::runtime_error("BF16 round-up conversion failed");
    }
}

void test_shard_bounds() {
    for (uint32_t shards : {1u, 8u, 32u}) {
        for (uint32_t shard = 0; shard <= shards; ++shard) {
            if (tt_logits_block_bound(shard, shards) != TT_LOGITS_ROW_BLOCKS * shard / shards) {
                throw std::runtime_error("incorrect logits shard boundary");
            }
            for (uint32_t blocks : {TT_QUERY_ROW_BLOCKS, TT_KV_ROW_BLOCKS}) {
                if (tt_attention_block_bound(blocks, shard, shards) != blocks / shards * shard) {
                    throw std::runtime_error("incorrect QKV shard boundary");
                }
            }
            const uint32_t hidden = TT_DIM_ROW_BLOCKS * shard / shards;
            const uint32_t alignment = shards <= TT_MAX_SHARDS ? 4 : 1;
            const uint32_t residual = TT_DIM_ROW_BLOCKS / alignment * shard / shards * alignment;
            if (tt_hidden_block_bound(shard, shards) != hidden || tt_residual_block_bound(shard, shards) != residual) {
                throw std::runtime_error("incorrect expert shard boundary");
            }
        }
    }
    static_assert(tt_router_shards(TT_GPT_OSS_20B_EXPERTS, 1) == 1);
    static_assert(tt_router_shards(TT_GPT_OSS_20B_EXPERTS, TT_MAX_SHARDS) == 1);
    static_assert(tt_router_shards(TT_GPT_OSS_120B_EXPERTS, 1) == 1);
    static_assert(tt_router_shards(TT_GPT_OSS_120B_EXPERTS, TT_MAX_SHARDS) == TT_MAX_SHARDS);
    const uint32_t hidden[] = {0, 22, 45, 67, 90, 112, 135, 157, 180};
    const uint32_t residual[] = {0, 20, 44, 64, 88, 112, 132, 156, 180};

    static_assert(TT_MAX_SHARDS == 8 && TT_MAX_GLOBAL_SHARDS == 32 && TT_DIM_ROW_BLOCKS == 180);
    for (uint32_t i = 0; i <= TT_MAX_SHARDS; ++i) {
        if (tt_hidden_block_bound(i, TT_MAX_SHARDS) != hidden[i] ||
            tt_residual_block_bound(i, TT_MAX_SHARDS) != residual[i]) {
            throw std::runtime_error("incorrect expert shard boundary");
        }
    }
}

float unpack_bfp4_value(uint8_t bits, uint8_t exponent) {
    const uint8_t magnitude = bits & 7;
    if (!magnitude) {
        return 0.0f;
    }
    const float scale = std::ldexp(1.0f, int(exponent) - 127 - 2);
    const float value = scale * float(magnitude);
    return (bits & 8) ? -value : value;
}

float unpack_bfp8_value(uint8_t bits, uint8_t exponent) {
    const uint8_t magnitude = bits & 0x7f;
    if (!magnitude) {
        return 0.0f;
    }
    const float scale = std::ldexp(1.0f, int(exponent) - 127 - 6);
    const float value = scale * float(magnitude);
    return (bits & 0x80) ? -value : value;
}

void test_bfp4_packing() {
    const float values[16] = {0.0f,  0.25f,  0.5f,  0.75f, 1.0f,   1.5f, 1.75f, -0.25f,
                              -0.5f, -0.75f, -1.0f, -1.5f, -1.75f, 2.0f, -2.0f, 0.125f};
    uint8_t exponent = 0;
    uint8_t packed[8] = {};
    pack_bfp4_section(values, &exponent, packed);
    const float scale = std::ldexp(1.0f, int(exponent) - 127 - 2);
    for (size_t i = 0; i < 16; ++i) {
        const uint8_t bits = (i & 1) ? uint8_t(packed[i / 2] >> 4) : uint8_t(packed[i / 2] & 0x0f);
        check_close(unpack_bfp4_value(bits, exponent), values[i], scale, "BFP4 reconstruction");
    }
    if (requant_bfp4_strip_bytes != 25920) {
        throw std::runtime_error("unexpected BFP4 strip size");
    }
}

void test_bfp8_packing() {
    const float values[16] = {0.0f,  0.25f,  0.5f,  0.75f, 1.0f,   1.5f, 1.75f, -0.25f,
                              -0.5f, -0.75f, -1.0f, -1.5f, -1.75f, 2.0f, -2.0f, 0.125f};
    uint8_t exponent = 0;
    uint8_t packed[16] = {};
    pack_bfp8_section(values, &exponent, packed);
    const float scale = std::ldexp(1.0f, int(exponent) - 127 - 6);
    for (size_t i = 0; i < 16; ++i) {
        check_close(unpack_bfp8_value(packed[i], exponent), values[i], scale, "BFP8 reconstruction");
    }
    if (requant_bfp8_strip_bytes != 48960) {
        throw std::runtime_error("unexpected BFP8 strip size");
    }
}

void test_f32_matvec_and_norm() {
    const float weights[] = {1, 2, 3, 4};
    const GgufTensor matrix = tensor("f32", {2, 2}, 0, weights);
    const float bias_values[] = {10, 20};
    const GgufTensor bias = tensor("f32-bias", {2}, 0, bias_values);
    const float input[] = {5, 6};
    float output[2];
    matvec_add_f32(matrix, bias, input, output);
    check_close(output[0], 27, 0, "F32 matvec row zero");
    check_close(output[1], 59, 0, "F32 matvec row one");

    const float norm_weights[] = {2, 3};
    const GgufTensor norm = tensor("norm", {2}, 0, norm_weights);
    const float norm_input[] = {3, 4};
    rms_norm(norm_input, norm, 0.0f, output);
    const float scale = 1.0f / std::sqrt(12.5f);
    check_close(output[0], 6 * scale, 1e-6f, "RMSNorm zero");
    check_close(output[1], 12 * scale, 1e-6f, "RMSNorm one");
}

void test_q8_0_matvec() {
    std::array<std::byte, 68> data{};
    const uint16_t one = 0x3c00;
    std::memcpy(data.data(), &one, 2);
    std::memcpy(data.data() + 34, &one, 2);
    std::memset(data.data() + 2, 1, 32);
    std::memset(data.data() + 36, 0xff, 32);
    const GgufTensor matrix = tensor("q8", {32, 2}, 8, data.data());
    std::array<float, 32> input;
    input.fill(1.0f);
    float output[2];
    matvec_q8_0(matrix, input.data(), output);
    check_close(output[0], 32, 0, "Q8_0 positive row");
    check_close(output[1], -32, 0, "Q8_0 negative row");

    const float bias_values[] = {10, 20};
    const GgufTensor bias = tensor("q8-bias", {2}, 0, bias_values);
    matvec_add_q8_0(matrix, bias, input.data(), output);
    check_close(output[0], 42, 0, "Q8_0 positive row with bias");
    check_close(output[1], -12, 0, "Q8_0 negative row with bias");
}

} // namespace

int main() {
    try {
        test_float_formats();
        test_shard_bounds();
        test_bfp4_packing();
        test_bfp8_packing();
        test_f32_matvec_and_norm();
        test_q8_0_matvec();
    } catch (const std::exception &error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
