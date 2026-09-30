// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "constants.h"
#include "tensor.h"
#include "tt_matvec.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// Test both bodies directly as well as the IFUNC-selected entry points. These are GCC's
// multiversioned symbols; direct v4 calls must remain behind the CPU capability check.
extern void bf16_v3(const uint8_t *, const uint32_t *, size_t, uint32_t (*)[16]) asm("_Z14tt_matvec_bf16PKhPKjmPA16_j");
extern void bf16_v4(const uint8_t *, const uint32_t *, size_t,
                    uint32_t (*)[16]) asm("_Z14tt_matvec_bf16PKhPKjmPA16_j.arch_x86_64_v4");
extern void bfp8_v3(const uint8_t *, const uint32_t *, uint32_t (*)[16]) asm("_Z14tt_matvec_bfp8PKhPKjPA16_j");
extern void bfp8_v4(const uint8_t *, const uint32_t *,
                    uint32_t (*)[16]) asm("_Z14tt_matvec_bfp8PKhPKjPA16_j.arch_x86_64_v4");

static void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_proxy_matvec(bool v4) {
    std::vector<uint8_t> bf16(TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES);
    std::vector<uint8_t> bfp8(TT_BFP8_STRIP_BYTES);
    std::vector<uint32_t> b(TT_MVMUL_K_TILES * 16);
    uint32_t random = 1;
    const auto next = [&]() {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };
    uint32_t digest = 0;
    for (size_t sample = 0; sample < 16; ++sample) {
        for (uint8_t &value : bf16) {
            value = uint8_t(next());
        }
        for (uint8_t &value : bfp8) {
            value = uint8_t(next());
        }
        for (uint32_t &value : b) {
            value = (next() & 0xffffu) << 3;
        }
        // Include zeros, normal values, and signed/exceptional encodings in addition to random data.
        if (sample < 4) {
            const uint8_t fill[] = {0, 0x80, 0x3f, 0xff};
            std::memset(bf16.data(), fill[sample], bf16.size());
            std::memset(bfp8.data(), fill[sample], bfp8.size());
        }
        uint32_t reference[2][16] = {};
        uint32_t selected[2][16] = {};
        uint32_t wide[2][16] = {};
        const size_t k_tiles = sample % 2 ? TT_MVMUL_K_TILES : 4;
        bf16_v3(bf16.data(), b.data(), k_tiles, reference);
        tt_matvec_bf16(bf16.data(), b.data(), k_tiles, selected);
        check(std::memcmp(reference, selected, sizeof(reference)) == 0, "BF16 dispatch mismatch");
        if (v4) {
            bf16_v4(bf16.data(), b.data(), k_tiles, wide);
            check(std::memcmp(reference, wide, sizeof(reference)) == 0, "BF16 v3/v4 mismatch");
        }
        for (const auto &chain : reference) {
            for (uint32_t value : chain) {
                digest = digest * 31 + value;
            }
        }
        std::memset(reference, 0, sizeof(reference));
        std::memset(selected, 0, sizeof(selected));
        std::memset(wide, 0, sizeof(wide));
        bfp8_v3(bfp8.data(), b.data(), reference);
        tt_matvec_bfp8(bfp8.data(), b.data(), selected);
        check(std::memcmp(reference, selected, sizeof(reference)) == 0, "BFP8 dispatch mismatch");
        if (v4) {
            bfp8_v4(bfp8.data(), b.data(), wide);
            check(std::memcmp(reference, wide, sizeof(reference)) == 0, "BFP8 v3/v4 mismatch");
        }
        for (const auto &chain : reference) {
            for (uint32_t value : chain) {
                digest = digest * 31 + value;
            }
        }
    }
    std::printf("proxy matvec digest: %08x\n", digest);
}

static void test_mxfp4() {
    constexpr size_t n = 2880;
    constexpr size_t expert_bytes = n * (n / 32) * 17;
    std::vector<uint8_t> data(2 * expert_bytes, 0);
    std::vector<float> biases(2 * n, 0.25f);
    std::vector<float> input(n, 0.5f), output(n);
    for (size_t block = expert_bytes; block < data.size(); block += 17) {
        data[block] = 127;
        std::memset(data.data() + block + 1, 0x21, 16);
    }
    const GgufTensor matrix{"weights", {n, n, 2}, tensor_mxfp4, 0, (const std::byte *)data.data()};
    const GgufTensor bias{"bias", {n, 2}, tensor_f32, 0, (const std::byte *)biases.data()};
    matvec_add_mxfp4_2880(matrix, bias, input.data(), output.data(), 1);
    for (float value : output) {
        check(value == 1080.25f, "MXFP4 matvec mismatch");
    }
    for (size_t count : {size_t(3), size_t(16), size_t(17)}) {
        std::vector<size_t> tokens(count);
        std::vector<float> transposed(n * count), workspace(4 * count);
        output.assign((count + 1) * n, -1.0f);
        for (size_t s = 0; s < count; ++s) {
            tokens[s] = count - s;
            for (size_t k = 0; k < n; ++k) {
                transposed[k * count + s] = float(s + 1) * 0.5f;
            }
        }
        matmul_add_mxfp4_selected_transposed(matrix, bias, transposed.data(), tokens.data(), count, output.data(),
                                             workspace.data(), 1);
        for (size_t row = 0; row < n; ++row) {
            check(output[row] == -1.0f, "MXFP4 scatter overwrote unselected token");
            for (size_t s = 0; s < count; ++s) {
                check(output[tokens[s] * n + row] == 1080.0f * float(s + 1) + 0.25f, "MXFP4 matmul mismatch");
            }
        }
    }
}

int main() {
    try {
        const bool v4 = __builtin_cpu_supports("x86-64-v4");
        std::printf("host dispatch: %s\n", v4 ? "v4" : "v3");
        test_proxy_matvec(v4);
        test_mxfp4();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "host dispatch test failed: %s\n", error.what());
        return 1;
    }
}
