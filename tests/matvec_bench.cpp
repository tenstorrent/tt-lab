// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// CPU-only microbenchmark and digest for the host matvec kernels (device proxy and CPU
// reference). Digests must match across hosts and ISA variants: they are the bit-exactness
// oracle for new kernels. Usage: matvec-bench [iterations-scale]
#include "constants.h"
#include "tensor.h"
#include "tt_matvec.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

uint32_t random_state = 1;
uint32_t next_random() {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

template<typename F>
double time_ns(size_t iterations, F &&body) {
    const auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        body(i);
    }
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin).count() /
           double(iterations);
}

uint32_t digest_words(const void *data, size_t bytes, uint32_t digest) {
    const auto *words = (const uint32_t *)data;
    for (size_t i = 0; i < bytes / 4; ++i) {
        digest = digest * 31 + words[i];
    }
    return digest;
}

} // namespace

int main(int argc, char **argv) {
    const double scale = argc > 1 ? std::atof(argv[1]) : 1.0;
    const auto iters = [&](size_t n) { return size_t(double(n) * scale) + 1; };

    // Device proxy: BFP8 strips and BF16 tiles, random bytes plus special fills.
    constexpr size_t samples = 64;
    std::vector<uint8_t> bfp8(samples * TT_BFP8_STRIP_BYTES);
    std::vector<uint8_t> bf16(samples * TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES);
    std::vector<uint32_t> b(samples * TT_MVMUL_K_TILES * 16);
    for (uint8_t &v : bfp8) v = uint8_t(next_random());
    for (uint8_t &v : bf16) v = uint8_t(next_random());
    for (uint32_t &v : b) v = (next_random() & 0xffffu) << 3;
    const uint8_t fills[] = {0, 0x80, 0x3f, 0xff};
    for (size_t s = 0; s < 4; ++s) {
        std::memset(bfp8.data() + s * TT_BFP8_STRIP_BYTES, fills[s], TT_BFP8_STRIP_BYTES);
        std::memset(bf16.data() + s * TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES, fills[s],
                    TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES);
    }

    uint32_t digest = 0;
    for (size_t s = 0; s < samples; ++s) {
        uint32_t chains[2][16] = {};
        tt_matvec_bfp8(bfp8.data() + s * TT_BFP8_STRIP_BYTES, b.data() + s * TT_MVMUL_K_TILES * 16, chains);
        digest = digest_words(chains, sizeof(chains), digest);
    }
    const double bfp8_ns = time_ns(iters(2000), [&](size_t i) {
        uint32_t chains[2][16] = {};
        const size_t s = i % samples;
        tt_matvec_bfp8(bfp8.data() + s * TT_BFP8_STRIP_BYTES, b.data() + s * TT_MVMUL_K_TILES * 16, chains);
        __asm__ volatile("" : : "r"(chains) : "memory");
    });
    std::printf("proxy bfp8 strip (16x%u)   %10.0f ns/call  digest %08x\n", TT_MVMUL_K_TILES * 16, bfp8_ns, digest);
    // The proxy is integer emulation of the Tensix datapath: its digests are host- and ISA-independent.
    if (digest != 0x4ee04876u) {
        std::printf("FAIL: proxy bfp8 digest changed (expected 4ee04876)\n");
        return 1;
    }

    digest = 0;
    for (size_t s = 0; s < samples; ++s) {
        uint32_t chains[2][16] = {};
        tt_matvec_bf16(bf16.data() + s * TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES,
                       b.data() + s * TT_MVMUL_K_TILES * 16, s % 2 ? TT_MVMUL_K_TILES : 4, chains);
        digest = digest_words(chains, sizeof(chains), digest);
    }
    const double bf16_ns = time_ns(iters(1000), [&](size_t i) {
        uint32_t chains[2][16] = {};
        const size_t s = i % samples;
        tt_matvec_bf16(bf16.data() + s * TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES,
                       b.data() + s * TT_MVMUL_K_TILES * 16, TT_MVMUL_K_TILES, chains);
        __asm__ volatile("" : : "r"(chains) : "memory");
    });
    std::printf("proxy bf16 block (16x%u)   %10.0f ns/call  digest %08x\n", TT_MVMUL_K_TILES * 16, bf16_ns, digest);
    if (digest != 0xfa7b330au) {
        std::printf("FAIL: proxy bf16 digest changed (expected fa7b330a)\n");
        return 1;
    }

#if defined(__riscv_vector)
    // Bit-exactness of every RVV body against the generic body, plus per-body timing.
    {
        std::vector<int> lmuls = {4};
        uint32_t vlenb;
        __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
        if (vlenb >= 32) {
            lmuls.push_back(2);
        }
        std::printf("matvec kernel selected: %s (VLEN %u)\n", tt_matvec_kernel_name(), vlenb * 8);
        size_t compared = 0;
        for (size_t s = 0; s < samples; ++s) {
            const uint8_t *strip = bfp8.data() + s * TT_BFP8_STRIP_BYTES;
            const uint8_t *tiles = bf16.data() + s * TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES;
            const uint32_t *bv = b.data() + s * TT_MVMUL_K_TILES * 16;
            // Non-zero starting chains exercise the accumulate path from arbitrary Dst contents.
            uint32_t seed[2][16];
            for (auto &chain : seed) {
                for (uint32_t &v : chain) v = s < 8 ? 0 : next_random();
            }
            for (int lmul : lmuls) {
                uint32_t want[2][16], got[2][16];
                std::memcpy(want, seed, sizeof(seed));
                std::memcpy(got, seed, sizeof(seed));
                tt_matvec_bfp8_generic(strip, bv, want);
                tt_matvec_bfp8_rvv(strip, bv, got, lmul);
                if (std::memcmp(want, got, sizeof(want)) != 0) {
                    std::printf("FAIL: bfp8 rvv m%d differs from generic (sample %zu)\n", lmul, s);
                    return 1;
                }
                const size_t k_tiles = s % 2 ? TT_MVMUL_K_TILES : 4;
                std::memcpy(want, seed, sizeof(seed));
                std::memcpy(got, seed, sizeof(seed));
                tt_matvec_bf16_generic(tiles, bv, k_tiles, want);
                tt_matvec_bf16_rvv(tiles, bv, k_tiles, got, lmul);
                if (std::memcmp(want, got, sizeof(want)) != 0) {
                    std::printf("FAIL: bf16 rvv m%d differs from generic (sample %zu)\n", lmul, s);
                    return 1;
                }
                compared += 2;
            }
        }
        std::printf("rvv vs generic: %zu matvec calls bit-identical\n", compared);
        const auto bench = [&](const char *name, auto &&fn, size_t n) {
            const double ns = time_ns(iters(n), [&](size_t i) {
                uint32_t chains[2][16] = {};
                fn(i % samples, chains);
                __asm__ volatile("" : : "r"(chains) : "memory");
            });
            std::printf("  %-28s %10.0f ns/call\n", name, ns);
        };
        bench("bfp8 generic", [&](size_t s, uint32_t (*c)[16]) {
            tt_matvec_bfp8_generic(bfp8.data() + s * TT_BFP8_STRIP_BYTES, b.data() + s * TT_MVMUL_K_TILES * 16, c);
        }, 500);
        bench("bf16 generic", [&](size_t s, uint32_t (*c)[16]) {
            tt_matvec_bf16_generic(bf16.data() + s * TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES,
                                   b.data() + s * TT_MVMUL_K_TILES * 16, TT_MVMUL_K_TILES, c);
        }, 250);
        for (int lmul : lmuls) {
            char name[64];
            std::snprintf(name, sizeof(name), "bfp8 rvv m%d", lmul);
            bench(name, [&](size_t s, uint32_t (*c)[16]) {
                tt_matvec_bfp8_rvv(bfp8.data() + s * TT_BFP8_STRIP_BYTES, b.data() + s * TT_MVMUL_K_TILES * 16, c,
                                   lmul);
            }, 2000);
            std::snprintf(name, sizeof(name), "bf16 rvv m%d", lmul);
            bench(name, [&](size_t s, uint32_t (*c)[16]) {
                tt_matvec_bf16_rvv(bf16.data() + s * TT_MVMUL_K_TILES * TT_MVMUL_BF16_A_TILE_BYTES,
                                   b.data() + s * TT_MVMUL_K_TILES * 16, TT_MVMUL_K_TILES, c, lmul);
            }, 1000);
        }
    }
#endif

    // CPU reference: MXFP4 2880x2880 and Q8_0 2880x2880 matvec on random weights.
    random_state = 0x2880u;  // independent of how much randomness the sections above consumed
    constexpr size_t n = 2880;
    std::vector<uint8_t> mxfp4(n * (n / 32) * 17);
    for (size_t block = 0; block < mxfp4.size(); block += 17) {
        mxfp4[block] = uint8_t(118 + next_random() % 16);
        for (size_t j = 1; j < 17; ++j) mxfp4[block + j] = uint8_t(next_random());
    }
    std::vector<uint8_t> q8(n * (n / 32) * 34);
    for (size_t block = 0; block < q8.size(); block += 34) {
        const uint16_t half = uint16_t(0x2000 + next_random() % 0x800);  // small positive fp16 scale
        std::memcpy(q8.data() + block, &half, 2);
        for (size_t j = 2; j < 34; ++j) q8[block + j] = uint8_t(next_random());
    }
    std::vector<float> biases(n), input(n), output(n);
    for (size_t i = 0; i < n; ++i) {
        biases[i] = float(int(next_random() % 2001) - 1000) / 1024.0f;
        input[i] = float(int(next_random() % 2001) - 1000) / 512.0f;
    }
    const GgufTensor mx{"mx", {n, n}, tensor_mxfp4, 0, (const std::byte *)mxfp4.data()};
    const GgufTensor q{"q8", {n, n}, tensor_q8_0, 0, (const std::byte *)q8.data()};
    const GgufTensor bias{"bias", {n}, tensor_f32, 0, (const std::byte *)biases.data()};

    matvec_add_mxfp4_2880(mx, bias, input.data(), output.data(), 0);
    digest = digest_words(output.data(), n * 4, 0);
    const double mx_ns = time_ns(iters(20), [&](size_t) { matvec_add_mxfp4_2880(mx, bias, input.data(), output.data(), 0); });
    std::printf("cpu mxfp4 2880x2880        %10.0f ns/call  digest %08x\n", mx_ns, digest);

    matvec_add_q8_0(q, bias, input.data(), output.data());
    digest = digest_words(output.data(), n * 4, 0);
    const double q8_ns = time_ns(iters(20), [&](size_t) { matvec_add_q8_0(q, bias, input.data(), output.data()); });
    std::printf("cpu q8_0 2880x2880         %10.0f ns/call  digest %08x\n", q8_ns, digest);

#if defined(__riscv_vector)
    {
        std::vector<int> lmuls = {4};
        uint32_t vlenb;
        __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
        if (vlenb >= 32) {
            lmuls.push_back(2);
        }
        std::vector<float> want(n), got(n);
        // Several inputs, including signed zeros and large magnitudes.
        for (size_t trial = 0; trial < 4; ++trial) {
            if (trial) {
                for (size_t i = 0; i < n; ++i) {
                    input[i] = trial == 1 ? (i % 2 ? -0.0f : 0.0f)
                                          : float(int(next_random() % 2000001) - 1000000) / float(1u << (trial * 4));
                }
            }
            for (int lmul : lmuls) {
                matvec_add_q8_0_generic(q, bias, input.data(), want.data());
                matvec_q8_0_rvv(q, &bias, input.data(), got.data(), lmul);
                if (std::memcmp(want.data(), got.data(), n * 4) != 0) {
                    std::printf("FAIL: q8_0 (bias) rvv m%d differs from generic (trial %zu)\n", lmul, trial);
                    return 1;
                }
                matvec_q8_0_generic(q, input.data(), want.data());
                matvec_q8_0_rvv(q, nullptr, input.data(), got.data(), lmul);
                if (std::memcmp(want.data(), got.data(), n * 4) != 0) {
                    std::printf("FAIL: q8_0 rvv m%d differs from generic (trial %zu)\n", lmul, trial);
                    return 1;
                }
            }
        }
        std::printf("cpu rvv vs generic: q8_0 bit-identical (4 inputs x %zu LMULs)\n", lmuls.size());
        std::printf("  %-28s %10.0f ns/call\n", "q8_0 generic",
                    time_ns(iters(5), [&](size_t) { matvec_add_q8_0_generic(q, bias, input.data(), output.data()); }));
        for (int lmul : lmuls) {
            char name[64];
            std::snprintf(name, sizeof(name), "q8_0 rvv m%d", lmul);
            std::printf("  %-28s %10.0f ns/call\n", name, time_ns(iters(40), [&](size_t) {
                matvec_q8_0_rvv(q, &bias, input.data(), output.data(), lmul);
            }));
        }
    }
#endif
    return 0;
}
