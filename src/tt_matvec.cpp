// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tt_matvec.h"
#include "constants.h"

#include <bit>
#include <cstring>
#include <immintrin.h>

// Compile this arithmetic together for each ISA so vector ABIs cannot cross targets.
#if defined(__AVX512F__)
#define MATVEC_TARGET [[gnu::target("arch=x86-64-v4")]]
#else
#define MATVEC_TARGET [[gnu::target("default")]]
#endif

namespace {

typedef uint32_t u32x16 __attribute__((vector_size(64)));
typedef int32_t i32x16 __attribute__((vector_size(64)));
typedef uint16_t u16x16 __attribute__((vector_size(32)));

u32x16 splat_u32(uint32_t x) {
    return u32x16{} + x;
}

i32x16 splat_i32(int32_t x) {
    return i32x16{} + x;
}

u32x16 select_u32(u32x16 mask, u32x16 yes, u32x16 no) {
    return (mask & yes) | (~mask & no);
}

i32x16 select_i32(u32x16 mask, i32x16 yes, i32x16 no) {
    return (i32x16(mask) & yes) | (i32x16(~mask) & no);
}

i32x16 max_i32(i32x16 a, i32x16 b) {
#if defined(__AVX512F__)
    return (i32x16)_mm512_max_epi32((__m512i)a, (__m512i)b);
#else
    return select_i32(u32x16(a > b), a, b);
#endif
}

u32x16 min_u32(u32x16 a, u32x16 b) {
#if defined(__AVX512F__)
    return (u32x16)_mm512_min_epu32((__m512i)a, (__m512i)b);
#else
    return select_u32(u32x16(a < b), a, b);
#endif
}

u32x16 clz_u32_nonzero(u32x16 x) {
#if defined(__AVX512CD__)
    return (u32x16)_mm512_lzcnt_epi32((__m512i)x);
#else
    u32x16 result;
    for (size_t i = 0; i < 16; ++i) {
        result[i] = uint32_t(std::countl_zero(x[i]));
    }
    return result;
#endif
}

u32x16 align_man_16(u32x16 man, u32x16 sign, i32x16 exp_value, i32x16 exp) {
    const u32x16 active = u32x16(exp_value < exp);
    const u32x16 diff = u32x16(exp - exp_value);
    const u32x16 small = u32x16(diff < splat_u32(31));
    const u32x16 shift_m1 = select_u32(active & small, diff - splat_u32(1), splat_u32(0));
    const u32x16 shifted = (man + (splat_u32(1) << shift_m1) - sign) >> diff;
    return select_u32(active, select_u32(small, shifted, splat_u32(0)), man);
}

u32x16 normalize_encode_fp32_acc_16(u32x16 sign_sop0, i32x16 exp_sop0, u32x16 man_sop0, u32x16 sign_sop1,
                                    i32x16 exp_sop1, u32x16 man_sop1, u32x16 dst_value) {
    u32x16 sign_dst = dst_value >> 31;
    i32x16 exp_dst = i32x16((dst_value >> 23) & splat_u32(255));
    u32x16 man_dst = (dst_value & splat_u32(0x7FFFFF)) | splat_u32(0x800000);
    man_dst = select_u32(u32x16(exp_dst != splat_i32(0)), man_dst, splat_u32(0));

    i32x16 exp = max_i32(max_i32(exp_sop0, exp_sop1), exp_dst);
    const u32x16 exp_positive = u32x16(exp > splat_i32(0));
    man_sop0 = align_man_16(man_sop0, sign_sop0, exp_sop0, exp);
    man_sop1 = align_man_16(man_sop1, sign_sop1, exp_sop1, exp);
    man_dst = align_man_16(man_dst, splat_u32(0), exp_dst, exp);

    u32x16 sign = sign_dst;
    i32x16 man = i32x16(man_dst);
    man += select_i32(u32x16(sign_sop0 == sign), i32x16(man_sop0), -i32x16(man_sop0));
    man += select_i32(u32x16(sign_sop1 == sign), i32x16(man_sop1), -i32x16(man_sop1));
    const u32x16 negative = u32x16(man < splat_i32(0));
    man = select_i32(negative, -man, man);
    sign = select_u32(negative, sign ^ splat_u32(1), sign);

    const u32x16 nonzero = u32x16(man != splat_i32(0)) & exp_positive;
    u32x16 man_u = u32x16(man);
    i32x16 shift = i32x16(clz_u32_nonzero(man_u)) - splat_i32(8);
    exp -= shift;
    const u32x16 right_mask = u32x16(shift < splat_i32(0));
    const u32x16 right_shift = u32x16(-shift);
    const u32x16 right_shift_m1 = select_u32(right_mask, right_shift - splat_u32(1), splat_u32(0));
    const u32x16 right_value = (man_u + (splat_u32(1) << right_shift_m1)) >> right_shift;
    const u32x16 left_value = man_u << u32x16(shift);
    man_u = select_u32(right_mask, right_value, left_value);
    exp += i32x16(u32x16((man_u & splat_u32(0x1000000u)) != splat_u32(0)) & splat_u32(1));
    man_u &= splat_u32(0x7FFFFFu);

    const u32x16 valid = nonzero & u32x16(exp > splat_i32(0));
    const u32x16 inf = u32x16(exp >= splat_i32(255));
    const u32x16 finite = (sign << 31) | (u32x16(exp) << 23) | man_u;
    const u32x16 inf_value = (sign << 31) | splat_u32(255u << 23);
    return select_u32(valid, select_u32(inf, inf_value, finite), splat_u32(0));
}

u32x16 load_dst_values_16(const uint32_t *dst_internal) {
    alignas(64) uint32_t values[16];
    for (uint32_t i = 0; i < 16; ++i) {
        values[i] = dst_decode_fp32_proxy(dst_internal[i]);
    }
    return *(const u32x16 *)values;
}

void store_dst_values_16(uint32_t *dst_internal, u32x16 result) {
    alignas(64) uint32_t values[16];
    *(u32x16 *)values = result;
    for (uint32_t i = 0; i < 16; ++i) {
        dst_internal[i] = dst_encode_fp32_proxy(values[i]);
    }
}

u32x16 bfp8_tile_k_to_srca_16(const uint8_t *strip, uint32_t k_column) {
    // Each packed row shares one exponent across K. Gather a column to model UNPACR transpose.
    // Read aligned four-byte groups so the last column never reads beyond the tile.
    const uint8_t *source = strip + TT_MVMUL_BFP8_TILE_EXP_BYTES + (k_column & ~3u);
#if defined(__AVX512F__)
    const __m512i offsets = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240);
    const u32x16 words = (u32x16)_mm512_i32gather_epi32(offsets, source, 1);
    const u32x16 packed = (words >> ((k_column & 3u) * 8u)) & splat_u32(255u);
    const u32x16 shared_exp = (u32x16)_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)strip));
#else
    u32x16 packed, shared_exp;
    for (size_t i = 0; i < 16; ++i) {
        packed[i] = source[i * 16 + (k_column & 3u)];
        shared_exp[i] = strip[i];
    }
#endif
    const u32x16 sign = packed >> 7;
    const u32x16 magnitude = (packed << 1) & splat_u32(255u);
    const u32x16 lz = clz_u32_nonzero(magnitude) - splat_u32(24u);
    const u32x16 exponent = (shared_exp - lz) & splat_u32(255u);
    const u32x16 bf16 = (sign << 15) | (exponent << 7) | ((magnitude << lz) & splat_u32(0x7Eu));

    // Packed negative zero unpacks as negative infinity, not signed zero.
    return select_u32(u32x16(magnitude != splat_u32(0)), bf16, sign * splat_u32(0xFF80u)) << 3;
}

u32x16 bf16_tile_k_to_srca_16(const uint8_t *tile_bytes, uint32_t k_column) {
    alignas(32) uint16_t values[16];
    const uint16_t *tile = (const uint16_t *)tile_bytes;
    const uint16_t *source = tile + (k_column / 2) * 32 + (k_column & 1) * 16;
    std::memcpy(values, source, sizeof(values));
    const u16x16 raw = *(const u16x16 *)values;
    return __builtin_convertvector(raw, u32x16) << 3;
}

struct MvmulSop16 {
    u32x16 sign;
    i32x16 exp;
    u32x16 man;
};

template<uint32_t phase, uint32_t srca_format>
MvmulSop16 mvmul_sop_tile_16(const uint8_t *tile, const uint32_t *b_values, uint32_t k_base) {
    static_assert(phase < 2);
    static_assert(srca_format == 5 || srca_format == 6);
    u32x16 signs[8];
    i32x16 exps[8];
    u32x16 mans[8];
    const u32x16 zero_threshold = splat_u32(0x400u);
    const u32x16 low_mask = splat_u32(0x3FFFFu);
    const u32x16 exp_mask = splat_u32(255u);
    const u32x16 man_mask = splat_u32(0x3FFu);
    const u32x16 hidden_bit = splat_u32(0x400u);
    constexpr int32_t exp_prod_adj = -127 - (phase ? 5 : 0);

    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t k_column = k_base + i;
        u32x16 a;
        if constexpr (srca_format == 6) {
            a = bfp8_tile_k_to_srca_16(tile, k_column);
        } else {
            a = bf16_tile_k_to_srca_16(tile, k_column);
        }
        const u32x16 b = splat_u32(b_values[k_column]);
        const u32x16 zero = u32x16(((a & low_mask) < zero_threshold)) | u32x16(((b & low_mask) < zero_threshold));
        const u32x16 active = ~zero;
        const u32x16 exp_a = (a >> 10) & exp_mask;
        const u32x16 exp_b = (b >> 10) & exp_mask;
        u32x16 man_a = (a & man_mask) | hidden_bit;
        u32x16 man_b = (b & man_mask) | hidden_bit;
        if constexpr (phase & 1) {
            man_a = (man_a >> 1) & splat_u32(31u);
        } else {
            man_a >>= 6;
        }
        man_b >>= 4;
        signs[i] = ((a >> 18) ^ (b >> 18)) & active;
        exps[i] = select_i32(zero, splat_i32(0), i32x16(exp_a + exp_b) + splat_i32(exp_prod_adj));
        mans[i] = select_u32(zero, splat_u32(0), man_a * man_b);
    }

    i32x16 exp_sop = exps[0];
    for (uint32_t i = 1; i < 8; ++i) {
        exp_sop = max_i32(exp_sop, exps[i]);
    }

    i32x16 man_sop = splat_i32(0);
    const u32x16 positive = u32x16(exp_sop > splat_i32(0));
    for (uint32_t i = 0; i < 8; ++i) {
        u32x16 exp_diff = u32x16(exp_sop - exps[i]);
        exp_diff = min_u32(exp_diff, splat_u32(30u));
        const u32x16 man_u = mans[i];
        u32x16 man = ((man_u << 1) + (splat_u32(1u) << exp_diff)) >> (exp_diff + splat_u32(1u));
        i32x16 signed_man = select_i32(u32x16(signs[i] != splat_u32(0)), -i32x16(man), i32x16(man));
        signed_man = select_i32(positive, signed_man, splat_i32(0));
        man_sop += signed_man;
    }

    const u32x16 negative = u32x16(man_sop < splat_i32(0));
    MvmulSop16 sop = {};
    sop.sign = negative & splat_u32(1u);
    sop.man = u32x16(select_i32(negative, -man_sop, man_sop) << 13);
    sop.exp = select_i32(positive, exp_sop, splat_i32(0));
    return sop;
}

template<uint32_t phase, uint32_t srca_format>
[[gnu::always_inline]]
inline void mvmul_tile_add_16(const uint8_t *tile, const uint32_t *b_values, uint32_t *dst_internal) {
    const MvmulSop16 sop0 = mvmul_sop_tile_16<phase, srca_format>(tile, b_values, 0);
    const MvmulSop16 sop1 = mvmul_sop_tile_16<phase, srca_format>(tile, b_values, 8);
    const u32x16 dst_value = load_dst_values_16(dst_internal);
    const u32x16 result =
        normalize_encode_fp32_acc_16(sop0.sign, sop0.exp, sop0.man, sop1.sign, sop1.exp, sop1.man, dst_value);
    store_dst_values_16(dst_internal, result);
}

} // namespace

MATVEC_TARGET
void tt_matvec_bf16(const uint8_t *tiles, const uint32_t *b_value_rows, size_t k_tiles, uint32_t chains[2][16]) {
    // Each inner64 group runs all four tiles in phase 0, then all four in phase 1.
    for (size_t base = 0; base < k_tiles; base += 4) {
        for (size_t k = base; k < base + 4; ++k) {
            mvmul_tile_add_16<0, 5>(tiles + k * TT_MVMUL_BF16_A_TILE_BYTES, b_value_rows + k * TT_MVMUL_TILE_DIM,
                                    chains[k % 2]);
        }
        for (size_t k = base; k < base + 4; ++k) {
            mvmul_tile_add_16<1, 5>(tiles + k * TT_MVMUL_BF16_A_TILE_BYTES, b_value_rows + k * TT_MVMUL_TILE_DIM,
                                    chains[k % 2]);
        }
    }
}

MATVEC_TARGET
void tt_matvec_bfp8(const uint8_t *strip, const uint32_t *b_value_rows, uint32_t chains[2][16]) {
    uint8_t tile[TT_MVMUL_BFP8_TILE_EXP_BYTES + TT_MVMUL_BFP8_TILE_DATUM_BYTES];

    for (size_t k = 0; k < TT_MVMUL_K_TILES; ++k) {
        std::memcpy(tile, strip + k * TT_MVMUL_BFP8_TILE_EXP_BYTES, TT_MVMUL_BFP8_TILE_EXP_BYTES);
        std::memcpy(tile + TT_MVMUL_BFP8_TILE_EXP_BYTES,
                    strip + TT_BFP8_STRIP_EXP_BYTES + k * TT_MVMUL_BFP8_TILE_DATUM_BYTES,
                    TT_MVMUL_BFP8_TILE_DATUM_BYTES);
        const uint32_t *b_values = b_value_rows + k * TT_MVMUL_TILE_DIM;

        mvmul_tile_add_16<0, 6>(tile, b_values, chains[k % 2]);
    }
}
