// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "requant.h"
#include "tensor.h"

#include <cstring>
#include <immintrin.h>

// Compile this arithmetic together for each ISA so vector ABIs cannot cross targets.
#if defined(__AVX512F__)
#define REQUANT_TARGET [[gnu::target("arch=x86-64-v4")]]
#else
#define REQUANT_TARGET [[gnu::target("default")]]
#endif

// The 16 lanes are the 16 rows of a strip, so each lane runs one BFP8 section through the same IEEE operations, in
// the same order, as pack_bfp8_section: the MXFP4 product, the floor, the double error sums with one std::fma per
// element in element order, and the comparison. Only the integer conversion of the floor needs care: x86 converts it
// to a 64-bit integer and keeps the low 32 bits, which these kernels reproduce for values that do not fit.
namespace {

typedef float f32x16 __attribute__((vector_size(64)));
typedef double f64x16 __attribute__((vector_size(128)));
typedef uint32_t u32x16 __attribute__((vector_size(64)));
typedef int32_t i32x16 __attribute__((vector_size(64)));

constexpr size_t strip_rows = requant_bfp8_strip_rows;
constexpr size_t section_elements = 16;
constexpr size_t mxfp4_block_columns = 32;
constexpr size_t mxfp4_block_bytes = 17;

[[gnu::always_inline]] inline u32x16 splat_u32(uint32_t x) {
    return u32x16{} + x;
}

[[gnu::always_inline]] inline u32x16 select_u32(u32x16 mask, u32x16 yes, u32x16 no) {
    return (mask & yes) | (~mask & no);
}

// All-ones lanes where a > b as signed integers. (GCC lowers 16-lane comparisons element by element without AVX-512.)
[[gnu::always_inline]] inline u32x16 greater_i32(u32x16 a, u32x16 b) {
#if defined(__AVX512F__)
    return (u32x16)_mm512_movm_epi32(_mm512_cmpgt_epi32_mask((__m512i)a, (__m512i)b));
#else
    __m256i x[2];
    __m256i y[2];
    std::memcpy(x, &a, sizeof(a));
    std::memcpy(y, &b, sizeof(b));
    for (size_t i = 0; i < 2; ++i) {
        x[i] = _mm256_cmpgt_epi32(x[i], y[i]);
    }
    std::memcpy(&a, x, sizeof(a));
    return a;
#endif
}

[[gnu::always_inline]] inline f32x16 as_f32(u32x16 x) {
    return (f32x16)x;
}

[[gnu::always_inline]] inline u32x16 as_u32(f32x16 x) {
    return (u32x16)x;
}

[[gnu::always_inline]] inline f32x16 floor_16(f32x16 x) {
#if defined(__AVX512F__)
    return (f32x16)_mm512_roundscale_ps((__m512)x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
#else
    __m256 half[2];
    std::memcpy(half, &x, sizeof(x));
    for (__m256 &h : half) {
        h = _mm256_round_ps(h, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    }
    std::memcpy(&x, half, sizeof(x));
    return x;
#endif
}

[[gnu::always_inline]] inline u32x16 truncate_u32(f32x16 x) {
#if defined(__AVX512F__)
    return (u32x16)_mm512_cvttps_epi32((__m512)x);
#else
    __m256 in[2];
    __m256i out[2];
    std::memcpy(in, &x, sizeof(x));
    for (size_t i = 0; i < 2; ++i) {
        out[i] = _mm256_cvttps_epi32(in[i]);
    }
    u32x16 result;
    std::memcpy(&result, out, sizeof(result));
    return result;
#endif
}

[[gnu::always_inline]] inline u32x16 shift_left_u32(u32x16 x, u32x16 count) {
#if defined(__AVX512F__)
    return (u32x16)_mm512_sllv_epi32((__m512i)x, (__m512i)count);
#else
    __m256i value[2];
    __m256i shift[2];
    std::memcpy(value, &x, sizeof(x));
    std::memcpy(shift, &count, sizeof(count));
    for (size_t i = 0; i < 2; ++i) {
        value[i] = _mm256_sllv_epi32(value[i], shift[i]);
    }
    std::memcpy(&x, value, sizeof(x));
    return x;
#endif
}

[[gnu::always_inline]] inline f64x16 fma_16(f64x16 a, f64x16 b, f64x16 c) {
#if defined(__AVX512F__)
    __m512d x[2];
    __m512d y[2];
    __m512d z[2];
    std::memcpy(x, &a, sizeof(a));
    std::memcpy(y, &b, sizeof(b));
    std::memcpy(z, &c, sizeof(c));
    for (size_t i = 0; i < 2; ++i) {
        z[i] = _mm512_fmadd_pd(x[i], y[i], z[i]);
    }
#else
    __m256d x[4];
    __m256d y[4];
    __m256d z[4];
    std::memcpy(x, &a, sizeof(a));
    std::memcpy(y, &b, sizeof(b));
    std::memcpy(z, &c, sizeof(c));
    for (size_t i = 0; i < 4; ++i) {
        z[i] = _mm256_fmadd_pd(x[i], y[i], z[i]);
    }
#endif
    std::memcpy(&c, z, sizeof(c));
    return c;
}

// float(mxfp4_values[code]) for codes 0..15.
[[gnu::always_inline]] inline f32x16 mxfp4_code_values_16(u32x16 codes) {
    const f32x16 table = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
#if defined(__AVX512F__)
    return (f32x16)_mm512_permutexvar_ps((__m512i)codes, (__m512)table);
#else
    __m256 low_table;
    __m256 high_table;
    std::memcpy(&low_table, &table, sizeof(low_table));
    std::memcpy(&high_table, (const char *)&table + sizeof(low_table), sizeof(high_table));
    __m256i index[2];
    __m256 result[2];
    std::memcpy(index, &codes, sizeof(codes));
    for (size_t i = 0; i < 2; ++i) {
        const __m256 low = _mm256_permutevar8x32_ps(low_table, index[i]);
        const __m256 high = _mm256_permutevar8x32_ps(high_table, index[i]);
        result[i] = _mm256_blendv_ps(low, high, _mm256_castsi256_ps(_mm256_slli_epi32(index[i], 28)));
    }
    f32x16 values;
    std::memcpy(&values, result, sizeof(values));
    return values;
#endif
}

// The biased exponent of fabs(value); zeros and subnormals give 0, as they add nothing to the section maximum.
[[gnu::always_inline]] inline u32x16 biased_exponent_16(f32x16 values) {
    return (as_u32(values) << 1) >> 24;
}

// std::ldexp(1.0f, exponent - 127 - 6), exactly, for exponent 0..255.
[[gnu::always_inline]] inline f32x16 bfp8_scale_16(u32x16 exponent) {
    const u32x16 subnormal = shift_left_u32(splat_u32(1), (exponent & splat_u32(7)) + splat_u32(16));
    const u32x16 normal = (exponent - splat_u32(6)) << 23;
    return as_f32(select_u32(greater_i32(splat_u32(7), exponent), subnormal, normal));
}

// uint32_t(std::floor(std::fabs(value) / scale + 0.5f)) as x86-64 computes it: the floor converted to a 64-bit
// integer (0x8000000000000000 when it does not fit) and truncated to 32 bits.
[[gnu::always_inline]] inline u32x16 bfp_magnitude_16(f32x16 values, f32x16 scale) {
    const f32x16 magnitude = floor_16(as_f32(as_u32(values) & splat_u32(0x7FFFFFFF)) / scale + 0.5f);
    const u32x16 bits = as_u32(magnitude);
    const u32x16 large = shift_left_u32((bits & splat_u32(0x7FFFFF)) | splat_u32(0x800000), (bits >> 23) - 150);
    // The magnitude is at least 0.5 or +infinity, so its bits order like it; 0x4F000000 is 2^31.
    return select_u32(greater_i32(splat_u32(0x4F000000), bits), truncate_u32(magnitude), large);
}

[[gnu::always_inline]] inline u32x16 min_u32(u32x16 a, u32x16 b) {
#if defined(__AVX512F__)
    return (u32x16)_mm512_min_epu32((__m512i)a, (__m512i)b);
#else
    __m256i x[2];
    __m256i y[2];
    std::memcpy(x, &a, sizeof(a));
    std::memcpy(y, &b, sizeof(b));
    for (size_t i = 0; i < 2; ++i) {
        x[i] = _mm256_min_epu32(x[i], y[i]);
    }
    std::memcpy(&a, x, sizeof(a));
    return a;
#endif
}

// All-ones lanes where a < b (an ordered comparison, as the scalar operator).
[[gnu::always_inline]] inline u32x16 less_f64(f64x16 a, f64x16 b) {
    uint32_t mask = 0;
#if defined(__AVX512F__)
    __m512d x[2];
    __m512d y[2];
    std::memcpy(x, &a, sizeof(a));
    std::memcpy(y, &b, sizeof(b));
    for (size_t i = 0; i < 2; ++i) {
        mask |= uint32_t(_mm512_cmp_pd_mask(x[i], y[i], _CMP_LT_OQ)) << (8 * i);
    }
#else
    __m256d x[4];
    __m256d y[4];
    std::memcpy(x, &a, sizeof(a));
    std::memcpy(y, &b, sizeof(b));
    for (size_t i = 0; i < 4; ++i) {
        mask |= uint32_t(_mm256_movemask_pd(_mm256_cmp_pd(x[i], y[i], _CMP_LT_OQ))) << (4 * i);
    }
#endif
    const u32x16 lane = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    return -((splat_u32(mask) >> lane) & 1);
}

// double(unpacked) - double(value) for one element of bfp_section_error with seven magnitude bits.
[[gnu::always_inline]] inline f64x16 bfp8_delta_16(f32x16 values, f32x16 scale) {
    const u32x16 magnitude = min_u32(bfp_magnitude_16(values, scale), splat_u32(127));
    const f32x16 product = scale * __builtin_convertvector(i32x16(magnitude), f32x16);
    const f32x16 unpacked = as_f32(as_u32(product) ^ (as_u32(values) & splat_u32(0x80000000)));
    return __builtin_convertvector(unpacked, f64x16) - __builtin_convertvector(values, f64x16);
}

// pack_bfp8_value for one element.
[[gnu::always_inline]] inline u32x16 bfp8_pack_16(f32x16 values, f32x16 scale) {
    const u32x16 magnitude = min_u32(bfp_magnitude_16(values, scale), splat_u32(127));
    const u32x16 packed = magnitude | ((as_u32(values) >> 31) << 7);
    return select_u32(greater_i32(magnitude, splat_u32(0)), packed, splat_u32(0));
}

// choose_bfp_exponent with seven magnitude bits, for 16 sections at once.
[[gnu::always_inline]] inline u32x16 choose_bfp8_exponent_16(const f32x16 *values) {
    u32x16 max_exponent = splat_u32(0);
    for (size_t i = 0; i < section_elements; ++i) {
        const u32x16 exponent = biased_exponent_16(values[i]);
        max_exponent = select_u32(greater_i32(exponent, max_exponent), exponent, max_exponent);
    }
    const u32x16 smaller_exponent = max_exponent - splat_u32(1);
    const f32x16 max_scale = bfp8_scale_16(max_exponent);
    const f32x16 smaller_scale = bfp8_scale_16(smaller_exponent & splat_u32(255));
    f64x16 max_error = {};
    f64x16 smaller_error = {};
    for (size_t i = 0; i < section_elements; ++i) {
        const f64x16 max_delta = bfp8_delta_16(values[i], max_scale);
        const f64x16 smaller_delta = bfp8_delta_16(values[i], smaller_scale);
        max_error = fma_16(max_delta, max_delta, max_error);
        smaller_error = fma_16(smaller_delta, smaller_delta, smaller_error);
    }
    const u32x16 smaller = less_f64(smaller_error, max_error);
    const u32x16 chosen = select_u32(smaller, smaller_exponent, max_exponent);
    return select_u32(greater_i32(max_exponent, splat_u32(1)), chosen, max_exponent);
}

// Transposes 16 rows of 16 bytes: four rounds of byte interleaves rotate the 8-bit (row, column) index by four.
[[gnu::always_inline]] inline void transpose_16x16_u8(__m128i rows[16]) {
    for (size_t round = 0; round < 4; ++round) {
        __m128i next[16];
        for (size_t i = 0; i < 8; ++i) {
            next[2 * i] = _mm_unpacklo_epi8(rows[i], rows[i + 8]);
            next[2 * i + 1] = _mm_unpackhi_epi8(rows[i], rows[i + 8]);
        }
        std::memcpy(rows, next, sizeof(next));
    }
}

// Lanes 0..255 to bytes.
[[gnu::always_inline]] inline __m128i narrow_u8(u32x16 x) {
#if defined(__AVX512F__)
    return _mm512_cvtepi32_epi8((__m512i)x);
#else
    __m256i half[2];
    std::memcpy(half, &x, sizeof(x));
    const __m256i words = _mm256_permute4x64_epi64(_mm256_packus_epi32(half[0], half[1]), 0xD8);
    return _mm_packus_epi16(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
#endif
}

[[gnu::always_inline]] inline u32x16 widen_u8(__m128i x) {
#if defined(__AVX512F__)
    return (u32x16)_mm512_cvtepu8_epi32(x);
#else
    const __m256i half[2] = {_mm256_cvtepu8_epi32(x), _mm256_cvtepu8_epi32(_mm_srli_si128(x, 8))};
    u32x16 result;
    std::memcpy(&result, half, sizeof(result));
    return result;
#endif
}

} // namespace

REQUANT_TARGET
void pack_mxfp4_bfp8_strip(const std::byte *rows, size_t row_bytes, uint8_t *strip) {
    uint8_t *exponents = strip;
    uint8_t *data = strip + requant_bfp8_strip_exponent_bytes;

    for (size_t column = 0; column < requant_bfp8_strip_columns; column += mxfp4_block_columns) {
        const std::byte *block = rows + column / mxfp4_block_columns * mxfp4_block_bytes;
        // codes[i] holds element i's byte for all 16 rows; low nibbles are columns 0..15 of the block.
        __m128i codes[strip_rows];
        alignas(64) float scale_values[strip_rows];
        for (size_t row = 0; row < strip_rows; ++row) {
            const std::byte *source = block + row * row_bytes;
            scale_values[row] = mxfp4_scale(uint8_t(source[0]));
            codes[row] = _mm_loadu_si128((const __m128i *)(source + 1));
        }
        transpose_16x16_u8(codes);
        const f32x16 scale = *(const f32x16 *)scale_values;

        for (size_t half = 0; half < 2; ++half) {
            f32x16 values[section_elements];
            for (size_t i = 0; i < section_elements; ++i) {
                const u32x16 bytes = widen_u8(codes[i]);
                values[i] = scale * mxfp4_code_values_16(half ? bytes >> 4 : bytes & 15);
            }
            const u32x16 exponent = choose_bfp8_exponent_16(values);
            const f32x16 exponent_scale = bfp8_scale_16(exponent);
            __m128i packed[strip_rows];
            for (size_t i = 0; i < section_elements; ++i) {
                packed[i] = narrow_u8(bfp8_pack_16(values[i], exponent_scale));
            }
            transpose_16x16_u8(packed);
            const size_t section_column = column + half * section_elements;
            const __m128i exponent_bytes = narrow_u8(exponent);
            std::memcpy(exponents + section_column, &exponent_bytes, sizeof(exponent_bytes));
            std::memcpy(data + section_column * strip_rows, packed, sizeof(packed));
        }
    }
}
