// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

size_t checked_product(const std::vector<uint64_t> &shape, size_t first) {
    size_t result = 1;
    for (size_t i = first; i < shape.size(); ++i) {
        if (shape[i] > SIZE_MAX || result > SIZE_MAX / size_t(shape[i])) {
            throw std::runtime_error("tensor is too large");
        }
        result *= size_t(shape[i]);
    }
    return result;
}

uint16_t load_u16(const std::byte *data) {
    uint16_t result;
    std::memcpy(&result, data, sizeof(result));
    return result;
}

constexpr int8_t mxfp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

constexpr float make_mxfp4_scale(uint8_t exponent) {
    uint32_t bits;
    if (exponent < 2) {
        bits = 0x00200000u << exponent;
    } else {
        bits = uint32_t(exponent - 1) << 23;
    }
    return std::bit_cast<float>(bits);
}

struct Mxfp4ScaleTable {
    float values[256];
};

constexpr Mxfp4ScaleTable make_mxfp4_scale_table() {
    Mxfp4ScaleTable table{};
    for (size_t scale = 0; scale < 256; ++scale) {
        table.values[scale] = make_mxfp4_scale(uint8_t(scale));
    }
    return table;
}

struct Mxfp4WeightTable {
    float values[256][16];
};

Mxfp4WeightTable make_mxfp4_weight_table() {
    Mxfp4WeightTable table{};
    for (size_t scale = 0; scale < 256; ++scale) {
        const float multiplier = make_mxfp4_scale(uint8_t(scale));
        for (size_t value = 0; value < 16; ++value) {
            table.values[scale][value] = multiplier * float(mxfp4_values[value]);
        }
    }
    return table;
}

constexpr Mxfp4ScaleTable mxfp4_scale_table = make_mxfp4_scale_table();
const Mxfp4WeightTable mxfp4_weight_table = make_mxfp4_weight_table();

constexpr size_t gpt_oss_mxfp4_rows = 2880;
constexpr size_t gpt_oss_mxfp4_columns = 2880;
constexpr size_t gpt_oss_mxfp4_blocks = gpt_oss_mxfp4_columns / 32;
constexpr size_t gpt_oss_mxfp4_row_bytes = gpt_oss_mxfp4_blocks * 17;

using f32x16 = float __attribute__((vector_size(64)));
using i32x16 = int32_t __attribute__((vector_size(64)));
using i8x16 = int8_t __attribute__((vector_size(16)));

[[gnu::always_inline]]
inline float reduce_f32x16(const f32x16 *value) {
    float sum = 0.0f;
    for (size_t i = 0; i < 16; ++i) {
        sum += (*value)[i];
    }
    return sum;
}

[[gnu::always_inline]]
inline void q8_0_vector_fma_block(const std::byte *source, const float *input, f32x16 &acc0, f32x16 &acc1) {
    i8x16 q0_i8;
    i8x16 q1_i8;
    f32x16 x0;
    f32x16 x1;

    std::memcpy(&q0_i8, source + 2, sizeof(q0_i8));
    std::memcpy(&q1_i8, source + 18, sizeof(q1_i8));
    std::memcpy(&x0, input, sizeof(x0));
    std::memcpy(&x1, input + 16, sizeof(x1));

    const i32x16 q0_i32 = __builtin_convertvector(q0_i8, i32x16);
    const i32x16 q1_i32 = __builtin_convertvector(q1_i8, i32x16);
    const float scale = fp16_to_fp32(load_u16(source));
    const f32x16 q0 = __builtin_convertvector(q0_i32, f32x16) * scale;
    const f32x16 q1 = __builtin_convertvector(q1_i32, f32x16) * scale;

    acc0 += q0 * x0;
    acc1 += q1 * x1;
}

#if defined(__x86_64__)
[[gnu::target("arch=x86-64-v4")]]
inline void mxfp4_avx512_fma_block(const uint8_t *source, const float *input, __m128i lut, __m128i mask, __m512 &acc) {
    const __m128i packed = _mm_loadu_si128((const __m128i *)(source + 1));
    const __m128i low_nibbles = _mm_and_si128(packed, mask);
    const __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    const __m128i low_i8 = _mm_shuffle_epi8(lut, low_nibbles);
    const __m128i high_i8 = _mm_shuffle_epi8(lut, high_nibbles);
    const __m512 low = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(low_i8));
    const __m512 high = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(high_i8));
    const __m512 scale = _mm512_set1_ps(mxfp4_scale_table.values[source[0]]);
    const __m512 x0 = _mm512_mul_ps(_mm512_loadu_ps(input), scale);
    const __m512 x1 = _mm512_mul_ps(_mm512_loadu_ps(input + 16), scale);
    acc = _mm512_fmadd_ps(low, x0, acc);
    acc = _mm512_fmadd_ps(high, x1, acc);
}
#endif

} // namespace

size_t tensor_row_bytes(const GgufTensor &tensor) {
    if (tensor.shape.empty() || tensor.shape[0] > SIZE_MAX) {
        throw std::runtime_error("invalid tensor shape");
    }
    const size_t n = size_t(tensor.shape[0]);
    switch (tensor.type) {
        case tensor_f32:
            return n * 4;
        case tensor_q8_0:
            if (n % 32) {
                throw std::runtime_error("Q8_0 row is not a multiple of 32");
            }
            return n / 32 * 34; // fp16 scale + 32 signed bytes
        case tensor_mxfp4:
            if (n % 32) {
                throw std::runtime_error("MXFP4 row is not a multiple of 32");
            }
            return n / 32 * 17; // E8M0 scale + 32 packed E2M1 values
        default:
            throw std::runtime_error("unsupported tensor type " + std::to_string(tensor.type));
    }
}

size_t tensor_bytes(const GgufTensor &tensor) {
    return tensor_row_bytes(tensor) * checked_product(tensor.shape, 1);
}

float fp16_to_fp32(uint16_t half) {
    return float(std::bit_cast<_Float16>(half));
}

float mxfp4_scale(uint8_t exponent) {
    return mxfp4_scale_table.values[exponent];
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void load_token_embedding_q8_0(const GgufTensor &tensor, size_t row, float *output) {
    const size_t n = size_t(tensor.shape[0]);
    const size_t row_bytes = n / 32 * 34;
    const std::byte *data = tensor.data + row * row_bytes;
    for (size_t block = 0; block < n / 32; ++block) {
        const std::byte *source = data + block * 34;
        const float scale = fp16_to_fp32(load_u16(source));
        const auto *quants = (const int8_t *)(source + 2);
        for (size_t i = 0; i < 32; ++i) {
            output[block * 32 + i] = scale * quants[i];
        }
    }
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void matvec_add_f32(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output) {
    const size_t columns = size_t(matrix.shape[0]);
    const size_t rows = size_t(matrix.shape[1]);

    const float *weights = (const float *)matrix.data;
    const float *bias_values = (const float *)bias.data;
    for (size_t row = 0; row < rows; ++row) {
        float sum = bias_values[row];
        for (size_t column = 0; column < columns; ++column) {
            sum += weights[row * columns + column] * input[column];
        }
        output[row] = sum;
    }
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void matvec_q8_0(const GgufTensor &matrix, const float *input, float *output) {
    const size_t columns = size_t(matrix.shape[0]);
    const size_t rows = size_t(matrix.shape[1]);
    const size_t row_bytes = columns / 32 * 34;

    for (size_t row = 0; row < rows; ++row) {
        const std::byte *data = matrix.data + row * row_bytes;
        f32x16 acc0 = {};
        f32x16 acc1 = {};
        for (size_t block = 0; block < columns / 32; ++block) {
            q8_0_vector_fma_block(data + block * 34, input + block * 32, acc0, acc1);
        }
        const f32x16 acc = acc0 + acc1;
        output[row] = reduce_f32x16(&acc);
    }
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void matvec_add_q8_0(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output) {
    const size_t columns = size_t(matrix.shape[0]);
    const size_t rows = size_t(matrix.shape[1]);
    const size_t row_bytes = columns / 32 * 34;
    const float *bias_values = (const float *)bias.data;

    for (size_t row = 0; row < rows; ++row) {
        const std::byte *data = matrix.data + row * row_bytes;
        f32x16 acc0 = {};
        f32x16 acc1 = {};
        for (size_t block = 0; block < columns / 32; ++block) {
            q8_0_vector_fma_block(data + block * 34, input + block * 32, acc0, acc1);
        }
        const f32x16 acc = acc0 + acc1;
        output[row] = bias_values[row] + reduce_f32x16(&acc);
    }
}

#if defined(__x86_64__)
[[gnu::target("arch=x86-64-v4")]]
void matvec_add_mxfp4_2880(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output,
                           size_t expert) {
    const std::byte *expert_data = matrix.data + expert * gpt_oss_mxfp4_rows * gpt_oss_mxfp4_row_bytes;
    const float *bias_values = (const float *)bias.data + expert * gpt_oss_mxfp4_rows;
    const __m128i lut = _mm_loadu_si128((const __m128i *)mxfp4_values);
    const __m128i mask = _mm_set1_epi8(0x0f);

    for (size_t row = 0; row < gpt_oss_mxfp4_rows; ++row) {
        const uint8_t *data = (const uint8_t *)(expert_data + row * gpt_oss_mxfp4_row_bytes);
        __m512 acc = _mm512_setzero_ps();
        for (size_t block = 0; block < gpt_oss_mxfp4_blocks; ++block) {
            mxfp4_avx512_fma_block(data + block * 17, input + block * 32, lut, mask, acc);
        }
        output[row] = bias_values[row] + _mm512_reduce_add_ps(acc);
    }
}

[[gnu::target("default")]]
#endif
void matvec_add_mxfp4_2880(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output,
                           size_t expert) {
    const std::byte *expert_data = matrix.data + expert * gpt_oss_mxfp4_rows * gpt_oss_mxfp4_row_bytes;
    const float *bias_values = (const float *)bias.data + expert * gpt_oss_mxfp4_rows;
    for (size_t row = 0; row < gpt_oss_mxfp4_rows; row += 4) {
        const auto *data0 = (const uint8_t *)(expert_data + (row + 0) * gpt_oss_mxfp4_row_bytes);
        const auto *data1 = (const uint8_t *)(expert_data + (row + 1) * gpt_oss_mxfp4_row_bytes);
        const auto *data2 = (const uint8_t *)(expert_data + (row + 2) * gpt_oss_mxfp4_row_bytes);
        const auto *data3 = (const uint8_t *)(expert_data + (row + 3) * gpt_oss_mxfp4_row_bytes);
        float sum0 = bias_values[row + 0];
        float sum1 = bias_values[row + 1];
        float sum2 = bias_values[row + 2];
        float sum3 = bias_values[row + 3];
        for (size_t block = 0; block < gpt_oss_mxfp4_blocks; ++block) {
            const uint8_t *source0 = data0 + block * 17;
            const uint8_t *source1 = data1 + block * 17;
            const uint8_t *source2 = data2 + block * 17;
            const uint8_t *source3 = data3 + block * 17;
            const float *weights0 = mxfp4_weight_table.values[source0[0]];
            const float *weights1 = mxfp4_weight_table.values[source1[0]];
            const float *weights2 = mxfp4_weight_table.values[source2[0]];
            const float *weights3 = mxfp4_weight_table.values[source3[0]];
            for (size_t i = 0; i < 16; ++i) {
                const uint8_t packed0 = source0[1 + i];
                const uint8_t packed1 = source1[1 + i];
                const uint8_t packed2 = source2[1 + i];
                const uint8_t packed3 = source3[1 + i];
                const float x0 = input[block * 32 + i];
                const float x1 = input[block * 32 + 16 + i];
                sum0 += weights0[packed0 & 0x0f] * x0;
                sum0 += weights0[packed0 >> 4] * x1;
                sum1 += weights1[packed1 & 0x0f] * x0;
                sum1 += weights1[packed1 >> 4] * x1;
                sum2 += weights2[packed2 & 0x0f] * x0;
                sum2 += weights2[packed2 >> 4] * x1;
                sum3 += weights3[packed3 & 0x0f] * x0;
                sum3 += weights3[packed3 >> 4] * x1;
            }
        }
        output[row + 0] = sum0;
        output[row + 1] = sum1;
        output[row + 2] = sum2;
        output[row + 3] = sum3;
    }
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void transpose_selected(const GgufTensor &matrix, const float *input, const size_t *tokens, size_t selected_count,
                        float *transposed_input) {
    const size_t columns = size_t(matrix.shape[0]);

    for (size_t s = 0; s < selected_count; ++s) {
        const size_t token = tokens[s];
        for (size_t column = 0; column < columns; ++column) {
            transposed_input[column * selected_count + s] = input[token * columns + column];
        }
    }
}

#if defined(__x86_64__)
[[gnu::target("arch=x86-64-v4")]]
void matmul_add_mxfp4_selected_transposed(const GgufTensor &matrix, const GgufTensor &bias,
                                          const float *transposed_input, const size_t *tokens, size_t selected_count,
                                          float *output, float *workspace, size_t expert) {
    const std::byte *expert_data = matrix.data + expert * gpt_oss_mxfp4_rows * gpt_oss_mxfp4_row_bytes;
    const float *bias_values = (const float *)bias.data + expert * gpt_oss_mxfp4_rows;
    for (size_t row = 0; row < gpt_oss_mxfp4_rows; row += 4) {
        const auto *data0 = (const uint8_t *)(expert_data + (row + 0) * gpt_oss_mxfp4_row_bytes);
        const auto *data1 = (const uint8_t *)(expert_data + (row + 1) * gpt_oss_mxfp4_row_bytes);
        const auto *data2 = (const uint8_t *)(expert_data + (row + 2) * gpt_oss_mxfp4_row_bytes);
        const auto *data3 = (const uint8_t *)(expert_data + (row + 3) * gpt_oss_mxfp4_row_bytes);
        float *workspace0 = workspace + selected_count * 0;
        float *workspace1 = workspace + selected_count * 1;
        float *workspace2 = workspace + selected_count * 2;
        float *workspace3 = workspace + selected_count * 3;

        size_t s = 0;
        for (; s + 16 <= selected_count; s += 16) {
            __m512 acc0 = _mm512_set1_ps(bias_values[row + 0]);
            __m512 acc1 = _mm512_set1_ps(bias_values[row + 1]);
            __m512 acc2 = _mm512_set1_ps(bias_values[row + 2]);
            __m512 acc3 = _mm512_set1_ps(bias_values[row + 3]);
            for (size_t block = 0; block < gpt_oss_mxfp4_blocks; ++block) {
                const uint8_t *source0 = data0 + block * 17;
                const uint8_t *source1 = data1 + block * 17;
                const uint8_t *source2 = data2 + block * 17;
                const uint8_t *source3 = data3 + block * 17;
                const float scale0 = mxfp4_scale_table.values[source0[0]];
                const float scale1 = mxfp4_scale_table.values[source1[0]];
                const float scale2 = mxfp4_scale_table.values[source2[0]];
                const float scale3 = mxfp4_scale_table.values[source3[0]];
                for (size_t i = 0; i < 16; ++i) {
                    const uint8_t packed0 = source0[1 + i];
                    const uint8_t packed1 = source1[1 + i];
                    const uint8_t packed2 = source2[1 + i];
                    const uint8_t packed3 = source3[1 + i];
                    const __m512 x0 = _mm512_loadu_ps(transposed_input + (block * 32 + i) * selected_count + s);
                    const __m512 x1 = _mm512_loadu_ps(transposed_input + (block * 32 + 16 + i) * selected_count + s);
                    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(scale0 * float(mxfp4_values[packed0 & 0x0f])), x0, acc0);
                    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(scale0 * float(mxfp4_values[packed0 >> 4])), x1, acc0);
                    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(scale1 * float(mxfp4_values[packed1 & 0x0f])), x0, acc1);
                    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(scale1 * float(mxfp4_values[packed1 >> 4])), x1, acc1);
                    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(scale2 * float(mxfp4_values[packed2 & 0x0f])), x0, acc2);
                    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(scale2 * float(mxfp4_values[packed2 >> 4])), x1, acc2);
                    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(scale3 * float(mxfp4_values[packed3 & 0x0f])), x0, acc3);
                    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(scale3 * float(mxfp4_values[packed3 >> 4])), x1, acc3);
                }
            }
            _mm512_storeu_ps(workspace0 + s, acc0);
            _mm512_storeu_ps(workspace1 + s, acc1);
            _mm512_storeu_ps(workspace2 + s, acc2);
            _mm512_storeu_ps(workspace3 + s, acc3);
        }
        if (s < selected_count) {
            const __mmask16 mask = __mmask16((1u << (selected_count - s)) - 1u);
            __m512 acc0 = _mm512_set1_ps(bias_values[row + 0]);
            __m512 acc1 = _mm512_set1_ps(bias_values[row + 1]);
            __m512 acc2 = _mm512_set1_ps(bias_values[row + 2]);
            __m512 acc3 = _mm512_set1_ps(bias_values[row + 3]);
            for (size_t block = 0; block < gpt_oss_mxfp4_blocks; ++block) {
                const uint8_t *source0 = data0 + block * 17;
                const uint8_t *source1 = data1 + block * 17;
                const uint8_t *source2 = data2 + block * 17;
                const uint8_t *source3 = data3 + block * 17;
                const float scale0 = mxfp4_scale_table.values[source0[0]];
                const float scale1 = mxfp4_scale_table.values[source1[0]];
                const float scale2 = mxfp4_scale_table.values[source2[0]];
                const float scale3 = mxfp4_scale_table.values[source3[0]];
                for (size_t i = 0; i < 16; ++i) {
                    const uint8_t packed0 = source0[1 + i];
                    const uint8_t packed1 = source1[1 + i];
                    const uint8_t packed2 = source2[1 + i];
                    const uint8_t packed3 = source3[1 + i];
                    const __m512 x0 =
                        _mm512_maskz_loadu_ps(mask, transposed_input + (block * 32 + i) * selected_count + s);
                    const __m512 x1 =
                        _mm512_maskz_loadu_ps(mask, transposed_input + (block * 32 + 16 + i) * selected_count + s);
                    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(scale0 * float(mxfp4_values[packed0 & 0x0f])), x0, acc0);
                    acc0 = _mm512_fmadd_ps(_mm512_set1_ps(scale0 * float(mxfp4_values[packed0 >> 4])), x1, acc0);
                    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(scale1 * float(mxfp4_values[packed1 & 0x0f])), x0, acc1);
                    acc1 = _mm512_fmadd_ps(_mm512_set1_ps(scale1 * float(mxfp4_values[packed1 >> 4])), x1, acc1);
                    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(scale2 * float(mxfp4_values[packed2 & 0x0f])), x0, acc2);
                    acc2 = _mm512_fmadd_ps(_mm512_set1_ps(scale2 * float(mxfp4_values[packed2 >> 4])), x1, acc2);
                    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(scale3 * float(mxfp4_values[packed3 & 0x0f])), x0, acc3);
                    acc3 = _mm512_fmadd_ps(_mm512_set1_ps(scale3 * float(mxfp4_values[packed3 >> 4])), x1, acc3);
                }
            }
            _mm512_mask_storeu_ps(workspace0 + s, mask, acc0);
            _mm512_mask_storeu_ps(workspace1 + s, mask, acc1);
            _mm512_mask_storeu_ps(workspace2 + s, mask, acc2);
            _mm512_mask_storeu_ps(workspace3 + s, mask, acc3);
        }

        for (size_t t = 0; t < selected_count; ++t) {
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 0] = workspace0[t];
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 1] = workspace1[t];
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 2] = workspace2[t];
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 3] = workspace3[t];
        }
    }
}

[[gnu::target("default")]]
#endif
void matmul_add_mxfp4_selected_transposed(const GgufTensor &matrix, const GgufTensor &bias,
                                          const float *transposed_input, const size_t *tokens, size_t selected_count,
                                          float *output, float *workspace, size_t expert) {
    const std::byte *expert_data = matrix.data + expert * gpt_oss_mxfp4_rows * gpt_oss_mxfp4_row_bytes;
    const float *bias_values = (const float *)bias.data + expert * gpt_oss_mxfp4_rows;
    for (size_t row = 0; row < gpt_oss_mxfp4_rows; row += 4) {
        const auto *data0 = (const uint8_t *)(expert_data + (row + 0) * gpt_oss_mxfp4_row_bytes);
        const auto *data1 = (const uint8_t *)(expert_data + (row + 1) * gpt_oss_mxfp4_row_bytes);
        const auto *data2 = (const uint8_t *)(expert_data + (row + 2) * gpt_oss_mxfp4_row_bytes);
        const auto *data3 = (const uint8_t *)(expert_data + (row + 3) * gpt_oss_mxfp4_row_bytes);
        float *workspace0 = workspace + selected_count * 0;
        float *workspace1 = workspace + selected_count * 1;
        float *workspace2 = workspace + selected_count * 2;
        float *workspace3 = workspace + selected_count * 3;

        const float bias0 = bias_values[row + 0];
        const float bias1 = bias_values[row + 1];
        const float bias2 = bias_values[row + 2];
        const float bias3 = bias_values[row + 3];
        for (size_t s = 0; s < selected_count; ++s) {
            workspace0[s] = bias0;
            workspace1[s] = bias1;
            workspace2[s] = bias2;
            workspace3[s] = bias3;
        }
        for (size_t block = 0; block < gpt_oss_mxfp4_blocks; ++block) {
            const uint8_t *source0 = data0 + block * 17;
            const uint8_t *source1 = data1 + block * 17;
            const uint8_t *source2 = data2 + block * 17;
            const uint8_t *source3 = data3 + block * 17;
            const float scale0 = mxfp4_scale_table.values[source0[0]];
            const float scale1 = mxfp4_scale_table.values[source1[0]];
            const float scale2 = mxfp4_scale_table.values[source2[0]];
            const float scale3 = mxfp4_scale_table.values[source3[0]];
            for (size_t i = 0; i < 16; ++i) {
                const uint8_t packed0 = source0[1 + i];
                const uint8_t packed1 = source1[1 + i];
                const uint8_t packed2 = source2[1 + i];
                const uint8_t packed3 = source3[1 + i];
                const float weight00 = scale0 * mxfp4_values[packed0 & 0x0f];
                const float weight01 = scale0 * mxfp4_values[packed0 >> 4];
                const float weight10 = scale1 * mxfp4_values[packed1 & 0x0f];
                const float weight11 = scale1 * mxfp4_values[packed1 >> 4];
                const float weight20 = scale2 * mxfp4_values[packed2 & 0x0f];
                const float weight21 = scale2 * mxfp4_values[packed2 >> 4];
                const float weight30 = scale3 * mxfp4_values[packed3 & 0x0f];
                const float weight31 = scale3 * mxfp4_values[packed3 >> 4];
                const float *input0 = transposed_input + (block * 32 + i) * selected_count;
                const float *input1 = transposed_input + (block * 32 + 16 + i) * selected_count;
                for (size_t s = 0; s < selected_count; ++s) {
                    const float x0 = input0[s];
                    const float x1 = input1[s];
                    workspace0[s] += weight00 * x0;
                    workspace0[s] += weight01 * x1;
                    workspace1[s] += weight10 * x0;
                    workspace1[s] += weight11 * x1;
                    workspace2[s] += weight20 * x0;
                    workspace2[s] += weight21 * x1;
                    workspace3[s] += weight30 * x0;
                    workspace3[s] += weight31 * x1;
                }
            }
        }

        for (size_t t = 0; t < selected_count; ++t) {
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 0] = workspace0[t];
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 1] = workspace1[t];
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 2] = workspace2[t];
            output[tokens[t] * gpt_oss_mxfp4_rows + row + 3] = workspace3[t];
        }
    }
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void matmul_add_f32(const GgufTensor &matrix, const GgufTensor &bias, const float *input, size_t token_count,
                    float *output, float *workspace, float *transposed_input) {
    const size_t columns = size_t(matrix.shape[0]);
    const size_t rows = size_t(matrix.shape[1]);
    const float *bias_values = (const float *)bias.data;

    for (size_t token = 0; token < token_count; ++token) {
        for (size_t column = 0; column < columns; ++column) {
            transposed_input[column * token_count + token] = input[token * columns + column];
        }
    }

    const float *weights = (const float *)matrix.data;
    for (size_t row = 0; row < rows; ++row) {
        const float bias_value = bias_values[row];
        for (size_t token = 0; token < token_count; ++token) {
            workspace[token] = bias_value;
        }
        for (size_t column = 0; column < columns; ++column) {
            const float weight = weights[row * columns + column];
            const float *column_input = transposed_input + column * token_count;
            for (size_t token = 0; token < token_count; ++token) {
                workspace[token] += weight * column_input[token];
            }
        }
        for (size_t token = 0; token < token_count; ++token) {
            output[token * rows + row] = workspace[token];
        }
    }
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void matmul_add_q8_0(const GgufTensor &matrix, const GgufTensor &bias, const float *input, size_t token_count,
                     float *output, float *workspace, float *transposed_input) {
    const size_t columns = size_t(matrix.shape[0]);
    const size_t rows = size_t(matrix.shape[1]);
    const size_t row_bytes = columns / 32 * 34;
    const float *bias_values = (const float *)bias.data;

    for (size_t token = 0; token < token_count; ++token) {
        for (size_t column = 0; column < columns; ++column) {
            transposed_input[column * token_count + token] = input[token * columns + column];
        }
    }

    const std::byte *expert_data = matrix.data;
    for (size_t row = 0; row < rows; ++row) {
        const std::byte *data = expert_data + row * row_bytes;
        const float bias_value = bias_values[row];
        for (size_t token = 0; token < token_count; ++token) {
            workspace[token] = bias_value;
        }
        for (size_t block = 0; block < columns / 32; ++block) {
            const std::byte *source = data + block * 34;
            const float scale = fp16_to_fp32(load_u16(source));
            const auto *quants = (const int8_t *)(source + 2);
            for (size_t i = 0; i < 32; ++i) {
                const float weight = scale * float(quants[i]);
                const float *column_input = transposed_input + (block * 32 + i) * token_count;
                for (size_t token = 0; token < token_count; ++token) {
                    workspace[token] += weight * column_input[token];
                }
            }
        }
        for (size_t token = 0; token < token_count; ++token) {
            output[token * rows + row] = workspace[token];
        }
    }
}

[[gnu::target_clones("arch=x86-64-v4", "default")]]
void rms_norm(const float *input, const GgufTensor &weights, float epsilon, float *output) {
    const size_t n = size_t(weights.shape[0]);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += input[i] * input[i];
    }
    const float scale = 1.0f / std::sqrt(sum / float(n) + epsilon);
    const float *weight = (const float *)weights.data;
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}
