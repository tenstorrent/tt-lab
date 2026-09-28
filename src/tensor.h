// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gguf.h"

#include <cstddef>
#include <cstdint>

constexpr uint32_t tensor_f32 = 0;
constexpr uint32_t tensor_q8_0 = 8;
constexpr uint32_t tensor_mxfp4 = 39;

// The GGUF matrices used by gpt-oss are stored row by row. shape[0] is the
// number of inputs and shape[1] is the number of outputs. A third dimension,
// when present, selects an expert.
size_t tensor_row_bytes(const GgufTensor &tensor);
size_t tensor_bytes(const GgufTensor &tensor);

float fp16_to_fp32(uint16_t half);
float mxfp4_scale(uint8_t exponent);

void load_token_embedding_q8_0(const GgufTensor &tensor, size_t row, float *output);
void matvec_add_f32(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output);
void matvec_q8_0(const GgufTensor &matrix, const float *input, float *output);
void matvec_add_q8_0(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output);
#if defined(__x86_64__)
[[gnu::target("arch=x86-64-v4")]]
void matvec_add_mxfp4_2880(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output,
                           size_t expert);
[[gnu::target("default")]]
#endif
void matvec_add_mxfp4_2880(const GgufTensor &matrix, const GgufTensor &bias, const float *input, float *output,
                           size_t expert = 0);
void matmul_add_f32(const GgufTensor &matrix, const GgufTensor &bias, const float *input, size_t token_count,
                    float *output, float *workspace, float *transposed_input);
void matmul_add_q8_0(const GgufTensor &matrix, const GgufTensor &bias, const float *input, size_t token_count,
                     float *output, float *workspace, float *transposed_input);
void transpose_selected(const GgufTensor &matrix, const float *input, const size_t *tokens, size_t selected_count,
                        float *transposed_input);
#if defined(__x86_64__)
[[gnu::target("arch=x86-64-v4")]]
void matmul_add_mxfp4_selected_transposed(const GgufTensor &matrix, const GgufTensor &bias,
                                          const float *transposed_input, const size_t *tokens, size_t selected_count,
                                          float *output, float *workspace, size_t expert);
[[gnu::target("default")]]
#endif
void matmul_add_mxfp4_selected_transposed(const GgufTensor &matrix, const GgufTensor &bias,
                                          const float *transposed_input, const size_t *tokens, size_t selected_count,
                                          float *output, float *workspace, size_t expert = 0);
void rms_norm(const float *input, const GgufTensor &weights, float epsilon, float *output);
