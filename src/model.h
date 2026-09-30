// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gguf.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

class GptOssModel {
  public:
    static constexpr size_t context_length = 131072;

    explicit GptOssModel(const GgufFile &file, size_t token_capacity = 0, std::string dump_directory = {},
                         size_t batch_capacity = 0);
    ~GptOssModel();
    GptOssModel(const GptOssModel &) = delete;
    GptOssModel &operator=(const GptOssModel &) = delete;

    // Runs the prompt as a layer-major sequence pass and returns logits for the last token.
    const float *prefill(const int32_t *tokens, size_t count);

    // Advances the model by one token and returns logits for the following token.
    const float *forward(int32_t token, size_t position);
    int32_t greedy(const float *logits) const;

  private:
    static constexpr size_t layer_capacity = 36;
    static constexpr size_t embedding_length = 2880;
    static constexpr size_t vocabulary_size = 201088;
    static constexpr size_t query_heads = 64;
    static constexpr size_t kv_heads = 8;
    static constexpr size_t head_size = 64;
    static constexpr size_t query_width = query_heads * head_size;
    static constexpr size_t kv_width = kv_heads * head_size;
    static constexpr size_t expert_capacity = 128;
    static constexpr size_t experts_used = 4;
    static constexpr size_t expert_hidden_length = 2880;
    static constexpr size_t sliding_window = 128;
    static constexpr float norm_epsilon = 1.0e-5f;
    static constexpr float rope_base = 150000.0f;
    static constexpr float rope_scale = 1.0f / 32.0f;
    static constexpr size_t rope_original_context = 4096;
    static constexpr float rope_beta_fast = 32.0f;
    static constexpr float rope_beta_slow = 1.0f;

    struct Layer {
        const GgufTensor *attention_norm;
        const GgufTensor *post_attention_norm;
        const GgufTensor *query_weight;
        const GgufTensor *query_bias;
        const GgufTensor *key_weight;
        const GgufTensor *key_bias;
        const GgufTensor *value_weight;
        const GgufTensor *value_bias;
        const GgufTensor *attention_output_weight;
        const GgufTensor *attention_output_bias;
        const GgufTensor *attention_sinks;
        const GgufTensor *router_weight;
        const GgufTensor *router_bias;
        const GgufTensor *gate_weight;
        const GgufTensor *gate_bias;
        const GgufTensor *up_weight;
        const GgufTensor *up_bias;
        const GgufTensor *down_weight;
        const GgufTensor *down_bias;
        std::vector<float> key_cache;
        std::vector<float> value_cache;
    };

    const GgufFile &file_;
    const GgufTensor *embeddings_;
    const GgufTensor *output_norm_;
    const GgufTensor *output_;
    std::array<Layer, layer_capacity> layers_;

    size_t layer_count_;
    size_t token_capacity_;
    size_t batch_capacity_;
    size_t expert_count_;
    int dump_fd_;
    int manifest_fd_;

    std::array<float, vocabulary_size> logits_;
    std::vector<float> attention_scores_;
    std::vector<float> x_sequence_;
    std::vector<float> normalized_sequence_;
    std::vector<float> query_sequence_;
    std::vector<float> key_sequence_;
    std::vector<float> value_sequence_;
    std::vector<float> attended_sequence_;
    std::vector<float> projected_sequence_;
    std::vector<float> residual_sequence_;
    std::vector<float> moe_sequence_;
    std::vector<float> router_sequence_;
    std::vector<size_t> expert_indices_sequence_;
    std::vector<float> expert_weights_sequence_;
    std::vector<float> gate_sequence_;
    std::vector<float> up_sequence_;
    std::vector<float> hidden_sequence_;
    std::vector<float> down_sequence_;
    std::vector<float> gemm_workspace_;
    std::vector<float> matmul_input_transposed_;
    std::array<float, head_size / 2> rope_frequencies_;
    std::array<size_t, expert_capacity> expert_token_counts_;
    std::vector<size_t> expert_token_buckets_;
    std::vector<float> expert_weight_buckets_;
    std::array<float, embedding_length> x_;
    std::array<float, embedding_length> normalized_;
    std::array<float, query_width> query_;
    std::array<float, kv_width> key_;
    std::array<float, kv_width> value_;
    std::array<float, query_width> attended_;
    std::array<float, embedding_length> projected_;
    std::array<float, embedding_length> residual_;
    std::array<float, embedding_length> moe_output_;
    std::array<float, expert_capacity> router_;
    std::array<size_t, expert_capacity> expert_indices_;
    std::array<float, experts_used> expert_weights_;
    std::array<float, expert_hidden_length> gate_;
    std::array<float, expert_hidden_length> up_;
    std::array<float, expert_hidden_length> hidden_;
    std::array<float, embedding_length> down_;

    const GgufTensor *require_tensor(const std::string &name, uint32_t type, std::initializer_list<size_t> shape) const;
    void apply_rope(float *vector, size_t heads, size_t position) const;
    void store_kv(Layer &layer, size_t position, const float *key, const float *value);
    void attention_from_cache(const Layer &layer, size_t layer_index, size_t position, const float *query,
                              float *output);
    void moe(const Layer &layer, const float *input, float *output, size_t layer_index, size_t position);
    void moe_batch(const Layer &layer, const float *input, float *output, size_t layer_index, size_t count);
    void dump(size_t position, size_t layer, std::string_view name, const float *values, size_t count) const;
};
