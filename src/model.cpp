// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "model.h"

#include "tensor.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <unistd.h>

namespace {

constexpr size_t mxfp4_matvec_selected_max = 4;

void require_integer(const GgufFile &file, std::string_view key, uint64_t expected) {
    if (file.integer(key) != expected) {
        throw std::runtime_error("unsupported " + std::string(key));
    }
}

void require_number(const GgufFile &file, std::string_view key, double expected) {
    if (file.number(key) != expected) {
        throw std::runtime_error("unsupported " + std::string(key));
    }
}

float dot(const float *a, const float *b, size_t n) {
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

bool write_all(int fd, const void *bytes, size_t count) {
    const char *data = (const char *)bytes;
    while (count > 0) {
        const ssize_t written = write(fd, data, count);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (written == 0) {
            return false;
        }
        data += written;
        count -= size_t(written);
    }
    return true;
}

} // namespace

GptOssModel::GptOssModel(const GgufFile &file, size_t token_capacity, std::string dump_directory, size_t batch_capacity)
    : file_(file), layer_count_(0), token_capacity_(token_capacity), batch_capacity_(batch_capacity), dump_fd_(-1),
      manifest_fd_(-1) {
    if (file.string("general.architecture") != "gpt-oss") {
        throw std::runtime_error("model architecture is not gpt-oss");
    }
    const uint64_t layer_count = file.integer("gpt-oss.block_count");
    if (layer_count != 24 && layer_count != 36) {
        throw std::runtime_error("only gpt-oss-20b and gpt-oss-120b are supported");
    }
    layer_count_ = size_t(layer_count);

    require_integer(file, "gpt-oss.context_length", context_length);
    require_integer(file, "gpt-oss.embedding_length", embedding_length);
    require_integer(file, "gpt-oss.feed_forward_length", embedding_length);
    require_integer(file, "gpt-oss.attention.head_count", query_heads);
    require_integer(file, "gpt-oss.attention.head_count_kv", kv_heads);
    require_integer(file, "gpt-oss.attention.key_length", head_size);
    require_integer(file, "gpt-oss.attention.value_length", head_size);
    require_integer(file, "gpt-oss.expert_used_count", experts_used);
    require_integer(file, "gpt-oss.expert_feed_forward_length", expert_hidden_length);
    require_integer(file, "gpt-oss.attention.sliding_window", sliding_window);
    require_number(file, "gpt-oss.attention.layer_norm_rms_epsilon", norm_epsilon);
    require_number(file, "gpt-oss.rope.freq_base", rope_base);
    require_number(file, "gpt-oss.rope.scaling.factor", 32.0);
    require_integer(file, "gpt-oss.rope.scaling.original_context_length", rope_original_context);
    require_number(file, "gpt-oss.rope.scaling.yarn_beta_fast", rope_beta_fast);
    require_number(file, "gpt-oss.rope.scaling.yarn_beta_slow", rope_beta_slow);

    const uint64_t expert_count = file.integer("gpt-oss.expert_count");
    if (expert_count != 32 && expert_count != expert_capacity) {
        throw std::runtime_error("unsupported expert count");
    }
    expert_count_ = size_t(expert_count);
    if (token_capacity_ > context_length) {
        throw std::runtime_error("token capacity exceeds model context");
    }
    if (batch_capacity_ == 0) {
        batch_capacity_ = token_capacity_;
    }
    if (batch_capacity_ > token_capacity_) {
        throw std::runtime_error("batch capacity exceeds token capacity");
    }
    if (!dump_directory.empty()) {
        std::filesystem::create_directories(dump_directory);
        dump_fd_ = open(dump_directory.c_str(), O_RDONLY | O_DIRECTORY);
        if (dump_fd_ < 0) {
            throw std::runtime_error("cannot open tensor dump directory: " + dump_directory);
        }
        manifest_fd_ = openat(dump_fd_, "manifest.tsv", O_WRONLY | O_CREAT | O_APPEND, 0666);
        if (manifest_fd_ < 0) {
            throw std::runtime_error("cannot open tensor dump manifest: " + dump_directory);
        }
    }

    embeddings_ = require_tensor("token_embd.weight", tensor_q8_0, {embedding_length, vocabulary_size});
    output_norm_ = require_tensor("output_norm.weight", tensor_f32, {embedding_length});
    output_ = require_tensor("output.weight", tensor_q8_0, {embedding_length, vocabulary_size});

    const float corr = float(head_size) *
                       std::log(float(rope_original_context) / (rope_beta_fast * 2.0f * std::numbers::pi_v<float>)) /
                       (2.0f * std::log(rope_base));
    const float corr_slow =
        float(head_size) *
        std::log(float(rope_original_context) / (rope_beta_slow * 2.0f * std::numbers::pi_v<float>)) /
        (2.0f * std::log(rope_base));
    const float low = std::max(0.0f, std::floor(corr));
    const float high = std::min(float(head_size - 1), std::ceil(corr_slow));
    for (size_t pair = 0; pair < head_size / 2; ++pair) {
        const size_t dimension = pair * 2;
        const float theta_extrap = std::pow(rope_base, -float(dimension) / float(head_size));
        const float y = (float(pair) - low) / std::max(0.001f, high - low);
        const float ramp = 1.0f - std::clamp(y, 0.0f, 1.0f);
        rope_frequencies_[pair] = rope_scale * theta_extrap * (1.0f - ramp) + theta_extrap * ramp;
    }

    for (size_t i = 0; i < layer_count_; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        Layer &layer = layers_[i];
        layer.attention_norm = require_tensor(prefix + "attn_norm.weight", tensor_f32, {embedding_length});
        layer.post_attention_norm =
            require_tensor(prefix + "post_attention_norm.weight", tensor_f32, {embedding_length});
        layer.query_weight = require_tensor(prefix + "attn_q.weight", tensor_q8_0, {embedding_length, query_width});
        layer.query_bias = require_tensor(prefix + "attn_q.bias", tensor_f32, {query_width});
        layer.key_weight = require_tensor(prefix + "attn_k.weight", tensor_q8_0, {embedding_length, kv_width});
        layer.key_bias = require_tensor(prefix + "attn_k.bias", tensor_f32, {kv_width});
        layer.value_weight = require_tensor(prefix + "attn_v.weight", tensor_q8_0, {embedding_length, kv_width});
        layer.value_bias = require_tensor(prefix + "attn_v.bias", tensor_f32, {kv_width});
        layer.attention_output_weight =
            require_tensor(prefix + "attn_output.weight", tensor_q8_0, {query_width, embedding_length});
        layer.attention_output_bias = require_tensor(prefix + "attn_output.bias", tensor_f32, {embedding_length});
        layer.attention_sinks = require_tensor(prefix + "attn_sinks.weight", tensor_f32, {query_heads});
        layer.router_weight =
            require_tensor(prefix + "ffn_gate_inp.weight", tensor_f32, {embedding_length, expert_count_});
        layer.router_bias = require_tensor(prefix + "ffn_gate_inp.bias", tensor_f32, {expert_count_});
        layer.gate_weight = require_tensor(prefix + "ffn_gate_exps.weight", tensor_mxfp4,
                                           {embedding_length, expert_hidden_length, expert_count_});
        layer.gate_bias =
            require_tensor(prefix + "ffn_gate_exps.bias", tensor_f32, {expert_hidden_length, expert_count_});
        layer.up_weight = require_tensor(prefix + "ffn_up_exps.weight", tensor_mxfp4,
                                         {embedding_length, expert_hidden_length, expert_count_});
        layer.up_bias = require_tensor(prefix + "ffn_up_exps.bias", tensor_f32, {expert_hidden_length, expert_count_});
        layer.down_weight = require_tensor(prefix + "ffn_down_exps.weight", tensor_mxfp4,
                                           {expert_hidden_length, embedding_length, expert_count_});
        layer.down_bias = require_tensor(prefix + "ffn_down_exps.bias", tensor_f32, {embedding_length, expert_count_});
        if (token_capacity_ > 0) {
            layer.key_cache.resize(token_capacity_ * kv_heads * head_size);
            layer.value_cache.resize(token_capacity_ * kv_heads * head_size);
        }
    }
    if (token_capacity_ > 0) {
        attention_scores_.resize(token_capacity_);
        x_sequence_.resize(batch_capacity_ * embedding_length);
        normalized_sequence_.resize(batch_capacity_ * embedding_length);
        query_sequence_.resize(batch_capacity_ * query_heads * head_size);
        key_sequence_.resize(batch_capacity_ * kv_heads * head_size);
        value_sequence_.resize(batch_capacity_ * kv_heads * head_size);
        attended_sequence_.resize(batch_capacity_ * query_heads * head_size);
        projected_sequence_.resize(batch_capacity_ * embedding_length);
        residual_sequence_.resize(batch_capacity_ * embedding_length);
        moe_sequence_.resize(batch_capacity_ * embedding_length);
        router_sequence_.resize(batch_capacity_ * expert_count_);
        expert_indices_sequence_.resize(batch_capacity_ * experts_used);
        expert_weights_sequence_.resize(batch_capacity_ * experts_used);
        gate_sequence_.resize(batch_capacity_ * expert_hidden_length);
        up_sequence_.resize(batch_capacity_ * expert_hidden_length);
        hidden_sequence_.resize(batch_capacity_ * expert_hidden_length);
        down_sequence_.resize(batch_capacity_ * embedding_length);
        gemm_workspace_.resize(batch_capacity_ * 4);
        matmul_input_transposed_.resize(batch_capacity_ * query_width);
        expert_token_buckets_.resize(batch_capacity_ * expert_count_);
        expert_weight_buckets_.resize(batch_capacity_ * expert_count_);
    }
}

GptOssModel::~GptOssModel() {
    if (manifest_fd_ >= 0) {
        close(manifest_fd_);
    }
    if (dump_fd_ >= 0) {
        close(dump_fd_);
    }
}

const GgufTensor *GptOssModel::require_tensor(const std::string &name, uint32_t type,
                                              std::initializer_list<size_t> shape) const {
    const GgufTensor *result = file_.tensor(name);
    if (!result) {
        throw std::runtime_error("missing tensor " + name);
    }
    if (result->type != type) {
        throw std::runtime_error("wrong type for tensor " + name);
    }
    if (result->shape.size() != shape.size()) {
        throw std::runtime_error("wrong rank for tensor " + name);
    }
    size_t i = 0;
    for (size_t expected : shape) {
        if (result->shape[i++] != expected) {
            throw std::runtime_error("wrong shape for tensor " + name);
        }
    }
    return result;
}

void GptOssModel::apply_rope(float *vector, size_t heads, size_t position) const {
    const float magnitude = 1.0f + 0.1f * std::log(1.0f / rope_scale);

    for (size_t head = 0; head < heads; ++head) {
        float *values = vector + head * head_size;
        for (size_t pair = 0; pair < head_size / 2; ++pair) {
            const float theta = float(position) * rope_frequencies_[pair];
            const float cosine = std::cos(theta) * magnitude;
            const float sine = std::sin(theta) * magnitude;
            const float x = values[pair];
            const float yv = values[pair + head_size / 2];
            values[pair] = x * cosine - yv * sine;
            values[pair + head_size / 2] = x * sine + yv * cosine;
        }
    }
}

void GptOssModel::store_kv(Layer &layer, size_t position, const float *key, const float *value) {
    if (position >= token_capacity_) {
        throw std::runtime_error("position exceeds startup token capacity");
    }
    std::memcpy(layer.key_cache.data() + position * kv_width, key, kv_width * sizeof(float));
    std::memcpy(layer.value_cache.data() + position * kv_width, value, kv_width * sizeof(float));
}

void GptOssModel::attention_from_cache(const Layer &layer, size_t layer_index, size_t position, const float *query,
                                       float *output) {
    if (position >= token_capacity_) {
        throw std::runtime_error("position exceeds startup token capacity");
    }
    const size_t first = layer_index % 2 == 0 && position + 1 > sliding_window ? position + 1 - sliding_window : 0;
    const size_t count = position + 1 - first;
    float *scores = attention_scores_.data();
    const float scale = 1.0f / std::sqrt(float(head_size));
    const float *sinks = (const float *)layer.attention_sinks->data;

    for (size_t head = 0; head < query_heads; ++head) {
        const size_t kv_head = head / (query_heads / kv_heads);
        const float *q = query + head * head_size;
        float maximum = sinks[head];
        for (size_t i = 0; i < count; ++i) {
            const float *k = layer.key_cache.data() + ((first + i) * kv_heads + kv_head) * head_size;
            scores[i] = dot(q, k, head_size) * scale;
            maximum = std::max(maximum, scores[i]);
        }
        float denominator = std::exp(sinks[head] - maximum);
        for (size_t i = 0; i < count; ++i) {
            scores[i] = std::exp(scores[i] - maximum);
            denominator += scores[i];
        }
        float *result = output + head * head_size;
        std::fill(result, result + head_size, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            const float weight = scores[i] / denominator;
            const float *v = layer.value_cache.data() + ((first + i) * kv_heads + kv_head) * head_size;
            for (size_t d = 0; d < head_size; ++d) {
                result[d] += weight * v[d];
            }
        }
    }
}

void GptOssModel::moe(const Layer &layer, const float *input, float *output, size_t layer_index, size_t position) {
    matvec_add_f32(*layer.router_weight, *layer.router_bias, input, router_.data());
    dump(position, layer_index, "router", router_.data(), expert_count_);

    for (size_t i = 0; i < expert_count_; ++i) {
        expert_indices_[i] = i;
    }
    for (size_t i = 0; i < experts_used; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < expert_count_; ++j) {
            if (router_[expert_indices_[j]] > router_[expert_indices_[best]]) {
                best = j;
            }
        }
        const size_t tmp = expert_indices_[i];
        expert_indices_[i] = expert_indices_[best];
        expert_indices_[best] = tmp;
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < experts_used; ++i) {
        maximum = std::max(maximum, router_[expert_indices_[i]]);
    }
    float weight_sum = 0.0f;
    for (size_t i = 0; i < experts_used; ++i) {
        expert_weights_[i] = std::exp(router_[expert_indices_[i]] - maximum);
        weight_sum += expert_weights_[i];
    }
    for (size_t i = 0; i < experts_used; ++i) {
        expert_weights_[i] /= weight_sum;
    }

    std::fill(output, output + embedding_length, 0.0f);
    for (size_t selected = 0; selected < experts_used; ++selected) {
        const size_t expert = expert_indices_[selected];
        matvec_add_mxfp4_2880(*layer.gate_weight, *layer.gate_bias, input, gate_.data(), expert);
        matvec_add_mxfp4_2880(*layer.up_weight, *layer.up_bias, input, up_.data(), expert);
        for (size_t i = 0; i < expert_hidden_length; ++i) {
            const float x = std::min(gate_[i], 7.0f);
            const float y = std::clamp(up_[i], -7.0f, 7.0f);
            hidden_[i] = (x / (1.0f + std::exp(-1.702f * x))) * (y + 1.0f);
        }
        matvec_add_mxfp4_2880(*layer.down_weight, *layer.down_bias, hidden_.data(), down_.data(), expert);
        for (size_t i = 0; i < embedding_length; ++i) {
            output[i] += expert_weights_[selected] * down_[i];
        }
    }
}

void GptOssModel::moe_batch(const Layer &layer, const float *input, float *output, size_t layer_index, size_t count) {
    matmul_add_f32(*layer.router_weight, *layer.router_bias, input, count, router_sequence_.data(),
                   gemm_workspace_.data(), matmul_input_transposed_.data());

    for (size_t position = 0; position < count; ++position) {
        const float *router = router_sequence_.data() + position * expert_count_;
        size_t *experts = expert_indices_sequence_.data() + position * experts_used;
        float *weights = expert_weights_sequence_.data() + position * experts_used;
        dump(position, layer_index, "router", router, expert_count_);

        for (size_t i = 0; i < expert_count_; ++i) {
            expert_indices_[i] = i;
        }
        for (size_t i = 0; i < experts_used; ++i) {
            size_t best = i;
            for (size_t j = i + 1; j < expert_count_; ++j) {
                if (router[expert_indices_[j]] > router[expert_indices_[best]]) {
                    best = j;
                }
            }
            const size_t tmp = expert_indices_[i];
            expert_indices_[i] = expert_indices_[best];
            expert_indices_[best] = tmp;
            experts[i] = expert_indices_[i];
        }

        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < experts_used; ++i) {
            maximum = std::max(maximum, router[experts[i]]);
        }
        float weight_sum = 0.0f;
        for (size_t i = 0; i < experts_used; ++i) {
            weights[i] = std::exp(router[experts[i]] - maximum);
            weight_sum += weights[i];
        }
        for (size_t i = 0; i < experts_used; ++i) {
            weights[i] /= weight_sum;
        }
    }

    std::fill(output, output + count * embedding_length, 0.0f);
    for (size_t expert = 0; expert < expert_count_; ++expert) {
        expert_token_counts_[expert] = 0;
    }
    for (size_t position = 0; position < count; ++position) {
        const size_t *experts = expert_indices_sequence_.data() + position * experts_used;
        const float *weights = expert_weights_sequence_.data() + position * experts_used;
        for (size_t rank = 0; rank < experts_used; ++rank) {
            const size_t expert = experts[rank];
            const size_t offset = expert * batch_capacity_ + expert_token_counts_[expert];
            expert_token_buckets_[offset] = position;
            expert_weight_buckets_[offset] = weights[rank];
            ++expert_token_counts_[expert];
        }
    }

    for (size_t expert = 0; expert < expert_count_; ++expert) {
        const size_t selected_count = expert_token_counts_[expert];
        if (selected_count == 0) {
            continue;
        }
        const size_t *selected_tokens = expert_token_buckets_.data() + expert * batch_capacity_;
        const float *selected_weights = expert_weight_buckets_.data() + expert * batch_capacity_;

        if (selected_count <= mxfp4_matvec_selected_max) {
            for (size_t s = 0; s < selected_count; ++s) {
                const size_t position = selected_tokens[s];
                matvec_add_mxfp4_2880(*layer.gate_weight, *layer.gate_bias, input + position * embedding_length,
                                      gate_sequence_.data() + position * expert_hidden_length, expert);
                matvec_add_mxfp4_2880(*layer.up_weight, *layer.up_bias, input + position * embedding_length,
                                      up_sequence_.data() + position * expert_hidden_length, expert);
            }
        } else {
            transpose_selected(*layer.gate_weight, input, selected_tokens, selected_count,
                               matmul_input_transposed_.data());
            matmul_add_mxfp4_selected_transposed(*layer.gate_weight, *layer.gate_bias, matmul_input_transposed_.data(),
                                                 selected_tokens, selected_count, gate_sequence_.data(),
                                                 gemm_workspace_.data(), expert);
            matmul_add_mxfp4_selected_transposed(*layer.up_weight, *layer.up_bias, matmul_input_transposed_.data(),
                                                 selected_tokens, selected_count, up_sequence_.data(),
                                                 gemm_workspace_.data(), expert);
        }

        for (size_t s = 0; s < selected_count; ++s) {
            const size_t position = selected_tokens[s];
            float *gate = gate_sequence_.data() + position * expert_hidden_length;
            float *up = up_sequence_.data() + position * expert_hidden_length;
            float *hidden = hidden_sequence_.data() + position * expert_hidden_length;
            for (size_t i = 0; i < expert_hidden_length; ++i) {
                const float x = std::min(gate[i], 7.0f);
                const float y = std::clamp(up[i], -7.0f, 7.0f);
                hidden[i] = (x / (1.0f + std::exp(-1.702f * x))) * (y + 1.0f);
            }
        }

        if (selected_count <= mxfp4_matvec_selected_max) {
            for (size_t s = 0; s < selected_count; ++s) {
                const size_t position = selected_tokens[s];
                matvec_add_mxfp4_2880(*layer.down_weight, *layer.down_bias,
                                      hidden_sequence_.data() + position * expert_hidden_length,
                                      down_sequence_.data() + position * embedding_length, expert);
            }
        } else {
            transpose_selected(*layer.down_weight, hidden_sequence_.data(), selected_tokens, selected_count,
                               matmul_input_transposed_.data());
            matmul_add_mxfp4_selected_transposed(*layer.down_weight, *layer.down_bias, matmul_input_transposed_.data(),
                                                 selected_tokens, selected_count, down_sequence_.data(),
                                                 gemm_workspace_.data(), expert);
        }

        for (size_t s = 0; s < selected_count; ++s) {
            const size_t position = selected_tokens[s];
            const float expert_weight = selected_weights[s];

            const float *down = down_sequence_.data() + position * embedding_length;
            float *token_output = output + position * embedding_length;
            for (size_t i = 0; i < embedding_length; ++i) {
                token_output[i] += expert_weight * down[i];
            }
        }
    }
}

const float *GptOssModel::prefill(const int32_t *tokens, size_t count) {
    if (count == 0) {
        throw std::runtime_error("prefill needs at least one token");
    }
    if (count > context_length) {
        throw std::runtime_error("model context is full");
    }
    if (count > batch_capacity_) {
        throw std::runtime_error("prefill exceeds startup batch capacity");
    }

    for (size_t position = 0; position < count; ++position) {
        const int32_t token = tokens[position];
        if (token < 0 || size_t(token) >= vocabulary_size) {
            throw std::runtime_error("input token is out of range");
        }
        float *x = x_sequence_.data() + position * embedding_length;
        load_token_embedding_q8_0(*embeddings_, size_t(token), x);
        dump(position, 0, "embedding", x, embedding_length);
    }

    for (size_t i = 0; i < layer_count_; ++i) {
        Layer &layer = layers_[i];

        for (size_t position = 0; position < count; ++position) {
            const float *x = x_sequence_.data() + position * embedding_length;
            float *normalized = normalized_sequence_.data() + position * embedding_length;

            rms_norm(x, *layer.attention_norm, norm_epsilon, normalized);
            dump(position, i, "attention_norm", normalized, embedding_length);
        }

        matmul_add_q8_0(*layer.query_weight, *layer.query_bias, normalized_sequence_.data(), count,
                        query_sequence_.data(), gemm_workspace_.data(), matmul_input_transposed_.data());
        matmul_add_q8_0(*layer.key_weight, *layer.key_bias, normalized_sequence_.data(), count, key_sequence_.data(),
                        gemm_workspace_.data(), matmul_input_transposed_.data());
        matmul_add_q8_0(*layer.value_weight, *layer.value_bias, normalized_sequence_.data(), count,
                        value_sequence_.data(), gemm_workspace_.data(), matmul_input_transposed_.data());

        for (size_t position = 0; position < count; ++position) {
            float *query = query_sequence_.data() + position * query_width;
            float *key = key_sequence_.data() + position * kv_width;
            float *value = value_sequence_.data() + position * kv_width;
            apply_rope(query, query_heads, position);
            apply_rope(key, kv_heads, position);
            dump(position, i, "query", query, query_width);
            dump(position, i, "key", key, kv_width);
            dump(position, i, "value", value, kv_width);
            store_kv(layer, position, key, value);
        }

        for (size_t position = 0; position < count; ++position) {
            const float *query = query_sequence_.data() + position * query_width;
            float *attended = attended_sequence_.data() + position * query_width;
            attention_from_cache(layer, i, position, query, attended);
        }

        matmul_add_q8_0(*layer.attention_output_weight, *layer.attention_output_bias, attended_sequence_.data(), count,
                        projected_sequence_.data(), gemm_workspace_.data(), matmul_input_transposed_.data());

        for (size_t position = 0; position < count; ++position) {
            const float *x = x_sequence_.data() + position * embedding_length;
            float *projected = projected_sequence_.data() + position * embedding_length;
            float *residual = residual_sequence_.data() + position * embedding_length;
            float *normalized = normalized_sequence_.data() + position * embedding_length;

            for (size_t j = 0; j < embedding_length; ++j) {
                residual[j] = x[j] + projected[j];
            }
            dump(position, i, "attention_output", projected, embedding_length);
            rms_norm(residual, *layer.post_attention_norm, norm_epsilon, normalized);
        }

        moe_batch(layer, normalized_sequence_.data(), moe_sequence_.data(), i, count);

        for (size_t position = 0; position < count; ++position) {
            const float *residual = residual_sequence_.data() + position * embedding_length;
            const float *moe_output = moe_sequence_.data() + position * embedding_length;
            float *next_x = x_sequence_.data() + position * embedding_length;
            for (size_t j = 0; j < embedding_length; ++j) {
                next_x[j] = residual[j] + moe_output[j];
            }
            dump(position, i, "output", next_x, embedding_length);
        }
    }

    const size_t last = count - 1;
    const float *x = x_sequence_.data() + last * embedding_length;
    rms_norm(x, *output_norm_, norm_epsilon, normalized_.data());
    matvec_q8_0(*output_, normalized_.data(), logits_.data());
    dump(last, layer_count_, "logits", logits_.data(), vocabulary_size);
    return logits_.data();
}

const float *GptOssModel::forward(int32_t token, size_t position) {
    if (position >= context_length) {
        throw std::runtime_error("model context is full");
    }
    if (position >= token_capacity_) {
        throw std::runtime_error("position exceeds startup token capacity");
    }
    if (token < 0 || size_t(token) >= vocabulary_size) {
        throw std::runtime_error("input token is out of range");
    }
    load_token_embedding_q8_0(*embeddings_, size_t(token), x_.data());
    dump(position, 0, "embedding", x_.data(), embedding_length);

    for (size_t i = 0; i < layer_count_; ++i) {
        Layer &layer = layers_[i];
        rms_norm(x_.data(), *layer.attention_norm, norm_epsilon, normalized_.data());
        dump(position, i, "attention_norm", normalized_.data(), embedding_length);
        matvec_add_q8_0(*layer.query_weight, *layer.query_bias, normalized_.data(), query_.data());
        matvec_add_q8_0(*layer.key_weight, *layer.key_bias, normalized_.data(), key_.data());
        matvec_add_q8_0(*layer.value_weight, *layer.value_bias, normalized_.data(), value_.data());
        apply_rope(query_.data(), query_heads, position);
        apply_rope(key_.data(), kv_heads, position);
        dump(position, i, "query", query_.data(), query_heads * head_size);
        dump(position, i, "key", key_.data(), kv_heads * head_size);
        dump(position, i, "value", value_.data(), kv_heads * head_size);
        store_kv(layer, position, key_.data(), value_.data());
        attention_from_cache(layer, i, position, query_.data(), attended_.data());
        matvec_add_q8_0(*layer.attention_output_weight, *layer.attention_output_bias, attended_.data(),
                        projected_.data());
        for (size_t j = 0; j < embedding_length; ++j) {
            residual_[j] = x_[j] + projected_[j];
        }
        dump(position, i, "attention_output", projected_.data(), embedding_length);
        rms_norm(residual_.data(), *layer.post_attention_norm, norm_epsilon, normalized_.data());
        moe(layer, normalized_.data(), moe_output_.data(), i, position);
        for (size_t j = 0; j < embedding_length; ++j) {
            x_[j] = residual_[j] + moe_output_[j];
        }
        dump(position, i, "output", x_.data(), embedding_length);
    }

    rms_norm(x_.data(), *output_norm_, norm_epsilon, normalized_.data());
    matvec_q8_0(*output_, normalized_.data(), logits_.data());
    dump(position, layer_count_, "logits", logits_.data(), vocabulary_size);
    return logits_.data();
}

int32_t GptOssModel::greedy(const float *logits) const {
    const float *best = logits;
    for (size_t i = 1; i < vocabulary_size; ++i) {
        if (logits[i] > *best) {
            best = logits + i;
        }
    }
    return int32_t(best - logits);
}

void GptOssModel::dump(size_t position, size_t layer, std::string_view name, const float *values, size_t count) const {
    if (dump_fd_ < 0) {
        return;
    }
    char filename[160];
    const int filename_size = std::snprintf(filename, sizeof(filename), "p%06zu.blk%02zu.%.*s.f32", position, layer,
                                            int(name.size()), name.data());
    if (filename_size < 0 || size_t(filename_size) >= sizeof(filename)) {
        throw std::runtime_error("tensor dump filename is too long");
    }

    const int fd = openat(dump_fd_, filename, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        throw std::runtime_error("cannot create tensor dump");
    }
    bool ok = write_all(fd, values, count * sizeof(float));
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        throw std::runtime_error("cannot write tensor dump");
    }

    char manifest[256];
    const int manifest_size = std::snprintf(manifest, sizeof(manifest), "%s\tf32\t%zu\n", filename, count);
    if (manifest_size < 0 || size_t(manifest_size) >= sizeof(manifest)) {
        throw std::runtime_error("tensor dump manifest line is too long");
    }
    if (!write_all(manifest_fd_, manifest, size_t(manifest_size))) {
        throw std::runtime_error("cannot update tensor dump manifest");
    }
}
