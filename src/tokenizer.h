// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gguf.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class Tokenizer {
  public:
    explicit Tokenizer(const GgufFile &file);

    std::vector<int32_t> encode(std::string_view text) const;
    size_t decode_to(int32_t token, char *output, size_t capacity, bool render_special = true) const;

  private:
    std::vector<std::string_view> tokens_;
    std::vector<int32_t> token_types_;
    std::unordered_map<std::string_view, int32_t> token_ids_;
    std::unordered_map<std::string, int32_t> merge_ranks_;
    std::vector<std::string_view> special_tokens_;
    std::array<int16_t, 512> byte_decoder_;

    void encode_ordinary(std::string_view text, std::vector<int32_t> &output) const;
    void encode_word(std::string_view word, std::vector<int32_t> &output) const;
};

std::string harmony_prompt(std::string_view user_prompt);
