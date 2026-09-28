// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer.h"

#include <algorithm>
#include <clocale>
#include <cstddef>
#include <cstring>
#include <cwctype>
#include <limits>
#include <stdexcept>

namespace {

template<class T> T array_element(const GgufValue &value, size_t index) {
    T result;
    std::memcpy(&result, value.data + index * sizeof(T), sizeof(T));
    return result;
}

void append_utf8(std::string &result, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        result.push_back(char(codepoint));
    } else if (codepoint <= 0x7ff) {
        result.push_back(char(0xc0 | (codepoint >> 6)));
        result.push_back(char(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        result.push_back(char(0xe0 | (codepoint >> 12)));
        result.push_back(char(0x80 | ((codepoint >> 6) & 0x3f)));
        result.push_back(char(0x80 | (codepoint & 0x3f)));
    } else {
        result.push_back(char(0xf0 | (codepoint >> 18)));
        result.push_back(char(0x80 | ((codepoint >> 12) & 0x3f)));
        result.push_back(char(0x80 | ((codepoint >> 6) & 0x3f)));
        result.push_back(char(0x80 | (codepoint & 0x3f)));
    }
}

uint32_t next_utf8(std::string_view text, size_t &offset) {
    const uint8_t first = uint8_t(text[offset++]);
    if (first < 0x80) {
        return first;
    }
    int extra = 0;
    uint32_t result = 0;
    if ((first & 0xe0) == 0xc0) {
        extra = 1;
        result = first & 0x1f;
    } else if ((first & 0xf0) == 0xe0) {
        extra = 2;
        result = first & 0x0f;
    } else if ((first & 0xf8) == 0xf0) {
        extra = 3;
        result = first & 0x07;
    } else {
        return 0xfffd;
    }
    if (offset + size_t(extra) > text.size()) {
        offset = text.size();
        return 0xfffd;
    }
    for (int i = 0; i < extra; ++i) {
        const uint8_t byte = uint8_t(text[offset]);
        if ((byte & 0xc0) != 0x80) {
            return 0xfffd;
        }
        ++offset;
        result = (result << 6) | (byte & 0x3f);
    }
    return result;
}

bool ascii_upper(uint32_t c) {
    return c >= 'A' && c <= 'Z';
}
bool ascii_lower(uint32_t c) {
    return c >= 'a' && c <= 'z';
}

bool is_space(uint32_t c) {
    return c <= uint32_t(std::numeric_limits<wchar_t>::max()) && std::iswspace(wint_t(c));
}

bool is_letter(uint32_t c) {
    return c <= uint32_t(std::numeric_limits<wchar_t>::max()) && std::iswalpha(wint_t(c));
}

bool is_number(uint32_t c) {
    return c <= uint32_t(std::numeric_limits<wchar_t>::max()) && std::iswdigit(wint_t(c));
}

struct Character {
    uint32_t codepoint;
    size_t begin;
    size_t end;
};

std::vector<Character> characters(std::string_view text) {
    std::vector<Character> result;
    for (size_t offset = 0; offset < text.size();) {
        const size_t begin = offset;
        const uint32_t codepoint = next_utf8(text, offset);
        result.push_back({codepoint, begin, offset});
    }
    return result;
}

// A direct, deliberately pedestrian implementation of the gpt-4o pre-tokenizer
// pattern. libc supplies Unicode letter/number/space classification; the case
// boundaries in the pattern are specifically ASCII A-Z and a-z.
std::vector<std::string_view> split_gpt4o(std::string_view text) {
    const std::vector<Character> chars = characters(text);
    std::vector<std::string_view> words;
    size_t i = 0;
    auto emit = [&](size_t first, size_t last) {
        words.emplace_back(text.data() + chars[first].begin, chars[last - 1].end - chars[first].begin);
    };

    while (i < chars.size()) {
        const uint32_t c = chars[i].codepoint;

        if (is_number(c)) {
            const size_t first = i++;
            while (i < chars.size() && i - first < 3 && is_number(chars[i].codepoint)) {
                ++i;
            }
            emit(first, i);
            continue;
        }

        // The punctuation alternative permits one ordinary leading space.
        if (c == ' ' && i + 1 < chars.size() && !is_space(chars[i + 1].codepoint) &&
            !is_letter(chars[i + 1].codepoint) && !is_number(chars[i + 1].codepoint)) {
            const size_t first = i++;
            while (i < chars.size() && !is_space(chars[i].codepoint) && !is_letter(chars[i].codepoint) &&
                   !is_number(chars[i].codepoint)) {
                ++i;
            }
            while (i < chars.size() &&
                   (chars[i].codepoint == '\r' || chars[i].codepoint == '\n' || chars[i].codepoint == '/')) {
                ++i;
            }
            emit(first, i);
            continue;
        }

        size_t prefix = i;
        size_t first_letter = i;
        if (c != '\r' && c != '\n' && !is_letter(c) && !is_number(c) && i + 1 < chars.size() &&
            is_letter(chars[i + 1].codepoint)) {
            first_letter = ++i;
        }
        if (is_letter(chars[i].codepoint)) {
            size_t end = i + 1;
            while (end < chars.size() && is_letter(chars[end].codepoint)) {
                ++end;
            }

            // Split lower-to-upper transitions and acronym-to-word transitions.
            size_t part = first_letter;
            while (part < end) {
                size_t stop = part + 1;
                while (stop < end) {
                    const uint32_t previous = chars[stop - 1].codepoint;
                    const uint32_t current = chars[stop].codepoint;
                    const bool lower_to_upper = !ascii_upper(previous) && ascii_upper(current);
                    const bool acronym_boundary = ascii_upper(previous) && ascii_upper(current) && stop + 1 < end &&
                                                  ascii_lower(chars[stop + 1].codepoint);
                    if (lower_to_upper || acronym_boundary) {
                        break;
                    }
                    ++stop;
                }
                const size_t output_first = part == first_letter ? prefix : part;
                size_t output_end = stop;
                if (stop == end && end < chars.size() && chars[end].codepoint == '\'' && end + 1 < chars.size()) {
                    size_t contraction_end = end + 1;
                    while (contraction_end < chars.size() && contraction_end < end + 3 &&
                           ((chars[contraction_end].codepoint >= 'A' && chars[contraction_end].codepoint <= 'Z') ||
                            (chars[contraction_end].codepoint >= 'a' && chars[contraction_end].codepoint <= 'z'))) {
                        ++contraction_end;
                    }
                    std::string suffix;
                    for (size_t k = end + 1; k < contraction_end; ++k) {
                        char x = char(chars[k].codepoint);
                        suffix.push_back(char(x >= 'A' && x <= 'Z' ? x + ('a' - 'A') : x));
                    }
                    if (suffix == "s" || suffix == "t" || suffix == "re" || suffix == "ve" || suffix == "m" ||
                        suffix == "ll" || suffix == "d") {
                        output_end = contraction_end;
                    }
                }
                emit(output_first, output_end);
                part = stop;
                if (output_end > end) {
                    end = output_end;
                    part = end;
                }
            }
            i = end;
            continue;
        }

        if (c == '\r' || c == '\n' || is_space(c)) {
            const size_t first = i;
            bool has_newline = false;
            while (i < chars.size() && is_space(chars[i].codepoint)) {
                has_newline = has_newline || chars[i].codepoint == '\r' || chars[i].codepoint == '\n';
                ++i;
                if (has_newline && i < chars.size() && !is_space(chars[i].codepoint)) {
                    break;
                }
            }
            // An interior run leaves its final space to prefix the following punctuation.
            if (!has_newline && i < chars.size() && i - first > 1) {
                emit(first, i - 1);
                --i;
            } else {
                emit(first, i);
            }
            continue;
        }

        const size_t first = i++;
        while (i < chars.size() && !is_space(chars[i].codepoint) && !is_letter(chars[i].codepoint) &&
               !is_number(chars[i].codepoint)) {
            ++i;
        }
        while (i < chars.size() &&
               (chars[i].codepoint == '\r' || chars[i].codepoint == '\n' || chars[i].codepoint == '/')) {
            ++i;
        }
        emit(first, i);
    }
    return words;
}

uint32_t byte_codepoint(uint8_t byte) {
    if ((byte >= '!' && byte <= '~') || (byte >= 0xa1 && byte <= 0xac) || (byte >= 0xae)) {
        return byte;
    }
    uint32_t extra = 0;
    for (uint32_t b = 0; b < byte; ++b) {
        if (!((b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || b >= 0xae)) {
            ++extra;
        }
    }
    return 256 + extra;
}

std::string byte_encode(std::string_view bytes) {
    std::string result;
    result.reserve(bytes.size() * 2);
    for (const unsigned char byte : bytes) {
        append_utf8(result, byte_codepoint(byte));
    }
    return result;
}

std::vector<std::string> unicode_symbols(std::string_view text) {
    std::vector<std::string> result;
    for (size_t offset = 0; offset < text.size();) {
        const size_t begin = offset;
        (void)next_utf8(text, offset);
        result.emplace_back(text.substr(begin, offset - begin));
    }
    return result;
}

} // namespace

Tokenizer::Tokenizer(const GgufFile &file) {
    if (file.string("tokenizer.ggml.model") != "gpt2" || file.string("tokenizer.ggml.pre") != "gpt-4o") {
        throw std::runtime_error("only the gpt-oss gpt-4o BPE tokenizer is supported");
    }
    const GgufValue *token_value = file.find("tokenizer.ggml.tokens");
    const GgufValue *type_value = file.find("tokenizer.ggml.token_type");
    const GgufValue *merge_value = file.find("tokenizer.ggml.merges");
    if (!token_value || token_value->type != GgufType::array || token_value->element_type != GgufType::string ||
        !type_value || type_value->type != GgufType::array || type_value->element_type != GgufType::i32 ||
        !merge_value || merge_value->type != GgufType::array || merge_value->element_type != GgufType::string ||
        token_value->count != type_value->count) {
        throw std::runtime_error("invalid tokenizer metadata");
    }

    tokens_ = token_value->strings;
    token_types_.reserve(tokens_.size());
    token_ids_.reserve(tokens_.size());
    for (size_t i = 0; i < tokens_.size(); ++i) {
        const int32_t type = array_element<int32_t>(*type_value, i);
        token_types_.push_back(type);
        token_ids_.emplace(tokens_[i], int32_t(i));
        if (type != 1) {
            special_tokens_.push_back(tokens_[i]); // 1 is a normal token.
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(), [](auto a, auto b) { return a.size() > b.size(); });

    merge_ranks_.reserve(merge_value->strings.size());
    for (size_t i = 0; i < merge_value->strings.size(); ++i) {
        merge_ranks_.emplace(std::string(merge_value->strings[i]), int32_t(i));
    }
    byte_decoder_.fill(-1);
    for (uint32_t byte = 0; byte < 256; ++byte) {
        const uint32_t codepoint = byte_codepoint(uint8_t(byte));
        if (codepoint >= byte_decoder_.size()) {
            throw std::runtime_error("byte decoder table is too small");
        }
        byte_decoder_[codepoint] = int16_t(byte);
    }
    std::setlocale(LC_CTYPE, "");
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> output;
    size_t offset = 0;
    while (offset < text.size()) {
        size_t special_at = text.size();
        std::string_view special;
        for (std::string_view candidate : special_tokens_) {
            const size_t found = text.find(candidate, offset);
            if (found < special_at) {
                special_at = found;
                special = candidate;
            }
        }
        if (special_at > offset) {
            encode_ordinary(text.substr(offset, special_at - offset), output);
        }
        if (special.empty()) {
            break;
        }
        output.push_back(token_ids_.at(special));
        offset = special_at + special.size();
    }
    return output;
}

void Tokenizer::encode_ordinary(std::string_view text, std::vector<int32_t> &output) const {
    for (const std::string_view word : split_gpt4o(text)) {
        encode_word(word, output);
    }
}

void Tokenizer::encode_word(std::string_view word, std::vector<int32_t> &output) const {
    std::vector<std::string> symbols = unicode_symbols(byte_encode(word));
    while (symbols.size() > 1) {
        size_t best = symbols.size();
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best = i;
                best_rank = it->second;
            }
        }
        if (best == symbols.size()) {
            break;
        }
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + ptrdiff_t(best + 1));
    }
    for (const std::string &symbol : symbols) {
        const auto it = token_ids_.find(symbol);
        if (it == token_ids_.end()) {
            throw std::runtime_error("tokenizer produced an unknown BPE piece");
        }
        output.push_back(it->second);
    }
}

size_t Tokenizer::decode_to(int32_t token, char *output, size_t capacity, bool render_special) const {
    if (token < 0 || size_t(token) >= tokens_.size()) {
        throw std::runtime_error("token id is out of range");
    }
    if (token_types_[size_t(token)] != 1) {
        if (!render_special) {
            return 0;
        }
        const std::string_view special = tokens_[size_t(token)];
        if (special.size() > capacity) {
            throw std::runtime_error("decoded token buffer is too small");
        }
        std::memcpy(output, special.data(), special.size());
        return special.size();
    }

    size_t size = 0;
    const std::string_view encoded = tokens_[size_t(token)];
    for (size_t offset = 0; offset < encoded.size();) {
        const uint32_t codepoint = next_utf8(encoded, offset);
        if (codepoint >= byte_decoder_.size() || byte_decoder_[codepoint] < 0) {
            throw std::runtime_error("token contains an invalid byte encoding");
        }
        if (size == capacity) {
            throw std::runtime_error("decoded token buffer is too small");
        }
        output[size++] = char(byte_decoder_[codepoint]);
    }
    return size;
}

std::string harmony_prompt(std::string_view user_prompt) {
    return "<|start|>user<|message|>" + std::string(user_prompt) + "<|end|><|start|>assistant";
}
