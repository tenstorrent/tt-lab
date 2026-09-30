// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class GgufType : uint32_t {
    u8 = 0,
    i8,
    u16,
    i16,
    u32,
    i32,
    f32,
    boolean,
    string,
    array,
    u64,
    i64,
    f64,
};

struct GgufValue {
    GgufType type{};
    GgufType element_type{}; // Used only when type == array.
    uint64_t count = 1;
    const std::byte *data = nullptr;
    std::string_view text;
    std::vector<std::string_view> strings;
};

struct GgufTensor {
    std::string_view name;
    std::vector<uint64_t> shape; // GGML order: the contiguous dimension is first.
    uint32_t type = 0;
    uint64_t offset = 0;
    const std::byte *data = nullptr;
};

class GgufFile {
  public:
    explicit GgufFile(const char *path);
    ~GgufFile();

    GgufFile(const GgufFile &) = delete;
    GgufFile &operator=(const GgufFile &) = delete;

    uint32_t version() const {
        return version_;
    }
    uint64_t data_offset() const {
        return data_offset_;
    }
    const std::vector<GgufTensor> &tensors() const {
        return tensors_;
    }
    const std::vector<std::string_view> &keys() const {
        return keys_;
    }

    const GgufValue *find(std::string_view key) const;
    const GgufTensor *tensor(std::string_view name) const;

    std::string_view string(std::string_view key) const;
    uint64_t integer(std::string_view key) const;
    double number(std::string_view key) const;
    bool boolean(std::string_view key) const;
    bool aliases_path(const char *path) const;

  private:
    int fd_ = -1;
    size_t size_ = 0;
    const std::byte *bytes_ = nullptr;
    uint32_t version_ = 0;
    uint64_t data_offset_ = 0;
    std::vector<std::string_view> keys_;
    std::vector<GgufValue> values_;
    std::unordered_map<std::string_view, size_t> key_index_;
    std::vector<GgufTensor> tensors_;
    std::unordered_map<std::string_view, size_t> tensor_index_;
};

const char *gguf_type_name(GgufType type);
const char *tensor_type_name(uint32_t type);
