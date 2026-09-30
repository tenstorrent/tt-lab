// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gguf.h"
#include "tensor.h"

#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

class Cursor {
  public:
    Cursor(const std::byte *bytes, size_t size) : begin_(bytes), current_(bytes), end_(bytes + size) {}

    template<class T> T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        require(sizeof(T));
        T result;
        std::memcpy(&result, current_, sizeof(T));
        current_ += sizeof(T);
        return result;
    }

    const std::byte *take(size_t size) {
        require(size);
        const std::byte *result = current_;
        current_ += size;
        return result;
    }

    std::string_view string() {
        const uint64_t length = read<uint64_t>();
        if (length > SIZE_MAX) {
            fail("GGUF string is too large");
        }
        const auto *chars = (const char *)take(size_t(length));
        return {chars, size_t(length)};
    }

    size_t offset() const {
        return size_t(current_ - begin_);
    }

  private:
    void require(size_t size) const {
        if (size > size_t(end_ - current_)) {
            fail("truncated GGUF file");
        }
    }

    const std::byte *begin_;
    const std::byte *current_;
    const std::byte *end_;
};

size_t primitive_size(GgufType type) {
    switch (type) {
        case GgufType::u8:
        case GgufType::i8:
        case GgufType::boolean:
            return 1;
        case GgufType::u16:
        case GgufType::i16:
            return 2;
        case GgufType::u32:
        case GgufType::i32:
        case GgufType::f32:
            return 4;
        case GgufType::u64:
        case GgufType::i64:
        case GgufType::f64:
            return 8;
        default:
            fail("GGUF value is not a fixed-size primitive");
    }
}

GgufType read_type(Cursor &cursor) {
    const uint32_t raw = cursor.read<uint32_t>();
    if (raw > uint32_t(GgufType::f64)) {
        fail("unknown GGUF metadata type " + std::to_string(raw));
    }
    return GgufType(raw);
}

GgufValue read_value(Cursor &cursor, GgufType type) {
    GgufValue value;
    value.type = type;
    if (type == GgufType::string) {
        value.text = cursor.string();
    } else if (type == GgufType::array) {
        value.element_type = read_type(cursor);
        value.count = cursor.read<uint64_t>();
        if (value.element_type == GgufType::array) {
            fail("nested GGUF arrays are invalid");
        }
        if (value.element_type == GgufType::string) {
            if (value.count > SIZE_MAX) {
                fail("GGUF string array is too large");
            }
            value.strings.reserve(size_t(value.count));
            for (uint64_t i = 0; i < value.count; ++i) {
                value.strings.push_back(cursor.string());
            }
        } else {
            const size_t width = primitive_size(value.element_type);
            if (value.count > SIZE_MAX / width) {
                fail("GGUF array is too large");
            }
            value.data = cursor.take(size_t(value.count) * width);
        }
    } else {
        value.data = cursor.take(primitive_size(type));
    }
    return value;
}

template<class T> T load(const GgufValue &value) {
    T result;
    std::memcpy(&result, value.data, sizeof(result));
    return result;
}

uint64_t round_up(uint64_t n, uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        fail("invalid GGUF alignment");
    }
    return (n + alignment - 1) & ~(alignment - 1);
}

} // namespace

GgufFile::GgufFile(const char *path) {
    if constexpr (std::endian::native != std::endian::little) {
        fail("only little-endian hosts are supported");
    }

    fd_ = open(path, O_RDONLY);
    if (fd_ < 0) {
        fail("cannot open " + std::string(path) + ": " + std::strerror(errno));
    }

    struct stat status {};
    if (fstat(fd_, &status) != 0) {
        fail("cannot stat " + std::string(path) + ": " + std::strerror(errno));
    }
    if (status.st_size < 0 || uint64_t(status.st_size) > SIZE_MAX) {
        fail("GGUF file is too large");
    }
    size_ = size_t(status.st_size);
    void *mapping = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping == MAP_FAILED) {
        fail("cannot mmap " + std::string(path) + ": " + std::strerror(errno));
    }
    bytes_ = (const std::byte *)mapping;

    Cursor cursor(bytes_, size_);
    const uint32_t magic = cursor.read<uint32_t>();
    if (magic != 0x46554747) {
        fail("not a GGUF file"); // bytes: GGUF
    }
    version_ = cursor.read<uint32_t>();
    if (version_ != 3) {
        fail("unsupported GGUF version " + std::to_string(version_));
    }
    const uint64_t tensor_count = cursor.read<uint64_t>();
    const uint64_t key_count = cursor.read<uint64_t>();
    if (tensor_count > SIZE_MAX || key_count > SIZE_MAX) {
        fail("GGUF header counts are too large");
    }

    keys_.reserve(size_t(key_count));
    values_.reserve(size_t(key_count));
    key_index_.reserve(size_t(key_count));
    for (uint64_t i = 0; i < key_count; ++i) {
        const std::string_view key = cursor.string();
        GgufValue value = read_value(cursor, read_type(cursor));
        if (!key_index_.emplace(key, keys_.size()).second) {
            fail("duplicate GGUF key: " + std::string(key));
        }
        keys_.push_back(key);
        values_.push_back(std::move(value));
    }

    tensors_.reserve(size_t(tensor_count));
    tensor_index_.reserve(size_t(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensor tensor;
        tensor.name = cursor.string();
        const uint32_t dimensions = cursor.read<uint32_t>();
        if (dimensions == 0 || dimensions > 4) {
            fail("invalid tensor rank");
        }
        tensor.shape.reserve(dimensions);
        for (uint32_t d = 0; d < dimensions; ++d) {
            tensor.shape.push_back(cursor.read<uint64_t>());
        }
        tensor.type = cursor.read<uint32_t>();
        tensor.offset = cursor.read<uint64_t>();
        if (!tensor_index_.emplace(tensor.name, tensors_.size()).second) {
            fail("duplicate tensor: " + std::string(tensor.name));
        }
        tensors_.push_back(std::move(tensor));
    }

    uint64_t alignment = 32;
    if (const GgufValue *value = find("general.alignment")) {
        if (value->type != GgufType::u32) {
            fail("general.alignment is not uint32");
        }
        alignment = load<uint32_t>(*value);
    }
    data_offset_ = round_up(cursor.offset(), alignment);
    if (data_offset_ > size_) {
        fail("GGUF tensor data starts past end of file");
    }
    for (GgufTensor &tensor : tensors_) {
        if (tensor.offset > size_ - data_offset_) {
            fail("tensor starts past end of GGUF file");
        }
        tensor.data = bytes_ + data_offset_ + tensor.offset;
        const size_t bytes = tensor_bytes(tensor);
        if (bytes > size_ - data_offset_ - tensor.offset) {
            fail("tensor extends past end of GGUF file");
        }
    }
}

bool GgufFile::aliases_path(const char *path) const {
    struct stat input {
    }, output{};
    if (fstat(fd_, &input) != 0) {
        fail("cannot stat GGUF input: " + std::string(std::strerror(errno)));
    }
    if (stat(path, &output) != 0) {
        if (errno == ENOENT) {
            return false;
        }
        fail("cannot stat output " + std::string(path) + ": " + std::strerror(errno));
    }
    return input.st_dev == output.st_dev && input.st_ino == output.st_ino;
}

GgufFile::~GgufFile() {
    if (bytes_) {
        munmap((void *)bytes_, size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

const GgufValue *GgufFile::find(std::string_view key) const {
    const auto it = key_index_.find(key);
    return it == key_index_.end() ? nullptr : &values_[it->second];
}

const GgufTensor *GgufFile::tensor(std::string_view name) const {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

std::string_view GgufFile::string(std::string_view key) const {
    const GgufValue *value = find(key);
    if (!value) {
        fail("missing GGUF key: " + std::string(key));
    }
    if (value->type != GgufType::string) {
        fail("GGUF key is not a string: " + std::string(key));
    }
    return value->text;
}

uint64_t GgufFile::integer(std::string_view key) const {
    const GgufValue *value = find(key);
    if (!value) {
        fail("missing GGUF key: " + std::string(key));
    }
    switch (value->type) {
        case GgufType::u8:
            return load<uint8_t>(*value);
        case GgufType::i8:
            return uint64_t(load<int8_t>(*value));
        case GgufType::u16:
            return load<uint16_t>(*value);
        case GgufType::i16:
            return uint64_t(load<int16_t>(*value));
        case GgufType::u32:
            return load<uint32_t>(*value);
        case GgufType::i32:
            return uint64_t(load<int32_t>(*value));
        case GgufType::u64:
            return load<uint64_t>(*value);
        case GgufType::i64:
            return uint64_t(load<int64_t>(*value));
        default:
            fail("GGUF key is not an integer: " + std::string(key));
    }
}

double GgufFile::number(std::string_view key) const {
    const GgufValue *value = find(key);
    if (!value) {
        fail("missing GGUF key: " + std::string(key));
    }
    if (value->type == GgufType::f32) {
        return load<float>(*value);
    }
    if (value->type == GgufType::f64) {
        return load<double>(*value);
    }
    return double(integer(key));
}

bool GgufFile::boolean(std::string_view key) const {
    const GgufValue *value = find(key);
    if (!value) {
        fail("missing GGUF key: " + std::string(key));
    }
    if (value->type != GgufType::boolean) {
        fail("GGUF key is not boolean: " + std::string(key));
    }
    return load<uint8_t>(*value) != 0;
}

const char *gguf_type_name(GgufType type) {
    static constexpr const char *names[] = {
        "uint8", "int8",   "uint16", "int16",  "uint32", "int32",   "float32",
        "bool",  "string", "array",  "uint64", "int64",  "float64",
    };
    const uint32_t i = uint32_t(type);
    return i < std::size(names) ? names[i] : "unknown";
}

const char *tensor_type_name(uint32_t type) {
    static constexpr const char *names[] = {
        "F32",     "F16",     "Q4_0",    "Q4_1",  "removed", "removed", "Q5_0",    "Q5_1",    "Q8_0",
        "Q8_1",    "Q2_K",    "Q3_K",    "Q4_K",  "Q5_K",    "Q6_K",    "Q8_K",    "IQ2_XXS", "IQ2_XS",
        "IQ3_XXS", "IQ1_S",   "IQ4_NL",  "IQ3_S", "IQ2_S",   "IQ4_XS",  "I8",      "I16",     "I32",
        "I64",     "F64",     "IQ1_M",   "BF16",  "removed", "removed", "removed", "TQ1_0",   "TQ2_0",
        "removed", "removed", "removed", "MXFP4", "NVFP4",   "Q1_0",    "Q2_0",
    };
    return type < std::size(names) ? names[type] : "UNKNOWN";
}
