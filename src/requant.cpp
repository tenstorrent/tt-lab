// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "requant.h"

#include "tensor.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

class TtqOutput {
  public:
    explicit TtqOutput(const char *path) : temporary_(std::string(path) + ".tmp.XXXXXX") {
        const int fd = mkstemp(temporary_.data());
        if (fd < 0) {
            throw std::runtime_error("cannot create temporary TTQ file beside " + std::string(path) + ": " +
                                     std::strerror(errno));
        }
        stream = fdopen(fd, "wb");
        if (!stream) {
            const int error = errno;
            close(fd);
            unlink(temporary_.c_str());
            throw std::runtime_error("cannot open temporary TTQ stream: " + std::string(std::strerror(error)));
        }
    }

    ~TtqOutput() {
        if (stream) {
            std::fclose(stream);
        }
        if (!temporary_.empty()) {
            unlink(temporary_.c_str());
        }
    }

    TtqOutput(const TtqOutput &) = delete;
    TtqOutput &operator=(const TtqOutput &) = delete;

    void finish(const char *path) {
        const int result = std::fclose(stream);
        stream = nullptr;
        if (result != 0) {
            throw std::runtime_error("cannot close TTQ file " + std::string(path) + ": " + std::strerror(errno));
        }
        if (std::rename(temporary_.c_str(), path) != 0) {
            throw std::runtime_error("cannot replace TTQ file " + std::string(path) + ": " + std::strerror(errno));
        }
        temporary_.clear();
    }

    std::FILE *stream = nullptr;

  private:
    std::string temporary_;
};

constexpr size_t tile_dim = 16;
constexpr size_t bf16_tile_bytes = 512;
constexpr size_t bfp_section = 16;
constexpr size_t embedding_length = 2880;
constexpr size_t vocabulary_size = 201088;
constexpr size_t query_heads = 64;
constexpr size_t kv_heads = 8;
constexpr size_t head_size = 64;
constexpr size_t expert_hidden_length = 2880;
constexpr size_t experts_used = 4;

struct ModelShape {
    uint32_t model = 0;
    uint32_t layers = 0;
    uint32_t experts = 0;
};

constexpr int8_t mxfp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

void require_integer(const GgufFile &file, std::string_view key, uint64_t expected) {
    if (file.integer(key) != expected) {
        throw std::runtime_error("unsupported " + std::string(key));
    }
}

ModelShape validate_model(const GgufFile &file) {
    if (file.string("general.architecture") != "gpt-oss") {
        throw std::runtime_error("model architecture is not gpt-oss");
    }

    ModelShape result;
    const uint64_t layers = file.integer("gpt-oss.block_count");
    const uint64_t experts = file.integer("gpt-oss.expert_count");
    if (layers == 24 && experts == 32) {
        result.model = 20;
    } else if (layers == 36 && experts == 128) {
        result.model = 120;
    } else {
        throw std::runtime_error("only gpt-oss-20b and gpt-oss-120b are supported by requant");
    }
    result.layers = uint32_t(layers);
    result.experts = uint32_t(experts);

    require_integer(file, "gpt-oss.embedding_length", embedding_length);
    require_integer(file, "gpt-oss.feed_forward_length", embedding_length);
    require_integer(file, "gpt-oss.attention.head_count", query_heads);
    require_integer(file, "gpt-oss.attention.head_count_kv", kv_heads);
    require_integer(file, "gpt-oss.attention.key_length", head_size);
    require_integer(file, "gpt-oss.attention.value_length", head_size);
    require_integer(file, "gpt-oss.expert_used_count", experts_used);
    require_integer(file, "gpt-oss.expert_feed_forward_length", expert_hidden_length);
    return result;
}

void write_exact(std::FILE *file, const void *data, size_t size) {
    if (size && std::fwrite(data, 1, size, file) != size) {
        throw std::runtime_error("TTQ write failed");
    }
}

uint64_t tell_file(std::FILE *file) {
    const long offset = std::ftell(file);
    if (offset < 0) {
        throw std::runtime_error("TTQ ftell failed");
    }
    return uint64_t(offset);
}

void seek_file(std::FILE *file, uint64_t offset) {
    if (offset > uint64_t(std::numeric_limits<long>::max()) || std::fseek(file, long(offset), SEEK_SET) != 0) {
        throw std::runtime_error("TTQ seek failed");
    }
}

void align_file(std::FILE *file) {
    static const uint8_t zeros[ttq_payload_alignment] = {};
    const uint64_t offset = tell_file(file);
    const uint64_t aligned = (offset + ttq_payload_alignment - 1) & ~(uint64_t(ttq_payload_alignment) - 1);
    write_exact(file, zeros, size_t(aligned - offset));
}

uint16_t load_u16(const std::byte *data) {
    uint16_t result;
    std::memcpy(&result, data, sizeof(result));
    return result;
}

void pack_q8_0_row_bf16(const std::byte *source_row, size_t columns, uint16_t *output) {
    for (size_t block = 0; block < columns / 32; ++block) {
        const std::byte *source = source_row + block * 34;
        const float scale = fp16_to_fp32(load_u16(source));
        const auto *quants = (const int8_t *)(source + 2);
        for (size_t i = 0; i < 32; ++i) {
            output[block * 32 + i] = fp32_to_bf16(scale * float(quants[i]));
        }
    }
}

void unpack_mxfp4_row(const std::byte *source_row, float *values) {
    for (size_t block = 0; block < embedding_length / 32; ++block) {
        const auto *source = (const uint8_t *)(source_row + block * 17);
        const float scale = mxfp4_scale(source[0]);
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = source[1 + i];
            values[block * 32 + i] = scale * float(mxfp4_values[packed & 0x0f]);
            values[block * 32 + 16 + i] = scale * float(mxfp4_values[packed >> 4]);
        }
    }
}

void pack_bf16_tensix_block(const GgufTensor &tensor, size_t row_base, size_t columns, std::vector<uint8_t> &tiles) {
    const size_t rows = size_t(tensor.shape[1]);
    const size_t row_bytes = tensor_row_bytes(tensor);
    const size_t k_tiles = columns / tile_dim;
    std::vector<uint16_t> rows_bf16(tile_dim * columns);

    if ((columns % tile_dim) != 0) {
        throw std::runtime_error("BF16 Tensix columns must be a multiple of 16");
    }
    if ((row_base + tile_dim) > rows) {
        throw std::runtime_error("BF16 Tensix row block out of range");
    }
    tiles.assign(k_tiles * bf16_tile_bytes, 0);
    for (size_t row = 0; row < tile_dim; ++row) {
        const std::byte *source = tensor.data + (row_base + row) * row_bytes;
        uint16_t *row_output = rows_bf16.data() + row * columns;
        if (tensor.type == tensor_q8_0) {
            pack_q8_0_row_bf16(source, columns, row_output);
        } else if (tensor.type == tensor_f32) {
            const float *values = (const float *)source;
            for (size_t i = 0; i < columns; ++i) {
                row_output[i] = fp32_to_bf16(values[i]);
            }
        } else {
            throw std::runtime_error("unsupported BF16 Tensix source tensor");
        }
    }
    for (size_t k = 0; k < k_tiles; ++k) {
        uint16_t *tile = (uint16_t *)(tiles.data() + k * bf16_tile_bytes);
        for (size_t i = 0; i < tile_dim; ++i) {
            uint16_t *dst = tile + (i / 2) * 32 + (i & 1) * tile_dim;
            for (size_t row = 0; row < tile_dim; ++row) {
                dst[row] = rows_bf16[row * columns + k * tile_dim + i];
            }
        }
    }
}

uint64_t write_bf16_linear_payload(std::FILE *output, const GgufTensor &tensor) {
    const uint64_t begin = tell_file(output);
    const size_t columns = size_t(tensor.shape[0]);
    size_t rows = 1;
    for (size_t i = 1; i < tensor.shape.size(); ++i) {
        rows *= size_t(tensor.shape[i]);
    }
    std::vector<uint16_t> row(columns);
    const size_t row_bytes = tensor_row_bytes(tensor);

    for (size_t r = 0; r < rows; ++r) {
        const std::byte *source = tensor.data + r * row_bytes;
        if (tensor.type == tensor_q8_0) {
            pack_q8_0_row_bf16(source, columns, row.data());
        } else if (tensor.type == tensor_f32) {
            const float *values = (const float *)source;
            for (size_t i = 0; i < columns; ++i) {
                row[i] = fp32_to_bf16(values[i]);
            }
        } else {
            throw std::runtime_error("unsupported TTQ BF16 linear source tensor");
        }
        write_exact(output, row.data(), columns * sizeof(uint16_t));
    }
    return tell_file(output) - begin;
}

uint64_t write_bf16_tensix_payload(std::FILE *output, const GgufTensor &tensor) {
    const uint64_t begin = tell_file(output);
    const size_t columns = size_t(tensor.shape[0]);
    const size_t rows = size_t(tensor.shape[1]);
    std::vector<uint8_t> tiles;

    for (size_t row_base = 0; row_base < rows; row_base += tile_dim) {
        pack_bf16_tensix_block(tensor, row_base, columns, tiles);
        write_exact(output, tiles.data(), tiles.size());
    }
    return tell_file(output) - begin;
}

void pack_bfp_strip_source_rows(const GgufTensor &tensor, size_t expert, size_t row_base,
                                std::vector<float> &source_rows) {
    const size_t rows = size_t(tensor.shape[1]);
    const size_t src_row_bytes = tensor_row_bytes(tensor);

    source_rows.assign(tile_dim * embedding_length, 0.0f);
    for (size_t r = 0; r < tile_dim; ++r) {
        const size_t row_index = row_base + r;
        const std::byte *source = tensor.data + (expert * rows + row_index) * src_row_bytes;
        unpack_mxfp4_row(source, source_rows.data() + r * embedding_length);
    }
}

void pack_bfp8_strip(const GgufTensor &tensor, size_t expert, size_t row_base, std::vector<uint8_t> &strip) {
    std::vector<float> source_rows;

    pack_bfp_strip_source_rows(tensor, expert, row_base, source_rows);
    strip.resize(requant_bfp8_strip_bytes);
    uint8_t *exponents = strip.data();
    uint8_t *data = strip.data() + requant_bfp8_strip_exponent_bytes;
    for (size_t column = 0; column < embedding_length; column += tile_dim) {
        for (size_t row = 0; row < tile_dim; ++row) {
            pack_bfp8_section(source_rows.data() + row * embedding_length + column, exponents + column + row,
                              data + column * tile_dim + row * tile_dim);
        }
    }
}

uint64_t write_bfp8_strip_payload(std::FILE *output, const GgufTensor &tensor) {
    const uint64_t begin = tell_file(output);
    const size_t rows = size_t(tensor.shape[1]);
    const size_t experts = size_t(tensor.shape[2]);
    std::vector<uint8_t> strip;

    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row_base = 0; row_base < rows; row_base += tile_dim) {
            pack_bfp8_strip(tensor, expert, row_base, strip);
            write_exact(output, strip.data(), strip.size());
        }
    }
    return tell_file(output) - begin;
}

bool is_router_weight(std::string_view name) {
    constexpr std::string_view suffix = "ffn_gate_inp.weight";
    return name.size() >= suffix.size() && name.substr(name.size() - suffix.size()) == suffix;
}

bool is_mxfp4_expert_weight(const GgufTensor &tensor, const ModelShape &model) {
    if (tensor.type != tensor_mxfp4) {
        return false;
    }
    if (tensor.shape.size() != 3 || tensor.shape[0] != embedding_length || tensor.shape[1] != expert_hidden_length ||
        tensor.shape[2] != model.experts) {
        throw std::runtime_error("unexpected MXFP4 tensor shape in TTQ requant: " + std::string(tensor.name));
    }
    return true;
}

void init_record(TtqRecord &record, std::string_view name, const GgufTensor &tensor) {
    if (name.size() >= ttq_name_bytes) {
        throw std::runtime_error("TTQ tensor name is too long: " + std::string(name));
    }
    if (tensor.shape.size() > 4) {
        throw std::runtime_error("TTQ tensor rank is too large: " + std::string(tensor.name));
    }
    std::memcpy(record.name, name.data(), name.size());
    record.rank = uint32_t(tensor.shape.size());
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        record.shape[i] = tensor.shape[i];
    }
}

} // namespace

uint16_t fp32_to_bf16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return uint16_t(bits >> 16);
}

uint8_t pack_bfp_value(float value, uint8_t exponent, uint32_t magnitude_bits) {
    const uint32_t sign = std::signbit(value) ? 1u : 0u;
    const uint32_t max_magnitude = (1u << magnitude_bits) - 1u;
    const float scale = std::ldexp(1.0f, int(exponent) - 127 - int(magnitude_bits - 1));
    uint32_t magnitude = uint32_t(std::floor(std::fabs(value) / scale + 0.5f));

    if (!magnitude) {
        return 0;
    }
    magnitude = std::min(magnitude, max_magnitude);
    return uint8_t(magnitude | (sign << magnitude_bits));
}

uint8_t pack_bfp4_value(float value, uint8_t exponent) {
    return pack_bfp_value(value, exponent, 3);
}

uint8_t pack_bfp8_value(float value, uint8_t exponent) {
    return pack_bfp_value(value, exponent, 7);
}

namespace {

double bfp_section_error(const float *values, uint8_t exponent, uint32_t magnitude_bits) {
    double error = 0.0;
    const uint32_t max_magnitude = (1u << magnitude_bits) - 1u;
    const float scale = std::ldexp(1.0f, int(exponent) - 127 - int(magnitude_bits - 1));

    for (size_t i = 0; i < bfp_section; ++i) {
        uint32_t magnitude = uint32_t(std::floor(std::fabs(values[i]) / scale + 0.5f));
        magnitude = std::min(magnitude, max_magnitude);
        const float unpacked = std::signbit(values[i]) ? -scale * float(magnitude) : scale * float(magnitude);
        const double delta = double(unpacked) - double(values[i]);
        error += delta * delta;
    }
    return error;
}

uint8_t choose_bfp_exponent(const float *values, uint32_t magnitude_bits) {
    uint8_t max_exp = 0;
    for (size_t i = 0; i < bfp_section; ++i) {
        const float value = values[i];
        if (value == 0.0f) {
            continue;
        }
        const uint32_t raw = std::bit_cast<uint32_t>(std::fabs(value));
        max_exp = std::max(max_exp, uint8_t((raw >> 23) & 255));
    }
    if (max_exp <= 1) {
        return max_exp;
    }

    const double max_error = bfp_section_error(values, max_exp, magnitude_bits);
    const double smaller_error = bfp_section_error(values, uint8_t(max_exp - 1), magnitude_bits);
    if (smaller_error < max_error) {
        return uint8_t(max_exp - 1);
    }
    return max_exp;
}

} // namespace

void pack_bfp4_section(const float *values, uint8_t *exponent, uint8_t *bfp4_data) {
    *exponent = choose_bfp_exponent(values, 3);
    for (size_t i = 0; i < bfp_section; ++i) {
        const uint8_t packed4 = pack_bfp4_value(values[i], *exponent);
        if ((i & 1) == 0) {
            bfp4_data[i / 2] = packed4;
        } else {
            bfp4_data[i / 2] |= uint8_t(packed4 << 4);
        }
    }
}

void pack_bfp8_section(const float *values, uint8_t *exponent, uint8_t *bfp8_data) {
    *exponent = choose_bfp_exponent(values, 7);
    for (size_t i = 0; i < bfp_section; ++i) {
        bfp8_data[i] = pack_bfp8_value(values[i], *exponent);
    }
}

void write_requant(const char *output_path, const GgufFile &file) {
    if (file.aliases_path(output_path)) {
        throw std::runtime_error("TTQ output aliases the GGUF input: " + std::string(output_path));
    }
    const ModelShape model = validate_model(file);

    for (const GgufTensor &tensor : file.tensors()) {
        is_mxfp4_expert_weight(tensor, model);
    }

    if (file.tensors().size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("too many tensors for TTQ file");
    }
    std::vector<TtqRecord> records(file.tensors().size());
    TtqHeader header{};
    const char magic[8] = {'T', 'T', 'L', 'B', 'Q', 'N', 'T', '1'};
    std::memcpy(header.magic, magic, sizeof(header.magic));
    header.version = 1;
    header.model = model.model;
    header.tensor_count = uint32_t(records.size());
    header.directory_offset = sizeof(TtqHeader);
    header.data_offset = (sizeof(TtqHeader) + records.size() * sizeof(TtqRecord) + ttq_payload_alignment - 1) &
                         ~(uint64_t(ttq_payload_alignment) - 1);
    header.layer_count = model.layers;
    header.expert_count = model.experts;
    header.dim = embedding_length;
    header.vocab = vocabulary_size;

    TtqOutput destination(output_path);
    std::FILE *output = destination.stream;
    write_exact(output, &header, sizeof(header));
    if (!records.empty()) {
        write_exact(output, records.data(), records.size() * sizeof(TtqRecord));
    }

    size_t record_index = 0;
    for (size_t tensor_index = 0; tensor_index < file.tensors().size(); ++tensor_index) {
        const GgufTensor &tensor = file.tensors()[tensor_index];
        TtqRecord &record = records[record_index++];
        const std::string name = std::string(tensor.name) + (tensor.type == tensor_mxfp4 ? ".bfp8" : "");
        init_record(record, name, tensor);

        align_file(output);
        record.offset = tell_file(output);
        if (tensor.type == tensor_mxfp4) {
            record.format = ttq_format_bfp8_strip;
            record.size = write_bfp8_strip_payload(output, tensor);
        } else if (tensor.type == tensor_q8_0) {
            if (tensor.name == "token_embd.weight") {
                record.format = ttq_format_bf16_linear;
                record.size = write_bf16_linear_payload(output, tensor);
            } else {
                if (tensor.shape.size() != 2 || (tensor.shape[0] % tile_dim) != 0 ||
                    (tensor.shape[1] % tile_dim) != 0) {
                    throw std::runtime_error("unexpected Q8_0 tensor shape in TTQ requant: " +
                                             std::string(tensor.name));
                }
                record.format = ttq_format_bf16_tensix;
                record.size = write_bf16_tensix_payload(output, tensor);
            }
        } else if (tensor.type == tensor_f32) {
            if (is_router_weight(tensor.name)) {
                if (tensor.shape.size() != 2 || tensor.shape[0] != embedding_length ||
                    tensor.shape[1] != model.experts) {
                    throw std::runtime_error("unexpected router tensor shape in TTQ requant: " +
                                             std::string(tensor.name));
                }
                record.format = ttq_format_bf16_tensix;
                record.size = write_bf16_tensix_payload(output, tensor);
            } else {
                record.format = ttq_format_bf16_linear;
                record.size = write_bf16_linear_payload(output, tensor);
            }
        } else {
            throw std::runtime_error("unsupported tensor type in TTQ requant: " + std::string(tensor.name));
        }
        std::fprintf(stderr, "requantized %zu/%zu: %s, %.3f MiB\n", record_index, records.size(), name.c_str(),
                     double(record.size) / (1024.0 * 1024.0));
    }
    if (record_index != records.size()) {
        throw std::runtime_error("internal TTQ record count mismatch");
    }

    seek_file(output, 0);
    write_exact(output, &header, sizeof(header));
    if (!records.empty()) {
        write_exact(output, records.data(), records.size() * sizeof(TtqRecord));
    }
    destination.finish(output_path);
}
