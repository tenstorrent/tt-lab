// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gguf.h"

#include <cstddef>
#include <cstdint>

constexpr uint32_t ttq_format_bf16_linear = 1;
constexpr uint32_t ttq_format_bf16_tensix = 2;
constexpr uint32_t ttq_format_bfp4_strip = 3;
// Format 4 used output-row shared exponents. Format 5 transposes each 16x16 tile:
// exponents[k_tile * 16 + row], data[k_tile * 256 + row * 16 + column].
constexpr uint32_t ttq_format_bfp8_strip = 5;
constexpr size_t ttq_name_bytes = 128;
constexpr size_t ttq_payload_alignment = 64;

constexpr size_t requant_bfp4_strip_columns = 2880;
constexpr size_t requant_bfp4_strip_rows = 16;
constexpr size_t requant_bfp4_strip_exponent_bytes = requant_bfp4_strip_columns;
constexpr size_t requant_bfp4_strip_data_bytes = requant_bfp4_strip_columns * requant_bfp4_strip_rows / 2;
constexpr size_t requant_bfp4_strip_bytes = requant_bfp4_strip_exponent_bytes + requant_bfp4_strip_data_bytes;
constexpr size_t requant_bfp8_strip_columns = 2880;
constexpr size_t requant_bfp8_strip_rows = 16;
constexpr size_t requant_bfp8_strip_exponent_bytes = requant_bfp8_strip_columns;
constexpr size_t requant_bfp8_strip_data_bytes = requant_bfp8_strip_columns * requant_bfp8_strip_rows;
constexpr size_t requant_bfp8_strip_bytes = requant_bfp8_strip_exponent_bytes + requant_bfp8_strip_data_bytes;

struct TtqHeader {
    char magic[8];
    uint32_t version;
    uint32_t model;
    uint32_t tensor_count;
    uint32_t reserved;
    uint64_t directory_offset;
    uint64_t data_offset;
    uint32_t layer_count;
    uint32_t expert_count;
    uint32_t dim;
    uint32_t vocab;
};

struct TtqRecord {
    char name[ttq_name_bytes];
    uint32_t format;
    uint32_t rank;
    uint64_t shape[4];
    uint64_t offset;
    uint64_t size;
};

uint16_t fp32_to_bf16(float value);
uint8_t pack_bfp4_value(float value, uint8_t exponent);
uint8_t pack_bfp8_value(float value, uint8_t exponent);
void pack_bfp4_section(const float *values, uint8_t *exponent, uint8_t *bfp4_data);
void pack_bfp8_section(const float *values, uint8_t *exponent, uint8_t *bfp8_data);
void write_requant(const char *output_path, const GgufFile &file);
