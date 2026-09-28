// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// Only scalar/pointer arguments cross the v3/v4 boundary; vector types stay private.
// Accumulate into caller-initialized chains in internal Dst encoding. BF16 k_tiles is a multiple of four.
[[gnu::target("default")]]
void tt_matvec_bf16(const uint8_t *tiles, const uint32_t *b_value_rows, size_t k_tiles, uint32_t chains[2][16]);
[[gnu::target("arch=x86-64-v4")]]
void tt_matvec_bf16(const uint8_t *tiles, const uint32_t *b_value_rows, size_t k_tiles, uint32_t chains[2][16]);

[[gnu::target("default")]]
void tt_matvec_bfp8(const uint8_t *strip, const uint32_t *b_value_rows, uint32_t chains[2][16]);
[[gnu::target("arch=x86-64-v4")]]
void tt_matvec_bfp8(const uint8_t *strip, const uint32_t *b_value_rows, uint32_t chains[2][16]);

inline uint16_t dst_decode_bf16_proxy(uint16_t x) {
    const uint16_t e = x & 255;
    const uint16_t m = (x >> 8) & 127;
    return uint16_t((x & 0x8000) | (e << 7) | m);
}

inline uint16_t dst_encode_bf16_proxy(uint16_t x) {
    const uint16_t e = (x >> 7) & 255;
    const uint16_t m = x & 127;
    return uint16_t((x & 0x8000) | (m << 8) | e);
}

inline uint32_t dst_decode_fp32_proxy(uint32_t x) {
    return (uint32_t(dst_decode_bf16_proxy(uint16_t(x >> 16))) << 16) | (x & 0xFFFF);
}

inline uint32_t dst_encode_fp32_proxy(uint32_t x) {
    return (uint32_t(dst_encode_bf16_proxy(uint16_t(x >> 16))) << 16) | (x & 0xFFFF);
}
