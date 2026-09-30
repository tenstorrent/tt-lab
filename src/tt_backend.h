// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gguf.h"

#include <cstddef>
#include <cstdint>
#include <string>

class Tokenizer;

enum class TtFirmwareBackend {
    sim,
    silicon,
};

struct TtMeshConfig {
    static constexpr size_t max_chips = 4;
    static constexpr size_t tiles_per_chip = 8;
    size_t shard_count = 1;

    constexpr bool valid() const {
        return shard_count == 1 || shard_count == tiles_per_chip || shard_count == max_chips * tiles_per_chip;
    }
    constexpr size_t chip_count() const {
        return shard_count == max_chips * tiles_per_chip ? max_chips : 1;
    }
    constexpr size_t local_shard_count() const {
        return shard_count / chip_count();
    }
};

// Report device stage cycles and host phase timings at the end of a run.
void tt_enable_profiling();

void run_tt_firmware_inference(TtFirmwareBackend backend, const char *sim_path, const char *requant_path,
                               const GgufFile &file, const Tokenizer &tokenizer, const std::string &formatted_prompt,
                               const int32_t *prompt_tokens, size_t prompt_count, size_t generation_limit,
                               TtMeshConfig mesh);
void run_tt_firmware_proxy_check(TtFirmwareBackend backend, const char *sim_path, const char *requant_path,
                                 const GgufFile &file, int32_t token, size_t generation_limit, size_t layer_limit,
                                 TtMeshConfig mesh);
void run_device_proxy_generation(const char *requant_path, const GgufFile &file, const Tokenizer &tokenizer,
                                 const std::string &formatted_prompt, const int32_t *prompt_tokens, size_t prompt_count,
                                 size_t generation_limit);
void run_device_proxy_check(const char *requant_path, const GgufFile &file, const int32_t *prompt_tokens,
                            size_t prompt_count, size_t layer_limit);
