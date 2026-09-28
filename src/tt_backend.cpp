// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tt_backend.h"

#include "constants.h"
#include "requant.h"
#include "tensor.h"
#include "tokenizer.h"
#include "tt_matvec.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "ioctl.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

void close_device_fd(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}

constexpr uint64_t bar0_base = 0x100000000ull;
constexpr uint64_t bar4_base = 0x800000000ull;
constexpr uint32_t bar0_tlb_cfg_base = 0x1FC00000u;
constexpr uint64_t bar0_tlb_window_bytes = 2ull * 1024 * 1024;
constexpr uint64_t peer_bar_bytes = TtMeshConfig::tiles_per_chip * 2 * bar0_tlb_window_bytes;
constexpr uint64_t tile_reg_window_base = 0xFFA00000ull;
constexpr uint32_t bar4_first_tlb = 202;
constexpr size_t bh_dram_channels = 8;
bool profiling_enabled = false;
constexpr uint64_t bh_dram_channel_window_bytes = 4ull * 1024 * 1024 * 1024;
constexpr uint64_t bh_dram_channel_usable_bytes = 4080ull * 1024 * 1024;
constexpr uint32_t brisc_soft_reset_bit = 0x800;
constexpr uint32_t all_riscv_reset_bits = 0x47800;
constexpr uint32_t soft_reset_addr = 0xFFB121B0u;
constexpr uint32_t pcie_tile_coord = 19u | (24u << 6);
constexpr size_t gpt_oss_query_width = TT_GPT_OSS_QUERY_WIDTH;
constexpr size_t gpt_oss_kv_width = TT_GPT_OSS_KV_WIDTH;
constexpr size_t proxy_b_values_count = gpt_oss_query_width;
constexpr uint8_t brisc_program[] = {
#include "brisc.inc"
};
static_assert(sizeof(brisc_program) < TT_L1_SENTINEL_ADDR, "BRISC firmware overlaps the L1 command/control block");
static_assert(TT_BFP4_STRIP_BYTES == requant_bfp4_strip_bytes,
              "firmware BFP4 strip size must match TTQ requant layout");
static_assert(TT_BFP8_STRIP_BYTES == requant_bfp8_strip_bytes,
              "firmware BFP8 strip size must match TTQ requant layout");

struct TensixShard {
    uint32_t coord = 0;
    uint32_t dram_channel = 0;
    uint32_t dram_coord = 0;
};

/* Translated coordinates. Shard 0 reaches physical DRAM row 1; the other endpoints
   are on their consuming tile's row, keeping bulk responses on separate NoC rows. */
constexpr std::array<TensixShard, TT_MAX_SHARDS> default_tensix_shards = {{
    {1 | (2 << 6), 0, 17 | (13 << 6)},
    {1 | (3 << 6), 1, 17 | (17 << 6)},
    {1 | (4 << 6), 2, 17 | (19 << 6)},
    {1 | (5 << 6), 3, 17 | (21 << 6)},
    {1 | (6 << 6), 7, 18 | (23 << 6)},
    {1 | (8 << 6), 6, 18 | (20 << 6)},
    {1 | (10 << 6), 5, 18 | (16 << 6)},
    {1 | (11 << 6), 4, 18 | (14 << 6)},
}};

static_assert(
    [] {
        constexpr uint32_t dram_rows[] = {0, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6};
        for (size_t i = 0; i < default_tensix_shards.size(); ++i) {
            const auto &tile = default_tensix_shards[i];
            const uint32_t y = tile.dram_coord >> 6;
            if (y < 12 || y >= 24 || (tile.dram_coord & 63) != 17 + tile.dram_channel / 4 ||
                (y - 12) / 3 != tile.dram_channel % 4 || dram_rows[y - 12] != (i == 0 ? 1 : tile.coord >> 6)) {
                return false;
            }
            for (size_t j = 0; j < i; ++j) {
                if (tile.dram_channel == default_tensix_shards[j].dram_channel ||
                    (tile.coord >> 6) == (default_tensix_shards[j].coord >> 6)) {
                    return false;
                }
            }
        }
        return true;
    }(),
    "shards must own distinct DRAM channels and rows with matching endpoints");

void validate_mesh_config(TtMeshConfig mesh) {
    static_assert(TT_MAX_SHARDS == TtMeshConfig::tiles_per_chip);
    if (!mesh.valid()) {
        throw std::runtime_error("--tiles supports only 1, 8 or 32");
    }
}

struct SiliconMapping {
    uint32_t id = TENSTORRENT_MAPPING_UNUSED;
    uint64_t base = 0;
    uint64_t size = 0;
    uint8_t *map = nullptr;
};

struct SiliconDevice {
    int fd = -1;
    SiliconMapping resource0_uc;
    SiliconMapping resource2_wc;
};

struct TtSimApi {
    void *handle = nullptr;
    void (*init)() = nullptr;
    void (*exit)() = nullptr;
    uint32_t (*pci_config_rd32)(uint32_t bdf, uint32_t offset) = nullptr;
    void (*pci_mem_rd_bytes)(uint64_t paddr, void *dst, uint32_t size) = nullptr;
    void (*pci_mem_wr_bytes)(uint64_t paddr, const void *src, uint32_t size) = nullptr;
    void (*set_pci_dma_mem_callbacks)(void (*rd)(uint64_t paddr, void *dst, uint32_t size),
                                      void (*wr)(uint64_t paddr, const void *src, uint32_t size)) = nullptr;
    void (*clock)(uint32_t clocks) = nullptr;
};

struct TtFirmwareRuntime {
    TtFirmwareBackend backend = TtFirmwareBackend::sim;
    TtSimApi api;
    std::array<SiliconDevice, TtMeshConfig::max_chips> devices;
    struct ChipBars {
        uint64_t bar0 = 0;
        uint64_t bar4 = 0;
    };
    std::array<ChipBars, TtMeshConfig::max_chips> bars;
    std::array<uint32_t, TtMeshConfig::max_chips> worker_columns{1, 1, 1, 1};
    std::array<std::array<uint64_t, TtMeshConfig::max_chips>, TtMeshConfig::max_chips> peer_bars{};
    size_t chip_count = 0;
    bool sim_initialized = false;
    const std::byte *sim_embedding = nullptr;
    uint64_t sim_embedding_bytes = 0;
    std::byte *sim_logits = nullptr;
};

TtFirmwareRuntime *current_runtime;

uint32_t worker_coord(size_t chip, size_t local) {
    return (default_tensix_shards[local].coord & ~63u) | current_runtime->worker_columns[chip];
}

SiliconDevice *silicon_device(size_t chip = 0) {
    if (!current_runtime || current_runtime->backend != TtFirmwareBackend::silicon ||
        chip >= current_runtime->chip_count) {
        return nullptr;
    }
    return &current_runtime->devices[chip];
}

const TtFirmwareRuntime::ChipBars &chip_bars(size_t chip) {
    if (!current_runtime || chip >= current_runtime->chip_count) {
        throw std::runtime_error("PCI access to an unopened chip");
    }
    return current_runtime->bars[chip];
}

void copy_from_mmio_scalar(void *dst, const volatile uint8_t *src, uint32_t size) {
    if ((((uintptr_t)dst | (uintptr_t)src | size) & 3u) == 0) {
        auto *d = (uint32_t *)dst;
        const volatile auto *s = (const volatile uint32_t *)src;
        for (uint32_t i = 0; i < size / 4; ++i) {
            d[i] = s[i];
        }
    } else {
        auto *d = (uint8_t *)dst;
        for (uint32_t i = 0; i < size; ++i) {
            d[i] = src[i];
        }
    }
}

void copy_to_mmio_scalar(volatile uint8_t *dst, const void *src, uint32_t size) {
    if ((((uintptr_t)dst | (uintptr_t)src | size) & 3u) == 0) {
        volatile auto *d = (volatile uint32_t *)dst;
        const auto *s = (const uint32_t *)src;
        for (uint32_t i = 0; i < size / 4; ++i) {
            d[i] = s[i];
        }
    } else {
        const auto *s = (const uint8_t *)src;
        for (uint32_t i = 0; i < size; ++i) {
            dst[i] = s[i];
        }
    }
}

void copy_to_mmio_wc(volatile uint8_t *dst, const void *src, uint32_t size) {
    while (((uintptr_t)dst & 31u) != 0 && size >= 32) {
        if ((((uintptr_t)dst | (uintptr_t)src) & 3u) == 0) {
            *(volatile uint32_t *)dst = *(const uint32_t *)src;
            dst += 4;
            src = (const uint8_t *)src + 4;
            size -= 4;
        } else {
            *dst++ = *(const uint8_t *)src;
            src = (const uint8_t *)src + 1;
            --size;
        }
    }
    if (((uintptr_t)dst & 31u) == 0 && size >= 32) {
        const uint32_t vectors = size / 32;
        volatile auto *d = (volatile __m256i *)dst;
        const auto *s = (const __m256i *)src;
        for (uint32_t i = 0; i < vectors; ++i) {
            d[i] = _mm256_loadu_si256(s + i);
        }
        const uint32_t bytes = vectors * 32;
        dst += bytes;
        src = (const uint8_t *)src + bytes;
        size -= bytes;
    }
    if (size == 0) {
        return;
    }
    copy_to_mmio_scalar(dst, src, size);
}

struct SiliconAddress {
    volatile uint8_t *ptr;
    bool wc;
};

SiliconAddress silicon_pci_mem_ptr(uint64_t paddr, uint32_t size) {
    const size_t chip = size_t(paddr >> 36);
    const SiliconDevice *device = silicon_device(chip);
    if (!device) {
        throw std::runtime_error("silicon PCI memory access to an unopened chip");
    }
    const auto &bars = chip_bars(chip);
    const auto in_mapping = [&](const SiliconMapping &mapping, uint64_t base) {
        return paddr >= base && size <= mapping.size && paddr - base <= mapping.size - size;
    };
    if (in_mapping(device->resource0_uc, bars.bar0)) {
        return {device->resource0_uc.map + (paddr - bars.bar0), false};
    }
    if (in_mapping(device->resource2_wc, bars.bar4)) {
        return {device->resource2_wc.map + (paddr - bars.bar4), true};
    }
    throw std::runtime_error("silicon PCI memory address is outside mapped BARs");
}

void silicon_pci_mem_rd_bytes(uint64_t paddr, void *dst, uint32_t size) {
    copy_from_mmio_scalar(dst, silicon_pci_mem_ptr(paddr, size).ptr, size);
}

void silicon_pci_mem_wr_bytes(uint64_t paddr, const void *src, uint32_t size) {
    const SiliconAddress target = silicon_pci_mem_ptr(paddr, size);
    if (target.wc) {
        copy_to_mmio_wc(target.ptr, src, size);
    } else {
        copy_to_mmio_scalar(target.ptr, src, size);
    }
    _mm_sfence();
}

void silicon_clock(uint32_t) {}

struct DramRegion {
    uint32_t channel = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct TtqTensor {
    std::string name;
    uint32_t format = 0;
    std::vector<uint64_t> shape;
    const std::byte *data = nullptr;
    uint64_t size = 0;
    std::array<uint32_t, TT_MAX_GLOBAL_SHARDS> stripe_channel = {};
    std::array<uint64_t, TT_MAX_GLOBAL_SHARDS> stripe_offset = {};
    std::array<DramRegion, TtMeshConfig::max_chips> chip_regions = {};
    bool dram_striped = false;
    bool dram_loaded = false;
};

struct DramCursor {
    explicit DramCursor(TtMeshConfig mesh) : offset(mesh.chip_count() * bh_dram_channels) {
        validate_mesh_config(mesh);
    }
    std::vector<uint64_t> offset;
};

class TtqFile {
  public:
    explicit TtqFile(const char *path) {
        fd_ = open(path, O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open TTQ file " + std::string(path) + ": " + std::strerror(errno));
        }

        struct stat status {};
        if (fstat(fd_, &status) != 0) {
            throw std::runtime_error("cannot stat TTQ file " + std::string(path) + ": " + std::strerror(errno));
        }
        if (status.st_size < 0 || uint64_t(status.st_size) > SIZE_MAX) {
            throw std::runtime_error("TTQ file is too large");
        }
        size_ = size_t(status.st_size);
        void *mapping = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping == MAP_FAILED) {
            throw std::runtime_error("cannot mmap TTQ file " + std::string(path) + ": " + std::strerror(errno));
        }
        bytes_ = (const std::byte *)mapping;
        parse();
    }

    ~TtqFile() {
        if (bytes_) {
            munmap((void *)bytes_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    TtqFile(const TtqFile &) = delete;
    TtqFile &operator=(const TtqFile &) = delete;

    const TtqTensor &tensor(std::string_view name) const {
        const auto found = index_.find(std::string(name));
        if (found == index_.end()) {
            throw std::runtime_error("missing TTQ tensor " + std::string(name));
        }
        return tensors_[found->second];
    }

    std::vector<TtqTensor> &tensors() {
        return tensors_;
    }

    size_t layer_count() const {
        return layer_count_;
    }

    size_t expert_count() const {
        return expert_count_;
    }

    void validate_model(const GgufFile &file) const {
        if (file.string("general.architecture") != "gpt-oss" ||
            file.integer("gpt-oss.embedding_length") != TT_GPT_OSS_DIM ||
            file.integer("gpt-oss.block_count") != layer_count_ ||
            file.integer("gpt-oss.expert_count") != expert_count_) {
            throw std::runtime_error("GGUF model shape does not match TTQ");
        }
    }

  private:
    void require_range(uint64_t offset, uint64_t size) const {
        if (offset > size_ || size > size_ - offset) {
            throw std::runtime_error("TTQ record points outside file");
        }
    }

    void parse() {
        require_range(0, sizeof(TtqHeader));
        TtqHeader header;
        std::memcpy(&header, bytes_, sizeof(header));
        const char expected_magic[8] = {'T', 'T', 'L', 'B', 'Q', 'N', 'T', '1'};

        if (std::memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0) {
            throw std::runtime_error("TTQ file has wrong magic; expected TTLBQNT1");
        }
        const bool model = (header.model == 20 && header.layer_count == TT_GPT_OSS_20B_LAYERS &&
                            header.expert_count == TT_GPT_OSS_20B_EXPERTS) ||
                           (header.model == 120 && header.layer_count == TT_GPT_OSS_120B_LAYERS &&
                            header.expert_count == TT_GPT_OSS_120B_EXPERTS);
        if (header.version != 1 || !model || header.dim != TT_GPT_OSS_DIM || header.vocab != TT_GPT_OSS_VOCAB) {
            throw std::runtime_error("unsupported TTQ file version/model/shape");
        }
        layer_count_ = header.layer_count;
        expert_count_ = header.expert_count;
        require_range(header.directory_offset, uint64_t(header.tensor_count) * sizeof(TtqRecord));
        const std::byte *records = bytes_ + header.directory_offset;
        tensors_.resize(header.tensor_count);
        for (uint32_t i = 0; i < header.tensor_count; ++i) {
            TtqRecord record;
            std::memcpy(&record, records + uint64_t(i) * sizeof(TtqRecord), sizeof(record));
            require_range(record.offset, record.size);
            if (record.format == 4) {
                throw std::runtime_error("obsolete output-axis BFP8 TTQ; requant with the current writer");
            }
            size_t name_size = 0;
            while (name_size < ttq_name_bytes && record.name[name_size]) {
                ++name_size;
            }
            if (name_size == ttq_name_bytes) {
                throw std::runtime_error("unterminated TTQ tensor name");
            }
            if (record.rank > 4) {
                throw std::runtime_error("TTQ tensor rank is too large");
            }
            TtqTensor &tensor = tensors_[i];
            tensor.name.assign(record.name, name_size);
            tensor.format = record.format;
            tensor.shape.resize(record.rank);
            for (uint32_t d = 0; d < record.rank; ++d) {
                tensor.shape[d] = record.shape[d];
            }
            tensor.data = bytes_ + record.offset;
            tensor.size = record.size;
            index_.emplace(tensor.name, i);
        }
    }

    int fd_ = -1;
    size_t size_ = 0;
    size_t layer_count_ = 0;
    size_t expert_count_ = 0;
    const std::byte *bytes_ = nullptr;
    std::vector<TtqTensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

template<class T> void load_symbol(TtSimApi &api, T &dst, const char *name) {
    dlerror();
    dst = (T)dlsym(api.handle, name);
    const char *error = dlerror();
    if (error || !dst) {
        throw std::runtime_error("missing libttsim symbol " + std::string(name));
    }
}

bool ttsim_peer_address(uint64_t paddr, uint32_t size) {
    if (!current_runtime) {
        return false;
    }
    // Peer DMA targets the BAR0 L1/register windows, not host pointers.
    for (size_t chip = 0; chip < current_runtime->chip_count; ++chip) {
        const uint64_t base = chip_bars(chip).bar0;
        if (paddr >= base && size <= peer_bar_bytes && paddr - base <= peer_bar_bytes - size) {
            return true;
        }
    }
    return false;
}

constexpr uint64_t sim_embedding_noc_base = 0x800000000000ull;
constexpr uint64_t sim_logits_noc_base = 0x900000000000ull;
constexpr uint64_t dma_logits_bytes = TT_GPT_OSS_VOCAB * sizeof(uint16_t);

void ttsim_dma_rd_bytes(uint64_t paddr, void *dst, uint32_t size) {
    if (ttsim_peer_address(paddr, size)) {
        current_runtime->api.pci_mem_rd_bytes(paddr, dst, size);
        return;
    }
    const uint64_t at = paddr - sim_embedding_noc_base;
    if (!current_runtime || !current_runtime->sim_embedding || paddr < sim_embedding_noc_base ||
        at > current_runtime->sim_embedding_bytes || size > current_runtime->sim_embedding_bytes - at) {
        throw std::runtime_error("device read host memory outside the embedding table");
    }
    std::memcpy(dst, current_runtime->sim_embedding + at, size);
}

void ttsim_dma_wr_bytes(uint64_t paddr, const void *src, uint32_t size) {
    if (ttsim_peer_address(paddr, size)) {
        current_runtime->api.pci_mem_wr_bytes(paddr, src, size);
        return;
    }
    const uint64_t at = paddr - sim_logits_noc_base;
    if (!current_runtime || !current_runtime->sim_logits || paddr < sim_logits_noc_base || at > dma_logits_bytes ||
        size > dma_logits_bytes - at) {
        throw std::runtime_error("device wrote host memory outside the logits buffer");
    }
    std::memcpy(current_runtime->sim_logits + at, src, size);
}

void write_u32(TtSimApi &api, uint64_t addr, uint32_t value) {
    api.pci_mem_wr_bytes(addr, &value, sizeof(value));
}

uint32_t read_u32(TtSimApi &api, uint64_t addr) {
    uint32_t value = 0;
    api.pci_mem_rd_bytes(addr, &value, sizeof(value));
    return value;
}

uint32_t dram_coord(size_t channel) {
    static constexpr uint32_t coords[bh_dram_channels] = {
        17 | (12 << 6), 17 | (15 << 6), 17 | (18 << 6), 17 | (21 << 6),
        18 | (12 << 6), 18 | (15 << 6), 18 | (18 << 6), 18 | (21 << 6),
    };
    return coords[channel % bh_dram_channels];
}

uint32_t dram_coord(size_t channel, size_t shard) {
    if (channel / bh_dram_channels != shard / TtMeshConfig::tiles_per_chip) {
        throw std::runtime_error("firmware DRAM reference crosses chips");
    }
    const auto &tile = default_tensix_shards[shard % TtMeshConfig::tiles_per_chip];

    return channel % bh_dram_channels == tile.dram_channel ? tile.dram_coord : dram_coord(channel);
}

void write_tlb_cfg(TtSimApi &api, size_t chip, uint32_t tlb, uint32_t cfg0, uint32_t cfg1, uint32_t cfg2) {
    const uint64_t cfg_addr = chip_bars(chip).bar0 + bar0_tlb_cfg_base + uint64_t(tlb * 3) * 4;
    write_u32(api, cfg_addr + 0, cfg0);
    write_u32(api, cfg_addr + 4, cfg1);
    write_u32(api, cfg_addr + 8, cfg2);
}

void configure_bar0_window(TtSimApi &api, size_t chip, uint32_t window, uint32_t coord, uint64_t local_addr) {
    const uint64_t addr_bits = local_addr >> 21;
    const uint32_t cfg0 = uint32_t(addr_bits);
    const uint32_t cfg1 = uint32_t(addr_bits >> 32) | (coord << 11);
    write_tlb_cfg(api, chip, window, cfg0, cfg1, 0);
}

void configure_bar4_window(TtSimApi &api, size_t chip, uint32_t window, uint32_t coord, uint64_t local_addr) {
    write_tlb_cfg(api, chip, bar4_first_tlb + window, uint32_t(local_addr >> 32), coord, 0);
}

void configure_tile_windows(TtSimApi &api, TtMeshConfig mesh) {
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        const size_t chip = shard / TtMeshConfig::tiles_per_chip;
        const size_t local = shard % TtMeshConfig::tiles_per_chip;
        configure_bar0_window(api, chip, uint32_t(local * 2), worker_coord(chip, local), 0);
        configure_bar0_window(api, chip, uint32_t(local * 2 + 1), worker_coord(chip, local), tile_reg_window_base);
    }
}

void configure_dram_windows(TtSimApi &api, TtMeshConfig mesh) {
    for (size_t chip = 0; chip < mesh.chip_count(); ++chip) {
        for (size_t channel = 0; channel < bh_dram_channels; ++channel) {
            configure_bar4_window(api, chip, uint32_t(channel), dram_coord(channel), 0);
        }
    }
}

uint64_t tile_l1_addr(size_t shard, uint32_t addr) {
    return chip_bars(shard / TtMeshConfig::tiles_per_chip).bar0 +
           uint64_t((shard % TtMeshConfig::tiles_per_chip) * 2) * bar0_tlb_window_bytes + addr;
}

uint64_t tile_reg_addr(size_t shard, uint32_t addr) {
    return chip_bars(shard / TtMeshConfig::tiles_per_chip).bar0 +
           uint64_t((shard % TtMeshConfig::tiles_per_chip) * 2 + 1) * bar0_tlb_window_bytes + uint64_t(addr) -
           tile_reg_window_base;
}

void configure_worker_placement(TtSimApi &api, TtMeshConfig mesh) {
    if (mesh.chip_count() == 1) {
        return;
    }
    for (size_t chip = 0; chip < mesh.chip_count(); ++chip) {
        const size_t shard = chip * TtMeshConfig::tiles_per_chip;
        const auto reg = [&](uint32_t offset) { return read_u32(api, tile_reg_addr(shard, 0xFFB20000u + offset)); };
        const auto translated_x = [&](uint32_t x) { return (reg(0x118u + x / 6u * 4u) >> (x % 6u * 5u)) & 31u; };
        if (!(reg(0x100u) & (1u << 14)) || reg(0x150u) != 0 || reg(0x154u) != 3) {
            throw std::runtime_error("unexpected worker coordinate translation configuration");
        }
        const uint32_t pcie_column = translated_x(19u);
        const uint32_t harvested = reg(0x108u);
        uint32_t column = 0;
        uint32_t fallback = 0;
        for (uint32_t x : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 10u, 11u, 12u, 13u, 14u, 15u, 16u}) {
            const uint32_t physical = translated_x(x);
            if (physical < 1u || physical > 16u || physical == 8u || physical == 9u || (harvested & (1u << physical))) {
                continue;
            }
            if (!fallback) {
                fallback = x;
            }
            if (physical == pcie_column) {
                column = x;
            }
        }
        if (!column) {
            if (!fallback) {
                throw std::runtime_error("no active Tensix column available");
            }
            column = fallback;
            std::fprintf(stderr,
                         "warning: chip %zu PCIe-aligned physical Tensix column %u is unavailable (harvested); "
                         "using logical column %u (physical %u), which may reduce PCIe performance\n",
                         chip, pcie_column, column, translated_x(column));
        }
        const uint32_t physical = translated_x(column);
        current_runtime->worker_columns[chip] = column;
        // Use the bootstrap register window to verify every candidate before starting firmware.
        for (size_t local = 0; local < TtMeshConfig::tiles_per_chip; ++local) {
            const uint32_t coord = worker_coord(chip, local);
            configure_bar0_window(api, chip, 1u, coord, tile_reg_window_base);
            if ((reg(0x44u) & 4095u) != ((coord & ~63u) | physical) || (reg(0x148u) & 4095u) != coord ||
                (reg(0x108u) & (1u << physical)) || (reg(0x110u) & (1u << (coord >> 6)))) {
                throw std::runtime_error("worker placement failed physical/logical/harvesting verification");
            }
        }
    }
    configure_tile_windows(api, mesh);
}

uint64_t dram_addr(size_t channel, uint64_t offset) {
    return chip_bars(channel / bh_dram_channels).bar4 +
           uint64_t(channel % bh_dram_channels) * bh_dram_channel_window_bytes + offset;
}

void write_mesh_config_to_l1(TtSimApi &api, TtMeshConfig mesh) {
    static_assert(TtMeshConfig::max_chips - 1 <= TtMeshConfig::tiles_per_chip);
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        write_u32(api, tile_l1_addr(shard, TT_L1_SHARD_INDEX_ADDR), uint32_t(shard));
        write_u32(api, tile_l1_addr(shard, TT_L1_SHARD_COUNT_ADDR), uint32_t(mesh.shard_count));
        write_u32(api, tile_l1_addr(shard, TT_L1_PEER_INDEX_ADDR), uint32_t(shard % TtMeshConfig::tiles_per_chip));
        write_u32(api, tile_l1_addr(shard, TT_L1_PEER_COUNT_ADDR), uint32_t(mesh.local_shard_count()));
        TtPcieExchange exchange{};
        const size_t chip = shard / TtMeshConfig::tiles_per_chip;
        const size_t courier = shard % TtMeshConfig::tiles_per_chip;
        if (courier + 1 < mesh.chip_count()) {
            const uint64_t offset = courier * 2 * bar0_tlb_window_bytes;
            const uint64_t send = current_runtime->peer_bars[chip][(chip + courier + 1) % mesh.chip_count()] + offset;
            const uint64_t source =
                current_runtime->peer_bars[chip][(chip + mesh.chip_count() - courier - 1) % mesh.chip_count()] + offset;
            exchange.send_lo = uint32_t(send);
            exchange.send_hi = uint32_t(send >> 32);
            exchange.source_lo = uint32_t(source);
            exchange.source_hi = uint32_t(source >> 32);
            exchange.coord = pcie_tile_coord;
        }
        api.pci_mem_wr_bytes(tile_l1_addr(shard, TT_PCIE_EXCHANGE_ADDR), &exchange, sizeof(exchange));
        for (size_t peer = 0; peer < TT_MAX_SHARDS; ++peer) {
            write_u32(api, tile_l1_addr(shard, TT_L1_SHARD_COORD_ADDR + uint32_t(peer * sizeof(uint32_t))),
                      worker_coord(chip, peer));
        }
    }
}

uint64_t align_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

void write_dram_bytes(TtSimApi &api, uint32_t channel, uint64_t offset, const std::byte *data, uint64_t size) {
    constexpr uint64_t chunk_size = 64ull * 1024 * 1024;
    uint64_t copied = 0;
    while (copied < size) {
        const uint64_t remaining = size - copied;
        const uint32_t chunk = uint32_t(std::min(remaining, chunk_size));
        api.pci_mem_wr_bytes(dram_addr(channel, offset + copied), data + copied, chunk);
        copied += chunk;
    }
}

void validate_bfp_tensor_strip_geometry(const TtqTensor &tensor) {
    uint64_t strip_bytes = 0;
    constexpr uint64_t strips_per_expert = TT_GPT_OSS_DIM / TT_MVMUL_TILE_DIM;
    const char *format_name = nullptr;

    if (tensor.format == ttq_format_bfp4_strip) {
        strip_bytes = TT_BFP4_STRIP_BYTES;
        format_name = "BFP4";
    } else if (tensor.format == ttq_format_bfp8_strip) {
        strip_bytes = TT_BFP8_STRIP_BYTES;
        format_name = "BFP8";
    } else {
        return;
    }
    if ((tensor.size % strip_bytes) != 0) {
        throw std::runtime_error(std::string(format_name) +
                                 " tensor is not a whole number of 2880x16 strips: " + tensor.name);
    }
    if (tensor.shape.size() != 3 || tensor.shape[0] != TT_GPT_OSS_DIM || tensor.shape[1] != TT_GPT_OSS_DIM ||
        (tensor.shape[2] != TT_GPT_OSS_20B_EXPERTS && tensor.shape[2] != TT_GPT_OSS_120B_EXPERTS)) {
        throw std::runtime_error(std::string(format_name) + " tensor has unsupported silicon geometry: " + tensor.name);
    }
    if (tensor.size / strip_bytes != strips_per_expert * tensor.shape[2]) {
        throw std::runtime_error(std::string(format_name) +
                                 " tensor has unexpected silicon strip count: " + tensor.name);
    }
}

bool is_fp32_bias(const TtqTensor &tensor) {
    return tensor.name.ends_with("attn_q.bias") || tensor.name.ends_with("attn_k.bias") ||
           tensor.name.ends_with("attn_output.bias") || tensor.name.ends_with("ffn_gate_inp.bias");
}

bool should_preload_ttq_tensor(const TtqTensor &tensor) {
    return tensor.format != ttq_format_bfp4_strip && !is_fp32_bias(tensor);
}

bool should_widen_to_f32(const TtqTensor &tensor) {
    return tensor.format == ttq_format_bf16_linear && tensor.name.ends_with("_norm.weight");
}

bool is_bfp8_expert_weight(const TtqTensor &tensor) {
    if (tensor.format != ttq_format_bfp8_strip) {
        return false;
    }
    return tensor.name.ends_with("ffn_gate_exps.weight.bfp8") || tensor.name.ends_with("ffn_up_exps.weight.bfp8") ||
           tensor.name.ends_with("ffn_down_exps.weight.bfp8");
}

bool is_output_weight(const TtqTensor &tensor) {
    return tensor.name == "output.weight";
}

bool is_attention_output_weight(const TtqTensor &tensor) {
    return tensor.name.ends_with("attn_output.weight");
}

bool is_qkv_projection_weight(const TtqTensor &tensor) {
    return tensor.name.ends_with("attn_q.weight") || tensor.name.ends_with("attn_k.weight") ||
           tensor.name.ends_with("attn_v.weight");
}

bool is_expert_down_weight(const TtqTensor &tensor) {
    return tensor.name.ends_with("ffn_down_exps.weight.bfp8");
}

bool is_router_weight(const TtqTensor &tensor) {
    return tensor.format == ttq_format_bf16_tensix && tensor.name.ends_with("ffn_gate_inp.weight");
}

DramRegion alloc_dram_region_in_channel(DramCursor &cursor, uint32_t channel, uint64_t size, uint64_t alignment) {
    if (channel >= cursor.offset.size()) {
        throw std::runtime_error("DRAM allocation channel is outside the mesh");
    }
    uint64_t offset = align_u64(cursor.offset[channel], alignment);

    if (size > bh_dram_channel_usable_bytes) {
        throw std::runtime_error("requested simulator DRAM allocation is larger than one channel");
    }
    if (offset > bh_dram_channel_usable_bytes || size > bh_dram_channel_usable_bytes - offset) {
        const uint64_t available = offset < bh_dram_channel_usable_bytes ? bh_dram_channel_usable_bytes - offset : 0;

        throw std::runtime_error("simulator DRAM allocation of " + std::to_string(size >> 20) +
                                 " MiB does not fit in channel " + std::to_string(channel) + " (" +
                                 std::to_string(available >> 20) + " MiB free)");
    }

    DramRegion region;
    region.channel = channel;
    region.offset = offset;
    region.size = size;
    cursor.offset[channel] = offset + size;
    return region;
}

DramRegion alloc_dram_region(DramCursor &cursor, uint64_t size, uint64_t alignment, size_t chip = 0) {
    if (chip >= cursor.offset.size() / bh_dram_channels) {
        throw std::runtime_error("DRAM allocation chip is outside the mesh");
    }
    const size_t first = chip * bh_dram_channels;
    const size_t end = first + bh_dram_channels;
    size_t chosen = end;

    if (size > bh_dram_channel_usable_bytes) {
        throw std::runtime_error("requested simulator DRAM allocation is larger than one channel");
    }
    for (size_t channel = first; channel < end; ++channel) {
        const uint64_t offset = align_u64(cursor.offset[channel], alignment);

        if (offset <= bh_dram_channel_usable_bytes && size <= bh_dram_channel_usable_bytes - offset) {
            if (chosen == end || offset < align_u64(cursor.offset[chosen], alignment)) {
                chosen = channel;
            }
        }
    }
    if (chosen == end) {
        throw std::runtime_error("DRAM allocations do not fit in chip " + std::to_string(chip));
    }
    return alloc_dram_region_in_channel(cursor, uint32_t(chosen), size, alignment);
}

void report_dram_usage(const DramCursor &cursor) {
    const auto peak = std::max_element(cursor.offset.begin(), cursor.offset.end());

    std::fprintf(stderr, "peak %.1f of %.0f MiB (channel %zu)\n", double(*peak) / (1024.0 * 1024.0),
                 double(bh_dram_channel_usable_bytes) / (1024.0 * 1024.0), size_t(peak - cursor.offset.begin()));
}

uint32_t shard_dram_channel(size_t shard) {
    return uint32_t(shard / TtMeshConfig::tiles_per_chip * bh_dram_channels) +
           default_tensix_shards[shard % TtMeshConfig::tiles_per_chip].dram_channel;
}

void preload_embedding_stripes(TtSimApi &api, TtqTensor &tensor, DramCursor &cursor, TtMeshConfig mesh) {
    constexpr uint64_t row_bytes = TT_GPT_OSS_DIM * sizeof(uint16_t);
    const uint64_t stripe_bytes = (TT_GPT_OSS_VOCAB / mesh.shard_count) * row_bytes;

    if (tensor.format != ttq_format_bf16_linear || tensor.shape.size() != 2 || tensor.shape[0] != TT_GPT_OSS_DIM ||
        tensor.shape[1] != TT_GPT_OSS_VOCAB || tensor.size != TT_GPT_OSS_VOCAB * row_bytes) {
        throw std::runtime_error("token embedding has unsupported geometry");
    }
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        const DramRegion region =
            mesh.shard_count == 1 ? alloc_dram_region(cursor, stripe_bytes, 64)
                                  : alloc_dram_region_in_channel(cursor, shard_dram_channel(shard), stripe_bytes, 64);

        write_dram_bytes(api, region.channel, region.offset, tensor.data + shard * stripe_bytes, stripe_bytes);
        tensor.stripe_channel[shard] = region.channel;
        tensor.stripe_offset[shard] = region.offset;
    }
    tensor.dram_striped = true;
    tensor.dram_loaded = true;
}

void preload_tensor_per_chip(TtSimApi &api, TtqTensor &tensor, DramCursor &cursor, TtMeshConfig mesh,
                             const std::byte *data, uint64_t bytes) {
    for (size_t chip = 0; chip < mesh.chip_count(); ++chip) {
        const DramRegion region = alloc_dram_region(cursor, bytes, 64, chip);
        tensor.chip_regions[chip] = region;
        write_dram_bytes(api, region.channel, region.offset, data, bytes);
    }
    tensor.dram_loaded = true;
}

void widen_tensor_to_f32_dram(TtSimApi &api, TtqTensor &tensor, DramCursor &cursor, TtMeshConfig mesh) {
    const size_t count = size_t(tensor.size / sizeof(uint16_t));
    const uint16_t *source = (const uint16_t *)tensor.data;
    std::vector<float> widened(count);

    for (size_t i = 0; i < count; ++i) {
        widened[i] = std::bit_cast<float>(uint32_t(source[i]) << 16);
    }

    preload_tensor_per_chip(api, tensor, cursor, mesh, (const std::byte *)widened.data(),
                            uint64_t(count) * sizeof(float));
}

void preload_router_weight(TtSimApi &api, TtqTensor &tensor, DramCursor &cursor, TtMeshConfig mesh) {
    if (tensor.format != ttq_format_bf16_tensix || tensor.shape.size() != 2 || tensor.shape[0] != TT_GPT_OSS_DIM ||
        (tensor.shape[1] != TT_GPT_OSS_20B_EXPERTS && tensor.shape[1] != TT_GPT_OSS_120B_EXPERTS) ||
        tensor.size != tensor.shape[0] * tensor.shape[1] * sizeof(uint16_t)) {
        throw std::runtime_error("router weight has unsupported geometry: " + tensor.name);
    }
    const size_t parts = tt_router_shards(uint32_t(tensor.shape[1]), uint32_t(mesh.local_shard_count()));
    const uint64_t bytes = tensor.size / parts;
    uint64_t common_offset = 0;

    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        const uint32_t channel = shard_dram_channel(shard);
        const uint64_t offset = align_u64(cursor.offset[channel], 64);

        common_offset = std::max(common_offset, offset);
    }
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        const uint32_t channel = shard_dram_channel(shard);

        cursor.offset[channel] = common_offset;
        const DramRegion region = alloc_dram_region_in_channel(cursor, channel, bytes, 64);

        if (region.offset + region.size > uint64_t(UINT32_MAX)) {
            throw std::runtime_error("replicated tensor offset exceeds firmware address range: " + tensor.name);
        }
        tensor.stripe_channel[shard] = region.channel;
        tensor.stripe_offset[shard] = region.offset;
        write_dram_bytes(api, region.channel, region.offset, tensor.data + (shard % parts) * bytes, bytes);
    }
    tensor.dram_striped = true;
    tensor.dram_loaded = true;
}

enum class StripeKind {
    none,
    expert,
    logits,
    attention_output,
    qkv
};

StripeKind stripe_kind(const TtqTensor &tensor) {
    if (is_bfp8_expert_weight(tensor)) {
        return StripeKind::expert;
    }
    if (is_output_weight(tensor)) {
        return StripeKind::logits;
    }
    if (is_attention_output_weight(tensor)) {
        return StripeKind::attention_output;
    }
    if (is_qkv_projection_weight(tensor)) {
        return StripeKind::qkv;
    }
    return StripeKind::none;
}

// Each item is a matrix stored in output-row blocks; expert weights have multiple items.
void stripe_tensor_to_dram(TtSimApi &api, TtqTensor &tensor, DramCursor &cursor, TtMeshConfig mesh,
                           uint64_t block_bytes, uint32_t blocks, uint32_t items, const uint32_t *bounds) {
    if (uint64_t(items) * blocks * block_bytes != tensor.size || bounds[0] != 0 || bounds[mesh.shard_count] != blocks) {
        throw std::runtime_error("striped weight size does not match its block geometry: " + tensor.name);
    }
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        if (bounds[shard + 1] <= bounds[shard] || bounds[shard + 1] > blocks) {
            throw std::runtime_error("striped weight has invalid shard boundaries: " + tensor.name);
        }
        const uint64_t slice_bytes = uint64_t(bounds[shard + 1] - bounds[shard]) * block_bytes;
        const uint32_t channel = shard_dram_channel(shard);
        const DramRegion region = alloc_dram_region_in_channel(cursor, channel, uint64_t(items) * slice_bytes, 64);

        if (region.offset + region.size > uint64_t(UINT32_MAX)) {
            throw std::runtime_error("striped tensor offset exceeds firmware address range: " + tensor.name);
        }
        for (uint32_t item = 0; item < items; ++item) {
            write_dram_bytes(api, region.channel, region.offset + uint64_t(item) * slice_bytes,
                             tensor.data + (uint64_t(item) * blocks + bounds[shard]) * block_bytes, slice_bytes);
        }
        tensor.stripe_channel[shard] = region.channel;
        tensor.stripe_offset[shard] = region.offset;
    }
    tensor.dram_striped = true;
    tensor.dram_loaded = true;
}

void preload_weight_stripes(TtSimApi &api, TtqTensor &tensor, DramCursor &cursor, TtMeshConfig mesh, StripeKind kind) {
    const bool expert = kind == StripeKind::expert;
    const bool residual = kind == StripeKind::attention_output || (expert && is_expert_down_weight(tensor));
    const uint32_t columns = kind == StripeKind::attention_output ? TT_GPT_OSS_QUERY_WIDTH : TT_GPT_OSS_DIM;
    const uint32_t blocks = kind == StripeKind::logits ? TT_LOGITS_ROW_BLOCKS
                            : kind == StripeKind::qkv
                                ? (tensor.name.ends_with("attn_q.weight") ? TT_QUERY_ROW_BLOCKS : TT_KV_ROW_BLOCKS)
                                : TT_DIM_ROW_BLOCKS;
    const uint64_t block_bytes =
        expert ? TT_BFP8_STRIP_BYTES : uint64_t(columns / TT_MVMUL_TILE_DIM) * TT_MVMUL_BF16_A_TILE_BYTES;
    uint32_t items = 1;
    if (expert) {
        validate_bfp_tensor_strip_geometry(tensor);
        items = uint32_t(tensor.shape[2]);
    } else if (tensor.format != ttq_format_bf16_tensix || tensor.shape.size() != 2 || tensor.shape[0] != columns ||
               tensor.shape[1] != uint64_t(blocks) * TT_MVMUL_TILE_DIM) {
        throw std::runtime_error("striped weight has unsupported silicon geometry: " + tensor.name);
    }
    std::array<uint32_t, TT_MAX_GLOBAL_SHARDS + 1> bounds;
    for (uint32_t shard = 0; shard <= mesh.shard_count; ++shard) {
        const uint32_t shards = uint32_t(mesh.shard_count);
        bounds[shard] = residual                     ? tt_residual_block_bound(shard, shards)
                        : expert                     ? tt_hidden_block_bound(shard, shards)
                        : kind == StripeKind::logits ? tt_logits_block_bound(shard, shards)
                                                     : tt_attention_block_bound(blocks, shard, shards);
    }
    stripe_tensor_to_dram(api, tensor, cursor, mesh, block_bytes, blocks, items, bounds.data());
}

double preload_ttq_tensors_to_dram(TtSimApi &api, TtqFile &requant, const GgufFile &file, DramCursor &cursor,
                                   TtMeshConfig mesh) {
    uint64_t total = 0;
    const auto begin = std::chrono::steady_clock::now();

    if (mesh.shard_count > 1) {
        for (TtqTensor &tensor : requant.tensors()) {
            if (!is_router_weight(tensor)) {
                continue;
            }
            preload_router_weight(api, tensor, cursor, mesh);
            total += uint64_t(tensor.size) *
                     (requant.expert_count() == TT_GPT_OSS_120B_EXPERTS ? mesh.chip_count() : mesh.shard_count);
        }
        // Preserve placement order while sharing the stripe loader across tensor classes.
        for (StripeKind kind :
             {StripeKind::expert, StripeKind::logits, StripeKind::attention_output, StripeKind::qkv}) {
            for (TtqTensor &tensor : requant.tensors()) {
                if (stripe_kind(tensor) != kind) {
                    continue;
                }
                preload_weight_stripes(api, tensor, cursor, mesh, kind);
                total += tensor.size;
            }
        }
    }
    for (TtqTensor &tensor : requant.tensors()) {
        if (!should_widen_to_f32(tensor)) {
            continue;
        }
        widen_tensor_to_f32_dram(api, tensor, cursor, mesh);
        total += tensor.size * 2 * mesh.chip_count();
    }
    for (TtqTensor &tensor : requant.tensors()) {
        if (!should_preload_ttq_tensor(tensor)) {
            continue;
        }
        if (tensor.dram_loaded) {
            continue;
        }
        if (tensor.name == "token_embd.weight") {
            if (mesh.chip_count() > 1) {
                continue;
            }
            preload_embedding_stripes(api, tensor, cursor, mesh);
            total += tensor.size;
            continue;
        }
        validate_bfp_tensor_strip_geometry(tensor);
        preload_tensor_per_chip(api, tensor, cursor, mesh, tensor.data, tensor.size);
        total += tensor.size * mesh.chip_count();
    }

    // These TTQ records provide placement metadata only; never load their BF16 payloads.
    for (TtqTensor &tensor : requant.tensors()) {
        if (!is_fp32_bias(tensor)) {
            continue;
        }
        const GgufTensor *source = file.tensor(tensor.name);
        if (tensor.format != ttq_format_bf16_linear || tensor.shape.size() != 1 || !source ||
            source->type != tensor_f32 || source->shape != tensor.shape ||
            tensor.size != tensor.shape[0] * sizeof(uint16_t)) {
            throw std::runtime_error("FP32 bias has unsupported geometry: " + tensor.name);
        }
        const uint64_t bytes = tensor.shape[0] * sizeof(float);
        preload_tensor_per_chip(api, tensor, cursor, mesh, source->data, bytes);
        total += bytes * mesh.chip_count();
    }

    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    size_t used_channels = 0;
    for (uint64_t offset : cursor.offset) {
        if (offset != 0) {
            ++used_channels;
        }
    }
    /* report_dram_usage finishes this line after allocating KV cache and scratch. */
    std::fprintf(stderr, "loaded TTQ tensors into device DRAM: %.3f GiB, %u channels, %.3f s, ",
                 double(total) / (1024.0 * 1024.0 * 1024.0), unsigned(used_channels), seconds);
    return seconds;
}

float bf16_to_fp32(uint16_t value) {
    return std::bit_cast<float>(uint32_t(value) << 16);
}

uint32_t proxy_denormals_as_zeros(uint32_t value) {
    if ((value & 0x7FFFFFFFu) < 0x800000u) {
        value &= 0x80000000u;
    }
    return value;
}

uint32_t sfpu_mul_bh(uint32_t a, uint32_t b) {
    a = proxy_denormals_as_zeros(a);
    b = proxy_denormals_as_zeros(b);
    uint32_t ret = std::bit_cast<uint32_t>(std::bit_cast<float>(a) * std::bit_cast<float>(b));
    if ((ret & 0x7FFFFFFFu) > 0x7F800000u) {
        return 0x7FC00000u;
    }
    ret = proxy_denormals_as_zeros(ret);
    const uint32_t a_exp = (a >> 23) & 255u;
    const uint32_t b_exp = (b >> 23) & 255u;
    if (!(ret & 0x7FFFFFFFu) && !(a_exp && b_exp && ((a_exp + b_exp) >= 127u))) {
        ret = 0;
    }
    return ret;
}

uint32_t sfpu_add_bh(uint32_t a, uint32_t b) {
    a = proxy_denormals_as_zeros(a);
    b = proxy_denormals_as_zeros(b);
    const uint32_t ret = std::bit_cast<uint32_t>(std::bit_cast<float>(a) + std::bit_cast<float>(b));
    if ((ret & 0x7FFFFFFFu) > 0x7F800000u) {
        return 0x7FC00000u;
    }
    return proxy_denormals_as_zeros(ret);
}

uint32_t sfpu_mad_bh(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t a_exp = (a >> 23) & 255u;
    const uint32_t b_exp = (b >> 23) & 255u;
    const uint32_t c_exp = (c >> 23) & 255u;
    const uint32_t product_sign = (a ^ b) & 0x80000000u;
    const uint32_t c_sign = c & 0x80000000u;
    const int32_t product_exp = int32_t(a_exp + b_exp) - 127;

    if ((a_exp == 255u) || (b_exp == 255u) || (c_exp == 255u) || (product_exp >= 255)) {
        const bool invalid_product = ((a & 0x7FFFFFFFu) > 0x7F800000u) || ((b & 0x7FFFFFFFu) > 0x7F800000u) ||
                                     ((a_exp == 255u) && !b_exp) || ((b_exp == 255u) && !a_exp);
        const bool product_inf = (a_exp == 255u) || (b_exp == 255u);
        if (invalid_product || ((c & 0x7FFFFFFFu) > 0x7F800000u) ||
            ((c_exp == 255u) && product_inf && (c_sign != product_sign))) {
            return 0x7FC00000u;
        }
        return (c_exp == 255u) ? c : (product_sign | 0x7F800000u);
    }

    uint32_t product_man = 0;
    if (!a_exp || !b_exp || (product_exp < 0)) {
        return c_exp ? c : c_sign & product_sign;
    } else {
        const uint64_t product = uint64_t((a & 0x7FFFFFu) | 0x800000u) * ((b & 0x7FFFFFu) | 0x800000u);
        product_man = uint32_t(product >> 20) | ((product & 0xFFFFFu) != 0);
    }

    uint32_t c_man = c_exp ? ((c & 0x7FFFFFu) | 0x800000u) << 3 : 0;
    int32_t aligned_exp = product_exp;
    if (product_exp < int32_t(c_exp)) {
        const uint32_t shift = c_exp - uint32_t(product_exp);
        const uint32_t original = product_man;
        product_man = (shift < 28u) ? (product_man >> shift) : 0;
        if (product_man) {
            product_man |= ((product_man << shift) != original);
        }
        aligned_exp = int32_t(c_exp);
    } else if (product_exp > int32_t(c_exp)) {
        const uint32_t shift = uint32_t(product_exp) - c_exp;
        const uint32_t original = c_man;
        c_man = (shift < 27u) ? (c_man >> shift) : 0;
        if (c_man) {
            c_man |= ((c_man << shift) != original);
        }
    }

    const int32_t sum =
        (product_sign ? -int32_t(product_man) : int32_t(product_man)) + (c_sign ? -int32_t(c_man) : int32_t(c_man));
    if (sum == 0) {
        return product_sign & c_sign;
    }

    const uint64_t result64 = std::bit_cast<uint64_t>(double(sum)) + ((uint64_t(aligned_exp) - 153u) << 52);
    const uint32_t ret = std::bit_cast<uint32_t>(float(std::bit_cast<double>(result64)));
    return (ret & 0x7F800000u) ? ret : (ret & 0x80000000u);
}

uint32_t sfpu_min_bits(uint32_t a, uint32_t b) {
    const int32_t aa = (a & 0x80000000u) ? int32_t(a ^ 0x7FFFFFFFu) : int32_t(a);
    const int32_t bb = (b & 0x80000000u) ? int32_t(b ^ 0x7FFFFFFFu) : int32_t(b);
    return aa <= bb ? a : b;
}

uint32_t sfpu_max_bits(uint32_t a, uint32_t b) {
    const int32_t aa = (a & 0x80000000u) ? int32_t(a ^ 0x7FFFFFFFu) : int32_t(a);
    const int32_t bb = (b & 0x80000000u) ? int32_t(b ^ 0x7FFFFFFFu) : int32_t(b);
    return aa >= bb ? a : b;
}

uint32_t sfpu_arecip_bits(uint32_t x) {
    static constexpr uint8_t lut[128] = {
        127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, 97, 96, 94, 93, 91,
        90,  88,  87,  85,  84,  83,  81,  80,  79,  77,  76,  75,  74,  72,  71,  70,  69, 68, 66, 65, 64, 63,
        62,  61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,  47,  46, 45, 44, 43, 42, 41,
        40,  40,  39,  38,  37,  36,  35,  35,  34,  33,  32,  31,  31,  30,  29,  28,  28, 27, 26, 25, 25, 24,
        23,  23,  22,  21,  21,  20,  19,  19,  18,  17,  17,  16,  15,  15,  14,  14,  13, 12, 12, 11, 11, 10,
        9,   9,   8,   8,   7,   7,   6,   5,   5,   4,   4,   3,   3,   2,   2,   1,   1,  0,
    };

    const uint32_t sign = x & 0x80000000u;
    x &= 0x7FFFFFFFu;
    if (x < 0x800000u) {
        return sign | 0x7F800000u;
    }
    if (x < 0x7E800000u) {
        return sign | ((253u - (x >> 23)) << 23) | (uint32_t(lut[(x >> 16) & 0x7Fu]) << 16);
    }
    return sign;
}

float attention_exp_sfpu(float score, float maximum);

uint16_t device_swiglu_sfpu_bf16(float gate, float up) {
    uint32_t x = uint32_t(fp32_to_bf16(gate)) << 16;
    uint32_t y = uint32_t(fp32_to_bf16(up)) << 16;

    y = sfpu_min_bits(y, 0x40E00000u);
    y = sfpu_max_bits(y, 0xC0E00000u);
    y = sfpu_add_bh(0x3F800000u, y);

    x = sfpu_min_bits(x, 0x40E00000u);
    const uint32_t z = sfpu_mul_bh(x, 0x3FD9DB23u);
    const uint32_t e = std::bit_cast<uint32_t>(attention_exp_sfpu(std::bit_cast<float>(z | 0x80000000u), 0));
    const uint32_t denominator = sfpu_add_bh(0x3F800000u, e);
    uint32_t reciprocal = sfpu_arecip_bits(denominator);
    reciprocal = sfpu_mul_bh(reciprocal, sfpu_mad_bh(denominator ^ 0x80000000u, reciprocal, 0x40000000u));
    uint32_t sigmoid = (x & 0x80000000u) ? sfpu_mul_bh(e, reciprocal) : reciprocal;
    sigmoid = sfpu_mul_bh(x, sigmoid);
    sigmoid = sfpu_mul_bh(sigmoid, y);

    sigmoid = proxy_denormals_as_zeros(sigmoid);
    // SFPSTOCHRND FP32->BF16, nearest ties away; exceptional values are normalized.
    const uint32_t exponent = (sigmoid >> 23) & 255u;
    if (!exponent) {
        return 0;
    }
    if (exponent == 255u) {
        return uint16_t((sigmoid >> 16) & 0xFF80u);
    }
    return uint16_t((sigmoid + 0x8000u) >> 16);
}

float device_swiglu_sfpu_value(float gate, float up) {
    return bf16_to_fp32(device_swiglu_sfpu_bf16(gate, up));
}

TtSimApi load_ttsim(const char *path) {
    TtSimApi api;
    api.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!api.handle) {
        throw std::runtime_error("cannot dlopen libttsim: " + std::string(dlerror()));
    }
    try {
        load_symbol(api, api.init, "libttsim_init");
        load_symbol(api, api.exit, "libttsim_exit");
        load_symbol(api, api.pci_config_rd32, "libttsim_pci_config_rd32");
        load_symbol(api, api.pci_mem_rd_bytes, "libttsim_pci_mem_rd_bytes");
        load_symbol(api, api.pci_mem_wr_bytes, "libttsim_pci_mem_wr_bytes");
        load_symbol(api, api.set_pci_dma_mem_callbacks, "libttsim_set_pci_dma_mem_callbacks");
        load_symbol(api, api.clock, "libttsim_clock");
        api.set_pci_dma_mem_callbacks(ttsim_dma_rd_bytes, ttsim_dma_wr_bytes);
        return api;
    } catch (...) {
        dlclose(api.handle);
        throw;
    }
}

void cleanup_silicon_device(SiliconDevice &device) {
    if (device.resource2_wc.map) {
        munmap(device.resource2_wc.map, device.resource2_wc.size);
        device.resource2_wc.map = nullptr;
    }
    if (device.resource0_uc.map) {
        munmap(device.resource0_uc.map, device.resource0_uc.size);
        device.resource0_uc.map = nullptr;
    }
    close_device_fd(device.fd);
    device.fd = -1;
}

void assign_silicon_mapping(SiliconDevice &device, const tenstorrent_mapping &mapping) {
    if (mapping.mapping_id == TENSTORRENT_MAPPING_RESOURCE0_UC) {
        device.resource0_uc.id = mapping.mapping_id;
        device.resource0_uc.base = mapping.mapping_base;
        device.resource0_uc.size = mapping.mapping_size;
    } else if (mapping.mapping_id == TENSTORRENT_MAPPING_RESOURCE2_WC) {
        device.resource2_wc.id = mapping.mapping_id;
        device.resource2_wc.base = mapping.mapping_base;
        device.resource2_wc.size = mapping.mapping_size;
    }
}

void mmap_silicon_mapping(SiliconDevice &device, SiliconMapping &mapping, const char *name) {
    if (mapping.id == TENSTORRENT_MAPPING_UNUSED) {
        throw std::runtime_error(std::string("missing KMD mapping ") + name);
    }
    void *p = mmap(nullptr, mapping.size, PROT_READ | PROT_WRITE, MAP_SHARED, device.fd, mapping.base);
    if (p == MAP_FAILED) {
        throw std::runtime_error("cannot mmap KMD mapping " + std::string(name) + ": " + std::strerror(errno));
    }
    mapping.map = (uint8_t *)p;
}

void raise_silicon_power(int fd) {
    tenstorrent_power_state power_state{};

    power_state.argsz = sizeof(power_state);
    power_state.validity = TT_POWER_VALIDITY(4, 0);
    power_state.power_flags = TT_POWER_FLAG_MAX_AI_CLK | TT_POWER_FLAG_MRISC_PHY_WAKEUP | TT_POWER_FLAG_TENSIX_ENABLE |
                              TT_POWER_FLAG_L2CPU_ENABLE;
    if (ioctl(fd, TENSTORRENT_IOCTL_SET_POWER_STATE, &power_state) != 0) {
        throw std::runtime_error("SET_POWER_STATE ioctl failed: " + std::string(std::strerror(errno)));
    }
}

SiliconDevice open_silicon_device(uint32_t id) {
    const std::string device_path = "/dev/tenstorrent/" + std::to_string(id);
    SiliconDevice device;
    device.fd = open(device_path.c_str(), O_RDWR | O_CLOEXEC | O_APPEND | O_EXCL | O_NONBLOCK);
    if (device.fd < 0) {
        if (errno == EAGAIN || errno == EBUSY) {
            throw std::runtime_error(device_path + " is busy; stop the other device user and retry");
        }
        throw std::runtime_error("cannot open " + device_path + ": " + std::strerror(errno));
    }
    try {
        // Older KMDs ignore O_EXCL. A second, ordinary nonblocking open must be rejected.
        const int probe = open(device_path.c_str(), O_RDWR | O_CLOEXEC | O_APPEND | O_NONBLOCK);
        if (probe >= 0) {
            close(probe);
            throw std::runtime_error("KMD does not enforce exclusive device ownership; update the driver");
        }
        if (errno != EAGAIN && errno != EBUSY) {
            throw std::runtime_error("cannot verify exclusive ownership of " + device_path + ": " +
                                     std::strerror(errno));
        }
        tenstorrent_get_driver_info driver_info{};
        driver_info.in.output_size_bytes = sizeof(driver_info.out);
        if (ioctl(device.fd, TENSTORRENT_IOCTL_GET_DRIVER_INFO, &driver_info) != 0) {
            const int saved_errno = errno;
            throw std::runtime_error("GET_DRIVER_INFO ioctl failed: " + std::string(std::strerror(saved_errno)));
        }
        tenstorrent_get_device_info device_info{};
        device_info.in.output_size_bytes = sizeof(device_info.out);
        if (ioctl(device.fd, TENSTORRENT_IOCTL_GET_DEVICE_INFO, &device_info) != 0) {
            const int saved_errno = errno;
            throw std::runtime_error("GET_DEVICE_INFO ioctl failed: " + std::string(std::strerror(saved_errno)));
        }
        if (device_info.out.vendor_id != 0x1e52) {
            throw std::runtime_error("unsupported PCI vendor id");
        }
        if (device_info.out.device_id != 0xb140) {
            throw std::runtime_error("unsupported PCI device id; only Blackhole is supported by --device");
        }

        raise_silicon_power(device.fd);

        constexpr uint32_t mapping_count = 8;
        alignas(tenstorrent_mapping) unsigned char
            mapping_storage[sizeof(tenstorrent_query_mappings_in) + mapping_count * sizeof(tenstorrent_mapping)] = {};
        tenstorrent_query_mappings *query = (tenstorrent_query_mappings *)mapping_storage;
        tenstorrent_mapping *mappings =
            (tenstorrent_mapping *)(mapping_storage + sizeof(tenstorrent_query_mappings_in));
        query->in.output_mapping_count = mapping_count;
        if (ioctl(device.fd, TENSTORRENT_IOCTL_QUERY_MAPPINGS, query) != 0) {
            const int saved_errno = errno;
            throw std::runtime_error("QUERY_MAPPINGS ioctl failed: " + std::string(std::strerror(saved_errno)));
        }

        for (uint32_t i = 0; i < mapping_count; ++i) {
            const tenstorrent_mapping &mapping = mappings[i];
            if (mapping.mapping_id == TENSTORRENT_MAPPING_UNUSED) {
                continue;
            }
            assign_silicon_mapping(device, mapping);
        }

        mmap_silicon_mapping(device, device.resource0_uc, "resource0_uc");
        mmap_silicon_mapping(device, device.resource2_wc, "resource2_wc");
        return device;
    } catch (...) {
        cleanup_silicon_device(device);
        throw;
    }
}

void arm_shard0_reset_on_close() {
    tenstorrent_set_noc_cleanup cleanup = {};

    SiliconDevice *device = silicon_device();
    if (!device) {
        return;
    }
    cleanup.argsz = sizeof(cleanup);
    cleanup.enabled = 1;
    cleanup.addr = soft_reset_addr;
    cleanup.data = all_riscv_reset_bits;
    for (size_t chip = 0; chip < current_runtime->chip_count; ++chip) {
        const uint32_t coord = worker_coord(chip, 0);
        cleanup.x = uint8_t(coord & 63u);
        cleanup.y = uint8_t(coord >> 6);
        if (ioctl(current_runtime->devices[chip].fd, TENSTORRENT_IOCTL_SET_NOC_CLEANUP, &cleanup) < 0) {
            std::fprintf(stderr, "warning: could not arm shard 0 reset on close for chip %zu: %s\n", chip,
                         std::strerror(errno));
        }
    }
}

TtSimApi make_silicon_api() {
    TtSimApi api;
    api.pci_mem_rd_bytes = silicon_pci_mem_rd_bytes;
    api.pci_mem_wr_bytes = silicon_pci_mem_wr_bytes;
    api.clock = silicon_clock;
    return api;
}

void stop_brisc_firmware(TtSimApi &api, TtMeshConfig mesh);

void close_tt_firmware_runtime(TtFirmwareRuntime &runtime) {
    if (runtime.backend == TtFirmwareBackend::sim) {
        if (runtime.sim_initialized) {
            runtime.api.exit();
            runtime.sim_initialized = false;
        }
        if (runtime.api.handle) {
            dlclose(runtime.api.handle);
            runtime.api.handle = nullptr;
        }
    } else {
        for (SiliconDevice &device : runtime.devices) {
            cleanup_silicon_device(device);
        }
    }
    runtime.chip_count = 0;
    runtime.worker_columns.fill(1u);
    runtime.peer_bars = {};
    runtime.sim_embedding = nullptr;
    runtime.sim_embedding_bytes = 0;
    runtime.sim_logits = nullptr;
    if (current_runtime == &runtime) {
        current_runtime = nullptr;
    }
}

std::vector<uint32_t> silicon_device_ids() {
    DIR *directory = opendir("/dev/tenstorrent");
    if (!directory) {
        throw std::runtime_error("cannot enumerate /dev/tenstorrent: " + std::string(std::strerror(errno)));
    }
    std::vector<uint32_t> ids;
    try {
        for (;;) {
            errno = 0;
            const dirent *entry = readdir(directory);
            if (!entry) {
                if (errno) {
                    throw std::runtime_error("cannot read /dev/tenstorrent: " + std::string(std::strerror(errno)));
                }
                break;
            }
            const char *end = entry->d_name + std::strlen(entry->d_name);
            uint32_t id;
            const auto result = std::from_chars(entry->d_name, end, id);
            if (result.ec == std::errc{} && result.ptr == end) {
                ids.push_back(id);
            }
        }
    } catch (...) {
        closedir(directory);
        throw;
    }
    closedir(directory);
    std::sort(ids.begin(), ids.end());
    return ids;
}

void discover_sim_chips(TtFirmwareRuntime &runtime, size_t required) {
    const auto config = runtime.api.pci_config_rd32;
    for (uint32_t bdf = 0; bdf < 0x10000u && runtime.chip_count < required; bdf += 8) {
        if (config(bdf, 0) != 0xb1401e52u) {
            continue;
        }
        const auto bar = [&](uint32_t offset) {
            return (uint64_t(config(bdf, offset + 4)) << 32) | (config(bdf, offset) & ~0xFu);
        };
        runtime.bars[runtime.chip_count++] = {bar(0x10), bar(0x20)};
    }
    if (runtime.chip_count != required) {
        throw std::runtime_error("simulator exposes " + std::to_string(runtime.chip_count) + " Blackhole chips, need " +
                                 std::to_string(required));
    }
}

void require_direct_dma_range(uint64_t address, uint64_t bytes) {
    // BH's low PCIe NoC window passes the DMA address directly to the host IOMMU.
    constexpr uint64_t limit = 1ull << 58;
    if (!bytes || address >= limit || bytes > limit - address) {
        throw std::runtime_error("DMA mapping exceeds the Blackhole direct PCIe window");
    }
}

void map_peer_bars(TtFirmwareRuntime &runtime) {
    for (size_t src = 0; src < runtime.chip_count; ++src) {
        for (size_t dst = 0; dst < runtime.chip_count; ++dst) {
            if (src == dst) {
                continue;
            }
            uint64_t address = runtime.bars[dst].bar0;
            if (runtime.backend == TtFirmwareBackend::silicon) {
                tenstorrent_map_peer_bar peer{};
                peer.in.peer_fd = uint32_t(runtime.devices[dst].fd);
                peer.in.peer_bar_index = 0;
                peer.in.peer_bar_length = uint32_t(peer_bar_bytes);
                if (ioctl(runtime.devices[src].fd, TENSTORRENT_IOCTL_MAP_PEER_BAR, &peer) < 0) {
                    throw std::runtime_error("MAP_PEER_BAR " + std::to_string(src) + " -> " + std::to_string(dst) +
                                             " failed: " + std::strerror(errno));
                }
                // KMD owns this mapping until the source device fd is closed.
                address = peer.out.dma_address;
            }
            require_direct_dma_range(address, peer_bar_bytes);
            runtime.peer_bars[src][dst] = address;
        }
    }
}

void open_tt_firmware_runtime(TtFirmwareRuntime &runtime, TtFirmwareBackend backend, const char *sim_path,
                              TtMeshConfig mesh) {
    runtime.backend = backend;
    try {
        if (backend == TtFirmwareBackend::sim) {
            if (!sim_path) {
                throw std::runtime_error("--sim requires a libttsim path");
            }
            runtime.api = load_ttsim(sim_path);
            runtime.api.init();
            runtime.sim_initialized = true;
            discover_sim_chips(runtime, mesh.chip_count());
        } else {
            const auto ids = silicon_device_ids();
            if (ids.size() < mesh.chip_count()) {
                throw std::runtime_error("found " + std::to_string(ids.size()) + " TT devices, need " +
                                         std::to_string(mesh.chip_count()));
            }
            for (size_t chip = 0; chip < mesh.chip_count(); ++chip) {
                runtime.devices[chip] = open_silicon_device(ids[chip]);
                // Synthetic host addresses, matching libttsim's per-device 64-GiB stride.
                const uint64_t base = uint64_t(chip) << 36;
                runtime.bars[chip] = {base + bar0_base, base + bar4_base};
                ++runtime.chip_count;
            }
            runtime.api = make_silicon_api();
        }
        current_runtime = &runtime;
        configure_tile_windows(runtime.api, mesh);
        configure_worker_placement(runtime.api, mesh);
        configure_dram_windows(runtime.api, mesh);
        map_peer_bars(runtime);
    } catch (...) {
        close_tt_firmware_runtime(runtime);
        throw;
    }
}

void abort_tt_firmware_runtime(TtFirmwareRuntime &runtime, TtMeshConfig mesh) {
    stop_brisc_firmware(runtime.api, mesh);
    close_tt_firmware_runtime(runtime);
}

uint16_t pack_dst_bf16_proxy(uint32_t dst_internal) {
    uint32_t value = dst_decode_fp32_proxy(dst_internal);
    if ((value & 0x7FFFFFFFu) > 0x7F800000u) {
        value = (value & 0x80000000u) | 0x7F800000u;
    }
    value = (value + 0x8000u) >> 16;
    if ((value & 0x7FFFu) < 0x80u) {
        value = 0;
    }
    return uint16_t(value);
}

uint32_t pack_l1_acc_fp32_proxy(uint32_t l1_value, uint32_t packed_value) {
    l1_value = proxy_denormals_as_zeros(l1_value);
    packed_value = proxy_denormals_as_zeros(packed_value);

    uint32_t result = std::bit_cast<uint32_t>(std::bit_cast<float>(l1_value) + std::bit_cast<float>(packed_value));
    if ((result & 0x7FFFFFFFu) > 0x7F800000u) {
        return 0x7FC00000u;
    }
    return proxy_denormals_as_zeros(result);
}

uint16_t pack_l1_acc_bf16_proxy(uint16_t l1_value, uint16_t packed_value) {
    if (((l1_value == 0x7F80u) && (packed_value == 0xFF80u)) || ((l1_value == 0xFF80u) && (packed_value == 0x7F80u))) {
        return 0x7FC0u;
    }
    if ((l1_value & 0x7FFFu) < 0x80u) {
        l1_value = uint16_t(l1_value & 0x8000u);
    }
    if ((packed_value & 0x7FFFu) < 0x80u) {
        packed_value = uint16_t(packed_value & 0x8000u);
    }

    double l1_float = double(bf16_to_fp32(l1_value));
    if ((l1_value & 0x7FFFu) > 0x7F80u) {
        l1_float = std::bit_cast<double>((uint64_t(l1_value & 0x8000u) << 48) | ((0x7FFull << 52) - 1));
    }

    double packed_float = double(bf16_to_fp32(packed_value));
    if ((packed_value & 0x7FFFu) > 0x7F80u) {
        packed_float =
            std::bit_cast<double>((uint64_t(packed_value & 0x8000u) << 48) | (uint64_t(255 - 127 + 1023) << 52) |
                                  (uint64_t(packed_value & 0x7Fu) << (52 - 7)));
    }

    const uint64_t result64 = std::bit_cast<uint64_t>(l1_float + packed_float);
    const uint16_t sign = uint16_t(result64 >> 48) & 0x8000u;
    const int32_t exponent = int32_t((result64 >> 52) & 0x7FFu) - 1023 + 127;
    const uint64_t mantissa = result64 & ((1ull << 52) - 1);
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 255) {
        return uint16_t(sign | 0x7F80u);
    }
    return uint16_t(sign | uint16_t((uint32_t(exponent) << 7) +
                                    ((mantissa + (1ull << (51 - 7)) - 1 + ((mantissa >> (52 - 7)) & 1)) >> (52 - 7))));
}

uint32_t src_from_bf16_proxy(uint16_t value) {
    return uint32_t(value) << 3;
}

void validate_firmware_model(const GgufFile &file, TtMeshConfig mesh) {
    validate_mesh_config(mesh);
    const uint64_t layers = file.integer("gpt-oss.block_count");
    const uint64_t experts = file.integer("gpt-oss.expert_count");
    const bool small = layers == TT_GPT_OSS_20B_LAYERS && experts == TT_GPT_OSS_20B_EXPERTS;
    const bool large = layers == TT_GPT_OSS_120B_LAYERS && experts == TT_GPT_OSS_120B_EXPERTS;
    if (file.string("general.architecture") != "gpt-oss" ||
        file.integer("gpt-oss.embedding_length") != TT_GPT_OSS_DIM || (!small && !large)) {
        throw std::runtime_error("TT firmware requires gpt-oss-20b or gpt-oss-120b");
    }
    if (large && mesh.chip_count() == 1) {
        throw std::runtime_error("120b requires --tiles 32");
    }
}

double seconds_since(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

void print_generation_stats(size_t prompt_count, double prefill_seconds, size_t decoded_tokens,
                            size_t decode_evaluations, double decode_seconds) {
    std::fprintf(stderr, "prefill: %zu tokens, %.3f s, %.3f t/s\n", prompt_count, prefill_seconds,
                 double(prompt_count) / prefill_seconds);
    std::fprintf(stderr, "generated: %zu tokens\n", decoded_tokens);
    if (decode_evaluations > 0) {
        std::fprintf(stderr, "decode: %zu evals, %.3f s, %.3f t/s\n", decode_evaluations, decode_seconds,
                     double(decode_evaluations) / decode_seconds);
    } else {
        std::fprintf(stderr, "decode: 0 evals, 0.000 s, n/a t/s\n");
    }
}

const GgufTensor &require_sim_tensor(const GgufFile &file, const char *name, uint32_t type,
                                     std::initializer_list<uint64_t> shape) {
    const GgufTensor *tensor = file.tensor(name);
    if (!tensor) {
        throw std::runtime_error("missing sim tensor " + std::string(name));
    }
    if (tensor->type != type) {
        throw std::runtime_error("wrong sim tensor type " + std::string(name));
    }
    if (tensor->shape.size() != shape.size()) {
        throw std::runtime_error("wrong sim tensor rank " + std::string(name));
    }
    size_t i = 0;
    for (uint64_t expected : shape) {
        if (tensor->shape[i++] != expected) {
            throw std::runtime_error("wrong sim tensor shape " + std::string(name));
        }
    }
    return *tensor;
}

const TtqTensor &require_ttq_tensor(const TtqFile &file, const char *name, uint32_t format,
                                    std::initializer_list<uint64_t> shape) {
    const TtqTensor &tensor = file.tensor(name);
    if (tensor.format != format) {
        throw std::runtime_error("wrong TTQ tensor format " + std::string(name));
    }
    if (tensor.shape.size() != shape.size()) {
        throw std::runtime_error("wrong TTQ tensor rank " + std::string(name));
    }
    size_t i = 0;
    for (uint64_t expected : shape) {
        if (tensor.shape[i++] != expected) {
            throw std::runtime_error("wrong TTQ tensor shape " + std::string(name));
        }
    }
    return tensor;
}

float bf16_linear_value(const TtqTensor &tensor, size_t index) {
    const uint16_t *values = (const uint16_t *)tensor.data;
    return bf16_to_fp32(values[index]);
}

uint16_t bf16_linear_bits(const TtqTensor &tensor, size_t index) {
    const uint16_t *values = (const uint16_t *)tensor.data;
    return values[index];
}

uint32_t rv32f_fma_bits(uint32_t a, uint32_t b, uint32_t c) {
    return sfpu_mad_bh(a, b, c);
}

float rv32f_fma_value(float a, float b, float c) {
    const uint32_t result =
        rv32f_fma_bits(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b), std::bit_cast<uint32_t>(c));
    return std::bit_cast<float>(result);
}

float sfpu_add_value(float a, float b) {
    return std::bit_cast<float>(sfpu_add_bh(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b)));
}

float sfpu_mul_value(float a, float b) {
    return std::bit_cast<float>(sfpu_mul_bh(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b)));
}

float sfpu_mad_value(float a, float b, float c) {
    return std::bit_cast<float>(
        sfpu_mad_bh(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b), std::bit_cast<uint32_t>(c)));
}

float pack_sfpu_bf16_value(float value) {
    return bf16_to_fp32(pack_dst_bf16_proxy(dst_encode_fp32_proxy(std::bit_cast<uint32_t>(value))));
}

float rv32f_add_value(float a, float b) {
    return rv32f_fma_value(a, 1.0f, b);
}

float rv32f_max_value(float a, float b) {
    // Match ttsim's fp32_min_max: one NaN loses, two produce canonical NaN, +0 beats -0.
    const uint32_t ab = std::bit_cast<uint32_t>(a);
    const uint32_t bb = std::bit_cast<uint32_t>(b);
    if ((ab & 0x7fffffffu) > 0x7f800000u) {
        return (bb & 0x7fffffffu) > 0x7f800000u ? std::bit_cast<float>(0x7fc00000u) : b;
    }
    if ((bb & 0x7fffffffu) > 0x7f800000u) {
        return a;
    }
    if (a == b) {
        return std::bit_cast<float>(ab & bb);
    }
    return a > b ? a : b;
}

float rv32f_sub_value(float a, float b) {
    return rv32f_fma_value(a, 1.0f, std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ 0x80000000u));
}

float rv32f_mul_value(float a, float b) {
    const uint32_t zero = (std::bit_cast<uint32_t>(a) ^ std::bit_cast<uint32_t>(b)) & 0x80000000u;
    const uint32_t result = rv32f_fma_bits(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b), zero);
    return std::bit_cast<float>(result);
}

float rv32f_fnmsub_value(float a, float b, float c) {
    const uint32_t neg_a = std::bit_cast<uint32_t>(a) ^ 0x80000000u;
    const uint32_t result = rv32f_fma_bits(neg_a, std::bit_cast<uint32_t>(b), std::bit_cast<uint32_t>(c));
    return std::bit_cast<float>(result);
}

float rv32f_rsqrt_approx_value(float x) {
    float y = std::bit_cast<float>(0x5F3759DFu - (std::bit_cast<uint32_t>(x) >> 1u));

    for (size_t i = 0; i < 3; ++i) {
        const float half_x = rv32f_mul_value(-0.5f, x);
        const float yy = rv32f_mul_value(y, y);
        const float correction = rv32f_fma_value(half_x, yy, 1.5f);

        y = rv32f_mul_value(y, correction);
    }
    return y;
}

float proxy_add_residual_value(float moe_value, float residual_value) {
    return std::bit_cast<float>(
        pack_l1_acc_fp32_proxy(std::bit_cast<uint32_t>(moe_value), std::bit_cast<uint32_t>(residual_value)));
}

float proxy_make_residual_value(float x, float projected) {
    return std::bit_cast<float>(pack_l1_acc_fp32_proxy(std::bit_cast<uint32_t>(x), std::bit_cast<uint32_t>(projected)));
}

void load_token_embedding_bf16(const TtqTensor &tensor, size_t row, float *output) {
    const uint16_t *values = (const uint16_t *)tensor.data + row * TT_GPT_OSS_DIM;
    for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
        output[i] = bf16_to_fp32(values[i]);
    }
}

float proxy_rms_partial(const float *input, size_t first_index, size_t count) {
    std::array<float, 32> first = {};
    std::array<float, 32> second = {};
    std::array<float, 32> lanes = {};

    for (size_t base = first_index; base < first_index + count; base += TT_RMS_PARTIALS) {
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = 0; column < 8; ++column) {
                const size_t lane = row * 8 + column;
                const size_t i = base + row * 16 + column * 2;
                const float low = i < first_index + count ? input[i] : 0.0f;
                const float high = i + 1 < first_index + count ? input[i + 1] : 0.0f;

                first[lane] = sfpu_mad_value(low, low, first[lane]);
                second[lane] = sfpu_mad_value(high, high, second[lane]);
            }
        }
    }
    for (size_t lane = 0; lane < 32; ++lane) {
        lanes[lane] = sfpu_add_value(second[lane], first[lane]);
    }
    for (size_t column = 0; column < 8; ++column) {
        float total = lanes[column];

        for (size_t row = 1; row < 4; ++row) {
            total = sfpu_add_value(lanes[row * 8 + column], total);
        }
        for (size_t row = 0; row < 4; ++row) {
            lanes[row * 8 + column] = total;
        }
    }
    for (size_t shift : {size_t(1), size_t(2), size_t(4)}) {
        std::array<float, 32> rotated = lanes;

        for (size_t lane = 0; lane < 32; ++lane) {
            const size_t base = lane & ~size_t(7);

            rotated[lane] = lanes[base + ((lane - shift) & 7)];
        }
        for (size_t lane = 0; lane < 32; ++lane) {
            lanes[lane] = sfpu_add_value(rotated[lane], lanes[lane]);
        }
    }
    return lanes[0];
}

void proxy_rms_norm_bf16(const float *input, const TtqTensor &weights, float *output, size_t shards = 1) {
    float sum = 0.0f;
    for (size_t shard = 0; shard < shards; ++shard) {
        const size_t first = tt_residual_block_bound(uint32_t(shard), uint32_t(shards)) * TT_MVMUL_TILE_DIM;
        const size_t end = tt_residual_block_bound(uint32_t(shard + 1), uint32_t(shards)) * TT_MVMUL_TILE_DIM;
        sum = rv32f_add_value(sum, proxy_rms_partial(input, first, end - first));
    }
    const float variance = rv32f_fma_value(sum, 1.0f / float(TT_GPT_OSS_DIM), 1.0e-5f);
    const float scale = rv32f_rsqrt_approx_value(variance);

    for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
        const float value = sfpu_mul_value(input[i], scale);
        const float scaled = sfpu_mul_value(value, bf16_linear_value(weights, i));
        const uint32_t encoded = dst_encode_fp32_proxy(std::bit_cast<uint32_t>(scaled));

        output[i] = bf16_to_fp32(pack_dst_bf16_proxy(encoded));
    }
}

void reference_rms_norm_bf16(const float *input, const TtqTensor &weights, float *output) {
    float sum = 0.0f;

    for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
        sum = std::fma(input[i], input[i], sum);
    }
    const float scale = 1.0f / std::sqrt(sum / float(TT_GPT_OSS_DIM) + 1.0e-5f);
    for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
        output[i] = input[i] * scale * bf16_linear_value(weights, i);
    }
}

void start_brisc_firmware(TtSimApi &api, TtMeshConfig mesh) {
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        write_u32(api, tile_l1_addr(shard, TT_L1_SENTINEL_ADDR), 0);
        write_u32(api, tile_l1_addr(shard, TT_L1_GO_ADDR), 0);
        write_u32(api, tile_reg_addr(shard, soft_reset_addr), all_riscv_reset_bits);
        const uint32_t counters[TT_MAX_SHARDS] = {};
        api.pci_mem_wr_bytes(tile_l1_addr(shard, TT_L1_ALLGATHER_ARRIVE_ADDR), counters, sizeof(counters));
        api.pci_mem_wr_bytes(tile_l1_addr(shard, TT_L1_ALLGATHER_TARGET_ADDR), counters, sizeof(counters));
        api.pci_mem_wr_bytes(tile_l1_addr(shard, 0), brisc_program, sizeof(brisc_program));
    }
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        write_u32(api, tile_reg_addr(shard, soft_reset_addr), all_riscv_reset_bits & ~brisc_soft_reset_bit);
    }
}

void stop_brisc_firmware(TtSimApi &api, TtMeshConfig mesh) {
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        write_u32(api, tile_reg_addr(shard, soft_reset_addr), all_riscv_reset_bits);
    }
}

void issue_brisc_command(TtSimApi &api, TtMeshConfig mesh, size_t max_iterations, double max_wall_seconds = 0.0) {
    const auto begin = std::chrono::steady_clock::now();

    write_u32(api, tile_l1_addr(0, TT_L1_SENTINEL_ADDR), 0);
    for (size_t shard = mesh.shard_count; shard-- > 0;) {
        write_u32(api, tile_l1_addr(shard, TT_L1_GO_ADDR), 1);
    }

    for (size_t i = 0; i < max_iterations; ++i) {
        api.clock(100);
        const uint32_t result = read_u32(api, tile_l1_addr(0, TT_L1_SENTINEL_ADDR));
        if (result == TT_SENTINEL_TOKEN) {
            return;
        }
        if ((result & 0xFFF00000u) == 0xBAD00000u) {
            break;
        }
        if (max_wall_seconds > 0.0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - begin).count();

            if (elapsed > max_wall_seconds) {
                break;
            }
        }
    }

    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        const uint32_t result = read_u32(api, tile_l1_addr(shard, TT_L1_SENTINEL_ADDR));
        std::fprintf(stderr, "BRISC shard %zu: sentinel 0x%08x, waypoint 0x%08x, GO %u\n", shard, result,
                     read_u32(api, tile_l1_addr(shard, TT_L1_WAYPOINT_ADDR)),
                     read_u32(api, tile_l1_addr(shard, TT_L1_GO_ADDR)));
        for (size_t peer = 0; peer < mesh.local_shard_count(); ++peer) {
            const uint32_t offset = uint32_t(peer * sizeof(uint32_t));

            std::fprintf(stderr, "  sender %zu: arrivals %u, target %u\n", peer,
                         read_u32(api, tile_l1_addr(shard, TT_L1_ALLGATHER_ARRIVE_ADDR + offset)),
                         read_u32(api, tile_l1_addr(shard, TT_L1_ALLGATHER_TARGET_ADDR + offset)));
        }
    }
    throw std::runtime_error("BRISC command did not finish");
}

void make_proxy_b_values(const float *input, size_t k_tiles, uint32_t *b_values) {
    for (size_t k = 0; k < k_tiles; ++k) {
        for (size_t i = 0; i < TT_MVMUL_TILE_DIM; ++i) {
            b_values[k * TT_MVMUL_TILE_DIM + i] = src_from_bf16_proxy(fp32_to_bf16(input[k * TT_MVMUL_TILE_DIM + i]));
        }
    }
}

void proxy_tensix_matvec_block_bf16_dst(const uint8_t *tiles, const uint32_t *b_value_rows, size_t k_tiles,
                                        uint32_t *dst_internal) {
    uint32_t chains[2][TT_MVMUL_TILE_DIM] = {};

    if (k_tiles % 4) {
        throw std::runtime_error("incomplete BF16 unpack group");
    }
    tt_matvec_bf16(tiles, b_value_rows, k_tiles, chains);
    for (size_t row = 0; row < TT_MVMUL_TILE_DIM; ++row) {
        dst_internal[row] = dst_encode_fp32_proxy(
            sfpu_add_bh(dst_decode_fp32_proxy(chains[0][row]), dst_decode_fp32_proxy(chains[1][row])));
    }
}

void proxy_tensix_matvec_block_bf16(const uint8_t *tiles, const uint32_t *b_value_rows, size_t k_tiles,
                                    uint16_t *output) {
    uint32_t dst_internal[TT_MVMUL_TILE_DIM] = {};

    proxy_tensix_matvec_block_bf16_dst(tiles, b_value_rows, k_tiles, dst_internal);
    for (size_t row = 0; row < TT_MVMUL_TILE_DIM; ++row) {
        output[row] = pack_dst_bf16_proxy(dst_internal[row]);
    }
}

void proxy_tensix_matvec_block_bf16_f32(const uint8_t *tiles, const uint32_t *b_value_rows, size_t k_tiles,
                                        uint32_t *output) {
    uint32_t dst_internal[TT_MVMUL_TILE_DIM] = {};

    proxy_tensix_matvec_block_bf16_dst(tiles, b_value_rows, k_tiles, dst_internal);
    for (size_t row = 0; row < TT_MVMUL_TILE_DIM; ++row) {
        output[row] = dst_decode_fp32_proxy(dst_internal[row]);
    }
}

void proxy_tensix_matvec_block_bfp8_strip(const uint8_t *strip, const uint32_t *b_value_rows, uint16_t *output) {
    uint32_t chains[2][TT_MVMUL_TILE_DIM] = {};
    tt_matvec_bfp8(strip, b_value_rows, chains);

    for (size_t row = 0; row < TT_MVMUL_TILE_DIM; ++row) {
        const uint32_t dst_internal = dst_encode_fp32_proxy(
            sfpu_add_bh(dst_decode_fp32_proxy(chains[0][row]), dst_decode_fp32_proxy(chains[1][row])));
        output[row] = pack_dst_bf16_proxy(dst_internal);
    }
}

void proxy_bf16_matvec_f32(const TtqTensor &weight, const GgufTensor *bias, const float *input, uint32_t *b_values,
                           float *output) {
    const size_t columns = size_t(weight.shape[0]);
    const size_t rows = size_t(weight.shape[1]);
    if (columns > proxy_b_values_count) {
        throw std::runtime_error("device proxy BF16 matvec source width exceeds scratch size");
    }
    const size_t k_tiles = columns / TT_MVMUL_TILE_DIM;
    const size_t block_bytes = k_tiles * TT_MVMUL_BF16_A_TILE_BYTES;
    uint32_t block_output[TT_MVMUL_TILE_DIM];
    const float *bias_values = bias ? (const float *)bias->data : nullptr;

    make_proxy_b_values(input, k_tiles, b_values);
    for (size_t row_base = 0; row_base < rows; row_base += TT_MVMUL_TILE_DIM) {
        const uint8_t *block = (const uint8_t *)weight.data + (row_base / TT_MVMUL_TILE_DIM) * block_bytes;

        proxy_tensix_matvec_block_bf16_f32(block, b_values, k_tiles, block_output);
        for (size_t i = 0; i < TT_MVMUL_TILE_DIM; ++i) {
            uint32_t value = block_output[i];
            if (bias) {
                const uint32_t bias_value = std::bit_cast<uint32_t>(bias_values[row_base + i]);
                value = pack_l1_acc_fp32_proxy(bias_value, value);
            }
            output[row_base + i] = std::bit_cast<float>(value);
        }
    }
}

void proxy_bf16_matvec(const TtqTensor &weight, const TtqTensor *bias, const float *input, uint32_t *b_values,
                       float *output) {
    const size_t columns = size_t(weight.shape[0]);
    const size_t rows = size_t(weight.shape[1]);
    if (columns > proxy_b_values_count) {
        throw std::runtime_error("device proxy BF16 matvec source width exceeds scratch size");
    }
    const size_t k_tiles = columns / TT_MVMUL_TILE_DIM;
    const size_t block_bytes = k_tiles * TT_MVMUL_BF16_A_TILE_BYTES;
    uint16_t block_output[TT_MVMUL_TILE_DIM];

    make_proxy_b_values(input, k_tiles, b_values);
    for (size_t row_base = 0; row_base < rows; row_base += TT_MVMUL_TILE_DIM) {
        const uint8_t *block = (const uint8_t *)weight.data + (row_base / TT_MVMUL_TILE_DIM) * block_bytes;

        proxy_tensix_matvec_block_bf16(block, b_values, k_tiles, block_output);
        for (size_t i = 0; i < TT_MVMUL_TILE_DIM; ++i) {
            uint16_t value = block_output[i];
            if (bias) {
                value = pack_l1_acc_bf16_proxy(bf16_linear_bits(*bias, row_base + i), value);
            }
            output[row_base + i] = bf16_to_fp32(value);
        }
    }
}

void proxy_bfp8_matvec_2880(const TtqTensor &weight, const TtqTensor &bias, const float *input, size_t expert,
                            uint32_t *b_values, float *output) {
    constexpr size_t blocks_per_expert = TT_GPT_OSS_DIM / TT_MVMUL_TILE_DIM;
    constexpr size_t block_bytes = TT_BFP8_STRIP_BYTES;
    uint16_t block_output[TT_MVMUL_TILE_DIM];

    make_proxy_b_values(input, TT_MVMUL_K_TILES, b_values);
    for (size_t row_base = 0; row_base < TT_GPT_OSS_DIM; row_base += TT_MVMUL_TILE_DIM) {
        const size_t block_index = expert * blocks_per_expert + row_base / TT_MVMUL_TILE_DIM;
        const uint8_t *block = (const uint8_t *)weight.data + block_index * block_bytes;

        proxy_tensix_matvec_block_bfp8_strip(block, b_values, block_output);
        for (size_t i = 0; i < TT_MVMUL_TILE_DIM; ++i) {
            const uint16_t bias_value = bf16_linear_bits(bias, expert * TT_GPT_OSS_DIM + row_base + i);

            output[row_base + i] = bf16_to_fp32(pack_l1_acc_bf16_proxy(bias_value, block_output[i]));
        }
    }
}

void proxy_bfp8_matvec_bf16_2880(const TtqTensor &weight, const TtqTensor &bias, const float *input, size_t expert,
                                 uint32_t *b_values, uint16_t *output) {
    constexpr size_t blocks_per_expert = TT_GPT_OSS_DIM / TT_MVMUL_TILE_DIM;
    constexpr size_t block_bytes = TT_BFP8_STRIP_BYTES;
    uint16_t block_output[TT_MVMUL_TILE_DIM];

    make_proxy_b_values(input, TT_MVMUL_K_TILES, b_values);
    for (size_t row_base = 0; row_base < TT_GPT_OSS_DIM; row_base += TT_MVMUL_TILE_DIM) {
        const size_t block_index = expert * blocks_per_expert + row_base / TT_MVMUL_TILE_DIM;
        const uint8_t *block = (const uint8_t *)weight.data + block_index * block_bytes;

        proxy_tensix_matvec_block_bfp8_strip(block, b_values, block_output);
        for (size_t i = 0; i < TT_MVMUL_TILE_DIM; ++i) {
            const uint16_t bias_value = bf16_linear_bits(bias, expert * TT_GPT_OSS_DIM + row_base + i);

            output[row_base + i] = pack_l1_acc_bf16_proxy(bias_value, block_output[i]);
        }
    }
}

void proxy_moe_reduce_down4(const uint16_t *down4, const std::array<float, 4> &scales, float *output) {
    for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
        uint32_t accum = 0;
        for (size_t selected = 0; selected < 4; ++selected) {
            const float value = bf16_to_fp32(down4[selected * TT_GPT_OSS_DIM + i]);
            const uint32_t scale_bits = std::bit_cast<uint32_t>(scales[selected]);
            const uint32_t value_bits = std::bit_cast<uint32_t>(value);
            accum = selected ? sfpu_mad_bh(value_bits, scale_bits, accum) : sfpu_mul_bh(value_bits, scale_bits);
        }
        output[i] = std::bit_cast<float>(accum);
    }
}

void proxy_swiglu(const float *gate, const float *up, float *output) {
    for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
        output[i] = device_swiglu_sfpu_value(gate[i], up[i]);
    }
}

void write_tensor_ref(TtSimApi &api, size_t shard, uint32_t coord_addr, uint32_t src_addr, const TtqTensor &tensor) {
    if (!tensor.dram_loaded) {
        throw std::runtime_error("TTQ tensor is not loaded in simulator DRAM: " + tensor.name);
    }
    if (tensor.dram_striped) {
        write_u32(api, tile_l1_addr(shard, coord_addr), dram_coord(tensor.stripe_channel[shard], shard));
        write_u32(api, tile_l1_addr(shard, src_addr), uint32_t(tensor.stripe_offset[shard]));
    } else {
        const DramRegion &region = tensor.chip_regions[shard / TtMeshConfig::tiles_per_chip];
        write_u32(api, tile_l1_addr(shard, coord_addr), dram_coord(region.channel, shard));
        write_u32(api, tile_l1_addr(shard, src_addr), uint32_t(region.offset));
    }
}

void write_dram_ref(TtSimApi &api, size_t shard, uint32_t coord_addr, uint32_t src_addr, const DramRegion &region) {
    write_u32(api, tile_l1_addr(shard, coord_addr), dram_coord(region.channel, shard));
    write_u32(api, tile_l1_addr(shard, src_addr), uint32_t(region.offset));
}

void put_dram_ref(uint32_t *desc, size_t ref_index, uint32_t channel, uint64_t offset, size_t shard) {
    if (offset > UINT32_MAX) {
        throw std::runtime_error("simulator DRAM reference offset does not fit in firmware mailbox");
    }
    desc[ref_index * 2] = dram_coord(channel, shard);
    desc[ref_index * 2 + 1] = uint32_t(offset);
}

void put_tensor_ref(uint32_t *desc, size_t ref_index, const TtqTensor &tensor, size_t shard) {
    if (!tensor.dram_loaded) {
        throw std::runtime_error("TTQ tensor is not loaded in simulator DRAM: " + tensor.name);
    }
    if (tensor.dram_striped) {
        put_dram_ref(desc, ref_index, tensor.stripe_channel[shard], tensor.stripe_offset[shard], shard);
    } else {
        const DramRegion &region = tensor.chip_regions[shard / TtMeshConfig::tiles_per_chip];
        put_dram_ref(desc, ref_index, region.channel, region.offset, shard);
    }
}

void put_region_ref(uint32_t *desc, size_t ref_index, const DramRegion &region, size_t shard) {
    put_dram_ref(desc, ref_index, region.channel, region.offset, shard);
}

struct SimLayer {
    const TtqTensor *attention_norm = nullptr;
    const TtqTensor *post_attention_norm = nullptr;
    const TtqTensor *query_weight = nullptr;
    const TtqTensor *query_bias = nullptr;
    const GgufTensor *query_bias_f32 = nullptr;
    const TtqTensor *key_weight = nullptr;
    const TtqTensor *key_bias = nullptr;
    const GgufTensor *key_bias_f32 = nullptr;
    const TtqTensor *value_weight = nullptr;
    const TtqTensor *value_bias = nullptr;
    const GgufTensor *value_bias_f32 = nullptr;
    const TtqTensor *attention_output_weight = nullptr;
    const TtqTensor *attention_output_bias = nullptr;
    const GgufTensor *attention_output_bias_f32 = nullptr;
    const TtqTensor *attention_sinks = nullptr;
    const TtqTensor *router_weight = nullptr;
    const TtqTensor *router_bias = nullptr;
    const GgufTensor *router_bias_f32 = nullptr;
    const TtqTensor *gate_weight = nullptr;
    const TtqTensor *gate_bias = nullptr;
    const TtqTensor *up_weight = nullptr;
    const TtqTensor *up_bias = nullptr;
    const TtqTensor *down_weight = nullptr;
    const TtqTensor *down_bias = nullptr;
    const GgufTensor *cpu_attention_norm = nullptr;
    const GgufTensor *cpu_post_attention_norm = nullptr;
    const GgufTensor *cpu_query_weight = nullptr;
    const GgufTensor *cpu_query_bias = nullptr;
    const GgufTensor *cpu_key_weight = nullptr;
    const GgufTensor *cpu_key_bias = nullptr;
    const GgufTensor *cpu_value_weight = nullptr;
    const GgufTensor *cpu_value_bias = nullptr;
    const GgufTensor *cpu_attention_output_weight = nullptr;
    const GgufTensor *cpu_attention_output_bias = nullptr;
    const GgufTensor *cpu_attention_sinks = nullptr;
    const GgufTensor *cpu_router_weight = nullptr;
    const GgufTensor *cpu_router_bias = nullptr;
    const GgufTensor *cpu_gate_weight = nullptr;
    const GgufTensor *cpu_gate_bias = nullptr;
    const GgufTensor *cpu_up_weight = nullptr;
    const GgufTensor *cpu_up_bias = nullptr;
    const GgufTensor *cpu_down_weight = nullptr;
    const GgufTensor *cpu_down_bias = nullptr;
    std::array<DramRegion, TT_MAX_GLOBAL_SHARDS> key_cache_dram;
    std::array<DramRegion, TT_MAX_GLOBAL_SHARDS> value_cache_dram;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

struct HostProfile {
    std::array<uint64_t, TT_PROFILE_STAGES> device = {};
    double stage = 0;
    double command = 0;
    double convert = 0;
    size_t tokens = 0;
    size_t logits_tokens = 0;
};

struct SimModelState {
    HostProfile profile;
    const TtqTensor *embeddings = nullptr;
    const TtqTensor *output_norm = nullptr;
    const TtqTensor *output = nullptr;
    const GgufTensor *cpu_embeddings = nullptr;
    const GgufTensor *cpu_output_norm = nullptr;
    const GgufTensor *cpu_output = nullptr;
    std::vector<SimLayer> layers;
    std::array<DramRegion, TT_MAX_GLOBAL_SHARDS> attention_scores_dram;
    std::vector<float> rope_cosines;
    std::vector<float> rope_sines;
    std::vector<float> x;
    std::vector<float> normalized;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> attended;
    std::vector<float> projected;
    std::vector<float> residual;
    std::vector<float> router;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::vector<float> down;
    std::vector<uint16_t> down4;
    std::vector<float> moe_output;
    std::vector<float> logits;
    std::vector<uint16_t> logits_storage;
    uint16_t *logits_bf16 = nullptr;
    size_t norm_shards = 1;
    std::array<uint64_t, TtMeshConfig::max_chips> logits_noc_address = {};
    std::array<uint64_t, TtMeshConfig::max_chips> embedding_noc_address = {};
    std::vector<std::byte> embedding_storage;
    std::vector<float> attention_scores;
    std::array<uint32_t, proxy_b_values_count> proxy_b_values = {};
    std::vector<size_t> expert_indices;
    std::array<float, 4> expert_weights = {};
};

void proxy_forward_token(SimModelState &state, int32_t token, size_t position, size_t layer_limit);
double proxy_finish_logits(SimModelState &state);
int32_t sim_argmax(const float *values, size_t count);

uint64_t pin_host_buffer(size_t chip, void *buffer, uint64_t bytes) {
    tenstorrent_pin_pages pin{};
    pin.in.output_size_bytes = sizeof(pin.out);
    pin.in.virtual_address = uint64_t(uintptr_t(buffer));
    pin.in.size = bytes;
    if (ioctl(current_runtime->devices[chip].fd, TENSTORRENT_IOCTL_PIN_PAGES, &pin) < 0) {
        throw std::runtime_error("cannot pin host buffer for NOC DMA: " + std::string(std::strerror(errno)));
    }
    require_direct_dma_range(pin.out.physical_address, bytes);
    return pin.out.physical_address;
}

void enable_host_dma(SimModelState &state, TtMeshConfig mesh, bool is_silicon) {
    const uint64_t logits_bytes = align_u64(TT_GPT_OSS_VOCAB * sizeof(uint16_t), 4096);
    std::byte *embedding = nullptr;
    const uint64_t embedding_bytes = align_u64(state.embeddings->size, 4096);
    if (is_silicon) {
        arm_shard0_reset_on_close();
    }
    if (mesh.chip_count() > 1) {
        if (is_silicon) {
            state.embedding_storage.resize(embedding_bytes + 4096);
            embedding = (std::byte *)align_u64(uint64_t(uintptr_t(state.embedding_storage.data())), 4096);
            std::memcpy(embedding, state.embeddings->data, state.embeddings->size);
        } else {
            embedding = (std::byte *)state.embeddings->data;
        }
    }
    if (!is_silicon) {
        current_runtime->sim_embedding = embedding;
        current_runtime->sim_embedding_bytes = embedding ? state.embeddings->size : 0;
        current_runtime->sim_logits = (std::byte *)state.logits_bf16;
    }
    for (size_t chip = 0; chip < mesh.chip_count(); ++chip) {
        state.logits_noc_address[chip] =
            is_silicon ? pin_host_buffer(chip, state.logits_bf16, logits_bytes) : sim_logits_noc_base;
        if (embedding) {
            state.embedding_noc_address[chip] =
                is_silicon ? pin_host_buffer(chip, embedding, embedding_bytes) : sim_embedding_noc_base;
        }
    }
}

void write_logits_ref(TtSimApi &api, size_t shard, const SimModelState &state) {
    const uint64_t address = state.logits_noc_address[shard / TtMeshConfig::tiles_per_chip];
    write_u32(api, tile_l1_addr(shard, TT_L1_TOKEN_LOGITS_COORD_ADDR), pcie_tile_coord);
    write_u32(api, tile_l1_addr(shard, TT_L1_TOKEN_LOGITS_SRC_ADDR), uint32_t(address));
    write_u32(api, tile_l1_addr(shard, TT_L1_TOKEN_LOGITS_MID_ADDR), uint32_t(address >> 32));
}

void write_token_config(TtSimApi &api, TtMeshConfig mesh, const SimModelState &state, size_t layer_count) {
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        for (size_t stripe = 0; mesh.chip_count() == 1 && stripe < mesh.shard_count; ++stripe) {
            const uint32_t addr = TT_L1_EMBEDDING_STRIPE_ADDR + uint32_t(stripe * 8u);

            write_u32(api, tile_l1_addr(shard, addr), dram_coord(state.embeddings->stripe_channel[stripe], shard));
            write_u32(api, tile_l1_addr(shard, addr + 4u), uint32_t(state.embeddings->stripe_offset[stripe]));
        }
        const uint64_t embedding = state.embedding_noc_address[shard / TtMeshConfig::tiles_per_chip];
        write_u32(api, tile_l1_addr(shard, TT_L1_EMBEDDING_DMA_ADDR), uint32_t(embedding));
        write_u32(api, tile_l1_addr(shard, TT_L1_EMBEDDING_DMA_ADDR + 4), uint32_t(embedding >> 32));
        write_tensor_ref(api, shard, TT_L1_TOKEN_OUTPUT_NORM_COORD_ADDR, TT_L1_TOKEN_OUTPUT_NORM_SRC_ADDR,
                         *state.output_norm);
        write_tensor_ref(api, shard, TT_L1_TOKEN_OUTPUT_WEIGHT_COORD_ADDR, TT_L1_TOKEN_OUTPUT_WEIGHT_SRC_ADDR,
                         *state.output);
        write_logits_ref(api, shard, state);
        write_dram_ref(api, shard, TT_L1_ATTENTION_SCORE_COORD_ADDR, TT_L1_ATTENTION_SCORE_SRC_ADDR,
                       state.attention_scores_dram[shard]);
        write_u32(api, tile_l1_addr(shard, TT_L1_LAYER_COUNT_ADDR), uint32_t(layer_count));
        write_u32(api, tile_l1_addr(shard, TT_L1_EXPERT_COUNT_ADDR), uint32_t(state.router.size()));
        write_u32(api, tile_l1_addr(shard, TT_L1_COMMAND_ADDR), TT_COMMAND_TOKEN);
    }
}

void device_token(TtSimApi &api, TtMeshConfig mesh, SimModelState &state, int32_t token, size_t position,
                  size_t layer_count, bool compute_logits, bool is_silicon) {
    const auto profile_time = [] {
        return profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    };
    const auto stage_begin = profile_time();
    const float *position_cosines = state.rope_cosines.data() + position * (TT_GPT_OSS_HEAD_SIZE / 2);
    const float *position_sines = state.rope_sines.data() + position * (TT_GPT_OSS_HEAD_SIZE / 2);
    std::array<float, TT_ROPE_TABLE_ENTRIES> rope_cos_table = {};
    std::array<float, TT_ROPE_TABLE_ENTRIES> rope_sin_table = {};

    if (layer_count > TT_GPT_OSS_MAX_LAYERS || layer_count > state.layers.size()) {
        throw std::runtime_error("simulator layer count is too large");
    }
    for (size_t i = 0; i < TT_GPT_OSS_HEAD_SIZE / 2; ++i) {
        rope_cos_table[i] = position_cosines[i];
        rope_cos_table[i + TT_GPT_OSS_HEAD_SIZE / 2] = position_cosines[i];
        rope_sin_table[i] = -position_sines[i];
        rope_sin_table[i + TT_GPT_OSS_HEAD_SIZE / 2] = position_sines[i];
    }
    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        /* Every shard ropes its own Q and K slice, so all of them need the tables. */
        api.pci_mem_wr_bytes(tile_l1_addr(shard, TT_ROPE_COS_L1_ADDR), rope_cos_table.data(),
                             TT_ROPE_TABLE_ENTRIES * sizeof(float));
        api.pci_mem_wr_bytes(tile_l1_addr(shard, TT_ROPE_SIN_L1_ADDR), rope_sin_table.data(),
                             TT_ROPE_TABLE_ENTRIES * sizeof(float));
        write_u32(api, tile_l1_addr(shard, TT_L1_TOKEN_INDEX_ADDR), uint32_t(token));
        write_u32(api, tile_l1_addr(shard, TT_L1_LAYER_POSITION_ADDR), uint32_t(position));
        write_u32(api, tile_l1_addr(shard, TT_L1_TOKEN_FLAGS_ADDR),
                  (compute_logits ? TT_TOKEN_FLAG_LOGITS : 0u) | (profiling_enabled ? TT_TOKEN_FLAG_PROFILE : 0u));
    }
    const auto command_begin = profile_time();
    issue_brisc_command(api, mesh, 400000000u * (uint32_t(layer_count) + 1u), is_silicon ? 10.0 : 0.0);
    const auto command_end = profile_time();
    if (compute_logits) {
        for (size_t i = 0; i < TT_GPT_OSS_VOCAB; ++i) {
            state.logits[i] = bf16_to_fp32(state.logits_bf16[i]);
        }
    }
    if (profiling_enabled) {
        HostProfile &profile = state.profile;
        profile.stage += seconds_since(stage_begin, command_begin);
        profile.command += seconds_since(command_begin, command_end);
        ++profile.tokens;
        if (compute_logits) {
            profile.convert += seconds_since(command_end, profile_time());
            ++profile.logits_tokens;
        }
        std::array<uint32_t, TT_PROFILE_STAGES> cycles;
        api.pci_mem_rd_bytes(tile_l1_addr(0, TT_L1_PROFILE_STAGE_ADDR), cycles.data(), sizeof(cycles));
        for (size_t i = 0; i < cycles.size(); ++i) {
            profile.device[i] += cycles[i];
        }
    }
}

void report_profile(const HostProfile &profile) {
    if (!profile.tokens) {
        return;
    }
    static const char *const names[TT_PROFILE_STAGES] = {"embedding", "rms norm",  "q/k/v",     "rope",    "kv cache",
                                                         "attention", "o project", "router",    "experts", "swiglu",
                                                         "residual",  "logits",    "completion"};
    uint64_t total = 0;
    for (uint64_t cycles : profile.device) {
        total += cycles;
    }
    std::fprintf(stderr, "profile: %zu tokens, shard 0 BRISC timeline (includes waits and overlap)\n", profile.tokens);
    for (size_t i = 0; i < profile.device.size(); ++i) {
        std::fprintf(stderr, "  %-10s %12.0f cycles/token  %5.1f%%\n", names[i],
                     double(profile.device[i]) / double(profile.tokens),
                     total ? 100.0 * double(profile.device[i]) / double(total) : 0.0);
    }
    std::fprintf(stderr, "  total      %12.0f cycles/token\n", double(total) / double(profile.tokens));
    std::fprintf(stderr, "host: setup %.3f ms/token, command %.3f ms/token",
                 profile.stage * 1000 / double(profile.tokens), profile.command * 1000 / double(profile.tokens));
    if (profile.logits_tokens) {
        std::fprintf(stderr, ", convert %.3f ms/logits token (%zu tokens)",
                     profile.convert * 1000 / double(profile.logits_tokens), profile.logits_tokens);
    }
    std::fputc('\n', stderr);
}

float sim_dot(const float *a, const float *b, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void sim_apply_rope(const float *cosines, const float *sines, float *vector, size_t heads, size_t position) {
    const float *position_cosines = cosines + position * (TT_GPT_OSS_HEAD_SIZE / 2);
    const float *position_sines = sines + position * (TT_GPT_OSS_HEAD_SIZE / 2);

    for (size_t head = 0; head < heads; ++head) {
        float *values = vector + head * TT_GPT_OSS_HEAD_SIZE;
        for (size_t pair = 0; pair < TT_GPT_OSS_HEAD_SIZE / 2; ++pair) {
            const float cosine = position_cosines[pair];
            const float sine = position_sines[pair];
            const float x = values[pair];
            const float y = values[pair + TT_GPT_OSS_HEAD_SIZE / 2];
            values[pair] = x * cosine - y * sine;
            values[pair + TT_GPT_OSS_HEAD_SIZE / 2] = x * sine + y * cosine;
        }
    }
}

void proxy_apply_rope_sfpu(const float *cosines, const float *sines, float *vector, size_t heads, size_t position) {
    const float *position_cosines = cosines + position * (TT_GPT_OSS_HEAD_SIZE / 2);
    const float *position_sines = sines + position * (TT_GPT_OSS_HEAD_SIZE / 2);

    for (size_t head = 0; head < heads; ++head) {
        float *values = vector + head * TT_GPT_OSS_HEAD_SIZE;
        for (size_t pair = 0; pair < TT_GPT_OSS_HEAD_SIZE / 2; ++pair) {
            const float cosine = position_cosines[pair];
            const float sine = position_sines[pair];
            const float x = values[pair];
            const float y = values[pair + TT_GPT_OSS_HEAD_SIZE / 2];
            const float x_cos = sfpu_mul_value(x, cosine);
            const float y_cos = sfpu_mul_value(y, cosine);

            values[pair] = pack_sfpu_bf16_value(sfpu_mad_value(-y, sine, x_cos));
            values[pair + TT_GPT_OSS_HEAD_SIZE / 2] = pack_sfpu_bf16_value(sfpu_mad_value(x, sine, y_cos));
        }
    }
}

void attention_window(size_t layer_index, size_t position, size_t *first, size_t *count) {
    constexpr size_t sliding_window = 128;

    *first = layer_index % 2 == 0 && position + 1 > sliding_window ? position + 1 - sliding_window : 0;
    *count = position + 1 - *first;
}

void sim_attention_from_cache(const SimLayer &layer, size_t layer_index, size_t position, const float *query,
                              float *output, std::vector<float> &scores, bool cpu_sinks) {
    size_t first;
    size_t count;
    const float scale = 1.0f / std::sqrt(float(TT_GPT_OSS_HEAD_SIZE));

    attention_window(layer_index, position, &first, &count);

    for (size_t head = 0; head < TT_GPT_OSS_QUERY_HEADS; ++head) {
        const size_t kv_head = head / (TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS);
        const float *q = query + head * TT_GPT_OSS_HEAD_SIZE;
        const float sink = cpu_sinks ? ((const float *)layer.cpu_attention_sinks->data)[head]
                                     : bf16_linear_value(*layer.attention_sinks, head);
        float maximum = sink;
        for (size_t i = 0; i < count; ++i) {
            const float *k =
                layer.key_cache.data() + ((first + i) * TT_GPT_OSS_KV_HEADS + kv_head) * TT_GPT_OSS_HEAD_SIZE;
            scores[i] = sim_dot(q, k, TT_GPT_OSS_HEAD_SIZE) * scale;
            maximum = std::max(maximum, scores[i]);
        }
        float denominator = std::exp(sink - maximum);
        for (size_t i = 0; i < count; ++i) {
            scores[i] = std::exp(scores[i] - maximum);
            denominator += scores[i];
        }
        float *result = output + head * TT_GPT_OSS_HEAD_SIZE;
        std::fill(result, result + TT_GPT_OSS_HEAD_SIZE, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            const float weight = scores[i] / denominator;
            const float *v =
                layer.value_cache.data() + ((first + i) * TT_GPT_OSS_KV_HEADS + kv_head) * TT_GPT_OSS_HEAD_SIZE;
            for (size_t d = 0; d < TT_GPT_OSS_HEAD_SIZE; ++d) {
                result[d] += weight * v[d];
            }
        }
    }
}

float attention_exp_approx(float x) {
    constexpr float inv_ln2 = 1.4426950408889634f;
    constexpr float ln2 = 0.6931471805599453f;

    if (x <= -20.0f) {
        return 0.0f;
    }
    if (x >= 0.0f) {
        x = 0.0f;
    }

    int n = int(std::nearbyint(rv32f_mul_value(x, inv_ln2)));
    const float r = rv32f_sub_value(x, rv32f_mul_value(float(n), ln2));
    const float r2 = rv32f_mul_value(r, r);
    const float poly1 = rv32f_add_value(r, 1.0f);
    const float r3 = rv32f_mul_value(r, r2);
    const float poly2 = rv32f_fma_value(r2, 0.5f, poly1);
    const float r4 = rv32f_mul_value(r2, r2);
    const float poly3 = rv32f_fma_value(r3, 0.1666666716f, poly2);
    const float r5 = rv32f_mul_value(r, r4);
    const float poly4 = rv32f_fma_value(r4, 0.0416666679f, poly3);
    const float poly5 = rv32f_fma_value(r5, 0.0083333338f, poly4);

    if (n < -126) {
        return 0.0f;
    }

    const uint32_t scale_bits = uint32_t(n + 127) << 23;
    const float scale = std::bit_cast<float>(scale_bits);
    return rv32f_mul_value(scale, poly5);
}

// Match sfpu_exp_negative: separate ADD/MUL instructions and fused polynomial terms.
float attention_exp_sfpu(float score, float maximum) {
    constexpr uint32_t one = 0x3F800000u;
    const uint32_t x = sfpu_add_bh(std::bit_cast<uint32_t>(score), std::bit_cast<uint32_t>(maximum) ^ 0x80000000u);
    const uint32_t bound = sfpu_add_bh(0x41A00000u, x);

    if (int32_t(bound - 1u) < 0) {
        return 0.0f;
    }

    const uint32_t t = sfpu_mul_bh(x, 0x3FB8AA3Bu);
    const uint32_t rounded = sfpu_add_bh(t, 0x4B400000u);
    const uint32_t n = sfpu_add_bh(rounded, 0x4B400000u ^ 0x80000000u);
    const uint32_t reduced = sfpu_mul_bh(n, 0x3F317218u);
    const uint32_t r = sfpu_add_bh(x, reduced ^ 0x80000000u);
    const uint32_t r2 = sfpu_mul_bh(r, r);
    uint32_t poly = sfpu_add_bh(r, one);

    poly = sfpu_mad_bh(r2, 0x3F000000u, poly);
    const uint32_t r3 = sfpu_mul_bh(r, r2);
    poly = sfpu_mad_bh(r3, 0x3E2AAAABu, poly);
    const uint32_t r4 = sfpu_mul_bh(r2, r2);
    poly = sfpu_mad_bh(r4, 0x3D2AAAABu, poly);
    const uint32_t r5 = sfpu_mul_bh(r, r4);
    poly = sfpu_mad_bh(r5, 0x3C088889u, poly);

    const uint32_t scale = ((rounded + 127u) & 0xFFu) << 23;
    return std::bit_cast<float>(sfpu_mul_bh(scale, poly));
}

float attention_recip_approx(float x) {
    float y = std::bit_cast<float>(0x7EF311C3u - std::bit_cast<uint32_t>(x));

    y = rv32f_mul_value(y, rv32f_fnmsub_value(x, y, 2.0f));
    y = rv32f_mul_value(y, rv32f_fnmsub_value(x, y, 2.0f));
    y = rv32f_mul_value(y, rv32f_fnmsub_value(x, y, 2.0f));
    return y;
}

float proxy_attention_score_sfpu(const uint16_t *query, const float *key, size_t query_base) {
    float partials[8];

    for (size_t col = 0; col < 8; ++col) {
        float rows[4];
        for (size_t row = 0; row < 4; ++row) {
            const size_t dim = row * 16 + col * 2;
            const float q0 = bf16_to_fp32(query[query_base + dim]);
            const float k0 = bf16_to_fp32(fp32_to_bf16(key[dim]));
            const float q1 = bf16_to_fp32(query[query_base + dim + 1]);
            const float k1 = bf16_to_fp32(fp32_to_bf16(key[dim + 1]));
            const float product = sfpu_mul_value(q0, k0);
            rows[row] = sfpu_mad_value(q1, k1, product);
        }
        partials[col] = sfpu_add_value(sfpu_add_value(rows[3], rows[2]), sfpu_add_value(rows[1], rows[0]));
    }

    for (size_t stage = 1; stage < 8; stage <<= 1) {
        float rotated[8];

        for (size_t i = 0; i < 8; ++i) {
            rotated[i] = partials[(i + 8 - stage) & 7];
        }
        for (size_t i = 0; i < 8; ++i) {
            partials[i] = sfpu_add_value(rotated[i], partials[i]);
        }
    }
    return sfpu_mul_value(0.125f, partials[0]);
}

void proxy_attention_rv32f(const SimLayer &layer, size_t layer_index, size_t position, const float *query,
                           float *scores, float *output) {
    size_t first;
    size_t count;
    uint16_t query_bf16[gpt_oss_query_width];
    float maximum[TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS];
    float denominator[TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS];
    float reciprocal[TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS];
    float head_output[TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS][TT_GPT_OSS_HEAD_SIZE];

    attention_window(layer_index, position, &first, &count);
    for (size_t i = 0; i < gpt_oss_query_width; ++i) {
        query_bf16[i] = fp32_to_bf16(query[i]);
    }

    for (size_t kv_head = 0; kv_head < TT_GPT_OSS_KV_HEADS; ++kv_head) {
        const size_t head_base = kv_head * (TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS);
        for (size_t local_head = 0; local_head < TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS; ++local_head) {
            maximum[local_head] = bf16_linear_value(*layer.attention_sinks, head_base + local_head);
        }
        for (size_t token = 0; token < count; ++token) {
            const float *key =
                layer.key_cache.data() + ((first + token) * TT_GPT_OSS_KV_HEADS + kv_head) * TT_GPT_OSS_HEAD_SIZE;
            for (size_t local_head = 0; local_head < TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS; ++local_head) {
                const size_t query_base = (head_base + local_head) * TT_GPT_OSS_HEAD_SIZE;
                const float sum = proxy_attention_score_sfpu(query_bf16, key, query_base);

                scores[token * (TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS) + local_head] = sum;
                maximum[local_head] = rv32f_max_value(maximum[local_head], sum);
            }
        }

        for (size_t local_head = 0; local_head < TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS; ++local_head) {
            const float sink = bf16_linear_value(*layer.attention_sinks, head_base + local_head);

            denominator[local_head] = attention_exp_approx(sink - maximum[local_head]);
        }
        for (size_t token = 0; token < count; ++token) {
            for (size_t local_head = 0; local_head < TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS; ++local_head) {
                const size_t score_index = token * (TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS) + local_head;

                scores[score_index] = attention_exp_sfpu(scores[score_index], maximum[local_head]);
            }
        }

        for (size_t chunk = 0; chunk < count; chunk += TT_ATTENTION_SCORE_CHUNK_TOKENS) {
            const size_t tokens = std::min(count - chunk, size_t(TT_ATTENTION_SCORE_CHUNK_TOKENS));
            const size_t complete = tokens >= 16 ? tokens / 8 * 8 : 0;
            for (size_t head = 0; head < TT_GPT_OSS_QUERY_GROUP; ++head) {
                if (complete) {
                    for (size_t lane = 0; lane < 8; ++lane) {
                        uint32_t sum = std::bit_cast<uint32_t>(scores[(chunk + lane) * TT_GPT_OSS_QUERY_GROUP + head]);
                        for (size_t token = lane + 8; token < complete; token += 8) {
                            sum = pack_l1_acc_fp32_proxy(
                                sum, std::bit_cast<uint32_t>(scores[(chunk + token) * TT_GPT_OSS_QUERY_GROUP + head]));
                        }
                        denominator[head] = rv32f_add_value(denominator[head], std::bit_cast<float>(sum));
                    }
                }
                for (size_t token = complete; token < tokens; ++token) {
                    denominator[head] =
                        rv32f_add_value(denominator[head], scores[(chunk + token) * TT_GPT_OSS_QUERY_GROUP + head]);
                }
            }
        }

        for (size_t local_head = 0; local_head < TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS; ++local_head) {
            reciprocal[local_head] = attention_recip_approx(denominator[local_head]);
            for (size_t dim = 0; dim < TT_GPT_OSS_HEAD_SIZE; ++dim) {
                head_output[local_head][dim] = 0.0f;
            }
        }
        for (size_t token = 0; token < count; ++token) {
            const float *value =
                layer.value_cache.data() + ((first + token) * TT_GPT_OSS_KV_HEADS + kv_head) * TT_GPT_OSS_HEAD_SIZE;
            for (size_t local_head = 0; local_head < TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS; ++local_head) {
                const size_t score_index = token * (TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS) + local_head;
                const float weight = scores[score_index];

                for (size_t dim = 0; dim < TT_GPT_OSS_HEAD_SIZE; ++dim) {
                    head_output[local_head][dim] =
                        sfpu_mad_value(bf16_to_fp32(fp32_to_bf16(value[dim])), weight, head_output[local_head][dim]);
                }
            }
        }
        for (size_t local_head = 0; local_head < TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS; ++local_head) {
            const size_t query_base = (head_base + local_head) * TT_GPT_OSS_HEAD_SIZE;

            for (size_t dim = 0; dim < TT_GPT_OSS_HEAD_SIZE; ++dim) {
                output[query_base + dim] =
                    pack_sfpu_bf16_value(sfpu_mul_value(head_output[local_head][dim], reciprocal[local_head]));
            }
        }
    }
}

void sim_select_top4(const std::vector<float> &router, std::vector<size_t> &indices, std::array<float, 4> &weights) {
    for (size_t i = 0; i < router.size(); ++i) {
        indices[i] = i;
    }
    for (size_t i = 0; i < 4; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < router.size(); ++j) {
            if (router[indices[j]] > router[indices[best]]) {
                best = j;
            }
        }
        const size_t tmp = indices[i];
        indices[i] = indices[best];
        indices[best] = tmp;
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < 4; ++i) {
        maximum = std::max(maximum, router[indices[i]]);
    }
    float weight_sum = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        weights[i] = std::exp(router[indices[i]] - maximum);
        weight_sum += weights[i];
    }
    for (size_t i = 0; i < 4; ++i) {
        weights[i] /= weight_sum;
    }
}

void proxy_select_top4(const std::vector<float> &router, std::vector<size_t> &indices, std::array<float, 4> &weights) {
    float exp_weights[4];
    // Mirror the SFPSWAP network, including its sign-dependent handling of equal values.
    // Each entry represents a paired value/index lane; transpose exchanges register/subvector axes.
    struct Entry {
        uint32_t value;
        size_t index;
    };
    Entry reg[4][4][8];
    for (size_t r = 0; r < 4; ++r) {
        for (size_t s = 0; s < 4; ++s) {
            for (size_t c = 0; c < 8; ++c) {
                const size_t i = (r / 2) * 64 + (r & 1) + 16 * s + 2 * c;
                reg[r][s][c] = {i < router.size() ? std::bit_cast<uint32_t>(router[i]) : 0u, i};
            }
        }
    }
    const auto swap = [&](size_t a, size_t b, unsigned mode) {
        for (size_t s = 0; s < 4; ++s) {
            const bool a_max = mode == 1 || (mode == 2 ? s < 2 : (s & 1) == 0);
            for (size_t c = 0; c < 8; ++c) {
                const uint32_t av = reg[a][s][c].value;
                const uint32_t bv = reg[b][s][c].value;
                const int32_t ak = int32_t(av ^ (uint32_t(int32_t(av) >> 30) >> 1));
                const int32_t bk = int32_t(bv ^ (uint32_t(int32_t(bv) >> 30) >> 1));
                const bool smaller = ak < bk || (av == bv && (av >> 31));
                if (smaller == a_max) {
                    std::swap(reg[a][s][c], reg[b][s][c]);
                }
            }
        }
    };
    const auto transpose = [&] {
        for (size_t r = 0; r < 4; ++r) {
            for (size_t s = r + 1; s < 4; ++s) {
                for (size_t c = 0; c < 8; ++c) {
                    std::swap(reg[r][s][c], reg[s][r][c]);
                }
            }
        }
    };
    const auto merge4 = [&](unsigned mode) {
        swap(0, 2, mode);
        swap(1, 3, mode);
        swap(0, 1, mode);
        swap(2, 3, mode);
    };
    if (router.size() == 128) {
        swap(0, 1, 3);
        swap(2, 3, 3);
        swap(0, 2, 3);
        swap(1, 3, 3);
        swap(1, 2, 3);
        transpose();
        swap(0, 1, 1);
        swap(2, 3, 1);
        transpose();
        merge4(2);
        transpose();
        swap(0, 2, 1);
        transpose();
        merge4(1);
    } else {
        swap(0, 1, 3);
        transpose();
        swap(0, 1, 1);
        transpose();
        swap(0, 1, 1);
    }
    const auto candidate = [&](size_t c, size_t rank) {
        return router.size() == 128 ? reg[rank][0][c] : reg[rank & 1][rank / 2][c];
    };
    size_t position[8] = {};
    for (size_t i = 0; i < 4; ++i) {
        size_t best = 0;
        for (size_t c = 1; c < 8; ++c) {
            if (std::bit_cast<float>(candidate(c, position[c]).value) >
                std::bit_cast<float>(candidate(best, position[best]).value)) {
                best = c;
            }
        }
        indices[i] = candidate(best, position[best]++).index;
    }
    const float maximum = router[indices[0]];

    float weight_sum = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        exp_weights[i] = attention_exp_approx(rv32f_sub_value(router[indices[i]], maximum));
        weight_sum = rv32f_add_value(weight_sum, exp_weights[i]);
    }
    const float reciprocal = attention_recip_approx(weight_sum);
    for (size_t i = 0; i < 4; ++i) {
        weights[i] = rv32f_mul_value(exp_weights[i], reciprocal);
    }
}

void init_sim_state(const GgufFile &file, const TtqFile &requant, size_t token_count, SimModelState &state,
                    bool host_kv_cache = true, bool cpu_tensors = false) {
    requant.validate_model(file);
    const size_t experts = requant.expert_count();
    state.layers.resize(requant.layer_count());
    state.expert_indices.resize(experts);
    constexpr float rope_base = 150000.0f;
    constexpr float rope_scale = 1.0f / 32.0f;
    constexpr size_t rope_original_context = 4096;
    constexpr float rope_beta_fast = 32.0f;
    constexpr float rope_beta_slow = 1.0f;

    state.embeddings =
        &require_ttq_tensor(requant, "token_embd.weight", ttq_format_bf16_linear, {TT_GPT_OSS_DIM, 201088});
    state.output_norm = &require_ttq_tensor(requant, "output_norm.weight", ttq_format_bf16_linear, {TT_GPT_OSS_DIM});
    state.output = &require_ttq_tensor(requant, "output.weight", ttq_format_bf16_tensix, {TT_GPT_OSS_DIM, 201088});
    if (cpu_tensors) {
        state.cpu_embeddings = &require_sim_tensor(file, "token_embd.weight", tensor_q8_0, {TT_GPT_OSS_DIM, 201088});
        state.cpu_output_norm = &require_sim_tensor(file, "output_norm.weight", tensor_f32, {TT_GPT_OSS_DIM});
        state.cpu_output = &require_sim_tensor(file, "output.weight", tensor_q8_0, {TT_GPT_OSS_DIM, 201088});
    }

    const float corr = float(TT_GPT_OSS_HEAD_SIZE) *
                       std::log(float(rope_original_context) / (rope_beta_fast * 2.0f * std::numbers::pi_v<float>)) /
                       (2.0f * std::log(rope_base));
    const float corr_slow =
        float(TT_GPT_OSS_HEAD_SIZE) *
        std::log(float(rope_original_context) / (rope_beta_slow * 2.0f * std::numbers::pi_v<float>)) /
        (2.0f * std::log(rope_base));
    const float magnitude = 1.0f + 0.1f * std::log(1.0f / rope_scale);
    const float low = std::max(0.0f, std::floor(corr));
    const float high = std::min(float(TT_GPT_OSS_HEAD_SIZE - 1), std::ceil(corr_slow));
    state.rope_cosines.resize(token_count * (TT_GPT_OSS_HEAD_SIZE / 2));
    state.rope_sines.resize(token_count * (TT_GPT_OSS_HEAD_SIZE / 2));
    for (size_t pair = 0; pair < TT_GPT_OSS_HEAD_SIZE / 2; ++pair) {
        const size_t dimension = pair * 2;
        const float theta_extrap = std::pow(rope_base, -float(dimension) / float(TT_GPT_OSS_HEAD_SIZE));
        const float y = (float(pair) - low) / std::max(0.001f, high - low);
        const float ramp = 1.0f - std::clamp(y, 0.0f, 1.0f);
        const float frequency = rope_scale * theta_extrap * (1.0f - ramp) + theta_extrap * ramp;

        for (size_t position = 0; position < token_count; ++position) {
            const float theta = float(position) * frequency;
            state.rope_cosines[position * (TT_GPT_OSS_HEAD_SIZE / 2) + pair] = std::cos(theta) * magnitude;
            state.rope_sines[position * (TT_GPT_OSS_HEAD_SIZE / 2) + pair] = std::sin(theta) * magnitude;
        }
    }

    for (size_t i = 0; i < state.layers.size(); ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        SimLayer &layer = state.layers[i];
        layer.attention_norm = &require_ttq_tensor(requant, (prefix + "attn_norm.weight").c_str(),
                                                   ttq_format_bf16_linear, {TT_GPT_OSS_DIM});
        layer.post_attention_norm = &require_ttq_tensor(requant, (prefix + "post_attention_norm.weight").c_str(),
                                                        ttq_format_bf16_linear, {TT_GPT_OSS_DIM});
        layer.query_weight = &require_ttq_tensor(requant, (prefix + "attn_q.weight").c_str(), ttq_format_bf16_tensix,
                                                 {TT_GPT_OSS_DIM, gpt_oss_query_width});
        layer.query_bias_f32 =
            &require_sim_tensor(file, (prefix + "attn_q.bias").c_str(), tensor_f32, {gpt_oss_query_width});
        layer.query_bias = &require_ttq_tensor(requant, (prefix + "attn_q.bias").c_str(), ttq_format_bf16_linear,
                                               {gpt_oss_query_width});
        layer.key_weight = &require_ttq_tensor(requant, (prefix + "attn_k.weight").c_str(), ttq_format_bf16_tensix,
                                               {TT_GPT_OSS_DIM, gpt_oss_kv_width});
        layer.key_bias_f32 =
            &require_sim_tensor(file, (prefix + "attn_k.bias").c_str(), tensor_f32, {gpt_oss_kv_width});
        layer.key_bias =
            &require_ttq_tensor(requant, (prefix + "attn_k.bias").c_str(), ttq_format_bf16_linear, {gpt_oss_kv_width});
        layer.value_weight = &require_ttq_tensor(requant, (prefix + "attn_v.weight").c_str(), ttq_format_bf16_tensix,
                                                 {TT_GPT_OSS_DIM, gpt_oss_kv_width});
        layer.value_bias =
            &require_ttq_tensor(requant, (prefix + "attn_v.bias").c_str(), ttq_format_bf16_linear, {gpt_oss_kv_width});
        layer.value_bias_f32 =
            &require_sim_tensor(file, (prefix + "attn_v.bias").c_str(), tensor_f32, {gpt_oss_kv_width});
        layer.attention_output_weight =
            &require_ttq_tensor(requant, (prefix + "attn_output.weight").c_str(), ttq_format_bf16_tensix,
                                {gpt_oss_query_width, TT_GPT_OSS_DIM});
        layer.attention_output_bias_f32 =
            &require_sim_tensor(file, (prefix + "attn_output.bias").c_str(), tensor_f32, {TT_GPT_OSS_DIM});
        layer.attention_output_bias = &require_ttq_tensor(requant, (prefix + "attn_output.bias").c_str(),
                                                          ttq_format_bf16_linear, {TT_GPT_OSS_DIM});
        layer.attention_sinks = &require_ttq_tensor(requant, (prefix + "attn_sinks.weight").c_str(),
                                                    ttq_format_bf16_linear, {TT_GPT_OSS_QUERY_HEADS});
        layer.router_weight = &require_ttq_tensor(requant, (prefix + "ffn_gate_inp.weight").c_str(),
                                                  ttq_format_bf16_tensix, {TT_GPT_OSS_DIM, experts});
        layer.router_bias_f32 =
            &require_sim_tensor(file, (prefix + "ffn_gate_inp.bias").c_str(), tensor_f32, {experts});
        layer.router_bias =
            &require_ttq_tensor(requant, (prefix + "ffn_gate_inp.bias").c_str(), ttq_format_bf16_linear, {experts});
        layer.gate_weight = &require_ttq_tensor(requant, (prefix + "ffn_gate_exps.weight.bfp8").c_str(),
                                                ttq_format_bfp8_strip, {TT_GPT_OSS_DIM, TT_GPT_OSS_DIM, experts});
        layer.gate_bias = &require_ttq_tensor(requant, (prefix + "ffn_gate_exps.bias").c_str(), ttq_format_bf16_linear,
                                              {TT_GPT_OSS_DIM, experts});
        layer.up_weight = &require_ttq_tensor(requant, (prefix + "ffn_up_exps.weight.bfp8").c_str(),
                                              ttq_format_bfp8_strip, {TT_GPT_OSS_DIM, TT_GPT_OSS_DIM, experts});
        layer.up_bias = &require_ttq_tensor(requant, (prefix + "ffn_up_exps.bias").c_str(), ttq_format_bf16_linear,
                                            {TT_GPT_OSS_DIM, experts});
        layer.down_weight = &require_ttq_tensor(requant, (prefix + "ffn_down_exps.weight.bfp8").c_str(),
                                                ttq_format_bfp8_strip, {TT_GPT_OSS_DIM, TT_GPT_OSS_DIM, experts});
        layer.down_bias = &require_ttq_tensor(requant, (prefix + "ffn_down_exps.bias").c_str(), ttq_format_bf16_linear,
                                              {TT_GPT_OSS_DIM, experts});
        if (cpu_tensors) {
            layer.cpu_attention_norm =
                &require_sim_tensor(file, (prefix + "attn_norm.weight").c_str(), tensor_f32, {TT_GPT_OSS_DIM});
            layer.cpu_post_attention_norm = &require_sim_tensor(file, (prefix + "post_attention_norm.weight").c_str(),
                                                                tensor_f32, {TT_GPT_OSS_DIM});
            layer.cpu_query_weight = &require_sim_tensor(file, (prefix + "attn_q.weight").c_str(), tensor_q8_0,
                                                         {TT_GPT_OSS_DIM, gpt_oss_query_width});
            layer.cpu_query_bias = layer.query_bias_f32;
            layer.cpu_key_weight = &require_sim_tensor(file, (prefix + "attn_k.weight").c_str(), tensor_q8_0,
                                                       {TT_GPT_OSS_DIM, gpt_oss_kv_width});
            layer.cpu_key_bias = layer.key_bias_f32;
            layer.cpu_value_weight = &require_sim_tensor(file, (prefix + "attn_v.weight").c_str(), tensor_q8_0,
                                                         {TT_GPT_OSS_DIM, gpt_oss_kv_width});
            layer.cpu_value_bias = layer.value_bias_f32;
            layer.cpu_attention_output_weight = &require_sim_tensor(file, (prefix + "attn_output.weight").c_str(),
                                                                    tensor_q8_0, {gpt_oss_query_width, TT_GPT_OSS_DIM});
            layer.cpu_attention_output_bias = layer.attention_output_bias_f32;
            layer.cpu_attention_sinks =
                &require_sim_tensor(file, (prefix + "attn_sinks.weight").c_str(), tensor_f32, {TT_GPT_OSS_QUERY_HEADS});
            layer.cpu_router_weight = &require_sim_tensor(file, (prefix + "ffn_gate_inp.weight").c_str(), tensor_f32,
                                                          {TT_GPT_OSS_DIM, experts});
            layer.cpu_router_bias = layer.router_bias_f32;
            layer.cpu_gate_weight = &require_sim_tensor(file, (prefix + "ffn_gate_exps.weight").c_str(), tensor_mxfp4,
                                                        {TT_GPT_OSS_DIM, TT_GPT_OSS_DIM, experts});
            layer.cpu_gate_bias = &require_sim_tensor(file, (prefix + "ffn_gate_exps.bias").c_str(), tensor_f32,
                                                      {TT_GPT_OSS_DIM, experts});
            layer.cpu_up_weight = &require_sim_tensor(file, (prefix + "ffn_up_exps.weight").c_str(), tensor_mxfp4,
                                                      {TT_GPT_OSS_DIM, TT_GPT_OSS_DIM, experts});
            layer.cpu_up_bias =
                &require_sim_tensor(file, (prefix + "ffn_up_exps.bias").c_str(), tensor_f32, {TT_GPT_OSS_DIM, experts});
            layer.cpu_down_weight = &require_sim_tensor(file, (prefix + "ffn_down_exps.weight").c_str(), tensor_mxfp4,
                                                        {TT_GPT_OSS_DIM, TT_GPT_OSS_DIM, experts});
            layer.cpu_down_bias = &require_sim_tensor(file, (prefix + "ffn_down_exps.bias").c_str(), tensor_f32,
                                                      {TT_GPT_OSS_DIM, experts});
        }
        if (host_kv_cache) {
            layer.key_cache.resize(token_count * gpt_oss_kv_width);
            layer.value_cache.resize(token_count * gpt_oss_kv_width);
        }
    }

    state.x.resize(TT_GPT_OSS_DIM);
    state.normalized.resize(TT_GPT_OSS_DIM);
    state.query.resize(gpt_oss_query_width);
    state.key.resize(gpt_oss_kv_width);
    state.value.resize(gpt_oss_kv_width);
    state.attended.resize(gpt_oss_query_width);
    state.projected.resize(TT_GPT_OSS_DIM);
    state.residual.resize(TT_GPT_OSS_DIM);
    state.router.resize(experts);
    state.gate.resize(TT_GPT_OSS_DIM);
    state.up.resize(TT_GPT_OSS_DIM);
    state.hidden.resize(TT_GPT_OSS_DIM);
    state.down.resize(TT_GPT_OSS_DIM);
    state.down4.resize(4 * TT_GPT_OSS_DIM);
    state.moe_output.resize(TT_GPT_OSS_DIM);
    state.logits.resize(TT_GPT_OSS_VOCAB);
    /* Cover both the aligned start and the page-rounded pinning range. */
    state.logits_storage.resize((align_u64(TT_GPT_OSS_VOCAB * sizeof(uint16_t), 4096) + 4096) / sizeof(uint16_t));
    state.logits_bf16 = (uint16_t *)align_u64(uint64_t(uintptr_t(state.logits_storage.data())), 4096);
    if (host_kv_cache) {
        state.attention_scores.resize(token_count * (TT_GPT_OSS_QUERY_HEADS / TT_GPT_OSS_KV_HEADS));
    }
}

void allocate_sim_kv_cache(DramCursor &cursor, SimModelState &state, size_t token_capacity, size_t layer_limit,
                           TtMeshConfig mesh) {
    static_assert(TT_MAX_GLOBAL_SHARDS % TT_GPT_OSS_KV_HEADS == 0);
    static_assert(TtMeshConfig::tiles_per_chip % (TT_MAX_GLOBAL_SHARDS / TT_GPT_OSS_KV_HEADS) == 0);
    const size_t kv_heads = std::max(size_t(1), TT_GPT_OSS_KV_HEADS / mesh.shard_count);
    const size_t shards_per_head = std::max(size_t(1), mesh.shard_count / TT_GPT_OSS_KV_HEADS);
    const uint64_t cache_bytes = uint64_t(token_capacity) * kv_heads * TT_GPT_OSS_HEAD_SIZE * sizeof(uint16_t);

    for (size_t i = 0; i < layer_limit; ++i) {
        for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
            const size_t owner = shard / shards_per_head * shards_per_head;

            // At 32 shards, four chip-local peers share one complete KV head.
            if (shard != owner) {
                state.layers[i].key_cache_dram[shard] = state.layers[i].key_cache_dram[owner];
                state.layers[i].value_cache_dram[shard] = state.layers[i].value_cache_dram[owner];
            } else if (mesh.shard_count == 1) {
                state.layers[i].key_cache_dram[shard] = alloc_dram_region(cursor, cache_bytes, 64);
                state.layers[i].value_cache_dram[shard] = alloc_dram_region(cursor, cache_bytes, 64);
            } else {
                const uint32_t channel = shard_dram_channel(shard);

                state.layers[i].key_cache_dram[shard] = alloc_dram_region_in_channel(cursor, channel, cache_bytes, 64);
                state.layers[i].value_cache_dram[shard] =
                    alloc_dram_region_in_channel(cursor, channel, cache_bytes, 64);
            }
        }
    }
}

void fill_layer_descriptor(uint32_t *desc, const SimLayer &layer, size_t shard) {
    put_tensor_ref(desc, TT_LAYER_REF_ATTENTION_NORM, *layer.attention_norm, shard);
    put_tensor_ref(desc, TT_LAYER_REF_POST_ATTENTION_NORM, *layer.post_attention_norm, shard);
    put_tensor_ref(desc, TT_LAYER_REF_QUERY_WEIGHT, *layer.query_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_QUERY_BIAS, *layer.query_bias, shard);
    put_tensor_ref(desc, TT_LAYER_REF_KEY_WEIGHT, *layer.key_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_KEY_BIAS, *layer.key_bias, shard);
    put_tensor_ref(desc, TT_LAYER_REF_VALUE_WEIGHT, *layer.value_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_VALUE_BIAS, *layer.value_bias, shard);
    put_tensor_ref(desc, TT_LAYER_REF_ATTENTION_OUTPUT_WEIGHT, *layer.attention_output_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_ATTENTION_OUTPUT_BIAS, *layer.attention_output_bias, shard);
    put_tensor_ref(desc, TT_LAYER_REF_ATTENTION_SINKS, *layer.attention_sinks, shard);
    put_tensor_ref(desc, TT_LAYER_REF_ROUTER_WEIGHT, *layer.router_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_ROUTER_BIAS, *layer.router_bias, shard);
    put_tensor_ref(desc, TT_LAYER_REF_GATE_WEIGHT, *layer.gate_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_GATE_BIAS, *layer.gate_bias, shard);
    put_tensor_ref(desc, TT_LAYER_REF_UP_WEIGHT, *layer.up_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_UP_BIAS, *layer.up_bias, shard);
    put_tensor_ref(desc, TT_LAYER_REF_DOWN_WEIGHT, *layer.down_weight, shard);
    put_tensor_ref(desc, TT_LAYER_REF_DOWN_BIAS, *layer.down_bias, shard);
    put_region_ref(desc, TT_LAYER_REF_ATTENTION_KEY_CACHE, layer.key_cache_dram[shard], shard);
    put_region_ref(desc, TT_LAYER_REF_ATTENTION_VALUE_CACHE, layer.value_cache_dram[shard], shard);
}

void write_layer_descriptors_to_l1(TtSimApi &api, TtMeshConfig mesh, SimModelState &state, size_t layer_limit) {
    if (layer_limit > TT_GPT_OSS_MAX_LAYERS || layer_limit > state.layers.size()) {
        throw std::runtime_error("firmware layer descriptor count is too large");
    }
    std::array<uint32_t, TT_LAYER_DESC_WORDS * TT_GPT_OSS_MAX_LAYERS> descriptors = {};
    const uint32_t descriptor_bytes = uint32_t(layer_limit * TT_LAYER_DESC_BYTES);

    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        for (size_t i = 0; i < layer_limit; ++i) {
            fill_layer_descriptor(descriptors.data() + i * TT_LAYER_DESC_WORDS, state.layers[i], shard);
        }
        api.pci_mem_wr_bytes(tile_l1_addr(shard, TT_LAYER_DESC_L1_ADDR), descriptors.data(), descriptor_bytes);
    }
}

void allocate_sim_attention_scores(DramCursor &cursor, SimModelState &state, size_t token_capacity, TtMeshConfig mesh) {
    const uint64_t score_bytes = uint64_t(token_capacity) * TT_ATTENTION_SCORE_STRIDE_BYTES;

    for (size_t shard = 0; shard < mesh.shard_count; ++shard) {
        if (mesh.shard_count == 1) {
            state.attention_scores_dram[shard] = alloc_dram_region(cursor, score_bytes, 64);
        } else {
            const uint32_t channel = shard_dram_channel(shard);

            state.attention_scores_dram[shard] = alloc_dram_region_in_channel(cursor, channel, score_bytes, 64);
        }
    }
}

void cpu_moe(const SimLayer &layer, SimModelState &state, const float *input, float *output) {
    matvec_add_f32(*layer.cpu_router_weight, *layer.cpu_router_bias, input, state.router.data());
    sim_select_top4(state.router, state.expert_indices, state.expert_weights);

    std::fill(output, output + TT_GPT_OSS_DIM, 0.0f);
    for (size_t selected = 0; selected < 4; ++selected) {
        const size_t expert = state.expert_indices[selected];
        matvec_add_mxfp4_2880(*layer.cpu_gate_weight, *layer.cpu_gate_bias, input, state.gate.data(), expert);
        matvec_add_mxfp4_2880(*layer.cpu_up_weight, *layer.cpu_up_bias, input, state.up.data(), expert);
        for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
            const float x = std::min(state.gate[i], 7.0f);
            const float y = std::clamp(state.up[i], -7.0f, 7.0f);
            state.hidden[i] = (x / (1.0f + std::exp(-1.702f * x))) * (y + 1.0f);
        }
        matvec_add_mxfp4_2880(*layer.cpu_down_weight, *layer.cpu_down_bias, state.hidden.data(), state.down.data(),
                              expert);
        for (size_t i = 0; i < TT_GPT_OSS_DIM; ++i) {
            output[i] += state.expert_weights[selected] * state.down[i];
        }
    }
}

void cpu_attention_layer(SimModelState &state, size_t layer_index, size_t position) {
    SimLayer &layer = state.layers[layer_index];

    rms_norm(state.x.data(), *layer.cpu_attention_norm, 1.0e-5f, state.normalized.data());
    matvec_add_q8_0(*layer.cpu_query_weight, *layer.cpu_query_bias, state.normalized.data(), state.query.data());
    matvec_add_q8_0(*layer.cpu_key_weight, *layer.cpu_key_bias, state.normalized.data(), state.key.data());
    matvec_add_q8_0(*layer.cpu_value_weight, *layer.cpu_value_bias, state.normalized.data(), state.value.data());
    sim_apply_rope(state.rope_cosines.data(), state.rope_sines.data(), state.query.data(), TT_GPT_OSS_QUERY_HEADS,
                   position);
    sim_apply_rope(state.rope_cosines.data(), state.rope_sines.data(), state.key.data(), TT_GPT_OSS_KV_HEADS, position);
    std::memcpy(layer.key_cache.data() + position * gpt_oss_kv_width, state.key.data(),
                gpt_oss_kv_width * sizeof(float));
    std::memcpy(layer.value_cache.data() + position * gpt_oss_kv_width, state.value.data(),
                gpt_oss_kv_width * sizeof(float));
    sim_attention_from_cache(layer, layer_index, position, state.query.data(), state.attended.data(),
                             state.attention_scores, true);
    matvec_add_q8_0(*layer.cpu_attention_output_weight, *layer.cpu_attention_output_bias, state.attended.data(),
                    state.projected.data());
    for (size_t j = 0; j < TT_GPT_OSS_DIM; ++j) {
        state.residual[j] = state.x[j] + state.projected[j];
    }

    rms_norm(state.residual.data(), *layer.cpu_post_attention_norm, 1.0e-5f, state.normalized.data());
}

struct CompareMetrics {
    double rms_a = 0.0;
    double rms_b = 0.0;
    double rms_error = 0.0;
    double relative_rms_error = 0.0;
    double cosine = 0.0;
    float max_abs_error = 0.0f;
    size_t max_index = 0;
};

CompareMetrics compare_vectors(const float *a, const float *b, size_t count) {
    CompareMetrics metrics;
    double sum_a2 = 0.0;
    double sum_b2 = 0.0;
    double sum_error2 = 0.0;
    double dot_ab = 0.0;

    for (size_t i = 0; i < count; ++i) {
        const double av = double(a[i]);
        const double bv = double(b[i]);
        const double error = av - bv;
        const float abs_error = std::fabs(a[i] - b[i]);
        sum_a2 += av * av;
        sum_b2 += bv * bv;
        sum_error2 += error * error;
        dot_ab += av * bv;
        if (abs_error > metrics.max_abs_error) {
            metrics.max_abs_error = abs_error;
            metrics.max_index = i;
        }
    }

    metrics.rms_a = std::sqrt(sum_a2 / double(count));
    metrics.rms_b = std::sqrt(sum_b2 / double(count));
    metrics.rms_error = std::sqrt(sum_error2 / double(count));
    metrics.relative_rms_error = metrics.rms_error / std::max(1.0e-30, metrics.rms_b);
    metrics.cosine = dot_ab / std::sqrt(std::max(1.0e-30, sum_a2 * sum_b2));
    return metrics;
}

struct DeviceProxyCheckStats {
    size_t layers = 0;
    size_t expert_overlap_sum = 0;
    size_t expert_top1_matches = 0;
    double residual_rel_rms_sum = 0.0;
    double residual_cosine_min = 1.0;
    double norm_rel_rms_sum = 0.0;
    double norm_cosine_min = 1.0;
    double rms_approx_rel_rms_sum = 0.0;
    double rms_approx_cosine_min = 1.0;
    double moe_rel_rms_sum = 0.0;
    double moe_cosine_min = 1.0;
};

void proxy_moe(const SimLayer &layer, SimModelState &state, const float *input, float *output);
void proxy_moe_from_residual(const SimLayer &layer, SimModelState &state, const float *residual, float *output);
void proxy_moe_from_attention_output(const SimLayer &layer, SimModelState &state, const float *x,
                                     const float *projected, float *output);

size_t expert_overlap(const std::vector<size_t> &a, const std::vector<size_t> &b) {
    size_t result = 0;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (a[i] == b[j]) {
                ++result;
            }
        }
    }
    return result;
}

void add_check_metrics(const float *proxy, const float *cpu, size_t count, double *rel_rms_sum, double *cosine_min) {
    const CompareMetrics metrics = compare_vectors(proxy, cpu, count);
    *rel_rms_sum += metrics.relative_rms_error;
    *cosine_min = std::min(*cosine_min, metrics.cosine);
}

CompareMetrics proxy_cpu_forward_token_check(SimModelState &proxy_state, SimModelState &cpu_state, int32_t token,
                                             size_t position, size_t layer_limit, DeviceProxyCheckStats &stats) {
    if (token < 0 || size_t(token) >= 201088) {
        throw std::runtime_error("device proxy check token is out of range");
    }
    load_token_embedding_bf16(*proxy_state.embeddings, size_t(token), proxy_state.x.data());
    load_token_embedding_q8_0(*cpu_state.cpu_embeddings, size_t(token), cpu_state.x.data());

    for (size_t i = 0; i < layer_limit; ++i) {
        SimLayer &proxy_layer = proxy_state.layers[i];
        SimLayer &cpu_layer = cpu_state.layers[i];

        proxy_rms_norm_bf16(proxy_state.x.data(), *proxy_layer.attention_norm, proxy_state.normalized.data());
        reference_rms_norm_bf16(proxy_state.x.data(), *proxy_layer.attention_norm, proxy_state.down.data());
        add_check_metrics(proxy_state.normalized.data(), proxy_state.down.data(), TT_GPT_OSS_DIM,
                          &stats.rms_approx_rel_rms_sum, &stats.rms_approx_cosine_min);
        proxy_bf16_matvec_f32(*proxy_layer.query_weight, proxy_layer.query_bias_f32, proxy_state.normalized.data(),
                              proxy_state.proxy_b_values.data(), proxy_state.query.data());
        proxy_bf16_matvec_f32(*proxy_layer.key_weight, proxy_layer.key_bias_f32, proxy_state.normalized.data(),
                              proxy_state.proxy_b_values.data(), proxy_state.key.data());
        proxy_bf16_matvec(*proxy_layer.value_weight, proxy_layer.value_bias, proxy_state.normalized.data(),
                          proxy_state.proxy_b_values.data(), proxy_state.value.data());
        proxy_apply_rope_sfpu(proxy_state.rope_cosines.data(), proxy_state.rope_sines.data(), proxy_state.query.data(),
                              TT_GPT_OSS_QUERY_HEADS, position);
        proxy_apply_rope_sfpu(proxy_state.rope_cosines.data(), proxy_state.rope_sines.data(), proxy_state.key.data(),
                              TT_GPT_OSS_KV_HEADS, position);
        std::memcpy(proxy_layer.key_cache.data() + position * gpt_oss_kv_width, proxy_state.key.data(),
                    gpt_oss_kv_width * sizeof(float));
        std::memcpy(proxy_layer.value_cache.data() + position * gpt_oss_kv_width, proxy_state.value.data(),
                    gpt_oss_kv_width * sizeof(float));
        proxy_attention_rv32f(proxy_layer, i, position, proxy_state.query.data(), proxy_state.attention_scores.data(),
                              proxy_state.attended.data());
        proxy_bf16_matvec_f32(*proxy_layer.attention_output_weight, proxy_layer.attention_output_bias_f32,
                              proxy_state.attended.data(), proxy_state.proxy_b_values.data(),
                              proxy_state.projected.data());
        for (size_t j = 0; j < TT_GPT_OSS_DIM; ++j) {
            proxy_state.residual[j] = proxy_make_residual_value(proxy_state.x[j], proxy_state.projected[j]);
        }

        cpu_attention_layer(cpu_state, i, position);
        add_check_metrics(proxy_state.residual.data(), cpu_state.residual.data(), TT_GPT_OSS_DIM,
                          &stats.residual_rel_rms_sum, &stats.residual_cosine_min);

        proxy_rms_norm_bf16(proxy_state.residual.data(), *proxy_layer.post_attention_norm,
                            proxy_state.normalized.data());
        reference_rms_norm_bf16(proxy_state.residual.data(), *proxy_layer.post_attention_norm, proxy_state.down.data());
        add_check_metrics(proxy_state.normalized.data(), proxy_state.down.data(), TT_GPT_OSS_DIM,
                          &stats.rms_approx_rel_rms_sum, &stats.rms_approx_cosine_min);
        add_check_metrics(proxy_state.normalized.data(), cpu_state.normalized.data(), TT_GPT_OSS_DIM,
                          &stats.norm_rel_rms_sum, &stats.norm_cosine_min);

        proxy_moe(proxy_layer, proxy_state, proxy_state.normalized.data(), proxy_state.moe_output.data());
        cpu_moe(cpu_layer, cpu_state, cpu_state.normalized.data(), cpu_state.moe_output.data());
        stats.expert_overlap_sum += expert_overlap(proxy_state.expert_indices, cpu_state.expert_indices);
        if (proxy_state.expert_indices[0] == cpu_state.expert_indices[0]) {
            ++stats.expert_top1_matches;
        }
        add_check_metrics(proxy_state.moe_output.data(), cpu_state.moe_output.data(), TT_GPT_OSS_DIM,
                          &stats.moe_rel_rms_sum, &stats.moe_cosine_min);

        for (size_t j = 0; j < TT_GPT_OSS_DIM; ++j) {
            proxy_state.x[j] = proxy_add_residual_value(proxy_state.moe_output[j], proxy_state.residual[j]);
            cpu_state.x[j] = cpu_state.residual[j] + cpu_state.moe_output[j];
        }
        ++stats.layers;
    }

    return compare_vectors(proxy_state.x.data(), cpu_state.x.data(), TT_GPT_OSS_DIM);
}

void require_exact_match(const char *label, const float *sim, const float *proxy, size_t count, size_t position,
                         size_t layer_index) {
    for (size_t i = 0; i < count; ++i) {
        if (std::bit_cast<uint32_t>(sim[i]) != std::bit_cast<uint32_t>(proxy[i])) {
            const CompareMetrics metrics = compare_vectors(sim, proxy, count);
            char message[256];

            std::snprintf(message, sizeof(message),
                          "sim/device-proxy mismatch at token %zu layer %zu %s: first %zu sim %.9g proxy %.9g "
                          "rel RMS %.6g max abs %.6g at %zu",
                          position, layer_index, label, i, double(sim[i]), double(proxy[i]), metrics.relative_rms_error,
                          double(metrics.max_abs_error), metrics.max_index);
            throw std::runtime_error(message);
        }
    }
}

void run_firmware_proxy_check(TtFirmwareBackend backend, const char *sim_path, const char *requant_path,
                              const GgufFile &file, int32_t token, size_t generation_limit, size_t layer_limit,
                              TtMeshConfig mesh) {
    if (!requant_path) {
        throw std::runtime_error("TT firmware check requires --ttq");
    }
    if (generation_limit == 0) {
        throw std::runtime_error("TT firmware check needs at least one token command");
    }
    if (layer_limit == 0) {
        layer_limit = size_t(file.integer("gpt-oss.block_count"));
    }
    if (layer_limit > file.integer("gpt-oss.block_count")) {
        throw std::runtime_error("--debug-layers is too large");
    }
    validate_firmware_model(file, mesh);
    validate_mesh_config(mesh);
    TtqFile requant(requant_path);
    requant.validate_model(file);

    TtFirmwareRuntime runtime;
    open_tt_firmware_runtime(runtime, backend, sim_path, mesh);
    try {
        write_mesh_config_to_l1(runtime.api, mesh);

        DramCursor dram_cursor(mesh);
        preload_ttq_tensors_to_dram(runtime.api, requant, file, dram_cursor, mesh);

        SimModelState state;
        init_sim_state(file, requant, generation_limit, state, false);
        SimModelState proxy_state;
        init_sim_state(file, requant, generation_limit, proxy_state);
        proxy_state.norm_shards = mesh.chip_count() > 1 ? 1 : mesh.local_shard_count();
        allocate_sim_kv_cache(dram_cursor, state, generation_limit, layer_limit, mesh);
        write_layer_descriptors_to_l1(runtime.api, mesh, state, layer_limit);
        allocate_sim_attention_scores(dram_cursor, state, generation_limit, mesh);
        report_dram_usage(dram_cursor);
        enable_host_dma(state, mesh, backend == TtFirmwareBackend::silicon);
        write_token_config(runtime.api, mesh, state, layer_limit);

        double total_command_seconds = 0.0;
        double total_proxy_logits_seconds = 0.0;
        int32_t current_token = token;
        const bool is_silicon = backend == TtFirmwareBackend::silicon;
        const char *label = is_silicon ? "silicon" : "sim";

        start_brisc_firmware(runtime.api, mesh);
        for (size_t position = 0; position < generation_limit; ++position) {
            const auto command_begin = std::chrono::steady_clock::now();

            device_token(runtime.api, mesh, state, current_token, position, layer_limit, true, is_silicon);
            const auto command_end = std::chrono::steady_clock::now();
            const double command_seconds = seconds_since(command_begin, command_end);
            std::cout << label << ' ' << layer_limit << "-layer token " << (position + 1u) << '/' << generation_limit
                      << " command " << command_seconds << " s, next token " << current_token << '\n';

            proxy_forward_token(proxy_state, current_token, position, layer_limit);
            const double proxy_logits_seconds = proxy_finish_logits(proxy_state);

            require_exact_match("logits", state.logits.data(), proxy_state.logits.data(), TT_GPT_OSS_VOCAB, position,
                                layer_limit);
            current_token = sim_argmax(state.logits.data(), state.logits.size());
            total_command_seconds += command_seconds;
            total_proxy_logits_seconds += proxy_logits_seconds;
            std::cout << "  logits exact\n";
        }
        stop_brisc_firmware(runtime.api, mesh);
        std::cout << label << " logits exact match: " << generation_limit << " tokens, " << TT_GPT_OSS_VOCAB
                  << " values each\n";
        std::cout << label << " token commands total " << total_command_seconds << " s, avg "
                  << (total_command_seconds / double(generation_limit)) << " s/token\n";
        std::cout << "proxy logits total " << total_proxy_logits_seconds << " s\n";
        report_profile(state.profile);
        close_tt_firmware_runtime(runtime);
    } catch (...) {
        abort_tt_firmware_runtime(runtime, mesh);
        throw;
    }
}

void proxy_moe(const SimLayer &layer, SimModelState &state, const float *input, float *output) {
    proxy_bf16_matvec_f32(*layer.router_weight, layer.router_bias_f32, input, state.proxy_b_values.data(),
                          state.router.data());
    proxy_select_top4(state.router, state.expert_indices, state.expert_weights);

    std::fill(output, output + TT_GPT_OSS_DIM, 0.0f);
    for (size_t selected = 0; selected < 4; ++selected) {
        const size_t expert = state.expert_indices[selected];
        proxy_bfp8_matvec_2880(*layer.gate_weight, *layer.gate_bias, input, expert, state.proxy_b_values.data(),
                               state.gate.data());
        proxy_bfp8_matvec_2880(*layer.up_weight, *layer.up_bias, input, expert, state.proxy_b_values.data(),
                               state.up.data());
        proxy_swiglu(state.gate.data(), state.up.data(), state.hidden.data());
        proxy_bfp8_matvec_bf16_2880(*layer.down_weight, *layer.down_bias, state.hidden.data(), expert,
                                    state.proxy_b_values.data(), state.down4.data() + selected * TT_GPT_OSS_DIM);
    }
    proxy_moe_reduce_down4(state.down4.data(), state.expert_weights, output);
}

void proxy_moe_from_residual(const SimLayer &layer, SimModelState &state, const float *residual, float *output) {
    proxy_rms_norm_bf16(residual, *layer.post_attention_norm, state.normalized.data(), state.norm_shards);
    proxy_moe(layer, state, state.normalized.data(), output);
}

void proxy_moe_from_attention_output(const SimLayer &layer, SimModelState &state, const float *x,
                                     const float *projected, float *output) {
    for (size_t j = 0; j < TT_GPT_OSS_DIM; ++j) {
        state.residual[j] = proxy_make_residual_value(x[j], projected[j]);
    }
    proxy_moe_from_residual(layer, state, state.residual.data(), output);
}

void proxy_forward_token(SimModelState &state, int32_t token, size_t position, size_t layer_limit) {
    if (token < 0 || size_t(token) >= 201088) {
        throw std::runtime_error("device proxy token is out of range");
    }
    load_token_embedding_bf16(*state.embeddings, size_t(token), state.x.data());

    for (size_t i = 0; i < layer_limit; ++i) {
        SimLayer &layer = state.layers[i];

        proxy_rms_norm_bf16(state.x.data(), *layer.attention_norm, state.normalized.data(), state.norm_shards);
        proxy_bf16_matvec_f32(*layer.query_weight, layer.query_bias_f32, state.normalized.data(),
                              state.proxy_b_values.data(), state.query.data());
        proxy_bf16_matvec_f32(*layer.key_weight, layer.key_bias_f32, state.normalized.data(),
                              state.proxy_b_values.data(), state.key.data());
        proxy_bf16_matvec(*layer.value_weight, layer.value_bias, state.normalized.data(), state.proxy_b_values.data(),
                          state.value.data());
        proxy_apply_rope_sfpu(state.rope_cosines.data(), state.rope_sines.data(), state.query.data(),
                              TT_GPT_OSS_QUERY_HEADS, position);
        proxy_apply_rope_sfpu(state.rope_cosines.data(), state.rope_sines.data(), state.key.data(), TT_GPT_OSS_KV_HEADS,
                              position);
        std::memcpy(layer.key_cache.data() + position * gpt_oss_kv_width, state.key.data(),
                    gpt_oss_kv_width * sizeof(float));
        std::memcpy(layer.value_cache.data() + position * gpt_oss_kv_width, state.value.data(),
                    gpt_oss_kv_width * sizeof(float));
        proxy_attention_rv32f(layer, i, position, state.query.data(), state.attention_scores.data(),
                              state.attended.data());
        proxy_bf16_matvec_f32(*layer.attention_output_weight, layer.attention_output_bias_f32, state.attended.data(),
                              state.proxy_b_values.data(), state.projected.data());
        proxy_moe_from_attention_output(layer, state, state.x.data(), state.projected.data(), state.moe_output.data());
        for (size_t j = 0; j < TT_GPT_OSS_DIM; ++j) {
            state.x[j] = proxy_add_residual_value(state.moe_output[j], state.residual[j]);
        }
    }
}

int32_t sim_argmax(const float *values, size_t count) {
    size_t best = 0;
    for (size_t i = 1; i < count; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return int32_t(best);
}

std::vector<size_t> top_indices(const float *values, size_t count, size_t top_count) {
    std::vector<size_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = i;
    }
    std::partial_sort(indices.begin(), indices.begin() + long(top_count), indices.end(),
                      [&](size_t a, size_t b) { return values[a] > values[b]; });
    indices.resize(top_count);
    return indices;
}

size_t rank_of(const float *values, size_t count, size_t token) {
    size_t rank = 1;
    const float target = values[token];
    for (size_t i = 0; i < count; ++i) {
        if (values[i] > target) {
            ++rank;
        }
    }
    return rank;
}

size_t top_overlap(const std::vector<size_t> &a, const std::vector<size_t> &b, size_t count) {
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < count; ++j) {
            if (a[i] == b[j]) {
                ++result;
            }
        }
    }
    return result;
}

double softmax_kl_on_union(const float *p_logits, const float *q_logits, const std::vector<size_t> &p_top,
                           const std::vector<size_t> &q_top, size_t count) {
    std::vector<size_t> tokens;
    tokens.reserve(count * 2);
    for (size_t i = 0; i < count; ++i) {
        tokens.push_back(p_top[i]);
        bool seen = false;
        for (size_t j = 0; j < tokens.size(); ++j) {
            if (tokens[j] == q_top[i]) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            tokens.push_back(q_top[i]);
        }
    }

    float p_max = -std::numeric_limits<float>::infinity();
    float q_max = -std::numeric_limits<float>::infinity();
    for (size_t token : tokens) {
        p_max = std::max(p_max, p_logits[token]);
        q_max = std::max(q_max, q_logits[token]);
    }
    double p_sum = 0.0;
    double q_sum = 0.0;
    for (size_t token : tokens) {
        p_sum += std::exp(double(p_logits[token] - p_max));
        q_sum += std::exp(double(q_logits[token] - q_max));
    }

    double kl = 0.0;
    for (size_t token : tokens) {
        const double p = std::exp(double(p_logits[token] - p_max)) / p_sum;
        const double q = std::exp(double(q_logits[token] - q_max)) / q_sum;
        kl += p * std::log(p / std::max(q, 1.0e-300));
    }
    return kl;
}

void print_logit_diagnostics(const char *a_label, const float *a_logits, const char *b_label, const float *b_logits,
                             size_t count) {
    constexpr size_t top_count = 20;
    const std::vector<size_t> a_top = top_indices(a_logits, count, top_count);
    const std::vector<size_t> b_top = top_indices(b_logits, count, top_count);
    const size_t a_best = a_top[0];
    const size_t b_best = b_top[0];
    const CompareMetrics metrics = compare_vectors(a_logits, b_logits, count);

    double a_mean = 0.0;
    double b_mean = 0.0;
    for (size_t i = 0; i < count; ++i) {
        a_mean += a_logits[i];
        b_mean += b_logits[i];
    }
    a_mean /= double(count);
    b_mean /= double(count);

    double centered_dot = 0.0;
    double centered_a2 = 0.0;
    double centered_b2 = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double a = double(a_logits[i]) - a_mean;
        const double b = double(b_logits[i]) - b_mean;
        centered_dot += a * b;
        centered_a2 += a * a;
        centered_b2 += b * b;
    }
    const double pearson = centered_dot / std::sqrt(std::max(1.0e-30, centered_a2 * centered_b2));

    std::printf("logit distribution: cosine %.6f, pearson %.6f, rel RMS %.6g, RMS error %.6g, max abs %.6g at %zu\n",
                metrics.cosine, pearson, metrics.relative_rms_error, metrics.rms_error, double(metrics.max_abs_error),
                metrics.max_index);
    std::printf("top overlap: @5 %zu/5, @10 %zu/10, @20 %zu/20\n", top_overlap(a_top, b_top, 5),
                top_overlap(a_top, b_top, 10), top_overlap(a_top, b_top, 20));
    std::printf("reciprocal ranks: %s top token %zu is %s rank %zu; %s top token %zu is %s rank %zu\n", b_label, b_best,
                a_label, rank_of(a_logits, count, b_best), a_label, a_best, b_label, rank_of(b_logits, count, a_best));
    std::printf("KL on top-20 union: %s||%s %.6g, %s||%s %.6g\n", b_label, a_label,
                softmax_kl_on_union(b_logits, a_logits, b_top, a_top, top_count), a_label, b_label,
                softmax_kl_on_union(a_logits, b_logits, a_top, b_top, top_count));
    std::printf("top 10 logits:\n");
    for (size_t i = 0; i < 10; ++i) {
        std::printf("  rank %2zu: %s token %6zu logit %9.5g | %s token %6zu logit %9.5g\n", i, b_label, b_top[i],
                    double(b_logits[b_top[i]]), a_label, a_top[i], double(a_logits[a_top[i]]));
    }
}

double proxy_finish_logits(SimModelState &state) {
    const auto begin = std::chrono::steady_clock::now();
    proxy_rms_norm_bf16(state.x.data(), *state.output_norm, state.normalized.data(), state.norm_shards);
    proxy_bf16_matvec(*state.output, nullptr, state.normalized.data(), state.proxy_b_values.data(),
                      state.logits.data());
    const auto end = std::chrono::steady_clock::now();
    return seconds_since(begin, end);
}

void cpu_finish_logits(SimModelState &state) {
    rms_norm(state.x.data(), *state.cpu_output_norm, 1.0e-5f, state.normalized.data());
    matvec_q8_0(*state.cpu_output, state.normalized.data(), state.logits.data());
}

void run_firmware_generation(TtSimApi &api, TtqFile &requant, DramCursor &dram_cursor, const GgufFile &file,
                             const Tokenizer &tokenizer, const std::string &formatted_prompt,
                             const int32_t *prompt_tokens, size_t prompt_count, size_t generation_limit,
                             TtMeshConfig mesh, bool is_silicon) {
    const size_t layer_count = requant.layer_count();
    const size_t token_capacity = prompt_count + generation_limit;

    SimModelState state;
    init_sim_state(file, requant, token_capacity, state, false);
    allocate_sim_kv_cache(dram_cursor, state, token_capacity, layer_count, mesh);
    write_layer_descriptors_to_l1(api, mesh, state, layer_count);
    allocate_sim_attention_scores(dram_cursor, state, token_capacity, mesh);
    report_dram_usage(dram_cursor);
    enable_host_dma(state, mesh, is_silicon);
    write_token_config(api, mesh, state, layer_count);

    const int32_t eos = int32_t(file.integer("tokenizer.ggml.eos_token_id"));
    std::array<char, 8192> decoded;
    size_t decoded_tokens = 0;
    size_t decode_evaluations = 0;
    double decode_seconds = 0.0;

    start_brisc_firmware(api, mesh);
    const auto prefill_begin = std::chrono::steady_clock::now();
    for (size_t position = 0; position < prompt_count; ++position) {
        const bool compute_logits = position + 1 == prompt_count;
        device_token(api, mesh, state, prompt_tokens[position], position, layer_count, compute_logits, is_silicon);
    }
    const auto prefill_end = std::chrono::steady_clock::now();

    int32_t current_top = sim_argmax(state.logits.data(), state.logits.size());
    std::cout << formatted_prompt << std::flush;
    for (size_t i = 0; i < generation_limit; ++i) {
        const int32_t next = current_top;
        if (next == eos) {
            break;
        }
        ++decoded_tokens;
        const size_t decoded_size = tokenizer.decode_to(next, decoded.data(), decoded.size(), true);
        std::fwrite(decoded.data(), 1, decoded_size, stdout);
        std::fflush(stdout);
        if (i + 1 < generation_limit) {
            const auto decode_begin = std::chrono::steady_clock::now();
            device_token(api, mesh, state, next, prompt_count + i, layer_count, true, is_silicon);
            current_top = sim_argmax(state.logits.data(), state.logits.size());
            const auto decode_end = std::chrono::steady_clock::now();
            decode_seconds += seconds_since(decode_begin, decode_end);
            ++decode_evaluations;
        }
    }
    std::fputc('\n', stdout);
    stop_brisc_firmware(api, mesh);

    const double prefill_seconds = seconds_since(prefill_begin, prefill_end);
    print_generation_stats(prompt_count, prefill_seconds, decoded_tokens, decode_evaluations, decode_seconds);
    report_profile(state.profile);
}

} // namespace

void tt_enable_profiling() {
    profiling_enabled = true;
}

void run_tt_firmware_inference(TtFirmwareBackend backend, const char *sim_path, const char *requant_path,
                               const GgufFile &file, const Tokenizer &tokenizer, const std::string &formatted_prompt,
                               const int32_t *prompt_tokens, size_t prompt_count, size_t generation_limit,
                               TtMeshConfig mesh) {
    if (prompt_count == 0) {
        throw std::runtime_error("TT firmware inference needs at least one prompt token");
    }
    if (!requant_path || !prompt_tokens) {
        throw std::runtime_error("TT firmware inference requires prompt tokens and --ttq");
    }
    validate_firmware_model(file, mesh);
    validate_mesh_config(mesh);
    TtqFile requant(requant_path);
    requant.validate_model(file);

    TtFirmwareRuntime runtime;
    open_tt_firmware_runtime(runtime, backend, sim_path, mesh);
    try {
        write_mesh_config_to_l1(runtime.api, mesh);
        DramCursor dram_cursor(mesh);
        preload_ttq_tensors_to_dram(runtime.api, requant, file, dram_cursor, mesh);

        run_firmware_generation(runtime.api, requant, dram_cursor, file, tokenizer, formatted_prompt, prompt_tokens,
                                prompt_count, generation_limit, mesh, backend == TtFirmwareBackend::silicon);
        close_tt_firmware_runtime(runtime);
    } catch (...) {
        abort_tt_firmware_runtime(runtime, mesh);
        throw;
    }
}

void run_tt_firmware_proxy_check(TtFirmwareBackend backend, const char *sim_path, const char *requant_path,
                                 const GgufFile &file, int32_t token, size_t generation_limit, size_t layer_limit,
                                 TtMeshConfig mesh) {
    run_firmware_proxy_check(backend, sim_path, requant_path, file, token, generation_limit, layer_limit, mesh);
}

void run_device_proxy_check(const char *requant_path, const GgufFile &file, const int32_t *prompt_tokens,
                            size_t prompt_count, size_t layer_limit) {
    if (prompt_count == 0) {
        throw std::runtime_error("device proxy check needs at least one prompt token");
    }
    TtqFile requant(requant_path);
    if (layer_limit == 0 || layer_limit > requant.layer_count()) {
        layer_limit = requant.layer_count();
    }

    SimModelState proxy_state;
    init_sim_state(file, requant, prompt_count, proxy_state);
    SimModelState cpu_state;
    init_sim_state(file, requant, prompt_count, cpu_state, true, true);

    double sum_x_rel_rms = 0.0;
    double max_x_rel_rms = 0.0;
    double min_x_cosine = 1.0;
    size_t worst_x_token = 0;
    DeviceProxyCheckStats stats;

    const auto begin = std::chrono::steady_clock::now();
    for (size_t position = 0; position < prompt_count; ++position) {
        const auto token_begin = std::chrono::steady_clock::now();
        const CompareMetrics metrics = proxy_cpu_forward_token_check(proxy_state, cpu_state, prompt_tokens[position],
                                                                     position, layer_limit, stats);
        const auto token_end = std::chrono::steady_clock::now();
        sum_x_rel_rms += metrics.relative_rms_error;
        if (metrics.relative_rms_error > max_x_rel_rms) {
            max_x_rel_rms = metrics.relative_rms_error;
            worst_x_token = position;
        }
        min_x_cosine = std::min(min_x_cosine, metrics.cosine);
        std::printf("device proxy check token %zu/%zu: %.3f s, x cosine %.6f, rel RMS %.6g, max abs %.6g at %zu\n",
                    position + 1, prompt_count, seconds_since(token_begin, token_end), metrics.cosine,
                    metrics.relative_rms_error, double(metrics.max_abs_error), metrics.max_index);
    }

    proxy_finish_logits(proxy_state);
    cpu_finish_logits(cpu_state);
    const auto end = std::chrono::steady_clock::now();

    const int32_t proxy_top = sim_argmax(proxy_state.logits.data(), proxy_state.logits.size());
    const int32_t cpu_top = sim_argmax(cpu_state.logits.data(), cpu_state.logits.size());
    std::printf("device proxy check complete: %zu tokens, layers %zu/%zu, %.3f s, %.3f s/token\n", prompt_count,
                layer_limit, requant.layer_count(), seconds_since(begin, end),
                seconds_since(begin, end) / double(prompt_count));
    std::printf("x summary: avg rel RMS %.6g, max rel RMS %.6g at token %zu, min cosine %.6f\n",
                sum_x_rel_rms / double(prompt_count), max_x_rel_rms, worst_x_token + 1, min_x_cosine);
    if (stats.layers) {
        std::printf("RMSNorm approximation summary: avg rel RMS %.6g min cosine %.6f vs exact BF16-weight RMSNorm "
                    "on same inputs\n",
                    stats.rms_approx_rel_rms_sum / double(2 * stats.layers), stats.rms_approx_cosine_min);
        std::printf("layer drift summary: residual avg rel RMS %.6g min cosine %.6f; norm avg rel RMS %.6g "
                    "min cosine %.6f; MoE avg rel RMS %.6g min cosine %.6f\n",
                    stats.residual_rel_rms_sum / double(stats.layers), stats.residual_cosine_min,
                    stats.norm_rel_rms_sum / double(stats.layers), stats.norm_cosine_min,
                    stats.moe_rel_rms_sum / double(stats.layers), stats.moe_cosine_min);
        std::printf("expert selection summary: avg top-4 overlap %.3f/4, top-1 match %.3f\n",
                    double(stats.expert_overlap_sum) / double(stats.layers),
                    double(stats.expert_top1_matches) / double(stats.layers));
    }
    std::printf("CPU top token %d logit %.6g; device proxy top token %d logit %.6g%s\n", cpu_top,
                double(cpu_state.logits[size_t(cpu_top)]), proxy_top, double(proxy_state.logits[size_t(proxy_top)]),
                cpu_top == proxy_top ? ", top matches" : ", top differs");
    print_logit_diagnostics("device proxy", proxy_state.logits.data(), "CPU", cpu_state.logits.data(),
                            proxy_state.logits.size());
}

void run_device_proxy_generation(const char *requant_path, const GgufFile &file, const Tokenizer &tokenizer,
                                 const std::string &formatted_prompt, const int32_t *prompt_tokens, size_t prompt_count,
                                 size_t generation_limit) {
    if (prompt_count == 0) {
        throw std::runtime_error("device proxy needs at least one prompt token");
    }
    TtqFile requant(requant_path);
    const size_t layer_count = requant.layer_count();

    const size_t token_capacity = prompt_count + generation_limit;
    SimModelState state;
    init_sim_state(file, requant, token_capacity, state);

    const int32_t eos = int32_t(file.integer("tokenizer.ggml.eos_token_id"));
    std::array<char, 8192> decoded;

    const auto prefill_begin = std::chrono::steady_clock::now();
    for (size_t position = 0; position < prompt_count; ++position) {
        proxy_forward_token(state, prompt_tokens[position], position, layer_count);
    }
    proxy_finish_logits(state);
    const auto prefill_end = std::chrono::steady_clock::now();
    int32_t current_top = sim_argmax(state.logits.data(), state.logits.size());

    size_t decoded_tokens = 0;
    size_t decode_evaluations = 0;
    double decode_seconds = 0.0;
    std::cout << formatted_prompt << std::flush;
    for (size_t i = 0; i < generation_limit; ++i) {
        const int32_t next = current_top;
        if (next == eos) {
            break;
        }
        ++decoded_tokens;
        const size_t decoded_size = tokenizer.decode_to(next, decoded.data(), decoded.size(), true);
        std::fwrite(decoded.data(), 1, decoded_size, stdout);
        std::fflush(stdout);
        if (i + 1 < generation_limit) {
            const auto decode_begin = std::chrono::steady_clock::now();
            proxy_forward_token(state, next, prompt_count + i, layer_count);
            proxy_finish_logits(state);
            current_top = sim_argmax(state.logits.data(), state.logits.size());
            const auto decode_end = std::chrono::steady_clock::now();
            decode_seconds += seconds_since(decode_begin, decode_end);
            ++decode_evaluations;
        }
    }
    std::fputc('\n', stdout);

    const double prefill_seconds = seconds_since(prefill_begin, prefill_end);
    print_generation_stats(prompt_count, prefill_seconds, decoded_tokens, decode_evaluations, decode_seconds);
}
