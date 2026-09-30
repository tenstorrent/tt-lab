# tt-lab

`tt-lab` runs gpt-oss-20b and gpt-oss-120b on Tenstorrent Blackhole hardware and provides the tools
to understand and improve their execution. It combines end-to-end inference, custom firmware, a
small compiler for SFPU (the Tensix Vector Unit) kernels, a
bit-accurate host-side device proxy, and CPU reference inference in one inspectable codebase.

The hardware path runs the transformer on Tensix tiles, including attention, expert matvecs,
normalization, and activations. Four-chip execution uses direct PCIe peer-to-peer communication,
not host-mediated tensor exchange. The host handles tokenization and greedy token selection.

| Model | Hardware | Worker tiles |
|---|---|---|
| gpt-oss-20b | One Blackhole chip | 1 or 8 |
| gpt-oss-20b or gpt-oss-120b | Four Blackhole chips | 32, eight per chip |

Eight workers are only a fraction of a Blackhole chip's Tensix tiles. The goal is to make the
mapping efficient and understandable, with explicit data movement and no framework or compute
library between the model and the hardware. Each worker runs one firmware instance on BRISC, one of
the RISC-V cores in each Tensix tile.

The same tool supports several development workflows:

- **Run models:** load MXFP4 (4-bit microscaling floating point) GGUF weights, create a
  hardware-layout `.ttq` sidecar, and generate text.
- **Check correctness:** compare every output logit from silicon or simulation bit-for-bit against
  the device proxy, independently of differences from CPU reference arithmetic.
- **Study numerics:** compare CPU and proxy intermediate tensors, layer drift, and final logits.
- **Tune kernels:** edit C++ firmware or SFPU domain-specific language (DSL) fragments and measure
  per-stage device cycles.
- **Work without hardware:** use CPU inference, the device proxy, or `ttsim`.

Current scope is Linux/little-endian x86-64, the supported gpt-oss MXFP4 GGUF layouts, one sequence,
and greedy generation. CPU prefill is batched; hardware prefill processes tokens serially.
This is an inference and hardware experimentation project, not a production serving framework:
there is no multi-sequence batching or serving endpoint. The host inference code is single-threaded
and uses no compute libraries; Python is needed for building, not for running the executable.

## Getting Started

### Building

Build prerequisites are Python 3.12+, GCC with GNU++20 support, GNU binutils, and the standard Linux
shell utilities. The executable has no Python runtime dependency. All executable builds, including
CPU-only use, currently compile and embed the Tenstorrent (TT) firmware and therefore require
[SFPI](https://github.com/tenstorrent/sfpi); the tested version is 7.76.0.

```sh
./make.py :build
```

All build output is written under `_out/`; the main binary is `_out/tt-lab`.

The build uses `g++` and GNU++20 and requires x86-64-v3 (including AVX2). CPU tensor kernels and
device-proxy matvecs automatically use AVX-512 on x86-64-v4 hosts.

Ubuntu 24.04's GCC 13 is tested; newer compilers can generate better AVX-512 code. The TT backend
firmware build expects SFPI under `~/sfpi-7.76.0` by default; set `SFPI_PATH` to override that
location.

### Models

`tt-lab` does not download models automatically. You give it a local `.gguf` file with
`-m`.

GGUF is a model file format used by small local LLM runners. In practical terms, a GGUF
file is the downloaded model: it contains the neural-network weights plus metadata such as
tensor names, shapes, tokenizer information, and model settings. The `MXFP4` part of these
filenames describes the model's weight format; you do not need to convert it before using
it with the CPU path.

Create a model directory and download one or both supported GGUF files:

```sh
mkdir -p ~/models

wget -O ~/models/gpt-oss-20b-MXFP4.gguf \
    https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf

wget -O ~/models/gpt-oss-120b-MXFP4.gguf \
    https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-MXFP4.gguf
```

The `20b` model is the better first test because it is smaller and faster. The `120b` model
is much larger and will need substantially more disk space, memory, and patience on the
current single-threaded CPU backend.

The README examples assume these local filenames:

```sh
~/models/gpt-oss-20b-MXFP4.gguf
~/models/gpt-oss-120b-MXFP4.gguf
~/models/gpt-oss-20b.ttq
~/models/gpt-oss-120b.ttq
```

You can verify either GGUF file without generating text:

```sh
_out/tt-lab inspect -m ~/models/gpt-oss-20b-MXFP4.gguf
_out/tt-lab inspect -m ~/models/gpt-oss-120b-MXFP4.gguf
```

### Running

On one Blackhole chip, after building and creating the `.ttq` sidecar (see
[Requant Sidecar](#requant-sidecar)):

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf \
    --device --tiles 8 --ttq ~/models/gpt-oss-20b.ttq -p "Implement strdup()" -n 256
```

For gpt-oss-120b on four chips, use the 120b GGUF/TTQ pair and `--tiles 32`.
The [backend section](#tt-blackhole-backend) covers device ownership and setup requirements.

For CPU reference inference, omit the device options:

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf -p "How do you like your steak?"
```

Generation is greedy. By default, `tt-lab run` runs until the end-of-sequence (EOS) token or until
the remaining context is full. Use `-n N` to set a smaller output-token limit:

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf \
    -p "What is 2+2? Answer briefly." -n 16
```

The program applies the gpt-oss Harmony prompt wrapper internally for inference. Harmony
channel markers and reasoning text are streamed as the model emits them. CPU prompt processing
uses layer-major batched prefill; TT paths process prompt tokens serially. Decode appends one
token at a time.

To inspect raw tokenization:

```sh
_out/tt-lab tokenize -m ~/models/gpt-oss-20b-MXFP4.gguf -p "Hello"
```

The commands are `run`, `inspect`, `requant`, and `tokenize`. The common options also have long
forms: `-m`/`--model`, `-p`/`--prompt`, `-o`/`--output` (`requant` only), and `-n`/`--n-predict`
(also `--predict`; `run` only). `-h` or `--help` prints the usage summary. The hardware and
simulator options are listed under [Backend Commands](#backend-commands).

## Requant Sidecar

Use `requant` to convert a GGUF file into a `tt-lab` sidecar file:

```sh
_out/tt-lab requant -m ~/models/gpt-oss-20b-MXFP4.gguf -o ~/models/gpt-oss-20b.ttq
_out/tt-lab requant -m ~/models/gpt-oss-120b-MXFP4.gguf -o ~/models/gpt-oss-120b.ttq
```

Requant writes a temporary sibling named `OUTPUT.tmp.XXXXXX`, then atomically replaces `OUTPUT`
only after all writes and close succeed. Input/output aliases (including hardlinks and symlinks)
are rejected. Replacing another symlink replaces the link itself, not its target. The new file is
owner-readable/writable only. Allow disk space for both an existing output and its replacement.
Failures clean up the temporary file; forced termination can leave a temporary sibling that can
be removed once the process has stopped. This does not promise durability across power loss.

The sidecar is not GGUF and is not a compatibility format. It is a simple local binary
container with a fixed `TTLBQNT1` header, a fixed-size tensor directory, 64-byte-aligned
payloads, and tensor records for names, formats, shapes, offsets, and sizes. There is no
runtime repacking in simulator/proxy modes.

Current payload formats are:

- Q8_0 matrix tensors converted to BF16 (bfloat16) Tensix 16x16 tiles.
- `token_embd.weight` converted to linear BF16.
- F32 router weights converted to BF16 Tensix 16x16 tiles.
- Other F32 tensors, biases, norms, attention sinks, and scalar tensors converted to
  linear BF16.
- MXFP4 gate/up/down expert tensors stored only in BFP8 (8-bit block floating point) Tensix strips.
  Each strip is a `2880x16` block, equivalent to `16x16x180`, with all 2880 shared exponent bytes
  first, followed by the packed datum bytes. Each 16x16 tile is row-major: each exponent covers 16
  consecutive input columns within one output row, matching the native MXFP4 axis. The proxy reads
  these tiles transposed into SrcA order. BFP8 records have a `.bfp8` name suffix.

The current backend uses BFP8 for all expert matrices, with no format-selection knob.
Expert format ID 5 distinguishes the new axis from the old format ID 4; old sidecars must be
regenerated. The `.ttq` filename itself has no format significance. Sim/device use the unpacker's
16x16 SrcA transpose to consume the same layout as the proxy.
Superseded BF16 biases do not consume device DRAM. Q/K, attention-output,
and router biases are loaded directly from the GGUF in FP32; V and expert biases remain BF16.

The BFP8 packer is optimized for what the Tensix unpack path reconstructs; it is not
intended to match another packer bit-for-bit. The quantizer rounds stored
magnitudes, clamps to seven magnitude bits, and chooses between the section maximum
exponent and one smaller exponent by reconstructed squared error.

## TT Blackhole Backend

Supported hardware execution:

- gpt-oss-20b on one Blackhole chip with `--tiles 1` or `--tiles 8` (default: one);
- either model on four Blackhole chips with `--tiles 32`, using PCIe peer-to-peer DMA (required for
  120b);
- shared C++20 BRISC firmware (`src/brisc.cpp`), with one BRISC per tile;
- host tokenization;
- one sequence, greedy generation;
- sharded matvecs and attention, with local NOC (network-on-chip) all-gathers and cross-chip PCIe
  exchange.

Four-chip runs place each chip's workers in its active PCIe tile's physical column, using the
chip's coordinate translation tables to account for harvesting. If that Tensix column is
unavailable, startup warns and selects the first active translated column instead. Worker rows
and DRAM ownership stay unchanged. Single-chip runs retain their existing placement.

The simulator backend and host-side device proxy both require the offline requantized
`.ttq` sidecar. `--sim`, `--device`, `--device-proxy`, and `--check` require `--ttq`.

The simulator path loads active weight payloads into simulated Blackhole DRAM once at startup; all
device matvecs then stream weights from simulated DRAM rather than copying weights from the
host on demand. The device-proxy path mmaps the same `.ttq` and runs the same device
arithmetic model directly on the host, so it is the fast oracle for simulator/device
numerics. Simulator and silicon outputs must match the device proxy bit-for-bit.

`--device-proxy` supports both models with a matching `.ttq`. Standalone proxy runs use
single-shard arithmetic; `--sim/--device --check` matches the selected mapping's RMSNorm
reduction order (eight partial sums for `--tiles 8`, one full sum for `--tiles 1/32`).

Use dedicated, idle devices: `--device` changes power state, resets worker RISCs, and overwrites
device storage. Do not run it concurrently with another tt-lab instance or another hardware runtime
such as TT-Metal. The program acquires exclusive ownership through the Tenstorrent kernel-mode
driver (KMD) for each device and fails immediately if it is busy. A driver supporting `O_EXCL`
device opens is required; startup verifies support rather than silently proceeding unlocked.
Ownership lasts until the device file descriptors are released, including on process exit.

### Hardware Setup

Install the Tenstorrent KMD and board firmware appropriate to the machine. The account running
`tt-lab` needs read/write access to the selected `/dev/tenstorrent/<n>` nodes and permission to
use their ioctls and mmap regions. Use the driver's udev rules or an administrator-managed group;
running inference as root is not required on the tested setup. Containers and sandboxes must also
expose the device nodes and allow access to them.

Device selection is currently automatic: numeric names under `/dev/tenstorrent` are sorted
numerically, then the first one (`--tiles 1/8`) or first four (`--tiles 32`) are opened. There is no
device-selection flag, and busy or unsupported devices are not skipped. Each selected device must be
Blackhole. A numeric device ID is not necessarily a physical board number or a UMD (user-mode
driver) chip ID. Check the mapping before running on a machine with other accelerators or users:

```sh
ls -l /dev/tenstorrent /dev/tenstorrent/by-id
tt-smi --offline -ls
cat /sys/module/tenstorrent/version
tt-smi --offline -s --snapshot_no_tty
```

These `tt-smi` commands were checked with version 6.1.0; older versions may expose different
options. The listing associates PCI bus/device/function (BDF) addresses and board information with
device nodes. Run management tools before inference, not concurrently: their open device handles can
conflict with exclusive ownership. `tt-smi` is useful for setup and diagnostics but is not a runtime
dependency of `tt-lab`.

The driver must implement `O_EXCL` ownership, `PIN_PAGES`, and, for four chips, `MAP_PEER_BAR`.
Exclusive-open support was added by KMD commit `3d5abc9f8a916bacc761a42cdd194e1ce0b3045c`.
Do not rely on the `2.10.1-pre` version string alone: development builds with that string can
differ, and `tt-lab` checks that the ownership mechanism actually works.

Host DMA is mandatory, with no unpinned fallback. Every silicon run pins about 396 KiB for
returned logits. Four-chip runs additionally pin one shared host embedding table, about 1.08 GiB,
and map it for all four chips. The current KMD requires a contiguous DMA address range for each
buffer. A translating IOMMU supplies this on the tested machine; without translation, the KMD
requires physically contiguous pages, which ordinary host allocations do not guarantee.
Do not assume IOMMU passthrough or disabling the IOMMU is equivalent. This path uses KMD page
pinning, not `mlock`; changing `ulimit -l` alone does not resolve a fragmented DMA mapping.
For `cannot pin host buffer for NOC DMA`, inspect kernel driver diagnostics and the IOMMU setup.
No preallocated hugepages are used by this implementation.

Four-chip inference requires working PCIe peer-to-peer reads and writes between every pair of
selected chips. It maps peer BAR0 (PCIe base address register 0) through KMD and uses the returned
DMA addresses directly; there is no Ethernet or host-copy fallback. The host PCIe routing, isolation
policy, and IOMMU configuration must permit those transactions. Successful enumeration or BAR
mapping alone is not an end-to-end P2P test. The four-chip `--check` command below exercises the
actual path; arbitrary four-card host topologies have not been validated. Do not disable platform
isolation features blindly to work around a failure.

The development setup observed on 2026-09-17 is below. These are tested versions, not established
minimum requirements or a claim that every firmware/driver combination is interchangeable.

| Component | Development configuration |
|---|---|
| Hardware | TT QuietBox2, two P300 boards (`p300c`), four Blackhole chips |
| Host | Ryzen 7 9700X, 256 GiB installed RAM, Ubuntu 24.04.4 LTS |
| Kernel / KMD | `7.0.0-31-generic` / `2.10.1-pre` with exclusive-open support |
| IOMMU | Translating DMA domains (`DMA-FQ`), not passthrough |
| Firmware bundle | `19.4.1.0`, as reported by tt-smi on all four chips |
| Firmware components | Chip management (CM) `0.26.1.0`, device management (DM) application `0.20.1.0`, GDDR `2.11` |
| SFPI | `7.76.0`, compiled for `tt-bh`; see [Building](#building) |

Allow roughly 34 GiB of disk for the 20b GGUF plus BFP8 TTQ, or 177 GiB for the 120b pair. Requant
replacement temporarily needs another full TTQ's worth of space. Files are memory-mapped, not wholly
pinned, but CPU/proxy runs and `--check` benefit from enough host RAM to cache both files, plus
context-dependent key/value (KV) caches and working buffers. Silicon-only inference does not require
both files to remain resident after loading. The 256 GiB development host is not a measured minimum;
no universal minimum host-RAM requirement has been established.

### Backend Commands

For a four-chip silicon correctness check:

```sh
_out/tt-lab run -m ~/models/gpt-oss-120b-MXFP4.gguf \
    --device --tiles 32 --ttq ~/models/gpt-oss-120b.ttq -p "Hi" -n 5 --check
```

For four-chip simulation, replace `--device` with
`--sim ~/ttsim/src/_out/release_bh_x4/libttsim.so`. The 120b layout is close to the DRAM
capacity limit; requested KV/score storage must fit on every channel or startup fails.

Example simulator run:

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf \
    --sim ~/ttsim/src/_out/release_bh/libttsim.so \
    --ttq ~/models/gpt-oss-20b.ttq \
    -p "Hi" --tiles 8 -n 1 --check
```

Example device-proxy run:

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf \
    --device-proxy --ttq ~/models/gpt-oss-20b.ttq \
    -p "What is 2+2? Answer briefly." -n 32
```

Example device-proxy-vs-CPU drift check:

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf \
    --device-proxy --check --ttq ~/models/gpt-oss-20b.ttq \
    -p "How do you like your steak?"
```

Example silicon run:

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf \
    --device --ttq ~/models/gpt-oss-20b.ttq \
    -p "Hi" --tiles 8 -n 32
```

Supported Blackhole-path options:

- `--sim PATH`: load `libttsim.so` and run the BRISC/Tensix simulator path for generation.
- `--device`: run generation on the first one or four enumerated `/dev/tenstorrent` devices.
- `--tiles 1|8|32`: select 1/8 tiles for 20b or 32 tiles across four chips for either model;
  defaults to 1.
- `--profile`: report per-token stage cycles on shard 0 and host setup/command/logits-conversion
  timings with `--sim` or `--device`, including `--check`. Stages include waits and overlap;
  they are not isolated kernel timings. Totals include prefill and decode. Device cycles use
  `WALL_CLOCK_0`; simulator cycles are not silicon performance predictions.
- `--ttq PATH`: load the offline requantized tensor file. Required by all Blackhole paths.
- `--device-proxy`: run the host-side bit-exact proxy for the current device arithmetic
  and `.ttq` layout for generation.
- `--check`: validate the selected backend instead of generating text. With `--device-proxy`,
  compare proxy and classic CPU numerics on the prompt tokens and print drift stats. With
  `--sim` or `--device`, exact-check final logits against the device proxy, starting with the
  first prompt token and following greedy outputs for `-n` tokens. Use an explicit small `-n`.
- `--debug-layers N`: run only the first `N` transformer layers before final
  norm/projection. This is only valid with `--check`; plain inference always runs the full
  layer stack.
- `-n N`: generation limit. If omitted, normal inference runs until EOS or context
  exhaustion; for simulator/proxy/device work, an explicit small `-n` is usually more
  practical.

### Current Simulator/Device Dataflow

At startup the host loads `libttsim.so` or opens the selected devices, configures tile L1/register
windows (L1 is each tile's local SRAM) and BAR4 DRAM windows, preloads active tensors, allocates
KV/score scratch, and initializes descriptors and persistent firmware on each tile. Eight-tile
placement pairs each tile with a distinct DRAM channel, on the same NOC row wherever possible. The
allocator limits each channel to 4080 MiB and reports peak channel usage.

Logits are sent directly from tile L1 to host memory over PCIe, not staged through DRAM.
Silicon requires KMD-pinned host memory on every participating chip; pinning failure aborts
startup. The simulator uses PCIe DMA callbacks instead.
Four-chip runs map the same host logits buffer on each chip. Their embedding table also
lives in host memory, with each tile reading its residual slice directly over PCIe.

For each token, the host stages the token index, position, flags, and FP32 RoPE sine/cosine
slice on each tile, then sets its GO flag. The persistent BRISC firmware executes the configured
layer stack and optional final logits under that single token command.

All tiles run the token as peers. Each owns one KV head in eight-tile mode; at 32 tiles,
four adjacent chip-local shards assemble and share a KV head, with one cache writer and
two query outputs per shard. Cross-chip all-gathers use three courier tiles per chip and
direct peer BAR mappings, without host-assisted interchange. The simulator runs the same courier
protocol through PCIe DMA callbacks. Within one token command, BRISC currently performs:

1. Load BF16 token embedding from DRAM (one chip) or host memory (four chips) and expand to
   the FP32 residual buffer.
2. For each layer:
   - attention RMSNorm;
   - BF16 Tensix matvecs for Q, K, and V;
   - SFPU RoPE using host-precomputed FP32 sine/cosine tables;
   - BF16 KV-cache writes to DRAM;
   - attention over the active cache window;
   - BF16 Tensix attention-output matvec;
   - residual add and post-attention RMSNorm;
   - BF16 Tensix router matvec, router bias, SFPU top-4 network with a BRISC merge tail,
     and router softmax on BRISC;
   - BFP8 Tensix gate/up/down expert matvecs;
   - SFPU SwiGLU;
   - SFPU reduction of the four selected down outputs into FP32 `TT_MOE_OUT`;
   - residual update.
3. If logits are requested, final RMSNorm plus sharded BF16 Tensix `output.weight` matvec,
   sending BF16 logits directly to host memory.

The simulator KV cache is in simulated DRAM. Attention is structured around the real cache
layout rather than a fixed host-staged L1 window. Alternating layers use the model's
sliding/full attention policy.

Current attention mapping:

- K and V are streamed from DRAM in chunks of up to 256 tokens per KV head.
- Q/K score dot products, softmax exponentials, and value accumulation use the SFPU on both
  simulator and silicon. Score maxima and reciprocals remain on BRISC; the SFPU produces partial
  denominator sums, which BRISC combines along with any tail scores.
- Scores stay in L1 whenever the attended range fits in one chunk. This always covers the
  128-token sliding-window layers. Longer full-context attention spills scores between passes.
- Value accumulation stays in FP32 Dst until each KV head's output is packed.
- Final attention output is rounded to BF16 before the output projection, matching the proxy
  model.
- At 32 tiles, each KV head's cache lives on the first of its four shards' DRAM channels;
  it is not partitioned by token range. Even sliding-window layers allocate the requested token
  capacity. For 120b, uneven channel headroom can limit context before total free DRAM runs out.

Current mixture-of-experts (MoE) mapping:

- Router weights use the same 2-phase BF16 matvec path as other BF16 matvecs.
- Gate/up/down expert weights use the 1-phase BFP8 matvec path. All matvecs accumulate in FP32 Dst.
- Both paths process 64 input elements per iteration using broadcast MVMUL and two accumulation
  chains, folded with SFPU addition before packing. BF16 runs both phases in phase-major order
  within each group. Three whole-strip buffers keep up to two strips in flight using two NOC
  transaction IDs.
- FP32 biases use the packer's FP32 L1 accumulation; BF16 biases retain BF16 packing/accumulation.
- Q/K, attention-output, and router matvecs preserve FP32 packed output; V retains BF16 output.
- Gate and up matvecs pack directly into SwiGLU input tile buffers.
- SwiGLU uses an exp-based sigmoid with SFPU reciprocal and one Newton refinement.
- Down outputs are materialized as `down4[4][2880]` BF16 and reduced on the SFPU into FP32
  using dynamically loaded FP32 expert scales.

### Simulator Compatibility

The matvec kernels require SrcB broadcast MVMUL (`MVMUL` with `instr_mod19=1`). `ttsim` has
implemented it since v1.10.10, modeling all four destination rows; older versions report
`UnimplementedFunctionality`. Build `src/_out/release_bh/libttsim.so` for the one-chip mappings, or
`src/_out/release_bh_x4/libttsim.so` for the four-chip mapping.

The simulator's standard harvesting and PCIe topology need not match the physical machine;
four-chip placement follows each backend's translation tables. There is no host command queue: the
host issues one token command and waits for completion from the participating tiles.

### Numerics Status

Simulator and silicon treat the host-side device proxy as the exact reference. Any mismatch
is a bug. CPU-vs-proxy drift is measured separately with `--device-proxy --check`
because the proxy deliberately uses TTQ tensor formats and Tensix/SFPU arithmetic rather
than the classic CPU kernels.

Current qualitative status:

- Full 24-layer (20b) and 36-layer (120b) simulator/silicon checks match proxy logits exactly
  on tested inputs.
- BFP8 expert weights, FP32 accumulation, and improved SwiGLU have materially improved numerics
  and short-prompt generation quality.
- Expert routing and greedy token selection remain discontinuous: small numeric differences
  can change the generated sequence. Exact agreement with the CPU is not expected.
- Bit-exact device validation is distinct from model-quality evaluation. Broader teacher-forced
  and task-level evaluation is still needed before making application-level quality claims.

### Performance Snapshot

Recent short-context checks on the development machine, a TT QuietBox2 with a Ryzen 7 9700X
host and four Blackhole chips:

| Model | Mapping | Mean device command time |
|---|---|---|
| 20b | Eight tiles on one of the four chips | 10.98 ms/token |
| 20b | 32 tiles across all four chips | 3.83 ms/token |

These measurements used `-p "Hi" -n 2 --check`, with all 24 layers and exact proxy agreement.
They measure device token commands, not end-to-end generation throughput, and exclude model
loading and proxy checks. They are short-context sanity checks, not sustained-throughput or
long-context benchmarks.

These are development snapshots, not portable performance guarantees. Context length, compiler, and
host load matter. `--check` adds proxy execution outside the reported device/simulator command time.
The printed `proxy logits total` measures only the final vocabulary projection and its norm, not the
full proxy forward pass.

20b eight-tile startup loads about `22.3 GiB` of active tensors across eight DRAM channels, with
peak usage around `2947 MiB` per channel. Recent preload times were about `2.6 s` on silicon
and `2 s` in the simulator. For 120b, startup loads about `116.7 GiB` across 32 channels,
with peak usage around `3969 MiB` of the usable `4080 MiB` per channel and about `14 s` preload
time on silicon. Reduced-layer checks remain useful for fast iteration.

### Kernel Development

Firmware lives in `src/brisc.cpp`; the corresponding host arithmetic lives in `src/tt_backend.cpp`.
Use reduced-layer `--check --debug-layers N` runs for quick iteration, then full-model checks
on the affected tile mappings. Changes to arithmetic need matching proxy changes and a separate
CPU drift assessment. Attention changes also need longer-context testing.

Use `--profile` for per-stage device timing. Simulator speed is useful for iteration, but it is
not a prediction of silicon timing.

## SFPU Kernel Compiler

`src/kernels.sfpu` contains SFPU kernels written in a small, Python-shaped DSL. The build runs
`tools/sfpu.py` to generate C++ instruction fragments used by the firmware. This is a compiler,
not Python executing on the device or a runtime dependency of the installed program.

The DSL gives intermediate values names instead of requiring hand-assigned registers:

```python
@sfpu()
def add_rows():
    a = sfpload(0)
    b = sfpload(2)
    sfpstore(sfpadd(a, b), 0)
```

The compiler handles register allocation, immediate lowering, supported instruction hazards,
and explicitly declared REPLAY bodies. It preserves instruction order, so kernel scheduling
remains visible in the source. Register exhaustion is a compile error, not an implicit spill.
The firmware still owns unpacking, packing, NOC transfers, configuration, and synchronization.

The DSL is used by real model kernels, including RoPE, SwiGLU, RMSNorm, attention, and MoE
reduction. Its purpose is to make low-level optimization easier to review without hiding the
hardware behind a general tensor framework.

## Tensor Dumps

Pass `--dump-tensors DIRECTORY` to write intermediate tensors from CPU inference as raw,
little-endian FP32 files. It cannot be combined with `--check`, `--sim`, or `--device`:

```sh
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf \
    -p "Hello" -n 1 --dump-tensors tensors
sha256sum tensors/*.f32
```

The dump includes embeddings, attention inputs and projections, router logits, layer
outputs, and final logits. Names include the token position, layer, and operation.
`manifest.tsv` records the filename, dtype, and element count for each vector. Dumping
every prompt position can consume a large amount of storage, so use a new or empty
directory for each run.

## Tests

```sh
./make.py :test
```

This target needs no model files or hardware. It covers core arithmetic and quantization,
requant file safety with synthetic fixtures, host ISA dispatch, the SFPU compiler, and the router
top-four networks. Host dispatch tests compare v3/v4 proxy arithmetic when v4 is available and
exercise MXFP4 decode and prefill tails on the selected ISA.
Tests use incremental stamps.

For the additional tokenizer integration checks against the 20b model:

```sh
./make.py :test-model
```

Set `GPT_OSS_TEST_MODEL` if the model is somewhere other than
`~/models/gpt-oss-20b-MXFP4.gguf`. The integration target requires that file and tracks it as
a build dependency. It also includes the model-free tests.

For a quick smoke test against both locally downloaded model files:

```sh
_out/tt-lab inspect -m ~/models/gpt-oss-20b-MXFP4.gguf
_out/tt-lab inspect -m ~/models/gpt-oss-120b-MXFP4.gguf
_out/tt-lab run -m ~/models/gpt-oss-20b-MXFP4.gguf -p "Hello" -n 1
_out/tt-lab run -m ~/models/gpt-oss-120b-MXFP4.gguf -p "Hello" -n 1
```

## Project Direction

`tt-lab` is meant to stay small enough that the important behavior is visible in source
form. TT hardware support should preserve that character: plain files, explicit data
movement, clear numerical contracts, and backend code that can be inspected without first
learning a framework.

The CPU path provides reference inference and a baseline for numerical comparisons. The device
proxy is the exact reference for the TT simulator and silicon paths; CPU drift is tracked
separately because TTQ tensor formats and Tensix/SFPU arithmetic intentionally differ from
the classic CPU kernels.

## Security

See the [Security Policy](SECURITY.md) to report vulnerabilities. Use trusted GGUF/TTQ inputs;
this experimental runner has not undergone a dedicated malformed-model security audit. `--sim`
loads and executes the supplied shared library.

## Contributing

This project does not accept pull requests. Report bugs and send questions or suggestions through
GitHub Issues. See [CONTRIBUTING.md](CONTRIBUTING.md) for details and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

- [LICENSE](LICENSE): Overall license for this project, except where specified (Apache-2.0).
- `make.py`: MIT license, as stated in the file header.

See also [NOTICE](NOTICE) and [LICENSE_understanding.txt](LICENSE_understanding.txt). Downloaded
model weights are separate artifacts governed by their own licenses and terms.
