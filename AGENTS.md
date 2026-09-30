Goal: simple, readable `tt-lab` implementation for running gpt-oss-20b/120b MXFP4 GGUF files, with a narrow TT Blackhole backend for bring-up and tensor experiments.

Style: gnu++20, mostly C-like, Linux/x86_64 first, no threads. Intrinsics only for isolated measured hot kernels.

Allocate dynamic memory at startup only; no inner-loop or decode-path allocation in normal inference.

K&R braces, 4 spaces. Always brace control flow, one statement per line. Prefer `float(x)` casts; use C-style casts for pointers.

Run `clang-format` on touched C/C++ sources before finishing code changes.

Keep dependencies minimal. No Python runtime dependency for the installed program, no frameworks, no compute libraries, and no hidden service layer.

TT backend scope: BH only, gpt-oss-20b on one chip with one or eight Tensix tiles, either model on four chips with eight tiles per chip over PCIe peer-to-peer, one BRISC per tile, host tokenizer, `.ttq` required. Maintain the CPU path, host-side device proxy, `ttsim` path, silicon path, and `tt-lab requant` flow as related pieces of the same backend. Simulator/device must bit-match the device proxy; CPU drift is tracked separately.
