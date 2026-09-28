#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

model=${GPT_OSS_TEST_MODEL:-$HOME/models/gpt-oss-20b-MXFP4.gguf}

actual=$(_out/tt-lab tokenize -m "$model" -p 'How do you like your steak?' | tail -n 1)
expected='tokens (7): [5299, 621, 481, 1299, 634, 67314, 30]'
[[ "$actual" == "$expected" ]]

actual=$(_out/tt-lab tokenize -m "$model" -p $'HTTPServer can\x27t count 123456 — café 中文 👋\nNext line.' | tail -n 1)
expected='tokens (16): [17893, 6444, 8535, 3605, 220, 7633, 19354, 2733, 30469, 83711, 61138, 233, 198, 7695, 2543, 13]'
[[ "$actual" == "$expected" ]]
