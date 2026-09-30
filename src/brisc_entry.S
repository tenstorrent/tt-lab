// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

.section .text
.global _start
.extern brisc_main

_start:
    li sp, 0xFFB02000
    call brisc_main
loop:
    j loop
