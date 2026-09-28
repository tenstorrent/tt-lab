/* SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc. */
/* SPDX-License-Identifier: Apache-2.0 */

ENTRY(_start)

SECTIONS
{
    . = 0;
    .text : {
        *(.text*)
    }
    .rodata : {
        *(.rodata*)
    }
    .data : {
        *(.data*)
    }
    .bss : {
        *(.bss*)
        *(COMMON)
    }
}
