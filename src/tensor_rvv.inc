// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// RISC-V Vector body of the CPU reference Q8_0 matvec (see tensor.cpp). Included once per LMUL with
//   RVV_NS namespace, RVV_L LMUL for 16 x 32-bit (m4: VLEN >= 128, m2: VLEN >= 256), RVV_L8 for 16 x 8-bit.
// Each body performs the generic code's floating-point operations in the same order and with the
// same fusion (GCC contracts `acc += a * b` to FMA), so results are bit-identical; the digests in
// tests/matvec_bench.cpp check this. (MXFP4 stays generic: its per-row FMA chain is serial, and the
// rows-in-lanes layout needs strided byte loads, which made it slower than scalar on the SpacemiT X100.)

#define RVV_CAT_(a, b) a##b
#define RVV_CAT(a, b) RVV_CAT_(a, b)
#define RVV_FN(name, type) RVV_CAT(RVV_CAT(__riscv_, name), RVV_CAT(type, RVV_L))
#define RVV_FN8(name, type) RVV_CAT(RVV_CAT(__riscv_, name), RVV_CAT(type, RVV_L8))

namespace RVV_NS {

using f32v = RVV_CAT(RVV_CAT(vfloat32, RVV_L), _t);
constexpr size_t vl = 16;

// Q8_0: two 16-lane accumulators per row (columns 0-15 and 16-31 of each block), as in
// q8_0_vector_fma_block, then an in-order horizontal sum (reduce_f32x16).
[[gnu::always_inline]] inline float q8_0_row(const std::byte *data, const float *input, size_t blocks) {
    f32v acc0 = RVV_FN(vfmv_v_f_, f32)(0.0f, vl);
    f32v acc1 = acc0;
    for (size_t block = 0; block < blocks; ++block) {
        const auto *source = (const int8_t *)(data + block * 34);
        const float scale = fp16_to_fp32(load_u16(data + block * 34));
        const f32v q0 = __riscv_vfmul(__riscv_vfcvt_f(__riscv_vsext_vf4(RVV_FN8(vle8_v_, i8)(source + 2, vl), vl), vl),
                                      scale, vl);
        const f32v q1 = __riscv_vfmul(__riscv_vfcvt_f(__riscv_vsext_vf4(RVV_FN8(vle8_v_, i8)(source + 18, vl), vl), vl),
                                      scale, vl);
        acc0 = __riscv_vfmacc(acc0, q0, RVV_FN(vle32_v_, f32)(input + block * 32, vl), vl);
        acc1 = __riscv_vfmacc(acc1, q1, RVV_FN(vle32_v_, f32)(input + block * 32 + 16, vl), vl);
    }
    const f32v acc = __riscv_vfadd(acc0, acc1, vl);
    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f(__riscv_vfredosum(acc, zero, vl));
}

void matvec_q8_0(const std::byte *matrix, size_t columns, size_t rows, const float *bias, const float *input,
                 float *output) {
    const size_t row_bytes = columns / 32 * 34;
    for (size_t row = 0; row < rows; ++row) {
        const float sum = q8_0_row(matrix + row * row_bytes, input, columns / 32);
        output[row] = bias ? bias[row] + sum : sum;
    }
}

} // namespace RVV_NS

#undef RVV_CAT_
#undef RVV_CAT
#undef RVV_FN
#undef RVV_FN8
