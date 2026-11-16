// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// RISC-V Vector body of the device-proxy matvec (see tt_matvec.cpp). Included once per LMUL with
//   RVV_NS   namespace name
//   RVV_L    LMUL suffix for 16 x u32 (m4: any VLEN >= 128; m2: VLEN >= 256)
//   RVV_L8   matching suffix for 16 x u8, RVV_L16 for 16 x u16, RVV_B mask type
// It is a lane-for-lane transcription of the generic vector-extension code: same operations in
// the same order, so the results are bit-identical (checked by tests/matvec_bench.cpp digests).
// Out-of-range shift counts only occur in lanes whose result the generic code discards too.

#define RVV_CAT_(a, b) a##b
#define RVV_CAT(a, b) RVV_CAT_(a, b)
#define RVV_U32 RVV_CAT(RVV_CAT(vuint32, RVV_L), _t)
#define RVV_I32 RVV_CAT(RVV_CAT(vint32, RVV_L), _t)
#define RVV_FN(name, type) RVV_CAT(RVV_CAT(__riscv_, name), RVV_CAT(type, RVV_L))
#define RVV_FN_L(name, type, lmul) RVV_CAT(RVV_CAT(__riscv_, name), RVV_CAT(type, lmul))

namespace RVV_NS {

using u32v = RVV_U32;
using i32v = RVV_I32;
using bv = RVV_B;
constexpr size_t vl = 16;

[[gnu::always_inline]] inline u32v splat(uint32_t x) {
    return RVV_FN(vmv_v_x_, u32)(x, vl);
}
[[gnu::always_inline]] inline i32v splati(int32_t x) {
    return RVV_FN(vmv_v_x_, i32)(x, vl);
}
[[gnu::always_inline]] inline i32v as_i(u32v x) {
    return RVV_CAT(__riscv_vreinterpret_i32, RVV_L)(x);
}
[[gnu::always_inline]] inline u32v as_u(i32v x) {
    return RVV_CAT(__riscv_vreinterpret_u32, RVV_L)(x);
}
// select(mask, yes, no)
[[gnu::always_inline]] inline u32v sel(bv mask, u32v yes, u32v no) {
    return __riscv_vmerge(no, yes, mask, vl);
}
[[gnu::always_inline]] inline i32v seli(bv mask, i32v yes, i32v no) {
    return __riscv_vmerge(no, yes, mask, vl);
}

[[gnu::always_inline]] inline u32v clz(u32v x) {
#if defined(__riscv_zvbb)
    return __riscv_vclz(x, vl);
#else
    u32v n = splat(0);
    for (uint32_t s : {16u, 8u, 4u, 2u, 1u}) {
        const bv high_clear = __riscv_vmsltu(x, 1u << (32 - s), vl);
        n = sel(high_clear, __riscv_vadd(n, s, vl), n);
        x = sel(high_clear, __riscv_vsll(x, s, vl), x);
    }
    return sel(__riscv_vmseq(x, 0u, vl), splat(32), n);
#endif
}

[[gnu::always_inline]] inline u32v align_man(u32v man, u32v sign, i32v exp_value, i32v exp) {
    const bv active = __riscv_vmslt(exp_value, exp, vl);
    const u32v diff = as_u(__riscv_vsub(exp, exp_value, vl));
    const bv small = __riscv_vmsltu(diff, 31u, vl);
    const u32v shift_m1 = sel(__riscv_vmand(active, small, vl), __riscv_vsub(diff, 1u, vl), splat(0));
    const u32v rounded = __riscv_vsub(__riscv_vadd(man, __riscv_vsll(splat(1), shift_m1, vl), vl), sign, vl);
    const u32v shifted = __riscv_vsrl(rounded, diff, vl);
    return sel(active, sel(small, shifted, splat(0)), man);
}

[[gnu::always_inline]] inline u32v normalize_encode_fp32_acc(u32v sign_sop0, i32v exp_sop0, u32v man_sop0, u32v sign_sop1,
                                                       i32v exp_sop1, u32v man_sop1, u32v dst_value) {
    const u32v sign_dst = __riscv_vsrl(dst_value, 31u, vl);
    const i32v exp_dst = as_i(__riscv_vand(__riscv_vsrl(dst_value, 23u, vl), 255u, vl));
    u32v man_dst = __riscv_vor(__riscv_vand(dst_value, 0x7FFFFFu, vl), 0x800000u, vl);
    man_dst = sel(__riscv_vmsne(exp_dst, 0, vl), man_dst, splat(0));

    i32v exp = __riscv_vmax(__riscv_vmax(exp_sop0, exp_sop1, vl), exp_dst, vl);
    const bv exp_positive = __riscv_vmsgt(exp, 0, vl);
    man_sop0 = align_man(man_sop0, sign_sop0, exp_sop0, exp);
    man_sop1 = align_man(man_sop1, sign_sop1, exp_sop1, exp);
    man_dst = align_man(man_dst, splat(0), exp_dst, exp);

    u32v sign = sign_dst;
    i32v man = as_i(man_dst);
    const i32v m0 = as_i(man_sop0);
    const i32v m1 = as_i(man_sop1);
    man = __riscv_vadd(man, seli(__riscv_vmseq(sign_sop0, sign, vl), m0, __riscv_vneg(m0, vl)), vl);
    man = __riscv_vadd(man, seli(__riscv_vmseq(sign_sop1, sign, vl), m1, __riscv_vneg(m1, vl)), vl);
    const bv negative = __riscv_vmslt(man, 0, vl);
    man = seli(negative, __riscv_vneg(man, vl), man);
    sign = sel(negative, __riscv_vxor(sign, 1u, vl), sign);

    const bv nonzero = __riscv_vmand(__riscv_vmsne(man, 0, vl), exp_positive, vl);
    u32v man_u = as_u(man);
    const i32v shift = __riscv_vsub(as_i(clz(man_u)), 8, vl);
    exp = __riscv_vsub(exp, shift, vl);
    const bv right_mask = __riscv_vmslt(shift, 0, vl);
    const u32v right_shift = as_u(__riscv_vneg(shift, vl));
    const u32v right_shift_m1 = sel(right_mask, __riscv_vsub(right_shift, 1u, vl), splat(0));
    const u32v right_value =
        __riscv_vsrl(__riscv_vadd(man_u, __riscv_vsll(splat(1), right_shift_m1, vl), vl), right_shift, vl);
    const u32v left_value = __riscv_vsll(man_u, as_u(shift), vl);
    man_u = sel(right_mask, right_value, left_value);
    exp = __riscv_vadd_mu(__riscv_vmsne(__riscv_vand(man_u, 0x1000000u, vl), 0u, vl), exp, exp, 1, vl);
    man_u = __riscv_vand(man_u, 0x7FFFFFu, vl);

    const bv valid = __riscv_vmand(nonzero, __riscv_vmsgt(exp, 0, vl), vl);
    const bv inf = __riscv_vmsge(exp, 255, vl);
    const u32v sign_bit = __riscv_vsll(sign, 31u, vl);
    const u32v finite = __riscv_vor(__riscv_vor(sign_bit, __riscv_vsll(as_u(exp), 23u, vl), vl), man_u, vl);
    const u32v inf_value = __riscv_vor(sign_bit, 255u << 23, vl);
    return sel(valid, sel(inf, inf_value, finite), splat(0));
}

[[gnu::always_inline]] inline u32v dst_decode(u32v x) {
    const u32v hi = __riscv_vsrl(x, 16u, vl);
    u32v dec = __riscv_vand(hi, 0x8000u, vl);
    dec = __riscv_vor(dec, __riscv_vsll(__riscv_vand(hi, 255u, vl), 7u, vl), vl);
    dec = __riscv_vor(dec, __riscv_vand(__riscv_vsrl(hi, 8u, vl), 127u, vl), vl);
    return __riscv_vor(__riscv_vsll(dec, 16u, vl), __riscv_vand(x, 0xFFFFu, vl), vl);
}

[[gnu::always_inline]] inline u32v dst_encode(u32v x) {
    const u32v hi = __riscv_vsrl(x, 16u, vl);
    u32v enc = __riscv_vand(hi, 0x8000u, vl);
    enc = __riscv_vor(enc, __riscv_vsll(__riscv_vand(hi, 127u, vl), 8u, vl), vl);
    enc = __riscv_vor(enc, __riscv_vand(__riscv_vsrl(hi, 7u, vl), 255u, vl), vl);
    return __riscv_vor(__riscv_vsll(enc, 16u, vl), __riscv_vand(x, 0xFFFFu, vl), vl);
}

// A tile operand: BF16 tiles are one pointer; BFP8 tiles read exponents and data in place from the strip.
struct TileRef {
    const uint8_t *data;
    const uint8_t *exponents;
};

[[gnu::always_inline]] inline u32v bfp8_srca(TileRef tile, uint32_t k_column) {
    // Column k_column of the 16x16 datum block: byte k_column of each 16-byte row (UNPACR transpose).
    const u32v packed = __riscv_vzext_vf4(RVV_FN_L(vlse8_v_, u8, RVV_L8)(tile.data + k_column, 16, vl), vl);
    const u32v shared_exp = __riscv_vzext_vf4(RVV_FN_L(vle8_v_, u8, RVV_L8)(tile.exponents, vl), vl);
    const u32v sign = __riscv_vsrl(packed, 7u, vl);
    const u32v magnitude = __riscv_vand(__riscv_vsll(packed, 1u, vl), 255u, vl);
    const u32v lz = __riscv_vsub(clz(magnitude), 24u, vl);
    const u32v exponent = __riscv_vand(__riscv_vsub(shared_exp, lz, vl), 255u, vl);
    u32v bf16 = __riscv_vor(__riscv_vsll(sign, 15u, vl), __riscv_vsll(exponent, 7u, vl), vl);
    bf16 = __riscv_vor(bf16, __riscv_vand(__riscv_vsll(magnitude, lz, vl), 0x7Eu, vl), vl);
    const u32v zero_value = __riscv_vmul(sign, 0xFF80u, vl);
    return __riscv_vsll(sel(__riscv_vmsne(magnitude, 0u, vl), bf16, zero_value), 3u, vl);
}

[[gnu::always_inline]] inline u32v bf16_srca(TileRef tile, uint32_t k_column) {
    const uint16_t *source = (const uint16_t *)tile.data + (k_column / 2) * 32 + (k_column & 1) * 16;
    return __riscv_vsll(__riscv_vzext_vf2(RVV_FN_L(vle16_v_, u16, RVV_L16)(source, vl), vl), 3u, vl);
}

template<uint32_t phase, uint32_t srca_format>
[[gnu::always_inline]] inline void mvmul_sop_tile(TileRef tile, const uint32_t *b_values, uint32_t k_base, u32v &sop_sign,
                                            i32v &sop_exp, u32v &sop_man) {
    static_assert(phase < 2);
    static_assert(srca_format == 5 || srca_format == 6);
    alignas(64) uint32_t signs[8][16];
    alignas(64) int32_t exps[8][16];
    alignas(64) uint32_t mans[8][16];
    constexpr int32_t exp_prod_adj = -127 - (phase ? 5 : 0);

    i32v exp_sop = splati(0);
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t k_column = k_base + i;
        u32v a;
        if constexpr (srca_format == 6) {
            a = bfp8_srca(tile, k_column);
        } else {
            a = bf16_srca(tile, k_column);
        }
        const uint32_t b = b_values[k_column];
        bv zero = __riscv_vmsltu(__riscv_vand(a, 0x3FFFFu, vl), 0x400u, vl);
        if ((b & 0x3FFFFu) < 0x400u) {
            zero = __riscv_vmor(zero, __riscv_vmnot(zero, vl), vl);  // every lane
        }
        const u32v exp_a = __riscv_vand(__riscv_vsrl(a, 10u, vl), 255u, vl);
        const uint32_t exp_b = (b >> 10) & 255u;
        u32v man_a = __riscv_vor(__riscv_vand(a, 0x3FFu, vl), 0x400u, vl);
        const uint32_t man_b = ((b & 0x3FFu) | 0x400u) >> 4;
        if constexpr (phase & 1) {
            man_a = __riscv_vand(__riscv_vsrl(man_a, 1u, vl), 31u, vl);
        } else {
            man_a = __riscv_vsrl(man_a, 6u, vl);
        }
        const u32v sign = sel(zero, splat(0), __riscv_vxor(__riscv_vsrl(a, 18u, vl), b >> 18, vl));
        const i32v exp = seli(zero, splati(0), __riscv_vadd(as_i(__riscv_vadd(exp_a, exp_b, vl)), exp_prod_adj, vl));
        const u32v man = sel(zero, splat(0), __riscv_vmul(man_a, man_b, vl));
        RVV_FN(vse32_v_, u32)(signs[i], sign, vl);
        RVV_FN(vse32_v_, i32)(exps[i], exp, vl);
        RVV_FN(vse32_v_, u32)(mans[i], man, vl);
        exp_sop = i == 0 ? exp : __riscv_vmax(exp_sop, exp, vl);
    }

    i32v man_sop = splati(0);
    const bv positive = __riscv_vmsgt(exp_sop, 0, vl);
    for (uint32_t i = 0; i < 8; ++i) {
        const i32v exp_i = RVV_FN(vle32_v_, i32)(exps[i], vl);
        const u32v exp_diff = __riscv_vminu(as_u(__riscv_vsub(exp_sop, exp_i, vl)), 30u, vl);
        const u32v man_u = RVV_FN(vle32_v_, u32)(mans[i], vl);
        const u32v man = __riscv_vsrl(__riscv_vadd(__riscv_vsll(man_u, 1u, vl), __riscv_vsll(splat(1), exp_diff, vl), vl),
                                      __riscv_vadd(exp_diff, 1u, vl), vl);
        const bv negative = __riscv_vmsne(RVV_FN(vle32_v_, u32)(signs[i], vl), 0u, vl);
        i32v signed_man = seli(negative, __riscv_vneg(as_i(man), vl), as_i(man));
        signed_man = seli(positive, signed_man, splati(0));
        man_sop = __riscv_vadd(man_sop, signed_man, vl);
    }

    const bv negative = __riscv_vmslt(man_sop, 0, vl);
    sop_sign = sel(negative, splat(1), splat(0));
    sop_man = __riscv_vsll(as_u(seli(negative, __riscv_vneg(man_sop, vl), man_sop)), 13u, vl);
    sop_exp = seli(positive, exp_sop, splati(0));
}

template<uint32_t phase, uint32_t srca_format>
[[gnu::always_inline]] inline void mvmul_tile_add(TileRef tile, const uint32_t *b_values, uint32_t *dst_internal) {
    u32v sign0, man0, sign1, man1;
    i32v exp0, exp1;
    mvmul_sop_tile<phase, srca_format>(tile, b_values, 0, sign0, exp0, man0);
    mvmul_sop_tile<phase, srca_format>(tile, b_values, 8, sign1, exp1, man1);
    const u32v dst_value = dst_decode(RVV_FN(vle32_v_, u32)(dst_internal, vl));
    const u32v result = normalize_encode_fp32_acc(sign0, exp0, man0, sign1, exp1, man1, dst_value);
    RVV_FN(vse32_v_, u32)(dst_internal, dst_encode(result), vl);
}

void matvec_bf16(const uint8_t *tiles, const uint32_t *b_value_rows, size_t k_tiles, uint32_t chains[2][16]) {
    for (size_t base = 0; base < k_tiles; base += 4) {
        for (size_t k = base; k < base + 4; ++k) {
            mvmul_tile_add<0, 5>({tiles + k * TT_MVMUL_BF16_A_TILE_BYTES, nullptr}, b_value_rows + k * TT_MVMUL_TILE_DIM,
                                 chains[k % 2]);
        }
        for (size_t k = base; k < base + 4; ++k) {
            mvmul_tile_add<1, 5>({tiles + k * TT_MVMUL_BF16_A_TILE_BYTES, nullptr}, b_value_rows + k * TT_MVMUL_TILE_DIM,
                                 chains[k % 2]);
        }
    }
}

void matvec_bfp8(const uint8_t *strip, const uint32_t *b_value_rows, uint32_t chains[2][16]) {
    for (size_t k = 0; k < TT_MVMUL_K_TILES; ++k) {
        const TileRef tile = {strip + TT_BFP8_STRIP_EXP_BYTES + k * TT_MVMUL_BFP8_TILE_DATUM_BYTES,
                              strip + k * TT_MVMUL_BFP8_TILE_EXP_BYTES};
        mvmul_tile_add<0, 6>(tile, b_value_rows + k * TT_MVMUL_TILE_DIM, chains[k % 2]);
    }
}

} // namespace RVV_NS

#undef RVV_CAT_
#undef RVV_CAT
#undef RVV_U32
#undef RVV_I32
#undef RVV_FN
#undef RVV_FN_L
