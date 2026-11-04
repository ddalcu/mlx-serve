// MiMo prefill combine (fused): the prompt pass's routed sum read in place from the call's <= 24 expert-tile sources.
// The stock graph per 256-token tile (math.zig reduceGraph): f32(bf16 row) * f32 w (MLX Multiply, its own kernel: one
// f32 rounding, never contracted), the axis-1 sum of [n, 8, H] = col_reduce_small with threadgroup y = 8 (reduce.cpp
// strided_reduce_general_dispatch: 8 rows < 32), so each product is first added to the 0 init, then lid.y 0 folds
// k = 1..7 in order as op(product_k, total) (reduce_col.h); then astype bf16 (one rounding). No contraction into FMA.
#pragma METAL fp contract(off)
template <typename T>
inline const device T* mimo_pc_pick(int s, const device T* s0, const device T* s1, const device T* s2, const device T* s3, const device T* s4, const device T* s5, const device T* s6, const device T* s7, const device T* s8, const device T* s9, const device T* s10, const device T* s11, const device T* s12, const device T* s13, const device T* s14, const device T* s15, const device T* s16, const device T* s17, const device T* s18, const device T* s19, const device T* s20, const device T* s21, const device T* s22, const device T* s23) {
    switch (s) {
        case 0: return s0;
        case 1: return s1;
        case 2: return s2;
        case 3: return s3;
        case 4: return s4;
        case 5: return s5;
        case 6: return s6;
        case 7: return s7;
        case 8: return s8;
        case 9: return s9;
        case 10: return s10;
        case 11: return s11;
        case 12: return s12;
        case 13: return s13;
        case 14: return s14;
        case 15: return s15;
        case 16: return s16;
        case 17: return s17;
        case 18: return s18;
        case 19: return s19;
        case 20: return s20;
        case 21: return s21;
        case 22: return s22;
        default: return s23;
    }
}
// P7 `inplace` (prefill_combine.zig `callInplace`): the same pick through the call's table of its sources' GPU addresses
// (each source's MTLBuffer gpuAddress + offset): every source read in place, past Metal's 31 buffer arguments.
template <typename T>
inline const device T* mimo_pc_pick(int s, const device uint64_t* tbl) {
    return reinterpret_cast<const device T* const device*>(tbl)[s];
}
