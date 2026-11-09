// MLX's argmax order on (value, index) pairs.
//
// MLX's ArgMax (backend/metal/kernels/arg_reduce.metal): a NaN beats every number, and among NaNs the lower index wins;
// among numbers the larger value wins, and among equal values (-0 equals +0) the lower index. On distinct indices that is
// a strict total order, so any reduction tree over the pairs ends at MLX's argmax.
inline bool dsv41_mk_better(float va, uint ia, float vb, uint ib) {
  const bool na = metal::isnan(va), nb = metal::isnan(vb);
  if (na != nb) return na;
  if (na) return ia < ib;
  if (va != vb) return va > vb;
  return ia < ib;
}

// An empty lane's or slot's index (its value -inf): every real pair beats it.
#define DSV41_MK_NONE 0xFFFFFFFFu

// The simdgroup's best pair, in lane 0.
inline void dsv41_mk_simd_best(thread float& v, thread uint& i) {
  for (ushort d = 16; d >= 1; d >>= 1) {
    const float ov = metal::simd_shuffle_down(v, d);
    const uint oi = metal::simd_shuffle_down(i, d);
    if (dsv41_mk_better(ov, oi, v, i)) {
      v = ov;
      i = oi;
    }
  }
}

// The index of the best of the `n` (value bits, index) pairs at `p`, for every thread of a threadgroup of TGT threads:
// each thread's share, its simdgroup's best, then simdgroup 0 over the simdgroups' bests. `tv` / `ti` hold a slot a
// simdgroup, `bc` the answer; the threadgroup may reuse all three once it returns.
template <uint TGT>
inline uint dsv41_mk_tg_best(const device uint* p, uint n, uint tid, uint lane, uint sg, threadgroup float* tv,
                             threadgroup uint* ti, threadgroup uint* bc) {
  float bv = -metal::numeric_limits<float>::infinity();
  uint bi = DSV41_MK_NONE;
  for (uint j = tid; j < n; j += TGT) {
    const float v = as_type<float>(p[2u * j]);
    const uint ix = p[2u * j + 1u];
    if (dsv41_mk_better(v, ix, bv, bi)) {
      bv = v;
      bi = ix;
    }
  }
  dsv41_mk_simd_best(bv, bi);
  if (lane == 0) {
    tv[sg] = bv;
    ti[sg] = bi;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (sg == 0) {
    constexpr uint NSG = TGT / 32u;
    bv = lane < NSG ? tv[lane] : -metal::numeric_limits<float>::infinity();
    bi = lane < NSG ? ti[lane] : DSV41_MK_NONE;
    dsv41_mk_simd_best(bv, bi);
    if (lane == 0) *bc = bi;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint r = *bc;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  return r;
}
