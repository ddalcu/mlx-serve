// dec_moe_fp4_rows: the all-rows down launch's readers (after dec_moe_fp4_header.metal). Each loads a weight unit once
// and adds it into every row that reads that weight row: per row the same units in the same order as the one-row
// readers (`dec_fp4_down_routed`, `dec_fp4_down_shared`, at the same FR and DW), so each row's sum is the one-row
// launch's.

// Routed expert row `row` (fp4, K inputs) against the rows whose pick index `js[r]` is not 0xFF: row r reads its
// activation slot r * (TOPK + 1) + js[r] of `x`. A lane takes units first, first + step...
template <int K, int GS, int FR, int R, int TOPK, typename T, int DW = 0>
inline void dec_fp4_down_routed_rows(const device uint32_t* w, const device uint8_t* sc, size_t row, const device T* x, const threadgroup uchar* js, uint first, uint step, thread float (&acc)[R]) {
  constexpr uint NU = K / 32, SPU = 32 / GS;
  const device uint4* wp = (const device uint4*)(w + row * (K / 8));
  const device uint8_t* sp = sc + row * (K / GS);
  const device uint4* xr[R];
  bool on[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const uint j = uint(js[r]);
    on[r] = j != 0xFFu;
    xr[r] = (const device uint4*)(x + size_t(r * (TOPK + 1) + (on[r] ? j : 0u)) * K);
    acc[r] = 0.0f;
  }
  for (uint u = first; u < NU; u += step) {
    const uint4 q = wp[u];
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (on[r]) acc[r] = dec_fp4_unit<GS, FR, DW>(q, xr[r] + u * 4, sp + u * SPU, acc[r]);
  }
}

// The shared expert's down row `row` (mxfp8, K inputs: a unit is 32 bytes) against every row's slot TOPK of `x`, each
// unit decoded once for all rows: `dec_fp4_down_shared_unit<FR>`'s chain a row.
template <int K, int FR, int R, int TOPK, typename T>
inline void dec_fp4_down_shared_rows(const device uint32_t* sw, const device uint8_t* ss, size_t row, const device T* x, uint first, uint step, thread float (&acc)[R]) {
  constexpr uint NU = K / 32;
  const device uint4* wp = (const device uint4*)(sw + row * (K / 4));
  const device uint8_t* sp = ss + row * (K / 32);
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  for (uint u = first; u < NU; u += step) {
    const uint4 q0 = wp[2 * u], q1 = wp[2 * u + 1];
    const uint qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float4 a[4], b[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      a[c] = dsv41_e4m3x4<FR>(qs[2 * c]);
      b[c] = dsv41_e4m3x4<FR>(qs[2 * c + 1]);
    }
    const uint e = uint(sp[u]);
    const float f = FR == 0 ? mlxserve_fp4_e8m0(e) : as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const device uint4* xs = (const device uint4*)(x + size_t(r * (TOPK + 1) + TOPK) * K);
      float d = 0.0f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 xq = xs[u * 4 + c];
        const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          d = fma(xf[i], a[c][i], d);
          d = fma(xf[4 + i], b[c][i], d);
        }
      }
      acc[r] = FR == 0 ? fma(f, d * 256.0f, acc[r]) : fma(f, d, acc[r]);
    }
  }
}
