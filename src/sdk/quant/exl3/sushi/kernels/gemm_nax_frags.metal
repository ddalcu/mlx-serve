using nfrag = vec<half, 8>;
static inline nfrag nax_wfrag(const device uint *words, uint lane) {
  constexpr uint N = 64u;
  const uint source0 = (lane & 16u) + ((lane & 7u) << 1u);
  const uint2 current = *(const device uint2 *)(words + source0);
  const uint previous = words[(source0 + 31u) & 31u];
  const uint slot = ((lane >> 3u) & 1u) * 2u;
  const ulong w0 = ((ulong)previous << 32) | (ulong)current.x;
  const ulong w1 = ((ulong)current.x << 32) | (ulong)current.y;
  const uint s0 = 28u - slot * 4u;
  const uint s1 = 28u - (slot + 4u) * 4u;
  const half2 p00 = exl3_pairh(uint2(uint(w0 >> s0) & 0xffffu, uint(w0 >> (s0 - 4u)) & 0xffffu));
  const half2 p01 = exl3_pairh(uint2(uint(w1 >> s0) & 0xffffu, uint(w1 >> (s0 - 4u)) & 0xffffu));
  const half2 p10 = exl3_pairh(uint2(uint(w0 >> s1) & 0xffffu, uint(w0 >> (s1 - 4u)) & 0xffffu));
  const half2 p11 = exl3_pairh(uint2(uint(w1 >> s1) & 0xffffu, uint(w1 >> (s1 - 4u)) & 0xffffu));
  return nfrag(p00.x, p00.y, p01.x, p01.y, p10.x, p10.y, p11.x, p11.y);
}
// A lane's weights are 16m + 2b + {0, 1, 4, 5} and the same + 8: one funnel per quad, ending at
// its last weight. 16 weights span N whole bits, so the shifts depend on b alone.
static inline constexpr uint nax_end(uint N, uint b, uint j) { return ((2u * b + j + 1u) * N) >> 4u; }
// Past 32 bits (n > 50) a quad's funnel reads 64 bits from three words.
static inline constexpr bool nax_quads_fit(uint N) {
  return nax_end(N, 0u, 5u) - nax_end(N, 0u, 0u) <= 16u && nax_end(N, 1u, 5u) - nax_end(N, 1u, 0u) <= 16u;
}
static inline uint nax_sh(uint N, uint b, uint j) {
  const uint s0 = nax_end(N, 0u, 5u) - nax_end(N, 0u, j);
  return s0 + b * (nax_end(N, 1u, 5u) - nax_end(N, 1u, j) - s0);
}
template<uint N, typename F>
static inline nfrag nax_quads(F lo, F hi, uint s0, uint s1, uint s4) {
  const half2 p0 = exl3_pairh(uint2(uint(lo >> s0) & 0xffffu, uint(lo >> s1) & 0xffffu));
  const half2 p1 = exl3_pairh(uint2(uint(lo >> s4) & 0xffffu, uint(lo) & 0xffffu));
  const half2 p2 = exl3_pairh(uint2(uint(hi >> s0) & 0xffffu, uint(hi >> s1) & 0xffffu));
  const half2 p3 = exl3_pairh(uint2(uint(hi >> s4) & 0xffffu, uint(hi) & 0xffffu));
  return nfrag(p0.x, p0.y, p2.x, p2.y, p1.x, p1.y, p3.x, p3.y);
}
template<uint N>
static inline nfrag nax_wfrag_k(const device uint *words, uint lane) {
  if (N == 64u) return nax_wfrag(words, lane);
  constexpr uint W = N / 2u;
  const uint end = 8u * N * (lane >> 4u) + N * (lane & 7u) + (nax_end(N, 1u, 5u) - nax_end(N, 0u, 5u)) * ((lane >> 3u) & 1u) + nax_end(N, 0u, 5u);
  const uint b = (lane >> 3u) & 1u;
  const uint s0 = nax_sh(N, b, 0u);
  const uint s1 = nax_sh(N, b, 1u);
  const uint s4 = nax_sh(N, b, 4u);
  if (nax_quads_fit(N)) return nax_quads<N>(uint(exl3_window(words, end, W, false)), uint(exl3_window(words, end + W, W, false)), s0, s1, s4);
  return nax_quads<N>(exl3_window(words, end, W, true), exl3_window(words, end + W, W, true), s0, s1, s4);
}
static inline short2 nax_origin(uint lane) {
  const short qid = short(lane >> 2u);
  return short2(short(((qid & 2) | short(lane & 1u)) * 4), short((qid & 4) | short((lane >> 1u) & 3u)));
}
template <typename D>
static inline void nax_zero(thread D &dst) {
  for (uint s = 0u; s < dst.get_capacity(); s++) dst[s] = 0.0f;
}
// `rem` is the block's live row count: the 16-row tile is padded with zeros
// so a short run still runs the full mma. The half4 read is 8-byte aligned
// because IDIM is a multiple of 16 and origin.x a multiple of 4.
template <typename L, typename X>
static inline void nax_left(thread L &left, const device X *x, uint row, uint idim, uint kbase, short2 origin, uint rem) {
  using x4 = vec<X, 4>;
  const size_t r0 = (size_t)(row + uint(origin.y)) * (size_t)idim + kbase + uint(origin.x);
  const size_t r1 = r0 + 8u * (size_t)idim;
  const x4 v0 = (uint(origin.y) < rem) ? *(const device x4 *)(x + r0) : x4(0);
  const x4 v1 = (uint(origin.y) + 8u < rem) ? *(const device x4 *)(x + r1) : x4(0);
  for (uint c = 0u; c < 4u; c++) {
    left[c] = v0[c];
    left[4 + c] = v1[c];
  }
}
// Element offset of row `row`, clamped to the last input row, at the lane's column.
static inline size_t nax_row(uint row, uint last, uint idim, short2 origin) {
  return (size_t)min(row, last) * (size_t)idim + uint(origin.x);
}
template <typename L, typename X>
static inline void nax_left_rows(thread L &left, const device X *p0, const device X *p1) {
  using x4 = vec<X, 4>;
  const x4 v0 = *(const device x4 *)p0;
  const x4 v1 = *(const device x4 *)p1;
  for (uint c = 0u; c < 4u; c++) {
    left[c] = v0[c];
    left[4 + c] = v1[c];
  }
}
template <typename D, typename Y>
static inline void nax_store(const thread D &dst, device Y *y, uint row, uint odim, uint obase, short2 origin, uint rem) {
  const size_t o0 = (size_t)(row + uint(origin.y)) * (size_t)odim;
  const size_t o1 = o0 + 8u * (size_t)odim;
  const bool l0 = uint(origin.y) < rem;
  const bool l1 = uint(origin.y) + 8u < rem;
  for (uint ct = 0u; ct < 2u; ct++) {
    for (uint c = 0u; c < 4u; c++) {
      const uint col = obase + ct * 16u + uint(origin.x) + c;
      if (col >= odim) continue;
      if (l0) y[o0 + col] = Y(dst[ct * 8u + c]);
      if (l1) y[o1 + col] = Y(dst[ct * 8u + 4u + c]);
    }
  }
}