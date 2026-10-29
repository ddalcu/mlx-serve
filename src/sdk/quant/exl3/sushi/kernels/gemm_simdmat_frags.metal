#define SMAT_UNROLL _Pragma("clang loop unroll(full)")
template<uint N>
static inline half2 smat_pair(uint f, uint s0, uint s1) {
  return exl3_pairh(uint2((f >> s0) & 0xffffu, (f >> s1) & 0xffffu));
}
// The group's weights 0..L come from one 32-bit funnel ending at weight L, the rest from
// one ending at weight 7: L is the widest split whose first codeword still fits.
static inline constexpr uint smat_split(uint N) {
  return exl3_end(N, 7u) - exl3_end(N, 0u) <= 16u ? 7u : exl3_end(N, 5u) - exl3_end(N, 0u) <= 16u ? 5u : 3u;
}
template<uint N>
static inline void smat_group(const device uint *words, uint g, thread half2 *p) {
  if (N == 64u) {
    const ulong m = ((ulong)words[(g + 31u) & 31u] << 32u) | (ulong)words[g];
    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) p[j] = smat_pair<N>(uint(m >> (24u - 8u * j)), 4u, 0u);
  } else if (exl3_end(N, 3u) - exl3_end(N, 0u) > 16u || N / 2u - exl3_end(N, 4u) > 16u) {
    constexpr uint W = N / 2u;
    const ulong lo = exl3_window(words, W * g + exl3_end(N, 3u), W, true);
    const ulong hi = exl3_window(words, W * g + W, W, true);
    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) {
      const ulong bits = j < 2u ? lo : hi;
      const uint end = j < 2u ? exl3_end(N, 3u) : W;
      p[j] = exl3_pairh(uint2(uint(bits >> (end - exl3_end(N, 2u * j))) & 0xffffu, uint(bits >> (end - exl3_end(N, 2u * j + 1u))) & 0xffffu));
    }
  } else {
    constexpr uint W = N / 2u;
    constexpr uint L = smat_split(N);
    const uint lo = uint(exl3_window(words, W * g + exl3_end(N, L), W, false));
    const uint hi = uint(exl3_window(words, W * g + W, W, false));
    constexpr uint E = exl3_end(N, L);
    p[0] = smat_pair<N>(lo, E - exl3_end(N, 0u), E - exl3_end(N, 1u));
    p[1] = smat_pair<N>(lo, E - exl3_end(N, 2u), E - exl3_end(N, 3u));
    p[2] = L >= 5u ? smat_pair<N>(lo, E - exl3_end(N, 4u), E - exl3_end(N, 5u)) : smat_pair<N>(hi, W - exl3_end(N, 4u), W - exl3_end(N, 5u));
    p[3] = L == 7u ? smat_pair<N>(lo, E - exl3_end(N, 6u), 0u) : smat_pair<N>(hi, W - exl3_end(N, 6u), 0u);
  }
}