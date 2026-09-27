// Token c of `prev ++ ids`, the history the n-gram hash walks. MLX binds a small input in the
// `constant` address space, so the pointer types are left open.
template <typename P, typename I>
static inline uint ple_tok(P prev, I ids, long ctx, long c) {
  return c < ctx ? uint(prev[c]) : uint(ids[c - ctx]);
}
// Little-endian reads at byte offsets: no region of ngram_table.bin is promised any alignment.
static inline ushort ple_u16(const device uint8_t* b, ulong o) {
  return ushort(b[o]) | ushort(ushort(b[o + 1]) << 8);
}
static inline ulong ple_word(const device uint8_t* b, ulong o, int n) {
  ulong v = 0;
  for (int k = 0; k < n; k++) v |= ulong(b[o + k]) << (8 * k);
  return v;
}
