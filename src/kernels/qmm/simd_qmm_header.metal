#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
inline float bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
inline float sum8(uint4 v, float one) {
  float t = bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(bf8(v, e), one, t);
  return t;
}
// Value 8j + p of a 64-group's words (j = k-slot, p = pass): 4 and 8 bits
// masked in place, the input pre-scaled by wpre(p) to cancel the shift;
// 6 bits straddle words, so they are shifted down and wpre is 1.
template <int BITS> inline float wpre(int p) {
  return BITS == 6 ? 1.0f : as_type<float>(uint(127 - (BITS == 4 ? 4 * p : 8 * (p % 4))) << 23);
}
template <int BITS> inline float wfield(const thread uint* lw, int j, int p) {
  const int bit = (8 * j + p) * BITS, off = bit % 32;
  const uint w0 = lw[bit / 32];
  if (BITS == 6) return float((off <= 26 ? w0 >> off : (w0 >> off) | (lw[bit / 32 + 1] << (32 - off))) & 63u);
  return float(w0 & (((1u << BITS) - 1u) << off));
}