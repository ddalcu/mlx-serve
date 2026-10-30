template <typename OT, int RD, int D, typename P>
inline OT dsv41_unrope(OT v, int d, int pos, P periods) {
  const float pv = metal::simd_shuffle_xor(float(v), ushort(1));
  if (RD == 0 || d < D - RD) return v;
  const bool even = (d - (D - RD)) % 2 == 0;
  const float x1 = even ? float(v) : pv, x2 = even ? pv : float(v);
  const float theta = float(pos) * (1.0f / periods[(d - (D - RD)) / 2]);
  const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
  return OT(even ? x1 * cs - x2 * sn : x1 * sn + x2 * cs);
}