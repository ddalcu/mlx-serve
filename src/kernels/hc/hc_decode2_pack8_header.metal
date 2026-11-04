template <typename T, bool DENSE>
inline float hc_val(uint q) {
  if (DENSE) return float(as_type<T>(ushort(q)));
  return float(q);
}
template <int B>
inline ulong hc_pack8(const device uchar* p) {
  ulong r = 0;
  for (int b = 0; b < B; ++b) r |= ulong(p[b]) << (8 * b);
  return r;
}