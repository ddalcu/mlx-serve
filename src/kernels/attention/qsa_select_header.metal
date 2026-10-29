// Monotone f32 -> uint32: ascending order preserved, NaN above every
// number, -0.0 and +0.0 the SAME key (torch compares them equal, and a
// relu sum can produce either).
inline uint msv_qsa_ord(float v) {
  if (metal::isnan(v)) { return 0xFFFFFFFFu; }
  if (v == 0.0f) { return 0x80000000u; }
  uint u = as_type<uint>(v);
  return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}
