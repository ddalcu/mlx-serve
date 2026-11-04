inline void h2dec(uint u, thread half2* q, half o) {
  uint u6 = u >> 6;
  q[0] = as_type<half2>((u  & 0x00030003u) | 0x64006400u) - half2(1024.0h + o);
  q[1] = as_type<half2>((u  & 0x000C000Cu) | 0x5C005C00u) - half2(256.0h + o);
  q[2] = as_type<half2>((u  & 0x00300030u) | 0x54005400u) - half2(64.0h + o);
  q[3] = as_type<half2>((u  & 0x00C000C0u) | 0x4C004C00u) - half2(16.0h + o);
  q[4] = as_type<half2>((u  & 0x03000300u) | 0x44004400u) - half2(4.0h + o);
  q[5] = as_type<half2>((u6 & 0x00300030u) | 0x54005400u) - half2(64.0h + o);
  q[6] = as_type<half2>((u6 & 0x00C000C0u) | 0x4C004C00u) - half2(16.0h + o);
  q[7] = as_type<half2>((u6 & 0x03000300u) | 0x44004400u) - half2(4.0h + o);
}
inline float f2dot(thread const half2* q, thread const float2* xf) {
  float2 acc = float2(0);
  for (int j = 0; j < 8; ++j) acc = fma(float2(q[j]), xf[j], acc);
  return acc.x + acc.y;
}