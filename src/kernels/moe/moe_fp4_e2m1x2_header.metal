inline float2 mlxserve_e2m1x2(uint w, int j) {
  const uint v = w >> (4 * j);
  return float2(as_type<half2>(((v & 0x70007u) << 9) | ((v & 0x80008u) << 12)));
}
