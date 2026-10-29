inline float2 mlxserve_e4m3x2(uint w) {
  const uint v = w & 0x00FF00FFu;
  return float2(as_type<half2>(((v & 0x007F007Fu) << 7) | ((v & 0x00800080u) << 8)));
}
