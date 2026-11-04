inline float mlxserve_fp4_scale(uint b) {
  return as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
}
