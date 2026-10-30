inline float mlxserve_fp4_scale(uint b) {
  return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8))));
}
