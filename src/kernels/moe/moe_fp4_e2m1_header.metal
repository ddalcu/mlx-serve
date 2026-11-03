inline float mlxserve_e2m1(uint c) {
  return float(as_type<half>(ushort(((c & 0x7u) << 9) | ((c & 0x8u) << 12))));
}
