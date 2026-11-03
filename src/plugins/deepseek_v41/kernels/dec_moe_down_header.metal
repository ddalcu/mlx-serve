inline float2 dsv41_bf2(uint w) {
  return float2(as_type<float>(w << 16), as_type<float>(w & 0xFFFF0000u));
}

// A 3-bit group's dot over 32 values (3 words, value i at bit 3i) against their 34 prescaled terms: values 0..9 in word 0,
// 10 = (word 0 top two bits, word 1 bit 0), 11..20 in word 1 (shifts 1, 4, ..), 21 = (word 1 bit 31, word 2 bits 0..1),
// 22..31 in word 2 (shifts 2, 5, ..). A term is its value times 2^-shift, so a field is masked, never shifted.
inline float dsv41_dot96(const thread float* p, uint w0, uint w1, uint w2, float d) {
  for (int i = 0; i < 10; ++i) d = metal::fma(p[i], float(w0 & (7u << (3 * i))), d);
  d = metal::fma(p[10], float(w0 & 0xC0000000u), d);
  d = metal::fma(p[11], float(w1 & 0x1u), d);
  for (int i = 0; i < 10; ++i) d = metal::fma(p[12 + i], float(w1 & (7u << (1 + 3 * i))), d);
  d = metal::fma(p[22], float(w1 & 0x80000000u), d);
  d = metal::fma(p[23], float(w2 & 0x3u), d);
  for (int i = 0; i < 10; ++i) d = metal::fma(p[24 + i], float(w2 & (7u << (2 + 3 * i))), d);
  return d;
}
