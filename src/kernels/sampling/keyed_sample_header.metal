inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
inline float tf_val(uint k) { uint b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; return as_type<float>(b); }
inline ulong tf_mix(ulong x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9UL; x ^= x >> 27; x *= 0x94D049BB133111EBUL; return x ^ (x >> 31);
}
inline float tf_uniform(ulong seed, uint pos, uint id) {
  ulong x = tf_mix(seed + 0x9E3779B97F4A7C15UL);
  x = tf_mix(x ^ (ulong(pos) * 0xD1B54A32D192ED03UL));
  x = tf_mix(x ^ ulong(id));
  return (float(uint(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
}