// MLX_SERVE_DSV41_PF_KV_ROWS: kv_row.metal's rms, rope and fp8 / 32 round trip with one simdgroup a row, NSG rows a
// threadgroup, and D (the row width) and RD compiled in. Lane `lane` holds element g * 32 + lane of each 32-wide group
// g, so every sum and max runs over the same 32 elements as kv_row's simdgroups: the squares' simd_sum a group, the
// groups' partials added in order, the fp8 scale's simd_max a group; a rope pair (b, b + 1) is lanes (2i, 2i + 1) of
// one group (a shuffle). Every value is kv_row's expression: the same words, with no threadgroup memory or barrier.
constexpr int G = D / 32;
const uint lane = thread_index_in_simdgroup;
const int l = int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup);
if (l >= x_shape[0]) return;
const size_t base = size_t(l) * D;
float xv[G];
for (int g = 0; g < G; ++g) xv[g] = float(x[base + g * 32 + lane]);
float tot = 0.0f;
for (int g = 0; g < G; ++g) tot += metal::simd_sum(xv[g] * xv[g]);
const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
constexpr int head = D - RD;
for (int g = 0; g < G; ++g) {
  const int d = g * 32 + int(lane);
  const T y = w[d] * T(xv[g] * inv);
  float r = float(y);
  const float yo = metal::simd_shuffle_xor(float(y), ushort(1));
  if (d >= head) {
    const int p = (d - head) / 2;
    const bool even = (d - head) % 2 == 0;
    const float x1 = even ? float(y) : yo, x2 = even ? yo : float(y);
    const float theta = float(l + off) * (1.0f / periods[p]);
    const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
    r = float(T(even ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
  }
  if (QAT) {
    const float scale = dsv41_pow2ceil(metal::max(metal::simd_max(metal::abs(r)), 1e-4f) * (1.0f / 448.0f));
    r = dsv41_fp8_roundtrip(metal::clamp(r / scale, -448.0f, 448.0f)) * scale;
  }
  out[base + d] = T(r);
}
