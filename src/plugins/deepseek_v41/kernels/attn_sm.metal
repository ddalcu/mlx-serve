// Row (l, h) of `attn_qk`'s scores: absent keys masked, scaled in f32, softmax with the head's sink in the denominator;
// probabilities [L, H, NK] in PT.
const uint lane = thread_index_in_simdgroup;
const int h = int(thread_position_in_grid.x / 32), l = int(thread_position_in_grid.y);
const int H = sink_shape[1];
const int wp = meta[0], W = meta[1], K = meta[3];
const int NK = W + K;
const float scale = scal[0];
const size_t rw = (size_t(l) * H + h) * NK;
float v[NJ];
float mx = -INFINITY;
#pragma clang loop unroll(full)
for (int j = 0; j < NJ; ++j) {
  const int c = int(lane) + 32 * j;
  float x = -INFINITY;
  if (c < W) {
    if (wp + l - (W - 1) + c >= 0) x = scale * s[rw + c];
  } else if (c < NK) {
    if (picks[size_t(l) * K + c - W] >= 0) x = scale * s[rw + c];
  }
  v[j] = x;
  mx = metal::max(mx, x);
}
mx = metal::simd_max(mx);
const float sk = sink[h];
const float M = metal::max(mx, sk);
float den = 0.0f;
#pragma clang loop unroll(full)
for (int j = 0; j < NJ; ++j) { const float p = metal::exp(v[j] - M); v[j] = p; den += p; }
den = metal::simd_sum(den) + metal::exp(sk - M);
const float inv = 1.0f / den;
#pragma clang loop unroll(full)
for (int j = 0; j < NJ; ++j) {
  const int c = int(lane) + 32 * j;
  if (c < NK) p[rw + c] = PT(v[j] * inv);
}
