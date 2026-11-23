// pf_route_rows (MLX_SERVE_DSV41_PF_ROUTE_PACK): a prompt's router top-K over E experts, route.metal's words with no
// threadgroup barrier. route.metal gives each row a threadgroup of E threads and runs K rounds of a simdgroup max, a
// threadgroup barrier and one thread's pass over the simdgroups' bests (2K barriers a row); here one simdgroup owns a
// row and SG rows share a threadgroup. Lane j holds experts i * 128 + 4 j + c (i < E / 128, c < 4: a float4 of the
// logits and of the bias each i, 128-bit loads), its scores computed by route.metal's expressions. A round takes the
// lane's best (the largest biased score, the lowest expert on a tie, as route.metal's simd_max then simd_min of the
// tied experts), then the simdgroup's: the same pick. The weights are route.metal's: the picks' scores summed in pick
// order from 0, the divide and the scale; with DS, times the picked expert's down scale `dsc` (`routeScaled`'s take and
// multiply: one f32 product of the weight and the scale, as there).
static_assert(E % 128 == 0 && E <= 1024, "a lane's float4 runs over the experts");
static_assert(K >= 1 && K <= 16, "the picks a lane keeps");
constexpr int NI = E / 128;
const uint lane = thread_index_in_simdgroup;
const int l = int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup);
const int L = logits_shape[0];
if (l >= L) return;
const device float4* zr = (const device float4*)(logits + size_t(l) * E);
const device float4* br = (const device float4*)(bias + size_t(bias_shape[0] == 1 ? 0 : l) * E);
float s[NI * 4];
float v[NI * 4];
for (int i = 0; i < NI; ++i) {
  const float4 z4 = zr[i * 32 + int(lane)];
  const float4 b4 = br[i * 32 + int(lane)];
  for (int c = 0; c < 4; ++c) {
    const float z = z4[c];
    const float y = metal::exp(-metal::abs(z));
    // log1p(y) below 1/4 as 2 atanh(y / (2 + y)): log(1 + y) loses y's low bits when 1 + y rounds.
    const float lt = y / (2.0f + y), lt2 = lt * lt;
    const float lp = y < 0.25f ? 2.0f * lt * (1.0f + lt2 * (1.0f / 3.0f + lt2 * (0.2f + lt2 * (1.0f / 7.0f + lt2 * (1.0f / 9.0f + lt2 * (1.0f / 11.0f)))))) : metal::log(1.0f + y);
    const float sv = metal::sqrt(metal::max(z, 0.0f) + lp);
    s[i * 4 + c] = sv;
    v[i * 4 + c] = sv + b4[c];
  }
}
int pick[K];
float ps[K];
for (int r = 0; r < K; ++r) {
  float b = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = 0; i < NI; ++i)
    for (int c = 0; c < 4; ++c) {
      const int e = i * 128 + 4 * int(lane) + c;
      const float x = v[i * 4 + c];
      if (x > b || (x == b && e < bi)) {
        b = x;
        bi = e;
      }
    }
  const float bv = metal::simd_max(b);
  const int at = metal::simd_min(b == bv ? bi : 0x7fffffff);
  // The owner drops the pick and hands over its unbiased score.
  const uint owner = uint((at % 128) / 4);
  float mine = 0.0f;
  for (int i = 0; i < NI; ++i)
    for (int c = 0; c < 4; ++c)
      if (i * 128 + 4 * int(lane) + c == at) {
        mine = s[i * 4 + c];
        v[i * 4 + c] = -INFINITY;
      }
  pick[r] = at;
  ps[r] = metal::simd_shuffle(mine, owner);
}
if (lane == 0) {
  float sum = 0.0f;
  for (int r = 0; r < K; ++r) sum += ps[r];
  const float div = NORM ? sum + 1e-20f : 1.0f;
  for (int r = 0; r < K; ++r) {
    idx[l * K + r] = uint(PLANT != 0 && r == 0 ? pick[K - 1] : pick[r]);
    const float wr = ps[r] / div * scal[0];
    wts[l * K + r] = DS != 0 ? wr * float(dsc[pick[r]]) : wr;
  }
}
