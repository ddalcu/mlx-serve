// `dsv41_moe_gu` over GGUF expert banks `wg`, `wu` [experts, N, ROWB] (32-weight units), a z per decode row and slot:
// threadgroup (y, z) is a simdgroup's output row of pair z (row z / TOPK's slot z % TOPK), or past the rows' pairs of
// row z - rows * TOPK's shared expert (affine or mxfp8 rows [gate N | up N]). A routed lane loads a unit of x once for
// the gate and the up.
const uint lane = thread_index_in_simdgroup;
const uint n0 = threadgroup_position_in_grid.y * simdgroups_per_threadgroup + simdgroup_index_in_threadgroup;
const uint z = threadgroup_position_in_grid.z;
if (n0 >= N) return;
const uint pairs = threadgroups_per_grid.z / (TOPK + 1) * TOPK;
const bool routed = z < pairs;
const device T* xl = x + size_t(routed ? z / TOPK : z - pairs) * K;
float g = 0.0f, v = 0.0f;
if (routed) {
  const size_t eb = size_t(inds[z]) * N * ROWB;
  for (uint u = lane; u < K / UNIT; u += 32) {
    GgDotR o;
    for (uint i = 0; i < UNIT / 4; ++i) o.x[i] = float4(*(const device vec<T, 4>*)(xl + u * UNIT + 4 * i));
    o.acc = 0.0f;
    DEQ_AT(wg + eb, n0, u / UPB, u % UPB, o);
    g += o.acc.x + o.acc.y + o.acc.z + o.acc.w;
    o.acc = 0.0f;
    DEQ_AT(wu + eb, n0, u / UPB, u % UPB, o);
    v += o.acc.x + o.acc.y + o.acc.z + o.acc.w;
  }
  g = metal::simd_sum(g);
  v = metal::simd_sum(v);
} else {
  float a[2];
  size_t rows[2] = {n0, N + n0};
  dsv41_dots<SBITS, SGS, K, 2, SMX>(sw, sscales, sbiases, rows, xl, lane, a);
  g = a[0];
  v = a[1];
}
if (lane == 0) {
  float gg = float(T(g)), uu = float(T(v));
  if (lim[0] > 0.0f) { gg = metal::min(gg, lim[0]); uu = metal::clamp(uu, -lim[0], lim[0]); }
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gg)));
  out[size_t(z) * N + n0] = T((gg * (gg < 0.0f ? y : 1.0f - y)) * uu * (routed ? wts[z] : 1.0f));
}
