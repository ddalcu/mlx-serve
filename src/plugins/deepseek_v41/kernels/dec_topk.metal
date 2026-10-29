// dec_topk: the K largest of each row of `scores` as key ids in ascending order (ties to the lower id), -1 past the
// row's reach. One threadgroup a row, the row in registers. GATHER: slot s holds the key cand[s / BS] * BS + s % BS.
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int row = threadgroup_position_in_grid.y, n = scores_shape[1];
const int lens = (meta[0] + row + 1) / meta[1];
const auto sr = scores + size_t(row) * n;
const auto cr = cand + size_t(row) * KB;
device int* out = picks + size_t(row) * K;
threadgroup atomic_uint hist[2 * 256 + 2];
threadgroup uint part[TG / 32];
uint kk[EPT];
#pragma clang loop unroll(full)
for (int e = 0; e < EPT; ++e) {
  const int s = (int(sg) * EPT + e) * 32 + int(lane);
  kk[e] = s < n && (!GATHER || cr[s / BS] >= 0) ? dec_key(sr[s]) : 0u;
}
const uint want = metal::min(uint(K), uint(n));
uint thr, rem, kept;
dec_radix<TG, EPT>(kk, want, hist, tid, lane, thr, rem);
uint bef[EPT];
dec_scan<TG, EPT>(kk, thr, thr, true, rem, part, sg, lane, bef, kept);
#pragma clang loop unroll(full)
for (int e = 0; e < EPT; ++e) {
  const bool gt = kk[e] > thr, eq = kk[e] == thr;
  const uint g = bef[e] >> 16, q = bef[e] & 0xFFFFu;
  if (gt || (eq && q < rem)) {
    const int s = (int(sg) * EPT + e) * 32 + int(lane);
    int id = s;
    if (GATHER) { const int b = cr[s / BS]; id = b >= 0 ? b * BS + s % BS : -1; }
    out[g + (gt ? metal::min(q, rem) : q)] = id >= 0 && id < lens ? id : -1;
  }
}
for (uint p = want + tid; p < K; p += TG) out[p] = -1;
