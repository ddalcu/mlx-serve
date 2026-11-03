// dec_topk_merge: a long row's chunks (their KBLK largest blocks each, in id order) to its candidate blocks and picks.
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int row = threadgroup_position_in_grid.y, n = scores_shape[1];
const int lens = (meta[0] + row + 1) / meta[1];
const uint nblk = (n + BS - 1) / BS;
const uint len = uint(ck_shape[1]) * KBLK;
const auto sr = scores + size_t(row) * n;
const auto kr = ck + size_t(row) * len;
const auto ir = cid + size_t(row) * len;
threadgroup atomic_uint hist[2 * 256 + 2];
threadgroup uint part[TG / 32];
threadgroup int cs[KBLK];
uint bk[EPT];
int bid[EPT];
#pragma clang loop unroll(full)
for (int e = 0; e < EPT; ++e) {
  const uint i = (sg * EPT + e) * 32 + lane;
  bk[e] = i < len ? kr[i] : 0u;
  bid[e] = i < len ? ir[i] : -1;
}
dec_pick_blocks<TG, EPT, K, KBLK, BS>(bk, bid, nblk, sr, n, lens, hist, part, cs, cand + size_t(row) * KBLK, picks + size_t(row) * K, tid, lane, sg);
