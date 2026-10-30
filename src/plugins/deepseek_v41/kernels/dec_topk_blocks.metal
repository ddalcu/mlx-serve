// dec_topk_blocks: a row of up to TG * BPT blocks in one threadgroup: the candidate blocks (the newest group's block
// pinned) and the picks inside them.
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int row = threadgroup_position_in_grid.y, n = scores_shape[1];
const int lens = (meta[0] + row + 1) / meta[1];
const int pinned = lens > 0 ? (lens - 1) / BS : -1;
const uint nblk = (n + BS - 1) / BS;
const auto sr = scores + size_t(row) * n;
threadgroup atomic_uint hist[2 * 256 + 2];
threadgroup uint part[TG / 32];
threadgroup int cs[KBLK];
uint bk[BPT];
int bid[BPT];
#pragma clang loop unroll(full)
for (int e = 0; e < BPT; ++e) {
  const int b = (int(sg) * BPT + e) * 32 + int(lane);
  bid[e] = b;
  bk[e] = b >= int(nblk) ? 0u : (b == pinned ? DEC_KEY_PINF : dec_block_key<BS>(sr, n, b));
}
dec_pick_blocks<TG, BPT, K, KBLK, BS>(bk, bid, nblk, sr, n, lens, hist, part, cs, cand + size_t(row) * KBLK, picks + size_t(row) * K, tid, lane, sg);
