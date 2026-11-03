// dec_topk_chunk: chunk c of a long row, TG * BPT blocks: its KBLK largest blocks as (key, id) pairs in id order, for
// dec_topk_merge to select from.
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int c = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
const int n = scores_shape[1], lens = (meta[0] + row + 1) / meta[1];
const int pinned = lens > 0 ? (lens - 1) / BS : -1;
const int nblk = (n + BS - 1) / BS, first = c * TG * BPT, C = (nblk + TG * BPT - 1) / (TG * BPT);
const auto sr = scores + size_t(row) * n;
device uint* kout = ck + (size_t(row) * C + c) * KBLK;
device int* iout = cid + (size_t(row) * C + c) * KBLK;
threadgroup atomic_uint hist[2 * 256 + 2];
threadgroup uint part[TG / 32];
uint bk[BPT];
int bid[BPT];
#pragma clang loop unroll(full)
for (int e = 0; e < BPT; ++e) {
  const int b = first + (int(sg) * BPT + e) * 32 + int(lane);
  bid[e] = b;
  bk[e] = b >= nblk ? 0u : (b == pinned ? DEC_KEY_PINF : dec_block_key<BS>(sr, n, b));
}
uint thr, rem, kept;
dec_radix<TG, BPT>(bk, metal::min(uint(KBLK), uint(nblk - first)), hist, tid, lane, thr, rem);
uint bef[BPT];
dec_scan<TG, BPT>(bk, thr, thr, true, rem, part, sg, lane, bef, kept);
#pragma clang loop unroll(full)
for (int e = 0; e < BPT; ++e) {
  const bool gt = bk[e] > thr, eq = bk[e] == thr;
  const uint g = bef[e] >> 16, q = bef[e] & 0xFFFFu;
  if (gt || (eq && q < rem)) {
    const uint pos = g + (gt ? metal::min(q, rem) : q);
    kout[pos] = bk[e];
    iout[pos] = bid[e];
  }
}
for (uint p = kept + tid; p < KBLK; p += TG) { kout[p] = 0u; iout[p] = -1; }
