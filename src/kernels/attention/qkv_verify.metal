constexpr int VPW = 32 / BITS;
constexpr int NT = 128;
constexpr int LQ = GQA * TQ;

const uint tid = thread_position_in_threadgroup.x;
const uint hkv = threadgroup_position_in_grid.y;
const uint blk = threadgroup_position_in_grid.z;
const uint nblk = threadgroups_per_grid.z;

const int Tk = tk[0];
const float scale = scl[0];
const uint mask_bits = (1u << BITS) - 1u;
const int t0 = int(blk) * BLOCK;
const int rows = min(BLOCK, Tk - t0);

threadgroup float p_tg[LQ * BLOCK];
threadgroup float red[NT];
threadgroup float ml_tg[2 * LQ];

const long kq_base = (long)hkv * (long)kq_strides[1];
const long ks_base = (long)hkv * (long)ksc_strides[1];
const long kb_base = (long)hkv * (long)kbi_strides[1];

// Phase A: one thread per kv row (strided): dequant the row once, dot
// against all LQ q rows; causal tail from geometry at score write.
for (int t = int(tid); t < rows; t += NT) {
  const long row = kq_base + (long)(t0 + t) * (long)kq_strides[2];
  const long grow = ks_base + (long)(t0 + t) * (long)ksc_strides[2];
  const long brow = kb_base + (long)(t0 + t) * (long)kbi_strides[2];
  float sc[LQ];
  for (int r = 0; r < LQ; ++r) sc[r] = 0.0f;
  for (int wi = 0; wi < DK / VPW; ++wi) {
    const uint w = uint(kq[row + (long)wi]);
    const int g_idx = (wi * VPW) / GS;
    const float sj = float(ksc[grow + (long)g_idx]);
    const float bj = float(kbi[brow + (long)g_idx]);
    for (int u = 0; u < VPW; ++u) {
      const float kv = float((w >> (u * BITS)) & mask_bits) * sj + bj;
      const int e = wi * VPW + u;
      for (int gi = 0; gi < GQA; ++gi) {
        const long qh = ((long)hkv * GQA + gi) * (long)q_strides[1];
        for (int tq = 0; tq < TQ; ++tq)
          sc[gi * TQ + tq] += float(q[qh + (long)tq * (long)q_strides[2] + (long)e]) * kv;
      }
    }
  }
  const int gpos = t0 + t;
  for (int gi = 0; gi < GQA; ++gi)
    for (int tq = 0; tq < TQ; ++tq)
      p_tg[(gi * TQ + tq) * BLOCK + t] =
        (gpos < Tk - TQ + 1 + tq) ? sc[gi * TQ + tq] * scale : -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Phase M: per q row — block max, exp, row-weight sum (tree in red[]).
for (int r = 0; r < LQ; ++r) {
  float pm = -INFINITY;
  for (int t = int(tid); t < rows; t += NT) pm = max(pm, p_tg[r * BLOCK + t]);
  red[tid] = pm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int off = NT / 2; off > 0; off >>= 1) {
    if (int(tid) < off) red[tid] = max(red[tid], red[tid + off]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Clamp: a fully-tail-masked block row otherwise feeds exp(-inf - -inf).
  const float M = max(red[0], -1e30f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ls = 0.0f;
  for (int t = int(tid); t < rows; t += NT) {
    const float p = metal::exp(p_tg[r * BLOCK + t] - M);
    p_tg[r * BLOCK + t] = p;
    ls += p;
  }
  red[tid] = ls;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int off = NT / 2; off > 0; off >>= 1) {
    if (int(tid) < off) red[tid] += red[tid + off];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) { ml_tg[r] = M; ml_tg[LQ + r] = red[0]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Partial m/l out: [Hq*TQ, NBLK], rows (head-major, tq inner).
if (tid < uint(LQ)) {
  const long r = (long)hkv * LQ + int(tid);
  m_out[r * (long)nblk + (long)blk] = ml_tg[tid];
  l_out[r * (long)nblk + (long)blk] = ml_tg[LQ + tid];
}

const long vq_base = (long)hkv * (long)vq_strides[1];
const long vs_base = (long)hkv * (long)vsc_strides[1];
const long vb_base = (long)hkv * (long)vbi_strides[1];

// Phase B: one thread per output element: walk the block's V rows,
// dequant each element ONCE, feed all LQ row accumulators.
for (int e = int(tid); e < DV; e += NT) {
  float accg[LQ];
  for (int r = 0; r < LQ; ++r) accg[r] = 0.0f;
  const int wi = e / VPW;
  const int sh = (e % VPW) * BITS;
  const int g_idx = e / GS;
  for (int t = 0; t < rows; ++t) {
    const long row = vq_base + (long)(t0 + t) * (long)vq_strides[2];
    const uint w = uint(vq[row + (long)wi]);
    const float vv = float((w >> sh) & mask_bits) * float(vsc[vs_base + (long)(t0 + t) * (long)vsc_strides[2] + (long)g_idx]) + float(vbi[vb_base + (long)(t0 + t) * (long)vbi_strides[2] + (long)g_idx]);
    for (int r = 0; r < LQ; ++r) accg[r] += p_tg[r * BLOCK + t] * vv;
  }
  for (int r = 0; r < LQ; ++r) {
    const long orow = (long)hkv * LQ + r;
    o_out[(orow * (long)nblk + (long)blk) * (long)DV + (long)e] = accg[r];
  }
}