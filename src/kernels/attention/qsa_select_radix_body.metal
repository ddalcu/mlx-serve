uint pref = 0u;
uint fixed = 0u;
uint need = KTOP;
uint T = 0u;
uint need_eq = 0u;

for (uint lv = 0u; lv < NLEV; ++lv) {
  const uint width = (DIGITS == 8) ? 8u : ((lv == 2u) ? 10u : 11u);
  const uint nbins = 1u << width;
  const uint shift = 32u - fixed - width;
  const uint hi_b  = (fixed == 0u) ? 31u : (32u - fixed);

  for (uint b = tid; b < nbins; b += TGN) metal::atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed);
  if (tid == 0u) { sh[0] = 0u; sh[1] = 0u; sh[2] = 0u; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  uint last_d = 0xFFFFFFFFu;
  uint run = 0u;
  for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
    const uint i = base + tid;
    uint u = 0u;
    int out_idx = SENTINEL;
    uint ok = 0u;
    SEL_LOAD(i, u, out_idx, ok)
    if (ok != 0u) {
      if (!(fixed != 0u && (u >> hi_b) != pref)) {
        const uint d = (u >> shift) & (nbins - 1u);
        if (d == last_d) { run += 1u; }
        else {
          if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
          last_d = d;
          run = 1u;
        }
      }
    }
  }
  if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (sg == 0u) {
    const uint chunk = nbins / SIMDW;
    const uint base  = nbins - (lane + 1u) * chunk;
    uint tot = 0u;
    for (uint j = 0u; j < chunk; ++j) tot += metal::atomic_load_explicit(&hist[base + j], metal::memory_order_relaxed);
    const uint pre = metal::simd_prefix_exclusive_sum(tot);
    if (pre < need && need <= pre + tot) {
      uint acc = pre;
      for (uint jj = chunk; jj > 0u; --jj) {
        const uint bidx = base + jj - 1u;
        const uint c = metal::atomic_load_explicit(&hist[bidx], metal::memory_order_relaxed);
        if (acc + c >= need) { sh[0] = bidx; sh[1] = acc; sh[2] = c; break; }
        acc += c;
      }
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint d_sel   = sh[0];
  const uint above_w = sh[1];
  const uint cnt_d   = sh[2];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  need = need - above_w;
  pref = (pref << width) | d_sel;
  fixed += width;
  if (cnt_d == need || fixed >= 32u) {
    T = (fixed >= 32u) ? pref : (pref << (32u - fixed));
    need_eq = need;
    break;
  }
}

for (uint base = 0u; base < KTOP; base += TGN) {
  const uint i = base + tid;
  if (i < KTOP) outp[i] = SENTINEL;
}
threadgroup_barrier(metal::mem_flags::mem_device);

uint run_gt = 0u;
uint run_eq = 0u;
for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
  const uint i = base + tid;
  uint gtf = 0u;
  uint e = 0u;
  int out_idx = SENTINEL;
  uint u = 0u;
  uint ok = 0u;
  SEL_LOAD(i, u, out_idx, ok)
  if (ok != 0u) {
    gtf = (u > T) ? 1u : 0u;
    e = (u == T) ? 1u : 0u;
  }
  const uint pg = metal::simd_prefix_exclusive_sum(gtf);
  const uint pe = metal::simd_prefix_exclusive_sum(e);
  const uint sg_g = metal::simd_sum(gtf);
  const uint sg_e = metal::simd_sum(e);
  if (lane == 0u) { sgs[sg] = sg_g; sgs[NSIMD + sg] = sg_e; }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  uint off_g = 0u, off_e = 0u, tot_g = 0u, tot_e = 0u;
  {
    const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
    const uint b = (lane < NSIMD) ? sgs[NSIMD + lane] : 0u;
    const uint pa = metal::simd_prefix_exclusive_sum(a);
    const uint pb = metal::simd_prefix_exclusive_sum(b);
    off_g = metal::simd_shuffle(pa, sg);
    off_e = metal::simd_shuffle(pb, sg);
    tot_g = metal::simd_sum(a);
    tot_e = metal::simd_sum(b);
  }
  const uint gb = run_gt + off_g + pg;
  const uint eb = run_eq + off_e + pe;
  if (ok != 0u) {
    if (gtf != 0u) {
      const uint pos = gb + metal::min(eb, need_eq);
      if (pos < KTOP) outp[pos] = out_idx;
    } else if (e != 0u && eb < need_eq) {
      const uint pos = gb + eb;
      if (pos < KTOP) outp[pos] = out_idx;
    }
  }
  run_gt += tot_g;
  run_eq += tot_e;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (run_gt + metal::min(run_eq, need_eq) >= KTOP) break;
}