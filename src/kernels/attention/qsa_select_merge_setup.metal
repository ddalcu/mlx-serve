const uint nb = (uint)scores_shape[2];
const uint nc = (uint)local_ids_shape[2];
const auto sc = scores + (ulong)row * (ulong)nb; // `constant` under 8 elements
const auto loc = local_ids + (ulong)row * (ulong)nc;
device int* outp = ids + (ulong)row * (ulong)KTOP;

uint nvalid = 0u;
for (uint base = 0u; base < nc; base += TGN) {
  const uint i = base + tid;
  uint v = 0u;
  if (i < nc) v = (loc[i] != SENTINEL) ? 1u : 0u;
  nvalid += metal::simd_sum(v);
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (lane == 0u) sgs[sg] = nvalid;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
{
  const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
  nvalid = metal::simd_sum(a);
}

if (nvalid <= KTOP) {
  for (uint i = tid; i < KTOP; i += TGN) outp[i] = SENTINEL;
  threadgroup_barrier(metal::mem_flags::mem_device);
  uint run = 0u;
  for (uint base = 0u; base < nc; base += TGN) {
    const uint i = base + tid;
    uint gtf = 0u;
    int idx = SENTINEL;
    if (i < nc) {
      idx = loc[i];
      gtf = (idx != SENTINEL) ? 1u : 0u;
    }
    const uint pg = metal::simd_prefix_exclusive_sum(gtf);
    const uint sg_g = metal::simd_sum(gtf);
    if (lane == 0u) sgs[sg] = sg_g;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint off_g = 0u, tot_g = 0u;
    {
      const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
      const uint pa = metal::simd_prefix_exclusive_sum(a);
      off_g = metal::simd_shuffle(pa, sg);
      tot_g = metal::simd_sum(a);
    }
    const uint pos = run + off_g + pg;
    if (gtf != 0u && pos < KTOP) outp[pos] = idx;
    run += tot_g;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  return;
}
#define SEL_LO 0u
#define SEL_HI nc
#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { out_idx = loc[i]; if (out_idx != SENTINEL) { u = msv_qsa_ord(sc[out_idx]); ok = 1u; } } }
