const uint g    = threadgroup_position_in_grid.x;
const uint G    = threadgroups_per_grid.x;
const uint nb = (uint)scores_shape[2];
const int  vbi = bounds[row];
const uint vb  = (vbi > 0) ? (uint)vbi : 0u;
const uint lo  = (uint)(((ulong)g * (ulong)vb) / (ulong)G);
const uint hi  = (uint)(((ulong)(g + 1u) * (ulong)vb) / (ulong)G);
const uint nsl = hi - lo;
const auto sc = scores + (ulong)row * (ulong)nb; // `constant` under 8 elements
device int* outp = ids + (((ulong)row * (ulong)G) + (ulong)g) * (ulong)KTOP;
if (nsl <= KTOP) {
  for (uint i = tid; i < KTOP; i += TGN) outp[i] = (i < nsl) ? int(lo + i) : SENTINEL;
  return;
}
#define SEL_LO lo
#define SEL_HI hi
#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { out_idx = int(i); u = msv_qsa_ord(sc[i]); ok = 1u; } }
