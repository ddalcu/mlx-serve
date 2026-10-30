const uint nb = (uint)scores_shape[2];
const int  vbi = bounds[row];
const uint vb  = (vbi > 0) ? (uint)vbi : 0u;
const auto sc = scores + (ulong)row * (ulong)nb; // `constant` under 8 elements
device int* outp = ids + (ulong)row * (ulong)KTOP;
if (vb <= KTOP) {
  for (uint i = tid; i < KTOP; i += TGN) outp[i] = (i < vb) ? int(i) : SENTINEL;
  return;
}
#define SEL_LO 0u
#define SEL_HI vb
#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { out_idx = int(i); u = msv_qsa_ord(sc[i]); ok = 1u; } }
