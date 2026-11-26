// MLX's argmax order on (value, index): a NaN beats every number, the lower index among NaNs; else the larger value,
// the lower index among equal ones. A strict total order, so any reduction tree ends at MLX's argmax.
inline bool msv_ink2_better(float va, uint ia, float vb, uint ib) {
  if (isnan(va)) return !isnan(vb) || ia < ib;
  return !isnan(vb) && (va > vb || (va == vb && ia < ib));
}
