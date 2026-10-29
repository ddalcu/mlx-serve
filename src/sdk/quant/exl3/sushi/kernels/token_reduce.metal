uint col = uint(thread_position_in_grid.x);
uint row = uint(thread_position_in_grid.y);
if (col >= uint(ODIM)) return;
half acc = half(0.0f);
for (uint k = 0u; k < uint(TOPK); k++) {
  const uint orig = row * uint(TOPK) + k;
  const uint si = uint(inv[orig]);
  const half p = half(float(d[(size_t)si * (size_t)(ODIM) + col]) * float(half(sc[orig])));
  acc += p;
}
y[(size_t)row * (size_t)(ODIM) + col] = acc;