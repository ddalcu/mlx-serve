uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
if (c >= uint(H)) return;
size_t pbase = (size_t)b * 3 * H;
T g = proj[pbase + 2 * H + c] * proj[pbase + c];
float acc = 0.0;
size_t sbase = (size_t)b * (KS - 1) * H;
for (int i = 0; i < KS - 1; i++) {
  acc += static_cast<float>(state[sbase + i * H + c]) * w[c * KS + i];
  if (i + 1 < KS - 1) nstate[sbase + i * H + c] = state[sbase + (i + 1) * H + c];
}
acc += static_cast<float>(g) * w[c * KS + KS - 1];
nstate[sbase + (KS - 2) * H + c] = g;
y[(size_t)b * H + c] = proj[pbase + H + c] * static_cast<T>(acc);