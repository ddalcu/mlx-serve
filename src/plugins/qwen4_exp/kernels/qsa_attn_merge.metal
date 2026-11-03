constexpr int BD = 256;
const int d = int(thread_index_in_threadgroup);
const int s = int(threadgroup_position_in_grid.y);
const int hq = int(threadgroup_position_in_grid.z);
const int qL = pacc_shape[1];
const int NS = pacc_shape[2];
const long pbase = ((long)hq * (long)qL + (long)s) * NS;

// Every thread reads all NS (m, l) pairs. NS <= 64 and the pairs land in
// cache after the first thread of the threadgroup touches them, which is
// cheaper than a barrier plus threadgroup storage for two floats.
float M = -INFINITY;
for (int j = 0; j < NS; ++j) {
  if (pml[(pbase + j) * 2 + 1] > 0.0f) M = metal::max(M, pml[(pbase + j) * 2]);
}
float Z = 0.0f;
float acc = 0.0f;
for (int j = 0; j < NS; ++j) {
  const float lj = pml[(pbase + j) * 2 + 1];
  if (lj <= 0.0f) continue;
  const float w = metal::exp2(pml[(pbase + j) * 2] - M);
  Z += w * lj;
  acc += w * pacc[(pbase + j) * BD + d];
}
out[((long)hq * (long)qL + (long)s) * BD + d] = T(acc / Z);