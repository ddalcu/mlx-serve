const int l = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_threadgroup;
constexpr int MIX = (2 + HC) * HC;
const float e = eps[1];
const device float* zr = z + l * MIX;
if (lane < HC) {
  pre[l * HC + lane] = 1.0f / (1.0f + metal::exp(-zr[lane])) + e;
  post[l * HC + lane] = 2.0f / (1.0f + metal::exp(-zr[HC + lane]));
}
// Lane a * HC + b holds comb[a][b]; each sum gathers its HC terms in the serial order.
const bool act = lane < HC * HC;
const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
float v = zr[2 * HC + a * HC + b];
float mx = metal::simd_shuffle(v, a * HC);
for (ushort k = 1; k < HC; ++k) mx = metal::max(mx, metal::simd_shuffle(v, a * HC + k));
v = metal::exp(v - mx);
float sm = 0.0f;
for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, a * HC + k);
v = v / sm + e;
for (int t = 0; t < 2 * ITERS - 1; ++t) {
  const bool rows = t % 2 == 1;
  sm = 0.0f;
  for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, rows ? a * HC + k : k * HC + b);
  v = v / (sm + e);
}
if (act) comb[l * HC * HC + lane] = v;