// dec_engram_kv: an Engram layer's wkv over R decode rows read straight from the table's codes, kv[R, N] = x . W^T.
// Row r of x is its K / HD table rows (`codes`, `tsc`: e2m1 nibbles at TB 4 or e4m3 bytes at TB 8, an e8m0 scale per
// 32, MLX's reads), dequantized in the launch into threadgroup memory CH inputs at a time; W is mxfp8 (e4m3 words, an
// e8m0 exponent per 32). A threadgroup is SG simdgroups of RT output rows each; a lane takes 16 weights of a row per
// 512 inputs, decoded once and applied to every row of x.
constexpr int SG = 8;
constexpr int RT = 2;
constexpr int CH = 1024;
constexpr int KW = K / 4;
constexpr int KG = K / 32;
constexpr int TW = HD * TB / 32;
constexpr int TS = HD / 32;
threadgroup float xs[R * CH];
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = thread_index_in_threadgroup;
const int n0 = int(threadgroup_position_in_grid.x) * (SG * RT) + int(sg) * RT;
float acc[RT][R];
for (int j = 0; j < RT; ++j)
  for (int r = 0; r < R; ++r) acc[j][r] = 0.0f;
for (int c = 0; c < K; c += CH) {
  // Task q: inputs [k, k + 16) of row r, k = c + s * 16, stored where lane s % 32 of half s / 32 reads them (input i
  // of a lane's 16 at i * 32 + lane: no bank conflicts on either side).
  for (int q = int(tid); q < R * (CH / 16); q += SG * 32) {
    const int r = q / (CH / 16);
    const int s = q % (CH / 16);
    const int k = c + s * 16;
    const int row = r * (K / HD) + k / HD;
    const int p = k % HD;
    const float e = mlxserve_fp4_e8m0(uint(tsc[row * TS + p / 32]));
    float v[16];
    if constexpr (TB == 4) {
      const uint2 cw = *(const device uint2*)(codes + row * TW + p / 8);
      for (int i = 0; i < 8; ++i) {
        v[i] = (mlxserve_e2m1(cw.x >> (4 * i)) * 16384.0f) * e;
        v[8 + i] = (mlxserve_e2m1(cw.y >> (4 * i)) * 16384.0f) * e;
      }
    } else {
      const uint4 cw = *(const device uint4*)(codes + row * TW + p / 4);
      const uint words[4] = {cw.x, cw.y, cw.z, cw.w};
      for (int i = 0; i < 4; ++i) {
        const float2 lo = mlxserve_e4m3x2(words[i]);
        const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
        v[4 * i + 0] = (lo.x * 256.0f) * e;
        v[4 * i + 1] = (hi.x * 256.0f) * e;
        v[4 * i + 2] = (lo.y * 256.0f) * e;
        v[4 * i + 3] = (hi.y * 256.0f) * e;
      }
    }
    threadgroup float* dst = xs + r * CH + (s / 32) * 512 + (s % 32);
    for (int i = 0; i < 16; ++i) dst[i * 32] = v[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int h = 0; h < CH / 512; ++h) {
    const int k = c + h * 512 + int(lane) * 16;
    for (int j = 0; j < RT; ++j) {
      const int n = n0 + j;
      const uint4 qw = ((const device uint4*)(w + (size_t)n * KW))[k / 16];
      const float sc = mlxserve_fp4_e8m0(uint(wsc[(size_t)n * KG + k / 32]));
      float wv[16];
      const uint words[4] = {qw.x, qw.y, qw.z, qw.w};
      for (int i = 0; i < 4; ++i) {
        const float2 lo = mlxserve_e4m3x2(words[i]);
        const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
        wv[4 * i + 0] = lo.x;
        wv[4 * i + 1] = hi.x;
        wv[4 * i + 2] = lo.y;
        wv[4 * i + 3] = hi.y;
      }
      for (int r = 0; r < R; ++r) {
        const threadgroup float* xr = xs + r * CH + h * 512 + lane;
        float d = 0.0f;
        for (int i = 0; i < 16; ++i) d += xr[i * 32] * wv[i];
        acc[j][r] += (d * 256.0f) * sc;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (int j = 0; j < RT; ++j)
  for (int r = 0; r < R; ++r) {
    const float v = simd_sum(acc[j][r]);
    if (lane == 0) kv[(size_t)r * N + n0 + j] = T(v);
  }
