// engram_rows: a prompt chunk's Engram table rows gathered by id and dequantized to T, out[i, :] = row ids[i] of
// `codes` / `tsc` (bytes where `geo` puts them, as dec_engram_kv reads them: e2m1 nibbles at TB 4 or e4m3 bytes at
// TB 8, an e8m0 scale per 32, MLX's reads). A thread takes 16 values of a row; the grid is exactly the rows' values.
const ulong wbase = ulong(geo[0]) | (ulong(geo[1]) << 32);
const ulong sbase = ulong(geo[3]) | (ulong(geo[4]) << 32);
const uint gid = thread_position_in_grid.x;
const uint i = gid / uint(HD / 16);
const int p = int(gid % uint(HD / 16)) * 16;
const ulong id = ulong(ids[i]);
const device uchar* rc = (const device uchar*)codes + wbase + id * ulong(geo[2]);
const float e = mlxserve_fp4_e8m0(uint(((const device uchar*)tsc)[sbase + id * ulong(geo[5]) + p / 32]));
device T* o = out + size_t(i) * HD + p;
if constexpr (TB == 4) {
  const uint2 cw = *(const device uint2*)(rc + p / 2);
  for (int k = 0; k < 8; ++k) {
    o[k] = T(mlxserve_fp4_e2m1_value(cw.x >> (4 * k)) * e);
    o[8 + k] = T(mlxserve_fp4_e2m1_value(cw.y >> (4 * k)) * e);
  }
} else {
  const uint4 cw = *(const device uint4*)(rc + p);
  const uint words[4] = {cw.x, cw.y, cw.z, cw.w};
  for (int k = 0; k < 4; ++k) {
    const float2 lo = mlxserve_e4m3x2(words[k]);
    const float2 hi = mlxserve_e4m3x2(words[k] >> 8);
    o[4 * k + 0] = T((lo.x * 256.0f) * e);
    o[4 * k + 1] = T((hi.x * 256.0f) * e);
    o[4 * k + 2] = T((lo.y * 256.0f) * e);
    o[4 * k + 3] = T((hi.y * 256.0f) * e);
  }
}
