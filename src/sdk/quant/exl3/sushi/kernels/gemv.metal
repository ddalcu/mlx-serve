uint gid = uint(thread_position_in_grid.x);
if (gid >= uint(ODIM)) return;
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_HW = N;
constexpr uint word_count = N / 2u;
constexpr uint IN_TILES = uint(IDIM) / TILE;
constexpr uint OUT_TILES = uint(ODIM) / TILE;
const uint tn = gid / TILE;
const uint local = gid % TILE;
float acc = 0.0f;
for (uint tk = 0u; tk < IN_TILES; tk++) {
  const device ushort* tile = trellis + (tk * OUT_TILES + tn) * PACKED_HW;
  ushort cw[256];
  for (uint th = 0u; th < 128u; th++) {
    const exl3_win wv = exl3_pair_window(th * 2u, N);
    const uint a = uint(tile[wv.i0 * 2u]) | (uint(tile[wv.i0 * 2u + 1u]) << 16u);
    const uint b = uint(tile[wv.i1 * 2u]) | (uint(tile[wv.i1 * 2u + 1u]) << 16u);
    const ulong merged = (ulong(a) << 32) | ulong(b);
    const uint funnel = uint(merged >> wv.sh);
    cw[th * 2u] = ushort((funnel >> wv.fresh) & 0xffffu);
    cw[th * 2u + 1u] = ushort(funnel & 0xffffu);
  }
  for (uint slot = 0u; slot < 256u; slot++) {
    const uint lane = slot / 8u;
    const uint s = slot % 8u;
    const uint row0 = (lane & 3u) * 2u;
    const uint col0 = lane >> 2u;
    uint pos;
    switch (s) {
      case 0u: pos = row0 * 16u + col0; break;
      case 1u: pos = (row0 + 1u) * 16u + col0; break;
      case 2u: pos = (row0 + 8u) * 16u + col0; break;
      case 3u: pos = (row0 + 9u) * 16u + col0; break;
      case 4u: pos = row0 * 16u + col0 + 8u; break;
      case 5u: pos = (row0 + 1u) * 16u + col0 + 8u; break;
      case 6u: pos = (row0 + 8u) * 16u + col0 + 8u; break;
      default: pos = (row0 + 9u) * 16u + col0 + 8u; break;
    }
    if ((pos % 16u) != local) continue;
    const float w = exl3_slot(cw, slot);
    acc += float(x[tk * TILE + (pos / 16u)]) * w;
  }
}
y[gid] = half(exl3_out(acc));