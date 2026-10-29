uint win = uint(threadgroup_position_in_grid.y);
uint sg = uint(simdgroup_index_in_threadgroup);
ushort lane = ushort(thread_index_in_simdgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
const uint start = wstarts[win];
const uint n = wnlive[win];
if (n == 0u || n > uint(WIN)) return;
const uint col0 = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
const ushort qid = lane >> 2;
const ushort fm = (qid & 4) + ((lane >> 1) & 3);
const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
const uint g = uint(fm) * 4u + uint(fn >> 1);
constexpr uint MB = (uint(WIN) + 7u) / 8u;
uint row = start;
const uint end = start + n;
while (row < end) {
const uint run0 = row;
const uint eid = uint(eids[row]);
uint run_end = row + 1u;
while (run_end < end && uint(eids[run_end]) == eid) run_end++;
const uint nlive = run_end - row;
size_t xo[4][2];
SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
  xo[mb][0] = (size_t)(run0 + min(mb * 8u + uint(fn), nlive - 1u)) * (size_t)(IDIM);
  xo[mb][1] = (size_t)(run0 + min(mb * 8u + uint(fn) + 1u, nlive - 1u)) * (size_t)(IDIM);
}
simdgroup_matrix<float, 8, 8> acc[4][4];
SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) {
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) acc[nb][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
const device uint *tiles = (const device uint *)(trellis + (size_t)eid * (size_t)IT * (size_t)OT * (size_t)N) + (col0 / TILE) * PACKED_W;
for (uint tk = 0u; tk < IT; tk++) {
  const uint kc = tk * TILE + uint(fm);
  simdgroup_matrix<half, 8, 8> b[2][4];
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    SMAT_UNROLL for (uint kb = 0u; kb < 2u; kb++) {
      b[kb][mb].thread_elements()[0] = x[xo[mb][0] + kc + kb * 8u];
      b[kb][mb].thread_elements()[1] = x[xo[mb][1] + kc + kb * 8u];
    }
  }
  const device uint *words = tiles + (size_t)tk * (size_t)OT * (size_t)PACKED_W;
  simdgroup_matrix<half, 8, 8> a[4][2];
  SMAT_UNROLL for (uint t = 0u; t < 2u; t++) {
    half2 p[4];
    smat_group<N>(words + t * PACKED_W, g, p);
    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) {
      a[2u * t + (j >> 1u)][j & 1u].thread_elements()[0] = p[j].x;
      a[2u * t + (j >> 1u)][j & 1u].thread_elements()[1] = p[j].y;
    }
  }
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    SMAT_UNROLL for (uint kb = 0u; kb < 2u; kb++) {
      SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) simdgroup_multiply_accumulate(acc[nb][mb], a[nb][kb], b[kb][mb], acc[nb][mb]);
    }
  }
}
SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) {
  const size_t oc = (size_t)(col0 + nb * 8u + uint(fm));
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    const uint m = mb * 8u + uint(fn);
    if (m < nlive) y[(size_t)(run0 + m) * (size_t)(ODIM) + oc] = half(acc[nb][mb].thread_elements()[0]);
    if (m + 1u < nlive) y[(size_t)(run0 + m + 1u) * (size_t)(ODIM) + oc] = half(acc[nb][mb].thread_elements()[1]);
  }
}
row = run_end;
}