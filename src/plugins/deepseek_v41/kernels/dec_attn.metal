// dec_attn: decode attention partials on the tensor units. A threadgroup is one query row, one split and 16 heads: the
// split's keys are the row's window slots, then its picks, cut into S runs of at most BK. Simdgroup j scores key
// quarter j, then owns dims 128 j .. 128 j + 127 of P V. pacc is a run's unnormalised weighted keys, pm its max score
// (-inf: no key counted) and pl its weight sum, the partials `attn_merge` combines.
constexpr int D = 512;
constexpr int BK = 128;
constexpr int RING = 1 << 24;

const int p0 = meta[0], nc = meta[1], window = meta[2], cap = meta[3];
const int qi = int(threadgroup_position_in_grid.y);
const int sp = int(threadgroup_position_in_grid.z);
const int head0 = int(threadgroup_position_in_grid.x) * 16;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32 + lane;

const int pos = p0 + qi;
const int lo = metal::max(0, pos - window + 1);
const int nw = pos - lo + 1;
const int total = nw + K;
const int chunk = (total + S - 1) / S;
const int t_lo = sp * chunk;
const int tile_keys = metal::max(0, metal::min(total, t_lo + chunk) - t_lo);

if (tile_keys == 0) {
  for (int idx = int(tid); idx < 16 * D; idx += 128)
    pacc[((size_t(qi) * H + head0 + idx / D) * S + sp) * D + idx % D] = 0.0f;
  if (tid < 16) {
    const size_t at = (size_t(qi) * H + head0 + int(tid)) * S + sp;
    pm[at] = -INFINITY;
    pl[at] = 0.0f;
  }
  return;
}

const short qid = short(lane >> 2);
const short fm = short((qid & 4) | ((lane >> 1) & 3));
const short fn = short(((qid & 2) | (lane & 1)) * 4);
const bool row_writer = (lane & 9u) == 0u;

threadgroup int sel[BK];
threadgroup float red_max[4][16];
threadgroup float red_sum[4][16];
threadgroup half p_hi[BK / 16][32 * 8];
threadgroup half p_lo[BK / 16][32 * 8];

// A key is a ring slot (tagged) or a compressed row; -1 is a pick that is unset or past the cache.
for (int j = int(tid); j < BK; j += 128) {
  int s = -1;
  if (j < tile_keys) {
    const int t = t_lo + j;
    if (t < nw) s = RING | ((lo + t) % cap);
    else {
      const int c = topk[qi * K + t - nw];
      if (c >= 0 && c < nc) s = c;
    }
  }
  sel[j] = s;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

auto krow = [&](int s) -> const device T* {
  return s >= RING ? ring + size_t(s - RING) * D : comp + size_t(metal::max(s, 0)) * D;
};

constexpr auto qk_desc = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_desc, execution_simdgroup> qk_op;
constexpr auto pv_desc = matmul2d_descriptor(16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<pv_desc, execution_simdgroup> pv_op;

// The weights go through the tensor units as two fp16 pieces (the high one and the rest): 22 bits of each.
auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
auto pm_ = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
auto o0 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
auto o1 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
auto o2 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
auto o3 = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
for (short e = 0; e < 16; ++e) {
  o0[e] = 0.0f;
  o1[e] = 0.0f;
  o2[e] = 0.0f;
  o3[e] = 0.0f;
}

// ---- S = Q K^T for key quarter sg (16 heads x 32 keys)
auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
auto sc = qk_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
for (short e = 0; e < 16; ++e) sc[e] = 0.0f;
if (int(sg) * 32 < tile_keys) {
  const device T* qr0 = q + (size_t(qi) * H + head0 + fm) * D + fn;
  const device T* qr1 = qr0 + 8 * D;
  const device T* kr[2][2];
  for (short tn = 0; tn < 2; ++tn)
    for (short i = 0; i < 2; ++i)
      kr[tn][i] = krow(sel[sg * 32 + tn * 16 + fm + i * 8]) + fn;
  _Pragma("clang loop unroll_count(4)")
  for (short kk = 0; kk < D; kk += 16) {
    for (short j = 0; j < 4; ++j) {
      qa[j] = qr0[kk + j];
      qa[4 + j] = qr1[kk + j];
    }
    for (short tn = 0; tn < 2; ++tn)
      for (short i = 0; i < 2; ++i)
        for (short j = 0; j < 4; ++j) kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
    qk_op.run(qa, kb, sc);
  }
}
bool ok[2][4];
for (short tn = 0; tn < 2; ++tn)
  for (short j = 0; j < 4; ++j) ok[tn][j] = sel[sg * 32 + tn * 16 + fn + j] >= 0;

float pmx[2];
for (short i = 0; i < 2; ++i) {
  float m = -FLT_MAX;
  for (short tn = 0; tn < 2; ++tn)
    for (short j = 0; j < 4; ++j) m = ok[tn][j] ? metal::max(m, float(sc[tn * 8 + i * 4 + j])) : m;
  m = metal::max(m, metal::simd_shuffle_xor(m, ushort(1)));
  m = metal::max(m, metal::simd_shuffle_xor(m, ushort(8)));
  pmx[i] = m;
}
if (row_writer) {
  red_max[sg][fm] = pmx[0];
  red_max[sg][fm + 8] = pmx[1];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

const float scale = scal[0];
float mn[2];
for (short i = 0; i < 2; ++i) {
  const int r = fm + i * 8;
  const float m = metal::max(metal::max(red_max[0][r], red_max[1][r]), metal::max(red_max[2][r], red_max[3][r]));
  mn[i] = m == -FLT_MAX ? -FLT_MAX : m * scale;
}
float rs[2] = {0.0f, 0.0f};
for (short tn = 0; tn < 2; ++tn) {
  metal::vec<half, 8> h, l;
  for (short i = 0; i < 2; ++i) {
    for (short j = 0; j < 4; ++j) {
      const float e = ok[tn][j] ? metal::fast::exp(float(sc[tn * 8 + i * 4 + j]) * scale - mn[i]) : 0.0f;
      const half hi = half(e);
      h[i * 4 + j] = hi;
      l[i * 4 + j] = half(e - float(hi));
      rs[i] += e;
    }
  }
  const int ks = sg * 2 + tn;
  *(threadgroup metal::vec<half, 8>*)(&p_hi[ks][lane * 8]) = h;
  *(threadgroup metal::vec<half, 8>*)(&p_lo[ks][lane * 8]) = l;
}
for (short i = 0; i < 2; ++i) {
  rs[i] += metal::simd_shuffle_xor(rs[i], ushort(1));
  rs[i] += metal::simd_shuffle_xor(rs[i], ushort(8));
}
if (row_writer) {
  red_sum[sg][fm] = rs[0];
  red_sum[sg][fm + 8] = rs[1];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

// ---- O = P V for dim quarter sg over the whole tile
const int n_ks = (tile_keys + 15) / 16;
for (short ks = 0; ks < n_ks; ++ks) {
  const metal::vec<half, 8> h = *(const threadgroup metal::vec<half, 8>*)(&p_hi[ks][lane * 8]);
  const metal::vec<half, 8> l = *(const threadgroup metal::vec<half, 8>*)(&p_lo[ks][lane * 8]);
  for (short e = 0; e < 8; ++e) {
    pa[e] = h[e];
    pm_[e] = l[e];
  }
  const device T* v0 = krow(sel[ks * 16 + fm]) + sg * 128 + fn;
  const device T* v1 = krow(sel[ks * 16 + fm + 8]) + sg * 128 + fn;
  for (short np = 0; np < 4; ++np) {
    for (short tn = 0; tn < 2; ++tn) {
      for (short j = 0; j < 4; ++j) {
        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
      }
    }
    if (np == 0) {
      pv_op.run(pa, pb, o0);
      pv_op.run(pm_, pb, o0);
    } else if (np == 1) {
      pv_op.run(pa, pb, o1);
      pv_op.run(pm_, pb, o1);
    } else if (np == 2) {
      pv_op.run(pa, pb, o2);
      pv_op.run(pm_, pb, o2);
    } else {
      pv_op.run(pa, pb, o3);
      pv_op.run(pm_, pb, o3);
    }
  }
}

for (short i = 0; i < 2; ++i) {
  const int r = fm + i * 8;
  const size_t at = (size_t(qi) * H + head0 + r) * S + sp;
  device float* orow = pacc + at * D + sg * 128 + fn;
  for (short tn = 0; tn < 2; ++tn) {
    const short e = tn * 8 + i * 4;
    *(device float4*)(orow + tn * 16) = float4(o0[e], o0[e + 1], o0[e + 2], o0[e + 3]);
    *(device float4*)(orow + 32 + tn * 16) = float4(o1[e], o1[e + 1], o1[e + 2], o1[e + 3]);
    *(device float4*)(orow + 64 + tn * 16) = float4(o2[e], o2[e + 1], o2[e + 2], o2[e + 3]);
    *(device float4*)(orow + 96 + tn * 16) = float4(o3[e], o3[e + 1], o3[e + 2], o3[e + 3]);
  }
  if (sg == 0 && row_writer) {
    const float l = ((red_sum[0][r] + red_sum[1][r]) + red_sum[2][r]) + red_sum[3][r];
    pm[at] = l > 0.0f ? mn[i] : -INFINITY;
    pl[at] = l;
  }
}
