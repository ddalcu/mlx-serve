// fa1 (MLX_SERVE_DSV41_FLASH_ATTN): a wide prompt span's attention as one kernel, flash style. One threadgroup is one
// query row and HGN groups of 16 heads; the heads share the row's keys (one KV head, the keys are the values). The
// row's keys go in blocks of 128 (the window's 128 rows, then the picks in their order; c < W is keys_w row
// wp + l - W + 1 + c, absent outside [0, Wk); else comp row picks[l, c - W], absent below 0). Per block, simdgroup
// (head group, quarter) scores its quarter's 32 keys for its 16 heads on the tensor units (16 x 32 x 16 steps over the
// 512 dims, `q` for dims below D - 64 and Q_TAIL's roped tail `qt` for the last 64 when QT), the block's max per head
// joins the running max through threadgroup memory, exp(scale * s - m) goes to threadgroup memory as f16 (0 for absent
// keys), and the simdgroup then accumulates its quarter's 128 output dims of its 16 heads over the block's keys
// (16 x 32 x 16 steps), its f32 accumulator scaled by exp(m_old - m_new) first. The head's sink starts the running state
// (m = sink, l = 1), so it joins the denominator only. Epilogue: O / l, the last RD dims inverse-roped at position
// meta[6] + l (rope_tail's rotation on the f32 values), stored as attnOut's grouped projection reads them,
// [G, L, H / G * D]. Nothing of the scores or probabilities leaves the threadgroup.
constexpr int D = 512;
constexpr int BK = 128;
constexpr int NSG = 4 * HGN;
constexpr int RING = 1 << 30;

const int l = int(threadgroup_position_in_grid.y);
const int wp = meta[0], W = meta[1], Wk = meta[2], K = meta[3];
const int H = meta[4];
const int NK = W + K;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32 + lane;
const int hgl = int(sg) >> 2;
const int hg = int(threadgroup_position_in_grid.x) * HGN + hgl;
const int qq = int(sg) & 3;
const short qid = short(lane >> 2);
const short fm = short((qid & 4) | ((lane >> 1) & 3));
const short fn = short(((qid & 2) | (lane & 1)) * 4);
const bool row_writer = (lane & 9u) == 0u;

threadgroup int sel[2][BK];
threadgroup float red_max[NSG][16];
threadgroup float red_sum[NSG][16];
threadgroup half p_tg[HGN][BK / 16][32 * 8];

auto krow = [&](int s) -> const device T* {
  return s >= RING ? kw + size_t(s - RING) * D : comp + size_t(metal::max(s, 0)) * D;
};

constexpr auto qk_desc = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_desc, execution_simdgroup> qk_op;
constexpr auto pv_desc = matmul2d_descriptor(16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<pv_desc, execution_simdgroup> pv_op;

auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
auto sc = qk_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
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

const float scale = scal[0];
// This lane's two head rows: h0 and h0 + 8 (the tensor op's rows fm and fm + 8).
const int h0 = 16 * hg + fm;
float m_run[2] = {sink[h0], sink[h0 + 8]};
float l_run[2] = {1.0f, 1.0f};
const device T* qr0 = q + (size_t(l) * H + h0) * D + fn;
const device T* qr1 = qr0 + 8 * D;
const device T* qt0 = qt + (size_t(l) * H + h0) * 64 + fn;
const device T* qt1 = qt0 + 8 * 64;

const int nb = (NK + BK - 1) / BK;
for (int b = 0; b < nb; ++b) {
  const int sb = b & 1;
  if (int(tid) < BK) {
    const int c = b * BK + int(tid);
    int s = -1;
    if (c < W) {
      const int r = wp + l - (W - 1) + c;
      if (r >= 0 && r < Wk) s = RING | r;
    } else if (c < NK) {
      const int r = picks[size_t(l) * K + (c - W)];
      if (r >= 0) s = r;
    }
    sel[sb][tid] = s;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // ---- S = Q K^T: head group hg, keys 32 qq .. 32 qq + 31 of the block
  for (short e = 0; e < 16; ++e) sc[e] = 0.0f;
  if (metal::simd_any(sel[sb][qq * 32 + int(lane)] >= 0)) {
    const device T* kr[2][2];
    for (short tn = 0; tn < 2; ++tn)
      for (short i = 0; i < 2; ++i) kr[tn][i] = krow(sel[sb][qq * 32 + tn * 16 + fm + i * 8]) + fn;
    _Pragma("clang loop unroll_count(4)")
    for (short kk = 0; kk < D; kk += 16) {
      metal::vec<T, 4> a0, a1;
      if (QT && kk >= D - 64) {
        a0 = *(const device metal::vec<T, 4>*)(qt0 + (kk - (D - 64)));
        a1 = *(const device metal::vec<T, 4>*)(qt1 + (kk - (D - 64)));
      } else {
        a0 = *(const device metal::vec<T, 4>*)(qr0 + kk);
        a1 = *(const device metal::vec<T, 4>*)(qr1 + kk);
      }
      for (short j = 0; j < 4; ++j) {
        qa[j] = a0[j];
        qa[4 + j] = a1[j];
      }
      for (short tn = 0; tn < 2; ++tn)
        for (short i = 0; i < 2; ++i) {
          const metal::vec<T, 4> kv = *(const device metal::vec<T, 4>*)(kr[tn][i] + kk);
          for (short j = 0; j < 4; ++j) kb[tn * 8 + i * 4 + j] = kv[j];
        }
      qk_op.run(qa, kb, sc);
    }
  }

  // ---- the block's max per head (scaled, live keys only), joined across the head group's quarters
  bool ok[2][4];
  for (short tn = 0; tn < 2; ++tn)
    for (short j = 0; j < 4; ++j) ok[tn][j] = sel[sb][qq * 32 + tn * 16 + fn + j] >= 0;
  float bmx[2];
  for (short i = 0; i < 2; ++i) {
    float m = -INFINITY;
    for (short tn = 0; tn < 2; ++tn)
      for (short j = 0; j < 4; ++j)
        if (ok[tn][j]) m = metal::max(m, scale * float(sc[tn * 8 + i * 4 + j]));
    m = metal::max(m, metal::simd_shuffle_xor(m, ushort(1)));
    m = metal::max(m, metal::simd_shuffle_xor(m, ushort(8)));
    bmx[i] = m;
  }
  if (row_writer) {
    red_max[sg][fm] = bmx[0];
    red_max[sg][fm + 8] = bmx[1];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float mnew[2], alpha[2];
  for (short i = 0; i < 2; ++i) {
    const int r = fm + i * 8;
    const int s0 = hgl * 4;
    const float bm = metal::max(metal::max(red_max[s0][r], red_max[s0 + 1][r]), metal::max(red_max[s0 + 2][r], red_max[s0 + 3][r]));
    mnew[i] = metal::max(m_run[i], bm);
    alpha[i] = metal::exp(m_run[i] - mnew[i]);
    m_run[i] = mnew[i];
  }

  // ---- P = exp(scale * s - m) as f16 into the head group's tile, its f32 row sums joined across the quarters
  float rs[2] = {0.0f, 0.0f};
  for (short tn = 0; tn < 2; ++tn) {
    metal::vec<half, 8> hv;
    for (short i = 0; i < 2; ++i)
      for (short j = 0; j < 4; ++j) {
        const float e = ok[tn][j] ? metal::exp(scale * float(sc[tn * 8 + i * 4 + j]) - mnew[i]) : 0.0f;
        hv[i * 4 + j] = half(e);
        rs[i] += e;
      }
    *(threadgroup metal::vec<half, 8>*)(&p_tg[hgl][qq * 2 + tn][lane * 8]) = hv;
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
  for (short i = 0; i < 2; ++i) {
    const int r = fm + i * 8;
    const int s0 = hgl * 4;
    l_run[i] = l_run[i] * alpha[i] + (((red_sum[s0][r] + red_sum[s0 + 1][r]) + red_sum[s0 + 2][r]) + red_sum[s0 + 3][r]);
  }

  // ---- O = O * alpha + P V: dims 128 qq .. 128 qq + 127 of head group hg, the block's keys 16 a step
  for (short e = 0; e < 16; ++e) {
    const float a = alpha[(e >> 2) & 1];
    o0[e] *= a;
    o1[e] *= a;
    o2[e] *= a;
    o3[e] *= a;
  }
  for (short ks = 0; ks < BK / 16; ++ks) {
    if (!metal::simd_any(sel[sb][ks * 16 + int(lane & 15u)] >= 0)) continue;
    const metal::vec<half, 8> hv = *(const threadgroup metal::vec<half, 8>*)(&p_tg[hgl][ks][lane * 8]);
    for (short e = 0; e < 8; ++e) pa[e] = hv[e];
    const device T* v0 = krow(sel[sb][ks * 16 + fm]) + qq * 128 + fn;
    const device T* v1 = krow(sel[sb][ks * 16 + fm + 8]) + qq * 128 + fn;
#define FA1_PV_STEP(OO, NP)                                                                        \
    for (short tn = 0; tn < 2; ++tn) {                                                             \
      const metal::vec<T, 4> x0 = *(const device metal::vec<T, 4>*)(v0 + (NP) * 32 + tn * 16);    \
      const metal::vec<T, 4> x1 = *(const device metal::vec<T, 4>*)(v1 + (NP) * 32 + tn * 16);    \
      for (short j = 0; j < 4; ++j) {                                                              \
        pb[tn * 8 + j] = x0[j];                                                                    \
        pb[tn * 8 + 4 + j] = x1[j];                                                                \
      }                                                                                            \
    }                                                                                              \
    pv_op.run(pa, pb, OO);
    FA1_PV_STEP(o0, 0)
    FA1_PV_STEP(o1, 1)
    FA1_PV_STEP(o2, 2)
    FA1_PV_STEP(o3, 3)
#undef FA1_PV_STEP
  }
}

// ---- epilogue: O / l, the inverse-roped tail, the grouped store
const float inv[2] = {1.0f / l_run[0], 1.0f / l_run[1]};
const int L = meta[7], G = meta[8];
const int hgr = H / G;
const float pos = float(meta[6] + l);
const int head = D - RD;
auto dst_of = [&](int h, int d) -> device OT* {
  return o + (size_t(h / hgr) * L + l) * size_t(hgr * D) + size_t(h % hgr) * D + d;
};
#define FA1_STORE(OO, NP)                                                                           \
  for (short e = 0; e < 16; e += 2) {                                                               \
    const short i = (e >> 2) & 1, tn = e >> 3, j = e & 3;                                           \
    const int d = qq * 128 + (NP) * 32 + tn * 16 + fn + j;                                          \
    const int h = h0 + i * 8;                                                                       \
    float a = OO[e] * inv[i], bb = OO[e + 1] * inv[i];                                              \
    if (d >= head) {                                                                                \
      const float theta = pos * (1.0f / periods[(d - head) / 2]);                                   \
      const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);                       \
      const float ra = a * cs - bb * sn;                                                            \
      bb = a * sn + bb * cs;                                                                        \
      a = ra;                                                                                       \
    }                                                                                               \
    *(device metal::vec<OT, 2>*)dst_of(h, d) = metal::vec<OT, 2>(OT(a), OT(bb));                    \
  }
FA1_STORE(o0, 0)
FA1_STORE(o1, 1)
FA1_STORE(o2, 2)
FA1_STORE(o3, 3)
#undef FA1_STORE
