// dec_attn_tile (`MLX_SERVE_DSV41_ATTN_DEC`): decode attention partials, `dec_attn`'s outputs and split bounds, cut for
// the release's shapes (64 heads of 512, 128 window keys + 512 picks, rows 1..6). A threadgroup is one query row, 16
// heads and one split of at most TK keys, taken 32 keys (a group) at a time. Simdgroup s owns channels DS s .. DS s +
// DS - 1 of every key: it loads that slice of a group's 32 rows ONCE, in 16-byte loads, and the same registers are the
// right operand of its share of Q K^T and of its P V (keys and values are the same latent rows), so no row is read
// twice and no simdgroup waits on another's loads. The simdgroups' Q K^T sums over their slices meet in threadgroup
// memory; one pass there adds them, takes the online softmax (a running max per head, the sum rescaled) and hands the
// weights back as two fp16 pieces (22 bits). With PF the next group's rows load while this one computes.
// The contraction channels are permuted inside a slice (logical chunk t of 4 channels at 32 (t >> 1) + 8 f + 4 (t & 1)),
// the same for q, the rows and the output, which is what makes a lane's loads 16 bytes. With DSP = 2 a split runs in
// two threadgroups: both make the whole Q K^T and its softmax, each P V over half the channels (twice the grid at
// the same partial bytes, a third less matrix work a threadgroup).
constexpr int D = 512;
constexpr int RING = 1 << 24;
constexpr int DS = D / NSG;
constexpr int NU = DS / 32;
constexpr int NT = NSG * 32;
constexpr int NG = TK / 32;
constexpr int EPT = 512 / NT;
constexpr int TPH = 32 / EPT;
typedef metal::vec<T, 8> T8;

const int p0 = meta[0], nc = meta[1], window = meta[2], cap = meta[3];
const int qi = int(threadgroup_position_in_grid.y);
const int sp = int(threadgroup_position_in_grid.z);
const int head0 = int(threadgroup_position_in_grid.x) / DSP * 16;
const int dh = int(threadgroup_position_in_grid.x) % DSP;
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

threadgroup int sel[TK];
threadgroup float red[NSG][32][20];
threadgroup half p_hi[32][16];
threadgroup half p_lo[32][16];
threadgroup float alpha_s[16];

if (tile_keys == 0) {
  for (int idx = int(tid); idx < 16 * D / DSP; idx += NT)
    pacc[((size_t(qi) * H + head0 + idx / (D / DSP)) * S + sp) * D + dh * (D / DSP) + idx % (D / DSP)] = 0.0f;
  if (tid < 16 && dh == 0) {
    const size_t at = (size_t(qi) * H + head0 + int(tid)) * S + sp;
    pm[at] = -INFINITY;
    pl[at] = 0.0f;
  }
  return;
}

const short qid = short(lane >> 2);
const short fm = short((qid & 4) | ((lane >> 1) & 3));
const short ff = short((qid & 2) | (lane & 1));
const int d0 = int(sg) * DS + 8 * ff;
// This simdgroup's channels are in the threadgroup's half of P V.
const bool pv_mine = int(sg) / (NSG / DSP) == dh;

// The row's 16 heads over this simdgroup's channels, issued before the picks resolve.
T8 qv[2][NU];
#pragma clang loop unroll(full)
for (short i = 0; i < 2; ++i)
  #pragma clang loop unroll(full)
  for (short u = 0; u < NU; ++u)
    qv[i][u] = *(const device T8*)(q + (size_t(qi) * H + head0 + fm + 8 * i) * D + d0 + 32 * u);

// A key is a ring slot (tagged) or a compressed row; -1 is a pick that is unset or past the cache, or past the split.
for (int j = int(tid); j < TK; j += NT) {
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

// A group's rows: kv[a][i][u] is key 16 a + fm + 8 i of the group, chunk u of the lane's channels.
T8 kc[2][2][NU], kn[2][2][NU];
#define DSV41_LOAD_GROUP(DST, G) \
  _Pragma("clang loop unroll(full)") for (short a = 0; a < 2; ++a) \
  _Pragma("clang loop unroll(full)") for (short i = 0; i < 2; ++i) { \
    const device T* kr_ = krow(sel[(G) * 32 + a * 16 + fm + 8 * i]) + d0; \
    _Pragma("clang loop unroll(full)") for (short u = 0; u < NU; ++u) DST[a][i][u] = *(const device T8*)(kr_ + 32 * u); }

constexpr auto qk_desc = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_desc, execution_simdgroup> qk_op;
constexpr auto pv_desc = matmul2d_descriptor(16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<pv_desc, execution_simdgroup> pv_op;

auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
auto sc = qk_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
auto pz = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
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

// The softmax pass: thread tid owns head sh and keys sk .. sk + EPT - 1 of each group, held in (lane sl, element se..)
// of the Q K^T fragments (row fm + 8 i, column 16 tn + 4 f + j at element 8 tn + 4 i + j).
const int sh = int(tid) / TPH;
const int sk = (int(tid) % TPH) * EPT;
const int sf = (sk >> 2) & 3, sfm = sh & 7;
const int sl = (((sfm >> 2) & 1) << 4) | (((sf >> 1) & 1) << 3) | ((sfm & 3) << 1) | (sf & 1);
const int se = (sk >> 4) * 8 + (sh >> 3) * 4 + (sk & 3);
const float scale = scal[0];
float m_run = -FLT_MAX, l_run = 0.0f;

DSV41_LOAD_GROUP(kc, 0)
#pragma clang loop unroll(full)
for (int g = 0; g < NG; ++g) {
  if (g * 32 >= tile_keys) break;
  const bool more = g + 1 < NG && (g + 1) * 32 < tile_keys;
  if (PF && more) { DSV41_LOAD_GROUP(kn, g + 1) }

  // ---- this simdgroup's channels of S = Q K^T (16 heads x 32 keys)
  for (short e = 0; e < 16; ++e) sc[e] = 0.0f;
  #pragma clang loop unroll(full)
  for (short t = 0; t < 2 * NU; ++t) {
    #pragma clang loop unroll(full)
    for (short j = 0; j < 4; ++j) {
      qa[j] = qv[0][t >> 1][4 * (t & 1) + j];
      qa[4 + j] = qv[1][t >> 1][4 * (t & 1) + j];
    }
    #pragma clang loop unroll(full)
    for (short tn = 0; tn < 2; ++tn)
      #pragma clang loop unroll(full)
      for (short i = 0; i < 2; ++i)
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j) kb[tn * 8 + i * 4 + j] = kc[tn][i][t >> 1][4 * (t & 1) + j];
    qk_op.run(qa, kb, sc);
  }
  {
    threadgroup float4* rp = (threadgroup float4*)&red[sg][lane][0];
    rp[0] = float4(sc[0], sc[1], sc[2], sc[3]);
    rp[1] = float4(sc[4], sc[5], sc[6], sc[7]);
    rp[2] = float4(sc[8], sc[9], sc[10], sc[11]);
    rp[3] = float4(sc[12], sc[13], sc[14], sc[15]);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // ---- the slices' sums, the group's max, the rescale and the weights
  {
    float v[EPT];
    #pragma clang loop unroll(full)
    for (int x = 0; x < EPT; ++x) v[x] = 0.0f;
    #pragma clang loop unroll(full)
    for (int s = 0; s < NSG; ++s)
      #pragma clang loop unroll(full)
      for (int x = 0; x < EPT; ++x) v[x] += red[s][sl][se + x];
    bool ok[EPT];
    float gm = -FLT_MAX;
    #pragma clang loop unroll(full)
    for (int x = 0; x < EPT; ++x) {
      ok[x] = sel[g * 32 + sk + x] >= 0;
      gm = ok[x] ? metal::max(gm, v[x]) : gm;
    }
    #pragma clang loop unroll(full)
    for (ushort o = 1; o < TPH; o <<= 1) gm = metal::max(gm, metal::simd_shuffle_xor(gm, o));
    const float m_new = metal::max(m_run, gm);
    const float mo = m_run == -FLT_MAX ? -FLT_MAX : m_run * scale;
    const float mn = m_new == -FLT_MAX ? -FLT_MAX : m_new * scale;
#if DSV41_ATTN_DEC_PLANT
    const float alpha = 1.0f;
#else
    const float alpha = metal::fast::exp(mo - mn);
#endif
    float rs = 0.0f;
    #pragma clang loop unroll(full)
    for (int x = 0; x < EPT; ++x) {
      const float e = ok[x] ? metal::fast::exp(v[x] * scale - mn) : 0.0f;
      const half hi = half(e);
      p_hi[sl][se + x] = hi;
      p_lo[sl][se + x] = half(e - float(hi));
      rs += e;
    }
    #pragma clang loop unroll(full)
    for (ushort o = 1; o < TPH; o <<= 1) rs += metal::simd_shuffle_xor(rs, o);
    l_run = l_run * alpha + rs;
    m_run = m_new;
    if (int(tid) % TPH == 0) alpha_s[sh] = alpha;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // ---- O += P V on this simdgroup's channels, the running sums rescaled first
  if (pv_mine && g > 0) {
    const float a0 = alpha_s[fm], a1 = alpha_s[fm + 8];
    for (short e = 0; e < 16; ++e) {
      const float a = (e & 4) ? a1 : a0;
      o0[e] *= a;
      if (NU > 1) o1[e] *= a;
      if (NU > 2) o2[e] *= a;
      if (NU > 3) o3[e] *= a;
    }
  }
  #pragma clang loop unroll(full)
  for (short ks = 0; ks < 2; ++ks) {
    if (!pv_mine || g * 32 + ks * 16 >= tile_keys) break;
    const metal::vec<half, 8> h = *(const threadgroup metal::vec<half, 8>*)(&p_hi[lane][ks * 8]);
    const metal::vec<half, 8> l = *(const threadgroup metal::vec<half, 8>*)(&p_lo[lane][ks * 8]);
    for (short e = 0; e < 8; ++e) {
      pa[e] = h[e];
      pz[e] = l[e];
    }
    #pragma clang loop unroll(full)
    for (short np = 0; np < NU; ++np) {
      #pragma clang loop unroll(full)
      for (short tn = 0; tn < 2; ++tn)
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; ++i)
          #pragma clang loop unroll(full)
          for (short j = 0; j < 4; ++j) pb[tn * 8 + i * 4 + j] = kc[ks][i][np][4 * tn + j];
      if (np == 0) {
        pv_op.run(pa, pb, o0);
        pv_op.run(pz, pb, o0);
      } else if (np == 1) {
        pv_op.run(pa, pb, o1);
        pv_op.run(pz, pb, o1);
      } else if (np == 2) {
        pv_op.run(pa, pb, o2);
        pv_op.run(pz, pb, o2);
      } else {
        pv_op.run(pa, pb, o3);
        pv_op.run(pz, pb, o3);
      }
    }
  }
  if (more) {
    if (PF) {
      #pragma clang loop unroll(full)
      for (short a = 0; a < 2; ++a)
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; ++i)
          #pragma clang loop unroll(full)
          for (short u = 0; u < NU; ++u) kc[a][i][u] = kn[a][i][u];
    } else {
      DSV41_LOAD_GROUP(kc, g + 1)
    }
  }
}
#undef DSV41_LOAD_GROUP

// ---- the split's partials: unnormalised weighted rows, the max (scaled) and the weight sum
if (int(tid) % TPH == 0 && dh == 0) {
  const size_t at = (size_t(qi) * H + head0 + sh) * S + sp;
  pm[at] = l_run > 0.0f ? m_run * scale : -INFINITY;
  pl[at] = l_run;
}
if (pv_mine) {
#pragma clang loop unroll(full)
for (short i = 0; i < 2; ++i) {
  device float* orow = pacc + ((size_t(qi) * H + head0 + fm + 8 * i) * S + sp) * D + d0;
  const short e = i * 4;
  *(device float4*)(orow) = float4(o0[e], o0[e + 1], o0[e + 2], o0[e + 3]);
  *(device float4*)(orow + 4) = float4(o0[e + 8], o0[e + 9], o0[e + 10], o0[e + 11]);
  if (NU > 1) {
    *(device float4*)(orow + 32) = float4(o1[e], o1[e + 1], o1[e + 2], o1[e + 3]);
    *(device float4*)(orow + 36) = float4(o1[e + 8], o1[e + 9], o1[e + 10], o1[e + 11]);
  }
  if (NU > 2) {
    *(device float4*)(orow + 64) = float4(o2[e], o2[e + 1], o2[e + 2], o2[e + 3]);
    *(device float4*)(orow + 68) = float4(o2[e + 8], o2[e + 9], o2[e + 10], o2[e + 11]);
  }
  if (NU > 3) {
    *(device float4*)(orow + 96) = float4(o3[e], o3[e + 1], o3[e + 2], o3[e + 3]);
    *(device float4*)(orow + 100) = float4(o3[e + 8], o3[e + 9], o3[e + 10], o3[e + 11]);
  }
}
}
