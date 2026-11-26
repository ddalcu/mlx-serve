// Inkling 2, wide-tile sorted MoE prefill (non-NAX Apple9+): BM x 64 x 32 simdgroup_matrix<T, 8, 8> GEMM over
// tile_scan's (row_start, expert, rows, 0) tiles. Supports NVFP4 (FMT 0), MXFP4 (FMT 1), MXFP8 (FMT 2) and affine
// (FMT 3, BITS 2..6/8, GS 32/64/128). With USE_RMAP=1 gathers input rows directly from X [1, S, K] via RMAP[row];
// with SWIGLU=1 fuses per-expert scale SC[expert] and soft-capped SwiGLU over adjacent (gate, up) columns in registers.
// grid (max_tiles * 128, ceil(N / 64), 1), threadgroup (128, 1, 1): 4 SIMD groups (WM=2, WN=2).
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int WM = 2;
constexpr int WN = 2;
constexpr int LDA = BK + 8;
constexpr int LDB = BK + 8;
constexpr int TM = (BM / WM) / 8;
constexpr int TN = (BN / WN) / 8;
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint t_idx = threadgroup_position_in_grid.x;
const uint n_blk = threadgroup_position_in_grid.y;
if (t_idx >= tile_count[0]) return;
const uint4 desc = *((const device uint4*)tiles + t_idx);
const uint row0 = desc.x;
const uint expert = desc.y;
const uint rows = desc.z;
const int n0 = int(n_blk) * BN;
const uint sg_m = sg / WN;
const uint sg_n = sg % WN;
const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
threadgroup T As[BM * LDA];
threadgroup T Bs[BN * LDB];
simdgroup_float8x8 acc[TM][TN];
#pragma clang loop unroll(full)
for (int mi = 0; mi < TM; ++mi)
  #pragma clang loop unroll(full)
  for (int ni = 0; ni < TN; ++ni)
    acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

const uint r_load = tid >> 1;
const uint col_load = (tid & 1u) * 16;
const uint src_r = (r_load < rows) ? (USE_RMAP ? RMAP[row0 + r_load] : (row0 + r_load)) : 0u;
const device T* x_row = X + size_t(src_r) * K;
const uint SSTR = K / GS;
const size_t w_row_idx = size_t(expert) * N + size_t(n0) + size_t(r_load);
const uint WSTR_U32 = (FMT == 2) ? (K / 4) : (K / 8);
const uint WSTR_B = K * BITS / 8;
const device uint* w_row = (const device uint*)W + w_row_idx * WSTR_U32;
const device uchar* s_row = (const device uchar*)WS + w_row_idx * SSTR;
const device uchar* a_row = (const device uchar*)W + w_row_idx * WSTR_B;
const device S* sa_row = (const device S*)WS + w_row_idx * SSTR;
const device S* ba_row = (const device S*)WB + w_row_idx * SSTR;

for (uint k0 = 0; k0 < uint(K); k0 += BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (r_load < uint(BM)) {
    threadgroup vec<T, 4>* adst4 = (threadgroup vec<T, 4>*)(As + r_load * LDA + col_load);
    if (r_load < rows) {
      const device vec<T, 4>* xp = (const device vec<T, 4>*)(x_row + k0 + col_load);
      adst4[0] = xp[0];
      adst4[1] = xp[1];
      adst4[2] = xp[2];
      adst4[3] = xp[3];
    } else {
      adst4[0] = vec<T, 4>(0);
      adst4[1] = vec<T, 4>(0);
      adst4[2] = vec<T, 4>(0);
      adst4[3] = vec<T, 4>(0);
    }
  }
  {
    threadgroup vec<T, 4>* bdst4 = (threadgroup vec<T, 4>*)(Bs + r_load * LDB + col_load);
    if (n0 + int(r_load) < N) {
      const uint c16 = (k0 + col_load) >> 4;
      half4 w0, w1, w2, w3;
      float sc;
      if constexpr (FMT == 3) {
        float bi;
        msv_ink2_decode16_affine_v4<S, BITS, GS>(a_row, sa_row, ba_row, c16, w0, w1, w2, w3, sc, bi);
        bdst4[0] = vec<T, 4>(float4(w0) * sc + bi);
        bdst4[1] = vec<T, 4>(float4(w1) * sc + bi);
        bdst4[2] = vec<T, 4>(float4(w2) * sc + bi);
        bdst4[3] = vec<T, 4>(float4(w3) * sc + bi);
      } else {
        msv_ink2_decode16_v4(FMT, w_row, s_row, c16, w0, w1, w2, w3, sc);
        bdst4[0] = vec<T, 4>(float4(w0) * sc);
        bdst4[1] = vec<T, 4>(float4(w1) * sc);
        bdst4[2] = vec<T, 4>(float4(w2) * sc);
        bdst4[3] = vec<T, 4>(float4(w3) * sc);
      }
    } else {
      bdst4[0] = vec<T, 4>(0);
      bdst4[1] = vec<T, 4>(0);
      bdst4[2] = vec<T, 4>(0);
      bdst4[3] = vec<T, 4>(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for (int kb = 0; kb < BK; kb += 8) {
    metal::simdgroup_matrix<T, 8, 8> af[TM];
    metal::simdgroup_matrix<T, 8, 8> bf[TN];
    #pragma clang loop unroll(full)
    for (int mi = 0; mi < TM; ++mi)
      simdgroup_load(af[mi], As + (sg_m * (BM / WM) + mi * 8) * LDA + kb, LDA);
    #pragma clang loop unroll(full)
    for (int ni = 0; ni < TN; ++ni)
      simdgroup_load(bf[ni], Bs + (sg_n * (BN / WN) + ni * 8) * LDB + kb, LDB, ulong2(0, 0), true);
    #pragma clang loop unroll(full)
    for (int mi = 0; mi < TM; ++mi)
      #pragma clang loop unroll(full)
      for (int ni = 0; ni < TN; ++ni)
        simdgroup_multiply_accumulate(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
  }
}

const float esc = SCALED ? SC[expert] : 1.0f;
#pragma clang loop unroll(full)
for (int mi = 0; mi < TM; ++mi) {
  const int rm = int(sg_m * (BM / WM) + mi * 8) + fm;
  if (rm >= int(rows)) continue;
  const size_t yrow = size_t(row0 + uint(rm));
  #pragma clang loop unroll(full)
  for (int ni = 0; ni < TN; ++ni) {
    const int cn = n0 + int(sg_n * (BN / WN) + ni * 8) + fn;
    if (cn >= N) continue;
    if constexpr (SWIGLU) {
      const float g = acc[mi][ni].thread_elements()[0] * esc;
      const float u = acc[mi][ni].thread_elements()[1] * esc;
      const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
      Y[yrow * (N / 2) + uint(cn / 2)] = T(capped / (1.0f + precise::exp(-g)));
    } else {
      Y[yrow * N + uint(cn)] = T(acc[mi][ni].thread_elements()[0]);
      Y[yrow * N + uint(cn + 1)] = T(acc[mi][ni].thread_elements()[1]);
    }
  }
}
