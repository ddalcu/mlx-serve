constexpr uint NSG = (uint)NSGVAL;
constexpr uint TGN = NSG * 32u;
static_assert(4096u % TGN == 0);
constexpr uint USEH4 = (uint)USEH4VAL;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
threadgroup bfloat tile[128*32];
const uint nb   = (uint)POOL_shape[1];
const uint rows = (uint)Q_shape[2];
const uint tid  = thread_position_in_threadgroup.x;
const uint sg   = tid / 32u;
const uint lane = tid % 32u;
const uint rg   = threadgroup_position_in_grid.y;
const uint gsh  = threadgroup_position_in_grid.x;
const uint NSH  = (uint)threadgroups_per_grid.x;
const short qid = lane >> 2;
const short fm  = ((qid & 4) | ((lane >> 1) & 3));
const short fn  = ((qid & 2) | (lane & 1)) * 4;
const uint nslab = (nb + 31u) / 32u;
auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
auto ct_c = op.get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(ct_a)>,
    metal::remove_addrspace_t<decltype(ct_b)>, float>();
if (USEH4 != 0u) {
  const uint qrow0 = rg * (NSG*16u) + sg * 16u;
  bfloat A[4][8][8];
  #pragma unroll
  for (short h = 0; h < 4; h++) {
    #pragma unroll
    for (short ks = 0; ks < 8; ks++) {
      #pragma unroll
      for (short i = 0; i < 2; i++) {
        #pragma unroll
        for (short j = 0; j < 4; j++) {
          const uint qr = qrow0 + (uint)(fm + i*8), c = (uint)ks*16u + (uint)(fn + j);
          A[h][ks][i*4+j] = (qr < rows) ? Q[((ulong)h*(ulong)rows + qr)*128ul + c] : (bfloat)0;
        }
      }
    }
  }
  float acc[16];
  for (uint sl = gsh; sl < nslab; sl += NSH) {
    const uint n0 = sl * 32u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint i = 0; i < 4096u/TGN; i++) {
      const uint idx = tid + i*TGN;
      const uint n = idx / 128u, k = idx % 128u;
      tile[k*32u + n] = ((n0 + n) < nb) ? POOL[(ulong)(n0 + n)*128ul + k] : (bfloat)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (short e = 0; e < 16; e++) acc[e] = 0.0f;
    #pragma unroll
    for (short h = 0; h < 4; h++) {
      #pragma unroll
      for (short e = 0; e < 16; e++) ct_c[e] = 0.0f;
      #pragma unroll
      for (short ks = 0; ks < 8; ks++) {
        #pragma unroll
        for (short i = 0; i < 8; i++) ct_a[i] = A[h][ks][i];
        #pragma unroll
        for (short i = 0; i < 2; i++) {
          #pragma unroll
          for (short j = 0; j < 4; j++) {
            const uint kk = (uint)ks*16u + (uint)(fm + i*8);
            ct_b[i*4+j]     = tile[kk*32u + (uint)(fn + j)];
            ct_b[8 + i*4+j] = tile[kk*32u + (uint)(fn + j + 16)];
          }
        }
        op.run(ct_a, ct_b, ct_c);
      }
      #pragma unroll
      // heads accumulate ascending in f32: mlx_sum_axis reduces a length-4 axis sequentially
      for (short e = 0; e < 16; e++) acc[e] += metal::max(ct_c[e], 0.0f);
    }
    #pragma unroll
    for (short i = 0; i < 2; i++) {
      #pragma unroll
      for (short j = 0; j < 4; j++) {
        const uint qr = qrow0 + (uint)(fm + i*8);
        const uint c0 = n0 + (uint)(fn + j), c1 = c0 + 16u;
        if (qr < rows) {
          if (c0 < nb) OUT[(ulong)qr*(ulong)nb + c0] = acc[i*4+j];
          if (c1 < nb) OUT[(ulong)qr*(ulong)nb + c1] = acc[8 + i*4+j];
        }
      }
    }
  }
} else {
  const uint arow0 = rg * (NSG*16u) + sg * 16u;
  const uint qrow0 = arow0 / 4u;
  bfloat a0[8],a1[8],a2[8],a3[8],a4[8],a5[8],a6[8],a7[8];
  #define LOADA(KS, DST) { _Pragma("unroll") for (short i = 0; i < 2; i++) { _Pragma("unroll") for (short j = 0; j < 4; j++) { const uint ar = arow0 + fm + i*8, c = (KS)*16u + fn + j; const uint qr = ar >> 2, hh = ar & 3u; DST[i*4+j] = (qr < rows) ? Q[((ulong)hh*(ulong)rows + qr)*128ul + c] : (bfloat)0; } } }
  LOADA(0u,a0) LOADA(1u,a1) LOADA(2u,a2) LOADA(3u,a3)
  LOADA(4u,a4) LOADA(5u,a5) LOADA(6u,a6) LOADA(7u,a7)
  for (uint sl = gsh; sl < nslab; sl += NSH) {
    const uint n0 = sl * 32u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint i = 0; i < 4096u/TGN; i++) {
      const uint idx = tid + i*TGN;
      const uint n = idx / 128u, k = idx % 128u;
      tile[k*32u + n] = ((n0 + n) < nb) ? POOL[(ulong)(n0 + n)*128ul + k] : (bfloat)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
    #define STEP(KS, SRCA) { _Pragma("unroll") for (short i = 0; i < 8; i++) ct_a[i] = SRCA[i]; _Pragma("unroll") for (short i = 0; i < 2; i++) { _Pragma("unroll") for (short j = 0; j < 4; j++) { const uint kk = (KS)*16u + fm + i*8; ct_b[i*4+j] = tile[kk*32u + (fn + j)]; ct_b[8 + i*4+j] = tile[kk*32u + (fn + j + 16)]; } } op.run(ct_a, ct_b, ct_c); }
    STEP(0u,a0) STEP(1u,a1) STEP(2u,a2) STEP(3u,a3) STEP(4u,a4) STEP(5u,a5) STEP(6u,a6) STEP(7u,a7)
    float sv[16];
    #pragma unroll
    for (short e = 0; e < 16; e++) {
      float v = metal::max(ct_c[e], 0.0f);
      // lanes l^2, l^4, l^6 hold heads 1..3; the sum stays ascending, matching mlx_sum_axis
      v += metal::max(simd_shuffle_xor(ct_c[e], 2u), 0.0f);
      v += metal::max(simd_shuffle_xor(ct_c[e], 4u), 0.0f);
      v += metal::max(simd_shuffle_xor(ct_c[e], 6u), 0.0f);
      sv[e] = v;
    }
    if ((fm & 3) == 0) {
      #pragma unroll
      for (short i = 0; i < 2; i++) {
        #pragma unroll
        for (short j = 0; j < 4; j++) {
          const uint qr = qrow0 + (uint)(fm + i*8) / 4u;
          const uint c0 = n0 + fn + j, c1 = c0 + 16u;
          if (qr < rows) {
            if (c0 < nb) OUT[(ulong)qr*(ulong)nb + c0] = sv[i*4+j];
            if (c1 < nb) OUT[(ulong)qr*(ulong)nb + c1] = sv[8 + i*4+j];
          }
        }
      }
    }
  }
}
