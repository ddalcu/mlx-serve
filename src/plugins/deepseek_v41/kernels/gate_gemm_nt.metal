    // MLX's fused NAX GEMM kernel (steel/gemm/kernels/steel_gemm_fused_nax.h `gemm`, MLX 0.32.3) at transpose_a
    // false and transpose_b true (a [M, K] and b [N, K], both row-major: MLX's `matmul(a, transpose(b))`), no batch and no
    // C, as one metal_kernel launch: y float [M, N] = a b^T, the accumulators rounded to OUT_R as MLX's store rounds them
    // into an OUT_R output (NAXFrag::store's static_cast) and widened back to float (an astype to f32 after MLX's matmul),
    // then stored as MLX stores them. MLX's statements, its params from the operands' shapes, the tile as template
    // constants (BM BN BK WM WN SWZ; AM AN AK the alignments MLX passes as function constants); the loop is f3_gemm_loop,
    // MLX's gemm_loop but A's and B's memory types TA and TB (each element static_cast to the MMA's type T as it loads).
    const int M = int(a_shape[0]);
    const int K = int(a_shape[1]);
    const int N = int(b_shape[0]);
    const int lda = K;
    const int ldb = K;
    const int ldd = N;
    const int tiles_n = (N + BN - 1) / BN;
    const int tiles_m = (M + BM - 1) / BM;
    const int swizzle_log = SWZ;
    const int gemm_k_iterations_aligned = K / BK;
    constexpr bool align_M = AM != 0;
    constexpr bool align_N = AN != 0;
    constexpr bool align_K = AK != 0;
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    // Find block
    const int tid_y = ((tid.y) << swizzle_log) +
        ((tid.x) & ((1 << swizzle_log) - 1));
    const int tid_x = (tid.x) >> swizzle_log;
    // Exit early if out of bounds
    if (tiles_n <= tid_x || tiles_m <= tid_y) {
      return;
    }
    const device TA* A = a;
    const device TB* B = b;
    device float* D = y;
    // Prepare threadgroup memory
    threadgroup_barrier(mem_flags::mem_none);
    // Find block in A, B, C
    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const size_t c_row_long = size_t(c_row);
    const size_t c_col_long = size_t(c_col);
    A += c_row_long * lda;
    B += c_col_long * ldb;
    D += c_row_long * ldd + c_col_long;
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    const short tm = SM * (simd_group_id / WN);
    const short tn = SN * (simd_group_id % WN);
    const int sgp_sm_int =
        align_M ? int(SM) : min(int(SM), M - (c_row + tm));
    const short sgp_sm = short(sgp_sm_int);
    const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
    const int sgp_sn_int =
        align_N ? int(SN) : min(int(SN), N - (c_col + tn));
    const short sgp_sn = short(sgp_sn_int);
    const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);
    A += tm * lda;
    B += tn * ldb;
    D += tm * ldd + tn;
    NAXTile<float, TM, TN> Dtile;
    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
          Dtile = f3_gemm_loop<
              T,
              TA,
              TB,
              SM,
              SN,
              SK,
              BK,
              false,
              true,
              kAlignedM.value,
              kAlignedN.value,
              kAlignedK.value,
              float>(
              A,
              B,
              lda,
              ldb,
              K,
              gemm_k_iterations_aligned,
              sgp_sm,
              sgp_sn);
          if ((kAlignedM.value || sgp_sm > 0) &&
              (kAlignedN.value || sgp_sn > 0)) {
            // MLX's store into an OUT_R output (static_cast), widened back to float.
            constexpr short kFrags = NAXTile<float, TM, TN>::kNumFrags;
            constexpr short kElems = NAXTile<float, TM, TN>::kElemsPerFrag;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kFrags; ++i) {
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < kElems; ++j) {
                Dtile.val_frags[i][j] = float(static_cast<OUT_R>(Dtile.val_frags[i][j]));
              }
            }
            if constexpr (kAlignedM && kAlignedN) {
              Dtile.store(D, int(ldd));
            } else {
              Dtile.store_safe(D, int(ldd), short2(sgp_sn, sgp_sm));
            }
          }
        });
      });
    });
