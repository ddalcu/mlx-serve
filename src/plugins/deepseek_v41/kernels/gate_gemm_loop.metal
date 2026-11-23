// MLX steel's gemm_loop (mlx/backend/metal/kernels/steel/gemm/gemm_nax.h, MLX 0.32.3; turboexl3's pg2 copy,
// kernels/pg2/gemm_loop.metal) verbatim but for its name (f3_gemm_loop) and A's and B's memory types (TA, TB): the A and
// B tiles in the MMA's type T, each element static_cast to T as it loads (NAXFrag::load), SK 32, the 16 x 32 x 16 MMA,
// k in MLX's order. `kBlocked` (MLX_SERVE_DSV41_PF_COMP_BF16): each BK block's products summed into a tile of its own
// and that block's sum added to the output's (a blocked sum: about as close to float64 as MLX's split K, where one
// running sum over all of K drifts further); off, MLX's single running sum.
template <
    typename T,
    typename TA,
    typename TB,
    short SM,
    short SN,
    short SK,
    short BK,
    bool transpose_a,
    bool transpose_b,
    bool kAlignedM,
    bool kAlignedN,
    bool kAlignedK,
    typename AccumType = float,
    bool kBlocked = false>
auto f3_gemm_loop(
    const device TA* A,
    const device TB* B,
    int lda,
    int ldb,
    int K,
    int gemm_k_iterations_aligned,
    const short sgp_sm,
    const short sgp_sn) {
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  constexpr int RA = transpose_a ? TK : TM;
  constexpr int CA = transpose_a ? TM : TK;

  constexpr int RB = transpose_b ? TN : TK;
  constexpr int CB = transpose_b ? TK : TN;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();
  NAXTile<AccumType, TM, TN> Ctile;

  const bool has_output = sgp_sm > 0 && sgp_sn > 0;

  int gemm_k_iterations_ = gemm_k_iterations_aligned;

  STEEL_PRAGMA_NO_UNROLL
  for (int kk0 = 0; kk0 < gemm_k_iterations_; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    if constexpr (!kAlignedM || !kAlignedN) {
      if (!has_output)
        continue;
    }
    if constexpr (kBlocked) {
      Ctile.clear();
    }

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;
      const int k = kk1;

      volatile int compiler_barrier;

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      if constexpr (kAlignedM) {
        Atile.load(A + A_offset, lda);
      } else {
        const short rmax = transpose_a ? SK : sgp_sm;
        const short cmax = transpose_a ? sgp_sm : SK;
        Atile.load_safe(A + A_offset, lda, short2(cmax, rmax));
      }

      if constexpr (kAlignedN) {
        Btile.load(B + B_offset, ldb);
      } else {
        const short rmax = transpose_b ? sgp_sn : SK;
        const short cmax = transpose_b ? SK : sgp_sn;
        Btile.load_safe(B + B_offset, ldb, short2(cmax, rmax));
      }

      if constexpr (kBlocked) {
        tile_matmad_nax(
            Ctile,
            Atile,
            metal::bool_constant<transpose_a>{},
            Btile,
            metal::bool_constant<transpose_b>{});
      } else {
        tile_matmad_nax(
            Dtile,
            Atile,
            metal::bool_constant<transpose_a>{},
            Btile,
            metal::bool_constant<transpose_b>{});
      }

      (void)compiler_barrier;
    }
    if constexpr (kBlocked) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < NAXTile<AccumType, TM, TN>::kNumFrags; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < NAXTile<AccumType, TM, TN>::kElemsPerFrag; ++j) {
          Dtile.val_frags[i][j] += Ctile.val_frags[i][j];
        }
      }
    }

    A += transpose_a ? (BK * lda) : BK;
    B += transpose_b ? BK : (BK * ldb);
  }

  if constexpr (!kAlignedK) {
    simdgroup_barrier(mem_flags::mem_none);
    if constexpr (!kAlignedM || !kAlignedN) {
      if (!has_output)
        return Dtile;
    }

    const short rem_bk = K - gemm_k_iterations_ * BK;

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;

      const int k = kk1;
      const short psk = max(0, rem_bk - k);

      const short2 Aklims =
          transpose_a ? short2(sgp_sm, psk) : short2(psk, sgp_sm);
      const short2 Bklims =
          transpose_b ? short2(psk, sgp_sn) : short2(sgp_sn, psk);

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      Atile.load_safe(A + A_offset, lda, Aklims);
      Btile.load_safe(B + B_offset, ldb, Bklims);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});
    }
  }

  return Dtile;
}
