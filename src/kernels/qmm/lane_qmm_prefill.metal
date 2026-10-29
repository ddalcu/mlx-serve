  const ushort simd_gid = simdgroup_index_in_threadgroup;
  const ushort simd_lid = thread_index_in_simdgroup;
  const int M = mdims[0];
  constexpr int BK = 64, BN = 64, WM = 2, WN = 2, KG = K / BK;
  constexpr int BK_padded = BK + 8;
  threadgroup bfloat Ws[BN * BK_padded];
  const int y_row = threadgroup_position_in_grid.y * BM;
  const int y_col = threadgroup_position_in_grid.x * BN;
  // thread t dequantizes 32 codes (4 * BITS bytes) of weight row t / 2 per group
  constexpr int GB = 8 * BITS; // bytes per column per group
  const short tidx = simd_gid * 32 + simd_lid;
  const short bi = tidx / 2, bj = (tidx % 2) * (GB / 2);
  const bool row_ok = y_col + bi < N;
  const int nrow = R0 + y_col + bi;
  const device uint8_t* wsrc = (const device uint8_t*)W + ((int64_t)(nrow / NT) * KG * (NT * GB) + (nrow % NT) * GB + bj);
  threadgroup bfloat* wdst = Ws + bi * BK_padded + (tidx % 2) * 32;
  constexpr short SM = BM / WM, SN = BN / WN, SKK = 32;
  constexpr short TM = SM / 16, TN = SN / 16, TK = SKK / 16;
  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);
  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);
  const short sgp_sn = ALIGNED_N ? SN : min(int(SN), N - (y_col + tn));
  const short tgp_bn = ALIGNED_N ? BN : min(BN, int(N - y_col));
  const bool is_unaligned_bn = ALIGNED_N ? false : (tgp_bn != BN);
  device bfloat* y = Y + (int64_t)y_row * N + y_col;
  const device bfloat* x = X + (int64_t)(y_row + tm) * K;
  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(ALIGNED_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int g = 0; g < KG; g++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_ok) {
          const device bfloat* sb = (g < KG / 2 ? ST : BT) + (size_t(g % (KG / 2)) * NTOT + nrow) * 2;
          const float s = float(sb[0]), b = float(sb[1]);
          const device uint8_t* w = wsrc + (int64_t)g * (NT * GB);
#if BITS == 4
          const float sc[2] = {s, s / 16.0f};
          for (int i = 0; i < 16; i++) {
            wdst[2 * i] = static_cast<bfloat>(sc[0] * (w[i] & 0x0f) + b);
            wdst[2 * i + 1] = static_cast<bfloat>(sc[1] * (w[i] & 0xf0) + b);
          }
#else
          // code i sits at bit i * BITS of the little-endian stream (MLX's `code * s + b`)
          for (int i = 0; i < 32; i++) {
            const int bit = i * BITS, by = bit >> 3, sh = bit & 7;
            const uint hi = sh + BITS > 8 ? uint(w[by + 1]) << 8 : 0u;
            const uint code = ((uint(w[by]) | hi) >> sh) & ((1u << BITS) - 1u);
            wdst[i] = static_cast<bfloat>(code * s + b);
          }
#endif
        } else {
          for (int i = 0; i < 32; i++) wdst[i] = bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SKK) {
          NAXTile<bfloat, TM, TK> Atile;
          NAXTile<bfloat, TN, TK> Btile;
          volatile int compiler_barrier;
          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SKK, sgp_sm));
          }
          Btile.template load<bfloat, BK_padded, 1>(Ws + tn * BK_padded + kk1);
          tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
          (void)compiler_barrier;
        }
        x += BK;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
