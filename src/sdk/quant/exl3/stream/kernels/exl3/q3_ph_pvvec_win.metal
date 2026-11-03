
    // q3_attnfuse_pv: o = (ex @ KV) / denom for query row z; stock order steel_gemm_fused_nax nn
    // (2 x 256 full BK iterations for k 640, then the K tail with load_safe zero fill) + Divide.
    static_assert(metal::is_same_v<TKV, bfloat16_t> || metal::is_same_v<TKV, float>, "KV is bf16 or f32");
    constexpr short SM = 32, SN = 32, SK = 32, BK = 256, TM = 2, TN = 2, TK = 2;
    constexpr int HD = 512;
    const int tile = int(threadgroup_position_in_grid.x);
    const int z = int(threadgroup_position_in_grid.z);
    const short sg = short(simdgroup_index_in_threadgroup);
    const int S = widx_shape[1];
    const int W = widx_shape[2];
    const int KT = W;
    const int H = ex_shape[2];
    const int bi = z / S;
    const int si = z - bi * S;

    auto kv_row = [&](int j, thread bool& v, thread int64_t& es) -> const device TKV* {
        v = wval[bi * wval_strides[0] + si * wval_strides[1] + j * wval_strides[2]];
        const int r = v ? int(widx[bi * widx_strides[0] + si * widx_strides[1] + j * widx_strides[2]]) : 0;
        es = win_strides[2];
        return win + bi * win_strides[0] + int64_t(r) * win_strides[1];
    };

    const int c_col = tile * 128;
    const short tm = SM * (sg / 4);
    const short tn = SN * (sg % 4);
    const short2 sc = BaseNAXFrag::get_coord();
    const device float* A = ex + int64_t(z) * H * KT;           // ex: row-contiguous [b, s, H, KT]
    const int iters = KT / BK;
    const int rem = KT - iters * BK;
    auto load_ab = [&](thread NAXTile<float, TM, TK>& At, thread NAXTile<float, TK, TN>& Bt, const int k0,
                       const short psk) {
        STEEL_PRAGMA_UNROLL
        for (short fr = 0; fr < 2; fr++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const short kr = fr * 16 + sc.y + i * 8;           // B row (key) inside the chunk
                bool v;
                int64_t es;
                const device TKV* row = kv_row(kr < psk ? k0 + kr : k0, v, es);
                const int head = tm + fr * 16 + sc.y + i * 8;      // A row
                STEEL_PRAGMA_UNROLL
                for (short fc = 0; fc < 2; fc++) {
                    thread auto& a = At.frag_at(fr, fc);
                    thread auto& b = Bt.frag_at(fr, fc);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        const short kc = fc * 16 + sc.x + j;       // A col (key) inside the chunk
                        a[i * 4 + j] = kc < psk ? A[int64_t(head) * KT + k0 + kc] : float(0);
                        b[i * 4 + j] = kr < psk ? static_cast<float>(row[int64_t(c_col + tn + kc)]) : float(0);
                    }
                }
            }
        }
    };
    NAXTile<float, TM, TN> Dt;
    Dt.clear();
    STEEL_PRAGMA_NO_UNROLL
    for (int kk0 = 0; kk0 < iters; kk0++) {
        STEEL_PRAGMA_NO_UNROLL
        for (short kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<float, TM, TK> At;
            NAXTile<float, TK, TN> Bt;
            load_ab(At, Bt, kk0 * BK + kk1, SK);
            tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<false>{});
        }
    }
    STEEL_PRAGMA_NO_UNROLL
    for (short kk1 = 0; kk1 < rem; kk1 += SK) {
        NAXTile<float, TM, TK> At;
        NAXTile<float, TK, TN> Bt;
        load_ab(At, Bt, iters * BK + kk1, short(max(0, rem - kk1)));
        tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<false>{});
    }
    STEEL_PRAGMA_UNROLL
    for (short fr = 0; fr < 2; fr++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const int head = tm + fr * 16 + sc.y + i * 8;
            const float dn = denom[int64_t(z) * H + head];
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < 2; fc++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const int dim = c_col + tn + fc * 16 + sc.x + j;
                    o[(int64_t(z) * H + head) * HD + dim] = Dt.frag_at(fr, fc)[i * 4 + j] / dn;
                }
            }
        }
    }
