
    // q3_attnfuse_qk: unscaled QK^T of query row z against its gathered keys + the valid mask.
    // Stock order: steel_gemm_fused_nax nt (SPLIT = 0) / steel_gemm_splitk_nax nt + accum (SPLIT = 1).
    static_assert(metal::is_same_v<TQ, bfloat16_t> || metal::is_same_v<TQ, float>, "q is bf16 or f32");
    static_assert(metal::is_same_v<TKV, bfloat16_t> || metal::is_same_v<TKV, float>, "KV is bf16 or f32");
    constexpr short SM = 32, SN = 32, SK = 32, BK = 256, TM = 2, TN = 2, TK = 2;
    constexpr int KD = 512;
    const int tile = int(threadgroup_position_in_grid.x);
    const int z = int(threadgroup_position_in_grid.z);
    const short sg = short(simdgroup_index_in_threadgroup);
    const short lane = short(thread_index_in_simdgroup);
    const int S = widx_shape[1];
    const int W = widx_shape[2];
    const int KT = W;
    const int H = q_shape[2];
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
    const int sgp_sn = min(int(SN), KT - (c_col + tn));
    if (sg < 4 && lane < sgp_sn) {
        bool v;
        int64_t es;
        (void)kv_row(c_col + tn + lane, v, es);
        valid[int64_t(z) * KT + c_col + tn + lane] = v;
    }
    if (sgp_sn <= 0) {
        return;
    }
    const short2 sc = BaseNAXFrag::get_coord();
    const device TQ* qz = q + bi * q_strides[0] + si * q_strides[1];
    // q3_prefill_attnhalf ropefuse: q arrives UN-roped; its last 2 * HALF dims are rotated in the load
    const int HALF = qcos_shape[1];
    const int R0 = KD - 2 * HALF;
    const device float* qc = qcos + int64_t(si) * HALF;
    const device float* qs = qsin + int64_t(si) * HALF;
    const int64_t qh = q_strides[2];
    const device TQ* arow[2][2];
    const device TKV* brow[2][2];
    int64_t bes[2][2];
    bool bin[2][2];
    STEEL_PRAGMA_UNROLL
    for (short fr = 0; fr < 2; fr++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const short rr = fr * 16 + sc.y + i * 8;
            arow[fr][i] = qz + int64_t(tm + rr) * qh;
            bin[fr][i] = rr < sgp_sn;                       // load_safe row limit (the N tail)
            bool v;
            brow[fr][i] = kv_row(bin[fr][i] ? c_col + tn + rr : c_col + tn, v, bes[fr][i]);
        }
    }
    auto load_ab = [&](thread NAXTile<float, TM, TK>& At, thread NAXTile<float, TN, TK>& Bt, const int k0) {
        STEEL_PRAGMA_UNROLL
        for (short fr = 0; fr < 2; fr++) {
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < 2; fc++) {
                thread auto& a = At.frag_at(fr, fc);
                thread auto& b = Bt.frag_at(fr, fc);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const int kb = k0 + fc * 16 + sc.x;         // a multiple of 4: 2 whole rope pairs
                    if (kb >= R0) {
                        // _rope_last(q): r0 = x0*c - x1*s, r1 = x0*s + x1*c (f32, one rounding each), cast
                        // to the q dtype (the stock astype + concatenate), widened as the stock load does
                        const int p = (kb - R0) >> 1;
                        STEEL_PRAGMA_UNROLL
                        for (short h = 0; h < 2; h++) {
                            const float x0 = static_cast<float>(arow[fr][i][kb + 2 * h]);
                            const float x1 = static_cast<float>(arow[fr][i][kb + 2 * h + 1]);
                            const float cv = qc[p + h];
                            const float sv = qs[p + h];
                            a[i * 4 + 2 * h] = static_cast<float>(static_cast<TQ>(
                                metal::fma(x0, cv, -0.0f) - metal::fma(x1, sv, -0.0f)));
                            a[i * 4 + 2 * h + 1] = static_cast<float>(static_cast<TQ>(
                                metal::fma(x0, sv, -0.0f) + metal::fma(x1, cv, -0.0f)));
                        }
                    } else {
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++) {
                            a[i * 4 + j] = static_cast<float>(arow[fr][i][kb + j]);
                        }
                    }
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        b[i * 4 + j] = bin[fr][i] ? static_cast<float>(brow[fr][i][kb + j]) : float(0);
                    }
                }
            }
        }
    };
    auto store = [&](const int fr, const int fc, const int i, const int j, const float val) {
        const int head = tm + fr * 16 + sc.y + i * 8;
        const int kc = fc * 16 + sc.x + j;
        if (kc < sgp_sn) {
            scores[(int64_t(z) * H + head) * KT + c_col + tn + kc] = val;
        }
    };
    if constexpr (SPLIT) {
        // gemm_splitk_nax: partition p = [256 p, 256 p + 256), C cleared per partition; then accum
        NAXTile<float, TM, TN> D0;
        NAXTile<float, TM, TN> D1;
        D0.clear();
        D1.clear();
        STEEL_PRAGMA_NO_UNROLL
        for (short kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<float, TM, TK> At;
            NAXTile<float, TN, TK> Bt;
            load_ab(At, Bt, kk1);
            tile_matmad_nax(D0, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
        }
        STEEL_PRAGMA_NO_UNROLL
        for (short kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<float, TM, TK> At;
            NAXTile<float, TN, TK> Bt;
            load_ab(At, Bt, BK + kk1);
            tile_matmad_nax(D1, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
        }
        STEEL_PRAGMA_UNROLL
        for (short fr = 0; fr < 2; fr++) {
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < 2; fc++) {
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < 8; e++) {
                    float out = 0;                              // gemm_splitk_accum: out = 0; out += C[p]
                    out += D0.frag_at(fr, fc)[e];
                    out += D1.frag_at(fr, fc)[e];
                    store(fr, fc, e / 4, e % 4, out);
                }
            }
        }
    } else {
        // steel_gemm_fused_nax: one accumulator over K = 512 (2 BK iterations of 8 SK steps)
        NAXTile<float, TM, TN> Dt;
        Dt.clear();
        STEEL_PRAGMA_NO_UNROLL
        for (int kk0 = 0; kk0 < KD / BK; kk0++) {
            STEEL_PRAGMA_NO_UNROLL
            for (short kk1 = 0; kk1 < BK; kk1 += SK) {
                NAXTile<float, TM, TK> At;
                NAXTile<float, TN, TK> Bt;
                load_ab(At, Bt, kk0 * BK + kk1);
                tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short fr = 0; fr < 2; fr++) {
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < 2; fc++) {
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < 8; e++) {
                    store(fr, fc, e / 4, e % 4, Dt.frag_at(fr, fc)[e]);
                }
            }
        }
    }
