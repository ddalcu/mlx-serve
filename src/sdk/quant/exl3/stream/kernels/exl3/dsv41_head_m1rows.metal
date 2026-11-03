
    constexpr int BM = 8, SN = 32, TM = 4, TN = 4;
    constexpr int blockM = BM * TM;              // BM * SM * TM, SM = 1
    constexpr int blockN = SN * TN;              // BN * SN * TN, BN = 1
    static_assert(K % blockN == 0, "leftover path not transcribed");
    static_assert(N % blockM == 0, "tail row shift not transcribed");
    const uint3 tid = threadgroup_position_in_grid;
    const int simd_gid = int(simdgroup_index_in_threadgroup);
    const int simd_lid = int(thread_index_in_simdgroup);
    const int thrN = simd_lid;                   // SN == 32
    const int simdM = simd_gid;                  // BN == 1: SM * simd_gid, SM = 1
    const int bm = simdM * TM;                   // (simdM + thrM) * TM, thrM = 0
    int bn = thrN * TN;                          // (simdN + thrN) * TN, simdN = 0
    const int out_row = int(tid.x) * blockM + bm;
    const device bfloat16_t* mat = w + size_t(out_row) * K;
    float result[MV][TM];
    for (int v = 0; v < MV; v++) {
        for (int tm = 0; tm < TM; tm++) {
            result[v][tm] = 0;
        }
    }
    bfloat16_t inter[TM][TN];
    float v_coeff[TN];
    for (int i = 0; i < K / blockN; ++i) {
        int mat_offset = 0;
        for (int tm = 0; tm < TM; tm++) {
            for (int tn = 0; tn < TN; tn++) {
                inter[tm][tn] = mat[mat_offset + bn + tn];
            }
            mat_offset += K;
        }
        for (int v = 0; v < MV; v++) {
            for (int tn = 0; tn < TN; tn++) {
                v_coeff[tn] = static_cast<float>(x[v * K + bn + tn]);
            }
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    result[v][tm] += inter[tm][tn] * v_coeff[tn];
                }
            }
        }
        bn += blockN;
    }
    for (int v = 0; v < MV; v++) {
        for (int tm = 0; tm < TM; tm++) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[v][tm] += simd_shuffle_down(result[v][tm], sn);
            }
        }
    }
    if (thrN == 0) {
        for (int v = 0; v < MV; v++) {
            for (int tm = 0; tm < TM; tm++) {
                y[v * N + out_row + tm] = static_cast<bfloat16_t>(result[v][tm]);
            }
        }
    }
