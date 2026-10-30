
    constexpr int SN = 32, TN = 4;
    constexpr int blockN = BN * SN * TN;
    constexpr int ROWS_TG = BMK * TMK;
    static_assert(K % blockN == 0, "leftover path not transcribed");
    static_assert(N % ROWS_TG == 0, "row tail not transcribed");
    threadgroup float tgp_memory[BN > 1 ? MV * BMK * BN * TMK : 1];
    const int simd_gid = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int sgN = simd_gid % BN;
    const int sgM = simd_gid / BN;
    int bn = (SN * sgN + lane) * TN;
    const int row0 = int(threadgroup_position_in_grid.x) * ROWS_TG + sgM * TMK;
    const auto mat = w + size_t(row0) * K;
    float result[MV][TMK];
    for (int v = 0; v < MV; v++) {
        for (int tm = 0; tm < TMK; tm++) {
            result[v][tm] = 0;
        }
    }
    for (int i = 0; i < K / blockN; ++i) {
        for (int tm = 0; tm < TMK; tm++) {
            T inter[TN];
            for (int tn = 0; tn < TN; tn++) {
                inter[tn] = static_cast<T>(mat[tm * K + bn + tn]);
            }
            for (int v = 0; v < MV; v++) {
                float v_coeff[TN];
                for (int tn = 0; tn < TN; tn++) {
                    v_coeff[tn] = static_cast<float>(a[v * K + bn + tn]);
                }
                for (int tn = 0; tn < TN; tn++) {
                    result[v][tm] += inter[tn] * v_coeff[tn];
                }
            }
        }
        bn += blockN;
    }
    for (int v = 0; v < MV; v++) {
        for (int tm = 0; tm < TMK; tm++) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[v][tm] += simd_shuffle_down(result[v][tm], sn);
            }
        }
    }
    if (BN > 1) {
        if (lane == 0) {
            for (int v = 0; v < MV; v++) {
                for (int tm = 0; tm < TMK; tm++) {
                    tgp_memory[((v * BMK + sgM) * BN + sgN) * TMK + tm] = result[v][tm];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && sgN == 0) {
            for (int sgn = 1; sgn < BN; sgn++) {
                for (int v = 0; v < MV; v++) {
                    for (int tm = 0; tm < TMK; tm++) {
                        result[v][tm] += tgp_memory[((v * BMK + sgM) * BN + sgn) * TMK + tm];
                    }
                }
            }
        }
    }
    if (lane == 0 && sgN == 0) {
        for (int v = 0; v < MV; v++) {
            for (int tm = 0; tm < TMK; tm++) {
                out[v * N + row0 + tm] = static_cast<TO>(result[v][tm]);
            }
        }
    }
