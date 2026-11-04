
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int LDB = BK + 8;
    constexpr int KT = BK / 16;
    constexpr int NT = BN / 16;
    constexpr int NSG = 4;
    constexpr int KD = 2304;
    constexpr int ND = 5120;
    constexpr int WPT = 24;
    constexpr int K16 = KD / 16;
    constexpr int N16 = ND / 16;
    constexpr int NTILES = ND / BN;
    threadgroup uint4 dig_smem[BN * LDB * 2 / 16];
    threadgroup half* Bs = reinterpret_cast<threadgroup half*>(dig_smem);
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = tid & 31u;
    const int g = int(threadgroup_position_in_grid.x);
    const int s = g / NTILES;
    const int n0 = (g - s * NTILES) * BN;
    const uint h = sg & 1u;
    const uint c7 = lane & 7u;
    int dst_off[2];
    int src_off[2];
    for (int jj = 0; jj < 2; ++jj) {
        const int t = int(lane >> 3u) + 4 * (int(sg >> 1u) + (NSG / 2) * jj);
        const int kt = t / NT;
        const int nt = t % NT;
        dst_off[jj] = (nt * 16 + 8 * int(h) + int(c7)) * LDB + kt * 16;
        src_off[jj] = (kt * N16 + nt) * WPT;
    }
    const uint o0 = (3u * c7 + 23u) % 24u;
    const uint o1 = (3u * c7 + 24u) % 24u;
    const uint o2 = (3u * c7 + 25u) % 24u;
    const uint o3 = (3u * c7 + 26u) % 24u;
    const device uint* cd = reinterpret_cast<const device uint*>(code) + size_t(n0 / 16) * WPT;
    for (int jj = 0; jj < 2; ++jj) {
        const device uint* tile = cd + size_t(s) * size_t(KT * N16 * WPT) + src_off[jj];
        if (h != 0u) {
            dig_dec_job<half, 1>(tile, o0, o1, o2, o3, Bs + dst_off[jj]);
        } else {
            dig_dec_job<half, 0>(tile, o0, o1, o2, o3, Bs + dst_off[jj]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < uint(BN * BK); e += 128u) {
        const uint n = e / uint(BK);
        const uint k = e - n * uint(BK);
        bt[(size_t(n0) + n) * size_t(KD) + size_t(s * BK) + k] = Bs[n * LDB + k];
    }
    (void)K16;
