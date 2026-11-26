// Inkling 2 DFlash2: merge SHARDS sorted top-K lists per row into exact top-K [R_TOTAL, K].
// grid (32, R_TOTAL, 1), threadgroup (32, 1, 1). SHARDS <= 32.
const uint row = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
uint head = 0;
const size_t in_base = size_t(row) * SHARDS * K + size_t(lane) * K;
const size_t out_base = size_t(row) * K;
for (uint m = 0; m < K; ++m) {
    float cur_v = (lane < SHARDS && head < K) ? PVALS[in_base + head] : -INFINITY;
    int cur_i = (lane < SHARDS && head < K) ? PIDS[in_base + head] : 0x7FFFFFFF;
    uint cur_l = lane;
    for (uint off = 16u; off > 0u; off >>= 1) {
        const float ov = simd_shuffle_xor(cur_v, off);
        const int oi = simd_shuffle_xor(cur_i, off);
        const uint ol = simd_shuffle_xor(cur_l, off);
        if (ov > cur_v || (ov == cur_v && oi < cur_i)) {
            cur_v = ov;
            cur_i = oi;
            cur_l = ol;
        }
    }
    if (lane == 0u) {
        IDS[out_base + m] = cur_i;
        VALS[out_base + m] = cur_v;
    }
    const uint win = simd_broadcast(cur_l, 0);
    if (lane == win && head < K) ++head;
}
