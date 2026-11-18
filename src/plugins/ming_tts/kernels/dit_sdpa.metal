// Per head h (one threadgroup of 64 threads), attention over the q/k/v rows `ming_dit_qkv` wrote:
// 8 lanes a row, 8 dims a lane.
const int t = int(thread_position_in_threadgroup.x), lane = int(thread_index_in_simdgroup);
const int h = int(threadgroup_position_in_grid.x);
#define HQ(row, d) LDW(QKV, (row) * 1024 + h * 64 + (d))
for (int r = t >> 3; r < R; r += 8) {
    const int b = r / L, d = (lane & 7) * 8;
    const float4 qa = 0.125f * HQ(r, d), qb = 0.125f * HQ(r, d + 4);
    float sc[L];
    float m = -INFINITY;
    for (int j = 0; j < L; j++) {
        const int kr = R + b * L + j;
        float v = dot(qa, HQ(kr, d)) + dot(qb, HQ(kr, d + 4));
        v += simd_shuffle_xor(v, 1);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 4);
        sc[j] = v;
        m = max(m, v);
    }
    float sum = 0.0f;
    float4 oa = 0.0f, oc = 0.0f;
    for (int j = 0; j < L; j++) {
        const float p = metal::exp(sc[j] - m);
        const int vr = 2 * R + b * L + j;
        sum += p; oa += p * HQ(vr, d); oc += p * HQ(vr, d + 4);
    }
    for (int k = 0; k < 4; k++) {
        OUT[r * 1024 + h * 64 + d + k] = bfloat16_t(oa[k] / sum);
        OUT[r * 1024 + h * 64 + d + 4 + k] = bfloat16_t(oc[k] / sum);
    }
}
