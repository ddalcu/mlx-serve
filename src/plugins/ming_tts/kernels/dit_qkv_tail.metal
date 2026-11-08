for (int ei = int(t); ei < NO / 2 * R; ei += 32 * SG) {
    const int n = (ei / R) * 2, r = ei % R, col = col0 + n;
    const float inv = rinv(r);
    float e = rb(rsum(n, r) * inv + float(Bv[col]));
    float od = rb(rsum(n + 1, r) * inv + float(Bv[col + 1]));
    if (which < 2) {
        // Interleaved-pair rotary in f32 (MLX: bf16 * f32 table), rounded once.
        const int pi = (col % 64) / 2, l = r % L;
        const float c = float(COS[l * 32 + pi]), s = float(SIN[l * 32 + pi]);
        const float ne = e * c - od * s, no = e * s + od * c;
        e = ne; od = no;
    }
    OUT[(which * R + r) * 1024 + col] = bfloat16_t(e);
    OUT[(which * R + r) * 1024 + col + 1] = bfloat16_t(od);
}
// Per head h, the last threadgroup to finish h's q/k/v columns computes
// h's attention into ATT. CNT[h] is an output MLX zeroes before the dispatch
// (`init_value` 0), and the NTG = 3 * 64 / NO threadgroups of h bump it.
threadgroup_barrier(mem_flags::mem_device);
threadgroup bool last;
if (t == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const uint old = atomic_fetch_add_explicit((device atomic_uint*)CNT + col0 / 64, 1u, memory_order_relaxed);
    last = old == uint(NTG - 1);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (!last) return;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
// Stage head h's q/k/v rows through atomic loads: they are served from
// the coherent L2, so no stale L1 line (MLX recycles buffers) is read.
threadgroup uint hq[3 * R * 32];
const int h = col0 / 64;
for (int u = int(t); u < 3 * R * 32; u += 32 * SG) {
    const int row = u / 32, w = u % 32;
    hq[u] = atomic_load_explicit((device atomic_uint*)(OUT + row * 1024 + h * 64) + w, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
#define HQ(row, d) bf4(uint2(hq[(row) * 32 + (d) / 2], hq[(row) * 32 + (d) / 2 + 1]))
for (int r = int(t) >> 3; r < R; r += 4 * SG) {
    const int b = r / L, d = (int(lane) & 7) * 8;
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
        ATT[r * 1024 + h * 64 + d + k] = bfloat16_t(oa[k] / sum);
        ATT[r * 1024 + h * 64 + d + 4 + k] = bfloat16_t(oc[k] / sum);
    }
}
