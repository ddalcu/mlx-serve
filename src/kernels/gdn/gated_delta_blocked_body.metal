constexpr int DB = 32;                             // dv rows per threadgroup
const int tid = thread_position_in_threadgroup.x;  // 0..255
const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
const int hv  = threadgroup_position_in_grid.y;
const int b   = threadgroup_position_in_grid.z;
const int hk  = hv / (Hv / Hk);
const int dv0 = blk * DB;

// thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
// the same simdgroup (lane = (dvr%4)*8 + seg).
const int dvr = tid / 8;            // 0..31
const int seg = tid % 8;            // 0..7
const int d0  = seg * 16;

threadgroup InT k_s[TB][Dk + 8];
threadgroup InT q_s[TB][Dk + 8];
threadgroup InT v_s[TB][DB + 8];
threadgroup float g_s[TB];
threadgroup float b_s[TB];

auto k_base = k + ((size_t)b * T * Hk + hk) * Dk;
auto q_base = q + ((size_t)b * T * Hk + hk) * Dk;
auto v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
const size_t krow = (size_t)Hk * Dk;

// state fragment in registers: [dv0+dvr][d0..d0+16]
float4 st[4];
{
    const device vec<StT,4>* S_in = (const device vec<StT,4>*)(
        state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    for (int i = 0; i < 4; ++i) st[i] = float4(S_in[i]);
}

device OutT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

for (int t0 = 0; t0 < T; t0 += TB) {
    const int tt = min(TB, T - t0);
    // cooperative staging (coalesced): k/q rows, v slice, g/beta
    for (int p = tid; p < tt * Dk; p += 256) {
        const int r = p / Dk, d = p % Dk;
        k_s[r][d] = static_cast<InT>(k_base[(size_t)(t0 + r) * krow + d]);
        q_s[r][d] = static_cast<InT>(q_base[(size_t)(t0 + r) * krow + d]);
    }
    for (int p = tid; p < tt * DB; p += 256) {
        const int r = p / DB, d = p % DB;
        v_s[r][d] = static_cast<InT>(v_base[(size_t)(t0 + r) * Hv * Dv + d]);
    }
    for (int p = tid; p < tt; p += 256) {
        g_s[p] = (float)g[((size_t)b * T + t0 + p) * Hv + hv];
        b_s[p] = (float)beta[((size_t)b * T + t0 + p) * Hv + hv];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < tt; ++t) {
        const float gt = g_s[t];
        const float bt = b_s[t];
        const threadgroup vec<InT,4>* k4 =
            (const threadgroup vec<InT,4>*)&k_s[t][d0];
        const threadgroup vec<InT,4>* q4 =
            (const threadgroup vec<InT,4>*)&q_s[t][d0];
        float4 kf[4];
        for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
        // kv_mem = (g*state) . k ; decay applied to state first
        float4 p4 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[i] *= gt;
            p4 += st[i] * kf[i];
        }
        float part = p4.x + p4.y + p4.z + p4.w;
        // reduce across the 8 segment-threads of this dv row
        part += simd_shuffle_down(part, 4);
        part += simd_shuffle_down(part, 2);
        part += simd_shuffle_down(part, 1);
        const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
        const float delta = ((float)v_s[t][dvr] - kv_mem) * bt;

        float4 o4 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[i] += kf[i] * delta;
            o4 += st[i] * float4(q4[i]);
        }
        float out = o4.x + o4.y + o4.z + o4.w;
        out += simd_shuffle_down(out, 4);
        out += simd_shuffle_down(out, 2);
        out += simd_shuffle_down(out, 1);
        if (seg == 0) {
            y_base[(size_t)(t0 + t) * Hv * Dv + dvr] = static_cast<OutT>(out);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

{
    device vec<StT,4>* S_out = (device vec<StT,4>*)(
        state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    for (int i = 0; i < 4; ++i) S_out[i] = vec<StT,4>(st[i]);
}