const uint t = thread_position_in_threadgroup.x;
const uint r = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
threadgroup float local_sums[32];
float hv[NC * 4];
float acc = 0.0f;
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {{
    const int c = p * TN * 4 + int(t) * 4 + i;
    hv[p * 4 + i] = 0.0f;
    if (c < D) {{
        const size_t at = size_t(r) * D + c;
        float delta;
        {s}
        const T hn = T(float(H[at]) + delta);
        {s}
        hv[p * 4 + i] = float(hn);
    }}
    acc += hv[p * 4 + i] * hv[p * 4 + i];
}}
acc = simd_sum(acc);
if (sg == 0) local_sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) local_sums[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float total = simd_sum(local_sums[lane]);
const float inv = metal::precise::rsqrt(total / float(D) + eps);
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {{
    const int c = p * TN * 4 + int(t) * 4 + i;
    if (c < D) {s}
}}