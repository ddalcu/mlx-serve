{{
float acc[NO][R];
float ss[R];
for (int r = 0; r < R; r++) {{ ss[r] = 0.0f; for (int n = 0; n < NO; n++) acc[n][r] = 0.0f; }}
for (int i = int(sg) * 128 + int(lane) * 4; i < {0d}; i += SG * 128) {{
    float4 xv[R];
    for (int r = 0; r < R; r++) xv[r] = {2s};
    {3s}
    for (int n = 0; n < NO; n++) {{
        const float4 w = LDW({1s}, i);
        for (int r = 0; r < R; r++) acc[n][r] += dot(w, xv[r]);
    }}
}}
for (int r = 0; r < R; r++) {{
    for (int n = 0; n < NO; n++) {{
        const float v = simd_sum(acc[n][r]);
        if (lane == 0) red[(int(sg) * NO + n) * R + r] = v;
    }}
    {4s}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
}}
#define rsum(n, r) rsum_(red, SG, NO, R, n, r)
#define rinv(r) rinv_(ssr, SG, R, r, {0d})
