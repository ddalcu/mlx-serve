// GroupNorm statistics, pass 1: threadgroups of 256 threads, each over a chunk of PIX pixels of one image of
// X [B, H, W, C] (channels-last, C / GROUPS a multiple of 4). Each thread owns 4 adjacent channels (all in one group)
// and walks the chunk's pixel rows; per group, the sum and the sum of squares of x - pivot (PV [B, GROUPS] f32, the
// group's first value) go to OP [B, NCH, GROUPS, 2] f32.
threadgroup float red[2][256];
const uint C = X_shape[3];
const uint HW = X_shape[1] * X_shape[2];
const uint cg = C / GROUPS;
const uint tpp = C / 4;
const uint rows = 256 / tpp;
const uint lid = thread_position_in_threadgroup.x;
const uint chunk = threadgroup_position_in_grid.x;
const uint b = threadgroup_position_in_grid.y;
const uint nch = (HW + PIX - 1) / PIX;
const uint col = lid % tpp, r0 = lid / tpp;
const uint c0 = col * 4;
const device T* xb = X + size_t(b) * HW * C;
float s1 = 0.0f, s2 = 0.0f;
if (r0 < rows) {
    const float pivot = PV[b * GROUPS + c0 / cg];
    const uint p1 = min(HW, (chunk + 1) * PIX);
    for (uint p = chunk * PIX + r0; p < p1; p += rows) {
        const float4 v = float4(*(const device vec<T, 4>*)(xb + size_t(p) * C + c0)) - pivot;
        s1 += (v.x + v.y) + (v.z + v.w);
        s2 += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    }
}
red[0][lid] = s1;
red[1][lid] = s2;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < uint(GROUPS)) {
    float a = 0.0f, q = 0.0f;
    const uint q4 = cg / 4;
    for (uint rr = 0; rr < rows; rr++)
        for (uint cc = lid * q4; cc < (lid + 1) * q4; cc++) {
            a += red[0][rr * tpp + cc];
            q += red[1][rr * tpp + cc];
        }
    const size_t at = ((size_t(b) * nch + chunk) * GROUPS + lid) * 2;
    OP[at] = a;
    OP[at + 1] = q;
}
