const uint lane = thread_index_in_simdgroup;
const int o = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
const int r0 = (L - 1) * 1024, r1 = (2 * L - 1) * 1024;
float a0 = 0.0f, a1 = 0.0f, s0 = 0.0f, s1 = 0.0f;
for (int b = 0; b < 1024; b += 128) {
    const int i = b + int(lane) * 4;
    const float4 w = LDW(WF + o * 1024, i) * LDW(NW, i);
    const float4 x0 = LDW(X, r0 + i), x1 = LDW(X, r1 + i);
    s0 += dot(x0, x0); s1 += dot(x1, x1);
    a0 += dot(w, x0); a1 += dot(w, x1);
}
a0 = simd_sum(a0); a1 = simd_sum(a1); s0 = simd_sum(s0); s1 = simd_sum(s1);
if (lane == 0) {
    a0 *= metal::precise::rsqrt(s0 / 1024.0f + 1e-6f);
    a1 *= metal::precise::rsqrt(s1 / 1024.0f + 1e-6f);
    const float pc = rb(a0 + float(BF[o])), pu = rb(a1 + float(BF[o]));
    const float f0 = rb(pc + rb(rb(pc - pu) * rb(as_type<float>(CFG))));
    float y = rb(float(Y[o]) + rb(float(DTT[S]) * f0));
    if (SHIFT != 0) y = rb(y + rb(rb(rb(as_type<float>(SHIFT)) * float(SQ[S])) * float(SDE[S * 64 + o])));
    OUT[o] = bfloat16_t(y);
}
