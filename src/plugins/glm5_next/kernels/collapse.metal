uint tid  = thread_position_in_threadgroup.x;
uint row  = threadgroup_position_in_grid.x;
uint lane = tid % 32;
uint sg   = tid / 32;
constexpr int MIX = (2 + HC) * HC;
constexpr float EPS = EPS_INT * 1e-9;
threadgroup float pre_shared[HC];
if (sg == 0) {
    msv_hc_gates<HC, ITERS>(mixes + row * MIX, base, scale[0], scale[1], scale[2], EPS, lane,
                           pre_shared, post + row * HC, comb + row * HC * HC);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float p0 = pre_shared[0];
const float p1 = pre_shared[1];
const float p2 = pre_shared[2];
const float p3 = pre_shared[3];
const device T* x_row  = (const device T*)x_in + row * (HC * D);
device U*       out_row = (device U*)collapsed + row * D;
using T4 = vec<T, 4>;
using U4 = vec<U, 4>;
const device T4* x_row0 = (const device T4*)(x_row + 0*D);
const device T4* x_row1 = (const device T4*)(x_row + 1*D);
const device T4* x_row2 = (const device T4*)(x_row + 2*D);
const device T4* x_row3 = (const device T4*)(x_row + 3*D);
device U4*       out4   = (device U4*)out_row;
constexpr uint D4 = (uint)D / 4;
for (uint d4 = tid; d4 < D4; d4 += 256) {
    float4 x0 = float4(x_row0[d4]);
    float4 x1 = float4(x_row1[d4]);
    float4 x2 = float4(x_row2[d4]);
    float4 x3 = float4(x_row3[d4]);
    float4 result = fma(float4(p0), x0, fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
    out4[d4] = U4(result);
}