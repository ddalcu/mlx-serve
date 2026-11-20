const int t = int(thread_position_in_threadgroup.x), h = int(threadgroup_position_in_grid.x), kvh = h / 7;
const int Lc = KC_shape[2], N = Lc + 1;
threadgroup float q[64];
threadgroup float sc[512];
threadgroup float rd[256];
if (t < 64) q[t] = 0.125f * float(QKV[h * 64 + t]);
for (int idx = h * 256 + t; idx < 2 * N * 64; idx += 14 * 256) {
    const int row = idx / 64, d = idx % 64, kh = row / N, j = row % N;
    KN[idx] = j < Lc ? KC[(kh * Lc + j) * 64 + d] : QKV[896 + kh * 64 + d];
    VN[idx] = j < Lc ? VC[(kh * Lc + j) * 64 + d] : QKV[1024 + kh * 64 + d];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float m = -INFINITY;
for (int j = t; j < N; j += 256) {
    const device bfloat16_t* kr = j < Lc ? KC + (kvh * Lc + j) * 64 : QKV + 896 + kvh * 64;
    float a = 0.0f;
    for (int d = 0; d < 64; d += 4) a += dot(LDW(kr, d), *(threadgroup const float4*)(q + d));
    sc[j] = a;
    m = max(m, a);
}
const int lane = t % 32, sgi = t / 32;
m = simd_max(m);
if (lane == 0) rd[sgi] = m;
threadgroup_barrier(mem_flags::mem_threadgroup);
m = rd[0];
for (int g = 1; g < 8; g++) m = max(m, rd[g]);
threadgroup_barrier(mem_flags::mem_threadgroup);
float sum = 0.0f;
for (int j = t; j < N; j += 256) { const float p = metal::exp(sc[j] - m); sc[j] = p; sum += p; }
sum = simd_sum(sum);
if (lane == 0) rd[sgi] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
sum = 0.0f;
for (int g = 0; g < 8; g++) sum += rd[g];
const int d = t % 64, grp = t / 64;
float acc = 0.0f;
for (int j = grp; j < N; j += 4)
    acc += sc[j] * float(j < Lc ? VC[(kvh * Lc + j) * 64 + d] : QKV[1024 + kvh * 64 + d]);
threadgroup_barrier(mem_flags::mem_threadgroup);
rd[t] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t < 64) OUT[h * 64 + t] = bfloat16_t((rd[t] + rd[t + 64] + rd[t + 128] + rd[t + 192]) / sum);
