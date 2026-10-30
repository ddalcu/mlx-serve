const uint t = thread_position_in_threadgroup.x;
const uint grp = threadgroup_position_in_grid.y;
const uint r = threadgroup_position_in_grid.z;
constexpr int TN = GS / 4;
threadgroup float partial[TN / 32];
const size_t base = size_t(r) * XD + size_t(grp) * GS + size_t(t) * 4;
float v[4];
float ss = 0.0f;
for (int i = 0; i < 4; i++) { v[i] = float(X[base + i]); ss = metal::fma(v[i], v[i], ss); }
ss = simd_sum(ss);
if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = 0.0f;
for (int s = 0; s < TN / 32; s++) total += partial[s];
const float scale = metal::rsqrt(total / float(GS) + eps);
for (int i = 0; i < 4; i++) {
    const int c = int(grp) * GS + int(t) * 4 + i;
    OUT[base + i] = T(float(W[c]) * float(T(v[i] * scale)));
}