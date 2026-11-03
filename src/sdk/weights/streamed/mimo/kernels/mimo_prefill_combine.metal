constexpr uint C = 4;
constexpr uint DC = uint(H) / C;
const uint gid = thread_position_in_grid.x;
const uint t = gid / DC;
const uint col = (gid - t * DC) * C;
const ulong lb = ulong(t) * ulong(TOPK);
float acc[C];
{
    const float wk = w[lb];
    const device T* p = mimo_pc_pick<T>(loc[2 * lb], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23) + ulong(loc[2 * lb + 1]) * ulong(H) + col;
    for (uint i = 0; i < C; ++i) {
        acc[i] = static_cast<float>(p[i]) * wk + 0.0f;
    }
}
for (uint k = 1; k < uint(TOPK); ++k) {
    const float wk = w[lb + k];
    const device T* p = mimo_pc_pick<T>(loc[2 * (lb + k)], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23) + ulong(loc[2 * (lb + k) + 1]) * ulong(H) + col;
    for (uint i = 0; i < C; ++i) {
        const float q = static_cast<float>(p[i]) * wk + 0.0f;
        acc[i] = q + acc[i];
    }
}
device T* o = y + ulong(t) * ulong(H) + col;
for (uint i = 0; i < C; ++i) {
    o[i] = static_cast<T>(acc[i]);
}
