const uint tg = threadgroup_position_in_grid.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32 + lane;
constexpr int K = HC * D;
constexpr int MIX = (2 + HC) * HC;
constexpr int KS = K / SG;
const int rows = params;
const short qid = short(lane / 4);
const short mr = (qid & 4) + short((lane / 2) % 4);
const short mc = (qid & 2) * 2 + short(lane % 2) * 2;
const int r = int(tg) * 8 + mr;
const bool rv = r < rows;
const device T* xrow = stream + (size_t)(rv ? r : 0) * K;
simdgroup_matrix<float, 8, 8> a, b;
simdgroup_matrix<float, 8, 8> c[MIX / 8];
for (int t = 0; t < MIX / 8; ++t) c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
float ss = 0.0f;
for (int k0 = int(sg) * KS; k0 < int(sg + 1) * KS; k0 += 8) {
    const float2 av = rv ? float2(float(xrow[k0 + mc]), float(xrow[k0 + mc + 1])) : float2(0.0f);
    ss = fma(av.x, av.x, fma(av.y, av.y, ss));
    reinterpret_cast<thread float2&>(a.thread_elements()) = av;
    const device float* wcol = fn + (size_t)mc * K + k0 + mr;
    for (int t = 0; t < MIX / 8; ++t) {
        reinterpret_cast<thread float2&>(b.thread_elements()) = float2(wcol[t * 8 * K], wcol[(t * 8 + 1) * K]);
        simdgroup_multiply_accumulate(c[t], a, b, c[t]);
    }
}
threadgroup float part[SG][8][MIX];
threadgroup float ssp[SG][32];
threadgroup float rowss[8];
for (int t = 0; t < MIX / 8; ++t) {
    const float2 cv = reinterpret_cast<thread float2&>(c[t].thread_elements());
    part[sg][mr][t * 8 + mc] = cv.x;
    part[sg][mr][t * 8 + mc + 1] = cv.y;
}
ssp[sg][lane] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 8) {
    float acc = 0.0f;
    for (uint g = 0; g < SG; ++g)
        for (uint l = 0; l < 32; ++l) {
            const short q = short(l / 4);
            if ((q & 4) + short((l / 2) % 4) == short(tid)) acc += ssp[g][l];
        }
    rowss[tid] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 8 * MIX) {
    const int row = int(tid) / MIX;
    const int col = int(tid) % MIX;
    const int gr = int(tg) * 8 + row;
    if (gr < rows) {
        float acc = 0.0f;
        for (uint g = 0; g < SG; ++g) acc += part[g][row][col];
        mixes[(size_t)gr * MIX + col] = acc * metal::precise::rsqrt(rowss[row] / float(K) + eps[0]);
    }
}