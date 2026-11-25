// Inkling 2: compact the unique active experts across S rows (top-K picks each) into a dense list of at most NMAX
// slots so grouped MoE kernels read each expert's weights once for all rows that picked it.
// IDX [1, S, KSEL] u32. grid (TG, 1, 1), threadgroup (TG, 1, 1), TG = ceil(E / 32) * 32.
// ACT_E [NMAX] u32 (0xFFFFFFFF for unused), ACT_CNT [NMAX] u32, ACT_PAIR [NMAX, S_CAP] u32 (flattened pair indices).
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = t >> 5;
constexpr uint NSG = TG / 32;
const uint N = IDX_shape[1] * KSEL;
threadgroup uint sh_idx[S_CAP * KSEL];
threadgroup uint sg_tot[NSG];

for (uint p = t; p < N; p += TG) sh_idx[p] = IDX[p];
threadgroup_barrier(mem_flags::mem_threadgroup);

uint my_pairs[S_CAP];
uint cnt = 0u;
if (t < E) {
    for (uint p = 0u; p < N; ++p) {
        if (sh_idx[p] == t && cnt < S_CAP) my_pairs[cnt++] = p;
    }
}
const uint has = cnt > 0u ? 1u : 0u;
const uint p_in_sg = simd_prefix_exclusive_sum(has);
if (lane == 31u) sg_tot[sg] = p_in_sg + has;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = 0u;
for (uint g = 0u; g < sg; ++g) base += sg_tot[g];
uint total = 0u;
for (uint g = 0u; g < NSG; ++g) total += sg_tot[g];

if (cnt > 0u) {
    const uint slot = base + p_in_sg;
    ACT_E[slot] = t;
    ACT_CNT[slot] = cnt;
    for (uint m = 0u; m < cnt; ++m) ACT_PAIR[slot * S_CAP + m] = my_pairs[m];
    for (uint m = cnt; m < S_CAP; ++m) ACT_PAIR[slot * S_CAP + m] = 0u;
}
for (uint u = total + t; u < NMAX; u += TG) {
    ACT_E[u] = 0xffffffffu;
    ACT_CNT[u] = 0u;
    for (uint m = 0u; m < S_CAP; ++m) ACT_PAIR[u * S_CAP + m] = 0u;
}
