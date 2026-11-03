constexpr uint REAL_COUNT = (uint)RC;
constexpr uint TG_SIZE    = 256;
constexpr uint STRIDE     = 64u * 256u;
constexpr uint PER_THREAD = (REAL_COUNT + STRIDE - 1u) / STRIDE;
constexpr uint TOPK       = 32;
constexpr uint SIMD_SIZE  = 32;
constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
static_assert(PER_THREAD <= 32, "PER_THREAD exceeds taken-bitmask width");
static_assert(PB <= 32, "PB exceeds tk2-bitmask width");

uint tile = threadgroup_position_in_grid.x;
uint tid  = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;

uint ord[PER_THREAD];
uint idx[PER_THREAD];
for (uint t = 0; t < PER_THREAD; ++t) { ord[t] = 0u; idx[t] = 0u; }
uint n = 0;
for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {
    ord[n] = msv_top32_ordinal(float(logits[i]));
    idx[n] = i;
    n++;
}

threadgroup uint sc_ord[NSIMD * TOPK];
threadgroup uint sc_idx[NSIMD * TOPK];

uint taken = 0u;
for (uint r = 0; r < TOPK; ++r) {
    uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    for (uint t = 0; t < PER_THREAD; ++t) {
        if ((taken & (1u << t)) != 0u) { continue; }
        if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
            bo = ord[t]; bi = idx[t]; bs = t;
        }
    }
    uint mo = simd_max(bo);
    uint mi = simd_max((bo == mo) ? bi : 0u);
    if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
        taken |= (1u << bs);
    }
    if (lane == 0) {
        sc_ord[sg * TOPK + r] = mo;
        sc_idx[sg * TOPK + r] = mi;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sg == 0) {
    uint o2[PB];
    uint i2[PB];
    for (uint t = 0; t < PB; ++t) {
        uint p = t * SIMD_SIZE + lane;
        o2[t] = sc_ord[p];
        i2[t] = sc_idx[p];
    }
    uint tk2 = 0u;
    for (uint r = 0; r < TOPK; ++r) {
        uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
        for (uint t = 0; t < PB; ++t) {
            if ((tk2 & (1u << t)) != 0u) { continue; }
            if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
                bo = o2[t]; bi = i2[t]; bs = t;
            }
        }
        uint mo = simd_max(bo);
        uint mi = simd_max((bo == mo) ? bi : 0u);
        if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
            tk2 |= (1u << bs);
        }
        if (lane == 0) {
            cand_ord[tile * TOPK + r] = mo;
            cand_idx[tile * TOPK + r] = mi;
        }
    }
}