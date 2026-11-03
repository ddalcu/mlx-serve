constexpr uint TG_SIZE    = 256;
constexpr uint PER_THREAD = 8;
constexpr uint TOPK       = 32;
constexpr uint SIMD_SIZE  = 32;
constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;

uint tid  = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;

uint ord[PER_THREAD];
uint idx[PER_THREAD];
for (uint t = 0; t < PER_THREAD; ++t) {
    uint p = t * TG_SIZE + tid;
    ord[t] = cand_ord[p];
    idx[t] = cand_idx[p];
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
        if (lane == 0) { token_ids[TOPK - 1u - r] = mi; }
    }
}