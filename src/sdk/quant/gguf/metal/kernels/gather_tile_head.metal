#define GG_W we
#define GG_STORE(t, g) simdgroup_store(acc[t][g], stage + (8 * t) * TILE_ROWS + 8 * g, TILE_ROWS)
#define GG_T0 (lo / 8)
#define GG_T1 (hi / 8 + 1)
const uint lid = thread_index_in_threadgroup;
const uint r0 = threadgroup_position_in_grid.x * TILE_ROWS;
const uint m0 = threadgroup_position_in_grid.y * TILE_TOKENS;
const uint m_end = min(m0 + TILE_TOKENS, uint(ids_shape[0]));
for (uint m = m0; m < m_end; m++) {
    const uint e = ids[m];
    uint lo = TILE_TOKENS, hi = 0;
    for (uint p = m0; p < m_end; p++) if (ids[p] == e) { lo = min(lo, p - m0); hi = max(hi, p - m0); }
    if (lo < m - m0) continue;
    const device uint8_t *we = w + size_t(e) * N * ROWB;
