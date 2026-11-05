// IQ1: 8 ternary weights per grid entry (byte j = weights j and j + 4, a nibble
// each, value + 1), shifted by +-1/8 per group. `base` = dl * (-1 +- 1/8).
template <typename O>
inline void gg_iq1_group(uint idx, uint l, float dl, float base, thread O &o) {
    const uint32_t grid = iq1s_grid_gpu[idx];
    const uint4 q = uint4(grid, grid >> 8, grid >> 16, grid >> 24);
    o.emit(2 * l, dl * float4(q & 0xF) + base);
    o.emit(2 * l + 1, dl * float4((q >> 4) & 0xF) + base);
}

template <typename O>
inline void deq_iq1_s(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *qs = b + 2 + 4 * ib;
    const uint qh = uint(b[34 + 2 * ib]) | (uint(b[35 + 2 * ib]) << 8);
    const float dl = gg_half(b, 0) * float(2 * ((qh >> 12) & 7) + 1);
    const float base = dl * ((qh & 0x8000) ? -1.125f : -0.875f);
    for (uint l = 0; l < 4; l++) gg_iq1_group(uint(qs[l]) | (((qh >> (3 * l)) & 7) << 8), l, dl, base, o);
}

template <typename O>
inline void deq_iq1_m(const device uint8_t *b, uint ib, thread O &o) {
    // A block is 56 bytes, so it starts 8-aligned: the unit's grid bytes load as a word, its high bits as a half, the
    // four scale halves as two words.
    const uint qs = *(const device uint *)(b + 4 * ib);
    const uint qh2 = *(const device ushort *)(b + 32 + 2 * ib);
    const uint2 sw = *(const device uint2 *)(b + 48);
    const uint sc[4] = {sw.x & 0xFFFF, sw.x >> 16, sw.y & 0xFFFF, sw.y >> 16};
    // The block's f16 scale is the top nibble of each of the 4 u16 scale words.
    const uint16_t d_bits = uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = float(as_type<half>(d_bits));
    const uint s16 = sc[ib / 2] >> (6 * (ib % 2));
    for (uint l = 0; l < 4; l++) {
        const float dl = d * float(2 * ((s16 >> (3 * (l / 2))) & 7) + 1);
        const uint qh = (qh2 >> (8 * (l / 2))) >> (4 * (l % 2));
        gg_iq1_group(((qs >> (8 * l)) & 0xFF) | ((qh & 7) << 8), l, dl, dl * ((qh & 8) ? -1.125f : -0.875f), o);
    }
}
