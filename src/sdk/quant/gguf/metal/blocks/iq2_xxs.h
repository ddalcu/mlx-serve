// The IQ grids, unit = ggml's ib32: 8 weights per grid entry (iq2) or 4 (iq3).
// A block is 66 bytes, so it starts 2-aligned: the scale and a unit's 8 bytes load as halves.
template <typename O, typename G = constant uint64_t *>
inline void deq_iq2_xxs(const device uint8_t *b, uint ib, thread O &o, G grid_tab = iq2xxs_grid) {
    const device ushort *h = (const device ushort *)b;
    const uint aux0 = uint(h[1 + 4 * ib]) | (uint(h[2 + 4 * ib]) << 16);
    const uint aux1 = uint(h[3 + 4 * ib]) | (uint(h[4 + 4 * ib]) << 16);
    const float db = float(as_type<half>(h[0])) * (0.5f + float(aux1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; l++) {
        const uint64_t grid = grid_tab[(aux0 >> (8 * l)) & 0xFF];
        const uint signs = gg_ksigns((aux1 >> (7 * l)) & 127);
        o.emit(2 * l, db * gg_grid4(uint32_t(grid)) * gg_signs(signs, false));
        o.emit(2 * l + 1, db * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs, true));
    }
}

// The same against the grid staged as two half4 an entry (`stagedGrid`).
template <typename O>
inline void deq_iq2_xxs_h(const device uint8_t *b, uint ib, thread O &o, threadgroup const half4 *gh) {
    const device ushort *h = (const device ushort *)b;
    const uint aux0 = uint(h[1 + 4 * ib]) | (uint(h[2 + 4 * ib]) << 16);
    const uint aux1 = uint(h[3 + 4 * ib]) | (uint(h[4 + 4 * ib]) << 16);
    const float db = float(as_type<half>(h[0])) * (0.5f + float(aux1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; l++) {
        const uint e = 2 * ((aux0 >> (8 * l)) & 0xFF);
        const uint signs = gg_ksigns((aux1 >> (7 * l)) & 127);
        o.emit(2 * l, db * float4(gh[e]) * gg_signs(signs, false));
        o.emit(2 * l + 1, db * float4(gh[e + 1]) * gg_signs(signs, true));
    }
}
