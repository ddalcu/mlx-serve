// A block is 74 bytes, so it starts 2-aligned: the scale and the unit's four grid words load as halves.
template <typename O, typename G = constant uint64_t *>
inline void deq_iq2_xs(const device uint8_t *b, uint ib, thread O &o, G grid_tab = iq2xs_grid) {
    const device ushort *h = (const device ushort *)b;
    const float d = float(as_type<half>(h[0]));
    const uint8_t scales = b[66 + ib];
    const float db[2] = {
        d * (0.5f + float(scales & 0xF)) * 0.25f,
        d * (0.5f + float(scales >> 4)) * 0.25f,
    };
    for (uint l = 0; l < 4; l++) {
        const uint q = h[1 + 4 * ib + l];
        const uint64_t grid = grid_tab[q & 511];
        const uint signs = gg_ksigns(q >> 9);
        o.emit(2 * l, db[l / 2] * gg_grid4(uint32_t(grid)) * gg_signs(signs, false));
        o.emit(2 * l + 1, db[l / 2] * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs, true));
    }
}

// The same against the grid staged as two half4 an entry (`stagedGrid`).
template <typename O>
inline void deq_iq2_xs_h(const device uint8_t *b, uint ib, thread O &o, threadgroup const half4 *gh) {
    const device ushort *h = (const device ushort *)b;
    const float d = float(as_type<half>(h[0]));
    const uint8_t scales = b[66 + ib];
    const float db[2] = {
        d * (0.5f + float(scales & 0xF)) * 0.25f,
        d * (0.5f + float(scales >> 4)) * 0.25f,
    };
    for (uint l = 0; l < 4; l++) {
        const uint q = h[1 + 4 * ib + l];
        const uint e = 2 * (q & 511);
        const uint signs = gg_ksigns(q >> 9);
        o.emit(2 * l, db[l / 2] * float4(gh[e]) * gg_signs(signs, false));
        o.emit(2 * l + 1, db[l / 2] * float4(gh[e + 1]) * gg_signs(signs, true));
    }
}
