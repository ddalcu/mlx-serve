template <typename O>
inline void deq_iq2_xs(const device uint8_t *b, uint ib, thread O &o) {
    const float d = gg_half(b, 0);
    const device uint8_t *qs = b + 2 + 8 * ib;
    const uint8_t scales = b[66 + ib];
    const float db[2] = {
        d * (0.5f + float(scales & 0xF)) * 0.25f,
        d * (0.5f + float(scales >> 4)) * 0.25f,
    };
    for (uint l = 0; l < 4; l++) {
        const uint q = uint(qs[2 * l]) | (uint(qs[2 * l + 1]) << 8);
        const uint64_t grid = iq2xs_grid[q & 511];
        const uint signs = ksigns_iq2xs[q >> 9];
        o.emit(2 * l, db[l / 2] * gg_grid4(uint32_t(grid)) * gg_signs(signs, false));
        o.emit(2 * l + 1, db[l / 2] * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs, true));
    }
}
