template <typename O>
inline void deq_iq2_s(const device uint8_t *b, uint ib, thread O &o) {
    const float d = gg_half(b, 0);
    const device uint8_t *qs = b + 2 + 4 * ib;
    const device uint8_t *signs = b + 34 + 4 * ib;
    const uint qh = b[66 + ib];
    const uint8_t scales = b[74 + ib];
    const float db[2] = {
        d * (0.5f + float(scales & 0xF)) * 0.25f,
        d * (0.5f + float(scales >> 4)) * 0.25f,
    };
    for (uint l = 0; l < 4; l++) {
        const uint64_t grid = iq2s_grid[uint(qs[l]) | ((qh << (8 - 2 * l)) & 0x300)];
        o.emit(2 * l, db[l / 2] * gg_grid4(uint32_t(grid)) * gg_signs(signs[l], false));
        o.emit(2 * l + 1, db[l / 2] * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs[l], true));
    }
}
