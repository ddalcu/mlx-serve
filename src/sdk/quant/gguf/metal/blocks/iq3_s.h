template <typename O>
inline void deq_iq3_s(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *qs = b + 2 + 8 * ib;
    const uint qh = b[66 + ib];
    const device uint8_t *signs = b + 74 + 4 * ib;
    const uint8_t scales = b[106 + ib / 2];
    const uint nib = (ib % 2 == 0) ? (scales & 0xF) : (scales >> 4);
    const float db = gg_half(b, 0) * float(1 + 2 * nib);
    for (uint l = 0; l < 4; l++) {
        o.emit(2 * l, db * gg_grid4(iq3s_grid[uint(qs[2 * l]) | ((qh << (8 - 2 * l)) & 256)]) * gg_signs(signs[l], false));
        o.emit(2 * l + 1, db * gg_grid4(iq3s_grid[uint(qs[2 * l + 1]) | ((qh << (7 - 2 * l)) & 256)]) * gg_signs(signs[l], true));
    }
}
