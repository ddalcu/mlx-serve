template <typename O, typename G = constant uint32_t *>
inline void deq_iq3_xxs(const device uint8_t *b, uint ib, thread O &o, G grid_tab = iq3xxs_grid) {
    const device uint8_t *qs = b + 2 + 8 * ib;
    const uint32_t aux = gg_u32(b, 66 + 4 * ib);
    const float db = gg_half(b, 0) * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; l++) {
        const uint signs = gg_ksigns((aux >> (7 * l)) & 127);
        o.emit(2 * l, db * gg_grid4(grid_tab[qs[2 * l]]) * gg_signs(signs, false));
        o.emit(2 * l + 1, db * gg_grid4(grid_tab[qs[2 * l + 1]]) * gg_signs(signs, true));
    }
}
