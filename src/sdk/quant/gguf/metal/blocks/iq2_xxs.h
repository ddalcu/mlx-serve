// The IQ grids, unit = ggml's ib32: 8 weights per grid entry (iq2) or 4 (iq3).
template <typename O, typename G = constant uint64_t *>
inline void deq_iq2_xxs(const device uint8_t *b, uint ib, thread O &o, G grid_tab = iq2xxs_grid) {
    const device uint8_t *aux8 = b + 2 + 8 * ib;
    const uint32_t aux1 = gg_u32(aux8, 4);
    const float db = gg_half(b, 0) * (0.5f + float(aux1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; l++) {
        const uint64_t grid = grid_tab[aux8[l]];
        const uint signs = gg_ksigns((aux1 >> (7 * l)) & 127);
        o.emit(2 * l, db * gg_grid4(uint32_t(grid)) * gg_signs(signs, false));
        o.emit(2 * l + 1, db * gg_grid4(uint32_t(grid >> 32)) * gg_signs(signs, true));
    }
}
