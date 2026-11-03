template <typename O>
inline void deq_iq3_xxs(const device uint8_t *b, uint ib, thread O &o) {
    const device uint8_t *qs = b + 2 + 8 * ib;
    const uint32_t aux = gg_u32(b, 66 + 4 * ib);
    const float db = gg_half(b, 0) * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 4; l++) {
        const uint signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        o.emit(2 * l, db * gg_grid4(iq3xxs_grid[qs[2 * l]]) * gg_signs(signs, false));
        o.emit(2 * l + 1, db * gg_grid4(iq3xxs_grid[qs[2 * l + 1]]) * gg_signs(signs, true));
    }
}
