// Q4_0 / Q4_1: low nibbles are weights 0..15, high nibbles 16..31; y = q * d + m.
template <typename O>
inline void gg_q4_nibbles(const device uint8_t *qs, float d, float m, thread O &o) {
    for (uint i = 0; i < 4; i++) {
        const uint4 q = GG_U4(qs + 4 * i);
        o.emit(i, d * float4(q & 0xF) + m);
        o.emit(i + 4, d * float4(q >> 4) + m);
    }
}

template <typename O>
inline void deq_q4_0(const device uint8_t *b, uint, thread O &o) {
    const float d = gg_half(b, 0);
    gg_q4_nibbles(b + 2, d, -8.0f * d, o);
}

template <typename O>
inline void deq_q4_1(const device uint8_t *b, uint, thread O &o) {
    gg_q4_nibbles(b + 4, gg_half(b, 0), gg_half(b, 2), o);
}

// Split rows: the 16 nibble bytes as one aligned load.
template <typename O>
inline void gg_q4_nibbles_split(const device uint8_t *qs, float d, float m, thread O &o) {
    const device uchar4 *w = (const device uchar4 *)qs;
    for (uint i = 0; i < 4; i++) {
        const uint4 q = uint4(w[i]);
        o.emit(i, d * float4(q & 0xF) + m);
        o.emit(i + 4, d * float4(q >> 4) + m);
    }
}

template <typename O>
inline void deq_q4_0_split(const device uint8_t *h, const device uint8_t *qs, uint, thread O &o) {
    const float d = gg_half(h, 0);
    gg_q4_nibbles_split(qs, d, -8.0f * d, o);
}

template <typename O>
inline void deq_q4_1_split(const device uint8_t *h, const device uint8_t *qs, uint, thread O &o) {
    gg_q4_nibbles_split(qs, gg_half(h, 0), gg_half(h, 2), o);
}

// Q5_0 / Q5_1: the fifth bit of weight j is bit j of the u32 at `qh`.
template <typename O>
inline void gg_q5_nibbles(const device uint8_t *qh_p, const device uint8_t *qs, float d, float m, thread O &o) {
    const uint qh = uint(qh_p[0]) | (uint(qh_p[1]) << 8) | (uint(qh_p[2]) << 16) | (uint(qh_p[3]) << 24);
    for (uint i = 0; i < 4; i++) {
        const uint4 q = GG_U4(qs + 4 * i);
        const uint4 j = uint4(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
        const uint4 lo = (q & 0xF) | (((uint4(qh) >> j) << 4) & 0x10);
        const uint4 hi = (q >> 4) | ((uint4(qh) >> (j + 12)) & 0x10);
        o.emit(i, d * float4(lo) + m);
        o.emit(i + 4, d * float4(hi) + m);
    }
}

template <typename O>
inline void gg_q5_nibbles_split(const device uint8_t *qh_p, const device uint8_t *qs, float d, float m, thread O &o) {
    const uint qh = uint(qh_p[0]) | (uint(qh_p[1]) << 8) | (uint(qh_p[2]) << 16) | (uint(qh_p[3]) << 24);
    const device uchar4 *w = (const device uchar4 *)qs;
    for (uint i = 0; i < 4; i++) {
        const uint4 q = uint4(w[i]);
        const uint4 j = uint4(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
        const uint4 lo = (q & 0xF) | (((uint4(qh) >> j) << 4) & 0x10);
        const uint4 hi = (q >> 4) | ((uint4(qh) >> (j + 12)) & 0x10);
        o.emit(i, d * float4(lo) + m);
        o.emit(i + 4, d * float4(hi) + m);
    }
}

template <typename O>
inline void deq_q5_0_split(const device uint8_t *h, const device uint8_t *qs, uint, thread O &o) {
    const float d = gg_half(h, 0);
    gg_q5_nibbles_split(h + 2, qs, d, -16.0f * d, o);
}

template <typename O>
inline void deq_q5_1_split(const device uint8_t *h, const device uint8_t *qs, uint, thread O &o) {
    gg_q5_nibbles_split(h + 4, qs, gg_half(h, 0), gg_half(h, 2), o);
}

template <typename O>
inline void deq_q5_0(const device uint8_t *b, uint, thread O &o) {
    const float d = gg_half(b, 0);
    gg_q5_nibbles(b + 2, b + 6, d, -16.0f * d, o);
}

template <typename O>
inline void deq_q5_1(const device uint8_t *b, uint, thread O &o) {
    gg_q5_nibbles(b + 4, b + 8, gg_half(b, 0), gg_half(b, 2), o);
}
