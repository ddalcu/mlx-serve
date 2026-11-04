template <typename O>
inline void deq_q2_k(const device uint8_t *b, uint n, thread O &o) {
    const float d = gg_half(b, 80);
    const float mn = gg_half(b, 82);
    for (uint h = 0; h < 2; h++) {
        for (uint c = 0; c < 4; c++) {
            const uint4 q = GG_U4(b + 16 + 32 * n + 16 * h + 4 * c);
            for (uint j = 0; j < 4; j++) {
                const uint8_t s8 = b[8 * n + 2 * j + h];
                o.emit(8 * j + 4 * h + c, d * float(s8 & 0xF) * float4((q >> (2 * j)) & 3) - mn * float(s8 >> 4));
            }
        }
    }
}

template <typename O>
inline void deq_q3_k(const device uint8_t *b, uint n, thread O &o) {
    const float d_all = gg_half(b, 108);
    const device uint8_t *scales = b + 96;
    for (uint h = 0; h < 2; h++) {
        float dl[4];
        for (uint j = 0; j < 4; j++) {
            // 6-bit scale `is`: low nibbles live in scales[0..8), the high 2 bits in scales[8..12).
            const uint is = 8 * n + 2 * j + h;
            const uint low = is < 8 ? (scales[is] & 0xF) : (scales[is - 8] >> 4);
            const uint high = (scales[8 + is % 4] >> (2 * (is / 4))) & 3;
            dl[j] = d_all * float(int(low | (high << 4)) - 32);
        }
        for (uint c = 0; c < 4; c++) {
            const uint4 q = GG_U4(b + 32 + 32 * n + 16 * h + 4 * c);
            const uint4 hm = GG_U4(b + 16 * h + 4 * c);
            for (uint j = 0; j < 4; j++) {
                const int4 lo = int4((q >> (2 * j)) & 3);
                // hmask bit set = no -4 offset.
                const int4 hi = int4(((hm >> (4 * n + j)) & 1) ^ 1) * 4;
                o.emit(8 * j + 4 * h + c, dl[j] * float4(lo - hi));
            }
        }
    }
}

inline void gg_scale_min_k4(uint j, const device uint8_t *q, thread float &sc, thread float &mn) {
    if (j < 4) {
        sc = float(q[j] & 63);
        mn = float(q[j + 4] & 63);
    } else {
        sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Q4_K and Q5_K share the layout, Q5_K adds a high-bit plane (qh, bit 2j+h of
// qh[l]). Unit j: weight (h, l) = nibble h of qs[32j + l].
template <typename O>
inline void deq_q45_k(const device uint8_t *b, uint j, thread O &o, bool five) {
    const float d = gg_half(b, 0);
    const float mn = gg_half(b, 2);
    const device uint8_t *qs = b + (five ? 48 : 16) + 32 * j;
    float s0, m0, s1, m1;
    gg_scale_min_k4(2 * j, b + 4, s0, m0);
    gg_scale_min_k4(2 * j + 1, b + 4, s1, m1);
    for (uint c = 0; c < 8; c++) {
        const uint4 q = GG_U4(qs + 4 * c);
        uint4 lo = q & 0xF;
        uint4 hi = q >> 4;
        if (five) {
            const uint4 qh = GG_U4(b + 16 + 4 * c) >> (2 * j);
            lo += (qh & 1) * 16;
            hi += ((qh >> 1) & 1) * 16;
        }
        o.emit(c, d * s0 * float4(lo) - mn * m0);
        o.emit(8 + c, d * s1 * float4(hi) - mn * m1);
    }
}

template <typename O>
inline void deq_q4_k(const device uint8_t *b, uint j, thread O &o) { deq_q45_k(b, j, o, false); }
template <typename O>
inline void deq_q5_k(const device uint8_t *b, uint j, thread O &o) { deq_q45_k(b, j, o, true); }

// Split rows: ql / qh as aligned 16-byte loads, scales + d in the head.
template <typename O>
inline void deq_q6_k_split(const device uint8_t *h, const device uint8_t *qs, uint n, thread O &o) {
    const float d = gg_half(h, 16);
    const device uchar4 *ql = (const device uchar4 *)(qs + 64 * n);
    const device uchar4 *qh = (const device uchar4 *)(qs + 128 + 32 * n);
    const device uint8_t *sc = h + 8 * n;
    for (uint p = 0; p < 2; p++) {
        for (uint g = 0; g < 2; g++) {
            const float dlo = d * float(int8_t(sc[g + 2 * p]));
            const float dhi = d * float(int8_t(sc[g + 2 * p + 4]));
            for (uint k = 0; k < 4; k++) {
                const uint c = 4 * g + k;
                const uint4 q = uint4(ql[8 * p + c]);
                const uint4 hb = uint4(qh[c]) >> (2 * p);
                o.emit(8 * p + c, dlo * float4((q & 0xF) | ((hb & 3) << 4)) - 32.0f * dlo);
                o.emit(8 * p + 16 + c, dhi * float4((q >> 4) | (((hb >> 4) & 3) << 4)) - 32.0f * dhi);
            }
        }
    }
}

// Unit n, weight (k, l): low nibble from ql[64n + l + 32(k&1)] (high nibble for
// k >= 2), top 2 bits from bits 2k of qh[32n + l], scale sc[8n + 2k + l/16].
template <typename O>
inline void deq_q6_k(const device uint8_t *b, uint n, thread O &o) {
    const float d = gg_half(b, 208);
    const device uint8_t *ql = b + 64 * n;
    const device uint8_t *qh = b + 128 + 32 * n;
    const device uint8_t *sc = b + 192 + 8 * n;
    // Two passes (k = 0,2 off ql[l], then k = 1,3 off ql[l + 32]) keep the live state small.
    for (uint p = 0; p < 2; p++) {
        for (uint g = 0; g < 2; g++) {
            const float dlo = d * float(int8_t(sc[g + 2 * p]));
            const float dhi = d * float(int8_t(sc[g + 2 * p + 4]));
            for (uint c = 4 * g; c < 4 * g + 4; c++) {
                const uint4 q = GG_U4(ql + 32 * p + 4 * c);
                const uint4 h = GG_U4(qh + 4 * c) >> (2 * p);
                o.emit(8 * p + c, dlo * float4((q & 0xF) | ((h & 3) << 4)) - 32.0f * dlo);
                o.emit(8 * p + 16 + c, dhi * float4((q >> 4) | (((h >> 4) & 3) << 4)) - 32.0f * dhi);
            }
        }
    }
}

// 32-weight slices, for the kernels built on 32-weight units (the NAX and decode gathers, the fused decode FFN):
// deq_<type>_s(b, u, o) decodes weights [32u, 32u + 32) of the block, the slice the wide decoder above emits as
// groups [8s, 8s + 8) of its unit. The bytes a wide unit shares are read once per slice.
template <typename O>
inline void deq_q2_k_s(const device uint8_t *b, uint u, thread O &o) {
    const uint n = u / 4, j = u % 4;
    const float d = gg_half(b, 80);
    const float mn = gg_half(b, 82);
    for (uint h = 0; h < 2; h++) {
        const uint8_t s8 = b[8 * n + 2 * j + h];
        const float dl = d * float(s8 & 0xF), ml = mn * float(s8 >> 4);
        for (uint c = 0; c < 4; c++) o.emit(4 * h + c, dl * float4((GG_U4(b + 16 + 32 * n + 16 * h + 4 * c) >> (2 * j)) & 3) - ml);
    }
}

template <typename O>
inline void deq_q3_k_s(const device uint8_t *b, uint u, thread O &o) {
    const uint n = u / 4, j = u % 4;
    const float d_all = gg_half(b, 108);
    const device uint8_t *scales = b + 96;
    for (uint h = 0; h < 2; h++) {
        const uint is = 8 * n + 2 * j + h;
        const uint low = is < 8 ? (scales[is] & 0xF) : (scales[is - 8] >> 4);
        const uint high = (scales[8 + is % 4] >> (2 * (is / 4))) & 3;
        const float dl = d_all * float(int(low | (high << 4)) - 32);
        for (uint c = 0; c < 4; c++) {
            const int4 lo = int4((GG_U4(b + 32 + 32 * n + 16 * h + 4 * c) >> (2 * j)) & 3);
            const int4 hi = int4(((GG_U4(b + 16 * h + 4 * c) >> (4 * n + j)) & 1) ^ 1) * 4;
            o.emit(4 * h + c, dl * float4(lo - hi));
        }
    }
}

// Slice u: unit u / 2, its low (even u) or high nibbles.
template <typename O>
inline void deq_q45_k_s(const device uint8_t *b, uint u, thread O &o, bool five) {
    const uint j = u / 2, hi = u % 2;
    const float d = gg_half(b, 0);
    const float mn = gg_half(b, 2);
    const device uint8_t *qs = b + (five ? 48 : 16) + 32 * j;
    float sc, m;
    gg_scale_min_k4(u, b + 4, sc, m);
    for (uint c = 0; c < 8; c++) {
        uint4 v = (GG_U4(qs + 4 * c) >> (4 * hi)) & 0xF;
        if (five) v += ((GG_U4(b + 16 + 4 * c) >> u) & 1) * 16;
        o.emit(c, d * sc * float4(v) - mn * m);
    }
}

template <typename O>
inline void deq_q4_k_s(const device uint8_t *b, uint u, thread O &o) { deq_q45_k_s(b, u, o, false); }
template <typename O>
inline void deq_q5_k_s(const device uint8_t *b, uint u, thread O &o) { deq_q45_k_s(b, u, o, true); }

// Slice u: unit n = u / 4, pass p = u & 1, nibble u & 2 (weights 32p and 64 + 32p of the unit).
template <typename O>
inline void deq_q6_k_s(const device uint8_t *b, uint u, thread O &o) {
    const uint n = u / 4, p = u & 1, nib = (u >> 1) & 1;
    const float d = gg_half(b, 208);
    const device uint8_t *ql = b + 64 * n + 32 * p;
    const device uint8_t *qh = b + 128 + 32 * n;
    for (uint g = 0; g < 2; g++) {
        const float dl = d * float(int8_t(b[192 + 8 * n + g + 2 * p + 4 * nib]));
        for (uint c = 4 * g; c < 4 * g + 4; c++) {
            const uint4 q = GG_U4(ql + 4 * c) >> (4 * nib);
            const uint4 h = GG_U4(qh + 4 * c) >> (2 * p + 4 * nib);
            o.emit(c, dl * float4((q & 0xF) | ((h & 3) << 4)) - 32.0f * dl);
        }
    }
}

template <typename O>
inline void deq_q6_k_s_split(const device uint8_t *h, const device uint8_t *qs, uint u, thread O &o) {
    const uint n = u / 4, p = u & 1, nib = (u >> 1) & 1;
    const float d = gg_half(h, 16);
    const device uchar4 *ql = (const device uchar4 *)(qs + 64 * n);
    const device uchar4 *qh = (const device uchar4 *)(qs + 128 + 32 * n);
    for (uint g = 0; g < 2; g++) {
        const float dl = d * float(int8_t(h[8 * n + g + 2 * p + 4 * nib]));
        for (uint k = 0; k < 4; k++) {
            const uint c = 4 * g + k;
            const uint4 q = uint4(ql[8 * p + c]) >> (4 * nib);
            const uint4 hb = uint4(qh[c]) >> (2 * p + 4 * nib);
            o.emit(c, dl * float4((q & 0xF) | ((hb & 3) << 4)) - 32.0f * dl);
        }
    }
}
