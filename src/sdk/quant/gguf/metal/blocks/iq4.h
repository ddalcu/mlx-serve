// 32 weights from 16 bytes of codebook nibbles, shared by IQ4_NL and IQ4_XS.
template <typename O>
inline void gg_iq4_nibbles(const device uint8_t *qs, float d, thread O &o) {
    for (uint i = 0; i < 4; i++) {
        const uint4 q = GG_U4(qs + 4 * i);
        const uint4 lo = q & 0xF;
        const uint4 hi = q >> 4;
        o.emit(i, d * float4(kvalues_iq4nl[lo.x], kvalues_iq4nl[lo.y], kvalues_iq4nl[lo.z], kvalues_iq4nl[lo.w]));
        o.emit(i + 4, d * float4(kvalues_iq4nl[hi.x], kvalues_iq4nl[hi.y], kvalues_iq4nl[hi.z], kvalues_iq4nl[hi.w]));
    }
}

template <typename O>
inline void deq_iq4_nl(const device uint8_t *b, uint, thread O &o) { gg_iq4_nibbles(b + 2, gg_half(b, 0), o); }

template <typename O>
inline void deq_iq4_nl_split(const device uint8_t *h, const device uint8_t *qs, uint, thread O &o) {
    const float d = gg_half(h, 0);
    const device uchar4 *w = (const device uchar4 *)qs;
    for (uint i = 0; i < 4; i++) {
        const uint4 q = uint4(w[i]);
        const uint4 lo = q & 0xF;
        const uint4 hi = q >> 4;
        o.emit(i, d * float4(kvalues_iq4nl[lo.x], kvalues_iq4nl[lo.y], kvalues_iq4nl[lo.z], kvalues_iq4nl[lo.w]));
        o.emit(i + 4, d * float4(kvalues_iq4nl[hi.x], kvalues_iq4nl[hi.y], kvalues_iq4nl[hi.z], kvalues_iq4nl[hi.w]));
    }
}

// 6-bit scale of unit ib: low nibble in scales_l (b + 4), top 2 bits in scales_h (b + 2).
template <typename O>
inline void deq_iq4_xs(const device uint8_t *b, uint ib, thread O &o) {
    const uint scales_h = uint(b[2]) | (uint(b[3]) << 8);
    const uint ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    gg_iq4_nibbles(b + 8 + 16 * ib, gg_half(b, 0) * float(int(ls) - 32), o);
}

// Q2_K / Q3_K, unit n: weight (j, l) = bits 2j of q[32n + l], l < 32, scale group 8n + 2j + l/16.
