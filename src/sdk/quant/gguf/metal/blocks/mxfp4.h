// MXFP4: an E8M0 scale byte and 16 bytes of FP4 nibbles, read as MLX reads its mxfp4: the e2m1 value (-0 kept) times
// the shared fp4 decode's scale (kernels/common/fp4_scale.metal, MLX's fp8_e8m0: 0xff is NaN). ggml's halved scale
// times its doubled codebook is the same product, except where the halved scale is a subnormal the GPU flushes.
constant float kvalues_e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

template <typename O>
inline void deq_mxfp4_split(const device uint8_t *h, const device uint8_t *qs, uint, thread O &o) {
    const float d = mlxserve_fp4_e8m0(h[0]);
    const device uchar4 *w = (const device uchar4 *)qs;
    for (uint i = 0; i < 4; i++) {
        const uint4 q = uint4(w[i]);
        const uint4 lo = q & 0xF;
        const uint4 hi = q >> 4;
        o.emit(i, d * float4(kvalues_e2m1[lo.x], kvalues_e2m1[lo.y], kvalues_e2m1[lo.z], kvalues_e2m1[lo.w]));
        o.emit(i + 4, d * float4(kvalues_e2m1[hi.x], kvalues_e2m1[hi.y], kvalues_e2m1[hi.z], kvalues_e2m1[hi.w]));
    }
}

template <typename O>
inline void deq_mxfp4(const device uint8_t *b, uint, thread O &o) {
    const float d = mlxserve_fp4_e8m0(b[0]);
    for (uint i = 0; i < 4; i++) {
        const uint4 q = GG_U4(b + 1 + 4 * i);
        const uint4 lo = q & 0xF;
        const uint4 hi = q >> 4;
        o.emit(i, d * float4(kvalues_e2m1[lo.x], kvalues_e2m1[lo.y], kvalues_e2m1[lo.z], kvalues_e2m1[lo.w]));
        o.emit(i + 4, d * float4(kvalues_e2m1[hi.x], kvalues_e2m1[hi.y], kvalues_e2m1[hi.z], kvalues_e2m1[hi.w]));
    }
}
