for (int n = int(t); n < NO / 2; n += 32 * SG) {
    const float inv = rinv(0);
    const float g = rb(rsum(n, 0) * inv), u = rb(rsum(n + NO / 2, 0) * inv);
    const float sg_ = rb(1.0f / (1.0f + metal::exp(-g)));
    OUT[c0 + n] = bfloat16_t(rb(g * sg_) * u);
}
