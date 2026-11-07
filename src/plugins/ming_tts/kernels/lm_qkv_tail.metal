for (int e = int(t); e < NO / 2; e += 32 * SG) {
    const int n = 2 * e, c1 = COL(n), c2 = COL(n + 1);
    const float inv = rinv(0);
    float x1 = rb(rsum(n, 0) * inv + float(Bv[c1]));
    float x2 = rb(rsum(n + 1, 0) * inv + float(Bv[c2]));
    if (which < 2) {
        const float th = float(POS) * metal::exp2(-(float(c1 % 64) / 32.0f) * as_type<float>(LOG2BASE));
        const float c = metal::fast::cos(th), sn = metal::fast::sin(th);
        const float r1 = x1 * c - x2 * sn, r2 = x1 * sn + x2 * c;
        x1 = r1; x2 = r2;
    }
    const int off = which == 0 ? 0 : (which == 1 ? 896 : 1024);
    OUT[off + c1] = bfloat16_t(x1);
    OUT[off + c2] = bfloat16_t(x2);
}
