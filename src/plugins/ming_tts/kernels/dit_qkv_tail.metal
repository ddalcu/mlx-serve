for (int ei = int(t); ei < NO / 2 * R; ei += 32 * SG) {
    const int n = (ei / R) * 2, r = ei % R, col = col0 + n;
    const float inv = rinv(r);
    float e = rb(rsum(n, r) * inv + float(Bv[col]));
    float od = rb(rsum(n + 1, r) * inv + float(Bv[col + 1]));
    if (which < 2) {
        // Interleaved-pair rotary in f32 (MLX: bf16 * f32 table), rounded once.
        const int pi = (col % 64) / 2, l = r % L;
        const float c = float(COS[l * 32 + pi]), s = float(SIN[l * 32 + pi]);
        const float ne = e * c - od * s, no = e * s + od * c;
        e = ne; od = no;
    }
    OUT[(which * R + r) * 1024 + col] = bfloat16_t(e);
    OUT[(which * R + r) * 1024 + col + 1] = bfloat16_t(od);
}
