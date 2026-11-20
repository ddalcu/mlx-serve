for (int e = int(t); e < NO * R; e += 32 * SG) {
    const int n = e / R, r = e % R, o = ob + n;
    const float x = rb(rsum(n, r) * rinv(r) + float(B1[o]));
    const float x3 = rb(x * x * x);
    const float inner = rb(x + rb(rb(0.044715f) * x3));
    const float th = rb(metal::precise::tanh(rb(rb(0.7978845608f) * inner)));
    OUT[r * 4096 + o] = bfloat16_t(rb(0.5f * x) * rb(1.0f + th));
}
