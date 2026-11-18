for (int e = int(t); e < NO * R; e += 32 * SG) {
    const int n = e / R, r = e % R, o = ob + n;
    OUT[r * 1024 + o] = bfloat16_t(float(X[r * 1024 + o]) + rb(rsum(n, r) + float(B2[o])));
}
