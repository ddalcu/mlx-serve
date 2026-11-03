{
    const T n = T(float(W[c]) * float(T(hv[p * 4 + i] * inv)));
    const size_t mi = size_t(ROWMOD[r]) * D + c;
    const T sc1 = T(float(SC[mi]) + 1.0f);
    const T mm = T(float(n) * float(sc1));
    HN[size_t(r) * D + c] = T(float(mm) + float(SH[mi]));
}