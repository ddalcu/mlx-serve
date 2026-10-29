    float xs0[RT], xs1[RT];
    PRAGMA_UNROLL
    for (int rt = 0; rt < RT; rt++) { xs0[rt] = XS[size_t(xr0[rt]) * G + g]; xs1[rt] = XS[size_t(xr1[rt]) * G + g]; }