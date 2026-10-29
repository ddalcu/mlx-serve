    uint4 xa[RT], xb[RT];
    float xs0[RT], xs1[RT];
    PRAGMA_UNROLL
    for (int rt = 0; rt < RT; rt++) {
      xa[rt] = LOAD8(xr0[rt], 8 * g + fm);
      xb[rt] = LOAD8(xr1[rt], 8 * g + fm);
      float v = sum8(xa[rt], one), u = sum8(xb[rt], one);
      v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
      v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
      v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
      xs0[rt] = v; xs1[rt] = u;
    }