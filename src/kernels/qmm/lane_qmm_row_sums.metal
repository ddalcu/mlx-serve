  const int xr = lane >> 1, xh = lane & 1;
  bool xlive[TMR];
  const device bfloat* xp[TMR];
  for (int t = 0; t < TMR; t++) {
    xlive[t] = rb + t * 16 + xr < M;
    xp[t] = X + size_t(xlive[t] ? rb + t * 16 + xr : 0) * K + xh * (GS / 2);
  }
