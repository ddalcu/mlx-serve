  const int R = XS_shape[0];
  const int T8 = (R + 7) / 8;
  const device float2* XF2 = (const device float2*)XF;