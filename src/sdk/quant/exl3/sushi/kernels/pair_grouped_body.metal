for (uint proj = 0u; proj < 2u; proj++) {
  const device half *suh = (proj == 0u) ? suhg : suhu;
  const device ushort *trellis = (proj == 0u) ? tg : tu;
  device float *y = (proj == 0u) ? yg : yu;
