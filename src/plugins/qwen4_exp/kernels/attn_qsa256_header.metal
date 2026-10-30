inline int msv_qsa_pos(const device int* blk, int vi, int sel_len, int tail_start, int ratio) {
  const int b = vi / ratio;
  return (vi < sel_len) ? (blk[b] * ratio + (vi - b * ratio)) : (tail_start + (vi - sel_len));
}
