template <int B>
inline uint32_t mlxserve_qpack(const device uint32_t* w, size_t wbase_words, int pack) {
  if (B == 3) {
    const device uchar* wb = (const device uchar*)w + wbase_words * 4 + (size_t)pack * 3;
    return uint32_t(wb[0]) | (uint32_t(wb[1]) << 8) | (uint32_t(wb[2]) << 16);
  }
  return w[wbase_words + (size_t)pack];
}