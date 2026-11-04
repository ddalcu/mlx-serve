template <int B> struct mlxserve_pack { typedef uint32_t type; };
template <> struct mlxserve_pack<5> { typedef ulong type; };
template <int B> using mlxserve_pack_t = typename mlxserve_pack<B>::type;

// Values per packed unit: a word at 2/4/8-bit; mx.quantize's 8 values in 3 bytes at 3-bit, 8 in 5 at 5-bit, 4 in 3 at 6-bit.
template <int B> constexpr int mlxserve_vpw() { return (B == 3 || B == 5) ? 8 : (B == 6 ? 4 : 32 / B); }

template <int B>
inline mlxserve_pack_t<B> mlxserve_qpack(const device uint32_t* w, size_t wbase_words, int pack) {
  if constexpr (B == 3 || B == 5 || B == 6) {
    constexpr int NB = B == 5 ? 5 : 3;
    const device uchar* wb = (const device uchar*)w + wbase_words * 4 + (size_t)pack * NB;
    mlxserve_pack_t<B> v = mlxserve_pack_t<B>(wb[0]) | (mlxserve_pack_t<B>(wb[1]) << 8) | (mlxserve_pack_t<B>(wb[2]) << 16);
    if constexpr (NB == 5) v |= (mlxserve_pack_t<B>(wb[3]) << 24) | (mlxserve_pack_t<B>(wb[4]) << 32);
    return v;
  }
  return w[wbase_words + (size_t)pack];
}