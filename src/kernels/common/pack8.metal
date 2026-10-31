// Eight 2-byte T to and from one uint4, low half first. A thread T* cast of a uint4 is
// undefined under the aliasing rule MSL inherits from C++; as_type is MSL's defined reinterpretation.
template <typename T>
inline void msv_unpack8(uint4 w, thread T* e) {
  for (int i = 0; i < 4; ++i) {
    const vec<T, 2> p = as_type<vec<T, 2>>(w[i]);
    e[2 * i] = p.x;
    e[2 * i + 1] = p.y;
  }
}

template <typename T>
inline uint4 msv_pack8(thread const T* e) {
  return uint4(as_type<uint>(vec<T, 2>(e[0], e[1])), as_type<uint>(vec<T, 2>(e[2], e[3])),
               as_type<uint>(vec<T, 2>(e[4], e[5])), as_type<uint>(vec<T, 2>(e[6], e[7])));
}
