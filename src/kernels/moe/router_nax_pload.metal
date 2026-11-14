// A partial read device-coherently by the group that arrives last: the f32 bits an atomic store left.
static inline float pload(device float* p, size_t i) {
  return as_type<float>(atomic_load_explicit(reinterpret_cast<device atomic_uint*>(p + i), memory_order_relaxed));
}
