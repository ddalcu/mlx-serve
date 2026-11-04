#define STORE_H(k, v) { static_assert(SL % 2 == 0 && TPG % 2 == 0, "pairs need even k on even lanes"); const float v1 = simd_shuffle_down((v), 1u); if (((k) & 1) == 0) atomic_store_explicit((device atomic_uint*)(h + row * K + (k)), as_type<uint>(vec<T, 2>(T(v), T(v1))), memory_order_relaxed); }
#define TAIL4(k) msv_hc_load4<T>(h + row * K, (k))
