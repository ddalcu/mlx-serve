#define GG_W w
#define GG_STORE(t, g) simdgroup_store(acc[t][g], out + size_t(m0 + 8 * t) * N + r0 + 8 * g, N)
#define GG_T0 0u
#define GG_T1 4u
