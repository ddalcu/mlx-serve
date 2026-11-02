// The EXL3 prefill tiles on GPUs without NAX. Device loads are A, the threadgroup
// loads are transposed B, and all stores are the row-major accumulator.
template <typename T, int R, int C>
struct Dsv41SimdTile {
    metal::simdgroup_matrix<T, 8, 8> values[R * 2][C * 2];

    static short2 coord() {
        const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
        const short q = lane / 4;
        return short2((q & 2) * 2 + (lane % 2) * 2,
                      (q & 4) + ((lane / 2) % 4));
    }

    void clear() thread {
        for (int i = 0; i < R * 2; ++i)
            for (int j = 0; j < C * 2; ++j)
                for (int e = 0; e < 2; ++e) values[i][j].thread_elements()[e] = T(0);
    }

    template <typename U>
    void load(const device U* p, int ld) thread {
        for (int i = 0; i < R * 2; ++i)
            for (int j = 0; j < C * 2; ++j)
                metal::simdgroup_load(values[i][j], p + i * 8 * ld + j * 8, ld);
    }

    template <typename U>
    void load_safe(const device U* p, int ld, short2 size) thread {
        const short2 c = coord();
        for (int i = 0; i < R * 2; ++i)
            for (int j = 0; j < C * 2; ++j)
                for (int e = 0; e < 2; ++e) {
                    const int row = i * 8 + c.y;
                    const int col = j * 8 + c.x + e;
                    values[i][j].thread_elements()[e] =
                        row < size.y && col < size.x ? T(p[row * ld + col]) : T(0);
                }
    }

    template <typename U, int LD, int STRIDE>
    void load(const threadgroup U* p) thread {
        static_assert(R == C, "DIG B tiles are square");
        static_assert(STRIDE == 1, "SIMD matrix loads require contiguous columns");
        for (int i = 0; i < R * 2; ++i)
            for (int j = 0; j < C * 2; ++j)
                metal::simdgroup_load(values[i][j], p + j * 8 * LD + i * 8, LD,
                                     metal::ulong2(0), true);
    }

    template <typename U>
    void store(device U* p, int ld) const thread {
        for (int i = 0; i < R * 2; ++i)
            for (int j = 0; j < C * 2; ++j)
                metal::simdgroup_store(values[i][j], p + i * 8 * ld + j * 8, ld);
    }

    template <typename U>
    void store_safe(device U* p, int ld, short2 size) const thread {
        const short2 c = coord();
        for (int i = 0; i < R * 2; ++i)
            for (int j = 0; j < C * 2; ++j)
                for (int e = 0; e < 2; ++e) {
                    const int row = i * 8 + c.y;
                    const int col = j * 8 + c.x + e;
                    if (row < size.y && col < size.x)
                        p[row * ld + col] = U(values[i][j].thread_elements()[e]);
                }
    }

    template <typename U, int LD, int STRIDE>
    void store(threadgroup U* p) const thread {
        const short2 c = coord();
        for (int i = 0; i < R * 2; ++i)
            for (int j = 0; j < C * 2; ++j)
                for (int e = 0; e < 2; ++e)
                    p[(i * 8 + c.y) * LD + (j * 8 + c.x + e) * STRIDE] =
                        U(values[i][j].thread_elements()[e]);
    }
};

template <int M, int N, int K>
void dsv41_simd_matmad(thread Dsv41SimdTile<float, M, N>& d,
                      thread Dsv41SimdTile<half, M, K>& a,
                      metal::bool_constant<false>,
                      thread Dsv41SimdTile<half, N, K>& b,
                      metal::bool_constant<true>) {
    static_assert(N == K, "DIG B tiles are square");
    for (int m = 0; m < M * 2; ++m)
        for (int n = 0; n < N * 2; ++n)
            for (int k = 0; k < K * 2; ++k)
                metal::simdgroup_multiply_accumulate(d.values[m][n], a.values[m][k],
                                                     b.values[k][n], d.values[m][n]);
}
#define NAXTile Dsv41SimdTile
#define tile_matmad_nax dsv41_simd_matmad
