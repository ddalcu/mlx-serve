// One decoder per ggml block format: deq_<type>(block, u, o) decodes UNIT u of
// a block and hands every group of 4 weights to the emitter: o.emit(i, v) is
// weights [4i, 4i+4) of that unit. A unit is the smallest run of weights that
// shares loaded bytes, so each byte is read once: a whole 32-weight block
// (Q8_0, IQ4_NL), a 32-weight ib32 group of the IQ grids, 64 weights for
// Q4_K/Q5_K (both nibbles of 32 bytes), 128 for Q2_K/Q3_K/Q6_K (all bit
// planes of 32 bytes). Units are what a matvec stripes across GPU threads.
// A matvec multiplies and accumulates on the fly (GgDot, against up to 8
// activation rows at once), a dequant writes
// straight to the output (GgOut). Neither keeps a scratch array: 32 floats of
// thread memory already spill out of registers and cost ~10x.
// Same math as src/quants.zig, which is checked against ggml.
// Loads go through packed (alignment 1) vector types, four bytes at a time,
// since blocks sit at arbitrary byte offsets.

template <typename T>
struct GgOut {
    device T *out;
    void emit(uint i, float4 v) {
        out[4 * i] = T(v.x);
        out[4 * i + 1] = T(v.y);
        out[4 * i + 2] = T(v.z);
        out[4 * i + 3] = T(v.w);
    }
};

// One unit against M activation rows of K values each (`x` = the unit's first
// value in row 0): a verify step or a decode batch pays for the weights once.
// M is a constant so the loop unrolls into M vector registers.
template <typename T, uint M, uint K>
struct GgDot {
    const device T *x;
    float4 acc[M];
    void emit(uint i, float4 v) {
        for (uint m = 0; m < M; m++) {
            const device T *xm = x + m * K + 4 * i;
            acc[m] += v * float4(xm[0], xm[1], xm[2], xm[3]);
        }
    }
};

// One unit against M activation rows already in registers (`x`: the unit's G groups of 4 values, row after row):
// the column-major matvec loads a unit's activations once and reuses them for every weight row it decodes.
template <uint M, uint G>
struct GgDotRegs {
    thread const float4 *x;
    float4 acc[M];
    void emit(uint i, float4 v) {
        for (uint m = 0; m < M; m++) acc[m] += v * x[m * G + i];
    }
};

// Weights of one row into a [column][row] tile of 8 rows, the layout an 8x8
// simdgroup_matrix load reads column block by column block.
struct GgTile {
    threadgroup float *tile;
    uint row;
    void emit(uint i, float4 v) {
        tile[(4 * i) * 8 + row] = v.x;
        tile[(4 * i + 1) * 8 + row] = v.y;
        tile[(4 * i + 2) * 8 + row] = v.z;
        tile[(4 * i + 3) * 8 + row] = v.w;
    }
};

// One 32-weight unit against activations already in registers (`gather_body`).
struct GgDotR {
    float4 x[8];
    float4 acc;
    void emit(uint i, float4 v) { acc += v * x[i]; }
};

// Weights of one row into its row of a threadgroup tile, four at a store: the right operand `gather_nax`'s matrix
// units read transposed.
template <typename T>
struct GgRow {
    threadgroup T *row;
    void emit(uint i, float4 v) {
        *(threadgroup vec<T, 4> *)(row + 4 * i) = vec<T, 4>(v);
    }
};

#define GG_U4(p) uint4(uchar4(*(const device packed_uchar4 *)(p)))
// The 4 bytes of a word as lanes: a split row's payload comes in as aligned 16-byte loads.
inline uint4 gg_bytes(uint w) {
    return uint4(w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, w >> 24);
}
#define GG_I4(p) int4(char4(*(const device packed_char4 *)(p)))

inline float gg_half(const device uint8_t *b, uint off) {
    return float(as_type<half>(uint16_t(uint16_t(b[off]) | (uint16_t(b[off + 1]) << 8))));
}

inline uint32_t gg_u32(const device uint8_t *b, uint off) {
    return uint32_t(b[off]) | (uint32_t(b[off + 1]) << 8) | (uint32_t(b[off + 2]) << 16) | (uint32_t(b[off + 3]) << 24);
}

// +-1 per weight from a sign byte, bit j set = weight j negative. `hi` picks bits 4..7.
inline float4 gg_signs(uint bits, bool hi) {
    const uint4 mask = hi ? uint4(16, 32, 64, 128) : uint4(1, 2, 4, 8);
    return select(float4(1.0f), float4(-1.0f), (uint4(bits) & mask) != 0);
}

// ggml's ksigns_iq2xs[i]: 7 sign bits plus an eighth that makes the count of minus signs even.
inline uint gg_ksigns(uint i) {
    return i | ((popcount(i) & 1) << 7);
}

// The 4 magnitudes packed in one 32-bit grid word, low byte first.
inline float4 gg_grid4(uint32_t word) {
    return float4(uint4(word, word >> 8, word >> 16, word >> 24) & 0xFF);
}
