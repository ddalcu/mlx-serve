
// ---------------------------------------------------------------- DIG2 epilogue (q3_prefill_dig2_candidate.py)
// MLX 0.32.2 functors, bodies verbatim (the float branch): what vsn_Maximum / vsn_Minimum and the compiled
// silu * up kernel run in the stock chain (all compiled in MathMode::Safe)
struct Dig2Sigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct Dig2Maximum {
  template <typename T>
  T operator()(T x, T y) thread {
    if (metal::isnan(x)) {
      return x;
    }
    return x > y ? x : y;
  }
};
struct Dig2Minimum {
  template <typename T>
  T operator()(T x, T y) thread {
    if (metal::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }
};

// the stock _clamped_swiglu on one element, in the stock op order: mx.clip(up) = minimum(maximum(up, -L), L),
// mx.minimum(gate, L), then mlx_lm swiglu = nn.silu(gate) * up = (gate * sigmoid(gate)) * up
METAL_FUNC float dig2_clamped_swiglu(const float g, const float u) {
    const float LIMIT = as_type<float>(1092616192u);
    const float NLIMIT = as_type<float>(3240099840u);
    const float uc = Dig2Minimum()(Dig2Maximum()(u, NLIMIT), LIMIT);
    const float gc = Dig2Minimum()(g, LIMIT);
    return (gc * Dig2Sigmoid()(gc)) * uc;
}

// rot_widen2 for one (row, 128-block): t128(z) * rout, lane l holds columns c0 + 32 q (the stock layout)
METAL_FUNC void dig2_widen(const device float* act, const size_t src, const device half* rout, const size_t rb,
                           const uint c0, const uint l, thread float* out) {
#pragma METAL fp contract(off)
    const float SCALE = as_type<float>(1035273459u);
    float w[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        w[q] = float(act[src + size_t(32u * q)]);
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 32u; h <<= 1u) {
        bool upper = (l & h) != 0u;
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) {
            float o = simd_shuffle_xor(w[q], (ushort) h);
            w[q] = upper ? (o - w[q]) : (w[q] + o);
        }
    }
    float a0 = w[0] + w[1];
    float a1 = w[0] - w[1];
    float a2 = w[2] + w[3];
    float a3 = w[2] - w[3];
    w[0] = a0 + a2;
    w[2] = a0 - a2;
    w[1] = a1 + a3;
    w[3] = a1 - a3;
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        float ro = float(rout[rb + size_t(c0 + 32u * q)]);
        out[q] = (w[q] * SCALE) * ro;
    }
}

// rot_round for one (row, 128-block): f16(t128(h) ), the down GEMM's input
METAL_FUNC void dig2_round(thread const float* h, const device half* rin, const size_t rb, const uint c0,
                           device half* out, const size_t dst, const uint l) {
#pragma METAL fp contract(off)
    const float SCALE = as_type<float>(1035273459u);
    float w[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        w[q] = h[q] * float(rin[rb + size_t(c0 + 32u * q)]);
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 32u; h <<= 1u) {
        bool upper = (l & h) != 0u;
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) {
            float o = simd_shuffle_xor(w[q], (ushort) h);
            w[q] = upper ? (o - w[q]) : (w[q] + o);
        }
    }
    float a0 = w[0] + w[1];
    float a1 = w[0] - w[1];
    float a2 = w[2] + w[3];
    float a3 = w[2] - w[3];
    w[0] = a0 + a2;
    w[2] = a0 - a2;
    w[1] = a1 + a3;
    w[3] = a1 - a3;
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        out[dst + size_t(32u * q)] = static_cast<half>(w[q] * SCALE);
    }
}

// one (row, 128-column block), one simdgroup: rot_widen2 (g, u) -> clamped swiglu -> rot_round, the stock order
METAL_FUNC void dig2_swiglu_one(const device float* zg, const device float* zu, device half* hd,
                                const device half* rout_g, const device half* rout_u, const device half* rin_d,
                                const int slot, const int row, const int cb, const uint l) {
    const uint c0 = uint(cb) * 128u + l;
    const size_t rb = size_t(slot) * size_t(2304);
    const size_t src = size_t(row) * size_t(2304) + size_t(c0);
    float g[4];
    float u[4];
    float h[4];
    dig2_widen(zg, src, rout_g, rb, c0, l, g);
    dig2_widen(zu, src, rout_u, rb, c0, l, u);
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        h[q] = dig2_clamped_swiglu(g[q], u[q]);
    }
    dig2_round(h, rin_d, rb, c0, hd, src, l);
}

// epilogue: the threadgroup's rows of one 128-column block; simdgroup sg takes rows sg, sg + 4, ... (uniform)
METAL_FUNC void dig2_swiglu_rows(const device float* zg, const device float* zu, device half* hd,
                                 const device half* rout_g, const device half* rout_u, const device half* rin_d,
                                 const int slot, const int grow0, const int nrows, const int cb,
                                 const uint sg, const uint l) {
    for (int rr = int(sg); rr < nrows; rr += 4) {
        dig2_swiglu_one(zg, zu, hd, rout_g, rout_u, rin_d, slot, grow0 + rr, cb, l);
    }
}
