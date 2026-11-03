
// q3_attnfuse: MLX 0.32.2 functors, verbatim semantics (reduction/ops.h Max, binary_ops.h Maximum)
static METAL_FUNC float q3af_max_op(float a, float b) {          // Max<float>::operator()(input, total)
    if (metal::isnan(a) || metal::isnan(b)) {
        return static_cast<float>(NAN);
    } else {
        return a > b ? a : b;
    }
}
static METAL_FUNC float q3af_simd_max(float val) {               // Max<float>::simd_reduce_impl
    if (simd_any(val != val)) {
        return static_cast<float>(NAN);
    }
    return simd_max(val);
}
static METAL_FUNC float q3af_maximum(float x, float y) {         // Maximum::operator()(x, y), float
    if (metal::isnan(x)) {
        return x;
    }
    return x > y ? x : y;
}
static METAL_FUNC float q3af_val(const device float* s, int64_t ss, const device bool* v, int64_t vs,
                                 int j, float scale) {
    // vs_Multiply (x * scale, one correctly rounded product; fma with -0.0 keeps +-0 and cannot be
    // contracted with a later add) then Select(valid, p, -inf)
    const float p = metal::fma(s[j * ss], scale, -0.0f);
    return v[j * vs] ? p : -metal::numeric_limits<float>::infinity();
}
