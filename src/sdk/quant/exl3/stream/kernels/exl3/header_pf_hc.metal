
// q3_prefill_attn HC: one correctly rounded product per stock Multiply / Square (fma with -0.0: no contraction)
template <typename U>
static METAL_FUNC float q3pf_ld(const device U* p, int64_t off) {
    return static_cast<float>(p[off]);
}
