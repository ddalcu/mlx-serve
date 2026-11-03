// mlx-serve port addendum to header_pf_hc (appended at bind; not the lane's text: the lane pin covers
// header_pf_hc.metal alone). MLX binds an input of fewer than 8 elements as `const constant T*`
// (mlx/backend/common/metal_kernel.cpp:102), e.g. q3pf_hc_pre_norm's pre [1, 1, 4] f32 at one row;
// this twin of q3pf_ld makes every pf_hc text compile for either binding. Same load, same float.
template <typename U>
static METAL_FUNC float q3pf_ld(const constant U* p, int64_t off) {
    return static_cast<float>(p[off]);
}
