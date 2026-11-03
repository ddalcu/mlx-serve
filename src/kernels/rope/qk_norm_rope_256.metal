constexpr uint HD = 256;
constexpr uint half_rd = uint(RD) / 2;
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint simd_lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint q_rows = uint(HQ) * uint(S);
uint k_rows = uint(HK) * uint(S);
bool is_q = row < q_rows;
bool is_k = !is_q && row < q_rows + k_rows;
bool is_v = !is_q && !is_k;   // v rows (HV > 0): norm only, no rotation
uint local_row = is_q ? row : (is_k ? row - q_rows : row - q_rows - k_rows);
uint head = local_row / uint(S);
uint seq = local_row % uint(S);
// Inputs are row-contiguous [1, S, H, 256]; outputs head-major [1, H, S, 256].
const device T* src = is_q ? (q + (seq * uint(HQ) + head) * HD)
                    : (is_k ? (k + (seq * uint(HK) + head) * HD)
                            : (v + (seq * uint(HV) + head) * HD));
const device T* w = is_q ? qw : (is_k ? kw : vw);
device T* dst = is_q ? (oq + local_row * HD) : (is_k ? (ok + local_row * HD) : (ov + local_row * HD));
uint base = tid * 4;
threadgroup float local_sums[32];
threadgroup float local_inv[1];
threadgroup T nrm[HD];
float acc = 0.0f;
for (uint i = 0; i < 4; ++i) {
    float v = float(src[base + i]);
    acc += v * v;
}
acc = simd_sum(acc);
if (simd_group == 0) {
    local_sums[simd_lane] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_lane == 0) {
    local_sums[simd_group] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_group == 0) {
    acc = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0) {
        local_inv[0] = metal::precise::rsqrt(acc / float(HD) + float(eps));
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = local_inv[0];
for (uint i = 0; i < 4; ++i) {
    nrm[base + i] = NORM ? T(w[base + i] * T(float(src[base + i]) * inv)) : src[base + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const device float* arow = angles + seq * uint(RD);
for (uint i = 0; i < 4; ++i) {
    uint j = base + i;
    T r;
    if (is_v || j >= uint(RD)) {
        r = nrm[j];
    } else if (j < half_rd) {
        r = T(float(nrm[j]) * arow[j] - float(nrm[j + half_rd]) * arow[half_rd + j]);
    } else {
        r = T(float(nrm[j - half_rd]) * arow[j] + float(nrm[j]) * arow[j - half_rd]);
    }
    dst[j] = r;
}