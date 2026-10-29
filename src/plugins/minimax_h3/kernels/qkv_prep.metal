constexpr uint HD = 128;
const uint slot = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint lane = thread_position_in_threadgroup.x;
const uint which = slot / uint(H);
const uint head = slot - which * uint(H);
const uint seq = uint(COS_shape[0]);
const device T* src = QKV + size_t(row) * (3 * uint(H) * HD) + size_t(slot) * HD;
const size_t o = (size_t(head) * seq + row) * HD;
const uint base = lane * 4;
if (which == 2) {
    for (uint i = 0; i < 4; ++i) OV[o + base + i] = src[base + i];
    return;
}
float acc = 0.0f;
for (uint i = 0; i < 4; ++i) {
    float v = float(src[base + i]);
    acc += v * v;
}
acc = simd_sum(acc);
const float inv = metal::precise::rsqrt(acc / float(HD) + eps);
const device T* w = which == 0 ? QW : KW;
T nrm[4];
for (uint i = 0; i < 4; ++i) nrm[i] = w[base + i] * T(float(src[base + i]) * inv);
constexpr uint QL = uint(HALF) / 4;
const uint pl = lane < QL ? lane + QL : (lane < 2 * QL ? lane - QL : lane);
device T* dst = which == 0 ? OQ : OK;
const size_t tab = size_t(row) * uint(HALF);
for (uint i = 0; i < 4; ++i) {
    const uint j = base + i;
    const float pv = simd_shuffle(float(nrm[i]), pl);
    T res;
    if (j < uint(HALF)) {
        const T a1 = T(float(nrm[i]) * float(COS[tab + j]));
        const T a2 = T(pv * float(SIN[tab + j]));
        res = T(float(a1) - float(a2));
    } else if (j < 2 * uint(HALF)) {
        const uint jj = j - uint(HALF);
        const T b1 = T(pv * float(SIN[tab + jj]));
        const T b2 = T(float(nrm[i]) * float(COS[tab + jj]));
        res = T(float(b1) + float(b2));
    } else {
        res = nrm[i];
    }
    dst[o + j] = res;
}