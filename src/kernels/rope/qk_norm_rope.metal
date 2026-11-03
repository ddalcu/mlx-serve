constexpr uint HD = 128;
uint head = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
bool is_q = head < uint(HQ);
const device T* src = is_q ? (q + head * HD) : (k + (head - uint(HQ)) * HD);
const device T* w = is_q ? qw : kw;
device T* dst = is_q ? (oq + head * HD) : (ok + (head - uint(HQ)) * HD);
uint base = lane * 4;
float acc = 0.0f;
for (uint i = 0; i < 4; ++i) {
    float v = float(src[base + i]);
    acc += v * v;
}
acc = simd_sum(acc);
float inv = metal::precise::rsqrt(acc / float(HD) + float(eps));
T nrm[4];
for (uint i = 0; i < 4; ++i) {
    nrm[i] = w[base + i] * T(float(src[base + i]) * inv);
}
constexpr uint half_rd = uint(RD) / 2;
float partner[4];
for (uint i = 0; i < 4; ++i) {
    partner[i] = simd_shuffle(float(nrm[i]), lane ^ (half_rd / 4));
}
for (uint i = 0; i < 4; ++i) {
    uint j = base + i;
    T r;
    if (j < half_rd) {
        r = T(float(nrm[i]) * angles[j] - partner[i] * angles[half_rd + j]);
    } else if (j < uint(RD)) {
        r = T(partner[i] * angles[j] + float(nrm[i]) * angles[j - half_rd]);
    } else {
        r = nrm[i];
    }
    if (SCALED != 0) {
        r = r * ms[j];
    }
    dst[j] = r;
}