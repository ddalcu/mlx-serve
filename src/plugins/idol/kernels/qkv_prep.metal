constexpr uint HD = 64;
const uint lane = thread_index_in_simdgroup;
const uint slot = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
const uint row = threadgroup_position_in_grid.y;
if (slot >= 3 * uint(H)) return;
const uint which = slot / uint(H);
const uint head = slot - which * uint(H);
const size_t o = (size_t(head) * uint(Y_shape[0]) + row) * HD + lane * 2;
const vec<T, 2> y = *(const device vec<T, 2>*)(Y + size_t(row) * (3 * uint(H) * HD) + size_t(slot) * HD + lane * 2);
const vec<T, 2> b = *(const device vec<T, 2>*)(B + size_t(slot) * HD + lane * 2);
// The projection's bias, rounded to T as the chain's add rounds it.
const vec<T, 2> v = vec<T, 2>(T(float(y[0]) + float(b[0])), T(float(y[1]) + float(b[1])));
if (which == 2) {
    *(device vec<T, 2>*)(OV + o) = v;
    return;
}
const size_t tab = (size_t(row) * uint(H) + head) * (HD / 2) + lane;
const float c = COS[tab], s = SIN[tab];
const float x0 = float(v[0]), x1 = float(v[1]);
// Each product rounds on its own, as the chain's separate multiplies do; a fused one flips roundings.
const float m00 = x0 * c, m11 = x1 * s, m01 = x0 * s, m10 = x1 * c;
const T r0 = T(m00 - m11);
const T r1 = T(m01 + m10);
const float f0 = float(r0), f1 = float(r1);
// MLX's rms_single_row at width 64: thread t sums elements 4t..4t+3 in order (lane 2t holds them) and lane t < 16
// enters the simd_sum, so the f32 sum is the chain's bit for bit.
const float p0 = simd_shuffle_xor(f0, 1), p1 = simd_shuffle_xor(f1, 1);
float acc = f0 * f0;
acc += f1 * f1;
acc += p0 * p0;
acc += p1 * p1;
const float moved = simd_shuffle(acc, (lane & 15) * 2);
const float inv = metal::precise::rsqrt(simd_sum(lane < 16 ? moved : 0.0f) / float(HD) + eps);
const device float* g = (which == 0 ? QG : KG) + head * HD + lane * 2;
*(device vec<T, 2>*)((which == 0 ? OQ : OK) + o) = vec<T, 2>(T(f0 * inv * g[0]), T(f1 * inv * g[1]));
