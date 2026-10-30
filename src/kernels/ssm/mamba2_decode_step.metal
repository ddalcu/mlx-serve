const uint lane = thread_position_in_threadgroup.x;
const uint d = thread_position_in_grid.y;
const uint n = thread_position_in_grid.z;
const uint h = n % H;
const uint b = n / H;
const uint g = h / (H / NG);
const int R = rows;
constexpr int NS = DS / 32;
constexpr int CD = XD + 2 * NG * DS;
constexpr int XOFF = XD;
constexpr int DTOFF = XD + CD;
const int cx = int(h) * DH + int(d);
const device T* Pb = P + size_t(b) * R * PROJ;
const device T* CSb = CS_IN + size_t(b) * (KC - 1) * CD;
float st[NS];
const size_t sbase = size_t(n) * DH * DS + size_t(d) * DS + size_t(lane) * NS;
for (int i = 0; i < NS; i++) st[i] = S_IN[sbase + i];
const float A = -float(T(metal::exp(float(A_LOG[h]))));
const float dtb = float(DT_BIAS[h]);
#define TAP(ch, pos) ((pos) < 0 ? float(CSb[((pos) + KC - 1) * CD + (ch)]) : float(Pb[(pos) * PROJ + XOFF + (ch)]))
#define CONV(ch, rr, out) { \
    float a_ = float(CB[ch]); \
    for (int k_ = 0; k_ < KC; k_++) a_ = metal::fma(float(CW[(ch) * KC + k_]), TAP(ch, (rr) - (KC - 1) + k_), a_); \
    const float cv_ = float(T(a_)); \
    out = float(T(cv_ / (1.0f + metal::exp(-cv_)))); }
threadgroup float bc[MAXR * 2 * DS];
const uint tid = thread_position_in_threadgroup.y * 32 + lane;
// Passes of MAXR rows: stage their B/C, then run the recurrence over them.
for (int r0 = 0; r0 < R; r0 += MAXR) {
const int RB = metal::min(MAXR, R - r0);
for (int j = 0; j < RB; j++) {
    for (uint c = tid; c < 2 * DS; c += 32 * TGY) {
        const int ch = c < DS ? XD + int(g) * DS + int(c) : XD + NG * DS + int(g) * DS + int(c - DS);
        float v; CONV(ch, r0 + j, v);
        bc[j * 2 * DS + c] = v;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int j = 0; j < RB; j++) {
    const int rr = r0 + j;
    float xv = 0.0f;
    if (lane == 0) { CONV(cx, rr, xv); }
    xv = simd_broadcast(xv, 0);
    float dt = float(Pb[rr * PROJ + DTOFF + int(h)]) + dtb;
    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));
    dt = metal::clamp(dt, dt_lo, dt_hi);
    const float dA = metal::exp(A * dt);
    const float xdt = xv * dt;
    float acc = 0.0f;
    for (int i = 0; i < NS; i++) {
        const float s = dA * st[i] + xdt * bc[j * 2 * DS + int(lane) * NS + i];
        st[i] = s;
        acc += s * bc[j * 2 * DS + DS + int(lane) * NS + i];
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        const float dx = float(T(T(xv) * DSKIP[h]));
        const float y = float(T(acc + dx));
        const float z = float(Pb[rr * PROJ + cx]);
        const float sz = float(T(z / (1.0f + metal::exp(-z))));
        Y[(size_t(b) * R + rr) * XD + cx] = T(sz * y);
    }
    if (ALLROWS || rr == R - 1) {
        const size_t so = ALLROWS ? size_t(rr) * B * H * DH * DS : 0;
        for (int i = 0; i < NS; i++) S_OUT[so + sbase + i] = st[i];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
// conv state after the last row: its last KC-1 inputs, one thread per channel.
if (lane == 0) {
    for (int k = 0; k < KC - 1; k++) {
        const int pos = R - 1 - (KC - 2) + k;
        CS_OUT[(size_t(b) * (KC - 1) + k) * CD + cx] = pos < 0 ? CSb[(pos + KC - 1) * CD + cx] : Pb[pos * PROJ + XOFF + cx];
    }
}
if ((h % (H / NG)) == 0 && d == 0) {
    for (int c = int(lane); c < 2 * DS; c += 32) {
        const int ch = c < DS ? XD + int(g) * DS + c : XD + NG * DS + int(g) * DS + (c - DS);
        for (int k = 0; k < KC - 1; k++) {
            const int pos = R - 1 - (KC - 2) + k;
            CS_OUT[(size_t(b) * (KC - 1) + k) * CD + ch] = pos < 0 ? CSb[(pos + KC - 1) * CD + ch] : Pb[pos * PROJ + XOFF + ch];
        }
    }
}