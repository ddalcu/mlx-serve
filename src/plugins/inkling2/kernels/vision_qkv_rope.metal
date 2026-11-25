#pragma clang fp contract(off)
// Inkling 2 vision: fused QKV bias add, 2-D pairwise rotary on Q and K, and heads-major split.
// QKV [L, 3*H*D] (T), B [3*H*D] (T), COS [L, D] f32, SIN [L, D] f32 -> OQ, OK, OV [1, H, L, D] (T).
// grid (ceil(3*H*(D/2) / TG) * TG, L, 1), threadgroup (TG, 1, 1).
const uint L = QKV_shape[0];
constexpr uint PAIRS = uint(D) / 2u;
constexpr uint TOTAL = 3u * uint(H) * PAIRS;
const uint id = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (id >= TOTAL || row >= L) return;

const uint slot = id / PAIRS;
const uint pair = id - slot * PAIRS;
const uint which = slot / uint(H);
const uint head = slot - which * uint(H);
const uint col = slot * uint(D) + pair * 2u;
const size_t src = size_t(row) * (3u * uint(H) * uint(D)) + col;
const size_t dst = (size_t(head) * L + row) * uint(D) + pair * 2u;

const T v0 = T(float(QKV[src]) + float(B[col]));
const T v1 = T(float(QKV[src + 1u]) + float(B[col + 1u]));
if (which == 2u) {
    OV[dst] = v0;
    OV[dst + 1u] = v1;
    return;
}

const size_t tab = size_t(row) * uint(D) + pair * 2u;
const float c0 = COS[tab];
const float c1 = COS[tab + 1u];
const float s0 = SIN[tab];
const float s1 = SIN[tab + 1u];
const float f0 = float(v0);
const float f1 = float(v1);
const T r0 = T(f0 * c0 + (-f1) * s0);
const T r1 = T(f1 * c1 + f0 * s1);
device T* out = which == 0u ? OQ : OK;
out[dst] = r0;
out[dst + 1u] = r1;
