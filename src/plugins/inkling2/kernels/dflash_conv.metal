// Inkling 2 DFlash2: fused grouped dynamic convolution (and optional residual add on STAGE == 1).
// X [B, L, H] (T), DYN [B, L, 2, KT, G] (T), BASE [2, KT, H] (BT), RES [B, L, H] (T, unread when STAGE == 0).
// OUT [B, L, H] (T). grid (B * L * H, 1, 1), threadgroup (64, 1, 1).
const uint idx = thread_position_in_grid.x;
const uint B = X_shape[0];
const uint L = X_shape[1];
const uint total = B * L * H;
if (idx >= total) return;
const uint c = idx % H;
const uint bl = idx / H;
const uint l = bl % L;
const uint b = bl / L;
const uint G = H / GS;
const uint g = c / GS;
float acc = 0.0f;
for (uint t = 0; t < KT; ++t) {
    if (l < t) continue;
    const float v = float(X[(size_t(b) * L + (l - t)) * H + c]);
    const float bk = float(T(BASE[(size_t(STAGE) * KT + t) * H + c]));
    const float dk = float(DYN[(((size_t(b) * L + l) * 2u + STAGE) * KT + t) * G + g]);
    acc += bk * v + dk * v;
}
if constexpr (STAGE == 1) {
    acc = float(T(acc)) + float(RES[idx]);
}
OUT[idx] = T(acc);
