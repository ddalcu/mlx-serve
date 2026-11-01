// GroupNorm, pass 2, with the SiLU after it: one thread per 4 channels of a pixel of X [B, H, W, C] (channels-last,
// C / GROUPS a multiple of 4). ST [B, GROUPS, 2] f32 holds each group's mean and rstd, W and BI [C] f32 the affine.
// O = silu((x - mean) * rstd * W + BI), in f32 and rounded once to O's dtype.
const uint C = X_shape[3];
const size_t total = size_t(X_shape[0]) * X_shape[1] * X_shape[2] * C;
const size_t e = size_t(thread_position_in_grid.x) * 4;
if (e >= total) return;
const uint c0 = uint(e % C);
const size_t b = e / (size_t(X_shape[1]) * X_shape[2] * C);
const size_t at = (b * GROUPS + c0 / (C / GROUPS)) * 2;
const float mean = ST[at], rstd = ST[at + 1];
const float4 x = float4(*(const device vec<T, 4>*)(X + e));
const float4 y = (x - mean) * rstd * *(const device float4*)(W + c0) + *(const device float4*)(BI + c0);
*(device vec<OT, 4>*)(O + e) = vec<OT, 4>(y / (1.0f + metal::precise::exp(-y)));
