#pragma clang fp contract(off)
// One thread per (token, head, rotation pair j < D/2): DINOv3's rotate-half RoPE on a token-major projection
// X [L, H*D], written heads-major O [1, H, L, D]; the first P tokens (CLS, registers) pass through. COS and SIN
// [L - P, D] are of X's dtype. Each product and the sum round to T, as the chain's separate multiplies and add do.
const uint L = X_shape[0];
constexpr uint HALF = D / 2;
const uint id = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (id >= uint(H) * HALF) return;
const uint head = id / HALF, j = id - head * HALF;
const device T* x = X + size_t(row) * (uint(H) * D) + head * D;
device T* o = O + (size_t(head) * L + row) * D;
const T x0 = x[j], x1 = x[j + HALF];
if (row < uint(P)) {
    o[j] = x0;
    o[j + HALF] = x1;
    return;
}
const size_t t = size_t(row - uint(P)) * D;
const float a0 = float(T(float(x0) * float(COS[t + j])));
const float b0 = float(T(-float(x1) * float(SIN[t + j])));
const float a1 = float(T(float(x1) * float(COS[t + j + HALF])));
const float b1 = float(T(float(x0) * float(SIN[t + j + HALF])));
o[j] = T(a0 + b0);
o[j + HALF] = T(a1 + b1);
