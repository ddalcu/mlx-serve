// torchvision's deform_conv2d columns: for output position `pos` and tap p (row-major in the k x k kernel), the
// bilinear sample of x at the tap's grid point plus its predicted offset (zero outside the plane, corner by corner),
// times the tap's modulator; f32 positions, weights and sum, one rounding to T. One thread per 4 channels.
const uint c4 = thread_position_in_grid.x * 4;
const uint p = thread_position_in_grid.y;
const uint pos = thread_position_in_grid.z;
const uint C = X_shape[1];
if (c4 >= C) return;
const int y = int(pos) / W, x = int(pos) % W;
const int i = int(p) / KS, j = int(p) % KS;
const float py = float(y + i - KS / 2) + OFF[size_t(pos) * 2 * KS * KS + 2 * p];
const float px = float(x + j - KS / 2) + OFF[size_t(pos) * 2 * KS * KS + 2 * p + 1];
const float fy = metal::floor(py), fx = metal::floor(px);
const float ly = py - fy, lx = px - fx;
float4 acc = 0;
for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
        const int yy = int(fy) + dy, xx = int(fx) + dx;
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        const float wgt = (dy == 0 ? 1.0f - ly : ly) * (dx == 0 ? 1.0f - lx : lx);
        acc += wgt * float4(*(const device vec<T, 4>*)(X + size_t(yy * W + xx) * C + c4));
    }
}
const float m = float(MSK[size_t(pos) * KS * KS + p]);
*(device vec<T, 4>*)(O + size_t(pos) * KS * KS * C + p * C + c4) = vec<T, 4>(acc * m);
