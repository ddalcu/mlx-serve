const int o = int(thread_position_in_grid.x);
float a = 0.0f;
for (int k = 0; k < 64; k += 4)
    a += dot(LDW(WX + o * 64, k), float4(float(Y[k]), float(Y[k + 1]), float(Y[k + 2]), float(Y[k + 3])));
const bfloat16_t xe = bfloat16_t(a + float(BX[o]));
for (int b = 0; b < 2; b++) {
    OUT[(b * 3 + 0) * 1024 + o] = bfloat16_t(float(TE[b * 1024 + o]) + float(CE[b * 1024 + o]));
    OUT[(b * 3 + 1) * 1024 + o] = XH[o];
    OUT[(b * 3 + 2) * 1024 + o] = xe;
}
