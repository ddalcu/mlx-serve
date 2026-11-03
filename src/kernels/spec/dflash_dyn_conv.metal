uint i = thread_position_in_grid.x;
if (i >= uint(L * H)) return;
const int t = int(i) / H, c = int(i) % H;
T acc = T(0);
for (int tap = 0; tap < KS; ++tap) {
  const T v = t >= tap ? x[(t - tap) * H + c] : T(0);
  acc = T(float(acc) + float(T(float(base[tap * H + c]) * float(v))));
  acc = T(float(acc) + float(T(float(dyn[(t * KS + tap) * (H / GS) + c / GS]) * float(v))));
}
y[i] = acc;