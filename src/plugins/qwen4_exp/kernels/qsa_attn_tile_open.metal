
threadgroup T KVs[LDK * BD];
threadgroup T* Ks = KVs;
threadgroup T* Vs = KVs;

const short2 sc = msv_coord(lane);
const short sn = sc.x;
const short sm = sc.y;
const short tm = 8 * short(warp);
const int Ks_off = sm * LDK + sn;
const int Vs_off = sm * LDV + sn;
const int row = tm + sm;
const bool row_ok = row < gqa;

float2 Qfrag[BD / 8];
if (row_ok) {
  const device T* Qrow = q + bb * qS[0] + (long)(hk * gqa + row) * qS[1] + (long)s * qS[2];
  for (int dd = 0; dd < BD / 8; ++dd) {
    const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    Qfrag[dd] = float2(float(pr.x), float(pr.y));
  }
} else {
  for (int dd = 0; dd < BD / 8; ++dd) Qfrag[dd] = float2(0.0f);
}
float2 Ofrag[BD / 8];
for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
float max_score = -3.0e38f;
float sum_score = 0.0f;

for (int t0 = t_lo; t0 < t_hi; t0 += BK) {
  const int rows_k = metal::min(BK, t_hi - t0);

