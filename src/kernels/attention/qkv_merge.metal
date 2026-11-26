const uint e = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.y;
const auto mr = m_in + (long)row * NS;
const auto lr = l_in + (long)row * NS;
float M = -1e30f;
for (int i = 0; i < NS; ++i) M = max(M, mr[i]);
// A sink is one more key of zero value: its logit joins the max and the denominator, not the sum.
const float sk = HAS_SINK ? float(sink[row]) : 0.0f;
if (HAS_SINK) M = max(M, sk);
float L = HAS_SINK ? metal::exp(sk - M) : 0.0f, acc = 0.0f;
const auto orow = o_in + (long)row * NS * DV + e;
for (int i = 0; i < NS; ++i) {
  const float w = metal::exp(mr[i] - M);
  L += lr[i] * w;
  acc += orow[(long)i * DV] * w;
}
out[(long)row * DV + e] = T(acc / L);
