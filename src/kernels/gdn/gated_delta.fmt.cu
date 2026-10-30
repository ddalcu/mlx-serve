const int n = blockIdx.z * blockDim.z + threadIdx.z;
const int b_idx = n / Hv;
const int hv_idx = n % Hv;
const int hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
const int T = T_arr[0];
const int dk_idx = threadIdx.x;
const int dv0 = (blockIdx.y * blockDim.y + threadIdx.y) * RPW;
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
auto i_state = state_in + (n * Dv + dv0) * Dk;
auto o_state = state_out + (n * Dv + dv0) * Dk;
float state[RPW][n_per_t];
for (int r = 0; r < RPW; ++r)
  for (int i = 0; i < n_per_t; ++i) state[r][i] = static_cast<float>(i_state[r * Dk + n_per_t * dk_idx + i]);
{s}
auto beta_ = beta + b_idx * T * Hv;
float kn[n_per_t], qn[n_per_t], gn[{s}], vn[RPW], bn;
auto load = [&](int tt) {{
  for (int i = 0; i < n_per_t; ++i) {{
    const int s_idx = n_per_t * dk_idx + i;
    kn[i] = static_cast<float>(k_[(long)tt * Hk * Dk + s_idx]);
    qn[i] = static_cast<float>(q_[(long)tt * Hk * Dk + s_idx]);
    {s}
  }}
  for (int r = 0; r < RPW; ++r) vn[r] = static_cast<float>(v_[(long)tt * Hv * Dv + dv0 + r]);
  bn = static_cast<float>(beta_[(long)tt * Hv + hv_idx]);
}};
if (T > 0) load(0);
for (int t = 0; t < T; ++t) {{
  float kc[n_per_t], qc[n_per_t], gc[{s}], vc[RPW];
  for (int i = 0; i < n_per_t; ++i) {{ kc[i] = kn[i]; qc[i] = qn[i]; }}
  for (int i = 0; i < {s}; ++i) gc[i] = gn[i];
  for (int r = 0; r < RPW; ++r) vc[r] = vn[r];
  const float bc = bn;
  if (t + 1 < T) load(t + 1);
  float kv_mem[RPW];
  for (int r = 0; r < RPW; ++r) kv_mem[r] = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {{
    for (int r = 0; r < RPW; ++r) {{
      state[r][i] = state[r][i] * {s};
      kv_mem[r] += state[r][i] * kc[i];
    }}
  }}
  for (int r = 0; r < RPW; ++r) kv_mem[r] = gdn_warp_sum(kv_mem[r]);
  float delta[RPW];
  for (int r = 0; r < RPW; ++r) delta[r] = (vc[r] - kv_mem[r]) * bc;
  float out[RPW];
  for (int r = 0; r < RPW; ++r) out[r] = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {{
    for (int r = 0; r < RPW; ++r) {{
      state[r][i] = state[r][i] + kc[i] * delta[r];
      out[r] += state[r][i] * qc[i];
    }}
  }}
  for (int r = 0; r < RPW; ++r) out[r] = gdn_warp_sum(out[r]);
  if (dk_idx == 0)
    for (int r = 0; r < RPW; ++r) y_[(long)t * Hv * Dv + dv0 + r] = static_cast<OutT>(out[r]);
{s}}}
for (int r = 0; r < RPW; ++r)
  for (int i = 0; i < n_per_t; ++i) o_state[r * Dk + n_per_t * dk_idx + i] = static_cast<StT>(state[r][i]);
