auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;
{s}
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {{
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}}

{s}
auto beta_ = beta + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {{
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {{
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * {s};
    kv_mem += state[i] * k_[s_idx];
  }}
  kv_mem = simd_sum(kv_mem);

  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {{
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }}
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {{
    y[dv_idx] = static_cast<OutT>(out);
  }}
{s}  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
{s}
  beta_ += Hv;
}}
for (int i = 0; i < n_per_t; ++i) {{
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<StT>(state[i]);
}}
