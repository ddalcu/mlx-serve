const int n = thread_position_in_grid.x, l = thread_position_in_grid.y;
const uint tid = thread_index_in_threadgroup, tgn = threads_per_threadgroup.x;
const int nb = keys_shape[0];
threadgroup float qs[IH * IHD];
for (uint k = tid; k < IH * IHD; k += tgn) qs[k] = float(q[l * IH * IHD + k]);
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (n >= nb) return;
float dots[IH];
for (int h = 0; h < IH; ++h) dots[h] = 0.0f;
const device KT* kr = keys + size_t(n) * IHD;
for (int j = 0; j < IHD; ++j) {
  const float kv = float(kr[j]);
  for (int h = 0; h < IH; ++h) dots[h] += qs[h * IHD + j] * kv;
}
float s = 0.0f;
for (int h = 0; h < IH; ++h) s += (float(w[l * IH + h]) * wsc[0]) * metal::max(dots[h], 0.0f);
const int seen = (meta[0] + l + 1) / meta[1];
scores[size_t(l) * nb + n] = n < seen ? s : -INFINITY;