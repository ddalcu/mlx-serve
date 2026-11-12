// A one-row router call in one launch (a std.fmt template): each threadgroup runs MLX's GEMVKernel::run at the stock
// gemv's template arguments (gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4), each logit stored atomically; the group that
// counts last on the call's own zeroed counter runs the stock router text, its read device-coherent.
threadgroup float ro1_tgp[1];
GEMVKernel<float, 4, 1, 1, 32, 4, 4, false>::run(w, x, logits, logits, ro1_in, ro1_out, ro1_ld, ro1_alpha, ro1_beta, ro1_bias_stride, ro1_tgp, threadgroup_position_in_grid, thread_position_in_threadgroup, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
constexpr uint RO1_NTG = {d}u;
threadgroup uint ro1_last;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
threadgroup_barrier(mem_flags::mem_device);
if (thread_position_in_threadgroup.x == 0u) {{
  device atomic_uint* ro1_arrivals = reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(counter));
  ro1_last = (atomic_fetch_add_explicit(ro1_arrivals, 1u, memory_order_relaxed) == RO1_NTG - 1u) ? 1u : 0u;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (ro1_last == 0u) {{
  return;
}}
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
{s}
