//! Ling 3.0's multi-head latent attention (the naive DeepSeek-V3 form: per-head keys and values expanded before the
//! cache) on one slot through `CacheOps`.
//! copied_from: sdk/blocks/attention/mla.zig (`mlaAttnWith`, the cache update replaced by `Slots.append`).

const mlx = @import("../../mlx/ffi.zig");
const Slots = @import("../glm5_next/slots.zig").Slots;
const trunk = @import("../glm5_next/trunk.zig");

pub fn mlaAttn(
    self: *const trunk.Model,
    sl: Slots,
    x: mlx.mlx_array,
    fa: *const trunk.Mla,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
) !mlx.mlx_array {
    const cfg = self.c;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const nope: c_int = @intCast(cfg.mla_qk_nope_head_dim);
    const rope_d: c_int = @intCast(cfg.mla_qk_rope_head_dim);
    const qk_dim: c_int = nope + rope_d;
    const v_dim: c_int = @intCast(cfg.mla_v_head_dim);
    const kv_lora: c_int = @intCast(cfg.mla_kv_lora_rank);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const strides3 = [_]c_int{ 1, 1, 1 };
    const strides4 = [_]c_int{ 1, 1, 1, 1 };

    // ── Q, then split into the non-positional and rope halves.
    //
    // Two arms. The low-rank pair (q_a → norm → q_b) is tiny's; a direct
    // q_proj is flash's, where `q_lora_rank` is null. Both produce the same
    // [batch, seq, heads * qk_head_dim] flat query, so everything below is
    // shared — only the projection differs.
    const q_flat = if (fa.q.w.ctx != null)
        try self.mm(&fa.q, x)
    else blk: {
        const q_a = try self.mm(&fa.q_a, x);
        defer _ = mlx.mlx_array_free(q_a);
        const q_a_n = try self.rmsNorm(q_a, fa.q_a_norm);
        defer _ = mlx.mlx_array_free(q_a_n);
        break :blk try self.mm(&fa.q_b, q_a_n);
    };
    defer _ = mlx.mlx_array_free(q_flat);

    var q_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_heads);
    try mlx.check(mlx.mlx_reshape(&q_heads, q_flat, &[_]c_int{ batch, seq_len, h_count, qk_dim }, 4, self.s));
    var q_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_heads, &perm, 4, self.s));

    var q_nope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_nope);
    try mlx.check(mlx.mlx_slice(&q_nope, q_t, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, h_count, seq_len, nope }, 4, &strides4, 4, self.s));
    var q_rot_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rot_raw);
    try mlx.check(mlx.mlx_slice(&q_rot_raw, q_t, &[_]c_int{ 0, 0, 0, nope }, 4, &[_]c_int{ batch, h_count, seq_len, qk_dim }, 4, &strides4, 4, self.s));

    // ── KV: one compressed latent per token, plus one shared rope key.
    const kv_a = try self.mm(&fa.kv_a, x);
    defer _ = mlx.mlx_array_free(kv_a);
    var latent_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(latent_raw);
    try mlx.check(mlx.mlx_slice(&latent_raw, kv_a, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, kv_lora }, 3, &strides3, 3, self.s));
    var k_pe_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_pe_flat);
    try mlx.check(mlx.mlx_slice(&k_pe_flat, kv_a, &[_]c_int{ 0, 0, kv_lora }, 3, &[_]c_int{ batch, seq_len, kv_lora + rope_d }, 3, &strides3, 3, self.s));

    const latent = try self.rmsNorm(latent_raw, fa.kv_a_norm);
    defer _ = mlx.mlx_array_free(latent);
    const kv_flat = try self.mm(&fa.kv_b, latent);
    defer _ = mlx.mlx_array_free(kv_flat);
    var kv_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_heads);
    try mlx.check(mlx.mlx_reshape(&kv_heads, kv_flat, &[_]c_int{ batch, seq_len, h_count, nope + v_dim }, 4, self.s));
    var kv_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_t);
    try mlx.check(mlx.mlx_transpose_axes(&kv_t, kv_heads, &perm, 4, self.s));
    var k_nope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_nope);
    try mlx.check(mlx.mlx_slice(&k_nope, kv_t, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, h_count, seq_len, nope }, 4, &strides4, 4, self.s));
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_slice(&v_t, kv_t, &[_]c_int{ 0, 0, 0, nope }, 4, &[_]c_int{ batch, h_count, seq_len, nope + v_dim }, 4, &strides4, 4, self.s));

    var k_pe_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_pe_t);
    {
        var k_pe_h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_pe_h);
        try mlx.check(mlx.mlx_reshape(&k_pe_h, k_pe_flat, &[_]c_int{ batch, seq_len, 1, rope_d }, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&k_pe_t, k_pe_h, &perm, 4, self.s));
    }

    // RoPE over the whole rope slice, adjacent-pair rotation.
    const traditional = cfg.rope_interleaved_pairs;
    var q_rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rot);
    var k_rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rot);
    try mlx.check(mlx.mlx_fast_rope(&q_rot, q_rot_raw, rope_d, traditional, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, offset, .{ .ctx = null }, self.s));
    try mlx.check(mlx.mlx_fast_rope(&k_rot, k_pe_t, rope_d, traditional, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, offset, .{ .ctx = null }, self.s));

    // The rope key is MQA — one head, replicated so the cached K is a
    // plain [B,H,S,qk_dim] tensor the standard cache path can carry.
    var k_rot_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rot_b);
    try mlx.check(mlx.mlx_broadcast_to(&k_rot_b, k_rot, &[_]c_int{ batch, h_count, seq_len, rope_d }, 4, self.s));

    var queries = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(queries);
    {
        const parts = [_]mlx.mlx_array{ q_nope, q_rot };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&queries, vec, -1, self.s));
    }
    var keys = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keys);
    {
        const parts = [_]mlx.mlx_array{ k_nope, k_rot_b };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&keys, vec, -1, self.s));
    }
    // The cache stores views; a strided V slice must own its bytes before
    // it is written into the buffer (mlx `slice` is a VIEW op).
    var v_contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_contig);
    try mlx.check(mlx.mlx_contiguous(&v_contig, v_t, false, self.s));

    var kv_view = try sl.append(0, layer, keys, v_contig, self.s);
    defer kv_view.deinit();

    // Plain SDPA: MLX has a fused vector kernel for exactly this
    // asymmetric (192-wide query, 128-wide value) decode shape, and falls
    // back to the composed path at prefill widths. The quantized-KV fused
    // kernels assume one head width for both, so MLA never opts in.
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    const sel_mode: [*:0]const u8 = if (is_prefill) "causal" else "";
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, queries, kv_view.k, kv_view.v, attn_scale, sel_mode, none_mask, .{ .ctx = null }, false, self.s));

    // [B,H,S,Dv] → [B,S,H,Dv]; the gate is per (token, head).
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));

    var gated: mlx.mlx_array = .{ .ctx = null };
    defer if (gated.ctx != null) {
        _ = mlx.mlx_array_free(gated);
    };
    if (cfg.mla_head_gate and fa.g.w.ctx != null) {
        const gate_logits = try self.mm(&fa.g, x);
        defer _ = mlx.mlx_array_free(gate_logits);
        // The reference takes the sigmoid in fp32 and casts back.
        var g_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_f32);
        try mlx.check(mlx.mlx_astype(&g_f32, gate_logits, .float32, self.s));
        var g_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_sig);
        try mlx.check(mlx.mlx_sigmoid(&g_sig, g_f32, self.s));
        var g_cast = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_cast);
        try mlx.check(mlx.mlx_astype(&g_cast, g_sig, mlx.mlx_array_dtype(attn_t), self.s));
        var g_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_4d);
        try mlx.check(mlx.mlx_reshape(&g_4d, g_cast, &[_]c_int{ batch, seq_len, h_count, 1 }, 4, self.s));
        gated = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&gated, attn_t, g_4d, self.s));
    }

    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, if (gated.ctx != null) gated else attn_t, &[_]c_int{ batch, seq_len, h_count * v_dim }, 3, self.s));
    // `qmatmul`, NOT `attnProj`: the decode-only int8 requant is a LOSSY
    // default measured on qwen3.5/gemma-shaped dense attention, and a
    // shared lossy flag's default is not transferable between archs (the
    // dsv4 comp_in rule). It would also apply to exactly ONE of this
    // layer's five projections — the other four are low-rank and go
    // through qmatmul — so a dense checkpoint would get an inconsistent
    // transform nobody characterized. Quantized checkpoints (every
    // shipped mirror) skip the path either way; this only binds a future
    // bf16 mirror, and it can opt in once someone A/Bs it.
    return self.mm(&fa.o, attn_flat);
}
