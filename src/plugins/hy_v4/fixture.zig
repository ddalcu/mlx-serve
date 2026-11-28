//! The conformance pack: the tiny hy_v4_flash config and the matching micro safetensors set. Every name
//! `weights.zig` validates exists at its config shape; the Engram table holds the rows the config's head-
//! vocab lists sum to (800), at the packed geometry the stream bank opens. The SDK loader's dequantise-small
//! line rewrites the quant triples into dense rows, so the pack reaches the map half-dense — the fixture
//! exercises both arms of every shape check.
//!
//! The suite loads this pack and runs `forward` in every declared mode.

const std = @import("std");
const File = @import("../../sdk/testing/fixture.zig").File;
const Tensors = @import("../../sdk/testing/tensors.zig").Tensors;

pub const tiny_config =
    \\{"model_type":"hy_v4_flash","text_config":{"model_type":"hy_v4_flash",
        \\"vocab_size":64,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":2,"head_dim":64,
        \\"rms_norm_eps":1e-5,"max_position_embeddings":4096,"eos_token_id":2,
        \\"q_lora_rank":64,"kv_lora_rank":64,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":64,
        \\"gated_mla":true,"learnable_sink":true,"sandwich_norm":true,"sliding_window":512,"swa_num_attention_heads":4,
        \\"rope_parameters":{"deepseek_sparse_attention":{"rope_type":"default","rope_theta":10000000,
        \\"mrope_interleaved":true,"mrope_section":[12,10,10]},
        \\"sliding_attention":{"rope_type":"default","rope_theta":10000,
        \\"mrope_interleaved":true,"mrope_section":[12,10,10]}},
        \\"layer_types":["deepseek_sparse_attention","sliding_attention"],
        \\"swa_pattern":[0,1],"indexer_types":["full","full"],
        \\"use_dsa":true,"use_compressed_indexer":true,"compressed_indexer_skip_fine_select":true,
        \\"compressed_indexer_reuse_fine_indexer":false,
        \\"index_topk":8,"index_n_heads":16,"index_head_dim":2,"compressed_indexer_compress_ratio":4,
        \\"compressed_indexer_group_topk":2,"compressed_indexer_stage2_query_block_size":64,
        \\"first_k_dense_replace":1,"mlp_layer_types":["dense","sparse"],
        \\"n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,"moe_intermediate_size":64,
        \\"routed_expert_hidden_size":64,"intermediate_size":64,"routed_scaling_factor":3.998,"swiglu_limit":10.0,
        \\"norm_topk_prob":true,"n_group":1,"topk_group":1,"moe_router_use_sigmoid":true,
        \\"moe_router_enable_expert_bias":true,"gating_type":"elementwise",
        \\"enable_ihc":true,"hc_mult":4,"hc_magnitude":2.0,"hc_eps":1e-6,
        \\"engram_max_ngram_size":3,"engram_n_head_per_ngram":8,"engram_n_embed_per_ngram":768,"engram_pad_id":3,
        \\"engram_lookup_table_size":64,
        \\"engram_vocab_size":[50,50],"engram_layer_ids":[2],
        \\"engram_layer_head_vocab_sizes":{"2":[[50,50,50,50,50,50,50,50],[50,50,50,50,50,50,50,50]]},
        \\"num_nextn_predict_layers":0},
        \\"quantization":{"bits":4,"group_size":64,"mode":"affine",
        \\"model.layers.1.engram_layer.multi_head_embedding.embedding":{"bits":4,"group_size":32,"mode":"affine"}}}
    ;

pub fn build(a: std.mem.Allocator) ![]const File {
    var t = Tensors.init(a, 7);
    try q4(&t, "model.embed_tokens", .{}, &.{64}, 64); // 64 vocab rows, 64-group quant
    try t.fill("model.norm.weight", .{}, &.{@as(u64, 64)}, .BF16, 1);
    try q4(&t, "lm_head", .{}, &.{64}, 64);
    try t.fill("model.hc_head.hc_head_base", .{}, &.{@as(u64, 4)}, .F32, 0.5);
    try t.fill("model.hc_head.hc_head_fn", .{}, &.{ 4, 256 }, .F32, 0.1);
    try t.fill("model.hc_head.hc_head_scale", .{}, &.{@as(u64, 1)}, .F32, 1);
    for (0..2) |li| {
        try t.fill("model.layers.{d}.input_layernorm.weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
        try t.fill("model.layers.{d}.post_attention_layernorm.weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
        try q4(&t, "model.layers.{d}.self_attn.q_a_proj", .{li}, &.{64}, 64);
        try t.fill("model.layers.{d}.self_attn.q_a_layernorm.weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
        // Layer 0 is a DSA layer (2 heads), layer 1 a SWA layer (4 heads): both geometries live in the pack.
        const h: u64 = if (li == 0) 2 else 4;
        try q4(&t, "model.layers.{d}.self_attn.q_b_proj", .{li}, &.{h * 256}, 64);
        try q4(&t, "model.layers.{d}.self_attn.kv_a_proj_with_mqa", .{li}, &.{128}, 64);
        try t.fill("model.layers.{d}.self_attn.kv_a_layernorm.weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
        try q4(&t, "model.layers.{d}.self_attn.kv_b_proj", .{li}, &.{h * 256}, 64);
        try q4(&t, "model.layers.{d}.self_attn.linear_gate", .{li}, &.{h * 64}, 64);
        try q4(&t, "model.layers.{d}.self_attn.o_proj", .{li}, &.{64}, h * 64);
        try t.fill("model.layers.{d}.self_attn.post_proj_norm.weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
        try t.fill("model.layers.{d}.self_attn.learnable_sink_param", .{li}, &.{h}, .F32, 0);
        try t.fill("model.layers.{d}.hc_attn_layer.hc_pre.hc_base", .{li}, &.{@as(u64, 8)}, .F32, 0);
        try t.fill("model.layers.{d}.hc_attn_layer.hc_pre.hc_fn", .{li}, &.{ 8, 256 }, .F32, 0.1);
        try t.fill("model.layers.{d}.hc_attn_layer.hc_pre.hc_scale", .{li}, &.{@as(u64, 2)}, .F32, 0);
        try t.fill("model.layers.{d}.hc_mlp_layer.hc_pre.hc_base", .{li}, &.{@as(u64, 8)}, .F32, 0);
        try t.fill("model.layers.{d}.hc_mlp_layer.hc_pre.hc_fn", .{li}, &.{ 8, 256 }, .F32, 0.1);
        try t.fill("model.layers.{d}.hc_mlp_layer.hc_pre.hc_scale", .{li}, &.{@as(u64, 2)}, .F32, 0);
        if (li == 0) {
            // The DSA layer: the compressed indexer arms.
            try q4(&t, "model.layers.{d}.self_attn.compressed_indexer.query_projection", .{li}, &.{32}, 64);
            try q4(&t, "model.layers.{d}.self_attn.compressed_indexer.token_key_projection", .{li}, &.{2}, 64);
            try q4(&t, "model.layers.{d}.self_attn.compressed_indexer.head_weight_projection", .{li}, &.{16}, 64);
            try t.fill("model.layers.{d}.self_attn.compressed_indexer.key_value_projection.weight", .{li}, &.{ 2, 2 }, .F32, 0.1);
            try t.fill("model.layers.{d}.self_attn.compressed_indexer.key_gate_projection.weight", .{li}, &.{ 2, 2 }, .F32, 0.1);
            try t.fill("model.layers.{d}.self_attn.compressed_indexer.key_norm.weight", .{li}, &.{@as(u64, 2)}, .F32, 1);
            try t.fill("model.layers.{d}.self_attn.compressed_indexer.token_key_norm.weight", .{li}, &.{@as(u64, 2)}, .BF16, 1);
            try t.fill("model.layers.{d}.self_attn.compressed_indexer.intra_group_ape", .{li}, &.{ 2, 2 }, .F32, 0);
            try q4(&t, "model.layers.{d}.mlp.gate_proj", .{li}, &.{64}, 64);
            try q4(&t, "model.layers.{d}.mlp.up_proj", .{li}, &.{64}, 64);
            try q4(&t, "model.layers.{d}.mlp.down_proj", .{li}, &.{64}, 64);
            try t.fill("model.layers.{d}.mlp.post_fc2_norm.weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
        } else {
            // The engram + MoE layer.
            try t.fill("model.layers.{d}.mlp.gate.weight", .{li}, &.{ 4, 64 }, .BF16, 0.1);
            try t.fill("model.layers.{d}.mlp.gate.e_score_correction_bias", .{li}, &.{@as(u64, 4)}, .F32, 0);
            try q4Experts(&t, "model.layers.{d}.mlp.experts.gate_up_proj", .{li}, 4, 128, 64);
            try q4Experts(&t, "model.layers.{d}.mlp.experts.down_proj", .{li}, 4, 64, 64);
            try q4(&t, "model.layers.{d}.mlp.routed_expert_up_proj", .{li}, &.{64}, 64);
            try q4(&t, "model.layers.{d}.mlp.routed_expert_down_proj", .{li}, &.{64}, 64);
            try q4(&t, "model.layers.{d}.mlp.shared_experts.gate_proj", .{li}, &.{64}, 64);
            try q4(&t, "model.layers.{d}.mlp.shared_experts.up_proj", .{li}, &.{64}, 64);
            try q4(&t, "model.layers.{d}.mlp.shared_experts.down_proj", .{li}, &.{64}, 64);
            try t.fill("model.layers.{d}.mlp.combine_output_norm.weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
            inline for (.{ @as(u64, 0), 1, 2, 3 }) |pi| {
                const nm = "model.layers.{d}.engram_layer.key_projs." ++ "0123"[pi .. pi + 1];
                // The memory is 16 heads x 96 = 1536 wide; each lane's key projection reads it.
                try q4(&t, nm, .{li}, &.{64}, 1536);
                try t.fill(nm ++ ".bias", .{li}, &.{@as(u64, 64)}, .BF16, 0);
                try t.fill("model.layers.{d}.engram_layer.norm_query." ++ "0123"[pi .. pi + 1] ++ ".weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
                try t.fill("model.layers.{d}.engram_layer.norm_key." ++ "0123"[pi .. pi + 1] ++ ".weight", .{li}, &.{@as(u64, 64)}, .BF16, 1);
                try t.fill("model.layers.{d}.engram_layer.unified_per_hc_scales." ++ "0123"[pi .. pi + 1], .{li}, &.{@as(u64, 64)}, .BF16, 1);
            }
            try q4(&t, "model.layers.{d}.engram_layer.value_proj", .{li}, &.{64}, 1536);
            try t.fill("model.layers.{d}.engram_layer.value_proj.bias", .{li}, &.{@as(u64, 64)}, .BF16, 0);
            try t.intsSeq("model.layers.{d}.engram_layer.lookup_table", .{li}, &.{@as(u64, 64)}, .I32);
            try t.fill("model.layers.{d}.engram_layer.unified_pre_gamma", .{li}, &.{@as(u64, 64)}, .BF16, 1);
            const heads = [_][]const u8{ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15" };
            inline for (heads) |hn| try t.fill(
                "model.layers.{d}.engram_layer.lookup_per_head_norms." ++ hn ++ ".weight",
                .{li}, &.{@as(u64, 96)}, .BF16, 1);
            // 16 heads x 96 dims, packed 4-bit/g32: 12 U32 words of 8 nibbles per row, 3 bf16 groups.
            try t.words("model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight", .{li}, &.{ 800, 12 });
            try t.fill("model.layers.{d}.engram_layer.multi_head_embedding.embedding.scales", .{li}, &.{ 800, 3 }, .BF16, 1);
            try t.fill("model.layers.{d}.engram_layer.multi_head_embedding.embedding.biases", .{li}, &.{ 800, 3 }, .BF16, 1);
        }
    }

    const st = try t.bytes();
    const files = try a.alloc(File, 2);
    files[0] = .{ .name = "config.json", .bytes = tiny_config };
    files[1] = .{ .name = "model.safetensors", .bytes = st };
    return files;
}

/// One affine packed triple, laid out the way the pack stores quant triples: `.weight` as U32 words of
/// eight 4-bit weights (out x in/8), `.scales`/`.biases` as bf16 at one value per 64-column group rounded
/// up. `out_shape` is the logical out-feature shape (1-D for a 2-D linear).
pub fn q4(t: *Tensors, comptime fmt: []const u8, args: anytype, out_shape: []const u64, in_features: u64) !void {
    var wshape = try t.a.alloc(u64, out_shape.len + 1);
    @memcpy(wshape[0..out_shape.len], out_shape);
    wshape[wshape.len - 1] = @divExact(in_features, 8); // 8 nibbles per U32
    try t.words(fmt ++ ".weight", args, wshape);
    const groups = @divExact(in_features, 64);
    var gshape = try t.a.alloc(u64, out_shape.len + 1);
    @memcpy(gshape[0..out_shape.len], out_shape);
    gshape[gshape.len - 1] = groups;
    try t.rand(fmt ++ ".scales", args, gshape, .BF16, 1);
    try t.rand(fmt ++ ".biases", args, gshape, .BF16, 0);
}

/// One fused expert triple, three axes like the pack: `[experts, out, in/8]` U32 words with bf16
/// `[experts, out, in/64]` scales and biases. The in-features are the routed latent width.
pub fn q4Experts(t: *Tensors, comptime fmt: []const u8, args: anytype, experts: u64, out_rows: u64, in_features: u64) !void {
    const groups = @divExact(in_features, 64);
    try t.words(fmt ++ ".weight", args, &.{ experts, out_rows, @divExact(in_features, 8) });
    try t.rand(fmt ++ ".scales", args, &.{ experts, out_rows, groups }, .BF16, 1);
    try t.rand(fmt ++ ".biases", args, &.{ experts, out_rows, groups }, .BF16, 0);
}
