//! The trunk's routed experts streamed from the pack as downloaded (the `stream_experts` setting: `Streamed.openPack`,
//! the `Routed` trunk arm), over tiny packs (MXFP8 linears, affine 6-bit routed experts, BF16 embedding / head /
//! compressor / indexer keys, F32 hyper-connections, no Engram, no DSpark), against the same experts held in memory.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const model = @import("../../sdk/config/model.zig");
const dsv41 = @import("deepseek_v41.zig");
const streamed_w = @import("../../sdk/weights/streamed.zig");

const testing = std.testing;

const config_json =
    \\{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "bos_token_id": 0, "eos_token_id": 1,
    \\ "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 96, "hidden_size": 128,
    \\  "moe_intermediate_size": 128, "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 1,
    \\  "head_dim": 64, "qk_rope_head_dim": 16, "q_lora_rank": 64, "o_lora_rank": 32, "o_groups": 2, "hidden_act": "silu", "swiglu_limit": 1.0,
    \\  "rms_norm_eps": 1e-20, "attention_bias": false, "tie_word_embeddings": false,
    \\  "max_position_embeddings": 128, "rope_theta": 10000.0,
    \\  "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64},
    \\  "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 3, "scoring_func": "sqrtsoftplus",
    \\  "topk_method": "noaux_tc", "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 8,
    \\  "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1], "compress_rope_theta": 40000.0, "kv_source_layer_ids": [2, 5],
    \\  "index_source_layer_ids": [2, 5, 7], "index_n_heads": 8, "index_head_dim": 32, "index_topk": 4,
    \\  "candidate_source_layer_id": 5, "candidate_topk_blocks": 3, "candidate_block_size": 2, "hc_mult": 4,
    \\  "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "engram_layer_ids": [], "engram_num_embeddings": [],
    \\  "engram_max_ngram_size": 2, "engram_vocab_size": 50, "engram_n_heads": 1, "engram_head_dim": 64,
    \\  "engram_pad_token_id": 2, "engram_compressed_vocab_size": 96, "num_nextn_predict_layers": 0}}
;

/// A tiny pack's widths and layer table (`config_json`'s).
const Geo = struct { d: usize, inter: usize, ratios: []const u8, kv_sources: []const usize, index_sources: []const usize };
const tiny: Geo = .{ .d = 128, .inter = 128, .ratios = &.{ 0, 0, 2, 2, 2, 1, 1, 1 }, .kv_sources = &.{ 2, 5 }, .index_sources = &.{ 2, 5, 7 } };

/// The tiny pack's tensors (the geometry of `fixture.v41Weights`, in the repack layout).
fn repackWeights(a: std.mem.Allocator, seed: u64, geo: Geo, n_experts: u32) ![]u8 {
    const d = geo.d;
    const hd = 64;
    const ql = 64;
    const h = 4;
    const og = 2;
    const ol = 32;
    const inter = geo.inter;
    const e: usize = n_experts;
    const ih = 8;
    const ihd = 32;
    const vocab = 96;
    const ratios = geo.ratios;
    const kv_sources = geo.kv_sources;
    const index_sources = geo.index_sources;
    var p = fixture.Pack.init(a, seed);
    var buf: [160]u8 = undefined;
    try p.dense("embed.weight", &.{ vocab, d }, 0.5, 0, true);
    try p.dense("head.weight", &.{ vocab, d }, 0.1, 0, true);
    try p.norm("norm.weight", d);
    for (ratios, 0..) |ratio, li| {
        var pfx_buf: [32]u8 = undefined;
        const pfx = try std.fmt.bufPrint(&pfx_buf, "layers.{d}", .{li});
        const at = struct {
            fn f(b: []u8, pf: []const u8, name: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "{s}.{s}", .{ pf, name });
            }
        }.f;
        try fixture.hc(&p, pfx, d);
        try p.norm(try at(&buf, pfx, "attn_norm.weight"), d);
        try p.norm(try at(&buf, pfx, "ffn_norm.weight"), d);
        try p.mxfp8(try at(&buf, pfx, "attn.wq_a"), &.{ql}, d);
        try p.norm(try at(&buf, pfx, "attn.q_norm.weight"), ql);
        try p.mxfp8(try at(&buf, pfx, "attn.wq_b"), &.{h * hd}, ql);
        try p.mxfp8(try at(&buf, pfx, "attn.wkv"), &.{hd}, d);
        try p.norm(try at(&buf, pfx, "attn.kv_norm.weight"), hd);
        try p.mxfp8(try at(&buf, pfx, "attn.wo_a"), &.{og * ol}, h * hd / og);
        try p.mxfp8(try at(&buf, pfx, "attn.wo_b"), &.{d}, og * ol);
        try p.dense(try at(&buf, pfx, "attn.attn_sink"), &.{h}, 0.5, 0, false);
        const kv = std.mem.indexOfScalar(usize, kv_sources, li) != null;
        const ix = std.mem.indexOfScalar(usize, index_sources, li) != null;
        if (kv) {
            try p.dense(try at(&buf, pfx, "attn.compressor.wkv.weight"), &.{ hd, d }, 0.09, 0, true);
            if (ratio > 1) try p.dense(try at(&buf, pfx, "attn.compressor.wgate.weight"), &.{ hd, d }, 0.09, 0, true);
            try p.norm(try at(&buf, pfx, "attn.compressor.norm.weight"), hd);
            try p.dense(try at(&buf, pfx, "attn.indexer.wk.weight"), &.{ ihd, hd }, 0.12, 0, true);
            try p.norm(try at(&buf, pfx, "attn.indexer.k_norm.weight"), ihd);
        }
        if (ix) {
            try p.mxfp8(try at(&buf, pfx, "attn.indexer.wq_b"), &.{ih * ihd}, ql);
            try p.dense(try at(&buf, pfx, "attn.indexer.weights_proj.weight"), &.{ ih, d }, 0.09, 0, true);
        }
        try p.dense(try at(&buf, pfx, "ffn.gate.weight"), &.{ e, d }, 0.09, 0, true);
        try p.dense(try at(&buf, pfx, "ffn.gate.bias"), &.{e}, 0.3, 0, false);
        try p.mxfp8(try at(&buf, pfx, "ffn.shared_experts.w1"), &.{inter}, d);
        try p.mxfp8(try at(&buf, pfx, "ffn.shared_experts.w3"), &.{inter}, d);
        try p.mxfp8(try at(&buf, pfx, "ffn.shared_experts.w2"), &.{d}, inter);
        // The routed experts as mlx-lm stacks them, every expert's rows in one tensor: affine 6-bit at group 64, which the
        // in-memory decode's fused affine kernel (2, 3, 4 and 8 bits, `ffnDecode`, no streamed twin) declines.
        try p.affine(try at(&buf, pfx, "ffn.experts.w1"), &.{ e, inter }, d, 6, 64);
        try p.affine(try at(&buf, pfx, "ffn.experts.w3"), &.{ e, inter }, d, 6, 64);
        try p.affine(try at(&buf, pfx, "ffn.experts.w2"), &.{ e, d }, inter, 6, 64);
    }
    return p.bytes();
}

/// The resident module itself (not the served table) at the numerics `opts` names, its routed experts from `streamed`
/// when set.
const Numerics = struct { act: mlx.mlx_dtype, qat: bool, head_8bit: bool, streamed_experts: bool = false };

const Resident = struct {
    cfg: @import("config.zig").Config,
    w: model.Weights,
    s: mlx.mlx_stream,
    m: *dsv41.Dsv41Model,

    fn open(dir: []const u8, opts: Numerics) !*Resident {
        const gpa = testing.allocator;
        const r = try gpa.create(Resident);
        errdefer gpa.destroy(r);
        r.cfg = try @import("../../sdk/testing/pack_config.zig").packConfig(@import("config.zig"), testing.io, dir);
        r.w = try dsv41.loadPack(testing.io, gpa, dir, false);
        errdefer r.w.deinit();
        r.s = mlx.gpuStream();
        errdefer _ = mlx.mlx_stream_free(r.s);
        r.m = try dsv41.init(gpa, &r.cfg, &r.w, r.s, .{ .act = opts.act, .head_8bit = opts.head_8bit, .streamed_experts = opts.streamed_experts, .pack_dir = dir });
        r.m.qat_off = !opts.qat;
        return r;
    }

    fn close(r: *Resident) void {
        r.m.deinit();
        _ = mlx.mlx_stream_free(r.s);
        r.w.deinit();
        testing.allocator.destroy(r);
    }

    /// `ids` in `calls` from a fresh state: each call's last row of logits, host f32. A streamed source settles after
    /// each call's eval.
    fn logits(r: *Resident, a: std.mem.Allocator, ids: []const u32, calls: []const u32, vocab: usize) ![]f32 {
        const gpa = testing.allocator;
        var st = try dsv41.initState(r.m, gpa);
        defer st.deinit();
        const out = try a.alloc(f32, calls.len * vocab);
        var at: usize = 0;
        for (calls, 0..) |n, i| {
            const l = try dsv41.extend(r.m, gpa, &st, ids[at..][0..n], .last);
            defer _ = mlx.mlx_array_free(l);
            var f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f);
            try mlx.check(mlx.mlx_astype(&f, l, .float32, r.s));
            try mlx.check(mlx.mlx_array_eval(f));
            if (r.m.streamed) |src| try src.settle();
            @memcpy(out[i * vocab ..][0..vocab], (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..vocab]);
            at += n;
        }
        return out;
    }
};

// ── tests ──

/// The index of `row`'s highest value.
fn topOf(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}

test "deepseek_v41 streamed: the trunk with its experts streamed from the pack (stream_experts) is bit-equal to its in-memory banks" {
    if (!streamed_w.built or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    // 8 experts: the prompt's gathers take the sorted row-block kernels (60 pairs, 7 a row); 32: the per-pair ones.
    try streamedParity(8);
    try streamedParity(32);
}

fn streamedParity(n_experts: u32) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = try std.mem.replaceOwned(u8, a, config_json, "\"swiglu_limit\": 1.0", "\"swiglu_limit\": 10.0");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = try std.mem.replaceOwned(u8, a, cfg, "\"n_routed_experts\": 8,", try std.fmt.allocPrint(a, "\"n_routed_experts\": {d},", .{n_experts})) });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = try repackWeights(a, 41, tiny, n_experts) });
    var buf: [512]u8 = undefined;
    const dir = try a.dupe(u8, buf[0..try tmp.dir.realPath(io, &buf)]);
    const vocab = 96;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    // f32 activations and the served bf16 ones (QAT on, the 8-bit head).
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |act| {
        const resident = try Resident.open(dir, .{ .act = act, .qat = true, .head_8bit = true });
        defer resident.close();

        // The module as served (its in-memory banks): the prompt (the wide MoE) and 6 greedy decodes (the compiled
        // stretches); then the same calls eager.
        const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90, 33, 5, 71, 18, 44, 6, 29, 50, 12 };
        var ids: std.ArrayList(u32) = .empty;
        try ids.appendSlice(a, &prompt);
        var calls: [7]u32 = @splat(1);
        calls[0] = prompt.len;
        for (1..calls.len) |i| {
            const rows = try resident.logits(a, ids.items, calls[0..i], vocab);
            try ids.append(a, topOf(rows[(i - 1) * vocab ..][0..vocab]));
        }
        const native = try resident.logits(a, ids.items, &calls, vocab);
        const regions = resident.m.regions;
        resident.m.regions = regions[0..0];
        const native_eager = try resident.logits(a, ids.items, &calls, vocab);
        resident.m.regions = regions;

        // Streamed: every expert's row (all reads land in the prompt rows), then 3 rows a layer (misses through the
        // other banks every call); each with the compiled stretches and without.
        for ([_]?u32{ null, 3 }) |rows| {
            var sw = try streamed_w.Streamed.openPack(testing.allocator, dir, s, rows);
            defer sw.close();
            resident.m.streamed = sw.routedSource();
            defer resident.m.streamed = null;
            const got = try resident.logits(a, ids.items, &calls, vocab);
            resident.m.regions = regions[0..0];
            const got_eager = try resident.logits(a, ids.items, &calls, vocab);
            resident.m.regions = regions;
            const st = sw.src.stats();
            std.debug.print("deepseek_v41 streamed ({d} experts, {t}, {s}): {d} routes, {d} hits, {d} misses, {d} evictions\n", .{ n_experts, act, if (rows == null) "every expert resident" else "3 rows a layer", st.route_calls, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions });
            try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(native)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(got)));
            try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(native_eager)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(got_eager)));
            if (rows != null) try testing.expect(st.expert_cache_evictions > 0);
        }
    }
}
