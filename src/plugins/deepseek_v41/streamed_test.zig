//! The trunk's routed experts read from a streamed bank, over tiny packs in OpensourceWTF's repack layout (MXFP8
//! linears, BF16 embedding / head / compressor / indexer keys, F32 hyper-connections, no Engram, no DSpark): an MXFP4
//! `experts.bin` through `Streamed` against the same records held in memory, and the served EXL3 pack (`experts.bin` +
//! its v2 manifest) through the plugin's load against the trunk over the same bank at fewer slot rows.

const std = @import("std");
const build_options = @import("build_options");
const mlx = @import("../../mlx/ffi.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const oracle = @import("../../sdk/weights/streamed/dupes_test.zig");
const model = @import("../../sdk/config/model.zig");
const dsv41 = @import("deepseek_v41.zig");
const streamed_w = @import("../../sdk/weights/streamed.zig");
const stream = if (build_options.mlx_stream) @import("mlx_stream") else struct {};

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

/// A tiny pack's widths and layer table (`config_json`'s, or `exl3_config_json`'s).
const Geo = struct { d: usize, inter: usize, e: usize, ratios: []const u8, kv_sources: []const usize, index_sources: []const usize };
const tiny: Geo = .{ .d = 128, .inter = 128, .e = 8, .ratios = &.{ 0, 0, 2, 2, 2, 1, 1, 1 }, .kv_sources = &.{ 2, 5 }, .index_sources = &.{ 2, 5, 7 } };
/// The EXL3 lane's widths (its texts are compiled for the release's 5120 / 2304).
const exl3_geo: Geo = .{ .d = 5120, .inter = 2304, .e = 8, .ratios = &.{ 0, 2, 2, 1 }, .kv_sources = &.{ 1, 3 }, .index_sources = &.{ 1, 3 } };

/// Where a pack's routed experts are: MXFP4 stacked from `Bank`'s records, or none (a streamed bank beside the shards).
const Routed = union(enum) { mxfp4: *const Bank, streamed };

/// The tiny pack's tensors (the geometry of `fixture.v41Weights`, in the repack layout).
fn repackWeights(a: std.mem.Allocator, seed: u64, geo: Geo, experts: Routed) ![]u8 {
    const d = geo.d;
    const hd = 64;
    const ql = 64;
    const h = 4;
    const og = 2;
    const ol = 32;
    const inter = geo.inter;
    const e: usize = switch (experts) {
        .mxfp4 => |r| r.geo.num_experts,
        else => geo.e,
    };
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
        switch (experts) {
            .mxfp4 => |r| try r.stacked(&p, pfx, @intCast(li)),
            .streamed => {},
        }
    }
    return p.bytes();
}

/// The routed experts as MXFP4 records in the streamed bank's layout (`streamed.Geometry`), every layer's experts in
/// order: random E2M1 words, E8M0 scales of 2^-4 and 2^-3.
const Bank = struct {
    geo: Geometry,
    bytes: []u8,

    const Geometry = if (build_options.mlx_stream) stream.bank_source.Geometry else void;

    fn init(a: std.mem.Allocator, seed: u64, n_experts: u32) !Bank {
        const geo = try Geometry.init(128, 128, n_experts, 0, 8);
        const bytes = try a.alloc(u8, geo.total_bytes);
        var rnd = std.Random.DefaultPrng.init(seed);
        for (0..geo.num_moe_layers) |l| for (0..geo.num_experts) |e| for (geo.segments) |seg| {
            const at = bytes[geo.recordOffset(@intCast(l), @intCast(e)) + seg.offset ..][0..seg.length];
            for (at) |*b| b.* = if (seg.dtype == .U32) rnd.random().int(u8) else @as(u8, 123) + rnd.random().int(u1);
        };
        return .{ .geo = geo, .bytes = bytes };
    }

    /// Layer `l`'s banks as the resident reads them: `ffn.experts.{gate,up,down}_proj` stacked over the experts.
    fn stacked(b: *const Bank, p: *fixture.Pack, pfx: []const u8, l: u32) !void {
        const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
        for (projs, 0..) |proj, pi| for ([_][]const u8{ "weight", "scales" }, 0..) |part, ci| {
            const seg = b.geo.segments[pi * 2 + ci];
            const data = try p.a.alloc(u8, b.geo.num_experts * seg.length);
            for (0..b.geo.num_experts) |e| @memcpy(data[e * seg.length ..][0..seg.length], b.bytes[b.geo.recordOffset(l, @intCast(e)) + seg.offset ..][0..seg.length]);
            var buf: [96]u8 = undefined;
            try p.put(try std.fmt.bufPrint(&buf, "{s}.ffn.experts.{s}.{s}", .{ pfx, proj, part }), @tagName(seg.dtype), &.{ b.geo.num_experts, seg.shape[0], seg.shape[1] }, data);
        };
    }
};

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

test "deepseek_v41 streamed: the trunk with its experts streamed from an MXFP4 experts.bin is bit-equal to its in-memory banks" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
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
    const bank = try Bank.init(a, 5, n_experts);
    // The same records twice: stacked MXFP4 banks in the pack, and the streamed bank in its own directory.
    const cfg = try std.mem.replaceOwned(u8, a, config_json, "\"swiglu_limit\": 1.0", "\"swiglu_limit\": 10.0");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = try std.mem.replaceOwned(u8, a, cfg, "\"n_routed_experts\": 8,", try std.fmt.allocPrint(a, "\"n_routed_experts\": {d},", .{n_experts})) });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = try repackWeights(a, 41, tiny, .{ .mxfp4 = &bank }) });
    try tmp.dir.createDir(io, "bank", .default_dir);
    var bank_dir = try tmp.dir.openDir(io, "bank", .{});
    defer bank_dir.close(io);
    try stream.bank_source.writeBank(a, io, bank_dir, bank.geo, bank.bytes, "synthetic/dsv41-tiny", "0000000000000000000000000000000000000000");
    var buf: [512]u8 = undefined;
    const dir = try a.dupe(u8, buf[0..try tmp.dir.realPath(io, &buf)]);
    const bank_path = try std.fmt.allocPrint(a, "{s}/bank", .{dir});
    const vocab = 96;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const few: [8]u32 = @splat(3);
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
            try ids.append(a, oracle.topOf(rows[(i - 1) * vocab ..][0..vocab])[0][0]);
        }
        const native = try resident.logits(a, ids.items, &calls, vocab);
        const regions = resident.m.regions;
        resident.m.regions = regions[0..0];
        const native_eager = try resident.logits(a, ids.items, &calls, vocab);
        resident.m.regions = regions;

        // Streamed: every expert's row (all reads land in the prompt rows), then 3 rows a layer (misses through the
        // other banks every call); each with the compiled stretches and without.
        for ([_]?[]const u32{ null, &few }) |rows| {
            var sw = try streamed_w.Streamed.open(testing.allocator, io, bank_path, s, .{ .rows = rows });
            defer sw.close();
            resident.m.streamed = sw.routedSource();
            defer resident.m.streamed = null;
            const got = try resident.logits(a, ids.items, &calls, vocab);
            resident.m.regions = regions[0..0];
            const got_eager = try resident.logits(a, ids.items, &calls, vocab);
            resident.m.regions = regions;
            const st = sw.src.mxfp4.stats();
            std.debug.print("deepseek_v41 streamed ({d} experts, {t}, {s}): {d} routes, {d} hits, {d} misses, {d} evictions\n", .{ n_experts, act, if (rows == null) "every expert resident" else "3 rows a layer", st.route_calls, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions });
            try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(native)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(got)));
            try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(native_eager)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(got_eager)));
            if (rows != null) try testing.expect(st.expert_cache_evictions > 0);
        }
    }
}

/// The EXL3 tiny pack's config: `config_json` at the EXL3 lane's widths, four layers (the stream's lookahead reads 8 experts).
const exl3_config_json =
    \\{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "bos_token_id": 0, "eos_token_id": 1,
    \\ "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 96, "hidden_size": 5120,
    \\  "moe_intermediate_size": 2304, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 1,
    \\  "head_dim": 64, "qk_rope_head_dim": 16, "q_lora_rank": 64, "o_lora_rank": 32, "o_groups": 2, "hidden_act": "silu", "swiglu_limit": 10.0,
    \\  "rms_norm_eps": 1e-6, "attention_bias": false, "tie_word_embeddings": false,
    \\  "max_position_embeddings": 1024, "rope_theta": 10000.0,
    \\  "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 256},
    \\  "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 3, "scoring_func": "sqrtsoftplus",
    \\  "topk_method": "noaux_tc", "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 8,
    \\  "compress_ratios": [0, 2, 2, 1], "compress_rope_theta": 40000.0, "kv_source_layer_ids": [1, 3],
    \\  "index_source_layer_ids": [1, 3], "index_n_heads": 8, "index_head_dim": 32, "index_topk": 4,
    \\  "candidate_source_layer_id": 3, "candidate_topk_blocks": 3, "candidate_block_size": 2, "hc_mult": 4,
    \\  "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "engram_layer_ids": [], "engram_num_embeddings": [],
    \\  "engram_max_ngram_size": 2, "engram_vocab_size": 50, "engram_n_heads": 1, "engram_head_dim": 64,
    \\  "engram_pad_token_id": 2, "engram_compressed_vocab_size": 96, "num_nextn_predict_layers": 0}}
;

/// The EXL3 stream kernels' plans fix threadgroups of up to 1024 threads, more than M1/M2 grant some of them.
fn skipOnThreadCap(e: anyerror) anyerror {
    return if (e == error.MlxError and mlx.takeErrorIf("Thread group size")) error.SkipZigTest else e;
}

test "deepseek_v41 streamed exl3: a served EXL3 pack (its prompt in wide calls, a phase change to decode) gives the trunk's logits over the same bank at fewer slot rows" {
    // The repack's prompt GEMMs need NAX: below the M5 the load refuses it by name.
    if (!build_options.mlx_stream or mlx.noGpuBackend() or !@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    _ = try streamed_w.Exl3.writeSynth(a, &tmp, .{ .hidden = 5120, .inter = 2304, .n_experts = 8, .k = &.{ 3, 2, 3, 3 }, .finite_scales = true });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = exl3_config_json });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = try repackWeights(a, 43, exl3_geo, .streamed) });
    var buf: [512]u8 = undefined;
    const dir = try a.dupe(u8, buf[0..try tmp.dir.realPath(io, &buf)]);
    const vocab = 96;
    // Two 256-row spans (the wide layer, its experts read ahead), then greedy decodes.
    const prompt = try a.alloc(u32, 300);
    var rng = std.Random.DefaultPrng.init(7);
    for (prompt) |*t| t.* = 3 + rng.random().uintLessThan(u32, vocab - 3);
    const decode = 3;

    // As served: the plugin's load opens the bank at the rows the box holds (every expert here).
    var ids: std.ArrayList(u32) = .empty;
    try ids.appendSlice(a, prompt);
    const want = try a.alloc(f32, (1 + decode) * vocab);
    {
        const s = mlx.gpuStream();
        // A box without the free memory for the bank's slot rows refuses the load (16 GB).
        const p = oracle.Served.open(io, testing.allocator, dir, s) catch |e| return if (e == error.InsufficientMemory) error.SkipZigTest else e;
        defer p.close(testing.allocator);
        try testing.expectEqualStrings("deepseek_v41", p.implName());
        var row = oracle.forwardIds(&p.xfm, a, prompt, s) catch |e| return skipOnThreadCap(e);
        for (0..1 + decode) |i| {
            @memcpy(want[i * vocab ..][0..vocab], row);
            for (row) |v| try testing.expect(std.math.isFinite(v));
            if (i == decode) break;
            const next = oracle.topOf(row)[0][0];
            try ids.append(a, next);
            row = oracle.forwardIds(&p.xfm, a, &.{next}, s) catch |e| return skipOnThreadCap(e);
        }
    }
    // The trunk over the same bank at 2 prompt rows and 3 decode rows a layer: misses through the transient bank, the
    // phase change growing the rows.
    const r = try Resident.open(dir, .{ .act = .bfloat16, .qat = true, .head_8bit = dsv41.head8Wanted(), .streamed_experts = true });
    defer r.close();
    const Exl3 = streamed_w.Exl3;
    const c = &r.cfg;
    const geo: Exl3.Geometry = .{ .hidden_size = c.hidden_size, .moe_intermediate_size = c.moe_intermediate_size, .n_experts_per_tok = c.num_experts_per_tok, .n_routed_experts = c.num_experts, .n_layers = @intCast(r.m.n_layers), .swiglu_limit = c.dsv4_swiglu_limit };
    const z = try Exl3.billOf(testing.allocator, io, dir, geo, .{ .prompt = std.math.maxInt(u64) / 2, .decode = std.math.maxInt(u64) / 2 });
    var gates: [4]Exl3.Gate = undefined;
    for (&gates, r.m.layers[0..4]) |*g, *ly| g.* = .{ .w = ly.gate_w, .bias = ly.gate_bias };
    const src = try Exl3.Source.open(testing.allocator, io, dir, r.s, geo, &gates, .{ .prompt = z.prompt_windows + z.staging + 2 * z.row_bytes, .decode = z.decode_windows + z.staging + 3 * z.row_bytes });
    defer src.deinit();
    try testing.expectEqual(@as(u32, 2), src.bill.rows.prompt);
    try testing.expectEqual(@as(u32, 3), src.bill.rows.decode);
    r.m.streamed = streamed_w.exl3Routed(src);
    defer r.m.streamed = null;
    var calls: [1 + decode]u32 = @splat(1);
    calls[0] = @intCast(prompt.len);
    const got = r.logits(a, ids.items, &calls, vocab) catch |e| return skipOnThreadCap(e);
    try testing.expectEqual(Exl3.Phase.decode, src.phase);
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(want)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(got)));
}
