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
const Geo = struct { d: usize, inter: usize, e: usize, ratios: []const u8, kv_sources: []const usize, index_sources: []const usize, engram: bool = false };
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
    if (geo.engram) {
        try p.dense("layers.1.engram.q_weight", &.{ 4, d }, 0.1, 0.9, true);
        try p.dense("layers.1.engram.k_weight", &.{ 4, d }, 0.1, 0.9, true);
        try p.mxfp8("layers.1.engram.wkv", &.{5 * d}, 64);
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

fn requireM2Readiness() !void {
    if (!@import("../../env.zig").on(.DSV41_M2_READINESS)) return error.SkipZigTest;
    if (!build_options.mlx_stream) return error.ReadinessNeedsMlxStream;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.ReadinessNeedsMetal;
    const device = @import("../../mlx/device.zig");
    var buf: [128]u8 = undefined;
    const arch = device.gpuArchitecture(&buf) orelse return error.ReadinessDeviceUnknown;
    const gen = device.naxArchGeneration(arch);
    if (gen.gen != 14 or gen.phone or device.device_gen_override != null) return error.ReadinessNeedsActualM2;
    try testing.expect(!device.naxAvailable());
    mlx.installErrorHandler();
    std.debug.print("M2 EXL3 readiness: device={s}, NAX=false\n", .{arch});
}

fn writeReadinessPack(tmp: *testing.TmpDir, path: []u8) ![]const u8 {
    return writeReadinessPackWithEngram(tmp, path, false);
}

fn writeReadinessPackWithEngram(tmp: *testing.TmpDir, path: []u8, with_engram: bool) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try streamed_w.Exl3.writeSynth(a, tmp, .{ .hidden = 5120, .inter = 2304, .n_experts = 8, .k = &.{ 3, 2, 3, 3 }, .finite_scales = true });
    const cfg = if (with_engram) try std.mem.replaceOwned(u8, a, exl3_config_json, "\"engram_layer_ids\": [], \"engram_num_embeddings\": []", "\"engram_layer_ids\": [1], \"engram_num_embeddings\": [53]") else exl3_config_json;
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = cfg });
    var geo = exl3_geo;
    geo.engram = with_engram;
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors", .data = try repackWeights(a, 43, geo, .streamed) });
    if (with_engram) {
        try tmp.dir.createDirPath(testing.io, "engram");
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "engram/engram-manifest.json", .data = "{\"layers\":[{\"layer_id\":1,\"file\":\"engram-L1.bin\",\"rows\":53,\"record_bytes\":66," ++
            "\"quant\":{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\",\"head_dim\":64}}]}" });
        var records: [53 * 66]u8 = undefined;
        for (0..53) |row| {
            for (0..64) |col| records[row * 66 + col] = @intCast(32 + (row * 7 + col * 3) % 24);
            @memset(records[row * 66 + 64 ..][0..2], 123);
        }
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "engram/engram-L1.bin", .data = &records });
        var token_map: [96 * 4]u8 = undefined;
        for (0..96) |i| std.mem.writeInt(u32, token_map[i * 4 ..][0..4], @intCast((i * 5) % 96), .little);
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "engram-token-map.u32", .data = &token_map });
    }
    return path[0..try tmp.dir.realPath(testing.io, path)];
}

const readiness_calls = [_]u32{ 300, 1, 1, 300, 1, 1 };
const readiness_tokens = 604;
const readiness_vocab = 96;

fn readinessIds() [readiness_tokens]u32 {
    var ids: [readiness_tokens]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(7);
    for (&ids) |*id| id.* = 3 + rng.random().uintLessThan(u32, readiness_vocab - 3);
    return ids;
}

fn expectReadinessProbes(m: *const dsv41.Dsv41Model) !void {
    try testing.expect(m.dec.topk != null);
    try testing.expect(m.dec.moe_down != null and m.dec.moe_down_rb > 0);
    try testing.expect(m.dec.attn == null);
    std.debug.print("M2 EXL3 readiness: load probes completed, topk threads={d}, moe-down rows={d}\n", .{ m.dec.topk.?.tg, m.dec.moe_down_rb });
}

fn readReadinessRow(logits: mlx.mlx_array, dst: []f32) !void {
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(logits));
    try testing.expectEqual(dst.len, mlx.mlx_array_size(logits));
    try mlx.check(mlx.mlx_array_eval(logits));
    const row = (mlx.mlx_array_data_float32(logits) orelse return error.NoData)[0..dst.len];
    for (row) |v| try testing.expect(std.math.isFinite(v));
    @memcpy(dst, row);
}

/// Evaluate before a phase may free the bank arrays referenced by the previous call.
fn readinessInternalRows(r: *Resident, src: *streamed_w.Exl3.Source, ids: []const u32, out: []f32) !void {
    var st = try dsv41.initState(r.m, testing.allocator);
    defer st.deinit();
    var at: usize = 0;
    for (readiness_calls, 0..) |n, i| {
        const logits = try dsv41.extend(r.m, testing.allocator, &st, ids[at..][0..n], .last);
        defer _ = mlx.mlx_array_free(logits);
        try readReadinessRow(logits, out[i * readiness_vocab ..][0..readiness_vocab]);
        try src.settle();
        at += n;
        try testing.expectEqual(at, st.n);
        try testing.expectEqual(if (n > dsv41.region_rows) streamed_w.Exl3.Phase.prompt else .decode, src.phase);
    }
    try testing.expectEqual(ids.len, at);
}

const ReadinessRows = struct { prompt: u32, decode: u32 };

fn readinessAtRows(dir: []const u8, ids: []const u32, rows: ReadinessRows, out: []f32) !void {
    const Exl3 = streamed_w.Exl3;
    var replay: [readiness_calls.len * readiness_vocab]f32 = undefined;
    const r = try Resident.open(dir, .{ .act = .bfloat16, .qat = true, .head_8bit = dsv41.head8Wanted(), .streamed_experts = true });
    defer r.close();
    try expectReadinessProbes(r.m);
    const c = &r.cfg;
    const geo: Exl3.Geometry = .{ .hidden_size = c.hidden_size, .moe_intermediate_size = c.moe_intermediate_size, .n_experts_per_tok = c.num_experts_per_tok, .n_routed_experts = c.num_experts, .n_layers = @intCast(r.m.n_layers), .swiglu_limit = c.dsv4_swiglu_limit };
    const z = try Exl3.billOf(testing.allocator, testing.io, dir, geo, .{ .prompt = std.math.maxInt(u64) / 2, .decode = std.math.maxInt(u64) / 2 });
    var gates: [4]Exl3.Gate = undefined;
    for (&gates, r.m.layers[0..4]) |*g, *ly| g.* = .{ .w = ly.gate_w, .bias = ly.gate_bias };
    const src = try Exl3.Source.open(testing.allocator, testing.io, dir, r.s, geo, &gates, .{
        .prompt = z.prompt_windows + z.staging + rows.prompt * z.row_bytes,
        .decode = z.decode_windows + z.staging + rows.decode * z.row_bytes,
    });
    defer src.deinit();
    try testing.expectEqual(rows.prompt, src.bill.rows.prompt);
    try testing.expectEqual(rows.decode, src.bill.rows.decode);
    r.m.streamed = streamed_w.exl3Routed(src);
    defer r.m.streamed = null;
    try readinessInternalRows(r, src, ids, out);
    try readinessInternalRows(r, src, ids, &replay);
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(out)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&replay)));
    const stats = src.stats();
    try testing.expect(stats.route_calls > 0 and stats.expert_cache_misses > 0);
    if (rows.prompt < r.cfg.num_experts_per_tok) try testing.expect(stats.transient_loads > 0);
    std.debug.print("M2 EXL3 readiness: internal {d}/{d} bank rows, {d} calls plus replay, misses={d}, transient loads={d}\n", .{ rows.prompt, rows.decode, readiness_calls.len, stats.expert_cache_misses, stats.transient_loads });
}

test "deepseek_v41 M2 readiness: internal EXL3 load probes, phase reversals and continuation at different bank capacities" {
    if (comptime !build_options.mlx_stream) return requireM2Readiness();
    try requireM2Readiness();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path: [512]u8 = undefined;
    const dir = try writeReadinessPack(&tmp, &path);
    const ids = readinessIds();
    var want: [readiness_calls.len * readiness_vocab]f32 = undefined;
    var got: @TypeOf(want) = undefined;
    try readinessAtRows(dir, &ids, .{ .prompt = 8, .decode = 8 }, &want);
    try readinessAtRows(dir, &ids, .{ .prompt = 2, .decode = 3 }, &got);
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&want)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&got)));
}

fn readinessServedRows(dir: []const u8, ids: []const u32, out: []f32) !void {
    const abi = @import("../../abi/plugin.zig");
    const plugin = @import("plugin.zig");
    const FsPackDir = @import("../../sdk/util/pack_dir.zig").FsPackDir;
    var fs = try FsPackDir.open(testing.io, dir);
    defer fs.close();
    const pd = fs.packDir();
    var parsed: abi.Parsed = .{};
    try testing.expect(plugin.table.parse(&pd, .of(exl3_config_json), .{}, null, false, &parsed).isOk());
    defer plugin.table.free_parsed(&parsed);
    try testing.expect(plugin.config.of(&parsed).dsv41_exl3_bank);
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var verdict: abi.Verdict = .{};
    const mp = plugin.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &parsed, .settings_json = .of("{\"ctx_size\":1024}"), .stream = .{ .ctx = s.ctx } }, &verdict) orelse {
        std.debug.print("M2 EXL3 served readiness blocked: {s}\n", .{verdict.why.slice()});
        return error.M2ServedExl3Refused;
    };
    defer plugin.table.unload(mp);
    const m: *plugin.Model = @ptrCast(@alignCast(mp));
    try expectReadinessProbes(m.mdl);
    try testing.expectEqual(@as(u64, 1024), m.bank_context);
    const src = (m.bank orelse return error.ReadinessBankMissing).exl3;
    var at: u32 = 0;
    for (readiness_calls, 0..) |n, i| {
        var result: abi.ForwardOut = .{};
        const v = plugin.table.forward(mp, &.{ .mode = if (n > 1) .prefill else .decode, .rows = n, .tokens = .of(ids[at..][0..n]), .slots = .of(&.{1}), .positions = .of(&.{at}) }, &result);
        if (!v.isOk()) {
            std.debug.print("M2 EXL3 served readiness forward failed: {s}\n", .{v.why.slice()});
            return error.M2ServedExl3ForwardFailed;
        }
        const logits: mlx.mlx_array = .{ .ctx = result.logits.ctx };
        defer _ = mlx.mlx_array_free(logits);
        try readReadinessRow(logits, out[i * readiness_vocab ..][0..readiness_vocab]);
        try src.settle();
        at += n;
        try testing.expectEqual(if (n > dsv41.region_rows) streamed_w.Exl3.Phase.prompt else .decode, src.phase);
    }
    try testing.expectEqual(@as(u32, readiness_tokens), at);
    plugin.table.release_slot.?(mp, 1);
}

test "deepseek_v41 M2 readiness: served EXL3 plugin accepts the real device and preserves continuation through phase reversals" {
    if (comptime !build_options.mlx_stream) return requireM2Readiness();
    try requireM2Readiness();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path: [512]u8 = undefined;
    const dir = try writeReadinessPack(&tmp, &path);
    const ids = readinessIds();
    var served: [readiness_calls.len * readiness_vocab]f32 = undefined;
    var internal: @TypeOf(served) = undefined;
    try readinessServedRows(dir, &ids, &served);
    try readinessAtRows(dir, &ids, .{ .prompt = 2, .decode = 3 }, &internal);
    try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&internal)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&served)));
}

const SlotReadiness = struct {
    const abi = @import("../../abi/plugin.zig");
    const plugin = @import("plugin.zig");
    parsed: abi.Parsed,
    mp: *anyopaque,
    stream: mlx.mlx_stream,

    fn open(dir: []const u8) !SlotReadiness {
        return openWithCache(dir, 0);
    }

    fn openWithCache(dir: []const u8, cache_bytes: u64) !SlotReadiness {
        const FsPackDir = @import("../../sdk/util/pack_dir.zig").FsPackDir;
        var fs = try FsPackDir.open(testing.io, dir);
        defer fs.close();
        const pd = fs.packDir();
        var parsed: abi.Parsed = .{};
        const config_bytes = try fs.root.readFileAlloc(testing.io, "config.json", testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(config_bytes);
        try testing.expect(plugin.table.parse(&pd, .of(config_bytes), .{}, null, false, &parsed).isOk());
        errdefer plugin.table.free_parsed(&parsed);
        try testing.expect(!parsed.caps.slot_state and parsed.caps.whole_prompt_prefill);
        try testing.expectEqual(@as(u8, 0), parsed.caps.active_slots);
        var runtime_caps = parsed.caps;
        var verdict: abi.Verdict = .{};
        const s = mlx.gpuStream();
        errdefer _ = mlx.mlx_stream_free(s);
        const mp = plugin.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &parsed, .runtime_caps = &runtime_caps, .aux_cache_budget_bytes = cache_bytes, .settings_json = .of("{\"ctx_size\":1024}"), .stream = .{ .ctx = s.ctx } }, &verdict) orelse return error.M2ServedExl3Refused;
        errdefer plugin.table.unload(mp);
        try testing.expect(runtime_caps.slot_state and !runtime_caps.batches_decode and !runtime_caps.whole_prompt_prefill);
        try testing.expectEqual(@as(u8, 2), runtime_caps.active_slots);
        const loaded: *plugin.Model = @ptrCast(@alignCast(mp));
        try testing.expectEqual(@as(u64, @bitCast(runtime_caps)), @as(u64, @bitCast(loaded.caps)));
        parsed.caps = runtime_caps;
        return .{ .parsed = parsed, .mp = mp, .stream = s };
    }

    fn close(self: *SlotReadiness) void {
        plugin.table.unload(self.mp);
        _ = mlx.mlx_stream_free(self.stream);
        plugin.table.free_parsed(&self.parsed);
    }

    fn model(self: *SlotReadiness) *plugin.Model {
        return @ptrCast(@alignCast(self.mp));
    }

    fn step(self: *SlotReadiness, slot: u32, ids: []const u32, at: u32, n: u32, capture: bool) !mlx.mlx_array {
        const start: u32 = if (capture and n > 1)
            @intCast(plugin.resumePrompt(self.mp, slot, .of(ids[0 .. at + n]), .{}))
        else
            at;
        var result: abi.ForwardOut = .{};
        const count = at + n - start;
        std.debug.print("two-slot step slot={d} at={d} n={d} capture={} resumed={d} count={d}\n", .{ slot, at, n, capture, start, count });
        const v = plugin.table.forward(self.mp, &.{ .mode = if (count > 1) .prefill else .decode, .rows = count, .tokens = .of(ids[start .. at + n]), .slots = .of(&.{slot}), .positions = .of(&.{start}) }, &result);
        if (!v.isOk()) {
            std.debug.print("two-slot forward: {s}\n", .{v.why.slice()});
            return error.TwoSlotForwardFailed;
        }
        try testing.expectEqual(slot, self.model().owner);
        try testing.expectEqual(@as(usize, at + n), self.model().mdl.dec_state.?.n);
        return .{ .ctx = result.logits.ctx };
    }
};

test "deepseek_v41 M2 readiness: two streamed owners share phase changes, cap the third and retain continuation after release" {
    if (comptime !build_options.mlx_stream) return requireM2Readiness();
    try requireM2Readiness();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = try writeReadinessPack(&tmp, &path_buf);
    const ids_a = readinessIds() ++ [_]u32{12};
    var ids_b = ids_a;
    for (&ids_b) |*id| id.* = 3 + (id.* + 7) % (readiness_vocab - 3);
    const plugin = @import("plugin.zig");
    const abi = @import("../../abi/plugin.zig");
    const bill = @import("bill.zig");
    const calls = readiness_calls ++ [_]u32{1};
    for ([_]bool{ false, true }) |capture| {
        var want: [2][calls.len * readiness_vocab]f32 = undefined;
        {
            var serial = try SlotReadiness.open(dir);
            defer serial.close();
            for ([_][]const u32{ &ids_a, &ids_b }, 0..) |ids, arm| {
                var at: u32 = 0;
                for (calls, 0..) |n, i| {
                    const out = try serial.step(1, ids, at, n, capture);
                    defer _ = mlx.mlx_array_free(out);
                    try readReadinessRow(out, want[arm][i * readiness_vocab ..][0..readiness_vocab]);
                    at += n;
                }
                plugin.table.release_slot.?(serial.mp, 1);
            }
        }
        var interleaved = try SlotReadiness.open(dir);
        defer interleaved.close();
        const m = interleaved.model();
        const src = m.bank.?.exl3;
        try testing.expect(!plugin.table.cache.aux.?.enabled(interleaved.mp));
        var at: u32 = 0;
        for (readiness_calls, 0..) |n, i| {
            const a = try interleaved.step(1, &ids_a, at, n, capture);
            defer _ = mlx.mlx_array_free(a);
            // Leave A's output pending while B takes the shared bank; its bytes must remain A's.
            const b = try interleaved.step(2, &ids_b, at, n, capture);
            defer _ = mlx.mlx_array_free(b);
            for ([_]mlx.mlx_array{ a, b }, 0..) |out, arm| {
                var got: [readiness_vocab]f32 = undefined;
                try readReadinessRow(out, &got);
                try testing.expectEqualSlices(f32, want[arm][i * readiness_vocab ..][0..readiness_vocab], &got);
            }
            try src.settle();
            if (!capture) try testing.expectEqual(if (n > dsv41.region_rows) streamed_w.Exl3.Phase.prompt else .decode, src.phase);
            try testing.expectEqual(@as(usize, 1), m.parked.count());
            try testing.expect(m.bank.?.exl3 == src);
            try testing.expect(dsv41.stateBytes(&m.mdl.dec_state.?) <= bill.streamRetainedBytes(&m.cfg, m.bank_context));
            const refused = plugin.table.activate_slot.?(interleaved.mp, 3);
            try testing.expectEqual(abi.Verdict.Code.too_large, refused.code);
            try testing.expectEqual(@as(u32, 2), m.owner);
            try testing.expectEqual(@as(usize, 1), m.parked.count());
            at += n;
        }
        // Cancellation and normal finish use this same release hook. Its capacity is reusable.
        plugin.table.release_slot.?(interleaved.mp, 1);
        try testing.expect(m.idle != null);
        try testing.expect(plugin.table.activate_slot.?(interleaved.mp, 3).isOk());
        const fresh = try interleaved.step(3, &ids_a, 0, readiness_calls[0], capture);
        defer _ = mlx.mlx_array_free(fresh);
        var row: [readiness_vocab]f32 = undefined;
        try readReadinessRow(fresh, &row);
        try testing.expect(m.idle == null);
        const continued = try interleaved.step(2, &ids_b, at, 1, capture);
        defer _ = mlx.mlx_array_free(continued);
        try readReadinessRow(continued, &row);
        try testing.expectEqualSlices(f32, want[1][readiness_calls.len * readiness_vocab ..][0..readiness_vocab], &row);
        plugin.table.release_slot.?(interleaved.mp, 2);
        plugin.table.release_slot.?(interleaved.mp, 3);
    }
}

test "deepseek_v41 M2 shared cache: released EXL3 snapshots restore through RAM and codec without disturbing another owner" {
    try sharedCacheRoundTrip(false);
}

test "deepseek_v41 M2 shared cache: real Engram history survives RAM and codec restore with another owner" {
    try sharedCacheRoundTrip(true);
}

fn sharedCacheRoundTrip(with_engram: bool) !void {
    if (comptime !build_options.mlx_stream) return requireM2Readiness();
    try requireM2Readiness();
    const abi = @import("../../abi/plugin.zig");
    const plugin = @import("plugin.zig");
    const codec = @import("cache_codec.zig");
    const aux = plugin.table.cache.aux.?;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path: [512]u8 = undefined;
    const dir = try writeReadinessPackWithEngram(&tmp, &path, with_engram);
    const ids_a = readinessIds();
    var ids_b = ids_a;
    for (&ids_b) |*id| id.* = 3 + (id.* + 7) % (readiness_vocab - 3);
    var expected: [2][readiness_vocab]f32 = undefined;
    for ([_][]const u32{ &ids_a, &ids_b }, 0..) |ids, i| {
        var serial = try SlotReadiness.open(dir);
        defer serial.close();
        const prompt = try serial.step(1, ids, 0, 300, false);
        defer _ = mlx.mlx_array_free(prompt);
        var scratch: [readiness_vocab]f32 = undefined;
        try readReadinessRow(prompt, &scratch);
        const next = try serial.step(1, ids, 300, 1, false);
        defer _ = mlx.mlx_array_free(next);
        try readReadinessRow(next, &expected[i]);
    }
    // Capture splits the donor at 299; preserve that partition and the restored forward width in each oracle.
    var restored_expected: [2][readiness_vocab]f32 = undefined;
    var next_expected: [2][readiness_vocab]f32 = undefined;
    for (0..2) |arm| {
        var serial = try SlotReadiness.open(dir);
        defer serial.close();
        const prefix = try serial.step(1, &ids_a, 0, 299, false);
        defer _ = mlx.mlx_array_free(prefix);
        var scratch: [readiness_vocab]f32 = undefined;
        try readReadinessRow(prefix, &scratch);
        if (arm == 1) {
            const tail = try serial.step(1, &ids_a, 299, 1, false);
            defer _ = mlx.mlx_array_free(tail);
            try readReadinessRow(tail, &scratch);
        }
        const suffix = try serial.step(1, &ids_a, if (arm == 0) 299 else 300, if (arm == 0) 2 else 1, false);
        defer _ = mlx.mlx_array_free(suffix);
        try readReadinessRow(suffix, &restored_expected[arm]);
        const next = try serial.step(1, &ids_a, 301, 1, false);
        defer _ = mlx.mlx_array_free(next);
        try readReadinessRow(next, &next_expected[arm]);
        var differing: usize = 0;
        for (expected[0], restored_expected[arm]) |old, matched| if (@as(u32, @bitCast(old)) != @as(u32, @bitCast(matched))) {
            differing += 1;
        };
        std.debug.print("EXL3 cache oracle partition 299+{s}: {d} logits differ from unsplit300+1\n", .{ if (arm == 0) "2" else "1+1", differing });
    }
    for ([_]bool{ false, true }, 0..) |through_codec, arm| {
        var live = try SlotReadiness.openWithCache(dir, 8 << 20);
        defer live.close();
        try testing.expectEqual(@as(u64, 8 << 20), live.model().aux_cache_budget_bytes);
        try testing.expect(aux.enabled(live.mp));
        try testing.expectEqual(@as(usize, if (with_engram) 1 else 0), live.model().mdl.engrams.len);
        try testing.expect(aux.begin_prompt(live.mp, 11, .of(ids_a[0..300]), .{}, 0, true).isOk());
        const first = try live.step(11, &ids_a, 0, 300, false);
        defer _ = mlx.mlx_array_free(first);
        var got: [readiness_vocab]f32 = undefined;
        try readReadinessRow(first, &got);
        var saved: abi.Handle = null;
        try testing.expect(aux.snapshot(live.mp, 11, &saved).isOk());
        defer aux.free(live.mp, saved);
        try testing.expect(saved != null);
        var info: abi.AuxSnapshotInfo = .{};
        try testing.expect(aux.describe(live.mp, saved, &info).isOk());
        try testing.expectEqual(@as(u64, 299), info.n);
        try testing.expectEqualSlices(u32, ids_a[0..299], info.ids.slice());
        try testing.expect(aux.begin_prompt(live.mp, 22, .of(ids_b[0..300]), .{}, 0, false).isOk());
        const neighbor = try live.step(22, &ids_b, 0, 300, false);
        defer _ = mlx.mlx_array_free(neighbor);
        try readReadinessRow(neighbor, &got);
        plugin.table.release_slot.?(live.mp, 11);
        try testing.expect(live.model().idle == null);
        try testing.expect(!live.model().parked.contains(11));
        try testing.expectEqual(abi.Verdict.Code.unsupported, aux.restore_at(live.mp, 22, saved, 298).code);
        try testing.expectEqual(@as(u32, 22), live.model().owner);
        try testing.expectEqual(@as(usize, 300), live.model().mdl.dec_state.?.n);

        var imported: abi.Handle = null;
        defer aux.free(live.mp, imported);
        if (through_codec) {
            var tensors: [codec.max_tensors]abi.AuxTensor = undefined;
            var names: [16 * 1024]u8 = undefined;
            var header: [codec.max_header]u8 = undefined;
            var out: abi.AuxExport = .{ .tensors = &tensors, .tensor_cap = tensors.len, .names = &names, .names_cap = names.len, .header = &header, .header_cap = header.len };
            try testing.expect(aux.export_arrays(live.mp, saved, &out).isOk());
            defer for (tensors[0..out.tensor_len]) |tensor| {
                _ = mlx.mlx_array_free(.{ .ctx = tensor.array.ctx });
            };
            var peak: u64 = 0;
            try testing.expect(aux.import_plan(live.mp, .of(header[0..out.header_len]), &peak).isOk());
            try testing.expect(peak > 0);
            try testing.expect(!aux.import_arrays(live.mp, .of(ids_a[0..299]), .of(header[0 .. out.header_len - 1]), .of(tensors[0..out.tensor_len]), &imported).isOk());
            try testing.expect(imported == null);
            try testing.expectEqual(@as(u32, 22), live.model().owner);
            try testing.expectEqual(@as(usize, 300), live.model().mdl.dec_state.?.n);
            try testing.expect(aux.import_arrays(live.mp, .of(ids_a[0..299]), .of(header[0..out.header_len]), .of(tensors[0..out.tensor_len]), &imported).isOk());
            try testing.expect(imported != null);
            try testing.expect(peak >= aux.bytes(live.mp, imported));
        }
        try testing.expect(aux.restore_at(live.mp, 33, if (through_codec) imported else saved, 299).isOk());
        if (with_engram) {
            const hist = live.model().mdl.dec_state.?.hist.items;
            try testing.expectEqual(@as(usize, 299), hist.len);
            for (hist, ids_a[0..299]) |compressed, id| try testing.expectEqual((id * 5) % 96, compressed);
        }
        try testing.expect(aux.begin_prompt(live.mp, 33, .of(ids_a[0..301]), .{}, 299, false).isOk());
        if (through_codec) {
            const tail = try live.step(33, &ids_a, 299, 1, false);
            defer _ = mlx.mlx_array_free(tail);
            try readReadinessRow(tail, &got);
        }
        const restored = try live.step(33, &ids_a, if (through_codec) 300 else 299, if (through_codec) 1 else 2, false);
        defer _ = mlx.mlx_array_free(restored);
        try readReadinessRow(restored, &got);
        try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&restored_expected[arm])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&got)));
        const next = try live.step(33, &ids_a, 301, 1, false);
        defer _ = mlx.mlx_array_free(next);
        try readReadinessRow(next, &got);
        try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&next_expected[arm])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&got)));
        plugin.table.release_slot.?(live.mp, 33);
        const continued = try live.step(22, &ids_b, 300, 1, false);
        defer _ = mlx.mlx_array_free(continued);
        try readReadinessRow(continued, &got);
        try testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&expected[1])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&got)));
        plugin.table.release_slot.?(live.mp, 22);
        try testing.expect(live.model().mdl.dec_state == null and live.model().idle == null);
        try testing.expectEqual(@as(usize, 0), live.model().parked.count());
    }
}
