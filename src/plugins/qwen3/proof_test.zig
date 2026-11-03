//! Parity proof for the qwen3 plugin on tiny random packs. Dense: served end to end (`parseConfig` hands the pack to
//! the plugin, `driver.load` builds the shell) and every forward the engine makes — prefill, decode, verify, captures,
//! the lazy pipelined decode, batched rows — bit-equal to the legacy `Transformer`'s; the served `ModelConfig` equal
//! to the legacy parse on every generic field. MoE: the table's forward bit-equal to the legacy layered forward.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const discovery = @import("../../host/registry/discovery.zig");
const lazyForward = @import("../../engines/mlx/generate/step.zig").lazyForward;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const twin = @import("twin_test.zig");
const dflash_mod = @import("../../engines/mlx/spec/dflash/dflash.zig");
const pooling = @import("../../engines/mlx/embed/pooling.zig");
const qwen3 = @import("plugin.zig");
const served_test = @import("../qwen3_5/served_test.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// A tiny qwen3 pack's geometry and options.
pub const Geo = struct {
    hidden: u32 = 128,
    heads: u32 = 4,
    kv_heads: u32 = 2,
    head_dim: u32 = 32,
    inter: u32 = 256,
    vocab: u32 = 256,
    layers: u32 = 2,
    experts: u32 = 0,
    quantized: bool = false,
    tied: bool = false,
    /// Qwen3-Embedding's explicit `pooling_mode`.
    last_token_pooling: bool = false,
};

const MOE_INTER = 64;

pub fn configJson(a: std.mem.Allocator, g: Geo) ![]u8 {
    const experts = if (g.experts > 0) try std.fmt.allocPrint(a, "\"num_experts\":{d},\"num_experts_per_tok\":2,\"moe_intermediate_size\":64,\"decoder_sparse_step\":1,\"mlp_only_layers\":[],\"norm_topk_prob\":true,", .{g.experts}) else try a.dupe(u8, "");
    defer a.free(experts);
    const model_type = if (g.experts > 0) "qwen3_moe" else "qwen3";
    const quant = if (g.quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "";
    const pool = if (g.last_token_pooling) ",\"pooling_mode\":\"last_token\"" else "";
    return std.fmt.allocPrint(a, "{{\"model_type\":\"{s}\",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_hidden_layers\":{d}," ++
        "\"num_attention_heads\":{d},\"num_key_value_heads\":{d},\"head_dim\":{d},{s}\"vocab_size\":{d},\"rms_norm_eps\":1e-6," ++
        "\"rope_theta\":1000000.0,\"max_position_embeddings\":40960,\"hidden_act\":\"silu\",\"bos_token_id\":1,\"eos_token_id\":[2,3]," ++
        "\"sliding_window\":null,\"use_sliding_window\":false,\"tie_word_embeddings\":{}{s}{s}}}", .{ model_type, g.hidden, g.inter, g.layers, g.heads, g.kv_heads, g.head_dim, experts, g.vocab, g.tied, quant, pool });
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

pub fn writePack(p: *fixture.Pack, g: Geo) !void {
    var buf: [128]u8 = undefined;
    const h: c_int = @intCast(g.hidden);
    const hd: c_int = @intCast(g.head_dim);
    const q = g.quantized;
    try p.weight("model.embed_tokens", @intCast(g.vocab), h, q);
    try p.norm("model.norm.weight", h);
    if (!g.tied) try p.weight("lm_head", @intCast(g.vocab), h, q);
    for (0..g.layers) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), h);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), h);
        try p.weight(try at(&buf, i, "self_attn.q_proj"), @as(c_int, @intCast(g.heads)) * hd, h, q);
        try p.weight(try at(&buf, i, "self_attn.k_proj"), @as(c_int, @intCast(g.kv_heads)) * hd, h, q);
        try p.weight(try at(&buf, i, "self_attn.v_proj"), @as(c_int, @intCast(g.kv_heads)) * hd, h, q);
        try p.weight(try at(&buf, i, "self_attn.o_proj"), h, @as(c_int, @intCast(g.heads)) * hd, q);
        try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), hd);
        try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), hd);
        if (g.experts > 0) {
            const e: c_int = @intCast(g.experts);
            try p.weight(try at(&buf, i, "mlp.gate"), e, h, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.gate_proj"), &.{ e, MOE_INTER, h }, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.up_proj"), &.{ e, MOE_INTER, h }, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.down_proj"), &.{ e, h, MOE_INTER }, q);
        } else {
            try p.weight(try at(&buf, i, "mlp.gate_proj"), @intCast(g.inter), h, q);
            try p.weight(try at(&buf, i, "mlp.up_proj"), @intCast(g.inter), h, q);
            try p.weight(try at(&buf, i, "mlp.down_proj"), h, @intCast(g.inter), q);
        }
    }
}

// ── dense, served through the engine's plugin path ──

/// The pack served by its plugin, against its tape.
const Both = struct {
    tmp: testing.TmpDir,
    dir: []u8,
    cfg: model.ModelConfig,
    xfm: Transformer,
    layers: u32,
    oracle: served_test.Oracle,
    sides: [1]*Transformer = undefined,

    fn open(b: *Both, g: Geo, comptime seed: u64) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        var p = fixture.Pack.init(seed);
        defer p.deinit();
        try writePack(&p, g);
        const json = try configJson(a, g);
        defer a.free(json);
        b.* = .{ .tmp = testing.tmpDir(.{}), .dir = &.{}, .cfg = undefined, .xfm = undefined, .layers = g.layers, .oracle = try served_test.oracle("qwen3", std.fmt.comptimePrint("both-{x}", .{seed})) };
        errdefer b.tmp.cleanup();
        try b.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
        var path_buf: [512]u8 = undefined;
        b.dir = try a.dupe(u8, path_buf[0..try b.tmp.dir.realPath(io, &path_buf)]);
        errdefer a.free(b.dir);
        try p.save(b.dir);
        b.cfg = try model.parseConfig(io, a, b.dir);
        errdefer b.cfg.deinit(a);
        if (b.cfg.plugin == null) return error.TestExpectedServedPack;
        b.xfm = try driver.load(io, a, &b.cfg, b.dir, fixture.stream());
    }

    fn close(b: *Both) void {
        b.xfm.deinit();
        b.cfg.deinit(testing.allocator);
        testing.allocator.free(b.dir);
        b.tmp.cleanup();
    }

    /// The shell to run (a slice, as the proofs iterate it).
    fn shells(b: *Both) []const *Transformer {
        b.sides = .{&b.xfm};
        return &b.sides;
    }

    /// The shell's value against the tape.
    pub fn check(b: *Both, vals: []const Array) !void {
        try b.oracle.check(vals[vals.len - 1], false);
    }
};

const Slot = struct {
    kv: KVCache,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    fn init(s: *Slot, layers: u32) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, layers) };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    }
    fn deinit(s: *Slot) void {
        s.kv.deinit();
    }
};

const steps = [_]struct { ids: []const i32, verify: bool = false }{
    .{ .ids = &.{ 3, 17, 42, 99, 7, 250, 1 } },
    .{ .ids = &.{88} },
    .{ .ids = &.{ 5, 61, 200 }, .verify = true },
    .{ .ids = &.{12} },
    .{ .ids = &.{ 9, 33, 77, 140, 2 }, .verify = true },
    .{ .ids = &.{64} },
};

/// Prefill, decode and verify steps: logits, the last hidden row and a captured layer bit-equal.
fn expectSteps(b: *Both) !void {
    const xs = b.shells();
    var slots: [2]Slot = undefined;
    for (slots[0..xs.len]) |*s| try s.init(b.layers);
    defer for (slots[0..xs.len]) |*s| s.deinit();
    for (steps) |st| {
        var g = ops.Graph.init(testing.allocator, fixture.stream());
        defer g.deinit();
        const ids = try fixture.tokens(&g, st.ids);
        var outs: [2][3]Array = undefined;
        for (xs, slots[0..xs.len], 0..) |x, *s, k| {
            var hidden = mlx.mlx_array_new();
            var caps: [2]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
            var cl: CaptureLayers = .{ .ids = &.{ 0, 1 }, .out = &caps };
            s.ctx.capture_layers = &cl;
            s.ctx.spec_verify = st.verify;
            defer s.ctx.capture_layers = null;
            const logits = try g.adopt(try x.forwardWithCapture(&s.ctx, ids, &hidden));
            _ = try g.adopt(hidden);
            _ = try g.adopt(caps[0]);
            _ = try g.adopt(caps[1]);
            outs[k] = .{ logits, hidden, caps[1] };
        }
        try served_test.checkRows(b, &outs, xs.len);
    }
}

/// Greedy decode through the pipelined step (`lazyForward`), tokens equal.
fn expectGreedy(b: *Both) !void {
    var tokens: [2][8]u32 = undefined;
    const xs = b.shells();
    for (xs, 0..) |x, k| {
        var s: Slot = undefined;
        try s.init(b.layers);
        defer s.deinit();
        var g = ops.Graph.init(testing.allocator, fixture.stream());
        defer g.deinit();
        var logits = try g.adopt(try x.forwardWith(&s.ctx, try fixture.tokens(&g, &.{ 7, 8, 9, 10 })));
        for (&tokens[k]) |*t| {
            const sh = ops.shape(logits);
            const last = try g.reshape(try g.sliceAxis(logits, 1, sh[1] - 1, sh[1]), &.{ 1, sh[2] });
            const next = try g.argmax(last, -1, false);
            logits = try g.adopt(try lazyForward(x, &s.ctx, next));
            const host = try g.astype(next, .uint32);
            try mlx.check(mlx.mlx_array_eval(host));
            t.* = mlx.mlx_array_data_uint32(host).?[0];
        }
    }
    try b.oracle.stream(&tokens[xs.len - 1]);
}

/// Two slots prefilled apart, then decoded together: each slot's logits bit-equal to the legacy batched tick's.
fn expectBatched(b: *Both) !void {
    const a = testing.allocator;
    const xs = b.shells();
    var slots: [2][2]Slot = undefined;
    for (slots[0..xs.len]) |*pair| for (pair) |*s| try s.init(b.layers);
    defer for (slots[0..xs.len]) |*pair| for (pair) |*s| s.deinit();
    const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23 }, &.{ 42, 101, 7 } };
    var positions: [2]u32 = undefined;
    for (prompts, 0..) |pr, i| {
        var g = ops.Graph.init(a, fixture.stream());
        defer g.deinit();
        for (xs, 0..) |x, k| _ = try g.adopt(try x.forwardWith(&slots[k][i].ctx, try fixture.tokens(&g, pr)));
        positions[i] = @intCast(pr.len);
    }
    for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 } }) |next| {
        var g = ops.Graph.init(a, fixture.stream());
        defer g.deinit();
        var rows: [2][]Array = undefined;
        for (xs, 0..) |x, k| {
            const ctxs = [_]*ForwardCtx{ &slots[k][0].ctx, &slots[k][1].ctx };
            rows[k] = try x.forwardBatchedDecode(&next, &ctxs, &positions);
            for (rows[k]) |r| _ = try g.adopt(r);
        }
        defer for (rows[0..xs.len]) |r| a.free(r);
        for (0..2) |i| {
            const col = [2]Array{ rows[0][i], rows[xs.len - 1][i] };
            try b.check(col[2 - xs.len ..]);
        }
        for (&positions) |*p| p.* += 1;
    }
}

test "qwen3 plugin: a 4-bit pack served by its plugin is bit-equal to the legacy forward (steps, greedy, batched)" {
    var b: Both = undefined;
    try b.open(.{ .quantized = true }, 0x9e31);
    defer b.close();
    try testing.expect(b.xfm.plugin != null);
    try expectSteps(&b);
    try expectGreedy(&b);
    try expectBatched(&b);
    try b.oracle.done();
}

test "qwen3 plugin: a bf16 pack at head width 256 (the fused decode norm+RoPE) with a tied head is bit-equal" {
    var b: Both = undefined;
    try b.open(.{ .head_dim = 256, .tied = true }, 0x9e32);
    defer b.close();
    try expectSteps(&b);
    try expectGreedy(&b);
    try expectBatched(&b);
    try b.oracle.done();
}

/// Fields the forward alone reads: the plugin keeps them, the adapter leaves their defaults.
const plugin_owned = [_][]const u8{ "rms_norm_eps", "rope_theta", "rope_local_base_freq", "rope_llama3", "tie_word_embeddings", "query_pre_attn_scalar", "has_qk_norm", "plugin" };

fn expectGenericFieldsEqual(served: *const model.ModelConfig, legacy: *const model.ModelConfig) !void {
    @setEvalBranchQuota(100_000);
    inline for (@typeInfo(model.ModelConfig).@"struct".field_names) |name| {
        const owned = comptime for (plugin_owned) |n| {
            if (std.mem.eql(u8, n, name)) break true;
        } else false;
        const idle = comptime std.mem.startsWith(u8, name, "sliding_window");
        if (!owned and !(idle and !legacy.has_sliding_window)) {
            const x = @field(served, name);
            const y = @field(legacy, name);
            const same = switch (@typeInfo(@FieldType(model.ModelConfig, name))) {
                .pointer => std.mem.eql(u8, x, y),
                .optional => |o| if (@typeInfo(o.child) == .pointer) (x == null) == (y == null) and (x == null or std.mem.eql(u8, x.?, y.?)) else std.meta.eql(x, y),
                else => std.meta.eql(x, y),
            };
            if (!same) {
                std.debug.print("plugin config: field {s} differs from the legacy parse\n", .{name});
                return error.ConfigFieldDiffers;
            }
        }
    }
    try testing.expectEqual(legacy.contextCap(), served.contextCap());
    try testing.expectEqualSlices(u32, legacy.eosTokenSlice(), served.eosTokenSlice());
}

test "qwen3 plugin: the served ModelConfig equals the legacy parse on every generic field, and discovery's facts too" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    for ([_]Geo{ .{ .quantized = true, .head_dim = 128 }, .{ .head_dim = 256, .tied = true } }) |geo| {
        const json = try configJson(a, geo);
        defer a.free(json);
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
        try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = "{\"temperature\":0.6,\"top_p\":0.9,\"eos_token_id\":[2,9]}" });
        var path_buf: [512]u8 = undefined;
        const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        var served = try model.parseConfig(io, a, dir);
        defer served.deinit(a);
        try testing.expect(served.plugin != null);
        var legacy = try model.parseConfigFromJson(a, json);
        defer legacy.deinit(a);
        legacy.gen_temperature = 0.6;
        legacy.gen_top_p = 0.9;
        legacy.mergeEosTokens(&.{ 2, 9 });
        legacy.applyFamilySamplingDefaults();
        try expectGenericFieldsEqual(&served, &legacy);

        const stub = discovery.readStubMeta(io, a, dir);
        const want = discovery.parseStubMeta(a, json, false);
        inline for (.{ "vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings", "quant_bits", "is_moe", "has_vision", "has_video" }) |f|
            try testing.expectEqual(@field(want, f), @field(stub, f));
    }
}

test "qwen3 plugin: a dense pack named as an embedding model is served, last-token pooled by its folder name; MoE is refused by name" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    for ([_]Geo{ .{ .quantized = true }, .{ .quantized = true, .experts = 4 } }) |geo| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.createDirPath(io, "Qwen3-Embedding-0.6B-4bit");
        const json = try configJson(a, geo);
        defer a.free(json);
        try tmp.dir.writeFile(io, .{ .sub_path = "Qwen3-Embedding-0.6B-4bit/config.json", .data = json });
        var path_buf: [512]u8 = undefined;
        const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        const dir = try std.fmt.allocPrint(a, "{s}/Qwen3-Embedding-0.6B-4bit", .{root});
        defer a.free(dir);
        if (geo.experts > 0) {
            try testing.expectError(error.PluginDeclinedNoLegacy, model.parseConfig(io, a, dir));
            continue;
        }
        var cfg = try model.parseConfig(io, a, dir);
        defer cfg.deinit(a);
        try testing.expectEqual(model.PoolingMode.last_token, cfg.pooling_mode.?);
        try testing.expect(cfg.plugin != null);
    }
}

/// `seqs`' pooled rows from both shells through the engine's embedding batch.
fn expectEmbeddings(b: *Both, seqs: []const []const u32) !void {
    const a = testing.allocator;
    const xs = b.shells();
    var rows: [2][][]f32 = undefined;
    for (rows[0..xs.len], xs) |*r, x| r.* = try pooling.computeEmbeddingsBatchWith(a, x, seqs, null);
    defer for (rows[0..xs.len]) |r| {
        for (r) |row| a.free(row);
        a.free(r);
    };
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    for (0..seqs.len) |i| {
        var col: [2]Array = undefined;
        for (rows[0..xs.len], col[0..xs.len]) |r, *c| c.* = try g.fromSlice(f32, r[i], &.{@intCast(r[i].len)});
        try b.check(col[0..xs.len]);
    }
}

const embed_batches = [_][]const []const u32{
    &.{&.{ 5, 9, 33, 70, 2 }},
    &.{ &.{ 5, 9, 33, 70, 2 }, &.{ 8, 2 }, &.{ 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 3 }, &.{7} },
    &.{ &.{7}, &.{9}, &.{11} },
};

test "qwen3 plugin: a Qwen3-Embedding pack's pooled rows equal the legacy dense forward's (one sequence, padded batches)" {
    inline for (.{ .{ Geo{ .quantized = true, .last_token_pooling = true }, 0x3e01 }, .{ Geo{ .head_dim = 256, .tied = true, .last_token_pooling = true }, 0x3e02 } }) |case| {
        var b: Both = undefined;
        try b.open(case[0], case[1]);
        defer b.close();
        try testing.expect(b.xfm.plugin != null and b.cfg.pooling_mode.? == .last_token);
        for (embed_batches) |seqs| try expectEmbeddings(&b, seqs);
        try b.oracle.done();
    }
}

// ── MoE, through the table ──

const moe_steps = [_]twin.Step{
    .{ .mode = .prefill, .ids = &twin.seq(20, 3) },
    .{ .mode = .prefill, .ids = &twin.seq(70, 11) },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{64} },
    .{ .mode = .prefill, .ids = &twin.seq(6, 5) },
    .{ .mode = .decode, .ids = &.{17} },
};

fn moeCase(g: Geo, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, g);
    const T = twin.Twin(qwen3);
    const json = try configJson(testing.allocator, g);
    defer testing.allocator.free(json);
    const name = std.fmt.comptimePrint("fixtures/twin-{x}.tape", .{seed});
    const t = try T.open(&p, json, &.{}, @embedFile(name));
    defer t.close();
    const l = try t.lane();
    defer l.deinit();
    for (moe_steps) |st| try t.step(l, st);
    try t.expectDone();
}

test "qwen3 plugin: qwen3_moe 4-bit through the table is bit-equal to the legacy layered forward" {
    try moeCase(.{ .experts = 8, .quantized = true }, 0x3e01);
}

test "qwen3 plugin: qwen3_moe bf16 at head width 256 (dense experts, decode side copies) is bit-equal" {
    try moeCase(.{ .experts = 8, .head_dim = 256 }, 0x3e02);
}

// ── a DFlash drafter on the served dense trunk ──

test "qwen3 plugin: a DFlash drafter on a served trunk (dense and MoE) reads its embedding and head and emits the legacy stream" {
    // The tiny assistant's geometry (`dflash.TinyFix`): hidden 64, targets layers 0 and 2, mask id 127.
    try expectDflash(.{ .hidden = 64, .head_dim = 16, .inter = 128, .vocab = 128, .layers = 4, .tied = true }, 0xdf1a6);
    try expectDflash(.{ .hidden = 64, .head_dim = 16, .inter = 128, .vocab = 128, .layers = 4, .tied = true, .experts = 8 }, 0xdf1a7);
}

fn expectDflash(geo: Geo, comptime seed: u64) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var b: Both = undefined;
    try b.open(geo, seed);
    defer b.close();
    try testing.expect(b.xfm.supportsLayerCapture());
    const xs = b.shells();
    {
        var g = ops.Graph.init(a, fixture.stream());
        defer g.deinit();
        const ids = try fixture.tokens(&g, &.{ 4, 90, 127, 0, 33 });
        const hidden = try g.reshape(try g.adopt(try xs[0].rawEmbedding(ids)), &.{ 1, 5, 64 });
        var emb: [2]Array = undefined;
        var head: [2]Array = undefined;
        for (xs, emb[0..xs.len], head[0..xs.len]) |x, *e, *h| {
            e.* = try g.adopt(try x.rawEmbedding(ids));
            h.* = try g.adopt(try x.lmHeadForDraft(hidden));
        }
        try b.check(emb[0..xs.len]);
        try b.check(head[0..xs.len]);
    }
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const asst = buf[0..try tmp.dir.realPath(io, &buf)];
    try dflash_mod.TinyFix.writeAssistant(io, tmp.dir, asst, fixture.stream());
    var tokens: [2][16]u32 = undefined;
    var accepted: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&accepted) |*l| l.deinit(a);
    for (xs, 0..) |x, k| try served_test.dflashRun(x, asst, &tokens[k], &accepted[k]);
    const last = xs.len - 1;
    try b.oracle.stream(&tokens[last]);
    try b.oracle.stream(accepted[last].items);
    try b.oracle.done();
}

test "qwen3 plugin: a qwen3_moe pack through the driver (legacy ModelConfig) equals the legacy shell" {
    const served = @import("../qwen3_5/served_test.zig");
    const g: Geo = .{ .experts = 8, .quantized = true };
    var pk = fixture.Pack.init(0x3e03);
    defer pk.deinit();
    try writePack(&pk, g);
    const json = try configJson(testing.allocator, g);
    defer testing.allocator.free(json);
    var p: served.Pair = undefined;
    try p.open(&qwen3.table, &pk, json, try served.oracle("qwen3", "served-3e03"));
    defer p.close();
    try served.expectSolo(&p);
}
