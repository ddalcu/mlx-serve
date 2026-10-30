//! The Gemma 3 plugin proven against the legacy engine on tiny random packs written to disk: the plugin parses the
//! pack, the engine's driver loads it into a served shell, and every forward the engine makes (prefill chunks past
//! the window and through the hd-256 prefill kernels, decode through the fused norm+RoPE step, verify, the lazy
//! pipelined decode, captures, batched rows) is bit-equal to the legacy `Transformer`'s; the parse equals the legacy
//! parse field for field on every claimed fixture config. The served set is untouched: the test binds the plugin
//! itself. `Both`, the step runners and the field diff are shared with Gemma 4's proof.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const lazyForward = @import("../../engines/mlx/generate/step.zig").lazyForward;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const FsPackDir = @import("../../host/registry/pack_dir.zig").FsPackDir;
const fused_act = @import("../../sdk/blocks/mlp/fused_activations.zig");
const gemma3 = @import("plugin.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");
const Tape = tape_mod.Tape;
pub const Oracle = @import("../../engines/mlx/plugin/both.zig").Oracle;

const testing = std.testing;
const Array = mlx.mlx_array;

/// The plugin's shell against a tape recorded from the legacy forward (`plugin/tape.zig`).
pub const Both = struct {
    tmp: testing.TmpDir,
    dir: []u8,
    cfg: model.ModelConfig,
    xfm: Transformer,
    tape: Tape = .{},
    /// The process's GELU tables before the pack: one built here from the compiled GELU goes at `close`, so the
    /// uncompiled-GELU tests that follow see their own.
    tables: [2]?Array,
    declined: [2]bool,

    pub fn openAs(table: *const abi.ModelTable, p: *fixture.Pack, json: []const u8, from: Oracle) !Both {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const tables = fused_act.gelu_tanh_tables;
        const declined = fused_act.gelu_tanh_declined;
        // First: a tape that skips here (off Metal) must do so before anything is allocated.
        const t: Tape = switch (from) {
            .tape => |bytes| try Tape.player(bytes),
            .none => .{},
        };
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
        var path_buf: [512]u8 = undefined;
        const dir = try a.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
        errdefer a.free(dir);
        try p.save(dir);
        var cfg = try bind(a, io, table, dir, json);
        errdefer cfg.deinit(a);
        const xfm = try driver.load(io, a, &cfg, dir, fixture.stream());
        return .{ .tmp = tmp, .dir = dir, .cfg = cfg, .xfm = xfm, .tape = t, .tables = tables, .declined = declined };
    }

    /// A playback read every recorded value (the last call of a proof).
    pub fn expectDone(b: *const Both) !void {
        try b.tape.expectDone();
    }

    pub fn close(b: *Both) void {
        b.xfm.deinit();
        b.cfg.deinit(testing.allocator);
        testing.allocator.free(b.dir);
        b.tmp.cleanup();
        for (&fused_act.gelu_tanh_tables, b.tables) |*t, before| {
            if (t.*) |now| if (before == null) {
                _ = mlx.mlx_array_free(now);
            };
            t.* = before;
        }
        fused_act.gelu_tanh_declined = b.declined;
    }

    /// The shell to run (a slice, as the proofs iterate it).
    pub fn sides(b: *Both, buf: *[2]*Transformer) []const *Transformer {
        buf[0] = &b.xfm;
        return buf[0..1];
    }

    /// `got` against the tape; `tokens` marks a greedy run.
    pub fn check(b: *Both, got: Array, tokens: bool) !void {
        try b.tape.expectNext(got, tokens);
    }

    /// Host values as `check` reads them; `tokens` marks a greedy run.
    pub fn checkSlice(b: *Both, comptime T: type, got: []const T, tokens: bool) !void {
        const dt: mlx.mlx_dtype = if (T == u32) .uint32 else .float32;
        const g = mlx.mlx_array_new_data(got.ptr, &[_]c_int{@intCast(got.len)}, 1, dt);
        defer _ = mlx.mlx_array_free(g);
        try b.check(g, tokens);
    }

    pub fn checkIds(b: *Both, got: []const u32) !void {
        try b.checkSlice(u32, got, true);
    }
};

/// The engine's config of the pack at `dir` bound to `table`, as `parseConfig` builds it for a served type: the
/// legacy parse with the plugin's binding attached.
pub fn bind(a: std.mem.Allocator, io: std.Io, table: *const abi.ModelTable, dir: []const u8, json: []const u8) !model.ModelConfig {
    var cfg = try model.parseConfigFromJson(a, json);
    errdefer cfg.deinit(a);
    cfg.plugin = try binding(a, io, table, dir, json);
    return cfg;
}

/// The plugin's facts alone (`plugin_config.adapted`), as the engine builds them for a type the legacy parse refuses.
pub fn bindAdapted(a: std.mem.Allocator, io: std.Io, table: *const abi.ModelTable, dir: []const u8, json: []const u8) !model.ModelConfig {
    return plugin_config.adapted(try binding(a, io, table, dir, json));
}

fn binding(a: std.mem.Allocator, io: std.Io, table: *const abi.ModelTable, dir: []const u8, json: []const u8) !*plugin_config.Binding {
    var fs = try FsPackDir.open(io, dir);
    defer fs.close();
    const pd = fs.packDir();
    const bnd = try a.create(plugin_config.Binding);
    bnd.* = .{ .table = table };
    const v = table.parse(&pd, abi.Str.of(json), .{}, null, false, &bnd.parsed);
    if (!v.isOk()) {
        a.destroy(bnd);
        std.debug.print("plugin parse: {s}\n", .{v.why.slice()});
        return error.PluginDeclined;
    }
    return bnd;
}

/// One slot's cache and context.
pub const Slot = struct {
    kv: KVCache,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    pub fn init(s: *Slot, layers: u32) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, layers) };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    }
    pub fn deinit(s: *Slot) void {
        s.kv.deinit();
    }
};

pub const Step = struct { ids: []const i32, verify: bool = false };

/// A prompt past the window and through the hd-256 prefill kernels (16+ rows), decodes, verify widths, a second
/// chunk: every sliding window crossed.
pub const steps = [_]Step{
    .{ .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 4, 5, 6, 7, 8, 9, 10, 11 } },
    .{ .ids = &.{88} },
    .{ .ids = &.{ 5, 61, 200 }, .verify = true },
    .{ .ids = &.{12} },
    .{ .ids = &.{ 9, 33, 77, 140, 2, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 } },
    .{ .ids = &.{64} },
    .{ .ids = &.{ 9, 33, 77, 140, 2 }, .verify = true },
    .{ .ids = &.{12} },
};

/// Prefill, decode and verify steps on caches of their own: logits, the last hidden row and the captured layers
/// bit-equal.
pub fn expectSteps(b: *Both) !void {
    const n = b.cfg.num_hidden_layers;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    var slots: [2]Slot = undefined;
    for (slots[0..xs.len]) |*s| try s.init(n);
    defer for (slots[0..xs.len]) |*s| s.deinit();
    for (steps) |st| {
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
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
        for (0..3) |i| try b.check(outs[xs.len - 1][i], false);
    }
}

/// Greedy decode through the pipelined step (`lazyForward`: the next token stays a lazy argmax), tokens bit-equal.
pub fn expectGreedy(b: *Both) !void {
    // Off Metal the tiny pack's pipelined generation can stall in MLX's CUDA wait; served packs complete.
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    var tokens: [2][12]u32 = undefined;
    for (xs, 0..) |x, k| {
        var s: Slot = undefined;
        try s.init(b.cfg.num_hidden_layers);
        defer s.deinit();
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
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
    try b.checkIds(&tokens[xs.len - 1]);
}

/// A prompt with three media rows at placeholder `img` (the engine's `vision_embeddings`, the shell's `Forward.media`),
/// then a decode step: logits bit-equal, and the rows reach the forward.
pub fn expectMedia(b: *Both, img: i32) !void {
    const n = b.cfg.num_hidden_layers;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    var g = ops.Graph.init(testing.allocator, b.xfm.s);
    defer g.deinit();
    const hidden: c_int = @intCast(b.cfg.hidden_size);
    const vals = try testing.allocator.alloc(f32, @intCast(3 * hidden));
    defer testing.allocator.free(vals);
    var prng = std.Random.DefaultPrng.init(0x1a6e);
    for (vals) |*v| v.* = prng.random().float(f32) - 0.5;
    const rows = try g.astype(try g.fromSlice(f32, vals, &.{ 1, 3, hidden }), .bfloat16);
    const prompt = [_]i32{ 2, 17, img, img, img, 42, 99, 7 };
    var outs: [2][2]Array = undefined;
    for (xs, 0..) |x, k| {
        var s: Slot = undefined;
        try s.init(n);
        defer s.deinit();
        s.ctx.vision_embeddings = rows;
        outs[k][0] = try g.adopt(try x.forwardWith(&s.ctx, try fixture.tokens(&g, &prompt)));
        s.ctx.vision_embeddings = null;
        outs[k][1] = try g.adopt(try x.forwardWith(&s.ctx, try fixture.tokens(&g, &.{64})));
    }
    const got = outs[xs.len - 1];
    for (0..2) |i| try b.check(got[i], false);
    var s: Slot = undefined;
    try s.init(n);
    defer s.deinit();
    const plain = try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, &prompt)));
    const eq = try g.call(mlx.mlx_array_equal, .{ plain, got[0], false });
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = true;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(!same);
}

/// Two slots prefilled apart, then decoded together past the window: each slot's logits bit-equal to the legacy
/// batched tick's.
pub fn expectBatched(b: *Both) !void {
    // Off Metal the tiny pack's pipelined generation can stall in MLX's CUDA wait; served packs complete.
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const n = b.cfg.num_hidden_layers;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    var slots: [2][2]Slot = undefined;
    for (slots[0..xs.len]) |*pair| for (pair) |*s| try s.init(n);
    defer for (slots[0..xs.len]) |*pair| for (pair) |*s| s.deinit();
    const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23, 42, 9 }, &.{ 42, 101, 7 } };
    var positions: [2]u32 = undefined;
    for (prompts, 0..) |pr, i| {
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        for (xs, 0..) |x, k| _ = try g.adopt(try x.forwardWith(&slots[k][i].ctx, try fixture.tokens(&g, pr)));
        positions[i] = @intCast(pr.len);
    }
    for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 }, .{ 11, 12 }, .{ 13, 250 } }) |next| {
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        var rows: [2][]Array = undefined;
        for (xs, 0..) |x, k| {
            const ctxs = [_]*ForwardCtx{ &slots[k][0].ctx, &slots[k][1].ctx };
            rows[k] = try x.forwardBatchedDecode(&next, &ctxs, &positions);
            for (rows[k]) |r| _ = try g.adopt(r);
        }
        defer for (rows[0..xs.len]) |r| a.free(r);
        for (rows[xs.len - 1]) |got| try b.check(got, false);
        for (&positions) |*p| p.* += 1;
    }
}

/// The plugin's parse of `json` (+ a generation sidecar) against the engine's: every `ModelConfig` field the
/// family's `Config` names equals the legacy value; every other field equals the served adapter's, except
/// `waived` (fields the engine reads only on the legacy path and the adapter leaves at its generic defaults).
pub fn expectParseEqual(comptime Config: type, table: *const abi.ModelTable, json: []const u8, comptime waived: []const []const u8) !void {
    @setEvalBranchQuota(200_000);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = "{\"temperature\":0.7,\"top_k\":40,\"eos_token_id\":[1,50]}" });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var served = try bindAdapted(a, io, table, dir, json);
    defer served.deinit(a);
    const c: *const Config = @ptrCast(@alignCast(served.plugin.?.parsed.private.?));
    var legacy = try model.parseConfigFromJson(a, json);
    defer legacy.deinit(a);
    // The sidecar half of the legacy load (`parseConfig` past `parseConfigFromJson`), then the family defaults.
    legacy.gen_temperature = 0.7;
    legacy.gen_top_k = 40;
    legacy.mergeEosTokens(&.{ 1, 50 });
    legacy.applyFamilySamplingDefaults();

    inline for (@typeInfo(model.ModelConfig).@"struct".field_names) |name| {
        const in_config = comptime @hasField(Config, name);
        const skip = comptime for (waived ++ [_][]const u8{"plugin"}) |n| {
            if (std.mem.eql(u8, n, name)) break true;
        } else false;
        if (!skip) {
            const want = @field(legacy, name);
            const got = if (in_config) @field(c, name) else @field(served, name);
            const same = switch (@typeInfo(@FieldType(model.ModelConfig, name))) {
                .pointer => std.mem.eql(u8, got, want),
                .optional => |o| if (@typeInfo(o.child) == .pointer) (got == null) == (want == null) and (got == null or std.mem.eql(u8, got.?, want.?)) else std.meta.eql(got, want),
                else => std.meta.eql(got, want),
            };
            if (!same) {
                std.debug.print("plugin config: field {s} differs from the legacy parse ({s})\n", .{ name, if (in_config) "family Config" else "served adapter" });
                return error.ConfigFieldDiffers;
            }
        }
    }
    try testing.expectEqual(legacy.contextCap(), served.contextCap());
    try testing.expectEqualSlices(u32, legacy.eosTokenSlice(), served.eosTokenSlice());
    try testing.expectEqual(legacy.quant_bits, c.quant.bits);
}

// ── Gemma 3 packs ──

/// How a proof pack stores its linears when quantized: 4-bit affine at group 64, one of MLX's float modes, or a
/// ternary 2-bit layout (every bias -scale).
pub const Quant = struct { mode: [*:0]const u8 = "affine", bits: c_int = 4, group: c_int = 64, ternary: bool = false };

pub const q2: Quant = .{ .bits = 2, .group = 128 };
pub const ternary: Quant = .{ .bits = 2, .group = 128, .ternary = true };
pub const q2_json = "\"quantization\":{\"group_size\":128,\"bits\":2}";
pub const q4_json = "\"quantization\":{\"group_size\":64,\"bits\":4}";

/// `base` stored as `q`; bf16 when its input width is not a multiple of the group, as mlx-lm leaves such a layer.
pub fn put(p: *fixture.Pack, base: []const u8, shape: []const c_int, q: ?Quant) !void {
    const x = q orelse return p.tensor(base, shape, false);
    if (@rem(shape[shape.len - 1], x.group) != 0) return p.tensor(base, shape, false);
    if (x.ternary) return p.ternary(base, shape, x.group);
    return p.quantize(base, shape, x.mode, x.bits, x.group);
}

const H = 128;
const VOCAB = 256;
const INTER = 256;
const HD = 256;

fn at(buf: []u8, prefix: []const u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, name });
}

/// Four layers of hd 256 (2 heads over 1 KV head) under `prefix`; an untied head when asked.
fn writeGemma3(p: *fixture.Pack, prefix: []const u8, q: ?Quant, head: bool) !void {
    var buf: [192]u8 = undefined;
    var nb: [96]u8 = undefined;
    try put(p, try std.fmt.bufPrint(&nb, "{s}.embed_tokens", .{prefix}), &.{ VOCAB, H }, q);
    try p.norm(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{prefix}), H);
    if (head) try put(p, "lm_head", &.{ VOCAB, H }, q);
    for (0..4) |i| {
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |n|
            try p.norm(try at(&buf, prefix, i, n), H);
        try put(p, try at(&buf, prefix, i, "self_attn.q_proj"), &.{ 2 * HD, H }, q);
        try put(p, try at(&buf, prefix, i, "self_attn.k_proj"), &.{ HD, H }, q);
        try put(p, try at(&buf, prefix, i, "self_attn.v_proj"), &.{ HD, H }, q);
        try put(p, try at(&buf, prefix, i, "self_attn.o_proj"), &.{ H, 2 * HD }, q);
        try p.norm(try at(&buf, prefix, i, "self_attn.q_norm.weight"), HD);
        try p.norm(try at(&buf, prefix, i, "self_attn.k_norm.weight"), HD);
        try put(p, try at(&buf, prefix, i, "mlp.gate_proj"), &.{ INTER, H }, q);
        try put(p, try at(&buf, prefix, i, "mlp.up_proj"), &.{ INTER, H }, q);
        try put(p, try at(&buf, prefix, i, "mlp.down_proj"), &.{ H, INTER }, q);
    }
}

/// Gemma 3's own pattern (layers 1 and 3 global), the linear rope scale of 8 it defaults to, a logit softcap.
const gemma3_4bit =
    \\{"model_type":"gemma3_text","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,
    \\"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"vocab_size":256,"rms_norm_eps":1e-6,
    \\"rope_theta":1000000.0,"rope_local_base_freq":10000.0,"sliding_window":4,"sliding_window_pattern":2,
    \\"query_pre_attn_scalar":256,"final_logit_softcapping":30.0,"max_position_embeddings":4096,
    \\"eos_token_id":[1,106],"quantization":{"group_size":64,"bits":4}}
;

/// bf16, an untied head, an explicit layer map (layer 2 global), a named rope factor, no softcap.
const gemma3_bf16 =
    \\{"model_type":"gemma3_text","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,
    \\"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"vocab_size":256,"rms_norm_eps":1e-6,
    \\"rope_theta":1000000.0,"rope_local_base_freq":10000.0,"sliding_window":4,"query_pre_attn_scalar":64,
    \\"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention"],
    \\"rope_scaling":{"factor":4.0,"rope_type":"linear"},"tie_word_embeddings":false,"max_position_embeddings":4096,
    \\"eos_token_id":1}
;

/// Fields the engine reads off `ModelConfig` on the legacy path only; the served adapter leaves its generic defaults
/// (the flip needs them from the cache layout and facts first).
pub const flip_waived = [_][]const u8{ "has_qk_norm", "has_pre_ff_norm", "hidden_act" };


fn oracle(comptime name: []const u8) Oracle {
    const rel = "fixtures/" ++ name ++ ".tape";
    return .{ .tape = @embedFile(rel) };
}

test "gemma3 plugin: a 4-bit pack (pattern layers, softcap, rope scale 8) is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e3a);
    defer p.deinit();
    try writeGemma3(&p, "model", .{}, false);
    var b = try Both.openAs(&gemma3.table, &p, gemma3_4bit, oracle("4bit"));
    defer b.close();
    try testing.expect(b.xfm.plugin != null and b.cfg.rope_scaling_factor == 8.0);
    try expectSteps(&b);
    try expectGreedy(&b);
    try expectBatched(&b);
    try b.expectDone();
}

test "gemma3 plugin: a bf16 pack (layer map, untied head, named rope factor) is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e3b);
    defer p.deinit();
    try writeGemma3(&p, "model", null, true);
    var b = try Both.openAs(&gemma3.table, &p, gemma3_bf16, oracle("bf16"));
    defer b.close();
    try expectSteps(&b);
    try expectGreedy(&b);
    try expectBatched(&b);
    try b.expectDone();
}

test "gemma3 plugin: an mxfp4 pack is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e3d);
    defer p.deinit();
    try writeGemma3(&p, "model", .{ .mode = "mxfp4", .bits = 4, .group = 32 }, false);
    const json = try std.mem.replaceOwned(u8, testing.allocator, gemma3_4bit, "\"quantization\":{\"group_size\":64,\"bits\":4}", "\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}");
    defer testing.allocator.free(json);
    var b = try Both.openAs(&gemma3.table, &p, json, oracle("mxfp4"));
    defer b.close();
    try expectSteps(&b);
    try expectGreedy(&b);
    try expectBatched(&b);
    try b.expectDone();
}

test "gemma3 plugin: 2-bit group-128 packs, plain (stock qmm) and ternary (qmv2's kernels on both sides), are bit-equal" {
    inline for (.{ .{ q2, "q2", 0x9e3e }, .{ ternary, "ternary", 0x9e3f } }) |cs| {
        var p = fixture.Pack.init(cs[2]);
        defer p.deinit();
        try writeGemma3(&p, "model", cs[0], true);
        const json = try std.mem.replaceOwned(u8, testing.allocator, gemma3_4bit, q4_json, q2_json ++ ",\"tie_word_embeddings\":false");
        defer testing.allocator.free(json);
        var b = try Both.openAs(&gemma3.table, &p, json, oracle(cs[1]));
        defer b.close();
        const m: *const gemma3.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
        try testing.expectEqual(cs[0].ternary, m.base.weights.head.head.ternary);
        try expectSteps(&b);
        try expectGreedy(&b);
        try expectBatched(&b);
        try b.expectDone();
    }
}

test "gemma3 plugin: a vision pack's image rows splice bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e3c);
    defer p.deinit();
    try writeGemma3(&p, "language_model.model", .{}, false);
    const json = "{\"model_type\":\"gemma3\",\"image_token_index\":250,\"vision_config\":{\"hidden_size\":64},\"quantization\":{\"group_size\":64,\"bits\":4},\"text_config\":" ++ gemma3_4bit ++ "}";
    var b = try Both.openAs(&gemma3.table, &p, json, oracle("vision"));
    defer b.close();
    try testing.expectEqual(@as(u32, 250), b.cfg.image_token_id);
    try expectMedia(&b, 250);
    try b.expectDone();
}

test "gemma3 plugin: the parse equals the legacy parse on the proof packs and the harrier fixture" {
    for ([_][]const u8{ gemma3_4bit, gemma3_bf16, @embedFile("../../fixtures/model-configs/gemma3-text.json") }) |json|
        try expectParseEqual(gemma3.config.Config, &gemma3.table, json, &flip_waived);
}
