//! Family proofs of a plugin table against tapes recorded from the legacy forward (`tape.zig`): one tiny random pack
//! on disk served behind the driver's shell (`driver.load`). Steps (prefill, decode, verify, captures), the pipelined
//! greedy decode and batched rows replay their recording; `expectGenericFieldsEqual` diffs the engine's `ModelConfig`
//! of the plugin's parse and the legacy parse.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const model = @import("../../../sdk/config/model.zig");
const driver = @import("driver.zig");
const plugin_config = @import("config.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const lazyForward = @import("../generate/step.zig").lazyForward;
const FsPackDir = @import("../../../host/registry/pack_dir.zig").FsPackDir;
const KVCache = @import("../kv/cache.zig").KVCache;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../forward_ctx.zig").CaptureLayers;
const Transformer = @import("../transformer.zig").Transformer;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const tape_mod = @import("tape.zig");
const pooling = @import("../embed/pooling.zig");
const Tape = tape_mod.Tape;

const testing = std.testing;
const Array = mlx.mlx_array;

pub const Step = fixture.Step;

/// Prefill, then decode and verify windows of several widths.
pub const steps = fixture.steps;

/// A longer run: a prompt past a small window, a second chunk, verify widths and decodes across it.
pub const long_steps = [_]Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .prefill, .ids = &.{ 9, 33, 77, 140, 2, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 } },
    .{ .mode = .decode, .ids = &.{64} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{12} },
};

/// Where a proof's expected values come from: a tape recorded from the legacy forward, or none (a proof that
/// compares two of the plugin's own runs).
pub const Oracle = union(enum) {
    tape: []const u8,
    none,
};

/// A tiny pack served by its plugin, against its tape.
pub const Both = struct {
    tmp: testing.TmpDir,
    dir: []u8,
    cfg: model.ModelConfig,
    xfm: Transformer,
    tape: Tape = .{},

    /// The plugin against `recorded` (`tape.zig`), on the legacy parse's `ModelConfig` when `legacy_config`.
    pub fn replay(table: *const abi.ModelTable, pack: *fixture.Pack, config_json: []const u8, legacy_config: bool, recorded: []const u8) !Both {
        return openAs(table, pack, config_json, legacy_config, .{ .tape = recorded });
    }

    pub fn openAs(table: *const abi.ModelTable, pack: *fixture.Pack, config_json: []const u8, legacy_config: bool, oracle: Oracle) !Both {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        // First: a tape that skips here (off Metal) must do so before anything is allocated.
        const t: Tape = switch (oracle) {
            .tape => |bytes| try Tape.player(bytes),
            .none => .{},
        };
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
        var path_buf: [512]u8 = undefined;
        const dir = try a.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
        errdefer a.free(dir);
        try pack.save(dir);
        var cfg = try parse(table, dir, config_json);
        errdefer cfg.deinit(a);
        if (legacy_config) {
            var lc = try model.parseConfigFromJson(a, config_json);
            lc.plugin = cfg.plugin;
            cfg.plugin = null;
            cfg.deinit(a);
            cfg = lc;
        }
        const xfm = try driver.load(io, a, &cfg, dir, fixture.stream());
        return .{ .tmp = tmp, .dir = dir, .cfg = cfg, .xfm = xfm, .tape = t };
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
    }

    pub fn layers(b: *const Both) usize {
        return b.cfg.num_hidden_layers;
    }

    /// `got` against the tape's next value.
    pub fn check(b: *Both, got: Array, tokens: bool) !void {
        try b.tape.expectNext(got, tokens);
    }

    /// Prefill, decode and verify steps on a cache of its own: logits, the last hidden row and every layer's captured
    /// output.
    pub fn expectSteps(b: *Both, seq: []const Step) !void {
        const n = b.layers();
        var s: Slot = undefined;
        try s.init(b, &b.xfm);
        defer s.deinit();
        const ids_all = try testing.allocator.alloc(u32, n);
        defer testing.allocator.free(ids_all);
        for (ids_all, 0..) |*id, i| id.* = @intCast(i);
        for (seq) |st| {
            var g = ops.Graph.init(testing.allocator, b.xfm.s);
            defer g.deinit();
            const ids = try fixture.tokens(&g, st.ids);
            var hidden = mlx.mlx_array_new();
            const caps = try testing.allocator.alloc(Array, n);
            defer testing.allocator.free(caps);
            for (caps) |*c| c.* = mlx.mlx_array_new();
            var cl: CaptureLayers = .{ .ids = ids_all, .out = caps };
            s.ctx.capture_layers = &cl;
            s.ctx.spec_verify = st.mode == .verify;
            defer s.ctx.capture_layers = null;
            const logits = try g.adopt(try b.xfm.forwardWithCapture(&s.ctx, ids, &hidden));
            _ = try g.adopt(hidden);
            try b.check(logits, false);
            try b.check(hidden, false);
            for (caps) |c| try b.check(try g.adopt(c), false);
        }
    }

    /// Greedy decode through the pipelined step (`lazyForward`: the next token stays a lazy argmax), tokens equal.
    pub fn expectGreedy(b: *Both) !void {
        // Off Metal the tiny pack's pipelined generation can stall in MLX's CUDA wait; served packs complete.
        if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
        var tokens: [8]u32 = undefined;
        var s: Slot = undefined;
        try s.init(b, &b.xfm);
        defer s.deinit();
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
        defer g.deinit();
        var logits = try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, &.{ 7, 8, 9, 10 })));
        for (&tokens) |*t| {
            const sh = ops.shape(logits);
            const last = try g.reshape(try g.sliceAxis(logits, 1, sh[1] - 1, sh[1]), &.{ 1, sh[2] });
            const next = try g.argmax(last, -1, false);
            logits = try g.adopt(try lazyForward(&b.xfm, &s.ctx, next));
            const host = try g.astype(next, .uint32);
            try mlx.check(mlx.mlx_array_eval(host));
            t.* = mlx.mlx_array_data_uint32(host).?[0];
        }
        const got = mlx.mlx_array_new_data(&tokens, &[_]c_int{8}, 1, .uint32);
        defer _ = mlx.mlx_array_free(got);
        try b.check(got, true);
    }

    /// A prompt with `n` media rows at placeholder id `image` (one split across a chunk boundary, the second chunk
    /// resuming at its row offset), then decode: logits equal.
    pub fn expectMedia(b: *Both, image: i32, hidden: c_int) !void {
        const a = testing.allocator;
        var s: Slot = undefined;
        try s.init(b, &b.xfm);
        defer s.deinit();
        var g0 = ops.Graph.init(a, b.xfm.s);
        defer g0.deinit();
        var prng = std.Random.DefaultPrng.init(0x3ed1a);
        const buf = try a.alloc(f32, @intCast(5 * hidden));
        defer a.free(buf);
        for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
        s.ctx.vision_embeddings = try g0.astype(try g0.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, 5, hidden }, 3, .float32)), .bfloat16);
        const chunks = [_]struct { ids: []const i32, offset: usize }{
            .{ .ids = &.{ 3, 17, image, image, image }, .offset = 0 },
            .{ .ids = &.{ image, image, 7, 8 }, .offset = 3 },
            .{ .ids = &.{9}, .offset = 5 },
        };
        for (chunks) |ch| {
            var g = ops.Graph.init(a, b.xfm.s);
            defer g.deinit();
            s.ctx.vision_splice_offset = ch.offset;
            try b.check(try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, ch.ids))), false);
        }
    }

    /// Two slots prefilled apart, then decoded together: each slot's logits.
    pub fn expectBatched(b: *Both) !void {
        // Off Metal the tiny pack's pipelined generation can stall in MLX's CUDA wait; served packs complete.
        if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
        const a = testing.allocator;
        var slots: [2]Slot = undefined;
        for (&slots) |*s| try s.init(b, &b.xfm);
        defer for (&slots) |*s| s.deinit();
        const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23, 42, 9 }, &.{ 42, 101, 7 } };
        var positions: [2]u32 = undefined;
        for (prompts, &slots, 0..) |pr, *s, i| {
            var g = ops.Graph.init(a, b.xfm.s);
            defer g.deinit();
            try b.check(try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, pr))), false);
            positions[i] = @intCast(pr.len);
        }
        for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 }, .{ 11, 12 }, .{ 13, 250 } }) |next| {
            var g = ops.Graph.init(a, b.xfm.s);
            defer g.deinit();
            const ctxs = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
            const rows = try b.xfm.forwardBatchedDecode(&next, &ctxs, &positions);
            defer a.free(rows);
            for (rows) |r| _ = try g.adopt(r);
            for (rows) |r| try b.check(r, false);
            for (&positions) |*p| p.* += 1;
            // The scheduler's tick advances each slot's offset after the forward (`batched/tick.zig`).
            for (&slots) |*s| s.off += 1;
        }
    }
};

/// `table`'s parse of the pack at `dir` as the engine's `ModelConfig`, bound to the table (as `parseServed` does for
/// a served type).
pub fn parse(table: *const abi.ModelTable, dir: []const u8, config_json: []const u8) !model.ModelConfig {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fs = try FsPackDir.open(io, dir);
    defer fs.close();
    const pd = fs.packDir();
    const bind = try a.create(plugin_config.Binding);
    bind.* = .{ .table = table };
    const json = try plugin_config.withOverrides(a, config_json);
    defer if (json.ptr != config_json.ptr) a.free(json);
    const v = table.parse(&pd, abi.Str.of(json), .{}, null, false, &bind.parsed);
    if (!v.isOk()) {
        a.destroy(bind);
        std.debug.print("plugin parse refused: {s}\n", .{v.why.slice()});
        return error.PluginParseRefused;
    }
    var cfg = plugin_config.modelConfig(&bind.parsed);
    cfg.plugin = bind;
    return cfg;
}

/// One slot's cache, recurrent entries (when the model has recurrent layers) and context on `xfm`.
const Slot = struct {
    kv: KVCache,
    off: usize = 0,
    ssm: ?[]SSMCacheEntry = null,
    xfm: *Transformer,
    ctx: ForwardCtx = undefined,

    fn init(s: *Slot, b: *const Both, xfm: *Transformer) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, @intCast(b.layers())), .xfm = xfm };
        errdefer s.kv.deinit();
        if (b.cfg.needsSsmEntries()) s.ssm = try fixture.newEntries(b.layers());
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = s.ssm, .capture_hidden = null, .vision_embeddings = null };
    }
    fn deinit(s: *Slot) void {
        if (s.ssm) |e| {
            s.xfm.ssmGroupDrop(e);
            fixture.freeEntries(e);
        }
        s.kv.deinit();
    }
};

/// The served `ModelConfig` equals the legacy one on every field but `owned` (fields only the plugin's forward reads),
/// `plugin`, and an idle `sliding_window*` (read only when `has_sliding_window`).
pub fn expectGenericFieldsEqual(served: *const model.ModelConfig, legacy: *const model.ModelConfig, owned: []const []const u8) !void {
    @setEvalBranchQuota(100_000);
    inline for (@typeInfo(model.ModelConfig).@"struct".field_names) |name| {
        const skip = for (owned) |n| {
            if (std.mem.eql(u8, n, name)) break true;
        } else false;
        const idle = comptime std.mem.startsWith(u8, name, "sliding_window");
        if (comptime !std.mem.eql(u8, name, "plugin") and !std.mem.eql(u8, name, "legacy")) if (!skip and !(idle and !legacy.has_sliding_window)) {
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
        };
    }
    try testing.expectEqual(legacy.contextCap(), served.contextCap());
    try testing.expectEqualSlices(u32, legacy.eosTokenSlice(), served.eosTokenSlice());
}

/// Parses `config_json` both ways (a generation_config sidecar beside it on the plugin side, the same values merged
/// into the legacy parse) and diffs them.
pub fn expectConfigEqual(table: *const abi.ModelTable, config_json: []const u8, owned: []const []const u8) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
    try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = "{\"temperature\":0.6,\"top_p\":0.9,\"eos_token_id\":[2,9]}" });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var served = try parse(table, dir, config_json);
    defer served.deinit(a);
    var legacy = try model.parseConfigFromJson(a, config_json);
    defer legacy.deinit(a);
    // The sidecar half of the legacy load (`parseConfig` past `parseConfigFromJson`).
    legacy.gen_temperature = 0.6;
    legacy.gen_top_p = 0.9;
    legacy.mergeEosTokens(&.{ 2, 9 });
    try expectGenericFieldsEqual(&served, &legacy, owned);
}

/// A pack of a type whose legacy forward is deleted, with `key` (a field its plugin declines) added: the served parse
/// refuses it by name instead of handing it to the legacy path.
pub fn expectNoLegacyFallback(config_json: []const u8, comptime key: []const u8) !void {
    const json = try std.mem.concat(testing.allocator, u8, &.{ "{\"" ++ key ++ "\":{},", config_json[1..] });
    defer testing.allocator.free(json);
    try expectRefused(json);
}

/// A shape a legacy-removed type's plugin declines, and what the legacy reading did with it (the pre-deletion audit,
/// `.bb/reports/p3-fam-dense.md` round 4): its parse failed (`parse`), its load or forward failed or emitted
/// non-finite logits (`failed`), or it served the pack on another family's default or with the key dropped
/// (`dropped`).
pub const Decline = struct {
    /// A root key, or `text_config.<key>` inside the text config.
    key: []const u8,
    /// The key's JSON value; null removes it.
    value: ?[]const u8,
    legacy: enum { parse, failed, dropped },
};

/// Every shape in `shapes` applied to `config_json`: a `.parse` one fails the legacy parse (the served load stops
/// there), every other one is refused by name.
pub fn expectDeclines(config_json: []const u8, shapes: []const Decline) !void {
    const a = testing.allocator;
    for (shapes) |sh| {
        var v = try std.json.parseFromSlice(std.json.Value, a, config_json, .{});
        defer v.deinit();
        const nested = std.mem.startsWith(u8, sh.key, "text_config.");
        const obj = if (nested) &v.value.object.getPtr("text_config").?.object else &v.value.object;
        const key = if (nested) sh.key["text_config.".len..] else sh.key;
        if (sh.value) |val| {
            const x = try std.json.parseFromSliceLeaky(std.json.Value, v.arena.allocator(), val, .{});
            try obj.put(v.arena.allocator(), key, x);
        } else _ = obj.orderedRemove(key);
        const json = try std.json.Stringify.valueAlloc(a, v.value, .{});
        defer a.free(json);
        errdefer std.debug.print("decline shape {s}={s}\n", .{ sh.key, sh.value orelse "(removed)" });
        if (sh.legacy == .parse) {
            if (model.parseConfigFromJson(a, json)) |c| {
                var cfg = c;
                cfg.deinit(a);
                return error.TestExpectedLegacyParseError;
            } else |_| {}
        } else try expectRefused(json);
    }
}

/// `json` as a pack's config: the plugin declines it and no legacy path takes it.
fn expectRefused(json: []const u8) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try testing.expectError(error.PluginDeclinedNoLegacy, plugin_config.bindServed(io, a, dir, json));
}

/// The engine config of `config_json` served under the caller's `--config-overrides`: the plugin binds and the
/// engine config is the legacy parse with the overrides in it. Caller deinits.
pub fn servedUnderOverrides(config_json: []const u8) !model.ModelConfig {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var cfg = try model.parseConfig(io, a, dir);
    errdefer cfg.deinit(a);
    if (cfg.plugin == null) return error.TestExpectedPluginServed;
    return cfg;
}

/// A released pack's config at a proof pack's geometry: `real` with every key of `fixture` but `keep` taken from the
/// fixture (inside both `text_config`s too), `drop` removed, and its `quantization_config` (when present) mlx-lm's copy of the result's
/// `quantization`. Caller frees.
pub fn atGeometry(real: []const u8, fixture_json: []const u8, keep: []const []const u8, drop: []const []const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out = try std.json.parseFromSliceLeaky(std.json.Value, a, real, .{});
    const fix = try std.json.parseFromSliceLeaky(std.json.Value, a, fixture_json, .{});
    try overlay(a, &out.object, fix.object, keep, drop);
    if (out.object.get("quantization_config") != null) try out.object.put(a, "quantization_config", out.object.get("quantization") orelse .null);
    return std.json.Stringify.valueAlloc(testing.allocator, out, .{});
}

/// A greedy DFlash run on `xfm` with `dflash.TinyFix`'s assistant (hidden 64, vocab 128, target layers 0 and 2): the
/// emitted tokens into `tokens` and each round's accepted count into `accepted`.
pub fn dflashRun(xfm: *Transformer, tokens: []u32, accepted: *std.ArrayList(u32)) !void {
    // Off Metal the tiny assistant's run can stall in MLX's CUDA wait; a served DFlash pair completes.
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const dflash_mod = @import("../spec/dflash/dflash.zig");
    const Generator = @import("../generate/generator.zig").Generator;
    const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const asst = buf[0..try tmp.dir.realPath(io, &buf)];
    try dflash_mod.TinyFix.writeAssistant(io, tmp.dir, asst, fixture.stream());
    var dm = try dflash_mod.loadDflash(io, a, fixture.stream(), asst);
    defer dm.deinit();
    try dm.bind(xfm);
    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    var gen = try Generator.initWithOptions(io, a, xfm, &tok_dummy, &prompt, @intCast(tokens.len), .{ .temperature = 0.0 }, &.{}, .{
        .dflash_enabled = true,
        .dflash = &dm,
        .dflash_min_accepted_per_round = 0,
    });
    defer gen.deinit(a);
    var n: usize = 0;
    while (try gen.nextDflash(a)) |res| {
        defer a.free(res.tokens);
        @memcpy(tokens[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
        try accepted.append(a, res.accepted_tokens);
    }
    try testing.expectEqual(tokens.len, n);
    try testing.expect(gen.dflash_attempted > 0);
}

/// A pack of `config_json` carrying an in-dir DFlash sidecar (`drafter/`) binds its plugin.
pub fn expectServedWithDrafter(config_json: []const u8) !void {
    const dflash_mod = @import("../spec/dflash/dflash.zig");
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
    try tmp.dir.createDirPath(io, "drafter");
    var sub = try tmp.dir.openDir(io, "drafter", .{});
    defer sub.close(io);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var dbuf: [600]u8 = undefined;
    const drafter = try std.fmt.bufPrint(&dbuf, "{s}/drafter", .{dir});
    try dflash_mod.TinyFix.writeAssistant(io, sub, drafter, fixture.stream());
    const p = dflash_mod.resolveInDirDrafter(io, a, dir) orelse return error.TestExpectedDrafter;
    a.free(p);
    const b = (try plugin_config.bindServed(io, a, dir, config_json)) orelse return error.TestExpectedPluginServed;
    b.destroy(a);
}

/// The served shell embeds a padded batch exactly as it embeds each sequence alone.
pub fn expectEmbeddingBatch(xfm: *Transformer) !void {
    const a = testing.allocator;
    const seqs = [_][]const u32{ &.{ 5, 9, 33, 70, 2 }, &.{ 8, 2 }, &.{7} };
    const batch = try pooling.computeEmbeddingsBatch(a, xfm, &seqs);
    defer {
        for (batch) |r| a.free(r);
        a.free(batch);
    }
    for (seqs, batch) |seq, got| {
        const one = try pooling.computeEmbeddingsBatch(a, xfm, &.{seq});
        defer {
            a.free(one[0]);
            a.free(one);
        }
        try testing.expectEqualSlices(f32, one[0], got);
    }
}

/// `dflashRun` on `xfm` against a run recorded from the legacy forward on `device` (tokens and accepted counts are
/// compared only there, as a tape's greedy run).
pub fn expectDflashRecorded(xfm: *Transformer, device: []const u8, want_tokens: []const u32, want_accepted: []const u32) !void {
    const a = testing.allocator;
    const tokens = try a.alloc(u32, want_tokens.len);
    defer a.free(tokens);
    var accepted: std.ArrayList(u32) = .empty;
    defer accepted.deinit(a);
    try dflashRun(xfm, tokens, &accepted);
    var dbuf: [64]u8 = undefined;
    if (!std.mem.eql(u8, tape_mod.deviceName(&dbuf), device)) return;
    try testing.expectEqualSlices(u32, want_tokens, tokens);
    try testing.expectEqualSlices(u32, want_accepted, accepted.items);
}

fn overlay(a: std.mem.Allocator, out: *std.json.ObjectMap, fix: std.json.ObjectMap, keep: []const []const u8, drop: []const []const u8) !void {
    var it = fix.iterator();
    next: while (it.next()) |e| {
        for (keep) |k| if (std.mem.eql(u8, k, e.key_ptr.*)) continue :next;
        if (std.mem.eql(u8, e.key_ptr.*, "text_config")) if (out.getPtr("text_config")) |tc| if (tc.* == .object and e.value_ptr.* == .object) {
            try overlay(a, &tc.object, e.value_ptr.object, keep, drop);
            continue;
        };
        try out.put(a, e.key_ptr.*, e.value_ptr.*);
    }
    for (drop) |k| _ = out.orderedRemove(k);
}
