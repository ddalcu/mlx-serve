//! Parity proof for the KDA-hybrid plugins: on a tiny random pack, the table's forward over the engine's own cache
//! adapter (`engines/mlx/kv/cache_ops.zig`) is bit-equal to the deleted legacy forward at every step: logits of every
//! row, captured layers, each KDA layer's recurrent state and a verify's per-row capture. Shared by glm5_next and
//! bailing_hybrid (`expectSteps`); GLM's batched rows, MTP head and media rows too. The legacy values replay from
//! `fixtures/*.tape`, recorded from that forward before it was deleted.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../sdk/config/model.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const sdk = @import("../../sdk/root.zig");
const fixture = @import("fixture.zig");
const parity = @import("../../sdk/testing/parity.zig");
const glm = @import("plugin.zig");
const ling = @import("../bailing_hybrid/plugin.zig");
const fwd = @import("table_forward.zig");

const testing = std.testing;
const Array = mlx.mlx_array;
const Tape = tape_mod.Tape;

/// One slot's engine cache: the KV cache and a recurrent state per layer behind the engine's adapter.
const Slot = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    adapter: cache_ops.Slot = undefined,

    fn init(s: *Slot, layers: u32) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, layers), .ssm = try testing.allocator.alloc(SSMCacheEntry, layers) };
        for (s.ssm) |*e| e.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
        s.adapter = .{ .kv = &s.kv, .ssm = s.ssm };
    }

    fn deinit(s: *Slot) void {
        for (s.ssm) |*e| for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        testing.allocator.free(s.ssm);
        s.kv.deinit();
    }
};

/// A pack on disk served by plugin `P` over the engine's resident source, checked against the tape `tape` (under
/// `src/plugins/`; none for a proof comparing the plugin with itself).
fn Twin(comptime P: type) type {
    return struct {
        tmp: testing.TmpDir,
        cfg: model.ModelConfig,
        res: Resident,
        src: quant_abi.Source = undefined,
        parsed: abi.Parsed = .{},
        m: *anyopaque,
        tape: Tape = .{},
        s: mlx.mlx_stream,

        fn open(files: []const sdk.fixture.File, comptime tape: ?[]const u8) !*@This() {
            const a = testing.allocator;
            const io = std.Io.Threaded.global_single_threaded.io();
            const t = try a.create(@This());
            errdefer a.destroy(t);
            t.* = .{ .tmp = testing.tmpDir(.{}), .cfg = undefined, .res = undefined, .m = undefined, .s = mlx.gpuStream() };
            errdefer t.tmp.cleanup();
            for (files) |f| try t.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
            var buf: [512]u8 = undefined;
            const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
            t.cfg = try model.parseConfigLegacy(io, a, dir);
            errdefer t.cfg.deinit(a);
            t.res = Resident.adopt(try model.loadModelWeights(io, a, dir, &t.cfg, false));
            errdefer t.res.deinit();
            t.src = t.res.source();
            if (tape) |rel| {
                t.tape = try Tape.player(@embedFile("../" ++ rel));
                t.tape.compact_over = tape_compact_over;
            }
            const mem: sdk.fixture.MemDir = .{ .files = files };
            const pd = mem.packDir();
            try testing.expect(P.table.parse(&pd, .of(files[0].bytes), .{}, null, false, &t.parsed).isOk());
            errdefer P.table.free_parsed(&t.parsed);
            var why: abi.Verdict = .{};
            t.m = P.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = t.s.ctx }, .weights = &t.src }, &why) orelse {
                std.debug.print("load: {s}\n", .{why.why.slice()});
                return error.PluginLoad;
            };
            return t;
        }

        fn close(t: *@This()) void {
            P.table.unload(t.m);
            P.table.free_parsed(&t.parsed);
            t.res.deinit();
            t.cfg.deinit(testing.allocator);
            t.tmp.cleanup();
            testing.allocator.destroy(t);
        }

        fn layers(t: *const @This()) u32 {
            return t.cfg.num_hidden_layers;
        }

        /// `got` (the plugin's) against the tape's next value.
        fn check(t: *@This(), got: Array) !void {
            try t.tape.expectNext(got, false);
        }

        /// The plugin's `[S, V]` logits and two captured layers for one step of slot `s`.
        fn plugin(t: *@This(), s: *Slot, mode: abi.Mode, ids: []const u32, caps: *[2]Array) !Array {
            var captured: [2]abi.Array = .{ .{}, .{} };
            var out: abi.ForwardOut = .{ .captured = &captured };
            const h = [_]abi.Handle{s.adapter.handle()};
            const pos = [_]u32{@intCast(s.off)};
            const v = P.table.forward(t.m, &.{ .mode = mode, .logits = .all_rows, .rows = @intCast(ids.len), .tokens = .of(ids), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops, .capture_layers = .of(&capture_ids) }, &out);
            if (!v.isOk()) {
                std.debug.print("forward: {s}\n", .{v.why.slice()});
                return error.PluginForward;
            }
            s.off += ids.len;
            caps.* = .{ .{ .ctx = captured[0].ctx }, .{ .ctx = captured[1].ctx } };
            const l: Array = .{ .ctx = out.logits.ctx };
            try settle(s, &.{ l, caps[0], caps[1] });
            return l;
        }

        /// Every KDA layer's state of `ps` against the tape: which states are set, then each set one, and a
        /// verify's per-row capture up to its last row (which the verify kernel may leave unwritten: the final state
        /// is its own output).
        fn expectStates(t: *@This(), ps: *const Slot) !void {
            const flags = try stateFlags(ps);
            defer _ = mlx.mlx_array_free(flags);
            try t.check(flags);
            for (ps.ssm) |y| {
                for ([_]Array{ y.conv_state, y.ssm_state, y.spec_conv_input }) |got| if (got.ctx != null) try t.check(got);
                if (y.spec_state_seq.ctx == null or mlx.getShape(y.spec_state_seq)[0] < 2) continue;
                const got = try leadingRows(y.spec_state_seq);
                defer _ = mlx.mlx_array_free(got);
                try t.check(got);
            }
        }
    };
}

/// Per layer: initialized, then which of conv, ssm, conv input and state sequence are set, as int32 `[layers, 5]`.
fn stateFlags(s: *const Slot) !Array {
    var buf: [64 * 5]i32 = undefined;
    for (s.ssm, 0..) |e, i| {
        buf[i * 5] = @intFromBool(e.initialized);
        for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }, 1..) |a, k| buf[i * 5 + k] = @intFromBool(a.ctx != null);
    }
    return mlx.mlx_array_new_data(&buf, &[_]c_int{ @intCast(s.ssm.len), 5 }, 2, .int32);
}

/// Every row of a state sequence `[T, ...]` but the last.
fn leadingRows(seq: Array) !Array {
    const sh = mlx.getShape(seq);
    var hi: [5]c_int = undefined;
    for (sh, 0..) |v, i| hi[i] = v;
    hi[0] -= 1;
    var rows = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&rows, seq, &(@as([5]c_int, @splat(0))), sh.len, &hi, sh.len, &(@as([5]c_int, @splat(1))), sh.len, mlx.gpuStream()));
    return rows;
}

const capture_ids = [_]u32{ 1, 3 };

/// Every array past 1 KiB (the logits, captures and recurrent states) is kept as a digest and a sample (`Tape.compact_over`).
pub const tape_compact_over = 1 << 10;

/// Evaluates a forward's outputs and the slot's recurrent states before anything else is built.
fn settle(s: *const Slot, outs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (outs) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, a);
    };
    for (s.ssm) |e| for ([_]Array{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, a);
    };
    try mlx.check(mlx.mlx_eval(vec));
}

fn idsArray(ids: []const u32) !Array {
    var i32s: [64]i32 = undefined;
    for (ids, 0..) |id, k| i32s[k] = @intCast(id);
    return mlx.mlx_array_new_data(&i32s, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
}

fn freeAll(arrs: []const Array) void {
    for (arrs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
}

const Step = struct { mode: abi.Mode, ids: []const u32 };

const long_prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 88, 140, 61, 5, 230, 99, 12, 400, 33, 77, 2, 64, 301, 18, 45 };

/// A long prompt (GLM: dense causal rows on lifted keys, then indexed rows), decodes, verify blocks past the
/// indexer's budget, a later chunk wide enough for the NAX sparse kernel, and a short prompt on its own slot.
const runs = [_][]const Step{
    &.{
        .{ .mode = .prefill, .ids = &long_prompt },
        .{ .mode = .decode, .ids = &.{8} },
        .{ .mode = .verify, .ids = &.{ 5, 61, 20 } },
        .{ .mode = .decode, .ids = &.{12} },
        .{ .mode = .prefill, .ids = long_prompt[0..12] },
        .{ .mode = .decode, .ids = &.{2} },
        .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    },
    &.{
        .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 9, 7 } },
        .{ .mode = .decode, .ids = &.{55} },
        .{ .mode = .verify, .ids = &.{ 6, 7 } },
        .{ .mode = .decode, .ids = &.{400} },
        .{ .mode = .decode, .ids = &.{401} },
        .{ .mode = .decode, .ids = &.{402} },
        .{ .mode = .decode, .ids = &.{403} },
        .{ .mode = .verify, .ids = &.{ 1, 2, 3 } },
    },
};

fn expectSteps(comptime P: type, t: *Twin(P)) !void {
    for (runs) |run| {
        var ps: Slot = undefined;
        try ps.init(t.layers());
        defer ps.deinit();
        for (run) |st| {
            var pc: [2]Array = undefined;
            const got = try t.plugin(&ps, st.mode, st.ids, &pc);
            defer freeAll(&.{ got, pc[0], pc[1] });
            try t.check(got);
            for (pc) |g| try t.check(g);
            try t.expectStates(&ps);
        }
    }
}

test "glm5_next plugin: prefill, decode, verify and later chunks equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "glm5_next/fixtures/steps.tape");
    defer t.close();
    try expectSteps(glm, t);
    try t.tape.expectDone();
}

test "bailing_hybrid plugin: prefill, decode, verify and later chunks equal the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(ling).open(try fixture.lingFiles(arena.allocator(), 11), "bailing_hybrid/fixtures/steps.tape");
    defer t.close();
    try expectSteps(ling, t);
    try t.tape.expectDone();
}

// Two MoE row groups: a tick of two groups is two trunk forwards in one evaluation.
test "glm5_next plugin: batched rows equal the legacy batched decode bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "glm5_next/fixtures/batched.tape");
    defer t.close();
    try expectBatched(t);
    try t.tape.expectDone();
}

// Bar: a Sushi pack (EXL3 routed experts through `sdk/quant/exl3/format.zig`) runs as the legacy forward runs it,
// per slot and batched.
test "glm5_next plugin: a Sushi pack's EXL3 experts equal the legacy forward bit for bit" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmSushiFiles(arena.allocator(), 7), "glm5_next/fixtures/sushi.tape");
    defer t.close();
    try testing.expect(t.cfg.exl3 != null);
    try expectSteps(glm, t);
    try expectBatched(t);
    try t.tape.expectDone();
}

/// The plugin's batched tick over `ps` (`toks`, one next token per slot): logits `[N, V]`.
fn pluginTick(t: *Twin(glm), ps: []Slot, toks: []const u32) !Array {
    var handles: [16]abi.Handle = undefined;
    var positions: [16]u32 = undefined;
    for (ps, 0..) |*p, i| {
        handles[i] = p.adapter.handle();
        positions[i] = @intCast(p.off);
    }
    var out: abi.ForwardOut = .{};
    try testing.expect(glm.table.forward(t.m, &.{ .mode = .batched_rows, .rows = 1, .tokens = .of(toks), .positions = .of(positions[0..ps.len]), .caches = .of(handles[0..ps.len]), .cache_ops = &cache_ops.ops }, &out).isOk());
    return .{ .ctx = out.logits.ctx };
}

fn expectBatched(t: *Twin(glm)) !void {
    const N = fwd.MOE_DECODE_BATCH_MAX + 2;
    var ps: [N]Slot = undefined;
    var made: usize = 0;
    defer for (ps[0..made]) |*p| p.deinit();
    for (&ps, 0..) |*p, i| {
        try p.init(t.layers());
        made += 1;
        // Prompts of different lengths, so the slots attend over different histories.
        var c: [2]Array = undefined;
        freeAll(&.{try t.plugin(p, .prefill, long_prompt[0 .. 3 + i], &c)});
        freeAll(&c);
        try t.expectStates(p);
    }
    for (0..3) |tick| {
        var toks: [N]u32 = undefined;
        for (&toks, 0..) |*tk, i| tk.* = @intCast((tick * 31 + i * 7 + 5) % fixture.glm.V);
        const got = try pluginTick(t, &ps, &toks);
        defer _ = mlx.mlx_array_free(got);
        for (&ps) |*p| try settle(p, &.{got});
        for (&ps, 0..) |*p, i| {
            p.off += 1;
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, got, &[_]c_int{ @intCast(i), 0 }, 2, &[_]c_int{ @intCast(i + 1), @intCast(fixture.glm.V) }, 2, &[_]c_int{ 1, 1 }, 2, t.s));
            try t.check(row);
            try t.expectStates(p);
        }
    }
}

/// One MTP head cache: a one-layer KV cache behind the engine's adapter, as `MtpCacheRef.PluginRef` holds it.
const HeadCache = struct {
    kv: KVCache,
    ssm: [1]SSMCacheEntry = .{.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false }},
    adapter: cache_ops.Slot = undefined,

    fn init(c: *HeadCache) !void {
        c.* = .{ .kv = try KVCache.init(testing.allocator, 1) };
        c.adapter = .{ .kv = &c.kv, .ssm = &c.ssm };
    }

    fn view(c: *HeadCache) !abi.KvView {
        var v: abi.KvView = .{};
        try testing.expect(cache_ops.ops.view(c.adapter.handle(), 0, .{ .ctx = mlx.gpuStream().ctx }, &v).isOk());
        return v;
    }
};

// Bar: the head behind `SpecHooks.head` gives the legacy `mtp.0` head's normed output and cache rows bit for bit over
// history rows, chained drafts and a rollback.
test "glm5_next plugin: the MTP head equals the legacy head" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "glm5_next/fixtures/head.tape");
    defer t.close();
    const s = t.s;
    const hooks = glm.table.spec.?.head.?;
    try testing.expect(hooks.load(t.m, undefined).isOk());
    var pc: HeadCache = undefined;
    try pc.init();
    defer pc.kv.deinit();

    var prng = std.Random.DefaultPrng.init(0x4ead);
    // History rows, a chained draft at depths 1 and 2, a rollback to the history, then more history.
    const Op = union(enum) { rows: []const u32, chain: u32, truncate: usize };
    const script = [_]Op{ .{ .rows = &.{ 5, 77, 140, 9, 301, 12 } }, .{ .chain = 33 }, .{ .chain = 410 }, .{ .truncate = 6 }, .{ .rows = &.{ 64, 2, 18 } }, .{ .chain = 7 } };
    var prev: Array = .{ .ctx = null };
    defer if (prev.ctx != null) {
        _ = mlx.mlx_array_free(prev);
    };
    for (script) |op| {
        const ids: []const u32 = switch (op) {
            .rows => |r| r,
            .chain => |*c| c[0..1],
            .truncate => |len| {
                try pc.kv.truncate(len, s);
                continue;
            },
        };
        const n: c_int = @intCast(ids.len);
        // A chain step reads the head's own output; history rows read fresh trunk hiddens.
        const hidden: Array = if (op == .chain) prev else try parity.testRandWeightBf16(prng.random(), &.{ 1, n, fixture.glm.H }, s);
        defer if (op != .chain) {
            _ = mlx.mlx_array_free(hidden);
        };
        const flat = try idsArray(ids);
        defer _ = mlx.mlx_array_free(flat);
        const handles = [_]abi.Handle{pc.adapter.handle()};
        const pos = [_]u32{0};
        var out: abi.ForwardOut = .{};
        try testing.expect(hooks.forward(t.m, &.{ .mode = if (n == 1) .decode else .prefill, .logits = .none, .want_hidden = true, .rows = @intCast(n), .tokens = .{}, .ids = .{ .ctx = flat.ctx }, .positions = .of(&pos), .caches = .of(&handles), .cache_ops = &cache_ops.ops, .hidden_in = .{ .ctx = hidden.ctx } }, &out).isOk());
        const got: Array = .{ .ctx = out.hidden.ctx };
        try t.check(got);
        var pv = try pc.view();
        defer cache_ops.ops.release(&pv);
        try t.check(.{ .ctx = pv.k.ctx });
        try t.check(.{ .ctx = pv.v.ctx });
        if (prev.ctx != null) _ = mlx.mlx_array_free(prev);
        prev = got;
    }
    try t.tape.expectDone();
}

/// The plugin's logits over `ids` on slot `s`, with `media` rows from `offset` (none when `media` is null).
fn plainStep(t: *Twin(glm), s: *Slot, ids: []const u32, media: Array, offset: usize) !Array {
    const h = [_]abi.Handle{s.adapter.handle()};
    const pos = [_]u32{@intCast(s.off)};
    var out: abi.ForwardOut = .{};
    try testing.expect(glm.table.forward(t.m, &.{ .mode = if (ids.len == 1) .decode else .prefill, .logits = .all_rows, .rows = @intCast(ids.len), .tokens = .of(ids), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops, .media = .{ .ctx = media.ctx }, .media_offset = @intCast(offset) }, &out).isOk());
    s.off += ids.len;
    const got: Array = .{ .ctx = out.logits.ctx };
    try settle(s, &.{got});
    return got;
}

// Bar: image rows the engine hands the plugin (`Forward.media`) splice at the placeholders as the legacy forward
// splices `vision_embeddings`, across two prompt chunks (the second resuming at its row offset), then decode.
test "glm5_next plugin: media rows splice as the legacy forward splices them" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), "glm5_next/fixtures/media.tape");
    defer t.close();
    const img: u32 = 500;
    try testing.expectEqual(img, t.cfg.image_token_id);
    var prng = std.Random.DefaultPrng.init(0x1a6e);
    const media = try parity.testRandWeightBf16(prng.random(), &.{ 1, 5, fixture.glm.H }, t.s);
    defer _ = mlx.mlx_array_free(media);
    var ps: Slot = undefined;
    try ps.init(t.layers());
    defer ps.deinit();
    const chunks = [_]struct { ids: []const u32, offset: usize }{
        .{ .ids = &.{ 3, 17, img, img, img, 42, 9 }, .offset = 0 },
        .{ .ids = &.{ img, img, 25, 1, 88 }, .offset = 3 },
        .{ .ids = &.{140}, .offset = 0 },
    };
    for (chunks) |c| {
        const got = try plainStep(t, &ps, c.ids, media, c.offset);
        defer _ = mlx.mlx_array_free(got);
        try t.check(got);
        try t.expectStates(&ps);
    }
    // The splice moved the logits: without the rows the same prompt reads differently.
    var bare: Slot = undefined;
    try bare.init(t.layers());
    defer bare.deinit();
    const plain = try plainStep(t, &bare, chunks[0].ids, .{ .ctx = null }, 0);
    defer _ = mlx.mlx_array_free(plain);
    var ws: Slot = undefined;
    try ws.init(t.layers());
    defer ws.deinit();
    const with = try plainStep(t, &ws, chunks[0].ids, media, 0);
    defer _ = mlx.mlx_array_free(with);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, plain, with, false, t.s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same_bits: bool = true;
    try mlx.check(mlx.mlx_array_item_bool(&same_bits, eq));
    try testing.expect(!same_bits);
    try t.tape.expectDone();
}

// Bar: a batched tick's bits never depend on whether the slots' prefills were evaluated before it was built.
test "glm5_next plugin: a batched tick over unevaluated prefill states equals one over evaluated states" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), null);
    defer t.close();
    const N = fwd.MOE_DECODE_BATCH_MAX + 2;
    var arms: [2][N]Slot = undefined;
    var made: [2]usize = .{ 0, 0 };
    defer for (&arms, made) |*arm, m| for (arm[0..m]) |*sl| sl.deinit();
    var toks: [N]u32 = undefined;
    for (&toks, 0..) |*tk, i| tk.* = @intCast((i * 7 + 5) % fixture.glm.V);
    var got: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer freeAll(&got);
    for (&arms, &made, 0..) |*arm, *m, lazy| {
        for (arm, 0..) |*sl, i| {
            try sl.init(t.layers());
            m.* += 1;
            const prompt = long_prompt[0 .. 3 + i];
            const h = [_]abi.Handle{sl.adapter.handle()};
            const pos = [_]u32{0};
            var out: abi.ForwardOut = .{};
            try testing.expect(glm.table.forward(t.m, &.{ .mode = .prefill, .rows = @intCast(prompt.len), .tokens = .of(prompt), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops }, &out).isOk());
            sl.off = prompt.len;
            const l: Array = .{ .ctx = out.logits.ctx };
            defer _ = mlx.mlx_array_free(l);
            if (lazy == 0) try settle(sl, &.{l});
        }
        got[lazy] = try pluginTick(t, arm, &toks);
        try mlx.check(mlx.mlx_array_eval(got[lazy]));
    }
    try ops.expectBitEqual(got[1], got[0]);
}

test "glm5_next plugin: a decode token on the device equals the host token's decode" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(glm).open(try fixture.glmFiles(arena.allocator(), 7), null);
    defer t.close();
    var hs: Slot = undefined;
    try hs.init(t.layers());
    defer hs.deinit();
    var ds: Slot = undefined;
    try ds.init(t.layers());
    defer ds.deinit();
    for ([_]*Slot{ &hs, &ds }) |s| freeAll(&.{try plainStep(t, s, long_prompt[0..5], .{ .ctx = null }, 0)});
    for ([_]u32{ 8, 55, 12 }) |id| {
        const want = try plainStep(t, &hs, &.{id}, .{ .ctx = null }, 0);
        defer _ = mlx.mlx_array_free(want);
        const dev = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(dev);
        const h = [_]abi.Handle{ds.adapter.handle()};
        const pos = [_]u32{@intCast(ds.off)};
        var out: abi.ForwardOut = .{};
        try testing.expect(glm.table.forward(t.m, &.{ .mode = .decode, .logits = .all_rows, .rows = 1, .tokens = .{}, .ids = .{ .ctx = dev.ctx }, .positions = .of(&pos), .caches = .of(&h), .cache_ops = &cache_ops.ops }, &out).isOk());
        ds.off += 1;
        const got: Array = .{ .ctx = out.logits.ctx };
        defer _ = mlx.mlx_array_free(got);
        try settle(&ds, &.{got});
        try ops.expectBitEqual(got, want);
    }
}
