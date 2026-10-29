//! The qwen3_5 table served through the engine's plugin driver, proven against the legacy shell over the same tiny
//! pack: prompt chunks, decode, captured verifies with partial-accept rollbacks from the captures, the prompt-lookup
//! round loop, batched decode ticks and a batched captured verify, and an image prompt (media rows spliced across a
//! chunk boundary on the M-RoPE table). Logits, hidden rows and every recurrent state bit-equal. The served side's
//! `ModelConfig` is the legacy parse with the plugin bound, as the engine builds it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const sdk = @import("../../sdk/root.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_pack.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const ssm_entry = @import("../../engines/mlx/ssm/entry.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const SSMCacheEntry = ssm_entry.SSMCacheEntry;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const qwen3_5 = @import("plugin.zig");
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");
const tree_mod = @import("../../engines/mlx/spec/tree.zig");
const lane_qmm = @import("../../sdk/blocks/qmm/lane_qmm.zig");
const naxAvailable = @import("../../mlx/device.zig").naxAvailable;
const dflash_mod = @import("../../engines/mlx/spec/dflash/dflash.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

/// The served shell over one pack, bound to `table`, against its tape.
pub const Pair = struct {
    dir: []u8,
    cfg: model.ModelConfig,
    xfm: Transformer,
    /// The values every proof compares (`Oracle`).
    oracle: Oracle,
    tmp: testing.TmpDir,
    sides: [1]*Transformer = undefined,

    pub fn open(p: *Pair, table: *const abi.ModelTable, pk: *fixture.Pack, json: []const u8, o: Oracle) !void {
        return p.openAs(table, pk, json, null, null, o);
    }

    /// `open`, with `edit` applied to the plugin's parse.
    pub fn openWith(p: *Pair, table: *const abi.ModelTable, pk: *fixture.Pack, json: []const u8, edit: ?*const fn (*abi.Parsed) void, o: Oracle) !void {
        return p.openAs(table, pk, json, edit, null, o);
    }

    /// `open`, with `settings` applied to the config before it loads (a per-model load setting).
    pub fn openSettings(p: *Pair, table: *const abi.ModelTable, pk: *fixture.Pack, json: []const u8, settings: *const fn (*model.ModelConfig) void, o: Oracle) !void {
        return p.openAs(table, pk, json, null, settings, o);
    }

    fn openAs(p: *Pair, table: *const abi.ModelTable, pk: *fixture.Pack, json: []const u8, edit: ?*const fn (*abi.Parsed) void, settings: ?*const fn (*model.ModelConfig) void, o: Oracle) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        p.* = .{ .dir = &.{}, .cfg = undefined, .xfm = undefined, .oracle = o, .tmp = testing.tmpDir(.{}) };
        errdefer p.tmp.cleanup();
        var buf: [512]u8 = undefined;
        p.dir = try a.dupe(u8, buf[0..try p.tmp.dir.realPath(io, &buf)]);
        errdefer a.free(p.dir);
        try pk.save(p.dir);
        const b = try a.create(plugin_config.Binding);
        b.* = .{ .table = table };
        const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
        const dir = mem.packDir();
        const v = table.parse(&dir, abi.Str.of(json), .{}, null, false, &b.parsed);
        if (!v.isOk()) {
            a.destroy(b);
            std.debug.print("parse refused: {s}\n", .{v.why.slice()});
            return error.ParseRefused;
        }
        if (edit) |e| e(&b.parsed);
        p.cfg = try model.parseConfigFromJson(a, json);
        p.cfg.plugin = b;
        errdefer p.cfg.deinit(a);
        if (settings) |e| e(&p.cfg);
        p.xfm = try driver.load(io, a, &p.cfg, p.dir, fixture.stream());
    }

    pub fn close(p: *Pair) void {
        p.xfm.deinit();
        p.cfg.deinit(testing.allocator);
        testing.allocator.free(p.dir);
        p.tmp.cleanup();
    }

    /// The shell to run (a slice, as the proofs iterate it).
    pub fn shells(p: *Pair) []const *Transformer {
        p.sides = .{&p.xfm};
        return &p.sides;
    }

    /// The shell's value (`vals` in `shells` order) against the tape.
    pub fn check(p: *Pair, vals: []const Array) !void {
        try p.oracle.check(vals[vals.len - 1], false);
    }
};

/// What a proof compares: a tape recorded from the legacy forward (`tape.zig`).
pub const Oracle = struct {
    tape: tape_mod.Tape = .{},

    /// A playback read every recorded value (a proof's last call).
    pub fn done(o: *const Oracle) !void {
        try o.tape.expectDone();
    }

    /// `got` against the tape's next value.
    pub fn check(o: *Oracle, got: Array, greedy: bool) !void {
        if (o.tape.playing()) try o.tape.expectNext(got, greedy);
    }

    /// A greedy run's tokens.
    pub fn stream(o: *Oracle, got: []const u32) !void {
        const g = mlx.mlx_array_new_data(got.ptr, &[_]c_int{@intCast(got.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(g);
        try o.check(g, true);
    }
};

/// The oracle of one proof: tape `src/plugins/<family>/fixtures/<name>.tape`.
pub fn oracle(comptime family: []const u8, comptime name: []const u8) !Oracle {
    return .{ .tape = try tape_mod.Tape.player(@embedFile("../" ++ family ++ "/fixtures/" ++ name ++ ".tape")) };
}

/// One request's cache, recurrent entries and position.
pub const Lane = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    pub fn init(l: *Lane, x: *Transformer) !void {
        const n = x.config.num_hidden_layers;
        l.* = .{ .kv = try KVCache.init(testing.allocator, n), .ssm = try fixture.newEntries(n) };
        l.ctx = x.defaultCtx();
        l.ctx.cache = &l.kv;
        l.ctx.moe_seq_offset = &l.off;
        l.ctx.ssm_entries = l.ssm;
    }
    pub fn deinit(l: *Lane) void {
        l.kv.deinit();
        fixture.freeEntries(l.ssm);
    }
};

fn ids(comptime n: usize, seed: u32) [n]i32 {
    var out: [n]i32 = undefined;
    for (&out, 0..) |*v, i| v.* = @intCast((seed + 37 * @as(u32, @intCast(i))) % 256);
    return out;
}

const Step = struct { ids: []const i32, verify: bool = false, accept: ?u32 = null };

/// Prefill chunks, decode, captured verifies with partial accepts rolled back from the captures, then decode on the
/// rolled-back state. Never a rollback to a verify's last row: the GDN verify kernels leave it unwritten (a whole
/// accept keeps the state as it is), and the legacy layered forward writes it on some layers.
const steps = [_]Step{
    .{ .ids = &ids(20, 3) },
    .{ .ids = &ids(70, 11) },
    .{ .ids = &.{88} },
    .{ .ids = &.{ 5, 61, 200 }, .verify = true, .accept = 1 },
    .{ .ids = &.{12} },
    .{ .ids = &.{ 9, 33, 77, 140, 2 }, .verify = true, .accept = 3 },
    .{ .ids = &.{64} },
    .{ .ids = &.{ 6, 7, 8, 9 }, .verify = true, .accept = 0 },
    .{ .ids = &ids(6, 5) },
    .{ .ids = &.{17} },
};

/// The solo steps (every one capturing layers 0 and 1, as a DFlash drafter reads them) and the prompt-lookup round
/// loop, served vs legacy.
pub fn expectSolo(p: *Pair) !void {
    try soloSteps(p);
    try p.oracle.done();
}

fn soloSteps(p: *Pair) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    const xs = p.shells();
    var lane_buf: [2]Lane = undefined;
    const lanes = lane_buf[0..xs.len];
    for (lanes, xs) |*l, x| try l.init(x);
    defer for (lanes) |*l| l.deinit();
    const recurrent = driver.positionsOf(&p.xfm) != .kv_step;
    for (steps) |st| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const t = try fixture.tokens(&g, st.ids);
        var outs: [2]Array = undefined;
        var caps: [2][2]Array = @splat(@splat(.{ .ctx = null }));
        defer for (caps) |pair| for (pair) |c| if (c.ctx != null) {
            _ = mlx.mlx_array_free(c);
        };
        // A trunk positioned off the KV step leaves `off` to the scheduler.
        const before = if (recurrent) lanes[0].off else lanes[0].kv.step;
        for (lanes, xs, outs[0..xs.len], caps[0..xs.len]) |*l, x, *o, *cv| {
            for (cv) |*c| c.* = mlx.mlx_array_new();
            var cl: CaptureLayers = .{ .ids = &.{ 0, 1 }, .out = cv };
            l.ctx.capture_ssm_seq = st.verify;
            l.ctx.capture_layers = &cl;
            defer {
                l.ctx.capture_ssm_seq = false;
                l.ctx.capture_layers = null;
            }
            o.* = try g.adopt(try x.forwardWith(&l.ctx, t));
        }
        try p.check(outs[0..xs.len]);
        for (0..2) |c| {
            var col: [2]Array = undefined;
            for (caps[0..xs.len], col[0..xs.len]) |cv, *x| x.* = cv[c];
            try p.check(col[0..xs.len]);
        }
        // A trunk the driver positions off the KV step leaves `moe_seq_offset` to the scheduler.
        if (recurrent) try testing.expectEqual(lanes[0].off, lanes[xs.len - 1].off);
        if (st.accept) |acc| {
            for (lanes) |*l| {
                const keep = before + 1 + acc;
                const step_keep = l.kv.step;
                try l.kv.truncate(keep, s);
                // A GDN trunk positions off `moe_seq_offset` and keeps its KV step (the engine's PLD rollback).
                if (recurrent) l.kv.step = step_keep;
                for (l.ssm) |*e| {
                    try ssm_entry.ssmRollbackFromCapture(e, acc, @intCast(st.ids.len), s);
                    ssm_entry.ssmFreeSpecCapture(e);
                }
                l.off = keep;
            }
            }
    }

    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 3, 7, 1, 12, 30, 5, 9, 22, 3, 7, 1 };
    var streams: [2][24]u32 = undefined;
    for (xs, streams[0..xs.len]) |x, *out| {
        var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, x, &tok_dummy, &prompt, out.len, .{ .temperature = 0.0 }, &.{}, .{ .pld_enabled = true });
        defer gen.deinit(a);
        var n: usize = 0;
        while (try gen.nextPld(a, 4, 2)) |res| {
            defer a.free(res.tokens);
            @memcpy(out[n..][0..res.tokens.len], res.tokens);
            n += res.tokens.len;
        }
        try testing.expectEqual(out.len, n);
    }
    try p.oracle.stream(&streams[xs.len - 1]);
}

/// A greedy DFlash run on `xfm` with the tiny assistant at `asst`: every emitted token and each round's accepted count.
pub fn dflashRun(xfm: *Transformer, asst: []const u8, tokens: []u32, accepted: *std.ArrayList(u32)) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
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

/// What a DFlash drafter reads of the trunk (the embedding of five ids, the `head` readout of those rows), then a
/// greedy DFlash run with the assistant at `asst`: its tokens and each round's accepted count, served against the
/// legacy shell or the tape.
pub fn expectDflash(p: *Pair, asst: []const u8, hidden: c_int, comptime head: []const u8) !void {
    const a = testing.allocator;
    const xs = p.shells();
    try testing.expect(p.xfm.plugin.?.drafters and p.xfm.supportsLayerCapture());
    {
        var g = ops.Graph.init(a, fixture.stream());
        defer g.deinit();
        const t = try fixture.tokens(&g, &.{ 4, 90, 127, 0, 33 });
        const rows = try g.reshape(try g.adopt(try xs[0].rawEmbedding(t)), &.{ 1, 5, hidden });
        var emb: [2]Array = undefined;
        var out: [2]Array = undefined;
        for (xs, emb[0..xs.len], out[0..xs.len]) |x, *e, *o| {
            e.* = try g.adopt(try x.rawEmbedding(t));
            o.* = try g.adopt(try @field(Transformer, head)(x, rows));
        }
        try p.check(emb[0..xs.len]);
        try p.check(out[0..xs.len]);
    }
    var tokens: [2][16]u32 = undefined;
    var accepted: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&accepted) |*l| l.deinit(a);
    for (xs, 0..) |x, k| try dflashRun(x, asst, &tokens[k], &accepted[k]);
    const last = xs.len - 1;
    try p.oracle.stream(&tokens[last]);
    try p.oracle.stream(accepted[last].items);
    try p.oracle.done();
}

/// With a drafter bound (the row-exact lanes on both sides): the solo steps equal the legacy shell, and after a prompt
/// a captured verify gives every row the logits serial decode of the same tokens gives, bit for bit.
fn expectBound(p: *Pair) !void {
    for (p.shells()) |x| {
        x.config.dflash_bound = true;
        try testing.expect(x.config.rowExactDecode());
    }
    try expectSolo(p);
    for (p.shells()) |x| try expectVerifyEqualsSerial(x);
}

fn expectVerifyEqualsSerial(x: *Transformer) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    var lanes: [2]Lane = undefined;
    for (&lanes) |*l| try l.init(x);
    defer for (&lanes) |*l| l.deinit();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const prompt = try fixture.tokens(&g, &ids(19, 7));
    for (&lanes) |*l| _ = try g.adopt(try x.forwardWith(&l.ctx, prompt));
    const drafts = [_]i32{ 88, 5, 61, 200, 12, 9, 33, 77 };
    lanes[1].ctx.capture_ssm_seq = true;
    const verify = try g.adopt(try x.forwardWith(&lanes[1].ctx, try fixture.tokens(&g, &drafts)));
    const vocab = ops.shape(verify)[ops.shape(verify).len - 1];
    for (drafts, 0..) |d, i| {
        const want = try g.adopt(try x.forwardWith(&lanes[0].ctx, try fixture.tokens(&g, &.{d})));
        const r: c_int = @intCast(i);
        const row = try g.slice(verify, &.{ 0, r, 0 }, &.{ 1, r + 1, vocab });
        try ops.expectBitEqual(row, want);
    }
}

/// Two slots: batched decode ticks and a batched captured verify, served vs legacy.
pub fn expectBatched(p: *Pair) !void {
    // Off Metal the tiny pack's pipelined generation can stall in MLX's CUDA wait; served packs complete.
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = fixture.stream();
    for (p.shells()) |x| try testing.expect(x.supportsBatchedGdnDecode());
    const xs = p.shells();
    var lane_buf: [2][2]Lane = undefined;
    const lanes = lane_buf[0..xs.len];
    for (lanes, xs) |*pair, x| for (pair) |*l| try l.init(x);
    defer for (lanes) |*pair| for (pair) |*l| l.deinit();
    for ([_][]const i32{ &ids(20, 3), &ids(7, 9) }, 0..) |pr, j| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        for (lanes, xs) |*pair, x| _ = try g.adopt(try x.forwardWith(&pair[j].ctx, try fixture.tokens(&g, pr)));
    }
    for (lanes, xs) |*pair, x| try testing.expect(x.batchedGdnReady(&.{ &pair[0].ctx, &pair[1].ctx }));
    for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 } }) |next| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        var outs: [2][4]Array = undefined;
        for (lanes, xs, outs[0..xs.len]) |*pair, x, *o| {
            const ctxs = [_]*ForwardCtx{ &pair[0].ctx, &pair[1].ctx };
            const offs = [_]u32{ @intCast(pair[0].off), @intCast(pair[1].off) };
            var hidden: ?[]Array = null;
            const rows = try x.forwardMoeBatchedDecode(&next, &ctxs, &offs, &hidden);
            defer a.free(rows);
            defer a.free(hidden.?);
            o.* = .{ try g.adopt(rows[0]), try g.adopt(rows[1]), try g.adopt(hidden.?[0]), try g.adopt(hidden.?[1]) };
            for (pair) |*l| l.off += 1;
        }
        try checkRows(p, &outs, xs.len);
        try slotEntries(lanes);
    }
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const t = try g.adopt(mlx.mlx_array_new_data(&[_]i32{ 5, 61, 200, 9, 33, 77 }, &[_]c_int{ 2, 3 }, 2, .int32));
    var outs: [2][3]Array = undefined;
    for (lanes, xs, outs[0..xs.len]) |*pair, x, *o| {
        const ctxs = [_]*ForwardCtx{ &pair[0].ctx, &pair[1].ctx };
        const offs = [_]u32{ @intCast(pair[0].off), @intCast(pair[1].off) };
        var last = mlx.mlx_array_new();
        var all = mlx.mlx_array_new();
        const logits = try x.forwardMoeBatchedVerify(t, &ctxs, &offs, &last, &all);
        o.* = .{ try g.adopt(logits), try g.adopt(last), try g.adopt(all) };
    }
    try checkRows(p, &outs, xs.len);
    try slotEntries(lanes);
    try p.oracle.done();
}

/// Row `i` of each shell's outputs (`outs[side][i]`), the served side's against the legacy's or the tape.
pub fn checkRows(p: anytype, outs: anytype, sides: usize) !void {
    for (0..outs[0].len) |i| {
        var col: [2]Array = undefined;
        for (outs[0..sides], col[0..sides]) |o, *x| x.* = o[i];
        try p.check(col[0..sides]);
    }
}

/// Each slot's recurrent states, served against legacy, while recording.
fn slotEntries(lanes: []const [2]Lane) !void {
    if (lanes.len == 2) for (&lanes[0], &lanes[1]) |*w, *gl| try fixture.expectEntries(w.ssm, gl.ssm);
}

/// 4 text tokens, a 2x3 image (id 250), 3 text tokens on the M-RoPE table; the rows splice across a chunk boundary,
/// then decode and a captured verify past the table.
pub fn expectImage(p: *Pair) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    try testing.expectEqual(@as(u32, 250), p.cfg.image_token_id);
    const T = 13;
    const tokens = [T]i32{ 3, 17, 42, 99, 250, 250, 250, 250, 250, 250, 7, 8, 9 };
    var table: [3 * T]i32 = undefined;
    for (0..T) |i| {
        const img = i >= 4 and i < 10;
        const r: i32 = if (img) @intCast((i - 4) / 3) else 0;
        const c: i32 = if (img) @intCast((i - 4) % 3) else 0;
        const t: i32 = if (i < 4) @intCast(i) else if (img) 4 else @intCast(i - 10 + 7);
        table[i] = t;
        table[T + i] = if (img) t + r else t;
        table[2 * T + i] = if (img) t + c else t;
    }
    var g0 = ops.Graph.init(a, s);
    defer g0.deinit();
    const rows = try g0.astype(try g0.adopt(try fixtureRows(6, @intCast(p.cfg.hidden_size), 9)), p.cfg.actDtype());
    const xs = p.shells();
    var lane_buf: [2]Lane = undefined;
    const lanes = lane_buf[0..xs.len];
    for (lanes, xs) |*l, x| {
        try l.init(x);
        l.ctx.mrope_pos = &table;
        l.ctx.mrope_total = T;
        l.ctx.mrope_delta = 10 - T;
        l.ctx.vision_embeddings = rows;
    }
    defer for (lanes) |*l| l.deinit();
    const chunks = [_]struct { ids: []const i32, offset: usize = 0, verify: bool = false }{
        .{ .ids = tokens[0..7] },
        .{ .ids = tokens[7..], .offset = 3 },
        .{ .ids = &.{88} },
        .{ .ids = &.{ 5, 61, 200 }, .verify = true },
        .{ .ids = &.{12} },
    };
    for (chunks) |ch| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const t = try fixture.tokens(&g, ch.ids);
        var outs: [2]Array = undefined;
        for (lanes, xs, outs[0..xs.len]) |*l, x, *o| {
            l.ctx.vision_splice_offset = ch.offset;
            l.ctx.capture_ssm_seq = ch.verify;
            defer l.ctx.capture_ssm_seq = false;
            o.* = try g.adopt(try x.forwardWith(&l.ctx, t));
        }
        try p.check(outs[0..xs.len]);
    }
    try p.oracle.done();
}

/// A draft tree of 8 rows over three branches (the root, then each row's parent and depth).
const tree_parents = [_]i32{ -1, 0, 1, 1, 3, 0, 5, 2 };
const tree_depth = [_]i32{ 0, 1, 2, 2, 3, 1, 2, 3 };
const tree_toks = [_]i32{ 88, 5, 61, 200, 12, 9, 33, 77 };
/// The accepted path the commit keeps: rows 0, 1, 3, 4.
const tree_kept = [_]u32{ 0, 1, 3, 4 };

/// Row `r`'s tokens from the root down to it.
fn treePath(r: usize, out: *[4]i32) []const i32 {
    const d: usize = @intCast(tree_depth[r]);
    var cur: i32 = @intCast(r);
    var i = d + 1;
    while (cur >= 0) : (cur = tree_parents[@intCast(cur)]) {
        i -= 1;
        out[i] = tree_toks[@intCast(cur)];
    }
    return out[0 .. d + 1];
}

/// With a drafter bound: a draft-tree verify (captures on) and the commit of an accepted path, served vs legacy, bit
/// for bit (logits, captures, every state the commit replays from, the committed states, the next step). With `exact`
/// every tree row equals serial decode along its own ancestors, and the committed state the serial state after the
/// kept path (a bf16 matmul's bits follow its row count).
fn expectTree(p: *Pair, exact: bool) !void {
    try treeSteps(p, exact);
    try p.oracle.done();
}

fn treeSteps(p: *Pair, exact: bool) !void {
    const a = testing.allocator;
    const s = fixture.stream();
    const xs = p.shells();
    for (xs) |x| try testing.expect(x.specTreeSupported());
    var lane_buf: [2]Lane = undefined;
    const lanes = lane_buf[0..xs.len];
    for (lanes, xs) |*l, x| try l.init(x);
    defer for (lanes) |*l| l.deinit();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const prompt = try fixture.tokens(&g, &ids(19, 7));
    for (lanes, xs) |*l, x| _ = try g.adopt(try x.forwardWith(&l.ctx, prompt));
    const anchor = lanes[0].off;
    const last = xs.len - 1;

    const tree = tree_mod.specTreeFor(&tree_parents, &tree_depth, 3);
    defer tree.deinit();
    var outs: [2]Array = undefined;
    var caps: [2][2]Array = @splat(@splat(.{ .ctx = null }));
    defer for (caps) |pair| for (pair) |c| if (c.ctx != null) {
        _ = mlx.mlx_array_free(c);
    };
    for (lanes, xs, outs[0..xs.len], caps[0..xs.len]) |*l, x, *o, *cv| {
        for (cv) |*c| c.* = mlx.mlx_array_new();
        var cl: CaptureLayers = .{ .ids = &.{ 0, 1 }, .out = cv };
        l.ctx.capture_layers = &cl;
        l.ctx.capture_ssm_seq = true;
        l.ctx.tree = &tree;
        // As `dflash_tree.zig` runs it: the residual handed to the GPU between layers.
        l.ctx.pipeline_build = 1;
        defer {
            l.ctx.capture_layers = null;
            l.ctx.capture_ssm_seq = false;
            l.ctx.tree = null;
            l.ctx.pipeline_build = 0;
        }
        o.* = try g.adopt(try x.forwardWith(&l.ctx, try fixture.tokens(&g, &tree_toks)));
    }
    try p.check(outs[0..xs.len]);
    try checkRows(p, &caps, xs.len);

    if (xs.len == 2) try expectTreeCaptures(lanes[0].ssm, lanes[1].ssm);
    try testing.expect(lanes[last].ssm[0].spec_prework.k.ctx != null);

    // Every row against serial decode along its ancestors, from the prompt's state.
    const vocab = ops.shape(outs[last])[2];
    for (0..if (exact) tree_toks.len else 0) |r| {
        var buf: [4]i32 = undefined;
        const path = treePath(r, &buf);
        var ser: Lane = undefined;
        try ser.init(&p.xfm);
        defer ser.deinit();
        _ = try g.adopt(try p.xfm.forwardWith(&ser.ctx, prompt));
        var serial: Array = undefined;
        for (path) |t| serial = try g.adopt(try p.xfm.forwardWith(&ser.ctx, try fixture.tokens(&g, &.{t})));
        const rc: c_int = @intCast(r);
        try ops.expectBitEqual(try g.slice(outs[last], &.{ 0, rc, 0 }, &.{ 1, rc + 1, vocab }), serial);
    }

    // The commit (`dflash_tree.zig`): the kept rows' KV moved to consecutive positions, the states replayed.
    const accepted = tree_kept.len - 1;
    var conv_rows: [3]i32 = undefined;
    for (0..3) |i| {
        const dd: i32 = @as(i32, @intCast(accepted)) - 2 + @as(i32, @intCast(i));
        conv_rows[i] = if (dd < 0) dd + 3 else 3 + @as(i32, @intCast(tree_kept[@intCast(dd)]));
    }
    var kept: [tree_kept.len]i32 = undefined;
    for (tree_kept, &kept) |r, *k| k.* = @intCast(r);
    for (lanes) |*l| {
        const step = l.kv.step;
        try l.kv.compactRows(anchor, tree_kept[1..], s);
        try l.kv.truncate(anchor + tree_kept.len, s);
        l.kv.step = step;
        for (l.ssm) |*e| {
            try ssm_entry.ssmCommitTreePath(e, &kept, conv_rows, s);
            ssm_entry.ssmFreeSpecCapture(e);
        }
        l.off = anchor + tree_kept.len;
    }

    var next: [2]Array = undefined;
    for (lanes, xs, next[0..xs.len]) |*l, x, *o| o.* = try g.adopt(try x.forwardWith(&l.ctx, try fixture.tokens(&g, &.{17})));
    try p.check(next[0..xs.len]);
    if (!exact) return;
    var ser: Lane = undefined;
    try ser.init(&p.xfm);
    defer ser.deinit();
    _ = try g.adopt(try p.xfm.forwardWith(&ser.ctx, prompt));
    for (tree_kept) |r| _ = try g.adopt(try p.xfm.forwardWith(&ser.ctx, try fixture.tokens(&g, &.{tree_toks[r]})));
    try ops.expectBitEqual(next[last], try g.adopt(try p.xfm.forwardWith(&ser.ctx, try fixture.tokens(&g, &.{17}))));
}

/// What a tree verify leaves for the commit to replay from (the round's input state, every row's prework), bit-equal.
fn expectTreeCaptures(want: []const SSMCacheEntry, got: []const SSMCacheEntry) !void {
    for (want, got) |*w, *gt| for (
        [_]Array{ w.spec_state_in, w.spec_prework.k, w.spec_prework.v, w.spec_prework.gb },
        [_]Array{ gt.spec_state_in, gt.spec_prework.k, gt.spec_prework.v, gt.spec_prework.gb },
    ) |wa, ga| {
        try testing.expectEqual(wa.ctx == null, ga.ctx == null);
        if (wa.ctx != null) try ops.expectBitEqual(ga, wa);
    };
}

/// The dense pack (4-bit or bf16), a drafter bound on every shell; tape `fixtures/tree-<seed>.tape`.
fn openBound(p: *Pair, comptime quantized: bool, comptime seed: u64) !void {
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try pack.writePack(&pk, false, quantized, 8);
    try p.open(&qwen3_5.table, &pk, pack.config(false, quantized, 8), try oracle("qwen3_5", std.fmt.comptimePrint("tree-{x}", .{seed})));
    for (p.shells()) |x| x.config.dflash_bound = true;
}

pub fn fixtureRows(n: c_int, h: c_int, seed: u64) !Array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, @intCast(n * h));
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    return mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, n, h }, 3, .float32);
}

fn openPack(p: *Pair, comptime moe: bool, comptime quantized: bool, comptime experts: c_int, comptime seed: u64, json: []const u8) !void {
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try pack.writePack(&pk, moe, quantized, experts);
    try p.open(&qwen3_5.table, &pk, json, try oracle("qwen3_5", std.fmt.comptimePrint("served-{x}", .{seed})));
}

/// The pack's config with an image tower: id 250 and interleaved M-RoPE sections [11,11,10].
fn vlConfig(comptime moe: bool) ![]u8 {
    const a = testing.allocator;
    const text = pack.config(moe, true, 8);
    const json = try std.mem.concat(a, u8, &.{ "{\"image_token_id\":250,\"vision_config\":{\"depth\":1},", text[1..] });
    defer a.free(json);
    return std.mem.replaceOwned(u8, a, json, "\"partial_rotary_factor\":0.25}", "\"partial_rotary_factor\":0.25,\"mrope_section\":[11,11,10],\"mrope_interleaved\":true}");
}

test "qwen3_5 served: dense 4-bit steps, rollbacks and the lookup loop equal the legacy shell" {
    var p: Pair = undefined;
    try openPack(&p, false, true, 8, 0x5e01, pack.config(false, true, 8));
    defer p.close();
    try testing.expectEqual(driver.Positions.gdn, driver.positionsOf(&p.xfm));
    try expectSolo(&p);
}

test "qwen3_5 served: a chat pack embeds a padded batch as it embeds each sequence alone" {
    var p: Pair = undefined;
    try openPack(&p, false, true, 8, 0x5e01, pack.config(false, true, 8));
    defer p.close();
    try @import("../../engines/mlx/plugin/both.zig").expectEmbeddingBatch(&p.xfm);
}

test "qwen3_5 served: a qwen3_5_moe 4-bit pack's steps and lookup loop equal the legacy shell" {
    var p: Pair = undefined;
    try openPack(&p, true, true, 64, 0x5e02, pack.config(true, true, 64));
    defer p.close();
    try expectSolo(&p);
}

test "qwen3_5 served: a bf16 pack (decode side copies) equals the legacy shell" {
    var p: Pair = undefined;
    try openPack(&p, false, false, 8, 0x5e03, pack.config(false, false, 8));
    defer p.close();
    try expectSolo(&p);
}

test "qwen3_5 served: lm_head_mxfp8 (LOSSY) quantizes a bf16 pack's head as the legacy load does, steps equal" {
    const cli = &@import("../../engines/mlx/transformer.zig").cli().lm_head_mxfp8_cli;
    const was = cli.*;
    defer cli.* = was;
    cli.* = true;
    var p: Pair = undefined;
    try openPack(&p, false, false, 8, 0x5e0a, pack.config(false, false, 8));
    defer p.close();
    try testing.expect(p.xfm.lm_head_s.ctx != null);
    try expectSolo(&p);
}

test "qwen3_5 served: dense and MoE batched decode ticks and a batched verify equal the legacy GDN batched path" {
    inline for (.{ false, true }, .{ 0x5e04, 0x5e05 }) |moe, seed| {
        var p: Pair = undefined;
        try openPack(&p, moe, true, 8, seed, pack.config(moe, true, 8));
        defer p.close();
        try expectBatched(&p);
    }
}

test "qwen3_5 served: an image prompt on the M-RoPE table (dense and MoE) equals the legacy shell" {
    inline for (.{ false, true }, .{ 0x5e06, 0x5e07 }) |moe, seed| {
        const json = try vlConfig(moe);
        defer testing.allocator.free(json);
        var p: Pair = undefined;
        try openPack(&p, moe, true, 8, seed, json);
        defer p.close();
        try expectImage(&p);
    }
}

test "qwen3_5 served: with a drafter bound, the row-exact lanes equal the legacy rowExactDecode and a verify row equals serial decode" {
    var p: Pair = undefined;
    try openPack(&p, false, true, 8, 0x5e08, pack.config(false, true, 8));
    defer p.close();
    try testing.expect(p.xfm.plugin.?.drafters);
    try testing.expectEqual(.mma, @as(*qwen3_5.Loaded, @ptrCast(@alignCast(p.xfm.plugin.?.model))).body.dense.base.row_exact);
    try expectBound(&p);
}

test "qwen3_5 served: a bf16 pack takes DFlash without exact_drafted (its matmul bits follow the row count)" {
    var p: Pair = undefined;
    try openPack(&p, false, false, 8, 0x5e09, pack.config(false, false, 8));
    defer p.close();
    try testing.expect(p.xfm.plugin.?.drafters and !p.cfg.plugin.?.parsed.caps.exact_drafted);
}

test "qwen3_5 served: with a drafter bound, a draft-tree verify and its commit equal the legacy trunk and serial decode along each row's ancestors" {
    var p: Pair = undefined;
    try openBound(&p, true, 0x5e0a);
    defer p.close();
    try testing.expect(p.cfg.plugin.?.parsed.caps.spec_tree);
    try expectTree(&p, true);
}

test "qwen3_5 served: a bf16 dense pack verifies draft trees as the legacy trunk does" {
    var p: Pair = undefined;
    try openBound(&p, false, 0x5e0c);
    defer p.close();
    try expectTree(&p, false);
}

test "qwen3_5 served: trunk projections tiled in place for the lane give the legacy tiled trunk's bits (steps, verify, tree)" {
    if (!naxAvailable()) return error.SkipZigTest;
    var p: Pair = undefined;
    try openBound(&p, true, 0x5e0b);
    defer p.close();
    _ = try p.xfm.tileLaneWeights();
    const m: *qwen3_5.Loaded = @ptrCast(@alignCast(p.xfm.plugin.?.model));
    const o = &m.body.dense.layers[1].mixer.second.o.lin;
    var g = ops.Graph.init(testing.allocator, fixture.stream());
    defer g.deinit();
    const x = try g.astype(try g.adopt(try fixtureRows(1, ops.shape(o.w)[1] * 8, 3)), .bfloat16);
    try testing.expect(try lane_qmm.tiledQmm(x, o.w, fixture.stream()) != null);
    try soloSteps(&p);
    for (p.shells()) |sh| try expectVerifyEqualsSerial(sh);
    try expectTree(&p, true);
}
