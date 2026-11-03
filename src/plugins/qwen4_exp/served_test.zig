//! The served path end to end: a tiny qwen4_exp pack with its MTP head loaded as the scheduler loads it, through the
//! plugin table behind the driver's shell (`parseConfig`, `driver.load`, the MTP head on the plugin's trunk through
//! `plugin/intree.zig`). Captured steps, the pipelined greedy decode (deferred n-gram PLE settled through
//! `forward_settle`), batched rows, a batched verify and MTP rounds (pinned and auto depth) emit the tokens, accepted
//! counts and logits the legacy path recorded on the tapes, with the host and the GPU n-gram gather.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const intree = @import("../../engines/mlx/plugin/intree.zig");
const depth_bounds = @import("../../engines/mlx/spec/mtp_depth_bounds.zig");
const lazyForward = @import("../../engines/mlx/generate/step.zig").lazyForward;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const qwen4 = @import("plugin.zig");
const ple_gpu = @import("ple_gpu.zig");
const fixture = @import("fixture.zig");
const Oracle = @import("oracle_test.zig").Oracle;

const testing = std.testing;
const Array = mlx.mlx_array;

/// The pack loaded as the scheduler loads it, with the steps the scheduler's load runs after the trunk.
const Pair = struct {
    tmp: testing.TmpDir,
    dir_buf: [std.fs.max_path_bytes]u8 = undefined,
    dir_len: usize = 0,
    scfg: model.ModelConfig,
    served: Transformer,

    fn open(precision: fixture.Precision, seed: u64) !*Pair {
        return openWith(precision, seed, .{});
    }

    fn openWith(precision: fixture.Precision, seed: u64, opts: fixture.Opts) !*Pair {
        const a = testing.allocator;
        const io = testing.io;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const files = try fixture.filesWith(arena.allocator(), seed, precision, opts);
        const p = try a.create(Pair);
        errdefer a.destroy(p);
        p.* = .{ .tmp = testing.tmpDir(.{}), .scfg = undefined, .served = undefined };
        errdefer p.tmp.cleanup();
        for (files) |f| try p.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        p.dir_len = try p.tmp.dir.realPath(io, &p.dir_buf);
        p.scfg = try model.parseConfig(io, a, p.dir());
        if (p.scfg.plugin == null) return error.NotServed;
        errdefer p.scfg.deinit(a);
        p.served = try driver.load(io, a, &p.scfg, p.dir(), mlx.gpuStream());
        errdefer p.served.deinit();
        const x = intree.inner(&p.served) orelse return error.NoInnerTrunk;
        try x.qwen4MtpApplyKvQuant(.dense);
        _ = x.qwen4BuildDraftRerank();
        return p;
    }

    fn dir(p: *Pair) []const u8 {
        return p.dir_buf[0..p.dir_len];
    }

    fn close(p: *Pair) void {
        p.served.deinit();
        p.scfg.deinit(testing.allocator);
        p.tmp.cleanup();
        testing.allocator.destroy(p);
    }
};

/// A slot's engine caches and context.
const Slot = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    off: usize = 0,
    ctx: ForwardCtx = undefined,

    fn init(s: *Slot, layers: u32) !void {
        const ssm = try testing.allocator.alloc(SSMCacheEntry, layers);
        const gen = nextSsmMemberGen();
        for (ssm) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false, .member_gen = gen };
        s.* = .{ .kv = try KVCache.init(testing.allocator, layers), .ssm = ssm };
        s.ctx = .{ .cache = &s.kv, .moe_seq_offset = &s.off, .ssm_entries = s.ssm, .ssm_member_gen = gen, .capture_hidden = null, .vision_embeddings = null };
    }

    fn deinit(s: *Slot, x: *Transformer) void {
        x.discardDeferredPle(&s.ctx);
        @import("../../engines/mlx/plugin/settle.zig").release(x, &s.ctx);
        x.qwen4Trunk().?.ssmGroupDrop(s.ssm);
        for (s.ssm) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        testing.allocator.free(s.ssm);
        s.kv.deinit();
    }
};

fn tokens(g: *ops.Graph, ids: []const i32) !Array {
    return g.adopt(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32));
}

/// A request's emitted tokens and, with MTP, each round's accepted count.
const Run = struct { tokens: [24]u32 = @splat(0), n: usize = 0, accepted: std.ArrayList(u32) = .empty };

fn generate(x: *Transformer, head: ?MtpHeadRef, run: *Run) !void {
    const a = testing.allocator;
    try x.resetCache();
    var tok: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28, 40, 41, 3, 7 };
    var gen = try Generator.initWithOptions(testing.io, a, x, &tok, &prompt, run.tokens.len, .{ .temperature = 0.0 }, &.{}, .{
        .mtp_enabled = head != null,
        .mtp = head,
        .model_has_mtp = head != null,
    });
    defer gen.deinit(a);
    if (head == null) {
        while (run.n < run.tokens.len) : (run.n += 1) run.tokens[run.n] = (try gen.next(a)) orelse break;
        return;
    }
    while (try gen.nextMtp(a)) |res| {
        defer a.free(res.tokens);
        @memcpy(run.tokens[run.n..][0..res.tokens.len], res.tokens);
        run.n += res.tokens.len;
        try run.accepted.append(a, res.accepted_tokens);
    }
    try testing.expect(gen.mtp_attempted > 0);
}

/// Returns the drafts the plugin's rounds accepted.
fn expectSameRuns(p: *Pair, mtp: bool, oracle: *Oracle) !u32 {
    const head: ?MtpHeadRef = if (mtp) .{ .qwen4 = p.served.qwen4Trunk().? } else null;
    var run: Run = .{};
    defer run.accepted.deinit(testing.allocator);
    try generate(&p.served, head, &run);
    try testing.expect(run.n > 1);
    try oracle.checkIds(run.tokens[0..run.n], null);
    if (mtp) try oracle.checkIds(run.accepted.items, null);
    var sum: u32 = 0;
    for (run.accepted.items) |n| sum += n;
    return sum;
}

// ── tests ──

test "qwen4_exp served: the plugin claims the pack and the engine config is the legacy parse" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const p = try Pair.open(.q4, 31);
    defer p.close();
    var lcfg = try model.parseConfigLegacy(testing.io, testing.allocator, p.dir());
    defer lcfg.deinit(testing.allocator);
    // The load settles what the weights decide (the norm convention), on both configs.
    var w = try model.loadModelWeights(testing.io, testing.allocator, p.dir(), &lcfg, false);
    w.deinit();
    var cfg = p.scfg;
    cfg.plugin = null;
    try testing.expectEqualDeep(lcfg, cfg);
    try testing.expect(p.served.plugin != null and p.served.qwen4Trunk().?.qwen4_mtp != null);
    // load.zig compiles nothing on a served shell, and every test here forwards through shells like this one.
    const x = &p.served;
    try testing.expect(x.compiled_gdn_gate == null and x.compiled_moe_routing == null and x.compiled_gelu == null);
}

test "qwen4_exp served: captured steps and a pipelined greedy decode equal the legacy path bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    for ([_]fixture.Precision{ .bf16, .q4 }) |prec| {
        const p = try Pair.open(prec, 41);
        defer p.close();
        var oracle = try if (prec == .bf16) Oracle.open("served-steps-bf16") else Oracle.open("served-steps-q4");
        const x = &p.served;
        var slot: Slot = undefined;
        try slot.init(p.scfg.num_hidden_layers);
        defer slot.deinit(x);
        const steps = [_]struct { ids: []const i32, verify: bool = false }{
            .{ .ids = &.{ 3, 17, 42, 99, 7, 50, 1, 8, 66, 31 } },
            .{ .ids = &.{88} },
            .{ .ids = &.{ 5, 61, 100 }, .verify = true },
            .{ .ids = &.{ 9, 33, 77, 120, 2, 18, 19, 20, 21 } },
            .{ .ids = &.{64} },
        };
        for (steps) |st| {
            var g = ops.Graph.init(a, mlx.gpuStream());
            defer g.deinit();
            var last = mlx.mlx_array_new();
            var all = mlx.mlx_array_new();
            slot.ctx.capture_ssm_seq = st.verify;
            slot.ctx.spec_verify = st.verify;
            defer {
                slot.ctx.capture_ssm_seq = false;
                slot.ctx.spec_verify = false;
            }
            const l = try g.adopt(try x.forwardWithCaptureAll(&slot.ctx, try tokens(&g, st.ids), &last, &all));
            _ = try g.adopt(last);
            try oracle.check(l, null, false);
            try oracle.check(try g.adopt(all), null, false);
        }
        // The pipelined step: each next token a lazy argmax, the n-gram PLE deferred past the build.
        var toks: [6]u32 = undefined;
        {
            var g = ops.Graph.init(a, mlx.gpuStream());
            defer g.deinit();
            var logits = try g.adopt(try x.forwardWith(&slot.ctx, try tokens(&g, &.{ 11, 12 })));
            for (&toks) |*t| {
                const sh = ops.shape(logits);
                const next = try g.argmax(try g.reshape(try g.sliceAxis(logits, 1, sh[1] - 1, sh[1]), &.{ 1, sh[2] }), -1, false);
                logits = try g.adopt(try lazyForward(x, &slot.ctx, next));
                const host = try g.astype(next, .uint32);
                try mlx.check(mlx.mlx_array_eval(host));
                t.* = mlx.mlx_array_data_uint32(host).?[0];
            }
            try x.flushDeferredPle(&slot.ctx);
        }
        try oracle.checkIds(&toks, null);
        try oracle.close();
    }
}

test "qwen4_exp served: batched rows and a batched verify equal the legacy tick per slot" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const p = try Pair.open(.q4, 43);
    defer p.close();
    var oracle = try Oracle.open("served-batched");
    const x = &p.served;
    var slots: [2]Slot = undefined;
    for (&slots) |*s| try s.init(p.scfg.num_hidden_layers);
    defer for (&slots) |*s| s.deinit(x);
    const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23, 42, 9, 10, 11 }, &.{ 42, 101, 7, 1, 2 } };
    for (prompts, 0..) |pr, i| {
        var g = ops.Graph.init(a, mlx.gpuStream());
        defer g.deinit();
        try oracle.check(try g.adopt(try x.forwardWith(&slots[i].ctx, try tokens(&g, pr))), null, false);
    }
    const ctxs = [_]*ForwardCtx{ &slots[0].ctx, &slots[1].ctx };
    for ([_][2]u32{ .{ 77, 3 }, .{ 100, 64 }, .{ 11, 12 } }) |next| {
        var g = ops.Graph.init(a, mlx.gpuStream());
        defer g.deinit();
        var hidden: ?[]Array = null;
        const offs = [_]u32{ @intCast(slots[0].off), @intCast(slots[1].off) };
        const rows = try x.forwardMoeBatchedDecode(&next, &ctxs, &offs, &hidden);
        for (rows) |r| _ = try g.adopt(r);
        for (hidden.?) |h| _ = try g.adopt(h);
        for (&slots) |*s| s.off += 1;
        defer {
            a.free(rows);
            a.free(hidden.?);
        }
        for (rows, hidden.?) |r, h| {
            try oracle.check(r, null, false);
            try oracle.check(h, null, false);
        }
    }
    var g = ops.Graph.init(a, mlx.gpuStream());
    defer g.deinit();
    const ids = [_]i32{ 5, 6, 7, 8, 9, 10 };
    const offs = [_]u32{ @intCast(slots[0].off), @intCast(slots[1].off) };
    var last = mlx.mlx_array_new();
    var all = mlx.mlx_array_new();
    const arr = try g.adopt(mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 3 }, 2, .int32));
    const l = try g.adopt(try x.forwardMoeBatchedVerify(arr, &ctxs, &offs, &last, &all));
    for ([_]Array{ l, try g.adopt(last), try g.adopt(all) }) |got| try oracle.check(got, null, false);
    try oracle.close();
}

test "qwen4_exp served: greedy decode and MTP rounds emit the legacy tokens and accepted counts" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    // A random head drafts what the trunk rejects (every round rolls back); a sign-reading lm_head makes the head
    // agree often, so rounds accept all, some or none of their drafts.
    // The GPU n-gram gather (`--ple-gpu`) leaves the history owed past each decode step (`Owed.history`).
    defer ple_gpu.pleGpu().enabled = false;
    for ([_]u64{ 47, 51, 51 }, [_]fixture.Opts{ .{}, .{ .sign_head = true }, .{ .sign_head = true } }, [_]bool{ false, false, true }, 0..) |seed, opts, gpu, k| {
        // The GPU arm keeps `HEADROOM` past the table: a 16 GB Mac's working set never has it.
        if (gpu and mlx.maxRecommendedWorkingSet() <= ple_gpu.HEADROOM) continue;
        ple_gpu.pleGpu().enabled = gpu;
        var oracle = try switch (k) {
            0 => Oracle.open("served-runs-0"),
            1 => Oracle.open("served-runs-1"),
            else => Oracle.open("served-runs-2"),
        };
        const p = try Pair.openWith(.q4, seed, opts);
        defer p.close();
        if (gpu) try testing.expect(p.served.qwen4Trunk().?.qwen4.?.gpu != null);
        _ = try expectSameRuns(p, false, &oracle);
        const auto = try expectSameRuns(p, true, &oracle);
        depth_bounds.active = .{ .min = 2, .max = 2 };
        defer depth_bounds.active = .{};
        const pinned = try expectSameRuns(p, true, &oracle);
        if (opts.sign_head) try testing.expect(auto > 0 and pinned > 0);
        try oracle.close();
    }
}

test "qwen4_exp served: an image prompt (M-RoPE table, rows spliced across a chunk boundary) equals the legacy path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const p = try Pair.openWith(.q4, 53, .{ .vision = true });
    defer p.close();
    var oracle = try Oracle.open("served-image");
    try testing.expectEqual(@as(u32, fixture.image_id), p.scfg.image_token_id);
    try testing.expect(p.scfg.plugin.?.parsed.mrope.rope_dims == 8);
    // 4 text tokens, a 2x3 image, 6 text tokens: past the image the positions resume after its widest axis.
    const T = 16;
    const img = fixture.image_id;
    const prompt = [T]i32{ 3, 17, 42, 99, img, img, img, img, img, img, 7, 8, 9, 10, 11, 12 };
    var table: [3 * T]i32 = undefined;
    for (0..T) |i| {
        const in_img = i >= 4 and i < 10;
        const r: i32 = if (in_img) @intCast((i - 4) / 3) else 0;
        const c: i32 = if (in_img) @intCast((i - 4) % 3) else 0;
        const t: i32 = if (i < 4) @intCast(i) else if (in_img) 4 else @intCast(i - 10 + 7);
        table[i] = t;
        table[T + i] = if (in_img) t + r else t;
        table[2 * T + i] = if (in_img) t + c else t;
    }
    var prng = std.Random.DefaultPrng.init(9);
    var buf: [6 * 64]f32 = undefined;
    for (&buf) |*v| v.* = prng.random().floatNorm(f32) * 0.5;
    var g0 = ops.Graph.init(a, mlx.gpuStream());
    defer g0.deinit();
    const rows = try g0.astype(try g0.adopt(mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, 6, 64 }, 3, .float32)), .bfloat16);
    const x = &p.served;
    var slot: Slot = undefined;
    try slot.init(p.scfg.num_hidden_layers);
    slot.ctx.mrope_pos = &table;
    slot.ctx.mrope_total = T;
    slot.ctx.mrope_delta = 13 - T;
    slot.ctx.vision_embeddings = rows;
    defer slot.deinit(x);
    const chunks = [_]struct { ids: []const i32, offset: usize = 0, verify: bool = false }{
        .{ .ids = prompt[0..7] },
        .{ .ids = prompt[7..], .offset = 3 },
        .{ .ids = &.{88} },
        .{ .ids = &.{ 5, 61, 100 }, .verify = true },
        .{ .ids = &.{12} },
    };
    for (chunks) |ch| {
        var g = ops.Graph.init(a, mlx.gpuStream());
        defer g.deinit();
        slot.ctx.vision_splice_offset = ch.offset;
        slot.ctx.capture_ssm_seq = ch.verify;
        slot.ctx.spec_verify = ch.verify;
        defer {
            slot.ctx.capture_ssm_seq = false;
            slot.ctx.spec_verify = false;
        }
        try oracle.check(try g.adopt(try x.forwardWith(&slot.ctx, try tokens(&g, ch.ids))), null, false);
    }
    try oracle.close();
}
