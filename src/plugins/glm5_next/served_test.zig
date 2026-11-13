//! Served proofs of the KDA hybrids: the pack through the driver's shell on the legacy parse's config against the
//! deleted legacy forward's values (`engines/mlx/plugin/both.zig`, replayed from `fixtures/*.tape`): steps past a
//! chunk with every layer captured, the pipelined greedy decode and media rows (the batched tick is
//! `proof_test.zig`'s); GLM's MTP head through `MtpHeadRef.plugin` against the legacy `.glm` arm's recorded values,
//! its engine calls and a pinned-depth round loop.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const proof_fixture = @import("../../sdk/bases/proof/fixture.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const plugin_head = @import("../../engines/mlx/plugin/head.zig");
const mtpRun = @import("../../engines/mlx/plugin/proof_mtp.zig").mtpRun;
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const driver = @import("../../engines/mlx/plugin/driver.zig");
const File = @import("../../sdk/root.zig").fixture.File;
const fixture = @import("fixture.zig");
const glm = @import("plugin.zig");
const host = @import("host.zig");
const trunk = @import("trunk.zig");
const checkpoint = @import("../../engines/mlx/ssm/checkpoint.zig");
const ssmRestore = @import("../../engines/mlx/ssm/entry.zig").ssmRestore;
const ling = @import("../bailing_hybrid/plugin.zig");
const tape_compact_over = @import("proof_test.zig").tape_compact_over;
const kda_rows = @import("../../sdk/blocks/kda/verify_rows.zig");

const testing = std.testing;

/// The fixture's safetensors as a proof `Pack` (the arrays `Both` writes back out).
fn packOf(files: []const File) !proof_fixture.Pack {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = files[1].bytes });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir}, 0);
    defer testing.allocator.free(path);
    var p = proof_fixture.Pack.init(0);
    errdefer p.deinit();
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_load_safetensors(&p.map, &meta, path.ptr, mlx.weightLoadStream()));
    // The load is lazy and the file goes with `tmp`: read every tensor now.
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    const it = mlx.mlx_map_string_to_array_iterator_new(p.map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        if (mlx.mlx_map_string_to_array_iterator_next(&key, &v, it) != 0 or key == null) break;
        _ = mlx.mlx_vector_array_append_value(vec, v);
    }
    try mlx.check(mlx.mlx_eval(vec));
    return p;
}

/// The tape at `rel` (under `src/plugins/`).
fn oracle(comptime rel: []const u8) both.Oracle {
    return .{ .tape = @embedFile("../" ++ rel) };
}

fn glmServed(files: []const File, comptime tape: []const u8) !void {
    var p = try packOf(files);
    defer p.deinit();
    var b = try both.Both.openAs(&glm.table, &p, files[0].bytes, true, oracle(tape));
    defer b.close();
    b.tape.compact_over = tape_compact_over;
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectMedia(500, fixture.glm.H);
    try b.expectDone();
}

fn rows(g: *ops.Graph, n: c_int, seed: u64) !mlx.mlx_array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, @intCast(n * fixture.glm.H));
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    return g.astype(try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, n, fixture.glm.H }, 3, .float32)), .bfloat16);
}

// ── tests ──

test "glm5_next served: a 4-bit pack through the driver on the legacy config is bit-equal, greedy decode and media too" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try glmServed(try fixture.glmFiles(arena.allocator(), 7), "glm5_next/fixtures/served.tape");
}

test "glm5_next served: a Sushi pack (EXL3 experts) through the driver is bit-equal, greedy decode and media too" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try glmServed(try fixture.glmSushiFiles(arena.allocator(), 7), "glm5_next/fixtures/served_sushi.tape");
}

test "bailing_hybrid served: a 4-bit pack through the driver on the legacy config is bit-equal, greedy decode too" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.lingFiles(arena.allocator(), 11);
    var p = try packOf(files);
    defer p.deinit();
    var b = try both.Both.openAs(&ling.table, &p, files[0].bytes, true, oracle("bailing_hybrid/fixtures/served.tape"));
    defer b.close();
    b.tape.compact_over = tape_compact_over;
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

// Bar: the engine drives the served head as it drives the legacy `.glm` head: the same readout built, history, a
// multi-row `.logits` step, a `.mixed` step and its draft pick bit-equal, and a pinned-depth round loop with equal
// tokens and accepted counts.
test "glm5_next served: the MTP head through MtpHeadRef.plugin equals the legacy .glm arm" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const files = try fixture.glmFiles(arena.allocator(), 7);
    var p = try packOf(files);
    defer p.deinit();
    var b = try both.Both.openAs(&glm.table, &p, files[0].bytes, true, oracle("glm5_next/fixtures/served_head.tape"));
    defer b.close();
    b.tape.compact_over = tape_compact_over;
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    // A GLM vocab is past the coarse readout's floor: the served head builds it and keeps no low-bit draft head.
    try testing.expect(served.rerank_coarse != null and served.draft_head == null);
    const h: MtpHeadRef = .{ .plugin = served };
    const x = &b.xfm;
    {
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        const hist = try rows(&g, 3, 0x91);
        const step = try rows(&g, 2, 0x92);
        const one = try rows(&g, 1, 0x93);
        var cr = try h.makeCache(a);
        defer cr.deinit();
        try h.appendHistory(x, &cr, &.{ 5, 9, 40 }, hist, 0, null, a);
        const merged = try h.forward(x, &cr, try g.fromSlice(i32, &.{ 7, 11 }, &.{2}), step, 3, .logits, null, null);
        const mixed = try h.forward(x, &cr, try g.fromSlice(i32, &.{60}, &.{1}), one, 5, .mixed, null, null);
        if (mixed.rerank_x.ctx != null) _ = try g.adopt(mixed.rerank_x);
        const outs = [_]mlx.mlx_array{ try g.adopt(merged.logits), try g.adopt(merged.hidden_next), try g.adopt(mixed.hidden_next), try g.adopt(try h.draftSelect(x, mixed.hidden_next, null)) };
        try testing.expectEqual(@as(usize, 6), cr.step());
        for (outs) |got| try b.check(got, false);
    }
    var tokens: [16]u32 = undefined;
    var accepted: std.ArrayList(u32) = .empty;
    defer accepted.deinit(a);
    try mtpRun(x, h, &tokens, &accepted);
    const got_t = mlx.mlx_array_new_data(&tokens, &[_]c_int{16}, 1, .uint32);
    defer _ = mlx.mlx_array_free(got_t);
    try b.check(got_t, true);
    const got_a = mlx.mlx_array_new_data(accepted.items.ptr, &[_]c_int{@intCast(accepted.items.len)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(got_a);
    try b.check(got_a, true);
    try b.expectDone();
}

/// A quant table claiming the Ling streaming fixture: every projection stays the pack's own (`.weight`), and its bank's
/// routed sum is zero; it counts the phases, settles and geometry the trunk hands it.
const ToyLing = struct {
    const outside = @import("../../sdk/quant/outside.zig");
    const qa = @import("../../abi/quant.zig");
    const common = @import("../../abi/common.zig");
    var phases: u32 = 0;
    var settles: u32 = 0;
    var opened: u32 = 0;
    var layers: u32 = 0;
    var limits: bool = false;
    var aheads: u32 = 0;
    var ahead_ids: usize = 0;
    fn claims(block: qa.Str, _: qa.Str) callconv(.c) common.Claim {
        return if (std.mem.indexOf(u8, block.slice(), "\"exl3\"") != null) .by_marker else .no;
    }
    fn validate(_: qa.Str, _: qa.Str, _: *const common.PackDir) callconv(.c) common.Verdict {
        return .ok;
    }
    fn bill(_: qa.Str, _: qa.Str, stored: u64) callconv(.c) u64 {
        return stored;
    }
    fn open(_: *qa.LoadCtx) callconv(.c) common.Verdict {
        return .ok;
    }
    fn close(_: *qa.LoadCtx) callconv(.c) void {}
    fn load(_: *const qa.LoadCtx, _: qa.Str, _: u32, _: *qa.Weight) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.failed, "MissingWeight");
    }
    fn release(w: *qa.Weight) callconv(.c) void {
        w.* = .{};
    }
    fn matmul(_: *const qa.Weight, _: qa.Array, _: qa.Stream, _: *qa.Array) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.unsupported, "toy");
    }
    fn moe(_: *const qa.Weight, _: qa.Array, _: qa.Array, _: qa.Array, _: bool, _: qa.Stream, _: *qa.Array) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.unsupported, "toy");
    }
    fn bankRows(_: *const qa.LoadCtx, _: *const qa.BankGeometry, _: qa.BankBudget, out: *qa.BankRows) callconv(.c) common.Verdict {
        out.* = .{ .prompt = 8, .decode = 8, .row_bytes = 1 };
        return .ok;
    }
    fn openBank(_: *const qa.LoadCtx, geo: *const qa.BankGeometry, _: qa.BankBudget, out: *qa.Bank) callconv(.c) common.Verdict {
        opened += 1;
        layers = geo.n_layers;
        limits = geo.limits_out != null;
        out.* = .{ .ctx = null, .close = &bankClose, .about = .of("toy"), .decode_ids = 16, .rows = &zeroRows, .settle = &settle, .phase = &phase, .read_ahead = &readAhead, .exact_rows = true };
        return .ok;
    }
    fn bankClose(_: ?*anyopaque) callconv(.c) void {}
    fn zeroRows(_: ?*anyopaque, _: u32, x: qa.Array, _: qa.Array, _: qa.Array, out: *qa.Array) callconv(.c) common.Verdict {
        const xa: mlx.mlx_array = .{ .ctx = x.ctx };
        var z = mlx.mlx_array_new();
        mlx.check(mlx.mlx_zeros(&z, mlx.getShape(xa).ptr, mlx.getShape(xa).len, .float32, mlx.gpuStream())) catch return common.Verdict.refuse(.failed, "zeros");
        out.* = .{ .ctx = z.ctx };
        return .ok;
    }
    fn settle(_: ?*anyopaque) callconv(.c) common.Verdict {
        settles += 1;
        return .ok;
    }
    fn phase(_: ?*anyopaque, _: bool) callconv(.c) common.Verdict {
        phases += 1;
        return .ok;
    }
    fn readAhead(_: ?*anyopaque, _: u32, _: [*]const u16, n: usize) callconv(.c) common.Verdict {
        aheads += 1;
        ahead_ids += n;
        return .ok;
    }
    const table: qa.QuantTable = .{ .name = .of("toy-ling"), .claims = &claims, .validate = &validate, .bill = &bill, .open = &open, .close = &close, .load_tensor = &load, .load_bank = &load, .release = &release, .matmul = &matmul, .moe = &moe, .bank_rows = &bankRows, .open_bank = &openBank };

    var row_calls: u32 = 0;
    /// A routed sum each row computes alone, from its own input and route weights: x times the weights' sum, f32.
    fn scaledRows(_: ?*anyopaque, _: u32, x: qa.Array, _: qa.Array, w: qa.Array, out: *qa.Array) callconv(.c) common.Verdict {
        row_calls += 1;
        const s = mlx.gpuStream();
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        var wf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wf);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        var y = mlx.mlx_array_new();
        mlx.check(mlx.mlx_astype(&xf, .{ .ctx = x.ctx }, .float32, s)) catch return common.Verdict.refuse(.failed, "toy rows");
        mlx.check(mlx.mlx_astype(&wf, .{ .ctx = w.ctx }, .float32, s)) catch return common.Verdict.refuse(.failed, "toy rows");
        mlx.check(mlx.mlx_sum_axis(&ws, wf, -1, true, s)) catch return common.Verdict.refuse(.failed, "toy rows");
        mlx.check(mlx.mlx_multiply(&y, xf, ws, s)) catch {
            _ = mlx.mlx_array_free(y);
            return common.Verdict.refuse(.failed, "toy rows");
        };
        out.* = .{ .ctx = y.ctx };
        return .ok;
    }
    fn openBankRows(ctx: *const qa.LoadCtx, geo: *const qa.BankGeometry, budget: qa.BankBudget, out: *qa.Bank) callconv(.c) common.Verdict {
        const v = openBank(ctx, geo, budget, out);
        out.rows = &scaledRows;
        return v;
    }
    /// `table` whose bank's routed sum is `scaledRows`: a row routed or placed wrong in a call shows in its bytes.
    const table_rows: qa.QuantTable = blk: {
        var t = table;
        t.open_bank = &openBankRows;
        break :blk t;
    };
};

// Bar: a Ling streaming pack loads through the table that claims it (its bank bound to every routed layer, one phase
// before and one settle after each forward, a prompt's routes read ahead) and serves its NextN layer as the engine's
// MTP head; an MLX pack loads no head. No legacy oracle exists for the head (the legacy path serves none).
test "bailing_hybrid served: a streaming pack's bank takes each forward's phase and settle, and its NextN layer is its MTP head; an MLX pack has none" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    {
        const files = try fixture.lingMtpFiles(arena.allocator(), 11);
        var p = try packOf(files);
        defer p.deinit();
        var b = try both.Both.openAs(&ling.table, &p, files[0].bytes, false, .none);
        defer b.close();
        try testing.expect(plugin_head.load(io, a, &b.xfm, b.dir) == null);
    }
    const saved = ToyLing.outside.tables.get().*;
    defer ToyLing.outside.tables.get().* = saved;
    ToyLing.outside.tables.get().* = .{};
    ToyLing.outside.register(&ToyLing.table);
    const files = try fixture.lingMtpFiles(arena.allocator(), 11);
    var p = try packOf(files);
    defer p.deinit();
    var b = try both.Both.openWith(&ling.table, &p, fixture.ling_stream_config, false, .none, &.{.{ .name = "experts.bin", .bytes = "" }});
    defer b.close();
    try testing.expectEqual(@as(u32, 1), ToyLing.opened);
    try testing.expectEqual(@as(u32, 4), ToyLing.layers);
    try testing.expect(!ToyLing.limits);
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    var tokens: [16]u32 = undefined;
    var accepted: std.ArrayList(u32) = .empty;
    defer accepted.deinit(a);
    try mtpRun(&b.xfm, .{ .plugin = served }, &tokens, &accepted);
    try testing.expect(accepted.items.len > 0);
    for (accepted.items) |n| try testing.expect(n <= 2);
    // Every trunk forward told the bank its phase first and settled it after.
    try testing.expect(ToyLing.phases > 0 and ToyLing.settles == ToyLing.phases);
    // The prompt read each routed layer's predicted experts ahead: its rows times top_k ids a call.
    try testing.expect(ToyLing.aheads > 0 and ToyLing.ahead_ids % b.xfm.config.num_experts_per_tok == 0);
}

// Bar: a streaming pack's load leaves its NextN layer resident (the bank's open reads the memory it takes), before any
// `HeadHooks.load`.
test "bailing_hybrid served: a streaming pack's load makes its NextN layer resident before the bank opens" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const m = host.of(b.xfm.plugin.?.model);
            try testing.expect(m.ling_mtp != null and m.bank != null);
            try testing.expect(m.mtp_ready);
        }
    }.run);
}

/// The Ling streaming fixture served through `ToyLing` (registered for the call's life).
fn withToyLing(comptime f: fn (*both.Both) anyerror!void) !void {
    return withToy(&ToyLing.table, f);
}

/// The Ling streaming fixture served through the toy quant table `table` (registered for the call's life).
fn withToy(table: *const ToyLing.qa.QuantTable, comptime f: fn (*both.Both) anyerror!void) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const saved = ToyLing.outside.tables.get().*;
    defer ToyLing.outside.tables.get().* = saved;
    ToyLing.outside.tables.get().* = .{};
    ToyLing.outside.register(table);
    var p = try packOf(try fixture.lingMtpFiles(arena.allocator(), 11));
    defer p.deinit();
    var b = try both.Both.openWith(&ling.table, &p, fixture.ling_stream_config, false, .none, &.{.{ .name = "experts.bin", .bytes = "" }});
    defer b.close();
    try f(&b);
}

// Bar: a streaming pack's verify of 2 to 9 rows (past the bank's 8-row decode lane) gives every row serial decode's
// logits and leaves serial decode's state, bit for bit; up to 8 rows its KDA layers take the one-launch rows.
test "bailing_hybrid served: a streaming pack's verify rows are serial decode's, bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const prompt = [_]i32{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31 };
            const window = [_]i32{ 5, 61, 200, 9, 33, 77, 140, 2, 18 };
            for ([_]usize{ 2, 3, 5, 9 }) |n| {
                const before = kda_rows.launches;
                try b.expectVerifyIsSerial(&prompt, window[0..n], 64);
                // The fixture's 3 KDA layers, one launch each, while the rows fit one.
                try testing.expectEqual(before + @as(usize, if (n <= @as(usize, @intCast(kda_rows.max_rows))) 3 else 0), kda_rows.launches);
            }
        }
    }.run);
}

/// Greedy tokens after a fixed prompt on a fresh trunk: serial decode (`depth` 0) or MTP rounds through the head.
fn lingGreedy(b: *both.Both, depth: u32, out: []u32) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var x = try driver.load(io, a, &b.cfg, b.dir, proof_fixture.stream());
    defer x.deinit();
    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    if (depth == 0) {
        var gen = try Generator.initWithOptions(io, a, &x, &tok_dummy, &prompt, @intCast(out.len), .{ .temperature = 0.0 }, &.{}, .{ .skip_lazy_preforward = true });
        defer gen.deinit(a);
        for (out) |*t| t.* = (try gen.next(a)) orelse return error.TestStreamEnded;
        return;
    }
    const served = plugin_head.load(io, a, &x, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    var gen = try Generator.initWithOptions(io, a, &x, &tok_dummy, &prompt, @intCast(out.len), .{ .temperature = 0.0 }, &.{}, .{
        .mtp_enabled = true,
        .mtp = .{ .plugin = served },
        .model_has_mtp = true,
        .mtp_depth = depth,
    });
    defer gen.deinit(a);
    var n: usize = 0;
    while (try gen.nextMtp(a)) |res| {
        defer a.free(res.tokens);
        @memcpy(out[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
    }
    try testing.expectEqual(out.len, n);
    try testing.expect(gen.mtp_attempted > 0);
}

// Bar: on a streaming pack greedy MTP drafting at depth 2 and 4 emits the stream greedy decode with MTP off emits.
test "bailing_hybrid served: a streaming pack's greedy MTP stream equals its greedy stream with MTP off" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            var plain: [24]u32 = undefined;
            try lingGreedy(b, 0, &plain);
            for ([_]u32{ 2, 4 }) |depth| {
                var drafted: [24]u32 = undefined;
                try lingGreedy(b, depth, &drafted);
                try testing.expectEqualSlices(u32, &plain, &drafted);
            }
        }
    }.run);
}

/// A whole-prompt plan of the span-major proof: the span ends and checkpoint positions the engine hands the forward,
/// the routed calls' width, the routed calls each MoE layer then makes, and its read-aheads (the calls over wider spans)
/// and their rows.
const SpanCase = struct { ends: []const u64, cps: []const u64, calls: trunk.PromptCalls = .{}, routed: u32, ahead: u32, ahead_rows: u32 };

/// `cs`'s whole prompt in one forward against the chunk loop's forward a span: every row's logits and hidden, each KDA
/// layer's state after the prompt and at each checkpoint, and a next-token decode, bit for bit; each checkpoint filed
/// restores a chunk-loop slot stopped there to the chunk loop's next-token logits.
fn expectSpanMajor(b: *both.Both, prompt: []const i32, cs: SpanCase) !void {
    const a = testing.allocator;
    const m = host.of(b.xfm.plugin.?.model);
    m.prompt_calls = cs.calls;
    defer m.prompt_calls = .{};
    var g = ops.Graph.init(a, b.xfm.s);
    defer g.deinit();
    const n_layers = b.layers();
    const next = [_]i32{77};

    var loop: both.Slot = undefined;
    try loop.init(b, &b.xfm);
    defer loop.deinit();
    var logits_parts: [8]mlx.mlx_array = undefined;
    var hidden_parts: [8]mlx.mlx_array = undefined;
    // Each checkpoint's KDA states (conv, ssm) by layer, as the chunk loop leaves them at its span's end.
    var want_cp: [4][2][8]mlx.mlx_array = undefined;
    var lo: u64 = 0;
    ToyLing.row_calls = 0;
    ToyLing.aheads = 0;
    for (cs.ends, 0..) |hi, k| {
        var last = mlx.mlx_array_new();
        var all = mlx.mlx_array_new();
        logits_parts[k] = try g.adopt(try b.xfm.forwardWithCaptureAll(&loop.ctx, try proof_fixture.tokens(&g, prompt[lo..hi]), &last, &all));
        _ = try g.adopt(last);
        hidden_parts[k] = try g.adopt(all);
        for (cs.cps, 0..) |cp, c| if (cp == hi) for (m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
            want_cp[c][0][li] = try g.keep(loop.ssm.?[li].conv_state);
            want_cp[c][1][li] = try g.keep(loop.ssm.?[li].ssm_state);
        };
        lo = hi;
    }
    const loop_calls = ToyLing.row_calls;
    try testing.expectEqual(@as(u32, 0), ToyLing.aheads);

    var whole: both.Slot = undefined;
    try whole.init(b, &b.xfm);
    defer whole.deinit();
    var filed: std.ArrayList(checkpoint.SSMCheckpoint) = .empty;
    defer {
        for (filed.items) |*cp| cp.deinit(a);
        filed.deinit(a);
    }
    var pc: checkpoint.PromptCheckpoints = .{ .allocator = a, .list = &filed, .positions = cs.cps, .layers = n_layers, .s = b.xfm.s };
    const sink = pc.sink();
    var progress = std.atomic.Value(u64).init(0);
    whole.ctx.prompt_spans = cs.ends;
    whole.ctx.ssm_cp_positions = cs.cps;
    whole.ctx.cp_sink = &sink;
    whole.ctx.prefill_progress = &progress;
    ToyLing.row_calls = 0;
    ToyLing.aheads = 0;
    ToyLing.ahead_ids = 0;
    var w_last = mlx.mlx_array_new();
    var w_all = mlx.mlx_array_new();
    const got = try g.adopt(try b.xfm.forwardWithCaptureAll(&whole.ctx, try proof_fixture.tokens(&g, prompt[0..lo]), &w_last, &w_all));
    _ = try g.adopt(w_last);
    _ = try g.adopt(w_all);
    whole.ctx.prompt_spans = &.{};
    whole.ctx.ssm_cp_positions = &.{};
    whole.ctx.cp_sink = null;
    whole.ctx.prefill_progress = null;
    // Three routed layers, each in its calls over whole spans; the chunk loop calls each layer once a span.
    try testing.expectEqual(3 * cs.routed, ToyLing.row_calls);
    try testing.expectEqual(@as(u32, @intCast(3 * cs.ends.len)), loop_calls);
    // Each routed layer reads ahead once a call over wider spans, before its attention: the call's rows x top_k ids.
    try testing.expectEqual(3 * cs.ahead, ToyLing.aheads);
    try testing.expectEqual(@as(usize, 3 * 2 * cs.ahead_rows), ToyLing.ahead_ids);
    try testing.expectEqual(lo, progress.load(.monotonic));

    try ops.expectBitEqual(got, try g.concat(logits_parts[0..cs.ends.len], 1));
    try ops.expectBitEqual(w_all, try g.concat(hidden_parts[0..cs.ends.len], 1));
    for (m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
        try ops.expectBitEqual(whole.ssm.?[li].conv_state, loop.ssm.?[li].conv_state);
        try ops.expectBitEqual(whole.ssm.?[li].ssm_state, loop.ssm.?[li].ssm_state);
    };
    try testing.expectEqual(cs.cps.len, filed.items.len);
    for (filed.items, cs.cps, 0..) |cp, pos, c| {
        try testing.expectEqual(pos, cp.pos);
        for (m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
            try ops.expectBitEqual(cp.layers[li].conv_state, want_cp[c][0][li]);
            try ops.expectBitEqual(cp.layers[li].ssm_state, want_cp[c][1][li]);
        };
    }
    const after = try g.adopt(try b.xfm.forwardWith(&loop.ctx, try proof_fixture.tokens(&g, &next)));
    try ops.expectBitEqual(try g.adopt(try b.xfm.forwardWith(&whole.ctx, try proof_fixture.tokens(&g, &next))), after);

    // A slot the chunk loop stopped at a checkpoint, its KDA states the filed ones, then the rest of the prompt.
    for (filed.items) |cp| {
        var rs: both.Slot = undefined;
        try rs.init(b, &b.xfm);
        defer rs.deinit();
        lo = 0;
        for (cs.ends) |hi| {
            if (lo == cp.pos) for (m.layers, 0..) |*lw, li| if (lw.mixer == .kda) try ssmRestore(&rs.ssm.?[li], &cp.layers[li]);
            _ = try g.adopt(try b.xfm.forwardWith(&rs.ctx, try proof_fixture.tokens(&g, prompt[lo..hi])));
            lo = hi;
        }
        try ops.expectBitEqual(try g.adopt(try b.xfm.forwardWith(&rs.ctx, try proof_fixture.tokens(&g, &next))), after);
    }
}

// Bar: a streaming pack's whole prompt layer by layer over the chunk loop's spans (2, 3 and 5 spans; decode-lane spans
// among them; routed calls joined and split) equals the chunk loop bit for bit, and its checkpoints restore.
test "bailing_hybrid served: a streaming pack's span-major prompt is the chunk loop's, bit for bit, at 2, 3 and 5 spans" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToy(&ToyLing.table_rows, struct {
        fn run(b: *both.Both) anyerror!void {
            var prompt: [40]i32 = undefined;
            for (&prompt, 0..) |*t, i| t.* = @intCast(3 + (i * 53) % 500);
            // The toy's decode lane holds 8 rows: an 8-row span and a 1-row span run as forwards of their own.
            const cases = [_]SpanCase{
                .{ .ends = &.{ 32, 40 }, .cps = &.{32}, .routed = 2, .ahead = 1, .ahead_rows = 32 },
                .{ .ends = &.{ 12, 30, 40 }, .cps = &.{ 12, 30 }, .calls = .{ .rows = 30, .tail = 4 }, .routed = 2, .ahead = 2, .ahead_rows = 40 },
                .{ .ends = &.{ 10, 20, 21, 33, 40 }, .cps = &.{ 20, 21, 33 }, .routed = 4, .ahead = 2, .ahead_rows = 32 },
            };
            for (cases) |cs| try expectSpanMajor(b, &prompt, cs);
        }
    }.run);
}
