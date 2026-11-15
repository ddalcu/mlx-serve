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
const fixture = @import("../../sdk/bases/kda_hybrid/fixture.zig");
const glm = @import("plugin.zig");
const host = @import("../../sdk/bases/kda_hybrid/host.zig");
const ling = @import("../bailing_hybrid/plugin.zig");
const kda = @import("../../sdk/bases/kda_hybrid/kda.zig");
const trunk = @import("../../sdk/bases/kda_hybrid/trunk.zig");
const tape_compact_over = @import("proof_test.zig").tape_compact_over;

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
    fn readAhead(_: ?*anyopaque, _: u32, _: [*]const u16, _: usize) callconv(.c) common.Verdict {
        return .ok;
    }
    const table: qa.QuantTable = .{ .name = .of("toy-ling"), .claims = &claims, .validate = &validate, .bill = &bill, .open = &open, .close = &close, .load_tensor = &load, .load_bank = &load, .release = &release, .matmul = &matmul, .moe = &moe, .bank_rows = &bankRows, .open_bank = &openBank };
};

// Bar: a Ling streaming pack loads through the table that claims it (its bank bound to every routed layer, one phase
// before and one settle after each forward) and serves its NextN layer as the engine's MTP head; an MLX pack loads no
// head. No legacy oracle exists for the head (the legacy path serves none).
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
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const saved = ToyLing.outside.tables.get().*;
    defer ToyLing.outside.tables.get().* = saved;
    ToyLing.outside.tables.get().* = .{};
    ToyLing.outside.register(&ToyLing.table);
    var p = try packOf(try fixture.lingMtpFiles(arena.allocator(), 11));
    defer p.deinit();
    var b = try both.Both.openWith(&ling.table, &p, fixture.ling_stream_config, false, .none, &.{.{ .name = "experts.bin", .bytes = "" }});
    defer b.close();
    try f(&b);
}

// Bar: a streaming pack's verify of 2 to 9 rows (past the bank's 8-row decode lane) gives every row serial decode's
// logits and leaves serial decode's state, bit for bit.
test "bailing_hybrid served: a streaming pack's verify rows are serial decode's, bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const prompt = [_]i32{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31 };
            const window = [_]i32{ 5, 61, 200, 9, 33, 77, 140, 2, 18 };
            for ([_]usize{ 2, 3, 5, 9 }) |n| try b.expectVerifyIsSerial(&prompt, window[0..n], 64);
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

/// A model plugin's verify rows: counts each call and keeps its operands; declines while `out` is null, else hands
/// back new references on `out`; fails while `fail` is set.
const ProbeRows = struct {
    var calls: usize = 0;
    var seen: kda.VerifyRowsIn = undefined;
    var rows_seen: c_int = 0;
    var state_in: mlx.mlx_array = .{};
    var conv_in: mlx.mlx_array = .{};
    var out: ?kda.VerifyRowsOut = null;
    var fail = false;

    fn run(_: mlx.mlx_stream, in: kda.VerifyRowsIn) anyerror!?kda.VerifyRowsOut {
        calls += 1;
        seen = in;
        rows_seen = mlx.getShape(in.qkv)[1];
        keep(&state_in, in.state);
        keep(&conv_in, in.conv_state);
        if (fail) return error.TestProbeFailed;
        const o = out orelse return null;
        return .{ .y = ref(o.y), .conv_state = ref(o.conv_state), .state = ref(o.state), .state_seq = ref(o.state_seq) };
    }

    fn ref(a: mlx.mlx_array) mlx.mlx_array {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, a);
        return r;
    }

    fn keep(slot: *mlx.mlx_array, a: mlx.mlx_array) void {
        if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
        slot.* = ref(a);
    }

    fn reset() void {
        for ([_]*mlx.mlx_array{ &state_in, &conv_in }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a.*);
            a.* = .{};
        };
        calls = 0;
        rows_seen = 0;
        out = null;
        fail = false;
    }
};

/// The first KDA layer of the trunk.
fn kdaLayer(m: *trunk.Model) !*const trunk.Kda {
    for (m.layers) |*l| if (l.mixer == .kda) return &l.mixer.kda;
    return error.TestNoKdaLayer;
}

/// A slot's state at `la` after a 10-token prompt, as a prefill leaves it.
fn primed(m: *trunk.Model, la: *const trunk.Kda, g: *ops.Graph) !kda.Entry {
    var e: kda.Entry = .{};
    errdefer e.deinit();
    const prompt = [_]i32{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31 };
    const x = try g.adopt(try m.embedding(try proof_fixture.tokens(g, &prompt)));
    _ = try g.adopt(try kda.forward(m, x, la, &e, 1, prompt.len, true, false));
    return e;
}

/// A second reference on every array of `e`.
fn copyOf(e: *const kda.Entry) kda.Entry {
    var c: kda.Entry = .{ .initialized = e.initialized };
    inline for (.{ "conv_state", "ssm_state", "spec_conv_input", "spec_state_seq" }) |f| if (@field(e.*, f).ctx != null) {
        @field(c, f) = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&@field(c, f), @field(e.*, f));
    };
    return c;
}

/// normal(0, 1) of shape `sh` in `dt`.
fn noise(g: *ops.Graph, sh: []const c_int, dt: mlx.mlx_dtype, seed: u64) !mlx.mlx_array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    return g.astype(try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, @as(mlx.mlx_dtype, .float32), @as(f32, 0), @as(f32, 1), key }), dt);
}

// Bar: a serial-rows verify hands a KDA layer's rows to the model's verify rows once, on the slot's state and the
// layer's geometry; y goes through the out projection and the window, state and every row's state into the slot.
test "glm5_next kda: a model's verify rows serve a serial-rows verify, and the layer and the slot take their outputs" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const m = host.of(b.xfm.plugin.?.model);
            const c = m.c;
            const la = try kdaLayer(m);
            var g = ops.Graph.init(testing.allocator, m.s);
            defer g.deinit();
            var base = try primed(m, la, &g);
            defer base.deinit();
            const window = [_]i32{ 5, 61, 200, 9 };
            const n: c_int = window.len;
            const x = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &window)));
            m.serial_rows = true;
            defer m.serial_rows = false;
            // The trunk's own rows, for the window's conv inputs it captures on either path.
            var stock = copyOf(&base);
            defer stock.deinit();
            _ = try g.adopt(try kda.forward(m, x, la, &stock, 1, n, false, true));

            const hk: c_int = @intCast(c.linear_num_key_heads);
            const hv: c_int = @intCast(c.linear_num_value_heads);
            const dk: c_int = @intCast(c.linear_key_head_dim);
            const dv: c_int = @intCast(c.linear_value_head_dim);
            const act = ops.dtype(stock.conv_state);
            const st = ops.dtype(base.ssm_state);
            ProbeRows.reset();
            defer ProbeRows.reset();
            ProbeRows.out = kda.VerifyRowsOut{
                .y = try noise(&g, &.{ 1, n, hv * dv }, act, 1),
                .conv_state = try noise(&g, &.{ 1, 3, 2 * hk * dk + hv * dv }, act, 2),
                .state = try noise(&g, &.{ 1, hv, dv, dk }, st, 3),
                .state_seq = try noise(&g, &.{ n, 1, hv, dv, dk }, st, 4),
            };
            const want = ProbeRows.out.?;
            m.kda_verify_rows = &ProbeRows.run;
            defer m.kda_verify_rows = null;
            var served = copyOf(&base);
            defer served.deinit();
            const y = try g.adopt(try kda.forward(m, x, la, &served, 1, n, false, true));

            try testing.expectEqual(@as(usize, 1), ProbeRows.calls);
            try testing.expectEqual(n, ProbeRows.rows_seen);
            const in = ProbeRows.seen;
            try testing.expectEqual([4]c_int{ hk, hv, dk, dv }, [4]c_int{ in.hk, in.hv, in.dk, in.dv });
            try testing.expectEqual(c.kda_gate_lower_bound, in.bound);
            try testing.expectEqual(c.rms_norm_eps, in.norm_eps);
            try testing.expectEqual(@as(f32, 1e-6), in.qk_eps);
            try ops.expectBitEqual(ProbeRows.state_in, base.ssm_state);
            try ops.expectBitEqual(ProbeRows.conv_in, base.conv_state);

            try ops.expectBitEqual(y, try g.adopt(try m.mm(&la.out, want.y)));
            try ops.expectBitEqual(served.conv_state, want.conv_state);
            try ops.expectBitEqual(served.ssm_state, want.state);
            try ops.expectBitEqual(served.spec_state_seq, want.state_seq);
            try ops.expectBitEqual(served.spec_conv_input, stock.spec_conv_input);
        }
    }.run);
}

// Bar: an error from the model's verify rows is the forward's, and the slot keeps the state it had.
test "glm5_next kda: a model's verify rows that fail leave the slot's state as it was" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const m = host.of(b.xfm.plugin.?.model);
            const la = try kdaLayer(m);
            var g = ops.Graph.init(testing.allocator, m.s);
            defer g.deinit();
            var base = try primed(m, la, &g);
            defer base.deinit();
            const window = [_]i32{ 5, 61 };
            const x = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &window)));
            ProbeRows.reset();
            defer ProbeRows.reset();
            ProbeRows.fail = true;
            m.kda_verify_rows = &ProbeRows.run;
            defer m.kda_verify_rows = null;
            m.serial_rows = true;
            defer m.serial_rows = false;
            var failed = copyOf(&base);
            defer failed.deinit();
            try testing.expectError(error.TestProbeFailed, kda.forward(m, x, la, &failed, 1, window.len, false, true));
            try testing.expectEqual(@as(usize, 1), ProbeRows.calls);
            try ops.expectBitEqual(failed.conv_state, base.conv_state);
            try ops.expectBitEqual(failed.ssm_state, base.ssm_state);
            try testing.expect(failed.spec_state_seq.ctx == null and failed.spec_conv_input.ctx == null);
        }
    }.run);
}

// Bar: with no verify rows set, a serial-rows verify's rows, final state, conv window and every row's state are the
// one-row decode steps', bit for bit.
test "glm5_next kda: without a model's verify rows a serial-rows verify steps each row as one-row decode does" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const m = host.of(b.xfm.plugin.?.model);
            try testing.expect(m.kda_verify_rows == null);
            const la = try kdaLayer(m);
            var g = ops.Graph.init(testing.allocator, m.s);
            defer g.deinit();
            var base = try primed(m, la, &g);
            defer base.deinit();
            const window = [_]i32{ 5, 61, 200, 9 };
            var verify = copyOf(&base);
            defer verify.deinit();
            m.serial_rows = true;
            const got = kda.forward(m, try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &window))), la, &verify, 1, window.len, false, true);
            m.serial_rows = false;
            const y = try g.adopt(try got);
            var serial = copyOf(&base);
            defer serial.deinit();
            var ys: [window.len]mlx.mlx_array = undefined;
            var states: [window.len]mlx.mlx_array = undefined;
            for (&ys, &states, window) |*yr, *sr, t| {
                const xr = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &.{t})));
                yr.* = try g.adopt(try kda.forward(m, xr, la, &serial, 1, 1, false, false));
                sr.* = try g.keep(serial.ssm_state);
            }
            try ops.expectBitEqual(y, try g.concat(&ys, 1));
            try ops.expectBitEqual(verify.ssm_state, serial.ssm_state);
            try ops.expectBitEqual(verify.conv_state, serial.conv_state);
            for (states, 0..) |sr, r| {
                const row = try g.sliceAxis(verify.spec_state_seq, 0, @intCast(r), @intCast(r + 1));
                try ops.expectBitEqual(try g.reshape(row, ops.shape(sr)), sr);
            }
        }
    }.run);
}

// Bar: a streaming pack's verify offers every KDA layer's rows to the model's verify rows, at any width; where they
// decline, every row is still serial decode's, bit for bit.
test "bailing_hybrid served: a streaming pack's verify offers each KDA layer's rows to the model's verify rows; a decline keeps serial decode's bytes" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const m = host.of(b.xfm.plugin.?.model);
            ProbeRows.reset();
            defer ProbeRows.reset();
            m.kda_verify_rows = &ProbeRows.run;
            defer m.kda_verify_rows = null;
            const prompt = [_]i32{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31 };
            const window = [_]i32{ 5, 61, 200, 9, 33, 77, 140, 2, 18 };
            for ([_]usize{ 2, 5, 9 }) |n| {
                const before = ProbeRows.calls;
                try b.expectVerifyIsSerial(&prompt, window[0..n], 64);
                // The fixture's 3 KDA layers, each offered the verify's rows once; decode steps are never offered.
                try testing.expectEqual(before + 3, ProbeRows.calls);
                try testing.expectEqual(@as(c_int, @intCast(n)), ProbeRows.rows_seen);
            }
        }
    }.run);
}
