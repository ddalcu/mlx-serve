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
const moe_block = @import("../../sdk/bases/blocks/moe.zig");
const moe_routing = @import("../../sdk/blocks/moe/routing.zig");
const abi = @import("../../abi/plugin.zig");
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

// Bar: a streaming pack reserves its whole request at the prompt pass (prompt, reply to the context limit and the
// verify rows), so a greedy decode to the limit, past the 256 rows a first grow takes, never grows the cache.
test "bailing_hybrid served: a streaming pack's KV is reserved once at the prompt pass and never grows in decode" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const a = testing.allocator;
            const io = std.Io.Threaded.global_single_threaded.io();
            const KVCache = @import("../../sdk/kv/cache.zig").KVCache;
            const verify_rows = @import("../../engines/mlx/generate/init.zig").verify_rows_max;
            var x = try driver.load(io, a, &b.cfg, b.dir, proof_fixture.stream());
            defer x.deinit();
            const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
            const reply: u32 = 260;
            const limit: usize = prompt.len + reply;
            x.config.ctx_override = @intCast(limit);
            var tok_dummy: Tokenizer = undefined; // never read by the Generator
            var gen = try Generator.initWithOptions(io, a, &x, &tok_dummy, &prompt, reply, .{ .temperature = 0.0 }, &.{}, .{ .skip_lazy_preforward = true });
            defer gen.deinit(a);
            try testing.expectEqual(limit + verify_rows, gen.ctx.cache.reserve_tokens);
            var layers: usize = 0;
            for (gen.ctx.cache.entries) |e| if (e.initialized) {
                layers += 1;
                try testing.expect(KVCache.bufferCapacity(e.keys) >= limit + verify_rows);
            };
            try testing.expect(layers > 0);
            const grows = KVCache.kv_cap_buf_grows;
            var n: usize = 0;
            while (try gen.next(a)) |_| n += 1;
            try testing.expectEqual(@as(usize, reply), n);
            try testing.expectEqual(grows, KVCache.kv_cap_buf_grows);
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

/// Array helpers for the probes: a second reference, and a slot that keeps one.
const ProbeRows = struct {
    fn ref(a: mlx.mlx_array) mlx.mlx_array {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, a);
        return r;
    }

    fn keep(slot: *mlx.mlx_array, a: mlx.mlx_array) void {
        if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
        slot.* = ref(a);
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

/// A model's own KDA stages (`kda.Stages`) for the tests: each counts its calls and keeps what it was handed, then
/// serves its `*_out` or declines (null; `serves` false) where that is null.
const ProbeStages = struct {
    var prologue_calls: usize = 0;
    var serves_calls: usize = 0;
    var run_calls: usize = 0;
    var gate_calls: usize = 0;
    var capture_calls: usize = 0;
    var cold = false;
    var conv_in: mlx.mlx_array = .{};
    var heads_in: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
    var state_in: mlx.mlx_array = .{};
    var a_in: mlx.mlx_array = .{};
    var g_in: mlx.mlx_array = .{};
    var prologue_out: ?kda.PrologueOut = null;
    var recur_out: ?kda.RecurOut = null;
    var gate_out: ?mlx.mlx_array = null;
    var capture_out: ?kda.CaptureOut = null;
    var prework_calls: usize = 0;
    var epilogue_calls: usize = 0;
    var epilogue_verify = false;
    var pw_conv_in: mlx.mlx_array = .{};
    var beta_in: mlx.mlx_array = .{};
    var prework_out: ?kda.PreworkOut = null;
    var epilogue_out: ?mlx.mlx_array = null;

    const all: kda.Stages = .{
        .prologue = &prologue,
        .recurrence = .{ .serves = &serves, .run = &run },
        .verify_gate = &gate,
        .verify_capture = &capture,
        .verify_prework = &prework,
        .epilogue = &epilogue,
    };

    fn prework(_: mlx.mlx_stream, in: kda.PreworkIn) anyerror!?kda.PreworkOut {
        prework_calls += 1;
        if (in.conv_state) |c| ProbeRows.keep(&pw_conv_in, c);
        const o = prework_out orelse return null;
        return .{ .q = ProbeRows.ref(o.q), .k = ProbeRows.ref(o.k), .v = ProbeRows.ref(o.v), .beta = ProbeRows.ref(o.beta), .window = ProbeRows.ref(o.window), .conv_input = ProbeRows.ref(o.conv_input) };
    }

    fn epilogue(_: mlx.mlx_stream, in: kda.EpilogueIn) anyerror!?mlx.mlx_array {
        epilogue_calls += 1;
        epilogue_verify = in.verify;
        const o = epilogue_out orelse return null;
        return ProbeRows.ref(o);
    }

    fn prologue(_: mlx.mlx_stream, in: kda.PrologueIn) anyerror!?kda.PrologueOut {
        prologue_calls += 1;
        cold = in.conv_state == null;
        if (in.conv_state) |c| ProbeRows.keep(&conv_in, c);
        const o = prologue_out orelse return null;
        return .{ .q = ProbeRows.ref(o.q), .k = ProbeRows.ref(o.k), .v = ProbeRows.ref(o.v), .window = ProbeRows.ref(o.window) };
    }

    fn serves(_: mlx.mlx_stream, _: c_int, _: c_int, _: c_int, _: c_int, _: mlx.mlx_dtype, _: mlx.mlx_dtype, _: mlx.mlx_array, _: mlx.mlx_array, _: f32) bool {
        serves_calls += 1;
        return recur_out != null;
    }

    fn run(_: mlx.mlx_stream, in: kda.RecurIn) anyerror!kda.RecurOut {
        run_calls += 1;
        for (&heads_in, [_]mlx.mlx_array{ in.q, in.k, in.v }) |*h, x| ProbeRows.keep(h, x);
        ProbeRows.keep(&state_in, in.state);
        ProbeRows.keep(&a_in, in.a);
        const o = recur_out.?;
        return .{ .y = ProbeRows.ref(o.y), .state = ProbeRows.ref(o.state) };
    }

    fn gate(_: mlx.mlx_stream, in: kda.GateIn) anyerror!?mlx.mlx_array {
        gate_calls += 1;
        ProbeRows.keep(&a_in, in.a);
        const o = gate_out orelse return null;
        return ProbeRows.ref(o);
    }

    fn capture(_: mlx.mlx_stream, in: kda.CaptureIn) anyerror!?kda.CaptureOut {
        capture_calls += 1;
        for (&heads_in, [_]mlx.mlx_array{ in.q, in.k, in.v }) |*h, x| ProbeRows.keep(h, x);
        ProbeRows.keep(&beta_in, in.beta);
        ProbeRows.keep(&g_in, in.g);
        ProbeRows.keep(&state_in, in.state);
        const o = capture_out orelse return null;
        return .{ .y = ProbeRows.ref(o.y), .state = ProbeRows.ref(o.state), .state_seq = ProbeRows.ref(o.state_seq) };
    }

    fn reset() void {
        for ([_]*mlx.mlx_array{ &conv_in, &heads_in[0], &heads_in[1], &heads_in[2], &state_in, &a_in, &g_in, &pw_conv_in, &beta_in }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a.*);
            a.* = .{};
        };
        prologue_calls = 0;
        serves_calls = 0;
        run_calls = 0;
        gate_calls = 0;
        capture_calls = 0;
        cold = false;
        prologue_out = null;
        recur_out = null;
        gate_out = null;
        capture_out = null;
        prework_calls = 0;
        epilogue_calls = 0;
        epilogue_verify = false;
        prework_out = null;
        epilogue_out = null;
    }
};

/// The layer's output for the recurrence's `y` `[1, S, hv, dv]`: the gated norm over the layer's z, then the out
/// projection (the trunk's epilogue off the fused prework).
fn epilogueOf(m: *trunk.Model, la: *const trunk.Kda, g: *ops.Graph, x: mlx.mlx_array, y: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(y);
    const y_shape = [_]c_int{ sh[0], sh[1], sh[2], sh[3] };
    const z = try g.reshape(try g.adopt(try m.mm(&la.z, x)), &y_shape);
    const gated = try g.mul(try g.adopt(try m.rmsNorm(y, la.norm_w)), try g.sigmoid(z));
    return g.adopt(try m.mm(&la.out, try g.reshape(gated, &.{ y_shape[0], y_shape[1], y_shape[2] * y_shape[3] })));
}

/// Every state array of `got` equals `want`'s (both unset, or bit-equal).
fn expectSameEntry(got: *const kda.Entry, want: *const kda.Entry) !void {
    try testing.expectEqual(want.initialized, got.initialized);
    inline for (.{ "conv_state", "ssm_state", "spec_conv_input", "spec_state_seq" }) |f| {
        try testing.expectEqual(@field(want.*, f).ctx == null, @field(got.*, f).ctx == null);
        if (@field(want.*, f).ctx != null) try ops.expectBitEqual(@field(got.*, f), @field(want.*, f));
    }
}

const stage_prompt = [_]i32{ 5, 61, 200, 9, 33, 14, 77, 120, 2, 48, 301, 6, 90, 11, 250, 19, 7, 64, 133, 40 };
const stage_window = [_]i32{ 5, 61, 200, 9 };

// Bar: where a model's KDA stages decline (prompt rows: the prologue and the recurrence; verify rows: the gate and the
// capture), each is offered its rows once and the layer gives the stock bytes, the slot the stock states.
test "glm5_next kda: a model's stages that decline leave a prompt's and a verify's bytes and the slot's states as without them" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try withToyLing(struct {
        fn run(b: *both.Both) anyerror!void {
            const m = host.of(b.xfm.plugin.?.model);
            const la = try kdaLayer(m);
            var g = ops.Graph.init(testing.allocator, m.s);
            defer g.deinit();
            var base = try primed(m, la, &g);
            defer base.deinit();
            const xp = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &stage_prompt)));
            const np: c_int = stage_prompt.len;
            const xv = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &stage_window)));
            const nv: c_int = stage_window.len;

            // The stock prompt and verify, each from the primed slot.
            var stock_p = copyOf(&base);
            defer stock_p.deinit();
            const want_p = try g.adopt(try kda.forward(m, xp, la, &stock_p, 1, np, true, false));
            var stock_v = copyOf(&base);
            defer stock_v.deinit();
            const want_v = try g.adopt(try kda.forward(m, xv, la, &stock_v, 1, nv, false, true));

            ProbeStages.reset();
            defer ProbeStages.reset();
            m.kda.stages = ProbeStages.all;
            defer m.kda.stages = .{};

            var p = copyOf(&base);
            defer p.deinit();
            const got_p = try g.adopt(try kda.forward(m, xp, la, &p, 1, np, true, false));
            try testing.expectEqual([5]usize{ 1, 1, 0, 0, 0 }, [5]usize{ ProbeStages.prologue_calls, ProbeStages.serves_calls, ProbeStages.run_calls, ProbeStages.gate_calls, ProbeStages.capture_calls });
            try testing.expectEqual([2]usize{ 0, 1 }, [2]usize{ ProbeStages.prework_calls, ProbeStages.epilogue_calls });
            try testing.expect(!ProbeStages.epilogue_verify);
            try testing.expect(!ProbeStages.cold);
            try ops.expectBitEqual(ProbeStages.conv_in, base.conv_state);
            try ops.expectBitEqual(got_p, want_p);
            try expectSameEntry(&p, &stock_p);

            var v = copyOf(&base);
            defer v.deinit();
            const got_v = try g.adopt(try kda.forward(m, xv, la, &v, 1, nv, false, true));
            try testing.expectEqual([5]usize{ 1, 1, 0, 1, 1 }, [5]usize{ ProbeStages.prologue_calls, ProbeStages.serves_calls, ProbeStages.run_calls, ProbeStages.gate_calls, ProbeStages.capture_calls });
            try testing.expectEqual([2]usize{ 1, 2 }, [2]usize{ ProbeStages.prework_calls, ProbeStages.epilogue_calls });
            try testing.expect(ProbeStages.epilogue_verify);
            try ops.expectBitEqual(ProbeStages.pw_conv_in, base.conv_state);
            try ops.expectBitEqual(ProbeStages.state_in, base.ssm_state);
            try ops.expectBitEqual(got_v, want_v);
            try expectSameEntry(&v, &stock_v);
        }
    }.run);
}

// Bar: a model's prologue and recurrence that serve a prompt's rows take the stock stages' places: the recurrence gets
// the prologue's heads and the slot's state, the slot takes the prologue's window and the recurrence's state, and the
// layer's output is the epilogue over the recurrence's y.
test "glm5_next kda: a model's prompt stages serve a prompt's rows, and the layer and the slot take their outputs" {
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
            const x = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &stage_prompt)));
            const n: c_int = stage_prompt.len;

            const hk: c_int = @intCast(c.linear_num_key_heads);
            const hv: c_int = @intCast(c.linear_num_value_heads);
            const dk: c_int = @intCast(c.linear_key_head_dim);
            const dv: c_int = @intCast(c.linear_value_head_dim);
            const act = ops.dtype(base.conv_state);
            const st = ops.dtype(base.ssm_state);
            ProbeStages.reset();
            defer ProbeStages.reset();
            ProbeStages.prologue_out = .{
                .q = try noise(&g, &.{ 1, n, hk, dk }, act, 11),
                .k = try noise(&g, &.{ 1, n, hk, dk }, act, 12),
                .v = try noise(&g, &.{ 1, n, hv, dv }, act, 13),
                .window = try noise(&g, mlx.getShape(base.conv_state), act, 14),
            };
            ProbeStages.recur_out = .{ .y = try noise(&g, &.{ 1, n, hv, dv }, act, 15), .state = try noise(&g, mlx.getShape(base.ssm_state), st, 16) };
            const pro = ProbeStages.prologue_out.?;
            const rec = ProbeStages.recur_out.?;
            m.kda.stages = ProbeStages.all;
            defer m.kda.stages = .{};

            var served = copyOf(&base);
            defer served.deinit();
            const y = try g.adopt(try kda.forward(m, x, la, &served, 1, n, true, false));

            try testing.expectEqual([5]usize{ 1, 1, 1, 0, 0 }, [5]usize{ ProbeStages.prologue_calls, ProbeStages.serves_calls, ProbeStages.run_calls, ProbeStages.gate_calls, ProbeStages.capture_calls });
            try ops.expectBitEqual(ProbeStages.conv_in, base.conv_state);
            for (ProbeStages.heads_in, [_]mlx.mlx_array{ pro.q, pro.k, pro.v }) |got, want| try ops.expectBitEqual(got, want);
            try ops.expectBitEqual(ProbeStages.state_in, base.ssm_state);
            try ops.expectBitEqual(ProbeStages.a_in, try g.adopt(try m.mm(&la.a, x)));

            try ops.expectBitEqual(y, try epilogueOf(m, la, &g, x, rec.y));
            try ops.expectBitEqual(served.conv_state, pro.window);
            try ops.expectBitEqual(served.ssm_state, rec.state);
        }
    }.run);
}

// Bar: a model's gate and capture that serve a verify's rows take the stock stages' places: the capture gets the gate's
// output and the slot's state, the slot takes its state and every row's state, keeps the conv window and conv inputs
// the stock conv captures, and the layer's output is the epilogue over the capture's y.
test "glm5_next kda: a model's verify stages serve a verify's rows, and the layer and the slot take their outputs" {
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
            const x = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &stage_window)));
            const n: c_int = stage_window.len;
            // The stock verify, for the conv window and conv inputs the served one keeps.
            var stock = copyOf(&base);
            defer stock.deinit();
            _ = try g.adopt(try kda.forward(m, x, la, &stock, 1, n, false, true));

            const hk: c_int = @intCast(c.linear_num_key_heads);
            const hv: c_int = @intCast(c.linear_num_value_heads);
            const dk: c_int = @intCast(c.linear_key_head_dim);
            const dv: c_int = @intCast(c.linear_value_head_dim);
            const act = ops.dtype(base.conv_state);
            const st = ops.dtype(base.ssm_state);
            ProbeStages.reset();
            defer ProbeStages.reset();
            ProbeStages.gate_out = try noise(&g, &.{ 1, n, hk * dk }, .float32, 21);
            ProbeStages.capture_out = .{
                .y = try noise(&g, &.{ 1, n, hv, dv }, act, 22),
                .state = try noise(&g, mlx.getShape(base.ssm_state), st, 23),
                .state_seq = try noise(&g, &.{ n, 1, hv, dv, dk }, st, 24),
            };
            const gate_out = ProbeStages.gate_out.?;
            const cap = ProbeStages.capture_out.?;
            m.kda.stages = ProbeStages.all;
            defer m.kda.stages = .{};

            var served = copyOf(&base);
            defer served.deinit();
            const y = try g.adopt(try kda.forward(m, x, la, &served, 1, n, false, true));

            try testing.expectEqual([5]usize{ 0, 0, 0, 1, 1 }, [5]usize{ ProbeStages.prologue_calls, ProbeStages.serves_calls, ProbeStages.run_calls, ProbeStages.gate_calls, ProbeStages.capture_calls });
            try ops.expectBitEqual(ProbeStages.a_in, try g.adopt(try m.mm(&la.a, x)));
            try ops.expectBitEqual(ProbeStages.g_in, gate_out);
            try ops.expectBitEqual(ProbeStages.state_in, base.ssm_state);

            try ops.expectBitEqual(y, try epilogueOf(m, la, &g, x, cap.y));
            try ops.expectBitEqual(served.ssm_state, cap.state);
            try ops.expectBitEqual(served.spec_state_seq, cap.state_seq);
            try ops.expectBitEqual(served.conv_state, stock.conv_state);
            try ops.expectBitEqual(served.spec_conv_input, stock.spec_conv_input);
        }
    }.run);
}

// Bar: a model's verify prework that serves a verify's rows takes the conv's, the norms' and beta's places: the capture
// gets its q, k, v and beta, the slot takes its conv window and conv input, and the layer's output is the epilogue over
// the capture's y.
test "glm5_next kda: a model's verify prework serves a verify's rows, and the capture and the slot take its outputs" {
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
            const x = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, &stage_window)));
            const n: c_int = stage_window.len;
            const hk: c_int = @intCast(c.linear_num_key_heads);
            const hv: c_int = @intCast(c.linear_num_value_heads);
            const dk: c_int = @intCast(c.linear_key_head_dim);
            const dv: c_int = @intCast(c.linear_value_head_dim);
            const act = ops.dtype(base.conv_state);
            const st = ops.dtype(base.ssm_state);
            ProbeStages.reset();
            defer ProbeStages.reset();
            ProbeStages.prework_out = .{
                .q = try noise(&g, &.{ 1, n, hk, dk }, act, 31),
                .k = try noise(&g, &.{ 1, n, hk, dk }, act, 32),
                .v = try noise(&g, &.{ 1, n, hv, dv }, act, 33),
                .beta = try noise(&g, &.{ 1, n, hv }, act, 34),
                .window = try noise(&g, mlx.getShape(base.conv_state), act, 35),
                .conv_input = try noise(&g, &.{ 1, n + 3, 2 * hk * dk + hv * dv }, act, 36),
            };
            ProbeStages.capture_out = .{
                .y = try noise(&g, &.{ 1, n, hv, dv }, act, 37),
                .state = try noise(&g, mlx.getShape(base.ssm_state), st, 38),
                .state_seq = try noise(&g, &.{ n, 1, hv, dv, dk }, st, 39),
            };
            const pw = ProbeStages.prework_out.?;
            const cap = ProbeStages.capture_out.?;
            m.kda.stages = ProbeStages.all;
            defer m.kda.stages = .{};

            var served = copyOf(&base);
            defer served.deinit();
            const y = try g.adopt(try kda.forward(m, x, la, &served, 1, n, false, true));

            try testing.expectEqual([3]usize{ 1, 1, 1 }, [3]usize{ ProbeStages.prework_calls, ProbeStages.capture_calls, ProbeStages.epilogue_calls });
            try ops.expectBitEqual(ProbeStages.pw_conv_in, base.conv_state);
            for (ProbeStages.heads_in, [_]mlx.mlx_array{ pw.q, pw.k, pw.v }) |got, want| try ops.expectBitEqual(got, want);
            try ops.expectBitEqual(ProbeStages.beta_in, pw.beta);
            try ops.expectBitEqual(ProbeStages.state_in, base.ssm_state);

            try ops.expectBitEqual(y, try epilogueOf(m, la, &g, x, cap.y));
            try ops.expectBitEqual(served.conv_state, pw.window);
            try ops.expectBitEqual(served.spec_conv_input, pw.conv_input);
            try ops.expectBitEqual(served.ssm_state, cap.state);
            try ops.expectBitEqual(served.spec_state_seq, cap.state_seq);
        }
    }.run);
}

// Bar: a model's epilogue that serves a prompt's or a verify's rows gives the out projection's input: the layer's output
// is the out projection of what it gave, and it changes no slot state.
test "glm5_next kda: a model's epilogue serves a prompt's and a verify's rows, and the out projection takes its output" {
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
            const hv: c_int = @intCast(c.linear_num_value_heads);
            const dv: c_int = @intCast(c.linear_value_head_dim);
            const act = ops.dtype(base.conv_state);
            for ([_]struct { ids: []const i32, prefill: bool, capture: bool }{
                .{ .ids = &stage_prompt, .prefill = true, .capture = false },
                .{ .ids = &stage_window, .prefill = false, .capture = true },
            }, 0..) |case, i| {
                const x = try g.adopt(try m.embedding(try proof_fixture.tokens(&g, case.ids)));
                const n: c_int = @intCast(case.ids.len);
                var stock = copyOf(&base);
                defer stock.deinit();
                _ = try g.adopt(try kda.forward(m, x, la, &stock, 1, n, case.prefill, case.capture));

                ProbeStages.reset();
                defer ProbeStages.reset();
                ProbeStages.epilogue_out = try noise(&g, &.{ 1, n, hv * dv }, act, 41 + i);
                const flat = ProbeStages.epilogue_out.?;
                m.kda.stages = .{ .epilogue = &ProbeStages.epilogue };
                defer m.kda.stages = .{};
                var served = copyOf(&base);
                defer served.deinit();
                const y = try g.adopt(try kda.forward(m, x, la, &served, 1, n, case.prefill, case.capture));

                try testing.expectEqual(@as(usize, 1), ProbeStages.epilogue_calls);
                try testing.expectEqual(case.capture, ProbeStages.epilogue_verify);
                try ops.expectBitEqual(y, try g.adopt(try m.mm(&la.out, flat)));
                try expectSameEntry(&served, &stock);
            }
        }
    }.run);
}

/// A router hook fill for the load test: it counts its calls, keeps each stored weight it was handed, and gives a hook
/// whose `ctx` is the fill's own.
const ProbeRouter = struct {
    var calls: usize = 0;
    var ctx_seen: ?*anyopaque = null;
    var stored: [8]mlx.mlx_array = @splat(.{});
    var token: u8 = 0;

    fn fill(ctx: ?*anyopaque, w: mlx.mlx_array) anyerror!?moe_block.RouterHook {
        if (calls < stored.len) ProbeRows.keep(&stored[calls], w);
        calls += 1;
        ctx_seen = ctx;
        return .{ .ctx = ctx, .run = &decline };
    }

    fn decline(_: ?*const anyopaque, _: *const moe_block.Moe.Weights, _: ops.Stream, _: mlx.mlx_array) anyerror!?moe_routing.MoeRouting {
        return null;
    }

    fn reset() void {
        for (&stored) |*a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a.*);
            a.* = .{};
        };
        calls = 0;
        ctx_seen = null;
    }

    fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
        return host.load(glm.plugin.name, ctx, .{ .router = .{ .ctx = &token, .fill = &fill } }, out);
    }

    const table = blk: {
        var t = glm.table;
        t.load = &load;
        break :blk t;
    };
};

// Bar: a load's router hook sees each trunk routed layer's router weight as the pack stores it, once a layer, and that
// layer keeps the hook it gives; the dense layer and the MTP head's routed layer get none.
test "glm5_next served: a load's router hook gets each trunk routed layer's stored router weight, and the head's keeps the block's route" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.glmFiles(arena.allocator(), 7);
    var p = try packOf(files);
    defer p.deinit();
    ProbeRouter.reset();
    defer ProbeRouter.reset();
    var b = try both.Both.openWith(&ProbeRouter.table, &p, files[0].bytes, true, .none, &.{});
    defer b.close();
    const m = host.of(b.xfm.plugin.?.model);

    var routed: usize = 0;
    for (m.layers, 0..) |*l, i| switch (l.mlp) {
        .moe => |*w| {
            try testing.expect(w.router_hook != null);
            try testing.expectEqual(@as(?*const anyopaque, &ProbeRouter.token), w.router_hook.?.ctx);
            var nb: [64]u8 = undefined;
            const name = try std.fmt.bufPrintSentinel(&nb, "model.layers.{d}.mlp.gate.weight", .{i}, 0);
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            try mlx.check(mlx.mlx_map_string_to_array_get(&want, p.map, name.ptr));
            try ops.expectBitEqual(ProbeRouter.stored[routed], want);
            routed += 1;
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, fixture.glm.LAYERS - 1), routed);
    try testing.expectEqual(routed, ProbeRouter.calls);
    try testing.expectEqual(@as(?*anyopaque, &ProbeRouter.token), ProbeRouter.ctx_seen);
    try testing.expect(m.mtp.?.moe.router_hook == null);
}
