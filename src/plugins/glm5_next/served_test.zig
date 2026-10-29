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
const File = @import("../../sdk/root.zig").fixture.File;
const fixture = @import("fixture.zig");
const glm = @import("plugin.zig");
const ling = @import("../bailing_hybrid/plugin.zig");
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
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try glmServed(try fixture.glmFiles(arena.allocator(), 7), "glm5_next/fixtures/served.tape");
}

test "glm5_next served: a Sushi pack (EXL3 experts) through the driver is bit-equal, greedy decode and media too" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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
