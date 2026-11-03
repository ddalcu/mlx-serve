//! The template model plugin: every member of the table, minimal or stubbed, over a tiny model (a tied embedding and
//! one RMS norm: `logits = rms(embed[ids]) @ embed^T`). Copy the folder to start a family, renaming `_template`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const Scope = @import("../../mlx/scope.zig").Scope;
pub const config = @import("config.zig");

pub const plugin = sdk.Plugin{
    .name = "_template",
    .kind = .model,
    .claims = &config.model_types,
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{},
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    _ = shape;
    const c = config.of(p);
    // f32 weights; no attention layer, so no KV.
    out.* = .{ .load_bytes = (@as(u64, c.vocab_size) * c.hidden_size + c.hidden_size) * 4 };
}

fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const n = @min(if (span) |s| s.count else config.of(p).num_hidden_layers, cap);
    for (out[0..n]) |*l| l.* = .{ .kind = .none };
    return n;
}

const Model = struct {
    embed: mlx.mlx_array,
    norm: mlx.mlx_array,
    eps: f32,
    vocab: u32,
    caps: abi.Caps,
    stream: mlx.mlx_stream,
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const c = config.of(ctx.parsed);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&path_buf, "{s}/model.safetensors", .{ctx.model_dir.slice()}, 0) catch {
        out.* = Verdict.refuse(.failed, "_template: model path too long");
        return null;
    };
    const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
    var w = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(w);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    if (mlx.mlx_load_safetensors(&w, &meta, path, s) != 0) {
        out.* = Verdict.refuse(.failed, "_template: model.safetensors does not load");
        return null;
    }
    const m = gpa.create(Model) catch {
        out.* = Verdict.refuse(.failed, "_template: out of memory");
        return null;
    };
    m.* = .{ .embed = mlx.mlx_array_new(), .norm = mlx.mlx_array_new(), .eps = c.rms_norm_eps, .vocab = c.vocab_size, .caps = ctx.parsed.caps, .stream = s };
    if (mlx.mlx_map_string_to_array_get(&m.embed, w, "embed_tokens.weight") != 0 or
        mlx.mlx_map_string_to_array_get(&m.norm, w, "norm.weight") != 0 or
        mlx.mlx_array_eval(m.embed) != 0 or mlx.mlx_array_eval(m.norm) != 0)
    {
        unload(m);
        out.* = Verdict.refuse(.failed, "_template: a weight is missing");
        return null;
    }
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    _ = mlx.mlx_array_free(m.embed);
    _ = mlx.mlx_array_free(m.norm);
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "_template: forward mode not declared");
    if (f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "_template: no layer capture");
    const slots: u32 = @intCast(@max(f.slots.len, 1));
    const ids = f.tokens.slice();
    if (f.rows == 0 or ids.len != @as(usize, f.rows) * slots) return Verdict.refuse(.failed, "_template: tokens are not rows x slots");
    for (ids) |t| if (t >= m.vocab) return Verdict.refuse(.failed, "_template: token id past the vocabulary");
    run(m, f, slots, out) catch return Verdict.refuse(.failed, "_template: an MLX op failed");
    return .ok;
}

fn run(m: *Model, f: *const abi.Forward, slots: u32, out: *abi.ForwardOut) !void {
    var sc = Scope.init(gpa, m.stream);
    defer sc.deinit();
    const ids = f.tokens.slice();
    const n: c_int = @intCast(ids.len);
    const x = try sc.take(m.embed, try sc.keep(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{n}, 1, .uint32)));
    var h = try sc.rms(x, m.norm, m.eps);
    if (f.logits == .last_row) {
        const rows: c_int = @intCast(f.rows);
        const hidden = try sc.reshape(h, &.{ @intCast(slots), rows, -1 });
        h = try sc.reshape(try sc.slice(hidden, 1, rows - 1, rows), &.{ @intCast(slots), -1 });
    }
    if (f.logits != .none) out.logits = .{ .ctx = sc.out(try sc.matmul(h, try sc.transpose(m.embed, &.{ 1, 0 }))).ctx };
    if (f.want_hidden) out.hidden = .{ .ctx = sc.out(h).ctx };
}

/// The tiny pack the conformance suite runs: 16 tokens, width 8, seeded random weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 16;
    const h = 8;
    const norm = try a.alloc(f32, h);
    @memset(norm, 1);
    const st = try sdk.fixture.safetensors(a, &.{
        .{ .name = "embed_tokens.weight", .shape = &.{ v, h }, .data = try sdk.fixture.randomF32(a, 1, v * h, 0.5) },
        .{ .name = "norm.weight", .shape = &.{h}, .data = norm },
    });
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes = "{\"model_type\":\"_template\",\"vocab_size\":16,\"hidden_size\":8,\"num_hidden_layers\":1,\"rms_norm_eps\":1e-6}" },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.7}" },
        .{ .name = "model.safetensors", .bytes = st },
    }) };
}

// ── tests ──

const testing = std.testing;
const conformance = @import("../../sdk/testing/conformance.zig");

test "_template: decode logits equal the host reference rms(embed[t]) . embed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try conformance.Pack(@This()).open(arena.allocator());
    defer pack.close();
    const m: *Model = @ptrCast(@alignCast(pack.model));
    const token: u32 = 5;
    var out: abi.ForwardOut = .{};
    try testing.expect(forward(m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&.{token}) }, &out).isOk());
    const logits = try conformance.consumeF32(arena.allocator(), out.logits);

    // The host reference, from the fixture's own weights.
    var e = mlx.mlx_array_new();
    try testing.expectEqual(@as(c_int, 0), mlx.mlx_array_set(&e, m.embed));
    const embed = try conformance.consumeF32(arena.allocator(), .{ .ctx = e.ctx });
    const h = 8;
    const x = embed[token * h ..][0..h];
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / h + m.eps);
    for (0..16) |j| {
        var dot: f32 = 0;
        for (0..h) |k| dot += x[k] * inv * embed[j * h + k];
        try testing.expectApproxEqAbs(dot, logits[j], 1e-4);
    }
}
