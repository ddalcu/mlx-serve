//! The template media plugin: every member of the media table over a tiny image generator (seeded noise, `steps`
//! tanh updates, a 4 -> RGB projection, PNG out). Copy the folder to start a backend, renaming `_template_media`.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const Scope = @import("../../mlx/scope.zig").Scope;
const png = @import("../../sdk/media/png.zig");
pub const config = @import("config.zig");

pub const plugin = sdk.Plugin{
    .name = "_template_media",
    .kind = .media,
    .claims = &config.model_types,
};

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of("model.safetensors"),
    .modality = .image,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;
const latent = 4;

const Engine = struct {
    proj: mlx.mlx_array,
    stream: mlx.mlx_stream,
    png: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&path_buf, "{s}/model.safetensors", .{ctx.model_dir.slice()}, 0) catch {
        out.* = Verdict.refuse(.failed, "_template_media: model path too long");
        return null;
    };
    const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
    // MLX has no GPU kernel for a file load: read on a CPU stream, compute on `s`.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var w = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(w);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    var proj = mlx.mlx_array_new();
    if (mlx.mlx_load_safetensors(&w, &meta, path, cpu) != 0 or mlx.mlx_map_string_to_array_get(&proj, w, "proj.weight") != 0 or mlx.mlx_array_eval(proj) != 0) {
        _ = mlx.mlx_array_free(proj);
        out.* = Verdict.refuse(.failed, "_template_media: proj.weight does not load");
        return null;
    }
    const e = gpa.create(Engine) catch {
        _ = mlx.mlx_array_free(proj);
        out.* = Verdict.refuse(.failed, "_template_media: out of memory");
        return null;
    };
    e.* = .{ .proj = proj, .stream = s };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    _ = mlx.mlx_array_free(e.proj);
    gpa.free(e.png);
    gpa.destroy(e);
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const body = config.Body.parse(req.body.slice()) orelse
        return req.refuseFmt(.unsupported, "_template_media: bad request body", "'size' must be in [1,{d}] and 'steps' in [1,{d}]", .{ config.max_dim, config.max_steps });
    const image = run(e, body, progress) catch |err| return switch (err) {
        error.ClientLeft => Verdict.refuse(.failed, "_template_media: the client left"),
        else => Verdict.refuse(.failed, "_template_media: the job failed"),
    };
    gpa.free(e.png);
    e.png = image;
    return .ok;
}

fn run(e: *Engine, b: config.Body, progress: *const abi.Progress) ![]u8 {
    var sc = Scope.init(gpa, e.stream);
    defer sc.deinit();
    const px: c_int = @intCast(b.size * b.size);
    var key = mlx.mlx_array_new();
    _ = try sc.res(mlx.mlx_random_key(&key, b.seed), &key);
    var noise = mlx.mlx_array_new();
    var x = try sc.res(mlx.mlx_random_normal(&noise, &[_]c_int{ px, latent }, 2, .float32, 0, 1, key, e.stream), &noise);
    for (0..b.steps) |i| {
        x = try sc.tanh(x);
        try mlx.check(mlx.mlx_array_eval(x));
        progress.report(progress.ctx, .of("denoise"), @intCast(i + 1), b.steps);
        if (progress.boundary(progress.ctx)) return error.ClientLeft;
    }
    const rgb = try sc.sigmoid(try sc.matmul(x, try sc.transpose(e.proj, &.{ 1, 0 })));
    try mlx.check(mlx.mlx_array_eval(rgb));
    const data = (mlx.mlx_array_data_float32(rgb) orelse return error.NoData)[0 .. @as(usize, @intCast(px)) * 3];
    const bytes = try gpa.alloc(u8, data.len);
    defer gpa.free(bytes);
    for (bytes, data) |*o, v| o.* = @intFromFloat(@round(std.math.clamp(v, 0, 1) * 255));
    return png.encodeRgb(gpa, bytes, b.size, b.size);
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.png.len == 0) return Verdict.refuse(.failed, "_template_media: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

/// The tiny pack the conformance suite runs: a seeded 4 -> RGB projection.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const st = try sdk.fixture.safetensors(a, &.{
        .{ .name = "proj.weight", .shape = &.{ 3, latent }, .data = try sdk.fixture.randomF32(a, 2, 3 * latent, 1) },
    });
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = "{\"model_type\":\"_template_media\"}" },
            .{ .name = "model.safetensors", .bytes = st },
        }),
        .body = "{\"size\":4,\"steps\":2,\"seed\":3}",
    };
}

// ── tests ──

const testing = std.testing;

test "_template_media: the load bill is the weights' bytes from the safetensors header" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const fx = try conformanceFixture(arena.allocator());
    const mem: sdk.fixture.MemDir = .{ .files = fx.files };
    try testing.expectEqual(@as(u64, 3 * latent * 4), config.billLoad(&mem.packDir(), &.{}));
}
