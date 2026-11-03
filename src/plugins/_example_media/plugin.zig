//! The example outside media plugin, built from the kit as a dylib: a sine tone scaled by the pack's one weight,
//! computed with MLX in chunks (a progress report and a boundary between them), returned as a WAV.
//! `mlx-serve plugin new <name> --media` copies this folder.

const std = @import("std");
const kit = @import("mlx_serve");
const c = @import("mlx_c");
const abi = kit.abi.media;
pub const config = @import("config.zig");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "_example_media",
    .kind = .media,
    .claims = &config.model_types,
};

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of("model.safetensors"),
    .modality = .audio,
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

const Engine = struct {
    gain: f32,
    stream: c.mlx_stream,
    wav: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&path_buf, "{s}/model.safetensors", .{ctx.model_dir.slice()}, 0) catch {
        out.* = Verdict.refuse(.failed, "_example_media: model path too long");
        return null;
    };
    const s: c.mlx_stream = .{ .ctx = ctx.stream.ctx };
    // MLX has no GPU kernel for a file load: read on a CPU stream, compute on `s`.
    const cpu = c.mlx_default_cpu_stream_new();
    defer _ = c.mlx_stream_free(cpu);
    var w = c.mlx_map_string_to_array_new();
    defer _ = c.mlx_map_string_to_array_free(w);
    var meta = c.mlx_map_string_to_string_new();
    defer _ = c.mlx_map_string_to_string_free(meta);
    var gain = c.mlx_array_new();
    defer _ = c.mlx_array_free(gain);
    var g: f32 = 0;
    if (c.mlx_load_safetensors(&w, &meta, path, cpu) != 0 or c.mlx_map_string_to_array_get(&gain, w, "gain") != 0 or
        c.mlx_array_eval(gain) != 0 or c.mlx_array_item_float32(&g, gain) != 0)
    {
        out.* = Verdict.refuse(.failed, "_example_media: the gain weight does not load");
        return null;
    }
    const e = gpa.create(Engine) catch {
        out.* = Verdict.refuse(.failed, "_example_media: out of memory");
        return null;
    };
    e.* = .{ .gain = g, .stream = s };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    gpa.free(e.wav);
    gpa.destroy(e);
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const body = config.Body.parse(req.body.slice()) orelse
        return req.refuseFmt(.unsupported, "_example_media: bad request body", "'seconds' must be in (0,{d}] and 'frequency' below {d} Hz", .{ config.max_seconds, config.sample_rate / 2 });
    const wav = run(e, body, progress) catch |err| return switch (err) {
        error.ClientLeft => Verdict.refuse(.failed, "_example_media: the client left"),
        else => Verdict.refuse(.failed, "_example_media: the job failed"),
    };
    gpa.free(e.wav);
    e.wav = wav;
    return .ok;
}

fn run(e: *Engine, b: config.Body, progress: *const abi.Progress) ![]u8 {
    const n = b.samples();
    const pcm = try gpa.alloc(f32, n);
    defer gpa.free(pcm);
    const per = (n + config.chunks - 1) / config.chunks;
    var at: u32 = 0;
    var i: u32 = 0;
    while (at < n) : (i += 1) {
        const len = @min(per, n - at);
        try tone(e, b.frequency, at, pcm[at..][0..len]);
        at += len;
        progress.report(progress.ctx, .of("tone"), i + 1, config.chunks);
        if (progress.boundary(progress.ctx)) return error.ClientLeft;
    }
    return kit.wav.encodePcm16Mono(gpa, pcm, config.sample_rate);
}

/// `gain * sin(2 pi f t)` for samples `first ..` into `out`, on the GPU stream the host handed over.
fn tone(e: *Engine, freq: f32, first: u32, out: []f32) !void {
    var t = c.mlx_array_new();
    defer _ = c.mlx_array_free(t);
    var phase = c.mlx_array_new();
    defer _ = c.mlx_array_free(phase);
    var y = c.mlx_array_new();
    defer _ = c.mlx_array_free(y);
    const w = c.mlx_array_new_float(2 * std.math.pi * freq / config.sample_rate);
    defer _ = c.mlx_array_free(w);
    const start: f64 = @floatFromInt(first);
    if (c.mlx_arange(&t, start, start + @as(f64, @floatFromInt(out.len)), 1, c.MLX_FLOAT32, e.stream) != 0 or
        c.mlx_multiply(&phase, t, w, e.stream) != 0 or c.mlx_sin(&y, phase, e.stream) != 0 or c.mlx_array_eval(y) != 0)
        return error.MlxOpFailed;
    if (c.mlx_array_size(y) != out.len) return error.MlxOpFailed;
    const data = c.mlx_array_data_float32(y) orelse return error.MlxOpFailed;
    for (out, data[0..out.len]) |*o, v| o.* = e.gain * v;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.wav.len == 0) return Verdict.refuse(.failed, "_example_media: no finished job to encode");
    out.* = .of(e.wav);
    return .ok;
}

/// The tiny pack the conformance suite runs: the one gain weight.
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    const fx = kit.sdk.fixture;
    return .{
        .files = try a.dupe(fx.File, &.{
            .{ .name = "config.json", .bytes = "{\"model_type\":\"_example_media\"}" },
            .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, &.{.{ .name = "gain", .shape = &.{1}, .data = &.{0.5} }}) },
        }),
        .body = "{\"seconds\":0.01,\"frequency\":440}",
    };
}
