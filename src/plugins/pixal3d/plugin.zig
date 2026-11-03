//! Pixal3D on the media table: TRELLIS.2's cascade and decoders conditioned on pixel-aligned features (each voxel
//! projected into the DINOv3 + NAF-upsampled map of every view), one photo or 2-8 posed views averaged.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const mesh_request = @import("../../sdk/media/mesh_request.zig");
const mlx = @import("../../mlx/ffi.zig");
const t2_plugin = @import("../trellis2/plugin.zig");
const t2_config = @import("../trellis2/config.zig");
const px = @import("pixal3d.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

/// The tiny pack's job peaks near 13 GB of RSS: a 16 GB runner skips it.
pub const test_min_ram_gib = 32;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .mesh,
    .caps = &config.caps,
    .bill_load = &t2_config.billLoad,
    .bill_request = &t2_config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    m: *px.Engine,
    glb: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = px.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |err| {
        out.* = job.failure(err, "pixal3d: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "pixal3d: out of memory");
        return null;
    };
    e.* = .{ .m = m };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.m.deinit();
    gpa.free(e.glb);
    gpa.destroy(e);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

const Parsed = union(enum) { ok: px.Opts, bad: Verdict };

/// TRELLIS.2's fields (no 512 pipeline), `fov` (horizontal, degrees, every view) and, with `images`, `cameras` (one
/// 4x4 camera-to-world per view; the shipped rig when omitted). `ok.cameras` is the caller's to free.
fn parseOpts(body: *const job.Body, n_views: usize, has_mv: bool) Parsed {
    var o = px.Opts{};
    o.base = switch (t2_plugin.parseOpts(body, false)) {
        .ok => |b| b,
        .bad => |v| return .{ .bad = v },
    };
    o.base.pixal3d_axes = true;
    if (body.float("fov")) |f| {
        if (!(f >= 1 and f <= 120)) return .{ .bad = bad("'fov' is the horizontal field of view in degrees, in [1,120]") };
        o.fov = std.math.degreesToRadians(@as(f32, @floatCast(f)));
    }
    if (mesh_request.has(&body.parsed.value.object, "cameras")) {
        if (n_views < 2) return .{ .bad = bad("'cameras' goes with 'images' (several views)") };
        o.cameras = switch (px.parseCameras(gpa, body.get("cameras").?) catch return .{ .bad = Verdict.refuse(.failed, "pixal3d: out of memory") }) {
            .ok => |c| c,
            .bad => |m| return .{ .bad = Verdict.named(.unsupported, m) },
        };
    }
    if (px.viewsRefusal(n_views, o.cameras.len, has_mv)) |m| {
        gpa.free(o.cameras);
        return .{ .bad = Verdict.named(.unsupported, m) };
    }
    return .{ .ok = o };
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    var views: mesh_request.Views = .{};
    defer views.deinit(gpa);
    if (mesh_request.decode(gpa, &body.parsed.value.object, req.inputs.slice(), px.MAX_VIEWS, &views)) |v| return v;
    const n = views.list.items.len;
    const opts = switch (parseOpts(&body, n, e.m.mv != null)) {
        .ok => |o| o,
        .bad => |v| return v,
    };
    defer gpa.free(opts.cameras);
    if (t2_plugin.resolutionRefusal(req, opts.base.resolution)) |v| return v;
    var photos: [px.MAX_VIEWS]px.Photo = undefined;
    for (views.list.items, 0..) |v, i| photos[i] = .{ .rgba = v.pix, .w = v.w, .h = v.h };
    const fov = opts.fov orelse (if (n > 1) px.RIG_FOV else px.DEFAULT_FOV);
    log.info("[mesh] generating Pixal3D res={d} views={d} cameras={s} fov={d:.1}deg steps={d} faces={d} tex={d} seed={d}\n", .{ opts.base.resolution, n, if (opts.cameras.len > 0) "given" else "default", std.math.radiansToDegrees(fov), opts.base.steps, opts.base.decimation_target, opts.base.texture_size, opts.base.seed });
    const out = e.m.generateGlb(gpa, photos[0..n], opts, job.progressOf(progress)) catch |err| return t2_plugin.jobFailure(err);
    if (job.latched()) |v| {
        gpa.free(out);
        return v;
    }
    gpa.free(e.glb);
    e.glb = out;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.glb.len == 0) return Verdict.refuse(.failed, "pixal3d: no finished job to encode");
    out.* = .of(e.glb);
    return .ok;
}

// ── the tiny pack ──

const T = sdk.tensors.Tensors;

fn conv(t: *T, comptime name: []const u8, out: u64, in: u64, k: u64) !void {
    try t.rand(name ++ ".weight", .{}, &.{ out, in, k, k }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(in * k * k))));
    try t.rand(name ++ ".bias", .{}, &.{out}, .F32, 0.05);
}

/// One NAF guide encoder: 3 to 8 channels, two GroupNorm blocks, the last conv out to half the 256 query width.
fn nafEncoder(t: *T, comptime prefix: []const u8, k: u64) !void {
    try conv(t, prefix ++ ".0", 8, 3, k);
    inline for (.{ ".1", ".2" }, .{ 8, 128 }) |b, last| {
        inline for (.{ ".norm1", ".norm2" }) |n| {
            try t.fill(prefix ++ b ++ n ++ ".weight", .{}, &.{8}, .F32, 1);
            try t.fill(prefix ++ b ++ n ++ ".bias", .{}, &.{8}, .F32, 0);
        }
        try conv(t, prefix ++ b ++ ".conv1", 8, 8, k);
        try conv(t, prefix ++ b ++ ".conv2", last, 8, k);
    }
}

/// TRELLIS.2's tiny pack with projecting flows, plus NAF: 1x1 and 3x3 guide encoders and the RoPE periods.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var files = try t2_plugin.tinyPack(a, true);
    var naf = T.init(a, 68);
    try nafEncoder(&naf, "image_encoder.encoder", 1);
    try nafEncoder(&naf, "image_encoder.sem_encoder", 3);
    try naf.fill("image_encoder.rope.periods", .{}, &.{16}, .F32, 4);
    try files.append(a, .{ .name = "naf/model.safetensors", .bytes = try naf.bytes() });
    return .{
        .files = try files.toOwnedSlice(a),
        .body = "{\"image\":\"x\",\"steps\":1,\"texture_size\":512,\"decimation_target\":10000,\"seed\":3}",
        .inputs = try a.dupe([]const u8, &.{try t2_plugin.cutout(a)}),
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const golden = @import("../../sdk/testing/golden.zig");

fn parsed(json: []const u8, n_views: usize) !Parsed {
    var b = job.Body.parse(testing.allocator, json).?;
    defer b.deinit();
    return parseOpts(&b, n_views, true);
}

test "pixal3d: request fields: FOV in degrees, a camera per view, the TRELLIS.2 finish fields" {
    const d = (try parsed("{}", 1)).ok;
    try testing.expectEqual(@as(u32, 1024), d.base.resolution);
    try testing.expect(d.base.pixal3d_axes and d.fov == null and d.cameras.len == 0);
    const f = (try parsed("{\"fov\":30,\"resolution\":1536,\"roughness_scale\":0.5,\"clearcoat\":0.6}", 1)).ok;
    try testing.expectApproxEqAbs(std.math.degreesToRadians(@as(f32, 30)), f.fov.?, 1e-6);
    try testing.expectEqual(@as(u32, 1536), f.base.resolution);
    try testing.expectApproxEqAbs(@as(f32, 0.5), f.base.roughness_scale, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.6), f.base.clearcoat, 1e-6);
    const cams = "{\"cameras\":[[[1,0,0,0],[0,0,-1,-3],[0,1,0,0],[0,0,0,1]],[[0,0,1,3],[1,0,0,0],[0,1,0,0],[0,0,0,1]]]}";
    const c = (try parsed(cams, 2)).ok;
    defer gpa.free(c.cameras);
    try testing.expectEqual(@as(usize, 2), c.cameras.len);
    try testing.expectEqual(@as(f32, 3), c.cameras[1][0][3]);
    try testing.expect(std.mem.indexOf(u8, (try parsed(cams, 3)).bad.why.slice(), "'cameras'") != null);
    try testing.expect(std.mem.indexOf(u8, (try parsed("{}", 5)).bad.why.slice(), "'cameras'") != null);
    try testing.expect(std.mem.indexOf(u8, (try parsed("{\"fov\":0.4}", 1)).bad.why.slice(), "'fov'") != null);
}

test "pixal3d: a job on the tiny pack is a GLB; views, cameras and fields are refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const one = [_]abi.Str{.of(pack.fx.inputs[0])};
    const glb = (try pack.runInputs(pack.fx.body, &one)).ok;
    try testing.expectEqualStrings("glTF", glb[0..4]);
    const two = [_]abi.Str{ one[0], one[0] };
    const cases = [_][2][]const u8{
        .{ "{}", "missing 'image' (base64 PNG/JPEG of the subject)" },
        .{ "{\"image\":\"x\",\"images\":[\"x\",\"y\"]}", "send 'image' or 'images', not both" },
        .{ "{\"images\":\"x\"}", mesh_request.BAD_IMAGES },
        .{ "{\"images\":[\"x\"]}", "'images' needs at least 2 views (send a single view as 'image')" },
        .{ "{\"images\":[\"x\",\"y\"]}", "this Pixal3D pack has no multi-view models (pipeline_mv.json); send one 'image'" },
        .{ "{\"image\":\"x\",\"resolution\":512}", "'resolution' must be 1024 or 1536 (this pack has no 512 texture model)" },
        .{ "{\"image\":\"x\",\"fov\":150}", "'fov' is the horizontal field of view in degrees, in [1,120]" },
        .{ "{\"image\":\"x\",\"cameras\":[]}", "'cameras' goes with 'images' (several views)" },
        .{ "{\"images\":[\"x\",\"y\"],\"cameras\":[[1,2]]}", px.BAD_CAMERAS },
    };
    for (cases) |c| try testing.expectEqualStrings(c[1], (try pack.runInputs(c[0], &two)).refused.why.slice());
}

test "pixal3d: the tiny pack's voxels (conditioning, structure, cascade, texture) are bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const img = image_pixels.decodeRgba(gpa, pack.fx.inputs[0]).?;
    defer img.deinit(gpa);
    const photo = [_]px.Photo{.{ .rgba = img.pix, .w = img.w, .h = img.h }};
    var vox = try e.m.generateVoxels(gpa, &photo, .{ .base = .{ .steps = 2, .seed = 3, .pixal3d_axes = true } }, null);
    defer vox.deinit(gpa);
    const arrays = t2_plugin.voxelArrays(&vox);
    defer for (arrays) |x| {
        _ = mlx.mlx_array_free(x.array);
    };
    try golden.expect("pixal3d_voxels", &arrays);
}
