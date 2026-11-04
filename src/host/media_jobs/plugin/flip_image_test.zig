//! The image and mesh flip bar: each backend's tiny pack reports the model type, completion marker and modality its
//! legacy discovery gave it, and bills the load and per-request reserve its legacy gen.zig route did (the legacy
//! numbers were measured on the legacy arms before they were deleted, on a 16 GiB and a 192 GiB working set).

const std = @import("std");
const testing = std.testing;
const abi = @import("../../../abi/media.zig");
const served = @import("served.zig");
const discovery = @import("../../registry/discovery.zig");
const modality = @import("../../registry/modality.zig");
const FsPackDir = @import("../../registry/pack_dir.zig").FsPackDir;
const mlx = @import("../../../mlx/ffi.zig");
const gen = @import("../gen.zig");
const http = @import("http.zig");
const Engine = @import("engine.zig").Engine;
const Conn = @import("../../http/server.zig").Conn;
const capture = @import("test_conn.zig").capture;

const Legacy = struct {
    type: []const u8,
    marker: ?[]const u8 = null,
    modality: modality.Modality,
    /// The load bill on a 16 GiB and a 192 GiB working set.
    load: [2]u64,
    /// The reserve for `bodies`, 0 = the load bill.
    reserve: [bodies.len]u64 = .{ 0, 0, 0, 0 },
};

const bodies = [_][]const u8{ "{}", "{\"resolution\":512}", "{\"resolution\":1536}", "{\"resolution\":1024}" };
const machines = [_]abi.Machine{ .{ .gpu_working_set = 16 << 30 }, .{ .gpu_working_set = 192 << 30 } };
const trellis_reserve = .{ 24117248000, 6081740800, 37539020800, 24117248000 };

/// Plugin root `P`'s fixture written into `tmp`; its body.
fn stage(comptime P: type, a: std.mem.Allocator, tmp: *testing.TmpDir) ![]const u8 {
    const fx = try P.conformanceFixture(a);
    for (fx.files) |f| {
        if (std.fs.path.dirname(f.name)) |d| try tmp.dir.createDirPath(testing.io, d);
        try tmp.dir.writeFile(testing.io, .{ .sub_path = f.name, .data = f.bytes });
    }
    return fx.body;
}

fn expectLegacy(comptime P: type, want: Legacy) !void {
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    _ = try stage(P, a, &tmp);
    const mt = served.claimTypeIn(io, tmp.dir, a).?;
    try testing.expectEqualStrings(want.type, mt);
    const t = served.byName(mt).?;
    try testing.expectEqual(&P.table, t);
    const marker = discovery.requiredMediaMarker(mt);
    try testing.expectEqual(want.marker == null, marker == null);
    if (want.marker) |w| try testing.expectEqualStrings(w, marker.?);
    try testing.expectEqual(want.modality, modality.fromType(mt).?);
    const fs: FsPackDir = .{ .io = io, .root = tmp.dir };
    const pd = fs.packDir();
    for (&machines, want.load) |*m, l| {
        try testing.expectEqual(l, t.bill_load(&pd, m));
        for (bodies, want.reserve) |b, r| try testing.expectEqual(r, t.bill_request(&pd, m, .of(b)));
    }
}

test "flip (image): flux2, krea, mage_flow, qwen_image and moss_image report and bill as their legacy routes" {
    try expectLegacy(@import("../../../plugins/flux2/plugin.zig"), .{ .type = "flux2-klein-4b", .modality = .image, .load = .{ 10860352, 10860352 } });
    try expectLegacy(@import("../../../plugins/krea/plugin.zig"), .{ .type = "krea2_turbo", .modality = .image, .load = .{ 29122109, 29122109 } });
    try expectLegacy(@import("../../../plugins/mage_flow/plugin.zig"), .{ .type = "mage_flow", .modality = .image, .load = .{ 3995376, 3995376 } });
    try expectLegacy(@import("../../../plugins/qwen_image/plugin.zig"), .{ .type = "qwen_image21", .modality = .image, .load = .{ 5102289015, 5102289015 } });
    try expectLegacy(@import("../../../plugins/moss_image/plugin.zig"), .{ .type = "moss_image", .marker = "vae/diffusion_pytorch_model.safetensors", .modality = .image, .load = .{ 6980439812, 6980439812 } });
}

test "flip (mesh): trellis2, pixal3d and hunyuan3d report and bill as their legacy routes" {
    try expectLegacy(@import("../../../plugins/trellis2/plugin.zig"), .{
        .type = "trellis2",
        .marker = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        .modality = .mesh,
        .load = .{ 6081740800, 24117248000 },
        .reserve = trellis_reserve,
    });
    try expectLegacy(@import("../../../plugins/pixal3d/plugin.zig"), .{
        .type = "pixal3d",
        .marker = "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        .modality = .mesh,
        .load = .{ 24117248000, 24117248000 },
        .reserve = trellis_reserve,
    });
    try expectLegacy(@import("../../../plugins/hunyuan3d/plugin.zig"), .{ .type = "hunyuan3d_2_1", .modality = .mesh, .load = .{ 471809, 471809 } });
}

fn imageRoute(conn: *Conn, a: std.mem.Allocator, body: []const u8, e: *Engine) !void {
    return http.handle(a, conn, body, e, gen.GenRoute.image, "m");
}

test "flip (image): a lazy result that fails at the encode answers 503 by name, never a closed socket" {
    const io = testing.io;
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const body = try stage(@import("../../../plugins/moss_image/plugin.zig"), arena.allocator(), &tmp);
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = pbuf[0..try tmp.dir.realPath(io, &pbuf)];
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const e = try Engine.load(io, a, path, served.claimPath(io, path).?, s);
    defer e.deinit();
    // The job's checked mlx calls, counted until two runs agree: the PNG encode's eval of the lazy image is the last
    // two, so a failure latched at the first of them is a denoise that failed inside the encode.
    var ops: u64 = 0;
    for (0..3) |_| {
        const n0 = mlx.op_count.load(.monotonic);
        const ok = try capture(a, imageRoute, .{ a, body, e });
        defer a.free(ok);
        try testing.expect(std.mem.startsWith(u8, ok, "HTTP/1.1 200"));
        const n = mlx.op_count.load(.monotonic) - n0;
        if (n == ops) break;
        ops = n;
    }
    var stale: [512]u8 = undefined;
    _ = mlx.takeError(&stale);
    defer _ = mlx.takeError(&stale);
    mlx.armLatchingFaultWithForTest(ops - 1, "[METAL] Command buffer execution failed: Insufficient Memory. at array.cpp:352");
    defer mlx.armLatchingFaultForTest(0);
    const got = try capture(a, imageRoute, .{ a, body, e });
    defer a.free(got);
    try testing.expect(mlx.latchingFaultFiredForTest());
    try testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 503"));
    try testing.expect(std.mem.indexOf(u8, got, "generation failed") != null);
}

