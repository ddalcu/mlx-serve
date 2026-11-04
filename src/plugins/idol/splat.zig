//! The decoded Gaussians as files: the standard 3D Gaussian Splatting `.ply` (what every splat viewer opens) and a
//! glTF binary with the ratified `KHR_gaussian_splatting` extension on one POINTS primitive, the form the 3D
//! endpoint returns. The PLY turns the model's z-up object the way the reference's `save_ply` does (what splat viewers
//! expect: their +y points down); the GLB into glTF's y-up. The GLB carries a `COLOR_0` too, so a viewer without the extension still draws the object as a coloured point cloud.

const std = @import("std");
const dec = @import("decoders.zig");
const png = @import("../../sdk/media/png.zig");

const Splats = dec.Splats;
const SH0: f32 = 0.28209479177387814;
const EPS: f32 = 1e-6;

const Axes = enum {
    /// The reference's `save_ply`: +90 degrees about x, (x, y, z) -> (x, -z, y).
    ply,
    /// glTF's y-up: -90 degrees about x, (x, y, z) -> (x, z, -y).
    gltf,
};

fn turn(comptime axes: Axes, x: f32, y: f32, z: f32) [3]f32 {
    return switch (axes) {
        .ply => .{ x, -z, y },
        .gltf => .{ x, z, -y },
    };
}

/// `q` (w, x, y, z) premultiplied by the turn about x, (w, x, y, z) again.
fn turnRot(comptime axes: Axes, q: []const f32) [4]f32 {
    const h: f32 = @sqrt(0.5);
    return switch (axes) {
        .ply => .{ h * q[0] - h * q[1], h * q[1] + h * q[0], h * q[2] - h * q[3], h * q[3] + h * q[2] },
        .gltf => .{ h * q[0] + h * q[1], h * q[1] - h * q[0], h * q[2] + h * q[3], h * q[3] - h * q[2] },
    };
}

fn inverseSigmoid(p: f32) f32 {
    const c = std.math.clamp(p, EPS, 1 - EPS);
    return @log(c / (1 - c));
}

const PLY_PROPS = [_][]const u8{ "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

/// `save_ply`: binary little-endian, 17 float32 properties per Gaussian; opacity and scale in their raw (logit, log)
/// form. Caller frees.
pub fn writePly(gpa: std.mem.Allocator, s: *const Splats) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.print(gpa, "ply\nformat binary_little_endian 1.0\nelement vertex {d}\n", .{s.n});
    for (PLY_PROPS) |p| try out.print(gpa, "property float {s}\n", .{p});
    try out.appendSlice(gpa, "end_header\n");
    try out.ensureUnusedCapacity(gpa, s.n * PLY_PROPS.len * 4);
    for (0..s.n) |i| {
        const p = turn(.ply, s.xyz[i * 3], s.xyz[i * 3 + 1], s.xyz[i * 3 + 2]);
        const q = turnRot(.ply, s.rot[i * 4 ..][0..4]);
        const row = [_]f32{
            p[0],                                 p[1],                                  p[2],
            0,                                    0,                                     0,
            s.dc[i * 3],                          s.dc[i * 3 + 1],                       s.dc[i * 3 + 2],
            inverseSigmoid(s.opacity[i]),
            @log(@max(s.scale[i * 3], EPS)),      @log(@max(s.scale[i * 3 + 1], EPS)),  @log(@max(s.scale[i * 3 + 2], EPS)),
            q[0],                                 q[1],                                  q[2],                                  q[3],
        };
        out.appendSliceAssumeCapacity(std.mem.sliceAsBytes(&row));
    }
    return out.toOwnedSlice(gpa);
}

fn pad4(list: *std.ArrayList(u8), gpa: std.mem.Allocator, fill: u8) !void {
    while (list.items.len % 4 != 0) try list.append(gpa, fill);
}

/// A GLB whose single POINTS primitive is a `KHR_gaussian_splatting` ellipse-kernel field. Caller frees.
pub fn writeGlb(gpa: std.mem.Allocator, s: *const Splats) ![]u8 {
    const n = s.n;
    var bin: std.ArrayList(u8) = .empty;
    defer bin.deinit(gpa);
    var views: [6][2]usize = undefined; // offset, length
    var lo = [3]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var hi = [3]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };

    const pos = try gpa.alloc(f32, n * 3);
    defer gpa.free(pos);
    const rot = try gpa.alloc(f32, n * 4);
    defer gpa.free(rot);
    const col = try gpa.alloc(u8, n * 4);
    defer gpa.free(col);
    for (0..n) |i| {
        const p = turn(.gltf, s.xyz[i * 3], s.xyz[i * 3 + 1], s.xyz[i * 3 + 2]);
        for (0..3) |a| {
            pos[i * 3 + a] = p[a];
            lo[a] = @min(lo[a], p[a]);
            hi[a] = @max(hi[a], p[a]);
        }
        const q = turnRot(.gltf, s.rot[i * 4 ..][0..4]); // glTF order x, y, z, w
        rot[i * 4 ..][0..4].* = .{ q[1], q[2], q[3], q[0] };
        for (0..3) |a| col[i * 4 + a] = @intFromFloat(@round(std.math.clamp(0.5 + SH0 * s.dc[i * 3 + a], 0, 1) * 255));
        col[i * 4 + 3] = @intFromFloat(@round(std.math.clamp(s.opacity[i], 0, 1) * 255));
    }
    const blobs = [_][]const u8{
        std.mem.sliceAsBytes(pos),
        std.mem.sliceAsBytes(rot),
        std.mem.sliceAsBytes(s.scale),
        std.mem.sliceAsBytes(s.opacity),
        std.mem.sliceAsBytes(s.dc),
        col,
    };
    for (blobs, 0..) |b, i| {
        try pad4(&bin, gpa, 0);
        views[i] = .{ bin.items.len, b.len };
        try bin.appendSlice(gpa, b);
    }
    try pad4(&bin, gpa, 0);

    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.print(gpa, "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"mlx-serve idol\"}}," ++
        "\"extensionsUsed\":[\"KHR_gaussian_splatting\"],\"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],\"nodes\":[{{\"mesh\":0}}]," ++
        "\"meshes\":[{{\"primitives\":[{{\"mode\":0,\"attributes\":{{\"POSITION\":0,\"COLOR_0\":5," ++
        "\"KHR_gaussian_splatting:ROTATION\":1,\"KHR_gaussian_splatting:SCALE\":2,\"KHR_gaussian_splatting:OPACITY\":3," ++
        "\"KHR_gaussian_splatting:SH_DEGREE_0_COEF_0\":4}},\"extensions\":{{\"KHR_gaussian_splatting\":" ++
        "{{\"kernel\":\"ellipse\",\"colorSpace\":\"srgb_rec709_display\",\"projection\":\"perspective\",\"sortingMethod\":\"cameraDistance\"}}}}}}]}}]," ++
        "\"buffers\":[{{\"byteLength\":{d}}}],\"bufferViews\":[", .{bin.items.len});
    for (views, 0..) |v, i| try json.print(gpa, "{s}{{\"buffer\":0,\"byteOffset\":{d},\"byteLength\":{d}}}", .{ if (i == 0) "" else ",", v[0], v[1] });
    try json.print(gpa, "],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":{d},\"type\":\"VEC3\"," ++
        "\"min\":[{d},{d},{d}],\"max\":[{d},{d},{d}]}}," ++
        "{{\"bufferView\":1,\"componentType\":5126,\"count\":{d},\"type\":\"VEC4\"}}," ++
        "{{\"bufferView\":2,\"componentType\":5126,\"count\":{d},\"type\":\"VEC3\"}}," ++
        "{{\"bufferView\":3,\"componentType\":5126,\"count\":{d},\"type\":\"SCALAR\"}}," ++
        "{{\"bufferView\":4,\"componentType\":5126,\"count\":{d},\"type\":\"VEC3\"}}," ++
        "{{\"bufferView\":5,\"componentType\":5121,\"normalized\":true,\"count\":{d},\"type\":\"VEC4\"}}]}}", .{ n, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], n, n, n, n, n });
    try pad4(&json, gpa, ' ');

    const total: u32 = @intCast(12 + 8 + json.items.len + 8 + bin.items.len);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, total);
    const w32 = struct {
        fn put(l: *std.ArrayList(u8), v: u32) void {
            l.appendSliceAssumeCapacity(std.mem.asBytes(&std.mem.nativeToLittle(u32, v)));
        }
    }.put;
    w32(&out, 0x46546C67);
    w32(&out, 2);
    w32(&out, total);
    w32(&out, @intCast(json.items.len));
    w32(&out, 0x4E4F534A);
    out.appendSliceAssumeCapacity(json.items);
    w32(&out, @intCast(bin.items.len));
    w32(&out, 0x004E4942);
    out.appendSliceAssumeCapacity(bin.items);
    return out.toOwnedSlice(gpa);
}

/// Gaussians fainter than this are left out of the disc mesh.
const DISC_MIN_OPACITY: f32 = 0.05;
/// A disc reaches this many standard deviations along its two longest axes, at most `DISC_MAX_REACH` units (the
/// object spans about one): a larger flat Gaussian only blurs what the blend would soften.
const DISC_REACH: f32 = 1.5;
const DISC_MAX_REACH: f32 = 0.012;

/// The splat as a plain triangle GLB any viewer draws solid (Preview, Quick Look, three.js without the extension):
/// one opaque quad per Gaussian in the plane of its two longest axes, its colour a texel of an atlas the four corners
/// all point at (viewers ignore per-vertex colour). An approximation of the blended splat: a glassy or hazy region
/// shows the layer a viewer happens to draw first. Caller frees.
pub fn writeDiscGlb(gpa: std.mem.Allocator, s: *const Splats) ![]u8 {
    var kept: std.ArrayList(u32) = .empty;
    defer kept.deinit(gpa);
    for (0..s.n) |i| if (s.opacity[i] >= DISC_MIN_OPACITY) try kept.append(gpa, @intCast(i));
    const n = kept.items.len;
    if (n == 0) return error.EmptySplat;
    const side: usize = std.math.sqrt(n - 1) + 1;

    const pos = try gpa.alloc(f32, n * 4 * 3);
    defer gpa.free(pos);
    const uv = try gpa.alloc(f32, n * 4 * 2);
    defer gpa.free(uv);
    const idx = try gpa.alloc(u32, n * 6);
    defer gpa.free(idx);
    const atlas = try gpa.alloc(u8, side * side * 3);
    defer gpa.free(atlas);
    @memset(atlas, 0);
    var lo = [3]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var hi = [3]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    for (kept.items, 0..) |g, k| {
        const q = s.rot[g * 4 ..][0..4]; // w, x, y, z
        const w = q[0];
        const x = q[1];
        const y = q[2];
        const z = q[3];
        const axes = [3][3]f32{
            .{ 1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w) },
            .{ 2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w) },
            .{ 2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y) },
        };
        const sc: []const f32 = s.scale[g * 3 ..][0..3];
        var order = [3]usize{ 0, 1, 2 };
        std.mem.sort(usize, &order, sc, struct {
            fn gt(c: []const f32, a: usize, b: usize) bool {
                return c[a] > c[b];
            }
        }.gt);
        const ra = @min(DISC_REACH * sc[order[0]], DISC_MAX_REACH);
        const rb = @min(DISC_REACH * sc[order[1]], DISC_MAX_REACH);
        const corners = [4][2]f32{ .{ -1, -1 }, .{ 1, -1 }, .{ -1, 1 }, .{ 1, 1 } };
        for (corners, 0..) |c, ci| {
            var p: [3]f32 = undefined;
            for (0..3) |a| p[a] = s.xyz[g * 3 + a] + c[0] * ra * axes[order[0]][a] + c[1] * rb * axes[order[1]][a];
            const t = turn(.gltf, p[0], p[1], p[2]);
            for (0..3) |a| {
                pos[(k * 4 + ci) * 3 + a] = t[a];
                lo[a] = @min(lo[a], t[a]);
                hi[a] = @max(hi[a], t[a]);
            }
            uv[(k * 4 + ci) * 2] = (@as(f32, @floatFromInt(k % side)) + 0.5) / @as(f32, @floatFromInt(side));
            uv[(k * 4 + ci) * 2 + 1] = (@as(f32, @floatFromInt(k / side)) + 0.5) / @as(f32, @floatFromInt(side));
        }
        const b: u32 = @intCast(k * 4);
        idx[k * 6 ..][0..6].* = .{ b, b + 1, b + 2, b + 1, b + 3, b + 2 };
        for (0..3) |a| atlas[k * 3 + a] = @intFromFloat(@round(std.math.clamp(0.5 + SH0 * s.dc[g * 3 + a], 0, 1) * 255));
    }
    const image = try png.encodeRgb(gpa, atlas, @intCast(side), @intCast(side));
    defer gpa.free(image);

    var bin: std.ArrayList(u8) = .empty;
    defer bin.deinit(gpa);
    var views: [4][2]usize = undefined;
    const blobs = [_][]const u8{ std.mem.sliceAsBytes(pos), std.mem.sliceAsBytes(uv), std.mem.sliceAsBytes(idx), image };
    for (blobs, 0..) |b, i| {
        try pad4(&bin, gpa, 0);
        views[i] = .{ bin.items.len, b.len };
        try bin.appendSlice(gpa, b);
    }
    try pad4(&bin, gpa, 0);

    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.print(gpa, "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"mlx-serve idol\"}},\"extensionsUsed\":[\"KHR_materials_unlit\"]," ++
        "\"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],\"nodes\":[{{\"mesh\":0}}]," ++
        "\"meshes\":[{{\"primitives\":[{{\"mode\":4,\"attributes\":{{\"POSITION\":0,\"TEXCOORD_0\":1}},\"indices\":2,\"material\":0}}]}}]," ++
        "\"materials\":[{{\"pbrMetallicRoughness\":{{\"baseColorTexture\":{{\"index\":0}},\"metallicFactor\":0,\"roughnessFactor\":1}}," ++
        "\"doubleSided\":true,\"extensions\":{{\"KHR_materials_unlit\":{{}}}}}}]," ++
        "\"textures\":[{{\"source\":0,\"sampler\":0}}],\"images\":[{{\"bufferView\":3,\"mimeType\":\"image/png\"}}]," ++
        "\"samplers\":[{{\"magFilter\":9728,\"minFilter\":9728,\"wrapS\":33071,\"wrapT\":33071}}]," ++
        "\"buffers\":[{{\"byteLength\":{d}}}],\"bufferViews\":[", .{bin.items.len});
    for (views, 0..) |v, i| try json.print(gpa, "{s}{{\"buffer\":0,\"byteOffset\":{d},\"byteLength\":{d}}}", .{ if (i == 0) "" else ",", v[0], v[1] });
    try json.print(gpa, "],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":{d},\"type\":\"VEC3\",\"min\":[{d},{d},{d}],\"max\":[{d},{d},{d}]}}," ++
        "{{\"bufferView\":1,\"componentType\":5126,\"count\":{d},\"type\":\"VEC2\"}}," ++
        "{{\"bufferView\":2,\"componentType\":5125,\"count\":{d},\"type\":\"SCALAR\"}}]}}", .{ n * 4, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], n * 4, n * 6 });
    try pad4(&json, gpa, ' ');
    return container(gpa, json.items, bin.items);
}

fn container(gpa: std.mem.Allocator, json: []const u8, bin: []const u8) ![]u8 {
    const total: u32 = @intCast(12 + 8 + json.len + 8 + bin.len);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, total);
    const w32 = struct {
        fn put(l: *std.ArrayList(u8), v: u32) void {
            l.appendSliceAssumeCapacity(std.mem.asBytes(&std.mem.nativeToLittle(u32, v)));
        }
    }.put;
    w32(&out, 0x46546C67);
    w32(&out, 2);
    w32(&out, total);
    w32(&out, @intCast(json.len));
    w32(&out, 0x4E4F534A);
    out.appendSliceAssumeCapacity(json);
    w32(&out, @intCast(bin.len));
    w32(&out, 0x004E4942);
    out.appendSliceAssumeCapacity(bin);
    return out.toOwnedSlice(gpa);
}

// ── tests ──

const testing = std.testing;

fn tiny(gpa: std.mem.Allocator) !Splats {
    var s: Splats = .{ .n = 2, .xyz = try gpa.alloc(f32, 6), .dc = try gpa.alloc(f32, 6), .scale = try gpa.alloc(f32, 6), .rot = try gpa.alloc(f32, 8), .opacity = try gpa.alloc(f32, 2) };
    s.xyz[0..6].* = .{ 0.1, 0.2, 0.3, -0.4, 0.0, 0.5 };
    s.dc[0..6].* = .{ 0.5, -0.5, 0.0, 1.0, 1.0, 1.0 };
    s.scale[0..6].* = .{ 0.01, 0.02, 0.03, 0.01, 0.01, 0.01 };
    s.rot[0..8].* = .{ 1, 0, 0, 0, 0, 1, 0, 0 };
    s.opacity[0..2].* = .{ 0.5, 0.9 };
    return s;
}

test "idol splat: the ply is a 3DGS file turned as the reference turns it, scale and opacity in their raw form" {
    var s = try tiny(testing.allocator);
    defer s.deinit(testing.allocator);
    const ply = try writePly(testing.allocator, &s);
    defer testing.allocator.free(ply);
    const end = std.mem.indexOf(u8, ply, "end_header\n").? + 11;
    try testing.expect(std.mem.startsWith(u8, ply, "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"));
    try testing.expectEqual(@as(usize, end + 2 * 17 * 4), ply.len);
    const row: [*]align(1) const f32 = @ptrCast(ply[end..].ptr);
    try testing.expectApproxEqAbs(@as(f32, 0.1), row[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.3), row[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), row[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), row[9], 1e-6); // logit(0.5)
    try testing.expectApproxEqAbs(@log(@as(f32, 0.01)), row[10], 1e-5);
    // identity rotation turned 90 degrees about x
    try testing.expectApproxEqAbs(@sqrt(@as(f32, 0.5)), row[13], 1e-6);
    try testing.expectApproxEqAbs(@sqrt(@as(f32, 0.5)), row[14], 1e-6);
}

test "idol splat: the glb is a well-formed container with the splat extension and every accessor in range" {
    var s = try tiny(testing.allocator);
    defer s.deinit(testing.allocator);
    const glb = try writeGlb(testing.allocator, &s);
    defer testing.allocator.free(glb);
    try testing.expectEqualStrings("glTF", glb[0..4]);
    try testing.expectEqual(@as(u32, @intCast(glb.len)), std.mem.readInt(u32, glb[8..12], .little));
    const jlen = std.mem.readInt(u32, glb[12..16], .little);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, glb[20 .. 20 + jlen], .{});
    defer parsed.deinit();
    const prim = parsed.value.object.get("meshes").?.array.items[0].object.get("primitives").?.array.items[0].object;
    try testing.expectEqual(@as(i64, 0), prim.get("mode").?.integer);
    try testing.expect(prim.get("extensions").?.object.get("KHR_gaussian_splatting") != null);
    const buf_len = parsed.value.object.get("buffers").?.array.items[0].object.get("byteLength").?.integer;
    for (parsed.value.object.get("bufferViews").?.array.items) |v| {
        try testing.expect(v.object.get("byteOffset").?.integer + v.object.get("byteLength").?.integer <= buf_len);
        try testing.expectEqual(@as(i64, 0), @mod(v.object.get("byteOffset").?.integer, 4));
    }
}

test "idol splat: the glb turns the model's up (+z) to glTF's +y, and a rotation with it" {
    var s = try tiny(testing.allocator);
    defer s.deinit(testing.allocator);
    const glb = try writeGlb(testing.allocator, &s);
    defer testing.allocator.free(glb);
    const jlen = std.mem.readInt(u32, glb[12..16], .little);
    const bin = glb[20 + jlen + 8 ..];
    const pos: [*]align(1) const f32 = @ptrCast(bin.ptr);
    // (0.1, 0.2, 0.3): z is up, so y' = z = 0.3 and z' = -y = -0.2
    try testing.expectApproxEqAbs(@as(f32, 0.1), pos[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.3), pos[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.2), pos[2], 1e-6);
    // The turn alone as a unit quaternion (x, y, z, w) = (-sqrt(.5), 0, 0, sqrt(.5)): a rotation of -90 degrees about x.
    const q = turnRot(.gltf, &.{ 1, 0, 0, 0 });
    try testing.expectApproxEqAbs(@sqrt(@as(f32, 0.5)), q[0], 1e-6);
    try testing.expectApproxEqAbs(-@sqrt(@as(f32, 0.5)), q[1], 1e-6);
}

test "idol splat: the disc glb is a triangle mesh with one atlas texel per kept Gaussian and nothing faint" {
    var s = try tiny(testing.allocator);
    defer s.deinit(testing.allocator);
    s.opacity[1] = 0.01; // below the cutoff: left out
    const glb = try writeDiscGlb(testing.allocator, &s);
    defer testing.allocator.free(glb);
    try testing.expectEqual(@as(u32, @intCast(glb.len)), std.mem.readInt(u32, glb[8..12], .little));
    const jlen = std.mem.readInt(u32, glb[12..16], .little);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, glb[20 .. 20 + jlen], .{});
    defer parsed.deinit();
    const prim = parsed.value.object.get("meshes").?.array.items[0].object.get("primitives").?.array.items[0].object;
    try testing.expectEqual(@as(i64, 4), prim.get("mode").?.integer);
    const acc = parsed.value.object.get("accessors").?.array.items;
    try testing.expectEqual(@as(i64, 4), acc[0].object.get("count").?.integer); // one kept Gaussian, four corners
    try testing.expectEqual(@as(i64, 6), acc[2].object.get("count").?.integer);
    try testing.expect(parsed.value.object.get("images").?.array.items.len == 1);
}
