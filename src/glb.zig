//! Minimal glTF 2.0 binary (.glb) writer. Serializes a `marching_cubes.Mesh`
//! (positions + normals + u32 indices) into a single self-contained GLB blob:
//! one buffer, three bufferViews, three accessors, one mesh/node/scene. No
//! material — the spec's default applies. Coordinates are written as-is (no axis
//! transform); CCW winding is the glTF front face, matching the mesh.
//!
//! The writer is intentionally append-only/structured so a later texture phase
//! can add TEXCOORD_0 + a material + embedded image bufferViews without
//! reshuffling the existing layout.

const std = @import("std");
const marching_cubes = @import("marching_cubes.zig");

const MAGIC: u32 = 0x46546C67; // "glTF"
const VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E4942; // "BIN\0"

const COMPONENT_FLOAT: u32 = 5126;
const COMPONENT_UINT: u32 = 5125;
const TARGET_ARRAY: u32 = 34962; // ARRAY_BUFFER
const TARGET_ELEMENT: u32 = 34963; // ELEMENT_ARRAY_BUFFER

/// Serialize `mesh` to GLB bytes. Caller owns the returned slice.
pub fn writeGlb(alloc: std.mem.Allocator, mesh: *const marching_cubes.Mesh) ![]u8 {
    const nverts: u32 = @intCast(mesh.vertices.len / 3);
    const nidx: u32 = @intCast(mesh.indices.len);

    const pos_bytes: u32 = @intCast(mesh.vertices.len * @sizeOf(f32));
    const nrm_bytes: u32 = @intCast(mesh.normals.len * @sizeOf(f32));
    const idx_bytes: u32 = @intCast(mesh.indices.len * @sizeOf(u32));

    // f32/u32 payloads keep every offset a multiple of 4, so no inter-view pad.
    const pos_off: u32 = 0;
    const nrm_off: u32 = pos_off + pos_bytes;
    const idx_off: u32 = nrm_off + nrm_bytes;
    const bin_len: u32 = idx_off + idx_bytes;

    // POSITION min/max over the actual data (required by the accessor).
    var pmin = [3]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var pmax = [3]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    var vi: usize = 0;
    while (vi < mesh.vertices.len) : (vi += 3) {
        for (0..3) |c| {
            const val = mesh.vertices[vi + c];
            pmin[c] = @min(pmin[c], val);
            pmax[c] = @max(pmax[c], val);
        }
    }
    if (nverts == 0) {
        pmin = .{ 0, 0, 0 };
        pmax = .{ 0, 0, 0 };
    }

    // --- JSON chunk ---
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(alloc);
    try json.print(alloc,
        \\{{"asset":{{"version":"2.0","generator":"mlx-serve"}},
    , .{});
    try json.print(alloc,
        \\"buffers":[{{"byteLength":{d}}}],
    , .{bin_len});
    try json.print(alloc,
        \\"bufferViews":[{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":{d}}},{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":{d}}},{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":{d}}}],
    , .{ pos_off, pos_bytes, TARGET_ARRAY, nrm_off, nrm_bytes, TARGET_ARRAY, idx_off, idx_bytes, TARGET_ELEMENT });
    try json.print(alloc,
        \\"accessors":[{{"bufferView":0,"componentType":{d},"count":{d},"type":"VEC3","min":[{d},{d},{d}],"max":[{d},{d},{d}]}},{{"bufferView":1,"componentType":{d},"count":{d},"type":"VEC3"}},{{"bufferView":2,"componentType":{d},"count":{d},"type":"SCALAR"}}],
    , .{
        COMPONENT_FLOAT, nverts,         pmin[0], pmin[1], pmin[2], pmax[0], pmax[1], pmax[2],
        COMPONENT_FLOAT, nverts,         COMPONENT_UINT,
        nidx,
    });
    try json.print(alloc,
        \\"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2,"mode":4}}]}}],
    , .{});
    try json.print(alloc,
        \\"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0}}
    , .{});

    const json_pad: u32 = pad4(@intCast(json.items.len));

    // --- Assemble ---
    const total: u32 = 12 + 8 + json_pad + 8 + bin_len;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.ensureTotalCapacity(alloc, total);

    // Header.
    try appendU32(&out, alloc, MAGIC);
    try appendU32(&out, alloc, VERSION);
    try appendU32(&out, alloc, total);

    // JSON chunk (space-padded).
    try appendU32(&out, alloc, json_pad);
    try appendU32(&out, alloc, CHUNK_JSON);
    try out.appendSlice(alloc, json.items);
    try out.appendNTimes(alloc, ' ', json_pad - @as(u32, @intCast(json.items.len)));

    // BIN chunk (zero-padded; already 4-aligned).
    try appendU32(&out, alloc, bin_len);
    try appendU32(&out, alloc, CHUNK_BIN);
    try out.appendSlice(alloc, std.mem.sliceAsBytes(mesh.vertices));
    try out.appendSlice(alloc, std.mem.sliceAsBytes(mesh.normals));
    try out.appendSlice(alloc, std.mem.sliceAsBytes(mesh.indices));

    return out.toOwnedSlice(alloc);
}

/// Textured mesh for the paint stage: seam-duplicated (post-unwrap) vertices
/// with per-vertex UVs and pre-encoded PNG textures. `mr_png` follows the glTF
/// metallicRoughness channel packing (G = roughness, B = metallic) — note the
/// generated paint image is R = metallic, G = roughness; the CALLER remaps.
pub const TexturedMesh = struct {
    positions: []const f32, // N*3
    normals: []const f32, // N*3
    uvs: []const f32, // N*2, glTF convention (v grows downward)
    indices: []const u32, // M*3
    albedo_png: []const u8,
    mr_png: ?[]const u8,
};

/// Serialize a textured mesh to GLB: POSITION/NORMAL/TEXCOORD_0 + u32 indices
/// + a PBR material with embedded PNG images. Same container discipline as
/// `writeGlb` (single BIN buffer, 4-aligned views, POSITION min/max).
pub fn writeGlbTextured(alloc: std.mem.Allocator, tm: *const TexturedMesh) ![]u8 {
    const nverts: u32 = @intCast(tm.positions.len / 3);
    const nidx: u32 = @intCast(tm.indices.len);
    std.debug.assert(tm.normals.len == tm.positions.len);
    std.debug.assert(tm.uvs.len / 2 == nverts);

    const pos_bytes: u32 = @intCast(tm.positions.len * @sizeOf(f32));
    const nrm_bytes: u32 = @intCast(tm.normals.len * @sizeOf(f32));
    const uv_bytes: u32 = @intCast(tm.uvs.len * @sizeOf(f32));
    const idx_bytes: u32 = @intCast(tm.indices.len * @sizeOf(u32));
    const alb_bytes: u32 = @intCast(tm.albedo_png.len);
    const mr_bytes: u32 = if (tm.mr_png) |p| @intCast(p.len) else 0;

    // f32/u32 payloads keep the first four views 4-aligned; PNG views get
    // explicit alignment padding.
    const pos_off: u32 = 0;
    const nrm_off: u32 = pos_off + pos_bytes;
    const uv_off: u32 = nrm_off + nrm_bytes;
    const idx_off: u32 = uv_off + uv_bytes;
    const alb_off: u32 = idx_off + idx_bytes;
    const mr_off: u32 = pad4(alb_off + alb_bytes);
    const bin_len: u32 = if (tm.mr_png != null) pad4(mr_off + mr_bytes) else pad4(alb_off + alb_bytes);

    var pmin = [3]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var pmax = [3]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    var vi: usize = 0;
    while (vi < tm.positions.len) : (vi += 3) {
        for (0..3) |c| {
            const val = tm.positions[vi + c];
            pmin[c] = @min(pmin[c], val);
            pmax[c] = @max(pmax[c], val);
        }
    }
    if (nverts == 0) {
        pmin = .{ 0, 0, 0 };
        pmax = .{ 0, 0, 0 };
    }

    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(alloc);
    try json.print(alloc,
        \\{{"asset":{{"version":"2.0","generator":"mlx-serve"}},
    , .{});
    try json.print(alloc,
        \\"buffers":[{{"byteLength":{d}}}],
    , .{bin_len});
    // Views: 0 pos, 1 nrm, 2 uv, 3 idx, 4 albedo png [, 5 mr png].
    try json.print(alloc,
        \\"bufferViews":[{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":{d}}},{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":{d}}},{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":{d}}},{{"buffer":0,"byteOffset":{d},"byteLength":{d},"target":{d}}},{{"buffer":0,"byteOffset":{d},"byteLength":{d}}}
    , .{
        pos_off, pos_bytes, TARGET_ARRAY,
        nrm_off, nrm_bytes, TARGET_ARRAY,
        uv_off,  uv_bytes,  TARGET_ARRAY,
        idx_off, idx_bytes, TARGET_ELEMENT,
        alb_off, alb_bytes,
    });
    if (tm.mr_png != null) {
        try json.print(alloc,
            \\,{{"buffer":0,"byteOffset":{d},"byteLength":{d}}}
        , .{ mr_off, mr_bytes });
    }
    try json.print(alloc,
        \\],"accessors":[{{"bufferView":0,"componentType":{d},"count":{d},"type":"VEC3","min":[{d},{d},{d}],"max":[{d},{d},{d}]}},{{"bufferView":1,"componentType":{d},"count":{d},"type":"VEC3"}},{{"bufferView":2,"componentType":{d},"count":{d},"type":"VEC2"}},{{"bufferView":3,"componentType":{d},"count":{d},"type":"SCALAR"}}],
    , .{
        COMPONENT_FLOAT, nverts, pmin[0],         pmin[1],
        pmin[2],         pmax[0], pmax[1],         pmax[2],
        COMPONENT_FLOAT, nverts, COMPONENT_FLOAT, nverts,
        COMPONENT_UINT,  nidx,
    });
    if (tm.mr_png != null) {
        try json.print(alloc,
            \\"images":[{{"bufferView":4,"mimeType":"image/png"}},{{"bufferView":5,"mimeType":"image/png"}}],"samplers":[{{}}],"textures":[{{"sampler":0,"source":0}},{{"sampler":0,"source":1}}],
        , .{});
        try json.print(alloc,
            \\"materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}},"metallicRoughnessTexture":{{"index":1}},"metallicFactor":1.0,"roughnessFactor":1.0}}}}],
        , .{});
    } else {
        try json.print(alloc,
            \\"images":[{{"bufferView":4,"mimeType":"image/png"}}],"samplers":[{{}}],"textures":[{{"sampler":0,"source":0}}],
        , .{});
        try json.print(alloc,
            \\"materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}},"metallicFactor":0.0,"roughnessFactor":1.0}}}}],
        , .{});
    }
    try json.print(alloc,
        \\"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":0,"mode":4}}]}}],
    , .{});
    try json.print(alloc,
        \\"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0}}
    , .{});

    const json_pad: u32 = pad4(@intCast(json.items.len));
    const total: u32 = 12 + 8 + json_pad + 8 + bin_len;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.ensureTotalCapacity(alloc, total);

    try appendU32(&out, alloc, MAGIC);
    try appendU32(&out, alloc, VERSION);
    try appendU32(&out, alloc, total);

    try appendU32(&out, alloc, json_pad);
    try appendU32(&out, alloc, CHUNK_JSON);
    try out.appendSlice(alloc, json.items);
    try out.appendNTimes(alloc, ' ', json_pad - @as(u32, @intCast(json.items.len)));

    try appendU32(&out, alloc, bin_len);
    try appendU32(&out, alloc, CHUNK_BIN);
    try out.appendSlice(alloc, std.mem.sliceAsBytes(tm.positions));
    try out.appendSlice(alloc, std.mem.sliceAsBytes(tm.normals));
    try out.appendSlice(alloc, std.mem.sliceAsBytes(tm.uvs));
    try out.appendSlice(alloc, std.mem.sliceAsBytes(tm.indices));
    try out.appendSlice(alloc, tm.albedo_png);
    if (tm.mr_png) |p| {
        try out.appendNTimes(alloc, 0, mr_off - (alb_off + alb_bytes));
        try out.appendSlice(alloc, p);
    }
    try out.appendNTimes(alloc, 0, bin_len - @as(u32, @intCast(out.items.len - (12 + 8 + json_pad + 8))));

    return out.toOwnedSlice(alloc);
}

inline fn pad4(n: u32) u32 {
    return (n + 3) & ~@as(u32, 3);
}

fn appendU32(out: *std.ArrayList(u8), alloc: std.mem.Allocator, v: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try out.appendSlice(alloc, &buf);
}

// ---------------------------------------------------------------------------
// Tests (hermetic).
// ---------------------------------------------------------------------------

const testing = std.testing;

fn readU32(bytes: []const u8, off: usize) u32 {
    return std.mem.readInt(u32, bytes[off..][0..4], .little);
}

test "glb: tetrahedron round-trips through a valid GLB" {
    const a = testing.allocator;
    // A hand-built tetrahedron.
    const verts = [_]f32{
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const norms = [_]f32{
        0, 0, -1,
        0, -1, 0,
        -1, 0, 0,
        0.577, 0.577, 0.577,
    };
    const idx = [_]u32{ 0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3 };
    var mesh = marching_cubes.Mesh{
        .vertices = try a.dupe(f32, &verts),
        .normals = try a.dupe(f32, &norms),
        .indices = try a.dupe(u32, &idx),
    };
    defer mesh.deinit(a);

    const glb = try writeGlb(a, &mesh);
    defer a.free(glb);

    // Header.
    try testing.expectEqual(@as(u32, 0x46546C67), readU32(glb, 0));
    try testing.expectEqual(@as(u32, 2), readU32(glb, 4));
    try testing.expectEqual(@as(u32, @intCast(glb.len)), readU32(glb, 8));

    // JSON chunk header.
    const json_len = readU32(glb, 12);
    try testing.expectEqual(@as(u32, 0x4E4F534A), readU32(glb, 16));
    try testing.expectEqual(@as(u32, 0), json_len % 4); // 4-byte aligned
    const json_bytes = glb[20 .. 20 + json_len];

    // BIN chunk header.
    const bin_hdr = 20 + json_len;
    const bin_len = readU32(glb, bin_hdr);
    try testing.expectEqual(@as(u32, 0x004E4942), readU32(glb, bin_hdr + 4));
    const bin_data_off = bin_hdr + 8;
    try testing.expectEqual(@as(usize, glb.len), bin_data_off + bin_len);

    // Parse JSON and validate the accessor structure.
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json_bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const accessors = root.get("accessors").?.array.items;
    try testing.expectEqual(@as(usize, 3), accessors.len);
    try testing.expectEqual(@as(i64, 5126), accessors[0].object.get("componentType").?.integer);
    try testing.expectEqual(@as(i64, 4), accessors[0].object.get("count").?.integer);
    try testing.expectEqual(@as(i64, 5126), accessors[1].object.get("componentType").?.integer);
    try testing.expectEqual(@as(i64, 5125), accessors[2].object.get("componentType").?.integer);
    try testing.expectEqual(@as(i64, @intCast(idx.len)), accessors[2].object.get("count").?.integer);

    // POSITION min/max correctness.
    const mn = accessors[0].object.get("min").?.array.items;
    const mx = accessors[0].object.get("max").?.array.items;
    try testing.expectApproxEqAbs(@as(f64, 0.0), jsonNum(mn[0]), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), jsonNum(mn[1]), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), jsonNum(mn[2]), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.0), jsonNum(mx[0]), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.0), jsonNum(mx[1]), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.0), jsonNum(mx[2]), 1e-6);

    // Round-trip: positions extracted via bufferView 0 must byte-match the input.
    const bv0 = root.get("bufferViews").?.array.items[0].object;
    const bv0_off: usize = @intCast(bv0.get("byteOffset").?.integer);
    const bv0_len: usize = @intCast(bv0.get("byteLength").?.integer);
    const pos_slice = glb[bin_data_off + bv0_off ..][0..bv0_len];
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mesh.vertices), pos_slice);
}

fn jsonNum(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

test "glb: sphere mesh from marching cubes serializes with in-bounds indices" {
    const a = testing.allocator;
    const np: usize = 40;
    const grid = try a.alloc(f32, np * np * np);
    defer a.free(grid);
    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    for (0..np) |i| {
        for (0..np) |j| {
            for (0..np) |k| {
                const x = @as(f32, @floatFromInt(i)) * step - 1.0;
                const y = @as(f32, @floatFromInt(j)) * step - 1.0;
                const z = @as(f32, @floatFromInt(k)) * step - 1.0;
                grid[(i * np + j) * np + k] = 0.4 - @sqrt(x * x + y * y + z * z);
            }
        }
    }
    var mesh = try marching_cubes.extract(a, grid, .{ np, np, np }, 0.0, .{ step, step, step }, .{ -1, -1, -1 });
    defer mesh.deinit(a);

    const glb = try writeGlb(a, &mesh);
    defer a.free(glb);

    // The header's total-length claim matches the actual blob.
    try testing.expectEqual(@as(u32, @intCast(glb.len)), readU32(glb, 8));

    // Every index is a valid vertex.
    const nverts: u32 = @intCast(mesh.vertices.len / 3);
    for (mesh.indices) |ix| try testing.expect(ix < nverts);
}

test "glb: textured quad round-trips with TEXCOORD_0, PBR material and embedded PNGs" {
    const a = testing.allocator;
    const png = @import("png.zig");
    // 2x2 albedo (red-ish) + 2x2 metallicRoughness (G=roughness, B=metallic).
    const albedo_rgb = [_]u8{ 255, 0, 0, 250, 5, 5, 245, 10, 10, 240, 15, 15 };
    const mr_rgb = [_]u8{ 0, 128, 64, 0, 128, 64, 0, 128, 64, 0, 128, 64 };
    const albedo_png = try png.encodeRgb(a, &albedo_rgb, 2, 2);
    defer a.free(albedo_png);
    const mr_png = try png.encodeRgb(a, &mr_rgb, 2, 2);
    defer a.free(mr_png);

    const tm = TexturedMesh{
        .positions = &[_]f32{ 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0 },
        .normals = &[_]f32{ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1 },
        .uvs = &[_]f32{ 0, 0, 1, 0, 1, 1, 0, 1 },
        .indices = &[_]u32{ 0, 1, 2, 0, 2, 3 },
        .albedo_png = albedo_png,
        .mr_png = mr_png,
    };
    const glb = try writeGlbTextured(a, &tm);
    defer a.free(glb);

    // Container validity.
    try testing.expectEqual(@as(u32, 0x46546C67), readU32(glb, 0));
    try testing.expectEqual(@as(u32, @intCast(glb.len)), readU32(glb, 8));
    const json_len = readU32(glb, 12);
    const json_bytes = glb[20 .. 20 + json_len];
    const bin_data_off = 20 + json_len + 8;

    const parsed = try std.json.parseFromSlice(std.json.Value, a, json_bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    // TEXCOORD_0 accessor: VEC2 float, count 4, wired into the primitive.
    const prim = root.get("meshes").?.array.items[0].object.get("primitives").?.array.items[0].object;
    const attrs = prim.get("attributes").?.object;
    const uv_acc_idx: usize = @intCast(attrs.get("TEXCOORD_0").?.integer);
    const accessors = root.get("accessors").?.array.items;
    const uv_acc = accessors[uv_acc_idx].object;
    try testing.expectEqualStrings("VEC2", uv_acc.get("type").?.string);
    try testing.expectEqual(@as(i64, 5126), uv_acc.get("componentType").?.integer);
    try testing.expectEqual(@as(i64, 4), uv_acc.get("count").?.integer);
    try testing.expectEqual(@as(i64, 0), prim.get("material").?.integer);

    // Material → texture → image → bufferView chain, both slots.
    const mat = root.get("materials").?.array.items[0].object.get("pbrMetallicRoughness").?.object;
    const base_tex: usize = @intCast(mat.get("baseColorTexture").?.object.get("index").?.integer);
    const mr_tex: usize = @intCast(mat.get("metallicRoughnessTexture").?.object.get("index").?.integer);
    const textures = root.get("textures").?.array.items;
    const images = root.get("images").?.array.items;
    const views = root.get("bufferViews").?.array.items;
    const base_img: usize = @intCast(textures[base_tex].object.get("source").?.integer);
    const mr_img: usize = @intCast(textures[mr_tex].object.get("source").?.integer);
    try testing.expectEqualStrings("image/png", images[base_img].object.get("mimeType").?.string);

    // PNG bytes round-trip exactly, and every bufferView offset is 4-aligned.
    const base_view = views[@intCast(images[base_img].object.get("bufferView").?.integer)].object;
    const base_off: usize = @intCast(base_view.get("byteOffset").?.integer);
    const base_len: usize = @intCast(base_view.get("byteLength").?.integer);
    try testing.expectEqualSlices(u8, albedo_png, glb[bin_data_off + base_off ..][0..base_len]);
    const mr_view = views[@intCast(images[mr_img].object.get("bufferView").?.integer)].object;
    const mr_off: usize = @intCast(mr_view.get("byteOffset").?.integer);
    const mr_len: usize = @intCast(mr_view.get("byteLength").?.integer);
    try testing.expectEqualSlices(u8, mr_png, glb[bin_data_off + mr_off ..][0..mr_len]);
    for (views) |v| {
        const off = v.object.get("byteOffset") orelse continue;
        try testing.expectEqual(@as(i64, 0), @mod(off.integer, 4));
    }

    // UV bytes round-trip.
    const uv_view = views[@intCast(uv_acc.get("bufferView").?.integer)].object;
    const uv_off: usize = @intCast(uv_view.get("byteOffset").?.integer);
    const uv_len: usize = @intCast(uv_view.get("byteLength").?.integer);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tm.uvs), glb[bin_data_off + uv_off ..][0..uv_len]);
}

test "glb: albedo-only textured mesh omits the metallicRoughness texture" {
    const a = testing.allocator;
    const png = @import("png.zig");
    const albedo_rgb = [_]u8{ 10, 200, 30, 10, 200, 30, 10, 200, 30, 10, 200, 30 };
    const albedo_png = try png.encodeRgb(a, &albedo_rgb, 2, 2);
    defer a.free(albedo_png);

    const tm = TexturedMesh{
        .positions = &[_]f32{ 0, 0, 0, 1, 0, 0, 0, 1, 0 },
        .normals = &[_]f32{ 0, 0, 1, 0, 0, 1, 0, 0, 1 },
        .uvs = &[_]f32{ 0, 0, 1, 0, 0, 1 },
        .indices = &[_]u32{ 0, 1, 2 },
        .albedo_png = albedo_png,
        .mr_png = null,
    };
    const glb = try writeGlbTextured(a, &tm);
    defer a.free(glb);

    const json_len = readU32(glb, 12);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, glb[20 .. 20 + json_len], .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expectEqual(@as(usize, 1), root.get("images").?.array.items.len);
    const mat = root.get("materials").?.array.items[0].object.get("pbrMetallicRoughness").?.object;
    try testing.expect(mat.get("metallicRoughnessTexture") == null);
    // Without an MR texture the factors pin a dielectric look.
    try testing.expectApproxEqAbs(@as(f64, 0.0), jsonNum(mat.get("metallicFactor").?), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.0), jsonNum(mat.get("roughnessFactor").?), 1e-9);
}

test "glb: write textured swift fixture" {
    // Writes the TEXTURED fixture GLBMeshLoaderTests loads through the Swift
    // GLB reader (checkerboard albedo on a unit quad). Run via:
    //   GLB_TEXTURED_FIXTURE_OUT=app/Tests/MLXCoreTests/Fixtures/hy3d_textured_quad.glb \
    //     zig build test -Doptimize=ReleaseFast -Dtest-filter="textured swift fixture"
    const out_path = std.mem.span(std.c.getenv("GLB_TEXTURED_FIXTURE_OUT") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const png = @import("png.zig");
    var rgb: [8 * 8 * 3]u8 = undefined;
    for (0..8) |y| for (0..8) |x| {
        const on = (x / 2 + y / 2) % 2 == 0;
        rgb[(y * 8 + x) * 3 + 0] = if (on) 230 else 30;
        rgb[(y * 8 + x) * 3 + 1] = if (on) 60 else 180;
        rgb[(y * 8 + x) * 3 + 2] = if (on) 60 else 230;
    };
    const albedo_png = try png.encodeRgb(a, &rgb, 8, 8);
    defer a.free(albedo_png);
    var mr: [8 * 8 * 3]u8 = undefined;
    for (0..8 * 8) |i| {
        mr[i * 3 + 0] = 0;
        mr[i * 3 + 1] = 200; // roughness
        mr[i * 3 + 2] = 40; // metallic
    }
    const mr_png = try png.encodeRgb(a, &mr, 8, 8);
    defer a.free(mr_png);

    const tm = TexturedMesh{
        .positions = &[_]f32{ -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0 },
        .normals = &[_]f32{ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1 },
        .uvs = &[_]f32{ 0, 1, 1, 1, 1, 0, 0, 0 },
        .indices = &[_]u32{ 0, 1, 2, 0, 2, 3 },
        .albedo_png = albedo_png,
        .mr_png = mr_png,
    };
    const glb = try writeGlbTextured(a, &tm);
    defer a.free(glb);

    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.createFileAbsolute(io, out_path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, glb);
}

test "glb: write swift fixture" {
    const out_path = std.mem.span(std.c.getenv("GLB_FIXTURE_OUT") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const np: usize = 64;
    const grid = try a.alloc(f32, np * np * np);
    defer a.free(grid);
    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    for (0..np) |i| {
        for (0..np) |j| {
            for (0..np) |k| {
                const x = @as(f32, @floatFromInt(i)) * step - 1.0;
                const y = @as(f32, @floatFromInt(j)) * step - 1.0;
                const z = @as(f32, @floatFromInt(k)) * step - 1.0;
                grid[(i * np + j) * np + k] = 0.6 - @sqrt(x * x + y * y + z * z);
            }
        }
    }
    var mesh = try marching_cubes.extract(a, grid, .{ np, np, np }, 0.0, .{ step, step, step }, .{ -1, -1, -1 });
    defer mesh.deinit(a);

    const glb = try writeGlb(a, &mesh);
    defer a.free(glb);

    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.createFileAbsolute(io, out_path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, glb);
}
