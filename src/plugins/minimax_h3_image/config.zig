//! MiniMax-H3-Image's claims, caps and bills, MLX-free: a `minimax_h3_image` config.json (the converter's
//! `--partition image`), its geometry and sampling grid, and the staged plan's peak (text encoder, then DiT, then VAE).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const config_fields = @import("../../sdk/media/config_fields.zig");
const geometry_mod = @import("../minimax_h3/geometry.zig");
const h3_bill = @import("../minimax_h3/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "minimax_h3_image",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/MiniMax-H3-Image-MLX-Serve-4bit"},
};

pub const model_types = [_][]const u8{"minimax_h3_image"};

pub const claims = sdk.mediaClaimsByType(&model_types);

pub const required_file = "transformer.safetensors";

/// Reference images one edit takes (the release's own cap).
pub const edit_ref_cap = 16;
/// The canvas the model was trained up to: 6272x2688 is its widest listed size.
pub const max_dim = 6272;
/// Canvases are generated on the 32-pixel grid: the VAE's 16x then the DiT's 2x2 patch.
pub const CANVAS_MULTIPLE: u32 = 32;
/// A smaller request is generated on a canvas of this area at its aspect and resized back (`min_generation_area`).
pub const DEFAULT_MIN_AREA: u32 = 2048 * 2048;
pub const MIN_AREA_RANGE = [2]u32{ 1024, 4096 * 4096 };

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .edit = true, .max_dim = max_dim, .edit_ref_cap = edit_ref_cap };
}

pub const Geometry = struct {
    h3: geometry_mod.Geometry,
    /// The PDD student's sampling grid, `pdd_sigmas` (descending to 0, one output head per step).
    sigmas: [65]f64 = undefined,
    n_sigmas: u32 = 0,

    pub fn grid(self: *const Geometry) []const f64 {
        return self.sigmas[0..self.n_sigmas];
    }
};

/// The pack's geometry: the H3 DiT and text encoder, the 2D image VAE decoder (`image_vae`), and the sampling grid.
pub fn geometry(a: std.mem.Allocator, config_json: []const u8) !Geometry {
    var g: Geometry = .{ .h3 = try h3_bill.geometry(a, config_json) };
    var parsed = std.json.parseFromSlice(std.json.Value, a, config_json, .{}) catch return error.BadPackConfig;
    defer parsed.deinit();
    const root = parsed.value.object;
    g.h3.vae = geometry_mod.DecCfg.image;
    try config_fields.fill(geometry_mod.DecCfg, .{ .{ "layers", "layers" }, .{ "heads", "heads" }, .{ "head_dim", "head_dim" } }, try config_fields.section(root, "image_vae"), &g.h3.vae);
    const sv = root.get("pdd_sigmas") orelse return error.BadPackConfig;
    if (sv != .array or sv.array.items.len < 2 or sv.array.items.len > g.sigmas.len) return error.BadPackConfig;
    for (sv.array.items, 0..) |v, i| g.sigmas[i] = switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => return error.BadPackConfig,
    };
    g.n_sigmas = @intCast(sv.array.items.len);
    return g;
}

/// Where a request is generated: the requested size, scaled up to `min_area` at its aspect when smaller, each side
/// rounded to the 32-pixel grid (diffusers `get_image_canvas_size`).
pub fn canvasFor(w: u32, h: u32, min_area: u32) struct { w: u32, h: u32 } {
    const area: f64 = @as(f64, @floatFromInt(w)) * @as(f64, @floatFromInt(h));
    const scale = @max(1.0, @sqrt(@as(f64, @floatFromInt(min_area)) / area));
    const m: f64 = @floatFromInt(CANVAS_MULTIPLE);
    return .{ .w = roundTo(@as(f64, @floatFromInt(w)) * scale, m), .h = roundTo(@as(f64, @floatFromInt(h)) * scale, m) };
}

fn roundTo(v: f64, m: f64) u32 {
    const q = v / m;
    var r = @round(q);
    if (@abs(q - @trunc(q)) == 0.5 and @mod(r, 2) != 0) r -= 1; // half to even, as Python's round()
    return @max(CANVAS_MULTIPLE, @as(u32, @intFromFloat(r * m)));
}

/// A reference's size for one consumer (diffusers `bound_short_edge`): kept when its short edge is within
/// [lo, hi], else scaled so the short edge lands on the nearer bound, both sides rounded to 32.
pub fn boundShortEdge(w: u32, h: u32, lo: u32, hi: u32) struct { w: u32, h: u32 } {
    const short = @min(w, h);
    if (short >= lo and short <= hi) return .{ .w = w, .h = h };
    const target: f64 = @floatFromInt(if (short < lo) lo else hi);
    const k = target / @as(f64, @floatFromInt(short));
    const m: f64 = @floatFromInt(CANVAS_MULTIPLE);
    return .{ .w = roundTo(@as(f64, @floatFromInt(w)) * k, m), .h = roundTo(@as(f64, @floatFromInt(h)) * k, m) };
}

/// Sequence rows a request puts through the DiT: the target and each reference on the 32-pixel grid. Prompt and
/// vision rows ride the margin.
pub fn requestRows(w: u32, h: u32, refs: []const [2]u32) u64 {
    var rows: u64 = @as(u64, w / CANVAS_MULTIPLE) * (h / CANVAS_MULTIPLE);
    for (refs) |r| rows += @as(u64, r[0] / CANVAS_MULTIPLE) * (r[1] / CANVAS_MULTIPLE);
    return rows;
}

/// The staged plan's peak: the text encoder is freed before the DiT loads and the DiT before the VAE.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    const dit = h3_bill.ditResidentBytes(h3_bill.ditFileBytes(dir, m), true);
    return h3_bill.peakBytes(dir.fileBytes("text_encoder.safetensors"), dit, dir.fileBytes("image_vae.safetensors"), 0);
}

/// The step refuses a canvas that does not fit against live memory, by name (`h3_bill.canvasBill`).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    _ = body;
    return 0;
}

// ── tests ──

const testing = std.testing;

test "minimax_h3_image config: claimed by its own model_type, never by the video packs'" {
    const mem: @import("../../sdk/testing/fixture.zig").MemDir = .{ .files = &.{} };
    const dir = mem.packDir();
    try testing.expectEqual(abi.Claim.by_type, claims(.of("minimax_h3_image"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.of("minimax_h3"), &dir));
}

test "minimax_h3_image config: the canvas grows to the minimum area at the request's aspect, on the 32 grid" {
    // diffusers get_image_canvas_size(1024, 1024, 2048*2048, 32) = (2048, 2048).
    try testing.expectEqual(@as(u32, 2048), canvasFor(1024, 1024, DEFAULT_MIN_AREA).w);
    // A 16 MP request is generated as asked.
    const big = canvasFor(4736, 3552, DEFAULT_MIN_AREA);
    try testing.expectEqual(@as(u32, 4736), big.w);
    try testing.expectEqual(@as(u32, 3552), big.h);
    // min_generation_area 1024 turns the upscale off; a side off the grid still rounds to it (1000 -> 992).
    const off = canvasFor(1000, 768, 1024);
    try testing.expectEqual(@as(u32, 992), off.w);
    try testing.expectEqual(@as(u32, 768), off.h);
    // 1200x800 at 4 MP: scale sqrt(4194304/960000) = 2.0903 -> 2508.3 / 1672.2 -> 2496 x 1664.
    const r = canvasFor(1200, 800, DEFAULT_MIN_AREA);
    try testing.expectEqual(@as(u32, 2496), r.w);
    try testing.expectEqual(@as(u32, 1664), r.h);
}

test "minimax_h3_image config: a reference keeps its size inside the bounds, else lands its short edge on one" {
    try testing.expectEqual(@as(u32, 1500), boundShortEdge(1500, 2000, 1024, 2048).w);
    const small = boundShortEdge(600, 800, 1024, 2048); // x1.7067 -> 1024 x 1365.3 -> 1024 x 1376
    try testing.expectEqual(@as(u32, 1024), small.w);
    try testing.expectEqual(@as(u32, 1376), small.h);
    const vision = boundShortEdge(1500, 2000, 1024, 1024); // short edge pinned to 1024
    try testing.expectEqual(@as(u32, 1024), vision.w);
    try testing.expectEqual(@as(u32, 1376), vision.h);
}

test "minimax_h3_image config: geometry carries the image VAE and the sampling grid; a pack without one is refused" {
    const a = testing.allocator;
    const g = try geometry(a, "{\"model_type\":\"minimax_h3_image\",\"pdd_sigmas\":[1.0,0.75,0.0],\"image_vae\":{\"layers\":2}}");
    try testing.expectEqual(@as(u32, 2), g.h3.vae.layers);
    try testing.expectEqual(@as(u32, 2), g.h3.vae.rope_axes);
    try testing.expectEqual(@as(u32, 1), g.h3.vae.patch_size_t);
    try testing.expectEqualSlices(f64, &.{ 1.0, 0.75, 0.0 }, g.grid());
    try testing.expectError(error.BadPackConfig, geometry(a, "{\"model_type\":\"minimax_h3_image\"}"));
}

test "minimax_h3_image config: the staged peak bills the biggest stage" {
    const GB: u64 = 1 << 30;
    // The 4-bit pack: TE 15.8 GB, DiT 18.7 GB shedding its AdaLN share, the VAE 2.3 GB: the TE stage binds.
    const te: u64 = 15_804_791_921;
    const dit = h3_bill.ditResidentBytes(18_700_000_000, true);
    try testing.expectEqual(te, h3_bill.peakBytes(te, dit, 2_340_000_000, 0));
    try testing.expect(te < 16 * GB);
}
