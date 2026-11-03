//! An image job's request, validated the same way for every image backend: the body's fields (prompt, size, seed,
//! steps, mode, strength, conditioning rebalance, guidance) checked against what the loaded pack can do (`Spec`),
//! and the edit/variation geometry helpers. Uploads arrive as `inputs`: `image` first, then each `ref_images`
//! entry, base64-decoded (a WebP already re-encoded as PNG). MLX-free.

const std = @import("std");
const abi = @import("../../abi/media.zig");

const Verdict = abi.Verdict;

/// What the loaded pack can do. Messages are static text naming the pack's limit.
pub const Spec = struct {
    img2img: bool = false,
    edit: bool = false,
    guidance: bool = false,
    default_guidance: f32 = 1.0,
    /// References one edit takes, the primary included.
    edit_ref_cap: u32 = 4,
    /// Tapped text-encoder layers `cond_weights` must cover; 0 = the field is refused.
    cond_weights: u32 = 0,
    /// Qwen-Image only: `transparent` and `ref_resolution`.
    transparent: bool = false,
    ref_resolution: bool = false,
    too_many_refs: []const u8 = "too many reference images ('ref_images' takes at most 3 beside 'image')",
    cond_weights_count: []const u8 = "'cond_weights' needs exactly one value per tapped layer for this model",
};

pub const Mode = enum { generate, variation, edit };

pub const max_cond_weights = 16;

pub const Job = struct {
    prompt: []const u8,
    /// The requested size (1024x1024 when omitted) and whether the client chose it.
    req_w: u32 = 1024,
    req_h: u32 = 1024,
    size_given: bool = false,
    seed: u64 = 42,
    /// 0 = the backend's default.
    steps: u32 = 0,
    mode: Mode = .generate,
    strength: f32 = 0.6,
    /// The edited or varied source, then the extra references (edit only).
    image: ?[]const u8 = null,
    refs: []const []const u8 = &.{},
    cond_gain: f32 = 1.0,
    cond_weights_buf: [max_cond_weights]f32 = undefined,
    cond_weights_len: usize = 0,
    guidance_scale: f32 = 1.0,
    negative_prompt: []const u8 = "",
    ref_resolution: u32 = 1024,
    transparent: bool = false,

    pub fn condWeights(j: *const Job) ?[]const f32 {
        return if (j.cond_weights_len == 0) null else j.cond_weights_buf[0..j.cond_weights_len];
    }
};

pub const Parsed = union(enum) { ok: Job, bad: Verdict };

fn bad(comptime why: []const u8) Parsed {
    return .{ .bad = Verdict.refuse(.unsupported, why) };
}

fn badText(why: []const u8) Parsed {
    return .{ .bad = .{ .code = .unsupported, .why = .of(why) } };
}

/// A request object's typed reads, the same semantics as the host's job body.
pub const Fields = struct {
    obj: std.json.ObjectMap,

    pub fn get(f: Fields, key: []const u8) ?std.json.Value {
        return f.obj.get(key);
    }
    pub fn str(f: Fields, key: []const u8) ?[]const u8 {
        const v = f.obj.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }
    /// True only for the literal `true`.
    pub fn flag(f: Fields, key: []const u8) bool {
        const v = f.obj.get(key) orelse return false;
        return v == .bool and v.bool;
    }
    pub fn float(f: Fields, key: []const u8) ?f64 {
        const v = f.obj.get(key) orelse return null;
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |x| x,
            else => null,
        };
    }
    /// A non-negative integer; anything else reads as absent.
    pub fn uint(f: Fields, key: []const u8) ?u64 {
        const v = f.obj.get(key) orelse return null;
        return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
    }
};

/// `inputs` borrow from the request and must outlive the job, as do the body's strings.
pub fn parse(body: Fields, inputs: []const []const u8, spec: Spec) Parsed {
    const prompt = body.str("prompt") orelse return bad("missing 'prompt'");
    if (prompt.len == 0) return bad("empty 'prompt'");
    var j: Job = .{ .prompt = prompt, .guidance_scale = spec.default_guidance };
    j.transparent = body.flag("transparent");
    if (j.transparent and !spec.transparent) return bad("'transparent' requires a Qwen-Image model");
    if (body.str("size")) |size| if (parseSize(size)) |wh| {
        j.req_w = wh.w;
        j.req_h = wh.h;
        j.size_given = true;
    };
    j.seed = body.uint("seed") orelse 42;
    if (body.get("steps")) |_| {
        const n = body.uint("steps") orelse return bad("'steps' must be in [0,100] (0 = the model's default)");
        if (n > 100) return bad("'steps' must be in [0,100] (0 = the model's default)");
        j.steps = @intCast(n);
    }
    if (body.str("mode")) |m| {
        if (std.mem.eql(u8, m, "edit")) {
            j.mode = .edit;
        } else if (!std.mem.eql(u8, m, "variation")) return bad("'mode' must be \"edit\" or \"variation\"");
    }
    const edit = j.mode == .edit;
    const has_image = body.str("image") != null;
    var rest = inputs;
    if (has_image) {
        if (edit and !spec.edit) return bad("instruction editing (mode:\"edit\") requires a FLUX.2, Mage-Flow-Edit, or Qwen-Image-2.1 model with its vision tower (re-convert with the current converter)");
        if (!edit and !spec.img2img) return bad("image-to-image (mode:\"variation\") isn't available for this model — either its VAE encoder failed to load, or this backend has no variation path");
        if (body.float("strength")) |sv| {
            if (!(sv > 0.0 and sv <= 1.0)) return bad("'strength' must be in (0,1]");
            j.strength = @floatCast(sv);
        }
        if (rest.len == 0) return bad("invalid base64 in 'image'");
        j.image = rest[0];
        rest = rest[1..];
        if (!edit) j.mode = .variation;
    } else if (edit) return bad("mode:\"edit\" needs an 'image' to edit");

    if (body.get("ref_images")) |v| if (v != .null) {
        if (!edit) return bad("'ref_images' requires mode:\"edit\"");
        if (v != .array) return bad("invalid 'ref_images' (must be a JSON array of base64 strings)");
        for (v.array.items) |item| if (item != .string) return bad("invalid 'ref_images' (must be a JSON array of base64 strings)");
        if (1 + v.array.items.len > spec.edit_ref_cap) return badText(spec.too_many_refs);
        if (rest.len != v.array.items.len) return bad("invalid base64 in 'ref_images'");
        j.refs = rest;
    };

    if (body.float("cond_gain")) |g| {
        if (!(g >= 0.0 and g <= 10.0)) return bad("'cond_gain' must be in [0,10]");
        j.cond_gain = @floatCast(g);
    }
    if (body.get("cond_weights")) |v| if (v != .null) {
        const n = floatList(v, &j.cond_weights_buf) orelse return bad("invalid 'cond_weights' (numbers, comma/space separated, or a JSON array)");
        if (n != spec.cond_weights) return badText(spec.cond_weights_count);
        j.cond_weights_len = n;
    };
    if (body.float("guidance_scale")) |g| {
        if (!(g >= 0.0 and g <= 20.0)) return bad("'guidance_scale' must be in [0,20]");
        j.guidance_scale = @floatCast(g);
    }
    if (body.str("negative_prompt")) |np| j.negative_prompt = np;
    if (j.guidance_scale != 1.0 and !spec.guidance) return bad("'guidance_scale' requires a FLUX.2, Qwen-Image or MOSS-Image model");
    if (body.get("ref_resolution")) |_| {
        const rr = body.uint("ref_resolution") orelse return bad("'ref_resolution' must be in [256,1024]");
        if (rr < 256 or rr > 1024) return bad("'ref_resolution' must be in [256,1024]");
        if (!spec.ref_resolution) return bad("'ref_resolution' requires a Qwen-Image model");
        j.ref_resolution = @intCast(rr);
    }
    return .{ .ok = j };
}

/// A JSON number array, or a string of numbers separated by commas or spaces; null past `buf` or on a bad token.
fn floatList(v: std.json.Value, buf: []f32) ?usize {
    switch (v) {
        .array => |arr| {
            if (arr.items.len == 0 or arr.items.len > buf.len) return null;
            for (arr.items, 0..) |x, i| {
                buf[i] = switch (x) {
                    .integer => |n| @floatFromInt(n),
                    .float => |f| @floatCast(f),
                    else => return null,
                };
                if (!std.math.isFinite(buf[i])) return null;
            }
            return arr.items.len;
        },
        .string => |text| {
            var n: usize = 0;
            var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
            while (it.next()) |tok| {
                if (n >= buf.len) return null;
                buf[n] = std.fmt.parseFloat(f32, tok) catch return null;
                if (!std.math.isFinite(buf[n])) return null;
                n += 1;
            }
            return if (n == 0) null else n;
        },
        else => return null,
    }
}

// ── geometry ──

pub const Dims = struct { w: u32, h: u32 };

/// "WxH" (or "WXH"), both positive; null otherwise.
pub fn parseSize(size: []const u8) ?Dims {
    const xi = std.mem.indexOfScalar(u8, size, 'x') orelse std.mem.indexOfScalar(u8, size, 'X') orelse return null;
    const w = std.fmt.parseInt(u32, size[0..xi], 10) catch return null;
    const h = std.fmt.parseInt(u32, size[xi + 1 ..], 10) catch return null;
    if (w == 0 or h == 0) return null;
    return .{ .w = w, .h = h };
}

/// A multiple of 32 in [256, 1536] (FLUX.2 klein's crop granularity); 0 = the 1024 default.
pub fn clamp32(v: u32) u32 {
    if (v == 0) return 1024;
    const rounded = ((v + 31) / 32) * 32;
    return std.math.clamp(rounded, 256, 1536);
}

/// A multiple of 16 in [256, 2048] (one latent token per 16x16 tile).
pub fn clamp16(v: u32) u32 {
    const rounded = ((v + 15) / 16) * 16;
    return std.math.clamp(rounded, 256, 2048);
}

/// An edit reference's own grid: its aspect at most ~1 MP, each side floored to /32, never upscaled, at least 32.
pub fn fitRefDims(w: u32, h: u32) Dims {
    const cap: f64 = 1024.0 * 1024.0;
    const area: f64 = @as(f64, @floatFromInt(w)) * @as(f64, @floatFromInt(h));
    const scale: f64 = @min(1.0, std.math.sqrt(cap / @max(area, 1.0)));
    const sw: u32 = @intFromFloat(@as(f64, @floatFromInt(w)) * scale);
    const sh: u32 = @intFromFloat(@as(f64, @floatFromInt(h)) * scale);
    return .{ .w = @max(32, (sw / 32) * 32), .h = @max(32, (sh / 32) * 32) };
}

/// `src`'s aspect at the pixel budget `budget_w x budget_h`, each side rounded to /16.
pub fn fitAspect(src_w: u32, src_h: u32, budget_w: u32, budget_h: u32) Dims {
    if (src_w == 0 or src_h == 0) return .{ .w = budget_w, .h = budget_h };
    const budget: f64 = @as(f64, @floatFromInt(budget_w)) * @as(f64, @floatFromInt(budget_h));
    const aspect: f64 = @as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(src_h));
    const h = std.math.sqrt(budget / aspect);
    const w = h * aspect;
    return .{
        .w = @max(16, (@as(u32, @intFromFloat(@round(w))) / 16) * 16),
        .h = @max(16, (@as(u32, @intFromFloat(@round(h))) / 16) * 16),
    };
}

/// (w, h) scaled down until neither side passes `cap`, aspect kept, floored to /16.
pub fn fitWithinCap(w: u32, h: u32, cap: u32) Dims {
    const longest = @max(w, h);
    if (longest <= cap or longest == 0 or cap == 0) return .{ .w = w, .h = h };
    const scale = @as(f64, @floatFromInt(cap)) / @as(f64, @floatFromInt(longest));
    const sw = @round(@as(f64, @floatFromInt(w)) * scale);
    const sh = @round(@as(f64, @floatFromInt(h)) * scale);
    return .{
        .w = @max(16, (@as(u32, @intFromFloat(sw)) / 16) * 16),
        .h = @max(16, (@as(u32, @intFromFloat(sh)) / 16) * 16),
    };
}

/// An edit's output: the primary reference's aspect at the budget, then into the backend's cap; both steps keep a
/// phone photo from being squared off by a per-side clamp.
pub fn resolveEditTargetSize(src_w: u32, src_h: u32, budget_w: u32, budget_h: u32, cap: u32) Dims {
    const fit = fitAspect(src_w, src_h, budget_w, budget_h);
    return fitWithinCap(fit.w, fit.h, cap);
}

/// The first denoise step of an img2img run: `steps - round(steps * strength)`, at least one step left to run.
/// A canvas whose transient (`need`) is past the GPU's live headroom (`room`), refused by name with both numbers.
pub fn canvasRefusal(req: *const abi.Request, w: u32, h: u32, need: u64, room: u64) ?Verdict {
    if (need <= room) return null;
    const gb = 1024.0 * 1024.0 * 1024.0;
    return req.refuseFmt(.unsupported, "this canvas needs more GPU memory than is free: shrink 'size'", "{d}x{d} needs ~{d:.1} GB of GPU memory but {d:.1} GB is free: shrink 'size' or unload another model", .{
        w, h, @as(f64, @floatFromInt(need)) / gb, @as(f64, @floatFromInt(room)) / gb,
    });
}

pub fn img2imgStartStep(steps: u32, strength: f32) u32 {
    const fsteps: f32 = @floatFromInt(steps);
    const run: u32 = @intFromFloat(@round(fsteps * std.math.clamp(strength, 0.0, 1.0)));
    const start = steps -| run;
    return @min(start, steps -| 1);
}

// ── tests ──

const testing = std.testing;

fn parsed(json: []const u8, inputs: []const []const u8, spec: Spec) !Parsed {
    var b = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer b.deinit();
    const p = parse(.{ .obj = b.value.object }, inputs, spec);
    // Strings borrow from the body: only the verdict and scalars are read after it goes.
    return switch (p) {
        .ok => |j| .{ .ok = .{ .prompt = "", .req_w = j.req_w, .req_h = j.req_h, .size_given = j.size_given, .seed = j.seed, .steps = j.steps, .mode = j.mode, .strength = j.strength, .image = j.image, .refs = j.refs, .cond_gain = j.cond_gain, .cond_weights_buf = j.cond_weights_buf, .cond_weights_len = j.cond_weights_len, .guidance_scale = j.guidance_scale, .ref_resolution = j.ref_resolution, .transparent = j.transparent } },
        .bad => p,
    };
}

fn expectBad(json: []const u8, inputs: []const []const u8, spec: Spec, why: []const u8) !void {
    const p = try parsed(json, inputs, spec);
    try testing.expect(p == .bad and p.bad.code == .unsupported);
    try testing.expectEqualStrings(why, p.bad.why.slice());
}

test "image request: defaults, size, seed and steps" {
    const j = (try parsed("{\"prompt\":\"a cat\"}", &.{}, .{})).ok;
    try testing.expect(j.req_w == 1024 and j.req_h == 1024 and !j.size_given and j.seed == 42 and j.steps == 0 and j.mode == .generate);
    const k = (try parsed("{\"prompt\":\"a\",\"size\":\"512x768\",\"seed\":7,\"steps\":12}", &.{}, .{ .guidance = true, .default_guidance = 3 })).ok;
    try testing.expect(k.req_w == 512 and k.req_h == 768 and k.size_given and k.seed == 7 and k.steps == 12 and k.guidance_scale == 3);
    try expectBad("{}", &.{}, .{}, "missing 'prompt'");
    try expectBad("{\"prompt\":\"\"}", &.{}, .{}, "empty 'prompt'");
    try expectBad("{\"prompt\":\"a\",\"steps\":101}", &.{}, .{}, "'steps' must be in [0,100] (0 = the model's default)");
    try expectBad("{\"prompt\":\"a\",\"mode\":\"x\"}", &.{}, .{}, "'mode' must be \"edit\" or \"variation\"");
}

test "image request: an upload is a variation unless the mode is edit, each gated by the pack" {
    const img: []const u8 = "png";
    const v = (try parsed("{\"prompt\":\"a\",\"image\":\"x\",\"strength\":0.3}", &.{img}, .{ .img2img = true })).ok;
    try testing.expect(v.mode == .variation and v.strength == 0.3 and v.image != null);
    try expectBad("{\"prompt\":\"a\",\"image\":\"x\"}", &.{img}, .{}, "image-to-image (mode:\"variation\") isn't available for this model — either its VAE encoder failed to load, or this backend has no variation path");
    try expectBad("{\"prompt\":\"a\",\"image\":\"x\",\"strength\":0}", &.{img}, .{ .img2img = true }, "'strength' must be in (0,1]");
    try expectBad("{\"prompt\":\"a\",\"mode\":\"edit\"}", &.{}, .{ .edit = true }, "mode:\"edit\" needs an 'image' to edit");
    const e = (try parsed("{\"prompt\":\"a\",\"mode\":\"edit\",\"image\":\"x\",\"ref_images\":[\"y\",\"z\"]}", &.{ img, img, img }, .{ .edit = true })).ok;
    try testing.expect(e.mode == .edit and e.refs.len == 2);
    try expectBad("{\"prompt\":\"a\",\"mode\":\"edit\",\"image\":\"x\",\"ref_images\":[\"y\",\"z\",\"w\",\"v\"]}", &.{ img, img, img, img, img }, .{ .edit = true }, "too many reference images ('ref_images' takes at most 3 beside 'image')");
    try expectBad("{\"prompt\":\"a\",\"image\":\"x\",\"ref_images\":[\"y\"]}", &.{ img, img }, .{ .img2img = true }, "'ref_images' requires mode:\"edit\"");
}

test "image request: rebalance, guidance and the Qwen-only fields follow the spec" {
    const j = (try parsed("{\"prompt\":\"a\",\"cond_weights\":\"1 0.5, 2\",\"cond_gain\":2}", &.{}, .{ .cond_weights = 3 })).ok;
    try testing.expectEqualSlices(f32, &.{ 1, 0.5, 2 }, j.condWeights().?);
    try testing.expectEqual(@as(f32, 2), j.cond_gain);
    const k = (try parsed("{\"prompt\":\"a\",\"cond_weights\":[1,2,3]}", &.{}, .{ .cond_weights = 3 })).ok;
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, k.condWeights().?);
    try expectBad("{\"prompt\":\"a\",\"cond_weights\":[1,2]}", &.{}, .{ .cond_weights = 3 }, "'cond_weights' needs exactly one value per tapped layer for this model");
    try expectBad("{\"prompt\":\"a\",\"guidance_scale\":3}", &.{}, .{}, "'guidance_scale' requires a FLUX.2, Qwen-Image or MOSS-Image model");
    try expectBad("{\"prompt\":\"a\",\"transparent\":true}", &.{}, .{}, "'transparent' requires a Qwen-Image model");
    try expectBad("{\"prompt\":\"a\",\"ref_resolution\":512}", &.{}, .{}, "'ref_resolution' requires a Qwen-Image model");
    try testing.expectEqual(@as(u32, 512), (try parsed("{\"prompt\":\"a\",\"ref_resolution\":512}", &.{}, .{ .ref_resolution = true })).ok.ref_resolution);
}

test "image request: the geometry helpers keep the aspect and the grid" {
    try testing.expectEqual(Dims{ .w = 1024, .h = 1024 }, Dims{ .w = clamp32(0), .h = clamp32(1000) });
    try testing.expectEqual(@as(u32, 1536), clamp32(4000));
    try testing.expectEqual(@as(u32, 256), clamp16(10));
    try testing.expectEqual(Dims{ .w = 1152, .h = 864 }, fitRefDims(4032, 3024));
    const t = resolveEditTargetSize(4032, 3024, 4032, 3024, 2048);
    try testing.expect(t.w == 2048 and t.h == 1536);
    try testing.expectEqual(@as(u32, 2), img2imgStartStep(4, 0.6));
    try testing.expectEqual(@as(u32, 3), img2imgStartStep(4, 0.0));
}

test "image request: a canvas past the headroom is refused with its size and both numbers" {
    var buf: [256]u8 = undefined;
    var len: usize = 0;
    const req: abi.Request = .{ .body = .of(""), .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    try testing.expect(canvasRefusal(&req, 64, 64, 1 << 30, 1 << 30) == null);
    const v = canvasRefusal(&req, 2048, 1024, 3 << 30, 1 << 30).?;
    try testing.expect(v.code == .unsupported);
    try testing.expectEqualStrings("2048x1024 needs ~3.0 GB of GPU memory but 1.0 GB is free: shrink 'size' or unload another model", buf[0..len]);
}
