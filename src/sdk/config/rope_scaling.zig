//! A `rope_scaling` block (or transformers v5's `rope_parameters`) as mlx-lm's `initialize_rope` reads it, and the
//! rope it builds as host values: the `mlx_fast_rope` denominators, the scale on positions, the factor on the rotated
//! dims, and dynamic NTK's base. MLX-free.

const std = @import("std");
const mrope = @import("../vision/mrope.zig");

/// Llama 3.x `rope_type: "llama3"` (HF `_compute_llama3_parameters`), read through f32 as the engine stores it.
pub const Llama3 = struct {
    factor: f64,
    low_freq_factor: f64 = 1.0,
    high_freq_factor: f64 = 4.0,
    original_max_position_embeddings: f64 = 8192.0,
};

/// mlx-lm's `YarnRoPE` arguments, its defaults.
pub const Yarn = struct {
    factor: f64,
    original_max_position_embeddings: u32 = 4096,
    beta_fast: f64 = 32,
    beta_slow: f64 = 1,
    mscale: f64 = 1,
    mscale_all_dim: f64 = 0,
};

pub const Scaling = union(enum) {
    none,
    /// Positions read as `p / factor`.
    linear: f64,
    /// NTK: past `max_position_embeddings` the base grows with the sequence (mlx-lm's `DynamicNTKScalingRoPE`).
    dynamic: f64,
    llama3: Llama3,
    yarn: Yarn,
    /// A block this engine does not build (`longrope`, a type outside mlx-lm's, a malformed one): served unscaled.
    unmodeled,
};

fn field(o: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn float(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    const f: f64 = switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return null,
    };
    return if (std.math.isFinite(f)) f else null;
}

/// The block's type as mlx-lm reads it (`type`, else `rope_type`, else "default"); null without a block.
pub fn typeName(cfg: std.json.ObjectMap) ?[]const u8 {
    const rs = field(cfg, "rope_scaling") orelse field(cfg, "rope_parameters") orelse return null;
    if (rs != .object) return null;
    const t = field(rs.object, "type") orelse field(rs.object, "rope_type") orelse return "default";
    return if (t == .string) t.string else null;
}

/// The scaling `cfg` declares; `.unmodeled` for a block this engine does not build (a type outside mlx-lm's, a
/// malformed or out-of-range one), which the engine always served unscaled.
pub fn parse(cfg: std.json.ObjectMap) Scaling {
    return parseBlock(cfg) orelse .unmodeled;
}

fn parseBlock(cfg: std.json.ObjectMap) ?Scaling {
    const rs_val = field(cfg, "rope_scaling") orelse field(cfg, "rope_parameters") orelse return .none;
    if (rs_val != .object) return null;
    const rs = rs_val.object;
    const t = typeName(cfg) orelse return null;
    const eql = std.mem.eql;
    if (eql(u8, t, "default") or eql(u8, t, "mrope")) return .none;
    const factor = float(field(rs, "factor")) orelse return null;
    if (factor <= 0) return null;
    if (eql(u8, t, "linear")) return .{ .linear = factor };
    if (eql(u8, t, "dynamic")) return .{ .dynamic = factor };
    if (eql(u8, t, "llama3")) {
        var r = Llama3{ .factor = factor };
        inline for (.{ "low_freq_factor", "high_freq_factor", "original_max_position_embeddings" }) |n|
            if (field(rs, n)) |v| {
                @field(r, n) = float(v) orelse return null;
            };
        if (r.high_freq_factor <= r.low_freq_factor or r.original_max_position_embeddings <= 0) return null;
        inline for (.{ "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings" }) |n|
            @field(r, n) = @as(f32, @floatCast(@field(r, n)));
        return .{ .llama3 = r };
    }
    if (eql(u8, t, "yarn") or eql(u8, t, "deepseek_yarn") or eql(u8, t, "telechat3-yarn")) {
        var y = Yarn{ .factor = factor };
        if (field(rs, "original_max_position_embeddings")) |v| {
            const om = float(v) orelse return null;
            if (om < 1 or om > std.math.maxInt(u32)) return null;
            y.original_max_position_embeddings = @intFromFloat(om);
        }
        inline for (.{ "beta_fast", "beta_slow", "mscale", "mscale_all_dim" }) |n|
            if (field(rs, n)) |v| {
                @field(y, n) = float(v) orelse return null;
            };
        return .{ .yarn = y };
    }
    return null;
}

/// The `mlx_fast_rope` denominators of the `out.len` rotary pairs over `dims` rotated dims, when the scaling reshapes
/// the spectrum (llama3, yarn); false when the base's own spectrum stands.
pub fn freqs(s: Scaling, out: []f64, dims: u32, base: f64) bool {
    switch (s) {
        .llama3 => |l3| {
            computeLlama3Freqs(out, dims, base, l3);
            return true;
        },
        .yarn => |y| {
            yarnSpec(y, dims, base).invFreq(out);
            for (out) |*v| v.* = 1.0 / v.*;
            return true;
        },
        else => return false,
    }
}

/// YaRN's spectrum over `dims` rotated dims (the M-RoPE tables read its inverse frequencies).
pub fn yarnSpec(y: Yarn, dims: u32, base: f64) mrope.Yarn {
    return .{ .theta = base, .factor = y.factor, .orig_max = y.original_max_position_embeddings, .rotary_dim = dims, .beta_fast = y.beta_fast, .beta_slow = y.beta_slow };
}

/// The scale `mlx_fast_rope` applies to positions.
pub fn positionScale(s: Scaling) f32 {
    return switch (s) {
        .linear => |f| @floatCast(1.0 / f),
        else => 1.0,
    };
}

fn yarnGetMscale(scale: f64, m: f64) f64 {
    if (scale <= 1) return 1.0;
    return 0.1 * m * @log(scale) + 1.0;
}

/// The factor YaRN puts on the rotated dims of q and k (1 for every other scaling).
pub fn mscale(s: Scaling) f64 {
    return switch (s) {
        .yarn => |y| yarnGetMscale(y.factor, y.mscale) / yarnGetMscale(y.factor, y.mscale_all_dim),
        else => 1.0,
    };
}

/// Dynamic NTK as a rope reads it per forward.
pub const Ntk = struct {
    factor: f64,
    max_pos: u32,

    pub fn base(n: Ntk, b: f64, dims: u32, seq: u64) f64 {
        return ntkBase(b, dims, n.factor, n.max_pos, seq);
    }
};

/// The scaling's dynamic NTK; null for every other.
pub fn ntk(s: Scaling, max_pos: u32) ?Ntk {
    return switch (s) {
        .dynamic => |f| .{ .factor = f, .max_pos = max_pos },
        else => null,
    };
}

/// Dynamic NTK's base for a forward reaching `seq` positions over `dims` rotated dims (`base` itself up to
/// `max_pos`).
pub fn ntkBase(base: f64, dims: u32, factor: f64, max_pos: u32, seq: u64) f64 {
    const mp: f64 = @floatFromInt(max_pos);
    const len: f64 = @floatFromInt(@max(seq, max_pos));
    const d: f64 = @floatFromInt(dims);
    return base * std.math.pow(f64, factor * len / mp - (factor - 1), d / (d - 2));
}

/// The SSD cache namespace of K/V the scaling changes (a scaled rope stores other keys than the unscaled one the
/// legacy load served); null for none.
pub fn layoutNamespace(s: Scaling) ?[]const u8 {
    return switch (s) {
        .none, .unmodeled => null,
        .llama3 => "llama3-rope-v1",
        .linear, .dynamic, .yarn => "scaled-rope-v1",
    };
}

// ── tests ──

const testing = std.testing;

fn parseText(json: []const u8) !Scaling {
    const doc = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer doc.deinit();
    return parse(doc.value.object);
}

/// HF `_compute_llama3_parameters` as mlx_fast_rope denominators (angle =
/// position / freqs[i]): low frequencies slow down by `factor`, high ones keep
/// their rate, the band between is smoothed. `out.len` = head_dim / 2.
/// `l3` is a `Llama3Rope` or any struct with its fields (f32 or f64).
pub fn computeLlama3Freqs(out: []f64, head_dim: u32, base: f64, l3: anytype) void {
    const factor: f64 = l3.factor;
    const low: f64 = l3.low_freq_factor;
    const high: f64 = l3.high_freq_factor;
    const window: f64 = l3.original_max_position_embeddings;
    for (out, 0..) |*o, i| {
        const period = std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim)));
        const wavelen = 2.0 * std.math.pi * period;
        o.* = if (wavelen < window / high) period else if (wavelen > window / low) period * factor else blk: {
            const smooth = (window / wavelen - low) / (high - low);
            break :blk period / ((1.0 - smooth) / factor + smooth);
        };
    }
}

test "rope scaling: each type mlx-lm builds, its keys and defaults; any other block is unmodeled" {
    try testing.expectEqual(Scaling.none, try parseText("{}"));
    try testing.expectEqual(Scaling.none, try parseText("{\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":1e6}}"));
    try testing.expectEqual(Scaling{ .linear = 2 }, try parseText("{\"rope_scaling\":{\"type\":\"linear\",\"factor\":2.0}}"));
    try testing.expectEqual(Scaling{ .dynamic = 4 }, try parseText("{\"rope_scaling\":{\"rope_type\":\"dynamic\",\"factor\":4}}"));
    // mlx-lm reads `type` before `rope_type`.
    try testing.expectEqual(Scaling{ .linear = 2 }, try parseText("{\"rope_scaling\":{\"type\":\"linear\",\"rope_type\":\"dynamic\",\"factor\":2}}"));
    const y = (try parseText("{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4,\"original_max_position_embeddings\":32768}}")).yarn;
    try testing.expectEqual(Yarn{ .factor = 4, .original_max_position_embeddings = 32768 }, y);
    for ([_][]const u8{
        "{\"rope_scaling\":{\"type\":\"longrope\",\"factor\":2,\"short_factor\":[1],\"long_factor\":[1]}}",
        "{\"rope_scaling\":{\"type\":\"linear\"}}",
        "{\"rope_scaling\":{\"type\":\"su-ntk\",\"factor\":2}}",
        "{\"rope_scaling\":\"linear\"}",
    }) |json| try testing.expectEqual(Scaling.unmodeled, try parseText(json));
    try testing.expectEqualStrings("scaled-rope-v1", layoutNamespace(.{ .linear = 2 }).?);
}

/// mlx-lm's `YarnRoPE.__init__`, transcribed: the denominators and the rotated-dim factor.
fn hostYarn(y: Yarn, dims: u32, base: f64, out: []f64) f64 {
    const d: f64 = @floatFromInt(dims);
    const om: f64 = @floatFromInt(y.original_max_position_embeddings);
    const corr = struct {
        fn f(r: f64, dd: f64, o: f64, b: f64) f64 {
            return (dd * @log(o / (r * 2 * std.math.pi))) / (2 * @log(b));
        }
    }.f;
    const low = @max(@floor(corr(y.beta_fast, d, om, base)), 0);
    var high = @min(@ceil(corr(y.beta_slow, d, om, base)), d - 1);
    if (low == high) high += 0.001;
    for (out, 0..) |*o, i| {
        const extra = std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / d);
        const inter = y.factor * extra;
        const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / (high - low), 0, 1);
        const mask = 1 - ramp;
        o.* = (inter * extra) / (inter * mask + extra * (1 - mask));
    }
    const gm = struct {
        fn f(s: f64, m: f64) f64 {
            return if (s <= 1) 1 else 0.1 * m * @log(s) + 1;
        }
    }.f;
    return gm(y.factor, y.mscale) / gm(y.factor, y.mscale_all_dim);
}

test "rope scaling: YaRN's denominators and rotated-dim factor equal mlx-lm's YarnRoPE" {
    for ([_]Yarn{ .{ .factor = 4, .original_max_position_embeddings = 32768 }, .{ .factor = 8, .beta_fast = 16, .mscale = 0.707, .mscale_all_dim = 0.707 } }) |y| {
        var got: [64]f64 = undefined;
        var want: [64]f64 = undefined;
        try testing.expect(freqs(.{ .yarn = y }, &got, 128, 1e6));
        const ms = hostYarn(y, 128, 1e6, &want);
        for (got, want) |g, w| try testing.expectApproxEqRel(w, g, 1e-12);
        try testing.expectApproxEqRel(ms, mscale(.{ .yarn = y }), 1e-12);
    }
}

test "rope scaling: linear scales positions, dynamic NTK grows the base only past the window, as mlx-lm's do" {
    try testing.expectEqual(@as(f32, 0.25), positionScale(.{ .linear = 4 }));
    try testing.expectEqual(@as(f32, 1), positionScale(.{ .dynamic = 4 }));
    // DynamicNTKScalingRoPE: seq_len = max(L + offset, max_pos); base * ((f * seq / max_pos) - (f - 1)) ** (d / (d - 2)).
    try testing.expectEqual(@as(f64, 10000), ntkBase(10000, 128, 2, 4096, 100));
    const want = 10000 * std.math.pow(f64, 2.0 * 8192.0 / 4096.0 - 1.0, 128.0 / 126.0);
    try testing.expectApproxEqRel(want, ntkBase(10000, 128, 2, 4096, 8192), 1e-12);
}
