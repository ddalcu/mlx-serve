//! Laya typed-decision model: a ModernBERT/mmBERT encoder plus a decision
//! head that scores one [MASK] marker per answer option. Mirrors the
//! `laya_mlx` reference (https://github.com/mizorewww/laya-mlx) op for op:
//! prompt layout, batching, masks, fp16 numerics, calibration, output JSON.
//!
//! Checkpoint layout (aac6fef/laya-multilingual-mlx): `encoder/config.json`,
//! `rl_agent_config.json`, `tokenizer/tokenizer.json` (+ `_config.json`),
//! `model.safetensors` with MLX parameter names. No top-level config.json.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const ltx = @import("ltx_video.zig");
const tokenizer_mod = @import("tokenizer.zig");

const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const none = A{ .ctx = null };

pub const QType = enum(u8) {
    choice = 0,
    score = 1,
    noul = 2,

    pub fn name(self: QType) []const u8 {
        return @tagName(self);
    }
    pub fn parse(s: []const u8) ?QType {
        return std.meta.stringToEnum(QType, s);
    }
};

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_layers: u32,
    num_heads: u32,
    head_dim: u32,
    norm_eps: f32,
    local_attention: u32,
    /// Per layer: true = full attention, false = sliding window.
    layer_global: []bool,
    rope_theta_global: f32,
    rope_theta_local: f32,
    head_layers: u32,
    max_len: u32,
    head_max_len: u32,
    /// `len(act_costs) + 1` output columns of the action head.
    n_actions: u32,
    temperature: [3]f32,
    /// "choice:3-5" style bucket -> temperature; keys owned.
    temperature_by_options: std.StringHashMap(f32),
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    mask_id: u32,
    mask_token: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Config) void {
        self.allocator.free(self.layer_global);
        var it = self.temperature_by_options.iterator();
        while (it.next()) |e| self.allocator.free(e.key_ptr.*);
        self.temperature_by_options.deinit();
        self.allocator.free(self.mask_token);
    }
};

/// A negative, fractional or out-of-range config integer is `error.InvalidLayaConfig`.
fn jsonInt(v: ?std.json.Value, default: u32) !u32 {
    const x = v orelse return default;
    return switch (x) {
        .integer => |i| std.math.cast(u32, i) orelse error.InvalidLayaConfig,
        .float => |f| if (f >= 0 and f <= std.math.maxInt(u32) and @floor(f) == f) @intFromFloat(f) else error.InvalidLayaConfig,
        .number_string => error.InvalidLayaConfig,
        else => default,
    };
}

fn badConfig(comptime fmt: []const u8, args: anytype) error{InvalidLayaConfig} {
    log.err("[laya] invalid checkpoint: " ++ fmt ++ "\n", args);
    return error.InvalidLayaConfig;
}

fn jsonF32(v: ?std.json.Value, default: f32) f32 {
    const x = v orelse return default;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => default,
    };
}

fn readWholeFile(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
}

fn readJsonFile(io: std.Io, a: std.mem.Allocator, dir: []const u8, rel: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, rel });
    defer a.free(path);
    const text = try readWholeFile(io, a, path);
    defer a.free(text);
    return std.json.parseFromSlice(std.json.Value, a, text, .{ .allocate = .alloc_always });
}

pub fn parseConfig(io: std.Io, a: std.mem.Allocator, dir: []const u8, tok: *const tokenizer_mod.Tokenizer) !Config {
    var enc = try readJsonFile(io, a, dir, "encoder/config.json");
    defer enc.deinit();
    var agent = try readJsonFile(io, a, dir, "rl_agent_config.json");
    defer agent.deinit();
    var tcfg = try readJsonFile(io, a, dir, "tokenizer/tokenizer_config.json");
    defer tcfg.deinit();
    if (enc.value != .object or agent.value != .object or tcfg.value != .object) return error.InvalidLayaConfig;
    const e = enc.value.object;
    const g = agent.value.object;

    if (e.get("model_type")) |mt| {
        if (mt != .string or !std.mem.eql(u8, mt.string, "modernbert")) return error.UnsupportedEncoder;
    }
    if (e.get("hidden_activation")) |act| {
        if (act != .string or !std.mem.eql(u8, act.string, "gelu")) return error.UnsupportedEncoder;
    }
    const hidden = try jsonInt(e.get("hidden_size"), 0);
    const heads = try jsonInt(e.get("num_attention_heads"), 0);
    const layers = try jsonInt(e.get("num_hidden_layers"), 0);
    if (hidden == 0 or heads == 0 or layers == 0 or hidden % heads != 0 or (hidden / heads) % 2 != 0) return error.InvalidLayaConfig;

    const layer_global = try a.alloc(bool, layers);
    errdefer a.free(layer_global);
    const every_n = try jsonInt(e.get("global_attn_every_n_layers"), 3);
    if (every_n == 0) return badConfig("global_attn_every_n_layers is 0", .{});
    for (layer_global, 0..) |*lg, i| lg.* = (i % every_n == 0);
    if (e.get("layer_types")) |lt| {
        if (lt != .array or lt.array.items.len != layers) return error.InvalidLayaConfig;
        for (lt.array.items, 0..) |v, i| {
            if (v != .string) return error.InvalidLayaConfig;
            if (std.mem.eql(u8, v.string, "full_attention")) {
                layer_global[i] = true;
            } else if (std.mem.eql(u8, v.string, "sliding_attention")) {
                layer_global[i] = false;
            } else return error.InvalidLayaConfig;
        }
    }
    var theta_global = jsonF32(e.get("global_rope_theta"), 160000.0);
    var theta_local = jsonF32(e.get("local_rope_theta"), 10000.0);
    if (e.get("rope_parameters")) |rp| {
        if (rp == .object) {
            if (rp.object.get("full_attention")) |fa| if (fa == .object) {
                theta_global = jsonF32(fa.object.get("rope_theta"), theta_global);
            };
            if (rp.object.get("sliding_attention")) |sa| if (sa == .object) {
                theta_local = jsonF32(sa.object.get("rope_theta"), theta_local);
            };
        }
    }

    var temperature = [3]f32{ 1.0, 1.0, 1.0 };
    if (g.get("temperature")) |t| {
        if (t != .array or t.array.items.len != 3) return error.InvalidLayaConfig;
        for (t.array.items, 0..) |v, i| temperature[i] = jsonF32(v, 1.0);
    }
    var by_opt = std.StringHashMap(f32).init(a);
    errdefer {
        var it = by_opt.iterator();
        while (it.next()) |kv| a.free(kv.key_ptr.*);
        by_opt.deinit();
    }
    if (g.get("temperature_by_options")) |tbo| {
        if (tbo == .object) {
            var it = tbo.object.iterator();
            while (it.next()) |kv| {
                const key = try a.dupe(u8, kv.key_ptr.*);
                errdefer a.free(key);
                try by_opt.put(key, jsonF32(kv.value_ptr.*, 1.0));
            }
        }
    }
    for (temperature) |t| if (!(t > 0) or !std.math.isFinite(t)) return error.InvalidLayaConfig;
    var tit = by_opt.valueIterator();
    while (tit.next()) |t| if (!(t.* > 0) or !std.math.isFinite(t.*)) return error.InvalidLayaConfig;
    for (&temperature, 0..) |*t, i| clampTemperature(@as(QType, @enumFromInt(i)).name(), t);
    var cit = by_opt.iterator();
    while (cit.next()) |kv| clampTemperature(kv.key_ptr.*, kv.value_ptr);

    const n_actions: u32 = if (g.get("act_costs")) |ac| (if (ac == .object) @as(u32, @intCast(ac.object.count())) + 1 else 1) else 1;
    const max_len = try jsonInt(g.get("max_len"), 512);
    const head_max_len = try jsonInt(g.get("head_max_len"), 192);
    const max_pos = try jsonInt(e.get("max_position_embeddings"), 8192);
    if (!(4 < head_max_len and head_max_len < max_len and max_len <= max_pos))
        return badConfig("want 4 < head_max_len ({d}) < max_len ({d}) <= max_position_embeddings ({d})", .{ head_max_len, max_len, max_pos });

    const specialId = struct {
        fn f(t: *const tokenizer_mod.Tokenizer, obj: std.json.ObjectMap, key: []const u8) !struct { id: u32, text: []const u8 } {
            const v = obj.get(key) orelse return error.TokenizerMissingSpecial;
            const text = switch (v) {
                .string => |s| s,
                .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else return error.TokenizerMissingSpecial) else return error.TokenizerMissingSpecial,
                else => return error.TokenizerMissingSpecial,
            };
            const id = t.specialTokenId(text) orelse t.vocab.get(text) orelse return error.TokenizerMissingSpecial;
            return .{ .id = id, .text = text };
        }
    }.f;
    const tc = tcfg.value.object;
    const cls = try specialId(tok, tc, "cls_token");
    const sep = try specialId(tok, tc, "sep_token");
    const pad = try specialId(tok, tc, "pad_token");
    const mask = try specialId(tok, tc, "mask_token");
    const vocab_size = try jsonInt(e.get("vocab_size"), 0);
    for ([_]u32{ cls.id, sep.id, pad.id, mask.id, @intCast(tok.definedVocabSize() -| 1) }) |id| {
        if (id >= vocab_size) return badConfig("token id {d} is outside vocab_size {d}", .{ id, vocab_size });
    }

    return .{
        .vocab_size = vocab_size,
        .hidden_size = hidden,
        .intermediate_size = try jsonInt(e.get("intermediate_size"), 0),
        .num_layers = layers,
        .num_heads = heads,
        .head_dim = hidden / heads,
        .norm_eps = jsonF32(e.get("norm_eps"), jsonF32(e.get("layer_norm_eps"), 1e-5)),
        .local_attention = try jsonInt(e.get("local_attention"), 128),
        .layer_global = layer_global,
        .rope_theta_global = theta_global,
        .rope_theta_local = theta_local,
        .head_layers = try jsonInt(g.get("head_layers"), 2),
        .max_len = max_len,
        .head_max_len = head_max_len,
        .n_actions = n_actions,
        .temperature = temperature,
        .temperature_by_options = by_opt,
        .cls_id = cls.id,
        .sep_id = sep.id,
        .pad_id = pad.id,
        .mask_id = mask.id,
        .mask_token = try a.dupe(u8, mask.text),
        .allocator = a,
    };
}

/// `TEMP_MIN`/`TEMP_MAX` of laya 0.3.5 and laya-mlx 0.2.0: a fitted temperature below 0.5
/// sharpens the logits enough to report a coin flip as a certainty.
const TEMP_MIN: f32 = 0.5;
const TEMP_MAX: f32 = 5.0;

fn clampTemperature(name: []const u8, t: *f32) void {
    const c = std.math.clamp(t.*, TEMP_MIN, TEMP_MAX);
    if (c != t.*) log.warn("[laya] calibration temperature {s}={d} is outside [{d}, {d}]; using {d}\n", .{ name, t.*, TEMP_MIN, TEMP_MAX, c });
    t.* = c;
}

// ── Prompt construction (laya_mlx.common.build_sequence) ──

/// Python `json.dumps(v, separators=(", ", ": "))` — key order, spacing,
/// escaping and number spelling must match because the result is TOKENIZED.
/// `ascii` mirrors `ensure_ascii`. Request bodies come from `parseRequestJson`
/// (numbers kept as text). A number past the f64 range is `error.NonFiniteNumber`:
/// Python would write `Infinity`, which is not JSON.
/// Nesting past `MAX_JSON_DEPTH` is `error.NestingTooDeep` (one call frame per level).
pub fn pyJson(a: std.mem.Allocator, out: *std.ArrayList(u8), v: std.json.Value, ascii: bool) !void {
    return pyJsonDepth(a, out, v, ascii, 0);
}

fn pyJsonDepth(a: std.mem.Allocator, out: *std.ArrayList(u8), v: std.json.Value, ascii: bool, depth: usize) !void {
    if ((v == .array or v == .object) and depth >= MAX_JSON_DEPTH) return error.NestingTooDeep;
    switch (v) {
        .null => try out.appendSlice(a, "null"),
        .bool => |b| try out.appendSlice(a, if (b) "true" else "false"),
        .integer => |i| try out.print(a, "{d}", .{i}),
        .float => |f| try pyFloat(a, out, f),
        .number_string => |s| {
            if (std.mem.eql(u8, s, "-0")) {
                try out.append(a, '0');
            } else if (std.json.isNumberFormattedLikeAnInteger(s)) {
                // JSON integers have no leading zeros: the text is Python's `int` repr.
                try out.appendSlice(a, s);
            } else {
                try pyFloat(a, out, std.fmt.parseFloat(f64, s) catch return error.NonFiniteNumber);
            }
        },
        .string => |s| try pyJsonString(a, out, s, ascii),
        .array => |arr| {
            try out.append(a, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try out.appendSlice(a, ", ");
                try pyJsonDepth(a, out, item, ascii, depth + 1);
            }
            try out.append(a, ']');
        },
        .object => |obj| {
            try out.append(a, '{');
            var it = obj.iterator();
            var i: usize = 0;
            while (it.next()) |kv| : (i += 1) {
                if (i > 0) try out.appendSlice(a, ", ");
                try pyJsonString(a, out, kv.key_ptr.*, ascii);
                try out.appendSlice(a, ": ");
                try pyJsonDepth(a, out, kv.value_ptr.*, ascii, depth + 1);
            }
            try out.append(a, '}');
        },
    }
}

/// Python `float.__repr__`: shortest round-trip digits; exponent form (`1e-05`,
/// `1.5e+300`) when the decimal point position is <= -4 or > 16, else fixed
/// notation with at least one fraction digit (`1.0`, `-0.0`).
fn pyFloat(a: std.mem.Allocator, out: *std.ArrayList(u8), f: f64) !void {
    if (!std.math.isFinite(f)) return error.NonFiniteNumber;
    var buf: [std.fmt.float.min_buffer_size]u8 = undefined;
    var sci = std.fmt.float.render(&buf, f, .{ .mode = .scientific }) catch unreachable; // "[-]D[.DDD]e[-]X"
    if (sci[0] == '-') {
        try out.append(a, '-');
        sci = sci[1..];
    }
    const e_at = std.mem.indexOfScalar(u8, sci, 'e').?;
    const exp10 = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch unreachable;
    var digits_buf: [24]u8 = undefined;
    var nd: usize = 0;
    for (sci[0..e_at]) |c| if (c != '.') {
        digits_buf[nd] = c;
        nd += 1;
    };
    const digits = digits_buf[0..nd];
    const decpt = exp10 + 1; // value = 0.DIGITS * 10^decpt
    if (decpt <= -4 or decpt > 16) {
        try out.append(a, digits[0]);
        if (nd > 1) {
            try out.append(a, '.');
            try out.appendSlice(a, digits[1..]);
        }
        try out.print(a, "e{c}{d:0>2}", .{ @as(u8, if (exp10 < 0) '-' else '+'), @abs(exp10) });
    } else if (decpt <= 0) {
        try out.appendSlice(a, "0.");
        try out.appendNTimes(a, '0', @intCast(-decpt));
        try out.appendSlice(a, digits);
    } else {
        const int_len: usize = @intCast(decpt);
        if (int_len >= nd) {
            try out.appendSlice(a, digits);
            try out.appendNTimes(a, '0', int_len - nd);
            try out.appendSlice(a, ".0");
        } else {
            try out.appendSlice(a, digits[0..int_len]);
            try out.append(a, '.');
            try out.appendSlice(a, digits[int_len..]);
        }
    }
}

fn pyJsonString(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8, ascii: bool) !void {
    try out.append(a, '"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c >= 0x80) {
            const n = std.unicode.utf8ByteSequenceLength(c) catch 1;
            const end = @min(i + n, s.len);
            if (ascii) {
                // WTF-8 (`parseRequestJson`): a lone surrogate re-emits as its `\udXXX` escape, like Python.
                const cp = std.unicode.wtf8Decode(s[i..end]) catch 0xFFFD;
                if (cp >= 0x10000) {
                    const u = cp - 0x10000;
                    try out.print(a, "\\u{x:0>4}\\u{x:0>4}", .{ 0xD800 + (u >> 10), 0xDC00 + (u & 0x3FF) });
                } else try out.print(a, "\\u{x:0>4}", .{cp});
            } else try out.appendSlice(a, s[i..end]);
            i = end;
            continue;
        }
        i += 1;
        switch (c) {
            '"' => try out.appendSlice(a, "\\\""),
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            0x08 => try out.appendSlice(a, "\\b"),
            0x0C => try out.appendSlice(a, "\\f"),
            0...7, 0x0B, 0x0E...0x1F => try out.print(a, "\\u{x:0>4}", .{c}),
            // ensure_ascii escapes everything outside ' '..'~', DEL included.
            0x7F => if (ascii) try out.appendSlice(a, "\\u007f") else try out.append(a, c),
            else => try out.append(a, c),
        }
    }
    try out.append(a, '"');
}

/// A string in the response body: raw UTF-8, except a string holding a lone surrogate (a
/// question id sent as `"\ud800"`), which goes out `\u`-escaped as Python's `json.dumps` writes it.
fn wireString(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    return pyJsonString(a, out, s, !std.unicode.utf8ValidateSlice(s));
}

/// Parse a `/v1/decisions` body the way Python's `json.loads` reads it, for `pyJson`:
/// numbers keep their text, a repeated key keeps its LAST value at its FIRST position, and a
/// lone UTF-16 surrogate escape (`"\ud800"`, which `std.json` rejects) is kept as WTF-8.
pub fn parseRequestJson(a: std.mem.Allocator, body: []const u8) !std.json.Parsed(std.json.Value) {
    const opts: std.json.ParseOptions = .{ .parse_numbers = false, .duplicate_field_behavior = .use_last, .allocate = .alloc_always };
    if (!try scanLoneSurrogates(body, null)) return std.json.parseFromSlice(std.json.Value, a, body, opts);
    // `std.json` sees each lone surrogate as NUL + its 4 hex digits, and a real `\u0000` as
    // NUL + "0000", so every NUL in a parsed string is followed by its marker; then the
    // strings are decoded back.
    var marked: std.ArrayList(u8) = .empty;
    defer marked.deinit(a);
    try marked.ensureTotalCapacity(a, body.len + body.len / 2);
    _ = try scanLoneSurrogates(body, .{ .a = a, .out = &marked });
    var parsed = try std.json.parseFromSlice(std.json.Value, a, marked.items, opts);
    errdefer parsed.deinit();
    try unmarkValue(parsed.arena.allocator(), &parsed.value);
    return parsed;
}

const MarkedOut = struct { a: std.mem.Allocator, out: *std.ArrayList(u8) };

/// Whether a string in the JSON text holds a lone surrogate escape; with `mark`, also
/// writes the body with the NUL markers of `parseRequestJson`. Malformed escapes are
/// copied as they are for the parser to reject.
fn scanLoneSurrogates(body: []const u8, mark: ?MarkedOut) !bool {
    var in_string = false;
    var lone = false;
    var i: usize = 0;
    var copied: usize = 0; // body[0..copied] already written to `mark`
    while (i < body.len) : (i += 1) {
        const c = body[i];
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string or c != '\\' or i + 1 >= body.len) continue;
        i += 1;
        if (body[i] != 'u') continue;
        const cp = hex4(body, i + 1) orelse continue;
        const esc_start = i - 1;
        i += 4; // at the last hex digit
        if (cp >= 0xD800 and cp <= 0xDBFF and i + 6 < body.len and body[i + 1] == '\\' and body[i + 2] == 'u') {
            if (hex4(body, i + 3)) |lo| if (lo >= 0xDC00 and lo <= 0xDFFF) {
                i += 6; // a proper pair
                continue;
            };
        }
        const surrogate = cp >= 0xD800 and cp <= 0xDFFF;
        lone = lone or surrogate;
        if (mark) |m| if (surrogate or cp == 0) {
            try m.out.appendSlice(m.a, body[copied..esc_start]);
            try m.out.print(m.a, "\\u0000{x:0>4}", .{cp});
            copied = i + 1;
        };
    }
    if (mark) |m| try m.out.appendSlice(m.a, body[copied..]);
    return lone;
}

fn hex4(body: []const u8, at: usize) ?u16 {
    if (at + 4 > body.len) return null;
    return std.fmt.parseInt(u16, body[at .. at + 4], 16) catch null;
}

/// Decode the NUL markers of `parseRequestJson` in every string and key under `v`.
fn unmarkValue(arena: std.mem.Allocator, v: *std.json.Value) !void {
    switch (v.*) {
        .string => |s| v.* = .{ .string = try unmarkString(arena, s) },
        .array => |*arr| for (arr.items) |*item| try unmarkValue(arena, item),
        .object => |*obj| {
            for (obj.keys()) |*k| k.* = try unmarkString(arena, k.*);
            try obj.reIndex(arena);
            for (obj.values()) |*item| try unmarkValue(arena, item);
        },
        else => {},
    }
}

fn unmarkString(arena: std.mem.Allocator, s: []const u8) ![]const u8 {
    if (std.mem.indexOfScalar(u8, s, 0) == null) return s;
    var out: std.ArrayList(u8) = try .initCapacity(arena, s.len);
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] != 0 or i + 5 > s.len) {
            out.appendAssumeCapacity(s[i]);
            i += 1;
            continue;
        }
        const cp = std.fmt.parseInt(u16, s[i + 1 .. i + 5], 16) catch {
            out.appendAssumeCapacity(0);
            i += 1;
            continue;
        };
        var buf: [4]u8 = undefined;
        const n = std.unicode.wtf8Encode(cp, &buf) catch unreachable; // cp < 0x10000
        out.appendSliceAssumeCapacity(buf[0..n]);
        i += 5;
    }
    return out.items;
}

/// Deepest `[`/`{` nesting a request may carry. Python's `json` stops at its recursion
/// limit; the serializer here takes one call frame per level.
pub const MAX_JSON_DEPTH = 512;

/// `error.NestingTooDeep` when JSON text nests past `MAX_JSON_DEPTH`. A byte scan, so it
/// runs before any recursive parse, serialize or free; brackets inside strings do not count.
pub fn checkJsonDepth(body: []const u8) !void {
    var depth: usize = 0;
    var in_string = false;
    var i: usize = 0;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            '"' => in_string = !in_string,
            '\\' => if (in_string) {
                i += 1;
            },
            '[', '{' => if (!in_string) {
                depth += 1;
                if (depth > MAX_JSON_DEPTH) return error.NestingTooDeep;
            },
            ']', '}' => if (!in_string) {
                depth -|= 1;
            },
            else => {},
        }
    }
}

/// `serialize_state` / `render_criterion`: strings pass through, anything
/// else is compact-ish JSON with non-ASCII kept.
fn renderValue(a: std.mem.Allocator, v: std.json.Value) ![]u8 {
    if (v == .string) return a.dupe(u8, v.string);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try pyJson(a, &out, v, false);
    return out.toOwnedSlice(a);
}

fn isNullOrEmpty(v: ?std.json.Value) bool {
    const x = v orelse return true;
    return switch (x) {
        .null => true,
        .string => |s| s.len == 0,
        else => false,
    };
}

/// One question in its internal form (`Agent._to_internal`). `crit` borrows
/// the parsed request; `ins` is owned.
pub const Question = struct {
    t: QType,
    ins: []u8,
    /// choice: option labels in order (borrowed keys); score: criteria array;
    /// noul: optional {false, true} object.
    labels: []const []const u8 = &.{},
    crit: ?std.json.Value = null,

    pub fn deinit(self: *Question, a: std.mem.Allocator) void {
        a.free(self.ins);
        if (self.labels.len > 0) a.free(self.labels);
    }

    /// `max_options` (`maxOptions`) bounds every option list before any per-option work.
    pub fn fromJson(a: std.mem.Allocator, def: std.json.Value, max_options: usize) !Question {
        if (def != .object) return error.QuestionNotObject;
        const o = def.object;
        const kind_v = o.get("type") orelse return error.UnknownQuestionType;
        if (kind_v != .string) return error.UnknownQuestionType;
        const kind = QType.parse(kind_v.string) orelse return error.UnknownQuestionType;
        const ins_v = o.get("instructions") orelse return error.MissingInstructions;
        const ins = blk: {
            if (ins_v == .string) break :blk try a.dupe(u8, ins_v.string);
            // json.dumps(instructions) — default ensure_ascii=True.
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(a);
            try pyJson(a, &out, ins_v, true);
            break :blk try out.toOwnedSlice(a);
        };
        errdefer a.free(ins);
        const crit = o.get("criteria");
        var labels: []const []const u8 = &.{};
        switch (kind) {
            .choice => {
                const c = crit orelse return error.BadChoiceCriteria;
                switch (c) {
                    .array => |arr| {
                        if (arr.items.len == 0) return error.BadChoiceCriteria;
                        if (arr.items.len > max_options) return error.TooManyOptions;
                        const ls = try a.alloc([]const u8, arr.items.len);
                        errdefer a.free(ls);
                        var seen: std.StringHashMapUnmanaged(void) = .empty;
                        defer seen.deinit(a);
                        try seen.ensureTotalCapacity(a, @intCast(arr.items.len));
                        for (arr.items, 0..) |v, i| {
                            if (v != .string) return error.BadChoiceCriteria;
                            if (seen.getOrPutAssumeCapacity(v.string).found_existing) return error.BadChoiceCriteria;
                            ls[i] = v.string;
                        }
                        labels = ls;
                    },
                    .object => |obj| {
                        if (obj.count() == 0) return error.BadChoiceCriteria;
                        if (obj.count() > max_options) return error.TooManyOptions;
                        const ls = try a.alloc([]const u8, obj.count());
                        errdefer a.free(ls);
                        var it = obj.iterator();
                        var i: usize = 0;
                        while (it.next()) |kv| : (i += 1) ls[i] = kv.key_ptr.*;
                        labels = ls;
                    },
                    else => return error.BadChoiceCriteria,
                }
            },
            .score => {
                const c = crit orelse return error.BadScoreCriteria;
                if (c != .array or c.array.items.len == 0) return error.BadScoreCriteria;
                if (c.array.items.len > max_options) return error.TooManyOptions;
            },
            .noul => {
                if (crit) |c| if (c != .null and c != .object) return error.BadNoulCriteria;
            },
        }
        return .{ .t = kind, .ins = ins, .labels = labels, .crit = crit };
    }

    /// Value of choice option `i` (null when the label has no description).
    fn choiceDesc(self: *const Question, i: usize) ?std.json.Value {
        const c = self.crit orelse return null;
        if (c != .object) return null;
        return c.object.get(self.labels[i]);
    }

    pub fn optionCount(self: *const Question) usize {
        return switch (self.t) {
            .choice => self.labels.len,
            .score => self.crit.?.array.items.len,
            .noul => 2,
        };
    }
};

/// Most options a question can carry: each option holds at least its marker token, placed
/// after [CLS], at least one head token and [SEP], and a marker at or past `max_len` is lost
/// (`build_sequence`), so more than `max_len - 3` options never fit.
pub fn maxOptions(cfg: *const Config) usize {
    return cfg.max_len -| 3;
}

/// The `questions` object of one request, validated. Needs no tokenizer and no
/// MLX, so the server builds it before the request queues for the inference
/// thread. Borrows `questions`.
pub const Questions = struct {
    ids: [][]const u8,
    qs: []Question,

    pub fn init(a: std.mem.Allocator, questions: std.json.Value, max_questions: usize, max_options: usize) !Questions {
        if (questions != .object) return error.QuestionsNotObject;
        const qobj = questions.object;
        if (qobj.count() > max_questions) return error.TooManyQuestions;
        const ids = try a.alloc([]const u8, qobj.count());
        errdefer a.free(ids);
        const qs = try a.alloc(Question, qobj.count());
        var n: usize = 0;
        errdefer {
            for (qs[0..n]) |*q| q.deinit(a);
            a.free(qs);
        }
        var it = qobj.iterator();
        while (it.next()) |kv| : (n += 1) {
            ids[n] = kv.key_ptr.*;
            qs[n] = try Question.fromJson(a, kv.value_ptr.*, max_options);
        }
        return .{ .ids = ids, .qs = qs };
    }

    pub fn deinit(self: *Questions, a: std.mem.Allocator) void {
        for (self.qs) |*q| q.deinit(a);
        a.free(self.qs);
        a.free(self.ids);
    }
};

/// `render_options`: option texts in label-index order. Caller frees each and the slice.
pub fn renderOptions(a: std.mem.Allocator, q: *const Question) ![][]u8 {
    const n = q.optionCount();
    const out = try a.alloc([]u8, n);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |s| a.free(s);
        a.free(out);
    }
    switch (q.t) {
        .choice => for (q.labels, 0..) |label, i| {
            const d = q.choiceDesc(i);
            if (isNullOrEmpty(d)) {
                out[i] = try a.dupe(u8, label);
            } else {
                const r = try renderValue(a, d.?);
                defer a.free(r);
                out[i] = try std.fmt.allocPrint(a, "{s}: {s}", .{ label, r });
            }
            made += 1;
        },
        .score => for (q.crit.?.array.items, 0..) |c, i| {
            const r = try renderValue(a, c);
            defer a.free(r);
            out[i] = try std.fmt.allocPrint(a, "level {d}: {s}", .{ i, r });
            made += 1;
        },
        .noul => {
            const obj: ?std.json.ObjectMap = if (q.crit) |c| (if (c == .object) c.object else null) else null;
            const f = if (obj) |o| o.get("false") else null;
            const t = if (obj) |o| o.get("true") else null;
            if (isNullOrEmpty(f)) {
                out[0] = try a.dupe(u8, "false: no, the statement does not hold");
            } else {
                const r = try renderValue(a, f.?);
                defer a.free(r);
                out[0] = try std.fmt.allocPrint(a, "false: {s}", .{r});
            }
            made = 1;
            if (isNullOrEmpty(t)) {
                out[1] = try a.dupe(u8, "true: yes, the statement holds");
            } else {
                const r = try renderValue(a, t.?);
                defer a.free(r);
                out[1] = try std.fmt.allocPrint(a, "true: {s}", .{r});
            }
            made = 2;
        },
    }
    return out;
}

/// Encode `text` with the mask token blanked (`text.replace(mask_tok, " ")`),
/// no specials added. A lone surrogate (WTF-8 from `parseRequestJson`) is
/// `error.LoneSurrogate`: Python's tokenizer refuses a str holding one.
fn encodeClean(a: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer, cache: ?*TokenCache, mask_token: []const u8, text: []const u8) ![]u32 {
    if (cache) |c| if (c.map.get(text)) |ids| return a.dupe(u32, ids);
    if (!std.unicode.utf8ValidateSlice(text)) return error.LoneSurrogate;
    const ids = if (std.mem.indexOf(u8, text, mask_token) == null) try tok.encode(a, text) else blk: {
        const cleaned = try std.mem.replaceOwned(u8, a, text, mask_token, " ");
        defer a.free(cleaned);
        break :blk try tok.encode(a, cleaned);
    };
    if (cache) |c| c.put(text, ids);
    return ids;
}

/// Token ids of question and option texts, keyed by text: a client polling
/// the same questions skips the tokenizer for them. At most `MAX` entries of
/// texts up to `MAX_TEXT` bytes; when full it starts over.
pub const TokenCache = struct {
    allocator: std.mem.Allocator,
    map: std.StringHashMapUnmanaged([]u32) = .empty,

    pub const MAX = 256;
    pub const MAX_TEXT = 4096;

    /// Store a copy of `ids` for `text`; out of memory only skips the entry.
    pub fn put(self: *TokenCache, text: []const u8, ids: []const u32) void {
        if (text.len > MAX_TEXT or self.map.contains(text)) return;
        if (self.map.count() >= MAX) self.clear();
        const key = self.allocator.dupe(u8, text) catch return;
        const val = self.allocator.dupe(u32, ids) catch {
            self.allocator.free(key);
            return;
        };
        self.map.put(self.allocator, key, val) catch {
            self.allocator.free(key);
            self.allocator.free(val);
        };
    }

    fn clear(self: *TokenCache) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            self.allocator.free(e.value_ptr.*);
        }
        self.map.clearRetainingCapacity();
    }

    pub fn deinit(self: *TokenCache) void {
        self.clear();
        self.map.deinit(self.allocator);
    }
};

pub const Sequence = struct {
    ids: []u32,
    markers: []u32,
    pub fn deinit(self: *Sequence, a: std.mem.Allocator) void {
        a.free(self.ids);
        a.free(self.markers);
    }
};

/// `build_sequence`: [CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP],
/// with the head capped at `head_max_len` and the state's token ids `state_ids` filling `max_len`.
/// `cache` (optional) holds the token ids of question and option texts.
pub fn buildSequence(a: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer, cache: ?*TokenCache, cfg: *const Config, state_ids: []const u32, q: *const Question) !Sequence {
    const opts = try renderOptions(a, q);
    defer {
        for (opts) |o| a.free(o);
        a.free(opts);
    }
    const head_text = try std.fmt.allocPrint(a, "{s} question: {s}", .{ q.t.name(), q.ins });
    defer a.free(head_text);
    var head_ids = try encodeClean(a, tok, cache, cfg.mask_token, head_text);
    defer a.free(head_ids);

    var opt_ids = try a.alloc([]u32, opts.len);
    var n_opt: usize = 0;
    defer {
        for (opt_ids[0..n_opt]) |o| a.free(o);
        a.free(opt_ids);
    }
    for (opts, 0..) |opt, i| {
        const spaced = try std.fmt.allocPrint(a, " {s}", .{opt});
        defer a.free(spaced);
        const enc = try encodeClean(a, tok, cache, cfg.mask_token, spaced);
        defer a.free(enc);
        const keep = @min(enc.len, 48);
        const o = try a.alloc(u32, 1 + keep);
        o[0] = cfg.mask_id;
        @memcpy(o[1..], enc[0..keep]);
        opt_ids[i] = o;
        n_opt += 1;
    }
    var opt_total: usize = 0;
    for (opt_ids) |o| opt_total += o.len;
    var opt_budget: isize = @as(isize, @intCast(cfg.head_max_len)) - @as(isize, @intCast(opt_total));
    if (opt_budget < 16) {
        const per: usize = @max(4, (@as(usize, cfg.head_max_len) -| 16) / @max(1, opt_ids.len));
        opt_total = 0;
        for (opt_ids) |*o| {
            if (o.len > per) o.* = try a.realloc(o.*, per);
            opt_total += o.len;
        }
        opt_budget = @as(isize, @intCast(cfg.head_max_len)) - @as(isize, @intCast(opt_total));
    }
    const head_keep: usize = @intCast(@max(8, opt_budget));
    if (head_ids.len > head_keep) head_ids = try a.realloc(head_ids, head_keep);

    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);
    var markers: std.ArrayList(u32) = .empty;
    errdefer markers.deinit(a);
    try ids.append(a, cfg.cls_id);
    try ids.appendSlice(a, head_ids);
    try ids.append(a, cfg.sep_id);
    for (opt_ids) |o| {
        try markers.append(a, @intCast(ids.items.len));
        try ids.appendSlice(a, o);
    }
    try ids.append(a, cfg.sep_id);

    const room: usize = @as(usize, cfg.max_len) -| (ids.items.len + 1);
    try ids.appendSlice(a, state_ids[0..@min(state_ids.len, room)]);
    try ids.append(a, cfg.sep_id);
    if (ids.items.len > cfg.max_len) ids.shrinkRetainingCapacity(cfg.max_len);
    var kept: usize = 0;
    for (markers.items) |m| {
        if (m < cfg.max_len) {
            markers.items[kept] = m;
            kept += 1;
        }
    }
    markers.shrinkRetainingCapacity(kept);
    const owned_markers = try markers.toOwnedSlice(a);
    errdefer a.free(owned_markers);
    return .{ .ids = try ids.toOwnedSlice(a), .markers = owned_markers };
}

// ── MLX primitives ──

fn free(x: A) void {
    _ = mlx.mlx_array_free(x);
}

fn matmul(x: A, w_t: A, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, x, w_t, s));
    return out;
}

fn add(x: A, y: A, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, x, y, s));
    return out;
}

/// `x @ w^T + b` with `w_t` already stored as `[in, out]`.
fn linear(x: A, w_t: A, b: ?A, s: S) !A {
    const y = try matmul(x, w_t, s);
    if (b) |bias| {
        defer free(y);
        return add(y, bias, s);
    }
    return y;
}

fn layerNorm(x: A, w: A, b: ?A, eps: f32, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b orelse none, eps, s));
    return out;
}

/// 0-d constant in `like`'s dtype. `mlx_array_new_float` is a float32 0-d
/// ARRAY, and MLX promotes fp16 operands against it to float32 (Python
/// scalars keep the operand dtype) — one such constant in the MLP turned the
/// whole residual stream fp32 (measured: encoder 5.4 ms -> 3.9 ms).
fn scalarLike(like: A, v: f32) A {
    const no_shape = [_]c_int{};
    switch (mlx.mlx_array_dtype(like)) {
        .float16 => {
            const h: f16 = @floatCast(v);
            return mlx.mlx_array_new_data(&h, &no_shape, 0, .float16);
        },
        .bfloat16 => {
            const b: u16 = @truncate(@as(u32, @bitCast(v)) >> 16);
            return mlx.mlx_array_new_data(&b, &no_shape, 0, .bfloat16);
        },
        else => return mlx.mlx_array_new_float(v),
    }
}

/// Exact (erf) GELU — `mlx.nn.gelu`.
fn gelu(x: A, s: S) !A {
    const inv_sqrt2 = scalarLike(x, 1.0 / @sqrt(2.0));
    defer free(inv_sqrt2);
    var scaled = mlx.mlx_array_new();
    defer free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, x, inv_sqrt2, s));
    var e = mlx.mlx_array_new();
    defer free(e);
    try mlx.check(mlx.mlx_erf(&e, scaled, s));
    const one = scalarLike(x, 1.0);
    defer free(one);
    var onep = mlx.mlx_array_new();
    defer free(onep);
    try mlx.check(mlx.mlx_add(&onep, e, one, s));
    var prod = mlx.mlx_array_new();
    defer free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, x, onep, s));
    const half = scalarLike(x, 0.5);
    defer free(half);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, prod, half, s));
    return out;
}

fn relu(x: A, s: S) !A {
    const zero = scalarLike(x, 0.0);
    defer free(zero);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_maximum(&out, x, zero, s));
    return out;
}

fn reshape(x: A, shape: []const c_int, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, s));
    return out;
}

fn transposeAxes(x: A, axes: []const c_int, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, x, axes.ptr, axes.len, s));
    return out;
}

/// `[N, T, 3*H*Dh]` -> three `[N, H, T, Dh]` (q, k, v). Caller frees all three.
fn splitQkv(qkv: A, n: c_int, t: c_int, heads: c_int, head_dim: c_int, s: S) ![3]A {
    return splitHeads(3, qkv, n, t, heads, head_dim, s);
}

/// `[N, T, P*H*Dh]` -> `P` arrays `[N, H, T, Dh]`. Caller frees all of them.
fn splitHeads(comptime P: usize, x: A, n: c_int, t: c_int, heads: c_int, head_dim: c_int, s: S) ![P]A {
    const r = try reshape(x, &[_]c_int{ n, t, P, heads, head_dim }, s);
    defer free(r);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_split(&parts, r, P, 2, s));
    var out: [P]A = @splat(none);
    errdefer for (out) |o| if (o.ctx != null) free(o);
    for (0..P) |i| {
        var part = mlx.mlx_array_new();
        defer free(part);
        try mlx.check(mlx.mlx_vector_array_get(&part, parts, i));
        const sq = try reshape(part, &[_]c_int{ n, t, heads, head_dim }, s);
        defer free(sq);
        out[i] = try transposeAxes(sq, &[_]c_int{ 0, 2, 1, 3 }, s);
    }
    return out;
}

/// `[N, H, T, Dh]` -> `[N, T, H*Dh]`.
fn mergeHeads(x: A, n: c_int, t: c_int, d: c_int, s: S) !A {
    const tr = try transposeAxes(x, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer free(tr);
    return reshape(tr, &[_]c_int{ n, t, d }, s);
}

fn rope(x: A, dims: c_int, base: f32, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&out, x, dims, false, mlx.mlx_optional_float.some(base), 1.0, 0, none, s));
    return out;
}

fn sdpa(q: A, k: A, v: A, scale: f32, mask: A, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, "array", mask, none, false, s));
    return out;
}

fn take(x: A, idx: A, axis: c_int, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&out, x, idx, axis, s));
    return out;
}

fn astype(x: A, dt: mlx.mlx_dtype, s: S) !A {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, x, dt, s));
    return out;
}

// ── Weights ──

const Linear = struct {
    /// `[in, out]` transpose view of the `[out, in]` checkpoint weight.
    w_t: A,
    b: ?A,
};

const EncoderLayer = struct {
    attn_norm: ?A, // layer 0 has none (identity)
    wqkv: Linear,
    wo: Linear,
    mlp_norm: A,
    wi: Linear,
    wo2: Linear,
    global: bool,
};

const HeadLayer = struct {
    norm1_w: A,
    norm1_b: A,
    in_proj: Linear,
    out_proj: Linear,
    norm2_w: A,
    norm2_b: A,
    linear1: Linear,
    linear2: Linear,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    tok: tokenizer_mod.Tokenizer,
    weights: ltx.Component,
    stream: S,
    tok_embeddings: A,
    emb_norm: A,
    layers: []EncoderLayer,
    final_norm: A,
    type_emb: A,
    head: []HeadLayer,
    scorer_norm_w: A,
    scorer_norm_b: A,
    scorer_l1: Linear,
    scorer_l3: Linear,
    act_l0: Linear,
    act_l2: Linear,
    /// The last head layer's `in_proj` split into its query and key/value
    /// columns: that layer computes queries for the CLS and marker rows only.
    last_q: Linear = undefined,
    last_kv: Linear = undefined,
    tok_cache: TokenCache,
    /// Transpose views (`Linear.w_t`) not owned by `weights`.
    owned: std.ArrayList(A),
    /// `forwardBody` wrapped by `mlx_compile`: traced once per input shape,
    /// then replayed with fused elementwise kernels (the reference wraps its
    /// model in `mx.compile` the same way). Null: lazy graph (compile off or
    /// failed).
    compiled: ?mlx.mlx_closure = null,
    compile_failed: bool = false,
    /// Keys `getW`/`optW` found; `load` refuses any other checkpoint key, then empties this.
    requested: std.StringHashMapUnmanaged(void) = .empty,

    /// Bucketed `[rows, tokens, options]` shapes the closure has traced. Each
    /// trace keeps its own graph and buffers, so past the cap a new shape runs lazy.
    compiled_shapes: std.AutoHashMapUnmanaged([3]usize, void) = .empty,

    pub const MAX_COMPILED_SHAPES: usize = 64;

    /// Width of the act head's hidden layer (`nn.Linear(dims + 4, 256)` in laya_mlx).
    const ACT_HIDDEN = 256;

    /// The tensor `fmt`/`args`, which must have the shape `want` the config implies
    /// (laya_mlx loads with `strict=True`).
    fn getW(self: *Model, comptime fmt: []const u8, args: anytype, want: []const u32) !A {
        var buf: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, fmt, args);
        return try self.optW("{s}", .{key}, want) orelse {
            log.err("[laya] missing weight {s}\n", .{key});
            return error.MissingWeight;
        };
    }

    fn optW(self: *Model, comptime fmt: []const u8, args: anytype, want: []const u32) !?A {
        var buf: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, fmt, args);
        const e = self.weights.map.getEntry(key) orelse return null;
        try checkShape(key, e.value_ptr.*, want);
        try self.requested.put(self.allocator, e.key_ptr.*, {});
        return e.value_ptr.*;
    }

    fn checkShape(key: []const u8, w: A, want: []const u32) !void {
        const got = mlx.mlx_array_shape(w)[0..mlx.mlx_array_ndim(w)];
        const same = got.len == want.len and for (got, want) |g, x| {
            if (g != x) break false;
        } else true;
        if (!same) {
            log.err("[laya] invalid checkpoint: {s} has shape {any}, the config implies {any}\n", .{ key, got, want });
            return error.WeightShapeMismatch;
        }
    }

    /// `w^T` as a view: matmul reads the transposed strides, so a contiguous
    /// copy would hold every linear weight twice for the same output.
    fn linearW(self: *Model, comptime prefix: []const u8, args: anytype, out_dim: u32, in_dim: u32) !Linear {
        const w = try self.getW(prefix ++ ".weight", args, &.{ out_dim, in_dim });
        const b = try self.optW(prefix ++ ".bias", args, &.{out_dim});
        var w_t = mlx.mlx_array_new();
        errdefer free(w_t);
        try mlx.check(mlx.mlx_transpose(&w_t, w, self.stream));
        try self.owned.append(self.allocator, w_t);
        return .{ .w_t = w_t, .b = b };
    }

    /// Output columns `[lo, hi)` of `l`: views of its weight and bias.
    fn sliceOut(self: *Model, l: Linear, lo: u32, hi: u32) !Linear {
        const rows = mlx.mlx_array_shape(l.w_t)[0];
        var w = mlx.mlx_array_new();
        errdefer free(w);
        try mlx.check(mlx.mlx_slice(&w, l.w_t, &[_]c_int{ 0, @intCast(lo) }, 2, &[_]c_int{ rows, @intCast(hi) }, 2, &[_]c_int{ 1, 1 }, 2, self.stream));
        try self.owned.append(self.allocator, w);
        const b = if (l.b) |bias| blk: {
            var out = mlx.mlx_array_new();
            errdefer free(out);
            try mlx.check(mlx.mlx_slice(&out, bias, &[_]c_int{@intCast(lo)}, 1, &[_]c_int{@intCast(hi)}, 1, &[_]c_int{1}, 1, self.stream));
            try self.owned.append(self.allocator, out);
            break :blk out;
        } else null;
        return .{ .w_t = w, .b = b };
    }

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !Model {
        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        var tok = try tokenizer_mod.loadTokenizer(io, allocator, tok_dir);
        errdefer tok.deinit();
        var cfg = try parseConfig(io, allocator, model_dir, &tok);
        errdefer cfg.deinit();

        // safetensors READS run on the CPU stream (mlx's Load has no GPU impl).
        const load_s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(load_s);
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{model_dir}, 0);
        defer allocator.free(path);
        var weights = try ltx.loadComponent(allocator, path, load_s);
        errdefer weights.deinit();

        var self = Model{
            .allocator = allocator,
            .cfg = cfg,
            .tok = tok,
            .weights = weights,
            .stream = s,
            .tok_embeddings = undefined,
            .emb_norm = undefined,
            .layers = &.{},
            .final_norm = undefined,
            .type_emb = undefined,
            .head = &.{},
            .scorer_norm_w = undefined,
            .scorer_norm_b = undefined,
            .scorer_l1 = undefined,
            .scorer_l3 = undefined,
            .act_l0 = undefined,
            .act_l2 = undefined,
            .tok_cache = .{ .allocator = allocator },
            .owned = .empty,
        };
        errdefer self.freeOwned();
        errdefer self.requested.deinit(allocator);

        const D = cfg.hidden_size;
        const I = cfg.intermediate_size;
        self.tok_embeddings = try self.getW("encoder.embeddings.tok_embeddings.weight", .{}, &.{ cfg.vocab_size, D });
        self.emb_norm = try self.getW("encoder.embeddings.norm.weight", .{}, &.{D});
        self.final_norm = try self.getW("encoder.final_norm.weight", .{}, &.{D});
        self.type_emb = try self.getW("type_emb.weight", .{}, &.{ 3, D });

        self.layers = try allocator.alloc(EncoderLayer, cfg.num_layers);
        for (self.layers, 0..) |*l, i| {
            l.* = .{
                .attn_norm = if (i == 0) null else try self.getW("encoder.layers.{d}.attn_norm.weight", .{i}, &.{D}),
                .wqkv = try self.linearW("encoder.layers.{d}.attn.Wqkv", .{i}, 3 * D, D),
                .wo = try self.linearW("encoder.layers.{d}.attn.Wo", .{i}, D, D),
                .mlp_norm = try self.getW("encoder.layers.{d}.mlp_norm.weight", .{i}, &.{D}),
                .wi = try self.linearW("encoder.layers.{d}.mlp.Wi", .{i}, 2 * I, D),
                .wo2 = try self.linearW("encoder.layers.{d}.mlp.Wo", .{i}, D, I),
                .global = cfg.layer_global[i],
            };
        }
        self.head = try allocator.alloc(HeadLayer, cfg.head_layers);
        for (self.head, 0..) |*h, i| {
            h.* = .{
                .norm1_w = try self.getW("head.layers.{d}.norm1.weight", .{i}, &.{D}),
                .norm1_b = try self.getW("head.layers.{d}.norm1.bias", .{i}, &.{D}),
                .in_proj = try self.linearW("head.layers.{d}.self_attn.in_proj", .{i}, 3 * D, D),
                .out_proj = try self.linearW("head.layers.{d}.self_attn.out_proj", .{i}, D, D),
                .norm2_w = try self.getW("head.layers.{d}.norm2.weight", .{i}, &.{D}),
                .norm2_b = try self.getW("head.layers.{d}.norm2.bias", .{i}, &.{D}),
                .linear1 = try self.linearW("head.layers.{d}.linear1", .{i}, 4 * D, D),
                .linear2 = try self.linearW("head.layers.{d}.linear2", .{i}, D, 4 * D),
            };
        }
        self.scorer_norm_w = try self.getW("scorer.layers.0.weight", .{}, &.{D});
        self.scorer_norm_b = try self.getW("scorer.layers.0.bias", .{}, &.{D});
        self.scorer_l1 = try self.linearW("scorer.layers.1", .{}, D, D);
        self.scorer_l3 = try self.linearW("scorer.layers.3", .{}, 1, D);
        // Pooled CLS row + 4 confidence features in, `len(act_costs) + 1` actions out.
        self.act_l0 = try self.linearW("act_head.layers.0", .{}, ACT_HIDDEN, D + 4);
        self.act_l2 = try self.linearW("act_head.layers.2", .{}, cfg.n_actions, ACT_HIDDEN);
        if (self.head.len > 0) {
            const last = self.head[self.head.len - 1].in_proj;
            self.last_q = try self.sliceOut(last, 0, D);
            self.last_kv = try self.sliceOut(last, D, 3 * D);
        }
        // A buffer of the reference model; calibration reads `temperature` from rl_agent_config.json.
        _ = try self.getW("temperature", .{}, &.{3});
        var keys = self.weights.map.keyIterator();
        while (keys.next()) |k| if (!self.requested.contains(k.*)) {
            log.err("[laya] invalid checkpoint: unexpected weight {s} (the config implies no such tensor)\n", .{k.*});
            return error.UnexpectedWeight;
        };
        self.requested.deinit(allocator);
        self.requested = .empty;

        // Read every tensor now: a tensor left lazy is read again, and kept, by
        // every compiled input shape that uses it.
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var wit = self.weights.map.valueIterator();
        while (wit.next()) |w| try mlx.check(mlx.mlx_vector_array_append_value(vec, w.*));
        for (self.owned.items) |w| try mlx.check(mlx.mlx_vector_array_append_value(vec, w));
        try mlx.check(mlx.mlx_eval(vec));
        log.info("[laya] ready — {d} tensors, {d} encoder layers, {d} head layers, max_len {d}\n", .{ weights.count(), cfg.num_layers, cfg.head_layers, cfg.max_len });
        return self;
    }

    fn freeOwned(self: *Model) void {
        self.tok_cache.deinit();
        for (self.owned.items) |w| free(w);
        self.owned.deinit(self.allocator);
        if (self.layers.len > 0) self.allocator.free(self.layers);
        if (self.head.len > 0) self.allocator.free(self.head);
    }

    pub fn deinit(self: *Model) void {
        if (self.compiled) |c| _ = mlx.mlx_closure_free(c);
        self.compiled_shapes.deinit(self.allocator);
        self.freeOwned();
        self.weights.deinit();
        self.cfg.deinit();
        self.tok.deinit();
    }

    /// One padded batch. `ids[i]` are the token ids of row i (no padding),
    /// `markers[i]` the marker positions, `qtype[i]` the question type.
    pub const Batch = struct {
        ids: []const []const u32,
        markers: []const []const u32,
        qtype: []const QType,
    };

    pub const Output = struct {
        /// Row-major `[n, k_pad]` raw scorer logits (masked slots = -1e4).
        logits: []f32,
        k_pad: usize,
        /// Row-major `[n, n_actions]` softmax of the action head.
        act: []f32,
        n_actions: usize,
        pub fn deinit(self: *Output, a: std.mem.Allocator) void {
            a.free(self.logits);
            a.free(self.act);
        }
    };

    /// Padding granularity of the input shape `[n, t, k]`: bounds the number of
    /// distinct shapes the compiled forward traces. Rows past 8 go to a multiple
    /// of 8 with dummy rows; neither those nor masked option slots change a real row.
    pub const PAD_T: usize = 16;
    pub const PAD_K: usize = 8;

    pub fn bucketShape(n: usize, t: usize, k: usize) [3]usize {
        const rows = if (n > 8) (n + 7) / 8 * 8 else @max(n, 1);
        return .{ rows, (t + PAD_T - 1) / PAD_T * PAD_T, (@max(k, 2) + PAD_K - 1) / PAD_K * PAD_K };
    }

    /// The five device inputs of `DecisionModel.__call__` (`input_ids`,
    /// `attention_mask`, `marker_pos`, `marker_mask`, `qtype`).
    pub const Inputs = struct {
        ids: A, // [n, t] int32
        valid: A, // [n, t] bool
        marker_pos: A, // [n, k] int32 (padded slots 0)
        marker_mask: A, // [n, k] bool
        qtype: A, // [n] int32

        pub fn fromBatch(a: std.mem.Allocator, batch: Batch, pad_id: u32) !Inputs {
            var t_max: usize = 0;
            var k_max: usize = 0;
            for (batch.ids, batch.markers) |row, m| {
                t_max = @max(t_max, row.len);
                k_max = @max(k_max, m.len);
            }
            const n, const t, const k = bucketShape(batch.ids.len, t_max, k_max);
            const ids = try a.alloc(i32, n * t);
            defer a.free(ids);
            const valid = try a.alloc(bool, n * t);
            defer a.free(valid);
            const mpos = try a.alloc(i32, n * k);
            defer a.free(mpos);
            const mmask = try a.alloc(bool, n * k);
            defer a.free(mmask);
            const qt = try a.alloc(i32, n);
            defer a.free(qt);
            for (0..n) |i| {
                // A dummy row: one valid pad token (no all-masked softmax row), no option.
                const row: []const u32 = if (i < batch.ids.len) batch.ids[i] else &.{pad_id};
                const marks: []const u32 = if (i < batch.ids.len) batch.markers[i] else &.{};
                for (0..t) |j| {
                    const in_row = j < row.len;
                    ids[i * t + j] = if (in_row) @intCast(row[j]) else @intCast(pad_id);
                    valid[i * t + j] = in_row;
                }
                for (0..k) |j| {
                    const in_row = j < marks.len;
                    mpos[i * k + j] = if (in_row) @intCast(marks[j]) else 0;
                    mmask[i * k + j] = in_row;
                }
                qt[i] = if (i < batch.ids.len) @intFromEnum(batch.qtype[i]) else 0;
            }
            const N: c_int = @intCast(n);
            const T: c_int = @intCast(t);
            const K: c_int = @intCast(k);
            return .{
                .ids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ N, T }, 2, .int32),
                .valid = mlx.mlx_array_new_data(valid.ptr, &[_]c_int{ N, T }, 2, .bool_),
                .marker_pos = mlx.mlx_array_new_data(mpos.ptr, &[_]c_int{ N, K }, 2, .int32),
                .marker_mask = mlx.mlx_array_new_data(mmask.ptr, &[_]c_int{ N, K }, 2, .bool_),
                .qtype = mlx.mlx_array_new_data(qt.ptr, &[_]c_int{N}, 1, .int32),
            };
        }

        pub fn deinit(self: *Inputs) void {
            for ([_]A{ self.ids, self.valid, self.marker_pos, self.marker_mask, self.qtype }) |x| free(x);
        }
    };

    /// Attention masks of one forward, built once and shared by every
    /// encoder and head layer. Additive fp16 (0 keeps a key, -inf drops it):
    /// the attention kernel adds them to the scores as they are.
    pub const Masks = struct {
        full: A, // [n, 1, 1, T]: every valid key
        local: A, // [n, 1, T, T]: `localMask`

        pub fn init(valid: A, half: c_int, s: S) !Masks {
            const shape = mlx.mlx_array_shape(valid);
            const N = shape[0];
            const T = shape[1];
            const full_b = try reshape(valid, &[_]c_int{ N, 1, 1, T }, s);
            defer free(full_b);
            const local_b = try localMask(valid, N, T, half, s);
            defer free(local_b);
            const full = try additiveMask(full_b, s);
            errdefer free(full);
            return .{ .full = full, .local = try additiveMask(local_b, s) };
        }

        pub fn deinit(self: *Masks) void {
            free(self.full);
            free(self.local);
        }
    };

    pub fn masks(self: *const Model, valid: A) !Masks {
        return Masks.init(valid, @intCast(self.cfg.local_attention / 2), self.stream);
    }

    /// Encoder only: `[n, T, D]` fp16 (final norm applied) from `ids` `[n, T]`
    /// int32 and the forward's `masks`. Caller frees.
    pub fn encode(self: *Model, ids_arr: A, m: Masks) !A {
        const s = self.stream;
        const D: c_int = @intCast(self.cfg.hidden_size);
        const H: c_int = @intCast(self.cfg.num_heads);
        const Dh: c_int = @intCast(self.cfg.head_dim);
        const shape = mlx.mlx_array_shape(ids_arr);
        const N = shape[0];
        const T = shape[1];
        const full_mask = m.full;
        const local_mask = m.local;

        const emb = try take(self.tok_embeddings, ids_arr, 0, s);
        defer free(emb);
        var x = try layerNorm(emb, self.emb_norm, null, self.cfg.norm_eps, s);
        errdefer free(x);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.cfg.head_dim)));

        for (self.layers) |*l| {
            const mask = if (l.global) full_mask else local_mask;
            const theta = if (l.global) self.cfg.rope_theta_global else self.cfg.rope_theta_local;
            const h = if (l.attn_norm) |w| try layerNorm(x, w, null, self.cfg.norm_eps, s) else x;
            defer if (l.attn_norm != null) free(h);
            const qkv = try linear(h, l.wqkv.w_t, l.wqkv.b, s);
            defer free(qkv);
            const parts = try splitQkv(qkv, N, T, H, Dh, s);
            defer for (parts) |p| free(p);
            const q = try rope(parts[0], Dh, theta, s);
            defer free(q);
            const k = try rope(parts[1], Dh, theta, s);
            defer free(k);
            const att = try sdpa(q, k, parts[2], scale, mask, s);
            defer free(att);
            const merged = try mergeHeads(att, N, T, D, s);
            defer free(merged);
            const proj = try linear(merged, l.wo.w_t, l.wo.b, s);
            defer free(proj);
            const x1 = try add(x, proj, s);
            free(x);
            x = x1;

            const h2 = try layerNorm(x, l.mlp_norm, null, self.cfg.norm_eps, s);
            defer free(h2);
            const wi = try linear(h2, l.wi.w_t, l.wi.b, s);
            defer free(wi);
            var halves = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(halves);
            try mlx.check(mlx.mlx_split(&halves, wi, 2, -1, s));
            var value = mlx.mlx_array_new();
            defer free(value);
            try mlx.check(mlx.mlx_vector_array_get(&value, halves, 0));
            var gate = mlx.mlx_array_new();
            defer free(gate);
            try mlx.check(mlx.mlx_vector_array_get(&gate, halves, 1));
            const g = try gelu(value, s);
            defer free(g);
            var act = mlx.mlx_array_new();
            defer free(act);
            try mlx.check(mlx.mlx_multiply(&act, g, gate, s));
            const down = try linear(act, l.wo2.w_t, l.wo2.b, s);
            defer free(down);
            const x2 = try add(x, down, s);
            free(x);
            x = x2;
        }
        const out = try layerNorm(x, self.final_norm, null, self.cfg.norm_eps, s);
        free(x);
        return out;
    }

    /// Decision head over encoder output `enc` (`[n, T, D]`): type embedding
    /// (`qt_arr` `[n]` int32), head layers with the padding key mask `mask`
    /// (`Masks.full`). With `rows` (`[n * S]` flat row indices into `[n * T]`)
    /// the last layer computes queries, out_proj and FFN for those rows only
    /// (keys and values still cover every row) and the result is `[n, S, D]`;
    /// otherwise `[n, T, D]`. Caller frees.
    pub fn headForward(self: *Model, enc: A, mask: A, qt_arr: A, rows: ?A) !A {
        const s = self.stream;
        const shape = mlx.mlx_array_shape(enc);
        const N = shape[0];
        const T = shape[1];
        const D: c_int = @intCast(self.cfg.hidden_size);
        const heads: c_int = @intCast(@max(1, self.cfg.hidden_size / 64));
        if (@rem(D, heads) != 0) return error.InvalidLayaConfig;
        const dh = @divExact(D, heads);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dh)));

        const te = try take(self.type_emb, qt_arr, 0, s);
        defer free(te);
        const te3 = try reshape(te, &[_]c_int{ N, 1, D }, s);
        defer free(te3);
        var x = try add(enc, te3, s);
        errdefer free(x);

        for (self.head, 0..) |*l, li| {
            const h = try layerNorm(x, l.norm1_w, l.norm1_b, 1e-5, s);
            defer free(h);
            const sel = if (li + 1 == self.head.len) rows else null;
            var q: A = none;
            defer if (q.ctx != null) free(q);
            var kv: [2]A = .{ none, none };
            defer for (kv) |p| if (p.ctx != null) free(p);
            var S_ = T;
            if (sel) |idx| {
                S_ = @divExact(mlx.mlx_array_shape(idx)[0], N);
                const kv_all = try linear(h, self.last_kv.w_t, self.last_kv.b, s);
                defer free(kv_all);
                kv = try splitHeads(2, kv_all, N, T, heads, dh, s);
                const h_flat = try reshape(h, &[_]c_int{ N * T, D }, s);
                defer free(h_flat);
                const hs = try take(h_flat, idx, 0, s);
                defer free(hs);
                const q_rows = try linear(hs, self.last_q.w_t, self.last_q.b, s);
                defer free(q_rows);
                q = (try splitHeads(1, q_rows, N, S_, heads, dh, s))[0];
            } else {
                const qkv = try linear(h, l.in_proj.w_t, l.in_proj.b, s);
                defer free(qkv);
                const parts = try splitQkv(qkv, N, T, heads, dh, s);
                q = parts[0];
                kv = .{ parts[1], parts[2] };
            }
            const att = try sdpa(q, kv[0], kv[1], scale, mask, s);
            defer free(att);
            const merged = try mergeHeads(att, N, S_, D, s);
            defer free(merged);
            const proj = try linear(merged, l.out_proj.w_t, l.out_proj.b, s);
            defer free(proj);
            const res = if (sel) |idx| blk: {
                const x_flat = try reshape(x, &[_]c_int{ N * T, D }, s);
                defer free(x_flat);
                const xs = try take(x_flat, idx, 0, s);
                defer free(xs);
                break :blk try reshape(xs, &[_]c_int{ N, S_, D }, s);
            } else x;
            defer if (sel != null) free(res);
            const x1 = try add(res, proj, s);
            free(x);
            x = x1;

            const h2 = try layerNorm(x, l.norm2_w, l.norm2_b, 1e-5, s);
            defer free(h2);
            const ff = try linear(h2, l.linear1.w_t, l.linear1.b, s);
            defer free(ff);
            const r = try relu(ff, s);
            defer free(r);
            const down = try linear(r, l.linear2.w_t, l.linear2.b, s);
            defer free(down);
            const x2 = try add(x, down, s);
            free(x);
            x = x2;
        }
        return x;
    }

    /// Full forward for one batch (`DecisionModel.__call__`): pads, runs the
    /// compiled graph (lazy graph when compile is off), one eval.
    pub fn forward(self: *Model, batch: Batch) !Output {
        const a = self.allocator;
        if (batch.ids.len == 0) return error.EmptyBatch;
        var in = try Inputs.fromBatch(a, batch, self.cfg.pad_id);
        defer in.deinit();
        const n = batch.ids.len; // rows past `n` are bucket padding
        const t: usize = @intCast(mlx.mlx_array_shape(in.ids)[1]);
        const k_pad: usize = @intCast(mlx.mlx_array_shape(in.marker_pos)[1]);
        const shape = [3]usize{ @intCast(mlx.mlx_array_shape(in.ids)[0]), t, k_pad };

        const outs = try self.runGraph(in, shape, n, t);
        const masked = outs[0];
        defer free(masked);
        const act32 = outs[1];
        defer free(act32);

        const logits = try a.alloc(f32, n * k_pad);
        errdefer a.free(logits);
        const raw = mlx.mlx_array_data_float32(masked) orelse return error.MlxError;
        @memcpy(logits, raw[0 .. n * k_pad]);
        const n_act: usize = self.cfg.n_actions;
        const act_raw = mlx.mlx_array_data_float32(act32) orelse return error.MlxError;
        const act = try a.alloc(f32, n * n_act);
        errdefer a.free(act);
        for (0..n) |i| {
            const row = act_raw[i * n_act .. (i + 1) * n_act];
            const pr = try a.alloc(f64, n_act);
            defer a.free(pr);
            softmaxInto(row, 1.0, pr);
            for (pr, 0..) |v, jj| act[i * n_act + jj] = @floatCast(v);
        }
        return .{ .logits = logits, .k_pad = k_pad, .act = act, .n_actions = n_act };
    }

    /// Build and evaluate the graph for `in`, compiled when `shape` is admitted.
    /// A compiled-path failure (apply or eval) turns compile off for good and
    /// reruns this batch once on the lazy graph; a lazy-graph failure is returned.
    fn runGraph(self: *Model, in: Inputs, shape: [3]usize, n: usize, t: usize) ![2]A {
        if (self.ensureCompiled()) |cls| if (self.admitShape(shape)) {
            const had_error = mlx.errorPending();
            if (self.runGraphOnce(in, cls, n, t)) |outs| return outs else |e| {
                log.warn("[laya] compiled forward failed ({s}); lazy graph from now on\n", .{@errorName(e)});
                self.compile_failed = true;
                _ = mlx.mlx_closure_free(cls);
                self.compiled = null;
                // Handled by the rerun: the latch must not fail the next, unrelated request.
                mlx.dropLatchedErrorUnless(had_error);
            }
        };
        const had_error = mlx.errorPending();
        return self.runGraphOnce(in, null, n, t) catch |e| {
            // The caller gets the error; a latch left set would fail the next request.
            mlx.dropLatchedErrorUnless(had_error);
            return e;
        };
    }

    fn runGraphOnce(self: *Model, in: Inputs, cls: ?mlx.mlx_closure, n: usize, t: usize) ![2]A {
        const t_start = std.Io.Timestamp.now(trace_io, .boot);
        const outs = if (cls) |c| try self.applyCompiled(c, in) else try self.forwardBody(in);
        errdefer for (outs) |o| free(o);
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_vector_array_append_value(ev, outs[0]));
        try mlx.check(mlx.mlx_vector_array_append_value(ev, outs[1]));
        const build_ms = msSince(t_start);
        const t_built = std.Io.Timestamp.now(trace_io, .boot);
        try mlx.check(mlx.mlx_eval(ev));
        log.debug("[laya] forward n={d} t={d} compiled={}: graph {d:.2} ms, eval {d:.2} ms\n", .{ n, t, cls != null, build_ms, msSince(t_built) });
        return outs;
    }

    /// May `shape` run compiled: already traced, or room for one more trace.
    fn admitShape(self: *Model, shape: [3]usize) bool {
        if (self.compiled_shapes.contains(shape)) return true;
        if (self.compiled_shapes.count() >= MAX_COMPILED_SHAPES) return false;
        self.compiled_shapes.put(self.allocator, shape, {}) catch return false;
        if (self.compiled_shapes.count() == MAX_COMPILED_SHAPES) log.info("[laya] {d} compiled input shapes; new shapes run the lazy graph\n", .{MAX_COMPILED_SHAPES});
        return true;
    }

    fn compileEnabled() bool {
        const raw = std.c.getenv("MLX_SERVE_LAYA_COMPILE") orelse return true;
        return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    }

    /// The compiled forward, built on first use. Null: lazy graph
    /// (`MLX_SERVE_LAYA_COMPILE=0`, or compile failed once).
    fn ensureCompiled(self: *Model) ?mlx.mlx_closure {
        if (self.compile_failed or !compileEnabled()) return null;
        if (self.compiled) |c| return c;
        const raw = mlx.mlx_closure_new_func_payload(&forwardClosure, @ptrCast(self), null);
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw, false);
        _ = mlx.mlx_closure_free(raw);
        if (rc != 0 or compiled.ctx == null) {
            self.compile_failed = true;
            log.warn("[laya] mlx_compile failed; lazy graph\n", .{});
            return null;
        }
        self.compiled = compiled;
        log.info("[laya] compiled forward engaged (one trace per input shape)\n", .{});
        return compiled;
    }

    fn forwardClosure(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Model = @ptrCast(@alignCast(payload.?));
        if (mlx.mlx_vector_array_size(input) != 5) return -1;
        var arrs: [5]A = @splat(.{ .ctx = null });
        defer for (&arrs) |*x| {
            if (x.ctx != null) free(x.*);
        };
        for (0..5) |i| {
            arrs[i] = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&arrs[i], input, i) != 0) return -1;
        }
        const out = self.forwardBody(.{ .ids = arrs[0], .valid = arrs[1], .marker_pos = arrs[2], .marker_mask = arrs[3], .qtype = arrs[4] }) catch return -1;
        res.* = mlx.mlx_vector_array_new_data(&out, 2);
        for (out) |o| free(o);
        return 0;
    }

    fn applyCompiled(self: *Model, cls: mlx.mlx_closure, in: Inputs) ![2]A {
        _ = self;
        const in_arr = [_]A{ in.ids, in.valid, in.marker_pos, in.marker_mask, in.qtype };
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, in_arr.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, cls, in_vec));
        if (mlx.mlx_vector_array_size(out_vec) != 2) return error.MlxError;
        var out: [2]A = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        errdefer for (out) |o| free(o);
        try mlx.check(mlx.mlx_vector_array_get(&out[0], out_vec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&out[1], out_vec, 1));
        return out;
    }

    /// The op sequence of `DecisionModel.__call__` on device inputs, stock
    /// ops only (this is what `mlx_compile` traces). Returns the masked
    /// scorer logits `[n, k]` f32 and the raw action logits `[n, n_actions]`
    /// f32; caller frees both.
    pub fn forwardBody(self: *Model, in: Inputs) ![2]A {
        const s = self.stream;
        const shape = mlx.mlx_array_shape(in.ids);
        const N_ = shape[0];
        const T_ = shape[1];
        const K_ = mlx.mlx_array_shape(in.marker_pos)[1];
        const D: c_int = @intCast(self.cfg.hidden_size);

        var m = try self.masks(in.valid);
        defer m.deinit();
        const enc = try self.encode(in.ids, m);
        defer free(enc);

        // Rows the answer reads: CLS (`b * T`) and the markers
        // (`b * T + max(marker_pos, 0)`; padded slots read their CLS row and
        // are masked below).
        const zero = mlx.mlx_array_new_int(0);
        defer free(zero);
        var cls_t = mlx.mlx_array_new();
        defer free(cls_t);
        try mlx.check(mlx.mlx_arange(&cls_t, 0, @floatFromInt(N_ * T_), @floatFromInt(T_), .int32, s));
        const cls_tc = try reshape(cls_t, &[_]c_int{ N_, 1 }, s);
        defer free(cls_tc);
        var mp0 = mlx.mlx_array_new();
        defer free(mp0);
        try mlx.check(mlx.mlx_maximum(&mp0, in.marker_pos, zero, s));
        var mrow = mlx.mlx_array_new();
        defer free(mrow);
        try mlx.check(mlx.mlx_add(&mrow, mp0, cls_tc, s));

        // The last head layer runs on those `1 + K` rows per sequence only;
        // `h` is then `[n, 1 + K, D]` with CLS at 0 and marker j at 1 + j.
        const pick = self.head.len > 0;
        const rows = if (pick) blk: {
            const pair = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(pair);
            try mlx.check(mlx.mlx_vector_array_append_value(pair, cls_tc));
            try mlx.check(mlx.mlx_vector_array_append_value(pair, mrow));
            var cat = mlx.mlx_array_new();
            defer free(cat);
            try mlx.check(mlx.mlx_concatenate_axis(&cat, pair, 1, s));
            break :blk try reshape(cat, &[_]c_int{N_ * (K_ + 1)}, s);
        } else null;
        defer if (rows) |r| free(r);
        const h = try self.headForward(enc, m.full, in.qtype, rows);
        defer free(h);
        const Tv: c_int = if (pick) K_ + 1 else T_;
        const h_flat = try reshape(h, &[_]c_int{ N_ * Tv, D }, s);
        defer free(h_flat);

        var row0 = mlx.mlx_array_new();
        defer free(row0);
        try mlx.check(mlx.mlx_arange(&row0, 0, @floatFromInt(N_ * Tv), @floatFromInt(Tv), .int32, s));
        const cidx_arr = row0;
        const row0c = try reshape(row0, &[_]c_int{ N_, 1 }, s);
        defer free(row0c);
        var midx2 = mlx.mlx_array_new();
        defer free(midx2);
        if (pick) {
            var ar = mlx.mlx_array_new();
            defer free(ar);
            try mlx.check(mlx.mlx_arange(&ar, 1, @floatFromInt(K_ + 1), 1, .int32, s));
            const ar2 = try reshape(ar, &[_]c_int{ 1, K_ }, s);
            defer free(ar2);
            try mlx.check(mlx.mlx_add(&midx2, ar2, row0c, s));
        } else try mlx.check(mlx.mlx_array_set(&midx2, mrow));
        const midx_arr = try reshape(midx2, &[_]c_int{N_ * K_}, s);
        defer free(midx_arr);
        // k = max(marker count, 2) as f32 [n, 1]
        var ksum = mlx.mlx_array_new();
        defer free(ksum);
        try mlx.check(mlx.mlx_sum_axis(&ksum, in.marker_mask, -1, true, s));
        const two = mlx.mlx_array_new_int(2);
        defer free(two);
        var k2 = mlx.mlx_array_new();
        defer free(k2);
        try mlx.check(mlx.mlx_maximum(&k2, ksum, two, s));
        const k_arr = try astype(k2, .float32, s);
        defer free(k_arr);
        const mmask_arr = in.marker_mask;

        const markers = try take(h_flat, midx_arr, 0, s);
        defer free(markers);
        const sn = try layerNorm(markers, self.scorer_norm_w, self.scorer_norm_b, 1e-5, s);
        defer free(sn);
        const s1 = try linear(sn, self.scorer_l1.w_t, self.scorer_l1.b, s);
        defer free(s1);
        const sg = try gelu(s1, s);
        defer free(sg);
        const s3 = try linear(sg, self.scorer_l3.w_t, self.scorer_l3.b, s);
        defer free(s3);
        const logits_arr = try astype(s3, .float32, s);
        defer free(logits_arr);
        const cls = try take(h_flat, cidx_arr, 0, s);
        defer free(cls);

        // Masked logits, confidence features and the action head all stay in
        // the graph (one eval), matching the reference op sequence.
        const lg = try reshape(logits_arr, &[_]c_int{ N_, K_ }, s);
        defer free(lg);
        const neg = mlx.mlx_array_new_float(-1e4);
        defer free(neg);
        var masked = mlx.mlx_array_new();
        defer free(masked);
        try mlx.check(mlx.mlx_where(&masked, mmask_arr, lg, neg, s));
        var p = mlx.mlx_array_new();
        defer free(p);
        try mlx.check(mlx.mlx_softmax_axis(&p, masked, -1, false, s));
        // entropy = -sum(p * log(max(p, 1e-9))) / log(k)
        const floor = mlx.mlx_array_new_float(1e-9);
        defer free(floor);
        var pf = mlx.mlx_array_new();
        defer free(pf);
        try mlx.check(mlx.mlx_maximum(&pf, p, floor, s));
        var lp = mlx.mlx_array_new();
        defer free(lp);
        try mlx.check(mlx.mlx_log(&lp, pf, s));
        var plp = mlx.mlx_array_new();
        defer free(plp);
        try mlx.check(mlx.mlx_multiply(&plp, p, lp, s));
        var ent_sum = mlx.mlx_array_new();
        defer free(ent_sum);
        try mlx.check(mlx.mlx_sum_axis(&ent_sum, plp, -1, true, s));
        var neg_ent = mlx.mlx_array_new();
        defer free(neg_ent);
        try mlx.check(mlx.mlx_negative(&neg_ent, ent_sum, s));
        var logk = mlx.mlx_array_new();
        defer free(logk);
        try mlx.check(mlx.mlx_log(&logk, k_arr, s));
        var ent = mlx.mlx_array_new();
        defer free(ent);
        try mlx.check(mlx.mlx_divide(&ent, neg_ent, logk, s));
        // top-2 via sort
        var sorted = mlx.mlx_array_new();
        defer free(sorted);
        try mlx.check(mlx.mlx_sort_axis(&sorted, p, -1, s));
        var top0 = mlx.mlx_array_new();
        defer free(top0);
        try mlx.check(mlx.mlx_slice(&top0, sorted, &[_]c_int{ 0, K_ - 2 }, 2, &[_]c_int{ N_, K_ - 1 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var top1 = mlx.mlx_array_new();
        defer free(top1);
        try mlx.check(mlx.mlx_slice(&top1, sorted, &[_]c_int{ 0, K_ - 1 }, 2, &[_]c_int{ N_, K_ }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var margin = mlx.mlx_array_new();
        defer free(margin);
        try mlx.check(mlx.mlx_subtract(&margin, top1, top0, s));
        const inv255 = mlx.mlx_array_new_float(1.0 / 255.0);
        defer free(inv255);
        var k255 = mlx.mlx_array_new();
        defer free(k255);
        try mlx.check(mlx.mlx_multiply(&k255, k_arr, inv255, s));
        const feat_parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(feat_parts);
        for ([_]A{ top1, margin, ent, k255 }) |f| try mlx.check(mlx.mlx_vector_array_append_value(feat_parts, f));
        var feats = mlx.mlx_array_new();
        defer free(feats);
        try mlx.check(mlx.mlx_concatenate_axis(&feats, feat_parts, -1, s));
        const feats16 = try astype(feats, .float16, s);
        defer free(feats16);
        const pair = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(pair);
        try mlx.check(mlx.mlx_vector_array_append_value(pair, cls));
        try mlx.check(mlx.mlx_vector_array_append_value(pair, feats16));
        var pooled = mlx.mlx_array_new();
        defer free(pooled);
        try mlx.check(mlx.mlx_concatenate_axis(&pooled, pair, -1, s));
        const a0 = try linear(pooled, self.act_l0.w_t, self.act_l0.b, s);
        defer free(a0);
        const ag = try gelu(a0, s);
        defer free(ag);
        const a2 = try linear(ag, self.act_l2.w_t, self.act_l2.b, s);
        defer free(a2);
        const act32 = try astype(a2, .float32, s);
        errdefer free(act32);
        var masked_out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&masked_out, masked));
        return .{ masked_out, act32 };
    }
};

/// `keep` (bool) as an additive fp16 mask: 0 where true, -inf where false.
fn additiveMask(keep: A, s: S) !A {
    const no_shape = [_]c_int{};
    const zero: f16 = 0;
    const ninf: f16 = -std.math.inf(f16);
    const z = mlx.mlx_array_new_data(&zero, &no_shape, 0, .float16);
    defer free(z);
    const n = mlx.mlx_array_new_data(&ninf, &no_shape, 0, .float16);
    defer free(n);
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_where(&out, keep, z, n, s));
    return out;
}

/// `attention_masks` sliding mask: `[n, 1, T, T]` bool, key within `half`
/// positions of the query and valid; padded queries see every valid key (no
/// all-masked softmax rows).
fn localMask(valid: A, N: c_int, T: c_int, half: c_int, s: S) !A {
    var pos = mlx.mlx_array_new();
    defer free(pos);
    try mlx.check(mlx.mlx_arange(&pos, 0, @floatFromInt(T), 1, .int32, s));
    const pi = try reshape(pos, &[_]c_int{ T, 1 }, s);
    defer free(pi);
    const pj = try reshape(pos, &[_]c_int{ 1, T }, s);
    defer free(pj);
    var diff = mlx.mlx_array_new();
    defer free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, pi, pj, s));
    var dist = mlx.mlx_array_new();
    defer free(dist);
    try mlx.check(mlx.mlx_abs(&dist, diff, s));
    const h = mlx.mlx_array_new_int(half);
    defer free(h);
    var near = mlx.mlx_array_new();
    defer free(near);
    try mlx.check(mlx.mlx_less_equal(&near, dist, h, s));
    const near4 = try reshape(near, &[_]c_int{ 1, 1, T, T }, s);
    defer free(near4);
    const vq = try reshape(valid, &[_]c_int{ N, 1, T, 1 }, s);
    defer free(vq);
    var pad_q = mlx.mlx_array_new();
    defer free(pad_q);
    try mlx.check(mlx.mlx_logical_not(&pad_q, vq, s));
    var q_ok = mlx.mlx_array_new();
    defer free(q_ok);
    try mlx.check(mlx.mlx_logical_or(&q_ok, near4, pad_q, s));
    const vk = try reshape(valid, &[_]c_int{ N, 1, 1, T }, s);
    defer free(vk);
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_logical_and(&out, q_ok, vk, s));
    return out;
}

const trace_io = std.Io.Threaded.global_single_threaded.io();

fn msSince(t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.untilNow(trace_io, .boot).nanoseconds)) / 1e6;
}

/// `softmax(z / scale)` in f64 into `out` (same length as `z`).
fn softmaxInto(z: []const f32, scale: f64, out: []f64) void {
    var m: f64 = -std.math.inf(f64);
    for (z) |v| m = @max(m, @as(f64, v) / scale);
    var sum: f64 = 0;
    for (z, 0..) |v, i| {
        out[i] = @exp(@as(f64, v) / scale - m);
        sum += out[i];
    }
    for (out) |*v| v.* /= sum;
}

/// `confidence_from_probs`: 1 - H(p)/log(k), clipped to [0, 1].
pub fn confidenceFromProbs(p: []const f64, k: usize) f64 {
    if (k < 2) return 1.0;
    var ent: f64 = 0;
    for (p[0..k]) |v| ent -= v * @log(std.math.clamp(v, 1e-12, 1.0));
    return std.math.clamp(1.0 - ent / @log(@as(f64, @floatFromInt(k))), 0.0, 1.0);
}

/// `temp_bucket`: "<type>:<2|3-5|6-10|11+>".
pub fn tempBucket(buf: []u8, qt: QType, k: usize) []const u8 {
    const size: []const u8 = if (k <= 2) "2" else if (k <= 5) "3-5" else if (k <= 10) "6-10" else "11+";
    return std.fmt.bufPrint(buf, "{s}:{s}", .{ qt.name(), size }) catch unreachable;
}

// ── Engine: request JSON in, laya `predict` JSON out ──

/// How `predictMany` splits rows into forwards: row indices sorted by token
/// length, cut into batches (`order[ends[b - 1]..ends[b]]`) of at most
/// `MAX_ROWS` rows and `MAX_TOKENS` tokens as `Model.bucketShape` pads them,
/// where a longer row may add at most `MAX_PAD_GROWTH` padding tokens to the
/// rows already in the batch.
pub const BatchPlan = struct {
    order: []u32,
    ends: []usize,

    pub const MAX_ROWS: usize = 64;
    pub const MAX_PAD_GROWTH: usize = 512;
    pub const MAX_TOKENS: usize = 16 * 1024;

    pub fn init(a: std.mem.Allocator, lens: []const usize) !BatchPlan {
        const order = try a.alloc(u32, lens.len);
        errdefer a.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        var ends: std.ArrayList(usize) = .empty;
        errdefer ends.deinit(a);
        std.mem.sort(u32, order, lens, struct {
            fn lessThan(l: []const usize, x: u32, y: u32) bool {
                return l[x] < l[y] or (l[x] == l[y] and x < y);
            }
        }.lessThan);
        var start: usize = 0;
        var t_max: usize = 0; // padded length of the current batch
        for (order, 0..) |o, i| {
            const n = i - start; // rows already in the batch
            const t = Model.bucketShape(1, lens[o], 0)[1];
            const tokens = Model.bucketShape(n + 1, 0, 0)[0] * @max(t, t_max);
            const growth = n * (@max(t, t_max) - t_max);
            if (n > 0 and (n == MAX_ROWS or tokens > MAX_TOKENS or growth > MAX_PAD_GROWTH)) {
                try ends.append(a, i);
                start = i;
                t_max = t;
            } else t_max = @max(t, t_max);
        }
        if (lens.len > 0) try ends.append(a, lens.len);
        return .{ .order = order, .ends = try ends.toOwnedSlice(a) };
    }

    pub fn deinit(self: *BatchPlan, a: std.mem.Allocator) void {
        a.free(self.order);
        a.free(self.ends);
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    model: Model,
    /// Per-request bounds, checked before any forward: one request runs to
    /// completion on the inference thread, so these bound how long it holds it.
    max_questions: usize = DEFAULT_MAX_QUESTIONS,
    max_input_tokens: usize = DEFAULT_MAX_INPUT_TOKENS,

    pub const DEFAULT_MAX_QUESTIONS: usize = 64;
    pub const DEFAULT_MAX_INPUT_TOKENS: usize = 32 * 1024;

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .model = try Model.load(io, allocator, model_dir, s),
            .max_questions = envLimit("MLX_SERVE_LAYA_MAX_QUESTIONS", DEFAULT_MAX_QUESTIONS),
            .max_input_tokens = envLimit("MLX_SERVE_LAYA_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS),
        };
        return self;
    }

    fn envLimit(name: [*:0]const u8, default: usize) usize {
        const raw = std.c.getenv(name) orelse return default;
        const v = std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        if (v == 0) {
            log.warn("[laya] ignoring {s}={s} (want a positive integer)\n", .{ name, raw });
            return default;
        }
        return v;
    }

    /// 400 text for a limit error, naming the limit in force; null for other errors.
    pub fn limitMessage(self: *const Engine, buf: []u8, err: anyerror) ?[]const u8 {
        return switch (err) {
            error.TooManyQuestions => std.fmt.bufPrint(buf, "too many questions in one request (limit {d}, MLX_SERVE_LAYA_MAX_QUESTIONS)", .{self.max_questions}) catch null,
            error.TooManyInputTokens => std.fmt.bufPrint(buf, "the questions total more than {d} input tokens (MLX_SERVE_LAYA_MAX_INPUT_TOKENS); split them over several requests", .{self.max_input_tokens}) catch null,
            error.TooManyOptions => std.fmt.bufPrint(buf, "a question has more options than fit in the {d}-token input (never more than {d})", .{ self.model.cfg.max_len, maxOptions(&self.model.cfg) }) catch null,
            else => null,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.model.deinit();
        self.allocator.destroy(self);
    }

    /// `questions` validated against this engine's limits (`limitMessage`).
    pub fn parseQuestions(self: *const Engine, a: std.mem.Allocator, questions: std.json.Value) !Questions {
        return Questions.init(a, questions, self.max_questions, maxOptions(&self.model.cfg));
    }

    /// `parseQuestions` + `predict`.
    pub fn predictJson(self: *Engine, a: std.mem.Allocator, model_id: []const u8, state: std.json.Value, questions: std.json.Value) ![]u8 {
        var q = try self.parseQuestions(a, questions);
        defer q.deinit(a);
        return self.predict(a, model_id, state, &q);
    }

    /// Run the reference `predict` for one state and validated questions;
    /// returns the response JSON (caller frees). Validation errors are the
    /// named `error.*` values in `errorMessage` and `limitMessage`.
    pub fn predict(self: *Engine, a: std.mem.Allocator, model_id: []const u8, state: std.json.Value, questions: *const Questions) ![]u8 {
        var jobs = [_]Job{.{ .a = a, .model_id = model_id, .state = state, .questions = questions }};
        self.predictMany(&jobs);
        return jobs[0].result;
    }

    /// One request of `predictMany`: `result` is its response JSON (owned by
    /// `a`) or its own error.
    pub const Job = struct {
        a: std.mem.Allocator,
        model_id: []const u8,
        state: std.json.Value,
        questions: *const Questions,
        result: anyerror![]u8 = error.NotRun,
    };

    /// Answer several requests with shared forwards: the questions of every
    /// job go through one `BatchPlan`. Each job fails on its own (its prompt,
    /// its input-token limit, a non-finite row of its own); a failed forward
    /// fails the jobs that had rows in it.
    pub fn predictMany(self: *Engine, jobs: []Job) void {
        const a = self.allocator;
        const works = a.alloc(Work, jobs.len) catch {
            for (jobs) |*j| j.result = error.OutOfMemory;
            return;
        };
        defer a.free(works);
        for (works, jobs) |*w, *j| {
            w.* = self.prepareJob(j) catch |err| .{ .err = err };
        }
        defer for (works) |*w| w.deinit(a);
        self.runJobs(jobs, works) catch |err| for (works) |*w| {
            if (w.err == null) w.err = err;
        };
        for (jobs, works) |*j, *w| {
            j.result = if (w.err) |err| err else writeAnswers(j.a, j.model_id, j.questions, w);
        }
    }

    /// A job's sequences and answers, in the order of its questions.
    const Work = struct {
        seqs: []Sequence = &.{},
        probs: [][]f64 = &.{},
        acts: []f64 = &.{},
        input_tokens: usize = 0,
        err: ?anyerror = null,

        fn deinit(self: *Work, a: std.mem.Allocator) void {
            for (self.seqs) |*sq| sq.deinit(a);
            a.free(self.seqs);
            for (self.probs) |p| a.free(p);
            a.free(self.probs);
            a.free(self.acts);
        }
    };

    fn prepareJob(self: *Engine, job: *const Job) !Work {
        const a = self.allocator;
        const qs = job.questions.qs;
        const cfg = &self.model.cfg;
        var w: Work = .{};
        // No question: Python never serializes the state and answers `{}`.
        if (qs.len == 0) return w;
        errdefer w.deinit(a);
        const state_text = try renderValue(a, job.state);
        defer a.free(state_text);
        const state_ids = try encodeClean(a, &self.model.tok, null, cfg.mask_token, state_text);
        defer a.free(state_ids);
        w.probs = try a.alloc([]f64, qs.len);
        for (w.probs) |*p| p.* = &.{};
        w.acts = try a.alloc(f64, qs.len);
        const seqs = try a.alloc(Sequence, qs.len);
        var nseq: usize = 0;
        errdefer {
            for (seqs[0..nseq]) |*sq| sq.deinit(a);
            a.free(seqs);
        }
        for (qs, seqs) |*q, *sq| {
            sq.* = try buildSequence(a, &self.model.tok, &self.model.tok_cache, cfg, state_ids, q);
            nseq += 1;
            if (sq.markers.len != q.optionCount()) return error.TooManyOptions;
            w.input_tokens += sq.ids.len;
        }
        if (w.input_tokens > self.max_input_tokens) return error.TooManyInputTokens;
        w.seqs = seqs;
        return w;
    }

    /// Forward every question of the jobs without an error, in length-sorted batches.
    fn runJobs(self: *Engine, jobs: []const Job, works: []Work) !void {
        const a = self.allocator;
        const cfg = &self.model.cfg;
        const Row = struct { job: u32, q: u32 };
        var rows: std.ArrayList(Row) = .empty;
        defer rows.deinit(a);
        var lens: std.ArrayList(usize) = .empty;
        defer lens.deinit(a);
        for (works, 0..) |*w, ji| {
            if (w.err != null) continue;
            for (w.seqs, 0..) |sq, qi| {
                try rows.append(a, .{ .job = @intCast(ji), .q = @intCast(qi) });
                try lens.append(a, sq.ids.len);
            }
        }
        var plan = try BatchPlan.init(a, lens.items);
        defer plan.deinit(a);
        const ids = try a.alloc([]const u32, BatchPlan.MAX_ROWS);
        defer a.free(ids);
        const markers = try a.alloc([]const u32, BatchPlan.MAX_ROWS);
        defer a.free(markers);
        const qtypes = try a.alloc(QType, BatchPlan.MAX_ROWS);
        defer a.free(qtypes);
        var b_start: usize = 0;
        for (plan.ends) |b_end| {
            const batch = plan.order[b_start..b_end];
            defer b_start = b_end;
            for (batch, 0..) |ri, r| {
                const row = rows.items[ri];
                const sq = works[row.job].seqs[row.q];
                ids[r] = sq.ids;
                markers[r] = sq.markers;
                qtypes[r] = jobs[row.job].questions.qs[row.q].t;
            }
            var res = self.model.forward(.{ .ids = ids[0..batch.len], .markers = markers[0..batch.len], .qtype = qtypes[0..batch.len] }) catch |err| {
                if (err == error.OutOfMemory) return err;
                for (batch) |ri| {
                    const w = &works[rows.items[ri].job];
                    if (w.err == null) w.err = err;
                }
                continue;
            };
            defer res.deinit(a);
            for (batch, 0..) |ri, r| {
                const row = rows.items[ri];
                const w = &works[row.job];
                const q = &jobs[row.job].questions.qs[row.q];
                const k = w.seqs[row.q].markers.len;
                const logits = res.logits[r * res.k_pad .. r * res.k_pad + k];
                const act = res.act[r * res.n_actions .. (r + 1) * res.n_actions];
                for (logits) |v| if (!std.math.isFinite(v)) {
                    w.err = error.NonFiniteOutput;
                };
                for (act) |v| if (!std.math.isFinite(v)) {
                    w.err = error.NonFiniteOutput;
                };
                var bucket_buf: [32]u8 = undefined;
                const bucket = tempBucket(&bucket_buf, q.t, k);
                const scale: f64 = cfg.temperature_by_options.get(bucket) orelse cfg.temperature[@intFromEnum(q.t)];
                w.probs[row.q] = try a.alloc(f64, k);
                softmaxInto(logits, scale, w.probs[row.q]);
                w.acts[row.q] = act[0];
            }
        }
    }

    /// Laya's `predict` JSON for one job's answers, in the order of its questions.
    fn writeAnswers(a: std.mem.Allocator, model_id: []const u8, questions: *const Questions, w: *const Work) ![]u8 {
        const qs = questions.qs;
        const qids = questions.ids;
        const probs = w.probs;
        const acts = w.acts;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        try out.appendSlice(a, "{\"model\":");
        try pyJsonString(a, &out, model_id, false);
        try out.appendSlice(a, ",\"answers\":{");
        for (qs, 0..) |*q, i| {
            const p = probs[i];
            const k = p.len;
            const act_prob = acts[i];
            if (i > 0) try out.append(a, ',');
            try wireString(a, &out, qids[i]);
            try out.appendSlice(a, ":{\"type\":\"");
            try out.appendSlice(a, q.t.name());
            try out.appendSlice(a, "\",\"confidence\":");
            const conf = switch (q.t) {
                .noul => @max(p[1], 1.0 - p[1]),
                else => confidenceFromProbs(p, k),
            };
            try appendRounded(a, &out, conf);
            try out.appendSlice(a, ",\"action\":{\"act_probability\":");
            try appendRounded(a, &out, act_prob);
            try out.append(a, '}');
            switch (q.t) {
                .choice => {
                    var best: usize = 0;
                    for (p, 0..) |v, j| if (v > p[best]) {
                        best = j;
                    };
                    try out.appendSlice(a, ",\"choice\":");
                    try wireString(a, &out, q.labels[best]);
                    try out.appendSlice(a, ",\"probabilities\":{");
                    for (q.labels, 0..) |label, j| {
                        if (j > 0) try out.append(a, ',');
                        try wireString(a, &out, label);
                        try out.append(a, ':');
                        try appendRounded(a, &out, p[j]);
                    }
                    try out.append(a, '}');
                },
                .score => {
                    var score: f64 = 0;
                    for (p, 0..) |v, j| score += @as(f64, @floatFromInt(j)) * v;
                    try out.appendSlice(a, ",\"score\":");
                    try appendRounded(a, &out, score);
                    try out.appendSlice(a, ",\"legend\":{");
                    for (q.crit.?.array.items, 0..) |c, j| {
                        if (j > 0) try out.append(a, ',');
                        try out.print(a, "\"{d}\":", .{j});
                        try pyJson(a, &out, c, false);
                    }
                    try out.appendSlice(a, "},\"probabilities\":{");
                    for (p, 0..) |v, j| {
                        if (j > 0) try out.append(a, ',');
                        try out.print(a, "\"{d}\":", .{j});
                        try appendRounded(a, &out, v);
                    }
                    try out.append(a, '}');
                },
                .noul => {
                    try out.appendSlice(a, ",\"noul\":");
                    try appendRounded(a, &out, p[1]);
                },
            }
            try out.append(a, '}');
        }
        try out.print(a, "}},\"usage\":{{\"input_tokens\":{d},\"output_tokens\":0}}}}", .{w.input_tokens});
        return out.toOwnedSlice(a);
    }
};

/// Python `round(x, 4)` rendered as a JSON number.
fn appendRounded(a: std.mem.Allocator, out: *std.ArrayList(u8), x: f64) !void {
    const r = @round(x * 10000.0) / 10000.0;
    try out.print(a, "{d}", .{r});
}

pub fn errorMessage(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.QuestionsNotObject => "'questions' must be an object keyed by question id",
        error.QuestionNotObject => "each question must be an object",
        error.UnknownQuestionType => "question 'type' must be one of choice, score, noul",
        error.MissingInstructions => "question is missing 'instructions'",
        error.BadChoiceCriteria => "choice 'criteria' must be a nonempty object or a list of unique string labels",
        error.BadScoreCriteria => "score 'criteria' must be a nonempty list",
        error.BadNoulCriteria => "noul 'criteria' must be an object with false/true descriptions",
        error.TooManyOptions => "a question has too many options for the token budget",
        error.NonFiniteNumber => "a number in the request is outside the float64 range",
        error.LoneSurrogate => "a text the model reads holds a lone UTF-16 surrogate escape (\\ud800-\\udfff)",
        error.NestingTooDeep => std.fmt.comptimePrint("the request nests arrays/objects more than {d} levels deep", .{MAX_JSON_DEPTH}),
        else => null,
    };
}

// ── Tests ──

const testing = std.testing;

fn testIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

fn testModelDir() ?[]const u8 {
    const p = std.c.getenv("LAYA_TEST_MODEL") orelse return null;
    return std.mem.span(p);
}

fn testFixturesDir() ?[]const u8 {
    const p = std.c.getenv("LAYA_FIXTURES") orelse return null;
    return std.mem.span(p);
}

test "laya: pyJson matches Python json.dumps spacing, escaping and ensure_ascii" {
    const a = testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\{"body": "I was \"charged\"\n twice", "n": 3, "f": 1.5, "ok": true, "z": null, "l": [1, "é"], "w": 2.0}
    , .{});
    defer parsed.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try pyJson(a, &out, parsed.value, false);
    try testing.expectEqualStrings(
        \\{"body": "I was \"charged\"\n twice", "n": 3, "f": 1.5, "ok": true, "z": null, "l": [1, "é"], "w": 2.0}
    , out.items);
    out.clearRetainingCapacity();
    try pyJson(a, &out, parsed.value.object.get("l").?, true);
    try testing.expectEqualStrings("[1, \"\\u00e9\"]", out.items);

    // Numbers as the HTTP handler parses them (text kept): Python repr spelling.
    var nums = try parseRequestJson(a,
        \\[1e-5, 0.0001, 1e16, 1e15, 1.0, -0.0, -0, 1.5e300, 12345678901234567890, 5e-324, 1E5]
    );
    defer nums.deinit();
    out.clearRetainingCapacity();
    try pyJson(a, &out, nums.value, false);
    try testing.expectEqualStrings("[1e-05, 0.0001, 1e+16, 1000000000000000.0, 1.0, -0.0, 0, 1.5e+300, 12345678901234567890, 5e-324, 100000.0]", out.items);
    var inf = try parseRequestJson(a, "{\"x\": 1e999}");
    defer inf.deinit();
    try testing.expectError(error.NonFiniteNumber, renderValue(a, inf.value));
}

test "laya: request JSON reads like Python json.loads (repeated keys, lone surrogates, DEL)" {
    const a = testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);

    var dup = try parseRequestJson(a, "{\"k\": 1, \"j\": 2, \"k\": [3]}");
    defer dup.deinit();
    try pyJson(a, &out, dup.value, false);
    try testing.expectEqualStrings("{\"k\": [3], \"j\": 2}", out.items);

    // Lone surrogates re-emit as escapes; a real NUL, a proper pair and marker-like text do not change.
    var lone = try parseRequestJson(a,
        \\{"\udfff": ["a\ud800b", "\u0000", "\ud83d\ude00", "\uDBFF\ud800", "x\u0000d800", "\\ud800"]}
    );
    defer lone.deinit();
    out.clearRetainingCapacity();
    try pyJson(a, &out, lone.value, true);
    try testing.expectEqualStrings(
        \\{"\udfff": ["a\ud800b", "\u0000", "\ud83d\ude00", "\udbff\ud800", "x\u0000d800", "\\ud800"]}
    , out.items);
    out.clearRetainingCapacity();
    try wireString(a, &out, lone.value.object.keys()[0]);
    try testing.expectEqualStrings("\"\\udfff\"", out.items);

    out.clearRetainingCapacity();
    try pyJsonString(a, &out, "a\x7fb", true);
    try testing.expectEqualStrings("\"a\\u007fb\"", out.items);

    for ([_][]const u8{ "{\"a\": \"\\ud800\"", "[\"\\uZZZZ\"]", "[\"\\ud800\" 1]" }) |bad| {
        if (parseRequestJson(a, bad)) |p| {
            var pp = p;
            pp.deinit();
            return error.TestUnexpectedResult;
        } else |_| {}
    }
}

test "laya: nesting past MAX_JSON_DEPTH is refused before parsing or serializing" {
    const a = testing.allocator;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    try body.appendSlice(a, "{\"state\": ");
    try body.appendNTimes(a, '[', 100_000);
    try body.appendNTimes(a, ']', 100_000);
    try body.append(a, '}');
    try testing.expectError(error.NestingTooDeep, checkJsonDepth(body.items));
    // Exactly MAX_JSON_DEPTH levels pass; brackets inside strings do not count.
    var ok: std.ArrayList(u8) = .empty;
    defer ok.deinit(a);
    try ok.appendNTimes(a, '[', MAX_JSON_DEPTH - 1);
    try ok.appendSlice(a, "[\"[[[[{{{{\\\"[[\"]");
    try ok.appendNTimes(a, ']', MAX_JSON_DEPTH - 1);
    try checkJsonDepth(ok.items);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, ok.items, .{});
    defer parsed.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try pyJson(a, &out, parsed.value, false);
    // A deeper value handed to the serializer directly is refused too.
    try ok.insert(a, 0, '[');
    try ok.append(a, ']');
    try testing.expectError(error.NestingTooDeep, checkJsonDepth(ok.items));
    var deeper = try std.json.parseFromSlice(std.json.Value, a, ok.items, .{});
    defer deeper.deinit();
    try testing.expectError(error.NestingTooDeep, pyJson(a, &out, deeper.value, false));
}

test "laya: option lists past maxOptions are refused before de-duplication" {
    const a = testing.allocator;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    try body.appendSlice(a, "{\"type\": \"choice\", \"instructions\": \"?\", \"criteria\": [");
    for (0..250_000) |i| try body.print(a, "{s}\"l{d}\"", .{ if (i > 0) "," else "", i });
    try body.appendSlice(a, "]}");
    var big = try std.json.parseFromSlice(std.json.Value, a, body.items, .{});
    defer big.deinit();
    try testing.expectError(error.TooManyOptions, Question.fromJson(a, big.value, 1021));
    // Every question type, both choice forms; at the bound duplicates are still found.
    var shapes = try std.json.parseFromSlice(std.json.Value, a,
        \\{"list": {"type": "choice", "instructions": "", "criteria": ["a", "b", "c"]},
        \\ "obj": {"type": "choice", "instructions": "", "criteria": {"a": 1, "b": 2, "c": 3}},
        \\ "score": {"type": "score", "instructions": "", "criteria": [1, 2, 3]},
        \\ "dup": {"type": "choice", "instructions": "", "criteria": ["a", "b", "a"]}}
    , .{});
    defer shapes.deinit();
    for ([_][]const u8{ "list", "obj", "score" }) |name| {
        try testing.expectError(error.TooManyOptions, Question.fromJson(a, shapes.value.object.get(name).?, 2));
        var q = try Question.fromJson(a, shapes.value.object.get(name).?, 3);
        q.deinit(a);
    }
    try testing.expectError(error.BadChoiceCriteria, Question.fromJson(a, shapes.value.object.get("dup").?, 3));
}

test "laya: renderOptions for the three question types" {
    const a = testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\{"department": {"type": "choice", "instructions": "Which team should handle this?",
        \\  "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages", "sales": ""}},
        \\ "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", {"desc": "soon"}, "blocking"]},
        \\ "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
        \\ "labels": {"type": "choice", "instructions": 42, "criteria": ["a", "b"]}}
    , .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    var q1 = try Question.fromJson(a, o.get("department").?, 1021);
    defer q1.deinit(a);
    const r1 = try renderOptions(a, &q1);
    defer {
        for (r1) |s| a.free(s);
        a.free(r1);
    }
    try testing.expectEqual(@as(usize, 3), r1.len);
    try testing.expectEqualStrings("billing: invoices, payments, refunds", r1[0]);
    try testing.expectEqualStrings("sales", r1[2]);

    var q2 = try Question.fromJson(a, o.get("urgency").?, 1021);
    defer q2.deinit(a);
    const r2 = try renderOptions(a, &q2);
    defer {
        for (r2) |s| a.free(s);
        a.free(r2);
    }
    try testing.expectEqualStrings("level 0: not urgent", r2[0]);
    try testing.expectEqualStrings("level 1: {\"desc\": \"soon\"}", r2[1]);

    var q3 = try Question.fromJson(a, o.get("churn").?, 1021);
    defer q3.deinit(a);
    const r3 = try renderOptions(a, &q3);
    defer {
        for (r3) |s| a.free(s);
        a.free(r3);
    }
    try testing.expectEqualStrings("false: no, the statement does not hold", r3[0]);
    try testing.expectEqualStrings("true: yes, the statement holds", r3[1]);

    var q4 = try Question.fromJson(a, o.get("labels").?, 1021);
    defer q4.deinit(a);
    try testing.expectEqualStrings("42", q4.ins);
    try testing.expectEqual(@as(usize, 2), q4.labels.len);

    var dup = try std.json.parseFromSlice(std.json.Value, a,
        \\{"type": "choice", "instructions": "x", "criteria": ["a", "a"]}
    , .{});
    defer dup.deinit();
    try testing.expectError(error.BadChoiceCriteria, Question.fromJson(a, dup.value, 1021));
}

test "laya: calibration helpers" {
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("choice:3-5", tempBucket(&buf, .choice, 3));
    try testing.expectEqualStrings("noul:2", tempBucket(&buf, .noul, 2));
    try testing.expectEqualStrings("score:11+", tempBucket(&buf, .score, 12));
    const p = [_]f64{ 0.0094, 0.1117, 0.8788 };
    try testing.expectApproxEqAbs(@as(f64, 0.6338), confidenceFromProbs(&p, 3), 2e-3);
    try testing.expectEqual(@as(f64, 1.0), confidenceFromProbs(&p, 1));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try appendRounded(testing.allocator, &out, 0.87884);
    try testing.expectEqualStrings("0.8788", out.items);
}

test "laya: batches are sorted by length and cut by rows, tokens and padding growth" {
    const a = testing.allocator;
    // Short and long rows interleaved: two batches, each sorted, short first.
    const lens = [_]usize{ 700, 20, 710, 25, 30, 690 };
    var plan = try BatchPlan.init(a, &lens);
    defer plan.deinit(a);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 4, 5, 0, 2 }, plan.order);
    // Adding a 704-token row to three 32-token rows would pad them by 3 * 672.
    try testing.expectEqualSlices(usize, &.{ 3, 6 }, plan.ends);

    var same: [100]usize = @splat(40);
    var p2 = try BatchPlan.init(a, &same);
    defer p2.deinit(a);
    try testing.expectEqualSlices(usize, &.{ 64, 100 }, p2.ends);
    var long: [40]usize = @splat(1024);
    var p3 = try BatchPlan.init(a, &long);
    defer p3.deinit(a);
    try testing.expectEqualSlices(usize, &.{ 16, 32, 40 }, p3.ends);
    const grow = [_]usize{ 100, 110, 120, 130, 140, 150 };
    var p4 = try BatchPlan.init(a, &grow);
    defer p4.deinit(a);
    try testing.expectEqualSlices(usize, &.{6}, p4.ends);
}

test "laya: the token cache stays bounded" {
    const a = testing.allocator;
    var cache: TokenCache = .{ .allocator = a };
    defer cache.deinit();
    var buf: [16]u8 = undefined;
    for (0..TokenCache.MAX + 10) |i| cache.put(try std.fmt.bufPrint(&buf, "text {d}", .{i}), &.{@intCast(i)});
    try testing.expect(cache.map.count() <= TokenCache.MAX);
    try testing.expectEqualSlices(u32, &.{TokenCache.MAX + 9}, cache.map.get("text 265").?);
    const big: [TokenCache.MAX_TEXT + 1]u8 = @splat('a');
    cache.put(&big, &.{1});
    try testing.expect(cache.map.get(&big) == null);
}

const Fixtures = struct {
    parsed: std.json.Parsed(std.json.Value),
    fn load(a: std.mem.Allocator, dir: []const u8) !Fixtures {
        const path = try std.fmt.allocPrint(a, "{s}/cases.json", .{dir});
        defer a.free(path);
        const text = try readWholeFile(testIo(), a, path);
        defer a.free(text);
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{ .allocate = .alloc_always }) };
    }
    fn cases(self: *const Fixtures) []std.json.Value {
        return self.parsed.value.object.get("cases").?.array.items;
    }
};

fn jsonU32Slice(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const out = try a.alloc(u32, v.array.items.len);
    for (v.array.items, 0..) |x, i| out[i] = @intCast(x.integer);
    return out;
}

/// Minimal `.npy` reader: little-endian float32, C order, any shape.
fn readNpyF32(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const bytes = try readWholeFile(testIo(), a, path);
    defer a.free(bytes);
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const major = bytes[6];
    const header_len: usize = if (major == 1) std.mem.readInt(u16, bytes[8..10], .little) else std.mem.readInt(u32, bytes[8..12], .little);
    const data_start: usize = (if (major == 1) @as(usize, 10) else 12) + header_len;
    const header = bytes[0..data_start];
    if (std.mem.indexOf(u8, header, "<f4") == null) return error.BadNpy;
    const n = (bytes.len - data_start) / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[data_start .. data_start + n * 4]);
    return out;
}

test "laya: prompt construction reproduces laya_mlx token ids and markers" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const tok_dir = try std.fmt.allocPrint(a, "{s}/tokenizer", .{dir});
    defer a.free(tok_dir);
    var tok = try tokenizer_mod.loadTokenizer(testIo(), a, tok_dir);
    defer tok.deinit();
    var cfg = try parseConfig(testIo(), a, dir, &tok);
    defer cfg.deinit();
    try testing.expectEqual(@as(u32, 22), cfg.num_layers);
    try testing.expectEqual(@as(u32, 1024), cfg.max_len);
    try testing.expectEqual(@as(u32, 4), cfg.mask_id);

    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    var cache: TokenCache = .{ .allocator = a };
    defer cache.deinit();
    // Twice through the token cache: the second pass reads every question and option from it.
    for (0..2) |_| for (fx.cases()) |c| {
        const o = c.object;
        const state_text = try renderValue(a, o.get("state").?);
        defer a.free(state_text);
        const state_ids = try encodeClean(a, &tok, null, cfg.mask_token, state_text);
        defer a.free(state_ids);
        var q = try Question.fromJson(a, o.get("question").?, 1021);
        defer q.deinit(a);
        var seq = try buildSequence(a, &tok, &cache, &cfg, state_ids, &q);
        defer seq.deinit(a);
        const want_ids = try jsonU32Slice(a, o.get("ids").?);
        defer a.free(want_ids);
        const want_markers = try jsonU32Slice(a, o.get("markers").?);
        defer a.free(want_markers);
        testing.expectEqualSlices(u32, want_ids, seq.ids) catch |err| {
            std.debug.print("case {s}/{s}: ids differ\n", .{ o.get("lang").?.string, o.get("qid").?.string });
            return err;
        };
        try testing.expectEqualSlices(u32, want_markers, seq.markers);
    };
}

test "laya: encoder and head hidden states match laya_mlx (en/department)" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();
    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    const c = fx.cases()[0].object;
    const ids = try jsonU32Slice(a, c.get("ids").?);
    defer a.free(ids);
    const markers = try jsonU32Slice(a, c.get("markers").?);
    defer a.free(markers);
    const batch = Model.Batch{ .ids = &.{ids}, .markers = &.{markers}, .qtype = &.{.choice} };

    var in = try Model.Inputs.fromBatch(a, batch, model.cfg.pad_id);
    defer in.deinit();
    var m = try model.masks(in.valid);
    defer m.deinit();
    const enc = try model.encode(in.ids, m);
    defer free(enc);
    const enc32 = try astype(enc, .float32, s);
    defer free(enc32);
    try mlx.check(mlx.mlx_array_eval(enc32));
    const want_enc = try readNpyF32(a, fx_dir, "encoder_en_q0.npy");
    defer a.free(want_enc);
    const got_enc = mlx.mlx_array_data_float32(enc32).?[0..want_enc.len];
    var max_abs: f32 = 0;
    var max_ref: f32 = 0;
    for (want_enc, got_enc) |w, g| {
        max_abs = @max(max_abs, @abs(w - g));
        max_ref = @max(max_ref, @abs(w));
    }
    std.debug.print("\n[laya] encoder max|diff| {d:.4} (max|ref| {d:.2})\n", .{ max_abs, max_ref });
    try testing.expect(max_abs < 0.05 * max_ref);

    const h = try model.headForward(enc, m.full, in.qtype, null);
    defer free(h);
    const h32 = try astype(h, .float32, s);
    defer free(h32);
    try mlx.check(mlx.mlx_array_eval(h32));
    const want_h = try readNpyF32(a, fx_dir, "head_en_q0.npy");
    defer a.free(want_h);
    const got_h = mlx.mlx_array_data_float32(h32).?[0..want_h.len];
    var hmax: f32 = 0;
    var href: f32 = 0;
    for (want_h, got_h) |w, g| {
        hmax = @max(hmax, @abs(w - g));
        href = @max(href, @abs(w));
    }
    std.debug.print("[laya] head max|diff| {d:.4} (max|ref| {d:.2})\n", .{ hmax, href });
    try testing.expect(hmax < 0.05 * href);

    // The last layer on the CLS and marker rows only matches those rows of the full head.
    const D = model.cfg.hidden_size;
    const picked = try a.alloc(i32, 1 + markers.len);
    defer a.free(picked);
    picked[0] = 0;
    for (markers, 1..) |mk, j| picked[j] = @intCast(mk);
    const rows = mlx.mlx_array_new_data(picked.ptr, &[_]c_int{@intCast(picked.len)}, 1, .int32);
    defer free(rows);
    const hs = try model.headForward(enc, m.full, in.qtype, rows);
    defer free(hs);
    const hs32 = try astype(hs, .float32, s);
    defer free(hs32);
    try mlx.check(mlx.mlx_array_eval(hs32));
    const got_rows = mlx.mlx_array_data_float32(hs32).?;
    var rmax: f32 = 0;
    for (picked, 0..) |row, j| {
        for (0..D) |d| rmax = @max(rmax, @abs(got_rows[j * D + d] - got_h[@as(usize, @intCast(row)) * D + d]));
    }
    try testing.expect(rmax <= 1e-3 * href);

    var out = try model.forward(batch);
    defer out.deinit(a);
    const want_logits = try readNpyF32(a, fx_dir, "logits_en_q0.npy");
    defer a.free(want_logits);
    for (want_logits, 0..) |w, i| {
        std.debug.print("[laya] logit[{d}] got {d:.4} want {d:.4}\n", .{ i, out.logits[i], w });
        try testing.expectApproxEqAbs(w, out.logits[i], 0.15);
    }
}

test "laya: predict reproduces laya_mlx answers for the en/fr/hi states (tolerance 0.01)" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const engine = try Engine.load(testIo(), a, dir, s);
    defer engine.deinit();
    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    const root = fx.parsed.value.object;
    const questions = root.get("questions").?;
    var states = root.get("states").?.object.iterator();
    var max_diff: f64 = 0;
    while (states.next()) |st| {
        const json = try engine.predictJson(a, "laya-test", st.value_ptr.*, questions);
        defer a.free(json);
        var got = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer got.deinit();
        const answers = got.value.object.get("answers").?.object;
        for (fx.cases()) |c| {
            const o = c.object;
            if (!std.mem.eql(u8, o.get("lang").?.string, st.key_ptr.*)) continue;
            const want = o.get("expected").?.object;
            const have = answers.get(o.get("qid").?.string).?.object;
            var it = want.iterator();
            while (it.next()) |kv| {
                const key = kv.key_ptr.*;
                const wv = kv.value_ptr.*;
                const hv = have.get(key) orelse return error.MissingAnswerField;
                switch (wv) {
                    .string => try testing.expectEqualStrings(wv.string, hv.string),
                    .integer, .float => {
                        const d = @abs(numF64(wv) - numF64(hv));
                        max_diff = @max(max_diff, d);
                        try testing.expect(d <= 0.01);
                    },
                    .object => {
                        var pit = wv.object.iterator();
                        while (pit.next()) |pkv| {
                            const hp = hv.object.get(pkv.key_ptr.*) orelse return error.MissingAnswerField;
                            if (pkv.value_ptr.* == .string) {
                                try testing.expectEqualStrings(pkv.value_ptr.string, hp.string);
                            } else {
                                const d = @abs(numF64(pkv.value_ptr.*) - numF64(hp));
                                max_diff = @max(max_diff, d);
                                try testing.expect(d <= 0.01);
                            }
                        }
                    },
                    else => {},
                }
            }
        }
    }
    std.debug.print("\n[laya] predict parity max|diff| {d:.4}\n", .{max_diff});
}

test "laya: requests answered in one pass match their serial answers; errors stay per request" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const engine = try Engine.load(testIo(), a, dir, s);
    defer engine.deinit();
    // The second request has more options than fit in the prompt.
    var many: std.ArrayList(u8) = .empty;
    defer many.deinit(a);
    try many.appendSlice(a, "[\"o0\"");
    for (1..400) |i| try many.print(a, ",\"o{d}\"", .{i});
    try many.append(a, ']');
    const text = try std.fmt.allocPrint(a,
        \\[{{"state": {{"board": "P . G", "moves": ["up", "left"]}},
        \\  "questions": {{"up": {{"type": "score", "instructions": "How good is up?", "criteria": ["bad", "ok", "good"]}},
        \\                "left": {{"type": "noul", "instructions": "Is left safe?"}}}}}},
        \\ {{"state": "x", "questions": {{"q": {{"type": "choice", "instructions": "?", "criteria": {s}}}}}}},
        \\ {{"state": "I was charged twice for my order and want a refund today",
        \\  "questions": {{"team": {{"type": "choice", "instructions": "Which team?", "criteria": ["billing", "sales", "tech"]}},
        \\                "angry": {{"type": "noul", "instructions": "Is the customer angry?"}}}}}}]
    , .{many.items});
    defer a.free(text);
    var req = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer req.deinit();
    const items = req.value.array.items;
    var qs: [3]Questions = undefined;
    for (items, &qs) |it, *q| q.* = try engine.parseQuestions(a, it.object.get("questions").?);
    defer for (&qs) |*q| q.deinit(a);
    var jobs: [3]Engine.Job = undefined;
    for (items, &qs, &jobs) |it, *q, *j| j.* = .{ .a = a, .model_id = "m", .state = it.object.get("state").?, .questions = q };
    engine.predictMany(&jobs);
    defer for (jobs) |j| if (j.result) |out| a.free(out) else |_| {};
    try testing.expectError(error.TooManyOptions, jobs[1].result);
    for (items, &qs, jobs, 0..) |it, *q, j, i| {
        if (i == 1) continue;
        const serial = try engine.predict(a, "m", it.object.get("state").?, q);
        defer a.free(serial);
        var ps = try std.json.parseFromSlice(std.json.Value, a, serial, .{});
        defer ps.deinit();
        var pm = try std.json.parseFromSlice(std.json.Value, a, try j.result, .{});
        defer pm.deinit();
        try expectJsonClose(ps.value, pm.value, 1e-4);
    }
}

fn expectJsonClose(want: std.json.Value, got: std.json.Value, tol: f64) !void {
    switch (want) {
        .object => |o| {
            try testing.expectEqual(o.count(), got.object.count());
            var it = o.iterator();
            while (it.next()) |e| try expectJsonClose(e.value_ptr.*, got.object.get(e.key_ptr.*) orelse return error.MissingKey, tol);
        },
        .float, .integer => try testing.expectApproxEqAbs(numF64(want), numF64(got), tol),
        .string => |x| try testing.expectEqualStrings(x, got.string),
        else => {},
    }
}

const FixtureRows = struct {
    ids: [18][]const u32,
    markers: [18][]const u32,
    qtype: [18]QType,

    fn load(a: std.mem.Allocator, fx: *const Fixtures) !FixtureRows {
        var r: FixtureRows = undefined;
        for (fx.cases(), 0..) |cv, i| {
            r.ids[i] = try jsonU32Slice(a, cv.object.get("ids").?);
            r.markers[i] = try jsonU32Slice(a, cv.object.get("markers").?);
            r.qtype[i] = @enumFromInt(cv.object.get("qtype").?.integer);
        }
        return r;
    }
    fn deinit(self: *FixtureRows, a: std.mem.Allocator) void {
        for (self.ids, self.markers) |x, m| {
            a.free(x);
            a.free(m);
        }
    }
};

test "laya: dummy rows of a bucketed batch leave the real rows bit-identical" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();
    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    var r = try FixtureRows.load(a, &fx);
    defer r.deinit(a);
    // 9 rows pad to 16 with dummy rows; 16 real rows (7 repeated) have the same shape.
    var ids: [16][]const u32 = undefined;
    var markers: [16][]const u32 = undefined;
    var qtype: [16]QType = undefined;
    for (0..16) |i| {
        ids[i] = r.ids[i % 9];
        markers[i] = r.markers[i % 9];
        qtype[i] = r.qtype[i % 9];
    }
    var nine = try model.forward(.{ .ids = ids[0..9], .markers = markers[0..9], .qtype = qtype[0..9] });
    defer nine.deinit(a);
    var sixteen = try model.forward(.{ .ids = &ids, .markers = &markers, .qtype = &qtype });
    defer sixteen.deinit(a);
    try testing.expectEqual(nine.k_pad, sixteen.k_pad);
    try testing.expectEqualSlices(f32, nine.logits, sixteen.logits[0..nine.logits.len]);
    try testing.expectEqualSlices(f32, nine.act, sixteen.act[0..nine.act.len]);
}

fn countForwardOps(model: *Model, len: usize, n: usize, fx: *const FixtureRows) !u64 {
    var long: [256]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = fx.ids[0][i % fx.ids[0].len];
    const rows: [8][]const u32 = @splat(long[0..len]);
    const markers: [8][]const u32 = @splat(fx.markers[0]);
    const qtype: [8]QType = @splat(.choice);
    const ops0 = mlx.op_count.load(.monotonic);
    var out = try model.forward(.{ .ids = rows[0..n], .markers = markers[0..n], .qtype = qtype[0..n] });
    out.deinit(model.allocator);
    _ = mlx.mlx_clear_cache();
    return mlx.op_count.load(.monotonic) - ops0;
}

test "laya: past the compiled-shape cap a new input shape runs the lazy graph" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();
    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    var r = try FixtureRows.load(a, &fx);
    defer r.deinit(a);
    // 1..8 rows of 32..144 tokens: one trace per shape up to the cap.
    for (0..Model.MAX_COMPILED_SHAPES) |i| _ = try countForwardOps(&model, 32 + 16 * (i / 8), 1 + i % 8, &r);
    if (model.compiled == null) return error.SkipZigTest;
    const replay = try countForwardOps(&model, 32, 1, &r);
    _ = try countForwardOps(&model, 160, 1, &r);
    // A replayed trace issues a handful of ops; the lazy graph issues every op again.
    try testing.expect(try countForwardOps(&model, 160, 1, &r) > 10 * replay);
}

test "laya: question and input-token limits reject a request before any forward" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const engine = try Engine.load(testIo(), a, dir, s);
    defer engine.deinit();
    var qs = try std.json.parseFromSlice(std.json.Value, a,
        \\{"a": {"type": "noul", "instructions": "x?"}, "b": {"type": "noul", "instructions": "y?"}, "c": {"type": "noul", "instructions": "z?"}}
    , .{});
    defer qs.deinit();
    const state = std.json.Value{ .string = "some state" };
    engine.max_questions = 2;
    try testing.expectError(error.TooManyQuestions, engine.predictJson(a, "m", state, qs.value));
    engine.max_questions = 3;
    engine.max_input_tokens = 10;
    try testing.expectError(error.TooManyInputTokens, engine.predictJson(a, "m", state, qs.value));
    var buf: [160]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, engine.limitMessage(&buf, error.TooManyInputTokens).?, "10 input tokens") != null);
    engine.max_input_tokens = Engine.DEFAULT_MAX_INPUT_TOKENS;
    a.free(try engine.predictJson(a, "m", state, qs.value));
}

test "laya: request serialization matches Python json.dumps text and token ids" {
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const path = try std.fmt.allocPrint(a, "{s}/numeric_cases.json", .{fx_dir});
    defer a.free(path);
    const text = try readWholeFile(testIo(), a, path);
    defer a.free(text);
    var fx = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer fx.deinit();
    const root = fx.value.object;

    var tok: ?tokenizer_mod.Tokenizer = null;
    defer if (tok) |*t| t.deinit();
    var cfg: ?Config = null;
    defer if (cfg) |*c| c.deinit();
    if (testModelDir()) |dir| {
        const tok_dir = try std.fmt.allocPrint(a, "{s}/tokenizer", .{dir});
        defer a.free(tok_dir);
        tok = try tokenizer_mod.loadTokenizer(testIo(), a, tok_dir);
        cfg = try parseConfig(testIo(), a, dir, &tok.?);
    }
    var state_q = try Question.fromJson(a, root.get("question").?, 1021);
    defer state_q.deinit(a);

    // `cases`: a state object; `ascii_cases`: non-string instructions over the state "x".
    for ([_][]const u8{ "cases", "ascii_cases" }) |kind| for (root.get(kind).?.array.items) |cv| {
        const c = cv.object;
        const is_state = std.mem.eql(u8, kind, "cases");
        const body = if (is_state) try a.dupe(u8, c.get("json").?.string) else try std.fmt.allocPrint(a, "{{\"type\": \"noul\", \"instructions\": {s}}}", .{c.get("json").?.string});
        defer a.free(body);
        var v = try parseRequestJson(a, body);
        defer v.deinit();
        const state_text = if (is_state) try renderValue(a, v.value) else try a.dupe(u8, "x");
        defer a.free(state_text);
        var q: ?Question = if (is_state) null else try Question.fromJson(a, v.value, 1021);
        defer if (q) |*qq| qq.deinit(a);
        testing.expectEqualStrings(c.get("dumps").?.string, if (is_state) state_text else q.?.ins) catch |e| {
            std.debug.print("{s} {s}\n", .{ kind, c.get("name").?.string });
            return e;
        };
        if (tok == null) continue;
        const state_ids = try encodeClean(a, &tok.?, null, cfg.?.mask_token, state_text);
        defer a.free(state_ids);
        var seq = try buildSequence(a, &tok.?, null, &cfg.?, state_ids, if (q) |*qq| qq else &state_q);
        defer seq.deinit(a);
        const want_ids = try jsonU32Slice(a, c.get("ids").?);
        defer a.free(want_ids);
        const want_markers = try jsonU32Slice(a, c.get("markers").?);
        defer a.free(want_markers);
        try testing.expectEqualSlices(u32, want_ids, seq.ids);
        try testing.expectEqualSlices(u32, want_markers, seq.markers);
    };
    for (root.get("rejected").?.array.items) |r| {
        var state = try parseRequestJson(a, r.string);
        defer state.deinit();
        try testing.expectError(error.NonFiniteNumber, renderValue(a, state.value));
    }
}

test "laya: predict answers {} for no question and refuses lone surrogates the tokenizer would see" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const engine = try Engine.load(testIo(), a, dir, s);
    defer engine.deinit();

    // No question: Python never serializes the state, so even an unrenderable one answers.
    var empty = try parseRequestJson(a, "{\"state\": {\"x\": 1e999}, \"questions\": {}}");
    defer empty.deinit();
    const ej = try engine.predictJson(a, "m", empty.value.object.get("state").?, empty.value.object.get("questions").?);
    defer a.free(ej);
    try testing.expectEqualStrings("{\"model\":\"m\",\"answers\":{},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}", ej);

    var req = try parseRequestJson(a,
        \\{"ok": {"\ud800": {"type": "noul", "instructions": ["\udc00"]}},
        \\ "bad_ins": {"q": {"type": "noul", "instructions": "\udc00"}},
        \\ "bad_state": "a\ud800b"}
    );
    defer req.deinit();
    const o = req.value.object;
    const json = try engine.predictJson(a, "m", .{ .string = "x" }, o.get("ok").?);
    defer a.free(json);
    try testing.expect(std.mem.startsWith(u8, json, "{\"model\":\"m\",\"answers\":{\"\\ud800\":{\"type\":\"noul\""));
    try testing.expectError(error.LoneSurrogate, engine.predictJson(a, "m", .{ .string = "x" }, o.get("bad_ins").?));
    try testing.expectError(error.LoneSurrogate, engine.predictJson(a, "m", o.get("bad_state").?, o.get("ok").?));
}

test "laya: a compiled-path failure reruns the batch on the lazy graph and is never retried" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();
    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    const c = fx.cases()[0].object;
    const ids = try jsonU32Slice(a, c.get("ids").?);
    defer a.free(ids);
    const markers = try jsonU32Slice(a, c.get("markers").?);
    defer a.free(markers);
    const batch = Model.Batch{ .ids = &.{ids}, .markers = &.{markers}, .qtype = &.{.choice} };

    var ref = try model.forward(batch);
    defer ref.deinit(a);
    if (model.compiled == null) return error.SkipZigTest;
    const ops0 = mlx.op_count.load(.monotonic);
    var warm = try model.forward(batch);
    warm.deinit(a);
    const replay_ops = mlx.op_count.load(.monotonic) - ops0;
    // A replayed trace: checked op 1 is the closure apply, the last one the eval.
    for ([_]u64{ 1, replay_ops }) |k| {
        model.compile_failed = false;
        var prime = try model.forward(batch);
        prime.deinit(a);
        mlx.armLatchingFaultForTest(k);
        var out = try model.forward(batch);
        defer out.deinit(a);
        try testing.expect(mlx.latchingFaultFiredForTest());
        try testing.expect(!mlx.errorPending());
        try testing.expect(model.ensureCompiled() == null);
        for (ref.logits, out.logits) |w, g| try testing.expectApproxEqAbs(w, g, 0.05);
    }
    // A failing lazy rerun is returned, not retried, and leaves no latch behind.
    model.compile_failed = false;
    var prime = try model.forward(batch);
    prime.deinit(a);
    mlx.fault.arm(1);
    mlx.armLatchingFaultForTest(2);
    try testing.expectError(error.MlxError, model.forward(batch));
    try testing.expect(mlx.latchingFaultFiredForTest());
    try testing.expect(!mlx.errorPending());
}

fn numF64(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => std.math.nan(f64),
    };
}

test "laya: the loaded model holds each checkpoint tensor once" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Finished work releases its buffers asynchronously; settle before reading.
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var active0: usize = 0;
    _ = mlx.mlx_get_active_memory(&active0);
    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();
    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    const c = fx.cases()[0].object;
    const ids = try jsonU32Slice(a, c.get("ids").?);
    defer a.free(ids);
    const markers = try jsonU32Slice(a, c.get("markers").?);
    defer a.free(markers);
    // Two input shapes: two graphs over the same tensors.
    for ([_][]const u32{ ids, ids[0..32] }) |row| {
        var out = try model.forward(.{ .ids = &.{row}, .markers = &.{markers}, .qtype = &.{.choice} });
        out.deinit(a);
    }
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    var tensor_bytes: usize = 0;
    var it = model.weights.map.valueIterator();
    while (it.next()) |w| tensor_bytes += mlx.mlx_array_size(w.*) * mlx.mlx_array_itemsize(w.*);
    try testing.expect(active -| active0 < tensor_bytes + tensor_bytes / 8);
}

test "laya: forward latency breakdown (LAYA_BENCH=1)" {
    // Not a correctness test: prints the median of 30 for the 3-question en
    // batch, compiled and lazy, plus encoder-only — the numbers REPORT.md quotes.
    if (std.c.getenv("LAYA_BENCH") == null) return error.SkipZigTest;
    const dir = testModelDir() orelse return error.SkipZigTest;
    const fx_dir = testFixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();
    var fx = try Fixtures.load(a, fx_dir);
    defer fx.parsed.deinit();
    var ids_l: std.ArrayList([]const u32) = .empty;
    defer {
        for (ids_l.items) |x| a.free(x);
        ids_l.deinit(a);
    }
    var mk_l: std.ArrayList([]const u32) = .empty;
    defer {
        for (mk_l.items) |x| a.free(x);
        mk_l.deinit(a);
    }
    var qt_l: std.ArrayList(QType) = .empty;
    defer qt_l.deinit(a);
    for (fx.cases()) |cv| {
        const c = cv.object;
        if (!std.mem.eql(u8, c.get("lang").?.string, "en")) continue;
        try ids_l.append(a, try jsonU32Slice(a, c.get("ids").?));
        try mk_l.append(a, try jsonU32Slice(a, c.get("markers").?));
        try qt_l.append(a, @enumFromInt(c.get("qtype").?.integer));
    }
    const batch = Model.Batch{ .ids = ids_l.items, .markers = mk_l.items, .qtype = qt_l.items };

    // LAYA_BENCH_GAP_MS: idle gap between requests (an HTTP client's pacing)
    // — the GPU/CPU clocks ramp down between calls.
    const gap_ms: u64 = if (std.c.getenv("LAYA_BENCH_GAP_MS")) |g| try std.fmt.parseInt(u64, std.mem.sliceTo(g, 0), 10) else 0;
    const gap_ts = std.c.timespec{ .sec = 0, .nsec = @intCast(gap_ms * 1_000_000) };
    var times: [30]f64 = undefined;
    for ([_]bool{ true, false }) |compiled| {
        model.compile_failed = !compiled;
        for (0..5) |_| {
            var o = try model.forward(batch);
            o.deinit(a);
        }
        for (&times) |*t| {
            if (gap_ms > 0) _ = std.c.nanosleep(&gap_ts, null);
            const t0 = std.Io.Timestamp.now(trace_io, .boot);
            var o = try model.forward(batch);
            o.deinit(a);
            t.* = msSince(t0);
        }
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        std.debug.print("\n[laya-bench] forward n={d} compiled={} gap={d}ms: median {d:.2} ms p10 {d:.2} p90 {d:.2}\n", .{ batch.ids.len, compiled, gap_ms, times[15], times[3], times[27] });
    }
    // Encoder only (lazy graph): 22 layers.
    var in = try Model.Inputs.fromBatch(a, batch, model.cfg.pad_id);
    defer in.deinit();
    for (0..5 + 30) |i| {
        const t0 = std.Io.Timestamp.now(trace_io, .boot);
        var m = try model.masks(in.valid);
        defer m.deinit();
        const enc = try model.encode(in.ids, m);
        defer free(enc);
        try mlx.check(mlx.mlx_array_eval(enc));
        if (i >= 5) times[i - 5] = msSince(t0);
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    std.debug.print("[laya-bench] encoder only: median {d:.2} ms p10 {d:.2} p90 {d:.2}\n", .{ times[15], times[3], times[27] });
}

/// A checkpoint dir in `tmp` linking `src`'s tokenizer and weights, with `src`'s two config
/// files each edited by one text substitution (`.{ needle, replacement }`, "" = none).
fn mutatedCheckpoint(a: std.mem.Allocator, tmp: *std.testing.TmpDir, src: []const u8, enc_edit: [2][]const u8, agent_edit: [2][]const u8) ![]u8 {
    const io = testIo();
    var buf: [1024]u8 = undefined;
    const root = try a.dupe(u8, buf[0..try tmp.dir.realPath(io, &buf)]);
    errdefer a.free(root);
    try tmp.dir.createDirPath(io, "encoder");
    for ([_][]const u8{ "tokenizer", "model.safetensors" }) |name| {
        const from = try std.fmt.allocPrint(a, "{s}/{s}", .{ src, name });
        defer a.free(from);
        const to = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, name });
        defer a.free(to);
        try std.Io.Dir.symLinkAbsolute(io, from, to, .{});
    }
    for ([_][]const u8{ "encoder/config.json", "rl_agent_config.json" }, [_][2][]const u8{ enc_edit, agent_edit }) |rel, edit| {
        const from = try std.fmt.allocPrint(a, "{s}/{s}", .{ src, rel });
        defer a.free(from);
        const text = try readWholeFile(io, a, from);
        defer a.free(text);
        if (edit[0].len > 0 and std.mem.indexOf(u8, text, edit[0]) == null) return error.TestEditNotFound;
        const data = if (edit[0].len > 0) try std.mem.replaceOwned(u8, a, text, edit[0], edit[1]) else try a.dupe(u8, text);
        defer a.free(data);
        try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = data });
    }
    return root;
}

test "laya: a config that does not match its weights or the reference fails the load by name" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const none_edit = [2][]const u8{ "", "" };
    const Case = struct { enc: [2][]const u8 = none_edit, agent: [2][]const u8 = none_edit, err: anyerror };
    const cases = [_]Case{
        .{ .enc = .{ "\"global_attn_every_n_layers\": 3", "\"global_attn_every_n_layers\": 0" }, .err = error.InvalidLayaConfig },
        .{ .agent = .{ "\"head_layers\": 2", "\"head_layers\": -1" }, .err = error.InvalidLayaConfig },
        .{ .agent = .{ "\"max_len\": 1024", "\"max_len\": 9000" }, .err = error.InvalidLayaConfig },
        .{ .agent = .{ "\"max_len\": 1024", "\"max_len\": 1024.5" }, .err = error.InvalidLayaConfig },
        .{ .enc = .{ "\"vocab_size\": 256000", "\"vocab_size\": 4" }, .err = error.InvalidLayaConfig },
        .{ .agent = .{ "\"escalate\": 0.5", "\"escalate\": 0.5, \"wait\": 1.0" }, .err = error.WeightShapeMismatch },
        .{ .enc = .{ "\"hidden_size\": 768", "\"hidden_size\": 1536" }, .err = error.WeightShapeMismatch },
        .{ .enc = .{ "\"intermediate_size\": 1152", "\"intermediate_size\": 1024" }, .err = error.WeightShapeMismatch },
        .{ .agent = .{ "\"head_layers\": 2", "\"head_layers\": 1" }, .err = error.UnexpectedWeight },
    };
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for (cases) |c| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const root = try mutatedCheckpoint(a, &tmp, dir, c.enc, c.agent);
        defer a.free(root);
        if (Model.load(testIo(), a, root, s)) |m| {
            var mm = m;
            mm.deinit();
            return error.TestUnexpectedResult;
        } else |e| try testing.expectEqual(c.err, e);
    }
}

test "laya: calibration temperatures are clamped to [0.5, 5] at load" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testIo();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [1024]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    for ([_][]const u8{ "encoder", "tokenizer" }) |name| {
        const from = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
        defer a.free(from);
        const to = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, name });
        defer a.free(to);
        try std.Io.Dir.symLinkAbsolute(io, from, to, .{});
    }
    try tmp.dir.writeFile(io, .{
        .sub_path = "rl_agent_config.json",
        .data =
        \\{"temperature": [0.1, 1.0, 9.0], "temperature_by_options": {"choice:11+": 0.1006, "noul:2": 2.0}}
        ,
    });
    const tok_dir = try std.fmt.allocPrint(a, "{s}/tokenizer", .{dir});
    defer a.free(tok_dir);
    var tok = try tokenizer_mod.loadTokenizer(io, a, tok_dir);
    defer tok.deinit();
    var cfg = try parseConfig(io, a, root, &tok);
    defer cfg.deinit();
    try testing.expectEqual([3]f32{ 0.5, 1.0, 5.0 }, cfg.temperature);
    try testing.expectEqual(@as(f32, 0.5), cfg.temperature_by_options.get("choice:11+").?);
    try testing.expectEqual(@as(f32, 2.0), cfg.temperature_by_options.get("noul:2").?);
}

fn buildSequenceOnce(a: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer, cfg: *const Config, state_ids: []const u32, q: *const Question) !void {
    var seq = try buildSequence(a, tok, null, cfg, state_ids, q);
    seq.deinit(a);
}

test "laya: buildSequence frees everything when any allocation fails" {
    const dir = testModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const tok_dir = try std.fmt.allocPrint(a, "{s}/tokenizer", .{dir});
    defer a.free(tok_dir);
    var tok = try tokenizer_mod.loadTokenizer(testIo(), a, tok_dir);
    defer tok.deinit();
    var cfg = try parseConfig(testIo(), a, dir, &tok);
    defer cfg.deinit();
    var qv = try std.json.parseFromSlice(std.json.Value, a,
        \\{"type": "choice", "instructions": "Which team?", "criteria": {"billing": "refunds", "sales": ""}}
    , .{});
    defer qv.deinit();
    var q = try Question.fromJson(a, qv.value, maxOptions(&cfg));
    defer q.deinit(a);
    const state_ids = try encodeClean(a, &tok, null, cfg.mask_token, "I was charged twice.");
    defer a.free(state_ids);
    // Refuse in-place shrinks so every toOwnedSlice allocates and can fail.
    var no_remap = std.testing.FailingAllocator.init(a, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(no_remap.allocator(), buildSequenceOnce, .{ &tok, &cfg, state_ids, &q });
}
