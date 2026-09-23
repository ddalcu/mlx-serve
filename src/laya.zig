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

fn jsonInt(v: ?std.json.Value, default: u32) u32 {
    const x = v orelse return default;
    return switch (x) {
        .integer => |i| @intCast(i),
        .float => |f| @intFromFloat(f),
        else => default,
    };
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
    const hidden = jsonInt(e.get("hidden_size"), 0);
    const heads = jsonInt(e.get("num_attention_heads"), 0);
    const layers = jsonInt(e.get("num_hidden_layers"), 0);
    if (hidden == 0 or heads == 0 or layers == 0 or hidden % heads != 0 or (hidden / heads) % 2 != 0) return error.InvalidLayaConfig;

    const layer_global = try a.alloc(bool, layers);
    errdefer a.free(layer_global);
    const every_n = jsonInt(e.get("global_attn_every_n_layers"), 3);
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

    const n_actions: u32 = if (g.get("act_costs")) |ac| (if (ac == .object) @as(u32, @intCast(ac.object.count())) + 1 else 1) else 1;
    const max_len = jsonInt(g.get("max_len"), 512);
    const head_max_len = jsonInt(g.get("head_max_len"), 192);
    if (!(4 < head_max_len and head_max_len < max_len)) return error.InvalidLayaConfig;

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

    return .{
        .vocab_size = jsonInt(e.get("vocab_size"), 0),
        .hidden_size = hidden,
        .intermediate_size = jsonInt(e.get("intermediate_size"), 0),
        .num_layers = layers,
        .num_heads = heads,
        .head_dim = hidden / heads,
        .norm_eps = jsonF32(e.get("norm_eps"), jsonF32(e.get("layer_norm_eps"), 1e-5)),
        .local_attention = jsonInt(e.get("local_attention"), 128),
        .layer_global = layer_global,
        .rope_theta_global = theta_global,
        .rope_theta_local = theta_local,
        .head_layers = jsonInt(g.get("head_layers"), 2),
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

// ── Prompt construction (laya_mlx.common.build_sequence) ──

/// Python `json.dumps(v, separators=(", ", ": "))` — key order, spacing and
/// escaping must match because the result is TOKENIZED. `ascii` mirrors
/// `ensure_ascii`. Ceiling: floats print as shortest decimal, never in
/// Python's exponent form (|x| >= 1e16 or < 1e-4); ints/strings are exact.
pub fn pyJson(a: std.mem.Allocator, out: *std.ArrayList(u8), v: std.json.Value, ascii: bool) !void {
    switch (v) {
        .null => try out.appendSlice(a, "null"),
        .bool => |b| try out.appendSlice(a, if (b) "true" else "false"),
        .integer => |i| try out.print(a, "{d}", .{i}),
        .float => |f| {
            if (f == @trunc(f) and @abs(f) < 1e16) {
                try out.print(a, "{d:.1}", .{f});
            } else {
                try out.print(a, "{d}", .{f});
            }
        },
        .number_string => |s| try out.appendSlice(a, s),
        .string => |s| try pyJsonString(a, out, s, ascii),
        .array => |arr| {
            try out.append(a, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try out.appendSlice(a, ", ");
                try pyJson(a, out, item, ascii);
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
                try pyJson(a, out, kv.value_ptr.*, ascii);
            }
            try out.append(a, '}');
        },
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
                const cp = std.unicode.utf8Decode(s[i..end]) catch 0xFFFD;
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
            else => try out.append(a, c),
        }
    }
    try out.append(a, '"');
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

    pub fn fromJson(a: std.mem.Allocator, def: std.json.Value) !Question {
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
                        const ls = try a.alloc([]const u8, arr.items.len);
                        errdefer a.free(ls);
                        for (arr.items, 0..) |v, i| {
                            if (v != .string) return error.BadChoiceCriteria;
                            for (ls[0..i]) |prev| if (std.mem.eql(u8, prev, v.string)) return error.BadChoiceCriteria;
                            ls[i] = v.string;
                        }
                        labels = ls;
                    },
                    .object => |obj| {
                        if (obj.count() == 0) return error.BadChoiceCriteria;
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
/// no specials added.
fn encodeClean(a: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer, mask_token: []const u8, text: []const u8) ![]u32 {
    if (std.mem.indexOf(u8, text, mask_token) == null) return tok.encode(a, text);
    const cleaned = try std.mem.replaceOwned(u8, a, text, mask_token, " ");
    defer a.free(cleaned);
    return tok.encode(a, cleaned);
}

pub const Sequence = struct {
    ids: []u32,
    markers: []u32,
    pub fn deinit(self: *Sequence, a: std.mem.Allocator) void {
        a.free(self.ids);
        a.free(self.markers);
    }
};

/// `build_sequence`: [CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP],
/// with the head capped at `head_max_len` and the state filling `max_len`.
pub fn buildSequence(a: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer, cfg: *const Config, state_text: []const u8, q: *const Question) !Sequence {
    const opts = try renderOptions(a, q);
    defer {
        for (opts) |o| a.free(o);
        a.free(opts);
    }
    const head_text = try std.fmt.allocPrint(a, "{s} question: {s}", .{ q.t.name(), q.ins });
    defer a.free(head_text);
    var head_ids = try encodeClean(a, tok, cfg.mask_token, head_text);
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
        const enc = try encodeClean(a, tok, cfg.mask_token, spaced);
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
    const st = try encodeClean(a, tok, cfg.mask_token, state_text);
    defer a.free(st);
    try ids.appendSlice(a, st[0..@min(st.len, room)]);
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
    return .{ .ids = try ids.toOwnedSlice(a), .markers = try markers.toOwnedSlice(a) };
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
    const r = try reshape(qkv, &[_]c_int{ n, t, 3, heads, head_dim }, s);
    defer free(r);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_split(&parts, r, 3, 2, s));
    var out: [3]A = .{ none, none, none };
    errdefer for (out) |o| if (o.ctx != null) free(o);
    for (0..3) |i| {
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
    /// Transpose views (`Linear.w_t`) not owned by `weights`.
    owned: std.ArrayList(A),
    /// `forwardBody` wrapped by `mlx_compile`: traced once per input shape,
    /// then replayed with fused elementwise kernels (the reference wraps its
    /// model in `mx.compile` the same way). Null: lazy graph (compile off or
    /// failed).
    compiled: ?mlx.mlx_closure = null,
    compile_failed: bool = false,

    fn getW(self: *Model, comptime fmt: []const u8, args: anytype) !A {
        var buf: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, fmt, args);
        return self.weights.get(key) orelse {
            log.err("[laya] missing weight {s}\n", .{key});
            return error.MissingWeight;
        };
    }

    fn optW(self: *Model, comptime fmt: []const u8, args: anytype) ?A {
        var buf: [160]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, fmt, args) catch return null;
        return self.weights.get(key);
    }

    /// `w^T` as a view: matmul reads the transposed strides, so a contiguous
    /// copy would hold every linear weight twice for the same output.
    fn linearW(self: *Model, comptime prefix: []const u8, args: anytype) !Linear {
        const w = try self.getW(prefix ++ ".weight", args);
        var w_t = mlx.mlx_array_new();
        errdefer free(w_t);
        try mlx.check(mlx.mlx_transpose(&w_t, w, self.stream));
        try self.owned.append(self.allocator, w_t);
        return .{ .w_t = w_t, .b = self.optW(prefix ++ ".bias", args) };
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
            .owned = .empty,
        };
        errdefer self.freeOwned();

        self.tok_embeddings = try self.getW("encoder.embeddings.tok_embeddings.weight", .{});
        self.emb_norm = try self.getW("encoder.embeddings.norm.weight", .{});
        self.final_norm = try self.getW("encoder.final_norm.weight", .{});
        self.type_emb = try self.getW("type_emb.weight", .{});

        self.layers = try allocator.alloc(EncoderLayer, cfg.num_layers);
        for (self.layers, 0..) |*l, i| {
            l.* = .{
                .attn_norm = if (i == 0) null else try self.getW("encoder.layers.{d}.attn_norm.weight", .{i}),
                .wqkv = try self.linearW("encoder.layers.{d}.attn.Wqkv", .{i}),
                .wo = try self.linearW("encoder.layers.{d}.attn.Wo", .{i}),
                .mlp_norm = try self.getW("encoder.layers.{d}.mlp_norm.weight", .{i}),
                .wi = try self.linearW("encoder.layers.{d}.mlp.Wi", .{i}),
                .wo2 = try self.linearW("encoder.layers.{d}.mlp.Wo", .{i}),
                .global = cfg.layer_global[i],
            };
        }
        self.head = try allocator.alloc(HeadLayer, cfg.head_layers);
        for (self.head, 0..) |*h, i| {
            h.* = .{
                .norm1_w = try self.getW("head.layers.{d}.norm1.weight", .{i}),
                .norm1_b = try self.getW("head.layers.{d}.norm1.bias", .{i}),
                .in_proj = try self.linearW("head.layers.{d}.self_attn.in_proj", .{i}),
                .out_proj = try self.linearW("head.layers.{d}.self_attn.out_proj", .{i}),
                .norm2_w = try self.getW("head.layers.{d}.norm2.weight", .{i}),
                .norm2_b = try self.getW("head.layers.{d}.norm2.bias", .{i}),
                .linear1 = try self.linearW("head.layers.{d}.linear1", .{i}),
                .linear2 = try self.linearW("head.layers.{d}.linear2", .{i}),
            };
        }
        self.scorer_norm_w = try self.getW("scorer.layers.0.weight", .{});
        self.scorer_norm_b = try self.getW("scorer.layers.0.bias", .{});
        self.scorer_l1 = try self.linearW("scorer.layers.1", .{});
        self.scorer_l3 = try self.linearW("scorer.layers.3", .{});
        self.act_l0 = try self.linearW("act_head.layers.0", .{});
        self.act_l2 = try self.linearW("act_head.layers.2", .{});

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
        for (self.owned.items) |w| free(w);
        self.owned.deinit(self.allocator);
        if (self.layers.len > 0) self.allocator.free(self.layers);
        if (self.head.len > 0) self.allocator.free(self.head);
    }

    pub fn deinit(self: *Model) void {
        if (self.compiled) |c| _ = mlx.mlx_closure_free(c);
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

    /// Sequence length padding granularity: bounds the number of distinct
    /// shapes the compiled forward has to trace (one trace per `[n, t, k]`).
    pub const PAD_T: usize = 16;

    /// The five device inputs of `DecisionModel.__call__` (`input_ids`,
    /// `attention_mask`, `marker_pos`, `marker_mask`, `qtype`).
    pub const Inputs = struct {
        ids: A, // [n, t] int32
        valid: A, // [n, t] bool
        marker_pos: A, // [n, k] int32 (padded slots 0)
        marker_mask: A, // [n, k] bool
        qtype: A, // [n] int32

        pub fn fromBatch(a: std.mem.Allocator, batch: Batch, pad_id: u32) !Inputs {
            const n = batch.ids.len;
            var t: usize = 0;
            var k: usize = 2;
            for (batch.ids, batch.markers) |row, m| {
                t = @max(t, row.len);
                k = @max(k, m.len);
            }
            t = (t + PAD_T - 1) / PAD_T * PAD_T;
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
                for (0..t) |j| {
                    const in_row = j < batch.ids[i].len;
                    ids[i * t + j] = if (in_row) @intCast(batch.ids[i][j]) else @intCast(pad_id);
                    valid[i * t + j] = in_row;
                }
                for (0..k) |j| {
                    const in_row = j < batch.markers[i].len;
                    mpos[i * k + j] = if (in_row) @intCast(batch.markers[i][j]) else 0;
                    mmask[i * k + j] = in_row;
                }
                qt[i] = @intFromEnum(batch.qtype[i]);
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

    /// Encoder only: `[n, T, D]` fp16 (final norm applied) from `ids` `[n, T]`
    /// int32 and `valid` `[n, T]` bool. Caller frees.
    pub fn encode(self: *Model, ids_arr: A, valid: A) !A {
        const s = self.stream;
        const D: c_int = @intCast(self.cfg.hidden_size);
        const H: c_int = @intCast(self.cfg.num_heads);
        const Dh: c_int = @intCast(self.cfg.head_dim);
        const shape = mlx.mlx_array_shape(ids_arr);
        const N = shape[0];
        const T = shape[1];

        const full_mask = try reshape(valid, &[_]c_int{ N, 1, 1, T }, s);
        defer free(full_mask);
        const local_mask = try localMask(valid, N, T, @intCast(self.cfg.local_attention / 2), s);
        defer free(local_mask);

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
    /// (`qt_arr` `[n]` int32), head layers with the padding key mask (`valid`
    /// `[n, T]` bool). Caller frees.
    pub fn headForward(self: *Model, enc: A, valid: A, qt_arr: A) !A {
        const s = self.stream;
        const shape = mlx.mlx_array_shape(enc);
        const N = shape[0];
        const T = shape[1];
        const D: c_int = @intCast(self.cfg.hidden_size);
        const heads: c_int = @intCast(@max(1, self.cfg.hidden_size / 64));
        if (@rem(D, heads) != 0) return error.InvalidLayaConfig;
        const dh = @divExact(D, heads);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dh)));

        const mask = try reshape(valid, &[_]c_int{ N, 1, 1, T }, s);
        defer free(mask);

        const te = try take(self.type_emb, qt_arr, 0, s);
        defer free(te);
        const te3 = try reshape(te, &[_]c_int{ N, 1, D }, s);
        defer free(te3);
        var x = try add(enc, te3, s);
        errdefer free(x);

        for (self.head) |*l| {
            const h = try layerNorm(x, l.norm1_w, l.norm1_b, 1e-5, s);
            defer free(h);
            const qkv = try linear(h, l.in_proj.w_t, l.in_proj.b, s);
            defer free(qkv);
            const parts = try splitQkv(qkv, N, T, heads, dh, s);
            defer for (parts) |p| free(p);
            const att = try sdpa(parts[0], parts[1], parts[2], scale, mask, s);
            defer free(att);
            const merged = try mergeHeads(att, N, T, D, s);
            defer free(merged);
            const proj = try linear(merged, l.out_proj.w_t, l.out_proj.b, s);
            defer free(proj);
            const x1 = try add(x, proj, s);
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
        const n: usize = @intCast(mlx.mlx_array_shape(in.ids)[0]);
        const t: usize = @intCast(mlx.mlx_array_shape(in.ids)[1]);
        const k_pad: usize = @intCast(mlx.mlx_array_shape(in.marker_pos)[1]);

        const t_start = std.Io.Timestamp.now(trace_io, .boot);
        const outs = if (self.ensureCompiled()) |cls| self.applyCompiled(cls, in) catch |e| blk: {
            log.warn("[laya] compiled forward failed ({s}); lazy graph from now on\n", .{@errorName(e)});
            self.compile_failed = true;
            break :blk try self.forwardBody(in);
        } else try self.forwardBody(in);
        const masked = outs[0];
        defer free(masked);
        const act32 = outs[1];
        defer free(act32);
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_vector_array_append_value(ev, masked));
        try mlx.check(mlx.mlx_vector_array_append_value(ev, act32));
        const build_ms = msSince(t_start);
        const t_built = std.Io.Timestamp.now(trace_io, .boot);
        try mlx.check(mlx.mlx_eval(ev));
        log.debug("[laya] forward n={d} t={d}: graph {d:.2} ms, eval {d:.2} ms\n", .{ n, t, build_ms, msSince(t_built) });

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

    fn compileEnabled() bool {
        const raw = std.c.getenv("MLX_SERVE_LAYA_COMPILE") orelse return true;
        return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    }

    /// The compiled forward, built on first use. Null: lazy graph
    /// (`MLX_SERVE_LAYA_COMPILE=0`, or compile failed once).
    fn ensureCompiled(self: *Model) ?mlx.mlx_closure {
        if (self.compiled) |c| return c;
        if (self.compile_failed or !compileEnabled()) return null;
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

        const enc = try self.encode(in.ids, in.valid);
        defer free(enc);
        const h = try self.headForward(enc, in.valid, in.qtype);
        defer free(h);
        const h_flat = try reshape(h, &[_]c_int{ N_ * T_, D }, s);
        defer free(h_flat);

        // Marker rows (`h[b, max(marker_pos, 0)]`; padded slots gather row 0
        // of their sequence and are masked below) and the CLS rows.
        var row0 = mlx.mlx_array_new();
        defer free(row0);
        try mlx.check(mlx.mlx_arange(&row0, 0, @floatFromInt(N_ * T_), @floatFromInt(T_), .int32, s));
        const cidx_arr = row0;
        const row0c = try reshape(row0, &[_]c_int{ N_, 1 }, s);
        defer free(row0c);
        const zero = mlx.mlx_array_new_int(0);
        defer free(zero);
        var mp0 = mlx.mlx_array_new();
        defer free(mp0);
        try mlx.check(mlx.mlx_maximum(&mp0, in.marker_pos, zero, s));
        var midx2 = mlx.mlx_array_new();
        defer free(midx2);
        try mlx.check(mlx.mlx_add(&midx2, mp0, row0c, s));
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

pub const Engine = struct {
    allocator: std.mem.Allocator,
    model: Model,

    pub const BATCH_SIZE: usize = 16;

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.model = try Model.load(io, allocator, model_dir, s);
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.model.deinit();
        self.allocator.destroy(self);
    }

    /// Run the reference `predict` for one state and a `questions` object;
    /// returns the response JSON (caller frees). Validation errors are the
    /// named `error.*` values in `errorMessage`.
    pub fn predictJson(self: *Engine, a: std.mem.Allocator, model_id: []const u8, state: std.json.Value, questions: std.json.Value) ![]u8 {
        if (questions != .object) return error.QuestionsNotObject;
        const qobj = questions.object;
        const nq = qobj.count();
        if (nq == 0) return error.NoQuestions;
        const cfg = &self.model.cfg;

        const state_text = try renderValue(a, state);
        defer a.free(state_text);

        const qs = try a.alloc(Question, nq);
        var nqs: usize = 0;
        defer {
            for (qs[0..nqs]) |*q| q.deinit(a);
            a.free(qs);
        }
        const seqs = try a.alloc(Sequence, nq);
        var nseq: usize = 0;
        defer {
            for (seqs[0..nseq]) |*sq| sq.deinit(a);
            a.free(seqs);
        }
        const qids = try a.alloc([]const u8, nq);
        defer a.free(qids);
        const t_prep = std.Io.Timestamp.now(trace_io, .boot);
        var it = qobj.iterator();
        var qi: usize = 0;
        while (it.next()) |kv| : (qi += 1) {
            qids[qi] = kv.key_ptr.*;
            qs[qi] = try Question.fromJson(a, kv.value_ptr.*);
            nqs += 1;
            seqs[qi] = try buildSequence(a, &self.model.tok, cfg, state_text, &qs[qi]);
            nseq += 1;
            if (seqs[qi].markers.len != qs[qi].optionCount()) return error.TooManyOptions;
        }

        log.debug("[laya] prepare {d} question(s): {d:.2} ms\n", .{ nq, msSince(t_prep) });
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        try out.appendSlice(a, "{\"model\":");
        try pyJsonString(a, &out, model_id, false);
        try out.appendSlice(a, ",\"answers\":{");
        var input_tokens: usize = 0;

        var start: usize = 0;
        while (start < nq) : (start += BATCH_SIZE) {
            const end = @min(nq, start + BATCH_SIZE);
            const ids = try a.alloc([]const u32, end - start);
            defer a.free(ids);
            const markers = try a.alloc([]const u32, end - start);
            defer a.free(markers);
            const qtypes = try a.alloc(QType, end - start);
            defer a.free(qtypes);
            for (start..end) |i| {
                ids[i - start] = seqs[i].ids;
                markers[i - start] = seqs[i].markers;
                qtypes[i - start] = qs[i].t;
                input_tokens += seqs[i].ids.len;
            }
            var res = try self.model.forward(.{ .ids = ids, .markers = markers, .qtype = qtypes });
            defer res.deinit(a);
            for (res.logits) |v| if (!std.math.isFinite(v)) return error.NonFiniteOutput;
            for (res.act) |v| if (!std.math.isFinite(v)) return error.NonFiniteOutput;

            for (start..end) |i| {
                const row = i - start;
                const q = &qs[i];
                const k = seqs[i].markers.len;
                var bucket_buf: [32]u8 = undefined;
                const bucket = tempBucket(&bucket_buf, q.t, k);
                const scale: f64 = cfg.temperature_by_options.get(bucket) orelse cfg.temperature[@intFromEnum(q.t)];
                const p = try a.alloc(f64, k);
                defer a.free(p);
                softmaxInto(res.logits[row * res.k_pad .. row * res.k_pad + k], @max(1e-3, scale), p);
                const act_prob = res.act[row * res.n_actions];
                if (i > 0) try out.append(a, ',');
                try pyJsonString(a, &out, qids[i], false);
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
                        try pyJsonString(a, &out, q.labels[best], false);
                        try out.appendSlice(a, ",\"probabilities\":{");
                        for (q.labels, 0..) |label, j| {
                            if (j > 0) try out.append(a, ',');
                            try pyJsonString(a, &out, label, false);
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
        }
        try out.print(a, "}},\"usage\":{{\"input_tokens\":{d},\"output_tokens\":0}}}}", .{input_tokens});
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
        error.NoQuestions => "'questions' is empty",
        error.QuestionNotObject => "each question must be an object",
        error.UnknownQuestionType => "question 'type' must be one of choice, score, noul",
        error.MissingInstructions => "question is missing 'instructions'",
        error.BadChoiceCriteria => "choice 'criteria' must be a nonempty object or a list of unique string labels",
        error.BadScoreCriteria => "score 'criteria' must be a nonempty list",
        error.BadNoulCriteria => "noul 'criteria' must be an object with false/true descriptions",
        error.TooManyOptions => "a question has too many options for the token budget",
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

    var q1 = try Question.fromJson(a, o.get("department").?);
    defer q1.deinit(a);
    const r1 = try renderOptions(a, &q1);
    defer {
        for (r1) |s| a.free(s);
        a.free(r1);
    }
    try testing.expectEqual(@as(usize, 3), r1.len);
    try testing.expectEqualStrings("billing: invoices, payments, refunds", r1[0]);
    try testing.expectEqualStrings("sales", r1[2]);

    var q2 = try Question.fromJson(a, o.get("urgency").?);
    defer q2.deinit(a);
    const r2 = try renderOptions(a, &q2);
    defer {
        for (r2) |s| a.free(s);
        a.free(r2);
    }
    try testing.expectEqualStrings("level 0: not urgent", r2[0]);
    try testing.expectEqualStrings("level 1: {\"desc\": \"soon\"}", r2[1]);

    var q3 = try Question.fromJson(a, o.get("churn").?);
    defer q3.deinit(a);
    const r3 = try renderOptions(a, &q3);
    defer {
        for (r3) |s| a.free(s);
        a.free(r3);
    }
    try testing.expectEqualStrings("false: no, the statement does not hold", r3[0]);
    try testing.expectEqualStrings("true: yes, the statement holds", r3[1]);

    var q4 = try Question.fromJson(a, o.get("labels").?);
    defer q4.deinit(a);
    try testing.expectEqualStrings("42", q4.ins);
    try testing.expectEqual(@as(usize, 2), q4.labels.len);

    var dup = try std.json.parseFromSlice(std.json.Value, a,
        \\{"type": "choice", "instructions": "x", "criteria": ["a", "a"]}
    , .{});
    defer dup.deinit();
    try testing.expectError(error.BadChoiceCriteria, Question.fromJson(a, dup.value));
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
    for (fx.cases()) |c| {
        const o = c.object;
        const state_text = try renderValue(a, o.get("state").?);
        defer a.free(state_text);
        var q = try Question.fromJson(a, o.get("question").?);
        defer q.deinit(a);
        var seq = try buildSequence(a, &tok, &cfg, state_text, &q);
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
    }
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
    const enc = try model.encode(in.ids, in.valid);
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

    const h = try model.headForward(enc, in.valid, in.qtype);
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
        const enc = try model.encode(in.ids, in.valid);
        defer free(enc);
        try mlx.check(mlx.mlx_array_eval(enc));
        if (i >= 5) times[i - 5] = msSince(t0);
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    std.debug.print("[laya-bench] encoder only: median {d:.2} ms p10 {d:.2} p90 {d:.2}\n", .{ times[15], times[3], times[27] });
}
