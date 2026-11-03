//! LiquidAI D1 typed decisions (LiquidAI/d1-3B): an LFM2.5-VL-3B trunk whose answer is a softmax over option
//! tokens at the answer slot of its ordinary LM head. The prompt, option codes and readout mirror the card's own
//! package (`prompt.py`, `api.py`); the state runs once and each question restores its caches (kev.scoreBranches).
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const laya = @import("../laya/laya.zig");
const kev = @import("../kev/kev.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const lfm2 = @import("../lfm2/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const params = @import("../../sdk/quant/params.zig");
const sushi_pack = @import("../../sdk/quant/exl3/sushi_pack.zig");
const qmatmulBits = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const array = @import("../../mlx/array.zig");
const testing = std.testing;

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const free = laya.free;

const BOS = "<|startoftext|>";
const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";
const MAX_OPTIONS = 255;
const MAX_SCORE_LEVELS = 10;

/// prompt.py's `_FALLBACK_POOL`: the codes tried, in order, once a label's own code is not one token.
const POOL_LEN = 26 + 100 + 26 + 200 + 26 * 26;
const FALLBACK_POOL: [POOL_LEN][]const u8 = blk: {
    @setEvalBranchQuota(100_000);
    var pool: [POOL_LEN][]const u8 = undefined;
    var n: usize = 0;
    for ('A'..'Z' + 1) |c| {
        pool[n] = &[_]u8{c};
        n += 1;
    }
    for (0..100) |i| {
        pool[n] = std.fmt.comptimePrint("{d:0>2}", .{i});
        n += 1;
    }
    for ('a'..'z' + 1) |c| {
        pool[n] = &[_]u8{c};
        n += 1;
    }
    for (0..200) |i| {
        pool[n] = std.fmt.comptimePrint("#{d}", .{i});
        n += 1;
    }
    for ('A'..'Z' + 1) |x| {
        for ('A'..'Z' + 1) |y| {
            pool[n] = &[_]u8{ x, y };
            n += 1;
        }
    }
    break :blk pool;
};

pub const Kind = enum { noul, choice, score };

pub const Question = struct {
    /// Borrowed from the parsed request.
    id: []const u8,
    kind: Kind,
    instructions: []u8,
    /// choice: the option labels; score: the level names (the legend).
    labels: [][]u8 = &.{},
    /// choice: each option's text, its description or else its label with `_` as spaces.
    texts: [][]u8 = &.{},
    /// noul: the `[yes, no]` criteria text, null when the criteria object is absent or empty.
    noul: ?[2][]u8 = null,

    pub fn deinit(self: *Question, a: std.mem.Allocator) void {
        a.free(self.instructions);
        for (self.labels) |s| a.free(s);
        a.free(self.labels);
        for (self.texts) |s| a.free(s);
        a.free(self.texts);
        if (self.noul) |n| for (n) |s| a.free(s);
    }
};

pub const Questions = struct {
    qs: []Question,

    /// The SystemOne question schema: a non-empty object of noul / choice / score questions, in request order.
    pub fn init(a: std.mem.Allocator, v: std.json.Value, max_questions: usize) !Questions {
        if (v != .object or v.object.count() == 0) return error.D1NoQuestions;
        if (v.object.count() > max_questions) return error.TooManyQuestions;
        var list: std.ArrayList(Question) = .empty;
        errdefer {
            for (list.items) |*q| q.deinit(a);
            list.deinit(a);
        }
        var it = v.object.iterator();
        while (it.next()) |kv| try list.append(a, try parseQuestion(a, kv.key_ptr.*, kv.value_ptr.*));
        return .{ .qs = try list.toOwnedSlice(a) };
    }

    pub fn deinit(self: *Questions, a: std.mem.Allocator) void {
        for (self.qs) |*q| q.deinit(a);
        a.free(self.qs);
    }
};

fn parseQuestion(a: std.mem.Allocator, id: []const u8, v: std.json.Value) !Question {
    if (v != .object) return error.D1BadQuestion;
    const tv = v.object.get("type") orelse return error.D1BadType;
    if (tv != .string) return error.D1BadType;
    const kind = std.meta.stringToEnum(Kind, tv.string) orelse return error.D1BadType;
    const iv = v.object.get("instructions") orelse return error.D1BadInstructions;
    if (iv != .string) return error.D1BadInstructions;
    var q: Question = .{ .id = id, .kind = kind, .instructions = try a.dupe(u8, iv.string) };
    errdefer q.deinit(a);
    const crit = v.object.get("criteria");
    switch (kind) {
        .noul => if (crit) |cv| switch (cv) {
            .null => {},
            .object => |o| if (o.count() > 0) {
                const yes = try criteriaText(a, o.get("true"));
                errdefer a.free(yes);
                q.noul = .{ yes, try criteriaText(a, o.get("false")) };
            },
            else => return error.D1NoulCriteria,
        },
        .choice => {
            const cv = crit orelse return error.D1ChoiceCriteria;
            switch (cv) {
                .object => |o| {
                    if (o.count() == 0 or o.count() > MAX_OPTIONS) return error.D1ChoiceCriteria;
                    var labels: std.ArrayList([]u8) = .empty;
                    var texts: std.ArrayList([]u8) = .empty;
                    errdefer {
                        for (labels.items) |s| a.free(s);
                        labels.deinit(a);
                        for (texts.items) |s| a.free(s);
                        texts.deinit(a);
                    }
                    var it = o.iterator();
                    while (it.next()) |kv| {
                        const label = try a.dupe(u8, kv.key_ptr.*);
                        try labels.append(a, label);
                        try texts.append(a, try optionText(a, label, kv.value_ptr.*));
                    }
                    q.labels = try labels.toOwnedSlice(a);
                    q.texts = try texts.toOwnedSlice(a);
                },
                .array => |arr| {
                    if (arr.items.len == 0 or arr.items.len > MAX_OPTIONS) return error.D1ChoiceCriteria;
                    for (arr.items, 0..) |x, i| {
                        if (x != .string) return error.D1ChoiceCriteria;
                        for (arr.items[0..i]) |y| if (std.mem.eql(u8, x.string, y.string)) return error.D1ChoiceCriteria;
                    }
                    // Emptied before any allocation can fail, so `deinit` frees only what was set.
                    q.labels = try a.alloc([]u8, arr.items.len);
                    @memset(q.labels, &.{});
                    q.texts = try a.alloc([]u8, arr.items.len);
                    @memset(q.texts, &.{});
                    for (arr.items, q.labels, q.texts) |x, *label, *text| {
                        label.* = try a.dupe(u8, x.string);
                        text.* = try optionText(a, label.*, .null);
                    }
                },
                else => return error.D1ChoiceCriteria,
            }
        },
        .score => {
            const cv = crit orelse return error.D1ScoreCriteria;
            if (cv != .array or cv.array.items.len < 2 or cv.array.items.len > MAX_SCORE_LEVELS) return error.D1ScoreCriteria;
            for (cv.array.items) |x| if (x != .string) return error.D1ScoreCriteria;
            q.labels = try a.alloc([]u8, cv.array.items.len);
            @memset(q.labels, &.{});
            for (cv.array.items, q.labels) |x, *s| s.* = try a.dupe(u8, x.string);
        },
    }
    return q;
}

/// `prompt.question_block`'s option text: a non-empty description, else the label with `_` read as spaces.
fn optionText(a: std.mem.Allocator, label: []const u8, desc: std.json.Value) ![]u8 {
    return switch (desc) {
        .string => |s| if (s.len > 0) try a.dupe(u8, s) else try spaced(a, label),
        .null => try spaced(a, label),
        else => error.D1ChoiceCriteria,
    };
}

fn spaced(a: std.mem.Allocator, label: []const u8) ![]u8 {
    const out = try a.dupe(u8, label);
    for (out) |*c| if (c.* == '_') {
        c.* = ' ';
    };
    return out;
}

/// A noul criterion as the prompt writes it: its text, or `None` when the key is absent.
fn criteriaText(a: std.mem.Allocator, v: ?std.json.Value) ![]u8 {
    const x = v orelse return a.dupe(u8, "None");
    return switch (x) {
        .string => |s| try a.dupe(u8, s),
        .null => try a.dupe(u8, "None"),
        else => error.D1NoulCriteria,
    };
}

/// One option's code in the prompt and the vocabulary id it reads.
pub const Alias = struct { code: []const u8, id: u32 };

/// The single-token lookup the prompt and readout need: a text's one token id, or null when it is not one token.
pub const Vocab = struct {
    ctx: *anyopaque,
    single_fn: *const fn (ctx: *anyopaque, text: []const u8) ?u32,

    pub fn single(self: Vocab, text: []const u8) ?u32 {
        return self.single_fn(self.ctx, text);
    }
};

/// A question's prompt plan: the codes its options are written with, and each option's readout token group.
pub const Plan = struct {
    codes: []Alias = &.{},
    /// Codes allocated past the letters; `codes` may point into it.
    owned: [][]u8 = &.{},
    /// One group per option, in option order (a noul's are `[yes, no]`).
    groups: [][]u32 = &.{},

    pub fn deinit(self: *Plan, a: std.mem.Allocator) void {
        a.free(self.codes);
        for (self.owned) |s| a.free(s);
        a.free(self.owned);
        for (self.groups) |g| a.free(g);
        a.free(self.groups);
    }
};

/// `prompt.state_block` for the default `json_only` style: a string as is, any other value as `json.dumps(indent=2)`.
pub fn stateBlock(a: std.mem.Allocator, state: std.json.Value) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    if (state == .string) {
        try out.appendSlice(a, state.string);
    } else {
        try laya.pyJsonIndent(a, &out, state, false);
    }
    try out.appendSlice(a, "\n\n");
    return out.toOwnedSlice(a);
}

/// Everything before the question: BOS, the user turn and the state, ending at `QUESTION:`.
pub fn prefixText(a: std.mem.Allocator, state: std.json.Value) ![]u8 {
    const block = try stateBlock(a, state);
    defer a.free(block);
    return std.fmt.allocPrint(a, BOS ++ IM_START ++ "user\n{s}\nQUESTION:\n", .{block});
}

/// `prompt.aliases` + `prompt.readout_ids`: codes and readout groups for one question.
pub fn plan(a: std.mem.Allocator, vocab: Vocab, q: *const Question) !Plan {
    switch (q.kind) {
        .noul => {
            const yes = try singleIds(a, vocab, &.{ "yes", "Yes", "YES" });
            errdefer a.free(yes);
            const no = try singleIds(a, vocab, &.{ "no", "No", "NO" });
            errdefer a.free(no);
            if (yes.len == 0 or no.len == 0) return error.D1NoSingleToken;
            const groups = try a.alloc([]u32, 2);
            groups[0] = yes;
            groups[1] = no;
            return .{ .groups = groups };
        },
        .score => {
            const groups = try a.alloc([]u32, q.labels.len);
            var n: usize = 0;
            errdefer {
                for (groups[0..n]) |g| a.free(g);
                a.free(groups);
            }
            for (groups, 0..) |*g, i| {
                const digit = try std.fmt.allocPrint(a, "{d}", .{i});
                defer a.free(digit);
                g.* = try singleIds(a, vocab, &.{digit});
                n += 1;
                if (g.len == 0) return error.D1NoSingleToken;
            }
            return .{ .groups = groups };
        },
        .choice => {
            var owned: std.ArrayList([]u8) = .empty;
            errdefer {
                for (owned.items) |s| a.free(s);
                owned.deinit(a);
            }
            const codes = try optionCodes(a, q.labels, &owned);
            defer a.free(codes);
            const aliases = try assignAliases(a, vocab, codes);
            errdefer a.free(aliases);
            const groups = try a.alloc([]u32, aliases.len);
            var n: usize = 0;
            errdefer {
                for (groups[0..n]) |g| a.free(g);
                a.free(groups);
            }
            for (aliases, groups) |al, *g| {
                const spaced_code = try std.fmt.allocPrint(a, " {s}", .{al.code});
                defer a.free(spaced_code);
                const extra = vocab.single(spaced_code);
                g.* = if (extra != null and extra.? != al.id)
                    try a.dupe(u32, &.{ al.id, extra.? })
                else
                    try a.dupe(u32, &.{al.id});
                n += 1;
            }
            return .{ .codes = aliases, .owned = try owned.toOwnedSlice(a), .groups = groups };
        },
    }
}

/// The distinct single-token ids of `texts`, in order (`prompt._ids`).
fn singleIds(a: std.mem.Allocator, vocab: Vocab, texts: []const []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    for (texts) |t| {
        const id = vocab.single(t) orelse continue;
        if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
    }
    return out.toOwnedSlice(a);
}

/// `prompt.option_codes`: the labels themselves when each is one letter, else A..Z, else 00.. (one rule per cardinality).
fn optionCodes(a: std.mem.Allocator, labels: []const []u8, owned: *std.ArrayList([]u8)) ![][]const u8 {
    var letters = true;
    for (labels) |l| {
        const t = std.mem.trim(u8, l, " \t\r\n");
        if (t.len != 1 or !std.ascii.isAlphabetic(t[0])) letters = false;
    }
    const out = try a.alloc([]const u8, labels.len);
    errdefer a.free(out);
    for (labels, out, 0..) |l, *code, i| {
        if (letters) {
            code.* = std.mem.trim(u8, l, " \t\r\n");
        } else if (labels.len <= 26) {
            code.* = FALLBACK_POOL[i];
        } else {
            const s = try std.fmt.allocPrint(a, "{d:0>2}", .{i});
            try owned.append(a, s);
            code.* = s;
        }
    }
    return out;
}

/// `prompt.aliases`: each code's own single token, or the next pool code that is one unused token. The code
/// written in the prompt is the one that won, so a fallback is shown to the model as the code it read.
fn assignAliases(a: std.mem.Allocator, vocab: Vocab, codes: []const []const u8) ![]Alias {
    var used = std.AutoHashMap(u32, void).init(a);
    defer used.deinit();
    const out = try a.alloc(Alias, codes.len);
    errdefer a.free(out);
    for (codes, out) |code, *o| {
        o.* = (try takeAlias(vocab, &used, code)) orelse blk: {
            for (FALLBACK_POOL) |raw| if (try takeAlias(vocab, &used, raw)) |al| break :blk al;
            return error.D1NoAlias;
        };
    }
    return out;
}

fn takeAlias(vocab: Vocab, used: *std.AutoHashMap(u32, void), text: []const u8) !?Alias {
    const id = vocab.single(text) orelse return null;
    if (used.contains(id)) return null;
    try used.put(id, {});
    return .{ .code = text, .id = id };
}

/// The question and the assistant header, up to the answer slot.
pub fn suffixText(a: std.mem.Allocator, q: *const Question, codes: []const Alias) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, q.instructions);
    switch (q.kind) {
        .choice => {
            try out.appendSlice(a, "\n\nOptions:\n");
            for (q.texts, codes, 0..) |text, al, i| {
                if (i > 0) try out.append(a, '\n');
                try out.print(a, "{s} {s}", .{ al.code, text });
            }
            try out.appendSlice(a, "\n\nReply with the option code only.");
        },
        .noul => {
            if (q.noul) |n| try out.print(a, "\nYes: {s}\nNo: {s}", .{ n[0], n[1] });
            try out.appendSlice(a, "\n\nReply with yes or no only.");
        },
        .score => {
            try out.appendSlice(a, "\n\n");
            for (q.labels, 0..) |name, i| {
                if (i > 0) try out.append(a, '\n');
                try out.print(a, "{d} {s}", .{ i, name });
            }
            try out.print(a, "\n\nReply with a single digit 0-{d} only.", .{q.labels.len - 1});
        },
    }
    try out.appendSlice(a, IM_END ++ "\n" ++ IM_START ++ "assistant\n");
    return out.toOwnedSlice(a);
}

/// Option probabilities: each group scores its best token's logit, then a softmax over the groups.
pub fn probabilities(a: std.mem.Allocator, logits: []const f32, groups: []const []const u32) ![]f64 {
    const out = try a.alloc(f64, groups.len);
    errdefer a.free(out);
    var top: f64 = -std.math.inf(f64);
    for (groups, out) |g, *o| {
        var best: f64 = -std.math.inf(f64);
        for (g) |id| {
            if (id >= logits.len) return error.D1BadLogits;
            best = @max(best, @as(f64, logits[id]));
        }
        o.* = best;
        top = @max(top, best);
    }
    var z: f64 = 0;
    for (out) |*o| {
        o.* = @exp(o.* - top);
        z += o.*;
    }
    for (out) |*o| o.* /= z;
    return out;
}

/// `api.answer` for every question, in request order: a noul's P(yes); a choice's pick, confidence and
/// probabilities; a score's expected level, confidence and legend. Confidence is the chosen probability.
pub fn appendAnswers(a: std.mem.Allocator, out: *std.ArrayList(u8), qs: []const Question, probs: []const []f64) !void {
    try out.appendSlice(a, "{");
    for (qs, probs, 0..) |q, p, qi| {
        if (qi > 0) try out.append(a, ',');
        try laya.wireString(a, out, q.id);
        try out.print(a, ":{{\"type\":\"{s}\"", .{@tagName(q.kind)});
        switch (q.kind) {
            .noul => {
                try out.appendSlice(a, ",\"noul\":");
                try appendProb(a, out, p[0]);
            },
            .choice => {
                const best = argmax(p);
                try out.appendSlice(a, ",\"choice\":");
                try laya.wireString(a, out, q.labels[best]);
                try out.appendSlice(a, ",\"confidence\":");
                try appendProb(a, out, p[best]);
                try out.appendSlice(a, ",\"probabilities\":{");
                for (q.labels, p, 0..) |label, v, j| {
                    if (j > 0) try out.append(a, ',');
                    try laya.wireString(a, out, label);
                    try out.append(a, ':');
                    try appendProb(a, out, v);
                }
                try out.append(a, '}');
            },
            .score => {
                var expected: f64 = 0;
                for (p, 0..) |v, j| expected += @as(f64, @floatFromInt(j)) * v;
                try out.appendSlice(a, ",\"score\":");
                try appendProb(a, out, expected);
                try out.appendSlice(a, ",\"confidence\":");
                try appendProb(a, out, p[argmax(p)]);
                try out.appendSlice(a, ",\"probabilities\":{");
                for (p, 0..) |v, j| {
                    if (j > 0) try out.append(a, ',');
                    try out.print(a, "\"{d}\":", .{j});
                    try appendProb(a, out, v);
                }
                try out.appendSlice(a, "},\"legend\":{");
                for (q.labels, 0..) |name, j| {
                    if (j > 0) try out.append(a, ',');
                    try out.print(a, "\"{d}\":", .{j});
                    try laya.wireString(a, out, name);
                }
                try out.append(a, '}');
            },
        }
        try out.append(a, '}');
    }
    try out.append(a, '}');
}

fn appendProb(a: std.mem.Allocator, out: *std.ArrayList(u8), x: f64) !void {
    try laya.pyFloat(a, out, kev.roundProb(x));
}

/// The first largest entry, as Python's `max(range(n), key=...)` picks it.
fn argmax(p: []const f64) usize {
    var best: usize = 0;
    for (p, 0..) |v, j| if (v > p[best]) {
        best = j;
    };
    return best;
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    stream: S,
    trunk: kev.Trunk,
    /// The trunk's LM head (borrowed from the trunk).
    head: quant_abi.Weight,
    tok: tokenizer_mod.Tokenizer,

    /// A dir whose config.json names `modeling_d1.D1Model` (bf16 LFM2.5-VL weights, the card's own layout).
    pub fn load(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, s: S) !*Engine {
        if (try hasTorchaoQuant(io, allocator, dir)) return error.D1Int8Unsupported;
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.stream = s;
        // The trunk is bound here, never by the LM's served path.
        self.trunk = kev.Trunk.parse(io, allocator, dir, &.{&lfm2.table}, "{}") catch |e| return if (e == error.TrunkUnsupported) error.D1UnsupportedBase else e;
        var loaded = false;
        errdefer self.trunk.close(loaded);
        if (!std.mem.startsWith(u8, self.trunk.parsed.shell.model_type.slice(), "lfm2")) return error.D1UnsupportedBase;
        self.tok = try tokenizer_mod.loadTokenizer(io, allocator, dir);
        errdefer self.tok.deinit();
        // The thread's default GPU stream, as the engine's own load takes.
        try self.trunk.load(io, allocator, dir, mlx.gpuStream(), null);
        loaded = true;
        self.head = try self.trunk.weight(.head);
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.trunk.close(true);
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    /// `x` through the trunk's LM head as the engine's shell projects a draft row (`lmHeadProject`): a dense head
    /// through a transposed view, a quantized one at its own per-weight parameters.
    pub fn headLogits(self: *const Engine, x: A) !A {
        const s = self.trunk.s;
        const w: A = .{ .ctx = self.head.w.ctx };
        const sc: A = .{ .ctx = self.head.scales.ctx };
        if (sc.ctx == null) {
            const wt = try array.transposeBf16Weight(w, s);
            defer free(wt);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
            return out;
        }
        if (sushi_pack.isRowScaledFp8(w, sc)) return sushi_pack.fp8Linear(x, w, sc, s);
        const q = self.trunk.parsed.shell.quant;
        const base: params.QuantParams = .{ .bits = q.bits, .group_size = q.group_size, .mode = std.meta.stringToEnum(params.QuantMode, @tagName(q.mode)) orelse .affine };
        const qp = params.computeQuantParams(base, w, sc, array.lastDim(x));
        return qmatmulBits(x, w, sc, .{ .ctx = self.head.biases.ctx }, qp.bits, qp.group_size, qp.mode, s);
    }

    pub fn parseQuestions(_: *const Engine, a: std.mem.Allocator, questions: std.json.Value, max_questions: usize) !Questions {
        return Questions.init(a, questions, max_questions);
    }

    /// The response JSON for one request: `{"model", "answers", "usage"}`.
    pub fn predict(self: *Engine, a: std.mem.Allocator, model_id: []const u8, state: std.json.Value, questions: *const Questions, max_input_tokens: usize) ![]u8 {
        const prefix = try prefixText(a, state);
        defer a.free(prefix);
        const prefix_ids = try self.tok.encode(a, prefix);
        defer a.free(prefix_ids);

        var tv: TokVocab = .{ .tok = &self.tok, .a = a };
        const vocab: Vocab = .{ .ctx = &tv, .single_fn = TokVocab.single };
        const n = questions.qs.len;
        const plans = try a.alloc(Plan, n);
        var planned: usize = 0;
        defer {
            for (plans[0..planned]) |*p| p.deinit(a);
            a.free(plans);
        }
        const suffixes = try a.alloc([]u32, n);
        var encoded: usize = 0;
        defer {
            for (suffixes[0..encoded]) |s| a.free(s);
            a.free(suffixes);
        }
        var input_tokens = prefix_ids.len;
        for (questions.qs, 0..) |*q, i| {
            plans[i] = try plan(a, vocab, q);
            planned = i + 1;
            const text = try suffixText(a, q, plans[i].codes);
            defer a.free(text);
            suffixes[i] = try self.tok.encode(a, text);
            encoded = i + 1;
            input_tokens += suffixes[i].len;
            if (input_tokens > max_input_tokens) return error.TooManyInputTokens;
        }

        const probs = try a.alloc([]f64, n);
        var emit: ReadoutEmit = .{ .eng = self, .a = a, .plans = plans, .probs = probs };
        defer {
            for (probs[0..emit.done]) |p| a.free(p);
            a.free(probs);
        }
        const branches = try a.alloc([]const u32, n);
        defer a.free(branches);
        for (suffixes, branches) |s, *b| b.* = s;
        try kev.scoreBranches(a, &self.trunk, prefix_ids, branches, &emit, ReadoutEmit.run);

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        try out.appendSlice(a, "{\"model\":");
        try laya.wireString(a, &out, model_id);
        try out.appendSlice(a, ",\"answers\":");
        try appendAnswers(a, &out, questions.qs, probs);
        try out.print(a, ",\"usage\":{{\"input_tokens\":{d},\"output_tokens\":0}}}}", .{input_tokens});
        return out.toOwnedSlice(a);
    }

    /// Logits at the last row of `h` `[1, L, H]`, through the trunk's own LM head.
    fn lastLogits(self: *Engine, a: std.mem.Allocator, h: A) ![]f32 {
        const s = self.stream;
        const last_index = [_]i32{@intCast(mlx.getShape(h)[1] - 1)};
        const index_shape = [_]c_int{1};
        const index = mlx.mlx_array_new_data(&last_index, &index_shape, 1, .int32);
        defer free(index);
        const last = try laya.take(h, index, 1, s);
        defer free(last);
        const logits = try self.headLogits(last);
        defer free(logits);
        const f32_logits = try laya.astype(logits, .float32, s);
        defer free(f32_logits);
        const flat = try laya.reshape(f32_logits, &[_]c_int{-1}, s);
        defer free(flat);
        try mlx.check(mlx.mlx_array_eval(flat));
        const raw = mlx.mlx_array_data_float32(flat) orelse return error.MlxError;
        return a.dupe(f32, raw[0..@intCast(mlx.getShape(flat)[0])]);
    }

    /// Each question's probabilities, from its branch's last hidden row.
    const ReadoutEmit = struct {
        eng: *Engine,
        a: std.mem.Allocator,
        plans: []const Plan,
        probs: [][]f64,
        done: usize = 0,

        fn run(self: *ReadoutEmit, i: usize, h: A) !void {
            const logits = try self.eng.lastLogits(self.a, h);
            defer self.a.free(logits);
            self.probs[i] = try probabilities(self.a, logits, self.plans[i].groups);
            self.done += 1;
        }
    };
};

/// Token lookups through the engine's tokenizer: a text is single when it encodes to exactly one id.
const TokVocab = struct {
    tok: *const tokenizer_mod.Tokenizer,
    a: std.mem.Allocator,

    fn single(ctx: *anyopaque, text: []const u8) ?u32 {
        const self: *TokVocab = @ptrCast(@alignCast(ctx));
        const ids = self.tok.encode(self.a, text) catch return null;
        defer self.a.free(ids);
        return if (ids.len == 1) ids[0] else null;
    }
};

/// A torchao int8 checkpoint (`quantization_config`, `_weight_qdata` tensors) is not loadable: refused by name.
fn hasTorchaoQuant(io: std.Io, allocator: std.mem.Allocator, dir: []const u8) !bool {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    var file = try d.openFile(io, "config.json", .{});
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const bytes = try rs.interface.allocRemaining(allocator, .limited(1 * 1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    return parsed.value == .object and parsed.value.object.get("quantization_config") != null;
}

/// The message for a 400 this file raises, or null for the generic ones the server words itself.
pub fn errorMessage(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.D1NoQuestions => "\"questions\" must be a non-empty object",
        error.D1BadQuestion => "each question must be an object",
        error.D1BadType => "\"type\" must be one of: choice, score, noul",
        error.D1BadInstructions => "\"instructions\" must be a string",
        error.D1NoulCriteria => "noul \"criteria\" must be an object with \"true\" and \"false\" strings",
        error.D1ChoiceCriteria => "choice \"criteria\" must be an object of labels or a list of unique labels",
        error.D1ScoreCriteria => "score \"criteria\" must be a list of 2 to 10 levels",
        error.D1Int8Unsupported => "this D1 pack is int8 (torchao); only the bf16 LiquidAI/d1-3B pack is served",
        error.D1UnsupportedBase => "this D1 checkpoint is not an LFM2 trunk",
        error.D1NoSingleToken => "the tokenizer has no single-token digit, yes/no or option code for this question",
        error.D1NoAlias => "too many options for the option codes",
        else => null,
    };
}

// ── tests ──

const fake_table = [_]struct { text: []const u8, id: u32 }{
    .{ .text = "A", .id = 10 },
    .{ .text = " A", .id = 20 },
    .{ .text = "C", .id = 12 },
    .{ .text = "0", .id = 30 },
    .{ .text = "1", .id = 31 },
    .{ .text = "2", .id = 32 },
    .{ .text = "yes", .id = 40 },
    .{ .text = "no", .id = 41 },
    .{ .text = "Yes", .id = 42 },
};

fn fakeSingle(_: *anyopaque, text: []const u8) ?u32 {
    for (fake_table) |e| if (std.mem.eql(u8, e.text, text)) return e.id;
    return null;
}

fn fakeVocab(dummy: *u8) Vocab {
    return .{ .ctx = dummy, .single_fn = fakeSingle };
}

test "d1 state block: a string as is, any other value as json.dumps(indent=2)" {
    const a = testing.allocator;
    const s = try laya.parseRequestJson(a, "\"Refund me now\"");
    defer s.deinit();
    const text = try stateBlock(a, s.value);
    defer a.free(text);
    try testing.expectEqualStrings("Refund me now\n\n", text);

    const obj = try laya.parseRequestJson(a, "{\"ticket\":{\"id\":7,\"tags\":[\"a\",\"b\"]},\"empty\":{}}");
    defer obj.deinit();
    const block = try stateBlock(a, obj.value);
    defer a.free(block);
    try testing.expectEqualStrings(
        "{\n  \"ticket\": {\n    \"id\": 7,\n    \"tags\": [\n      \"a\",\n      \"b\"\n    ]\n  },\n  \"empty\": {}\n}\n\n",
        block,
    );
}

test "d1 prefix: BOS, the user turn, the state, then QUESTION:" {
    const a = testing.allocator;
    const s = try laya.parseRequestJson(a, "\"Order 42 never arrived\"");
    defer s.deinit();
    const text = try prefixText(a, s.value);
    defer a.free(text);
    try testing.expectEqualStrings("<|startoftext|><|im_start|>user\nOrder 42 never arrived\n\n\nQUESTION:\n", text);
}

test "d1 plan: letters for labels, a readout group of the letter and its spaced form" {
    const a = testing.allocator;
    var dummy: u8 = 0;
    const req = try laya.parseRequestJson(a, "{\"team\":{\"type\":\"choice\",\"instructions\":\"Which team?\",\"criteria\":{\"billing\":\"Payments\",\"sales\":null}}}");
    defer req.deinit();
    var qs = try Questions.init(a, req.value, 8);
    defer qs.deinit(a);
    var p = try plan(a, fakeVocab(&dummy), &qs.qs[0]);
    defer p.deinit(a);
    try testing.expectEqualStrings("A", p.codes[0].code);
    try testing.expectEqual(@as(u32, 10), p.codes[0].id);
    try testing.expectEqualSlices(u32, &.{ 10, 20 }, p.groups[0]);
    try testing.expectEqual(@as(u32, 12), p.codes[1].id); // "B" is no vocabulary entry: the pool's next single token
    try testing.expectEqualStrings("C", p.codes[1].code);
}

test "d1 question text: choice options, noul criteria with None, score levels, then the header" {
    const a = testing.allocator;
    var dummy: u8 = 0;
    const req = try laya.parseRequestJson(a,
        \\{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments, refunds","not_urgent":null}},
        \\ "churn":{"type":"noul","instructions":"Does the customer threaten to cancel?","criteria":{"true":"explicit threat"}},
        \\ "urgency":{"type":"score","instructions":"How urgent is this?","criteria":["not urgent","soon","blocking"]}}
    );
    defer req.deinit();
    var qs = try Questions.init(a, req.value, 8);
    defer qs.deinit(a);
    var p0 = try plan(a, fakeVocab(&dummy), &qs.qs[0]);
    defer p0.deinit(a);
    const choice = try suffixText(a, &qs.qs[0], p0.codes);
    defer a.free(choice);
    try testing.expectEqualStrings(
        "Which team?\n\nOptions:\nA Payments, refunds\nC not urgent\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n",
        choice,
    );
    var p1 = try plan(a, fakeVocab(&dummy), &qs.qs[1]);
    defer p1.deinit(a);
    const noul = try suffixText(a, &qs.qs[1], p1.codes);
    defer a.free(noul);
    try testing.expectEqualStrings(
        "Does the customer threaten to cancel?\nYes: explicit threat\nNo: None\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n",
        noul,
    );
    var p2 = try plan(a, fakeVocab(&dummy), &qs.qs[2]);
    defer p2.deinit(a);
    const score = try suffixText(a, &qs.qs[2], p2.codes);
    defer a.free(score);
    try testing.expectEqualStrings(
        "How urgent is this?\n\n0 not urgent\n1 soon\n2 blocking\n\nReply with a single digit 0-2 only.<|im_end|>\n<|im_start|>assistant\n",
        score,
    );
    // The noul readout is [yes, no]: the surface forms that are one token each, duplicates dropped.
    try testing.expectEqualSlices(u32, &.{ 40, 42 }, p1.groups[0]);
    try testing.expectEqualSlices(u32, &.{41}, p1.groups[1]);
    try testing.expectEqualSlices(u32, &.{30}, p2.groups[0]);
    try testing.expectEqualSlices(u32, &.{32}, p2.groups[2]);
}

test "d1 plan: codes past the letters are two digits, and a missing digit token is refused" {
    const a = testing.allocator;
    var dummy: u8 = 0;
    const req = try laya.parseRequestJson(a, "{\"u\":{\"type\":\"score\",\"instructions\":\"x\",\"criteria\":[\"one\",\"two\",\"three\"]}}");
    defer req.deinit();
    var qs = try Questions.init(a, req.value, 8);
    defer qs.deinit(a);
    var p = try plan(a, fakeVocab(&dummy), &qs.qs[0]);
    defer p.deinit(a);
    try testing.expectEqual(@as(usize, 3), p.groups.len);
    try testing.expectEqual(@as(u32, 30), p.groups[0][0]);
}

test "d1 readout: a softmax over each group's best logit" {
    const a = testing.allocator;
    const logits = [_]f32{ 0, 1, 2, 3, 0.5, -1 };
    const groups = [_][]const u32{ &.{ 2, 3 }, &.{4}, &.{ 5, 1 } };
    const probs = try probabilities(a, &logits, &groups);
    defer a.free(probs);
    // Group scores 3, 0.5, 1: softmax -> e^0, e^-2.5, e^-2 over their sum.
    const z = 1.0 + @exp(@as(f64, -2.5)) + @exp(@as(f64, -2.0));
    try testing.expectApproxEqAbs(1.0 / z, probs[0], 1e-9);
    try testing.expectApproxEqAbs(@exp(@as(f64, -2.5)) / z, probs[1], 1e-9);
    try testing.expectApproxEqAbs(@exp(@as(f64, -2.0)) / z, probs[2], 1e-9);
}

test "d1 answers: noul is P(yes); choice and score confidence is the chosen probability" {
    const a = testing.allocator;
    const req = try laya.parseRequestJson(a,
        \\{"churn":{"type":"noul","instructions":"Cancel?"},
        \\ "team":{"type":"choice","instructions":"Which?","criteria":["billing","sales"]},
        \\ "urgency":{"type":"score","instructions":"How?","criteria":["not urgent","soon","blocking"]}}
    );
    defer req.deinit();
    var qs = try Questions.init(a, req.value, 8);
    defer qs.deinit(a);
    var p0 = [_]f64{ 0.8, 0.2 };
    var p1 = [_]f64{ 0.7, 0.3 };
    var p2 = [_]f64{ 0.0, 0.5, 0.5 };
    var probs = [_][]f64{ &p0, &p1, &p2 };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try appendAnswers(a, &out, qs.qs, &probs);
    try testing.expectEqualStrings(
        "{\"churn\":{\"type\":\"noul\",\"noul\":0.8}," ++
            "\"team\":{\"type\":\"choice\",\"choice\":\"billing\",\"confidence\":0.7,\"probabilities\":{\"billing\":0.7,\"sales\":0.3}}," ++
            "\"urgency\":{\"type\":\"score\",\"score\":1.5,\"confidence\":0.5,\"probabilities\":{\"0\":0.0,\"1\":0.5,\"2\":0.5},\"legend\":{\"0\":\"not urgent\",\"1\":\"soon\",\"2\":\"blocking\"}}}",
        out.items,
    );
}

test "d1 questions: a score needs 2 to 10 levels, a noul's criteria is an object, a choice is a map or a list" {
    const a = testing.allocator;
    const one = try laya.parseRequestJson(a, "{\"u\":{\"type\":\"score\",\"instructions\":\"x\",\"criteria\":[\"one\"]}}");
    defer one.deinit();
    try testing.expectError(error.D1ScoreCriteria, Questions.init(a, one.value, 8));
    const noul = try laya.parseRequestJson(a, "{\"u\":{\"type\":\"noul\",\"instructions\":\"x\",\"criteria\":[\"no\",\"yes\"]}}");
    defer noul.deinit();
    try testing.expectError(error.D1NoulCriteria, Questions.init(a, noul.value, 8));
    const choice = try laya.parseRequestJson(a, "{\"u\":{\"type\":\"choice\",\"instructions\":\"x\",\"criteria\":\"a\"}}");
    defer choice.deinit();
    try testing.expectError(error.D1ChoiceCriteria, Questions.init(a, choice.value, 8));
}
