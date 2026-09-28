//! Kev typed decisions (github.com/jaredpalmer/kev): a Qwen3.5 backbone whose final-norm hidden states at
//! option-close and decide markers are scored by a pointer head. Request mapping, prompt layout, confidences
//! and rounding mirror `kev.api` and `kev.model`; the tokens and answers are pinned by `tests/fixtures/kev`.
const std = @import("std");
const laya = @import("laya.zig");
const log = @import("log.zig");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");

const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const free = laya.free;
const matmul = laya.matmul;
const linear = laya.linear;
const reshape = laya.reshape;
const take = laya.take;
const astype = laya.astype;

pub const MAX_OPTIONS = 255;
/// kev SERVE_MAX_BRANCH: a whole question row (state + branch, delimiters included).
pub const MAX_ROW = 8192;
/// kev SERVE_MAX_STATE: the state delimiter plus at most MAX_ROW - 1 state tokens.
pub const MAX_STATE = 8192;

pub const QType = enum { noul, choice, score };

// ── Request mapping (kev.api.to_record) ──

/// Rendered bytes allowed per request part (the state, or all questions together). Every byte written counts,
/// copies of nested items included, so a small deeply nested body cannot fan out into a large rendering.
pub const MAX_RENDER_BYTES: usize = 1 << 20;

pub const Budget = struct {
    left: usize = MAX_RENDER_BYTES,

    fn take(self: *Budget, n: usize) !void {
        if (n > self.left) return error.KevRenderTooLarge;
        self.left -= n;
    }
};

/// `kev.api.render`: the text the model sees for any JSON value. Not JSON: objects become `key: value` lines.
pub fn render(a: std.mem.Allocator, out: *std.ArrayList(u8), v: std.json.Value, indent: usize, budget: *Budget) !void {
    return renderDepth(a, out, v, indent, 0, budget);
}

fn put(a: std.mem.Allocator, out: *std.ArrayList(u8), bytes: []const u8, budget: *Budget) !void {
    try budget.take(bytes.len);
    try out.appendSlice(a, bytes);
}

fn pad(a: std.mem.Allocator, out: *std.ArrayList(u8), indent: usize, budget: *Budget) !void {
    try budget.take(2 * indent);
    try out.appendNTimes(a, ' ', 2 * indent);
}

fn renderDepth(a: std.mem.Allocator, out: *std.ArrayList(u8), v: std.json.Value, indent: usize, depth: usize, budget: *Budget) !void {
    if ((v == .array or v == .object) and depth >= laya.MAX_JSON_DEPTH) return error.NestingTooDeep;
    switch (v) {
        .null => {},
        .bool => |b| try put(a, out, if (b) "True" else "False", budget),
        .integer, .float, .number_string => {
            // Integer text is copied as written (Python ints are unbounded); every other spelling fits in 32 bytes.
            const n = if (v == .number_string and std.json.isNumberFormattedLikeAnInteger(v.number_string)) v.number_string.len else 32;
            try budget.take(n);
            switch (v) {
                .integer => |i| try out.print(a, "{d}", .{i}),
                .float => |f| try laya.pyFloat(a, out, f),
                .number_string => |ns| try laya.pyNumber(a, out, ns),
                else => unreachable,
            }
        },
        .string => |str| try put(a, out, str, budget),
        .array => |arr| for (arr.items, 0..) |x, i| {
            if (i > 0) try put(a, out, "\n", budget);
            try pad(a, out, indent, budget);
            try put(a, out, "- ", budget);
            var item: std.ArrayList(u8) = .empty;
            defer item.deinit(a);
            try renderDepth(a, &item, x, indent + 1, depth + 1, budget);
            try put(a, out, lstripPy(item.items), budget);
        },
        .object => |obj| {
            var it = obj.iterator();
            var i: usize = 0;
            while (it.next()) |kv| : (i += 1) {
                if (i > 0) try put(a, out, "\n", budget);
                try pad(a, out, indent, budget);
                try put(a, out, kv.key_ptr.*, budget);
                const x = kv.value_ptr.*;
                if (x == .object or x == .array) {
                    try put(a, out, ":\n", budget);
                    try renderDepth(a, out, x, indent + 1, depth + 1, budget);
                } else {
                    try put(a, out, ": ", budget);
                    try renderDepth(a, out, x, 0, depth + 1, budget);
                }
            }
        },
    }
}

/// Python `str.lstrip()`: drops leading characters for which `str.isspace()` is true.
fn lstripPy(s: []const u8) []const u8 {
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return s[i..];
        if (i + n > s.len) return s[i..];
        const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return s[i..];
        if (!isSpacePy(cp)) return s[i..];
        i += n;
    }
    return s[i..];
}

fn isSpacePy(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// `kev.api.option_text`: the name alone when the description is None or "", else `name: <render(desc)>`.
fn optionText(a: std.mem.Allocator, name: []const u8, desc: ?std.json.Value, budget: *Budget) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try put(a, &out, name, budget);
    if (desc) |d| if (!(d == .null or (d == .string and d.string.len == 0))) {
        try put(a, &out, ": ", budget);
        try render(a, &out, d, 0, budget);
    };
    return out.toOwnedSlice(a);
}

pub const Question = struct {
    /// Borrowed from the parsed request, like `names`: the request JSON outlives its Questions.
    id: []const u8,
    t: QType,
    instr: []u8,
    /// Option texts in request order; for a score question they are also the legend.
    options: [][]u8,
    /// Choice criteria names, borrowed from the parsed request (null for noul and score).
    names: ?[][]const u8 = null,

    pub fn deinit(self: *Question, a: std.mem.Allocator) void {
        a.free(self.instr);
        for (self.options) |o| a.free(o);
        a.free(self.options);
        if (self.names) |n| a.free(n);
    }
};

pub const Questions = struct {
    qs: []Question,

    /// `SystemOneRequest.questions`: a non-empty object of noul / choice / score questions, in request order.
    pub fn init(a: std.mem.Allocator, v: std.json.Value, max_questions: usize) !Questions {
        if (v != .object or v.object.count() == 0) return error.KevNoQuestions;
        if (v.object.count() > max_questions) return error.TooManyQuestions;
        var list: std.ArrayList(Question) = .empty;
        errdefer {
            for (list.items) |*q| q.deinit(a);
            list.deinit(a);
        }
        try list.ensureTotalCapacityPrecise(a, v.object.count());
        var budget: Budget = .{};
        var it = v.object.iterator();
        while (it.next()) |kv| list.appendAssumeCapacity(try parseQuestion(a, kv.key_ptr.*, kv.value_ptr.*, &budget));
        return .{ .qs = try list.toOwnedSlice(a) };
    }

    pub fn deinit(self: *Questions, a: std.mem.Allocator) void {
        for (self.qs) |*q| q.deinit(a);
        a.free(self.qs);
    }
};

fn parseQuestion(a: std.mem.Allocator, id: []const u8, v: std.json.Value, budget: *Budget) !Question {
    if (v != .object) return error.KevBadQuestion;
    const tv = v.object.get("type") orelse return error.KevBadType;
    if (tv != .string) return error.KevBadType;
    const t = std.meta.stringToEnum(QType, tv.string) orelse return error.KevBadType;
    const crit = v.object.get("criteria");
    // Shape and count first: an invalid question costs no rendering.
    const n_opts: usize = switch (t) {
        .noul => blk: {
            if (crit) |cv| if (cv != .null and cv != .object) return error.KevNoulCriteria;
            break :blk 2;
        },
        .choice => blk: {
            // kev takes {label: description}; a list of unique labels, as Laya
            // also accepts, is the same with no descriptions.
            const cv = crit orelse return error.KevChoiceCriteria;
            const n = switch (cv) {
                .object => |o| o.count(),
                .array => |l| l.items.len,
                else => return error.KevChoiceCriteria,
            };
            if (n == 0 or n > MAX_OPTIONS) return error.KevChoiceCriteria;
            if (cv == .array) for (cv.array.items, 0..) |x, i| {
                if (x != .string) return error.KevChoiceCriteria;
                for (cv.array.items[0..i]) |y| if (std.mem.eql(u8, x.string, y.string)) return error.KevChoiceCriteria;
            };
            break :blk n;
        },
        .score => blk: {
            const cv = crit orelse return error.KevScoreCriteria;
            if (cv != .array or cv.array.items.len == 0 or cv.array.items.len > MAX_OPTIONS) return error.KevScoreCriteria;
            break :blk cv.array.items.len;
        },
    };
    var opts: std.ArrayList([]u8) = .empty;
    errdefer {
        for (opts.items) |o| a.free(o);
        opts.deinit(a);
    }
    try opts.ensureTotalCapacityPrecise(a, n_opts);
    var names: ?[][]const u8 = null;
    errdefer if (names) |n| a.free(n);
    switch (t) {
        .noul => {
            const c: ?std.json.ObjectMap = if (crit) |cv| (if (cv == .object) cv.object else null) else null;
            opts.appendAssumeCapacity(try optionText(a, "no", if (c) |o| o.get("false") else null, budget));
            opts.appendAssumeCapacity(try optionText(a, "yes", if (c) |o| o.get("true") else null, budget));
        },
        .choice => {
            const n = try a.alloc([]const u8, n_opts);
            names = n;
            switch (crit.?) {
                .object => |o| {
                    var it = o.iterator();
                    var i: usize = 0;
                    while (it.next()) |kv| : (i += 1) {
                        n[i] = kv.key_ptr.*;
                        opts.appendAssumeCapacity(try optionText(a, kv.key_ptr.*, kv.value_ptr.*, budget));
                    }
                },
                .array => |l| for (l.items, n) |x, *name| {
                    name.* = x.string;
                    opts.appendAssumeCapacity(try optionText(a, x.string, null, budget));
                },
                else => unreachable,
            }
        },
        .score => for (crit.?.array.items) |x| {
            var o: std.ArrayList(u8) = .empty;
            errdefer o.deinit(a);
            try render(a, &o, x, 0, budget);
            opts.appendAssumeCapacity(try o.toOwnedSlice(a));
        },
    }
    var instr: std.ArrayList(u8) = .empty;
    errdefer instr.deinit(a);
    if (v.object.get("instructions")) |ins| try render(a, &instr, ins, 0, budget);
    // A lone \ud800-style escape survives parsing as WTF-8, which the tokenizer cannot read.
    if (!std.unicode.utf8ValidateSlice(instr.items)) return error.LoneSurrogate;
    for (opts.items) |o| if (!std.unicode.utf8ValidateSlice(o)) return error.LoneSurrogate;
    const options = try opts.toOwnedSlice(a);
    errdefer {
        for (options) |o| a.free(o);
        a.free(options);
    }
    return .{ .id = id, .t = t, .instr = try instr.toOwnedSlice(a), .options = options, .names = names };
}

/// kev.model.user_tokens: `<|name|>` in caller text becomes `<¦name¦>`, so caller text can never spell a marker.
pub fn escapeMarkers(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < text.len) {
        if (i + 1 < text.len and text[i] == '<' and text[i + 1] == '|') {
            var j = i + 2;
            while (j < text.len and (std.ascii.isAlphanumeric(text[j]) or text[j] == '_')) j += 1;
            if (j > i + 2 and j + 1 < text.len and text[j] == '|' and text[j + 1] == '>') {
                try out.appendSlice(a, "<\u{A6}");
                try out.appendSlice(a, text[i + 2 .. j]);
                try out.appendSlice(a, "\u{A6}>");
                i = j + 2;
                continue;
            }
        }
        try out.append(a, text[i]);
        i += 1;
    }
    return out.toOwnedSlice(a);
}

/// 400 text for a request Kev cannot read; errors shared with Laya's parser fall through to its messages.
pub fn errorMessage(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.KevNoQuestions => "'questions' must be a non-empty object keyed by question id",
        error.KevBadQuestion => "each question must be an object",
        error.KevBadType => "question 'type' must be one of choice, score, noul",
        error.KevChoiceCriteria => std.fmt.comptimePrint("choice 'criteria' must be an object or a list of unique labels, 1 to {d} options", .{MAX_OPTIONS}),
        error.KevScoreCriteria => std.fmt.comptimePrint("score 'criteria' must be a list of 1 to {d} levels", .{MAX_OPTIONS}),
        error.KevNoulCriteria => "noul 'criteria' must be an object with false/true descriptions, or null",
        error.KevRowTooLong => std.fmt.comptimePrint("a question with the state exceeds {d} tokens", .{MAX_ROW}),
        error.KevRenderTooLarge => std.fmt.comptimePrint("the state or the questions render to more than {d} bytes", .{MAX_RENDER_BYTES}),
        else => laya.errorMessage(err),
    };
}

// ── Answers (kev.api.to_answers) ──

/// kev.api._normalize: all zeros -> uniform.
fn normalized(p: []const f64, i: usize) f64 {
    var t: f64 = 0;
    for (p) |x| t += x;
    return if (t == 0) 1.0 / @as(f64, @floatFromInt(p.len)) else p[i] / t;
}

/// First index of the largest value, like Python's `max(range(n), key=p.__getitem__)`.
fn argmax(p: []const f64) usize {
    var best: usize = 0;
    for (p, 0..) |v, j| if (v > p[best]) {
        best = j;
    };
    return best;
}

/// (p̂_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty; one option -> 1.
pub fn choiceConfidence(p: []const f64) f64 {
    const k: f64 = @floatFromInt(p.len);
    if (p.len == 1) return 1.0;
    return (normalized(p, argmax(p)) - 1.0 / k) / (1.0 - 1.0 / k);
}

/// max(0, 1 - E|level - mode| / D), D = mean |i - (L-1)/2|; mode = first most likely level; one level -> 1.
pub fn scoreConfidence(p: []const f64) f64 {
    if (p.len == 1) return 1.0;
    const l: f64 = @floatFromInt(p.len);
    var d: f64 = 0;
    for (0..p.len) |i| d += @abs(@as(f64, @floatFromInt(i)) - (l - 1) / 2);
    d /= l;
    const mode: f64 = @floatFromInt(argmax(p));
    var e: f64 = 0;
    for (0..p.len) |i| e += normalized(p, i) * @abs(@as(f64, @floatFromInt(i)) - mode);
    return @max(0.0, 1.0 - e / d);
}

/// Python `round(x, 4)`: x·10⁴ rounded on its EXACT binary value, ties to even (0.03125 -> 0.0312), then the
/// decimal read back correctly rounded. Zig's decimal formatting rounds ties away from zero, so it cannot be used.
pub fn roundProb(x: f64) f64 {
    if (!std.math.isFinite(x) or x == 0) return x;
    const parts = std.math.frexp(@abs(x)); // |x| = f · 2^e, f in [0.5, 1)
    const m: u128 = @intFromFloat(std.math.ldexp(parts.significand, 53)); // exact: 53-bit integer
    const n = m * 10_000; // |x|·10⁴ = n · 2^(e-53), n < 2^67
    const shift = @as(i32, parts.exponent) - 53;
    const q: u128 = if (shift >= 0) n << @intCast(shift) else if (shift <= -127) 0 else blk: {
        const k: u7 = @intCast(-shift);
        const fl = n >> k;
        const rem = n - (fl << k);
        const half = @as(u128, 1) << (k - 1);
        break :blk if (rem > half or (rem == half and fl & 1 == 1)) fl + 1 else fl;
    };
    var buf: [48]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d}e-4", .{q}) catch unreachable;
    const r = std.fmt.parseFloat(f64, s) catch unreachable;
    return if (x < 0) -r else r;
}

fn appendProb(a: std.mem.Allocator, out: *std.ArrayList(u8), x: f64) !void {
    try laya.pyFloat(a, out, roundProb(x));
}

/// `"answers": {...}` for one request: probabilities per question in option order.
pub fn appendAnswers(a: std.mem.Allocator, out: *std.ArrayList(u8), qs: []const Question, probs: []const []const f64) !void {
    try out.appendSlice(a, "{");
    for (qs, probs, 0..) |q, p, qi| {
        if (qi > 0) try out.append(a, ',');
        try laya.wireString(a, out, q.id);
        try out.print(a, ":{{\"type\":\"{s}\"", .{@tagName(q.t)});
        switch (q.t) {
            .noul => {
                try out.appendSlice(a, ",\"noul\":");
                try appendProb(a, out, p[1]);
            },
            .choice => {
                try out.appendSlice(a, ",\"choice\":");
                try laya.wireString(a, out, q.names.?[argmax(p)]);
                try out.appendSlice(a, ",\"confidence\":");
                try appendProb(a, out, choiceConfidence(p));
                try out.appendSlice(a, ",\"probabilities\":{");
                for (q.names.?, p, 0..) |name, v, j| {
                    if (j > 0) try out.append(a, ',');
                    try laya.wireString(a, out, name);
                    try out.append(a, ':');
                    try appendProb(a, out, v);
                }
                try out.append(a, '}');
            },
            .score => {
                var s: f64 = 0;
                for (p, 0..) |v, j| s += @as(f64, @floatFromInt(j)) * v;
                try out.appendSlice(a, ",\"score\":");
                try appendProb(a, out, s);
                try out.appendSlice(a, ",\"legend\":{");
                for (q.options, 0..) |o, j| {
                    if (j > 0) try out.append(a, ',');
                    try out.print(a, "\"{d}\":", .{j});
                    try laya.wireString(a, out, o);
                }
                try out.appendSlice(a, "},\"probabilities\":{");
                for (p, 0..) |v, j| {
                    if (j > 0) try out.append(a, ',');
                    try out.print(a, "\"{d}\":", .{j});
                    try appendProb(a, out, v);
                }
                try out.appendSlice(a, "},\"confidence\":");
                try appendProb(a, out, scoreConfidence(p));
            },
        }
        try out.append(a, '}');
    }
    try out.append(a, '}');
}


// ── Engine ──

/// The five delimiters kev reuses from Qwen's vocabulary, in kev.model.SPECIAL order.
const DELIMITERS = [_][]const u8{ "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>" };
const Delims = struct { state: u32, q: u32, opt: u32, close: u32, decide: u32 };

/// One question's branch: tokens after the state, with its readout offsets inside the branch.
const Branch = struct {
    ids: []u32,
    decide: usize,
    closes: []usize,

    fn deinit(self: *Branch, a: std.mem.Allocator) void {
        a.free(self.ids);
        a.free(self.closes);
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    stream: S,
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: transformer_mod.Transformer,
    tok: tokenizer_mod.Tokenizer,
    head_weights: model_mod.Weights,
    q_w_t: A,
    k_w_t: A,
    head_dim: usize,
    /// 1 / (sqrt(head_dim) · temperature): kev's logits are scaled, then divided by the calibration temperature.
    logit_scale: f32,
    delims: Delims,

    /// A dir with `kev_config.json` beside a qwen3_5 `config.json` (tests/convert_kev_weights.py). The head and
    /// its config are Kev's own and are checked here; the base checkpoint loads like any other model.
    pub fn load(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, s: S) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.stream = s;

        const kc = try readKevConfig(io, allocator, dir);
        self.head_dim = kc.head_dim;
        self.logit_scale = kc.logit_scale;

        self.config = try model_mod.parseConfig(io, allocator, dir);
        errdefer self.config.deinit(allocator);
        // The branch restore below snapshots exactly the Qwen3.5 caches (KV + GatedDeltaNet).
        if (!std.mem.startsWith(u8, self.config.model_type, "qwen3_5")) return error.KevUnsupportedBase;

        self.tok = try tokenizer_mod.loadTokenizer(io, allocator, dir);
        errdefer self.tok.deinit();
        if (self.tok.definedVocabSize() > self.config.vocab_size) return error.KevVocabMismatch;
        self.delims = try resolveDelims(&self.tok, self.config.vocab_size);

        const head_path = try std.fmt.allocPrint(allocator, "{s}/kev_head.safetensors", .{dir});
        defer allocator.free(head_path);
        self.head_weights = try model_mod.loadWeightsSingleFile(allocator, head_path);
        errdefer self.head_weights.deinit();
        try checkHead(&self.head_weights, kc.head_dim, self.config.hidden_size);
        self.q_w_t = try laya.transposeAxes(self.head_weights.get("q.weight").?, &.{ 1, 0 }, s);
        errdefer free(self.q_w_t);
        self.k_w_t = try laya.transposeAxes(self.head_weights.get("k.weight").?, &.{ 1, 0 }, s);
        errdefer free(self.k_w_t);

        self.weights = try model_mod.loadModelWeights(io, allocator, dir, &self.config, false);
        errdefer self.weights.deinit();
        model_mod.resolveWeightPrefix(&self.config, &self.weights);
        self.xfm = try transformer_mod.Transformer.init(io, allocator, self.config, &self.weights);
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.xfm.deinit();
        self.weights.deinit();
        free(self.q_w_t);
        free(self.k_w_t);
        self.head_weights.deinit();
        self.tok.deinit();
        self.config.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn parseQuestions(_: *const Engine, a: std.mem.Allocator, questions: std.json.Value, max_questions: usize) !Questions {
        return Questions.init(a, questions, max_questions);
    }


    /// The response JSON for one request: `{"model", "answers", "usage"}`.
    pub fn predict(self: *Engine, a: std.mem.Allocator, model_id: []const u8, state: std.json.Value, questions: *const Questions, max_input_tokens: usize) ![]u8 {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(a);
        var budget: Budget = .{};
        try render(a, &text, state, 0, &budget);
        if (!std.unicode.utf8ValidateSlice(text.items)) return error.LoneSurrogate;
        var input_tokens: usize = 0;
        const probs = try self.score(a, text.items, questions.qs, max_input_tokens, &input_tokens);
        defer {
            for (probs) |p| a.free(p);
            a.free(probs);
        }
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        try out.appendSlice(a, "{\"model\":");
        try laya.wireString(a, &out, model_id);
        try out.appendSlice(a, ",\"answers\":");
        try appendAnswers(a, &out, questions.qs, probs);
        try out.print(a, ",\"usage\":{{\"input_tokens\":{d},\"output_tokens\":0}}}}", .{input_tokens});
        return out.toOwnedSlice(a);
    }

    /// Caller text through the marker escape, then the tokenizer (kev.model.user_tokens).
    fn userTokens(self: *const Engine, a: std.mem.Allocator, text: []const u8) ![]u32 {
        const esc = try escapeMarkers(a, text);
        defer a.free(esc);
        return self.tok.encode(a, esc);
    }

    /// kev.model.encode for one question: `<q> instr (<opt> option </opt>)* <decide>`.
    fn buildBranch(self: *const Engine, a: std.mem.Allocator, q: *const Question) !Branch {
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(a);
        var closes = try a.alloc(usize, q.options.len);
        errdefer a.free(closes);
        try ids.append(a, self.delims.q);
        const instr = try self.userTokens(a, q.instr);
        defer a.free(instr);
        try ids.appendSlice(a, instr);
        for (q.options, 0..) |o, i| {
            try ids.append(a, self.delims.opt);
            const t = try self.userTokens(a, o);
            defer a.free(t);
            try ids.appendSlice(a, t);
            closes[i] = ids.items.len;
            try ids.append(a, self.delims.close);
        }
        const decide = ids.items.len;
        try ids.append(a, self.delims.decide);
        return .{ .ids = try ids.toOwnedSlice(a), .decide = decide, .closes = closes };
    }

    /// Probabilities per question. The state runs once; each question then runs from a restored snapshot of
    /// the state's KV cache, GatedDeltaNet state and position, so questions never see each other.
    fn score(self: *Engine, a: std.mem.Allocator, state_text: []const u8, qs: []const Question, max_input_tokens: usize, input_tokens: *usize) ![][]f64 {
        const s = self.stream;
        var state_ids: std.ArrayList(u32) = .empty;
        defer state_ids.deinit(a);
        try state_ids.append(a, self.delims.state);
        const st = try self.userTokens(a, state_text);
        defer a.free(st);
        try state_ids.appendSlice(a, st[0..@min(st.len, MAX_STATE - 1)]);

        const branches = try a.alloc(Branch, qs.len);
        var built: usize = 0;
        defer {
            for (branches[0..built]) |*b| b.deinit(a);
            a.free(branches);
        }
        var total = state_ids.items.len;
        for (qs, 0..) |*q, i| {
            branches[i] = try self.buildBranch(a, q);
            built = i + 1;
            if (state_ids.items.len + branches[i].ids.len > MAX_ROW) return error.KevRowTooLong;
            total += branches[i].ids.len;
            if (total > max_input_tokens) return error.TooManyInputTokens;
        }
        input_tokens.* = total;

        var cache = try transformer_mod.KVCache.init(a, self.config.num_hidden_layers);
        defer cache.deinit();
        const entries = try a.alloc(transformer_mod.SSMCacheEntry, self.config.num_hidden_layers);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer {
            for (entries) |*e| {
                free(e.conv_state);
                free(e.ssm_state);
                transformer_mod.ssmFreeQsaState(e);
            }
            a.free(entries);
        }
        var offset: usize = 0;
        var ctx = self.xfm.defaultCtx();
        ctx.cache = &cache;
        ctx.moe_seq_offset = &offset;
        ctx.ssm_entries = entries;
        ctx.capture_hidden = null;
        ctx.skip_lm_head = true;

        const h_state = try self.forwardIds(&ctx, state_ids.items);
        free(h_state);
        var kv_snap = try cache.snapshot();
        defer kv_snap.deinit();
        const ssm_snaps = try a.alloc(transformer_mod.SSMCacheEntrySnapshot, entries.len);
        for (entries, ssm_snaps) |*e, *sn| sn.* = transformer_mod.ssmSnapshot(e);
        defer {
            for (ssm_snaps) |*sn| transformer_mod.ssmSnapshotDeinit(sn);
            a.free(ssm_snaps);
        }
        const state_offset = offset;

        const probs = try a.alloc([]f64, qs.len);
        var done: usize = 0;
        errdefer {
            for (probs[0..done]) |p| a.free(p);
            a.free(probs);
        }
        for (branches) |*b| {
            try cache.restore(&kv_snap);
            for (entries, ssm_snaps) |*e, *sn| try transformer_mod.ssmRestore(e, sn);
            offset = state_offset;
            const h = try self.forwardIds(&ctx, b.ids);
            defer free(h);
            probs[done] = try self.headProbs(a, h, b, s);
            done += 1;
        }
        return probs;
    }

    fn forwardIds(self: *Engine, ctx: *transformer_mod.ForwardCtx, ids: []const u32) !A {
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(ids.ptr, &shape, 2, .uint32);
        defer free(arr);
        const h = try self.xfm.forwardWith(ctx, arr);
        errdefer free(h);
        try mlx.check(mlx.mlx_array_eval(h));
        return h;
    }

    /// kev.model.PointerHead on one branch: z_i = k(h_close_i) · q(h_decide) / sqrt(P) / T, softmax in f32.
    fn headProbs(self: *Engine, a: std.mem.Allocator, h: A, b: *const Branch, s: S) ![]f64 {
        const k = b.closes.len;
        const hidden: c_int = @intCast(self.config.hidden_size);
        const rows_idx = try a.alloc(i32, k + 1);
        defer a.free(rows_idx);
        rows_idx[0] = @intCast(b.decide);
        for (b.closes, 1..) |c, i| rows_idx[i] = @intCast(c);
        const idx_shape = [_]c_int{@intCast(k + 1)};
        const idx = mlx.mlx_array_new_data(rows_idx.ptr, &idx_shape, 1, .int32);
        defer free(idx);
        const flat = try reshape(h, &[_]c_int{ -1, hidden }, s);
        defer free(flat);
        const picked = try take(flat, idx, 0, s);
        defer free(picked);
        const rows = try astype(picked, .float32, s);
        defer free(rows);
        const zf = try self.headLogits(rows, s);
        defer free(zf);
        var p = mlx.mlx_array_new();
        defer free(p);
        try mlx.check(mlx.mlx_softmax_axis(&p, zf, -1, true, s));
        try mlx.check(mlx.mlx_array_eval(p));
        const raw = mlx.mlx_array_data_float32(p) orelse return error.MlxError;
        const out = try a.alloc(f64, k);
        for (out, raw[0..k]) |*o, v| {
            if (!std.math.isFinite(v)) {
                a.free(out);
                return error.KevNonFinite;
            }
            o.* = v;
        }
        return out;
    }

    /// Scaled logits `[K]` from f32 readout rows `[1+K, H]`: row 0 at `<decide>`, then each `</opt>`.
    fn headLogits(self: *Engine, rows: A, s: S) !A {
        const n: c_int = mlx.mlx_array_shape(rows)[0];
        const all_q = try linear(rows, self.q_w_t, self.head_weights.get("q.bias").?, s);
        defer free(all_q);
        const all_k = try linear(rows, self.k_w_t, self.head_weights.get("k.bias").?, s);
        defer free(all_k);
        const first = mlx.mlx_array_new_int(0);
        defer free(first);
        const q_row = try take(all_q, first, 0, s); // [P]
        defer free(q_row);
        const qd = try reshape(q_row, &[_]c_int{ -1, 1 }, s); // [P, 1]
        defer free(qd);
        var rest = mlx.mlx_array_new();
        defer free(rest);
        try mlx.check(mlx.mlx_arange(&rest, 1, @floatFromInt(n), 1, .int32, s));
        const ko = try take(all_k, rest, 0, s); // [K, P]
        defer free(ko);
        const z = try matmul(ko, qd, s); // [K, 1]
        defer free(z);
        const sc = mlx.mlx_array_new_float(self.logit_scale);
        defer free(sc);
        var zs = mlx.mlx_array_new();
        defer free(zs);
        try mlx.check(mlx.mlx_multiply(&zs, z, sc, s));
        return reshape(zs, &[_]c_int{n - 1}, s);
    }
};

const KevConfig = struct { head_dim: usize, logit_scale: f32 };

fn readKevConfig(io: std.Io, a: std.mem.Allocator, dir: []const u8) !KevConfig {
    const path = try std.fmt.allocPrint(a, "{s}/kev_config.json", .{dir});
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const text = try rs.interface.allocRemaining(a, .limited(64 * 1024));
    defer a.free(text);
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return error.KevBadConfig;
    defer parsed.deinit();
    return parseKevConfig(parsed.value);
}

/// `kev_config.json` as tests/convert_kev_weights.py writes it; anything else is `error.KevBadConfig`.
fn parseKevConfig(v: std.json.Value) !KevConfig {
    if (v != .object) return error.KevBadConfig;
    const o = v.object;
    const fmt = o.get("format") orelse return error.KevBadConfig;
    const ver = o.get("format_version") orelse return error.KevBadConfig;
    if (fmt != .string or !std.mem.eql(u8, fmt.string, "kev") or ver != .integer or ver.integer != 1) return error.KevBadConfig;
    const hd = o.get("head_dim") orelse return error.KevBadConfig;
    if (hd != .integer or hd.integer <= 0 or hd.integer > 65536) return error.KevBadConfig;
    const tv = o.get("temperature") orelse return error.KevBadConfig;
    const t: f64 = switch (tv) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => return error.KevBadConfig,
    };
    if (!std.math.isFinite(t) or t <= 0 or t > 1e6) return error.KevBadConfig;
    const dv = o.get("delimiters") orelse return error.KevBadConfig;
    if (dv != .object) return error.KevBadConfig;
    const keys = [_][]const u8{ "state", "question", "option", "option_end", "decide" };
    for (keys, DELIMITERS) |key, want| {
        const d = dv.object.get(key) orelse return error.KevBadConfig;
        if (d != .string or !std.mem.eql(u8, d.string, want)) return error.KevBadConfig;
    }
    // The scale is used in f32: it must survive the narrowing (a 1e-300 temperature would make it infinite).
    const scale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(hd.integer))) * t));
    if (!std.math.isFinite(scale) or scale <= 0) return error.KevBadConfig;
    return .{ .head_dim = @intCast(hd.integer), .logit_scale = scale };
}

fn resolveDelims(tok: *const tokenizer_mod.Tokenizer, vocab: u32) !Delims {
    var ids: [DELIMITERS.len]u32 = undefined;
    for (DELIMITERS, &ids, 0..) |name, *id, i| {
        id.* = tok.specialTokenId(name) orelse tok.vocab.get(name) orelse return error.KevMissingDelimiter;
        if (id.* >= vocab) return error.KevMissingDelimiter;
        for (ids[0..i]) |prev| if (prev == id.*) return error.KevMissingDelimiter;
    }
    return .{ .state = ids[0], .q = ids[1], .opt = ids[2], .close = ids[3], .decide = ids[4] };
}

/// Shapes, dtype and finiteness of the pointer head, read on the host before any MLX op uses it.
fn checkHead(w: *const model_mod.Weights, head_dim: usize, hidden: u32) !void {
    const specs = [_]struct { name: []const u8, rows: usize, cols: ?usize }{
        .{ .name = "q.weight", .rows = head_dim, .cols = hidden },
        .{ .name = "k.weight", .rows = head_dim, .cols = hidden },
        .{ .name = "q.bias", .rows = head_dim, .cols = null },
        .{ .name = "k.bias", .rows = head_dim, .cols = null },
    };
    for (specs) |sp| {
        const t = w.get(sp.name) orelse return error.KevBadHead;
        if (mlx.mlx_array_dtype(t) != .float32) return error.KevBadHead;
        const nd = mlx.mlx_array_ndim(t);
        const shape = mlx.mlx_array_shape(t);
        if (nd != (if (sp.cols == null) @as(usize, 1) else 2) or shape[0] != sp.rows) return error.KevBadHead;
        if (sp.cols) |c| if (shape[1] != c) return error.KevBadHead;
        try mlx.check(mlx.mlx_array_eval(t));
        const data = mlx.mlx_array_data_float32(t) orelse return error.KevBadHead;
        for (data[0..mlx.mlx_array_size(t)]) |x| if (!std.math.isFinite(x)) return error.KevBadHead;
    }
}


// ── Tests ──

const testing = std.testing;

fn renderJson(a: std.mem.Allocator, json: []const u8) ![]u8 {
    var parsed = try laya.parseRequestJson(a, json);
    defer parsed.deinit();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var budget: Budget = .{};
    try render(a, &out, parsed.value, 0, &budget);
    return out.toOwnedSlice(a);
}

fn expectRender(json: []const u8, want: []const u8) !void {
    const got = try renderJson(testing.allocator, json);
    defer testing.allocator.free(got);
    try testing.expectEqualStrings(want, got);
}

test "kev: render follows kev.api.render, not JSON" {
    try expectRender("\"text\"", "text");
    try expectRender("null", "");
    try expectRender("true", "True");
    try expectRender("-0", "0");
    try expectRender("1.0", "1.0");
    try expectRender("1e5", "100000.0");
    try expectRender("0.00001", "1e-05");
    try expectRender("{\"a\": 1, \"b\": false}", "a: 1\nb: False");
    try expectRender("{\"a\": {\"b\": [1, 2]}}", "a:\n  b:\n    - 1\n    - 2");
    try expectRender("[\"x\", {\"k\": \"v\", \"n\": null}]", "- x\n- k: v\n  n: ");
    try expectRender("{\"e\": {}, \"l\": []}", "e:\n\nl:\n");
}

test "kev: list items lstrip Python whitespace from the item start only" {
    try expectRender("[\"\\u3000\\u00a0 lead\"]", "- lead");
    try expectRender("[[\"a\", \"b\"]]", "- - a\n  - b");
}

test "kev: escapeMarkers rewrites only complete <|name|> markers" {
    const a = testing.allocator;
    const got = try escapeMarkers(a, "<|fim_suffix|> ok <|im_start|>x <| a|> <|x-y|> <||> <|end");
    defer a.free(got);
    try testing.expectEqualStrings("<\u{A6}fim_suffix\u{A6}> ok <\u{A6}im_start\u{A6}>x <| a|> <|x-y|> <||> <|end", got);
}

test "kev: questions map noul, choice and score like kev.api.to_record" {
    const a = testing.allocator;
    var parsed = try laya.parseRequestJson(a,
        \\{"r": {"type": "choice", "instructions": "Which?", "criteria": {"billing": "refunds", "sales": null, "other": ""}},
        \\ "n": {"type": "noul", "criteria": {"true": "killed"}},
        \\ "s": {"type": "score", "instructions": {"q": "How bad?"}, "criteria": ["low", {"level": "high"}]}}
    );
    defer parsed.deinit();
    var qs = try Questions.init(a, parsed.value, 64);
    defer qs.deinit(a);
    try testing.expectEqual(@as(usize, 3), qs.qs.len);
    try testing.expectEqualStrings("Which?", qs.qs[0].instr);
    try testing.expectEqualStrings("billing: refunds", qs.qs[0].options[0]);
    try testing.expectEqualStrings("sales", qs.qs[0].options[1]);
    try testing.expectEqualStrings("other", qs.qs[0].options[2]);
    try testing.expectEqualStrings("", qs.qs[1].instr);
    try testing.expectEqualStrings("no", qs.qs[1].options[0]);
    try testing.expectEqualStrings("yes: killed", qs.qs[1].options[1]);
    try testing.expectEqualStrings("q: How bad?", qs.qs[2].instr);
    try testing.expectEqualStrings("level: high", qs.qs[2].options[1]);
}

test "kev: choice criteria as a list of labels reads like an object without descriptions" {
    // Laya takes both shapes on the same endpoint; a list is kev's {label: null}.
    const a = testing.allocator;
    var parsed = try laya.parseRequestJson(a,
        \\{"l": {"type": "choice", "criteria": ["billing", "sales"]},
        \\ "o": {"type": "choice", "criteria": {"billing": null, "sales": null}}}
    );
    defer parsed.deinit();
    var qs = try Questions.init(a, parsed.value, 64);
    defer qs.deinit(a);
    for (0..2) |i| {
        try testing.expectEqualStrings(qs.qs[1].options[i], qs.qs[0].options[i]);
        try testing.expectEqualStrings(qs.qs[1].names.?[i], qs.qs[0].names.?[i]);
    }
}

test "kev: malformed questions are named errors" {
    const a = testing.allocator;
    const cases = [_]struct { json: []const u8, err: anyerror }{
        .{ .json = "{}", .err = error.KevNoQuestions },
        .{ .json = "[]", .err = error.KevNoQuestions },
        .{ .json = "{\"q\": 1}", .err = error.KevBadQuestion },
        .{ .json = "{\"q\": {\"type\": \"rank\"}}", .err = error.KevBadType },
        .{ .json = "{\"q\": {\"type\": \"choice\", \"criteria\": [\"a\", 1]}}", .err = error.KevChoiceCriteria },
        .{ .json = "{\"q\": {\"type\": \"choice\", \"criteria\": [\"a\", \"a\"]}}", .err = error.KevChoiceCriteria },
        .{ .json = "{\"q\": {\"type\": \"choice\", \"criteria\": []}}", .err = error.KevChoiceCriteria },
        .{ .json = "{\"q\": {\"type\": \"choice\", \"criteria\": {}}}", .err = error.KevChoiceCriteria },
        .{ .json = "{\"q\": {\"type\": \"score\", \"criteria\": []}}", .err = error.KevScoreCriteria },
        .{ .json = "{\"q\": {\"type\": \"noul\", \"criteria\": [1]}}", .err = error.KevNoulCriteria },
        .{ .json = "{\"a\": {\"type\": \"noul\"}, \"b\": {\"type\": \"noul\"}}", .err = error.TooManyQuestions },
    };
    for (cases) |c| {
        var parsed = try laya.parseRequestJson(a, c.json);
        defer parsed.deinit();
        try testing.expectError(c.err, Questions.init(a, parsed.value, 1));
    }
    var wide: std.ArrayList(u8) = .empty;
    defer wide.deinit(a);
    try wide.appendSlice(a, "{\"q\": {\"type\": \"choice\", \"criteria\": {");
    for (0..MAX_OPTIONS + 1) |i| try wide.print(a, "{s}\"{d}\": null", .{ if (i > 0) ", " else "", i });
    try wide.appendSlice(a, "}}}");
    var parsed = try laya.parseRequestJson(a, wide.items);
    defer parsed.deinit();
    try testing.expectError(error.KevChoiceCriteria, Questions.init(a, parsed.value, 64));
}

test "kev: roundProb matches Python round(x, 4) on the exact binary value" {
    try testing.expectEqual(@as(f64, 0.0312), roundProb(0.03125)); // exact tie -> even
    try testing.expectEqual(@as(f64, 0.0001), roundProb(0.00015)); // 1.4999..e-4 in binary
    try testing.expectEqual(@as(f64, 0.0003), roundProb(0.00025)); // 2.5000..01e-4 in binary
    try testing.expectEqual(@as(f64, 0.9632), roundProb(0.96315005));
    try testing.expectEqual(@as(f64, 1.0), roundProb(0.99996));
    try testing.expectEqual(@as(f64, 0.0), roundProb(0.00004999));
    try testing.expectEqual(@as(f64, 0.9688), roundProb(0.96875)); // exact tie -> even
    try testing.expectEqual(@as(f64, 254.0), roundProb(253.99999));
    try testing.expectEqual(@as(f64, 0.0), roundProb(1e-300));
}

test "kev: confidences follow the TypeSafe reference formulas" {
    try testing.expectEqual(@as(f64, 1.0), choiceConfidence(&.{0.3}));
    try testing.expectEqual(@as(f64, 0.0), choiceConfidence(&.{ 0.25, 0.25, 0.25, 0.25 }));
    try testing.expectEqual(@as(f64, 0.0), choiceConfidence(&.{ 0, 0 })); // zeros -> uniform
    try testing.expectApproxEqAbs(@as(f64, 0.6), choiceConfidence(&.{ 0.8, 0.2 }), 1e-12);
    try testing.expectEqual(@as(f64, 1.0), scoreConfidence(&.{ 0, 1, 0 }));
    try testing.expectEqual(@as(f64, 0.0), scoreConfidence(&.{ 0.5, 0, 0.5 }));
    // L=3: D = (1 + 0 + 1)/3; mode 1; E = 0.2 + 0.2 -> 1 - 0.4 / (2/3) = 0.4
    try testing.expectApproxEqAbs(@as(f64, 0.4), scoreConfidence(&.{ 0.2, 0.6, 0.2 }), 1e-12);
}

test "kev: answers JSON carries Kev's fields, rounded like Python" {
    const a = testing.allocator;
    var parsed = try laya.parseRequestJson(a,
        \\{"r": {"type": "choice", "criteria": {"a": null, "b": null}}, "n": {"type": "noul"},
        \\ "s": {"type": "score", "criteria": ["lo", "hi"]}}
    );
    defer parsed.deinit();
    var qs = try Questions.init(a, parsed.value, 64);
    defer qs.deinit(a);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try appendAnswers(a, &out, qs.qs, &.{ &.{ 0.25, 0.75 }, &.{ 0.03125, 0.96875 }, &.{ 0.5, 0.5 } });
    try testing.expectEqualStrings(
        \\{"r":{"type":"choice","choice":"b","confidence":0.5,"probabilities":{"a":0.25,"b":0.75}},"n":{"type":"noul","noul":0.9688},"s":{"type":"score","score":0.5,"legend":{"0":"lo","1":"hi"},"probabilities":{"0":0.5,"1":0.5},"confidence":0.0}}
    , out.items);
}

test "kev: kev_config.json is refused unless it is exactly the converter's format" {
    const a = testing.allocator;
    const good =
        \\{"format": "kev", "format_version": 1, "head_dim": 256, "temperature": 2.4, "delimiters":
        \\ {"state": "<|fim_prefix|>", "question": "<|fim_middle|>", "option": "<|box_start|>", "option_end": "<|box_end|>", "decide": "<|fim_suffix|>"}}
    ;
    {
        var p = try std.json.parseFromSlice(std.json.Value, a, good, .{});
        defer p.deinit();
        const kc = try parseKevConfig(p.value);
        try testing.expectEqual(@as(usize, 256), kc.head_dim);
        try testing.expectApproxEqAbs(@as(f32, 1.0 / (16.0 * 2.4)), kc.logit_scale, 1e-7);
    }
    const bad = [_][]const u8{
        "[]",
        "{\"format\": \"laya\", \"format_version\": 1}",
        "{\"format\": \"kev\", \"format_version\": 2}",
        "{\"format\": \"kev\", \"format_version\": 1, \"head_dim\": 0, \"temperature\": 1}",
        "{\"format\": \"kev\", \"format_version\": 1, \"head_dim\": 256, \"temperature\": 0}",
        "{\"format\": \"kev\", \"format_version\": 1, \"head_dim\": 256, \"temperature\": -1}",
        // Positive and finite in f64, but the f32 scale would be infinite or zero.
        "{\"format\": \"kev\", \"format_version\": 1, \"head_dim\": 256, \"temperature\": 1e-300, \"delimiters\": {\"state\": \"<|fim_prefix|>\", \"question\": \"<|fim_middle|>\", \"option\": \"<|box_start|>\", \"option_end\": \"<|box_end|>\", \"decide\": \"<|fim_suffix|>\"}}",
        "{\"format\": \"kev\", \"format_version\": 1, \"head_dim\": 256, \"temperature\": 1e300, \"delimiters\": {\"state\": \"<|fim_prefix|>\", \"question\": \"<|fim_middle|>\", \"option\": \"<|box_start|>\", \"option_end\": \"<|box_end|>\", \"decide\": \"<|fim_suffix|>\"}}",
        "{\"format\": \"kev\", \"format_version\": 1, \"head_dim\": 256, \"temperature\": 1, \"delimiters\": {}}",
        "{\"format\": \"kev\", \"format_version\": 1, \"head_dim\": 256, \"temperature\": 1, \"delimiters\": {\"state\": \"<|im_start|>\", \"question\": \"<|fim_middle|>\", \"option\": \"<|box_start|>\", \"option_end\": \"<|box_end|>\", \"decide\": \"<|fim_suffix|>\"}}",
    };
    for (bad) |b| {
        var p = try std.json.parseFromSlice(std.json.Value, a, b, .{});
        defer p.deinit();
        try testing.expectError(error.KevBadConfig, parseKevConfig(p.value));
    }
}

fn numOf(v: std.json.Value) f64 {
    return switch (v) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => std.math.nan(f64),
    };
}

fn testIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

// KEV_TEST_MODEL = a tests/convert_kev_weights.py pack, KEV_FIXTURES = tests/dump_kev_fixtures.py output. Bars: exact
// token ids; head within 1e-3 on kev's rows; trunk rows cos >= 0.99, RMS within 1%; probabilities within 0.01; argmax
// equal unless kev's top two are within 0.01 (every difference is printed).
test "kev: oracle against kev's MLX backend (KEV_TEST_MODEL + KEV_FIXTURES)" {
    const model_dir = std.mem.sliceTo(std.c.getenv("KEV_TEST_MODEL") orelse return error.SkipZigTest, 0);
    const fix_dir = std.mem.sliceTo(std.c.getenv("KEV_FIXTURES") orelse return error.SkipZigTest, 0);
    const a = testing.allocator;
    const io = testIo();
    const s = mlx.gpuStream();
    const engine = try Engine.load(io, a, model_dir, s);
    defer engine.deinit();

    const path = try std.fmt.allocPrint(a, "{s}/cases.json", .{fix_dir});
    defer a.free(path);
    const text = try readTestFile(a, path);
    defer a.free(text);
    var fx = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer fx.deinit();

    var worst: f64 = 0;
    var flips: usize = 0;
    for (fx.value.object.get("cases").?.array.items) |c| {
        const name = c.object.get("name").?.string;
        const req = c.object.get("request").?.object;
        var qs = try Questions.init(a, req.get("questions").?, 64);
        defer qs.deinit(a);
        var st: std.ArrayList(u8) = .empty;
        defer st.deinit(a);
        var budget: Budget = .{};
        try render(a, &st, req.get("state").?, 0, &budget);
        try testing.expectEqualStrings(c.object.get("record").?.object.get("state").?.string, st.items);

        // Token layout: the packed kev encoding is the state followed by every branch.
        var state_ids: std.ArrayList(u32) = .empty;
        defer state_ids.deinit(a);
        try state_ids.append(a, engine.delims.state);
        const stt = try engine.userTokens(a, st.items);
        defer a.free(stt);
        try state_ids.appendSlice(a, stt);
        var packed_ids: std.ArrayList(u32) = .empty;
        defer packed_ids.deinit(a);
        try packed_ids.appendSlice(a, state_ids.items);
        var b0 = try engine.buildBranch(a, &qs.qs[0]);
        defer b0.deinit(a);
        for (qs.qs) |*q| {
            var b = try engine.buildBranch(a, q);
            defer b.deinit(a);
            try packed_ids.appendSlice(a, b.ids);
        }
        const want_ids = c.object.get("ids").?.array.items;
        try testing.expectEqual(want_ids.len, packed_ids.items.len);
        for (want_ids, packed_ids.items) |w, got| try testing.expectEqual(@as(u32, @intCast(w.integer)), got);

        // Head alone, on kev's own readout rows (question 0, row form).
        const logits_name = try std.fmt.allocPrint(a, "{s}/logits_{s}.npy", .{ fix_dir, name });
        defer a.free(logits_name);
        const want_z = try readNpyF32(a, logits_name);
        defer a.free(want_z);
        var head_err: f32 = 0;
        var trunk_cos: f64 = std.math.nan(f64); // stays NaN when the case has no saved hidden rows
        var trunk_rms: f64 = std.math.nan(f64);
        const hidden_name = try std.fmt.allocPrint(a, "{s}/hidden_{s}.npy", .{ fix_dir, name });
        defer a.free(hidden_name);
        const has_hidden = !std.mem.eql(u8, name, "wide"); // the one case dumped without hidden rows (size)
        if (!has_hidden) {} else if (readNpyF32(a, hidden_name)) |want_h| {
            defer a.free(want_h);
            const hsz: usize = engine.config.hidden_size;
            const rows_n: usize = want_h.len / hsz;
            const shape = [_]c_int{ @intCast(rows_n), @intCast(hsz) };
            const kev_rows = mlx.mlx_array_new_data(want_h.ptr, &shape, 2, .float32);
            defer free(kev_rows);
            const z = try engine.headLogits(kev_rows, s);
            defer free(z);
            try mlx.check(mlx.mlx_array_eval(z));
            for (mlx.mlx_array_data_float32(z).?[0 .. rows_n - 1], want_z) |g, w| head_err = @max(head_err, @abs(g - w));

            // Trunk alone: question 0 as one causal row from an empty cache, readout rows vs kev's.
            const got_h = try rowReadouts(engine, a, state_ids.items, &b0);
            defer a.free(got_h);
            var dot: f64 = 0;
            var nw: f64 = 0;
            var ng: f64 = 0;
            for (got_h, want_h) |g, w| {
                dot += @as(f64, g) * w;
                ng += @as(f64, g) * g;
                nw += @as(f64, w) * w;
            }
            trunk_cos = dot / @sqrt(ng * nw);
            trunk_rms = @sqrt(ng / nw);
        } else |err| return err;

        var ntok: usize = 0;
        const probs = try engine.score(a, st.items, qs.qs, std.math.maxInt(usize), &ntok);
        defer {
            for (probs) |p| a.free(p);
            a.free(probs);
        }
        var case_worst: f64 = 0;
        for (c.object.get("probs").?.array.items, probs, 0..) |want, got, qi| {
            const wp = try a.alloc(f64, want.array.items.len);
            defer a.free(wp);
            for (want.array.items, wp) |wv, *x| x.* = numOf(wv);
            for (wp, got) |w, g| case_worst = @max(case_worst, @abs(w - g));
            const wb = argmax(wp);
            const gb = argmax(got);
            if (wb != gb) {
                flips += 1;
                std.debug.print("kev oracle {s} q{d}: argmax {d} (kev) vs {d}, kev margin {d:.5}\n", .{ name, qi, wb, gb, wp[wb] - wp[gb] });
                try testing.expect(wp[wb] - wp[gb] <= 1e-2);
            }
        }
        worst = @max(worst, case_worst);
        std.debug.print("kev oracle {s}: max|dp| {d:.5}  head|dz| {d:.6}  trunk cos {d:.6} rms {d:.4}\n", .{ name, case_worst, head_err, trunk_cos, trunk_rms });
        try testing.expect(case_worst <= 1e-2);
        try testing.expect(head_err <= 1e-3);
        if (has_hidden) {
            try testing.expect(trunk_cos >= 0.99);
            try testing.expect(@abs(trunk_rms - 1.0) <= 0.01);
        }
    }
    std.debug.print("kev oracle: {d} cases, max |dp| {e}, argmax flips {d}\n", .{ fx.value.object.get("cases").?.array.items.len, worst, flips });
}

/// Question 0's readout rows `[1+K, H]` (f32, flattened) as ONE causal row from an empty cache: kev's row form,
/// the shape the hidden fixtures were dumped in.
fn rowReadouts(engine: *Engine, a: std.mem.Allocator, state_ids: []const u32, b: *const Branch) ![]f32 {
    const s = engine.stream;
    const row = try std.mem.concat(a, u32, &.{ state_ids, b.ids });
    defer a.free(row);
    var cache = try transformer_mod.KVCache.init(a, engine.config.num_hidden_layers);
    defer cache.deinit();
    const entries = try a.alloc(transformer_mod.SSMCacheEntry, engine.config.num_hidden_layers);
    for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    defer {
        for (entries) |*e| {
            free(e.conv_state);
            free(e.ssm_state);
            transformer_mod.ssmFreeQsaState(e);
        }
        a.free(entries);
    }
    var offset: usize = 0;
    var ctx = engine.xfm.defaultCtx();
    ctx.cache = &cache;
    ctx.moe_seq_offset = &offset;
    ctx.ssm_entries = entries;
    ctx.capture_hidden = null;
    ctx.skip_lm_head = true;
    const h = try engine.forwardIds(&ctx, row);
    defer free(h);
    const idx_host = try a.alloc(i32, b.closes.len + 1);
    defer a.free(idx_host);
    idx_host[0] = @intCast(state_ids.len + b.decide);
    for (b.closes, 1..) |cl, i| idx_host[i] = @intCast(state_ids.len + cl);
    const idx_shape = [_]c_int{@intCast(idx_host.len)};
    const idx = mlx.mlx_array_new_data(idx_host.ptr, &idx_shape, 1, .int32);
    defer free(idx);
    const flat = try reshape(h, &[_]c_int{ -1, @intCast(engine.config.hidden_size) }, s);
    defer free(flat);
    const picked = try take(flat, idx, 0, s);
    defer free(picked);
    const rows = try astype(picked, .float32, s);
    defer free(rows);
    try mlx.check(mlx.mlx_array_eval(rows));
    return a.dupe(f32, mlx.mlx_array_data_float32(rows).?[0..mlx.mlx_array_size(rows)]);
}

fn readTestFile(a: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = testIo();
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
}

/// Minimal `.npy` reader (tests only): little-endian float32, C order.
fn readNpyF32(a: std.mem.Allocator, path: []const u8) ![]f32 {
    const bytes = try readTestFile(a, path);
    defer a.free(bytes);
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const major = bytes[6];
    const header_len: usize = if (major == 1) std.mem.readInt(u16, bytes[8..10], .little) else std.mem.readInt(u32, bytes[8..12], .little);
    const data_start: usize = (if (major == 1) @as(usize, 10) else 12) + header_len;
    if (std.mem.indexOf(u8, bytes[0..data_start], "<f4") == null) return error.BadNpy;
    const out = try a.alloc(f32, (bytes.len - data_start) / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes[data_start .. data_start + out.len * 4]);
    return out;
}

test "kev: rendering is budgeted while it runs, nested copies included" {
    const a = testing.allocator;
    const nested = struct {
        fn body(al: std.mem.Allocator, levels: usize) !std.ArrayList(u8) {
            var b: std.ArrayList(u8) = .empty;
            try b.appendNTimes(al, '[', levels);
            try b.append(al, '[');
            for (0..1000) |i| try b.appendSlice(al, if (i == 0) "null" else ",null");
            try b.append(al, ']');
            try b.appendNTimes(al, ']', levels);
            return b;
        }
    }.body;
    // ~5 KB of JSON: 100 single-element lists around 1,000 nulls renders ~200 KB, re-copied at every level.
    var deep = try nested(a, 100);
    defer deep.deinit(a);
    var pd = try laya.parseRequestJson(a, deep.items);
    defer pd.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var b1: Budget = .{};
    try testing.expectError(error.KevRenderTooLarge, render(a, &out, pd.value, 0, &b1));
    // Ordinary nesting renders fine under the same budget.
    var shallow = try nested(a, 3);
    defer shallow.deinit(a);
    var ps = try laya.parseRequestJson(a, shallow.items);
    defer ps.deinit();
    out.clearRetainingCapacity();
    var b2: Budget = .{};
    try render(a, &out, ps.value, 0, &b2);
    try testing.expect(out.items.len > 1000);
}

fn parseQuestionsOnce(a: std.mem.Allocator, v: std.json.Value) !void {
    var qs = try Questions.init(a, v, 64);
    qs.deinit(a);
}

test "kev: question parsing frees everything on every allocation failure" {
    const a = testing.allocator;
    var parsed = try laya.parseRequestJson(a,
        \\{"r": {"type": "choice", "instructions": {"q": "Which?", "l": [1, 2]}, "criteria": {"billing": "refunds", "sales": ["a", "b"], "x": null}},
        \\ "n": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "money back", "false": ""}},
        \\ "s": {"type": "score", "instructions": "How bad?", "criteria": ["low", {"level": "high"}, [1, 2]]},
        \\ "l": {"type": "choice", "criteria": ["billing", "sales"]}}
    );
    defer parsed.deinit();
    // Refuse in-place resizes so every toOwnedSlice allocates and can fail.
    var no_remap = std.testing.FailingAllocator.init(a, .{ .resize_fail_index = 0 });
    try testing.checkAllAllocationFailures(no_remap.allocator(), parseQuestionsOnce, .{parsed.value});
}

test "kev: a lone surrogate in text the model reads is refused by name" {
    const a = testing.allocator;
    var parsed = try laya.parseRequestJson(a, "{\"q\": {\"type\": \"noul\", \"instructions\": \"bad \\ud800 text\"}}");
    defer parsed.deinit();
    try testing.expectError(error.LoneSurrogate, Questions.init(a, parsed.value, 64));
    try testing.expect(errorMessage(error.LoneSurrogate) != null);
    try testing.expect(errorMessage(error.KevChoiceCriteria) != null);
}

test "kev: integer text is charged at its real length" {
    const a = testing.allocator;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    try body.appendSlice(a, "[");
    for (0..8) |i| {
        if (i > 0) try body.append(a, ',');
        try body.appendNTimes(a, '7', 4096);
    }
    try body.append(a, ']');
    var parsed = try laya.parseRequestJson(a, body.items);
    defer parsed.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var small: Budget = .{ .left = 16 * 1024 };
    try testing.expectError(error.KevRenderTooLarge, render(a, &out, parsed.value, 0, &small));
}
