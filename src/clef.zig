//! Cloudflare Clef joint schema decisions, matching the mlx-community MLX packs.
const std = @import("std");
const mlx = @import("mlx.zig");
const laya = @import("laya.zig");
const kev = @import("kev.zig");
const model = @import("model.zig");
const tokenizer = @import("tokenizer.zig");
const transformer = @import("transformer.zig");
const chat = @import("chat.zig");
const qwen_vision = @import("qwen_vision.zig");
const mrope = @import("mrope.zig");
const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const free = laya.free;
const V = std.json.Value;
const Allocator = std.mem.Allocator;

pub const MAX_LENGTH = 16384;
const MAX_OPTIONS = 255;
const MAX_RENDER_BYTES = 1 << 20;
const system_prompt = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const prefix = "<|im_start|>system\n" ++ system_prompt ++ "<|im_end|>\n<|im_start|>user\nSTATE:\n";
const suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";
pub const QType = enum { noul, choice, score };

fn less(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

fn json(a: Allocator, out: *std.ArrayList(u8), value: V, depth: usize) !void {
    if (depth > laya.MAX_JSON_DEPTH) return error.NestingTooDeep;
    switch (value) {
        .object => |obj| {
            const keys = try a.dupe([]const u8, obj.keys());
            defer a.free(keys);
            std.mem.sort([]const u8, keys, {}, less);
            try out.append(a, '{');
            for (keys, 0..) |key, i| {
                if (i != 0) try out.append(a, ',');
                try laya.wireString(a, out, key);
                try out.append(a, ':');
                try json(a, out, obj.get(key).?, depth + 1);
            }
            try out.append(a, '}');
        },
        .array => |arr| {
            try out.append(a, '[');
            for (arr.items, 0..) |v, i| {
                if (i != 0) try out.append(a, ',');
                try json(a, out, v, depth + 1);
            }
            try out.append(a, ']');
        },
        .string => |s| try laya.wireString(a, out, s),
        .number_string => |s| try laya.pyNumber(a, out, s),
        .float => |f| try laya.pyFloat(a, out, f),
        .integer => |i| try out.print(a, "{d}", .{i}),
        .bool => |b| try out.appendSlice(a, if (b) "true" else "false"),
        .null => try out.appendSlice(a, "null"),
    }
    if (out.items.len > MAX_RENDER_BYTES) return error.ClefRenderTooLarge;
}

fn render(a: Allocator, value: V) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    if (value == .string) try out.appendSlice(a, value.string) else try json(a, &out, value, 0);
    if (out.items.len > MAX_RENDER_BYTES) return error.ClefRenderTooLarge;
    if (!std.unicode.utf8ValidateSlice(out.items)) return error.LoneSurrogate;
    return out.toOwnedSlice(a);
}

fn truthy(v: V) bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .string => |s| s.len != 0,
        .array => |s| s.items.len != 0,
        .object => |s| s.count() != 0,
        .integer => |n| n != 0,
        .float => |n| n != 0,
        .number_string => |n| (std.fmt.parseFloat(f64, n) catch 1) != 0,
    };
}

pub const Option = struct { id: []u8, text: []u8, description: V, order: usize };
pub const Question = struct {
    id: []const u8,
    t: QType,
    instruction: []u8,
    options: []Option,

    fn deinit(self: *Question, a: Allocator) void {
        a.free(self.instruction);
        for (self.options) |o| {
            a.free(o.id);
            a.free(o.text);
        }
        a.free(self.options);
    }
};

pub const Questions = struct {
    qs: []Question,

    pub fn init(a: Allocator, v: V, max_questions: usize) !Questions {
        if (v != .object or v.object.count() == 0) return error.ClefNoQuestions;
        if (v.object.count() > max_questions) return error.TooManyQuestions;
        var qs: std.ArrayList(Question) = .empty;
        errdefer {
            for (qs.items) |*q| q.deinit(a);
            qs.deinit(a);
        }
        var it = v.object.iterator();
        var size: usize = 0;
        while (it.next()) |entry| {
            var q = try parseQuestion(a, entry.key_ptr.*, entry.value_ptr.*);
            errdefer q.deinit(a);
            size += q.id.len + q.instruction.len;
            for (q.options) |o| size += o.text.len;
            if (size > MAX_RENDER_BYTES) return error.ClefRenderTooLarge;
            try qs.append(a, q);
        }
        return .{ .qs = try qs.toOwnedSlice(a) };
    }

    pub fn deinit(self: *Questions, a: Allocator) void {
        for (self.qs) |*q| q.deinit(a);
        a.free(self.qs);
    }
};

fn makeOption(a: Allocator, id: []const u8, description: V, order: usize) !Option {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.append(a, '{');
    if (description != .null) {
        try text.appendSlice(a, "\"description\":");
        try json(a, &text, description, 0);
        try text.append(a, ',');
    }
    try text.appendSlice(a, "\"option_id\":");
    try laya.wireString(a, &text, id);
    try text.append(a, '}');
    if (!std.unicode.utf8ValidateSlice(text.items)) return error.LoneSurrogate;
    const owned_id = try a.dupe(u8, id);
    errdefer a.free(owned_id);
    return .{ .id = owned_id, .text = try text.toOwnedSlice(a), .description = description, .order = order };
}

fn optionLess(_: void, x: Option, y: Option) bool {
    return less({}, x.id, y.id);
}

fn parseQuestion(a: Allocator, id: []const u8, v: V) !Question {
    if (v != .object or id.len == 0 or !std.unicode.utf8ValidateSlice(id)) return error.ClefBadQuestion;
    const tv = v.object.get("type") orelse return error.ClefBadType;
    if (tv != .string) return error.ClefBadType;
    const t = std.meta.stringToEnum(QType, tv.string) orelse return error.ClefBadType;
    const criteria = v.object.get("criteria") orelse .null;
    switch (t) {
        .noul => if (criteria != .null and criteria != .object) {
            return error.ClefNoulCriteria;
        },
        .choice => if (criteria != .object or criteria.object.count() == 0 or criteria.object.count() > MAX_OPTIONS) {
            return error.ClefChoiceCriteria;
        },
        .score => if (criteria != .array or criteria.array.items.len == 0 or criteria.array.items.len > MAX_OPTIONS) {
            return error.ClefScoreCriteria;
        },
    }
    var instruction = v.object.get("instructions") orelse .null;
    if (!truthy(instruction)) instruction = .{ .string = id };
    const instr = try render(a, instruction);
    errdefer a.free(instr);
    var opts: std.ArrayList(Option) = .empty;
    errdefer {
        for (opts.items) |o| {
            a.free(o.id);
            a.free(o.text);
        }
        opts.deinit(a);
    }
    const count: usize = switch (t) {
        .noul => 2,
        .choice => criteria.object.count(),
        .score => criteria.array.items.len,
    };
    try opts.ensureTotalCapacityPrecise(a, count);
    switch (t) {
        .noul => {
            for ([_][]const u8{ "true", "false" }, [_][]const u8{ "The proposition is true or the answer is yes.", "The proposition is false or the answer is no." }, 0..) |key, desc, i| {
                const d = if (criteria == .object) criteria.object.get(key) orelse V{ .string = desc } else V{ .string = desc };
                opts.appendAssumeCapacity(try makeOption(a, key, d, i));
            }
        },
        .choice => {
            var it = criteria.object.iterator();
            while (it.next()) |entry| opts.appendAssumeCapacity(try makeOption(a, entry.key_ptr.*, entry.value_ptr.*, opts.items.len));
            std.mem.sort(Option, opts.items, {}, optionLess);
        },
        .score => for (criteria.array.items, 0..) |desc, i| {
            var buf: [32]u8 = undefined;
            opts.appendAssumeCapacity(try makeOption(a, try std.fmt.bufPrint(&buf, "{d}", .{i}), desc, i));
        },
    }
    return .{ .id = id, .t = t, .instruction = instr, .options = try opts.toOwnedSlice(a) };
}

const Span = struct { start: usize, end: usize };
const EncodedQuestion = struct { instruction: Span, options: []Span };
const Encoded = struct {
    ids: []u32,
    questions: []EncodedQuestion,
    fn deinit(self: *Encoded, a: Allocator) void {
        a.free(self.ids);
        for (self.questions) |q| a.free(q.options);
        a.free(self.questions);
    }
};

fn tokens(a: Allocator, tok: *const tokenizer.Tokenizer, out: *std.ArrayList(u32), text: []const u8) !void {
    const ids = try tok.encode(a, text);
    defer a.free(ids);
    try out.appendSlice(a, ids);
}

fn encode(a: Allocator, tok: *const tokenizer.Tokenizer, state: V, qs: []const Question, media_ids: []const u32, max_length: usize, truncate: bool) !Encoded {
    var schema: std.ArrayList(u32) = .empty;
    defer schema.deinit(a);
    var questions: std.ArrayList(EncodedQuestion) = .empty;
    errdefer {
        for (questions.items) |q| a.free(q.options);
        questions.deinit(a);
    }
    try tokens(a, tok, &schema, "\n\nSCHEMA FIELDS:\n");
    for (qs, 0..) |q, qi| {
        const label = try std.fmt.allocPrint(a, "\nFIELD {d}\nID: {s}\nTYPE: {s}\nINSTRUCTION: ", .{ qi + 1, q.id, @tagName(q.t) });
        defer a.free(label);
        try tokens(a, tok, &schema, label);
        const start = schema.items.len;
        try tokens(a, tok, &schema, q.instruction);
        const instruction = Span{ .start = start, .end = schema.items.len };
        if (instruction.start == instruction.end) return error.ClefBadQuestion;
        try tokens(a, tok, &schema, "\nALLOWED OPTIONS:\n");
        const options = try a.alloc(Span, q.options.len);
        errdefer a.free(options);
        for (q.options, 0..) |o, oi| {
            var buf: [64]u8 = undefined;
            try tokens(a, tok, &schema, try std.fmt.bufPrint(&buf, "OPTION {d}: ", .{oi + 1}));
            const os = schema.items.len;
            try tokens(a, tok, &schema, o.text);
            options[oi] = .{ .start = os, .end = schema.items.len };
            try tokens(a, tok, &schema, "\n");
        }
        try tokens(a, tok, &schema, "END FIELD\n");
        if (schema.items.len > max_length) return error.TooManyInputTokens;
        try questions.append(a, .{ .instruction = instruction, .options = options });
    }
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);
    try tokens(a, tok, &ids, prefix);
    try ids.appendSlice(a, media_ids);
    const end = try tok.encode(a, suffix);
    defer a.free(end);
    const fixed = ids.items.len + schema.items.len + end.len;
    if (fixed > max_length) return error.TooManyInputTokens;
    const state_text = try render(a, state);
    defer a.free(state_text);
    const state_ids = try tok.encode(a, state_text);
    defer a.free(state_ids);
    if (!truncate and fixed + state_ids.len > max_length) return error.TooManyInputTokens;
    try ids.appendSlice(a, state_ids[0..@min(state_ids.len, max_length - fixed)]);
    const offset = ids.items.len;
    for (questions.items) |*q| {
        q.instruction.start += offset;
        q.instruction.end += offset;
        for (q.options) |*o| {
            o.start += offset;
            o.end += offset;
        }
    }
    try ids.appendSlice(a, schema.items);
    try ids.appendSlice(a, end);
    const owned_q = try questions.toOwnedSlice(a);
    errdefer {
        for (owned_q) |q| a.free(q.options);
        a.free(owned_q);
    }
    return .{ .ids = try ids.toOwnedSlice(a), .questions = owned_q };
}

fn appendAnswers(a: Allocator, out: *std.ArrayList(u8), qs: []const Question, probabilities: []const []const f64) !void {
    try out.append(a, '{');
    for (qs, probabilities, 0..) |q, probs, qi| {
        if (qi != 0) try out.append(a, ',');
        try laya.wireString(a, out, q.id);
        try out.print(a, ":{{\"type\":\"{s}\",", .{@tagName(q.t)});
        if (q.t == .noul) {
            try out.print(a, "\"noul\":{d}}}", .{kev.roundProb(probs[0])});
            continue;
        }
        var best: usize = 0;
        for (probs, 0..) |p, i| if (p > probs[best] or (p == probs[best] and q.options[i].order < q.options[best].order)) {
            best = i;
        };
        if (q.t == .choice) {
            try out.appendSlice(a, "\"choice\":");
            try laya.wireString(a, out, q.options[best].id);
        } else {
            var score: f64 = 0;
            for (probs, 0..) |p, i| score += @as(f64, @floatFromInt(i)) * p;
            try out.print(a, "\"score\":{d},\"legend\":{{", .{kev.roundProb(score)});
            for (q.options, 0..) |o, i| {
                if (i != 0) try out.append(a, ',');
                try laya.wireString(a, out, o.id);
                try out.append(a, ':');
                try json(a, out, o.description, 0);
            }
            try out.append(a, '}');
        }
        try out.print(a, ",\"confidence\":{d},\"probabilities\":{{", .{kev.roundProb(probs[best])});
        for (0..q.options.len) |order| {
            for (q.options, 0..) |o, i| if (o.order == order) {
                if (order != 0) try out.append(a, ',');
                try laya.wireString(a, out, o.id);
                try out.print(a, ":{d}", .{kev.roundProb(probs[i])});
                break;
            };
        }
        try out.appendSlice(a, "}}");
    }
    try out.append(a, '}');
}

const HeadConfig = struct {
    hidden_size: c_int,
    width: c_int,
    routing_layers: usize,
    layers: usize,
    heads: c_int,
    feedforward: c_int,

    fn validate(self: HeadConfig, hidden: u32) !void {
        if (self.hidden_size != hidden or self.width <= 0 or self.width > 8192 or self.heads <= 0 or @rem(self.width, self.heads) != 0 or
            self.feedforward <= 0 or self.feedforward > 65536 or self.routing_layers > 32 or self.layers == 0 or self.layers > 32) return error.ClefBadHeadConfig;
    }
};

// Keep temporary array handles for one head evaluation; the backbone's caches have already been released.
const Ops = struct {
    a: Allocator,
    s: S,
    arrays: std.ArrayList(A) = .empty,

    fn deinit(self: *Ops) void {
        for (self.arrays.items) |x| free(x);
        self.arrays.deinit(self.a);
    }
    fn own(self: *Ops, x: A) !A {
        errdefer free(x);
        try self.arrays.append(self.a, x);
        return x;
    }
    fn binary(self: *Ops, comptime f: anytype, x: A, y: A) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(f(&out, x, y, self.s));
        try self.arrays.append(self.a, out);
        return out;
    }
    fn unary(self: *Ops, comptime f: anytype, x: A) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(f(&out, x, self.s));
        try self.arrays.append(self.a, out);
        return out;
    }
    fn scalar(self: *Ops, x: A, value: f32) !A {
        const f = mlx.mlx_array_new_float(value);
        defer free(f);
        return self.own(try laya.astype(f, mlx.mlx_array_dtype(x), self.s));
    }
    fn add(self: *Ops, x: A, y: A) !A {
        return self.binary(mlx.mlx_add, x, y);
    }
    fn mul(self: *Ops, x: A, y: A) !A {
        return self.binary(mlx.mlx_multiply, x, y);
    }
    fn reshape(self: *Ops, x: A, shape: []const c_int) !A {
        return self.own(try laya.reshape(x, shape, self.s));
    }
    fn transpose(self: *Ops, x: A, axes: []const c_int) !A {
        return self.own(try laya.transposeAxes(x, axes, self.s));
    }
    fn rows(self: *Ops, x: A, start: usize, end: usize) !A {
        const shape = mlx.getShape(x);
        var stops: [4]c_int = undefined;
        @memcpy(stops[0..shape.len], shape);
        stops[0] = @intCast(end);
        const starts = [_]c_int{ @intCast(start), 0, 0, 0 };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_slice(&out, x, &starts, shape.len, &stops, shape.len, &strides, shape.len, self.s));
        try self.arrays.append(self.a, out);
        return out;
    }
    fn reduce(self: *Ops, comptime f: anytype, x: A, axis: c_int) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(f(&out, x, axis, true, self.s));
        try self.arrays.append(self.a, out);
        return out;
    }
    fn meanSpan(self: *Ops, x: A, span: Span) !A {
        return self.reduce(mlx.mlx_mean_axis, try self.rows(x, span.start, span.end), 0);
    }
    fn cat(self: *Ops, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, axis, self.s));
        try self.arrays.append(self.a, out);
        return out;
    }
    fn softmax(self: *Ops, x: A, axis: c_int) !A {
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_softmax_axis(&out, x, axis, true, self.s));
        try self.arrays.append(self.a, out);
        return out;
    }
    fn norm(self: *Ops, x: A) !A {
        return self.unary(mlx.mlx_sqrt, try self.reduce(mlx.mlx_sum_axis, try self.mul(x, x), -1));
    }
    fn normalize(self: *Ops, x: A) !A {
        const denom = try self.binary(mlx.mlx_maximum, try self.norm(x), try self.scalar(x, 1e-12));
        return self.binary(mlx.mlx_divide, x, denom);
    }
};

fn weight(weights: *const model.Weights, base: []const u8, suffix_: []const u8) !A {
    var buf: [256]u8 = undefined;
    return weights.get(try std.fmt.bufPrint(&buf, "{s}{s}", .{ base, suffix_ })) orelse error.ClefMissingHeadWeight;
}

fn checkWeight(weights: *const model.Weights, base: []const u8, suffix_: []const u8, shape: []const c_int) !void {
    if (!std.mem.eql(c_int, mlx.getShape(try weight(weights, base, suffix_)), shape)) return error.ClefBadHeadWeight;
}

fn checkNorm(weights: *const model.Weights, base: []const u8, width: c_int) !void {
    try checkWeight(weights, base, ".weight", &.{width});
    try checkWeight(weights, base, ".bias", &.{width});
}

fn checkLinear(weights: *const model.Weights, base: []const u8, input: c_int, output: c_int, bias: bool) !void {
    try checkWeight(weights, base, ".weight", &.{ output, input });
    if (bias) try checkWeight(weights, base, ".bias", &.{output});
}

fn checkAttention(weights: *const model.Weights, base: []const u8, width: c_int) !void {
    try checkWeight(weights, base, ".in_proj_weight", &.{ 3 * width, width });
    try checkWeight(weights, base, ".in_proj_bias", &.{3 * width});
    try checkWeight(weights, base, ".out_proj.weight", &.{ width, width });
    try checkWeight(weights, base, ".out_proj.bias", &.{width});
}

fn checkHead(weights: *const model.Weights, c: HeadConfig) !void {
    try checkNorm(weights, "hidden_norm", c.hidden_size);
    for ([_][]const u8{ "memory_projection", "question_projection", "option_question_projection", "global_projection", "option_context_projection", "option_lexical_projection" }) |name| try checkLinear(weights, name, c.hidden_size, c.width, false);
    try checkWeight(weights, "type_embedding", ".weight", &.{ 3, c.width });
    for ([_][]const u8{ "option_summary_norm", "field_norm", "option_norm" }) |name| try checkNorm(weights, name, c.width);
    try checkLinear(weights, "residual_scorer.0", 4 * c.width, c.width, true);
    try checkLinear(weights, "residual_scorer.3", c.width, 1, true);
    for ([_][]const u8{ "prior_logit_scale", "joint_logit_scale", "residual_gate" }) |name| try checkWeight(weights, name, "", &.{});
    var buf: [256]u8 = undefined;
    for (0..c.routing_layers) |i| {
        for ([_][]const u8{ "query_norm", "memory_norm", "feedforward_norm" }) |name| try checkNorm(weights, try std.fmt.bufPrint(&buf, "evidence_layers.{d}.{s}", .{ i, name }), c.width);
        try checkAttention(weights, try std.fmt.bufPrint(&buf, "evidence_layers.{d}.attention", .{i}), c.width);
        try checkLinear(weights, try std.fmt.bufPrint(&buf, "evidence_layers.{d}.feedforward.0", .{i}), c.width, c.feedforward, true);
        try checkLinear(weights, try std.fmt.bufPrint(&buf, "evidence_layers.{d}.feedforward.3", .{i}), c.feedforward, c.width, true);
    }
    for (0..c.layers) |i| {
        for ([_][]const u8{ "norm1", "norm2", "norm3" }) |name| try checkNorm(weights, try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, name }), c.width);
        for ([_][]const u8{ "self_attn", "multihead_attn" }) |name| try checkAttention(weights, try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, name }), c.width);
        try checkLinear(weights, try std.fmt.bufPrint(&buf, "layers.{d}.linear1", .{i}), c.width, c.feedforward, true);
        try checkLinear(weights, try std.fmt.bufPrint(&buf, "layers.{d}.linear2", .{i}), c.feedforward, c.width, true);
    }
}

const Head = struct {
    weights: *const model.Weights,
    config: HeadConfig,
    ops: *Ops,
    gelu: mlx.mlx_closure,

    fn linear(self: Head, x: A, name: []const u8) !A {
        const wt = try self.ops.transpose(try weight(self.weights, name, ".weight"), &.{ 1, 0 });
        const bias = weight(self.weights, name, ".bias") catch null;
        if (bias) |b| {
            var out = mlx.mlx_array_new();
            errdefer free(out);
            try mlx.check(mlx.mlx_addmm(&out, b, x, wt, 1, 1, self.ops.s));
            try self.ops.arrays.append(self.ops.a, out);
            return out;
        }
        return self.ops.binary(mlx.mlx_matmul, x, wt);
    }
    fn norm(self: Head, x: A, name: []const u8) !A {
        return self.ops.own(try laya.layerNorm(x, try weight(self.weights, name, ".weight"), try weight(self.weights, name, ".bias"), 1e-5, self.ops.s));
    }
    fn attention(self: Head, query: A, memory: A, name: []const u8) !A {
        const o = self.ops;
        const w = try weight(self.weights, name, ".in_proj_weight");
        const b = try weight(self.weights, name, ".in_proj_bias");
        const d = self.config.width;
        const nh = self.config.heads;
        var qkv: [3]A = undefined;
        for (0..3) |i| {
            const x = if (i == 0) query else memory;
            const wt = try o.transpose(try o.rows(w, i * @as(usize, @intCast(d)), (i + 1) * @as(usize, @intCast(d))), &.{ 1, 0 });
            const bi = try o.rows(b, i * @as(usize, @intCast(d)), (i + 1) * @as(usize, @intCast(d)));
            const projected = try o.own(try laya.linear(x, wt, bi, o.s));
            qkv[i] = try o.transpose(try o.reshape(projected, &.{ 1, mlx.getShape(x)[0], nh, @divExact(d, nh) }), &.{ 0, 2, 1, 3 });
        }
        var attended = mlx.mlx_array_new();
        {
            errdefer free(attended);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attended, qkv[0], qkv[1], qkv[2], 1.0 / @sqrt(@as(f32, @floatFromInt(@divExact(d, nh)))), "", .{}, .{}, false, o.s));
            try o.arrays.append(o.a, attended);
        }
        const flat = try o.reshape(try o.transpose(attended, &.{ 0, 2, 1, 3 }), &.{ mlx.getShape(query)[0], d });
        const wt = try o.transpose(try weight(self.weights, name, ".out_proj.weight"), &.{ 1, 0 });
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_addmm(&out, try weight(self.weights, name, ".out_proj.bias"), flat, wt, 1, 1, o.s));
        try o.arrays.append(o.a, out);
        return out;
    }
    fn feedforward(self: Head, x: A, first: []const u8, second: []const u8) !A {
        const input = mlx.mlx_vector_array_new_data(&.{try self.linear(x, first)}, 1);
        defer _ = mlx.mlx_vector_array_free(input);
        var output = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(output);
        try mlx.check(mlx.mlx_closure_apply(&output, self.gelu, input));
        var g = mlx.mlx_array_new();
        {
            errdefer free(g);
            try mlx.check(mlx.mlx_vector_array_get(&g, output, 0));
            try self.ops.arrays.append(self.ops.a, g);
        }
        return self.linear(g, second);
    }
};

fn geluClosure(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const s: *S = @ptrCast(@alignCast(payload.?));
    var x = mlx.mlx_array_new();
    defer free(x);
    mlx.check(mlx.mlx_vector_array_get(&x, input, 0)) catch return -1;
    var o = Ops{ .a = std.heap.page_allocator, .s = s.* };
    defer o.deinit();
    const scaled = o.binary(mlx.mlx_divide, x, o.scalar(x, @sqrt(@as(f32, 2))) catch return -1) catch return -1;
    const cdf = o.add(o.unary(mlx.mlx_erf, scaled) catch return -1, o.scalar(x, 1) catch return -1) catch return -1;
    const result = o.binary(mlx.mlx_divide, o.mul(x, cdf) catch return -1, o.scalar(x, 2) catch return -1) catch return -1;
    res.* = mlx.mlx_vector_array_new_data(&.{result}, 1);
    return 0;
}

pub const Engine = struct {
    allocator: Allocator,
    stream: S,
    config: model.ModelConfig,
    weights: model.Weights,
    head_weights: model.Weights,
    head_config: HeadConfig,
    xfm: transformer.Transformer,
    tok: tokenizer.Tokenizer,
    vision: ?qwen_vision.QwenVision = null,
    gelu: mlx.mlx_closure,

    pub fn load(io: std.Io, a: Allocator, dir: []const u8, s: S) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        self.allocator = a;
        self.stream = s;
        self.config = try model.parseConfig(io, a, dir);
        errdefer self.config.deinit(a);
        if (!std.mem.startsWith(u8, self.config.model_type, "qwen3_5") or self.config.isMoe() or self.config.hadamard_block != 0) return error.ClefUnsupportedBase;
        const cfg_path = try std.fmt.allocPrint(a, "{s}/joint_head_config.json", .{dir});
        defer a.free(cfg_path);
        const f = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
        defer f.close(io);
        var rb: [4096]u8 = undefined;
        var reader = f.reader(io, &rb);
        const bytes = try reader.interface.allocRemaining(a, .limited(65536));
        defer a.free(bytes);
        const parsed = try std.json.parseFromSlice(HeadConfig, a, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        self.head_config = parsed.value;
        try self.head_config.validate(self.config.hidden_size);
        const head_path = try std.fmt.allocPrint(a, "{s}/joint_head.safetensors", .{dir});
        defer a.free(head_path);
        self.head_weights = try model.loadWeightsSingleFile(a, head_path);
        errdefer self.head_weights.deinit();
        try checkHead(&self.head_weights, self.head_config);
        var it = self.head_weights.map.valueIterator();
        while (it.next()) |w| {
            const cast = try laya.astype(w.*, .bfloat16, s);
            free(w.*);
            w.* = cast;
        }
        self.tok = try tokenizer.loadTokenizer(io, a, dir);
        errdefer self.tok.deinit();
        if (self.tok.definedVocabSize() > self.config.vocab_size) return error.ClefVocabMismatch;
        self.weights = try model.loadModelWeights(io, a, dir, &self.config, true);
        errdefer self.weights.deinit();
        model.resolveWeightPrefix(&self.config, &self.weights);
        self.xfm = try transformer.Transformer.init(io, a, self.config, &self.weights);
        errdefer self.xfm.deinit();
        self.xfm.compileGdnGate();
        const raw_gelu = mlx.mlx_closure_new_func_payload(&geluClosure, &self.stream, null);
        defer _ = mlx.mlx_closure_free(raw_gelu);
        self.gelu = .{};
        errdefer _ = mlx.mlx_closure_free(self.gelu);
        try mlx.check(mlx.mlx_compile(&self.gelu, raw_gelu, true));
        self.vision = if (self.config.qwen_vision and qwen_vision.resolveVisionPrefix(&self.weights) != null)
            try qwen_vision.QwenVision.init(a, self.config, &self.weights)
        else
            null;
        return self;
    }

    pub fn deinit(self: *Engine) void {
        _ = mlx.mlx_closure_free(self.gelu);
        if (self.vision) |*v| v.deinit();
        self.xfm.deinit();
        self.weights.deinit();
        self.head_weights.deinit();
        self.tok.deinit();
        self.config.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn parseQuestions(_: *const Engine, a: Allocator, questions: V, max_questions: usize) !Questions {
        return Questions.init(a, questions, max_questions);
    }

    fn imagePreproc(self: *const Engine) chat.VisionPreproc {
        var vp = @import("server.zig").visionPreprocFromConfig(&self.config);
        vp.composite_alpha = false;
        return vp;
    }

    pub fn prepareImages(self: *const Engine, a: Allocator, request: V) ![]chat.ImageData {
        for ([_][]const u8{ "videos", "media_kwargs" }) |key| if (request.object.get(key)) |v| {
            if (truthy(v)) return if (std.mem.eql(u8, key, "videos")) error.ClefVideosUnsupported else error.ClefMediaOptionsUnsupported;
        };
        const value = request.object.get("images") orelse return a.alloc(chat.ImageData, 0);
        if (value == .null) return a.alloc(chat.ImageData, 0);
        if (value != .array or value.array.items.len > 16) return error.ClefBadImages;
        if (value.array.items.len != 0 and self.vision == null) return error.ClefNoVision;
        var images: std.ArrayList(chat.ImageData) = .empty;
        errdefer {
            for (images.items) |im| a.free(im.pixels);
            images.deinit(a);
        }
        try images.ensureTotalCapacityPrecise(a, value.array.items.len);
        const server = @import("server.zig");
        const vp = self.imagePreproc();
        for (value.array.items) |item| {
            if (item != .string or item.string.len == 0) return error.ClefBadImages;
            const image = if (std.mem.startsWith(u8, item.string, "data:"))
                server.parseImageUrlContent(a, item.string, vp) orelse return error.ClefBadImages
            else blk: {
                if (std.mem.startsWith(u8, item.string, "http:") or std.mem.startsWith(u8, item.string, "https:")) return error.ClefImageUrlUnsupported;
                const decoder = std.base64.standard.Decoder;
                const bytes = try a.alloc(u8, decoder.calcSizeForSlice(item.string) catch return error.ClefBadImages);
                defer a.free(bytes);
                decoder.decode(bytes, item.string) catch return error.ClefBadImages;
                break :blk server.decodeImageToPixels(a, bytes, vp) orelse return error.ClefBadImages;
            };
            images.appendAssumeCapacity(image);
        }
        return images.toOwnedSlice(a);
    }

    fn mediaTokens(self: *const Engine, a: Allocator, images: []const chat.ImageData) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        const c = self.config;
        for (images) |im| {
            const n = @as(usize, im.grid_h) * im.grid_w / (@as(usize, c.qv_merge) * c.qv_merge);
            if (n == 0 or n > MAX_LENGTH or out.items.len + n + 2 > MAX_LENGTH) return error.TooManyInputTokens;
            try out.append(a, c.vision_start_token_id);
            try out.appendNTimes(a, c.image_token_id, n);
            try out.append(a, c.vision_end_token_id);
        }
        if (images.len != 0) try tokens(a, &self.tok, &out, "\n");
        return out.toOwnedSlice(a);
    }

    fn hidden(self: *Engine, a: Allocator, ids: []const u32, images: []const chat.ImageData) !A {
        var media_ops = Ops{ .a = a, .s = self.stream };
        defer media_ops.deinit();
        const grids = try a.alloc(mrope.ImageGrid, images.len);
        defer a.free(grids);
        const embeddings = try a.alloc(A, images.len);
        defer a.free(embeddings);
        for (images, 0..) |im, i| {
            const tower = if (self.vision) |*v| v else return error.ClefNoVision;
            const shape = [_]c_int{ @intCast(im.grid_h * im.grid_w), @intCast(3 * self.config.qv_temporal_patch * self.config.qv_patch * self.config.qv_patch) };
            const pixels = try media_ops.own(mlx.mlx_array_new_data(im.pixels.ptr, &shape, 2, .float32));
            embeddings[i] = try media_ops.own(try tower.forward(pixels, im.grid_h, im.grid_w));
            grids[i] = .{ .t = 1, .h = im.grid_h, .w = im.grid_w };
        }
        const positions = try a.alloc(i32, 3 * ids.len);
        defer a.free(positions);
        var delta: i32 = 0;
        {
            var ri = try mrope.getRopeIndex(a, ids, grids, &.{}, self.config.image_token_id, self.config.video_token_id, self.config.vision_start_token_id, self.config.qv_merge);
            defer ri.deinit();
            for (0..3) |axis| @memcpy(positions[axis * ids.len ..][0..ids.len], ri.pos[axis]);
            delta = ri.delta;
        }
        var cache = try transformer.KVCache.init(a, self.config.num_hidden_layers);
        defer cache.deinit();
        const entries = try a.alloc(transformer.SSMCacheEntry, self.config.num_hidden_layers);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer {
            for (entries) |*e| {
                free(e.conv_state);
                free(e.ssm_state);
                transformer.ssmFreeQsaState(e);
            }
            a.free(entries);
        }
        var offset: usize = 0;
        var ctx = self.xfm.defaultCtx();
        ctx.cache = &cache;
        ctx.ssm_entries = entries;
        ctx.moe_seq_offset = &offset;
        ctx.capture_hidden = null;
        ctx.skip_lm_head = true;
        ctx.mrope_pos = positions;
        ctx.mrope_total = ids.len;
        ctx.mrope_delta = delta;
        if (images.len != 0) {
            ctx.vision_embeddings = try media_ops.cat(embeddings, 1);
        }
        const input = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .uint32);
        defer free(input);
        const output = try self.xfm.forwardWith(&ctx, input);
        errdefer free(output);
        try mlx.check(mlx.mlx_array_eval(output));
        return output;
    }

    fn lexical(self: *Engine, o: *Ops, ids: []const u32) !A {
        const index = try o.own(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .uint32));
        const w = try o.own(try laya.take(self.xfm.lm_head_w, index, 0, o.s));
        if (self.xfm.lm_head_s.ctx == null) return w;
        const scales = try o.own(try laya.take(self.xfm.lm_head_s, index, 0, o.s));
        const biases = if (self.xfm.lm_head_b.ctx != null) try o.own(try laya.take(self.xfm.lm_head_b, index, 0, o.s)) else A{};
        const qp = transformer.computeQuantParams(&self.config, self.xfm.lm_head_w, self.xfm.lm_head_s, self.config.hidden_size);
        var out = mlx.mlx_array_new();
        errdefer free(out);
        try mlx.check(mlx.mlx_dequantize(&out, w, scales, biases, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), qp.mode.cstr(), .{}, .{ .value = .bfloat16, .has_value = true }, o.s));
        try o.arrays.append(o.a, out);
        return out;
    }

    fn score(self: *Engine, a: Allocator, enc: *const Encoded, qs: []const Question, images: []const chat.ImageData) ![][]f64 {
        const hidden_states = try self.hidden(a, enc.ids, images);
        defer free(hidden_states);
        return self.scoreHidden(a, enc, qs, hidden_states);
    }

    fn scoreHidden(self: *Engine, a: Allocator, enc: *const Encoded, qs: []const Question, hidden_states: A) ![][]f64 {
        var o = Ops{ .a = a, .s = self.stream };
        defer o.deinit();
        const head = Head{ .weights = &self.head_weights, .config = self.head_config, .ops = &o, .gelu = self.gelu };
        const h = try head.norm(try o.reshape(hidden_states, &.{ @intCast(enc.ids.len), @intCast(self.config.hidden_size) }), "hidden_norm");
        const memory = try head.linear(h, "memory_projection");
        const global = try o.rows(h, enc.ids.len - 1, enc.ids.len);
        const vectors = try a.alloc(A, qs.len);
        defer a.free(vectors);
        const lexical_options = try a.alloc(A, qs.len);
        defer a.free(lexical_options);
        const option_queries = try a.alloc(A, qs.len);
        defer a.free(option_queries);
        for (enc.questions, 0..) |q, i| {
            vectors[i] = try o.meanSpan(h, q.instruction);
            const contexts = try a.alloc(A, q.options.len);
            defer a.free(contexts);
            const lexical_rows = try a.alloc(A, q.options.len);
            defer a.free(lexical_rows);
            for (q.options, 0..) |span, j| {
                contexts[j] = try o.meanSpan(h, span);
                lexical_rows[j] = try o.reduce(mlx.mlx_mean_axis, try self.lexical(&o, enc.ids[span.start..span.end]), 0);
            }
            lexical_options[i] = try o.cat(lexical_rows, 0);
            option_queries[i] = try o.add(try o.add(try head.linear(try o.cat(contexts, 0), "option_context_projection"), try head.linear(lexical_options[i], "option_lexical_projection")), try head.linear(vectors[i], "option_question_projection"));
        }
        var routed = try o.cat(option_queries, 0);
        var b1: [256]u8 = undefined;
        var b2: [256]u8 = undefined;
        for (0..self.head_config.routing_layers) |i| {
            const m = try head.norm(memory, try std.fmt.bufPrint(&b1, "evidence_layers.{d}.memory_norm", .{i}));
            const q = try head.norm(routed, try std.fmt.bufPrint(&b1, "evidence_layers.{d}.query_norm", .{i}));
            routed = try o.add(routed, try head.attention(q, m, try std.fmt.bufPrint(&b1, "evidence_layers.{d}.attention", .{i})));
            const n = try head.norm(routed, try std.fmt.bufPrint(&b1, "evidence_layers.{d}.feedforward_norm", .{i}));
            routed = try o.add(routed, try head.feedforward(n, try std.fmt.bufPrint(&b1, "evidence_layers.{d}.feedforward.0", .{i}), try std.fmt.bufPrint(&b2, "evidence_layers.{d}.feedforward.3", .{i})));
        }
        const base_fields = try head.linear(try o.cat(vectors, 0), "question_projection");
        const options = try a.alloc(A, qs.len);
        defer a.free(options);
        const summaries = try a.alloc(A, qs.len);
        defer a.free(summaries);
        var offset: usize = 0;
        for (qs, 0..) |q, i| {
            options[i] = try o.rows(routed, offset, offset + q.options.len);
            offset += q.options.len;
            const field = try o.transpose(try o.rows(base_fields, i, i + 1), &.{ 1, 0 });
            const dots = try o.binary(mlx.mlx_matmul, options[i], field);
            const w = try o.softmax(try o.binary(mlx.mlx_divide, dots, try o.scalar(dots, @sqrt(@as(f32, @floatFromInt(self.head_config.width))))), 0);
            summaries[i] = try o.reduce(mlx.mlx_sum_axis, try o.mul(w, options[i]), 0);
        }
        const types = try a.alloc(u32, qs.len);
        defer a.free(types);
        for (qs, types) |q, *t| t.* = @backingInt(q.t);
        const type_ids = try o.own(mlx.mlx_array_new_data(types.ptr, &[_]c_int{@intCast(types.len)}, 1, .uint32));
        const type_embeddings = try o.own(try laya.take(try weight(&self.head_weights, "type_embedding", ".weight"), type_ids, 0, o.s));
        var fields = try o.add(try o.add(try o.add(base_fields, try head.norm(try o.cat(summaries, 0), "option_summary_norm")), try head.linear(global, "global_projection")), type_embeddings);
        for (0..self.head_config.layers) |i| {
            const norm1 = try head.norm(fields, try std.fmt.bufPrint(&b1, "layers.{d}.norm1", .{i}));
            fields = try o.add(fields, try head.attention(norm1, norm1, try std.fmt.bufPrint(&b1, "layers.{d}.self_attn", .{i})));
            const norm2 = try head.norm(fields, try std.fmt.bufPrint(&b1, "layers.{d}.norm2", .{i}));
            fields = try o.add(fields, try head.attention(norm2, memory, try std.fmt.bufPrint(&b1, "layers.{d}.multihead_attn", .{i})));
            const norm3 = try head.norm(fields, try std.fmt.bufPrint(&b1, "layers.{d}.norm3", .{i}));
            fields = try o.add(fields, try head.feedforward(norm3, try std.fmt.bufPrint(&b1, "layers.{d}.linear1", .{i}), try std.fmt.bufPrint(&b2, "layers.{d}.linear2", .{i})));
        }
        fields = try head.norm(fields, "field_norm");
        const prior_weight = try weight(&self.head_weights, "prior_logit_scale", "");
        const joint_weight = try weight(&self.head_weights, "joint_logit_scale", "");
        const max_log = try o.scalar(prior_weight, @log(@as(f32, 100)));
        const prior_scale = try o.unary(mlx.mlx_exp, try o.binary(mlx.mlx_minimum, prior_weight, max_log));
        const joint_scale = try o.unary(mlx.mlx_exp, try o.binary(mlx.mlx_minimum, joint_weight, max_log));
        const gate = try o.unary(mlx.mlx_sigmoid, try weight(&self.head_weights, "residual_gate", ""));
        var probs: std.ArrayList([]f64) = .empty;
        errdefer {
            for (probs.items) |p| a.free(p);
            probs.deinit(a);
        }
        try probs.ensureTotalCapacityPrecise(a, qs.len);
        for (qs, 0..) |q, i| {
            const anchor = try o.transpose(try o.normalize(try o.add(vectors[i], global)), &.{ 1, 0 });
            const prior = try o.mul(prior_scale, try o.binary(mlx.mlx_matmul, try o.normalize(lexical_options[i]), anchor));
            const opts = try head.norm(options[i], "option_norm");
            const field = try o.rows(fields, i, i + 1);
            var rf = mlx.mlx_array_new();
            {
                errdefer free(rf);
                try mlx.check(mlx.mlx_broadcast_to(&rf, field, mlx.getShape(opts).ptr, mlx.getShape(opts).len, o.s));
                try o.arrays.append(o.a, rf);
            }
            const prod = try o.mul(rf, opts);
            const denom = try o.binary(mlx.mlx_maximum, try o.mul(try o.norm(rf), try o.norm(opts)), try o.scalar(opts, 1e-8));
            const cosine = try o.binary(mlx.mlx_divide, try o.reduce(mlx.mlx_sum_axis, prod, -1), denom);
            const diff = try o.unary(mlx.mlx_abs, try o.binary(mlx.mlx_subtract, rf, opts));
            const features = try o.cat(&.{ rf, opts, prod, diff }, -1);
            const residual = try head.feedforward(features, "residual_scorer.0", "residual_scorer.3");
            const logits = try o.add(prior, try o.mul(gate, try o.add(try o.mul(joint_scale, cosine), residual)));
            const p = try o.softmax(try o.own(try laya.astype(logits, .float32, o.s)), 0);
            try mlx.check(mlx.mlx_array_eval(p));
            const values = mlx.mlx_array_data_float32(p).?[0..q.options.len];
            const result = try a.alloc(f64, values.len);
            for (values, result) |x, *y| y.* = x;
            probs.appendAssumeCapacity(result);
        }
        return probs.toOwnedSlice(a);
    }

    pub fn predict(self: *Engine, a: Allocator, model_id: []const u8, state: V, questions: *const Questions, max_input_tokens: usize, truncate: bool, images: []const chat.ImageData) ![]u8 {
        const media_ids = try self.mediaTokens(a, images);
        defer a.free(media_ids);
        var enc = try encode(a, &self.tok, state, questions.qs, media_ids, @min(MAX_LENGTH, max_input_tokens), truncate);
        defer enc.deinit(a);
        const probs = try self.score(a, &enc, questions.qs, images);
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
        try out.print(a, ",\"usage\":{{\"input_tokens\":{d},\"output_tokens\":0}}}}", .{enc.ids.len});
        return out.toOwnedSlice(a);
    }
};

pub fn errorMessage(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ClefNoQuestions => "questions must be a non-empty object",
        error.ClefBadQuestion => "each question must have a non-empty id and instruction",
        error.ClefBadType => "question type must be noul, choice, or score",
        error.ClefNoulCriteria => "noul criteria must be an object",
        error.ClefChoiceCriteria => "choice criteria must be an object with 1 to 255 options",
        error.ClefScoreCriteria => "score criteria must be an array with 1 to 255 levels",
        error.ClefRenderTooLarge => "rendered state or schema exceeds 1 MiB",
        error.ClefBadImages => "images must contain up to 16 valid base64 images or image data URLs",
        error.ClefNoVision => "this Clef checkpoint has no vision tower",
        error.ClefVideosUnsupported => "videos are not supported by the Clef HTTP API",
        error.ClefMediaOptionsUnsupported => "media_kwargs is not supported; images use the checkpoint's preprocessing settings",
        error.ClefImageUrlUnsupported => "remote image URLs are not supported; send a base64 image or data URL",
        else => null,
    };
}

test "clef: rendering, option ordering and confidence follow the joint schema contract" {
    const a = std.testing.allocator;
    const p = try laya.parseRequestJson(a, "{\"q\":{\"type\":\"choice\",\"criteria\":{\"z\":null,\"a\":{\"z\":2,\"a\":\"é\"}}},\"s\":{\"type\":\"score\",\"criteria\":[\"low\",\"high\"]}}");
    defer p.deinit();
    var qs = try Questions.init(a, p.value, 64);
    defer qs.deinit(a);
    try std.testing.expectEqualStrings("q", qs.qs[0].instruction);
    try std.testing.expectEqualStrings("{\"description\":{\"a\":\"é\",\"z\":2},\"option_id\":\"a\"}", qs.qs[0].options[0].text);
    try std.testing.expectEqualStrings("{\"option_id\":\"z\"}", qs.qs[0].options[1].text);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try appendAnswers(a, &out, qs.qs, &.{ &.{ 0.5, 0.5 }, &.{ 0.25, 0.75 } });
    const answer = try std.json.parseFromSlice(V, a, out.items, .{});
    defer answer.deinit();
    const q = answer.value.object.get("q").?.object;
    try std.testing.expectEqualStrings("z", q.get("choice").?.string);
    try std.testing.expectEqual(@as(f64, 0.5), q.get("confidence").?.float);
    const s = answer.value.object.get("s").?.object;
    try std.testing.expectEqual(@as(f64, 0.75), s.get("score").?.float);
    try std.testing.expectEqual(@as(f64, 0.75), s.get("confidence").?.float);
}

test "clef: reject malformed schemas before inference" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "{\"q\":{\"type\":\"choice\",\"criteria\":[]}}", "{\"q\":{\"type\":\"score\",\"criteria\":{}}}", "{\"q\":{\"type\":\"noul\",\"criteria\":2}}", "{\"q\":null}", "{}" }) |body| {
        const p = try laya.parseRequestJson(a, body);
        defer p.deinit();
        if (Questions.init(a, p.value, 64)) |result| {
            var qs = result;
            qs.deinit(a);
            return error.ExpectedInvalidSchema;
        } else |err| try std.testing.expect(errorMessage(err) != null);
    }
}

fn checkSchemaAllocation(a: Allocator) !void {
    const parsed = try laya.parseRequestJson(a, "{\"q\":{\"type\":\"choice\",\"criteria\":{\"a\":null,\"b\":\"second\"}},\"s\":{\"type\":\"score\",\"criteria\":[\"low\",\"high\"]}}");
    defer parsed.deinit();
    var qs = try Questions.init(a, parsed.value, 64);
    defer qs.deinit(a);
    var answer: std.ArrayList(u8) = .empty;
    defer answer.deinit(a);
    try appendAnswers(a, &answer, qs.qs, &.{ &.{ 0.25, 0.75 }, &.{ 0.75, 0.25 } });
}

test "clef: schema allocation failures release partial questions and answers" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, checkSchemaAllocation, .{});
}

test "clef: reject incompatible head geometry before allocating the model" {
    const valid = HeadConfig{ .hidden_size = 4096, .width = 1024, .routing_layers = 2, .layers = 4, .heads = 16, .feedforward = 4096 };
    try valid.validate(4096);
    try std.testing.expectError(error.ClefBadHeadConfig, valid.validate(5120));
    for ([_]c_int{ 0, -1, 3, 2048 }) |heads| {
        var invalid = valid;
        invalid.heads = heads;
        try std.testing.expectError(error.ClefBadHeadConfig, invalid.validate(4096));
    }
}

test "clef: live joint head agrees with the published MLX loader" {
    const dir = std.mem.span(std.c.getenv("CLEF_TEST_MODEL") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const fixture = @import("clef_http_test.zig");
    const engine = try Engine.load(io, a, dir, mlx.gpuStream());
    defer engine.deinit();
    const body = try fixture.readFile(io, "tests/fixtures/clef/request.json");
    defer a.free(body);
    const p = try laya.parseRequestJson(a, body);
    defer p.deinit();
    var qs = try Questions.init(a, p.value.object.get("questions").?, 64);
    defer qs.deinit(a);
    var enc = try encode(a, &engine.tok, p.value.object.get("state").?, qs.qs, &.{}, MAX_LENGTH, true);
    defer enc.deinit(a);
    var empty = try encode(a, &engine.tok, .{ .string = "" }, qs.qs, &.{}, MAX_LENGTH, true);
    defer empty.deinit(a);
    var truncated = try encode(a, &engine.tok, p.value.object.get("state").?, qs.qs, &.{}, empty.ids.len, true);
    defer truncated.deinit(a);
    try std.testing.expectEqualSlices(u32, empty.ids, truncated.ids);
    try std.testing.expectError(error.TooManyInputTokens, encode(a, &engine.tok, p.value.object.get("state").?, qs.qs, &.{}, enc.ids.len - 1, false));
    try std.testing.expectError(error.TooManyInputTokens, encode(a, &engine.tok, .{ .string = "" }, qs.qs, &.{}, empty.ids.len - 1, true));
    try std.testing.expectError(error.TooManyInputTokens, encode(a, &engine.tok, .{ .string = "" }, qs.qs, &.{}, 1, true));

    if (std.c.getenv("CLEF_TEST_REFERENCE")) |raw_path| {
        var reference = try model.loadWeightsSingleFile(a, std.mem.span(raw_path));
        defer reference.deinit();
        const ref_ids = try laya.astype(reference.get("ids").?, .uint32, engine.stream);
        defer free(ref_ids);
        try mlx.check(mlx.mlx_array_eval(ref_ids));
        try std.testing.expectEqual(enc.ids.len, @as(usize, @intCast(mlx.getShape(ref_ids)[0])));
        try std.testing.expectEqualSlices(u32, mlx.mlx_array_data_uint32(ref_ids).?[0..enc.ids.len], enc.ids);
        const scores = try engine.scoreHidden(a, &enc, qs.qs, reference.get("hidden").?);
        defer {
            for (scores) |v| a.free(v);
            a.free(scores);
        }
        for (scores, 0..) |values, i| {
            var key: [32]u8 = undefined;
            const expected = reference.get(try std.fmt.bufPrint(&key, "prob_{d}", .{i})).?;
            try mlx.check(mlx.mlx_array_eval(expected));
            for (values, mlx.mlx_array_data_float32(expected).?[0..values.len]) |actual, wanted| {
                try std.testing.expectApproxEqAbs(@as(f64, wanted), actual, 0.001);
            }
        }
    }
    const oracle_bytes = try fixture.readFile(io, fixture.oraclePath());
    defer a.free(oracle_bytes);
    const oracle = try std.json.parseFromSlice(V, a, oracle_bytes, .{});
    defer oracle.deinit();
    const expected = oracle.value.object.get(std.fs.path.basename(dir)) orelse return error.MissingReference;
    const response = try engine.predict(a, "clef", p.value.object.get("state").?, &qs, MAX_LENGTH, true, &.{});
    defer a.free(response);
    const result = try std.json.parseFromSlice(V, a, response, .{});
    defer result.deinit();
    try fixture.compare(expected.object.get("text").?.object.get("answers").?, result.value.object.get("answers").?);
    try fixture.compare(expected.object.get("text").?.object.get("usage").?, result.value.object.get("usage").?);

    const bytes = try fixture.readFile(io, "tests/fixtures/robot.png");
    defer a.free(bytes);
    const image = @import("server.zig").decodeImageToPixels(a, bytes, engine.imagePreproc()) orelse return error.BadImage;
    defer a.free(image.pixels);
    const request = try fixture.readFile(io, "tests/fixtures/clef/image-request.json");
    defer a.free(request);
    const image_req = try laya.parseRequestJson(a, request);
    defer image_req.deinit();
    var image_qs = try Questions.init(a, image_req.value.object.get("questions").?, 64);
    defer image_qs.deinit(a);
    const image_response = try engine.predict(a, "clef", image_req.value.object.get("state").?, &image_qs, MAX_LENGTH, true, &.{image});
    defer a.free(image_response);
    const image_result = try std.json.parseFromSlice(V, a, image_response, .{});
    defer image_result.deinit();
    if (std.c.getenv("CLEF_TEST_VISION_REFERENCE")) |raw_path| {
        var reference = try model.loadWeightsSingleFile(a, std.mem.span(raw_path));
        defer reference.deinit();
        const ref_pixels = reference.get("pixels").?;
        try mlx.check(mlx.mlx_array_eval(ref_pixels));
        const actual_pixels = @as([*]const f32, @ptrCast(@alignCast(image.pixels.ptr)))[0 .. image.pixels.len / @sizeOf(f32)];
        try std.testing.expectEqual(mlx.mlx_array_size(ref_pixels), actual_pixels.len);
        var pixel_error: f32 = 0;
        for (actual_pixels, mlx.mlx_array_data_float32(ref_pixels).?[0..actual_pixels.len]) |x, y| pixel_error = @max(pixel_error, @abs(x - y));
        try std.testing.expect(pixel_error < 0.000001);
    }
    try fixture.compare(expected.object.get("image").?.object.get("answers").?, image_result.value.object.get("answers").?);
    try fixture.compare(expected.object.get("image").?.object.get("usage").?, image_result.value.object.get("usage").?);
}
