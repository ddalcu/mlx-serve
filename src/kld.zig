const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const chat_mod = @import("chat.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const expert_stream_mod = @import("expert_stream.zig");
const scheduler_mod = @import("scheduler.zig");
const model_settings_mod = @import("model_settings.zig");
const server_mod = @import("server.zig");
const testing = std.testing;

pub const SCHEMA = "mlx-serve-kld-baseline-v1";
pub const COMPARE_SCHEMA = "mlx-serve-kld-compare-v1";
pub const TOOL = "mlx-serve";

pub const Command = enum { capture, compare };

pub const Options = struct {
    command: Command = .capture,
    help: bool = false,
    model_dir: []const u8 = "",
    prompts: []const u8 = "",
    out_dir: []const u8 = "",
    fixture: []const u8 = "",
    json_out: []const u8 = "",
    label: []const u8 = "mlx-serve",
    tokens: u32 = 64,
    top_k: u32 = 10,
    limit: u32 = 0,
    no_template: bool = false,
    ctx_size: u32 = 0,
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense,
    expert_cache_bytes: u64 = 0,
    ssd_budget_bytes: u64 = 0,
    enable_mtp: bool = false,
};

pub const ArgError = error{
    MissingSubcommand,
    UnknownSubcommand,
    UnknownFlag,
    MissingFlagValue,
    BadFlagValue,
    MissingModel,
    MissingPrompts,
    MissingOut,
    MissingFixture,
};

const ValueFlag = enum {
    model,
    prompts,
    out,
    fixture,
    json,
    label,
    tokens,
    top_k,
    limit,
    ctx_size,
    kv_quant,
    ssd_budget_gb,
    expert_cache_gb,
};

fn valueFlag(name: []const u8) ?ValueFlag {
    const table = [_]struct { []const u8, ValueFlag }{
        .{ "--model", .model },
        .{ "--prompts", .prompts },
        .{ "--out", .out },
        .{ "--fixture", .fixture },
        .{ "--json", .json },
        .{ "--label", .label },
        .{ "--tokens", .tokens },
        .{ "--top-k", .top_k },
        .{ "--limit", .limit },
        .{ "--ctx-size", .ctx_size },
        .{ "--kv-quant", .kv_quant },
        .{ "--ssd-budget-gb", .ssd_budget_gb },
        .{ "--expert-cache-gb", .expert_cache_gb },
    };
    for (table) |row| if (std.mem.eql(u8, name, row[0])) return row[1];
    return null;
}

pub fn parseArgs(args: []const []const u8) ArgError!Options {
    if (args.len == 0) return error.MissingSubcommand;
    var o = Options{};
    if (std.mem.eql(u8, args[0], "capture")) {
        o.command = .capture;
    } else if (std.mem.eql(u8, args[0], "compare")) {
        o.command = .compare;
    } else if (std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h")) {
        o.help = true;
        return o;
    } else {
        return error.UnknownSubcommand;
    }

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            o.help = true;
            return o;
        } else if (std.mem.eql(u8, a, "--no-template")) {
            o.no_template = true;
        } else if (std.mem.eql(u8, a, "--no-mtp")) {
            o.enable_mtp = false;
        } else if (std.mem.eql(u8, a, "--mtp")) {
            o.enable_mtp = true;
        } else if (valueFlag(a)) |flag| {
            if (i + 1 >= args.len) return error.MissingFlagValue;
            i += 1;
            const v = args[i];
            switch (flag) {
                .model => o.model_dir = v,
                .prompts => o.prompts = v,
                .out => o.out_dir = v,
                .fixture => o.fixture = v,
                .json => o.json_out = v,
                .label => o.label = v,
                .tokens => o.tokens = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .top_k => o.top_k = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .limit => o.limit = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .ctx_size => o.ctx_size = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .kv_quant => o.kv_quant_config = transformer_mod.KVQuantConfig.fromJsonValue(.{ .string = v }) orelse return error.BadFlagValue,
                .ssd_budget_gb => o.ssd_budget_bytes = server_mod.parseSsdBudgetGb(v) catch return error.BadFlagValue,
                .expert_cache_gb => o.expert_cache_bytes = server_mod.parseExpertCacheGb(v) catch return error.BadFlagValue,
            }
        } else {
            return error.UnknownFlag;
        }
    }
    if (o.tokens == 0) return error.BadFlagValue;
    if (o.model_dir.len == 0) return error.MissingModel;
    switch (o.command) {
        .capture => {
            if (o.prompts.len == 0) return error.MissingPrompts;
            if (o.out_dir.len == 0) return error.MissingOut;
        },
        .compare => {
            if (o.fixture.len == 0) return error.MissingFixture;
        },
    }
    return o;
}

pub const USAGE =
    \\usage:
    \\  mlx-serve kld capture --model <dir> --prompts <src> --out <dir> [options]
    \\  mlx-serve kld compare --model <dir> --fixture <dir> [options]
    \\
    \\  <src> is a captured fixture dir (its prompts are reused), a directory of
    \\  *.txt files (one prompt each, sorted by name), or a .jsonl of
    \\  {"id":...,"prompt":...} lines.
    \\
    \\options:
    \\  --tokens <n>          greedy tokens per prompt to capture (default 64)
    \\  --top-k <n>           top_k recorded in baseline.json (default 10)
    \\  --label <s>           label/run recorded in the output (default mlx-serve)
    \\  --limit <n>           only the first n prompts (0 = all)
    \\  --no-template         feed the raw prompt text, no chat template
    \\  --json <file>         compare: write the numbers as JSON
    \\  --ctx-size <n>        context length override
    \\  --kv-quant <off|4|8>  KV cache quantization
    \\  --ssd-budget-gb <n>   bf16 expert streaming budget (GiB)
    \\  --expert-cache-gb <n> bf16 expert cache size (GB), outranks --ssd-budget-gb
    \\  --mtp                 keep the MTP head resident (refused under streaming)
    \\
;

pub const RowScore = struct { kld: f64, nll: f64, top1: bool };

pub fn scoreRow(teacher: []const f32, model: []const f32, token: u32) !RowScore {
    if (model.len != teacher.len or token >= teacher.len) return error.KldShapeMismatch;
    var teacher_max = -std.math.inf(f64);
    var candidate_max = -std.math.inf(f64);
    var candidate_top: usize = 0;
    for (teacher, 0..) |value, i| {
        if (value > teacher_max) teacher_max = value;
        if (model[i] > candidate_max) {
            candidate_max = model[i];
            candidate_top = i;
        }
    }
    var teacher_sum: f64 = 0;
    var candidate_sum: f64 = 0;
    for (teacher, 0..) |value, i| {
        teacher_sum += @exp(@as(f64, value) - teacher_max);
        candidate_sum += @exp(@as(f64, model[i]) - candidate_max);
    }
    const teacher_log_z = teacher_max + @log(teacher_sum);
    const candidate_log_z = candidate_max + @log(candidate_sum);
    var kld: f64 = 0;
    for (teacher, 0..) |value, i| {
        const log_p = @as(f64, value) - teacher_log_z;
        const log_q = @as(f64, model[i]) - candidate_log_z;
        kld += @exp(log_p) * (log_p - log_q);
    }
    return .{
        .kld = kld,
        .nll = candidate_log_z - model[token],
        .top1 = candidate_top == token,
    };
}

fn rowNll(row: []const f32, chosen: u32) f64 {
    var max = -std.math.inf(f64);
    for (row) |v| {
        if (v > max) max = v;
    }
    var sum: f64 = 0;
    for (row) |v| sum += @exp(@as(f64, v) - max);
    return max + @log(sum) - @as(f64, row[chosen]);
}

fn argmaxOf(row: []const f32) u32 {
    var best: usize = 0;
    var best_v = -std.math.inf(f64);
    for (row, 0..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return @intCast(best);
}

pub const Prompt = struct { id: []u8, text: []u8 };

pub const PromptList = struct {
    allocator: std.mem.Allocator,
    items: []Prompt = &.{},

    pub fn deinit(self: *PromptList) void {
        for (self.items) |p| {
            self.allocator.free(p.id);
            self.allocator.free(p.text);
        }
        self.allocator.free(self.items);
        self.items = &.{};
    }
};

pub const SourceKind = enum { fixture, text_dir, jsonl };

pub fn classifySource(io: std.Io, path: []const u8) !SourceKind {
    if (path.len == 0) return error.PromptSourceUnreadable;
    if (std.mem.endsWith(u8, path, ".jsonl")) return .jsonl;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const baseline = std.fmt.bufPrint(&buf, "{s}/baseline.json", .{path}) catch return error.KldPathTooLong;
    if (std.Io.Dir.cwd().statFile(io, baseline, .{})) |_| {
        return .fixture;
    } else |_| {}
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return error.PromptSourceUnreadable;
    if (st.kind != .directory) return error.PromptSourceUnreadable;
    return .text_dir;
}

const MAX_PROMPT_BYTES = 8 * 1024 * 1024;

pub fn loadPrompts(allocator: std.mem.Allocator, io: std.Io, path: []const u8, limit: u32) !PromptList {
    var list = PromptList{ .allocator = allocator };
    var items: std.ArrayList(Prompt) = .empty;
    errdefer {
        for (items.items) |p| {
            allocator.free(p.id);
            allocator.free(p.text);
        }
        items.deinit(allocator);
    }
    switch (try classifySource(io, path)) {
        .fixture => {
            var base = try readBaseline(allocator, io, path);
            defer base.deinit();
            for (base.prompts) |fp| {
                if (limit > 0 and items.items.len >= limit) break;
                const text_path = try std.fmt.allocPrint(allocator, "{s}/{s}/prompt.txt", .{ path, fp.dir });
                defer allocator.free(text_path);
                const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, allocator, .limited(MAX_PROMPT_BYTES));
                errdefer allocator.free(text);
                try items.append(allocator, .{ .id = try allocator.dupe(u8, fp.id), .text = text });
            }
        },
        .text_dir => {
            var names: std.ArrayList([]u8) = .empty;
            defer {
                for (names.items) |n| allocator.free(n);
                names.deinit(allocator);
            }
            {
                var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch return error.PromptSourceUnreadable;
                defer dir.close(io);
                var it = dir.iterate();
                while (try it.next(io)) |dent| {
                    if (dent.kind == .directory) continue;
                    if (!std.mem.endsWith(u8, dent.name, ".txt")) continue;
                    try names.append(allocator, try allocator.dupe(u8, dent.name));
                }
            }
            std.mem.sort([]u8, names.items, {}, struct {
                fn lt(_: void, a: []u8, b: []u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lt);
            for (names.items) |name| {
                if (limit > 0 and items.items.len >= limit) break;
                const text_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ path, name });
                defer allocator.free(text_path);
                const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, allocator, .limited(MAX_PROMPT_BYTES));
                errdefer allocator.free(text);
                const id = try allocator.dupe(u8, name[0 .. name.len - 4]);
                errdefer allocator.free(id);
                try items.append(allocator, .{ .id = id, .text = text });
            }
        },
        .jsonl => {
            const body = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(MAX_PROMPT_BYTES)) catch return error.PromptSourceUnreadable;
            defer allocator.free(body);
            var lines = std.mem.splitScalar(u8, body, '\n');
            var seq: usize = 0;
            while (lines.next()) |raw_line| {
                const line = std.mem.trim(u8, raw_line, " \t\r\n");
                if (line.len == 0) continue;
                if (limit > 0 and items.items.len >= limit) break;
                var parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch return error.BadPromptJsonl;
                defer parsed.deinit();
                const obj = switch (parsed.value) {
                    .object => |o| o,
                    else => return error.BadPromptJsonl,
                };
                const text_value = obj.get("prompt") orelse return error.BadPromptJsonl;
                const text_str = switch (text_value) {
                    .string => |s| s,
                    else => return error.BadPromptJsonl,
                };
                const text = try allocator.dupe(u8, text_str);
                errdefer allocator.free(text);
                const id = if (obj.get("id")) |v| switch (v) {
                    .string => |s| try allocator.dupe(u8, s),
                    .integer => |n| try std.fmt.allocPrint(allocator, "{d}", .{n}),
                    else => try std.fmt.allocPrint(allocator, "prompt-{d:0>2}", .{seq}),
                } else try std.fmt.allocPrint(allocator, "prompt-{d:0>2}", .{seq});
                errdefer allocator.free(id);
                try items.append(allocator, .{ .id = id, .text = text });
                seq += 1;
            }
        },
    }
    list.items = try items.toOwnedSlice(allocator);
    return list;
}

pub const PromptRecord = struct {
    id: []u8,
    dir: []u8,
    prompt_tokens: usize,
    generated_tokens: usize,
    strict_nll_mean: f64,
    strict_perplexity: f64,

    pub fn deinit(self: *const PromptRecord, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.dir);
    }
};

pub const BaselineMeta = struct {
    label: []const u8,
    model: []const u8,
    run: []const u8,
    kv_cache_format: []const u8,
    inference_profile: []const u8,
    prompt_set: []const u8,
    ssd_budget_gb: u64,
    tokens_per_prompt: u32,
    top_k: u32,
    elapsed_secs: f64,
};

pub const FixturePrompt = struct {
    id: []u8,
    dir: []u8,
    prompt_tokens: usize = 0,
    generated_tokens: usize = 0,
    strict_nll_mean: f64 = 0,
    strict_perplexity: f64 = 0,
};

pub const Baseline = struct {
    allocator: std.mem.Allocator,
    schema: []u8 = &.{},
    tool: []u8 = &.{},
    label: []u8 = &.{},
    model: []u8 = &.{},
    kv_cache_format: []u8 = &.{},
    inference_profile: []u8 = &.{},
    prompt_set: []u8 = &.{},
    tokens_per_prompt: u32 = 0,
    top_k: u32 = 0,
    ssd_budget_gb: u64 = 0,
    prompts: []FixturePrompt = &.{},

    pub fn deinit(self: *Baseline) void {
        const a = self.allocator;
        a.free(self.schema);
        a.free(self.tool);
        a.free(self.label);
        a.free(self.model);
        a.free(self.kv_cache_format);
        a.free(self.inference_profile);
        a.free(self.prompt_set);
        for (self.prompts) |p| {
            a.free(p.id);
            a.free(p.dir);
        }
        a.free(self.prompts);
        self.prompts = &.{};
    }
};

fn dupStringField(allocator: std.mem.Allocator, obj: std.json.ObjectMap, key: []const u8) ![]u8 {
    if (obj.get(key)) |v| switch (v) {
        .string => |s| return allocator.dupe(u8, s),
        else => {},
    };
    return allocator.dupe(u8, "");
}

fn floatField(obj: std.json.ObjectMap, key: []const u8) f64 {
    if (obj.get(key)) |v| switch (v) {
        .float => |f| return f,
        .integer => |n| return @floatFromInt(n),
        .number_string => |t| return std.fmt.parseFloat(f64, t) catch 0,
        else => {},
    };
    return 0;
}

fn intField(obj: std.json.ObjectMap, key: []const u8) u64 {
    if (obj.get(key)) |v| switch (v) {
        .integer => |n| return if (n < 0) 0 else @intCast(n),
        .float => |f| return if (f < 0) 0 else @intFromFloat(f),
        else => {},
    };
    return 0;
}

pub fn readBaseline(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !Baseline {
    const path = try std.fmt.allocPrint(allocator, "{s}/baseline.json", .{dir});
    defer allocator.free(path);
    const body = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch return error.BaselineUnreadable;
    defer allocator.free(body);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.BadBaselineJson;
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadBaselineJson,
    };
    var b = Baseline{ .allocator = allocator };
    errdefer b.deinit();
    b.schema = try dupStringField(allocator, root, "schema");
    b.tool = try dupStringField(allocator, root, "tool");
    b.label = try dupStringField(allocator, root, "label");
    b.model = try dupStringField(allocator, root, "model");
    b.kv_cache_format = try dupStringField(allocator, root, "kv_cache_format");
    b.inference_profile = try dupStringField(allocator, root, "inference_profile");
    b.prompt_set = try dupStringField(allocator, root, "prompt_set");
    b.tokens_per_prompt = @intCast(intField(root, "tokens_per_prompt"));
    b.top_k = @intCast(intField(root, "top_k"));
    b.ssd_budget_gb = intField(root, "ssd_budget_gb");
    const arr = switch (root.get("prompts") orelse return error.BadBaselineJson) {
        .array => |a| a,
        else => return error.BadBaselineJson,
    };
    var prompts: std.ArrayList(FixturePrompt) = .empty;
    errdefer {
        for (prompts.items) |p| {
            allocator.free(p.id);
            allocator.free(p.dir);
        }
        prompts.deinit(allocator);
    }
    for (arr.items) |item| {
        const obj = switch (item) {
            .object => |o| o,
            else => return error.BadBaselineJson,
        };
        const id = try dupStringField(allocator, obj, "id");
        errdefer allocator.free(id);
        const sub = try dupStringField(allocator, obj, "dir");
        errdefer allocator.free(sub);
        if (sub.len == 0) return error.BadBaselineJson;
        try prompts.append(allocator, .{
            .id = id,
            .dir = sub,
            .prompt_tokens = @intCast(intField(obj, "prompt_tokens")),
            .generated_tokens = @intCast(intField(obj, "generated_tokens")),
            .strict_nll_mean = floatField(obj, "strict_nll_mean"),
            .strict_perplexity = floatField(obj, "strict_perplexity"),
        });
    }
    b.prompts = try prompts.toOwnedSlice(allocator);
    return b;
}

fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeAll("\"");
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => {
            if (c < 0x20) {
                try w.print("\\u{x:0>4}", .{c});
            } else {
                try w.writeByte(c);
            }
        },
    };
    try w.writeAll("\"");
}

pub fn writeBaseline(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_root: []const u8,
    meta: BaselineMeta,
    records: []const PromptRecord,
) !void {
    _ = allocator;
    try std.Io.Dir.cwd().createDirPath(io, out_root);
    var dir = try std.Io.Dir.cwd().openDir(io, out_root, .{});
    defer dir.close(io);
    var f = try dir.createFile(io, "baseline.json", .{});
    defer f.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = f.writer(io, &buf);
    const w = &fw.interface;
    try w.writeAll("{\n  \"schema\": ");
    try writeJsonString(w, SCHEMA);
    try w.writeAll(",\n  \"tool\": ");
    try writeJsonString(w, TOOL);
    try w.writeAll(",\n  \"label\": ");
    try writeJsonString(w, meta.label);
    try w.writeAll(",\n  \"model\": ");
    try writeJsonString(w, meta.model);
    try w.writeAll(",\n  \"run\": ");
    try writeJsonString(w, meta.run);
    try w.writeAll(",\n  \"kv_cache_format\": ");
    try writeJsonString(w, meta.kv_cache_format);
    try w.writeAll(",\n  \"inference_profile\": ");
    try writeJsonString(w, meta.inference_profile);
    try w.print(",\n  \"ssd_budget_gb\": {d}", .{meta.ssd_budget_gb});
    try w.writeAll(",\n  \"prompt_set\": ");
    try writeJsonString(w, meta.prompt_set);
    try w.print(",\n  \"tokens_per_prompt\": {d}", .{meta.tokens_per_prompt});
    try w.print(",\n  \"top_k\": {d}", .{meta.top_k});
    try w.print(",\n  \"elapsed_secs\": {d}", .{meta.elapsed_secs});
    try w.writeAll(",\n  \"prompts\": [\n");
    for (records, 0..) |r, i| {
        try w.writeAll("    {\"id\": ");
        try writeJsonString(w, r.id);
        try w.writeAll(", \"dir\": ");
        try writeJsonString(w, r.dir);
        try w.print(", \"prompt_tokens\": {d}, \"generated_tokens\": {d}, \"strict_nll_mean\": {d}, \"strict_perplexity\": {d}}}", .{
            r.prompt_tokens,
            r.generated_tokens,
            r.strict_nll_mean,
            r.strict_perplexity,
        });
        if (i + 1 < records.len) try w.writeAll(",");
        try w.writeAll("\n");
    }
    try w.writeAll("  ]\n}\n");
    try w.flush();
}

pub fn readIdList(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch return error.TokenListUnreadable;
    defer allocator.free(raw);
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(allocator);
    var parts = std.mem.splitScalar(u8, std.mem.trim(u8, raw, " \t\r\n"), ',');
    while (parts.next()) |part| {
        const t = std.mem.trim(u8, part, " \t\r\n");
        if (t.len == 0) continue;
        try ids.append(allocator, std.fmt.parseInt(u32, t, 10) catch return error.BadTokenList);
    }
    return ids.toOwnedSlice(allocator);
}

fn writeTextFile(io: std.Io, dir: std.Io.Dir, name: []const u8, body: []const u8) !void {
    var f = try dir.createFile(io, name, .{});
    defer f.close(io);
    var buf: [16 * 1024]u8 = undefined;
    var fw = f.writer(io, &buf);
    try fw.interface.writeAll(body);
    try fw.interface.flush();
}

fn writeIdListFile(io: std.Io, dir: std.Io.Dir, name: []const u8, ids: []const u32) !void {
    var f = try dir.createFile(io, name, .{});
    defer f.close(io);
    var buf: [16 * 1024]u8 = undefined;
    var fw = f.writer(io, &buf);
    for (ids, 0..) |id, i| {
        if (i > 0) try fw.interface.writeAll(",");
        try fw.interface.print("{d}", .{id});
    }
    try fw.interface.flush();
}

fn sanitizeDirName(allocator: std.mem.Allocator, id: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, id.len);
    for (id, 0..) |c, i| {
        out[i] = switch (c) {
            'a'...'z', 'A'...'Z', '0'...'9', '-', '_' => c,
            else => '_',
        };
    }
    return out;
}

fn writeAllFd(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (got < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.LogitsWriteFailed;
        }
        if (got == 0) return error.LogitsWriteFailed;
        done += @intCast(got);
    }
}

pub const PromptWriter = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    rel_dir: []u8,
    abs_dir: []u8,
    id: []u8,
    fd: std.c.fd_t,
    rows: usize = 0,
    vocab: usize = 0,
    nll_sum: f64 = 0,

    pub fn begin(
        allocator: std.mem.Allocator,
        io: std.Io,
        out_root: []const u8,
        index: usize,
        id: []const u8,
    ) !PromptWriter {
        const safe = try sanitizeDirName(allocator, id);
        defer allocator.free(safe);
        const rel_dir = try std.fmt.allocPrint(allocator, "prompts/{d:0>2}_{s}", .{ index, safe });
        errdefer allocator.free(rel_dir);
        const abs_dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ out_root, rel_dir });
        errdefer allocator.free(abs_dir);
        const own_id = try allocator.dupe(u8, id);
        errdefer allocator.free(own_id);
        try std.Io.Dir.cwd().createDirPath(io, abs_dir);
        const logits_path = try std.fmt.allocPrintSentinel(allocator, "{s}/logits.f32", .{abs_dir}, 0);
        defer allocator.free(logits_path);
        const fd = std.c.open(logits_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.LogitsCreateFailed;
        return .{
            .allocator = allocator,
            .io = io,
            .rel_dir = rel_dir,
            .abs_dir = abs_dir,
            .id = own_id,
            .fd = fd,
        };
    }

    pub fn appendRow(self: *PromptWriter, row: []const f32, chosen: u32) !void {
        if (self.rows == 0) {
            self.vocab = row.len;
        } else if (row.len != self.vocab) {
            return error.LogitsRowWidthChanged;
        }
        if (chosen >= row.len) return error.ChosenTokenOutOfRange;
        try writeAllFd(self.fd, std.mem.sliceAsBytes(row));
        self.nll_sum += rowNll(row, chosen);
        self.rows += 1;
    }

    pub fn finish(
        self: *PromptWriter,
        prompt_text: []const u8,
        rendered: []const u8,
        prompt_ids: []const u32,
        generated_ids: []const u32,
    ) !PromptRecord {
        if (self.fd >= 0) {
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
        var dir = try std.Io.Dir.cwd().openDir(self.io, self.abs_dir, .{});
        defer dir.close(self.io);
        try writeTextFile(self.io, dir, "id.txt", self.id);
        try writeTextFile(self.io, dir, "prompt.txt", prompt_text);
        try writeTextFile(self.io, dir, "rendered_prompt.txt", rendered);
        try writeIdListFile(self.io, dir, "prompt_tokens.txt", prompt_ids);
        try writeIdListFile(self.io, dir, "generated_tokens.txt", generated_ids);
        const mean = if (self.rows == 0) 0 else self.nll_sum / @as(f64, @floatFromInt(self.rows));
        const id = try self.allocator.dupe(u8, self.id);
        errdefer self.allocator.free(id);
        const rel = try self.allocator.dupe(u8, self.rel_dir);
        return .{
            .id = id,
            .dir = rel,
            .prompt_tokens = prompt_ids.len,
            .generated_tokens = generated_ids.len,
            .strict_nll_mean = mean,
            .strict_perplexity = @exp(mean),
        };
    }

    pub fn deinit(self: *PromptWriter) void {
        if (self.fd >= 0) {
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
        self.allocator.free(self.rel_dir);
        self.allocator.free(self.abs_dir);
        self.allocator.free(self.id);
    }
};

pub fn kvCacheFormat(cfg: transformer_mod.KVQuantConfig) []const u8 {
    return switch (cfg.scheme) {
        .off => "bf16",
        .affine => switch (cfg.bits) {
            4 => "affine4",
            8 => "affine8",
            else => "affine",
        },
    };
}

pub const Loaded = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: model_mod.ModelConfig,
    tok: tokenizer_mod.Tokenizer,
    chat_config: chat_mod.ChatConfig,
    weights: model_mod.Weights,
    xfm: transformer_mod.Transformer,

    pub fn deinit(self: *Loaded) void {
        const allocator = self.allocator;
        self.xfm.deinit();
        self.weights.deinit();
        self.chat_config.deinit();
        self.tok.deinit();
        self.config.deinit(allocator);
        allocator.destroy(self);
    }
};

pub fn loadModel(io: std.Io, allocator: std.mem.Allocator, opts: Options) !*Loaded {
    const self = try allocator.create(Loaded);
    errdefer allocator.destroy(self);
    self.* = .{
        .allocator = allocator,
        .io = io,
        .config = try model_mod.parseConfig(io, allocator, opts.model_dir),
        .tok = undefined,
        .chat_config = undefined,
        .weights = undefined,
        .xfm = undefined,
    };
    errdefer self.config.deinit(allocator);
    scheduler_mod.applyModelSettings(&self.config, model_settings_mod.overrideFor(allocator, io, opts.model_dir));
    if (opts.ctx_size > 0 and self.config.ctx_override == 0) self.config.ctx_override = opts.ctx_size;

    const kld_budget = scheduler_mod.resolveSsdBudget(opts.ssd_budget_bytes, self.config.ssd_budget_gb_override, self.config.supportsExpertStreaming());
    if (expert_stream_mod.expertStreamingEngaged(
        self.config.supportsExpertStreaming(),
        self.config.expertStreamingRequired(),
        opts.expert_cache_bytes,
        kld_budget.bytes,
    )) {
        const budget = kld_budget;
        if (opts.expert_cache_bytes == 0 and budget.bytes == 0) return error.ExpertStreamingRequired;
        switch (expert_stream_mod.mtpUnderStreaming(opts.enable_mtp, self.config.mtp_override)) {
            .refuse => {
                log.err("[expert-stream] {s}; drop --mtp\n", .{expert_stream_mod.MTP_UNSUPPORTED});
                return error.ExpertStreamingMtpUnsupported;
            },
            .drop_settings => {
                log.info("[expert-stream] model-settings mtp=true ignored: {s}\n", .{expert_stream_mod.MTP_UNSUPPORTED});
                self.config.mtp_override = false;
            },
            .off => {},
        }
        const mtp_resident = false;
        const geometry = scheduler_mod.streamingGeometryOf(&self.config);
        self.config.expert_layout = expert_stream_mod.quant.layoutOfDir(allocator, io, self.config.model_type, opts.model_dir, geometry.layers) orelse
            return error.ExpertStreamingUnsupportedLayout;
        const per_expert = try expert_stream_mod.expertBytesFor(allocator, opts.model_dir, geometry, self.config.expert_layout);
        const split = try model_mod.streamingResidentSplit(io, allocator, opts.model_dir, self.config.expert_layout);
        const resolved = try scheduler_mod.resolveExpertCache(opts.expert_cache_bytes, budget.bytes, &self.config, split, mtp_resident, per_expert);
        const plan = try expert_stream_mod.cachePlanBytes(
            resolved.cache_bytes,
            geometry.layers,
            geometry.experts,
            per_expert,
        );
        self.config.expert_streaming = true;
        if (self.config.expert_source_dir == null) self.config.expert_source_dir = try allocator.dupe(u8, opts.model_dir);
        self.config.expert_cache_bytes = plan.cache_bytes;
        self.config.expert_ssd_budget_bytes = if (resolved.ledger != null) budget.bytes else 0;
        self.config.expert_workspace_bytes = plan.workspace_bytes;
        self.config.expert_bounce_bytes = plan.bounce_bytes;
        self.config.expert_fill_peak_bytes = plan.prefill_peak_bytes;
        log.info("[kld] expert streaming: cache {d:.2} GB, {d} slots/layer\n", .{
            @as(f64, @floatFromInt(plan.cache_bytes)) / 1e9,
            plan.slots_per_layer,
        });
    }

    var metal_available: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&metal_available));
    if (metal_available) {
        const gpu = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu);
        try mlx.check(mlx.mlx_set_default_device(gpu));
    }

    self.tok = try tokenizer_mod.loadTokenizer(io, allocator, opts.model_dir);
    errdefer self.tok.deinit();
    self.chat_config = try chat_mod.loadChatConfig(io, allocator, opts.model_dir);
    errdefer self.chat_config.deinit();

    self.weights = if (self.config.expert_streaming)
        try model_mod.loadWeightsStreaming(io, allocator, opts.model_dir, self.config.expert_layout)
    else
        try model_mod.loadWeights(io, allocator, opts.model_dir);
    errdefer self.weights.deinit();
    model_mod.resolveWeightPrefix(&self.config, &self.weights);

    self.xfm = try transformer_mod.Transformer.init(io, allocator, self.config, &self.weights);
    errdefer self.xfm.deinit();
    if (opts.kv_quant_config.scheme != .off) {
        try self.xfm.cache.reinit(self.config.num_hidden_layers, opts.kv_quant_config);
    }
    try self.xfm.qwen4MtpApplyKvQuant(opts.kv_quant_config);
    _ = mlx.applyWiredPolicy();
    if (self.config.hidden_act == .gelu_approx) {
        self.xfm.compileGelu();
        self.xfm.compileGeglu();
    }
    if (self.config.final_logit_softcapping > 0.0) self.xfm.compileSoftcap();
    if (self.xfm.moe_layers != null) self.xfm.compileMoeRouting();
    if (self.config.linear_num_key_heads > 0) self.xfm.compileGdnGate();
    return self;
}

fn logitsVocab(logits: mlx.mlx_array) !usize {
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return error.KldLogitsShape;
    return @intCast(shape[2]);
}

fn readLastRow(xfm: *transformer_mod.Transformer, logits: mlx.mlx_array, dst: []f32) !void {
    const shape = mlx.getShape(logits);
    if (shape.len != 3 or @as(usize, @intCast(shape[2])) != dst.len) return error.KldLogitsShape;
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    if (shape[1] == 1) {
        try mlx.check(mlx.mlx_reshape(&row, logits, &[_]c_int{shape[2]}, 1, xfm.s));
    } else {
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &[_]c_int{ 0, shape[1] - 1, 0 }, 3, &[_]c_int{ 1, shape[1], shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, xfm.s));
        try mlx.check(mlx.mlx_reshape(&row, sliced, &[_]c_int{shape[2]}, 1, xfm.s));
    }
    var row_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_f32);
    try mlx.check(mlx.mlx_astype(&row_f32, row, .float32, xfm.s));
    var contiguous = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contiguous);
    try mlx.check(mlx.mlx_contiguous(&contiguous, row_f32, false, xfm.s));
    try mlx.check(mlx.mlx_array_eval(contiguous));
    const data = mlx.mlx_array_data_float32(contiguous) orelse return error.KldLogitsUnreadable;
    @memcpy(dst, data[0..dst.len]);
}

const Out = struct {
    fn print(self: *Out, comptime fmt: []const u8, args: anytype) void {
        _ = self;
        var buf: [32 * 1024]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
        writeAllFd(1, line) catch return;
    }
};

fn promptIds(allocator: std.mem.Allocator, l: *Loaded, opts: Options, text: []const u8) ![]u32 {
    if (opts.no_template) return l.tok.encode(allocator, text);
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = text }};
    return chat_mod.formatChat(allocator, &l.tok, &messages, &l.chat_config, null, null, false, null, false);
}

fn renderPrompt(allocator: std.mem.Allocator, l: *Loaded, opts: Options, text: []const u8) ![]const u8 {
    if (opts.no_template) return allocator.dupe(u8, text);
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = text }};
    return chat_mod.renderChatTemplate(allocator, &messages, &l.chat_config, null, null, false, null, false);
}

fn forwardPrompt(allocator: std.mem.Allocator, l: *Loaded, ctx: *transformer_mod.ForwardCtx, ids: []const u32) !mlx.mlx_array {
    const prompt_i32 = try allocator.alloc(i32, ids.len);
    defer allocator.free(prompt_i32);
    for (ids, 0..) |t, i| prompt_i32[i] = @intCast(t);
    const prompt_array = mlx.mlx_array_new_data(prompt_i32.ptr, &[_]c_int{ 1, @intCast(prompt_i32.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(prompt_array);
    return l.xfm.forwardWith(ctx, prompt_array);
}

fn forwardOne(l: *Loaded, ctx: *transformer_mod.ForwardCtx, token: u32) !mlx.mlx_array {
    const data = [_]i32{@intCast(token)};
    const input = mlx.mlx_array_new_data(&data, &[_]c_int{ 1, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(input);
    return l.xfm.forwardWith(ctx, input);
}

pub fn capturedSsdBudgetGb(config: *const model_mod.ModelConfig, flag_bytes: u64) u64 {
    if (!config.expert_streaming) return 0;
    if (config.expert_ssd_budget_bytes > 0) return config.expert_ssd_budget_bytes >> 30;
    return flag_bytes >> 30;
}

pub fn runCapture(io: std.Io, allocator: std.mem.Allocator, l: *Loaded, opts: Options, out: *Out) !void {
    var prompts = try loadPrompts(allocator, io, opts.prompts, opts.limit);
    defer prompts.deinit();
    if (prompts.items.len == 0) return error.NoPromptsFound;
    try std.Io.Dir.cwd().createDirPath(io, opts.out_dir);

    var records: std.ArrayList(PromptRecord) = .empty;
    defer {
        for (records.items) |r| r.deinit(allocator);
        records.deinit(allocator);
    }
    const started = std.Io.Timestamp.now(io, .awake);
    for (prompts.items, 0..) |p, index| {
        try l.xfm.resetCache();
        const rendered = try renderPrompt(allocator, l, opts, p.text);
        defer allocator.free(rendered);
        const ids = try promptIds(allocator, l, opts, p.text);
        defer allocator.free(ids);
        if (ids.len == 0) return error.EmptyPrompt;

        var writer = try PromptWriter.begin(allocator, io, opts.out_dir, index, p.id);
        defer writer.deinit();
        var generated: std.ArrayList(u32) = .empty;
        defer generated.deinit(allocator);

        var ctx = l.xfm.defaultCtx();
        var logits = try forwardPrompt(allocator, l, &ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        const vocab = try logitsVocab(logits);
        const row = try allocator.alloc(f32, vocab);
        defer allocator.free(row);

        var step: u32 = 0;
        while (step < opts.tokens) : (step += 1) {
            try readLastRow(&l.xfm, logits, row);
            const chosen = argmaxOf(row);
            try writer.appendRow(row, chosen);
            try generated.append(allocator, chosen);
            if (step + 1 == opts.tokens) break;
            const next = try forwardOne(l, &ctx, chosen);
            _ = mlx.mlx_array_free(logits);
            logits = next;
        }

        const record = try writer.finish(p.text, rendered, ids, generated.items);
        try records.append(allocator, record);
        out.print("[kld] {d}/{d} {s}: prompt={d} tok, generated={d}, nll={d:.9}, ppl={d:.6}\n", .{
            index + 1,
            prompts.items.len,
            record.id,
            record.prompt_tokens,
            record.generated_tokens,
            record.strict_nll_mean,
            record.strict_perplexity,
        });
    }
    const elapsed_ns: u64 = @intCast(started.untilNow(io, .awake).nanoseconds);
    const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    var nll_sum: f64 = 0;
    for (records.items) |r| nll_sum += r.strict_nll_mean;
    const mean_nll = nll_sum / @as(f64, @floatFromInt(records.items.len));

    try writeBaseline(allocator, io, opts.out_dir, .{
        .label = opts.label,
        .model = opts.model_dir,
        .run = opts.label,
        .kv_cache_format = kvCacheFormat(opts.kv_quant_config),
        .inference_profile = "greedy",
        .prompt_set = opts.prompts,
        .ssd_budget_gb = capturedSsdBudgetGb(&l.config, opts.ssd_budget_bytes),
        .tokens_per_prompt = opts.tokens,
        .top_k = opts.top_k,
        .elapsed_secs = elapsed_secs,
    }, records.items);
    out.print("[kld] captured {d} prompts x {d} tokens into {s} in {d:.1}s (mean strict NLL {d:.9})\n", .{
        records.items.len,
        opts.tokens,
        opts.out_dir,
        elapsed_secs,
        mean_nll,
    });
}

pub const PromptScore = struct {
    id: []const u8,
    kld: f64,
    top1: usize,
    positions: usize,
    nll: f64,
    eos_pos: ?usize = null,
    kld_to_eos: f64 = 0,
    top1_to_eos: usize = 0,
    positions_to_eos: usize = 0,
    nll_to_eos: f64 = 0,
    per_position_kld: []const f64 = &.{},
};

pub fn firstEosPosition(generated: []const u32, eos: []const u32) ?usize {
    for (generated, 0..) |t, i| {
        for (eos) |e| if (t == e) return i;
    }
    return null;
}

pub fn runCompare(io: std.Io, allocator: std.mem.Allocator, l: *Loaded, opts: Options, out: *Out) !void {
    var base = try readBaseline(allocator, io, opts.fixture);
    defer base.deinit();
    const count = if (opts.limit > 0 and opts.limit < base.prompts.len) opts.limit else base.prompts.len;
    if (count == 0) return error.NoPromptsFound;

    var scores: std.ArrayList(PromptScore) = .empty;
    defer scores.deinit(allocator);
    defer for (scores.items) |sc| allocator.free(sc.per_position_kld);
    var total_kld: f64 = 0;
    var total_kld_eos: f64 = 0;
    var total_nll_eos: f64 = 0;
    var total_top1_eos: usize = 0;
    var total_positions_eos: usize = 0;
    var eos_set: std.ArrayList(u32) = .empty;
    defer eos_set.deinit(allocator);
    for (l.config.eos_token_ids[0..l.config.num_eos_tokens]) |e| try eos_set.append(allocator, e);
    if (l.tok.encode(allocator, "<|im_end|>")) |im_end| {
        defer allocator.free(im_end);
        if (im_end.len == 1) try eos_set.append(allocator, im_end[0]);
    } else |_| {}
    var total_nll: f64 = 0;
    var total_top1: usize = 0;
    var total_positions: usize = 0;

    out.print("{s:<48} {s:>14} {s:>10} {s:>14}\n", .{ "prompt", "KLD", "top1", "NLL" });
    for (base.prompts[0..count]) |fp| {
        try l.xfm.resetCache();
        const prompt_path = try std.fmt.allocPrint(allocator, "{s}/{s}/prompt_tokens.txt", .{ opts.fixture, fp.dir });
        defer allocator.free(prompt_path);
        const prompt_ids = try readIdList(allocator, io, prompt_path);
        defer allocator.free(prompt_ids);
        const gen_path = try std.fmt.allocPrint(allocator, "{s}/{s}/generated_tokens.txt", .{ opts.fixture, fp.dir });
        defer allocator.free(gen_path);
        const generated = try readIdList(allocator, io, gen_path);
        defer allocator.free(generated);
        if (prompt_ids.len == 0 or generated.len == 0) return error.EmptyFixturePrompt;

        const logits_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}/logits.f32", .{ opts.fixture, fp.dir }, 0);
        defer allocator.free(logits_path);
        const teacher_fd = std.c.open(logits_path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (teacher_fd < 0) return error.TeacherLogitsMissing;
        defer _ = std.c.close(teacher_fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(teacher_fd, &st) != 0 or st.size < 0) return error.TeacherLogitsStatFailed;
        const size: u64 = @intCast(st.size);
        const row_bytes = generated.len * @sizeOf(f32);
        if (row_bytes == 0 or size % row_bytes != 0) return error.TeacherLogitsSizeMismatch;
        const vocab: usize = @intCast(size / row_bytes);

        const teacher_row = try allocator.alloc(f32, vocab);
        defer allocator.free(teacher_row);
        const model_row = try allocator.alloc(f32, vocab);
        defer allocator.free(model_row);

        var ctx = l.xfm.defaultCtx();
        var logits = try forwardPrompt(allocator, l, &ctx, prompt_ids);
        defer _ = mlx.mlx_array_free(logits);

        var prompt_kld: f64 = 0;
        var prompt_nll: f64 = 0;
        var prompt_top1: usize = 0;
        const eos_pos = firstEosPosition(generated, eos_set.items);
        const n_eos: usize = if (eos_pos) |e| e + 1 else generated.len;
        var kld_eos: f64 = 0;
        var nll_eos: f64 = 0;
        var top1_eos: usize = 0;
        const per_pos = try allocator.alloc(f64, generated.len);
        for (generated, 0..) |token, position| {
            try expert_stream_mod.readExact(teacher_fd, std.mem.sliceAsBytes(teacher_row), position * vocab * @sizeOf(f32));
            if (position > 0) {
                const next = try forwardOne(l, &ctx, generated[position - 1]);
                _ = mlx.mlx_array_free(logits);
                logits = next;
            }
            try readLastRow(&l.xfm, logits, model_row);
            const score = try scoreRow(teacher_row, model_row, token);
            prompt_kld += score.kld;
            prompt_nll += score.nll;
            if (score.top1) prompt_top1 += 1;
            per_pos[position] = score.kld;
            if (position < n_eos) {
                kld_eos += score.kld;
                nll_eos += score.nll;
                if (score.top1) top1_eos += 1;
            }
        }
        const positions: f64 = @floatFromInt(generated.len);
        const positions_eos: f64 = @floatFromInt(n_eos);
        try scores.append(allocator, .{
            .id = fp.id,
            .kld = prompt_kld / positions,
            .top1 = prompt_top1,
            .positions = generated.len,
            .nll = prompt_nll / positions,
            .eos_pos = eos_pos,
            .kld_to_eos = kld_eos / positions_eos,
            .top1_to_eos = top1_eos,
            .positions_to_eos = n_eos,
            .nll_to_eos = nll_eos / positions_eos,
            .per_position_kld = per_pos,
        });
        total_kld += prompt_kld;
        total_nll += prompt_nll;
        total_top1 += prompt_top1;
        total_positions += generated.len;
        total_kld_eos += kld_eos;
        total_nll_eos += nll_eos;
        total_top1_eos += top1_eos;
        total_positions_eos += n_eos;
        out.print("{s:<48} {d:>14.9} {d:>6}/{d:<3} {d:>14.9}   to-eos {d:>12.9} {d:>3}/{d:<3} {d:>12.9}\n", .{
            fp.id,
            prompt_kld / positions,
            prompt_top1,
            generated.len,
            prompt_nll / positions,
            kld_eos / positions_eos,
            top1_eos,
            n_eos,
            nll_eos / positions_eos,
        });
    }

    const positions_f: f64 = @floatFromInt(total_positions);
    const mean_kld = total_kld / positions_f;
    const mean_nll = total_nll / positions_f;
    const mean_top1 = @as(f64, @floatFromInt(total_top1)) / positions_f;
    const positions_eos_f: f64 = @floatFromInt(@max(total_positions_eos, 1));
    const mean_kld_eos = total_kld_eos / positions_eos_f;
    const mean_nll_eos = total_nll_eos / positions_eos_f;
    const mean_top1_eos = @as(f64, @floatFromInt(total_top1_eos)) / positions_eos_f;
    out.print("{s:<48} {d:>14.9} {d:>6}/{d:<3} {d:>14.9}   to-eos {d:>12.9} {d:>3}/{d:<3} {d:>12.9}\n", .{ "mean", mean_kld, total_top1, total_positions, mean_nll, mean_kld_eos, total_top1_eos, total_positions_eos, mean_nll_eos });
    out.print("[kld] {s}: to-first-EOS mean KLD={d:.9} top1={d:.6} NLL={d:.9} over {d} positions\n", .{ opts.label, mean_kld_eos, mean_top1_eos, mean_nll_eos, total_positions_eos });
    out.print("[kld] {s}: mean KLD={d:.9} top1={d:.6} NLL={d:.9} over {d} prompts / {d} positions\n", .{
        opts.label,
        mean_kld,
        mean_top1,
        mean_nll,
        scores.items.len,
        total_positions,
    });

    if (opts.json_out.len > 0) {
        var f = try std.Io.Dir.cwd().createFile(io, opts.json_out, .{});
        defer f.close(io);
        var buf: [64 * 1024]u8 = undefined;
        var fw = f.writer(io, &buf);
        const w = &fw.interface;
        try w.writeAll("{\n  \"schema\": ");
        try writeJsonString(w, COMPARE_SCHEMA);
        try w.writeAll(",\n  \"tool\": ");
        try writeJsonString(w, TOOL);
        try w.writeAll(",\n  \"label\": ");
        try writeJsonString(w, opts.label);
        try w.writeAll(",\n  \"model\": ");
        try writeJsonString(w, opts.model_dir);
        try w.writeAll(",\n  \"fixture\": ");
        try writeJsonString(w, opts.fixture);
        try w.writeAll(",\n  \"kv_cache_format\": ");
        try writeJsonString(w, kvCacheFormat(opts.kv_quant_config));
        try w.print(",\n  \"mean_kld\": {d},\n  \"mean_top1\": {d},\n  \"mean_nll\": {d},\n  \"positions\": {d}", .{
            mean_kld,
            mean_top1,
            mean_nll,
            total_positions,
        });
        try w.print(",\n  \"mean_kld_to_eos\": {d},\n  \"mean_top1_to_eos\": {d},\n  \"mean_nll_to_eos\": {d},\n  \"positions_to_eos\": {d}", .{
            mean_kld_eos,
            mean_top1_eos,
            mean_nll_eos,
            total_positions_eos,
        });
        try w.writeAll(",\n  \"prompts\": [\n");
        for (scores.items, 0..) |s, i| {
            try w.writeAll("    {\"id\": ");
            try writeJsonString(w, s.id);
            try w.print(", \"kld\": {d}, \"top1\": {d}, \"positions\": {d}, \"nll\": {d}", .{ s.kld, s.top1, s.positions, s.nll });
            if (s.eos_pos) |e| try w.print(", \"eos_pos\": {d}", .{e}) else try w.writeAll(", \"eos_pos\": null");
            try w.print(", \"kld_to_eos\": {d}, \"top1_to_eos\": {d}, \"positions_to_eos\": {d}, \"nll_to_eos\": {d}, \"per_position_kld\": [", .{ s.kld_to_eos, s.top1_to_eos, s.positions_to_eos, s.nll_to_eos });
            for (s.per_position_kld, 0..) |k, j| {
                if (j > 0) try w.writeAll(",");
                try w.print("{d}", .{k});
            }
            try w.writeAll("]}");
            if (i + 1 < scores.items.len) try w.writeAll(",");
            try w.writeAll("\n");
        }
        try w.writeAll("  ]\n}\n");
        try w.flush();
    }

    for (scores.items) |s| {
        if (!std.math.isFinite(s.kld) or !std.math.isFinite(s.nll)) return error.NonFiniteKld;
    }
}

pub fn cmdKld(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var out: Out = .{};
    const opts = parseArgs(args) catch |err| {
        out.print("mlx-serve kld: {s}\n\n{s}", .{ @errorName(err), USAGE });
        return err;
    };
    if (opts.help) {
        out.print("{s}", .{USAGE});
        return;
    }
    const loaded = try loadModel(io, allocator, opts);
    defer loaded.deinit();
    switch (opts.command) {
        .capture => try runCapture(io, allocator, loaded, opts, &out),
        .compare => try runCompare(io, allocator, loaded, opts, &out),
    }
}

const TEACHER_FIXTURE = "/Users/beam/llm/models/kld-teacher/Qwen3.8-Flash-Next-wikitext2-60x64";

test "kld: scoreRow on identical logits is zero divergence and the plain NLL" {
    const logits = [_]f32{ 0.5, -1.0, 2.0, 0.0, 3.5, -2.5, 1.0, 0.25 };
    const s = try scoreRow(&logits, &logits, 4);
    try testing.expectApproxEqAbs(@as(f64, 0), s.kld, 1e-12);
    try testing.expect(s.top1);
    var sum: f64 = 0;
    for (logits) |v| sum += @exp(@as(f64, v) - 3.5);
    try testing.expectApproxEqAbs(@log(sum), s.nll, 1e-12);
}

fn refScore(teacher: []const f32, model: []const f32, token: usize) RowScore {
    var tmax: f64 = -std.math.inf(f64);
    var mmax: f64 = -std.math.inf(f64);
    var mtop: usize = 0;
    for (teacher, 0..) |v, i| {
        if (v > tmax) tmax = v;
        if (model[i] > mmax) {
            mmax = model[i];
            mtop = i;
        }
    }
    var tsum: f64 = 0;
    var msum: f64 = 0;
    for (teacher, 0..) |v, i| {
        tsum += @exp(@as(f64, v) - tmax);
        msum += @exp(@as(f64, model[i]) - mmax);
    }
    const tlz = tmax + @log(tsum);
    const mlz = mmax + @log(msum);
    var kld: f64 = 0;
    for (teacher, 0..) |v, i| {
        const lp = @as(f64, v) - tlz;
        const lq = @as(f64, model[i]) - mlz;
        kld += @exp(lp) * (lp - lq);
    }
    return .{ .kld = kld, .nll = mlz - model[token], .top1 = mtop == token };
}

test "kld: scoreRow matches the closed form on two 8-vocab rows" {
    const teacher = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const model = [_]f32{ 1.0, 0.5, -0.5, 2.0, 0.0, -1.0, 0.25, 3.0 };

    var msum: f64 = 0;
    for (model) |v| msum += @exp(@as(f64, v));
    const log_z = @log(msum);
    var mean_logit: f64 = 0;
    for (model) |v| mean_logit += @as(f64, v) / 8.0;
    const expected_kld = -@log(@as(f64, 8.0)) - mean_logit + log_z;
    const expected_nll = log_z - 2.0;

    const s = try scoreRow(&teacher, &model, 3);
    try testing.expectApproxEqAbs(expected_kld, s.kld, 1e-12);
    try testing.expectApproxEqAbs(expected_nll, s.nll, 1e-12);
    try testing.expect(!s.top1);
    const s7 = try scoreRow(&teacher, &model, 7);
    try testing.expect(s7.top1);
    try testing.expectApproxEqAbs(refScore(&teacher, &model, 7).kld, s7.kld, 1e-12);
}

test "kld: scoreRow is log-sum-exp stable on large logits" {
    var teacher: [8]f32 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    var model: [8]f32 = .{ 1.0, 0.5, -0.5, 2.0, 0.0, -1.0, 0.25, 3.0 };
    const base = try scoreRow(&teacher, &model, 3);
    for (&teacher) |*v| v.* += 90000.0;
    for (&model) |*v| v.* += 90000.0;
    const shifted = try scoreRow(&teacher, &model, 3);
    try testing.expect(std.math.isFinite(shifted.kld));
    try testing.expect(std.math.isFinite(shifted.nll));
    try testing.expectApproxEqAbs(base.kld, shifted.kld, 1e-9);
    try testing.expectApproxEqAbs(base.nll, shifted.nll, 1e-9);
}

test "kld: a captured fixture round-trips through the reader byte for byte" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const out_root = path_buf[0..root_len];

    const ids = [_][]const u32{ &.{ 5, 6, 7 }, &.{ 1, 2 } };
    const gen = [_][]const u32{ &.{ 0, 3, 6 }, &.{ 7, 1, 2 } };
    const texts = [_][]const u8{ "first prompt", "second prompt" };
    const rendered = [_][]const u8{ "<|im_start|>user\nfirst prompt<|im_end|>\n", "<|im_start|>user\nsecond prompt<|im_end|>\n" };
    var rows_written: [2][3][8]f32 = undefined;

    var records: [2]PromptRecord = undefined;
    for (0..2) |p| {
        var w = try PromptWriter.begin(allocator, io, out_root, p, if (p == 0) "alpha" else "beta");
        defer w.deinit();
        for (0..3) |r| {
            var row: [8]f32 = undefined;
            for (&row, 0..) |*v, i| v.* = @as(f32, @floatFromInt(p * 100 + r * 10 + i)) * 0.125;
            row[gen[p][r]] = 9.0;
            rows_written[p][r] = row;
            try w.appendRow(&row, gen[p][r]);
        }
        records[p] = try w.finish(texts[p], rendered[p], ids[p], gen[p]);
    }
    defer for (&records) |*r| r.deinit(allocator);

    try writeBaseline(allocator, io, out_root, .{
        .label = "round-trip",
        .model = "/models/fake",
        .run = "round-trip",
        .kv_cache_format = "bf16",
        .inference_profile = "greedy",
        .prompt_set = "unit-test",
        .ssd_budget_gb = 0,
        .tokens_per_prompt = 3,
        .top_k = 10,
        .elapsed_secs = 1.5,
    }, &records);

    var base = try readBaseline(allocator, io, out_root);
    defer base.deinit();
    try testing.expectEqualStrings(SCHEMA, base.schema);
    try testing.expectEqualStrings(TOOL, base.tool);
    try testing.expectEqualStrings("round-trip", base.label);
    try testing.expectEqualStrings("bf16", base.kv_cache_format);
    try testing.expectEqualStrings("greedy", base.inference_profile);
    try testing.expectEqual(@as(u32, 3), base.tokens_per_prompt);
    try testing.expectEqual(@as(usize, 2), base.prompts.len);
    try testing.expectEqualStrings("alpha", base.prompts[0].id);
    try testing.expectEqualStrings("prompts/00_alpha", base.prompts[0].dir);
    try testing.expectEqualStrings("prompts/01_beta", base.prompts[1].dir);
    try testing.expectEqual(@as(usize, 3), base.prompts[0].prompt_tokens);
    try testing.expectEqual(@as(usize, 3), base.prompts[0].generated_tokens);
    try testing.expectEqual(@as(usize, 2), base.prompts[1].prompt_tokens);
    try testing.expectApproxEqAbs(records[0].strict_nll_mean, base.prompts[0].strict_nll_mean, 1e-9);
    try testing.expectApproxEqAbs(records[0].strict_perplexity, base.prompts[0].strict_perplexity, 1e-9);

    for (0..2) |p| {
        const dir_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ out_root, base.prompts[p].dir });
        defer allocator.free(dir_path);
        const tok_path = try std.fmt.allocPrint(allocator, "{s}/prompt_tokens.txt", .{dir_path});
        defer allocator.free(tok_path);
        const got_prompt = try readIdList(allocator, io, tok_path);
        defer allocator.free(got_prompt);
        try testing.expectEqualSlices(u32, ids[p], got_prompt);

        const gen_path = try std.fmt.allocPrint(allocator, "{s}/generated_tokens.txt", .{dir_path});
        defer allocator.free(gen_path);
        const got_gen = try readIdList(allocator, io, gen_path);
        defer allocator.free(got_gen);
        try testing.expectEqualSlices(u32, gen[p], got_gen);

        const text_path = try std.fmt.allocPrint(allocator, "{s}/prompt.txt", .{dir_path});
        defer allocator.free(text_path);
        const got_text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, allocator, .limited(1 << 16));
        defer allocator.free(got_text);
        try testing.expectEqualStrings(texts[p], got_text);

        const rend_path = try std.fmt.allocPrint(allocator, "{s}/rendered_prompt.txt", .{dir_path});
        defer allocator.free(rend_path);
        const got_rend = try std.Io.Dir.cwd().readFileAlloc(io, rend_path, allocator, .limited(1 << 16));
        defer allocator.free(got_rend);
        try testing.expectEqualStrings(rendered[p], got_rend);

        const id_path = try std.fmt.allocPrint(allocator, "{s}/id.txt", .{dir_path});
        defer allocator.free(id_path);
        const got_id = try std.Io.Dir.cwd().readFileAlloc(io, id_path, allocator, .limited(1 << 16));
        defer allocator.free(got_id);
        try testing.expectEqualStrings(base.prompts[p].id, got_id);

        const logits_path = try std.fmt.allocPrint(allocator, "{s}/logits.f32", .{dir_path});
        defer allocator.free(logits_path);
        const got_logits = try std.Io.Dir.cwd().readFileAlloc(io, logits_path, allocator, .limited(1 << 20));
        defer allocator.free(got_logits);
        try testing.expectEqual(@as(usize, 3 * 8 * 4), got_logits.len);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&rows_written[p]), got_logits);

        var nll: f64 = 0;
        for (0..3) |r| nll += refScore(&rows_written[p][r], &rows_written[p][r], gen[p][r]).nll;
        try testing.expectApproxEqAbs(nll / 3.0, records[p].strict_nll_mean, 1e-9);
        try testing.expectApproxEqAbs(@exp(nll / 3.0), records[p].strict_perplexity, 1e-9);
    }
}

test "kld: the real teacher fixture reads back at full width" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    _ = std.Io.Dir.cwd().statFile(io, TEACHER_FIXTURE ++ "/baseline.json", .{}) catch return error.SkipZigTest;

    var base = try readBaseline(allocator, io, TEACHER_FIXTURE);
    defer base.deinit();
    try testing.expectEqual(@as(usize, 60), base.prompts.len);
    try testing.expectEqual(@as(u32, 64), base.tokens_per_prompt);
    try testing.expectEqualStrings("wikitext2-test-00-robert-boulter", base.prompts[0].id);
    try testing.expectEqualStrings("prompts/00_wikitext2-test-00-robert-boulter", base.prompts[0].dir);

    const gen_path = TEACHER_FIXTURE ++ "/prompts/00_wikitext2-test-00-robert-boulter/generated_tokens.txt";
    const gen = try readIdList(allocator, io, gen_path);
    defer allocator.free(gen);
    try testing.expectEqual(@as(usize, 64), gen.len);

    const tok_path = TEACHER_FIXTURE ++ "/prompts/00_wikitext2-test-00-robert-boulter/prompt_tokens.txt";
    const ptok = try readIdList(allocator, io, tok_path);
    defer allocator.free(ptok);
    try testing.expectEqual(@as(usize, 343), ptok.len);

    const st = try std.Io.Dir.cwd().statFile(io, TEACHER_FIXTURE ++ "/prompts/00_wikitext2-test-00-robert-boulter/logits.f32", .{});
    const vocab = st.size / (64 * @sizeOf(f32));
    try testing.expectEqual(@as(u64, 248320), vocab);
    try testing.expectEqual(@as(u64, 0), st.size % (64 * @sizeOf(f32)));
}

test "kld: prompt sources parse from a fixture dir, a text dir and a jsonl file" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];

    try tmp.dir.createDirPath(io, "texts");
    {
        var d = try tmp.dir.openDir(io, "texts", .{});
        defer d.close(io);
        const names = [_][]const u8{ "b_second.txt", "a_first.txt", "skipme.md" };
        const bodies = [_][]const u8{ "second body", "first body", "not a prompt" };
        for (names, bodies) |n, b| {
            var f = try d.createFile(io, n, .{});
            defer f.close(io);
            var wb: [256]u8 = undefined;
            var w = f.writer(io, &wb);
            try w.interface.writeAll(b);
            try w.interface.flush();
        }
    }
    {
        var f = try tmp.dir.createFile(io, "prompts.jsonl", .{});
        defer f.close(io);
        var wb: [512]u8 = undefined;
        var w = f.writer(io, &wb);
        try w.interface.writeAll(
            \\{"id":"one","prompt":"prompt one"}
            \\
            \\{"id":"two","prompt":"prompt two"}
        );
        try w.interface.flush();
    }

    const texts_dir = try std.fmt.allocPrint(allocator, "{s}/texts", .{root});
    defer allocator.free(texts_dir);
    try testing.expectEqual(SourceKind.text_dir, try classifySource(io, texts_dir));
    var from_dir = try loadPrompts(allocator, io, texts_dir, 0);
    defer from_dir.deinit();
    try testing.expectEqual(@as(usize, 2), from_dir.items.len);
    try testing.expectEqualStrings("a_first", from_dir.items[0].id);
    try testing.expectEqualStrings("first body", from_dir.items[0].text);
    try testing.expectEqualStrings("b_second", from_dir.items[1].id);

    const jsonl = try std.fmt.allocPrint(allocator, "{s}/prompts.jsonl", .{root});
    defer allocator.free(jsonl);
    try testing.expectEqual(SourceKind.jsonl, try classifySource(io, jsonl));
    var from_jsonl = try loadPrompts(allocator, io, jsonl, 0);
    defer from_jsonl.deinit();
    try testing.expectEqual(@as(usize, 2), from_jsonl.items.len);
    try testing.expectEqualStrings("one", from_jsonl.items[0].id);
    try testing.expectEqualStrings("prompt two", from_jsonl.items[1].text);

    var limited = try loadPrompts(allocator, io, jsonl, 1);
    defer limited.deinit();
    try testing.expectEqual(@as(usize, 1), limited.items.len);

    if (std.Io.Dir.cwd().statFile(io, TEACHER_FIXTURE ++ "/baseline.json", .{})) |_| {
        try testing.expectEqual(SourceKind.fixture, try classifySource(io, TEACHER_FIXTURE));
        var from_fixture = try loadPrompts(allocator, io, TEACHER_FIXTURE, 2);
        defer from_fixture.deinit();
        try testing.expectEqual(@as(usize, 2), from_fixture.items.len);
        try testing.expectEqualStrings("wikitext2-test-00-robert-boulter", from_fixture.items[0].id);
        try testing.expect(std.mem.startsWith(u8, from_fixture.items[0].text, "= Robert Boulter ="));
    } else |_| {}
}

test "kld: the argument parser reads every flag and refuses an unknown one" {
    const capture = try parseArgs(&.{
        "capture",
        "--model",         "/models/pack",
        "--prompts",       "/fixtures/teacher",
        "--out",           "/out/run",
        "--tokens",        "32",
        "--top-k",         "5",
        "--label",         "run-a",
        "--limit",         "3",
        "--no-template",   "--ctx-size",
        "8192",            "--kv-quant",
        "8",               "--ssd-budget-gb",
        "94",              "--expert-cache-gb",
        "40",              "--no-mtp",
    });
    try testing.expectEqual(Command.capture, capture.command);
    try testing.expectEqualStrings("/models/pack", capture.model_dir);
    try testing.expectEqualStrings("/fixtures/teacher", capture.prompts);
    try testing.expectEqualStrings("/out/run", capture.out_dir);
    try testing.expectEqual(@as(u32, 32), capture.tokens);
    try testing.expectEqual(@as(u32, 5), capture.top_k);
    try testing.expectEqualStrings("run-a", capture.label);
    try testing.expectEqual(@as(u32, 3), capture.limit);
    try testing.expect(capture.no_template);
    try testing.expectEqual(@as(u32, 8192), capture.ctx_size);
    try testing.expectEqual(@as(u8, 8), capture.kv_quant_config.bits);
    try testing.expectEqual(@as(u64, 94) << 30, capture.ssd_budget_bytes);
    try testing.expectEqual(@as(u64, 40) * 1_000_000_000, capture.expert_cache_bytes);
    try testing.expect(!capture.enable_mtp);

    const compare = try parseArgs(&.{ "compare", "--model", "/models/pack", "--fixture", "/fixtures/teacher", "--json", "/tmp/out.json" });
    try testing.expectEqual(Command.compare, compare.command);
    try testing.expectEqualStrings("/fixtures/teacher", compare.fixture);
    try testing.expectEqualStrings("/tmp/out.json", compare.json_out);
    try testing.expect(!compare.enable_mtp);
    try testing.expectEqual(@as(u32, 64), compare.tokens);
    const with_mtp = try parseArgs(&.{ "compare", "--model", "/models/pack", "--fixture", "/fixtures/teacher", "--mtp" });
    try testing.expect(with_mtp.enable_mtp);

    try testing.expectError(error.UnknownFlag, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--nope" }));
    try testing.expectError(error.UnknownFlag, parseArgs(&.{ "capture", "--model=/m" }));
    try testing.expectError(error.UnknownSubcommand, parseArgs(&.{"replay"}));
    try testing.expectError(error.MissingSubcommand, parseArgs(&.{}));
    try testing.expectError(error.MissingFlagValue, parseArgs(&.{ "compare", "--model" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--tokens", "zero" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--kv-quant", "3" }));
    try testing.expectError(error.MissingModel, parseArgs(&.{ "capture", "--prompts", "/p", "--out", "/o" }));
    try testing.expectError(error.MissingPrompts, parseArgs(&.{ "capture", "--model", "/m", "--out", "/o" }));
    try testing.expectError(error.MissingOut, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p" }));
    try testing.expectError(error.MissingFixture, parseArgs(&.{ "compare", "--model", "/m" }));
    try testing.expect((try parseArgs(&.{ "capture", "--help" })).help);
}

test "kld: the recorded strict NLL is the teacher's own log-softmax, as the capture wrote it" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    _ = std.Io.Dir.cwd().statFile(io, TEACHER_FIXTURE ++ "/baseline.json", .{}) catch return error.SkipZigTest;

    var base = try readBaseline(allocator, io, TEACHER_FIXTURE);
    defer base.deinit();
    try testing.expectEqual(@as(usize, 343), base.prompts[0].prompt_tokens);

    const vocab: usize = 248320;
    const row = try allocator.alloc(f32, vocab);
    defer allocator.free(row);
    for (base.prompts[0..3]) |fp| {
        try testing.expectEqual(@as(usize, 64), fp.generated_tokens);
        const gen_path = try std.fmt.allocPrint(allocator, "{s}/{s}/generated_tokens.txt", .{ TEACHER_FIXTURE, fp.dir });
        defer allocator.free(gen_path);
        const generated = try readIdList(allocator, io, gen_path);
        defer allocator.free(generated);

        const logits_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}/logits.f32", .{ TEACHER_FIXTURE, fp.dir }, 0);
        defer allocator.free(logits_path);
        const fd = std.c.open(logits_path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        try testing.expect(fd >= 0);
        defer _ = std.c.close(fd);

        var nll_sum: f64 = 0;
        for (generated, 0..) |token, position| {
            try expert_stream_mod.readExact(fd, std.mem.sliceAsBytes(row), position * vocab * @sizeOf(f32));
            try testing.expectEqual(token, argmaxOf(row));
            const self_score = try scoreRow(row, row, token);
            try testing.expectApproxEqAbs(@as(f64, 0), self_score.kld, 1e-12);
            try testing.expect(self_score.top1);
            try testing.expectApproxEqAbs(self_score.nll, rowNll(row, token), 1e-12);
            nll_sum += self_score.nll;
        }
        const mean = nll_sum / @as(f64, @floatFromInt(generated.len));
        try testing.expectApproxEqAbs(fp.strict_nll_mean, mean, 1e-8);
        try testing.expectApproxEqAbs(fp.strict_perplexity, @exp(mean), 1e-7);
    }
}

test "kld: kvCacheFormat names the effective KV width" {
    try testing.expectEqualStrings("bf16", kvCacheFormat(transformer_mod.KVQuantConfig.dense));
    try testing.expectEqualStrings("affine4", kvCacheFormat(transformer_mod.KVQuantConfig.affine(4)));
    try testing.expectEqualStrings("affine8", kvCacheFormat(transformer_mod.KVQuantConfig.affine(8)));
}

test "kld: a prompt directory is named by index and id" {
    const allocator = testing.allocator;
    const plain = try sanitizeDirName(allocator, "wikitext2-test-00-robert-boulter");
    defer allocator.free(plain);
    try testing.expectEqualStrings("wikitext2-test-00-robert-boulter", plain);
    const messy = try sanitizeDirName(allocator, "a b/c.d:e");
    defer allocator.free(messy);
    try testing.expectEqualStrings("a_b_c_d_e", messy);
}

test "kld: the first EOS position bounds the scored span, and no EOS means the whole span" {
    const eos = [_]u32{ 248044, 248046 };
    try std.testing.expectEqual(@as(?usize, 2), firstEosPosition(&[_]u32{ 5, 6, 248046, 7 }, &eos));
    try std.testing.expectEqual(@as(?usize, 0), firstEosPosition(&[_]u32{ 248044, 1 }, &eos));
    try std.testing.expectEqual(@as(?usize, null), firstEosPosition(&[_]u32{ 1, 2, 3 }, &eos));
    try std.testing.expectEqual(@as(?usize, null), firstEosPosition(&[_]u32{}, &eos));
}

test "kld records the budget that shaped the load, never the raw flag" {
    const GiB: u64 = 1 << 30;
    var moe = model_mod.ModelConfig{ .model_type = "qwen3_5_moe" };
    try testing.expectEqual(@as(u64, 0), capturedSsdBudgetGb(&moe, 60 * GiB));
    moe.expert_ssd_budget_bytes = 60 * GiB;
    try testing.expectEqual(@as(u64, 0), capturedSsdBudgetGb(&moe, 60 * GiB));

    var q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true };
    q4.expert_ssd_budget_bytes = 60 * GiB;
    try testing.expectEqual(@as(u64, 60), capturedSsdBudgetGb(&q4, 0));
    q4.expert_ssd_budget_bytes = 0;
    try testing.expectEqual(@as(u64, 48), capturedSsdBudgetGb(&q4, 48 * GiB));
}
