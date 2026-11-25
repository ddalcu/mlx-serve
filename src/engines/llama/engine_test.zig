//! Model-free adapter contracts and optional real-GGUF parity tests.
const std = @import("std");
const engine = @import("engine.zig");
const Adapter = engine.Adapter;
const probe = engine.probe;
const default_context = engine.default_context;
const env = @import("../../env.zig");
const build_options = @import("build_options");
const abi = @import("../../abi/engine.zig");
const engine_pick = @import("../../sdk/quant/gguf/engine_pick.zig");
const book_mod = @import("../own_thread/book.zig");
const Reason = book_mod.Reason;
const io = book_mod.io;
const bridge = @import("bridge.zig");
const policy = @import("policy.zig");
const load = @import("load.zig");

const testing = std.testing;
const contract = @import("../../abi/contract_test.zig");

test {
    _ = book_mod;
    _ = @import("../own_thread/logits.zig");
    _ = @import("../own_thread/token_table.zig");
    _ = policy;
    _ = load;
}

/// A llama.cpp bridge with no model: a sequence's logits after n resident tokens favour `100 + n % 800` (1, every
/// other id 0) and it samples their argmax, so a shaped row changes what comes out; an MTP round accepts every draft.
const FakeBridge = struct {
    pub const vocab = 1024;
    pub const Error = bridge.Error;
    pub const Sampling = struct { temperature: f32, top_k: i32 = 0, top_p: f32 = 1, min_p: f32 = 0 };
    pub const Image = struct { at: u32, width: u32, height: u32, pixels: []const u8 };
    pub const LlamaEngine = struct {
        vision: bool = false,
        pub fn attachVision(e: *LlamaEngine, _: []const u8) !void {
            e.vision = true;
        }
        pub fn hasVision(e: *LlamaEngine) bool {
            return e.vision;
        }
        pub fn imageMarker(_: *LlamaEngine) []const u8 {
            return "<__media__>";
        }
        pub fn open(a: std.mem.Allocator, _: []const u8, _: bridge.OpenOptions) !*LlamaEngine {
            const e = try a.create(LlamaEngine);
            e.* = .{};
            return e;
        }
        pub fn close(e: *LlamaEngine) void {
            testing.allocator.destroy(e);
        }
        pub fn eosToken(_: *LlamaEngine) i32 {
            return 2;
        }
        pub fn isEog(_: *LlamaEngine, t: i32) bool {
            return t == 2 or t == 3;
        }
        pub fn nVocab(_: *LlamaEngine) i32 {
            return vocab;
        }
        pub fn tokenizeText(_: *LlamaEngine, a: std.mem.Allocator, text: []const u8, add_special: bool) ![]i32 {
            const ids = try a.alloc(i32, text.len + @intFromBool(add_special));
            if (add_special) ids[0] = 1;
            for (text, ids[@intFromBool(add_special)..]) |c, *id| id.* = c;
            return ids;
        }
        pub fn detokenizeOne(_: *LlamaEngine, a: std.mem.Allocator, id: i32) ![]u8 {
            return std.fmt.allocPrint(a, "<{d}>", .{id});
        }
        pub fn chatTemplate(_: *LlamaEngine) ?[]const u8 {
            return "{{ messages }}";
        }
        pub fn createContext(_: *LlamaEngine, opts: bridge.ContextOptions) !*LlamaContext {
            const c = try testing.allocator.create(LlamaContext);
            const seqs = try testing.allocator.alloc(LlamaSeq, opts.n_seq);
            for (seqs) |*s| s.* = .{};
            c.* = .{ .seqs = seqs, .drafts = opts.mtp_drafts };
            return c;
        }
    };
    pub const LlamaContext = struct {
        seqs: []LlamaSeq,
        drafts: u32,
        pub fn free(c: *LlamaContext) void {
            for (c.seqs) |*s| s.resident.deinit(testing.allocator);
            testing.allocator.free(c.seqs);
            testing.allocator.destroy(c);
        }
        pub fn mtpDrafts(c: *const LlamaContext) u32 {
            return c.drafts;
        }
        pub fn step(_: *LlamaContext, seqs: []const *LlamaSeq, toks: []const i32) !void {
            for (seqs, toks) |s, t| try s.resident.append(testing.allocator, t);
        }
    };
    pub const LlamaSeq = struct {
        resident: std.ArrayList(i32) = .empty,
        last_used_ns: i64 = 0,
        busy: bool = false,
        row: [vocab]f32 = @splat(0),
        lent: bool = false,
        fn next(s: *const LlamaSeq) i32 {
            return @intCast(100 + s.resident.items.len % 800);
        }
        pub fn logits(s: *LlamaSeq) ?[]f32 {
            s.row = @splat(0);
            s.row[@intCast(s.next())] = 1;
            s.lent = true;
            return &s.row;
        }
        pub fn syncImages(s: *LlamaSeq, p: []const i32, images: []const Image, _: []const i32, cancelled: *const std.atomic.Value(bool), budget: *u32) !u32 {
            if (cancelled.load(.acquire)) return error.Cancelled;
            s.resident.clearRetainingCapacity();
            try s.resident.appendSlice(testing.allocator, p);
            for (images) |image| {
                if (image.pixels.len != @as(usize, image.width) * image.height * 3) return error.InvalidImagePixels;
                // Different RGB must produce different logits even with identical placeholder ids.
                try s.resident.appendNTimes(testing.allocator, 50, image.pixels[0] + 2);
            }
            budget.* = @min(budget.*, 2); // Two decode cells left in this fixture.
            return @intCast(s.resident.items.len);
        }
        pub fn reset(s: *LlamaSeq) void {
            s.resident.clearRetainingCapacity();
        }
        pub fn syncWithFallback(s: *LlamaSeq, p: []const i32) !i32 {
            const common = @min(bridge.commonPrefixLen(s.resident.items, p), p.len -| 1);
            s.resident.shrinkRetainingCapacity(common);
            try s.resident.appendSlice(testing.allocator, p[common..]);
            return @intCast(common);
        }
        /// The argmax of the row `logits` handed out (reshaped or not), else of a fresh one.
        pub fn sample(s: *LlamaSeq, _: Sampling, _: *u64) i32 {
            if (!s.lent) return s.next();
            s.lent = false;
            return @intCast(std.mem.indexOfMax(f32, &s.row));
        }
        pub fn specStep(s: *LlamaSeq, id_last: i32, max_drafts: u32, _: Sampling, _: *u64, out: []i32) ![]i32 {
            try s.resident.append(testing.allocator, id_last);
            for (out[0..max_drafts]) |*o| {
                o.* = s.next();
                try s.resident.append(testing.allocator, o.*);
            }
            out[max_drafts] = s.next();
            return out[0 .. max_drafts + 1];
        }
    };
};

const Fake = Adapter(FakeBridge, testing.allocator);

/// A file whose GGUF header routes to llama.cpp, in `tmp`; returns its absolute path in `buf`.
fn llamaGguf(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    const header = try engine_pick.buildHeader(testing.allocator, &.{.{ .key = "general.architecture", .value = .{ .str = "qwen3" } }});
    defer testing.allocator.free(header);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = header });
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const dir = root[0..try tmp.dir.realPath(io, &root)];
    return std.fmt.bufPrint(buf, "{s}/m.gguf", .{dir});
}

test "llama adapter: the contract holds over a model-free bridge, batched and with MTP rounds" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try llamaGguf(&tmp, &buf);
    inline for (.{ "", "{\"seqs\":1,\"mtp_drafts\":3}", "{\"seqs\":2,\"mtp_drafts\":0}" }) |settings| {
        try contract.run(&Fake.table, .{ .model_dir = path, .model_type = "gguf", .config_json = "", .prompt = &.{ 5, 6, 7 }, .settings_json = settings, .gate_ms = 500 });
    }
}

test "llama adapter: probe reads the GGUF header" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try llamaGguf(&tmp, &buf);
    var v: abi.Verdict = .{};
    try testing.expect(probe(abi.span(u8, path), .{}, &v));
    const ds4 = try engine_pick.buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(ds4);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = ds4 });
    try testing.expect(!probe(abi.span(u8, path), .{}, &v));
    try testing.expect(std.mem.indexOf(u8, abi.items(u8, v.why), "ds4") != null);
    try testing.expect(!probe(abi.span(u8, "/nonexistent/m.gguf"), .{}, &v));
}

const Collect = struct {
    toks: std.ArrayList(u32) = .empty,
    cached: u32 = 0,
    prompt_tokens: u32 = 0,
    done: std.atomic.Value(bool) = .init(false),
    reason: Reason = .stop,

    fn sink(c: *Collect) abi.Sink {
        return .{ .ctx = c, .event = onEvent };
    }
    fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.toks.appendSlice(testing.allocator, abi.items(u32, ev.tokens)) catch {};
        if (ev.kind == .prefilled) {
            c.cached = ev.cached;
            c.prompt_tokens = ev.prompt_tokens;
        }
        if (ev.kind == .finished) {
            c.reason = ev.reason;
            c.done.store(true, .release);
        }
        return .go;
    }
    fn wait(c: *Collect) void {
        while (!c.done.load(.acquire)) io.sleep(.fromMicroseconds(100), .awake) catch {};
    }
};

fn openFake(path: []const u8, settings: []const u8, info: *abi.Info) !*anyopaque {
    const host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 8, .memory_budget = 0, .settings_json = abi.span(u8, settings), .options_json = .{}, .host = &host };
    var v: abi.Verdict = .{};
    return Fake.table.open(&o, info, &v) orelse error.Open;
}

test "llama adapter: an EOG id ends a request unemitted; a follow-up reuses its sequence's prefix; Info" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try llamaGguf(&tmp, &buf);
    var info: abi.Info = undefined;
    const e = try openFake(path, "{\"seqs\":2,\"mtp_drafts\":0}", &info);
    defer Fake.table.close(e);
    try testing.expectEqualSlices(u32, &.{ 2, 3 }, abi.items(u32, info.eos));
    try testing.expectEqual(@as(u32, 2), info.max_concurrent);
    try testing.expectEqual(default_context, info.context_window);
    try testing.expectEqual(@as(u32, FakeBridge.vocab), info.vocab_size);
    try testing.expect(info.caps.batches and !info.caps.drafts and info.caps.penalties and info.caps.logprobs and !info.caps.grammar);

    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(10 + i);
    const eos = [_]u32{121}; // the second token after a 20-token prompt
    var a: Collect = .{};
    defer a.toks.deinit(testing.allocator);
    _ = Fake.table.submit(e, 1, &.{ .prompt = abi.span(u32, &prompt), .max_tokens = 8, .eos = abi.span(u32, &eos) }, a.sink());
    a.wait();
    try testing.expectEqual(Reason.stop, a.reason);
    try testing.expectEqualSlices(u32, &.{120}, a.toks.items);

    var b: Collect = .{};
    defer b.toks.deinit(testing.allocator);
    _ = Fake.table.submit(e, 2, &.{ .prompt = abi.span(u32, &prompt), .max_tokens = 1 }, b.sink());
    b.wait();
    try testing.expectEqual(@as(u32, 19), b.cached); // the whole prompt but its last token
    try testing.expectEqualStrings("{{ messages }}", abi.items(u8, Fake.table.chat_template.?(e)));
    var ids: [4]u32 = undefined;
    try testing.expectEqual(@as(usize, 3), Fake.table.tokenize.?(e, abi.span(u8, "ab"), true, &ids, ids.len));
    try testing.expectEqualSlices(u32, &.{ 1, 'a', 'b' }, ids[0..3]);
    try testing.expectEqualStrings("<5>", abi.items(u8, Fake.table.token_bytes.?(e, 5)));
}

test "llama adapter: a logit bias and a penalty steer the sampled row; logprobs come off the raw row" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try llamaGguf(&tmp, &buf);
    var info: abi.Info = undefined;
    const e = try openFake(path, "{\"seqs\":1,\"mtp_drafts\":3}", &info);
    defer Fake.table.close(e);
    const LogCollect = struct {
        toks: std.ArrayList(u32) = .empty,
        top: std.ArrayList(u32) = .empty,
        done: std.atomic.Value(bool) = .init(false),
        fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.toks.appendSlice(testing.allocator, abi.items(u32, ev.tokens)) catch {};
            for (abi.items(abi.Logprob, ev.logprobs)) |lp| c.top.append(testing.allocator, abi.items(abi.Bias, lp.top)[0].token) catch {};
            if (ev.kind == .finished) c.done.store(true, .release);
            return .go;
        }
    };
    var c: LogCollect = .{};
    defer {
        c.toks.deinit(testing.allocator);
        c.top.deinit(testing.allocator);
    }
    // id 9 biased past the favoured id, then a presence penalty takes it back once it came out (no MTP round:
    // the row is shaped on the host).
    const bias = [_]abi.Bias{.{ .token = 9, .bias = 2 }};
    const req: abi.Request = .{ .prompt = abi.span(u32, &.{ 5, 6, 7 }), .max_tokens = 3, .sampling = .{ .temperature = 0, .presence_penalty = 3, .logit_bias = abi.span(abi.Bias, &bias) }, .logprobs = 1 };
    try testing.expectEqual(abi.Verdict.Code.ok, Fake.table.submit(e, 1, &req, .{ .ctx = &c, .event = LogCollect.onEvent }).code);
    while (!c.done.load(.acquire)) io.sleep(.fromMicroseconds(100), .awake) catch {};
    try testing.expectEqualSlices(u32, &.{ 9, 104, 105 }, c.toks.items);
    try testing.expectEqualSlices(u32, &.{ 103, 104, 105 }, c.top.items); // the raw row's favourite
}

test "llama adapter: a sidecar enables image atoms, distinct pixels miss cache and text follows" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try llamaGguf(&tmp, &buf);
    try tmp.dir.writeFile(io, .{ .sub_path = "mmproj-m.gguf", .data = "GGUF" });
    var info: abi.Info = undefined;
    const e = try openFake(path, "{\"seqs\":1,\"mtp_drafts\":3}", &info);
    defer Fake.table.close(e);
    var disabled_info: abi.Info = undefined;
    const disabled = try openFake(path, "{\"vision\":false}", &disabled_info);
    defer Fake.table.close(disabled);
    try testing.expect(!disabled_info.caps.images);
    try testing.expect(info.caps.images);
    try testing.expectEqualStrings("<__media__>", info.image_placeholder.slice());
    const prompt = [_]u32{ 1, '<', '_', '_', 'm', 'e', 'd', 'i', 'a', '_', '_', '>', 2 };
    const image = [_]abi.Media{.{ .kind = .image, .at = 1, .width = 1, .height = 1, .data = .of(&.{ 3, 0, 0 }), .hash = 7 }};
    var a: Collect = .{};
    defer a.toks.deinit(testing.allocator);
    const r: abi.Request = .{ .prompt = .of(&prompt), .media = .of(&image), .max_tokens = 1 };
    try testing.expectEqual(abi.Verdict.Code.ok, Fake.table.submit(e, 1, &r, a.sink()).code);
    a.wait();
    try testing.expectEqual(Reason.length, a.reason);
    try testing.expect(a.prompt_tokens > prompt.len);
    try testing.expectEqual(@as(u32, 0), a.cached);

    var blue = image;
    blue[0].data = .of(&.{ 0, 0, 3 });
    var next = r;
    next.media = .of(&blue);
    next.max_tokens = 5;
    var b: Collect = .{};
    defer b.toks.deinit(testing.allocator);
    try testing.expectEqual(abi.Verdict.Code.ok, Fake.table.submit(e, 2, &next, b.sink()).code);
    b.wait();
    try testing.expectEqual(Reason.length, b.reason);
    try testing.expectEqual(@as(usize, 2), b.toks.items.len);
    try testing.expect(a.toks.items[0] != b.toks.items[0]);
    try testing.expectEqual(@as(u32, 0), b.cached);

    next.max_tokens = 1;
    next.media = .{};
    next.prompt = .of(&.{ 5, 6, 7 });
    var c: Collect = .{};
    defer c.toks.deinit(testing.allocator);
    try testing.expectEqual(abi.Verdict.Code.ok, Fake.table.submit(e, 3, &next, c.sink()).code);
    c.wait();
    try testing.expectEqual(Reason.length, c.reason);
    try testing.expectEqualSlices(u32, &.{103}, c.toks.items);
}

test "llama adapter: tokenize keeps BOS across a leading media atom and stable atom ids" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try llamaGguf(&tmp, &buf);
    try tmp.dir.writeFile(io, .{ .sub_path = "mmproj-m.gguf", .data = "GGUF" });
    var info: abi.Info = undefined;
    const e = try openFake(path, "{\"seqs\":1}", &info);
    defer Fake.table.close(e);
    var ids: [24]u32 = undefined;
    // The atom at the front must not swallow BOS: it lands before any text id.
    var n = Fake.table.tokenize.?(e, abi.span(u8, "<__media__>abc"), true, &ids, ids.len);
    try testing.expect(n > 12);
    try testing.expectEqual(@as(u32, 1), ids[0]); // BOS
    try testing.expectEqual(@as(u32, '<'), ids[1]); // the atom's first id
    // A trailing atom after text: BOS once, atom ids identical both times.
    n = Fake.table.tokenize.?(e, abi.span(u8, "hi<__media__>"), true, &ids, ids.len);
    try testing.expectEqualSlices(u32, &.{ 1, 'h', 'i', '<', '_', '_', 'm', 'e', 'd', 'i', 'a', '_', '_', '>' }, ids[0..n]);
    // No atom: the plain path keeps its exact shape.
    n = Fake.table.tokenize.?(e, abi.span(u8, "abc"), true, &ids, ids.len);
    try testing.expectEqualSlices(u32, &.{ 1, 'a', 'b', 'c' }, ids[0..n]);
}

// ── against a real GGUF (`MLX_SERVE_TEST_LLAMA_GGUF`; skipped without it)

fn realGguf() ?[]const u8 {
    const raw = env.get(.MLX_SERVE_TEST_LLAMA_GGUF) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

test "llama adapter: the contract holds on a real GGUF" {
    if (!build_options.llama_engine) return error.SkipZigTest;
    const path = realGguf() orelse return error.SkipZigTest;
    const R = Adapter(bridge, testing.allocator);
    try contract.run(&R.table, .{ .model_dir = path, .model_type = "gguf", .config_json = "", .prompt = &.{ 1, 2, 3 }, .settings_json = "{\"seqs\":4}", .timeout_ms = 120_000, .gate_ms = 10_000 });
}

test "llama adapter: greedy tokens equal the plain bridge sequence on a real GGUF" {
    if (!build_options.llama_engine) return error.SkipZigTest;
    const path = realGguf() orelse return error.SkipZigTest;
    const R = Adapter(bridge, testing.allocator);
    const n = 16;
    // The plain bridge sequence (one sequence, no MTP): sync, draw,
    // then emit, feed and draw per tick.
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(testing.allocator);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(testing.allocator);
    {
        const eng = try bridge.LlamaEngine.open(testing.allocator, path, .{});
        defer eng.close();
        const prompt = try eng.tokenizeText(testing.allocator, "The capital of France is", true);
        defer testing.allocator.free(prompt);
        for (prompt) |t| try ids.append(testing.allocator, @intCast(t));
        var ctx = try eng.createContext(.{ .ctx_size = 4096 });
        defer ctx.free();
        const seq = &ctx.seqs[0];
        _ = try seq.syncWithFallback(prompt);
        var tok = seq.argmax();
        while (want.items.len < n and !eng.isEog(tok)) {
            try want.append(testing.allocator, @intCast(tok));
            try seq.eval(tok);
            tok = seq.argmax();
        }
    }

    const host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 4096, .max_concurrent = 1, .memory_budget = 0, .settings_json = abi.span(u8, "{\"seqs\":1,\"mtp_drafts\":0}"), .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = R.table.open(&o, &info, &v) orelse return error.Open;
    defer R.table.close(e);
    var got: Collect = .{};
    defer got.toks.deinit(testing.allocator);
    if (R.table.submit(e, 1, &.{ .prompt = abi.span(u32, ids.items), .max_tokens = n, .sampling = .{ .temperature = 0 }, .eos = info.eos }, got.sink()).code != .ok) return error.Refused;
    got.wait();
    try testing.expectEqualSlices(u32, want.items, got.toks.items);
}

// ── against a real GGUF with its mmproj (`MLX_SERVE_TEST_LLAMA_MMPROJ`; skipped without it)

fn realMmprojGguf() ?[]const u8 {
    const raw = env.get(.MLX_SERVE_TEST_LLAMA_MMPROJ) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

test "llama adapter: the real bridge clamps the decode budget after image expansion" {
    if (!build_options.llama_engine) return error.SkipZigTest;
    const path = realMmprojGguf() orelse return error.SkipZigTest;
    const gguf_files = @import("../../sdk/quant/gguf/files.zig");
    const mmproj = gguf_files.findVisionSidecar(io, testing.allocator, path) orelse return error.SkipZigTest;
    defer testing.allocator.free(mmproj);

    const eng = try bridge.LlamaEngine.open(testing.allocator, path, .{});
    defer eng.close();
    try eng.attachVision(mmproj);
    try testing.expect(eng.hasVision());

    const marker = try eng.tokenizeText(testing.allocator, eng.imageMarker(), false);
    defer testing.allocator.free(marker);
    const pre = try eng.tokenizeText(testing.allocator, "Describe this:", false);
    defer testing.allocator.free(pre);
    const post = try eng.tokenizeText(testing.allocator, " in one word.", false);
    defer testing.allocator.free(post);
    var prompt: std.ArrayList(i32) = .empty;
    defer prompt.deinit(testing.allocator);
    try prompt.appendSlice(testing.allocator, pre);
    const at: u32 = @intCast(prompt.items.len);
    try prompt.appendSlice(testing.allocator, marker);
    try prompt.appendSlice(testing.allocator, post);

    var ctx = try eng.createContext(.{ .ctx_size = 4096, .n_seq = 1 });
    defer ctx.free();
    const seq = &ctx.seqs[0];

    const pixels = try testing.allocator.alloc(u8, 64 * 64 * 3);
    defer testing.allocator.free(pixels);
    for (pixels, 0..) |*px, i| px.* = if (i % 3 == 0) 255 else 0; // solid red
    const images = [_]bridge.Image{.{ .at = at, .width = 64, .height = 64, .pixels = pixels }};
    var cancel = std.atomic.Value(bool).init(false);
    var budget: u32 = 64;
    const expanded = try seq.syncImages(prompt.items, &images, marker, &cancel, &budget);
    try testing.expect(expanded > prompt.items.len); // the image took KV cells
    var rng: u64 = 0;
    try testing.expect(seq.sample(.{ .temperature = 0 }, &rng) >= 0);

    budget = std.math.maxInt(u32);
    try testing.expectEqual(expanded, try seq.syncImages(prompt.items, &images, marker, &cancel, &budget));
    try testing.expectEqual(@as(u32, 4096), expanded + budget);
}
