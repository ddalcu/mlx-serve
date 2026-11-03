//! The ds4 engine behind the engine interface (`abi/engine.zig`): one ds4 session per opened model, decoded on the
//! adapter's own thread one request at a time; later requests wait in submit order.
//!
//! `Open.model_dir` names the `.gguf` file; `settings_json` is `load.Flags`. Penalties, logit bias and logprobs run
//! on the plain decode path over a host copy of the logits (ds4's speculative round reads its own). `render` takes
//! `{"messages":[{"role","content"}],"enable_thinking":bool}`, the transcript after the host's tool synthesis; it
//! and `tokenize` return the full count and write at most `cap` ids (0: nothing to encode or a malformed request).

const std = @import("std");
const env = @import("../../env.zig");
const build_options = @import("build_options");
const abi = @import("../../abi/engine.zig");
const engine_pick = @import("../../sdk/quant/gguf/engine_pick.zig");
const book_mod = @import("../own_thread/book.zig");
const logits = @import("../own_thread/logits.zig");
const TokenTable = @import("../own_thread/token_table.zig").TokenTable;
const load = @import("load.zig");
const round = @import("round.zig");

const Book = book_mod.Book;
const Req = book_mod.Req;
const Reason = book_mod.Reason;
const io = book_mod.io;

const bridge = @import("bridge.zig");

/// The table the host loads for ds4 (`host/engines/served.table`: builds that carry it).
pub const table = Adapter(bridge, std.heap.smp_allocator).table;

/// Today's probe: a GGUF whose header routes to ds4 (`engine_pick.preferredEngine`).
pub fn probe(model_dir: abi.Str, _: abi.Str, out: *abi.Verdict) callconv(.c) bool {
    const path = abi.items(u8, model_dir);
    var info = engine_pick.readFromFile(io, std.heap.smp_allocator, path) catch {
        out.* = abi.Verdict.refuse(.unsupported, "not a GGUF file: the ds4 engine opens one .gguf");
        return false;
    };
    defer info.deinit(std.heap.smp_allocator);
    if (info.split_count > 1) {
        out.* = abi.Verdict.refuse(.unsupported, "a split GGUF: the ds4 engine reads one file (serve it with llama.cpp, or merge the shards)");
        return false;
    }
    if (engine_pick.preferredEngine(info) != .ds4) {
        out.* = abi.Verdict.refuse(.unsupported, "not a ds4 GGUF: its header routes to llama.cpp");
        return false;
    }
    return true;
}

pub fn Adapter(comptime B: type, comptime gpa: std.mem.Allocator) type {
    return struct {
        pub const table: abi.Engine = .{
            .probe = probe,
            .load_bytes = loadBytes,
            .open = open,
            .close = close,
            .submit = submit,
            .cancel = cancel,
            .step = null,
            .status = status,
            .memory = memory,
            .report_json = null,
            .trim = null,
            .media_rows = null,
            .embed = null,
            .decide = null,
            .tokenize = tokenize,
            .token_bytes = tokenBytes,
            .chat_template = null,
            .render = render,
        };

        const State = struct {
            book: Book,
            engine: *B.Ds4Engine,
            session: *B.Ds4Session,
            caps: abi.Caps,
            ctx: u32,
            vocab: usize,
            eos: []u32,
            tokens: TokenTable,
            file_bytes: u64,
            thread: std.Thread = undefined,
            startup_buf: [96]u8 = undefined,
        };

        fn state(e: *anyopaque) *State {
            return @ptrCast(@alignCast(e));
        }

        fn fileBytes(path: []const u8) u64 {
            const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return 0;
            return st.size;
        }

        /// The GGUF and ds4's own estimate of the session's KV and scratch.
        fn loadBytes(o: *const abi.Open, bytes: *u64) callconv(.c) abi.Verdict {
            bytes.* = fileBytes(abi.items(u8, o.model_dir)) + B.contextBytes(B.clampSessionCtx(o.context));
            return .{};
        }

        fn parseFlags(json: []const u8) !load.Flags {
            if (json.len == 0) return .{};
            const parsed = try std.json.parseFromSlice(load.Flags, gpa, json, .{ .ignore_unknown_fields = true });
            defer parsed.deinit();
            return parsed.value;
        }

        fn open(o: *const abi.Open, info: *abi.Info, out: *abi.Verdict) callconv(.c) ?*anyopaque {
            return openOrError(o, info) catch |err| {
                out.* = .{ .code = if (err == error.InvalidSettings) .unsupported else .failed, .why = abi.span(u8, @errorName(err)) };
                return null;
            };
        }

        fn openOrError(o: *const abi.Open, info: *abi.Info) !*State {
            const path = abi.items(u8, o.model_dir);
            const flags = parseFlags(abi.items(u8, o.settings_json)) catch return error.InvalidSettings;
            const engine = try load.open(B, io, gpa, path, flags);
            errdefer engine.close();
            const ctx = B.clampSessionCtx(o.context);
            const session = try engine.createSession(@intCast(ctx));
            errdefer session.free();
            const vocab: usize = engine.vocabSize();
            var eos: std.ArrayList(u32) = .empty;
            errdefer eos.deinit(gpa);
            for (0..vocab) |t| if (engine.isStop(@intCast(t))) try eos.append(gpa, @intCast(t));
            var tokens = try TokenTable.init(gpa, vocab);
            errdefer tokens.deinit(gpa);
            const st = try gpa.create(State);
            errdefer gpa.destroy(st);
            st.* = .{
                .book = .{ .gpa = gpa, .max_waiting = @max(o.max_concurrent, 1) },
                .engine = engine,
                .session = session,
                .caps = .{
                    .penalties = true,
                    .logit_bias = true,
                    .logprobs = true,
                    .owns_tokenizer = true,
                    .owns_template = true,
                    .drafts = engine.mtpDraftTokens() > 1,
                    .prefix_cache = true,
                },
                .ctx = ctx,
                .vocab = vocab,
                .eos = try eos.toOwnedSlice(gpa),
                .tokens = tokens,
                .file_bytes = fileBytes(path),
            };
            errdefer gpa.free(st.eos);
            st.thread = try std.Thread.spawn(.{}, work, .{st});
            const startup = std.fmt.bufPrint(&st.startup_buf, "ds4: {d}-token session, {d} draft tokens per round", .{ ctx, engine.mtpDraftTokens() }) catch "";
            info.* = .{
                .name = abi.span(u8, "ds4"),
                .version = abi.span(u8, build_options.ds4_commit),
                .caps = st.caps,
                .threading = .own_thread,
                .context_window = ctx,
                .max_concurrent = 1,
                .embedding_dims = 0,
                .eos = abi.span(u32, st.eos),
                .startup = abi.span(u8, startup),
                .vocab_size = @intCast(vocab),
            };
            return st;
        }

        fn close(e: *anyopaque) callconv(.c) void {
            const st = state(e);
            st.book.close();
            st.thread.join();
            st.book.deinit();
            st.tokens.deinit(gpa);
            gpa.free(st.eos);
            st.session.free();
            st.engine.close();
            gpa.destroy(st);
        }

        fn submit(e: *anyopaque, id: u64, r: *const abi.Request, sink: abi.Sink) callconv(.c) abi.Verdict {
            const st = state(e);
            var v: abi.Verdict = .{};
            _ = st.book.add(st.caps, id, r, sink, &v);
            return v;
        }

        fn cancel(e: *anyopaque, id: u64) callconv(.c) void {
            state(e).book.cancel(id);
        }

        fn status(e: *anyopaque, out: *abi.Status) callconv(.c) void {
            state(e).book.status(out);
        }

        /// What props has always shown for ds4: the GGUF plus the session's estimate (ds4 allocates outside MLX).
        fn memory(e: *anyopaque, out: *abi.Memory) callconv(.c) void {
            const st = state(e);
            const active = st.file_bytes + B.contextBytes(st.ctx);
            out.* = .{ .active = active, .cache = 0, .peak = active };
        }

        fn tokenize(e: *anyopaque, text: abi.Str, _: bool, out: [*]u32, cap: usize) callconv(.c) usize {
            const st = state(e);
            const ids = st.engine.tokenizeText(gpa, abi.items(u8, text)) catch return 0;
            defer gpa.free(ids);
            for (ids[0..@min(ids.len, cap)], 0..) |t, i| out[i] = @intCast(t);
            return ids.len;
        }

        fn tokenBytes(e: *anyopaque, token: u32) callconv(.c) abi.Str {
            const st = state(e);
            return abi.span(u8, st.tokens.get(gpa, st.engine, token));
        }

        const Chat = struct {
            messages: []const struct { role: []const u8, content: []const u8 = "" } = &.{},
            enable_thinking: bool = false,
        };

        /// ds4's own template over the transcript: the first system message is its system prompt.
        fn render(e: *anyopaque, chat_json: abi.Str, out: [*]u32, cap: usize) callconv(.c) usize {
            const st = state(e);
            const parsed = std.json.parseFromSlice(Chat, gpa, abi.items(u8, chat_json), .{ .ignore_unknown_fields = true }) catch return 0;
            defer parsed.deinit();
            var system: ?[]const u8 = null;
            const turns = gpa.alloc(B.Ds4Engine.ChatTurn, parsed.value.messages.len) catch return 0;
            defer gpa.free(turns);
            var n: usize = 0;
            for (parsed.value.messages) |m| {
                if (system == null and std.mem.eql(u8, m.role, "system")) {
                    system = m.content;
                    continue;
                }
                turns[n] = .{ .role = m.role, .content = m.content };
                n += 1;
            }
            const think: B.ThinkMode = if (parsed.value.enable_thinking) .high else .none;
            const ids = st.engine.encodeChatTranscript(gpa, system, turns[0..n], think) catch return 0;
            defer gpa.free(ids);
            for (ids[0..@min(ids.len, cap)], 0..) |t, i| out[i] = @intCast(t);
            return ids.len;
        }

        fn work(st: *State) void {
            while (st.book.next(true)) |r| {
                const reason = serve(st, r) catch |err| {
                    // Leave nothing half-built behind for the next request.
                    st.session.invalidate();
                    st.book.done(r, .failed, @errorName(err));
                    continue;
                };
                st.book.done(r, reason, "");
            }
        }

        /// Sync the session to the prompt (ds4 keeps the common prefix in its live KV), then decode round by round.
        fn serve(st: *State, r: *Req) !Reason {
            if (r.isCancelled()) return .cancelled;
            const prompt = try gpa.alloc(i32, r.prompt.len);
            defer gpa.free(prompt);
            for (r.prompt, prompt) |t, *p| p.* = @intCast(t);
            const cached = st.session.commonPrefix(prompt) catch 0;
            try st.session.sync(prompt);
            if (r.prefilled(@intCast(@max(cached, 0)))) |why| return why;

            var rng = r.seed();
            const d: round.Draw = .{ .temperature = r.sampling.temperature, .top_k = r.sampling.top_k, .top_p = r.sampling.top_p, .min_p = r.sampling.min_p, .rng = &rng };
            const shaped = logits.shapes(r.sampling);
            const host = shaped or r.logprobs > 0;
            const raw = try gpa.alloc(f32, if (host) st.vocab else 0);
            defer gpa.free(raw);
            const work_row = try gpa.alloc(f32, if (shaped) st.vocab else 0);
            defer gpa.free(work_row);
            const top = try gpa.alloc(abi.Bias, r.logprobs);
            defer gpa.free(top);
            var history: logits.History = .{};
            defer history.deinit(gpa);
            const spec = !host and round.shouldSpeculate(st.engine.mtpDraftTokens(), d.temperature, st.engine.embedded_mtp);
            if (spec) r.stats.spec_mode = abi.span(u8, "mtp");

            while (true) {
                if (host) {
                    if (st.session.copyLogits(raw) != raw.len) return error.Ds4LogitsUnavailable;
                    if (shaped) {
                        @memcpy(work_row, raw);
                        logits.shape(f32, work_row, r.sampling, &history);
                        try st.session.setLogits(work_row);
                    }
                }
                const t = round.draw(st.session, d);
                if (t < 0) return error.Ds4SampleFailed;
                const tok: u32 = @intCast(t);
                if (r.isEos(tok)) return .stop;

                if (spec) {
                    var buf: [round.max_round_tokens]i32 = undefined;
                    const got = try round.speculate(st.engine, st.session, t, d, r.max_tokens - r.emitted, &buf);
                    var ids: [round.max_round_tokens]u32 = undefined;
                    for (got, ids[0..got.len]) |g, *id| id.* = @intCast(g);
                    r.stats.rounds += 1;
                    r.stats.accepted += got.len -| 1;
                    if (r.commit(ids[0..got.len], &.{})) |why| return why;
                    continue;
                }

                var lp: [1]abi.Logprob = undefined;
                if (r.logprobs > 0) lp[0] = logits.logprob(f32, raw, tok, top);
                const before = r.emitted;
                const why = r.commit(&.{tok}, &lp);
                // An emitted token goes into the KV: the next turn's prefix.
                if (r.emitted > before) {
                    try st.session.eval(t);
                    if (shaped) try history.add(gpa, tok);
                }
                if (why) |w| return w;
            }
        }
    };
}

// ── tests ──

const testing = std.testing;
const contract = @import("../../abi/contract_test.zig");

test {
    _ = book_mod;
    _ = logits;
    _ = @import("../own_thread/token_table.zig");
    _ = round;
    _ = load;
}

/// A ds4 bridge with no model: its logits always favour id 50 (2), then `100 + n % 800` after n tokens in the
/// session (1), so a shaped row changes what comes out; a speculative round commits three tokens.
const FakeBridge = struct {
    pub const vocab = 1024;
    pub const ThinkMode = enum(c_int) { none, high };
    pub fn clampSessionCtx(requested: u32) u32 {
        return if (requested == 0) 4096 else requested;
    }
    pub fn contextBytes(ctx: u32) u64 {
        return @as(u64, ctx) * 1024;
    }
    pub fn ggufDeclaresEmbeddedMtp(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
        return true;
    }
    pub const Ds4Engine = struct {
        pub const ChatTurn = struct { role: []const u8, content: []const u8 };
        embedded_mtp: bool,
        drafts: c_int,
        pub fn open(a: std.mem.Allocator, _: []const u8, opts: anytype) !*Ds4Engine {
            const e = try a.create(Ds4Engine);
            e.* = .{ .embedded_mtp = opts.embedded_mtp, .drafts = if (opts.prefill_chunk == 1) 0 else 4 };
            return e;
        }
        pub fn close(e: *Ds4Engine) void {
            testing.allocator.destroy(e);
        }
        pub fn mtpDraftTokens(e: *Ds4Engine) c_int {
            return e.drafts;
        }
        pub fn hasMtp(e: *Ds4Engine) bool {
            return e.drafts > 0;
        }
        pub fn eosToken(_: *Ds4Engine) i32 {
            return 2;
        }
        pub fn isStop(_: *Ds4Engine, t: i32) bool {
            return t == 2;
        }
        pub fn vocabSize(_: *Ds4Engine) u32 {
            return vocab;
        }
        pub fn tokenizeText(_: *Ds4Engine, a: std.mem.Allocator, text: []const u8) ![]i32 {
            const ids = try a.alloc(i32, text.len);
            for (text, ids) |c, *id| id.* = c;
            return ids;
        }
        pub fn detokenizeOne(_: *Ds4Engine, a: std.mem.Allocator, id: i32) ![]u8 {
            return std.fmt.allocPrint(a, "<{d}>", .{id});
        }
        /// [system length or 0] + each turn's (role length, content length) + 1 when thinking.
        pub fn encodeChatTranscript(_: *Ds4Engine, a: std.mem.Allocator, system: ?[]const u8, turns: []const ChatTurn, think: ThinkMode) ![]i32 {
            var ids: std.ArrayList(i32) = .empty;
            try ids.append(a, @intCast(if (system) |s| s.len else 0));
            for (turns) |t| try ids.appendSlice(a, &.{ @intCast(t.role.len), @intCast(t.content.len) });
            if (think == .high) try ids.append(a, 1);
            return ids.toOwnedSlice(a);
        }
        pub fn createSession(_: *Ds4Engine, _: i32) !*Ds4Session {
            const s = try testing.allocator.create(Ds4Session);
            s.* = .{};
            return s;
        }
    };
    pub const Ds4Session = struct {
        toks: std.ArrayList(i32) = .empty,
        row: [vocab]f32 = @splat(0),

        fn refresh(s: *Ds4Session) void {
            s.row = @splat(0);
            s.row[50] = 2;
            s.row[100 + s.toks.items.len % 800] = 1;
        }
        pub fn free(s: *Ds4Session) void {
            s.toks.deinit(testing.allocator);
            testing.allocator.destroy(s);
        }
        pub fn invalidate(s: *Ds4Session) void {
            s.toks.clearRetainingCapacity();
        }
        pub fn commonPrefix(s: *Ds4Session, p: []const i32) !i32 {
            var n: usize = 0;
            while (n < @min(p.len, s.toks.items.len) and p[n] == s.toks.items[n]) n += 1;
            return @intCast(n);
        }
        pub fn sync(s: *Ds4Session, p: []const i32) !void {
            s.toks.clearRetainingCapacity();
            try s.toks.appendSlice(testing.allocator, p);
            s.refresh();
        }
        pub fn eval(s: *Ds4Session, t: i32) !void {
            try s.toks.append(testing.allocator, t);
            s.refresh();
        }
        pub fn argmax(s: *Ds4Session) i32 {
            return @intCast(std.mem.indexOfMax(f32, &s.row));
        }
        pub fn sample(s: *Ds4Session, _: f32, _: c_int, _: f32, _: f32, _: *u64) i32 {
            return s.argmax();
        }
        pub fn copyLogits(s: *Ds4Session, out: []f32) usize {
            @memcpy(out, &s.row);
            return vocab;
        }
        pub fn setLogits(s: *Ds4Session, l: []const f32) !void {
            @memcpy(&s.row, l);
        }
        pub fn evalSpeculative(s: *Ds4Session, first: i32, max: i32, _: i32, out: []i32) !i32 {
            const n: usize = @intCast(@min(max, 3));
            out[0] = first;
            try s.eval(first);
            for (1..n) |i| {
                out[i] = s.argmax();
                try s.eval(out[i]);
            }
            return @intCast(n);
        }
        pub fn evalSpeculativeSampled(s: *Ds4Session, first: i32, max: i32, eos: i32, _: f32, _: c_int, _: f32, _: f32, _: *u64, out: []i32) !i32 {
            return s.evalSpeculative(first, max, eos, out);
        }
    };
};

const Fake = Adapter(FakeBridge, testing.allocator);

/// A file whose GGUF header routes to ds4, in `tmp`; returns its absolute path in `buf`.
fn ds4Gguf(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    const header = try engine_pick.buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(header);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = header });
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const dir = root[0..try tmp.dir.realPath(io, &root)];
    return std.fmt.bufPrint(buf, "{s}/m.gguf", .{dir});
}

test "ds4 adapter: the contract holds over a model-free bridge, speculative rounds and the plain path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    inline for (.{ "", "{\"prefill_chunk\":1}" }) |settings| { // drafts on, then off
        try contract.run(&Fake.table, .{ .model_dir = path, .model_type = "gguf", .config_json = "", .prompt = &.{ 5, 6, 7 }, .settings_json = settings, .gate_ms = 500 });
    }
}

test "ds4 adapter: probe reads the GGUF header" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    var v: abi.Verdict = .{};
    try testing.expect(probe(abi.span(u8, path), .{}, &v));
    const llama = try engine_pick.buildHeader(testing.allocator, &.{.{ .key = "general.architecture", .value = .{ .str = "llama" } }});
    defer testing.allocator.free(llama);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = llama });
    try testing.expect(!probe(abi.span(u8, path), .{}, &v));
    try testing.expect(std.mem.indexOf(u8, abi.items(u8, v.why), "llama.cpp") != null);
    try testing.expect(!probe(abi.span(u8, "/nonexistent/m.gguf"), .{}, &v));
}

const Collect = struct {
    toks: std.ArrayList(u32) = .empty,
    eos: std.ArrayList(u32) = .empty,
    lps: std.ArrayList(struct { token: u32, logprob: f32, top_len: usize, top_token: u32 }) = .empty,
    done: std.atomic.Value(bool) = .init(false),
    reason: Reason = .stop,

    fn sink(c: *Collect) abi.Sink {
        return .{ .ctx = c, .event = onEvent };
    }
    fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.toks.appendSlice(testing.allocator, abi.items(u32, ev.tokens)) catch {};
        for (abi.items(abi.Logprob, ev.logprobs)) |lp| c.lps.append(testing.allocator, .{ .token = lp.token, .logprob = lp.logprob, .top_len = lp.top.len, .top_token = if (lp.top.len > 0) abi.items(abi.Bias, lp.top)[0].token else 0 }) catch {};
        if (ev.kind == .finished) {
            c.reason = ev.reason;
            c.done.store(true, .release);
        }
        return .go;
    }
    fn deinit(c: *Collect) void {
        c.toks.deinit(testing.allocator);
        c.eos.deinit(testing.allocator);
        c.lps.deinit(testing.allocator);
    }
    fn wait(c: *Collect) void {
        while (!c.done.load(.acquire)) io.sleep(.fromMicroseconds(100), .awake) catch {};
    }
};

/// Opens `T` on `path`, runs one request to its end, closes; the model's Caps (its EOS set into `c.eos`).
fn runOne(comptime T: type, path: []const u8, settings: []const u8, req: abi.Request, c: *Collect) !abi.Caps {
    var host: abi.Host = .{ .ctx = undefined, .log = struct {
        fn f(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}
    }.f, .wake = struct {
        fn f(_: *anyopaque) callconv(.c) void {}
    }.f };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 4, .memory_budget = 0, .settings_json = abi.span(u8, settings), .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = T.table.open(&o, &info, &v) orelse return error.Open;
    defer T.table.close(e);
    try c.eos.appendSlice(testing.allocator, abi.items(u32, info.eos));
    if (T.table.submit(e, 1, &req, c.sink()).code != .ok) return error.Refused;
    c.wait();
    return info.caps;
}

test "ds4 adapter: a logit bias and a penalty steer the plain path; logprobs come with each token" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    const prompt = [_]u32{ 5, 6, 7 };

    var plain: Collect = .{};
    defer plain.deinit();
    const caps = try runOne(Fake, path, "", .{ .prompt = abi.span(u32, &prompt), .max_tokens = 4, .sampling = .{ .temperature = 0 } }, &plain);
    try testing.expectEqualSlices(u32, &.{ 50, 50, 50, 50 }, plain.toks.items);
    try testing.expectEqualSlices(u32, &.{2}, plain.eos.items);
    try testing.expect(caps.drafts and caps.owns_template);

    // The bias rules out 50: the session's own next id comes out instead.
    const bias = [_]abi.Bias{.{ .token = 50, .bias = -5 }};
    var biased: Collect = .{};
    defer biased.deinit();
    _ = try runOne(Fake, path, "", .{ .prompt = abi.span(u32, &prompt), .max_tokens = 2, .sampling = .{ .temperature = 0, .logit_bias = abi.span(abi.Bias, &bias) }, .logprobs = 1 }, &biased);
    try testing.expectEqualSlices(u32, &.{ 103, 104 }, biased.toks.items);
    try testing.expectEqual(@as(usize, 2), biased.lps.items.len);
    try testing.expectEqual(@as(usize, 1), biased.lps.items[0].top_len);
    // Read off the raw row, where 50 (not 103) is the likeliest.
    try testing.expectEqual(@as(u32, 50), biased.lps.items[0].top_token);

    // A presence penalty on the token that came out keeps it from coming out again.
    var pen: Collect = .{};
    defer pen.deinit();
    _ = try runOne(Fake, path, "", .{ .prompt = abi.span(u32, &prompt), .max_tokens = 3, .sampling = .{ .temperature = 0, .presence_penalty = 2 } }, &pen);
    try testing.expectEqualSlices(u32, &.{ 50, 104, 105 }, pen.toks.items);
}

test "ds4 adapter: requests past the one session wait their turn, in submit order" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    var host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 4, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v).?;
    defer Fake.table.close(e);
    var cs: [3]Collect = @splat(.{});
    defer for (&cs) |*c| c.deinit();
    const prompt = [_]u32{ 5, 6, 7 };
    for (&cs, 0..) |*c, i| try testing.expectEqual(abi.Verdict.Code.ok, Fake.table.submit(e, i, &.{ .prompt = abi.span(u32, &prompt), .max_tokens = 3 }, c.sink()).code);
    for (&cs) |*c| {
        c.wait();
        try testing.expectEqual(Reason.length, c.reason);
        try testing.expectEqual(@as(usize, 3), c.toks.items.len);
    }
}

test "ds4 adapter: render maps the transcript onto ds4's template; tokenize and token bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    var host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 1, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v).?;
    defer Fake.table.close(e);
    var out: [8]u32 = undefined;
    const chat = "{\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},{\"role\":\"user\",\"content\":\"hi\"}],\"enable_thinking\":true}";
    try testing.expectEqual(@as(usize, 4), Fake.table.render.?(e, abi.span(u8, chat), &out, out.len));
    try testing.expectEqualSlices(u32, &.{ 8, 4, 2, 1 }, out[0..4]);
    try testing.expectEqual(@as(usize, 0), Fake.table.render.?(e, abi.span(u8, "not json"), &out, out.len));
    try testing.expectEqual(@as(usize, 3), Fake.table.tokenize.?(e, abi.span(u8, "abc"), false, &out, 2));
    try testing.expectEqual(@as(u32, 'b'), out[1]);
    try testing.expectEqualStrings("<7>", abi.items(u8, Fake.table.token_bytes.?(e, 7)));
}

// ── against a real GGUF (`MLX_SERVE_TEST_DS4_GGUF`; skipped without it)

fn realGguf() ?[]const u8 {
    const raw = env.get(.MLX_SERVE_TEST_DS4_GGUF) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

test "ds4 adapter: the contract holds on a real GGUF" {
    if (!build_options.ds4_engine) return error.SkipZigTest;
    const path = realGguf() orelse return error.SkipZigTest;
    const R = Adapter(bridge, testing.allocator);
    try contract.run(&R.table, .{ .model_dir = path, .model_type = "gguf", .config_json = "", .prompt = &.{ 0, 1, 2 }, .settings_json = "{\"mtp\":false}", .timeout_ms = 600_000, .gate_ms = 60_000 });
}

test "ds4 adapter: greedy tokens equal the plain bridge sequence on a real GGUF" {
    if (!build_options.ds4_engine) return error.SkipZigTest;
    const path = realGguf() orelse return error.SkipZigTest;
    const R = Adapter(bridge, testing.allocator);
    const n = 16;
    var ids: [64]u32 = undefined;
    var got: Collect = .{};
    defer got.deinit();
    var prompt_len: usize = 0;
    {
        // The prompt through the adapter's own render.
        var host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
        const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 4096, .max_concurrent = 1, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &host };
        var info: abi.Info = undefined;
        var v: abi.Verdict = .{};
        const e = R.table.open(&o, &info, &v) orelse return error.Open;
        defer R.table.close(e);
        prompt_len = R.table.render.?(e, abi.span(u8, "{\"messages\":[{\"role\":\"user\",\"content\":\"Name three primes.\"}]}"), &ids, ids.len);
        try testing.expect(prompt_len > 0 and prompt_len <= ids.len);
        const req: abi.Request = .{ .prompt = abi.span(u32, ids[0..prompt_len]), .max_tokens = n, .sampling = .{ .temperature = 0 }, .eos = info.eos };
        if (R.table.submit(e, 1, &req, got.sink()).code != .ok) return error.Refused;
        got.wait();
    }
    // The plain bridge sequence: sync, then argmax and eval, or one
    // speculative round, per tick.
    const engine = try load.open(bridge, io, testing.allocator, path, .{});
    defer engine.close();
    const session = try engine.createSession(@intCast(bridge.clampSessionCtx(4096)));
    defer session.free();
    var p: [64]i32 = undefined;
    for (ids[0..prompt_len], p[0..prompt_len]) |t, *q| q.* = @intCast(t);
    try session.sync(p[0..prompt_len]);
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(testing.allocator);
    var rng: u64 = 0;
    const d: round.Draw = .{ .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0, .rng = &rng };
    decode: while (want.items.len < n) {
        const t = round.draw(session, d);
        if (engine.isStop(t)) break;
        if (round.shouldSpeculate(engine.mtpDraftTokens(), 0, engine.embedded_mtp)) {
            var spec: [round.max_round_tokens]i32 = undefined;
            for (try round.speculate(engine, session, t, d, @intCast(n - want.items.len), &spec)) |s| {
                if (engine.isStop(s)) break :decode;
                try want.append(testing.allocator, @intCast(s));
            }
            continue;
        }
        try want.append(testing.allocator, @intCast(t));
        try session.eval(t);
    }
    try testing.expectEqualSlices(u32, want.items[0..@min(want.items.len, n)], got.toks.items);
}
