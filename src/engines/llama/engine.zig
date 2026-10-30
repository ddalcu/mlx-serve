//! The llama.cpp engine behind the engine interface (`abi/engine.zig`): one context of `seqs` sequences per opened
//! model, decoded together on the adapter's own thread; requests past the free sequences wait in submit order.
//!
//! `Open.model_dir` names the `.gguf` file (a split GGUF's first shard); `settings_json` is `load.Settings`;
//! `Open.context` is each sequence's context (0: 8192, the scheduler's default). The host renders the GGUF's chat
//! template (`chat_template`) and tokenizes through `tokenize`, whose `specials` adds the model's BOS. Penalties,
//! logit bias and logprobs run on the sequence's own logits row before the shim's sampler draws from it (such
//! requests take no MTP round); grammar, the think budget and the tool gate are refused by name.

const std = @import("std");
const env = @import("../../env.zig");
const build_options = @import("build_options");
const abi = @import("../../abi/engine.zig");
const engine_pick = @import("../../sdk/quant/gguf/engine_pick.zig");
const log = @import("../../sdk/util/log.zig");
const book_mod = @import("../own_thread/book.zig");
const TokenTable = @import("../own_thread/token_table.zig").TokenTable;
const logits = @import("../own_thread/logits.zig");
const load = @import("load.zig");
const policy = @import("policy.zig");

const Book = book_mod.Book;
const Req = book_mod.Req;
const Reason = book_mod.Reason;
const io = book_mod.io;

const bridge = @import("bridge.zig");

/// The table the host loads for llama.cpp (`host/engines/served.table`: builds that carry it).
pub const table = Adapter(bridge, std.heap.smp_allocator).table;

/// The context a sequence gets when the host names none.
pub const default_context: u32 = 8192;

/// Every GGUF whose header does not route to ds4 (`engine_pick.preferredEngine`).
pub fn probe(model_dir: abi.Str, _: abi.Str, out: *abi.Verdict) callconv(.c) bool {
    var info = engine_pick.readFromFile(io, std.heap.smp_allocator, abi.items(u8, model_dir)) catch {
        out.* = abi.Verdict.refuse(.unsupported, "not a GGUF file: the llama.cpp engine opens a .gguf");
        return false;
    };
    defer info.deinit(std.heap.smp_allocator);
    if (engine_pick.preferredEngine(info) != .llama) {
        out.* = abi.Verdict.refuse(.unsupported, "a ds4 GGUF: its header routes to the ds4 engine");
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
            .chat_template = chatTemplate,
            .render = null,
        };

        const State = struct {
            book: Book,
            engine: *B.LlamaEngine,
            ctx: *B.LlamaContext,
            caps: abi.Caps,
            eos: []u32,
            tokens: TokenTable,
            file_bytes: u64,
            thread: std.Thread = undefined,
            batched_logged: bool = false,
            startup_buf: [96]u8 = undefined,
        };

        /// A request decoding on a sequence: the token it drew last round, not yet emitted nor fed, with its logprob
        /// when asked. A request that shapes its logits keeps the ids it emitted (`history`) and, for logprobs, the
        /// raw row (`raw`) the shaping overwrites.
        const Live = struct {
            r: *Req,
            seq: *B.LlamaSeq,
            next: i32,
            rng: u64,
            history: logits.History = .{},
            raw: []f32 = &.{},
            top: []abi.Bias = &.{},
            lp: [1]abi.Logprob = undefined,

            fn shaped(l: *const Live) bool {
                return logits.shapes(l.r.sampling);
            }

            /// Penalties, logit bias and logprobs need the row on the host; such a request takes no MTP round.
            fn onHost(l: *const Live) bool {
                return l.shaped() or l.r.logprobs > 0;
            }

            /// The next token from the row the last decode left: shaped first when the request asks, its logprob
            /// read off the raw row.
            fn draw(l: *Live) !i32 {
                const shaped_row = l.shaped();
                var row: []f32 = &.{};
                if (l.onHost()) {
                    row = l.seq.logits() orelse return error.LlamaLogitsUnavailable;
                    if (shaped_row and l.r.logprobs > 0) @memcpy(l.raw, row);
                    if (shaped_row) logits.shape(f32, row, l.r.sampling, &l.history);
                }
                const t = l.seq.sample(sampling(l.r), &l.rng);
                if (t < 0) return error.LlamaSampleFailed;
                if (l.r.logprobs > 0) l.lp[0] = logits.logprob(f32, if (shaped_row) l.raw else row, @intCast(t), l.top);
                return t;
            }

            /// Emits `next`; null: decode on.
            fn emit(l: *Live) !?Reason {
                const before = l.r.emitted;
                const why = l.r.commit(&.{@intCast(l.next)}, if (l.r.logprobs > 0) &l.lp else &.{});
                if (l.shaped() and l.r.emitted > before) try l.history.add(gpa, @intCast(l.next));
                return why;
            }

            fn deinit(l: *Live) void {
                l.history.deinit(gpa);
                gpa.free(l.raw);
                gpa.free(l.top);
            }
        };

        fn state(e: *anyopaque) *State {
            return @ptrCast(@alignCast(e));
        }

        fn fileBytes(path: []const u8) u64 {
            const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return 0;
            return st.size;
        }

        /// The GGUF, as the scheduler's preflight bills it: llama.cpp does not expose its KV ahead of a context.
        fn loadBytes(o: *const abi.Open, bytes: *u64) callconv(.c) abi.Verdict {
            bytes.* = fileBytes(abi.items(u8, o.model_dir));
            return .{};
        }

        fn parseSettings(json: []const u8) !load.Settings {
            if (json.len == 0) return .{};
            const parsed = try std.json.parseFromSlice(load.Settings, gpa, json, .{ .ignore_unknown_fields = true });
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
            const settings = parseSettings(abi.items(u8, o.settings_json)) catch return error.InvalidSettings;
            const n_ctx = if (o.context > 0) o.context else default_context;
            const opened = try load.open(B, io, gpa, path, settings, n_ctx);
            errdefer opened.engine.close();
            errdefer opened.ctx.free();
            const vocab: usize = @intCast(@max(opened.engine.nVocab(), 0));
            var eos: std.ArrayList(u32) = .empty;
            errdefer eos.deinit(gpa);
            for (0..vocab) |t| if (opened.engine.isEog(@intCast(t))) try eos.append(gpa, @intCast(t));
            var tokens = try TokenTable.init(gpa, vocab);
            errdefer tokens.deinit(gpa);
            const st = try gpa.create(State);
            errdefer gpa.destroy(st);
            const seqs: u32 = @intCast(opened.ctx.seqs.len);
            st.* = .{
                .book = .{ .gpa = gpa, .max_waiting = @max(o.max_concurrent, 1) },
                .engine = opened.engine,
                .ctx = opened.ctx,
                .caps = .{
                    .penalties = true,
                    .logit_bias = true,
                    .logprobs = true,
                    .owns_tokenizer = true,
                    .batches = seqs > 1,
                    .drafts = opened.ctx.mtpDrafts() > 0,
                    .prefix_cache = true,
                },
                .eos = try eos.toOwnedSlice(gpa),
                .tokens = tokens,
                .file_bytes = fileBytes(path),
            };
            errdefer gpa.free(st.eos);
            st.thread = try std.Thread.spawn(.{}, work, .{st});
            const startup = std.fmt.bufPrint(&st.startup_buf, "llama.cpp: {d} sequences x {d} tokens, {d} MTP drafts", .{ seqs, n_ctx, opened.ctx.mtpDrafts() }) catch "";
            info.* = .{
                .name = abi.span(u8, "llama.cpp"),
                .version = abi.span(u8, build_options.llama_tag),
                .caps = st.caps,
                .threading = .own_thread,
                .context_window = n_ctx,
                .max_concurrent = seqs,
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
            st.ctx.free();
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

        /// What props has always shown for llama.cpp: the GGUF (its allocations sit outside MLX, KV unexposed).
        fn memory(e: *anyopaque, out: *abi.Memory) callconv(.c) void {
            const st = state(e);
            out.* = .{ .active = st.file_bytes, .cache = 0, .peak = st.file_bytes };
        }

        fn tokenize(e: *anyopaque, text: abi.Str, specials: bool, out: [*]u32, cap: usize) callconv(.c) usize {
            const st = state(e);
            const ids = st.engine.tokenizeText(gpa, abi.items(u8, text), specials) catch return 0;
            defer gpa.free(ids);
            for (ids[0..@min(ids.len, cap)], 0..) |t, i| out[i] = @intCast(t);
            return ids.len;
        }

        fn tokenBytes(e: *anyopaque, token: u32) callconv(.c) abi.Str {
            const st = state(e);
            return abi.span(u8, st.tokens.get(gpa, st.engine, token));
        }

        /// The GGUF's embedded template (borrowed for the engine's lifetime); empty when it ships none.
        fn chatTemplate(e: *anyopaque) callconv(.c) abi.Str {
            return abi.span(u8, state(e).engine.chatTemplate() orelse "");
        }

        fn sampling(r: *const Req) B.Sampling {
            return .{
                .temperature = r.sampling.temperature,
                .top_k = @intCast(@min(r.sampling.top_k, std.math.maxInt(i32))),
                .top_p = r.sampling.top_p,
                .min_p = r.sampling.min_p,
            };
        }

        fn work(st: *State) void {
            var live: std.ArrayList(Live) = .empty;
            defer live.deinit(gpa);
            while (true) {
                // Fill the free sequences; wait for work only when nothing decodes.
                while (live.items.len < st.ctx.seqs.len) {
                    const r = st.book.next(live.items.len == 0) orelse break;
                    admit(st, &live, r);
                }
                if (live.items.len == 0) return; // closed, and every request finished
                tick(st, &live);
            }
        }

        /// Claim a sequence, sync it to the prompt (reusing the KV prefix a previous request left there) and draw
        /// the first token.
        fn admit(st: *State, live: *std.ArrayList(Live), r: *Req) void {
            if (r.isCancelled()) return st.book.done(r, .cancelled, "");
            const prompt = gpa.alloc(i32, r.prompt.len) catch return st.book.done(r, .failed, "OutOfMemory");
            defer gpa.free(prompt);
            for (r.prompt, prompt) |t, *p| p.* = @intCast(t);
            const seq = &st.ctx.seqs[policy.pickSeq(st.ctx.seqs, prompt).?]; // a free one: fewer live than seqs
            seq.busy = true;
            seq.last_used_ns = @intCast(book_mod.nowNs());
            var l: Live = .{ .r = r, .seq = seq, .next = 0, .rng = r.seed() };
            l.top = gpa.alloc(abi.Bias, r.logprobs) catch return end(st, &l, .failed, "OutOfMemory");
            if (l.shaped() and r.logprobs > 0) l.raw = gpa.alloc(f32, st.tokens.pieces.len) catch return end(st, &l, .failed, "OutOfMemory");
            const cached = seq.syncWithFallback(prompt) catch |err| return end(st, &l, .failed, @errorName(err));
            if (r.prefilled(@intCast(@max(cached, 0)))) |why| return end(st, &l, why, "");
            l.next = l.draw() catch |err| return end(st, &l, .failed, @errorName(err));
            live.append(gpa, l) catch return end(st, &l, .failed, "OutOfMemory");
        }

        fn end(st: *State, l: *Live, reason: Reason, why: []const u8) void {
            l.seq.busy = false;
            l.deinit();
            st.book.done(l.r, reason, why);
        }

        fn endAt(st: *State, live: *std.ArrayList(Live), i: usize, reason: Reason, why: []const u8) void {
            var l = live.orderedRemove(i);
            end(st, &l, reason, why);
        }

        /// One decode tick: each request emits the token it drew last
        /// round, then the ones still decoding feed theirs in ONE batched step and draw the next. A request alone
        /// on a context with an MTP head runs an MTP round instead.
        fn tick(st: *State, live: *std.ArrayList(Live)) void {
            var i: usize = 0;
            while (i < live.items.len) {
                const l = &live.items[i];
                const why = l.emit() catch |err| {
                    endAt(st, live, i, .failed, @errorName(err));
                    continue;
                };
                if (why) |w| {
                    endAt(st, live, i, w, "");
                    continue;
                }
                i += 1;
            }
            const n = live.items.len;
            if (n == 0) return;

            const first = &live.items[0];
            const drafts = if (first.onHost()) 0 else policy.draftsThisTick(n, st.ctx.mtpDrafts(), first.r.max_tokens - first.r.emitted);
            if (drafts > 0) {
                var buf: [policy.max_drafts + 1]i32 = undefined;
                const got = first.seq.specStep(first.next, drafts, sampling(first.r), &first.rng, &buf) catch |err| return endAt(st, live, 0, .failed, @errorName(err));
                first.r.stats.spec_mode = abi.span(u8, "mtp");
                first.r.stats.rounds += 1;
                first.r.stats.drafted += drafts;
                first.r.stats.accepted += got.len - 1;
                var ids: [policy.max_drafts]u32 = undefined;
                for (got[0 .. got.len - 1], ids[0 .. got.len - 1]) |g, *id| id.* = @intCast(g);
                if (got.len > 1) if (first.r.commit(ids[0 .. got.len - 1], &.{})) |why| return endAt(st, live, 0, why, "");
                first.next = got[got.len - 1];
                return;
            }

            var seqs: [load.max_seqs]*B.LlamaSeq = undefined;
            var toks: [load.max_seqs]i32 = undefined;
            for (live.items, seqs[0..n], toks[0..n]) |l, *q, *t| {
                q.* = l.seq;
                t.* = l.next;
            }
            st.ctx.step(seqs[0..n], toks[0..n]) catch |err| {
                while (live.items.len > 0) endAt(st, live, 0, .failed, @errorName(err));
                return;
            };
            if (n >= 2 and !st.batched_logged) {
                st.batched_logged = true;
                log.info("[batched] llama.cpp decode engaged (seqs={d})\n", .{n});
            }
            i = 0;
            while (i < live.items.len) {
                const l = &live.items[i];
                l.next = l.draw() catch |err| {
                    endAt(st, live, i, .failed, @errorName(err));
                    continue;
                };
                i += 1;
            }
        }
    };
}

// ── tests ──

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
    pub const Sampling = struct { temperature: f32, top_k: i32 = 0, top_p: f32 = 1, min_p: f32 = 0 };
    pub const LlamaEngine = struct {
        pub fn open(a: std.mem.Allocator, _: []const u8, _: bridge.OpenOptions) !*LlamaEngine {
            return a.create(LlamaEngine);
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
    done: std.atomic.Value(bool) = .init(false),
    reason: Reason = .stop,

    fn sink(c: *Collect) abi.Sink {
        return .{ .ctx = c, .event = onEvent };
    }
    fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.toks.appendSlice(testing.allocator, abi.items(u32, ev.tokens)) catch {};
        if (ev.kind == .prefilled) c.cached = ev.cached;
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
        const engine = try bridge.LlamaEngine.open(testing.allocator, path, .{});
        defer engine.close();
        const prompt = try engine.tokenizeText(testing.allocator, "The capital of France is", true);
        defer testing.allocator.free(prompt);
        for (prompt) |t| try ids.append(testing.allocator, @intCast(t));
        var ctx = try engine.createContext(.{ .ctx_size = 4096 });
        defer ctx.free();
        const seq = &ctx.seqs[0];
        _ = try seq.syncWithFallback(prompt);
        var tok = seq.argmax();
        while (want.items.len < n and !engine.isEog(tok)) {
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
