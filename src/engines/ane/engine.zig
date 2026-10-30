//! The Neural Engine LM engine behind the engine interface (`abi/engine.zig`): an MLX pack's programs loaded once
//! with `slots` engines on them, each decoding one request at a time on a thread of its own; later requests wait in
//! submit order. Never calls MLX.
//!
//! `Open.model_dir` is the pack; `settings_json` is `Settings` (what the host resolves for a load: slots, the
//! drafter, the reserved ids never drawn, the prefix cache). Penalties, logit bias and logprobs run on the host's
//! f16 logits rows (one token per call: a DFlash cycle verifies against the raw rows); grammar, the think budget and
//! the tool gate are refused by name.

const std = @import("std");
const env = @import("../../env.zig");
const build_options = @import("build_options");
const abi = @import("../../abi/engine.zig");
const model_mod = @import("../../sdk/config/model.zig");
const book_mod = @import("../own_thread/book.zig");
const logits = @import("../own_thread/logits.zig");
const lm = @import("lm.zig");

const Book = book_mod.Book;
const Req = book_mod.Req;
const Reason = book_mod.Reason;
const io = book_mod.io;

/// The table the host loads for the Neural Engine.
pub const table = Adapter(lm, std.heap.smp_allocator).table;

/// What shapes a load, resolved by the host (`engines/mlx/runtime/engine_load.zig`).
pub const Settings = struct {
    slots: u32 = 1,
    /// The DFlash2 drafter's dir; null decodes one token per call.
    drafter: ?[]const u8 = null,
    /// `Options.suppress_ids` / `defined_vocab`: the GPU's rule for ids never drawn.
    suppress_ids: []const u32 = &.{},
    defined_vocab: u32 = 0,
    /// The hot prefix cache; null keeps none.
    cache: ?lm.CacheLimits = null,
};

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
            .tokenize = null,
            .token_bytes = null,
            .chat_template = null,
            .render = null,
        };

        const State = struct {
            book: Book,
            pool: B.Pool,
            caps: abi.Caps,
            eos: []u32,
            threads: []std.Thread = &.{},
            startup_buf: [96]u8 = undefined,
        };

        fn state(e: *anyopaque) *State {
            return @ptrCast(@alignCast(e));
        }

        /// The pack's config as the host parses it (pooling and generation sidecars included).
        fn parse(model_dir: []const u8) !model_mod.ModelConfig {
            return model_mod.parseConfigLegacy(io, gpa, model_dir);
        }

        /// Dense qwen3_5 and Qwen3 packs (`lm.archOf`).
        pub fn probe(model_dir: abi.Str, _: abi.Str, out: *abi.Verdict) callconv(.c) bool {
            var cfg = parse(abi.items(u8, model_dir)) catch {
                out.* = abi.Verdict.refuse(.unsupported, "no readable config.json in the pack");
                return false;
            };
            defer cfg.deinit(gpa);
            _ = B.archOf(&cfg) catch {
                out.* = abi.Verdict.refuse(.unsupported, "does not run on the Neural Engine: dense qwen3_5 and Qwen3 packs only");
                return false;
            };
            return true;
        }

        fn parseSettings(json: []const u8) !std.json.Parsed(Settings) {
            return std.json.parseFromSlice(Settings, gpa, if (json.len == 0) "{}" else json, .{ .ignore_unknown_fields = true });
        }

        /// The int8 weights and each slot's KV at its top context (`lm.loadBytes`, the host's preflight).
        fn loadBytes(o: *const abi.Open, bytes: *u64) callconv(.c) abi.Verdict {
            const settings = parseSettings(abi.items(u8, o.settings_json)) catch return abi.Verdict.refuse(.unsupported, "settings_json does not parse");
            defer settings.deinit();
            var cfg = parse(abi.items(u8, o.model_dir)) catch return abi.Verdict.refuse(.unsupported, "no readable config.json in the pack");
            defer cfg.deinit(gpa);
            const arch = B.archOf(&cfg) catch return abi.Verdict.refuse(.unsupported, "does not run on the Neural Engine: dense qwen3_5 and Qwen3 packs only");
            bytes.* = B.loadBytes(arch, settings.value.slots);
            return .{};
        }

        fn open(o: *const abi.Open, info: *abi.Info, out: *abi.Verdict) callconv(.c) ?*anyopaque {
            return openOrError(o, info) catch |err| {
                out.* = .{ .code = if (err == error.InvalidSettings or err == error.UnsupportedArch) .unsupported else .failed, .why = abi.span(u8, @errorName(err)) };
                return null;
            };
        }

        fn openOrError(o: *const abi.Open, info: *abi.Info) !*State {
            const dir = abi.items(u8, o.model_dir);
            const settings = parseSettings(abi.items(u8, o.settings_json)) catch return error.InvalidSettings;
            defer settings.deinit();
            const s = settings.value;
            var cfg = try parse(dir);
            defer cfg.deinit(gpa);
            const arch = try B.archOf(&cfg);
            const pool = try B.Pool.open(gpa, io, dir, arch, .{
                .slots = s.slots,
                .drafter = s.drafter,
                .suppress_ids = s.suppress_ids,
                .defined_vocab = s.defined_vocab,
                .progress = o.host,
            }, s.cache);
            errdefer pool.close(gpa);
            for (pool.engines) |eng| eng.gauges = o.host;
            const eos = try gpa.dupe(u32, cfg.eosTokenSlice());
            errdefer gpa.free(eos);
            const st = try gpa.create(State);
            errdefer gpa.destroy(st);
            const engines: u32 = @intCast(pool.engines.len);
            st.* = .{
                .book = .{ .gpa = gpa, .max_waiting = @max(o.max_concurrent, 1) },
                .pool = pool,
                .caps = .{
                    .penalties = true,
                    .logit_bias = true,
                    .logprobs = true,
                    .drafts = pool.engines[0].drafter != null,
                    .prefix_cache = s.cache != null,
                },
                .eos = eos,
            };
            st.threads = try gpa.alloc(std.Thread, engines);
            for (st.threads, 0..) |*t, i| t.* = std.Thread.spawn(.{}, work, .{ st, i }) catch |err| {
                st.book.close();
                for (st.threads[0..i]) |u| u.join();
                gpa.free(st.threads);
                st.book.deinit();
                return err;
            };
            // The engine's top rung, or less where the pack was trained to less, or the host asks less.
            const top = pool.engines[0].model.maxPositions();
            var window = if (cfg.max_position_embeddings > 0) @min(cfg.max_position_embeddings, top) else top;
            if (o.context > 0) window = @min(window, o.context);
            const startup = std.fmt.bufPrint(&st.startup_buf, "ane: {d} slots, {d}-token context, drafter {s}", .{ engines, window, if (st.caps.drafts) "on" else "off" }) catch "";
            info.* = .{
                .name = abi.span(u8, "ane"),
                .version = abi.span(u8, build_options.version),
                .caps = st.caps,
                .threading = .own_thread,
                .context_window = window,
                .max_concurrent = engines,
                .embedding_dims = 0,
                .eos = abi.span(u32, st.eos),
                .startup = abi.span(u8, startup),
            };
            return st;
        }

        fn close(e: *anyopaque) callconv(.c) void {
            const st = state(e);
            st.book.close();
            for (st.threads) |t| t.join();
            gpa.free(st.threads);
            st.book.deinit();
            gpa.free(st.eos);
            st.pool.close(gpa);
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

        /// The registry's bill today: the compiled weights (one copy for every slot) and the pool's state.
        fn memory(e: *anyopaque, out: *abi.Memory) callconv(.c) void {
            const st = state(e);
            const active = st.pool.engines[0].model.weight_bytes + st.pool.stateBytes();
            out.* = .{ .active = active, .cache = 0, .peak = active };
        }

        fn work(st: *State, i: usize) void {
            const engine = st.pool.engines[i];
            while (st.book.next(true)) |r| {
                const reason = serve(engine, r) catch |err| {
                    st.book.done(r, if (err == error.Cancelled) .cancelled else .failed, @errorName(err));
                    continue;
                };
                st.book.done(r, reason, "");
                // After the last event: the next turn of this conversation starts from here.
                engine.checkpoint();
            }
        }

        /// A token drawn from `row` (shaped first when the request asks) and, when asked, its logprob off the raw row.
        const Pick = struct {
            r: *Req,
            s: B.Sampling,
            rng: std.Random,
            shaped: bool,
            work_row: []f16,
            top: []abi.Bias,
            history: logits.History = .{},
            lp: [1]abi.Logprob = undefined,

            fn draw(p: *Pick, engine: *B.Engine, row: []const f16) u32 {
                const t = if (p.shaped) blk: {
                    @memcpy(p.work_row, row);
                    logits.shape(f16, p.work_row, p.r.sampling, &p.history);
                    break :blk engine.draw(p.work_row, p.s, p.rng);
                } else engine.draw(row, p.s, p.rng);
                if (p.r.logprobs > 0) p.lp[0] = logits.logprob(f16, row, t, p.top);
                return t;
            }

            /// Emits `t`; null: decode on.
            fn emit(p: *Pick, t: u32) !?Reason {
                const why = p.r.commit(&.{t}, &p.lp);
                if (p.shaped) try p.history.add(gpa, t);
                return why;
            }
        };

        /// Sync the engine to the prompt, then sample, emit and feed one
        /// token at a time, or run DFlash cycles (anchor + accepted drafts + the next token) while they fit.
        fn serve(engine: *B.Engine, r: *Req) !Reason {
            if (r.isCancelled()) return .cancelled;
            const synced = synced: {
                defer engine.ungauge();
                break :synced try engine.sync(r.prompt, &r.cancelled);
            };
            if (r.prefilled(@intCast(synced.cached))) |why| return why;
            var prng = std.Random.DefaultPrng.init(r.seed());
            const shaped = logits.shapes(r.sampling);
            var p: Pick = .{
                .r = r,
                .s = .{ .temperature = r.sampling.temperature, .top_k = r.sampling.top_k, .top_p = r.sampling.top_p, .min_p = r.sampling.min_p },
                .rng = prng.random(),
                .shaped = shaped,
                .work_row = try gpa.alloc(f16, if (shaped) synced.logits.len else 0),
                .top = try gpa.alloc(abi.Bias, r.logprobs),
            };
            defer {
                gpa.free(p.work_row);
                gpa.free(p.top);
                p.history.deinit(gpa);
            }
            var tok = p.draw(engine, synced.logits);
            var emitted = false;
            if (engine.drafter != null and !shaped and r.logprobs == 0) {
                r.stats.spec_mode = abi.span(u8, "dflash");
                if (try p.emit(tok)) |why| return why;
                emitted = true;
                while (true) {
                    const c = engine.cycle(tok, p.s, p.rng, r.eos) catch |err| switch (err) {
                        error.ContextFull => break, // the last positions take one token per call
                        else => return err,
                    };
                    r.stats.rounds += 1;
                    r.stats.drafted += B.P - 1;
                    r.stats.accepted += c.accepted;
                    if (r.commit(c.tokens[0..c.n], &.{})) |why| return why;
                    tok = c.tokens[c.n - 1];
                }
            }
            while (true) {
                if (!emitted) if (try p.emit(tok)) |why| return why;
                emitted = false;
                const row = engine.step(tok) catch |err| switch (err) {
                    error.ContextFull => return .length,
                    else => return err,
                };
                tok = p.draw(engine, row);
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
}

/// `lm` with no Neural Engine: its logits rows favour id 50 (2), then `100 + n % 800` after n tokens fed (1); a
/// cycle accepts every draft.
const FakeLm = struct {
    pub const P = 4;
    pub const vocab = 1024;
    pub const Arch = struct {};
    pub const Sampling = lm.Sampling;
    pub const Synced = struct { logits: []const f16, cached: usize };
    const Error = error{ ContextFull, Cancelled, OutOfMemory };
    pub const Cycle = struct { tokens: [P]u32, n: usize, accepted: usize };
    pub fn archOf(cfg: anytype) error{UnsupportedArch}!Arch {
        if (cfg.num_experts > 0) return error.UnsupportedArch; // as lm.archOf: dense packs only
        return .{};
    }
    pub fn loadBytes(_: Arch, slots: u32) u64 {
        return @as(u64, slots) << 20;
    }
    pub const Model = struct {
        weight_bytes: u64 = 1 << 20,
        pub fn maxPositions(_: *const Model) u32 {
            return 4096;
        }
    };
    pub const Engine = struct {
        model: *Model,
        drafter: ?u8,
        gauges: ?*const abi.Host = null,
        fed: std.ArrayList(u32) = .empty,
        row: [vocab]f16 = @splat(0),

        fn refresh(e: *Engine) []const f16 {
            e.row = @splat(0);
            e.row[50] = 2;
            e.row[100 + e.fed.items.len % 800] = 1;
            return &e.row;
        }
        pub fn sync(e: *Engine, prompt: []const u32, stop: ?*const std.atomic.Value(bool)) Error!Synced {
            if (stop) |s| if (s.load(.acquire)) return error.Cancelled;
            var n: usize = 0;
            while (n < @min(prompt.len - 1, e.fed.items.len) and prompt[n] == e.fed.items[n]) n += 1;
            e.fed.shrinkRetainingCapacity(n);
            try e.fed.appendSlice(testing.allocator, prompt[n..]);
            return .{ .logits = e.refresh(), .cached = n };
        }
        pub fn step(e: *Engine, t: u32) Error![]const f16 {
            try e.fed.append(testing.allocator, t);
            return e.refresh();
        }
        pub fn draw(_: *Engine, row: []const f16, _: Sampling, _: std.Random) u32 {
            return lm.argmax(row);
        }
        pub fn cycle(e: *Engine, anchor: u32, _: Sampling, _: std.Random, _: []const u32) Error!Cycle {
            try e.fed.append(testing.allocator, anchor);
            var c: Cycle = .{ .tokens = undefined, .n = P, .accepted = P - 1 };
            for (c.tokens[0 .. P - 1]) |*t| {
                t.* = lm.argmax(e.refresh());
                try e.fed.append(testing.allocator, t.*);
            }
            c.tokens[P - 1] = lm.argmax(e.refresh());
            return c;
        }
        pub fn checkpoint(_: *Engine) void {}
        pub fn ungauge(_: *Engine) void {}
    };
    pub const Pool = struct {
        model: *Model,
        engines: []*Engine,
        pub fn open(a: std.mem.Allocator, _: std.Io, _: []const u8, _: Arch, opts: lm.Options, _: ?lm.CacheLimits) !Pool {
            const m = try a.create(Model);
            m.* = .{};
            const engines = try a.alloc(*Engine, opts.slots);
            for (engines) |*e| {
                e.* = try a.create(Engine);
                e.*.* = .{ .model = m, .drafter = if (opts.drafter != null) 1 else null };
            }
            return .{ .model = m, .engines = engines };
        }
        pub fn close(p: Pool, a: std.mem.Allocator) void {
            for (p.engines) |e| {
                e.fed.deinit(a);
                a.destroy(e);
            }
            a.free(p.engines);
            a.destroy(p.model);
        }
        pub fn stateBytes(p: Pool) u64 {
            return p.engines.len << 10;
        }
    };
};

const Fake = Adapter(FakeLm, testing.allocator);

/// A pack dir with a Qwen3 config.json in `tmp`; returns its absolute path in `buf`.
fn qwen3Pack(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3\",\"eos_token_id\":[2,3],\"max_position_embeddings\":2048}" });
    return buf[0..try tmp.dir.realPath(io, buf)];
}

test "ane adapter: the contract holds over a model-free lm, several slots, and DFlash cycles" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try qwen3Pack(&tmp, &buf);
    inline for (.{ "", "{\"slots\":3}", "{\"drafter\":\"/d\"}" }) |settings| {
        try contract.run(&Fake.table, .{ .model_dir = dir, .model_type = "qwen3", .config_json = "", .prompt = &.{ 5, 6, 7 }, .settings_json = settings, .gate_ms = 500 });
    }
}

test "ane adapter: probe and the load bill read the pack's config" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try qwen3Pack(&tmp, &buf);
    var v: abi.Verdict = .{};
    try testing.expect(Fake.table.probe(abi.span(u8, dir), .{}, &v));
    const host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, dir), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 1, .memory_budget = 0, .settings_json = abi.span(u8, "{\"slots\":2}"), .options_json = .{}, .host = &host };
    var bytes: u64 = 0;
    try testing.expectEqual(abi.Verdict.Code.ok, Fake.table.load_bytes(&o, &bytes).code);
    try testing.expectEqual(@as(u64, 2 << 20), bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3\",\"num_experts\":8}" });
    try testing.expect(!Fake.table.probe(abi.span(u8, dir), .{}, &v));
    try testing.expect(std.mem.indexOf(u8, abi.items(u8, v.why), "Neural Engine") != null);
}

const Collect = struct {
    toks: std.ArrayList(u32) = .empty,
    top: std.ArrayList(u32) = .empty,
    done: std.atomic.Value(bool) = .init(false),
    reason: Reason = .stop,

    fn sink(c: *Collect) abi.Sink {
        return .{ .ctx = c, .event = onEvent };
    }
    fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.toks.appendSlice(testing.allocator, abi.items(u32, ev.tokens)) catch {};
        for (abi.items(abi.Logprob, ev.logprobs)) |lp| c.top.append(testing.allocator, abi.items(abi.Bias, lp.top)[0].token) catch {};
        if (ev.kind == .finished) {
            c.reason = ev.reason;
            c.done.store(true, .release);
        }
        return .go;
    }
    fn deinit(c: *Collect) void {
        c.toks.deinit(testing.allocator);
        c.top.deinit(testing.allocator);
    }
    fn wait(c: *Collect) void {
        while (!c.done.load(.acquire)) io.sleep(.fromMicroseconds(100), .awake) catch {};
    }
};

test "ane adapter: bias, penalties and logprobs on the host rows; EOS from the config; the context window" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try qwen3Pack(&tmp, &buf);
    const host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, dir), .model_type = .{}, .config_json = .{}, .context = 1024, .max_concurrent = 4, .memory_budget = 0, .settings_json = abi.span(u8, "{\"drafter\":\"/d\"}"), .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v).?;
    defer Fake.table.close(e);
    try testing.expectEqualSlices(u32, &.{ 2, 3 }, abi.items(u32, info.eos));
    try testing.expectEqual(@as(u32, 1024), info.context_window);
    try testing.expect(info.caps.drafts and info.caps.penalties and !info.caps.prefix_cache);

    const prompt = [_]u32{ 5, 6, 7 };
    const bias = [_]abi.Bias{.{ .token = 50, .bias = -5 }};
    var biased: Collect = .{};
    defer biased.deinit();
    _ = Fake.table.submit(e, 1, &.{ .prompt = abi.span(u32, &prompt), .max_tokens = 2, .sampling = .{ .temperature = 0, .logit_bias = abi.span(abi.Bias, &bias) }, .logprobs = 1 }, biased.sink());
    biased.wait();
    try testing.expectEqualSlices(u32, &.{ 103, 104 }, biased.toks.items);
    try testing.expectEqualSlices(u32, &.{ 50, 50 }, biased.top.items); // off the raw rows

    var pen: Collect = .{};
    defer pen.deinit();
    _ = Fake.table.submit(e, 2, &.{ .prompt = abi.span(u32, &prompt), .max_tokens = 3, .sampling = .{ .temperature = 0, .presence_penalty = 2 } }, pen.sink());
    pen.wait();
    try testing.expectEqualSlices(u32, &.{ 50, 104, 105 }, pen.toks.items);
}

// ── against a real pack on the Neural Engine (`MLX_SERVE_TEST_ANE_MODEL`; skipped without it)

fn realPack() ?[]const u8 {
    const raw = env.get(.MLX_SERVE_TEST_ANE_MODEL) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

/// Plausible ids for a Qwen vocabulary (no tokenizer here).
const real_prompt = [_]u32{ 785, 6722, 315, 9625, 374, 12366, 13, 576, 6722, 315, 9856, 374 };

test "ane adapter: the contract holds on a real pack" {
    const dir = realPack() orelse return error.SkipZigTest;
    const R = Adapter(lm, testing.allocator);
    try contract.run(&R.table, .{ .model_dir = dir, .model_type = "qwen3", .config_json = "", .prompt = &real_prompt, .settings_json = "{\"slots\":2}", .timeout_ms = 600_000, .gate_ms = 10_000 });
}

test "ane adapter: greedy tokens equal the plain engine sequence on a real pack" {
    const dir = realPack() orelse return error.SkipZigTest;
    const R = Adapter(lm, testing.allocator);
    const n = 16;
    // The plain sequence (no drafter): sync, then draw, emit and step per token.
    var want: [n]u32 = undefined;
    {
        var cfg = try model_mod.parseConfig(io, testing.allocator, dir);
        defer cfg.deinit(testing.allocator);
        const pool = try lm.Pool.open(testing.allocator, io, dir, try lm.archOf(&cfg), .{}, null);
        defer pool.close(testing.allocator);
        const engine = pool.engines[0];
        var prng = std.Random.DefaultPrng.init(0);
        const synced = try engine.sync(&real_prompt, null);
        var tok = engine.draw(synced.logits, .{}, prng.random());
        for (&want) |*w| {
            w.* = tok;
            tok = engine.draw(try engine.step(tok), .{}, prng.random());
        }
    }
    const host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, dir), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 1, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = R.table.open(&o, &info, &v) orelse return error.Open;
    defer R.table.close(e);
    var got: Collect = .{};
    defer got.deinit();
    if (R.table.submit(e, 1, &.{ .prompt = abi.span(u32, &real_prompt), .max_tokens = n, .sampling = .{ .temperature = 0 } }, got.sink()).code != .ok) return error.Refused;
    got.wait();
    try testing.expectEqualSlices(u32, &want, got.toks.items);
}
