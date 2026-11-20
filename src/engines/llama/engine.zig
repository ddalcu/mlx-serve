//! The llama.cpp engine behind the engine interface (`abi/engine.zig`): one context of `seqs` sequences per opened
//! model, decoded together on the adapter's own thread; requests past the free sequences wait in submit order.
//!
//! `Open.model_dir` names the `.gguf` file (a split GGUF's first shard); `settings_json` is `load.Settings`;
//! `Open.context` is each sequence's context (0: 8192, the scheduler's default). The host renders the GGUF's chat
//! template (`chat_template`) and tokenizes through `tokenize`, whose `specials` adds the model's BOS. Penalties,
//! logit bias and logprobs run on the sequence's own logits row before the shim's sampler draws from it (such
//! requests take no MTP round); grammar, the think budget and the tool gate are refused by name.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/engine.zig");
const engine_pick = @import("../../sdk/quant/gguf/engine_pick.zig");
const log = @import("../../sdk/util/log.zig");
const book_mod = @import("../own_thread/book.zig");
const TokenTable = @import("../own_thread/token_table.zig").TokenTable;
const logits = @import("../own_thread/logits.zig");
const load = @import("load.zig");
const policy = @import("policy.zig");
const gguf_files = @import("../../sdk/quant/gguf/files.zig");

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
            image_marker: []i32,
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
            const path = abi.items(u8, o.model_dir);
            bytes.* = fileBytes(path);
            const settings = parseSettings(abi.items(u8, o.settings_json)) catch return .{ .code = .unsupported, .why = abi.span(u8, "InvalidSettings") };
            const vision = if (settings.vision) gguf_files.findVisionSidecar(io, gpa, path) else null;
            defer if (vision) |p| gpa.free(p);
            if (vision) |p| bytes.* +|= fileBytes(p);
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
            const image_marker = if (opened.engine.hasVision())
                try opened.engine.tokenizeText(gpa, opened.engine.imageMarker(), false)
            else
                try gpa.alloc(i32, 0);
            errdefer gpa.free(image_marker);
            if (opened.engine.hasVision() and image_marker.len == 0) return error.InvalidImageMarker;
            var file_bytes: u64 = 0;
            _ = loadBytes(o, &file_bytes);
            st.* = .{
                .book = .{ .gpa = gpa, .max_waiting = @max(o.max_concurrent, 1) },
                .engine = opened.engine,
                .ctx = opened.ctx,
                .caps = .{
                    .images = opened.engine.hasVision(),
                    .penalties = true,
                    .logit_bias = true,
                    .logprobs = true,
                    .owns_tokenizer = true,
                    .batches = seqs > 1,
                    .drafts = opened.ctx.mtpDrafts() > 0,
                    .prefix_cache = true,
                },
                .image_marker = image_marker,
                .eos = try eos.toOwnedSlice(gpa),
                .tokens = tokens,
                .file_bytes = file_bytes,
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
                .image_placeholder = if (st.caps.images) .of(opened.engine.imageMarker()) else .{},
            };
            return st;
        }

        fn close(e: *anyopaque) callconv(.c) void {
            const st = state(e);
            st.book.close();
            st.thread.join();
            st.book.deinit();
            st.tokens.deinit(gpa);
            gpa.free(st.image_marker);
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

        fn tokenizePrompt(st: *State, text: []const u8, specials: bool) ![]i32 {
            if (!st.caps.images) return st.engine.tokenizeText(gpa, text, specials);
            var ids: std.ArrayList(i32) = .empty;
            errdefer ids.deinit(gpa);
            var rest = text;
            // BOS rides the first part that tokenizes to something: an atom at
            // the very front would otherwise swallow the only add_special call.
            var bos_pending = specials;
            while (true) {
                const at = std.mem.indexOf(u8, rest, st.engine.imageMarker()) orelse rest.len;
                const part = try st.engine.tokenizeText(gpa, rest[0..at], bos_pending);
                defer gpa.free(part);
                if (part.len > 0) bos_pending = false;
                try ids.appendSlice(gpa, part);
                if (at == rest.len) break;
                // Keep atoms stable across adjacent text: BPE must not merge across an image boundary.
                try ids.appendSlice(gpa, st.image_marker);
                rest = rest[at + st.engine.imageMarker().len ..];
            }
            return ids.toOwnedSlice(gpa);
        }

        fn tokenize(e: *anyopaque, text: abi.Str, specials: bool, out: [*]u32, cap: usize) callconv(.c) usize {
            const st = state(e);
            const ids = tokenizePrompt(st, abi.items(u8, text), specials) catch return 0;
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
            const cached: i32 = if (r.media.len > 0) blk: {
                const images = gpa.alloc(B.Image, r.media.len) catch return end(st, &l, .failed, "OutOfMemory");
                defer gpa.free(images);
                for (r.media, images) |m, *image| image.* = .{ .at = m.at, .width = m.width, .height = m.height, .pixels = m.data.slice() };
                r.prompt_tokens = seq.syncImages(prompt, images, st.image_marker, &r.cancelled, &r.max_tokens) catch |err| {
                    seq.reset();
                    if (err == B.Error.MediaRefused) return end(st, &l, .failed, "the image prompt does not fit this model: reduce the prompt or image size (a non-causal projector may need a larger --llama-ubatch; the server log names the exact refusal)");
                    return end(st, &l, if (r.isCancelled()) .cancelled else .failed, @errorName(err));
                };
                break :blk 0;
            } else seq.syncWithFallback(prompt) catch |err| return end(st, &l, .failed, @errorName(err));
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
            const drafts = if (first.onHost() or first.r.media.len > 0 or first.r.no_drafts) 0 else policy.draftsThisTick(n, st.ctx.mtpDrafts(), first.r.max_tokens - first.r.emitted);
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

test {
    _ = @import("engine_test.zig");
}
