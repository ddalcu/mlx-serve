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
const log = @import("../../sdk/util/log.zig");
const vision_mod = @import("vision.zig");
const book_mod = @import("../own_thread/book.zig");
const logits = @import("../own_thread/logits.zig");
const TokenTable = @import("../own_thread/token_table.zig").TokenTable;
const load = @import("load.zig");
const round = @import("round.zig");
const reply = @import("reply.zig");
const think_tags = @import("../../host/chat/reasoning/think_tags.zig");
const prefix_cache = @import("prefix_cache.zig");

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
            .chat_template = chatTemplate,
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
            mapped_bytes: u64,
            template: []const u8,
            /// The last prompt as the client sent it, and where its reply began in the session (the length of the
            /// prompt synced for it): `keptReply` continues a conversation whose copies of past replies differ.
            client_prompt: []const i32 = &.{},
            reply_start: usize = 0,
            reply_opened_think: bool = false,
            /// The vision tokens, when the engine loaded a projector and the host renders the GGUF's template.
            vision: ?Vision = null,
            image_cache: vision_mod.Cache(B, gpa) = .{},
            disk: ?prefix_cache.Cache(B) = null,
            /// Only a completely synced text checkpoint may be persisted.
            disk_eligible: bool = false,
            thread: std.Thread = undefined,
            startup_buf: [96]u8 = undefined,
        };

        pub fn state(e: *anyopaque) *State {
            return @ptrCast(@alignCast(e));
        }

        fn fileBytes(path: []const u8) u64 {
            const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return 0;
            return st.size;
        }

        /// What the GGUF header says about the open: its trained window and whether
        /// ds4's own encoder is the chat format (DeepSeek-V4's DSML) rather than the template the GGUF ships.
        const Meta = struct { trained: u32 = 0, ds4_encodes: bool = true };

        fn ggufMeta(path: []const u8) Meta {
            var info = engine_pick.readFromFile(io, gpa, path) catch return .{};
            defer info.deinit(gpa);
            const arch = info.architecture orelse "";
            return .{
                .trained = info.context_length,
                .ds4_encodes = std.mem.eql(u8, arch, "deepseek4") or std.mem.eql(u8, arch, "deepseek41"),
            };
        }

        /// The GGUF's template for the host's Jinja: its native tool and reasoning format. Empty when ds4 encodes
        /// the chat itself or the GGUF ships none.
        fn hostTemplate(path: []const u8, meta: Meta) []const u8 {
            if (meta.ds4_encodes) return &.{};
            return (engine_pick.readChatTemplate(io, gpa, path) catch null) orelse &.{};
        }

        const Vision = vision_mod.Tokens;

        fn visionTokens(engine: *B.Ds4Engine, path: []const u8, template: []const u8) ?Vision {
            if (template.len == 0 or !engine.hasVision()) return null;
            return vision_mod.tokens(io, gpa, path);
        }

        /// The GGUF and ds4's own estimate of the session's KV and scratch.
        fn loadBytes(o: *const abi.Open, bytes: *u64) callconv(.c) abi.Verdict {
            const path = abi.items(u8, o.model_dir);
            const meta = ggufMeta(path);
            bytes.* = fileBytes(path) + B.contextBytes(askedContext(o, meta));
            return .{};
        }

        /// The session asked for: `o.context`, else the GGUF's trained window (`<arch>.context_length`), else ds4's default.
        fn askedContext(o: *const abi.Open, meta: Meta) u32 {
            return B.clampSessionCtx(if (o.context != 0) o.context else meta.trained);
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
            const meta = ggufMeta(path);
            const template = hostTemplate(path, meta);
            errdefer gpa.free(template);
            var ctx = askedContext(o, meta);
            const engine = try load.open(B, io, gpa, path, flags, ctx);
            errdefer engine.close();
            const session = engine.createSession(@intCast(ctx)) catch |err| retry: {
                // Only the trained window an unset context asked for falls back; an explicit one fails by name.
                const default_ctx = B.clampSessionCtx(0);
                if (o.context != 0 or ctx <= default_ctx) return err;
                log.warn("[ds4] the GGUF's trained {d}-token window was refused ({s}); opening ds4's default {d}\n", .{ ctx, @errorName(err), default_ctx });
                ctx = default_ctx;
                break :retry try engine.createSession(@intCast(ctx));
            };
            errdefer session.free();
            const vocab: usize = engine.vocabSize();
            var eos: std.ArrayList(u32) = .empty;
            errdefer eos.deinit(gpa);
            for (0..vocab) |t| if (engine.isStop(@intCast(t))) try eos.append(gpa, @intCast(t));
            const vision = visionTokens(engine, path, template);
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
                    .owns_template = template.len == 0,
                    .images = vision != null,
                    .drafts = engine.mtpDraftTokens() > 1,
                    .prefix_cache = true,
                },
                .ctx = ctx,
                .vocab = vocab,
                .eos = try eos.toOwnedSlice(gpa),
                .tokens = tokens,
                .mapped_bytes = fileBytes(path),
                .template = template,
                .vision = vision,
                .disk = openDisk(engine, path, ctx, template, flags),
            };
            errdefer gpa.free(st.eos);
            errdefer if (st.disk) |*disk| disk.deinit();
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
                .image_placeholder = if (vision) |v| .of(v.placeholder) else .{},
            };
            return st;
        }

        fn openDisk(engine: *B.Ds4Engine, path: []const u8, ctx: u32, template: []const u8, flags: load.Flags) ?prefix_cache.Cache(B) {
            // Unit adapters have no native graph handle. The native storage's C tests cover its boundary.
            if (comptime !@hasField(B.Ds4Engine, "handle")) return null;
            if (flags.prefix_cache_disk_bytes == 0) return null;
            const home = env.get(.HOME) orelse {
                log.warn("[ds4-cache] HOME is unset; SSD persistence unavailable\n", .{});
                return null;
            };
            const settings = std.json.Stringify.valueAlloc(gpa, .{
                .context = ctx,
                .mtp = flags.mtp,
                .dspark = flags.dspark,
                .ssd_streaming = flags.ssd_streaming,
                .prefill_chunk = flags.prefill_chunk,
            }, .{}) catch return null;
            defer gpa.free(settings);
            var files: [2][]const u8 = undefined;
            var count: usize = 0;
            if (engine.mtp_path_owned) |p| {
                files[count] = p;
                count += 1;
            }
            if (engine.vision_path_owned) |p| {
                files[count] = p;
                count += 1;
            }
            return prefix_cache.Cache(B).open(io, gpa, engine, path, .{
                .home = std.mem.span(home),
                .budget_bytes = flags.prefix_cache_disk_bytes,
                .identity = .{ .build = build_options.ds4_commit, .template = template, .settings = settings, .sidecars = files[0..count] },
            }) catch |err| {
                log.warn("[ds4-cache] SSD persistence unavailable: {s}\n", .{@errorName(err)});
                return null;
            };
        }

        fn persist(st: *State, reason: prefix_cache.Reason) void {
            if (comptime !@hasField(B.Ds4Session, "handle")) return;
            if (!st.disk_eligible) return;
            if (st.disk) |*disk| {
                const live = st.session.liveTokens(gpa) catch |err| {
                    log.warn("[ds4-cache] checkpoint tokens unavailable: {s}\n", .{@errorName(err)});
                    return;
                };
                defer gpa.free(live);
                const saved = disk.store(st.session, live, reason) catch |err| {
                    log.warn("[ds4-cache] store skipped: {s}\n", .{@errorName(err)});
                    return;
                };
                if (saved) log.info("[ds4-cache] saved {d} tokens ({s})\n", .{ live.len, @tagName(reason) });
            }
        }

        fn restore(st: *State, prompt: []const i32) usize {
            if (comptime !@hasField(B.Ds4Session, "handle")) return 0;
            if (st.disk) |*disk| {
                const hit = disk.restore(st.session, prompt) catch |err| {
                    log.warn("[ds4-cache] restore skipped: {s}\n", .{@errorName(err)});
                    return 0;
                };
                if (hit) |h| {
                    defer h.deinit(gpa);
                    log.info("[ds4-cache] restored {d}/{d} tokens\n", .{ h.cached_tokens, prompt.len });
                    return h.cached_tokens;
                }
            }
            return 0;
        }

        fn close(e: *anyopaque) callconv(.c) void {
            const st = state(e);
            st.book.close();
            st.thread.join();
            persist(st, .close);
            if (st.disk) |*disk| disk.deinit();
            st.book.deinit();
            st.tokens.deinit(gpa);
            gpa.free(st.eos);
            gpa.free(st.template);
            gpa.free(st.client_prompt);
            st.image_cache.deinit();
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
            const active = st.mapped_bytes + B.contextBytes(st.ctx);
            out.* = .{ .active = active, .cache = 0, .peak = active };
        }

        fn tokenize(e: *anyopaque, text: abi.Str, _: bool, out: [*]u32, cap: usize) callconv(.c) usize {
            const st = state(e);
            const ids = tokenizeText(st, abi.items(u8, text)) catch return 0;
            defer gpa.free(ids);
            for (ids[0..@min(ids.len, cap)], 0..) |t, i| out[i] = @intCast(t);
            return ids.len;
        }

        /// The host renders through the GGUF's template: its special tokens map to their ids, and each image placeholder
        /// to the three vision tokens with one pad (`expandImages` widens it to the tower's rows).
        fn tokenizeText(st: *State, text: []const u8) ![]i32 {
            if (st.template.len == 0) return st.engine.tokenizeText(gpa, text);
            const v = st.vision orelse return st.engine.tokenizeRenderedChat(gpa, text);
            var out: std.ArrayList(i32) = .empty;
            errdefer out.deinit(gpa);
            var rest = text;
            while (true) {
                const at = std.mem.indexOf(u8, rest, v.placeholder) orelse rest.len;
                if (at > 0) {
                    const ids = try st.engine.tokenizeRenderedChat(gpa, rest[0..at]);
                    defer gpa.free(ids);
                    try out.appendSlice(gpa, ids);
                }
                if (at == rest.len) break;
                try out.appendSlice(gpa, &.{ v.start, v.pad, v.end });
                rest = rest[at + v.placeholder.len ..];
            }
            return out.toOwnedSlice(gpa);
        }

        /// The GGUF's template when the host renders (`Caps.owns_template` false); empty when ds4 encodes the chat.
        fn chatTemplate(e: *anyopaque) callconv(.c) abi.Str {
            return abi.span(u8, state(e).template);
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
                    st.disk_eligible = false;
                    st.image_cache.deinit();
                    st.image_cache = .{};
                    forgetPrompt(st);
                    st.book.done(r, .failed, @errorName(err));
                    continue;
                };
                if (reason == .cancelled) st.disk_eligible = false;
                st.book.done(r, reason, "");
            }
        }

        /// How far before the end of its last prompt a client's copy of the reply may begin to differ: the reasoning
        /// opener of the generation prompt, which a template renders differently in history.
        const reply_opener_tokens = 4;
        const Kept = struct { tokens: []i32, covered: usize };

        /// A client's copy of a reply this session generated can differ from what was sampled (reasoning left out, a
        /// tool call re-serialized), and ds4 cannot rewind a hybrid model's state into it, so a plain sync replays the
        /// whole conversation. When `prompt` extends the client's last prompt with its copy of the reply, keep the
        /// session (every sampled reply in it) and prefill only what follows the copy, from its end-of-turn token on.
        /// Null: sync `prompt` as sent.
        fn keptReply(st: *State, prompt: []const i32, max_tokens: u32) !?Kept {
            if (st.reply_start == 0) return null;
            const common = std.mem.indexOfDiff(i32, st.client_prompt, prompt) orelse return null;
            if (common + reply_opener_tokens < st.client_prompt.len or common >= prompt.len) return null;
            const live = try st.session.liveTokens(gpa);
            defer gpa.free(live);
            if (live.len <= st.reply_start or std.mem.startsWith(i32, prompt, live)) return null;
            const end = endOfTurn(st, prompt, common) orelse return null;
            // A speculative round may have evaluated the reply's end-of-turn token and what follows it; the copy's
            // own end then continues that tail rather than repeating it.
            const live_end = endOfTurn(st, live, st.reply_start) orelse live.len;
            const copy = try decoded(st, prompt[common..end]);
            defer gpa.free(copy);
            const sampled = try decoded(st, live[st.reply_start..live_end]);
            defer gpa.free(sampled);
            const head = try decoded(st, prompt[common -| 8..common]);
            defer gpa.free(head);
            if (!try reply.equivalent(gpa, .{ .text = sampled, .opened_by_template = st.reply_opened_think }, .{ .text = copy, .opened_by_template = think_tags.promptTailOpensThink(head) })) return null;
            const len = live_end + (prompt.len - end);
            if (len + max_tokens > st.ctx) return null;
            const tokens = try gpa.alloc(i32, len);
            @memcpy(tokens[0..live_end], live[0..live_end]);
            @memcpy(tokens[live_end..], prompt[end..]);
            return .{ .tokens = tokens, .covered = end };
        }

        /// The first end-of-turn (stop) token in `tokens` at or after `from`.
        fn endOfTurn(st: *State, tokens: []const i32, from: usize) ?usize {
            for (tokens[from..], from..) |t, i| if (std.mem.indexOfScalar(u32, st.eos, @intCast(t)) != null) return i;
            return null;
        }

        fn decoded(st: *State, tokens: []const i32) ![]u8 {
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(gpa);
            for (tokens) |t| try out.appendSlice(gpa, st.tokens.get(gpa, st.engine, @intCast(t)));
            return out.toOwnedSlice(gpa);
        }

        /// The tower runs on this engine's own thread, never the host's request thread.
        fn expandImages(st: *State, prompt: []const i32, media: []const abi.Media) !?[]i32 {
            const v = st.vision orelse return if (media.len == 0) null else error.ImagesUnsupported;
            return try st.image_cache.expand(st.engine, v, prompt, media);
        }

        fn syncSession(st: *State, tokens: []const i32, r: *Req) !void {
            if (st.vision) |v| return st.image_cache.sync(st.engine, st.session, v, tokens, r.media, &r.cancelled);
            return st.session.sync(tokens, &r.cancelled);
        }

        /// The client's view of what the session now holds, for the next request's `keptReply`.
        fn rememberPrompt(st: *State, prompt: []const i32, synced: usize) !void {
            const copy = try gpa.dupe(i32, prompt);
            gpa.free(st.client_prompt);
            st.client_prompt = copy;
            st.reply_start = synced;
            const tail = try decoded(st, prompt[prompt.len -| 8..]);
            defer gpa.free(tail);
            st.reply_opened_think = think_tags.promptTailOpensThink(tail);
        }

        fn forgetPrompt(st: *State) void {
            gpa.free(st.client_prompt);
            st.client_prompt = &.{};
            st.reply_start = 0;
            st.reply_opened_think = false;
        }

        /// Sync the session to the prompt (ds4 keeps the common prefix in its live KV), then decode round by round.
        fn serve(st: *State, r: *Req) !Reason {
            if (r.isCancelled()) return .cancelled;
            if (r.cache.cold) {
                persist(st, .replacement);
                st.session.invalidate();
                st.disk_eligible = false;
                st.image_cache.deinit();
                st.image_cache = .{};
                forgetPrompt(st);
            }
            defer if (!r.cache.keep) {
                st.session.invalidate();
                st.disk_eligible = false;
                st.image_cache.deinit();
                st.image_cache = .{};
                forgetPrompt(st);
            };
            const prompt = try gpa.alloc(i32, r.prompt.len);
            defer gpa.free(prompt);
            for (r.prompt, prompt) |t, *p| p.* = @intCast(t);
            const expanded = try expandImages(st, prompt, r.media);
            defer if (expanded) |x| gpa.free(x);
            const view = expanded orelse prompt;
            if (view.len >= st.ctx) return error.ContextExceeded;
            const image_prefix = try st.image_cache.continues(st.engine, r.media);
            const kept = if (image_prefix) try keptReply(st, view, r.max_tokens) else null;
            defer if (kept) |k| gpa.free(k.tokens);
            const synced = if (kept) |k| k.tokens else view;
            r.prompt_tokens = @intCast(synced.len);
            r.max_tokens = @min(r.max_tokens, st.ctx - @as(u32, @intCast(synced.len)));
            var cached: usize = if (kept) |k| k.covered else if (image_prefix) @intCast(@max(st.session.commonPrefix(view) catch 0, 0)) else 0;
            if (kept == null) {
                const live = try st.session.liveTokens(gpa);
                defer gpa.free(live);
                if (cached < live.len or !image_prefix) {
                    persist(st, .replacement);
                    if (r.media.len == 0 and !r.cache.cold) cached = restore(st, view);
                } else if (live.len == 0 and r.media.len == 0 and !r.cache.cold) {
                    cached = restore(st, view);
                }
            }
            st.disk_eligible = false;
            syncSession(st, synced, r) catch |err| {
                st.session.invalidate();
                st.image_cache.deinit();
                st.image_cache = .{};
                forgetPrompt(st);
                // ds4 rebuilds a half-done pass on the next sync; the complete prefix stays cached.
                if (err == error.SyncInterrupted) return .cancelled;
                return err;
            };
            st.disk_eligible = r.cache.keep and r.media.len == 0 and !r.isCancelled();
            if (cached == 0) persist(st, .cold);
            try rememberPrompt(st, view, synced.len);
            if (r.prefilled(@intCast(cached))) |why| return why;

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
            const spec = !r.no_drafts and !host and !(r.eos.len == 0 and d.temperature > 0) and round.shouldSpeculate(st.engine.mtpDraftTokens(), d.temperature, st.engine.embedded_mtp);
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
                    const got = try round.speculate(st.engine, st.session, t, d, r.max_tokens - r.emitted, r.eos.len == 0, &buf);
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
    pub const Embedding = struct { data: ?[*]f32 = null, token_count: u32 = 0, fingerprint: [32]u8 = @splat(0) };
    pub const Span = struct { token_start: u32 = 0, embedding: Embedding = .{} };
    var fake_rows: [1]f32 = .{0};
    /// The largest session the fake admits: a bigger one is refused, as ds4's memory guard refuses one that does not fit.
    pub const session_cap: u32 = 1 << 20;
    pub const ThinkMode = enum(c_int) { none, high };
    pub fn clampSessionCtx(requested: u32) u32 {
        return if (requested == 0) 4096 else requested;
    }
    pub fn contextBytes(ctx: u32) u64 {
        return @as(u64, ctx) * 1024;
    }
    pub fn ggufSpec(_: std.Io, _: std.mem.Allocator, _: []const u8) engine_pick.Spec {
        return .{ .embedded_mtp = true };
    }
    pub fn ggufIsGlmFamily(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
        return false;
    }
    pub const Ds4Engine = struct {
        pub const ChatTurn = struct { role: []const u8, content: []const u8 };
        embedded_mtp: bool,
        drafts: c_int,
        ctx: u32,
        vision: bool,
        pub fn open(a: std.mem.Allocator, _: []const u8, opts: anytype) !*Ds4Engine {
            const e = try a.create(Ds4Engine);
            e.* = .{ .embedded_mtp = opts.embedded_mtp, .drafts = if (opts.prefill_chunk == 1) 0 else 4, .ctx = opts.context_size, .vision = opts.vision_path != null };
            return e;
        }
        pub fn close(e: *Ds4Engine) void {
            testing.allocator.destroy(e);
        }
        pub fn mtpDraftTokens(e: *Ds4Engine) c_int {
            return e.drafts;
        }
        pub fn hasVision(e: *Ds4Engine) bool {
            return e.vision;
        }
        /// Every image is three rows.
        pub fn encodeImage(_: *Ds4Engine, encoded: []const u8) !Embedding {
            var fp: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(encoded, &fp, .{});
            return .{ .data = &fake_rows, .token_count = 3, .fingerprint = fp };
        }
        pub fn freeEmbedding(_: *Embedding) void {}
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
        pub fn tokenizeRenderedChat(_: *Ds4Engine, a: std.mem.Allocator, text: []const u8) ![]i32 {
            const ids = try a.alloc(i32, text.len);
            for (text, ids) |c, *id| id.* = @as(i32, c) + 1000;
            return ids;
        }
        pub fn detokenizeOne(_: *Ds4Engine, a: std.mem.Allocator, id: i32) ![]u8 {
            // A history-only framing token in the replay fixtures carries no visible bytes.
            if (id == 9) return a.dupe(u8, "");
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
        pub fn createSession(_: *Ds4Engine, ctx: i32) !*Ds4Session {
            if (ctx > session_cap) return error.SessionCreateFailed;
            const s = try testing.allocator.create(Ds4Session);
            s.* = .{};
            return s;
        }
    };
    pub const Ds4Session = struct {
        toks: std.ArrayList(i32) = .empty,
        row: [vocab]f32 = @splat(0),
        /// Set by a prompt holding 998: from this length on the fake favours its end-of-turn token (2).
        stop_at: usize = 0,
        /// Where the last image sync placed each image.
        span_starts: std.ArrayList(u32) = .empty,

        fn refresh(s: *Ds4Session) void {
            s.row = @splat(0);
            s.row[50] = 2;
            s.row[100 + s.toks.items.len % 800] = 1;
            if (s.stop_at != 0 and s.toks.items.len >= s.stop_at) s.row[2] = 3;
        }
        pub fn free(s: *Ds4Session) void {
            s.toks.deinit(testing.allocator);
            s.span_starts.deinit(testing.allocator);
            testing.allocator.destroy(s);
        }
        pub fn invalidate(s: *Ds4Session) void {
            s.toks.clearRetainingCapacity();
        }
        pub fn syncMultimodal(s: *Ds4Session, p: []const i32, spans: []const Span, cancel: ?*std.atomic.Value(bool)) !void {
            s.span_starts.clearRetainingCapacity();
            for (spans) |sp| try s.span_starts.append(testing.allocator, sp.token_start);
            try s.sync(p, cancel);
        }
        pub fn liveTokens(s: *Ds4Session, a: std.mem.Allocator) ![]i32 {
            return a.dupe(i32, s.toks.items);
        }
        pub fn commonPrefix(s: *Ds4Session, p: []const i32) !i32 {
            var n: usize = 0;
            while (n < @min(p.len, s.toks.items.len) and p[n] == s.toks.items[n]) n += 1;
            return @intCast(n);
        }
        pub fn sync(s: *Ds4Session, p: []const i32, cancel: ?*std.atomic.Value(bool)) !void {
            if (std.mem.indexOfScalar(i32, p, 999) != null) if (cancel) |c| c.store(true, .release);
            if (cancel) |c| if (c.load(.acquire)) return error.SyncInterrupted;
            s.toks.clearRetainingCapacity();
            try s.toks.appendSlice(testing.allocator, p);
            s.stop_at = if (std.mem.indexOfScalar(i32, p, 998) != null) p.len + 2 else 0;
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
        pub fn evalSpeculativeIgnoringEos(s: *Ds4Session, first: i32, max: i32, eos: i32, out: []i32) !i32 {
            return s.evalSpeculative(first, max, eos, out);
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
    return ds4GgufTrained(tmp, buf, null);
}

/// `ds4Gguf`, declaring `trained` as the checkpoint's `deepseek4.context_length` when set.
fn ds4GgufTrained(tmp: *std.testing.TmpDir, buf: []u8, trained: ?u64) ![]const u8 {
    const kvs = [_]engine_pick.KV{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
        .{ .key = "deepseek4.context_length", .value = .{ .u64_v = trained orelse 0 } },
    };
    const header = try engine_pick.buildHeader(testing.allocator, if (trained != null) &kvs else kvs[0..2]);
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

/// The session the fake adapter opens when asked for `asked` on a ds4 GGUF declaring `trained`; null when the open fails.
fn openedContext(trained: ?u64, asked: u32) !?u32 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4GgufTrained(&tmp, &buf, trained);
    var host: abi.Host = .{ .ctx = undefined, .log = struct {
        fn f(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}
    }.f, .wake = struct {
        fn f(_: *anyopaque) callconv(.c) void {}
    }.f };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = asked, .max_concurrent = 1, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v) orelse return null;
    defer Fake.table.close(e);
    return info.context_window;
}

test "ds4 adapter: an unset context opens the checkpoint's trained window, ds4's default when that session does not fit" {
    const default = FakeBridge.clampSessionCtx(0);
    try testing.expectEqual(@as(?u32, 262144), try openedContext(262144, 0));
    try testing.expectEqual(@as(?u32, default), try openedContext(null, 0));
    try testing.expectEqual(@as(?u32, default), try openedContext(2 * FakeBridge.session_cap, 0));
}

test "ds4 adapter: an explicit context is the session, never the trained window or a silent fallback" {
    try testing.expectEqual(@as(?u32, 65536), try openedContext(262144, 65536));
    try testing.expectEqual(@as(?u32, null), try openedContext(262144, 2 * FakeBridge.session_cap));
}

/// A ds4-routed GGUF of `arch` in `tmp`, declaring `template` as its chat template when set.
fn archGguf(tmp: *std.testing.TmpDir, buf: []u8, arch: []const u8, template: ?[]const u8) ![]const u8 {
    const kvs = [_]engine_pick.KV{
        .{ .key = "general.architecture", .value = .{ .str = arch } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
        .{ .key = "tokenizer.chat_template", .value = .{ .str = template orelse "" } },
    };
    const header = try engine_pick.buildHeader(testing.allocator, if (template != null) &kvs else kvs[0..2]);
    defer testing.allocator.free(header);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = header });
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const dir = root[0..try tmp.dir.realPath(io, &root)];
    return std.fmt.bufPrint(buf, "{s}/m.gguf", .{dir});
}

const Rendering = struct { host_renders: bool, template: []const u8, first_token: u32 };

/// Who renders a chat for a GGUF of `arch` with `template`, and how the adapter tokenizes "a".
fn rendering(arch: []const u8, template: ?[]const u8, tbuf: []u8) !Rendering {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try archGguf(&tmp, &buf, arch, template);
    var host: abi.Host = .{ .ctx = undefined, .log = struct {
        fn f(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}
    }.f, .wake = struct {
        fn f(_: *anyopaque) callconv(.c) void {}
    }.f };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 4096, .max_concurrent = 1, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v) orelse return error.Open;
    defer Fake.table.close(e);
    const chat_template = Fake.table.chat_template orelse return error.NoChatTemplate;
    const t = abi.items(u8, chat_template(e));
    @memcpy(tbuf[0..t.len], t);
    var out: [4]u32 = undefined;
    _ = Fake.table.tokenize.?(e, abi.span(u8, "a"), false, &out, out.len);
    return .{ .host_renders = !info.caps.owns_template, .template = tbuf[0..t.len], .first_token = out[0] };
}

test "ds4 adapter: a GGUF's own chat template goes to the host's Jinja, except DeepSeek-V4's, which ds4 encodes" {
    var tbuf: [64]u8 = undefined;
    const qwen = try rendering("qwen4exp", "{{ qwen }}", &tbuf);
    try testing.expect(qwen.host_renders);
    try testing.expectEqualStrings("{{ qwen }}", qwen.template);
    // The rendered chat carries special tokens: the fake's rendered-chat tokenizer offsets ids by 1000.
    try testing.expectEqual(@as(u32, 'a' + 1000), qwen.first_token);

    const dsv4 = try rendering("deepseek4", "{{ dsv4 }}", &tbuf);
    try testing.expect(!dsv4.host_renders);
    try testing.expectEqual(@as(usize, 0), dsv4.template.len);
    try testing.expectEqual(@as(u32, 'a'), dsv4.first_token);

    const bare = try rendering("qwen4exp", null, &tbuf);
    try testing.expect(!bare.host_renders);
}

/// Runs `prompts` one after another on one opened session (plain greedy rounds, 3 tokens each); the tokens the
/// session holds afterwards, copied into `out`.
fn liveAfter(path: []const u8, prompts: []const []const u32, out: []i32) ![]i32 {
    return liveAfterWith(path, "{\"prefill_chunk\":1}", &.{}, prompts, out);
}

/// `liveAfter` with the open's `settings` (drafts on unless `prefill_chunk` is 1) and the requests' `eos`.
fn liveAfterWith(path: []const u8, settings: []const u8, eos: []const u32, prompts: []const []const u32, out: []i32) ![]i32 {
    var host: abi.Host = .{ .ctx = undefined, .log = struct {
        fn f(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}
    }.f, .wake = struct {
        fn f(_: *anyopaque) callconv(.c) void {}
    }.f };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 1, .memory_budget = 0, .settings_json = abi.span(u8, settings), .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v) orelse return error.Open;
    defer Fake.table.close(e);
    for (prompts, 1..) |p, id| {
        var c: Collect = .{};
        defer c.deinit();
        const req: abi.Request = .{ .prompt = abi.span(u32, p), .max_tokens = 3, .sampling = .{ .temperature = 0 }, .eos = abi.span(u32, eos) };
        if (Fake.table.submit(e, id, &req, c.sink()).code != .ok) return error.Refused;
        c.wait();
    }
    const live = Fake.state(e).session.toks.items;
    @memcpy(out[0..live.len], live);
    return out[0..live.len];
}

test "ds4 adapter: a client's copy of the last reply stands in for the sampled one; only what follows it is prefilled" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    const turn1 = [_]u32{ 5, 6, 7, 8, 10, 11, 12, 13, 14, 15 };
    var one_buf: [64]i32 = undefined;
    const one = try liveAfter(path, &.{&turn1}, &one_buf);
    try testing.expect(one.len > turn1.len);

    // The client's copy of the reply ([9, 50, 50, 50]) adds a framing token but preserves the entire visible body; then its end
    // of turn (2) and a new turn.
    const turn2 = turn1 ++ [_]u32{ 9, 50, 50, 50, 2, 16 };
    var two_buf: [64]i32 = undefined;
    const two = try liveAfter(path, &.{ &turn1, &turn2 }, &two_buf);
    try testing.expectEqualSlices(i32, one, two[0..one.len]);
    try testing.expectEqualSlices(i32, &.{ 2, 16 }, two[one.len .. one.len + 2]);

    // The next turn re-sends both copies: the client's view grows, the session keeps both sampled replies.
    const turn3 = turn2 ++ [_]u32{ 9, 50, 50, 50, 2, 17 };
    var three_live: [64]i32 = undefined;
    const kept3 = try liveAfter(path, &.{ &turn1, &turn2, &turn3 }, &three_live);
    try testing.expectEqualSlices(i32, two, kept3[0..two.len]);
    try testing.expectEqualSlices(i32, &.{ 2, 17 }, kept3[two.len .. two.len + 2]);

    // A client that dropped the reply and wrote a turn of its own keeps its turn: nothing it sent is replaced.
    const own_turn = turn1 ++ [_]u32{ 7, 7, 2, 16 };
    var own_live: [64]i32 = undefined;
    const own = try liveAfter(path, &.{ &turn1, &own_turn }, &own_live);
    var own_want: [own_turn.len]i32 = undefined;
    for (own_turn, &own_want) |t, *w| w.* = @intCast(t);
    try testing.expectEqualSlices(i32, &own_want, own[0..own_want.len]);

    // A history that changed before the reply is synced as sent.
    const edited = [_]u32{ 5, 99, 7, 8, 10, 11, 12, 13, 14, 15, 9, 50, 50, 50, 2, 16 };
    var three_buf: [64]i32 = undefined;
    const three = try liveAfter(path, &.{ &turn1, &edited }, &three_buf);
    const want = [_]i32{ 5, 99, 7, 8, 10, 11, 12, 13, 14, 15, 9, 50, 50, 50, 2, 16 };
    try testing.expectEqualSlices(i32, &want, three[0..want.len]);

    // A changed reply with exactly the same old suffix must remain the client's text.
    const edited_reply = turn1 ++ [_]u32{ 66, 50, 50, 50, 2, 16 };
    var edit_live: [64]i32 = undefined;
    const edited_state = try liveAfter(path, &.{ &turn1, &edited_reply }, &edit_live);
    var edit_want: [edited_reply.len]i32 = undefined;
    for (edited_reply, &edit_want) |id, *dst| dst.* = @intCast(id);
    try testing.expectEqualSlices(i32, &edit_want, edited_state[0..edit_want.len]);
}

test "ds4 adapter: a reply whose speculative round evaluated its end of turn is continued, not repeated" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    // 998: the fake drafts its end-of-turn token (2) two tokens into the reply, inside one speculative round.
    const turn1 = [_]u32{ 5, 998, 7, 8, 10, 11, 12, 13, 14, 15 };
    const turn2 = turn1 ++ [_]u32{ 9, 50, 50, 2, 16 };
    var live_buf: [64]i32 = undefined;
    const live = try liveAfterWith(path, "", &.{2}, &.{ &turn1, &turn2 }, &live_buf);
    var want: [turn1.len + 4]i32 = undefined;
    for (turn1, want[0..turn1.len]) |t, *w| w.* = @intCast(t);
    @memcpy(want[turn1.len..], &[_]i32{ 50, 50, 2, 16 });
    try testing.expectEqualSlices(i32, &want, live[0..want.len]);
}

/// A qwen4exp GGUF with its template, the vision tokens at ids 900..902 of its vocabulary, and an mmproj beside it.
fn visionGguf(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    var names: [903][]const u8 = undefined;
    for (&names) |*nm| nm.* = "t";
    names[900] = "<|vision_start|>";
    names[901] = "<|image_pad|>";
    names[902] = "<|vision_end|>";
    const header = try engine_pick.buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "qwen4exp" } },
        .{ .key = "tokenizer.chat_template", .value = .{ .str = "{{ messages }}" } },
        .{ .key = "tokenizer.ggml.tokens", .value = .{ .str_array = &names } },
    });
    defer testing.allocator.free(header);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = header });
    try tmp.dir.writeFile(io, .{ .sub_path = "mmproj-m.gguf", .data = "GGUF" });
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const dir = root[0..try tmp.dir.realPath(io, &root)];
    return std.fmt.bufPrint(buf, "{s}/m.gguf", .{dir});
}

test "ds4 adapter: an image's placeholder becomes a run of the tower's rows, synced with its span" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try visionGguf(&tmp, &buf);
    var host: abi.Host = .{ .ctx = undefined, .log = struct {
        fn f(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}
    }.f, .wake = struct {
        fn f(_: *anyopaque) callconv(.c) void {}
    }.f };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 4096, .max_concurrent = 1, .memory_budget = 0, .settings_json = abi.span(u8, "{\"prefill_chunk\":1}"), .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v) orelse return error.Open;
    defer Fake.table.close(e);
    try testing.expect(info.caps.images);

    // The template's placeholder tokenizes to the three vision tokens, one pad.
    var ids: [8]u32 = undefined;
    const n = Fake.table.tokenize.?(e, abi.span(u8, "x<|vision_start|><|image_pad|><|vision_end|>y"), false, &ids, ids.len);
    try testing.expectEqualSlices(u32, &.{ 'x' + 1000, 900, 901, 902, 'y' + 1000 }, ids[0..n]);

    // At submit the pad widens to the tower's three rows, and the image's span starts at the first of them.
    const rgb: [12]u8 = @splat(0);
    const media = [_]abi.Media{.{ .kind = .image, .at = 0, .width = 2, .height = 2, .data = abi.span(u8, &rgb), .hash = 7 }};
    const prompt = [_]u32{ 5, 900, 901, 902, 6 };
    var c: Collect = .{};
    defer c.deinit();
    const req: abi.Request = .{ .prompt = abi.span(u32, &prompt), .media = abi.span(abi.Media, &media), .max_tokens = 1, .sampling = .{ .temperature = 0 } };
    if (Fake.table.submit(e, 1, &req, c.sink()).code != .ok) return error.Refused;
    c.wait();
    const session = Fake.state(e).session;
    try testing.expectEqualSlices(i32, &.{ 5, 900, 901, 901, 901, 902, 6 }, session.toks.items[0..7]);
    try testing.expectEqualSlices(u32, &.{2}, session.span_starts.items);
    try testing.expectEqual(@as(u32, 7), c.prompt_tokens);
    try testing.expectEqualStrings("<|vision_start|><|image_pad|><|vision_end|>", abi.items(u8, info.image_placeholder));

    // Reusing the same literal placeholder with different pixels is a cache miss, not the old image's KV.
    const original_fp = Fake.state(e).image_cache.spans[0].embedding.fingerprint;
    const white: [12]u8 = @splat(255);
    var replacement = media;
    replacement[0].data = .of(&white);
    var replaced: Collect = .{};
    defer replaced.deinit();
    var next = req;
    next.media = .of(&replacement);
    if (Fake.table.submit(e, 2, &next, replaced.sink()).code != .ok) return error.Refused;
    replaced.wait();
    try testing.expectEqual(@as(u32, 0), replaced.cached);
    try testing.expect(!std.mem.eql(u8, &original_fp, &Fake.state(e).image_cache.spans[0].embedding.fingerprint));
    try testing.expectEqual(@as(u32, 1), Fake.state(e).image_cache.images.count());

    // Removed image parts clear old rows; a literal image atom without an attachment is refused.
    var bad: Collect = .{};
    defer bad.deinit();
    next.media = .{};
    if (Fake.table.submit(e, 3, &next, bad.sink()).code != .ok) return error.Refused;
    bad.wait();
    try testing.expectEqual(Reason.failed, bad.reason);
    try testing.expectEqual(@as(u32, 0), Fake.state(e).image_cache.images.count());
}

const Collect = struct {
    toks: std.ArrayList(u32) = .empty,
    eos: std.ArrayList(u32) = .empty,
    lps: std.ArrayList(struct { token: u32, logprob: f32, top_len: usize, top_token: u32 }) = .empty,
    done: std.atomic.Value(bool) = .init(false),
    reason: Reason = .stop,
    prompt_tokens: u32 = 0,
    cached: u32 = 0,
    rounds: u64 = 0,

    fn sink(c: *Collect) abi.Sink {
        return .{ .ctx = c, .event = onEvent };
    }
    fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        if (ev.kind == .prefilled) {
            c.prompt_tokens = ev.prompt_tokens;
            c.cached = ev.cached;
        }
        c.toks.appendSlice(testing.allocator, abi.items(u32, ev.tokens)) catch {};
        for (abi.items(abi.Logprob, ev.logprobs)) |lp| c.lps.append(testing.allocator, .{ .token = lp.token, .logprob = lp.logprob, .top_len = lp.top.len, .top_token = if (lp.top.len > 0) abi.items(abi.Bias, lp.top)[0].token else 0 }) catch {};
        if (ev.kind == .finished) {
            if (ev.stats) |stats| c.rounds = stats.rounds;
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

test "ds4 adapter: no_drafts selects the plain reference even with a loaded head" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    var plain: Collect = .{};
    defer plain.deinit();
    _ = try runOne(Fake, path, "", .{ .prompt = .of(&[_]u32{ 5, 6 }), .max_tokens = 3, .sampling = .{ .temperature = 0 }, .no_drafts = true }, &plain);
    try testing.expectEqual(@as(u64, 0), plain.rounds);
    var drafted: Collect = .{};
    defer drafted.deinit();
    _ = try runOne(Fake, path, "", .{ .prompt = .of(&[_]u32{ 5, 6 }), .max_tokens = 3, .sampling = .{ .temperature = 0 } }, &drafted);
    try testing.expect(drafted.rounds > 0);
}

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

test "ds4 adapter: the engine opens sized for the session's context, not ds4's 4096 default" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    var host: abi.Host = .{ .ctx = undefined, .log = undefined, .wake = undefined };
    const o: abi.Open = .{ .model_dir = abi.span(u8, path), .model_type = .{}, .config_json = .{}, .context = 65536, .max_concurrent = 1, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &host };
    var info: abi.Info = undefined;
    var v: abi.Verdict = .{};
    const e = Fake.table.open(&o, &info, &v).?;
    defer Fake.table.close(e);
    try testing.expectEqual(@as(u32, 65536), Fake.state(e).engine.ctx);
    try testing.expectEqual(@as(u32, 65536), info.context_window);
}

test "ds4 adapter: a cancel during prefill finishes the request cancelled; the next one serves" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try ds4Gguf(&tmp, &buf);
    var cancelled: Collect = .{};
    defer cancelled.deinit();
    // The fake's sync raises the request's cancel flag on token 999, as a client hanging up mid-prefill would.
    _ = try runOne(Fake, path, "", .{ .prompt = abi.span(u32, &[_]u32{ 5, 999, 7 }), .max_tokens = 3 }, &cancelled);
    try testing.expectEqual(Reason.cancelled, cancelled.reason);
    try testing.expectEqual(@as(usize, 0), cancelled.toks.items.len);
    var next: Collect = .{};
    defer next.deinit();
    _ = try runOne(Fake, path, "", .{ .prompt = abi.span(u32, &[_]u32{ 5, 6, 7 }), .max_tokens = 2, .sampling = .{ .temperature = 0 } }, &next);
    try testing.expectEqual(Reason.length, next.reason);
    try testing.expectEqual(@as(usize, 2), next.toks.items.len);
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
    const engine = try load.open(bridge, io, testing.allocator, path, .{}, bridge.clampSessionCtx(4096));
    defer engine.close();
    const session = try engine.createSession(@intCast(bridge.clampSessionCtx(4096)));
    defer session.free();
    var p: [64]i32 = undefined;
    for (ids[0..prompt_len], p[0..prompt_len]) |t, *q| q.* = @intCast(t);
    try session.sync(p[0..prompt_len], null);
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(testing.allocator);
    var rng: u64 = 0;
    const d: round.Draw = .{ .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0, .rng = &rng };
    decode: while (want.items.len < n) {
        const t = round.draw(session, d);
        if (engine.isStop(t)) break;
        if (round.shouldSpeculate(engine.mtpDraftTokens(), 0, engine.embedded_mtp)) {
            var spec: [round.max_round_tokens]i32 = undefined;
            for (try round.speculate(engine, session, t, d, @intCast(n - want.items.len), false, &spec)) |s| {
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
