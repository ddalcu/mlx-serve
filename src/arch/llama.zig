// Zig-side wrapper around llama.cpp's libllama, via the C shim in
// `lib/llama_shim/`. The C ABI lives in `src/llama_ffi.zig`; this module owns
// lifetimes and error mapping. A `LlamaContext` is ONE llama_context holding
// `n_seq` sequences (`LlamaSeq`), each with its own KV, decoded together by
// `LlamaContext.step`; a sequence keeps the `sync`/`eval`/`sample` shape of
// `src/arch/ds4.zig` so the scheduler drives either embedded engine alike.
//
// Routing: `src/main.zig` sends DeepSeek-V4-Flash GGUFs to ds4 (a bespoke
// engine for that architecture) and every other `.gguf` here.

const std = @import("std");
const ffi = @import("../llama_ffi.zig");
const log = @import("../log.zig");

pub const Error = error{
    EngineOpenFailed,
    SessionCreateFailed,
    SessionSyncFailed,
    SessionEvalFailed,
    TokenizeFailed,
    OutOfMemory,
};

pub const OpenOptions = struct {
    /// Layers to offload to Metal. 999 = "all" (every real model has fewer).
    n_gpu_layers: i32 = 999,
    /// MTP draft-head GGUF to load beside the trunk (llama.cpp's `mtp-*.gguf`).
    mtp_path: ?[]const u8 = null,
    /// Without `mtp_path`, load the trunk's own NextN heads when it ships them.
    load_mtp: bool = false,
};

/// Sequences one context holds at most.
pub const MAX_SEQS = 64;

pub const ContextOptions = struct {
    /// Context per sequence; 0 = the model's trained context.
    ctx_size: i32 = 0,
    /// Sequences decoded together, each with its own KV for `ctx_size` tokens.
    n_seq: u32 = 1,
    /// ggml types of the K and V caches (`ffi.GgmlType`); 0 = F16.
    type_k: i32 = 0,
    type_v: i32 = 0,
    /// Physical prefill batch; 0 = libllama's default (512).
    ubatch: u32 = 0,
    /// Draft tokens per MTP round; 0 = no MTP context.
    mtp_drafts: u32 = 0,
};

pub const LlamaEngine = struct {
    allocator: std.mem.Allocator,
    handle: *ffi.Engine,
    model_path_owned: [:0]u8,

    pub fn open(allocator: std.mem.Allocator, model_path: []const u8, opts: OpenOptions) Error!*LlamaEngine {
        const path_z = allocator.dupeSentinel(u8, model_path, 0) catch return Error.OutOfMemory;
        errdefer allocator.free(path_z);
        const mtp_z: ?[:0]u8 = if (opts.mtp_path) |p| (allocator.dupeSentinel(u8, p, 0) catch return Error.OutOfMemory) else null;
        defer if (mtp_z) |p| allocator.free(p);

        var err_buf: [256]u8 = undefined;
        const raw = ffi.mlx_llama_open(path_z.ptr, opts.n_gpu_layers, if (mtp_z) |p| p.ptr else null, opts.load_mtp, &err_buf, err_buf.len);
        if (raw == null) {
            log.err("[llama] open failed: {s} (model={s})\n", .{ std.mem.sliceTo(&err_buf, 0), model_path });
            return Error.EngineOpenFailed;
        }
        if (opts.mtp_path) |p| if (!ffi.mlx_llama_has_mtp(raw.?)) {
            log.warn("[llama] MTP head did not load, serving without drafts: {s}\n", .{p});
        };

        const wrapper = allocator.create(LlamaEngine) catch {
            ffi.mlx_llama_close(raw);
            return Error.OutOfMemory;
        };
        wrapper.* = .{
            .allocator = allocator,
            .handle = raw.?,
            .model_path_owned = path_z,
        };
        return wrapper;
    }

    pub fn close(self: *LlamaEngine) void {
        ffi.mlx_llama_close(self.handle);
        self.allocator.free(self.model_path_owned);
        self.allocator.destroy(self);
    }

    pub fn eosToken(self: *LlamaEngine) i32 {
        return ffi.mlx_llama_eos_token(self.handle);
    }

    pub fn isEog(self: *LlamaEngine, token: i32) bool {
        return ffi.mlx_llama_is_eog(self.handle, token);
    }

    pub fn nVocab(self: *LlamaEngine) i32 {
        return ffi.mlx_llama_n_vocab(self.handle);
    }

    /// An MTP head is loaded (the sidecar, or the trunk's own heads).
    pub fn hasMtp(self: *LlamaEngine) bool {
        return ffi.mlx_llama_has_mtp(self.handle);
    }

    /// Tokenize free-form text. `add_special` controls BOS/special insertion.
    /// Caller owns the returned slice and frees with `allocator.free`.
    pub fn tokenizeText(
        self: *LlamaEngine,
        allocator: std.mem.Allocator,
        text: []const u8,
        add_special: bool,
    ) Error![]i32 {
        if (text.len == 0) return allocator.alloc(i32, 0) catch Error.OutOfMemory;

        // First attempt with a generous estimate; on -required, grow and retry.
        var cap: i32 = @intCast(@min(text.len + 16, std.math.maxInt(i32)));
        while (true) {
            const buf = allocator.alloc(i32, @intCast(cap)) catch return Error.OutOfMemory;
            const n = ffi.mlx_llama_tokenize(
                self.handle,
                text.ptr,
                @intCast(@min(text.len, std.math.maxInt(i32))),
                add_special,
                true, // parse_special: honor template special tokens already in `text`
                buf.ptr,
                cap,
            );
            if (n >= 0) {
                return allocator.realloc(buf, @intCast(n)) catch buf[0..@intCast(n)];
            }
            // n < 0: -n is the required capacity. Grow and retry.
            allocator.free(buf);
            const needed = -n;
            if (needed <= cap) return Error.TokenizeFailed; // shouldn't happen; guard against a loop
            cap = needed;
        }
    }

    /// Single token → bytes lookup. Caller owns the returned buffer.
    /// Bytes are NOT NUL-terminated; the slice length is authoritative.
    pub fn detokenizeOne(self: *LlamaEngine, allocator: std.mem.Allocator, token_id: i32) Error![]u8 {
        var cap: i32 = 64;
        while (true) {
            const buf = allocator.alloc(u8, @intCast(cap)) catch return Error.OutOfMemory;
            const n = ffi.mlx_llama_token_to_piece(self.handle, token_id, buf.ptr, cap);
            if (n >= 0) {
                return allocator.realloc(buf, @intCast(n)) catch buf[0..@intCast(n)];
            }
            allocator.free(buf);
            const needed = -n;
            if (needed <= cap) return allocator.dupe(u8, "") catch Error.OutOfMemory;
            cap = needed;
        }
    }

    /// The model's embedded chat-template string (jinja source), or null.
    /// Borrowed; valid for the engine's lifetime. Prefer rendering this through
    /// mlx-serve's jinja engine (chat.zig) over `applyChatTemplate`.
    pub fn chatTemplate(self: *LlamaEngine) ?[]const u8 {
        const raw = ffi.mlx_llama_chat_template(self.handle);
        if (raw == null) return null;
        return std.mem.sliceTo(raw.?, 0);
    }

    pub const ChatTurn = struct { role: []const u8, content: []const u8 };

    /// Fallback chat rendering via llama_chat_apply_template (recognized formats
    /// only — not a full jinja parser). Returns the formatted prompt string;
    /// caller owns it. Use only when the jinja path is unavailable.
    pub fn applyChatTemplate(
        self: *LlamaEngine,
        allocator: std.mem.Allocator,
        turns: []const ChatTurn,
        add_assistant: bool,
    ) Error![]u8 {
        var roles = std.ArrayList([*:0]const u8).empty;
        defer roles.deinit(allocator);
        var contents = std.ArrayList([*:0]const u8).empty;
        defer contents.deinit(allocator);
        // Track the duped C strings so we can free them all at the end.
        var owned = std.ArrayList([:0]u8).empty;
        defer {
            for (owned.items) |s| allocator.free(s);
            owned.deinit(allocator);
        }

        for (turns) |t| {
            const role_z = allocator.dupeSentinel(u8, t.role, 0) catch return Error.OutOfMemory;
            owned.append(allocator, role_z) catch return Error.OutOfMemory;
            const content_z = allocator.dupeSentinel(u8, t.content, 0) catch return Error.OutOfMemory;
            owned.append(allocator, content_z) catch return Error.OutOfMemory;
            roles.append(allocator, role_z.ptr) catch return Error.OutOfMemory;
            contents.append(allocator, content_z.ptr) catch return Error.OutOfMemory;
        }

        var cap: i32 = 4096;
        while (true) {
            const buf = allocator.alloc(u8, @intCast(cap)) catch return Error.OutOfMemory;
            const n = ffi.mlx_llama_apply_chat_template(
                self.handle,
                roles.items.ptr,
                contents.items.ptr,
                @intCast(turns.len),
                add_assistant,
                buf.ptr,
                cap,
            );
            if (n < 0) {
                allocator.free(buf);
                return Error.TokenizeFailed;
            }
            if (n <= cap) {
                return allocator.realloc(buf, @intCast(n)) catch buf[0..@intCast(n)];
            }
            // n > cap: required size returned; grow and retry.
            allocator.free(buf);
            cap = n;
        }
    }

    pub fn createContext(self: *LlamaEngine, opts: ContextOptions) Error!*LlamaContext {
        const n_seq: u32 = std.math.clamp(opts.n_seq, 1, MAX_SEQS);
        const params: ffi.CtxParams = .{
            .n_ctx = opts.ctx_size,
            .n_seq = @intCast(n_seq),
            .type_k = opts.type_k,
            .type_v = opts.type_v,
            .n_ubatch = @intCast(opts.ubatch),
            .mtp_drafts = @intCast(opts.mtp_drafts),
        };
        var err_buf: [256]u8 = undefined;
        err_buf[0] = 0;
        const raw = ffi.mlx_llama_ctx_create(self.handle, &params, &err_buf, err_buf.len) orelse {
            log.err("[llama] context create failed: {s} (ctx={d} x {d} seqs, type_k={d}, type_v={d})\n", .{
                std.mem.sliceTo(&err_buf, 0), opts.ctx_size, n_seq, opts.type_k, opts.type_v,
            });
            return Error.SessionCreateFailed;
        };
        if (opts.mtp_drafts > 0 and self.hasMtp() and ffi.mlx_llama_ctx_mtp_drafts(raw) == 0) {
            log.warn("[llama] MTP head loaded but not used: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
        }
        const ctx = self.allocator.create(LlamaContext) catch {
            ffi.mlx_llama_ctx_free(raw);
            return Error.OutOfMemory;
        };
        const seqs = self.allocator.alloc(LlamaSeq, n_seq) catch {
            self.allocator.destroy(ctx);
            ffi.mlx_llama_ctx_free(raw);
            return Error.OutOfMemory;
        };
        for (seqs, 0..) |*s, i| s.* = .{ .ctx = ctx, .id = @intCast(i) };
        ctx.* = .{ .allocator = self.allocator, .engine = self, .handle = raw, .seqs = seqs };
        return ctx;
    }
};

/// KV quant for the llama.cpp engine. Mapped from the CLI flag onto ggml types;
/// F16 is the dense default, Q8_0 halves KV bytes, Q4_0 quarters them.
pub const LlamaKvQuant = enum(u8) {
    off, // F16 (default)
    q8, // Q8_0 (~2x compression, near-lossless on most archs)
    q4, // Q4_0 (~4x compression, some quality impact)

    pub fn fromString(s: []const u8) ?LlamaKvQuant {
        if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "f16") or std.mem.eql(u8, s, "F16")) return .off;
        if (std.mem.eql(u8, s, "8") or std.mem.eql(u8, s, "q8") or std.mem.eql(u8, s, "Q8_0") or std.mem.eql(u8, s, "q8_0")) return .q8;
        if (std.mem.eql(u8, s, "4") or std.mem.eql(u8, s, "q4") or std.mem.eql(u8, s, "Q4_0") or std.mem.eql(u8, s, "q4_0")) return .q4;
        return null;
    }

    pub fn ggmlType(self: LlamaKvQuant) i32 {
        return switch (self) {
            .off => 0, // shim treats 0 as "use libllama default"
            .q8 => ffi.GgmlType.Q8_0,
            .q4 => ffi.GgmlType.Q4_0,
        };
    }

    pub fn label(self: LlamaKvQuant) []const u8 {
        return switch (self) {
            .off => "F16",
            .q8 => "Q8_0",
            .q4 => "Q4_0",
        };
    }
};

/// Length of the longest common prefix of two token sequences. Pure helper so
/// the prompt-prefix reuse logic in `LlamaSeq.sync` is unit-testable without
/// a model. An off-by-one here would corrupt KV reuse, so it's covered directly.
pub fn commonPrefixLen(a: []const i32, b: []const i32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

/// One llama_context holding `seqs.len` sequences. Single-threaded: only the
/// inference thread calls in.
pub const LlamaContext = struct {
    allocator: std.mem.Allocator,
    engine: *LlamaEngine,
    handle: *ffi.Ctx,
    seqs: []LlamaSeq,

    pub fn free(self: *LlamaContext) void {
        for (self.seqs) |*s| s.resident.deinit(self.allocator);
        self.allocator.free(self.seqs);
        ffi.mlx_llama_ctx_free(self.handle);
        self.allocator.destroy(self);
    }

    /// Draft tokens per MTP round, 0 without an MTP context.
    pub fn mtpDrafts(self: *const LlamaContext) u32 {
        return @intCast(ffi.mlx_llama_ctx_mtp_drafts(self.handle));
    }

    /// One decode step for several sequences in ONE batch: `tokens[i]` goes to
    /// `seqs[i]` (distinct sequences). Each then samples from its own row.
    pub fn step(self: *LlamaContext, seqs: []const *LlamaSeq, tokens: []const i32) Error!void {
        std.debug.assert(seqs.len == tokens.len);
        var ids_buf: [MAX_SEQS]i32 = undefined;
        const ids = ids_buf[0..seqs.len];
        for (seqs, ids) |s, *id| id.* = s.id;
        var err_buf: [256]u8 = undefined;
        if (ffi.mlx_llama_step(self.handle, ids.ptr, tokens.ptr, @intCast(seqs.len), &err_buf, err_buf.len) != 0) {
            log.err("[llama] step failed ({d} seqs): {s}\n", .{ seqs.len, std.mem.sliceTo(&err_buf, 0) });
            // The failed batch left each sequence in an unknown state.
            for (seqs) |s| s.reset();
            return Error.SessionEvalFailed;
        }
        for (seqs, tokens) |s, t| s.resident.append(self.allocator, t) catch return Error.OutOfMemory;
    }
};

pub const Sampling = struct {
    /// < 0.01 is greedy, the MLX path's threshold.
    temperature: f32,
    top_k: i32 = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
};

/// One sequence of a `LlamaContext`: its own KV, prefix-reused across requests.
pub const LlamaSeq = struct {
    ctx: *LlamaContext,
    id: i32,
    /// Token ids resident in this sequence's KV (prompt + every token fed), in
    /// position order. `sync` diffs the next prompt against it to reuse the
    /// common prefix; it always mirrors the KV exactly.
    resident: std.ArrayList(i32) = .empty,
    /// Bumped on every pick; lowest = least recently used.
    last_used_ns: i64 = 0,
    /// A request is decoding on this sequence. Guarded by the scheduler's queue lock.
    busy: bool = false,

    pub fn pos(self: *LlamaSeq) i32 {
        return ffi.mlx_llama_seq_pos(self.ctx.handle, self.id);
    }

    /// Sync the KV to `prompt_ids`, reusing the longest prefix already
    /// resident from a previous request: trims the divergent tail, decodes only
    /// the suffix, and returns the number of tokens reused.
    ///
    /// At least the final prompt token is always (re)decoded so fresh logits
    /// exist for sampling — even when the whole prompt is already resident we
    /// back off one position.
    pub fn sync(self: *LlamaSeq, prompt_ids: []const i32) Error!i32 {
        var common = commonPrefixLen(self.resident.items, prompt_ids);
        if (common == prompt_ids.len and common > 0) common -= 1;

        if (common < self.resident.items.len) {
            // 1 = the tail could not be rolled back (recurrent state) and the
            // whole sequence was cleared: nothing is resident any more.
            if (ffi.mlx_llama_seq_trim(self.ctx.handle, self.id, @intCast(common)) == 1) common = 0;
            self.resident.shrinkRetainingCapacity(common);
        }

        const suffix = prompt_ids[common..];
        if (suffix.len > 0) {
            var err_buf: [256]u8 = undefined;
            if (ffi.mlx_llama_seq_prefill(self.ctx.handle, self.id, suffix.ptr, @intCast(suffix.len), &err_buf, err_buf.len) != 0) {
                log.err("[llama] prefill failed: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
                self.resident.clearRetainingCapacity();
                return Error.SessionSyncFailed;
            }
            self.resident.appendSlice(self.ctx.allocator, suffix) catch return Error.OutOfMemory;
        }
        return @intCast(common);
    }

    /// Drop all resident KV (and the mirror).
    pub fn reset(self: *LlamaSeq) void {
        ffi.mlx_llama_seq_reset(self.ctx.handle, self.id);
        self.resident.clearRetainingCapacity();
    }

    /// `sync` with a one-shot defense against libllama transients (the
    /// `failed to find a memory slot for batch of size N` class): on failure,
    /// drop the resident state and retry once cold. If the retry also fails the
    /// sequence is left clean for the next request.
    pub fn syncWithFallback(self: *LlamaSeq, prompt_ids: []const i32) Error!i32 {
        return self.sync(prompt_ids) catch |err| blk: {
            log.warn(
                "[llama] sync failed ({s}); resetting the sequence and retrying cold\n",
                .{@errorName(err)},
            );
            self.reset();
            break :blk self.sync(prompt_ids) catch |err2| {
                self.reset();
                return err2;
            };
        };
    }

    /// Advance this sequence alone by one already-sampled token.
    pub fn eval(self: *LlamaSeq, token: i32) Error!void {
        var one = [_]*LlamaSeq{self};
        return self.ctx.step(&one, &.{token});
    }

    /// Sample the next token from the logits this sequence's last decode left.
    pub fn sample(self: *LlamaSeq, s: Sampling, rng: *u64) i32 {
        const temp: f32 = if (s.temperature < 0.01) 0 else s.temperature;
        return ffi.mlx_llama_seq_sample(self.ctx.handle, self.id, temp, s.top_k, s.top_p, s.min_p, rng);
    }

    pub fn argmax(self: *LlamaSeq) i32 {
        var rng: u64 = 0;
        return self.sample(.{ .temperature = 0 }, &rng);
    }

    /// One MTP round: feed `id_last`, draft up to `max_drafts` tokens and keep
    /// those the target agrees with. Returns the accepted drafts followed by the
    /// next token (sampled, not yet fed), a slice of `out`.
    pub fn specStep(self: *LlamaSeq, id_last: i32, max_drafts: u32, s: Sampling, rng: *u64, out: []i32) Error![]i32 {
        std.debug.assert(out.len > max_drafts);
        const temp: f32 = if (s.temperature < 0.01) 0 else s.temperature;
        var err_buf: [256]u8 = undefined;
        const n = ffi.mlx_llama_seq_spec_step(self.ctx.handle, self.id, id_last, @intCast(max_drafts), temp, s.top_k, s.top_p, s.min_p, rng, out.ptr, &err_buf, err_buf.len);
        if (n < 1) {
            log.err("[llama] MTP step failed: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
            self.reset();
            return Error.SessionEvalFailed;
        }
        const got = out[0..@intCast(n)];
        // In the KV now: id_last and the accepted drafts, not the next token.
        self.resident.append(self.ctx.allocator, id_last) catch return Error.OutOfMemory;
        self.resident.appendSlice(self.ctx.allocator, got[0 .. got.len - 1]) catch return Error.OutOfMemory;
        return got;
    }
};

// ── Tests ────────────────────────────────────────────────────────────────
// Real-model tests gate on LLAMA_TEST_MODEL (a path to a small .gguf), matching
// the UD_MOE_MODEL / PLD_TEST_MODEL convention. Without it they skip so CI
// without the fixture stays green. The MTP tests also need a GGUF that ships a
// NextN head (e.g. unsloth/Qwen3.5-0.8B-MTP-GGUF).

fn testModelPath() ?[]const u8 {
    // libc getenv (same idiom as src/generate.zig / src/arch/ds4.zig); the
    // returned pointer is owned by the environment, so no free is needed.
    const raw = std.c.getenv("LLAMA_TEST_MODEL") orelse return null;
    const slice = std.mem.sliceTo(raw, 0);
    return if (slice.len == 0) null else slice;
}

/// Greedy-decode `out.len` tokens from the logits `seq`'s last decode left.
fn greedyDecode(seq: *LlamaSeq, out: []i32) !void {
    var tok = seq.argmax();
    for (out) |*o| {
        o.* = tok;
        try seq.eval(tok);
        tok = seq.argmax();
    }
}

test "llama: re-sync after a long generated tail is a cold decode, never a poisoned one (hybrid recurrent state, #286)" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const prompt = try engine.tokenizeText(allocator, "Read the file store/pricing.py and", true);
    defer allocator.free(prompt);

    var cold_out: [8]i32 = undefined;
    {
        var ctx = try engine.createContext(.{ .ctx_size = 4096 });
        defer ctx.free();
        _ = try ctx.seqs[0].sync(prompt);
        try greedyDecode(&ctx.seqs[0], &cold_out);
    }

    // A previous request left a long generated tail resident: longer than any
    // recurrent per-token snapshot window, so the tail cannot be rolled back.
    var ctx = try engine.createContext(.{ .ctx_size = 4096 });
    defer ctx.free();
    const sess = &ctx.seqs[0];
    _ = try sess.sync(prompt);
    const junk = try engine.tokenizeText(allocator, "cd /private/tmp/mlxcode && python3 -m pytest 2>&1 | head -60 and then glob every python file in the tree", false);
    defer allocator.free(junk);
    for (junk) |t| try sess.eval(t);
    try std.testing.expect(sess.resident.items.len > prompt.len + 16);

    _ = try sess.sync(prompt);
    try std.testing.expectEqual(prompt.len, sess.resident.items.len);
    var warm_out: [8]i32 = undefined;
    try greedyDecode(sess, &warm_out);
    try std.testing.expectEqualSlices(i32, &cold_out, &warm_out);
}

test "llama: tokenize round-trip and short greedy decode" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    try std.testing.expect(engine.nVocab() > 0);
    try std.testing.expect(engine.eosToken() >= 0);

    const ids = try engine.tokenizeText(allocator, "The capital of France is", true);
    defer allocator.free(ids);
    try std.testing.expect(ids.len > 0);

    // Every token detokenizes to some (possibly empty) byte slice without error.
    const piece = try engine.detokenizeOne(allocator, ids[ids.len - 1]);
    defer allocator.free(piece);

    var ctx = try engine.createContext(.{ .ctx_size = 2048 });
    defer ctx.free();
    const sess = &ctx.seqs[0];

    const cached0 = try sess.sync(ids);
    try std.testing.expectEqual(@as(i32, 0), cached0); // cold sequence: nothing reused
    const first = sess.argmax();
    try std.testing.expect(first >= 0 and first < engine.nVocab());

    // Decode a few greedy tokens; the loop must advance position and stay valid.
    var produced: usize = 0;
    var tok = first;
    while (produced < 5 and !engine.isEog(tok)) : (produced += 1) {
        try sess.eval(tok);
        tok = sess.argmax();
        try std.testing.expect(tok >= 0 and tok < engine.nVocab());
    }
    try std.testing.expect(sess.pos() >= @as(i32, @intCast(ids.len)));
}

// `syncWithFallback` is the public entry used by the scheduler. Happy path: it
// returns the same cached-prefix length and leaves the same state as `sync`.
test "llama: syncWithFallback matches sync on the happy path" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const a = try engine.tokenizeText(allocator, "Once upon a time, in a", true);
    defer allocator.free(a);
    const b = try engine.tokenizeText(allocator, "Once upon a time, in a galaxy far away", true);
    defer allocator.free(b);

    var ctx = try engine.createContext(.{ .ctx_size = 2048, .n_seq = 2 });
    defer ctx.free();
    const plain = &ctx.seqs[0];
    const fb = &ctx.seqs[1];

    // Cold: both return 0, leave the resident mirror == the prompt.
    try std.testing.expectEqual(try plain.sync(a), try fb.syncWithFallback(a));
    try std.testing.expectEqualSlices(i32, plain.resident.items, fb.resident.items);

    // Warm: prefix reuse, same cached-prefix count and same final resident.
    const c1 = try plain.sync(b);
    const c2 = try fb.syncWithFallback(b);
    try std.testing.expectEqual(c1, c2);
    try std.testing.expectEqualSlices(i32, plain.resident.items, fb.resident.items);

    // After reset it goes back to cold, and syncWithFallback still works.
    fb.reset();
    try std.testing.expectEqual(@as(usize, 0), fb.resident.items.len);
    try std.testing.expectEqual(@as(i32, 0), try fb.syncWithFallback(a));
}

test "commonPrefixLen: shared prefix, divergence, and bounds" {
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    // Identical → full length.
    try std.testing.expectEqual(@as(usize, 5), commonPrefixLen(&a, &a));
    // Shared 3-token prefix then diverge.
    try std.testing.expectEqual(@as(usize, 3), commonPrefixLen(&a, &[_]i32{ 1, 2, 3, 9, 9 }));
    // b is a strict prefix of a → bounded by b.len.
    try std.testing.expectEqual(@as(usize, 2), commonPrefixLen(&a, &[_]i32{ 1, 2 }));
    // Diverge at token 0 → 0.
    try std.testing.expectEqual(@as(usize, 0), commonPrefixLen(&a, &[_]i32{ 9, 2, 3 }));
    // Empty inputs → 0, no out-of-bounds.
    try std.testing.expectEqual(@as(usize, 0), commonPrefixLen(&a, &[_]i32{}));
    try std.testing.expectEqual(@as(usize, 0), commonPrefixLen(&[_]i32{}, &a));
}

// Prompt-prefix reuse must produce byte-identical greedy output to a cold
// decode: guards the KV-trim off-by-one.
test "llama: prefix reuse is byte-identical to cold decode" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const shared = try engine.tokenizeText(allocator, "The history of the Roman Empire is long and", true);
    defer allocator.free(shared);
    const full = try engine.tokenizeText(allocator, "The history of the Roman Empire is long and storied, beginning with", true);
    defer allocator.free(full);
    try std.testing.expect(commonPrefixLen(shared, full) >= shared.len - 1);

    var cold_out: [12]i32 = undefined;
    {
        var ctx = try engine.createContext(.{ .ctx_size = 4096 });
        defer ctx.free();
        _ = try ctx.seqs[0].sync(full);
        try greedyDecode(&ctx.seqs[0], &cold_out);
    }

    // Warm: prime with `shared` plus two generated tokens (the multi-turn
    // shape), then sync `full`. A recurrent memory that cannot roll its tail
    // back reports 0 and cold-prefills; both are correct if the bytes match.
    var ctx = try engine.createContext(.{ .ctx_size = 4096 });
    defer ctx.free();
    const sess = &ctx.seqs[0];
    _ = try sess.sync(shared);
    var two: [2]i32 = undefined;
    try greedyDecode(sess, &two);
    try std.testing.expect(try sess.sync(full) <= @as(i32, @intCast(full.len)));

    var warm_out: [12]i32 = undefined;
    try greedyDecode(sess, &warm_out);
    try std.testing.expectEqualSlices(i32, &cold_out, &warm_out);
}

/// Prefill each prompt into its own sequence, then greedy-decode `n` tokens
/// for all of them in one batched step per token.
fn batchedGreedy(ctx: *LlamaContext, prompts: []const []const i32, n: usize, out: [][]i32) !void {
    var seqs: [8]*LlamaSeq = undefined;
    var toks: [8]i32 = undefined;
    for (prompts, 0..) |p, i| {
        seqs[i] = &ctx.seqs[i];
        _ = try seqs[i].sync(p);
        toks[i] = seqs[i].argmax();
    }
    for (0..n) |t| {
        for (0..prompts.len) |i| out[i][t] = toks[i];
        try ctx.step(seqs[0..prompts.len], toks[0..prompts.len]);
        for (0..prompts.len) |i| toks[i] = seqs[i].argmax();
    }
}

test "llama: sequences batched in one step never read each other's KV (#547)" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const p = try engine.tokenizeText(allocator, "List three primary colors:", true);
    defer allocator.free(p);
    const q1 = try engine.tokenizeText(allocator, "Write a haiku about the sea and the moon over the harbor", true);
    defer allocator.free(q1);
    const q2 = try engine.tokenizeText(allocator, "def fib(n):", true);
    defer allocator.free(q2);

    var a1: [16]i32 = undefined;
    var b1: [16]i32 = undefined;
    var a2: [16]i32 = undefined;
    var b2: [16]i32 = undefined;
    var ctx = try engine.createContext(.{ .ctx_size = 1024, .n_seq = 2 });
    defer ctx.free();
    var run1 = [_][]i32{ &a1, &b1 };
    try batchedGreedy(ctx, &.{ p, q1 }, 16, &run1);
    ctx.seqs[0].reset();
    ctx.seqs[1].reset();
    var run2 = [_][]i32{ &a2, &b2 };
    try batchedGreedy(ctx, &.{ p, q2 }, 16, &run2);

    // Same batch shape, different neighbour: sequence 0 decodes the same tokens.
    try std.testing.expectEqualSlices(i32, &a1, &a2);
    try std.testing.expect(!std.mem.eql(i32, &b1, &b2));
}

/// Greedy-decode `out.len` tokens through MTP rounds; returns the drafts accepted.
fn mtpGreedy(seq: *LlamaSeq, max_drafts: u32, out: []i32) !usize {
    var rng: u64 = 0;
    var buf: [9]i32 = undefined;
    var next = seq.argmax();
    var n: usize = 0;
    var accepted: usize = 0;
    while (n < out.len) {
        out[n] = next;
        n += 1;
        const got = try seq.specStep(next, max_drafts, .{ .temperature = 0 }, &rng, &buf);
        for (got[0 .. got.len - 1]) |t| {
            if (n == out.len) break;
            out[n] = t;
            n += 1;
        }
        accepted += got.len - 1;
        next = got[got.len - 1];
    }
    return accepted;
}

fn expectMtpMatchesPlain(engine: *LlamaEngine) !void {
    const allocator = std.testing.allocator;
    const prompt = try engine.tokenizeText(allocator, "Count from one to twenty in words: one, two, three,", true);
    defer allocator.free(prompt);
    const turn2 = try engine.tokenizeText(allocator, "Count from one to twenty in words: one, two, three, four, five. Now backwards from ten:", true);
    defer allocator.free(turn2);

    var plain: [32]i32 = undefined;
    var plain2: [16]i32 = undefined;
    {
        var ctx = try engine.createContext(.{ .ctx_size = 2048 });
        defer ctx.free();
        try std.testing.expectEqual(@as(u32, 0), ctx.mtpDrafts());
        _ = try ctx.seqs[0].sync(prompt);
        try greedyDecode(&ctx.seqs[0], &plain);
        _ = try ctx.seqs[0].sync(turn2);
        try greedyDecode(&ctx.seqs[0], &plain2);
    }

    var ctx = try engine.createContext(.{ .ctx_size = 2048, .mtp_drafts = 2 });
    defer ctx.free();
    try std.testing.expectEqual(@as(u32, 2), ctx.mtpDrafts());
    var spec: [32]i32 = undefined;
    _ = try ctx.seqs[0].sync(prompt);
    const accepted = try mtpGreedy(&ctx.seqs[0], 2, &spec);
    try std.testing.expectEqualSlices(i32, &plain, &spec);
    try std.testing.expect(accepted > 0);

    // A second turn reuses the prefix; the head's KV is trimmed with the trunk's.
    var spec2: [16]i32 = undefined;
    _ = try ctx.seqs[0].sync(turn2);
    _ = try mtpGreedy(&ctx.seqs[0], 2, &spec2);
    try std.testing.expectEqualSlices(i32, &plain2, &spec2);
}

test "llama: greedy MTP rounds emit the plain greedy tokens and accept drafts (#548)" {
    const path = testModelPath() orelse return error.SkipZigTest;
    var engine = try LlamaEngine.open(std.testing.allocator, path, .{ .load_mtp = true });
    defer engine.close();
    if (!engine.hasMtp()) return error.SkipZigTest;
    try expectMtpMatchesPlain(engine);
}

test "llama: an MTP head loaded from a separate GGUF drafts like the trunk's own (#548)" {
    const path = testModelPath() orelse return error.SkipZigTest;
    // The fixture's own file stands in for an `mtp-*.gguf` sidecar: it carries the head.
    var engine = try LlamaEngine.open(std.testing.allocator, path, .{ .mtp_path = path });
    defer engine.close();
    if (!engine.hasMtp()) return error.SkipZigTest;
    try expectMtpMatchesPlain(engine);
}
