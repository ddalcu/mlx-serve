// Zig-side wrapper around llama.cpp's libllama, via the C shim in
// `lib/llama_shim/`. The C ABI lives in `src/llama_ffi.zig`; this module owns
// lifetimes and error mapping and exposes the same `sync`/`eval`/`argmax`/
// `sample` session shape as `src/arch/ds4.zig`, so the scheduler can drive a
// GGUF slot through either embedded engine with the same per-token loop.
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
};

pub const LlamaEngine = struct {
    allocator: std.mem.Allocator,
    handle: *ffi.Engine,
    model_path_owned: [:0]u8,

    pub fn open(allocator: std.mem.Allocator, model_path: []const u8, opts: OpenOptions) Error!*LlamaEngine {
        const path_z = allocator.dupeZ(u8, model_path) catch return Error.OutOfMemory;
        errdefer allocator.free(path_z);

        var err_buf: [256]u8 = undefined;
        const raw = ffi.mlx_llama_open(path_z.ptr, opts.n_gpu_layers, &err_buf, err_buf.len);
        if (raw == null) {
            log.err("[llama] open failed: {s} (model={s})\n", .{ std.mem.sliceTo(&err_buf, 0), model_path });
            return Error.EngineOpenFailed;
        }

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
            const role_z = allocator.dupeZ(u8, t.role) catch return Error.OutOfMemory;
            owned.append(allocator, role_z) catch return Error.OutOfMemory;
            const content_z = allocator.dupeZ(u8, t.content) catch return Error.OutOfMemory;
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

    pub fn createSession(self: *LlamaEngine, ctx_size: i32) Error!*LlamaSession {
        return self.createSessionWithKvQuant(ctx_size, 0, 0);
    }

    /// Plan 5 #2: create a session with non-default ggml types for the K and
    /// V halves of the KV cache. `type_k` / `type_v` are ggml_type integers
    /// (see `llama_ffi.GgmlType`). Passing 0 keeps the libllama default
    /// (F16 in current versions). Non-default values automatically enable
    /// flash attention in the shim (required for quantized KV).
    pub fn createSessionWithKvQuant(
        self: *LlamaEngine,
        ctx_size: i32,
        type_k: i32,
        type_v: i32,
    ) Error!*LlamaSession {
        var err_buf: [256]u8 = undefined;
        const raw = ffi.mlx_llama_session_create_kv_quant(
            self.handle,
            ctx_size,
            type_k,
            type_v,
            &err_buf,
            err_buf.len,
        );
        if (raw == null) {
            log.err("[llama] session_create_kv_quant failed: {s} (ctx={d}, type_k={d}, type_v={d})\n", .{
                std.mem.sliceTo(&err_buf, 0), ctx_size, type_k, type_v,
            });
            return Error.SessionCreateFailed;
        }
        const sess = self.allocator.create(LlamaSession) catch {
            ffi.mlx_llama_session_free(raw);
            return Error.OutOfMemory;
        };
        sess.* = .{
            .allocator = self.allocator,
            .engine = self,
            .handle = raw.?,
            .ctx_size = ctx_size,
            .resident = .empty,
        };
        return sess;
    }
};

/// Plan 5 #2: KV quant for the llama.cpp engine. Mapped from the CLI/body
/// flag onto ggml types; F16 is the dense default, Q8_0 halves KV bytes,
/// Q4_0 quarters them. See `mlx_llama_session_create_kv_quant`.
pub const LlamaKvQuant = enum(u8) {
    off,    // F16 (default)
    q8,     // Q8_0 (~2x compression, near-lossless on most archs)
    q4,     // Q4_0 (~4x compression, some quality impact)

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
/// the prompt-prefix reuse logic in `LlamaSession.sync` is unit-testable without
/// a model. An off-by-one here would corrupt KV reuse, so it's covered directly.
pub fn commonPrefixLen(a: []const i32, b: []const i32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

pub const LlamaSession = struct {
    allocator: std.mem.Allocator,
    engine: *LlamaEngine,
    handle: *ffi.Session,
    ctx_size: i32,
    /// Token ids currently resident in the KV cache (prompt + every token fed via
    /// `eval`), in position order. `sync` diffs the next prompt against this to
    /// reuse the common prefix; it always mirrors the C session's KV exactly.
    resident: std.ArrayList(i32),

    pub fn free(self: *LlamaSession) void {
        self.resident.deinit(self.allocator);
        ffi.mlx_llama_session_free(self.handle);
        self.allocator.destroy(self);
    }

    pub fn pos(self: *LlamaSession) i32 {
        return ffi.mlx_llama_session_pos(self.handle);
    }

    /// Sync the KV cache to `prompt_ids`, reusing the longest prefix already
    /// resident from a previous request. Trims the divergent tail, decodes only
    /// the suffix, and updates the resident-token mirror. Returns the number of
    /// tokens reused (the cached prefix length) so the caller can report it.
    ///
    /// At least the final prompt token is always (re)decoded so fresh logits
    /// exist for sampling — even when the whole prompt is already resident we
    /// back off one position. Decoding the suffix continues from the trim point
    /// because libllama tracks positions from the KV state.
    pub fn sync(self: *LlamaSession, prompt_ids: []const i32) Error!i32 {
        var common = commonPrefixLen(self.resident.items, prompt_ids);
        if (common == prompt_ids.len and common > 0) common -= 1;

        if (common < self.resident.items.len) {
            _ = ffi.mlx_llama_session_trim(self.handle, @intCast(common));
            self.resident.shrinkRetainingCapacity(common);
        }

        const suffix = prompt_ids[common..];
        if (suffix.len > 0) {
            var err_buf: [256]u8 = undefined;
            const rc = ffi.mlx_llama_session_sync(
                self.handle,
                suffix.ptr,
                @intCast(suffix.len),
                &err_buf,
                err_buf.len,
            );
            if (rc != 0) {
                log.err("[llama] session_sync rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
                return Error.SessionSyncFailed;
            }
            self.resident.appendSlice(self.allocator, suffix) catch return Error.OutOfMemory;
        }
        return @intCast(common);
    }

    /// Drop all resident KV (and the mirror). Used when the model wants a clean
    /// slate before a cold prefill.
    pub fn reset(self: *LlamaSession) void {
        ffi.mlx_llama_session_reset(self.handle);
        self.resident.clearRetainingCapacity();
    }

    /// `sync` with a one-shot defense against libllama transients (the
    /// `init_batch: failed to prepare attention ubatches` / `decode: failed to
    /// find a memory slot for batch of size N` class). On any failure we drop
    /// the resident state and retry once with a cold prefill — we lose this
    /// single request's prefix-reuse savings but the client gets a normal
    /// response instead of a 500. If the retry also fails the session is left
    /// clean so the *next* request can cold-prefill into a known-good state.
    ///
    /// Behaves identically to `sync` on the happy path, including the returned
    /// cached-prefix length.
    pub fn syncWithFallback(self: *LlamaSession, prompt_ids: []const i32) Error!i32 {
        return self.sync(prompt_ids) catch |err| blk: {
            log.warn(
                "[llama] sync failed ({s}); resetting session and retrying cold\n",
                .{@errorName(err)},
            );
            self.reset();
            break :blk self.sync(prompt_ids) catch |err2| {
                self.reset();
                return err2;
            };
        };
    }

    /// Advance the KV cache by one already-sampled token.
    pub fn eval(self: *LlamaSession, token: i32) Error!void {
        var err_buf: [256]u8 = undefined;
        const rc = ffi.mlx_llama_session_eval(self.handle, token, &err_buf, err_buf.len);
        if (rc != 0) {
            log.err("[llama] session_eval rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
            return Error.SessionEvalFailed;
        }
        self.resident.append(self.allocator, token) catch return Error.OutOfMemory;
    }

    pub fn argmax(self: *LlamaSession) i32 {
        return ffi.mlx_llama_session_argmax(self.handle);
    }

    pub fn sample(self: *LlamaSession, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: *u64) i32 {
        return ffi.mlx_llama_session_sample(self.handle, temperature, top_k, top_p, min_p, rng);
    }
};

// ── Tests ────────────────────────────────────────────────────────────────
// Real-model tests gate on LLAMA_TEST_MODEL (a path to a small .gguf), matching
// the UD_MOE_MODEL / PLD_TEST_MODEL convention. Without it they skip so CI
// without the fixture stays green.

fn testModelPath() ?[]const u8 {
    // libc getenv (same idiom as src/generate.zig / src/arch/ds4.zig); the
    // returned pointer is owned by the environment, so no free is needed.
    const raw = std.c.getenv("LLAMA_TEST_MODEL") orelse return null;
    const slice = std.mem.sliceTo(raw, 0);
    return if (slice.len == 0) null else slice;
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

    var sess = try engine.createSession(2048);
    defer sess.free();

    const cached0 = try sess.sync(ids);
    try std.testing.expectEqual(@as(i32, 0), cached0); // cold session: nothing reused
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

// Model-gated: `syncWithFallback` is the public entry used by the scheduler.
// Happy path — it must return the same cached-prefix length and leave the
// session in the same state as plain `sync`. The retry-on-error branch is
// covered by the integration test `tests/test_llama_persistent_session.sh`,
// which exercises the full HTTP path through the scheduler.
test "llama: syncWithFallback matches sync on the happy path" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const a = try engine.tokenizeText(allocator, "Once upon a time, in a", true);
    defer allocator.free(a);
    const b = try engine.tokenizeText(allocator, "Once upon a time, in a galaxy far away", true);
    defer allocator.free(b);

    var sess_plain = try engine.createSession(2048);
    defer sess_plain.free();
    var sess_fb = try engine.createSession(2048);
    defer sess_fb.free();

    // Cold: both return 0, leave the resident mirror == the prompt.
    try std.testing.expectEqual(
        try sess_plain.sync(a),
        try sess_fb.syncWithFallback(a),
    );
    try std.testing.expectEqualSlices(i32, sess_plain.resident.items, sess_fb.resident.items);

    // Warm: prefix reuse, same cached-prefix count and same final resident.
    const c1 = try sess_plain.sync(b);
    const c2 = try sess_fb.syncWithFallback(b);
    try std.testing.expectEqual(c1, c2);
    try std.testing.expectEqualSlices(i32, sess_plain.resident.items, sess_fb.resident.items);

    // After reset both go back to cold, and syncWithFallback still works.
    sess_fb.reset();
    try std.testing.expectEqual(@as(usize, 0), sess_fb.resident.items.len);
    const c3 = try sess_fb.syncWithFallback(a);
    try std.testing.expectEqual(@as(i32, 0), c3);
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

// Model-gated: prove prompt-prefix reuse produces byte-identical greedy output
// to a cold decode, and that the second request reuses the shared prefix. Guards
// the KV-trim off-by-one (the dangerous failure mode of Phase 3).
test "llama: prefix reuse is byte-identical to cold decode" {
    const allocator = std.testing.allocator;
    const path = testModelPath() orelse return error.SkipZigTest;

    var engine = try LlamaEngine.open(allocator, path, .{});
    defer engine.close();

    const shared = try engine.tokenizeText(allocator, "The history of the Roman Empire is long and", true);
    defer allocator.free(shared);
    // A divergent prompt that extends the shared prefix.
    const full = try engine.tokenizeText(allocator, "The history of the Roman Empire is long and storied, beginning with", true);
    defer allocator.free(full);
    try std.testing.expect(commonPrefixLen(shared, full) >= shared.len - 1);

    const N = 12; // first-N greedy tokens; short enough to stay byte-stable

    // Cold reference: a fresh session decodes `full` from scratch.
    var cold_out: [N]i32 = undefined;
    {
        var sess = try engine.createSession(4096);
        defer sess.free();
        _ = try sess.sync(full);
        var tok = sess.argmax();
        var i: usize = 0;
        while (i < N) : (i += 1) {
            cold_out[i] = tok;
            if (engine.isEog(tok)) break;
            try sess.eval(tok);
            tok = sess.argmax();
        }
    }

    // Warm: one session, prime with `shared`, then sync `full` (reuses prefix).
    var sess = try engine.createSession(4096);
    defer sess.free();

    _ = try sess.sync(shared);
    // Generate a couple tokens so the resident KV holds prompt + generated, the
    // realistic multi-turn shape the reuse must survive.
    var warm_tok = sess.argmax();
    try sess.eval(warm_tok);
    warm_tok = sess.argmax();
    try sess.eval(warm_tok);

    const cached = try sess.sync(full);
    try std.testing.expect(cached > 0); // reused the shared prefix
    try std.testing.expect(cached <= @as(i32, @intCast(full.len)));

    var tok = sess.argmax();
    var i: usize = 0;
    while (i < N) : (i += 1) {
        try std.testing.expectEqual(cold_out[i], tok);
        if (engine.isEog(tok)) break;
        try sess.eval(tok);
        tok = sess.argmax();
    }
}
