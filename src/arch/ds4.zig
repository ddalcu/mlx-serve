// Zig-side wrapper around the ds4 inference engine. The C ABI lives in
// `src/ds4_ffi.zig`; this module:
//
//   1. Extracts the embedded Metal kernels (`lib/ds4_metal_sources.zig`) to
//      `~/.mlx-serve/ds4-metal/<binary-hash>/` on first use and points ds4 at
//      them via its `DS4_METAL_<NAME>_SOURCE` env-var overrides — zero patches
//      to the upstream submodule.
//   2. Owns one `Ds4Engine` per loaded model (single-engine-per-process; ds4
//      enforces this via `~/.ds4.lock`).
//   3. Maps Zig's per-request slot model onto `ds4_session_*` so the existing
//      `Generator` shape in `src/generate.zig` carries over.
//
// The engine API surface tracks `ds4.h`'s `ds4_session_*` family. Functions
// here keep the same names (`sync`, `eval`, `argmax`, `sample`, …) so callers
// in `transformer.zig`/`generate.zig` switch on an `Engine` union and forward
// straight through.

const std = @import("std");
const ffi = @import("../ds4_ffi.zig");
const metal_sources = @import("ds4_metal_sources");
const log = @import("../log.zig");

pub const Error = error{
    EngineOpenFailed,
    MissingDsparkSupport,
    SessionCreateFailed,
    SessionSyncFailed,
    SessionEvalFailed,
    SessionSpecFailed,
    SnapshotFailed,
    LoadSnapshotFailed,
    KernelExtractionFailed,
    OutOfMemory,
};

/// ds4's prefill scratch is sized against the requested session ctx (the
/// engine's `prefill_chunk` is 2048 by default). A session SMALLER than the
/// prefill chunk produces junk output, so this is the hard floor for any
/// user-supplied `--ctx-size` — below it we raise rather than honor verbatim.
pub const ds4_prefill_chunk: u32 = 2048;
/// ds4's CLI default ctx, used when the user leaves `--ctx-size` unset (0).
pub const ds4_default_ctx: u32 = 32768;

/// Resolve a ds4 session context size from the user's `--ctx-size`.
/// `requested == 0` (unset) → ds4's default; otherwise the value, floored at
/// the prefill chunk so it can never drop into the junk-output regime. No
/// upper cap — a deliberately large ctx just allocates more KV scratch up
/// front, which is the caller's explicit choice. Idempotent for non-zero
/// inputs (re-clamping a clamped value is a no-op).
pub fn clampSessionCtx(requested: u32) u32 {
    if (requested == 0) return ds4_default_ctx;
    return @max(requested, ds4_prefill_chunk);
}

const KernelEntry = struct {
    env_var: [:0]const u8,
    file_name: []const u8,
    source: []const u8,
};

const kernel_entries = [_]KernelEntry{
    .{ .env_var = "DS4_METAL_FLASH_ATTN_SOURCE", .file_name = "flash_attn.metal", .source = metal_sources.flash_attn },
    .{ .env_var = "DS4_METAL_DENSE_SOURCE", .file_name = "dense.metal", .source = metal_sources.dense },
    .{ .env_var = "DS4_METAL_MOE_SOURCE", .file_name = "moe.metal", .source = metal_sources.moe },
    .{ .env_var = "DS4_METAL_DSV4_HC_SOURCE", .file_name = "dsv4_hc.metal", .source = metal_sources.dsv4_hc },
    .{ .env_var = "DS4_METAL_UNARY_SOURCE", .file_name = "unary.metal", .source = metal_sources.unary },
    .{ .env_var = "DS4_METAL_DSV4_KV_SOURCE", .file_name = "dsv4_kv.metal", .source = metal_sources.dsv4_kv },
    .{ .env_var = "DS4_METAL_DSV4_ROPE_SOURCE", .file_name = "dsv4_rope.metal", .source = metal_sources.dsv4_rope },
    .{ .env_var = "DS4_METAL_DSV4_MISC_SOURCE", .file_name = "dsv4_misc.metal", .source = metal_sources.dsv4_misc },
    .{ .env_var = "DS4_METAL_ARGSORT_SOURCE", .file_name = "argsort.metal", .source = metal_sources.argsort },
    .{ .env_var = "DS4_METAL_CPY_SOURCE", .file_name = "cpy.metal", .source = metal_sources.cpy },
    .{ .env_var = "DS4_METAL_CONCAT_SOURCE", .file_name = "concat.metal", .source = metal_sources.concat },
    .{ .env_var = "DS4_METAL_GET_ROWS_SOURCE", .file_name = "get_rows.metal", .source = metal_sources.get_rows },
    .{ .env_var = "DS4_METAL_SUM_ROWS_SOURCE", .file_name = "sum_rows.metal", .source = metal_sources.sum_rows },
    .{ .env_var = "DS4_METAL_SOFTMAX_SOURCE", .file_name = "softmax.metal", .source = metal_sources.softmax },
    .{ .env_var = "DS4_METAL_REPEAT_SOURCE", .file_name = "repeat.metal", .source = metal_sources.repeat },
    .{ .env_var = "DS4_METAL_GLU_SOURCE", .file_name = "glu.metal", .source = metal_sources.glu },
    .{ .env_var = "DS4_METAL_NORM_SOURCE", .file_name = "norm.metal", .source = metal_sources.norm },
    .{ .env_var = "DS4_METAL_BIN_SOURCE", .file_name = "bin.metal", .source = metal_sources.bin },
    .{ .env_var = "DS4_METAL_SET_ROWS_SOURCE", .file_name = "set_rows.metal", .source = metal_sources.set_rows },
};

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn mkdir(path: [*:0]const u8, mode: u16) c_int;
/// `creat()` is the non-variadic 2-arg form of `open()`. Equivalent to
/// `open(path, O_WRONLY|O_CREAT|O_TRUNC, mode)` and avoids the AArch64
/// variadic-arg calling-convention mismatch we'd hit declaring the 3-arg
/// `open()` as a non-variadic Zig `extern fn`.
extern "c" fn creat(path: [*:0]const u8, mode: u16) c_int;
extern "c" fn chmod(path: [*:0]const u8, mode: u16) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;

/// Atomic flag — `ensureMetalKernels` is idempotent and only writes the env
/// vars on the first call. In 0.16 std.Thread no longer exposes Mutex; we use
/// a CAS-once pattern instead since the work being protected is a single
/// O(n) staging step. Re-entry races are benign (each thread does identical
/// work and ends up with the same env-var values), but we cheaply gate
/// repeats anyway.
var kernels_extracted: std.atomic.Value(bool) = .{ .raw = false };

/// Hash the embedded kernel sources so we re-extract on binary upgrade and
/// skip the write otherwise. Stable across process restarts of the same
/// binary.
fn computeKernelHash() [16]u8 {
    var h = std.crypto.hash.Md5.init(.{});
    for (kernel_entries) |k| {
        h.update(k.env_var);
        h.update(k.source);
    }
    var out: [16]u8 = undefined;
    h.final(&out);
    return out;
}

fn hexEncode(bytes: []const u8, buf: []u8) []u8 {
    const hex = "0123456789abcdef";
    var i: usize = 0;
    while (i < bytes.len) : (i += 1) {
        buf[i * 2 + 0] = hex[bytes[i] >> 4];
        buf[i * 2 + 1] = hex[bytes[i] & 0x0f];
    }
    return buf[0 .. bytes.len * 2];
}

/// Idempotent: writes Metal sources to disk on first call and points ds4 at
/// them via `setenv()`. Subsequent calls return early. Safe to call before
/// each engine_open even though once-per-process is sufficient.
pub fn ensureMetalKernels(allocator: std.mem.Allocator) Error!void {
    if (kernels_extracted.load(.acquire)) return;

    const home_z = std.c.getenv("HOME");
    if (home_z == null) {
        log.err("[ds4] HOME not set; cannot stage Metal kernels\n", .{});
        return Error.KernelExtractionFailed;
    }
    const home = std.mem.sliceTo(home_z.?, 0);

    var hash_bytes: [16]u8 = computeKernelHash();
    var hex_buf: [32]u8 = undefined;
    const hash_hex = hexEncode(&hash_bytes, &hex_buf);

    const dir_path_slice = std.fmt.allocPrint(allocator, "{s}/.mlx-serve/ds4-metal/{s}", .{ home, hash_hex }) catch return Error.OutOfMemory;
    defer allocator.free(dir_path_slice);
    const dir_path = allocator.dupeSentinel(u8, dir_path_slice, 0) catch return Error.OutOfMemory;
    defer allocator.free(dir_path);

    // Best-effort recursive mkdir via libc. Walk the path component-by-component
    // and call mkdir(); ignore EEXIST. Simpler than dragging in std.fs.Dir,
    // which gained an Io parameter in 0.16 that we'd have to thread down.
    {
        var buf: [4096]u8 = undefined;
        var i: usize = 0;
        while (i <= dir_path.len) : (i += 1) {
            if (i == dir_path.len or dir_path[i] == '/') {
                if (i == 0) continue;
                if (i >= buf.len) return Error.KernelExtractionFailed;
                @memcpy(buf[0..i], dir_path[0..i]);
                buf[i] = 0;
                _ = mkdir(@ptrCast(&buf), 0o755);
            }
        }
    }

    for (kernel_entries) |k| {
        const file_path_slice = std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, k.file_name }) catch return Error.OutOfMemory;
        defer allocator.free(file_path_slice);
        const file_path = allocator.dupeSentinel(u8, file_path_slice, 0) catch return Error.OutOfMemory;
        defer allocator.free(file_path);

        // Always rewrite — staging is cheap (19 small kernel files), and the
        // content-hashed dir means we'd otherwise need a size check anyway.
        const fd = creat(file_path.ptr, 0o644);
        if (fd < 0) {
            log.err("[ds4] creat() failed: {s}\n", .{file_path});
            return Error.KernelExtractionFailed;
        }
        var written: usize = 0;
        while (written < k.source.len) {
            const n = write(fd, k.source.ptr + written, k.source.len - written);
            if (n <= 0) {
                _ = close(fd);
                log.err("[ds4] short write: {s}\n", .{file_path});
                return Error.KernelExtractionFailed;
            }
            written += @intCast(n);
        }
        _ = close(fd);
        // Belt and suspenders: explicit chmod since umask may have stripped
        // group/other read bits at creat() time.
        _ = chmod(file_path.ptr, 0o644);

        if (setenv(k.env_var.ptr, file_path.ptr, 1) != 0) {
            log.err("[ds4] setenv {s} failed\n", .{k.env_var});
            return Error.KernelExtractionFailed;
        }
    }

    log.info("[ds4] staged {d} Metal kernel sources at {s}\n", .{ kernel_entries.len, dir_path });
    kernels_extracted.store(true, .release);
}

pub const OpenOptions = struct {
    backend: ffi.Backend = .metal,
    n_threads: c_int = 0,
    /// Startup admission is sized to this context, before any persistent
    /// DSpark support view is registered.
    context_size: c_int = 0,
    warm_weights: bool = true,
    quality: bool = false,
    /// Optional support GGUF (ds4 speculative decode): the legacy MTP draft
    /// head OR the 0731 DSpark stage bundle — the engine classifies by
    /// tensors. It dupes the path to a NUL-terminated string, so a plain
    /// slice is fine here.
    mtp_path: ?[]const u8 = null,
    mtp_draft_tokens: c_int = 0,
    mtp_margin: f32 = 0,
    /// Select the DSpark runtime for a DSpark-kind support GGUF (upstream
    /// `--dspark`; without it the engine loads the stages but keeps
    /// target-only decode). Ignored for legacy-MTP support GGUFs. Confidence
    /// pruning stays at the engine default (0.9).
    dspark: bool = false,
    /// SSD weight-streaming (issue #39): skip full model residency + warmup and
    /// stream expert weights from disk, with an in-RAM cache. Lets DeepSeek-V4-Flash
    /// run on machines whose RAM can't hold the full model. When neither cache
    /// field is set, mlx-serve supplies its measured-safe 8 GiB budget instead
    /// of DwarfStar's device-limit-based auto budget, which cannot see other
    /// resident processes.
    ssd_streaming: bool = false,
    ssd_streaming_cold: bool = false,
    ssd_streaming_cache_experts: u32 = 0,
    ssd_streaming_cache_bytes: u64 = 0,
    ssd_streaming_preload_experts: u32 = 0,
};

/// Shared offline/server policy for ds4 support sidecars. `--no-ds4-mtp` is a
/// deliberate opt-out and therefore disables DSpark too. An explicit DSpark
/// request requires the deterministic lineage-matched DSpark sidecar; it must
/// never silently degrade to target-only or substitute the legacy MTP
/// checkpoint. Plain SSD remains target-only even when a sidecar is present.
pub const SupportSidecarPolicy = struct {
    lookup: bool,
    require_dspark: bool,
    dspark: bool,
};

pub fn supportSidecarPolicy(
    mtp_enabled: bool,
    ssd_streaming: bool,
    dspark_requested: bool,
) SupportSidecarPolicy {
    if (!mtp_enabled) return .{
        .lookup = false,
        .require_dspark = false,
        .dspark = false,
    };
    if (dspark_requested) return .{
        .lookup = true,
        .require_dspark = true,
        .dspark = true,
    };
    if (ssd_streaming) return .{
        .lookup = false,
        .require_dspark = false,
        .dspark = false,
    };
    return .{
        .lookup = true,
        .require_dspark = false,
        .dspark = false,
    };
}

/// SSD streaming is explicitly the bounded-memory route. Faulting every GGUF
/// page before the engine configures its streaming map defeats that contract
/// and can evict unrelated resident models, so the wrapper enforces the safe
/// combination even when a caller leaves its general warmup default enabled.
pub fn warmWeightsEffective(requested: bool, ssd_streaming: bool) bool {
    return requested and !ssd_streaming;
}

pub fn ssdStreamingCacheBytesEffective(
    ssd_streaming: bool,
    requested_experts: u32,
    requested_bytes: u64,
) u64 {
    if (!ssd_streaming or requested_experts != 0 or requested_bytes != 0) {
        return requested_bytes;
    }
    return 8 * 1024 * 1024 * 1024;
}

pub const Ds4Engine = struct {
    allocator: std.mem.Allocator,
    handle: *ffi.Engine,
    model_path_owned: [:0]u8,
    mtp_path_owned: ?[:0]u8,

    pub fn open(allocator: std.mem.Allocator, model_path: []const u8, opts: OpenOptions) Error!*Ds4Engine {
        const path_z = allocator.dupeSentinel(u8, model_path, 0) catch return Error.OutOfMemory;
        errdefer allocator.free(path_z);

        const mtp_z: ?[:0]u8 = if (opts.mtp_path) |p| (allocator.dupeSentinel(u8, p, 0) catch return Error.OutOfMemory) else null;
        errdefer if (mtp_z) |s| allocator.free(s);

        if (opts.dspark and mtp_z == null) {
            log.err("[ds4] DSpark requested but its lineage-matched support GGUF is missing; refusing target-only fallback\n", .{});
            return Error.MissingDsparkSupport;
        }
        try ensureMetalKernels(allocator);
        const dspark_on = opts.dspark;
        if (dspark_on) {
            // The nested engine still validates the candidate's actual tensor
            // content before it can arm DSpark or register support memory.
            log.info("[ds4] DSpark runtime requested (support GGUF candidate: {s})\n", .{opts.mtp_path.?});
        }

        var options = ffi.EngineOptions{
            .model_path = path_z.ptr,
            .mtp_path = if (mtp_z) |s| s.ptr else null,
            .backend = opts.backend,
            .n_threads = opts.n_threads,
            .context_size = opts.context_size,
            .mtp_draft_tokens = opts.mtp_draft_tokens,
            .mtp_margin = opts.mtp_margin,
            .dspark = dspark_on,
            .directional_steering_file = null,
            .directional_steering_attn = 0,
            .directional_steering_ffn = 0,
            .warm_weights = warmWeightsEffective(opts.warm_weights, opts.ssd_streaming),
            .quality = opts.quality,
            .ssd_streaming = opts.ssd_streaming,
            .ssd_streaming_cold = opts.ssd_streaming_cold,
            .ssd_streaming_cache_experts = opts.ssd_streaming_cache_experts,
            .ssd_streaming_cache_bytes = ssdStreamingCacheBytesEffective(
                opts.ssd_streaming,
                opts.ssd_streaming_cache_experts,
                opts.ssd_streaming_cache_bytes,
            ),
            .ssd_streaming_preload_experts = opts.ssd_streaming_preload_experts,
        };

        var raw: ?*ffi.Engine = null;
        if (opts.ssd_streaming) {
            log.info(
                "[ds4] SSD streaming cache budget: {d:.2} GiB ({s})\n",
                .{
                    @as(f64, @floatFromInt(options.ssd_streaming_cache_bytes)) / (1024.0 * 1024.0 * 1024.0),
                    if (opts.ssd_streaming_cache_experts == 0 and opts.ssd_streaming_cache_bytes == 0)
                        "mlx-serve safe default"
                    else
                        "explicit",
                },
            );
        }
        const rc = ffi.ds4_engine_open(&raw, &options);
        if (rc != 0 or raw == null) {
            log.err("[ds4] ds4_engine_open rc={d} model={s}\n", .{ rc, model_path });
            return Error.EngineOpenFailed;
        }

        const wrapper = allocator.create(Ds4Engine) catch {
            ffi.ds4_engine_close(raw);
            return Error.OutOfMemory;
        };
        wrapper.* = .{
            .allocator = allocator,
            .handle = raw.?,
            .model_path_owned = path_z,
            .mtp_path_owned = mtp_z,
        };
        return wrapper;
    }

    pub fn close(self: *Ds4Engine) void {
        ffi.ds4_engine_close(self.handle);
        self.allocator.free(self.model_path_owned);
        if (self.mtp_path_owned) |s| self.allocator.free(s);
        self.allocator.destroy(self);
    }

    pub fn hasMtp(self: *Ds4Engine) bool {
        return ffi.ds4_engine_has_mtp(self.handle);
    }

    pub fn mtpDraftTokens(self: *Ds4Engine) c_int {
        return ffi.ds4_engine_mtp_draft_tokens(self.handle);
    }

    pub fn eosToken(self: *Ds4Engine) i32 {
        return @intCast(ffi.ds4_token_eos(self.handle));
    }

    pub fn assistantToken(self: *Ds4Engine) i32 {
        return @intCast(ffi.ds4_token_assistant(self.handle));
    }

    pub fn userToken(self: *Ds4Engine) i32 {
        return @intCast(ffi.ds4_token_user(self.handle));
    }

    /// Tokenize free-form text (no chat scaffolding). Caller owns the returned
    /// slice and frees with `allocator.free`.
    pub fn tokenizeText(self: *Ds4Engine, allocator: std.mem.Allocator, text: []const u8) Error![]i32 {
        const text_z = allocator.dupeSentinel(u8, text, 0) catch return Error.OutOfMemory;
        defer allocator.free(text_z);

        var tv = ffi.Tokens{};
        ffi.ds4_tokenize_text(self.handle, text_z.ptr, &tv);
        defer ffi.ds4_tokens_free(&tv);

        return tokensToOwnedSlice(allocator, &tv);
    }

    /// Render the engine's built-in chat template for a single (system, user)
    /// turn and return token IDs. Use `encodeChatTranscript` for multi-turn.
    pub fn encodeChatPrompt(
        self: *Ds4Engine,
        allocator: std.mem.Allocator,
        system: ?[]const u8,
        user: []const u8,
        think_mode: ffi.ThinkMode,
    ) Error![]i32 {
        const sys_z: ?[:0]u8 = if (system) |s| (allocator.dupeSentinel(u8, s, 0) catch return Error.OutOfMemory) else null;
        defer if (sys_z) |s| allocator.free(s);
        const user_z = allocator.dupeSentinel(u8, user, 0) catch return Error.OutOfMemory;
        defer allocator.free(user_z);

        var tv = ffi.Tokens{};
        ffi.ds4_encode_chat_prompt(
            self.handle,
            if (sys_z) |s| s.ptr else null,
            user_z.ptr,
            think_mode,
            &tv,
        );
        defer ffi.ds4_tokens_free(&tv);
        return tokensToOwnedSlice(allocator, &tv);
    }

    pub const ChatTurn = struct { role: []const u8, content: []const u8 };

    /// Multi-turn chat encoding. `turns` is the full transcript; the engine
    /// emits BOS once and the trailing assistant prefix for the next reply.
    pub fn encodeChatTranscript(
        self: *Ds4Engine,
        allocator: std.mem.Allocator,
        system: ?[]const u8,
        turns: []const ChatTurn,
        think_mode: ffi.ThinkMode,
    ) Error![]i32 {
        var tv = ffi.Tokens{};
        defer ffi.ds4_tokens_free(&tv);

        ffi.ds4_chat_begin(self.handle, &tv);

        if (system) |s| {
            const z = allocator.dupeSentinel(u8, s, 0) catch return Error.OutOfMemory;
            defer allocator.free(z);
            const role_z: [*:0]const u8 = "system";
            ffi.ds4_chat_append_message(self.handle, &tv, role_z, z.ptr);
        }

        for (turns) |t| {
            const role_z = allocator.dupeSentinel(u8, t.role, 0) catch return Error.OutOfMemory;
            defer allocator.free(role_z);
            const content_z = allocator.dupeSentinel(u8, t.content, 0) catch return Error.OutOfMemory;
            defer allocator.free(content_z);
            ffi.ds4_chat_append_message(self.handle, &tv, role_z.ptr, content_z.ptr);
        }

        ffi.ds4_chat_append_assistant_prefix(self.handle, &tv, think_mode);
        return tokensToOwnedSlice(allocator, &tv);
    }

    /// Single token → bytes lookup. Caller owns the returned buffer.
    /// Returned bytes are NOT NUL-terminated; the slice length is authoritative.
    pub fn detokenizeOne(self: *Ds4Engine, allocator: std.mem.Allocator, token_id: i32) Error![]u8 {
        var len_out: usize = 0;
        const raw = ffi.ds4_token_text(self.handle, @intCast(token_id), &len_out);
        if (raw == null or len_out == 0) {
            return allocator.dupe(u8, "") catch Error.OutOfMemory;
        }
        const src = raw.?[0..len_out];
        return allocator.dupe(u8, src) catch Error.OutOfMemory;
    }

    pub fn createSession(self: *Ds4Engine, ctx_size: i32) Error!*Ds4Session {
        var raw: ?*ffi.Session = null;
        const rc = ffi.ds4_session_create(&raw, self.handle, @intCast(ctx_size));
        if (rc != 0 or raw == null) {
            log.err("[ds4] ds4_session_create rc={d} ctx={d}\n", .{ rc, ctx_size });
            return Error.SessionCreateFailed;
        }
        const sess = self.allocator.create(Ds4Session) catch {
            ffi.ds4_session_free(raw);
            return Error.OutOfMemory;
        };
        sess.* = .{
            .allocator = self.allocator,
            .engine = self,
            .handle = raw.?,
            .ctx_size = ctx_size,
        };
        return sess;
    }
};

fn tokensToOwnedSlice(allocator: std.mem.Allocator, tv: *const ffi.Tokens) Error![]i32 {
    if (tv.len <= 0 or tv.v == null) return allocator.alloc(i32, 0) catch Error.OutOfMemory;
    const n: usize = @intCast(tv.len);
    const buf = allocator.alloc(i32, n) catch return Error.OutOfMemory;
    const src = tv.v.?[0..n];
    for (src, 0..) |raw, i| buf[i] = @intCast(raw);
    return buf;
}

pub const Ds4Snapshot = struct {
    inner: ffi.SessionSnapshot,

    pub fn free(self: *Ds4Snapshot) void {
        ffi.ds4_session_snapshot_free(&self.inner);
        self.inner = .{};
    }
};

pub const Ds4Session = struct {
    allocator: std.mem.Allocator,
    engine: *Ds4Engine,
    handle: *ffi.Session,
    ctx_size: i32,

    pub fn free(self: *Ds4Session) void {
        ffi.ds4_session_free(self.handle);
        self.allocator.destroy(self);
    }

    pub fn pos(self: *Ds4Session) i32 {
        return @intCast(ffi.ds4_session_pos(self.handle));
    }

    pub fn rewind(self: *Ds4Session, to_pos: i32) void {
        ffi.ds4_session_rewind(self.handle, @intCast(to_pos));
    }

    pub fn invalidate(self: *Ds4Session) void {
        ffi.ds4_session_invalidate(self.handle);
    }

    /// Sync the live session to a full prompt prefix. Reuses common prefix
    /// against the live KV cache; otherwise rebuilds from scratch.
    pub fn sync(self: *Ds4Session, prompt_ids: []const i32) Error!void {
        var holder = try TokenHolder.init(self.allocator, prompt_ids);
        defer holder.deinit();
        var err_buf: [256]u8 = undefined;
        const rc = ffi.ds4_session_sync(self.handle, &holder.tv, &err_buf, err_buf.len);
        if (rc != 0) {
            log.err("[ds4] session_sync rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
            return Error.SessionSyncFailed;
        }
    }

    pub fn commonPrefix(self: *Ds4Session, prompt_ids: []const i32) Error!i32 {
        var holder = try TokenHolder.init(self.allocator, prompt_ids);
        defer holder.deinit();
        return @intCast(ffi.ds4_session_common_prefix(self.handle, &holder.tv));
    }

    /// Evaluate one new token. Caller has already synced to the prefix.
    pub fn eval(self: *Ds4Session, token: i32) Error!void {
        var err_buf: [256]u8 = undefined;
        const rc = ffi.ds4_session_eval(self.handle, @intCast(token), &err_buf, err_buf.len);
        if (rc != 0) {
            log.err("[ds4] session_eval rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
            return Error.SessionEvalFailed;
        }
    }

    pub fn argmax(self: *Ds4Session) i32 {
        return @intCast(ffi.ds4_session_argmax(self.handle));
    }

    pub fn argmaxExcluding(self: *Ds4Session, excluded_id: i32) i32 {
        return @intCast(ffi.ds4_session_argmax_excluding(self.handle, @intCast(excluded_id)));
    }

    pub fn sample(self: *Ds4Session, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: *u64) i32 {
        return @intCast(ffi.ds4_session_sample(self.handle, temperature, @intCast(top_k), top_p, min_p, rng));
    }

    /// MTP-driven speculative decode. Returns total tokens emitted into
    /// `out_tokens` (always >= 1: the verified first_token plus any accepted
    /// draft tokens). Stops at EOS or when max_tokens hit.
    pub fn evalSpeculative(
        self: *Ds4Session,
        first_token: i32,
        max_tokens: i32,
        eos_token: i32,
        out_tokens: []i32,
    ) Error!i32 {
        var c_out_buf = self.allocator.alloc(c_int, out_tokens.len) catch return Error.OutOfMemory;
        defer self.allocator.free(c_out_buf);

        var err_buf: [256]u8 = undefined;
        const n = ffi.ds4_session_eval_speculative_argmax(
            self.handle,
            @intCast(first_token),
            @intCast(max_tokens),
            @intCast(eos_token),
            c_out_buf.ptr,
            @intCast(c_out_buf.len),
            &err_buf,
            err_buf.len,
        );
        if (n < 0) {
            log.err("[ds4] eval_speculative rc={d} err={s}\n", .{ n, std.mem.sliceTo(&err_buf, 0) });
            return Error.SessionSpecFailed;
        }
        const n_usize: usize = @intCast(n);
        for (c_out_buf[0..n_usize], 0..) |v, i| out_tokens[i] = @intCast(v);
        return @intCast(n);
    }

    pub fn saveSnapshot(self: *Ds4Session) Error!Ds4Snapshot {
        var snap: ffi.SessionSnapshot = .{};
        var err_buf: [256]u8 = undefined;
        const rc = ffi.ds4_session_save_snapshot(self.handle, &snap, &err_buf, err_buf.len);
        if (rc != 0) {
            log.err("[ds4] save_snapshot rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
            return Error.SnapshotFailed;
        }
        return .{ .inner = snap };
    }

    pub fn loadSnapshot(self: *Ds4Session, snap: *const Ds4Snapshot) Error!void {
        var err_buf: [256]u8 = undefined;
        const rc = ffi.ds4_session_load_snapshot(self.handle, &snap.inner, &err_buf, err_buf.len);
        if (rc != 0) {
            log.err("[ds4] load_snapshot rc={d} err={s}\n", .{ rc, std.mem.sliceTo(&err_buf, 0) });
            return Error.LoadSnapshotFailed;
        }
    }
};

/// Stack-allocated RAII handle that materializes a `[]c_int` view of an i32
/// prompt slice and exposes the ds4_tokens C view. `deinit` frees the backing
/// buffer regardless of whether the FFI call succeeded.
const TokenHolder = struct {
    allocator: std.mem.Allocator,
    buf: []c_int,
    tv: ffi.Tokens,

    fn init(allocator: std.mem.Allocator, prompt_ids: []const i32) Error!TokenHolder {
        const buf = allocator.alloc(c_int, prompt_ids.len) catch return Error.OutOfMemory;
        for (prompt_ids, 0..) |t, i| buf[i] = @intCast(t);
        return .{
            .allocator = allocator,
            .buf = buf,
            .tv = .{
                .v = if (buf.len == 0) null else buf.ptr,
                .len = @intCast(buf.len),
                .cap = @intCast(buf.len),
            },
        };
    }

    fn deinit(self: *TokenHolder) void {
        self.allocator.free(self.buf);
    }
};

test "ds4: supportSidecarPolicy gives offline and scheduler the same fail-closed DSpark contract" {
    const normal = supportSidecarPolicy(true, false, false);
    try std.testing.expect(normal.lookup);
    try std.testing.expect(!normal.require_dspark);
    try std.testing.expect(!normal.dspark);

    const requested_dspark = supportSidecarPolicy(true, false, true);
    try std.testing.expect(requested_dspark.lookup);
    try std.testing.expect(requested_dspark.require_dspark);
    try std.testing.expect(requested_dspark.dspark);

    const ssd_default = supportSidecarPolicy(true, true, false);
    try std.testing.expect(!ssd_default.lookup);
    try std.testing.expect(!ssd_default.require_dspark);
    try std.testing.expect(!ssd_default.dspark);

    const disabled = supportSidecarPolicy(false, true, true);
    try std.testing.expect(!disabled.lookup);
    try std.testing.expect(!disabled.require_dspark);
    try std.testing.expect(!disabled.dspark);
}

test "ds4: open rejects missing explicit DSpark support before Metal staging" {
    try std.testing.expectError(
        Error.MissingDsparkSupport,
        Ds4Engine.open(std.testing.allocator, "unused-target.gguf", .{
            .dspark = true,
            .mtp_path = null,
        }),
    );
}

test "warmWeightsEffective: SSD streaming never faults the complete GGUF" {
    try std.testing.expect(warmWeightsEffective(true, false));
    try std.testing.expect(!warmWeightsEffective(true, true));
    try std.testing.expect(!warmWeightsEffective(false, false));
    try std.testing.expect(!warmWeightsEffective(false, true));
}

test "ssdStreamingCacheBytesEffective: wrapper default is bounded and explicit budgets win" {
    const safe_default: u64 = 8 * 1024 * 1024 * 1024;
    try std.testing.expectEqual(@as(u64, 0), ssdStreamingCacheBytesEffective(false, 0, 0));
    try std.testing.expectEqual(safe_default, ssdStreamingCacheBytesEffective(true, 0, 0));
    try std.testing.expectEqual(@as(u64, 0), ssdStreamingCacheBytesEffective(true, 701, 0));
    try std.testing.expectEqual(@as(u64, 12 * 1024 * 1024 * 1024), ssdStreamingCacheBytesEffective(true, 0, 12 * 1024 * 1024 * 1024));
}

test "clampSessionCtx: unset (0) → ds4 default" {
    try std.testing.expectEqual(ds4_default_ctx, clampSessionCtx(0));
}

test "clampSessionCtx: below prefill-chunk floor is raised (no junk-output regime)" {
    // ds4's prefill scratch is sized against the session ctx; a session smaller
    // than prefill_chunk produces junk. A user --ctx-size under the floor must
    // be raised, never honored verbatim.
    try std.testing.expectEqual(ds4_prefill_chunk, clampSessionCtx(1));
    try std.testing.expectEqual(ds4_prefill_chunk, clampSessionCtx(ds4_prefill_chunk - 1));
}

test "clampSessionCtx: at/above the floor is honored verbatim" {
    try std.testing.expectEqual(ds4_prefill_chunk, clampSessionCtx(ds4_prefill_chunk));
    try std.testing.expectEqual(@as(u32, 8192), clampSessionCtx(8192));
    try std.testing.expectEqual(@as(u32, 131072), clampSessionCtx(131072));
}

test "clampSessionCtx: idempotent (re-clamping a clamped value is a no-op except 0)" {
    try std.testing.expectEqual(clampSessionCtx(8192), clampSessionCtx(clampSessionCtx(8192)));
    try std.testing.expectEqual(ds4_default_ctx, clampSessionCtx(clampSessionCtx(0)));
}

test "kernel hash is stable across calls" {
    const a = computeKernelHash();
    const b = computeKernelHash();
    try std.testing.expectEqualSlices(u8, &a, &b);
}

test "embedded Metal source inventory matches upstream loader" {
    // The Objective-C loader table is the upstream authority. If DwarfStar
    // adds an entry, this fails before mlx-serve can silently ship an older
    // embedded-kernel closure.
    try std.testing.expectEqual(
        kernel_entries.len,
        std.mem.count(u8, metal_sources.upstream_loader, "@[@\"DS4_METAL_"),
    );

    for (kernel_entries) |entry| {
        try std.testing.expect(std.mem.indexOf(u8, metal_sources.upstream_loader, entry.env_var) != null);
        try std.testing.expect(std.mem.indexOf(u8, metal_sources.upstream_loader, entry.file_name) != null);
    }
}
