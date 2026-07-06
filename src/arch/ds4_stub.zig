//! iOS stub for `arch/ds4.zig`. See `ds4_ffi_stub.zig` for the rationale: the
//! ds4 engine is macOS-only, so on iOS these types exist only to satisfy the
//! type-checker for the shared scheduler/server/chat dispatch code. No method
//! is ever reached at runtime (iOS never loads a ds4/GGUF model), so the bodies
//! panic. Signatures mirror `arch/ds4.zig` exactly.

const std = @import("std");
const ffi = @import("../ds4_ffi_stub.zig");

const unavailable = "ds4 engine is unavailable on the iOS build (MLX safetensors only)";

pub const Error = error{
    EngineOpenFailed,
    SessionCreateFailed,
    SessionSyncFailed,
    SessionEvalFailed,
    SessionSpecFailed,
    SnapshotFailed,
    LoadSnapshotFailed,
    KernelExtractionFailed,
    OutOfMemory,
};

pub const ds4_prefill_chunk: u32 = 2048;
pub const ds4_default_ctx: u32 = 32768;

/// Pure helper mirrored from the real arch/ds4.zig (no engine dependency) so
/// shared scheduler code that clamps a requested ctx size analyzes cleanly.
pub fn clampSessionCtx(requested: u32) u32 {
    if (requested == 0) return ds4_default_ctx;
    return @max(requested, ds4_prefill_chunk);
}

pub const OpenOptions = struct {
    backend: ffi.Backend = .metal,
    n_threads: c_int = 0,
    warm_weights: bool = true,
    quality: bool = false,
    mtp_path: ?[:0]const u8 = null,
    mtp_draft_tokens: c_int = 0,
    mtp_margin: f32 = 0,
    ssd_streaming: bool = false,
    ssd_streaming_cold: bool = false,
    ssd_streaming_cache_experts: u32 = 0,
    ssd_streaming_cache_bytes: u64 = 0,
    ssd_streaming_preload_experts: u32 = 0,
};

pub const Ds4Snapshot = struct {
    pub fn free(self: *Ds4Snapshot) void {
        _ = self;
        @panic(unavailable);
    }
};

pub const Ds4Engine = struct {
    pub const ChatTurn = struct { role: []const u8, content: []const u8 };

    pub fn open(allocator: std.mem.Allocator, model_path: []const u8, opts: OpenOptions) Error!*Ds4Engine {
        _ = allocator;
        _ = model_path;
        _ = opts;
        return Error.EngineOpenFailed;
    }
    pub fn close(self: *Ds4Engine) void {
        _ = self;
        @panic(unavailable);
    }
    pub fn hasMtp(self: *Ds4Engine) bool {
        _ = self;
        @panic(unavailable);
    }
    pub fn mtpDraftTokens(self: *Ds4Engine) c_int {
        _ = self;
        @panic(unavailable);
    }
    pub fn eosToken(self: *Ds4Engine) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn assistantToken(self: *Ds4Engine) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn userToken(self: *Ds4Engine) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn tokenizeText(self: *Ds4Engine, allocator: std.mem.Allocator, text: []const u8) Error![]i32 {
        _ = self;
        _ = allocator;
        _ = text;
        @panic(unavailable);
    }
    pub fn encodeChatPrompt(
        self: *Ds4Engine,
        allocator: std.mem.Allocator,
        system: ?[]const u8,
        user: []const u8,
        think_mode: ffi.ThinkMode,
    ) Error![]i32 {
        _ = self;
        _ = allocator;
        _ = system;
        _ = user;
        _ = think_mode;
        @panic(unavailable);
    }
    pub fn encodeChatTranscript(
        self: *Ds4Engine,
        allocator: std.mem.Allocator,
        system: ?[]const u8,
        turns: []const ChatTurn,
        think_mode: ffi.ThinkMode,
    ) Error![]i32 {
        _ = self;
        _ = allocator;
        _ = system;
        _ = turns;
        _ = think_mode;
        @panic(unavailable);
    }
    pub fn detokenizeOne(self: *Ds4Engine, allocator: std.mem.Allocator, token_id: i32) Error![]u8 {
        _ = self;
        _ = allocator;
        _ = token_id;
        @panic(unavailable);
    }
    pub fn createSession(self: *Ds4Engine, ctx_size: i32) Error!*Ds4Session {
        _ = self;
        _ = ctx_size;
        @panic(unavailable);
    }
};

pub const Ds4Session = struct {
    pub fn free(self: *Ds4Session) void {
        _ = self;
        @panic(unavailable);
    }
    pub fn pos(self: *Ds4Session) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn rewind(self: *Ds4Session, to_pos: i32) void {
        _ = self;
        _ = to_pos;
        @panic(unavailable);
    }
    pub fn invalidate(self: *Ds4Session) void {
        _ = self;
        @panic(unavailable);
    }
    pub fn sync(self: *Ds4Session, prompt_ids: []const i32) Error!void {
        _ = self;
        _ = prompt_ids;
        @panic(unavailable);
    }
    pub fn commonPrefix(self: *Ds4Session, prompt_ids: []const i32) Error!i32 {
        _ = self;
        _ = prompt_ids;
        @panic(unavailable);
    }
    pub fn eval(self: *Ds4Session, token: i32) Error!void {
        _ = self;
        _ = token;
        @panic(unavailable);
    }
    pub fn argmax(self: *Ds4Session) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn argmaxExcluding(self: *Ds4Session, excluded_id: i32) i32 {
        _ = self;
        _ = excluded_id;
        @panic(unavailable);
    }
    pub fn sample(self: *Ds4Session, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: *u64) i32 {
        _ = self;
        _ = temperature;
        _ = top_k;
        _ = top_p;
        _ = min_p;
        _ = rng;
        @panic(unavailable);
    }
    pub fn evalSpeculative(
        self: *Ds4Session,
        first_token: i32,
        max_tokens: i32,
        eos_token: i32,
        out_tokens: []i32,
    ) Error!i32 {
        _ = self;
        _ = first_token;
        _ = max_tokens;
        _ = eos_token;
        _ = out_tokens;
        @panic(unavailable);
    }
    pub fn saveSnapshot(self: *Ds4Session) Error!Ds4Snapshot {
        _ = self;
        @panic(unavailable);
    }
    pub fn loadSnapshot(self: *Ds4Session, snap: *const Ds4Snapshot) Error!void {
        _ = self;
        _ = snap;
        @panic(unavailable);
    }
};
