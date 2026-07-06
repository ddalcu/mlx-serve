//! iOS stub for `arch/llama.zig`. The embedded llama.cpp engine (libllama) is
//! a macOS-only prebuilt dylib; iOS serves MLX safetensors models only, so
//! every llama code path is dead at runtime (`LoadedModel.llama_engine` is
//! always null). These types exist solely so the shared scheduler/server/chat
//! dispatch type-checks. Bodies that could be reached return defaults; the rest
//! panic. Signatures mirror `arch/llama.zig` exactly. Selected via
//! `build_options.ios`.

const std = @import("std");

const unavailable = "llama.cpp engine is unavailable on the iOS build (MLX safetensors only)";

pub const Error = error{
    EngineOpenFailed,
    SessionCreateFailed,
    TokenizeFailed,
    DetokenizeFailed,
    OutOfMemory,
};

pub const OpenOptions = struct {
    n_gpu_layers: i32 = 999,
};

pub const ChatTurn = struct { role: []const u8, content: []const u8 };

pub const LlamaKvQuant = enum(u8) {
    off,
    q8,
    q4,

    pub fn fromString(s: []const u8) ?LlamaKvQuant {
        if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "f16") or std.mem.eql(u8, s, "F16")) return .off;
        if (std.mem.eql(u8, s, "8") or std.mem.eql(u8, s, "q8") or std.mem.eql(u8, s, "Q8_0") or std.mem.eql(u8, s, "q8_0")) return .q8;
        if (std.mem.eql(u8, s, "4") or std.mem.eql(u8, s, "q4") or std.mem.eql(u8, s, "Q4_0") or std.mem.eql(u8, s, "q4_0")) return .q4;
        return null;
    }
    pub fn ggmlType(self: LlamaKvQuant) i32 {
        // Mirrors the real enum's mapping without the ffi.GgmlType dependency.
        return switch (self) {
            .off => 0,
            .q8 => 8, // GGML_TYPE_Q8_0
            .q4 => 2, // GGML_TYPE_Q4_0
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

pub fn commonPrefixLen(a: []const i32, b: []const i32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

pub const LlamaEngine = struct {
    pub fn open(allocator: std.mem.Allocator, model_path: []const u8, opts: OpenOptions) Error!*LlamaEngine {
        _ = allocator;
        _ = model_path;
        _ = opts;
        return Error.EngineOpenFailed;
    }
    pub fn close(self: *LlamaEngine) void {
        _ = self;
        @panic(unavailable);
    }
    pub fn eosToken(self: *LlamaEngine) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn isEog(self: *LlamaEngine, token: i32) bool {
        _ = self;
        _ = token;
        @panic(unavailable);
    }
    pub fn nVocab(self: *LlamaEngine) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn tokenizeText(
        self: *LlamaEngine,
        allocator: std.mem.Allocator,
        text: []const u8,
        add_special: bool,
    ) Error![]i32 {
        _ = self;
        _ = allocator;
        _ = text;
        _ = add_special;
        @panic(unavailable);
    }
    pub fn detokenizeOne(self: *LlamaEngine, allocator: std.mem.Allocator, token_id: i32) Error![]u8 {
        _ = self;
        _ = allocator;
        _ = token_id;
        @panic(unavailable);
    }
    pub fn chatTemplate(self: *LlamaEngine) ?[]const u8 {
        _ = self;
        @panic(unavailable);
    }
    pub fn applyChatTemplate(
        self: *LlamaEngine,
        allocator: std.mem.Allocator,
        turns: []const ChatTurn,
        add_assistant: bool,
    ) Error![]u8 {
        _ = self;
        _ = allocator;
        _ = turns;
        _ = add_assistant;
        @panic(unavailable);
    }
    pub fn createSession(self: *LlamaEngine, ctx_size: i32) Error!*LlamaSession {
        _ = self;
        _ = ctx_size;
        @panic(unavailable);
    }
    pub fn createSessionWithKvQuant(self: *LlamaEngine, ctx_size: i32, type_k: i32, type_v: i32) Error!*LlamaSession {
        _ = self;
        _ = ctx_size;
        _ = type_k;
        _ = type_v;
        @panic(unavailable);
    }
};

pub const LlamaSession = struct {
    allocator: std.mem.Allocator,
    resident: std.ArrayList(i32),

    pub fn free(self: *LlamaSession) void {
        _ = self;
        @panic(unavailable);
    }
    pub fn pos(self: *LlamaSession) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn sync(self: *LlamaSession, prompt_ids: []const i32) Error!i32 {
        _ = self;
        _ = prompt_ids;
        @panic(unavailable);
    }
    pub fn reset(self: *LlamaSession) void {
        _ = self;
        @panic(unavailable);
    }
    pub fn syncWithFallback(self: *LlamaSession, prompt_ids: []const i32) Error!i32 {
        _ = self;
        _ = prompt_ids;
        @panic(unavailable);
    }
    pub fn eval(self: *LlamaSession, token: i32) Error!void {
        _ = self;
        _ = token;
        @panic(unavailable);
    }
    pub fn argmax(self: *LlamaSession) i32 {
        _ = self;
        @panic(unavailable);
    }
    pub fn sample(self: *LlamaSession, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: *u64) i32 {
        _ = self;
        _ = temperature;
        _ = top_k;
        _ = top_p;
        _ = min_p;
        _ = rng;
        @panic(unavailable);
    }
};
