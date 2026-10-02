//! transformer surface for builds without MLX.

const std = @import("std");

pub const Error = error{MlxUnavailable};
pub fn decodeAttnQuantEnabled() bool {
    return false;
}
pub fn ssmFreeQsaState(_: *SSMCacheEntry) void {}

pub const Transformer = struct {
    pub fn qwen4MtpResetOwned(_: *Transformer, _: bool) void {}
    pub fn markQsaPooledRopeStale(_: *Transformer) void {}
    pub fn resetQsaPooledRope(_: *Transformer) void {}
    pub fn ssmGroupDrop(_: *Transformer, _: []SSMCacheEntry) void {}
    pub const PREFILL_EVAL_CADENCE_DEFAULT: u32 = 4;
    pub var mtp_head_kv_quant_flag: bool = false;
    pub const MOE_EVAL_EVERY_N_LAYERS: usize = 4;

    pub const module_owned_state_fields = [_][]const u8{};

    dense_attn_proj: bool = false,
    s: @import("mlx_stub.zig").mlx_stream = .{},
    round_cost: @import("round_cost.zig").Table = .{},
    round_cost_key_buf: [64]u8 = undefined,
    round_cost_key_len: u8 = 0,
    mtp_depth_free: u32 = 0,
    cache: KVCache = .{},
    config: @import("model.zig").ModelConfig = .{},
    ssm_entries: []SSMCacheEntry = &.{},
    vision_embeddings: ?@import("mlx_stub.zig").mlx_array = null,
    moe_layers: ?[]const u8 = null,
    dsv: ?*anyopaque = null,
    dsv4: ?*Dsv4State = null,
    ane_prefill: ?*@import("ane_stub.zig").AneEngine = null,

    pub fn init(_: std.Io, _: std.mem.Allocator, _: anytype, _: anytype) anyerror!Transformer {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *Transformer) void {}

    pub fn nativeMoeMtpHeadMeasured(_: anytype) bool {
        return false;
    }

    pub fn ownsModuleDecodeState(_: *const Transformer) bool {
        return false;
    }

    pub fn moduleStateSpecRollback(_: *const Transformer) bool {
        return false;
    }

    pub fn resetCache(_: *Transformer) anyerror!void {
        return Error.MlxUnavailable;
    }
    pub fn warmup(_: *Transformer) anyerror!void {
        return Error.MlxUnavailable;
    }
    pub fn defaultCtx(_: *Transformer) ForwardCtx {
        return .{};
    }

    pub fn compileForward(_: *Transformer) void {}
    pub fn compileGdnGate(_: *Transformer) void {}
    pub fn compileGeglu(_: *Transformer) void {}
    pub fn compileGelu(_: *Transformer) void {}
    pub fn compileMoeRouting(_: *Transformer) void {}
    pub fn compileSoftcap(_: *Transformer) void {}
    pub fn diagProjBench(_: *Transformer, _: usize, _: anytype) void {}
    pub fn buildAnePrefill(_: *Transformer, _: anytype, _: anytype, _: anytype, _: anytype) void {}

    pub fn supportsBatchedGdnDecode(_: *const Transformer) bool {
        return false;
    }
    pub fn batchedGdnReady(_: *const Transformer, _: anytype) bool {
        return false;
    }

    pub fn forwardWith(_: *Transformer, _: anytype, _: anytype) anyerror!@import("mlx_stub.zig").mlx_array {
        return Error.MlxUnavailable;
    }
    pub fn forwardBatchedDecode(_: *Transformer, _: anytype, _: anytype, _: anytype) anyerror![]@import("mlx_stub.zig").mlx_array {
        return Error.MlxUnavailable;
    }
    pub fn forwardMoeBatchedDecode(_: *Transformer, _: anytype, _: anytype, _: anytype) anyerror![]@import("mlx_stub.zig").mlx_array {
        return Error.MlxUnavailable;
    }
};

pub const KVCache = struct {
    pub fn residentBytes(_: *const KVCache) u64 {
        return 0;
    }
    pub fn truncate(_: *KVCache, _: usize, _: anytype) !void {
        return error.MlxUnavailable;
    }
    pub fn initWithConfig(a: std.mem.Allocator, layers: u32, cfg: KVQuantConfig) !KVCache {
        return initWithConfigAndHeadDim(a, layers, cfg, 0);
    }
    pub fn reservedTokens(_: u64, _: u64, _: u64, _: u64) u64 {
        return 0;
    }
    pub const RESERVE_GEN_HEADROOM: u64 = 8192;
    config: KVQuantConfig = .dense,
    step: usize = 0,

    pub fn deinit(_: *KVCache) void {}

    pub fn reinit(self: *KVCache, layers: anytype, _: anytype, _: anytype) anyerror!void {
        if (@as(usize, @intCast(layers)) != 0) return Error.MlxUnavailable;
        self.* = .{};
    }

    pub fn initWithConfigAndHeadDim(_: std.mem.Allocator, layers: anytype, _: anytype, _: anytype) anyerror!KVCache {
        if (@as(usize, @intCast(layers)) != 0) return Error.MlxUnavailable;
        return .{};
    }
};
pub const KVQuantScheme = @import("kv_quant_config.zig").Scheme;
pub const KVQuantConfig = @import("kv_quant_config.zig").KVQuantConfig;
pub const ForwardCtx = struct {
    ssm_member_gen: u64 = 0,
    cache: ?*KVCache = null,
    ssm_entries: ?[]SSMCacheEntry = null,
    vision_embeddings: ?@import("mlx_stub.zig").mlx_array = null,
    capture_hidden: ?*@import("mlx_stub.zig").mlx_array = null,
    skip_lm_head: bool = false,
    kv_attn_fused: bool = false,
    moe_seq_offset: ?*usize = null,
    mrope_pos: ?[]const i32 = null,
    mrope_delta: i32 = 0,
    mrope_total: usize = 0,
    decode_ns: u64 = 0,
};
pub const SSMCacheEntry = struct {
    conv_state: @import("mlx_stub.zig").mlx_array = .{},
    ssm_state: @import("mlx_stub.zig").mlx_array = .{},
    aux_state: @import("mlx_stub.zig").mlx_array = .{},
    initialized: bool = false,
};
pub const SSMCheckpoint = struct {
    pub fn deinit(_: *SSMCheckpoint, _: std.mem.Allocator) void {}
};

pub const Dsv4State = struct {
    n_mtp: u32 = 0,
};

pub const PREFILL_DQ_GEMM_MIN_M: usize = 0;

pub var decode_attn_quant_flag: bool = false;
pub var fused256_override: ?bool = null;
pub var prefill_dq_gemm_override: ?bool = null;

pub fn prefillDqGemmEnabled() bool {
    return false;
}

pub fn prefillHeadDimFused(_: u32) bool {
    return false;
}

pub fn verifyQmmNaxAvailable() bool {
    return false;
}

pub fn naxStatus() []const u8 {
    return "unavailable (built without MLX)";
}

const testing = std.testing;

test "KVCache init returns a zero-layer SHELL, it does not refuse" {
    var cache = try KVCache.initWithConfigAndHeadDim(testing.allocator, 0, KVQuantConfig.dense, 128);
    defer cache.deinit();
    try testing.expectEqual(@as(usize, 0), cache.step);
}

pub fn macosProductVersion(_: []u8) ?[]const u8 {
    return null;
}

pub fn warmQsaEnvCaches() void {}

pub const QSA_RING_ROWS: c_int = 32;
pub const FUSED256_MIN_Q_LEN: c_int = 16;
pub fn prefillDqGemmMinRows(_: u32) usize {
    return 2048;
}
pub fn qsaGatherEnabled() bool {
    return false;
}

pub fn qsaPrefillTransientBytes(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype) u64 {
    return 0;
}

pub fn nextSsmMemberGen() u64 {
    return 0;
}
pub fn ssmEntryBytes(_: *const SSMCacheEntry) u64 {
    return 0;
}
