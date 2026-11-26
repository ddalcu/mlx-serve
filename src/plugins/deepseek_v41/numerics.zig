//! DeepSeek-V4.1's levers that change the prefill or decode numerics, resolved as its kernels read them. Their hash
//! is the plugin's contribution to the peer settings key (`AuxHooks.settings_key`).

const std = @import("std");
const env = @import("../../env.zig");
const dsv41 = @import("deepseek_v41.zig");
const flash_attn = @import("prompt/flash_attn.zig");

pub fn attnF16() bool {
    return if (env.get(.MLX_SERVE_DSV41_ATTN_F16)) |v| v[0] != '0' else true;
}

pub fn ced() bool {
    return if (env.get(.MLX_SERVE_DSV41_CED)) |v| v[0] != '0' else true;
}

/// Effective tensor-parallel collective sum dtype for a load (`none` on a single box). Matches `dsv41.tpSum`:
/// `bf16` sums all parts in activation dtype; `f32` keeps wider parts in f32 (`auto` uses `f32` on a pair and `bf16` past a pair).
pub const TpSum = enum(u8) { none, bf16, f32 };

pub fn tpSumFor(tp_size: usize, mode: dsv41.TpSumMode) TpSum {
    if (tp_size <= 1) return .none;
    return switch (mode) {
        .bf16 => .bf16,
        .f32 => .f32,
        .auto => if (tp_size > 2) .bf16 else .f32,
    };
}

/// Flash attention and tensor-parallel sum mode depend on the load (pack / group size / rank 0's plan), so their
/// entries are the modes this load chose rather than raw process env text.
pub const Resolved = struct { attn_f16: bool, ced: bool, flash_attn: flash_attn.Mode, tp_sum: TpSum };

pub fn resolved(flash: flash_attn.Mode, tp_sum: TpSum) Resolved {
    return .{ .attn_f16 = attnF16(), .ced = ced(), .flash_attn = flash, .tp_sum = tp_sum };
}

/// `flash` and `tp_sum` are the modes the model was loaded with: the process-wide settings belong to whichever load ran last.
pub fn key(flash: flash_attn.Mode, tp_sum: TpSum) u64 {
    var h = std.hash.Wyhash.init(0x6473_7634_315f_6e75);
    std.hash.autoHash(&h, resolved(flash, tp_sum));
    return h.final();
}

/// What a `kill` lever does to the KV that DeepSeek-V4.1 computes for given token ids.
pub const KvEffect = enum {
    /// Changes the bytes: hashed by `key`.
    key,
    /// The same bytes by another route: a launch layout, a read order, a lossless encoding, which token is drawn.
    same,
    /// Read only on another model's path.
    other_model,
};

/// Every `kill` lever, classified; a new one fails the test below until it is. A `diag` lever never changes what is
/// served (`env.Kind`), except the three the engine hashes itself (`LAYER_CAP`, `DSV4_SKIP_MOE`,
/// `FORCE_GPU_FAMILY_FALLBACK`); of the `public` ones only `MLX_SERVE_KERNEL_DIR` does, and it refuses transfers.
pub const kill_levers = [_]struct { []const u8, KvEffect }{
    .{ "DSV41_TP_SUM", .key },
    .{ "MLX_SERVE_DSV41_ATTN_F16", .key },
    .{ "MLX_SERVE_DSV41_CED", .key },
    .{ "MLX_SERVE_DSV41_FLASH_ATTN", .key },
    .{ "MLX_SERVE_DSV41_HEAD8", .same },
    .{ "MLX_SERVE_DSV41_EXACT", .same },
    .{ "MLX_SERVE_DSV41_ENGRAM_LRU_MB", .same },
    .{ "MLX_SERVE_KEYED_SAMPLING", .same },
    .{ "MLX_SERVE_NATIVE_GROUPS", .same },
    .{ "MLX_SERVE_PREFIX_CACHE_PRESSURE", .same },
    .{ "MLX_SERVE_F16_PREFILL", .other_model },
    .{ "MLX_SERVE_KV_ATTN_MMA", .other_model },
    .{ "MLX_SERVE_GDN_STATE_F32", .other_model },
    .{ "MLX_SERVE_CUDA_QMV_TC", .other_model },
    .{ "MINIMAX_H3_GLUE_FUSED", .other_model },
    .{ "MINIMAX_H3_VSA", .other_model },
    .{ "MLX_SERVE_H3_RESIDENT", .other_model },
    .{ "MLX_SERVE_MING_FUSED", .other_model },
    .{ "MLX_SERVE_INKLING2_FUSED", .other_model },
    .{ "MLX_SERVE_INKLING2_TREE", .other_model },
};

fn classified(name: []const u8) bool {
    for (kill_levers) |c| if (std.mem.eql(u8, c[0], name)) return true;
    return false;
}

const testing = std.testing;

test "dsv41 numerics: every kill lever is classified once, and the table holds only kill levers" {
    var kills: usize = 0;
    inline for (@typeInfo(env.vars).@"struct".decl_names) |name| {
        const v = @field(env.vars, name);
        if (@TypeOf(v) == env.Var and v.kind == .kill) {
            kills += 1;
            try testing.expect(classified(name));
        }
    }
    try testing.expectEqual(kills, kill_levers.len);
    inline for (kill_levers) |c| {
        try testing.expect(@hasDecl(env.vars, c[0]));
        try testing.expectEqual(env.Kind.kill, @field(env.vars, c[0]).kind);
    }
}

test "dsv41 numerics: the key follows each lever and the load modes it is given, not raw flash or tp_sum text" {
    const unset_flash = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, null);
    defer unset_flash.restore();
    const unset_tp = env.override(.DSV41_TP_SUM, null);
    defer unset_tp.restore();
    const base = key(.off, .none);

    const flash_text = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, "1");
    try testing.expectEqual(base, key(.off, .none));
    flash_text.restore();

    const tp_text = env.override(.DSV41_TP_SUM, "bf16");
    try testing.expectEqual(base, key(.off, .none));
    tp_text.restore();

    try testing.expect(base != key(.on, .none));
    try testing.expect(base != key(.off, .bf16));
    try testing.expect(base != key(.off, .f32));
    try testing.expect(key(.off, .bf16) != key(.off, .f32));

    try testing.expectEqual(TpSum.none, tpSumFor(0, .bf16));
    try testing.expectEqual(TpSum.none, tpSumFor(1, .f32));
    try testing.expectEqual(TpSum.f32, tpSumFor(2, .auto));
    try testing.expectEqual(TpSum.bf16, tpSumFor(2, .bf16));
    try testing.expectEqual(TpSum.bf16, tpSumFor(4, .auto));
    try testing.expectEqual(TpSum.f32, tpSumFor(4, .f32));

    inline for (.{ .{ .MLX_SERVE_DSV41_ATTN_F16, "0" }, .{ .MLX_SERVE_DSV41_CED, "0" } }) |lever| {
        const saved = env.override(lever[0], lever[1]);
        defer saved.restore();
        try testing.expect(base != key(.off, .none));
    }
}
