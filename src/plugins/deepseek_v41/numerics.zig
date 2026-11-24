//! DeepSeek-V4.1's levers that change the prefill or decode numerics, resolved as its kernels read them. Their hash
//! is the plugin's contribution to the peer settings key (`AuxHooks.settings_key`).

const std = @import("std");
const env = @import("../../env.zig");
const flash_attn = @import("prompt/flash_attn.zig");

pub fn attnF16() bool {
    return if (env.get(.MLX_SERVE_DSV41_ATTN_F16)) |v| v[0] != '0' else true;
}

pub fn ced() bool {
    return if (env.get(.MLX_SERVE_DSV41_CED)) |v| v[0] != '0' else true;
}

/// Flash attention is the lever whose effect depends on the pack (a non-EXL3 pack takes the gather unless the setting
/// says `1`), so its entry is the mode this load chose.
pub const Resolved = struct { attn_f16: bool, ced: bool, flash_attn: flash_attn.Mode };

pub fn resolved(flash: flash_attn.Mode) Resolved {
    return .{ .attn_f16 = attnF16(), .ced = ced(), .flash_attn = flash };
}

/// `flash` is the mode the model was loaded with: the process-wide mode belongs to whichever load ran last.
pub fn key(flash: flash_attn.Mode) u64 {
    var h = std.hash.Wyhash.init(0x6473_7634_315f_6e75);
    std.hash.autoHash(&h, resolved(flash));
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
    .{ "MLX_SERVE_DSV41_ATTN_F16", .key },
    .{ "MLX_SERVE_DSV41_CED", .key },
    .{ "MLX_SERVE_DSV41_FLASH_ATTN", .key },
    .{ "MLX_SERVE_DSV41_HEAD8", .same },
    .{ "MLX_SERVE_DSV41_EXACT", .same },
    .{ "MLX_SERVE_DSV41_ENGRAM_LRU_MB", .same },
    .{ "MLX_SERVE_KEYED_SAMPLING", .same },
    .{ "MLX_SERVE_F16_PREFILL", .other_model },
    .{ "MLX_SERVE_KV_ATTN_MMA", .other_model },
    .{ "MLX_SERVE_GDN_STATE_F32", .other_model },
    .{ "MLX_SERVE_CUDA_QMV_TC", .other_model },
    .{ "MINIMAX_H3_GLUE_FUSED", .other_model },
    .{ "MINIMAX_H3_VSA", .other_model },
    .{ "MLX_SERVE_H3_RESIDENT", .other_model },
    .{ "MLX_SERVE_MING_FUSED", .other_model },
    .{ "MLX_SERVE_INKLING2_FUSED", .other_model },
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

test "dsv41 numerics: the key follows each lever and the flash mode it is given, not the flash lever's text" {
    const unset = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, null);
    defer unset.restore();
    const base = key(.off);

    const flash_text = env.override(.MLX_SERVE_DSV41_FLASH_ATTN, "1");
    try testing.expectEqual(base, key(.off));
    flash_text.restore();

    try testing.expect(base != key(.on));

    inline for (.{ .{ .MLX_SERVE_DSV41_ATTN_F16, "0" }, .{ .MLX_SERVE_DSV41_CED, "0" } }) |lever| {
        const saved = env.override(lever[0], lever[1]);
        defer saved.restore();
        try testing.expect(base != key(.off));
    }
}
