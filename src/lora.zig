//! Runtime (unfused) LoRA adapters for the image backends (FLUX.2 / Krea-2).
//!
//! A LoRA safetensors file carries pairs `<module>.lora_A.weight` [r,in] /
//! `<module>.lora_B.weight` [out,r] (diffusers naming; `lora_down`/`lora_up`,
//! the dotted `lora.down`/`lora.up` variant, and a `.default.` PEFT-adapter
//! infix are all accepted). Its net scale is alpha/rank, declared either per
//! module as a scalar `<module>.alpha` tensor (kohya) or once for the whole
//! file in the safetensors `__metadata__` (`fileAlphaScale` — PEFT, kohya
//! training metadata, plain flat pairs), with 1.0 for a file that declares
//! neither. Adapters are NOT fused into the base
//! weights — each attached linear computes y = base(x) + Σᵢ scaleᵢ·(x@Aᵀᵢ)@Bᵀᵢ
//! at runtime, one term per attached adapter. That keeps quantized
//! checkpoints lossless (no dequant→requant round-trip) and makes detach a
//! pointer clear.
//!
//! `<module>` itself shows up in the wild under several different naming
//! conventions depending on which tool exported the LoRA: mlx-serve's own
//! runtime module names (diffusers-style for FLUX.2, native for Krea-2),
//! BFL/ComfyUI-style names (`double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`,
//! `img_in`, `time_in.in_layer`, …), and Kohya's flat `lora_unet_...`/
//! `lora_transformer_...` key scheme (dots replaced with underscores). Some
//! of those source tensors are also *fused* where mlx-serve's runtime is
//! *split* (BFL packs Q/K/V for a double-stream block into one `qkv` linear;
//! FLUX.2's own attention keeps them separate) — `ArchTable` maps every
//! accepted spelling onto mlx-serve's own canonical module key, splitting a
//! fused up-projection into thirds when required (mirrors mflux's
//! `LoraTransforms.split_q_up` for the very same tensors). The
//! down-projection is always shared, full-rank, across every target of a
//! fused split — a single adapter trained over the fused linear has one A
//! applied identically to q/k/v. (An earlier version tried to infer a rarer
//! "three independently-packed rank-r/3 adapters" case from the rank being
//! divisible by 3, but that has no real signal behind it — ordinary shared
//! ranks like 24/48/96 hit it too — and silently corrupted those adapters
//! instead of erroring, so it was removed.)
//!
//! Multiple adapters attach simultaneously via `Stack` (mirrors mflux's
//! `lora_paths`/`lora_scales`): each loaded `File` keeps its own module→Ref
//! mapping, `Stack.findAll` collects every file's `Ref` for a given module,
//! and `deltaSum` adds their deltas — the same "sum, don't merge" semantics
//! as mflux's `FusedLoRALinear`.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// Non-owning adapter reference installed on a linear layer. `at`/`bt` are
/// pre-transposed bf16 so the hot path is two plain matmuls.
pub const Ref = struct {
    at: mlx.mlx_array, // [in, r]
    bt: mlx.mlx_array, // [r, out]
    scale: f32,
};

pub const Role = enum { a, b, alpha };
pub const KeyInfo = struct { module: []const u8, role: Role, flat: bool = false };

/// Classify one safetensors key: strip a wrapper prefix and a matrix-role
/// suffix, and report which flavor of prefix was stripped.
///
/// Dotted wrapper prefixes (`base_model.model.`, `transformer.`,
/// `diffusion_model.`) leave a dot-joined module name behind, e.g.
/// `transformer_blocks.3.attn.to_q`. Kohya's flat prefixes (`lora_unet_`,
/// `lora_transformer_`) leave an underscore-joined module name behind
/// instead (`flat = true`) — those two are never mixed, since a flat
/// export never uses dots in the module portion of the key.
///
/// Suffixes accept diffusers (`lora_A`/`lora_B`), Kohya (`lora_down`/
/// `lora_up`), and the dotted `lora.down`/`lora.up` variant, each with an
/// optional PEFT `.default.` adapter-name infix. `to_out.0` is normalized to
/// `to_out` (dotted keys) / `to_out_0` is normalized to `to_out` (flat keys)
/// so both forms line up with the single `to_out` linear on every backend.
pub fn parseKey(key: []const u8) ?KeyInfo {
    var k = key;
    var flat = false;
    inline for (.{ "lora_unet_", "lora_transformer_" }) |fpfx| {
        if (std.mem.startsWith(u8, k, fpfx)) {
            k = k[fpfx.len..];
            flat = true;
        }
    }
    if (!flat) {
        inline for (.{ "base_model.model.", "transformer.", "diffusion_model." }) |pfx| {
            if (std.mem.startsWith(u8, k, pfx)) k = k[pfx.len..];
        }
    }
    const suffixes = .{
        .{ ".lora_A.default.weight", Role.a },
        .{ ".lora_A.weight", Role.a },
        .{ ".lora_down.default.weight", Role.a },
        .{ ".lora_down.weight", Role.a },
        .{ ".lora.down.default.weight", Role.a },
        .{ ".lora.down.weight", Role.a },
        .{ ".lora_B.default.weight", Role.b },
        .{ ".lora_B.weight", Role.b },
        .{ ".lora_up.default.weight", Role.b },
        .{ ".lora_up.weight", Role.b },
        .{ ".lora.up.default.weight", Role.b },
        .{ ".lora.up.weight", Role.b },
        .{ ".alpha", Role.alpha },
    };
    inline for (suffixes) |sf| {
        if (std.mem.endsWith(u8, k, sf[0])) {
            var m = k[0 .. k.len - sf[0].len];
            if (flat) {
                if (std.mem.endsWith(u8, m, "_to_out_0")) m = m[0 .. m.len - 2];
            } else {
                if (std.mem.endsWith(u8, m, ".to_out.0")) m = m[0 .. m.len - 2];
            }
            return .{ .module = m, .role = sf[1], .flat = flat };
        }
    }
    return null;
}

// ════════════════════════════════════════════════════════════════════════
// Architecture-aware canonicalization
//
// mlx-serve's own runtime module names (the ones `attachLora` in flux.zig /
// krea.zig actually query) are the "canonical" names below. Every other
// spelling a LoRA file might use is an "alias" that maps onto one (or, for
// a fused source tensor, several) canonical names. See the module doc
// comment above for the taxonomy of naming schemes this covers.
// ════════════════════════════════════════════════════════════════════════

pub const Arch = enum { flux2, krea2, minimax_h3, generic };

/// Which third of a fused up-projection a canonical target draws from, when
/// the source tensor packs several linears together (BFL's fused QKV).
/// `none` means the source and target are already 1:1.
pub const Split = enum { none, third0, third1, third2 };

const AliasRow = struct {
    /// mlx-serve's own module key, e.g. "transformer_blocks.{}.attn.to_q"
    /// or "blocks.{}.attn.wq". At most one "{}" block placeholder.
    canonical: []const u8,
    /// A spelling that should resolve to `canonical` — either mlx-serve's
    /// own name again (a "self row", needed so every canonical target is
    /// also tried against Kohya's flattened key scheme) or a foreign one.
    alias: []const u8,
    split: Split = .none,
};

fn t(canonical: []const u8, alias: []const u8) AliasRow {
    return .{ .canonical = canonical, .alias = alias };
}
fn ts(canonical: []const u8, alias: []const u8, split: Split) AliasRow {
    return .{ .canonical = canonical, .alias = alias, .split = split };
}

// FLUX.2: doubles already use mlx-serve's own diffusers-style names
// 1:1 as canonical (self rows), so only the BFL/Kohya alternates below
// need real translation. Singles are fused in BOTH mlx-serve's runtime
// (`to_qkv_mlp_proj`) and BFL's native checkpoint (`linear1`) — a direct
// 1:1 alias, no splitting required. Doubles' Q/K/V are fused ONLY in the
// BFL/Kohya source (`img_attn.qkv` / `txt_attn.qkv`); mlx-serve keeps them
// split, so those three rows share one alias with different `split`s
// (mirrors mflux's `LoraTransforms.split_q_up`/`split_k_up`/`split_v_up`).
const flux2_table = [_]AliasRow{
    // Globals
    t("x_embedder", "x_embedder"),
    t("x_embedder", "img_in"),
    t("context_embedder", "context_embedder"),
    t("context_embedder", "txt_in"),
    t("time_guidance_embed.linear_1", "time_guidance_embed.linear_1"),
    t("time_guidance_embed.linear_1", "time_in.in_layer"),
    t("time_guidance_embed.linear_2", "time_guidance_embed.linear_2"),
    t("time_guidance_embed.linear_2", "time_in.out_layer"),
    t("double_stream_modulation_img.linear", "double_stream_modulation_img.linear"),
    t("double_stream_modulation_img.linear", "double_stream_modulation_img.lin"),
    t("double_stream_modulation_txt.linear", "double_stream_modulation_txt.linear"),
    t("double_stream_modulation_txt.linear", "double_stream_modulation_txt.lin"),
    t("single_stream_modulation.linear", "single_stream_modulation.linear"),
    t("single_stream_modulation.linear", "single_stream_modulation.lin"),
    t("norm_out.linear", "norm_out.linear"),
    t("proj_out", "proj_out"),
    t("proj_out", "final_layer.linear"),
    // Double-stream blocks
    t("transformer_blocks.{}.attn.to_q", "transformer_blocks.{}.attn.to_q"),
    ts("transformer_blocks.{}.attn.to_q", "double_blocks.{}.img_attn.qkv", .third0),
    t("transformer_blocks.{}.attn.to_k", "transformer_blocks.{}.attn.to_k"),
    ts("transformer_blocks.{}.attn.to_k", "double_blocks.{}.img_attn.qkv", .third1),
    t("transformer_blocks.{}.attn.to_v", "transformer_blocks.{}.attn.to_v"),
    ts("transformer_blocks.{}.attn.to_v", "double_blocks.{}.img_attn.qkv", .third2),
    t("transformer_blocks.{}.attn.to_out", "transformer_blocks.{}.attn.to_out"),
    t("transformer_blocks.{}.attn.to_out", "double_blocks.{}.img_attn.proj"),
    t("transformer_blocks.{}.attn.add_q_proj", "transformer_blocks.{}.attn.add_q_proj"),
    ts("transformer_blocks.{}.attn.add_q_proj", "double_blocks.{}.txt_attn.qkv", .third0),
    t("transformer_blocks.{}.attn.add_k_proj", "transformer_blocks.{}.attn.add_k_proj"),
    ts("transformer_blocks.{}.attn.add_k_proj", "double_blocks.{}.txt_attn.qkv", .third1),
    t("transformer_blocks.{}.attn.add_v_proj", "transformer_blocks.{}.attn.add_v_proj"),
    ts("transformer_blocks.{}.attn.add_v_proj", "double_blocks.{}.txt_attn.qkv", .third2),
    t("transformer_blocks.{}.attn.to_add_out", "transformer_blocks.{}.attn.to_add_out"),
    t("transformer_blocks.{}.attn.to_add_out", "double_blocks.{}.txt_attn.proj"),
    t("transformer_blocks.{}.ff.linear_in", "transformer_blocks.{}.ff.linear_in"),
    t("transformer_blocks.{}.ff.linear_out", "transformer_blocks.{}.ff.linear_out"),
    t("transformer_blocks.{}.ff_context.linear_in", "transformer_blocks.{}.ff_context.linear_in"),
    t("transformer_blocks.{}.ff_context.linear_out", "transformer_blocks.{}.ff_context.linear_out"),
    // BFL-style double-block MLP keys (ai-toolkit exports)
    t("transformer_blocks.{}.ff.linear_in", "double_blocks.{}.img_mlp.0"),
    t("transformer_blocks.{}.ff.linear_out", "double_blocks.{}.img_mlp.2"),
    t("transformer_blocks.{}.ff_context.linear_in", "double_blocks.{}.txt_mlp.0"),
    t("transformer_blocks.{}.ff_context.linear_out", "double_blocks.{}.txt_mlp.2"),
    // Single-stream blocks (fused on both sides — no split)
    t("single_transformer_blocks.{}.attn.to_qkv_mlp_proj", "single_transformer_blocks.{}.attn.to_qkv_mlp_proj"),
    t("single_transformer_blocks.{}.attn.to_qkv_mlp_proj", "single_blocks.{}.linear1"),
    t("single_transformer_blocks.{}.attn.to_out", "single_transformer_blocks.{}.attn.to_out"),
    t("single_transformer_blocks.{}.attn.to_out", "single_blocks.{}.linear2"),
};

// Krea-2: the runtime uses its own module names throughout (`blocks.{}.attn.wq`,
// `txtfusion.layerwise_blocks.{}...`, `first`, `tmlp.0`, …), never diffusers
// naming — every diffusers/community spelling needs an explicit alias. No
// fused source tensors are known for Krea-2, so no `split` rows here.
const krea2_table = [_]AliasRow{
    // Globals
    t("first", "first"),
    t("first", "img_in"),
    t("tmlp.0", "tmlp.0"),
    t("tmlp.0", "tmlp.linear_in"),
    t("tmlp.0", "time_embed.linear_1"),
    t("tmlp.2", "tmlp.2"),
    t("tmlp.2", "tmlp.linear_out"),
    t("tmlp.2", "time_embed.linear_2"),
    t("tproj.1", "tproj.1"),
    t("tproj.1", "tproj.linear"),
    t("tproj.1", "time_mod_proj"),
    t("txtmlp.1", "txtmlp.1"),
    t("txtmlp.1", "txtmlp.linear_in"),
    t("txtmlp.1", "txt_in.linear_1"),
    t("txtmlp.3", "txtmlp.3"),
    t("txtmlp.3", "txtmlp.linear_out"),
    t("txtmlp.3", "txt_in.linear_2"),
    t("txtfusion.projector", "txtfusion.projector"),
    t("txtfusion.projector", "text_fusion.projector"),
    t("last.linear", "last.linear"),
    t("last.linear", "final_layer.linear"),
} ++ kreaBlockRows("blocks.{}", "transformer_blocks.{}") ++
    kreaBlockRows("txtfusion.layerwise_blocks.{}", "text_fusion.layerwise_blocks.{}") ++
    kreaBlockRows("txtfusion.refiner_blocks.{}", "text_fusion.refiner_blocks.{}");

/// The 8 attn/mlp targets Krea-2 repeats identically across its main
/// blocks and both text-fusion groups — factored out so the three block
/// groups above can't drift out of sync with each other.
fn kreaBlockRows(comptime canon_pfx: []const u8, comptime alias_pfx: []const u8) [16]AliasRow {
    return .{
        t(canon_pfx ++ ".attn.wq", canon_pfx ++ ".attn.wq"),
        t(canon_pfx ++ ".attn.wq", alias_pfx ++ ".attn.to_q"),
        t(canon_pfx ++ ".attn.wk", canon_pfx ++ ".attn.wk"),
        t(canon_pfx ++ ".attn.wk", alias_pfx ++ ".attn.to_k"),
        t(canon_pfx ++ ".attn.wv", canon_pfx ++ ".attn.wv"),
        t(canon_pfx ++ ".attn.wv", alias_pfx ++ ".attn.to_v"),
        t(canon_pfx ++ ".attn.gate", canon_pfx ++ ".attn.gate"),
        t(canon_pfx ++ ".attn.gate", alias_pfx ++ ".attn.to_gate"),
        t(canon_pfx ++ ".attn.wo", canon_pfx ++ ".attn.wo"),
        t(canon_pfx ++ ".attn.wo", alias_pfx ++ ".attn.to_out"),
        t(canon_pfx ++ ".mlp.gate", canon_pfx ++ ".mlp.gate"),
        t(canon_pfx ++ ".mlp.gate", alias_pfx ++ ".ff.gate"),
        t(canon_pfx ++ ".mlp.up", canon_pfx ++ ".mlp.up"),
        t(canon_pfx ++ ".mlp.up", alias_pfx ++ ".ff.up"),
        t(canon_pfx ++ ".mlp.down", canon_pfx ++ ".mlp.down"),
        t(canon_pfx ++ ".mlp.down", alias_pfx ++ ".ff.down"),
    };
}

// MiniMax-H3: the checkpoint's own module names ARE mlx-serve's (the converter
// renames nothing and the reference node keys adapters as
// `diffusion_model.<module>.weight`, whose prefix `parseKey` already strips),
// so every row here is a self row. They exist for the FLAT scheme only: a
// Kohya-style export writes `lora_unet_blocks_0_attn_qkv_proj`, and without a
// table there is nothing to translate the underscores back into dots. A dotted
// H3 key needs no row at all — `canonicalize` passes unknown names through
// unchanged — which is why the Turbo file loaded correctly before this table
// existed.
const minimax_h3_table = [_]AliasRow{
    t("blocks.{}.attn.qkv_proj", "blocks.{}.attn.qkv_proj"),
    t("blocks.{}.attn.out_proj", "blocks.{}.attn.out_proj"),
    t("blocks.{}.mlp.fc1", "blocks.{}.mlp.fc1"),
    t("blocks.{}.mlp.fc2", "blocks.{}.mlp.fc2"),
    t("blocks.{}.adaln_proj.linear", "blocks.{}.adaln_proj.linear"),
    t("token_refiner.blocks.{}.attn.qkv_proj", "token_refiner.blocks.{}.attn.qkv_proj"),
    t("token_refiner.blocks.{}.attn.out_proj", "token_refiner.blocks.{}.attn.out_proj"),
    t("token_refiner.blocks.{}.mlp.fc1", "token_refiner.blocks.{}.mlp.fc1"),
    t("token_refiner.blocks.{}.mlp.fc2", "token_refiner.blocks.{}.mlp.fc2"),
    t("final_layer.adaln_proj.linear", "final_layer.adaln_proj.linear"),
};

fn archTable(arch: Arch) []const AliasRow {
    return switch (arch) {
        .flux2 => &flux2_table,
        .krea2 => &krea2_table,
        .minimax_h3 => &minimax_h3_table,
        .generic => &.{},
    };
}

/// Match `candidate` against `template`, which has at most one "{}" block
/// placeholder. Returns the block-number substring on a match (empty slice
/// when the template has no placeholder), null otherwise. Every character
/// in the placeholder position must be a digit — this is what lets a single
/// pass find the right block index without scanning every digit run in the
/// key the way a multi-placeholder scheme would need to.
fn matchTemplate(candidate: []const u8, template: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, template, "{}")) |idx| {
        const pre = template[0..idx];
        const suf = template[idx + 2 ..];
        if (candidate.len < pre.len + suf.len) return null;
        if (!std.mem.startsWith(u8, candidate, pre)) return null;
        if (!std.mem.endsWith(u8, candidate, suf)) return null;
        const mid = candidate[pre.len .. candidate.len - suf.len];
        if (mid.len == 0) return null;
        for (mid) |c| {
            if (c < '0' or c > '9') return null;
        }
        return mid;
    }
    return if (std.mem.eql(u8, candidate, template)) template[0..0] else null;
}

/// Copy `s` into `buf`, replacing `.` with `_` (Kohya's flat key scheme).
/// Truncates rather than overflows if `s` doesn't fit — every table entry
/// is well under the caller's buffer size, so truncation never happens in
/// practice; it just avoids ever writing out of bounds.
fn flattenInto(buf: []u8, s: []const u8) []const u8 {
    const n = @min(buf.len, s.len);
    for (s[0..n], 0..) |c, i| buf[i] = if (c == '.') '_' else c;
    return buf[0..n];
}

/// Render `template`'s "{}" placeholder (if any) as `block` into `buf`.
fn formatCanonical(buf: []u8, template: []const u8, block: []const u8) []const u8 {
    if (std.mem.indexOf(u8, template, "{}")) |idx| {
        return std.fmt.bufPrint(buf, "{s}{s}{s}", .{ template[0..idx], block, template[idx + 2 ..] }) catch template;
    }
    return template;
}

pub const MAX_FANOUT = 3; // widest fan-out is a fused QKV source → 3 targets
const CanonBuf = [96]u8;

pub const CanonMatch = struct { canon: []const u8, split: Split };

/// Resolve one classified key's module name to every canonical mlx-serve
/// module it targets. Almost always a single 1:1 match (including the
/// common case where the file already speaks mlx-serve's own naming — that
/// matches a self row directly). Fans out to up to `MAX_FANOUT` matches only
/// for a fused source tensor (BFL's packed QKV). Falls back to treating the
/// classified module name as already canonical when nothing in the arch
/// table recognizes it, so unlisted-but-already-compatible names and
/// genuinely-unknown ones both degrade to today's behavior rather than
/// silently dropping the tensor.
pub fn canonicalize(
    module: []const u8,
    flat: bool,
    arch: Arch,
    bufs: *[MAX_FANOUT]CanonBuf,
    out: *[MAX_FANOUT]CanonMatch,
) []CanonMatch {
    var n: usize = 0;
    var fbuf_alias: [128]u8 = undefined;

    for (archTable(arch)) |row| {
        if (n >= MAX_FANOUT) break;
        // ALIAS only, in both schemes. Every canonical is also listed as its
        // own alias (the self rows, invariant-tested), so a file already
        // speaking mlx-serve's naming resolves through those — and matching
        // the canonical as well would make every ALIAS row fire a second time
        // on its own target, producing a duplicate that for a fused row
        // carries a `.third*` split and a third-width B.
        const block = if (flat) blk: {
            const fa = flattenInto(&fbuf_alias, row.alias);
            break :blk matchTemplate(module, fa) orelse continue;
        } else (matchTemplate(module, row.alias) orelse continue);

        out[n] = .{ .canon = formatCanonical(&bufs[n], row.canonical, block), .split = row.split };
        n += 1;
    }

    if (n == 0) {
        out[0] = .{ .canon = formatCanonical(&bufs[0], module, ""), .split = .none };
        n = 1;
    }

    return out[0..n];
}

/// y_delta = scale · (x @ at) @ bt, returned in x's dtype.
pub fn delta(x: mlx.mlx_array, ref: Ref, s: mlx.mlx_stream) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    var xa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xa);
    try mlx.check(mlx.mlx_matmul(&xa, x, ref.at, s));
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_matmul(&xb, xa, ref.bt, s));
    const sc = mlx.mlx_array_new_float(ref.scale);
    defer _ = mlx.mlx_array_free(sc);
    var scaled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, xb, sc, s));
    if (mlx.mlx_array_dtype(scaled) != dt) {
        var back = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(back);
        try mlx.check(mlx.mlx_astype(&back, scaled, dt, s));
        _ = mlx.mlx_array_free(scaled);
        return back;
    }
    return scaled;
}

/// Sum of `scale_i · (x @ Aᵀᵢ) @ Bᵀᵢ` over every attached adapter — the
/// runtime realization of stacking multiple LoRAs on one linear (mirrors
/// mflux's `FusedLoRALinear`, which sums per-adapter deltas rather than
/// merging them into the base weight). Caller must ensure `refs.len >= 1`.
pub fn deltaSum(x: mlx.mlx_array, refs: []const Ref, s: mlx.mlx_stream) !mlx.mlx_array {
    var total = try delta(x, refs[0], s);
    errdefer _ = mlx.mlx_array_free(total);
    for (refs[1..]) |r| {
        const d = try delta(x, r, s);
        defer _ = mlx.mlx_array_free(d);
        var summed = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(summed);
        try mlx.check(mlx.mlx_add(&summed, total, d, s));
        _ = mlx.mlx_array_free(total);
        total = summed;
    }
    return total;
}

/// Max simultaneously-attached LoRA adapters per engine. Bounds the
/// per-linear `Ref` array in the image/video backends and the request-body
/// `lora_paths`/`lora_scales` arrays in gen.zig. mflux has no hard cap;
/// eight covers every practical multi-LoRA stack (style + character + a
/// couple of concept adapters) while keeping the per-linear footprint a
/// fixed-size array instead of a heap allocation on the hot path.
pub const MAX_LORAS: usize = 8;

/// One loaded adapter pair, keyed by the module it targets.
pub const Entry = struct {
    module: []u8, // owned
    at: mlx.mlx_array, // [in, r] bf16
    bt: mlx.mlx_array, // [r, out] bf16
    scale: f32, // alpha/rank when the file carries alpha, else 1.0
};

/// All adapters from one safetensors file. Owns the arrays the installed
/// `Ref`s point at — must outlive every attach until detach.
pub const File = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,

    pub fn deinit(self: *File) void {
        for (self.entries) |*e| {
            self.allocator.free(e.module);
            _ = mlx.mlx_array_free(e.at);
            _ = mlx.mlx_array_free(e.bt);
        }
        self.allocator.free(self.entries);
    }

    pub fn find(self: *const File, module: []const u8) ?*const Entry {
        for (self.entries) |*e| {
            if (std.mem.eql(u8, e.module, module)) return e;
        }
        return null;
    }
};

/// A bounded set of simultaneously-attached LoRA adapters (mflux's
/// `lora_paths`/`lora_scales` lists). Owns every loaded `File` plus a copy
/// of the request paths (for the no-op-reuse check in gen.zig's
/// `setLoras`) and each file's user-requested scale. Fixed-capacity —
/// `MAX_LORAS` — so attach/lookup never allocates on the hot path.
pub const Stack = struct {
    allocator: std.mem.Allocator,
    files: [MAX_LORAS]File = undefined,
    paths: [MAX_LORAS][]u8 = undefined, // owned, for reuse comparison
    scales: [MAX_LORAS]f32 = undefined,
    count: u8 = 0,

    pub fn deinit(self: *Stack) void {
        for (self.files[0..self.count]) |*f| f.deinit();
        for (self.paths[0..self.count]) |p| self.allocator.free(p);
        self.count = 0;
    }

    /// True when `paths`/`scales` are exactly the currently-attached set, in
    /// order — lets `setLoras` no-op on a repeat request (mirrors the
    /// single-adapter path's `cur == p and scale == self.lora_scale` check).
    pub fn matches(self: *const Stack, paths: []const []const u8, scales: []const f32) bool {
        if (paths.len != self.count or scales.len != self.count) return false;
        for (paths, scales, 0..) |p, sc, i| {
            if (!std.mem.eql(u8, self.paths[i], p) or sc != self.scales[i]) return false;
        }
        return true;
    }

    /// Collect every `Ref` across the stack's files that targets `module`,
    /// in attach order. Order does not change the result — deltas are
    /// summed, never fused, so it is not "later file wins" like a merged
    /// weight would be. Writes into `out` (capacity `MAX_LORAS`) and returns
    /// the filled prefix.
    pub fn findAll(self: *const Stack, module: []const u8, out: *[MAX_LORAS]Ref) []Ref {
        var n: usize = 0;
        for (self.files[0..self.count], self.scales[0..self.count]) |*f, user_scale| {
            if (f.find(module)) |e| {
                out[n] = .{ .at = e.at, .bt = e.bt, .scale = e.scale * user_scale };
                n += 1;
            }
        }
        return out[0..n];
    }
};

const Partial = struct {
    a: mlx.mlx_array = .{ .ctx = null },
    b: mlx.mlx_array = .{ .ctx = null },
    alpha: ?f32 = null,
    flat: bool = false,
};

fn metaGet(meta: mlx.mlx_map_string_to_string, key: [*:0]const u8) ?[]const u8 {
    var v: [*:0]const u8 = undefined;
    // A missing key returns non-zero WITHOUT raising, so this never reaches
    // the uncatchable mlx error path.
    if (mlx.mlx_map_string_to_string_get(&v, meta, key) != 0) return null;
    return std.mem.span(v);
}

/// alpha/rank. Null when either is missing or is not a usable positive finite
/// number — running an adapter at a scale we had to guess at is the bug this
/// whole thing exists to fix.
fn alphaRatio(alpha: ?f32, rank: ?f32) ?f32 {
    const al = alpha orelse return null;
    const rk = rank orelse return null;
    if (!std.math.isFinite(al) or al <= 0 or !std.math.isFinite(rk) or rk <= 0) return null;
    return al / rk;
}

fn parseNum(text: ?[]const u8) ?f32 {
    const raw = text orelse return null;
    return std.fmt.parseFloat(f32, std.mem.trim(u8, raw, " \t\r\n\"")) catch null;
}

fn jsonNumber(v: std.json.Value) ?f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        .number_string, .string => |s| parseNum(s),
        else => null,
    };
}

/// PEFT/diffusers store their whole adapter config as ONE JSON document
/// inside a single `__metadata__` string, with every key prefixed by the
/// pipeline component it configures (`transformer.lora_alpha`, `transformer.r`).
/// Match on the leaf, so the prefix can be anything.
fn alphaScaleFromJson(allocator: std.mem.Allocator, doc: []const u8) ?f32 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, doc, .{}) catch return null;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };
    var alpha: ?f32 = null;
    var rank: ?f32 = null;
    var it = obj.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        const leaf = if (std.mem.lastIndexOfScalar(u8, key, '.')) |d| key[d + 1 ..] else key;
        if (std.mem.eql(u8, leaf, "lora_alpha")) {
            alpha = jsonNumber(kv.value_ptr.*);
        } else if (std.mem.eql(u8, leaf, "r") or std.mem.eql(u8, leaf, "lora_rank")) {
            rank = jsonNumber(kv.value_ptr.*);
        }
    }
    return alphaRatio(alpha, rank);
}

/// The scale (alpha/rank) the FILE declares for itself, or null when it
/// declares none and 1.0 is therefore the honest answer.
///
/// A LoRA's net strength is alpha/rank, and WHERE that pair lives is per
/// exporter — four real community adapters, four conventions. This loader
/// originally read only kohya's per-module `.alpha` TENSOR, so every
/// PEFT/diffusers export silently ran at 1.0: for the common alpha 4 / r 32
/// pairing that is 8x too strong, which renders pure noise rather than a
/// stronger style. An adapter whose weights happen to be weak survives it and
/// looks fine, which is why "it produced an image" never caught this.
fn fileAlphaScale(allocator: std.mem.Allocator, meta: mlx.mlx_map_string_to_string) ?f32 {
    if (metaGet(meta, "lora_adapter_metadata")) |doc| {
        if (alphaScaleFromJson(allocator, doc)) |sc| return sc;
    }
    if (alphaRatio(
        parseNum(metaGet(meta, "lora_alpha")),
        parseNum(metaGet(meta, "lora_rank") orelse metaGet(meta, "r")),
    )) |sc| return sc;
    if (alphaRatio(
        parseNum(metaGet(meta, "ss_network_alpha")),
        parseNum(metaGet(meta, "ss_network_dim")),
    )) |sc| return sc;
    return null;
}

fn scalarValue(arr: mlx.mlx_array, s: mlx.mlx_stream) ?f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    mlx.check(mlx.mlx_astype(&f, arr, .float32, s)) catch return null;
    _ = mlx.mlx_array_eval(f);
    const d = mlx.mlx_array_data_float32(f) orelse return null;
    return d[0];
}

/// Load every complete A/B pair from a LoRA .safetensors file. A/B are
/// pre-transposed to [in,r]/[r,out], materialized, and cast to bf16.
/// All load-time ops run on a CPU stream — `Load::eval_gpu` is not
/// implemented, exactly like `model.loadWeights` (unified memory makes the
/// arrays GPU-usable afterwards).
/// Prove a client-supplied adapter path is loadable BEFORE mlx sees it.
/// `mlx_load_safetensors` on a missing path raises an MLX error, and an MLX
/// error is not a Zig error — it kills the process. One request with a stale
/// `lora_path` (a moved adapter, a typo) took the whole server down,
/// mid-conversation, for every other client too. Same class as the
/// non-text-model 400-before-prefill rule: validate on OUR side of an
/// uncatchable boundary.
///
/// Exported because a backend that loads its DiT PER REQUEST (MiniMax-H3)
/// reaches `loadFile` only after minutes of staging, far too late for the
/// handler's 400 — so its handler calls this up front. ONE rule either way:
/// a second copy is how the two would drift.
pub fn validatePath(path: []const u8) !void {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return error.BadLoraPath;
    const io = std.Io.Threaded.global_single_threaded.io();
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.BadLoraPath;
    defer f.close(io);
    const st = f.stat(io) catch return error.BadLoraPath;
    if (st.kind != .file) return error.BadLoraPath; // a directory opens fine
}

pub fn loadFile(allocator: std.mem.Allocator, path: []const u8, arch: Arch) !File {
    try validatePath(path);
    const pathz = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(pathz);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, pathz, s));
    const file_scale = fileAlphaScale(allocator, meta_map);
    log.info("[lora] {s}: scale {d:.4}{s}\n", .{
        std.fs.path.basename(path),
        file_scale orelse 1.0,
        if (file_scale == null) " (file declares no alpha)" else " from the file's own alpha",
    });

    var partials = std.StringHashMap(Partial).init(allocator);
    defer {
        var it = partials.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.a.ctx != null) _ = mlx.mlx_array_free(e.value_ptr.a);
            if (e.value_ptr.b.ctx != null) _ = mlx.mlx_array_free(e.value_ptr.b);
            allocator.free(e.key_ptr.*);
        }
        partials.deinit();
    }

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const info = parseKey(std.mem.span(key.?)) orelse {
            _ = mlx.mlx_array_free(value);
            continue;
        };
        const gop = try partials.getOrPut(info.module);
        if (!gop.found_existing) {
            gop.key_ptr.* = try allocator.dupe(u8, info.module);
            gop.value_ptr.* = .{ .flat = info.flat };
        }
        switch (info.role) {
            .a => {
                if (gop.value_ptr.a.ctx != null) _ = mlx.mlx_array_free(gop.value_ptr.a);
                gop.value_ptr.a = value;
            },
            .b => {
                if (gop.value_ptr.b.ctx != null) _ = mlx.mlx_array_free(gop.value_ptr.b);
                gop.value_ptr.b = value;
            },
            .alpha => {
                gop.value_ptr.alpha = scalarValue(value, s);
                _ = mlx.mlx_array_free(value);
            },
        }
    }

    var entries: std.ArrayList(Entry) = .empty;
    errdefer {
        for (entries.items) |*e| {
            allocator.free(e.module);
            _ = mlx.mlx_array_free(e.at);
            _ = mlx.mlx_array_free(e.bt);
        }
        entries.deinit(allocator);
    }
    var it = partials.iterator();
    while (it.next()) |e| {
        const p = e.value_ptr;
        if (p.a.ctx == null or p.b.ctx == null) continue; // incomplete pair
        const full_rank: c_int = mlx.getShape(p.a)[0]; // A [r,in]
        // The down-projection (A) is ALWAYS shared across every target of a
        // fused source tensor: a LoRA trained over a fused QKV linear has
        // ONE A applied identically to q/k/v, with only B's *output* rows
        // split per target (mirrors mflux's `_split_qkv_up` / the shared-A
        // half of `FusedLoRALinear`). We deliberately do NOT try to guess
        // "these are secretly three independent rank-r/3 adapters packed
        // block-diagonally" from the rank being divisible by 3 — that
        // heuristic has no real signal behind it (24, 48, 96... are
        // completely ordinary ranks for one shared adapter) and guessing
        // wrong doesn't error, it silently feeds the DiT a corrupted 1/3-rank
        // delta. Always sharing A is the correct behavior for every fused
        // export this module actually targets (BFL/ComfyUI/ai-toolkit).
        //
        // Canonicalize the module name(s) for the target architecture. A
        // fused source tensor (e.g. BFL's packed img_attn.qkv) fans out to
        // up to MAX_FANOUT canonical targets — one per `Split` third — and
        // EVERY match needs its own Entry, or two of every three fused
        // targets (to_k/to_v) would silently never get an adapter attached.
        var canon_bufs: [MAX_FANOUT]CanonBuf = undefined;
        var canon_out: [MAX_FANOUT]CanonMatch = undefined;
        const matches = canonicalize(e.key_ptr.*, p.flat, arch, &canon_bufs, &canon_out);
        for (matches) |m| {
            const idx = splitIndex(m.split);
            const at = try prepTransposed(p.a, s);
            errdefer _ = mlx.mlx_array_free(at);
            const bt = if (idx) |i| try prepTransposedThird(p.b, i, s) else try prepTransposed(p.b, s);
            errdefer _ = mlx.mlx_array_free(bt);
            // A per-module `.alpha` tensor is MORE SPECIFIC than a file-wide
            // declaration, so it wins; the file's own metadata is the default
            // for every module that ships none; 1.0 only when the file
            // declares nothing anywhere.
            const scale: f32 = if (p.alpha) |al|
                al / @as(f32, @floatFromInt(full_rank))
            else
                file_scale orelse 1.0;
            try entries.append(allocator, .{
                .module = try allocator.dupe(u8, m.canon),
                .at = at,
                .bt = bt,
                .scale = scale,
            });
        }
    }
    return .{ .allocator = allocator, .entries = try entries.toOwnedSlice(allocator) };
}

/// `Split` → which third (0/1/2) of a 3-way fused source it draws from, or
/// `null` for an already-1:1 (`.none`) match.
fn splitIndex(sp: Split) ?u2 {
    return switch (sp) {
        .none => null,
        .third0 => 0,
        .third1 => 1,
        .third2 => 2,
    };
}

/// [n, k] → the `idx`-th of 3 equal chunks along axis 0, still `[n/3, k]`
/// and untransposed. Used for the up-projection's output axis — always
/// split for a fused target (mirrors mflux's `_split_qkv_up`). The
/// down-projection is never split this way; see the shared-A note in
/// `loadFile`.
fn sliceThird(w: mlx.mlx_array, idx: u2, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(w);
    const n = shape[0];
    const third = @divTrunc(n, 3);
    const start = [_]c_int{ third * @as(c_int, idx), 0 };
    const stop = [_]c_int{ third * (@as(c_int, idx) + 1), shape[1] };
    const strides = [_]c_int{ 1, 1 };
    var res = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(res);
    try mlx.check(mlx.mlx_slice(&res, w, &start, 2, &stop, 2, &strides, 2, s));
    return res;
}

/// [n, k] → materialized bf16 [k, n/3] for the `idx`-th third, pre-transposed
/// exactly like `prepTransposed`.
fn prepTransposedThird(w: mlx.mlx_array, idx: u2, s: mlx.mlx_stream) !mlx.mlx_array {
    const piece = try sliceThird(w, idx, s);
    defer _ = mlx.mlx_array_free(piece);
    return prepTransposed(piece, s);
}

fn prepTransposed(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const axes = [_]c_int{ 1, 0 };
    var tensor = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tensor);
    try mlx.check(mlx.mlx_transpose_axes(&tensor, w, &axes, 2, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, tensor, false, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, c, .bfloat16, s));
    _ = mlx.mlx_array_eval(out); // settle the Load graph on the CPU stream
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "parseKey classifies diffusers + kohya LoRA keys and strips wrappers" {
    // diffusers A/B with transformer. prefix
    const a = parseKey("transformer.transformer_blocks.3.attn.to_q.lora_A.weight").?;
    try testing.expectEqualStrings("transformer_blocks.3.attn.to_q", a.module);
    try testing.expectEqual(Role.a, a.role);
    const b = parseKey("transformer.transformer_blocks.3.attn.to_q.lora_B.weight").?;
    try testing.expectEqual(Role.b, b.role);
    // to_out.0 normalization
    const o = parseKey("transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight").?;
    try testing.expectEqualStrings("transformer_blocks.0.attn.to_out", o.module);
    // kohya-style down/up aliases, no prefix
    const d = parseKey("single_transformer_blocks.7.attn.to_out.lora_down.weight").?;
    try testing.expectEqual(Role.a, d.role);
    try testing.expectEqualStrings("single_transformer_blocks.7.attn.to_out", d.module);
    const u = parseKey("blocks.2.mlp.gate.lora_up.weight").?;
    try testing.expectEqual(Role.b, u.role);
    // alpha
    const al = parseKey("blocks.2.mlp.gate.alpha").?;
    try testing.expectEqual(Role.alpha, al.role);
    try testing.expectEqualStrings("blocks.2.mlp.gate", al.module);
    // non-LoRA keys are ignored
    try testing.expect(parseKey("blocks.2.mlp.gate.weight") == null);
    try testing.expect(parseKey("bn.running_mean") == null);
}

test "delta computes scale·(x@Aᵀ)@Bᵀ" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // x [1,2] = [1,2]; A [1,2] (r=1,in=2) → at [2,1] = [[3],[4]];
    // B [2,1] (out=2,r=1) → bt [1,2] = [[5,6]]. x@at = 11; delta = 2·[55,66].
    const xv = [_]f32{ 1, 2 };
    const xs = [_]c_int{ 1, 2 };
    const x = mlx.mlx_array_new_data(&xv, &xs, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    const atv = [_]f32{ 3, 4 };
    const ats = [_]c_int{ 2, 1 };
    const at = mlx.mlx_array_new_data(&atv, &ats, 2, .float32);
    defer _ = mlx.mlx_array_free(at);
    const btv = [_]f32{ 5, 6 };
    const bts = [_]c_int{ 1, 2 };
    const bt = mlx.mlx_array_new_data(&btv, &bts, 2, .float32);
    defer _ = mlx.mlx_array_free(bt);
    const d = try delta(x, .{ .at = at, .bt = bt, .scale = 2.0 }, s);
    defer _ = mlx.mlx_array_free(d);
    _ = mlx.mlx_array_eval(d);
    const dd = mlx.mlx_array_data_float32(d) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 110), dd[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 132), dd[1], 1e-4);
}

test "deltaSum adds every attached adapter's delta — the whole point of stacking" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Same x/A/B as the single-delta test above, so the arithmetic is already
    // known: one ref at scale 2 gives [110, 132]. Stacking the SAME ref at
    // scale 1 must add exactly half of it again — summed, never merged, so
    // order cannot matter.
    const xv = [_]f32{ 1, 2 };
    const xs = [_]c_int{ 1, 2 };
    const x = mlx.mlx_array_new_data(&xv, &xs, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    const atv = [_]f32{ 3, 4 };
    const ats = [_]c_int{ 2, 1 };
    const at = mlx.mlx_array_new_data(&atv, &ats, 2, .float32);
    defer _ = mlx.mlx_array_free(at);
    const btv = [_]f32{ 5, 6 };
    const bts = [_]c_int{ 1, 2 };
    const bt = mlx.mlx_array_new_data(&btv, &bts, 2, .float32);
    defer _ = mlx.mlx_array_free(bt);

    const refs = [_]Ref{
        .{ .at = at, .bt = bt, .scale = 2.0 },
        .{ .at = at, .bt = bt, .scale = 1.0 },
    };
    const d = try deltaSum(x, &refs, s);
    defer _ = mlx.mlx_array_free(d);
    _ = mlx.mlx_array_eval(d);
    const dd = mlx.mlx_array_data_float32(d) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 165), dd[0], 1e-3); // 110 + 55
    try testing.expectApproxEqAbs(@as(f32, 198), dd[1], 1e-3); // 132 + 66

    // Reversed order is the same sum (no "last one wins" merge semantics).
    const rev = [_]Ref{ refs[1], refs[0] };
    const d2 = try deltaSum(x, &rev, s);
    defer _ = mlx.mlx_array_free(d2);
    _ = mlx.mlx_array_eval(d2);
    const dd2 = mlx.mlx_array_data_float32(d2) orelse return error.NoData;
    try testing.expectApproxEqAbs(dd[0], dd2[0], 1e-3);
    try testing.expectApproxEqAbs(dd[1], dd2[1], 1e-3);

    // One adapter must be identical to the un-stacked path.
    const one = try deltaSum(x, refs[0..1], s);
    defer _ = mlx.mlx_array_free(one);
    _ = mlx.mlx_array_eval(one);
    const od = mlx.mlx_array_data_float32(one) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 110), od[0], 1e-4);
}

test "Stack.findAll collects one Ref per file and folds in that file's user scale" {
    const a = testing.allocator;
    // Two hand-built files: both carry "m.shared", only the first carries
    // "m.only_a". Handles are dummies — findAll copies them, never reads them.
    const dummy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dummy);

    var e0 = [_]Entry{
        .{ .module = try a.dupe(u8, "m.shared"), .at = dummy, .bt = dummy, .scale = 0.5 },
        .{ .module = try a.dupe(u8, "m.only_a"), .at = dummy, .bt = dummy, .scale = 1.0 },
    };
    defer for (e0) |e| a.free(e.module);
    var e1 = [_]Entry{
        .{ .module = try a.dupe(u8, "m.shared"), .at = dummy, .bt = dummy, .scale = 2.0 },
    };
    defer for (e1) |e| a.free(e.module);

    var st: Stack = .{ .allocator = a };
    st.files[0] = .{ .allocator = a, .entries = &e0 };
    st.files[1] = .{ .allocator = a, .entries = &e1 };
    st.scales[0] = 3.0;
    st.scales[1] = 10.0;
    st.count = 2;
    // No `st.deinit()`: the entries are stack-allocated here, and paths were
    // never dup'd — deinit would free memory it does not own.

    var buf: [MAX_LORAS]Ref = undefined;
    const shared = st.findAll("m.shared", &buf);
    try testing.expectEqual(@as(usize, 2), shared.len);
    // The file's OWN alpha/rank scale multiplied by the user's per-file scale.
    try testing.expectApproxEqAbs(@as(f32, 1.5), shared[0].scale, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 20.0), shared[1].scale, 1e-6);

    // A module only one file carries yields exactly one Ref (a stack does not
    // demand that every adapter target every module).
    const only = st.findAll("m.only_a", &buf);
    try testing.expectEqual(@as(usize, 1), only.len);
    try testing.expectApproxEqAbs(@as(f32, 3.0), only[0].scale, 1e-6);

    // Unknown module: empty, never a stale prefix of a previous call.
    try testing.expectEqual(@as(usize, 0), st.findAll("m.nope", &buf).len);
}

test "Stack.matches is the no-op-reuse gate: order and scales are part of identity" {
    const a = testing.allocator;
    var st: Stack = .{ .allocator = a };
    st.paths[0] = try a.dupe(u8, "/a.safetensors");
    st.paths[1] = try a.dupe(u8, "/b.safetensors");
    defer {
        a.free(st.paths[0]);
        a.free(st.paths[1]);
    }
    st.scales[0] = 1.0;
    st.scales[1] = 0.5;
    st.count = 2;

    try testing.expect(st.matches(&.{ "/a.safetensors", "/b.safetensors" }, &.{ 1.0, 0.5 }));
    // A changed scale is a different attachment — reusing here would serve the
    // previous strength silently.
    try testing.expect(!st.matches(&.{ "/a.safetensors", "/b.safetensors" }, &.{ 1.0, 0.9 }));
    // Swapped order: the deltas sum, so the RESULT is the same, but the
    // paths/scales pairing is not — matching positionally keeps `setLoras`
    // honest rather than clever.
    try testing.expect(!st.matches(&.{ "/b.safetensors", "/a.safetensors" }, &.{ 1.0, 0.5 }));
    // Different length either way.
    try testing.expect(!st.matches(&.{"/a.safetensors"}, &.{1.0}));
    try testing.expect(!st.matches(&.{ "/a.safetensors", "/b.safetensors", "/c.safetensors" }, &.{ 1.0, 0.5, 1.0 }));
}

test "prepTransposedThird slices a fused up-projection into the right q/k/v third" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Fused B [out_fused=6, r=1]: rows 0-1 = "q" third, 2-3 = "k" third,
    // 4-5 = "v" third. Transposed+bf16, third `idx` should end up as
    // bt [1, 2] holding exactly rows [2*idx, 2*idx+1] of the source.
    const bv = [_]f32{ 1, 2, 10, 20, 100, 200 };
    const bs = [_]c_int{ 6, 1 };
    const b = mlx.mlx_array_new_data(&bv, &bs, 2, .float32);
    defer _ = mlx.mlx_array_free(b);

    const bt1 = try prepTransposedThird(b, 1, s);
    defer _ = mlx.mlx_array_free(bt1);
    _ = mlx.mlx_array_eval(bt1);
    const shape = mlx.getShape(bt1);
    try testing.expectEqual(@as(c_int, 1), shape[0]);
    try testing.expectEqual(@as(c_int, 2), shape[1]);
    // bf16 has limited precision but exactly represents small integers.
    var f32_bt1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_bt1);
    try mlx.check(mlx.mlx_astype(&f32_bt1, bt1, .float32, s));
    _ = mlx.mlx_array_eval(f32_bt1);
    const d = mlx.mlx_array_data_float32(f32_bt1) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 10), d[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 20), d[1], 1e-4);
}

test "loadFile: rank-divisible-by-3 fused source shares full-rank A across every target (regression)" {
    // A previous version guessed, from `rank % 3 == 0` alone, that a fused
    // source packs three INDEPENDENT rank-r/3 adapters block-diagonally,
    // and split A's rank axis into thirds to match. That guess has no real
    // signal behind it — ordinary shared ranks like 3/24/48/96 hit it just
    // as often as an actual block-diagonal export — and it silently fed the
    // DiT a corrupted 1/3-rank delta instead of erroring. r=3 here is
    // exactly the case that used to trigger the wrong guess; the fix is
    // that A is now ALWAYS shared, full-rank, across every fused target —
    // only B's output rows split by q/k/v third.
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    // `realPath` writes into the buffer and returns the BYTE COUNT, not a
    // slice — the sibling tests below already slice it this way, and passing
    // the usize straight to `{s}` is a compile error, not a runtime one.
    const root_len = try tmp.dir.realPath(io, &pbuf);
    const path = try std.fmt.allocPrint(a, "{s}/fused.safetensors", .{pbuf[0..root_len]});
    defer a.free(path);

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const tmap = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tmap);
    const in_dim = 8;
    const out_fused = 12; // 3 * out(4)
    const r = 3; // divisible by 3 -> used to (wrongly) trigger the split-A guess
    var av: [r * in_dim]f32 = undefined;
    for (&av, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;
    var bv: [out_fused * r]f32 = undefined;
    @memset(&bv, 0.1);
    const ash = [_]c_int{ r, in_dim };
    const bsh = [_]c_int{ out_fused, r };
    const aarr = mlx.mlx_array_new_data(&av, &ash, 2, .float32);
    const barr = mlx.mlx_array_new_data(&bv, &bsh, 2, .float32);
    _ = mlx.mlx_map_string_to_array_insert(tmap, "double_blocks.0.img_attn.qkv.lora_A.weight", aarr);
    _ = mlx.mlx_map_string_to_array_insert(tmap, "double_blocks.0.img_attn.qkv.lora_B.weight", barr);
    _ = mlx.mlx_array_free(aarr);
    _ = mlx.mlx_array_free(barr);
    const mmap = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(mmap);
    const pathz = try a.dupeSentinel(u8, path, 0);
    defer a.free(pathz);
    try mlx.check(mlx.mlx_save_safetensors(pathz, tmap, mmap));

    var file = try loadFile(a, path, .flux2);
    defer file.deinit();
    try testing.expect(file.entries.len == 3); // fans out to q/k/v

    // Reference: A transposed+bf16 the same way loadFile does, to compare
    // byte-for-byte against every entry's `at`.
    var full_at = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(full_at);
    {
        const ref_arr = mlx.mlx_array_new_data(&av, &ash, 2, .float32);
        defer _ = mlx.mlx_array_free(ref_arr);
        const axes = [_]c_int{ 1, 0 };
        var tr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tr);
        try mlx.check(mlx.mlx_transpose_axes(&tr, ref_arr, &axes, 2, s));
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, tr, false, s));
        try mlx.check(mlx.mlx_astype(&full_at, c, .bfloat16, s));
    }
    _ = mlx.mlx_array_eval(full_at);
    const full_at_shape = mlx.getShape(full_at); // [in, r] — NOT [in, r/3]

    for (file.entries) |ent| {
        const at_shape = mlx.getShape(ent.at);
        const bt_shape = mlx.getShape(ent.bt); // [r, out/3]
        // A is untouched: every target gets the SAME full-rank [in, r].
        try testing.expectEqual(full_at_shape[0], at_shape[0]);
        try testing.expectEqual(full_at_shape[1], at_shape[1]);
        try testing.expectEqual(@as(c_int, r), at_shape[1]);
        // Contracts cleanly against the full (unsplit) rank on both sides.
        try testing.expectEqual(at_shape[1], bt_shape[0]);
    }
}

const MetaPair = struct { key: [*:0]const u8, value: [*:0]const u8 };

/// Write a one-module rank-`r` adapter to `path` carrying the given
/// `__metadata__` pairs (plus, optionally, a kohya per-module `.alpha`
/// tensor), then return the scale `loadFile` resolves for it.
fn testLoraScale(
    a: std.mem.Allocator,
    path: []const u8,
    r: c_int,
    meta: []const MetaPair,
    module_alpha: ?f32,
) !f32 {
    const in_dim: c_int = 64;
    const out_dim: c_int = 64;
    const av = try a.alloc(f32, @intCast(r * in_dim));
    defer a.free(av);
    @memset(av, 0.01);
    const bv = try a.alloc(f32, @intCast(out_dim * r));
    defer a.free(bv);
    @memset(bv, 0.02);
    const ash = [_]c_int{ r, in_dim };
    const bsh = [_]c_int{ out_dim, r };

    const tmap = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tmap);
    const aarr = mlx.mlx_array_new_data(av.ptr, &ash, 2, .float32);
    defer _ = mlx.mlx_array_free(aarr);
    const barr = mlx.mlx_array_new_data(bv.ptr, &bsh, 2, .float32);
    defer _ = mlx.mlx_array_free(barr);
    _ = mlx.mlx_map_string_to_array_insert(tmap, "transformer.transformer_blocks.0.attn.to_q.lora_A.weight", aarr);
    _ = mlx.mlx_map_string_to_array_insert(tmap, "transformer.transformer_blocks.0.attn.to_q.lora_B.weight", barr);
    if (module_alpha) |ma| {
        const mv = [_]f32{ma};
        const msh = [_]c_int{1};
        const marr = mlx.mlx_array_new_data(&mv, &msh, 1, .float32);
        defer _ = mlx.mlx_array_free(marr);
        _ = mlx.mlx_map_string_to_array_insert(tmap, "transformer.transformer_blocks.0.attn.to_q.alpha", marr);
    }

    const mmap = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(mmap);
    for (meta) |kv| _ = mlx.mlx_map_string_to_string_insert(mmap, kv.key, kv.value);

    const pathz = try a.dupeSentinel(u8, path, 0);
    defer a.free(pathz);
    try mlx.check(mlx.mlx_save_safetensors(pathz, tmap, mmap));

    var file = try loadFile(a, path, .flux2);
    defer file.deinit();
    try testing.expectEqual(@as(usize, 1), file.entries.len);
    return file.entries[0].scale;
}

test "loadFile: the file's declared alpha wins, wherever the exporter put it" {
    // The whole reason this exists: four real community adapters, four
    // different homes for the alpha/rank pair. Reading only the kohya
    // per-module `.alpha` TENSOR ran every PEFT/diffusers export at 1.0 —
    // rank/alpha = 8x too strong for the common alpha 4 / r 32 pairing, which
    // renders pure noise, not a stronger style. No `want` here is 1.0 except
    // where 1.0 is the correct answer, so none of these can pass vacuously.
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &pbuf);
    const root = pbuf[0..root_len];

    const Case = struct { name: []const u8, meta: []const MetaPair, want: f32 };
    const cases = [_]Case{
        // linoyts klein-9B: a JSON DOCUMENT stored as ONE metadata string,
        // its keys prefixed with the pipeline component.
        .{ .name = "peft-nested", .want = 0.125, .meta = &.{
            .{ .key = "format", .value = "pt" },
            .{ .key = "lora_adapter_metadata", .value =
                \\{
                \\  "transformer.alpha_pattern": {},
                \\  "transformer.lora_alpha": 4,
                \\  "transformer.r": 32,
                \\  "transformer.rank_pattern": {},
                \\  "transformer.revision": null
                \\}
            },
        } },
        // gokaygokay Krea-2 Realism: flat pairs straight in __metadata__.
        .{ .name = "flat-lora_rank", .want = 0.5, .meta = &.{
            .{ .key = "lora_alpha", .value = "16" },
            .{ .key = "lora_rank", .value = "32" },
        } },
        // Same shape, PEFT's shorter spelling of the rank.
        .{ .name = "flat-r", .want = 0.25, .meta = &.{
            .{ .key = "lora_alpha", .value = "8" },
            .{ .key = "r", .value = "32" },
        } },
        // kohya's training metadata (its per-module `.alpha` tensor is the
        // more specific form and is covered by the precedence test below).
        .{ .name = "kohya-ss", .want = 0.25, .meta = &.{
            .{ .key = "ss_network_alpha", .value = "8.0" },
            .{ .key = "ss_network_dim", .value = "32" },
        } },
        // krea darkbrush: no __metadata__ at all.
        .{ .name = "none", .want = 1.0, .meta = &.{} },
        // Norod78 ai-toolkit: metadata, but no alpha in it (baked into the
        // weights) — must stay 1.0 rather than invent a scale.
        .{ .name = "no-alpha-key", .want = 1.0, .meta = &.{
            .{ .key = "format", .value = "pt" },
            .{ .key = "modelspec.title", .value = "vintage book cover" },
        } },
        // Guard rails: a scale we cannot trust is no scale. Silently running
        // an adapter at the wrong strength is exactly the bug being fixed.
        .{ .name = "zero-rank", .want = 1.0, .meta = &.{
            .{ .key = "lora_alpha", .value = "16" },
            .{ .key = "lora_rank", .value = "0" },
        } },
        .{ .name = "negative-alpha", .want = 1.0, .meta = &.{
            .{ .key = "lora_alpha", .value = "-4" },
            .{ .key = "lora_rank", .value = "32" },
        } },
        .{ .name = "not-a-number", .want = 1.0, .meta = &.{
            .{ .key = "lora_alpha", .value = "auto" },
            .{ .key = "lora_rank", .value = "32" },
        } },
        .{ .name = "malformed-json", .want = 1.0, .meta = &.{
            .{ .key = "lora_adapter_metadata", .value = "{not json" },
        } },
        // A nested doc that parses but declares no usable pair must fall
        // THROUGH to the flat keys, not stop the search at step 1.
        .{ .name = "nested-empty-falls-through", .want = 0.5, .meta = &.{
            .{ .key = "lora_adapter_metadata", .value = "{\"transformer.bias\": \"none\"}" },
            .{ .key = "lora_alpha", .value = "16" },
            .{ .key = "lora_rank", .value = "32" },
        } },
    };

    for (cases, 0..) |c, i| {
        const path = try std.fmt.allocPrint(a, "{s}/alpha-{d}.safetensors", .{ root, i });
        defer a.free(path);
        const got = try testLoraScale(a, path, 32, c.meta, null);
        testing.expectApproxEqAbs(c.want, got, 1e-6) catch |err| {
            std.debug.print("alpha case '{s}': want {d}, got {d}\n", .{ c.name, c.want, got });
            return err;
        };
    }
}

test "loadFile: a per-module .alpha tensor outranks the file-level metadata" {
    // Per-module beats file-wide because it is more specific — a kohya export
    // that carries both means the tensor for THIS module, and the metadata as
    // the default for modules that ship none.
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &pbuf);
    const path = try std.fmt.allocPrint(a, "{s}/both.safetensors", .{pbuf[0..root_len]});
    defer a.free(path);

    const meta = [_]MetaPair{
        .{ .key = "lora_alpha", .value = "4" },
        .{ .key = "lora_rank", .value = "32" }, // file-level would be 0.125
    };
    const got = try testLoraScale(a, path, 32, &meta, 16.0); // 16/32
    try testing.expectApproxEqAbs(@as(f32, 0.5), got, 1e-6);
}

test "canonicalize: a flat key naming our OWN module resolves ONCE, never a split alias" {
    // Live defect in the merged branch: flat mode matched a row's flattened
    // CANONICAL as well as its flattened alias, so a Kohya export spelling
    // mlx-serve's own module name matched the self row AND every alias row
    // that resolves to it — for a fused-QKV target that second match carries
    // a `.third*` split, i.e. an entry whose B is a THIRD of the right width.
    // `File.find` returns the first entry, so the correct one happened to win
    // and nothing failed loudly; the junk entry just sat there consuming a
    // fan-out slot and inflating `countLoraMatches`. Every canonical has a
    // self row (pinned by the invariant test below), so matching aliases
    // alone is complete — the fallback bought nothing and cost this.
    var bufs: [MAX_FANOUT]CanonBuf = undefined;
    var out: [MAX_FANOUT]CanonMatch = undefined;

    const own = canonicalize("transformer_blocks_3_attn_to_q", true, .flux2, &bufs, &out);
    try testing.expectEqual(@as(usize, 1), own.len);
    try testing.expectEqualStrings("transformer_blocks.3.attn.to_q", own[0].canon);
    try testing.expectEqual(Split.none, own[0].split);

    // A global with two alias rows pointing at one canonical: also once.
    const glob = canonicalize("x_embedder", true, .flux2, &bufs, &out);
    try testing.expectEqual(@as(usize, 1), glob.len);
    try testing.expectEqualStrings("x_embedder", glob[0].canon);

    // The fan-out that MUST still happen: a flat fused-QKV source.
    const fused = canonicalize("double_blocks_3_img_attn_qkv", true, .flux2, &bufs, &out);
    try testing.expectEqual(@as(usize, 3), fused.len);
    try testing.expectEqual(Split.third0, fused[0].split);
    try testing.expectEqual(Split.third2, fused[2].split);
}

test "arch tables: every canonical target has a SELF row" {
    // The invariant that makes flat-mode alias-only matching complete: a file
    // that already speaks mlx-serve's own naming (dotted OR flattened) is
    // recognized because the canonical name is also listed as an alias. A new
    // row added without its self row would silently stop resolving Kohya
    // exports for that module.
    inline for (.{ Arch.flux2, Arch.krea2, Arch.minimax_h3 }) |arch| {
        const table = archTable(arch);
        for (table) |row| {
            var found = false;
            for (table) |other| {
                if (std.mem.eql(u8, other.canonical, row.canonical) and
                    std.mem.eql(u8, other.alias, row.canonical) and other.split == .none)
                {
                    found = true;
                    break;
                }
            }
            if (!found) std.debug.print("no self row for canonical: {s}\n", .{row.canonical});
            try testing.expect(found);
        }
    }
}

test "canonicalize resolves Kohya flat-scheme module names, not just dotted ones" {
    // Dotted BFL-style key, e.g. from a diffusers/BFL export.
    var bufs: [MAX_FANOUT]CanonBuf = undefined;
    var out: [MAX_FANOUT]CanonMatch = undefined;
    const dotted = canonicalize("double_blocks.3.img_attn.qkv", false, .flux2, &bufs, &out);
    try testing.expectEqual(@as(usize, 3), dotted.len); // fused qkv fans out to q/k/v
    try testing.expectEqualStrings("transformer_blocks.3.attn.to_q", dotted[0].canon);

    // The same tensor, but as it actually appears in a Kohya `lora_unet_...`
    // flat export: dots replaced with underscores in the module portion.
    const flat = canonicalize("double_blocks_3_img_attn_qkv", true, .flux2, &bufs, &out);
    try testing.expectEqual(@as(usize, 3), flat.len);
    try testing.expectEqualStrings("transformer_blocks.3.attn.to_q", flat[0].canon);
}

test "loadFile rejects relative/empty paths (openFileAbsolute UB class)" {
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, "", .generic));
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, "rel/lora.safetensors", .generic));
}

test "loadFile rejects a MISSING file before mlx can kill the process" {
    // Live: `{"lora_path":"/tmp/nope.safetensors"}` printed
    // `MLX error: [load_safetensors] Failed to open file` and the server was
    // GONE — the client got a dropped connection, not a 400. mlx-c errors are
    // fatal, so a nonexistent path must never reach `mlx_load_safetensors`.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const missing = try std.fmt.allocPrint(testing.allocator, "{s}/definitely-not-here.safetensors", .{buf[0..root_len]});
    defer testing.allocator.free(missing);
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, missing, .generic));
    // A DIRECTORY exists, so an existence check alone would wave it through to
    // mlx and die the same way. It must be a regular file.
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, buf[0..root_len], .generic));
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, "rel/lora.safetensors", .generic));
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, "", .generic));
}
