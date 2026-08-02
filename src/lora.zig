//! Runtime (unfused) LoRA adapters for the image backends (FLUX.2 / Krea-2).
//!
//! A LoRA safetensors file carries pairs `<module>.lora_A.weight` [r,in] /
//! `<module>.lora_B.weight` [out,r] (diffusers naming; `lora_down`/`lora_up`,
//! the dotted `lora.down`/`lora.up` variant, and a `.default.` PEFT-adapter
//! infix are all accepted) plus an optional scalar `<module>.alpha` (net
//! scale alpha/rank, kohya convention). Adapters are NOT fused into the base
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
//! `LoraTransforms.split_q_up`/`split_q_down` for the very same tensors).
//! Down-projections are shared across a fused split unless their rank
//! happens to divide evenly by the fan-out (mirrors mflux exactly).
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

pub const Arch = enum { flux2, krea2, generic };

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

fn archTable(arch: Arch) []const AliasRow {
    return switch (arch) {
        .flux2 => &flux2_table,
        .krea2 => &krea2_table,
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
pub fn canonicalize(module: []const u8, flat: bool, arch: Arch, bufs: *[MAX_FANOUT]CanonBuf) []CanonMatch {
    var out: [MAX_FANOUT]CanonMatch = undefined;
    var n: usize = 0;
    var fbuf_alias: [128]u8 = undefined;
    var fbuf_canon: [128]u8 = undefined;
    for (archTable(arch)) |row| {
        if (n >= MAX_FANOUT) break;
        const block = if (flat) blk: {
            const fa = flattenInto(&fbuf_alias, row.alias);
            if (matchTemplate(module, fa)) |b| break :blk b;
            const fc = flattenInto(&fbuf_canon, row.canonical);
            if (matchTemplate(module, fc)) |b| break :blk b;
            continue;
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
};

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
pub fn loadFile(allocator: std.mem.Allocator, path: []const u8, arch: Arch) !File {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return error.BadLoraPath;
    // Prove the file is there BEFORE mlx sees it. `mlx_load_safetensors` on a
    // missing path raises an MLX error, and an MLX error is not a Zig error —
    // it kills the process. One request with a stale `lora_path` (a moved
    // adapter, a typo) took the whole server down, mid-conversation, for every
    // other client too. Same class as the non-text-model 400-before-prefill
    // rule: validate on OUR side of an uncatchable boundary.
    {
        const io = std.Io.Threaded.global_single_threaded.io();
        const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.BadLoraPath;
        defer f.close(io);
        const st = f.stat(io) catch return error.BadLoraPath;
        if (st.kind != .file) return error.BadLoraPath; // a directory opens fine
    }
    const pathz = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(pathz);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, pathz, s));

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
            gop.value_ptr.* = .{};
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
        const rank: c_int = mlx.getShape(p.a)[0]; // A [r,in]
        const at = try prepTransposed(p.a, s);
        errdefer _ = mlx.mlx_array_free(at);
        const bt = try prepTransposed(p.b, s);
        errdefer _ = mlx.mlx_array_free(bt);
        const scale: f32 = if (p.alpha) |al| al / @as(f32, @floatFromInt(rank)) else 1.0;
        
        // Canonicalize the module name for the target architecture
        var canon_bufs: [MAX_FANOUT]CanonBuf = undefined;
        const matches = canonicalize(e.key_ptr.*, false, arch, &canon_bufs);
        // For simple 1:1 mappings (including bypass LoRAs), use the first canonical name
        try entries.append(allocator, .{
            .module = try allocator.dupe(u8, matches[0].canon),
            .at = at,
            .bt = bt,
            .scale = scale,
        });
    }
    return .{ .allocator = allocator, .entries = try entries.toOwnedSlice(allocator) };
}

/// [o,i] → materialized bf16 [i,o].
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
