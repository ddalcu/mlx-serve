//! The model architectures we serve, picked by `general.architecture`. An arch
//! module brings three things: `Hparams.read` (GGUF metadata), `configJson`
//! (the HF config.json mlx-serve parses) and `mapTensor` (GGUF -> HF names).
const std = @import("std");
const gguf = @import("gguf.zig");
pub const qwen35 = @import("arch/qwen35.zig");
pub const gemma4 = @import("arch/gemma4.zig");
pub const gemma3 = @import("arch/gemma3.zig");
pub const lfm2 = @import("arch/lfm2.zig");
pub const gpt_oss = @import("arch/gpt_oss.zig");
pub const nemotron_h = @import("arch/nemotron_h.zig");
pub const qwen3next = @import("arch/qwen3next.zig");
pub const qwen4exp = @import("arch/qwen4exp.zig");
pub const qwen3 = @import("arch/qwen3.zig");
pub const flint = @import("arch/flint.zig");
pub const muse_glimmer = @import("arch/muse_glimmer.zig");
pub const deepseek41 = @import("arch/deepseek41.zig");

/// What loading has to do to a tensor's bytes. Most of them undo llama.cpp's
/// value-head tiling of the GatedDeltaNet, see arch/qwen35.zig.
pub const Transform = union(enum) {
    /// Bytes go through untouched.
    none,
    /// Un-tile value heads along the rows: `unit` rows per head, starting at `start`.
    untile_rows: struct { start: u32, unit: u32 },
    /// F32 vector with one entry per value head.
    untile_vec,
    /// ssm_a = -exp(A_log), per value head.
    a_log,
    /// [channels, kernel] depthwise conv, value channels un-tiled, stored [channels, kernel, 1].
    conv1d,
    /// The INPUT side is tiled and quant blocks straddle heads, so the bytes
    /// stay and the matmul tiles its activations instead.
    tiled_input,
    /// The shared expert's gate, a [hidden] vector in GGUF and a [1, hidden] linear in HF.
    shared_gate,
    /// F32 norm weight stored as HF's weight + 1 (Gemma 3): shift it back.
    minus_one,
    /// Float tensor that HF stores with one more trailing axis of size 1 (LFM2's conv kernel).
    trailing_axis,
    /// F32 `ssm_a = -exp(A_log)` back to A_log, as a flat vector.
    neg_log,
    /// Float tensor as a flat vector (ggml keeps a leading axis of 1 on Mamba's per-head vectors).
    flatten,
    /// Quantized [2 * nk * r, in] with rows [b heads of key head 0, a heads of key head 0, b of 1, ...]:
    /// split into `in_proj_b` and `in_proj_a` of [nk * r, in] (Qwen3-Next's fused b / a projection).
    split_ba: struct { nk: u32, r: u32 },
    /// Q or K rows llama.cpp interleaved per head for its NORM rope (`_unpermute_for_rope`: row `h*hd + 2j + p` holds
    /// HF's `h*hd + p*hd/2 + j`), back to HF's rotate-half order; whole rows move, nothing is requantized.
    rope_rows: u32,
    /// An expert bank `[experts, 2M, in]` holding gate rows then up rows: two banks, the mapped name (`gate_proj`)
    /// and its `up_proj` sibling, whole rows moved.
    split_gate_up,
    /// An F32 tensor kept f32 (every other float lands bf16): DeepSeek-V4.1's hyper-connection mixes, sinks and
    /// router bias, f32 in its MLX packs too (the bias picks experts at near-ties).
    keep_f32,
};

pub const Mapped = struct { name: []const u8, transform: Transform };

pub const Arch = union(enum) {
    qwen35: qwen35.Hparams,
    gemma4: gemma4.Hparams,
    gemma3: gemma3.Hparams,
    lfm2: lfm2.Hparams,
    gpt_oss: gpt_oss.Hparams,
    nemotron_h: nemotron_h.Hparams,
    qwen3next: qwen3next.Hparams,
    qwen4exp: qwen4exp.Hparams,
    qwen3: qwen3.Hparams,
    flint: flint.Hparams,
    muse_glimmer: muse_glimmer.Hparams,
    deepseek41: deepseek41.Hparams,

    /// error.UnsupportedArch for anything not in the table.
    pub fn read(f: *const gguf.File) !Arch {
        const name = f.getString("general.architecture") orelse return error.MissingMetadata;
        if (qwen35.handles(name)) return .{ .qwen35 = try qwen35.Hparams.read(f, name) };
        if (std.mem.eql(u8, name, gemma4.arch_name)) return .{ .gemma4 = try gemma4.Hparams.read(f) };
        if (std.mem.eql(u8, name, gemma3.arch_name)) return .{ .gemma3 = try gemma3.Hparams.read(f) };
        if (lfm2.handles(name)) return .{ .lfm2 = try lfm2.Hparams.read(f, name) };
        if (std.mem.eql(u8, name, gpt_oss.arch_name)) return .{ .gpt_oss = try gpt_oss.Hparams.read(f) };
        if (nemotron_h.handles(name)) return .{ .nemotron_h = try nemotron_h.Hparams.read(f, name) };
        if (std.mem.eql(u8, name, qwen3next.arch_name)) return .{ .qwen3next = try qwen3next.Hparams.read(f) };
        if (std.mem.eql(u8, name, qwen4exp.arch_name)) return .{ .qwen4exp = try qwen4exp.Hparams.read(f) };
        if (qwen3.handles(name)) return .{ .qwen3 = try qwen3.Hparams.read(f, name) };
        if (std.mem.eql(u8, name, flint.arch_name)) return .{ .flint = try flint.Hparams.read(f) };
        if (std.mem.eql(u8, name, muse_glimmer.arch_name)) return .{ .muse_glimmer = try muse_glimmer.Hparams.read(f) };
        if (std.mem.eql(u8, name, deepseek41.arch_name)) return .{ .deepseek41 = try deepseek41.Hparams.read(f) };
        return error.UnsupportedArch;
    }

    /// The config.json an mlx-community conversion of the same model would carry.
    pub fn configJson(self: Arch, allocator: std.mem.Allocator) ![]u8 {
        return switch (self) {
            .qwen35 => |hp| qwen35.configJson(allocator, hp),
            .gemma4 => |hp| gemma4.configJson(allocator, hp),
            .gemma3 => |hp| gemma3.configJson(allocator, hp),
            .lfm2 => |hp| lfm2.configJson(allocator, hp),
            .gpt_oss => |hp| gpt_oss.configJson(allocator, hp),
            .nemotron_h => |hp| nemotron_h.configJson(allocator, hp),
            .qwen3next => |hp| qwen3next.configJson(allocator, hp),
            .qwen4exp => |hp| qwen4exp.configJson(allocator, hp),
            .qwen3 => |hp| qwen3.configJson(allocator, hp),
            .flint => |hp| flint.configJson(allocator, hp),
            .muse_glimmer => |hp| muse_glimmer.configJson(allocator, hp),
            .deepseek41 => |hp| deepseek41.configJson(allocator, hp, null),
        };
    }

    /// HF name + load transform for a GGUF tensor, null when the tensor is
    /// unknown. An empty name (`skip`) = known and not needed.
    pub fn mapTensor(self: Arch, buf: []u8, name: []const u8) ?Mapped {
        return switch (self) {
            .qwen35 => |hp| qwen35.mapTensor(buf, name, hp),
            .gemma4 => |hp| gemma4.mapTensor(buf, name, hp),
            .gemma3 => |hp| gemma3.mapTensor(buf, name, hp),
            .lfm2 => |hp| lfm2.mapTensor(buf, name, hp),
            .gpt_oss => |hp| gpt_oss.mapTensor(buf, name, hp),
            .nemotron_h => |hp| nemotron_h.mapTensor(buf, name, hp),
            .qwen3next => |hp| qwen3next.mapTensor(buf, name, hp),
            .qwen4exp => |hp| qwen4exp.mapTensor(buf, name, hp),
            .qwen3 => |hp| qwen3.mapTensor(buf, name, hp),
            .flint => |hp| flint.mapTensor(buf, name, hp),
            .muse_glimmer => |hp| muse_glimmer.mapTensor(buf, name, hp),
            .deepseek41 => |hp| deepseek41.mapTensor(buf, name, hp),
        };
    }

    /// Whether the loader serves the mapped tensor `name` at `ty`; the affine-coded types load in MLX's affine layout
    /// (`transcodes`). DeepSeek-V4.1's forward reads raw ggml blocks only in its routed-expert gathers, so another
    /// tensor of a type that stays raw is refused here, by name, instead of failing the first request.
    pub fn serves(self: Arch, name: []const u8, ty: @import("quants.zig").GgmlType) bool {
        if (!ty.supported()) return false;
        if (self != .deepseek41 or !ty.isQuantized() or self.transcodes(ty)) return true;
        return std.mem.indexOf(u8, name, ".ffn.experts.") != null;
    }

    /// Whether `ty` loads in an MLX layout and runs on MLX's own kernels: MXFP4 as mxfp4, the affine-coded types as
    /// affine (`affine.zig`). Q8_0's copy grows by its bias (a dense Q8_0 decode is bandwidth-bound), so it keeps its
    /// blocks except where the model's fused kernels are affine-keyed.
    pub fn transcodes(self: Arch, ty: @import("quants.zig").GgmlType) bool {
        if (ty == .mxfp4) return true;
        if (@import("affine.zig").bitsOf(ty) == null) return false;
        return ty != .q8_0 or self == .qwen4exp or self == .deepseek41;
    }

    /// The GatedDeltaNet geometry the un-tiling transforms read (zeroes for an arch that has none).
    pub fn gdn(self: Arch) qwen35.Hparams {
        return switch (self) {
            .qwen35 => |hp| hp,
            .qwen4exp => |hp| hp.base,
            else => std.mem.zeroes(qwen35.Hparams),
        };
    }
};

/// For tensors mlx-serve derives itself (rope frequency tables).
pub const skip = Mapped{ .name = "", .transform = .none };

pub fn u32Key(f: *const gguf.File, key: []const u8) !u32 {
    return std.math.cast(u32, f.getInt(key) orelse return error.MissingMetadata) orelse error.BadMetadata;
}

/// Parses "blk.<n>.<leaf>" into its layer and leaf, null for anything else.
pub fn splitLayer(name: []const u8, n_layers: u32) ?struct { layer: u32, leaf: []const u8 } {
    if (!std.mem.startsWith(u8, name, "blk.")) return null;
    const rest = name[4..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    if (layer >= n_layers) return null;
    return .{ .layer = layer, .leaf = rest[dot + 1 ..] };
}

test "MXFP4 and the affine-coded types transcode for every arch, Q8_0 only where the fused kernels are affine-keyed" {
    const qwen: Arch = .{ .qwen35 = std.mem.zeroes(qwen35.Hparams) };
    const qwen4: Arch = .{ .qwen4exp = std.mem.zeroes(qwen4exp.Hparams) };
    try std.testing.expect(qwen.serves("head.weight", .q2_0) and qwen.serves("head.weight", .iq4_xs) and qwen.serves("head.weight", .q4_k));
    try std.testing.expect(!qwen.serves("head.weight", @fromBackingInt(@intCast(144))));
    try std.testing.expect(qwen.transcodes(.q4_k) and qwen.transcodes(.q2_0) and !qwen.transcodes(.q8_0) and !qwen.transcodes(.iq4_nl));
    try std.testing.expect(qwen4.transcodes(.q8_0) and !qwen4.transcodes(.q6_k) and qwen.transcodes(.mxfp4));
}

test "DeepSeek-V4.1 reads raw ggml blocks only in its routed experts: another tensor of a type that stays raw is refused" {
    const ds: Arch = .{ .deepseek41 = std.mem.zeroes(deepseek41.Hparams) };
    try std.testing.expect(ds.serves("layers.0.ffn.experts.w1.weight", .iq2_xxs) and ds.serves("layers.0.ffn.experts.w2.weight", .q6_k));
    try std.testing.expect(ds.serves("layers.0.attn.wq_a.weight", .q5_k) and ds.serves("layers.0.attn.wq_b.weight", .bf16));
    try std.testing.expect(!ds.serves("head.weight", .q6_k) and !ds.serves("layers.0.ffn.shared_experts.w1.weight", .iq2_xxs));
}

test {
    _ = qwen35;
    _ = gemma4;
    _ = gemma3;
    _ = lfm2;
    _ = gpt_oss;
    _ = nemotron_h;
    _ = qwen3next;
    _ = qwen4exp;
    _ = qwen3;
    _ = flint;
    _ = muse_glimmer;
    _ = deepseek41;
}
