//! The quant plugin table (`QuantTable`): a format claims a config's quantization block, validates and bills it,
//! loads its tensors and banks from a weight source (`Source`), runs its matmul and MoE dispatch, registers its
//! kernels. The built-in formats (`src/sdk/quant/`) implement the same table.

const Host = @import("host.zig").Host;
const std = @import("std");
const common = @import("common.zig");

pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;
pub const Array = common.Array;
pub const Stream = common.Stream;
pub const Claim = common.Claim;
pub const PackDir = common.PackDir;

pub const version: u32 = 1;

/// Where a format reads a pack's tensors: the resident source (`sdk/weights/resident.zig`) or the streamed one.
pub const Source = extern struct {
    size: u32 = @sizeOf(Source),
    ctx: ?*anyopaque,
    /// The tensor `name` (the full key, `<base>.weight`) as the source holds it, borrowed for the source's life; an
    /// empty array when the pack has none.
    get: *const fn (ctx: ?*anyopaque, name: Str) callconv(.c) Array,
    /// `ctx` is an SDK `Resident` whose map an in-tree model may take (`Resident.take`). A flag, not `get`'s address:
    /// every compile unit holds its own copy of the function.
    resident: bool = false,

    pub fn tensor(self: *const Source, name: []const u8) Array {
        return self.get(self.ctx, Str.of(name));
    }
};

/// A loaded weight in the format's own layout. It owns every array in it; `QuantTable.release` frees them.
pub const Weight = extern struct {
    size: u32 = @sizeOf(Weight),
    w: Array = .{},
    scales: Array = .{},
    biases: Array = .{},
    /// A side tensor the matmul reads (Hadamard: the sign vector).
    side: Array = .{},
    /// Resolved per weight: a tensor may differ from its block's declaration (mixed-precision packs).
    bits: u8 = 0,
    group_size: u32 = 0,
    /// The format's own layout tag (the MLX formats: dense, an MLX mode, row-scaled e4m3).
    layout: u8 = 0,
    /// The format's own flags (Hadamard: the biases are exactly -scales).
    flags: u8 = 0,
    /// A width the format reads beside the arrays (Hadamard: the rotation block).
    block: u32 = 0,
    /// The format's own state for this weight (a codebook, a bank's rate groups).
    state: ?*anyopaque = null,
};

pub const LoadCtx = extern struct {
    size: u32 = @sizeOf(LoadCtx),
    model_dir: Str = .{},
    dir: ?*const PackDir = null,
    /// The claimed block (`quantization`, `quantization_config`, `expert_quant`) as JSON.
    block_json: Str,
    /// The pack's whole config.json: a format may read a marker beside its block.
    config_json: Str = .{},
    source: *const Source,
    stream: Stream = .{},
    host: ?*const Host = null,
    /// The format's per-load state: set by `open`, read by the loads, freed by `close`.
    state: ?*anyopaque = null,
};

/// A quant plugin's table, exported as `exportName(name, version)`. `claims`, `validate` and `bill` never reach MLX.
/// A load runs `open`, then any number of `load_tensor`/`load_bank`, then `close`; the weights outlive `close`.
pub const QuantTable = extern struct {
    size: u32 = @sizeOf(QuantTable),
    version: u32 = version,
    name: Str,
    /// Claims one quantization block of `config_json`; a marker beside the block outranks the block's own mode.
    claims: *const fn (block_json: Str, config_json: Str) callconv(.c) Claim,
    /// Refuses by name a block the format cannot serve (an unknown mode, a width MLX has no kernels for).
    validate: *const fn (block_json: Str, config_json: Str, dir: *const PackDir) callconv(.c) Verdict,
    /// Resident bytes of the pack once loaded, given the `stored` bytes of the shards it loads.
    bill: *const fn (block_json: Str, model_dir: Str, stored: u64) callconv(.c) u64,
    open: *const fn (ctx: *LoadCtx) callconv(.c) Verdict,
    close: *const fn (ctx: *LoadCtx) callconv(.c) void,
    /// The projection `<name>.*`; `in_dim` (the activation's inner dim, 0 = unknown) pins its quant params exactly.
    load_tensor: *const fn (ctx: *const LoadCtx, name: Str, in_dim: u32, out: *Weight) callconv(.c) Verdict,
    /// Routed experts stacked under `<name>.*` (mlx-lm's `switch_mlp` layout), one bank.
    load_bank: *const fn (ctx: *const LoadCtx, name: Str, in_dim: u32, out: *Weight) callconv(.c) Verdict,
    release: *const fn (w: *Weight) callconv(.c) void,
    /// `x @ W^T`; `out` receives a new array.
    matmul: *const fn (w: *const Weight, x: Array, s: Stream, out: *Array) callconv(.c) Verdict,
    /// Routed experts: `x` through the bank rows `rhs` selects (`lhs` the gathered input rows, or empty; `sorted` when
    /// `rhs` is sorted). `out` receives a new array.
    moe: *const fn (bank: *const Weight, x: Array, lhs: Array, rhs: Array, sorted: bool, s: Stream, out: *Array) callconv(.c) Verdict,
    /// Registers the format's kernels with the kernel contract (P2 step 6).
    register_kernels: ?*const fn (host: ?*const Host) callconv(.c) Verdict = null,
    /// Routed SwiGLU experts in one dispatch, for a format whose kernels fuse the three projections (EXL3): `x`
    /// [B, S, D] through the banks' experts `inds` [B, S, K], weighted by `scores`. Null: compose `moe` per projection.
    swiglu: ?*const fn (gate: *const Weight, up: *const Weight, down: *const Weight, x: Array, inds: Array, scores: Array, opts: *const Swiglu, s: Stream, out: *Array) callconv(.c) Verdict = null,
};

pub const Swiglu = extern struct {
    size: u32 = @sizeOf(Swiglu),
    /// > 0: the SwiGLU's clamp (GLM's `swiglu_limit`).
    limit: i32 = 0,
    /// Spec-verify rows: the decode chain at any width.
    verify_rows: bool = false,
};

// ── tests ──

const testing = std.testing;

test "abi quant: every struct opens with its size" {
    const w: Weight = .{};
    try testing.expectEqual(@as(u32, @sizeOf(Weight)), w.size);
    inline for (.{ Source, Weight, LoadCtx, QuantTable, Swiglu }) |T|
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
}
