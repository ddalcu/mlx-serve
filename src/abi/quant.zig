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
    /// The slot rows per routed layer `budget` buys over the pack's streamed expert bank, no memory taken (`open_bank`'s
    /// rows); refused by name when none fit. Null: the format streams no bank.
    bank_rows: ?*const fn (ctx: *const LoadCtx, geo: *const BankGeometry, budget: BankBudget, out: *BankRows) callconv(.c) Verdict = null,
    /// The pack's streamed expert bank at the rows `budget` buys; it outlives `close`.
    open_bank: ?*const fn (ctx: *const LoadCtx, geo: *const BankGeometry, budget: BankBudget, out: *Bank) callconv(.c) Verdict = null,
    /// `swiglu` reads `Swiglu.limit_out`. Appended: the host passes a post-SiLU limit only to a table whose `size`
    /// reaches this flag and sets it (`clampsOut`), and refuses the call by name otherwise, so a table built before the
    /// field is never asked a clamp it would skip.
    swiglu_limit_out: bool = false,
    /// `matmul` over up to this many rows gives each row the bytes of its own one-row call; 0 = not declared, so a host
    /// that needs a row's one-row bytes calls one row at a time. Appended: read through `exactRows`.
    exact_rows: u32 = 0,

    /// Whether the table's `swiglu` honours `Swiglu.limit_out` (a table of any size).
    pub fn clampsOut(t: *const QuantTable) bool {
        return t.size >= @offsetOf(QuantTable, "swiglu_limit_out") + @sizeOf(bool) and t.swiglu_limit_out;
    }

    /// `exact_rows`, 0 on a table whose `size` ends before it.
    pub fn exactRows(t: *const QuantTable) u32 {
        return if (t.size >= @offsetOf(QuantTable, "exact_rows") + @sizeOf(u32)) t.exact_rows else 0;
    }

    /// The smallest table the loader admits: one built before `swiglu_limit_out`, whose appended members read as absent
    /// through their size-gated readers (`clampsOut`).
    pub const min_size: u32 = @offsetOf(QuantTable, "swiglu_limit_out");
};

/// A layer's router as the trunk loaded it. The `gates` list is valid for the call only; the arrays it names live
/// with the trunk, which outlives the bank.
pub const BankGate = extern struct { w: Array = .{}, bias: Array = .{} };

/// The trunk's routed MoE a streamed bank serves.
pub const BankGeometry = extern struct {
    size: u32 = @sizeOf(BankGeometry),
    hidden: u32,
    inter: u32,
    top_k: u32,
    n_experts: u32,
    /// The trunk's layers: `gates` and a call's `layer` index them alike.
    n_layers: u32,
    /// > 0: the SwiGLU's clamp.
    swiglu_limit: f32 = 0,
    /// Every layer's router (`n_layers`); null for `bank_rows`.
    gates: ?[*]const BankGate = null,
    /// Every layer's routed SwiGLU clamped AFTER the SiLU (`n_layers`, as `gates`; valid for the call only): > 0 the
    /// limit L of min(silu(g), L) * clamp(u, -L, L) (Ling-3.1's `expert_swiglu_limit_list`), 0 none; null: no layer
    /// clamps so. Appended: read it through `limitOut`, which a geometry from a host built before it (a smaller `size`)
    /// answers 0, so a plugin built with it loads on an older host, and a plugin built before it loads as before.
    limits_out: ?[*]const f32 = null,

    /// Layer `layer`'s post-SiLU limit (`limits_out`), 0 where the geometry does not reach the field or gives none.
    pub fn limitOut(g: *const BankGeometry, layer: u32) f32 {
        if (g.size < @offsetOf(BankGeometry, "limits_out") + @sizeOf(?[*]const f32)) return 0;
        const l = g.limits_out orelse return 0;
        return if (layer < g.n_layers) l[layer] else 0;
    }
};

/// Bytes the slot rows and the bank's own buffers may hold in each phase (a prompt pass, decode).
pub const BankBudget = extern struct { prompt: u64, decode: u64 };

pub const BankRows = extern struct {
    size: u32 = @sizeOf(BankRows),
    prompt: u32 = 0,
    decode: u32 = 0,
    /// One slot row in every layer.
    row_bytes: u64 = 0,
};

/// A streamed bank of routed experts, as the trunk's MoE reads one whose math it does not have (`sdk/weights/routed.zig`
/// `Routed`, its `source` arm): a call returns the routed sum (`[n, hidden]` f32, a new array) of the normed input `x`
/// over the routes `idx` weighted by `wts`. Every entry takes `ctx`; a refusal names its error.
pub const Bank = extern struct {
    size: u32 = @sizeOf(Bank),
    ctx: ?*anyopaque = null,
    close: *const fn (ctx: ?*anyopaque) callconv(.c) void,
    /// The load line's account of the bank, valid until `close`.
    about: Str = .{},
    /// Routed ids a decode-phase call takes: a wider call is a prompt pass, which the bank refuses in decode.
    decode_ids: u32,
    /// Every `layer` below is the trunk's layer index (`BankGeometry.n_layers`).
    rows: *const fn (ctx: ?*anyopaque, layer: u32, x: Array, idx: Array, wts: Array, out: *Array) callconv(.c) Verdict,
    /// After the eval that consumed a forward's calls.
    settle: *const fn (ctx: ?*anyopaque) callconv(.c) Verdict,
    /// Before each forward: a prompt pass or decode (the slot rows follow it).
    phase: *const fn (ctx: ?*anyopaque, prompt: bool) callconv(.c) Verdict,
    /// A prompt pass's predicted routes of `layer` (every row's ids), read while its attention runs.
    read_ahead: *const fn (ctx: ?*anyopaque, layer: u32, ids: [*]const u16, n: usize) callconv(.c) Verdict,
    /// The fused forms the trunk takes where offered (`Routed`'s of the same names); null: not offered.
    rows_shared: ?*const fn (ctx: ?*anyopaque, layer: u32, x: Array, idx: Array, wts: Array, shared: Array, out: *Array) callconv(.c) Verdict = null,
    rows_expand: ?*const fn (ctx: ?*anyopaque, layer: u32, x: Array, idx: Array, wts: Array, shared: Array, res: Array, pc: Array, out: *Array) callconv(.c) Verdict = null,
    rows_expand_into: ?*const fn (ctx: ?*anyopaque, layer: u32, x: Array, idx: Array, wts: Array, shared: Array, res: Array, pc: Array, out: *Array) callconv(.c) Verdict = null,
    rows_decode_expand: ?*const fn (ctx: ?*anyopaque, layer: u32, x: Array, idx: Array, wts: Array, shared: Array, res: Array, pc: Array, out: *Array) callconv(.c) Verdict = null,
    next_ahead: ?*const fn (ctx: ?*anyopaque, layer: u32, ids: Array) callconv(.c) Verdict = null,
    /// The bank's lines for the request's summary (`[stream-stats]` by phase, newline-separated), valid until the next
    /// call; the model logs them beside `[spec-stats]` at the request's end. Null: none.
    meters: ?*const fn (ctx: ?*anyopaque, out: *Str) callconv(.c) void = null,
    /// A decode-phase `rows` call (at most `decode_ids` ids) gives each row the bytes that row's call alone gets.
    /// Appended: read only where `size` reaches it.
    exact_rows: bool = false,
};

pub const Swiglu = extern struct {
    size: u32 = @sizeOf(Swiglu),
    /// > 0: the SwiGLU's clamp (GLM's `swiglu_limit`).
    limit: i32 = 0,
    /// Spec-verify rows: the decode chain at any width.
    verify_rows: bool = false,
    /// > 0: the SwiGLU clamped AFTER the SiLU, min(silu(g), L) * clamp(u, -L, L) (Ling-3.1's; `limit` is then 0).
    /// Appended: read only by a table that declares it (`QuantTable.clampsOut`).
    limit_out: f32 = 0,
};

// ── tests ──

const testing = std.testing;

test "abi quant: every struct opens with its size" {
    const w: Weight = .{};
    try testing.expectEqual(@as(u32, @sizeOf(Weight)), w.size);
    inline for (.{ Source, Weight, LoadCtx, QuantTable, Swiglu, BankGeometry, BankRows, Bank }) |T|
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
}

test "abi quant: a geometry's post-SiLU limits are read only where its size reaches them" {
    const lim = [_]f32{ 0, 10, 0 };
    var g: BankGeometry = .{ .hidden = 8, .inter = 4, .top_k = 1, .n_experts = 2, .n_layers = 3, .limits_out = &lim };
    try testing.expectEqual(@as(f32, 10), g.limitOut(1));
    try testing.expectEqual(@as(f32, 0), g.limitOut(0));
    try testing.expectEqual(@as(f32, 0), g.limitOut(3));
    // A host built before the field: its geometry ends at `gates`.
    g.size = @offsetOf(BankGeometry, "limits_out");
    try testing.expectEqual(@as(f32, 0), g.limitOut(1));
    g.size = @sizeOf(BankGeometry);
    g.limits_out = null;
    try testing.expectEqual(@as(f32, 0), g.limitOut(1));
}

test "abi quant: a table's row-exact matmul width is read only where its size reaches it" {
    var t: QuantTable = undefined;
    t.size = @sizeOf(QuantTable);
    t.exact_rows = 16;
    try testing.expectEqual(@as(u32, 16), t.exactRows());
    // A table built before the field (at the loader's minimum): not declared.
    t.size = QuantTable.min_size;
    try testing.expectEqual(@as(u32, 0), t.exactRows());
}

test "abi quant: a post-SiLU SwiGLU limit reaches only a table that declares it" {
    var t: QuantTable = undefined;
    t.size = @sizeOf(QuantTable);
    t.swiglu_limit_out = true;
    try testing.expect(t.clampsOut());
    t.swiglu_limit_out = false;
    try testing.expect(!t.clampsOut());
    // A table built before the flag: its size ends before it.
    t.swiglu_limit_out = true;
    t.size = @offsetOf(QuantTable, "swiglu_limit_out");
    try testing.expect(!t.clampsOut());
}
