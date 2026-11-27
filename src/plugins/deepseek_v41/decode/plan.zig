//! The one owner of the kernel and plan a decode op runs for a call's rows (`kernelFor`). The plain step and the
//! DSpark verify ask it for every op whose kernel or plan depends on the row count (the table below). Under
//! `MLX_SERVE_DSV41_EXACT` a verify row must give the served plain step's bits at its position, so plain decode stays
//! the reference: the owner gives each verify row the plain step's kernel and plan (`Cover.plain_kernel`), or an R-row
//! plan whose rows a standing byte test holds to the one-row launch (`Cover.r_invariant`). A call site that cannot
//! serve the owner's answer declines it, and the verify counts and logs a miss (`Dsv41Model.exact_misses`): no verify
//! row falls back in silence. Each lever row of `levers.zig` names the op that covers it here (`levers.Row.exact`).
//!
//! The answers depend only on the call (`Facts`, the rows, whether it is an EXACT verify): this file reads no switch
//! and no machine, and holds no state.

const std = @import("std");

/// The most rows a decode call carries (`decode.max_rows`, held equal there).
pub const max_rows = 6;

/// The decode ops whose kernel or plan depends on the row count.
pub const Op = enum { engram, in_proj, wq_b, compressor, indexer, attention, wo_a, wo_b, ffn_in, moe, head };

/// The owner's answer for one call.
pub const Kernel = enum {
    /// The op's served launch over all the call's rows, on the plan its module picks for that row count.
    rows,
    /// Each row on the plain step's one-row kernel: the one-row launch a row at a time, or one launch of a kernel whose
    /// every row is that launch's bytes (MLX's one-row qmv `rowQmv`, wo_a's vector units, WOA_PLAIN's GEMV).
    one_row,
    /// The Engram's slot kernel over GPU row ids (`engram_ids.write`), a row at a time: what the served plain step runs
    /// once the load arms `ENGRAM_GPU_IDS`.
    engram_slots,
};

/// What the plain kernels depend on beyond the rows: the GPU family and what the load armed. The model builds it
/// (`Dsv41Model.planFacts`); the owner reads it and nothing else.
pub const Facts = struct {
    /// A NAX GPU (M5 and newer): the fused decode kernels serve each verify row. Without NAX exact runs the whole-row
    /// path (`blockFusedRows`).
    nax: bool,
    /// The load armed the Engram's GPU row ids and the slot kernel (`ENGRAM_GPU_IDS`, `engram_ids.settle`).
    engram_slots: bool = false,
    /// This call is a plain step whose token is still on the GPU, its Engram ids hashed there (`extendChunk`).
    gpu_ids: bool = false,
};

/// How EXACT covers an op on one GPU family, and the test that holds it.
pub const Cover = union(enum) {
    /// Every R-row plan of the op gives each row its one-row launch's bytes, R = 1..`max_rows`.
    r_invariant: []const u8,
    /// A verify runs each row on the plain step's kernel (`Kernel.one_row`, or `.engram_slots` for the Engram).
    plain_kernel: []const u8,

    pub fn testName(c: Cover) []const u8 {
        return switch (c) {
            inline else => |t| t,
        };
    }
};

pub const Entry = struct {
    op: Op,
    /// What a plain step (one row) runs.
    plain: []const u8,
    /// What a verify of 2..`max_rows` rows runs off EXACT.
    rows: []const u8,
    /// EXACT's cover on a NAX GPU, and without NAX.
    nax: Cover,
    pre_nax: Cover,
};

/// The R-invariance and plain-kernel tests the covers name (`gates.sh`'s exact step checks each is in the suite).
pub const tests = struct {
    pub const engram = "dsv41 exact rows: engram";
    pub const in_proj = "dsv41 exact rows: in_proj";
    pub const wq_b = "dsv41 exact rows: wq_b";
    pub const attention = "dsv41 exact rows: attention";
    pub const wo_a = "dsv41 exact rows: wo_a";
    pub const ffn_in = "dsv41 exact rows: ffn_in";
    pub const moe = "dsv41 exact rows: moe";
    pub const head = "dsv41 exact: the one-row kernel over R rows gives each row's one-row quantized_matmul bits";
    /// The verify block through the fused decode layers against plain steps at every attention band edge and pooling
    /// group (`exact_test.zig`): the stateful ops (compressor, indexer) at the positions where their plans change.
    pub const layers = "dsv41 exact rows: verify layers";
    /// The whole-row gate without NAX (`DSV41_EXACT_GATE`; on NAX it needs force-gpu-family-fallback).
    pub const whole_row = "dsv41 exact: verify rows and the state they leave equal plain steps bit for bit";
};

/// One entry an op, in `Op` order.
pub const table = [_]Entry{
    .{
        .op = .engram,
        .plain = "the slot kernel over GPU-hashed ids (engram_ids.write, ENGRAM_GPU_IDS; its partials cache), else MLX's one-row chain",
        .rows = "MLX's M = R chain (engramApply, the gemv twin), or the kv made behind the draft (SPEC_ENGRAM_AHEAD)",
        .nax = .{ .plain_kernel = tests.engram },
        .pre_nax = .{ .plain_kernel = tests.engram },
    },
    .{
        .op = .in_proj,
        .plain = "dec_in_proj, one launch collapsing the stream",
        .rows = "the same at 2 rows; dec_in_proj_stage then the main launch at 3..6, its mixes a (mix, row) a threadgroup (IN_PROJ_MIXR)",
        .nax = .{ .r_invariant = tests.in_proj },
        .pre_nax = .{ .r_invariant = tests.in_proj },
    },
    .{
        .op = .wq_b,
        .plain = "dec_wq_b, its ring variant writing the window (FUSE_WQB_RING)",
        .rows = "the same launch over the rows' positions",
        .nax = .{ .r_invariant = tests.wq_b },
        .pre_nax = .{ .r_invariant = tests.wq_b },
    },
    .{
        .op = .compressor,
        .plain = "idx_score.pool (compressFused)",
        .rows = "the same, the open group kept for a rollback",
        .nax = .{ .r_invariant = tests.layers },
        .pre_nax = .{ .r_invariant = tests.whole_row },
    },
    .{
        .op = .indexer,
        .plain = "idx_score.scores then topk.select, index_topk picks a row",
        .rows = "the same over the rows, the scores at 320 tile ranges",
        .nax = .{ .r_invariant = tests.layers },
        .pre_nax = .{ .r_invariant = tests.whole_row },
    },
    .{
        .op = .attention,
        .plain = "the shape-cut tile kernel (ATTN_DEC, partialsTile) on NAX; dec_attn without",
        .rows = "the same plan (tilePlan reads no rows)",
        .nax = .{ .r_invariant = tests.attention },
        .pre_nax = .{ .plain_kernel = tests.whole_row },
    },
    .{
        .op = .wo_a,
        .plain = "WOA_PLAIN's merge and plain GEMV; else wo_a's vector units",
        .rows = "the same GEMV; else wo_a on the matrix units from 4 rows (NAX)",
        .nax = .{ .plain_kernel = tests.wo_a },
        .pre_nax = .{ .plain_kernel = tests.whole_row },
    },
    .{
        .op = .wo_b,
        .plain = "ffn_in launch 1 (dec_ffn_in)",
        .rows = "the same, 2 columns a simdgroup at 6 rows, 1 at 5 (FFN_IN_R5_COLS1)",
        .nax = .{ .r_invariant = tests.ffn_in },
        .pre_nax = .{ .plain_kernel = tests.whole_row },
    },
    .{
        .op = .ffn_in,
        .plain = "ffn_in's two launches (the mix tail in its own, FFN_IN_SINK)",
        .rows = "the same, up to 3 rows a router threadgroup",
        .nax = .{ .r_invariant = tests.ffn_in },
        .pre_nax = .{ .r_invariant = tests.whole_row },
    },
    .{
        .op = .moe,
        .plain = "the fused fp4 MoE (moe_fp4: FUSE_DOWN_SCALE, FP4_READS, FP4_DEC_WORD, FP4_UP_SPLIT), or the affine moe_up/moe_down (a GGUF bank is refused under EXACT)",
        .rows = "the same, the down launch in column-block order (MOE_DOWN_ORDER)",
        .nax = .{ .r_invariant = tests.moe },
        .pre_nax = .{ .plain_kernel = tests.whole_row },
    },
    .{
        .op = .head,
        .plain = "the final collapse and norm, then MLX's one-row qmv (HEAD8)",
        .rows = "MLX's M = R kernels (qmv_wide, qmv_mma)",
        .nax = .{ .plain_kernel = tests.head },
        .pre_nax = .{ .plain_kernel = tests.head },
    },
};

pub fn entry(op: Op) *const Entry {
    return &table[@intFromEnum(op)];
}

/// EXACT's cover of `op` on the GPU family `f` names.
pub fn cover(f: Facts, op: Op) Cover {
    return if (f.nax) entry(op).nax else entry(op).pre_nax;
}

/// The kernel `op` runs for a call of `rows` rows: `exact` when the call is an EXACT verify (`Dsv41Model.exactRows`).
/// Off EXACT every op runs its served launch, and a plain step whose ids are on the GPU writes the Engram on the slot
/// kernel. Under EXACT a verify row takes the plain step's kernel: the Engram's slot kernel where the load armed it
/// (else MLX's one-row chain), `one_row` for an op whose cover is the plain kernel, its R-row plan for one whose
/// cover is a standing R-invariance test.
pub fn kernelFor(f: Facts, op: Op, rows: usize, exact: bool) Kernel {
    if (op == .engram) {
        if (exact) return if (f.engram_slots) .engram_slots else .one_row;
        return if (f.engram_slots and f.gpu_ids and rows == 1) .engram_slots else .rows;
    }
    if (!exact) return .rows;
    return switch (cover(f, op)) {
        .r_invariant => .rows,
        .plain_kernel => .one_row,
    };
}

/// Whether a launch over all of a call's `rows` rows serves the answer `k`: `rows`, or `one_row` on a single row.
pub fn rowsServe(k: Kernel, rows: usize) bool {
    return k == .rows or (k == .one_row and rows == 1);
}

// ── tests ──

const testing = std.testing;

test "dsv41 decode plan: one entry an op in Op order, each with what it runs and a named cover on both GPU families" {
    try testing.expectEqual(std.enums.values(Op).len, table.len);
    for (table, 0..) |e, i| {
        try testing.expectEqual(i, @intFromEnum(e.op));
        try testing.expect(e.plain.len > 0 and e.rows.len > 0);
        for ([_]Cover{ e.nax, e.pre_nax }) |c| try testing.expect(std.mem.startsWith(u8, c.testName(), "dsv41 exact"));
    }
}

test "dsv41 decode plan: off EXACT every op runs its served launch and only a GPU-id plain step takes the Engram slot kernel" {
    for ([_]bool{ false, true }) |nax| for (std.enums.values(Op)) |op| for (1..max_rows + 1) |r| {
        const k = kernelFor(.{ .nax = nax, .engram_slots = true, .gpu_ids = true }, op, r, false);
        try testing.expectEqual(if (op == .engram and r == 1) Kernel.engram_slots else Kernel.rows, k);
        try testing.expectEqual(Kernel.rows, kernelFor(.{ .nax = nax, .engram_slots = true }, op, r, false));
        try testing.expectEqual(Kernel.rows, kernelFor(.{ .nax = nax, .gpu_ids = true }, op, r, false));
    };
}

test "dsv41 decode plan: under EXACT a verify row of any width takes the plain step's kernel or an R-invariant plan" {
    for ([_]bool{ false, true }) |nax| for (std.enums.values(Op)) |op| for (1..max_rows + 1) |r| for ([_]bool{ false, true }) |slots| {
        const f: Facts = .{ .nax = nax, .engram_slots = slots };
        const k = kernelFor(f, op, r, true);
        switch (op) {
            // The served plain step's Engram: the slot kernel once the load armed it, whatever the verify's ids.
            .engram => try testing.expectEqual(if (slots) Kernel.engram_slots else Kernel.one_row, k),
            else => try testing.expectEqual(if (cover(f, op) == .r_invariant) Kernel.rows else Kernel.one_row, k),
        }
        // The head always runs each row on MLX's one-row kernel: MLX's quantized matmul sums 2..16 rows otherwise.
        if (op == .head) try testing.expectEqual(Kernel.one_row, k);
    };
}

test "dsv41 decode plan: without NAX the verify tail (attention, wo_a, wo_b, the MoE) runs a row at a time together, as blockFusedRows serves them" {
    const f: Facts = .{ .nax = false };
    for ([_]Op{ .attention, .wo_a, .wo_b, .moe }) |op| try testing.expectEqual(Kernel.one_row, kernelFor(f, op, 4, true));
    // On NAX the fused decode kernels serve the tail at the verify's width; wo_a stays on its plain-step kernel.
    const n: Facts = .{ .nax = true };
    for ([_]Op{ .attention, .wo_b, .ffn_in, .moe, .in_proj, .wq_b, .compressor, .indexer }) |op| try testing.expectEqual(Kernel.rows, kernelFor(n, op, 4, true));
    try testing.expectEqual(Kernel.one_row, kernelFor(n, .wo_a, 4, true));
}
