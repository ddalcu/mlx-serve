//! Memory admission of the expert streamer: how many slot rows each layer may
//! hold in prefill and decode, and the prefill-boundary bound, from the live
//! baseline, before anything is allocated. A port of the DSV4.1 Python stack's
//! admission as the tier runs it: packed_admission.retarget over the static
//! envelope (resolve_admission at base 0) with the phase-memory proof,
//! native_projection_memory.projection_fixed_bounds and the retirement
//! credits; the composite's members as charges on every post-prefill phase; the
//! 109.5 GB peak fill (AUTO control + filled run); the ROWSX slack credit with
//! the gate restore of its EXL3 port. Pure: it reads nothing and allocates
//! nothing. Every refusal is a named error, before any allocation.

const std = @import("std");
const host_env = @import("mlx_host").env;

pub const box_ceiling_bytes: u64 = 110_000_000_000;
pub const wired_ceiling_bytes: u64 = 100 << 30;
const gib: u64 = 1 << 30;
const mib: u64 = 1 << 20;
const n_layers: u64 = 40;
const transient_slots: u64 = 48;
/// The mxfp4 predecessor's records: prefill (weights + raw scales), decode weights, packed scale bank.
const raw_record: u64 = 18_800_640;
const weights_record: u64 = 17_694_720;
const packed_scales: u64 = 3_086_136_060;
const prefill_engine_remainder: u64 = 89_686_016;
const transform_reserve: u64 = 5 * gib;
const physical_headroom: u64 = gib;
const wired_headroom: u64 = gib;
/// native_projection_memory.
const projection_bytes: u64 = 67_108_864;
const packed_projection_bytes: u64 = 34_603_008;
const restored_projection_bytes: u64 = 40 * projection_bytes - 3 * projection_bytes - packed_projection_bytes;
pub const base_pipeline_bytes: u64 = 6 * projection_bytes;
/// native_retirement_memory.
const embedding_bytes: u64 = 1_323_827_200;
const tail_credit_bytes: u64 = 1_006_632_960 - 125_829_120;
const tail_rows_proved: u32 = 2048;
/// q3_peakfill_memory: the fill's target window is [box - 1 GiB (the unfilled search ceiling), 109.5 GB].
pub const max_target_bytes: u64 = 109_500_000_000;

/// The static envelope: the mxfp4 predecessor's admission at base 0 (the
/// native proof) and the phase-memory proof's constants. A calibration input:
/// `dsv41_pass2` is the Python stack's, re-derived by its own functions.
pub const Envelope = struct {
    /// The box the proof admits against (decimal); the allocator limit below is its own.
    box_bytes: u64 = box_ceiling_bytes,
    predecessor_decode_rows: u32,
    predecessor_prefill_rows: u32,
    decode_cache_bytes: u64,
    allocator_limit_at_base0: u64,
    host_reserve_bytes: u64,
    embedding_credit_bytes: u64,
    projection_credit_bytes: u64,
    steady_active_bytes: u64,
    prefill_active_bytes: u64,
    transition_start_bytes: u64,
    prefill_cache_bytes: u64,
    post_prefill_reserve_bytes: u64,
    projection_seed_fixed_bytes: u64,
    retirement_owners_proved: bool,
    /// Resident draft-head bytes the envelope's measured stack did not hold:
    /// the stack of record's compact DSpark head (run_full.py
    /// `MTP_PRUNED_BYTES`, its `text_only_resident_discount`). A process whose
    /// head leaves out fewer bytes is charged the difference on every phase.
    draft_pruned_bytes: u64 = 0,

    pub const dsv41_pass2: Envelope = .{
        .predecessor_decode_rows = 112,
        .predecessor_prefill_rows = 84,
        .decode_cache_bytes = 268_435_456,
        .allocator_limit_at_base0 = 108_561_226_752,
        .host_reserve_bytes = 1_438_773_248,
        .embedding_credit_bytes = 1_323_827_200,
        .projection_credit_bytes = 2_448_424_960,
        .steady_active_bytes = 97_762_795_628,
        .prefill_active_bytes = 89_965_493_340,
        .transition_start_bytes = 79_683_400_948,
        .prefill_cache_bytes = 1_160_216_040,
        .post_prefill_reserve_bytes = 1_301_547_008,
        .projection_seed_fixed_bytes = 17_536_326_524,
        .retirement_owners_proved = true,
        // 201 of the head's 3 x 128 experts x 18,800,640 B (the 09-13 trace selection, 93 / 58 / 32 kept).
        .draft_pruned_bytes = 3_778_928_640,
    };
};

pub const Allocation = enum { uniform, prefill_excess };
/// Aligned I/O staging of the read pool: 36 MiB for gate/up-first reads, else 32 MiB.
pub const IoLayout = enum { segments, record, gate_up };

/// The box an admission fits: the physical ceiling of the process's box, the GPU's wired ceiling, the
/// modeled-peak target's upper end and the rows a layer may hold. Null in `Inputs`: the envelope's own box (the
/// 110 GB / 100 GiB calibration, fills to 109.5 GB, 160 rows).
pub const Ceiling = struct {
    box_bytes: u64,
    wired_bytes: u64,
    max_target_bytes: u64,
    max_rows: u32,

    /// A box whose ceiling is the GPU's working set (the wired limit): the admitted modeled peak lands within
    /// `stop_bytes` of it; the fill's window ends 0.5 GB under the box, as the envelope's does.
    pub fn ofWorkingSet(working_set_bytes: u64, stop_bytes: u64, max_rows: u32) Ceiling {
        const target = working_set_bytes - stop_bytes;
        return .{ .box_bytes = target + (box_ceiling_bytes - max_target_bytes), .wired_bytes = working_set_bytes, .max_target_bytes = target, .max_rows = max_rows };
    }
};

pub const PeakFill = struct {
    /// Decimal-GB modeled-peak target, in [box - 1 GiB, the ceiling's max target]; null: that max.
    target_bytes: ?u64 = null,
    /// The named evidence credit (wide allowance + compiler reserve + tensor margin).
    evidence_credit_bytes: u64 = 2_737_047_552,
};

/// The ROWSX slack credit (EXL3 port of record): its evidence and the stack
/// geometry it was measured on.
pub const Rowsx = struct {
    credit_bytes: u64 = 1_611_661_312,
    mlx_side_credit_bytes: u64 = 1_044_381_696,
    decode_fixed_bytes: u64 = 14_491_882_096,
    prime_fixed_bytes: u64 = 15_201_501_308,
    /// host reserve + decode cache on the evidence stack
    non_mlx_bytes: u64 = 1_886_515_200 + 268_435_456,
    target_bytes: u64 = max_target_bytes,
};

pub const Inputs = struct {
    /// Non-file baseline and wired bytes at construction.
    baseline_bytes: u64,
    wired_bytes: u64,
    /// One expert record = one slot (EXL3 3.0: 13,315,584).
    record_bytes: u64,
    /// Forced decode rows (84..the ceiling's rows), or null for the largest that fits.
    fixed_rows: ?u32 = null,
    /// The largest decode rows searched (null: the ceiling's rows); an oracle launch searches up to its rows.
    search_ceiling: ?u32 = null,
    /// The box (null: the envelope's own).
    ceiling: ?Ceiling = null,
    /// The oracle's prefill capacity, which the admitted prefill must equal.
    matched_prefill_rows: ?u32 = null,
    allocation: Allocation = .prefill_excess,
    io_layout: IoLayout = .gate_up,
    embedding_rows: bool = true,
    tail_rows: ?u32 = tail_rows_proved,
    /// Active bytes every post-prefill phase carries beyond the proof's own
    /// owners: the base pipeline reserve plus every member charge except the
    /// lookahead staging (below).
    phase_reserve_bytes: u64,
    /// The lookahead pool's speculative staging charge; paid out of the ROWSX
    /// credit when that is on, else charged like the other members.
    lookahead_staging_bytes: u64 = 0,
    /// The wide lane's read-ahead windows: transient rows past the first 48,
    /// resident from construction, charged in every phase, the prefill included.
    wide_window_bytes: u64 = 0,
    /// Host bytes beyond the envelope's (pipeline host, comparison reserve).
    host_reserve_bytes: u64,
    /// The prefill members' charge on the prefill cache allowance.
    prefill_charge_bytes: u64 = 0,
    peak_fill: ?PeakFill = .{},
    rowsx: ?Rowsx = null,
    /// Resident draft-head bytes this process leaves out (0: the full head;
    /// a subset head: its pruned experts' bytes). Null: the envelope's own
    /// head. The admission charges `envelope.draft_pruned_bytes` minus this
    /// on every phase, the prefill included.
    draft_pruned_bytes: ?u64 = null,
};

pub const Phase = enum { growth, seed, prime, decode };
pub const Phases = struct {
    growth: u64,
    seed: u64,
    prime: u64,
    decode: u64,

    fn max(p: Phases) u64 {
        return @max(@max(p.growth, p.seed), @max(p.prime, p.decode));
    }

    /// The binding phase (the first maximum in growth, seed, prime, decode order).
    fn binding(p: Phases) Phase {
        const m = p.max();
        return if (p.growth == m) .growth else if (p.seed == m) .seed else if (p.prime == m) .prime else .decode;
    }
};

/// One admission (packed_admission.retarget's result, the fields a caller uses).
pub const Admission = struct {
    /// The whole admission for one construction: rows, bounds, and the
    /// peak-fill / ROWSX records, or the refusal by name.
    pub const plan = planAdmission;

    decode_rows: u32,
    /// Prefill rows in the predecessor's record (the engine budget's unit).
    prefill_rows: u32,
    /// Expert records the prefill bank holds per layer at that budget.
    prefill_capacity: u32,
    prefill_max_rows: u32,
    search_ceiling: u32,
    /// The MLX active the prefill-boundary retirement and the growth start may reach.
    transition_start_bytes: u64,
    retirement_entry_bytes: u64,
    phases: Phases,
    steady_bytes: u64,
    resize_bytes: u64,
    prefill_active_bytes: u64,
    prefill_physical_bytes: u64,
    prefill_cache_bytes: u64,
    physical_bound_bytes: u64,
    active_bound_bytes: u64,
    host_reserve_bytes: u64,
    allocator_limit_bytes: u64,
    decode_cache_bytes: u64,
    final_bank_bytes: u64,
    prefill_bank_bound_bytes: u64,
    embedding_credit_bytes: u64,
    tail_credit_bytes: u64,
    baseline_bytes: u64,
    wired_bytes: u64,
};

/// q3_peakfill_memory's receipt summary of one admission.
pub const Summary = struct {
    decode_rows: u32,
    prefill_rows: u32,
    prefill_max_rows: u32,
    final_bank_bytes: u64,
    phases: Phases,
    binding: Phase,
    binding_active_bytes: u64,
    binding_physical_bytes: u64,
    prefill_active_bytes: u64,
    prefill_physical_bytes: u64,
    physical_bound_bytes: u64,
    allocator_limit_bytes: u64,
    active_plus_cache_bytes: u64,
    wired_plus_active_plus_cache_bytes: u64,

    fn of(a: Admission) Summary {
        const m = a.phases.max();
        return .{
            .decode_rows = a.decode_rows,
            .prefill_rows = a.prefill_rows,
            .prefill_max_rows = a.prefill_max_rows,
            .final_bank_bytes = a.final_bank_bytes,
            .phases = a.phases,
            .binding = a.phases.binding(),
            .binding_active_bytes = m,
            .binding_physical_bytes = a.baseline_bytes + a.host_reserve_bytes + m + a.decode_cache_bytes,
            .prefill_active_bytes = a.prefill_active_bytes,
            .prefill_physical_bytes = a.prefill_physical_bytes,
            .physical_bound_bytes = a.physical_bound_bytes,
            .allocator_limit_bytes = a.allocator_limit_bytes,
            .active_plus_cache_bytes = m + a.decode_cache_bytes,
            .wired_plus_active_plus_cache_bytes = a.wired_bytes + m + a.decode_cache_bytes,
        };
    }
};

pub const PeakFillRecord = struct {
    target_bytes: u64,
    control: Summary,
    filled: Summary,
    total_credit_bytes: u64,
    modeled_peak_bytes: u64,
};

pub const RowsxRecord = struct {
    uncredited_rows: u32,
    credited_rows: u32,
    added_rows: u32,
    added_bytes: u64,
    credit_bytes: u64,
    mlx_side_credit_bytes: u64,
    modeled_peak_uncredited_bytes: u64,
    modeled_peak_credited_bytes: u64,
    target_bytes: u64,
    allocator_margin_bytes: i64,
    wired_margin_bytes: i64,
    box_margin_bytes: i64,
    baseline_bytes: u64,
    /// The prefill-boundary bound the credit would have lowered, restored.
    restored_gate_bytes: u64,
};

pub const Plan = struct {
    admission: Admission,
    peak_fill: ?PeakFillRecord = null,
    rowsx: ?RowsxRecord = null,
};

pub const Error = error{
    InvalidMatchedRows,
    InvalidFixedRows,
    FixedRowsAboveCeiling,
    TailRowsNotProved,
    RetirementProofMissing,
    ProjectionGeometry,
    ProjectionEnvelopeTooSmall,
    PhaseCreditExceedsEnvelope,
    RetirementCreditExceedsOwners,
    PrefillDoesNotFit,
    NoDecodeCapacity,
    FixedRowsDoNotFit,
    NoCausalPrefill,
    MatchedPrefillDoesNotFit,
    FixedRowsIncompatibleWithPrefill,
    InvalidTarget,
    PeakFillMovedPrefill,
    PeakFillReducedRows,
    PeakFillMovedFixedRows,
    PeakFillOverTarget,
    PeakFillOverAllocator,
    PeakFillOverWired,
    CreditWithFixedRows,
    CreditGeometry,
    CreditMovedPrefill,
    CreditAddsNoRow,
    CreditOverTarget,
    CreditOverAllocator,
    CreditOverWired,
    CreditReachedGate,
};

fn bankBytes(rows: u64, record: u64) u64 {
    return (n_layers * rows + transient_slots) * record;
}

/// The lookahead pool's staging charge (q3_lookahead4_candidate.charge_bytes):
/// `slots` staging slots of the page-rounded record plus two pages, plus 1 MiB of tables.
pub fn lookaheadCharge(record_bytes: u64, slots: u64, page: u64) u64 {
    return slots * (std.mem.alignForward(u64, record_bytes, page) + 2 * page) + mib;
}

/// One admission at `credit` bytes off every post-prefill phase, `fixed` rows
/// forced; `filled`: the peak fill's run, whose credited phases must stay positive.
fn retarget(env: Envelope, in: Inputs, credit: u64, fixed: ?u32, filled: bool) Error!Admission {
    const base: i64 = @intCast(in.baseline_bytes);
    const wired: i64 = @intCast(in.wired_bytes);
    const record: i64 = @intCast(in.record_bytes);
    const io_staging: u64 = if (in.io_layout == .gate_up) 36 * mib else 32 * mib;
    const io_host = io_staging + mib;
    // native_retirement_memory.credits (scheduled projections).
    if (in.tail_rows) |t| if (t != tail_rows_proved) return error.TailRowsNotProved;
    if ((in.embedding_rows or in.tail_rows != null) and !env.retirement_owners_proved) return error.RetirementProofMissing;
    const retired_embedding: i64 = if (in.embedding_rows) embedding_bytes else 0;
    const retired_tail: i64 = if (in.tail_rows != null) tail_credit_bytes else 0;
    const box = boxOf(env, in);
    if (in.matched_prefill_rows) |m| if (m < 16 or m > box.max_rows) return error.InvalidMatchedRows;
    var ceiling = in.search_ceiling orelse box.max_rows;
    if (fixed) |f| {
        if (f < 84 or f > box.max_rows) return error.InvalidFixedRows;
        if (f > ceiling) return error.FixedRowsAboveCeiling;
        ceiling = f;
    }
    const post_reserve: i64 = @intCast(env.post_prefill_reserve_bytes);
    const cache: i64 = @intCast(env.decode_cache_bytes);
    const allocator_limit: i64 = @as(i64, @intCast(env.allocator_limit_at_base0 + box.box_bytes)) - @as(i64, @intCast(env.box_bytes)) - base;
    const host: i64 = @intCast(env.host_reserve_bytes + io_host + in.host_reserve_bytes);
    const restored: i64 = @intCast(env.embedding_credit_bytes + env.projection_credit_bytes);
    const pred_prefill: i64 = env.predecessor_prefill_rows;
    const source_prefill_bank: i64 = @intCast(bankBytes(env.predecessor_prefill_rows, raw_record));
    const steady_fixed: i64 = @as(i64, @intCast(env.steady_active_bytes)) - @as(i64, @intCast(bankBytes(env.predecessor_decode_rows, weights_record))) - packed_scales + restored;
    const transition_fixed: i64 = @as(i64, @intCast(env.transition_start_bytes + env.embedding_credit_bytes)) - source_prefill_bank;

    // projection_fixed_bounds at the proof's pipeline reserve; every member
    // adds its bytes to every phase, every credit takes them off.
    if (env.projection_credit_bytes != restored_projection_bytes or in.phase_reserve_bytes < base_pipeline_bytes)
        return error.ProjectionGeometry;
    const inherited = steady_fixed - @as(i64, restored_projection_bytes) - 3 * @as(i64, projection_bytes);
    if (inherited < 0 or steady_fixed < 0 or transition_fixed < 0) return error.ProjectionEnvelopeTooSmall;
    const seed_fixed: i64 = @intCast(env.projection_seed_fixed_bytes);
    const group: i64 = projection_bytes;
    // The draft head's residents beyond (or short of) the envelope's.
    const head_extra: i64 = if (in.draft_pruned_bytes) |p| @as(i64, @intCast(env.draft_pruned_bytes)) - @as(i64, @intCast(p)) else 0;
    const wide: i64 = @intCast(in.wide_window_bytes);
    const charge: i64 = @as(i64, @intCast(in.phase_reserve_bytes - base_pipeline_bytes)) +
        (if (in.rowsx == null) @as(i64, @intCast(in.lookahead_staging_bytes)) else 0) - @as(i64, @intCast(credit)) + head_extra + wide;
    var ph = [4]i64{
        transition_fixed + group,
        seed_fixed + group,
        seed_fixed + projection_bytes + group,
        inherited + 5 * @as(i64, projection_bytes) + group,
    };
    for (&ph) |*v| v.* += charge;
    if (filled) for (ph) |v| if (v <= 0) return error.PhaseCreditExceedsEnvelope;
    for (&ph, 0..) |*v, i| v.* -= retired_embedding + (if (i != 3) retired_tail else 0);
    for (ph) |v| if (v < 0) return error.RetirementCreditExceedsOwners;
    const fixed_peak = @max(@max(ph[0], ph[1]), @max(ph[2], ph[3]));
    const prefill_cache: i64 = @intCast(env.prefill_cache_bytes + in.prefill_charge_bytes);

    const Fit = struct {
        box: i64,
        base: i64,
        host: i64,
        wired: i64,
        wired_ceiling: i64,
        allocator_limit: i64,
        fn ok(f: @This(), active: i64, cache_bytes: i64) bool {
            return f.base + f.host + active + cache_bytes + @as(i64, physical_headroom) <= f.box and
                active + cache_bytes <= f.allocator_limit and
                f.wired + active + cache_bytes + @as(i64, wired_headroom) <= f.wired_ceiling;
        }
    };
    const fit: Fit = .{ .box = @intCast(box.box_bytes), .base = base, .host = host, .wired = wired, .wired_ceiling = @intCast(box.wired_bytes), .allocator_limit = allocator_limit };
    const entry_fixed = transition_fixed + projection_bytes - retired_tail + post_reserve + head_extra + wide;
    const prefill_base: i64 = @as(i64, @intCast(env.prefill_active_bytes)) + prefill_engine_remainder + transform_reserve + head_extra + wide;
    const raw_band: i64 = n_layers * raw_record;

    // The prefill: the largest predecessor-record row budget whose active and
    // un-retired entry both fit.
    var prefill_rows: i64 = -1;
    var prefill_active: i64 = 0;
    var prefill_physical: i64 = 0;
    var rows: i64 = pred_prefill;
    while (rows > 15) : (rows -= 1) {
        prefill_active = prefill_base - (pred_prefill - rows) * raw_band;
        prefill_physical = base + host + prefill_active + prefill_cache;
        const old_bank: i64 = @as(i64, @intCast(bankBytes(@intCast(rows), raw_record))) + prefill_engine_remainder;
        if (fit.ok(prefill_active, prefill_cache) and fit.ok(entry_fixed + old_bank, prefill_cache)) {
            prefill_rows = rows;
            break;
        }
    }
    if (prefill_rows < 0) return error.PrefillDoesNotFit;

    // The decode rows: the largest capacity whose binding phase fits.
    var chosen: i64 = -1;
    var cap: i64 = ceiling;
    while (cap >= pred_prefill) : (cap -= 1) {
        const active = fixed_peak + @as(i64, @intCast(bankBytes(@intCast(cap), @intCast(record)))) + post_reserve;
        if (fit.ok(active, cache)) {
            chosen = cap;
            break;
        }
    }
    if (chosen < 0) return error.NoDecodeCapacity;
    if (fixed) |f| if (chosen != f) return error.FixedRowsDoNotFit;

    // The causal allocator's room: eight average extension rows past the prefill bank.
    var prefill_max: i64 = chosen - (if (in.allocation == .prefill_excess) @as(i64, 8) else 0);
    if (in.matched_prefill_rows) |m| prefill_max = @min(prefill_max, m);
    const capacityOf = struct {
        fn f(r: i64, rec: i64) i64 {
            const bound: i64 = @as(i64, @intCast(bankBytes(@intCast(r), raw_record))) + prefill_engine_remainder;
            return @divFloor(bound - @as(i64, transient_slots) * rec, @as(i64, n_layers) * rec);
        }
    }.f;
    if (in.allocation == .prefill_excess or in.matched_prefill_rows != null) {
        var r = prefill_rows;
        var found = false;
        while (r > 15) : (r -= 1) {
            if (capacityOf(r, record) <= prefill_max) {
                prefill_rows = r;
                found = true;
                break;
            }
        }
        if (!found) return error.NoCausalPrefill;
        prefill_active = prefill_base - (pred_prefill - prefill_rows) * raw_band;
        prefill_physical = base + host + prefill_active + prefill_cache;
    }
    if (in.matched_prefill_rows) |m| if (capacityOf(prefill_rows, record) != m) return error.MatchedPrefillDoesNotFit;
    if (fixed != null and capacityOf(prefill_rows, record) > prefill_max) return error.FixedRowsIncompatibleWithPrefill;

    const final_bank: i64 = @intCast(bankBytes(@intCast(chosen), @intCast(record)));
    const phases: Phases = .{
        .growth = @intCast(ph[0] + final_bank + post_reserve),
        .seed = @intCast(ph[1] + final_bank + post_reserve),
        .prime = @intCast(ph[2] + final_bank + post_reserve),
        .decode = @intCast(ph[3] + final_bank + post_reserve),
    };
    const steady: i64 = ph[3] + final_bank + post_reserve;
    const resize: i64 = fixed_peak + final_bank + post_reserve;
    const prefill_bank_bound: i64 = @as(i64, @intCast(bankBytes(@intCast(prefill_rows), raw_record))) + prefill_engine_remainder;
    const entry = entry_fixed + prefill_bank_bound;
    const transition_start = ph[0] + prefill_bank_bound + post_reserve;
    const physical_bound = @max(@max(prefill_physical, base + host + entry + prefill_cache), base + host + @max(steady, resize) + cache);
    return .{
        .decode_rows = @intCast(chosen),
        .prefill_rows = @intCast(prefill_rows),
        .prefill_capacity = @intCast(capacityOf(prefill_rows, record)),
        .prefill_max_rows = @intCast(prefill_max),
        .search_ceiling = ceiling,
        .transition_start_bytes = @intCast(transition_start),
        .retirement_entry_bytes = @intCast(entry),
        .phases = phases,
        .steady_bytes = @intCast(steady),
        .resize_bytes = @intCast(resize),
        .prefill_active_bytes = @intCast(prefill_active),
        .prefill_physical_bytes = @intCast(prefill_physical),
        .prefill_cache_bytes = @intCast(prefill_cache),
        .physical_bound_bytes = @intCast(physical_bound),
        .active_bound_bytes = @intCast(@max(@max(prefill_active, entry), @max(steady, resize))),
        .host_reserve_bytes = @intCast(host),
        .allocator_limit_bytes = @intCast(allocator_limit),
        .decode_cache_bytes = env.decode_cache_bytes,
        .final_bank_bytes = @intCast(final_bank),
        .prefill_bank_bound_bytes = @intCast(prefill_bank_bound),
        .embedding_credit_bytes = @intCast(retired_embedding),
        .tail_credit_bytes = @intCast(retired_tail),
        .baseline_bytes = in.baseline_bytes,
        .wired_bytes = in.wired_bytes,
    };
}

/// The peak fill: the AUTO control, then the filled run at `in.fixed_rows`
/// with its credit off every post-prefill phase; the fill may move nothing
/// but the decode rows and must stay under its target.
fn peakFilled(env: Envelope, in: Inputs, pf: PeakFill, extra_credit: u64) Error!struct { admission: Admission, record: PeakFillRecord } {
    const box = boxOf(env, in);
    const target = pf.target_bytes orelse box.max_target_bytes;
    const unfilled = box.box_bytes - physical_headroom;
    if (target < unfilled or target > box.max_target_bytes) return error.InvalidTarget;
    const spend = target - unfilled;
    const credit = pf.evidence_credit_bytes + spend;
    const control = try retarget(env, in, extra_credit, null, false);
    const filled = try retarget(env, in, extra_credit + credit, in.fixed_rows, true);
    if (filled.prefill_rows != control.prefill_rows or filled.prefill_active_bytes != control.prefill_active_bytes or
        filled.prefill_physical_bytes != control.prefill_physical_bytes or filled.retirement_entry_bytes != control.retirement_entry_bytes or
        filled.host_reserve_bytes != control.host_reserve_bytes) return error.PeakFillMovedPrefill;
    if (in.fixed_rows) |f| {
        if (filled.decode_rows != f) return error.PeakFillMovedFixedRows;
    } else if (filled.decode_rows < control.decode_rows) return error.PeakFillReducedRows;
    const active = filled.phases.max();
    const modeled = @max(filled.physical_bound_bytes, filled.baseline_bytes + filled.host_reserve_bytes + active + filled.decode_cache_bytes + spend);
    if (modeled > target) return error.PeakFillOverTarget;
    if (active + filled.decode_cache_bytes > filled.allocator_limit_bytes) return error.PeakFillOverAllocator;
    if (filled.wired_bytes + active + filled.decode_cache_bytes + gib > box.wired_bytes) return error.PeakFillOverWired;
    return .{ .admission = filled, .record = .{ .target_bytes = target, .control = Summary.of(control), .filled = Summary.of(filled), .total_credit_bytes = credit, .modeled_peak_bytes = modeled } };
}

/// The box an admission fits: the caller's ceiling, else the envelope's own.
fn boxOf(env: Envelope, in: Inputs) Ceiling {
    return in.ceiling orelse .{ .box_bytes = env.box_bytes, .wired_bytes = wired_ceiling_bytes, .max_target_bytes = max_target_bytes, .max_rows = 160 };
}

fn planAdmission(env: Envelope, in: Inputs) Error!Plan {
    const pf = in.peak_fill orelse {
        if (in.rowsx != null) return error.CreditGeometry;
        return .{ .admission = try retarget(env, in, 0, in.fixed_rows, false) };
    };
    const rx = in.rowsx orelse {
        const r = try peakFilled(env, in, pf, 0);
        return .{ .admission = r.admission, .peak_fill = r.record };
    };
    // The lane pays the lookahead staging out of its credit (whole MiB).
    const staging = std.mem.alignForward(u64, in.lookahead_staging_bytes, mib);
    if (staging >= rx.credit_bytes) return error.CreditGeometry;
    const credit = rx.credit_bytes - staging;
    const ref = try peakFilled(env, in, pf, 0);
    if (in.fixed_rows != null) return error.CreditWithFixedRows;
    const r = ref.admission;
    // The credit's evidence was measured in the envelope's own box.
    if (in.ceiling != null) return error.CreditGeometry;
    const spend = ref.record.target_bytes - (env.box_bytes - physical_headroom);
    if (r.phases.decode - r.final_bank_bytes != rx.decode_fixed_bytes or r.phases.prime - r.final_bank_bytes != rx.prime_fixed_bytes or
        r.host_reserve_bytes + r.decode_cache_bytes != rx.non_mlx_bytes or ref.record.target_bytes != rx.target_bytes or env.box_bytes != box_ceiling_bytes)
        return error.CreditGeometry;
    const cred = try peakFilled(env, in, pf, credit);
    const c = cred.admission;
    if (c.prefill_rows != r.prefill_rows or c.prefill_active_bytes != r.prefill_active_bytes or c.prefill_physical_bytes != r.prefill_physical_bytes or
        c.retirement_entry_bytes != r.retirement_entry_bytes or c.host_reserve_bytes != r.host_reserve_bytes or c.decode_cache_bytes != r.decode_cache_bytes or
        c.baseline_bytes != r.baseline_bytes or c.allocator_limit_bytes != r.allocator_limit_bytes or c.wired_bytes != r.wired_bytes)
        return error.CreditMovedPrefill;
    if (c.decode_rows <= r.decode_rows) return error.CreditAddsNoRow;
    const added = c.decode_rows - r.decode_rows;
    const true_active = rx.prime_fixed_bytes + c.final_bank_bytes;
    const modeled_true = c.baseline_bytes + c.host_reserve_bytes + true_active + c.decode_cache_bytes + spend;
    if (modeled_true - credit > rx.target_bytes) return error.CreditOverTarget;
    const mlx_side = true_active - rx.mlx_side_credit_bytes + c.decode_cache_bytes;
    if (mlx_side > c.allocator_limit_bytes) return error.CreditOverAllocator;
    if (c.wired_bytes + mlx_side + gib > wired_ceiling_bytes) return error.CreditOverWired;
    // The credit lowers the prefill-boundary bound only through the growth
    // phase, at an unchanged prefill bank: give that bound back.
    if ((r.phases.growth - r.final_bank_bytes) - (c.phases.growth - c.final_bank_bytes) != credit or
        r.transition_start_bytes - c.transition_start_bytes != credit or r.prefill_bank_bound_bytes != c.prefill_bank_bound_bytes)
        return error.CreditReachedGate;
    var admitted = c;
    admitted.transition_start_bytes = r.transition_start_bytes;
    return .{ .admission = admitted, .peak_fill = cred.record, .rowsx = .{
        .uncredited_rows = r.decode_rows,
        .credited_rows = c.decode_rows,
        .added_rows = added,
        .added_bytes = added * n_layers * in.record_bytes,
        .credit_bytes = credit,
        .mlx_side_credit_bytes = rx.mlx_side_credit_bytes,
        .modeled_peak_uncredited_bytes = modeled_true,
        .modeled_peak_credited_bytes = modeled_true - credit,
        .target_bytes = rx.target_bytes,
        .allocator_margin_bytes = @as(i64, @intCast(c.allocator_limit_bytes)) - @as(i64, @intCast(mlx_side)),
        .wired_margin_bytes = @as(i64, @intCast(wired_ceiling_bytes)) - @as(i64, @intCast(c.wired_bytes + mlx_side + gib)),
        .box_margin_bytes = @as(i64, @intCast(env.box_bytes)) - @as(i64, @intCast(modeled_true - credit)),
        .baseline_bytes = c.baseline_bytes,
        .restored_gate_bytes = r.transition_start_bytes,
    } };
}

// ── Tests ──

const testing = std.testing;

test "dsv41 admission: the lookahead staging charge is the lane's" {
    // q3_lookahead4_candidate.charge_bytes: 4 slots (budget 2) of the page-rounded record + 2 pages, + 1 MiB.
    try testing.expectEqual(@as(u64, 54_460_416), lookaheadCharge(13_315_584, 4, 16384));
    try testing.expectEqual(@as(u64, 54_394_880), lookaheadCharge(13_290_496, 4, 16384));
    // Paid out of the ROWSX credit in whole MiB: 1,611,661,312 - 54,525,952.
    try testing.expectEqual(@as(u64, 1_557_135_360), 1_611_661_312 - std.mem.alignForward(u64, 54_460_416, mib));
}

const exl3_record: u64 = 13_315_584;
const tcq3_record: u64 = 13_290_496;

/// A pass-2 fast lane: pipeline + wide reserve + layer compile + member charge on every
/// post-prefill phase, the lookahead4 staging, and the prefill members' cache charge.
fn pass2Fast(base: u64, wired: u64, fixed: ?u32) Inputs {
    return .{
        .baseline_bytes = base,
        .wired_bytes = wired,
        .record_bytes = exl3_record,
        .fixed_rows = fixed,
        .phase_reserve_bytes = base_pipeline_bytes + 1536 * mib + 2 * projection_bytes,
        .lookahead_staging_bytes = lookaheadCharge(exl3_record, 4, 16384),
        .host_reserve_bytes = 408_944_640,
        .prefill_charge_bytes = 5_703_196_672,
    };
}

fn expectPhaseValues(p: Phases, growth: u64, seed: u64, prime: u64, decode: u64) !void {
    try testing.expectEqual(Phases{ .growth = growth, .seed = seed, .prime = prime, .decode = decode }, p);
}

test "dsv41 admission: a forced-147 fast receipt (host-fast-exact EXL3) admits its logged rows and bounds" {
    const p = try Admission.plan(.dsv41_pass2, pass2Fast(7_755_397_656, 3_377_741_824, 147));
    const a = p.admission;
    try testing.expectEqual(@as(u32, 147), a.decode_rows);
    try testing.expectEqual(@as(u32, 80), a.prefill_rows);
    try testing.expectEqual(@as(u32, 113), a.prefill_capacity);
    try testing.expectEqual(@as(u32, 139), a.prefill_max_rows);
    try testing.expectEqual(@as(u64, 75_741_338_100), a.transition_start_bytes);
    try testing.expectEqual(@as(u64, 78_576_663_796), a.retirement_entry_bytes);
    try testing.expectEqual(@as(u64, 108_921_111_644), a.physical_bound_bytes);
    try expectPhaseValues(a.phases, 93_521_955_316, 94_123_634_812, 94_190_743_676, 93_481_124_464);
    const pf = p.peak_fill.?;
    try testing.expectEqual(@as(u32, 149), pf.control.decode_rows);
    try testing.expectEqual(@as(u32, 147), pf.filled.decode_rows);
    try testing.expectEqual(Phase.prime, pf.filled.binding);
    try testing.expectEqual(@as(u64, 3_310_789_376), pf.total_credit_bytes);
    try testing.expectEqual(@as(u64, 108_921_111_644), pf.modeled_peak_bytes);
    try testing.expect(p.rowsx == null);
}

test "dsv41 admission: the full DSpark head pays the record's pruned experts on every phase; the record's own subset pays nothing" {
    const d = Envelope.dsv41_pass2.draft_pruned_bytes;
    try testing.expectEqual(@as(u64, 201 * 18_800_640), d);
    var rec_in = pass2Fast(7_755_397_656, 3_377_741_824, 130);
    rec_in.peak_fill = null;
    var full_in = rec_in;
    full_in.draft_pruned_bytes = 0;
    var same_in = rec_in;
    same_in.draft_pruned_bytes = d;
    const rec = (try Admission.plan(.dsv41_pass2, rec_in)).admission;
    const full = (try Admission.plan(.dsv41_pass2, full_in)).admission;
    const same = (try Admission.plan(.dsv41_pass2, same_in)).admission;
    try testing.expectEqual(rec, same);
    try expectPhaseValues(full.phases, rec.phases.growth + d, rec.phases.seed + d, rec.phases.prime + d, rec.phases.decode + d);
    // The prefill holds the head too: its active is the record's + d at the same rows, so it can admit fewer.
    try testing.expect(full.prefill_rows <= rec.prefill_rows);
    try testing.expectEqual(rec.prefill_active_bytes + d - (rec.prefill_rows - full.prefill_rows) * n_layers * raw_record, full.prefill_active_bytes);
    // Unforced, the full head admits the rows its 3.78 GB displaces (0.533 GB a row).
    var auto_rec = rec_in;
    auto_rec.fixed_rows = null;
    var auto_full = auto_rec;
    auto_full.draft_pruned_bytes = 0;
    const r0 = (try Admission.plan(.dsv41_pass2, auto_rec)).admission.decode_rows;
    const r1 = (try Admission.plan(.dsv41_pass2, auto_full)).admission.decode_rows;
    // 3,778,928,640 B = 7.09 rows of 40 x 13,315,584 B: 7 or 8 fewer, by the slack at r0.
    try testing.expect(r0 - r1 == d / (40 * exl3_record) or r0 - r1 == d / (40 * exl3_record) + 1);
}

test "dsv41 admission: an AUTO fast receipt (rowsx control) fills to its logged 156 rows" {
    const p = try Admission.plan(.dsv41_pass2, pass2Fast(7_490_976_280, 3_332_423_680, null));
    try testing.expectEqual(@as(u32, 156), p.admission.decode_rows);
    try testing.expectEqual(@as(u32, 80), p.admission.prefill_rows);
    try testing.expectEqual(@as(u32, 148), p.admission.prefill_max_rows);
    try testing.expectEqual(@as(u64, 75_741_338_100), p.admission.transition_start_bytes);
    try testing.expectEqual(@as(u64, 108_656_690_268), p.admission.physical_bound_bytes);
    try expectPhaseValues(p.admission.phases, 98_315_565_556, 98_917_245_052, 98_984_353_916, 98_274_734_704);
    try testing.expectEqual(@as(u32, 150), p.peak_fill.?.control.decode_rows);
    try testing.expectEqual(@as(u64, 109_204_022_676), p.peak_fill.?.modeled_peak_bytes);
}

test "dsv41 admission: the ROWSX slack receipt pays the lookahead staging from its credit and restores the gate" {
    var in = pass2Fast(7_044_415_488, 3_332_571_136, null);
    in.rowsx = .{};
    const p = try Admission.plan(.dsv41_pass2, in);
    const a = p.admission;
    try testing.expectEqual(@as(u32, 160), a.decode_rows);
    try testing.expectEqual(@as(u32, 80), a.prefill_rows);
    try testing.expectEqual(@as(u32, 152), a.prefill_max_rows);
    // The restored prefill-boundary bound; the receipt logged the pre-fix one, a credit lower.
    try testing.expectEqual(@as(u64, 75_686_877_684), a.transition_start_bytes);
    try testing.expectEqual(@as(u64, 108_702_617_724), a.physical_bound_bytes);
    try expectPhaseValues(a.phases, 98_834_463_220, 99_436_142_716, 99_503_251_580, 98_793_632_368);
    try testing.expectEqual(@as(u32, 154), p.peak_fill.?.control.decode_rows);
    try testing.expectEqual(@as(u64, 109_276_359_548), p.peak_fill.?.modeled_peak_bytes);
    try testing.expectEqualDeep(RowsxRecord{
        .uncredited_rows = 157,
        .credited_rows = 160,
        .added_rows = 3,
        .added_bytes = 1_597_870_080,
        .credit_bytes = 1_557_135_360,
        .mlx_side_credit_bytes = 1_044_381_696,
        .modeled_peak_uncredited_bytes = 110_833_494_908,
        .modeled_peak_credited_bytes = 109_276_359_548,
        .target_bytes = 109_500_000_000,
        .allocator_margin_bytes = 1_232_370_564,
        .wired_margin_bytes = 2_683_428_740,
        .box_margin_bytes = 723_640_452,
        .baseline_bytes = 7_044_415_488,
        .restored_gate_bytes = 75_686_877_684,
    }, p.rowsx.?);
    try testing.expectEqual(@as(u64, 74_129_742_324), p.rowsx.?.restored_gate_bytes - p.rowsx.?.credit_bytes);
}

test "dsv41 admission: a standard tcq3 receipt searches up to its oracle rows and matches its prefill" {
    var in = pass2Fast(6_935_067_184, 3_567_960_064, 147);
    in.record_bytes = tcq3_record;
    in.lookahead_staging_bytes = lookaheadCharge(tcq3_record, 4, 16384);
    in.host_reserve_bytes = 392_167_424;
    in.prefill_charge_bytes = 5_702_795_264;
    in.search_ceiling = 147;
    in.matched_prefill_rows = 115;
    const p = try Admission.plan(.dsv41_pass2, in);
    try testing.expectEqual(@as(u32, 147), p.admission.decode_rows);
    try testing.expectEqual(@as(u32, 81), p.admission.prefill_rows);
    try testing.expectEqual(@as(u32, 115), p.admission.prefill_capacity);
    try testing.expectEqual(@as(u64, 76_493_298_164), p.admission.transition_start_bytes);
    try testing.expectEqual(@as(u64, 79_328_689_396), p.admission.retirement_entry_bytes);
    try testing.expectEqual(@as(u64, 108_835_628_148), p.admission.physical_bound_bytes);
    try testing.expectEqual(@as(u32, 147), p.peak_fill.?.control.decode_rows);
}

test "dsv41 admission: every refusal is named before anything is allocated" {
    const env: Envelope = .dsv41_pass2;
    const t = pass2Fast(7_755_397_656, 3_377_741_824, 147);
    var c = t;
    c.fixed_rows = null;
    c.baseline_bytes = 48_000_000_000;
    try testing.expectError(error.NoDecodeCapacity, Admission.plan(env, c));
    c.baseline_bytes = 60_000_000_000;
    try testing.expectError(error.PrefillDoesNotFit, Admission.plan(env, c));
    c = t;
    c.fixed_rows = null;
    c.wired_bytes = 48_000_000_000;
    try testing.expectError(error.NoDecodeCapacity, Admission.plan(env, c));
    c.wired_bytes = 70_000_000_000;
    try testing.expectError(error.PrefillDoesNotFit, Admission.plan(env, c));
    c = t;
    c.fixed_rows = 160;
    c.baseline_bytes = 12_000_000_000;
    try testing.expectError(error.FixedRowsDoNotFit, Admission.plan(env, c));
    c = t;
    c.fixed_rows = 161;
    try testing.expectError(error.InvalidFixedRows, Admission.plan(env, c));
    c.fixed_rows = 83;
    try testing.expectError(error.InvalidFixedRows, Admission.plan(env, c));
    c = t;
    c.search_ceiling = 146;
    try testing.expectError(error.FixedRowsAboveCeiling, Admission.plan(env, c));
    // A forced 90 moves the causal prefill off the AUTO control's.
    c = t;
    c.fixed_rows = 90;
    c.baseline_bytes = 4_000_000_000;
    try testing.expectError(error.PeakFillMovedPrefill, Admission.plan(env, c));
    c = t;
    c.fixed_rows = null;
    c.matched_prefill_rows = 150;
    try testing.expectError(error.MatchedPrefillDoesNotFit, Admission.plan(env, c));
    c.matched_prefill_rows = 15;
    try testing.expectError(error.InvalidMatchedRows, Admission.plan(env, c));
    c = t;
    c.tail_rows = 1024;
    try testing.expectError(error.TailRowsNotProved, Admission.plan(env, c));
    var unproved = env;
    unproved.retirement_owners_proved = false;
    try testing.expectError(error.RetirementProofMissing, Admission.plan(unproved, t));
    c = t;
    c.peak_fill = .{ .target_bytes = 108_900_000_000 };
    try testing.expectError(error.InvalidTarget, Admission.plan(env, c));
    c.peak_fill = .{ .target_bytes = 109_600_000_000 };
    try testing.expectError(error.InvalidTarget, Admission.plan(env, c));
    c = t;
    c.phase_reserve_bytes = base_pipeline_bytes - 1;
    try testing.expectError(error.ProjectionGeometry, Admission.plan(env, c));

    // The ROWSX lane.
    var s = pass2Fast(7_044_415_488, 3_332_571_136, null);
    s.rowsx = .{};
    c = s;
    c.fixed_rows = 150;
    try testing.expectError(error.CreditWithFixedRows, Admission.plan(env, c));
    c = s;
    c.host_reserve_bytes = 392_167_424;
    try testing.expectError(error.CreditGeometry, Admission.plan(env, c));
    c = s;
    c.rowsx = .{ .credit_bytes = 100_000_000 + 52 * mib };
    try testing.expectError(error.CreditAddsNoRow, Admission.plan(env, c));
    c = s;
    c.baseline_bytes = 10_000_000_000;
    c.rowsx = .{ .credit_bytes = 4_000_000_000 + 52 * mib };
    try testing.expectError(error.CreditOverAllocator, Admission.plan(env, c));
    c = s;
    c.rowsx = .{ .credit_bytes = 52 * mib };
    try testing.expectError(error.CreditGeometry, Admission.plan(env, c));
    c = s;
    c.peak_fill = null;
    try testing.expectError(error.CreditGeometry, Admission.plan(env, c));
}

// The fixture's records (the reference runtime's dump_phase4a_admission_fixture.py).
const FixPhases = struct { growth: u64, seed: u64, prime: u64, decode: u64 };
const FixSummary = struct {
    decode_slots_per_layer: u32,
    prefill_slots_per_layer: u32,
    tcq3_prefill_max_slots_per_layer: u32,
    tcq3_final_slot_storage_bytes: u64,
    post_prefill_phase_active_bounds: FixPhases,
    binding_phase: []const u8,
    binding_phase_active_bytes: u64,
    binding_phase_physical_bytes: u64,
    prefill_active_bound_bytes: u64,
    prefill_physical_bound_bytes: u64,
    physical_bound_bytes: u64,
    allocator_limit_bytes: u64,
    active_plus_cache_bytes: u64,
    wired_plus_active_plus_cache_bytes: u64,
};
const FixRowsx = struct {
    first_capacity_uncredited: u32,
    first_capacity_credited: u32,
    added_rows: u32,
    added_bytes: u64,
    credit_bytes: u64,
    mlx_side_credit_bytes: u64,
    modeled_peak_uncredited_bytes: u64,
    modeled_peak_credited_bytes: u64,
    target_bytes: u64,
    allocator_margin_bytes: i64,
    wired_margin_bytes: i64,
    box_ceiling_margin_modeled_bytes: i64,
    baseline_bytes: u64,
    transition_start_active_bound_bytes: u64,
    transition_start_credit_restored_bytes: u64,
};
const FixOutputs = struct {
    decode_slots_per_layer: u32,
    prefill_slots_per_layer: u32,
    transition_start_active_bound_bytes: u64,
    tcq3_post_prefill_phase_active_bounds: FixPhases,
    prefill_active_bound_bytes: u64,
    prefill_physical_bound_bytes: u64,
    physical_bound_bytes: u64,
    active_bound_bytes: u64,
    retirement_entry_active_bound_bytes: u64,
    tcq3_prefill_slot_storage_bound_bytes: u64,
    tcq3_final_slot_storage_bytes: u64,
    tcq3_prefill_max_slots_per_layer: u32,
    host_reserve_bytes: u64,
    allocator_limit_bytes: u64,
    steady_decode_active_bound_bytes: u64,
    resize_active_bound_bytes: u64,
    seed_active_bound_bytes: u64,
    prime_active_bound_bytes: u64,
    prefill_cache_allowance_bytes: u64,
    decode_cache_allowance_bytes: u64,
    capacity_search_ceiling: u32,
    embedding_post_prefill_credit_bytes: u64,
    tail_transition_active_credit_bytes: u64,
    tcq3_prefill_capacity: u32,
    peak_fill: struct { control: FixSummary, filled: FixSummary, total_phase_credit_bytes: u64, modeled_peak_physical_bytes: u64 },
    rowsx: ?FixRowsx = null,
};
const FixInputs = struct {
    base: i64,
    wired: i64,
    slot: u64,
    fixed: ?i64 = null,
    active_reserve: i64,
    host_reserve: i64,
    allocation: []const u8,
    embedding_rows: bool,
    tail_rows: ?u32 = null,
    io_layout: []const u8,
    matched: ?i64 = null,
    prefill_charge: i64,
    target: i64,
    search_ceiling: u32 = 160,
    rowsx_credit: ?u64 = null,
    rowsx_mlx_credit: ?u64 = null,
};
const FixCell = struct {
    name: []const u8 = "",
    receipt: []const u8 = "",
    inputs: FixInputs,
    outputs: ?FixOutputs = null,
    refusal: ?[]const u8 = null,
    /// A perturbed static envelope (the variants), else the fixture's.
    envelope: ?FixEnvelope = null,
};
const FixEnvelope = struct {
    decode_slots_per_layer: u32,
    prefill_slots_per_layer: u32,
    decode_cache_allowance_bytes: u64,
    allocator_limit_bytes: u64,
    host_reserve_bytes: u64,
    embedding_post_prefill_credit_bytes: u64,
    projection_steady_credit_bytes: u64,
    steady_decode_active_bound_bytes: u64,
    prefill_active_bound_bytes: u64,
    transition_start_active_bound_bytes: u64,
    prefill_cache_allowance_bytes: u64,
    post_prefill_reserve_bytes: u64,
    projection_seed_fixed_bytes: u64,
    retirement_owners_proved: bool,
};

fn envelopeOf(e: FixEnvelope) Envelope {
    return .{
        .predecessor_decode_rows = e.decode_slots_per_layer,
        .predecessor_prefill_rows = e.prefill_slots_per_layer,
        .decode_cache_bytes = e.decode_cache_allowance_bytes,
        .allocator_limit_at_base0 = e.allocator_limit_bytes,
        .host_reserve_bytes = e.host_reserve_bytes,
        .embedding_credit_bytes = e.embedding_post_prefill_credit_bytes,
        .projection_credit_bytes = e.projection_steady_credit_bytes,
        .steady_active_bytes = e.steady_decode_active_bound_bytes,
        .prefill_active_bytes = e.prefill_active_bound_bytes,
        .transition_start_bytes = e.transition_start_active_bound_bytes,
        .prefill_cache_bytes = e.prefill_cache_allowance_bytes,
        .post_prefill_reserve_bytes = e.post_prefill_reserve_bytes,
        .projection_seed_fixed_bytes = e.projection_seed_fixed_bytes,
        .retirement_owners_proved = e.retirement_owners_proved,
    };
}

/// The fixture cell as Inputs, or null when a value has no Zig form (a type-level refusal).
fn inputsOf(c: FixInputs) ?Inputs {
    if (c.base < 0 or c.wired < 0 or c.active_reserve < 0 or c.host_reserve < 0 or c.prefill_charge < 0) return null;
    const allocation = std.meta.stringToEnum(Allocation, c.allocation) orelse return null;
    const io = std.meta.stringToEnum(IoLayout, c.io_layout) orelse return null;
    if (c.fixed) |f| if (f < 0 or f > std.math.maxInt(u32)) return null;
    if (c.matched) |m| if (m < 0 or m > std.math.maxInt(u32)) return null;
    return .{
        .baseline_bytes = @intCast(c.base),
        .wired_bytes = @intCast(c.wired),
        .record_bytes = c.slot,
        .fixed_rows = if (c.fixed) |f| @intCast(f) else null,
        .search_ceiling = c.search_ceiling,
        .matched_prefill_rows = if (c.matched) |m| @intCast(m) else null,
        .allocation = allocation,
        .io_layout = io,
        .embedding_rows = c.embedding_rows,
        .tail_rows = c.tail_rows,
        .phase_reserve_bytes = @intCast(c.active_reserve),
        .host_reserve_bytes = @intCast(c.host_reserve),
        .prefill_charge_bytes = @intCast(c.prefill_charge),
        .peak_fill = .{ .target_bytes = @intCast(c.target) },
        .rowsx = if (c.rowsx_credit) |cr| .{ .credit_bytes = cr, .mlx_side_credit_bytes = c.rowsx_mlx_credit.? } else null,
    };
}

/// The named refusal a Python admission message is.
fn refusalOf(msg: []const u8) ?Error {
    const table = [_]struct { []const u8, Error }{
        .{ "no tcq3 decode capacity fits", error.NoDecodeCapacity },
        .{ "tcq3 prefill plus transform reserve does not fit", error.PrefillDoesNotFit },
        .{ "exceeds the matched search ceiling", error.FixedRowsAboveCeiling },
        .{ "native fixed capacity must be an integer", error.InvalidFixedRows },
        .{ "is incompatible with prefill capacity", error.FixedRowsIncompatibleWithPrefill },
        .{ "does not fit the live memory envelope", error.FixedRowsDoNotFit },
        .{ "cannot fit the matched prefill capacity", error.MatchedPrefillDoesNotFit },
        .{ "native matched prefill capacity must be", error.InvalidMatchedRows },
        .{ "causal allocation has no feasible prefill", error.NoCausalPrefill },
        .{ "tail retirement requires exactly2048", error.TailRowsNotProved },
        .{ "retirement ownership proof is required", error.RetirementProofMissing },
        .{ "retirement credit exceeds its inherited owners", error.RetirementCreditExceedsOwners },
        .{ "peak fill credit exceeds a post-prefill phase envelope", error.PhaseCreditExceedsEnvelope },
        .{ "peak fill target is outside", error.InvalidTarget },
        .{ "peak fill moved the forced decode capacity", error.PeakFillMovedFixedRows },
        .{ "peak fill moved", error.PeakFillMovedPrefill },
        .{ "peak fill reduced the admitted decode capacity", error.PeakFillReducedRows },
        .{ "filled modeled peak exceeds the target", error.PeakFillOverTarget },
        .{ "filled decode envelope exceeds the allocator limit", error.PeakFillOverAllocator },
        .{ "filled decode envelope exceeds the 100 GiB wired ceiling", error.PeakFillOverWired },
        .{ "rowsx: a forced decode capacity is set", error.CreditWithFixedRows },
        .{ "rowsx: phase geometry differs", error.CreditGeometry },
        .{ "rowsx: the credit moved", error.CreditMovedPrefill },
        .{ "adds no decode row", error.CreditAddsNoRow },
        .{ "credited modeled peak exceeds", error.CreditOverTarget },
        .{ "credit exceeds the allocator limit", error.CreditOverAllocator },
        .{ "credit exceeds the 100 GiB wired ceiling", error.CreditOverWired },
        .{ "reached the prefill-boundary gate", error.CreditReachedGate },
    };
    for (table) |t| if (std.mem.indexOf(u8, msg, t[0]) != null) return t[1];
    return null;
}

fn expectSummary(want: FixSummary, got: Summary) !void {
    try testing.expectEqual(want.decode_slots_per_layer, got.decode_rows);
    try testing.expectEqual(want.prefill_slots_per_layer, got.prefill_rows);
    try testing.expectEqual(want.tcq3_prefill_max_slots_per_layer, got.prefill_max_rows);
    try testing.expectEqual(want.tcq3_final_slot_storage_bytes, got.final_bank_bytes);
    try expectPhases(want.post_prefill_phase_active_bounds, got.phases);
    try testing.expectEqualStrings(want.binding_phase, @tagName(got.binding));
    try testing.expectEqual(want.binding_phase_active_bytes, got.binding_active_bytes);
    try testing.expectEqual(want.binding_phase_physical_bytes, got.binding_physical_bytes);
    try testing.expectEqual(want.prefill_active_bound_bytes, got.prefill_active_bytes);
    try testing.expectEqual(want.prefill_physical_bound_bytes, got.prefill_physical_bytes);
    try testing.expectEqual(want.physical_bound_bytes, got.physical_bound_bytes);
    try testing.expectEqual(want.allocator_limit_bytes, got.allocator_limit_bytes);
    try testing.expectEqual(want.active_plus_cache_bytes, got.active_plus_cache_bytes);
    try testing.expectEqual(want.wired_plus_active_plus_cache_bytes, got.wired_plus_active_plus_cache_bytes);
}

fn expectPhases(want: FixPhases, got: Phases) !void {
    try testing.expectEqual(want.growth, got.growth);
    try testing.expectEqual(want.seed, got.seed);
    try testing.expectEqual(want.prime, got.prime);
    try testing.expectEqual(want.decode, got.decode);
}

/// Every field the fixture recorded equals the plan's.
fn expectPlan(want: FixOutputs, p: Plan) !void {
    const a = p.admission;
    try testing.expectEqual(want.decode_slots_per_layer, a.decode_rows);
    try testing.expectEqual(want.prefill_slots_per_layer, a.prefill_rows);
    try testing.expectEqual(want.tcq3_prefill_capacity, a.prefill_capacity);
    try testing.expectEqual(want.transition_start_active_bound_bytes, a.transition_start_bytes);
    try expectPhases(want.tcq3_post_prefill_phase_active_bounds, a.phases);
    try testing.expectEqual(want.prefill_active_bound_bytes, a.prefill_active_bytes);
    try testing.expectEqual(want.prefill_physical_bound_bytes, a.prefill_physical_bytes);
    try testing.expectEqual(want.physical_bound_bytes, a.physical_bound_bytes);
    try testing.expectEqual(want.active_bound_bytes, a.active_bound_bytes);
    try testing.expectEqual(want.retirement_entry_active_bound_bytes, a.retirement_entry_bytes);
    try testing.expectEqual(want.tcq3_prefill_slot_storage_bound_bytes, a.prefill_bank_bound_bytes);
    try testing.expectEqual(want.tcq3_final_slot_storage_bytes, a.final_bank_bytes);
    try testing.expectEqual(want.tcq3_prefill_max_slots_per_layer, a.prefill_max_rows);
    try testing.expectEqual(want.host_reserve_bytes, a.host_reserve_bytes);
    try testing.expectEqual(want.allocator_limit_bytes, a.allocator_limit_bytes);
    try testing.expectEqual(want.steady_decode_active_bound_bytes, a.steady_bytes);
    try testing.expectEqual(want.resize_active_bound_bytes, a.resize_bytes);
    try testing.expectEqual(want.seed_active_bound_bytes, a.phases.seed);
    try testing.expectEqual(want.prime_active_bound_bytes, a.phases.prime);
    try testing.expectEqual(want.prefill_cache_allowance_bytes, a.prefill_cache_bytes);
    try testing.expectEqual(want.decode_cache_allowance_bytes, a.decode_cache_bytes);
    try testing.expectEqual(want.capacity_search_ceiling, a.search_ceiling);
    try testing.expectEqual(want.embedding_post_prefill_credit_bytes, a.embedding_credit_bytes);
    try testing.expectEqual(want.tail_transition_active_credit_bytes, a.tail_credit_bytes);
    const pf = p.peak_fill.?;
    try expectSummary(want.peak_fill.control, pf.control);
    try expectSummary(want.peak_fill.filled, pf.filled);
    try testing.expectEqual(want.peak_fill.total_phase_credit_bytes, pf.total_credit_bytes);
    try testing.expectEqual(want.peak_fill.modeled_peak_physical_bytes, pf.modeled_peak_bytes);
    try testing.expectEqual(want.rowsx != null, p.rowsx != null);
    if (want.rowsx) |w| {
        const r = p.rowsx.?;
        try testing.expectEqual(w.first_capacity_uncredited, r.uncredited_rows);
        try testing.expectEqual(w.first_capacity_credited, r.credited_rows);
        try testing.expectEqual(w.added_rows, r.added_rows);
        try testing.expectEqual(w.added_bytes, r.added_bytes);
        try testing.expectEqual(w.credit_bytes, r.credit_bytes);
        try testing.expectEqual(w.mlx_side_credit_bytes, r.mlx_side_credit_bytes);
        try testing.expectEqual(w.modeled_peak_uncredited_bytes, r.modeled_peak_uncredited_bytes);
        try testing.expectEqual(w.modeled_peak_credited_bytes, r.modeled_peak_credited_bytes);
        try testing.expectEqual(w.target_bytes, r.target_bytes);
        try testing.expectEqual(w.allocator_margin_bytes, r.allocator_margin_bytes);
        try testing.expectEqual(w.wired_margin_bytes, r.wired_margin_bytes);
        try testing.expectEqual(w.box_ceiling_margin_modeled_bytes, r.box_margin_bytes);
        try testing.expectEqual(w.baseline_bytes, r.baseline_bytes);
        try testing.expectEqual(w.transition_start_active_bound_bytes, r.restored_gate_bytes);
        try testing.expectEqual(w.transition_start_credit_restored_bytes, r.credit_bytes);
    }
}

/// One fixture cell through Admission.plan: its outputs, or its named refusal.
fn checkCell(env: Envelope, c: FixCell) !enum { planned, refused, typed } {
    const in = inputsOf(c.inputs) orelse {
        try testing.expect(c.refusal != null);
        return .typed;
    };
    const got = Admission.plan(if (c.envelope) |e| envelopeOf(e) else env, in);
    if (c.refusal) |msg| {
        const want = refusalOf(msg) orelse {
            std.debug.print("unmapped refusal: {s}\n", .{msg});
            return error.UnmappedRefusal;
        };
        testing.expectError(want, got) catch |e| {
            std.debug.print("cell {s}{s}: want {s} ({s})\n", .{ c.name, c.receipt, @errorName(want), msg });
            return e;
        };
        return .refused;
    }
    const p = got catch |e| {
        std.debug.print("cell {s}{s}: refused {s}, the Python admitted\n", .{ c.name, c.receipt, @errorName(e) });
        return e;
    };
    expectPlan(c.outputs.?, p) catch |e| {
        std.debug.print("cell {s}{s} differs\n", .{ c.name, c.receipt });
        return e;
    };
    return .planned;
}

// DSV41_PHASE4A_FIXTURE=<json from the reference runtime's dump_phase4a_admission_fixture.py>
test "dsv41 admission: every pass-2 admission, synthetic cell and refusal equals the Python stack's" {
    const path = std.mem.span(host_env.get(.DSV41_PHASE4A_FIXTURE) orelse return error.SkipZigTest);
    const a = testing.allocator;
    const Fixture = struct {
        envelope: FixEnvelope,
        receipts: []const FixCell,
        synthetic: []const FixCell,
        refusals: []const FixCell,
        canned: []const FixCell,
        variants: []const FixCell,
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    var env = envelopeOf(f.envelope);
    // The fixture has no draft-head field: its cells all run the envelope's own head (null).
    env.draft_pruned_bytes = Envelope.dsv41_pass2.draft_pruned_bytes;
    // The built-in calibration is the fixture's envelope.
    try testing.expectEqualDeep(Envelope.dsv41_pass2, env);
    var counts: [5][3]u32 = @splat(@splat(0));
    for ([_][]const FixCell{ f.receipts, f.synthetic, f.refusals, f.canned, f.variants }, 0..) |cells, k| {
        for (cells) |c| counts[k][@intFromEnum(try checkCell(env, c))] += 1;
    }
    try testing.expect(f.receipts.len >= 10 and counts[0][0] == f.receipts.len);
    try testing.expect(counts[2][0] == 0 and counts[3][0] == f.canned.len and counts[4][0] >= 3);
    std.debug.print("admission parity: {d} pass-2 receipts planned equal; synthetic {d} planned + {d} refused + {d} typed; refusal cells {d} refused + {d} typed; canned {d} planned equal; envelope variants {d} planned + {d} refused\n", .{ counts[0][0], counts[1][0], counts[1][1], counts[1][2], counts[2][1], counts[2][2], counts[3][0], counts[4][0], counts[4][1] });
}

test "dsv41 admission: the wide lane's read-ahead window is charged in every phase, the prefill included" {
    const window: u64 = 48 * exl3_record; // one more window of max_route_ids transient rows
    try testing.expectEqual(@as(u64, 639_148_032), window);
    var in0 = pass2Fast(7_755_397_656, 3_377_741_824, 140);
    in0.peak_fill = null;
    var in1 = in0;
    in1.wide_window_bytes = window;
    const a0 = (try Admission.plan(.dsv41_pass2, in0)).admission;
    const a1 = (try Admission.plan(.dsv41_pass2, in1)).admission;
    try testing.expectEqual(a0.decode_rows, a1.decode_rows);
    try expectPhaseValues(a1.phases, a0.phases.growth + window, a0.phases.seed + window, a0.phases.prime + window, a0.phases.decode + window);
    // The prefill pays with rows when it must: each row it gives up is a raw record per layer.
    try testing.expect(a1.prefill_rows <= a0.prefill_rows);
    const rows_given: u64 = @as(u64, a0.prefill_rows - a1.prefill_rows) * n_layers * raw_record;
    try testing.expectEqual(a0.retirement_entry_bytes + window - rows_given, a1.retirement_entry_bytes);
    try testing.expectEqual(a0.prefill_active_bytes + window - rows_given, a1.prefill_active_bytes);
    // AUTO rows: the window costs rows, never headroom.
    var auto0 = in0;
    auto0.fixed_rows = null;
    var auto1 = in1;
    auto1.fixed_rows = null;
    const r0 = (try Admission.plan(.dsv41_pass2, auto0)).admission;
    const r1 = (try Admission.plan(.dsv41_pass2, auto1)).admission;
    try testing.expect(r1.decode_rows <= r0.decode_rows);
    try testing.expect(r1.physical_bound_bytes <= r0.physical_bound_bytes + window);
}
