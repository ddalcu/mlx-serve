//! GLM-5.3's memory bill (G4, `sdk.MemoryBill`), built the way DeepSeek-V4.1's is (`deepseek_v41_bill.zig`, on the
//! shared machinery of `phase_memory.zig`): each phase's terms by construction from the pack's shard headers, the
//! bank's manifest, the arrays the module allocates (their shapes, and the scopes that hold them) and the limits it
//! installs, a margin only where an allocation cannot be known before it runs (each named below and sized from the box's
//! readings); the phases' records of billed against measured at the module's four boundaries (`PhaseTerms`,
//! `phase_memory.Record`); the fill over the slot rows exactly as the module gives them (`slotTable`). The construction
//! bills the longest request the load takes (the billed context and `max_output` generated tokens); the decode handover
//! grows the rows for the request it starts, to the grown bytes both its bill and its live reading admit
//! (`requestBudget`, `Live.budget`). Pure host: no MLX, no device. The fill and the
//! admission (`sdk.fill`, `sdk.admit`) take the box baseline and the target; the ceiling comes only from the host
//! (`LoadCtx.ceiling`), the margin from `LoadFacts.wired_margin_bytes`.
//!
//! ASSUMPTIONS the bill rests on, each checked where it runs:
//! - The step creates no page cache (the bank reads past it, the residents load past it): each phase record carries
//!   `file_cache_created_bytes`, which the receipts hold to `phase_memory.checkPageCache`.
//! - The module reads the trunk's residents into their buffers at construction (`sdk.Weights.materialize`; the host's
//!   loader leaves them lazy), as the draft lane does its own: the construction holds them.
//! - A layer's scope holds every array the layer builds until its end (`MlxOps.mark` / `resetTo`); a sub-scope (a span,
//!   a routed group) frees its arrays once their consumers ran.

const std = @import("std");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const sdk = @import("sdk");
const glm = @import("glm_moe_dsa.zig");
const settings = @import("settings.zig");
const bank_mod = @import("bank.zig");
const cache_mod = @import("cache.zig");
const io_mod = @import("../../sdk/weights/streamed/sdk_ext.zig").expert.io;
const mtp_mod = @import("mtp.zig");
const exl3_bank = @import("exl3_bank.zig");
const phase_mem = @import("../../sdk/weights/streamed/phase_memory.zig");
const topk_mod = @import("topk.zig");
const glm53_quant = @import("glm53_exl3/quant.zig");
const sushi_quant = @import("exl3_quant.zig");

/// The request a construction bills when the model sets no context (`ctx_size`), and the fill's floor (shared with
/// DeepSeek-V4.1's bill, `phase_memory`).
pub const fill_prompt_tokens = phase_mem.fill_prompt_tokens;
pub const fill_max_tokens = phase_mem.fill_max_tokens;
pub const min_fill_rows = phase_mem.min_fill_rows;

/// The MLX allocator cache the module installs for each phase (`mlx_set_cache_limit`), which the bill charges at its
/// limit (one value for both: `Inputs.prompt_cache`, `decode_cache_bytes`). MLX recycles a freed buffer while the cache
/// is under its limit, so the cache can end one buffer over it (`mlx_cache_overshoot`). The prompt pass's holds the
/// buffers one layer frees for the next, which reuses them at their sizes: a buffer the cache dropped is a new one, made
/// resident and zeroed by the system on its first use. Decode's limit sits over what one decode step frees for the next
/// (so its waves reuse their buffers) and under what a cache with room keeps of older ones.
pub const decode_cache_bytes: u64 = 256 << 20;
/// The prompt pass's cache limit at most (on the 256 GiB M5 Ultra, a sixteenth of its RAM).
pub const prefill_cache_max_bytes: u64 = 16 << 30;

/// The prompt pass's cache limit on a box of `ram` bytes, which is also the bill's prompt-phase cache term (one value,
/// `Inputs.prompt_cache`, for both): a sixteenth of the RAM, at most `prefill_cache_max_bytes`; with no RAM reading (0),
/// the most. The cache fills to its limit: at 16 GiB it held 17.13 GB at the end of the coding workload's 16,387-token
/// prompt pass on the box (screen stack-9d157b5), and a charge of 2 GiB left the pass's footprint peak 6.07 GB over the
/// bill.
pub fn prefillCacheBytes(ram: u64) u64 {
    return if (ram == 0) prefill_cache_max_bytes else @min(prefill_cache_max_bytes, ram / 16);
}

/// The host side past the read pool's staging (`PhaseTerms.host_side`): the footprint less MLX's active and cache less
/// the pool's staging (the manifests, the module's host state, the server and the process itself), which nothing
/// before the run can size: a margin, measured. Each phase's bound is its largest reading past the staging
/// (`host_side_readings`) plus 0.1 GB, rounded up to the next 50 MB (the handover's and decode's together). Freed GPU
/// pages the kernel has not reclaimed yet read here too, but they were MLX's bytes a moment before, which MLX's terms
/// bound: the bounds leave them out (the phase's footprint peak, its residual, holds the sum). The construction's
/// readings are its footprint's high-water (the loaders' host buffers, freed before its end).
pub const host_side_construction_bytes: u64 = 1_150_000_000;
pub const host_side_prompt_bytes: u64 = 950_000_000;
pub const host_side_decode_bytes: u64 = 1_050_000_000;
/// The readings the host-side bounds cover (the box's runs, at --ctx-size 16448): the footprint, MLX active and MLX
/// cache at the boundary, the boundary (0 construction, 1 prompt pass, 2 handover, 3 decode) and the read pool's staging
/// (`poolStagingBytes`: 89,362,432 B for the box's stream). Runs m1, m2, k1 and k2 (plugins a2aa9bf and 4bf1112, MTP
/// off) read 0.240-0.475 GB constructed, 0.693-0.837 GB at the prompt pass's end and the handover (the largest, held
/// to the prompt pass's bound here), 0.789-1.216 GB at the requests' end (the 1.216 GB reading with MLX's cache at 0:
/// unreclaimed pages); screen stack-9d157b5 read 0.386 / 0.685 / 0.684 / 0.783 GB (its construction pinned here to the
/// byte); stack 3e37e40's runs 0.377-0.383 / 0.677-0.678 / 0.677-0.845 / 0.781 GB; stack 42abec0's (its kernel set on
/// the host) 0.592-0.594 / 0.894-0.895 / 0.894-0.929 / 0.997-0.998 GB (ABBAs stack-42abec0-vs-3e37e40,
/// decode-42abec0-vs-3e37e40, stack-42abec0-devroute); stack 27f97d6's (eight read workers: 102,141,952 B of staging)
/// 0.912-0.913 GB at the requests' end (ABBA rangejobs-27f97d6-w8-vs-stack-f2be1ba); the sampler's decode on stack
/// f2be1ba (diag cachemem 79d1891) at most 0.931 GB but in its two seconds of unreclaimed pages; the closing runs of
/// e96bfe4 on the uniform 3.0 pack 1.036 GB past the staging at their construction's high-water (0.531 GB steady), its
/// source not yet found (an open item). This bill's phase records are pinned to the byte (the receipts).
pub const host_side_readings = [_][5]u64{
    .{ 197_916_120_504, 197_530_209_800, 0, 0, 89_362_432 }, // stack-9d157b5 constructed: 0.386 GB
    .{ 183_255_346_344, 182_876_769_800, 0, 0, 89_362_432 }, // diag-coding constructed: 0.379 GB
    .{ 222_534_267_376, 204_727_402_052, 17_129_442_392, 1, 89_362_432 }, // diag-coding prompt pass: 0.677 GB
    .{ 235_028_278_184, 234_347_724_356, 2700, 2, 89_362_432 }, // diag-coding handover: 0.681 GB
    .{ 235_320_611_984, 234_146_381_380, 393_242_549, 3, 89_362_432 }, // diag-coding decode: 0.781 GB
    .{ 837_000_000, 0, 0, 1, 89_362_432 }, // runs m1, m2, k1 and k2: their largest, 0.837 GB
    .{ 222_481_478_224, 204_727_190_196, 17_076_501_692, 1, 89_362_432 }, // v2-coding-probe (3e37e40) prompt pass: 0.678 GB
    .{ 593_534_776, 0, 0, 0, 89_362_432 }, // stack 42abec0 constructed: 0.594 GB (its largest)
    .{ 188_386_562_656, 187_525_320_200, 0, 0, 89_362_432 }, // v2-42-prose-probe (42abec0) construction's high-water: 0.861 GB
    .{ 895_000_000, 0, 0, 1, 89_362_432 }, // stack 42abec0's prompt pass: 0.895 GB
    .{ 929_000_000, 0, 0, 2, 89_362_432 }, // stack 42abec0's handover: 0.929 GB
    .{ 998_000_000, 0, 0, 3, 89_362_432 }, // stack 42abec0's request end: 0.998 GB
    .{ 240_617_792_208, 239_216_970_488, 387_143_032, 3, 102_141_952 }, // rangejobs-27f97d6 run 2's request end: 0.912 GB
    .{ 240_651_068_424, 239_216_954_640, 419_342_868, 3, 102_141_952 }, // rangejobs-27f97d6 run 3's request end: 0.913 GB
    .{ 1_020_000_000, 0, 0, 3, 89_362_432 }, // diag cachemem (f2be1ba) decode, the sampler's steady seconds: 0.931 GB
    .{ 1_031_000_000, 0, 0, 3, 102_141_952 }, // diag cachemem (27f97d6, the pending copy) decode, the same: 0.929 GB
    .{ 195_611_398_824, 194_498_366_984, 0, 0, 76_976_128 }, // closing e96bfe4 (the 3.0 pack) construction's high-water: 1.036 GB
};
/// The page tables of `wired` bytes (`phase_memory.wireTables`, shared with DeepSeek-V4.1's bill).
pub const wireTables = phase_mem.wireTables;
/// What the fill leaves of the box's RAM past the box's used memory (`sdk.memory.boxUsedBytes`): the free pages, the
/// file cache and the speculative pages that macOS and the other processes run in. Measured on the 256 GB M5 Ultra at
/// --ctx-size 16448, the coding workload, EXL3 3.42 bpw, typical depth 3: decode at full speed with 21.6 GB of the RAM
/// left (run x4: 10.3 GB of it free) and with 20.0 GB left (run x5: 6.1 GB free), and 16 % slower with 20.6 GB left
/// when a 15.8 GB file cache held the free pages to 3.4 GB (run d3, affine). At 16 GiB the fill took the process to a
/// 238.0 GB peak beside 17.8 GB of the rest of the box (ABBA bill-exl3-cd65722-m2-vs-a7133a3-m14: decode +27.2 %).
/// The fill's wired pages pass 88 % of the RAM: decode needs the kernel's user wire limit above them
/// (vm.global_user_wire_limit; at 88 % decode ran at half speed, run x3').
pub const os_reserve_bytes: u64 = 16 << 30;

/// The box's used memory the fill reaches beside `others` (the rest of the box): the RAM less the OS reserve, and at
/// most what keeps the process's GPU memory within `gpu_target` (the host's ceiling less its wired margin; the
/// process's host side beside it). No RAM reading (0): the GPU bound alone.
pub fn targetOf(gpu_target: u64, ram: u64, others: u64) u64 {
    const gpu = others + host_side_decode_bytes + gpu_target;
    return if (ram == 0) gpu else @min(gpu, ram -| os_reserve_bytes);
}

/// The routed call's GEMM rows in flight in a prompt layer: the wide lane's slice window (`glm.prompt_gemm_rows_in_flight`,
/// two slices of `glm.group_slice_rows`), which the lane enforces, so the bound is the schedule's, not a reading. The
/// margins before the window (read off the wave probe on the M5 Ultra's coding and prose prompts: GLM-5.3's kernels
/// 18,893 rows, sushi's MoE 34,657, billed at 20,480 and 36,864) held only where the host's encode ran no further
/// ahead of the GPU than there; a run billed before it rebuilds at its margin (`Inputs.gemm_rows_in_flight`). The
/// attention spans in flight are the graph's span window (`glm.attentionSpansInFlight`).
pub const gemm_rows_in_flight_window: u64 = glm.prompt_gemm_rows_in_flight;

/// The read pool's tickets per mini record of a routed id (the module's stream: `Stream.Options.pool.tickets`).
pub const pool_tickets_per_mini: u32 = 1024;

/// The stream the module builds, as the bill charges it.
pub const StreamShape = struct {
    /// Widest route (`Stream.Options.max_route_ids`): a prompt group's experts.
    max_route_ids: u32 = 48,
    /// Prompt routes live at once in one layer (`settings.wideDepth`).
    wide_depth: u8 = 5,
    /// Decode's transient window (`Stream.Options.decode_route_ids`): the widest decode call's routed ids.
    decode_window_rows: u32 = 48,
    /// The decode lookahead's speculative records per call.
    lookahead_budget: u32 = 2,
    /// The read pool's workers (`read_workers`).
    workers: u32 = read_workers,
};

/// The read pool's workers: as many ranges in flight as the box's SSD needs for its full rate (it reads 16.1 GB/s at
/// queue depth 4 and 16.4 at 8). With a job per range, decode at 4 / 6 / 8 / 12 workers ran 13.2 / 13.5 / 13.4 / 13.5
/// tok/s (7a101e7, seed 1234, 85 units): flat past 6; 8 is the depth at which the drive reaches its full rate.
pub const read_workers: u32 = 8;

/// The MTP draft lane's part of a bill (`glm_moe_dsa_mtp.facts`): its depth, and its residents, its experts' arrays and
/// their slot table as MLX allocates them (the lane measures them at its open and refuses more, `checkMtp`).
pub const Mtp = struct { depth: u32, resident_bytes: u64, expert_bytes: u64 };

/// Whose kernels run the prompt's routed GEMMs, each with its own arrays per routed row (`prefillRowBytes`): MLX's
/// `gather_qmm` on the affine bank, sushi's EXL3 MoE (`glm_moe_dsa_exl3_quant`), GLM-5.3's own EXL3 kernels
/// (`glm53_exl3_quant`).
pub const PromptGemm = enum { affine, sushi, glm53 };

/// The prompt GEMM of the module's quant `Q`.
pub fn promptGemmOf(comptime Q: type) PromptGemm {
    if (Q == glm53_quant) return .glm53;
    return if (Q == sushi_quant) .sushi else .affine;
}

/// How a prompt span on the DSA kernels keeps each row's top keys (`glm_moe_dsa_graph.selectSpans`): the top-k kernel
/// (`glm_moe_dsa_topk`, where it serves the model's top-k), else MLX's argpartition over every key.
pub const Select = enum { kernel, argpartition };

/// The selection `selectSpans` runs for model `c`.
pub fn selectOf(c: *const glm.Config) Select {
    return if (topk_mod.serves(c.index_topk)) .kernel else .argpartition;
}

/// The bytes MLX's allocator takes for a request while its size classes are on (`glm_moe_dsa_cache.sizeClassBytes`).
pub const sizeClassBytes = cache_mod.sizeClassBytes;

/// The allocator's size-class switch (`mlx_set_alloc_size_classes`: mlx-serve's patched MLX,
/// patches/mlx-alloc-size-classes.patch), which the module turns on for the prompt's forward layers.
pub const SizeClassSwitch = *const fn (res: *bool, on: bool) callconv(.c) c_int;

/// The switch when the process's MLX exports it, looked up by name (null on a stock MLX, and off macOS): the plugin links
/// against either MLX. The module's construction (which binds it) and the plugin's view of its bill (`inputsOf`) read the
/// same lookup, so the bill charges size classes exactly when the module turns them on.
pub fn sizeClassSwitch() ?SizeClassSwitch {
    if (comptime @import("builtin").os.tag != .macos) return null;
    // RTLD_DEFAULT ((void *)-2 on Darwin): every image the process has loaded, libmlxc among them.
    const any_image: ?*anyopaque = @ptrFromInt(@as(usize, @bitCast(@as(isize, -2))));
    const f = std.c.dlsym(any_image, "mlx_set_alloc_size_classes") orelse return null;
    return @ptrCast(@alignCast(f));
}

/// The prompt layer's arrays as the allocator sizes them: each at its own bytes, or with the size classes on at its class
/// (`sizeClassBytes`); an array the routing sizes (a routed slice's output, a GEMM wave's arrays) at its class bound,
/// 9/8 of its bytes.
const PromptAlloc = struct {
    classes: bool,

    fn of(a: PromptAlloc, bytes: u64) u64 {
        return if (a.classes) sizeClassBytes(bytes) else bytes;
    }

    fn routed(a: PromptAlloc, bytes: u64) u64 {
        return if (a.classes) bytes + bytes / 8 else bytes;
    }
};

/// What one bill reads: the model, the bank's geometry, the residents' bytes, the stream's shape and the positions
/// each phase's KV lanes hold.
pub const Inputs = struct {
    model: *const glm.Config,
    bank: bank_mod.Geometry,
    /// The trunk's residents as MLX allocates them (`glm.residentBufferBytes`: with the size classes on, at their classes).
    resident_bytes: u64,
    stream: StreamShape = .{},
    /// The prompt tokens the bill covers (every length up to it): the prompt pass's waves and its KV lanes.
    prompt_tokens: u64,
    /// The positions decode's KV lanes hold (`decodeCap`): the longest request's at construction, a request's own at
    /// its handover.
    decode_positions: u64,
    /// The draft lane when the model sets `mtp_depth` (null: off, no term).
    mtp: ?Mtp = null,
    /// The rows the draft lane holds pending at a request's handover (null: the construction's, the prompt call's rows,
    /// the most the lane holds before its append).
    decode_pending_rows: ?u64 = null,
    /// The prompt's attention on the DSA kernels (`glm_moe_dsa_graph.Routes.dsa`), else in the reference's form.
    dsa: bool,
    /// The prompt's routed GEMMs (the module's quant, `promptGemmOf`) and its spans' selection (`selectOf`).
    gemm: PromptGemm,
    select: Select,
    /// The routed rows the prompt's GEMMs hold in flight: the wide lane's window (`gemm_rows_in_flight_window`); a
    /// receipt of a run before the window sets the margin the run billed.
    gemm_rows_in_flight: u64 = gemm_rows_in_flight_window,
    /// The allocator's size classes on for the prompt's forward layers (`sizeClassBytes`): its transient arrays at their
    /// classes.
    size_classes: bool = false,
    /// The prompt pass's MLX cache limit (`prefillCacheBytes`): the module installs it, the bill charges it.
    prompt_cache: u64,
};

/// One phase's billed terms (decimal bytes; the box baseline apart), as the phase record carries them
/// (`phase_memory.Record`). MLX's active bytes hold the device terms (`device`); its cache the limit the module installs
/// for the phase and one freed buffer past it; the footprint outside MLX the host terms; `wire_tables` lies outside the
/// footprint.
pub const PhaseTerms = struct {
    /// The persistent slot rows at the phase's units, exactly as `rowsAt` gives them (`slotBytes`).
    slot_rows: u64 = 0,
    /// The stream's transient rows: the prompt's scratch (every geometry's), decode's window 0.
    transient_rows: u64 = 0,
    residents: u64 = 0,
    /// The draft lane's residents, its experts' arrays and their slot table.
    mtp_residents: u64 = 0,
    kv: u64 = 0,
    mtp_kv: u64 = 0,
    /// The draft lane's pending hidden: the prompt's final-normed rows until its first round, decode's pending rows
    /// (`mtpDecodePendingBytes`).
    mtp_pending: u64 = 0,
    waves: u64 = 0,
    mlx_cache: u64 = 0,
    mlx_cache_overshoot: u64 = 0,
    /// The read pool's staging buffers and the lookahead's speculative slots (host pages).
    pool_staging: u64 = 0,
    /// The rest of the footprint outside MLX, a measured bound.
    host_side: u64 = 0,
    wire_tables: u64 = 0,

    pub fn sum(t: PhaseTerms) u64 {
        return phase_mem.sumFields(t);
    }

    /// The terms MLX's active bytes hold.
    pub fn device(t: PhaseTerms) u64 {
        return t.slot_rows + t.transient_rows + t.residents + t.mtp_residents + t.kv + t.mtp_kv + t.mtp_pending + t.waves;
    }

    /// The wired bytes the page tables cover: every MLX byte (its active bytes and its cache).
    pub fn wired(t: PhaseTerms) u64 {
        return t.device() + t.mlx_cache + t.mlx_cache_overshoot;
    }

    /// The terms at `slots` slot-row bytes, the page tables over the wired bytes then (the construction's none: they lie
    /// outside the footprint its check compares).
    pub fn at(t: PhaseTerms, slots: u64, tables: bool) PhaseTerms {
        var x = t;
        x.slot_rows = slots;
        x.wire_tables = if (tables) wireTables(x.wired()) else 0;
        return x;
    }
};

/// The bill's phases (`Bill.phases`).
pub const Phase = enum { construction, prompt, decode };

/// One load's bill, rows-free: each phase's terms but its slot rows and its page tables (`PhaseTerms.at` adds both at
/// the phase's units), and the geometry the slot rows follow.
pub const Bill = struct {
    geometry: glm.Geometry,
    phases: [3]PhaseTerms,

    /// The phase's terms at `units` fill units; decode's past the `prompt` units the grow kept (its rows in their own
    /// arrays, `decodeSlotBytes`).
    pub fn terms(b: Bill, phase: Phase, units: u32, prompt: u32) PhaseTerms {
        const slots = if (phase == .decode) decodeSlotBytes(b.geometry, prompt, units) else slotBytes(b.geometry, units);
        return b.termsAt(phase, slots);
    }

    /// The phase's terms at `slots` slot-row bytes (a handover's: the prompt's arrays and the rows it grew).
    pub fn termsAt(b: Bill, phase: Phase, slots: u64) PhaseTerms {
        return b.phases[@intFromEnum(phase)].at(slots, phase != .construction);
    }
};

/// A stream layer's expert slots at `units` (its share of them rounded down, at most its experts): never above what
/// the units bill.
fn slotsAt(l: glm.LayerSlots, units: u32) u32 {
    const experts = l.n_records / @max(l.rows_per_unit, 1);
    const want: u32 = @intFromFloat(@floor(@as(f64, @floatFromInt(units)) * l.share));
    return @min(want, experts);
}

/// Each stream layer's rows at `units` (`Stream.Options.rows`, the grow's rows), in `out`: its rows per expert slot
/// times its share of the units, rounded down.
pub fn rowsAt(g: glm.Geometry, units: u32, out: []u32) []u32 {
    for (g.layers, out[0..g.layers.len]) |l, *r| r.* = l.rows_per_unit * slotsAt(l, units);
    return out[0..g.layers.len];
}

/// One stream layer's slot arrays over `rows` rows (GLM-only: its bank's slot geometry): each segment's array, its buffer
/// page-rounded as MLX allocates it (`glm_moe_dsa_cache.bufferBytes`).
pub fn slotArrays(l: glm.LayerSlots, rows: u64) u64 {
    if (std.mem.allEqual(u64, &l.array_bytes, 0)) return rows * l.record_bytes;
    var n: u64 = 0;
    for (l.array_bytes) |b| n += cache_mod.bufferBytes(rows * b, std.heap.pageSize());
    return n;
}

/// The slot rows' bytes at `units`, exactly as `rowsAt` gives them (the stream's arrays at the prompt's rows).
pub fn slotBytes(g: glm.Geometry, units: u32) u64 {
    var n: u64 = 0;
    for (g.layers) |l| n += slotArrays(l, @as(u64, l.rows_per_unit) * slotsAt(l, units));
    return n;
}

/// The rows a grow from `from` units to `to` adds, in their own arrays (`Stream.grow`'s ext rows).
pub fn grownBytes(g: glm.Geometry, from: u32, to: u32) u64 {
    if (to <= from) return 0;
    var n: u64 = 0;
    for (g.layers) |l| n += slotArrays(l, @as(u64, l.rows_per_unit) * (slotsAt(l, to) - slotsAt(l, from)));
    return n;
}

/// The rows a grow from each stream layer's `prompt` rows to its `rows` adds, in their own arrays (`Stream.grow`'s ext
/// rows): the split's uneven rows or the even split's.
pub fn grownOf(g: glm.Geometry, prompt: []const u32, rows: []const u32) u64 {
    var n: u64 = 0;
    for (g.layers, prompt, rows) |l, p, r| n += slotArrays(l, r -| p);
    return n;
}

/// The most units, from `prompt` to `maxUnits`, whose even grow (`grownBytes`) fits in `budget` grown bytes.
pub fn unitsIn(g: glm.Geometry, prompt: u32, budget: u64) u32 {
    var u = prompt;
    while (u < maxUnits(g) and grownBytes(g, prompt, u + 1) <= budget) u += 1;
    return u;
}

/// Decode's slot rows at `units` after a grow from the `prompt` units: the prompt's arrays and the grown ones.
pub fn decodeSlotBytes(g: glm.Geometry, prompt: u32, units: u32) u64 {
    return slotBytes(g, @min(prompt, units)) + grownBytes(g, prompt, units);
}

/// The transient scratch's bytes at `rows` rows: each geometry's arrays (`Geometry.scratch`), page-rounded.
pub fn scratchBytes(g: glm.Geometry, rows: u64) u64 {
    var n: u64 = 0;
    var any = false;
    for (g.scratch) |geo| {
        if (std.mem.allEqual(u64, &geo, 0)) continue;
        any = true;
        for (geo) |b| n += cache_mod.bufferBytes(rows * b, std.heap.pageSize());
    }
    return if (any) n else rows * g.widest_record;
}

/// One fill unit's bytes at the shares, rounded up: the fill's linear step (`sdk.MemoryBill.per_row`); the slot table
/// holds the exact bytes.
pub fn unitBytes(g: glm.Geometry) u64 {
    var n: f64 = 0;
    for (g.layers) |l| n += @as(f64, @floatFromInt(l.rows_per_unit)) * l.share * @as(f64, @floatFromInt(l.record_bytes));
    return @intFromFloat(@ceil(n));
}

/// The most units the geometry takes: the most at which some layer still takes a slot more (past it every layer is
/// full).
pub fn maxUnits(g: glm.Geometry) u32 {
    if (g.layers.len == 0) return 0;
    var n: u32 = 0;
    for (g.layers) |l| {
        // An empty stream layer takes no slot at any units.
        if (l.empty()) continue;
        const experts: f64 = @floatFromInt(l.n_records / @max(l.rows_per_unit, 1));
        n = @max(n, @as(u32, @intFromFloat(@ceil(experts / l.share))));
    }
    return n;
}

/// The slot rows' bytes at every unit count `0 .. maxUnits` (`sdk.MemoryBill.slot_table`), in `a`: the prompt's arrays,
/// or with `prompt` set (a request's handover) decode's past those units (`decodeSlotBytes`).
pub fn slotTable(a: std.mem.Allocator, g: glm.Geometry, prompt: ?u32) ![]u64 {
    const t = try a.alloc(u64, maxUnits(g) + 1);
    for (t, 0..) |*x, u| x.* = if (prompt) |p| decodeSlotBytes(g, p, @intCast(u)) else slotBytes(g, @intCast(u));
    return t;
}

// ── The prompt pass (`glm_moe_dsa_graph.forwardRows` over a call of `rows` rows ending at `positions`) ──

/// bf16, the stream's dtype; f16, GLM-5.3's kernels' copy of the routed input; f32, the router's and the indexer's.
const bf16: u64 = 2;
const f16b: u64 = 2;
const f32b: u64 = 4;

/// One prompt layer's live set at its peak, by the stages the wave probe reads (`glm_moe_dsa_graph.WaveProbe`): the
/// layer's scope holds every array it builds until its end, a sub-scope's arrays live until their consumers ran.
pub const PromptWave = struct {
    /// What lives across the layers: the residual stream into the layer, the embedding's gathered rows and its
    /// dequantized stream (the forward's scope holds them to its end), the selection a full layer carries.
    cross: u64 = 0,
    /// The layer's attention arrays (its input norm through the residual after the output projection).
    attn: u64 = 0,
    /// The spans' own arrays while they run (`glm.attentionSpansInFlight` of the layer's kind, the graph's span window)
    /// and their outputs before the join.
    spans: u64 = 0,
    /// Every span's arrays and their outputs: what the layer's spans allocate over the call.
    all_spans: u64 = 0,
    /// The routed layer's input norm and the router's arrays (a dense layer's input norm).
    router: u64 = 0,
    /// Every routed row's output, held to the combine.
    routed: u64 = 0,
    /// The routed call's GEMM transients in flight (`Inputs.gemm_rows_in_flight` rows).
    gemm: u64 = 0,
    /// GLM-5.3's kernels' f16 copy of the routed call's input (`glm53_exl3_quant.beginWide`, MLX's astype: the gate and
    /// up GEMMs' A rows), dropped with the call's waves; the other quants read x itself.
    routed_x: u64 = 0,
    /// The combine's output, the shared expert or the dense MLP, the residual adds.
    tail: u64 = 0,

    /// The largest stage: the attention, the routed call, the tail.
    pub fn total(w: PromptWave) u64 {
        const attention = w.cross + w.attn + w.spans;
        const moe = w.cross + w.attn + w.router + w.routed + w.routed_x + w.gemm;
        const tail = w.cross + w.attn + w.router + w.routed + w.tail;
        return @max(attention, @max(moe, tail));
    }
};

/// A prompt span's rows of a `rows`-row call over `positions` keys (`glm.promptSpanRows`), and the spans of the call.
fn spansOf(c: *const glm.Config, rows: u64, positions: u64) struct { rows: u64, n: u64 } {
    const r = glm.promptSpanRows(c, rows, positions);
    return .{ .rows = r, .n = std.math.divCeil(u64, rows, r) catch unreachable };
}

/// The routed call's GEMM transient bytes per routed row through the module's quant (each a GLM-only allocation: the
/// quant's own arrays). GLM-5.3's kernels (`glm53_exl3_quant.prefill`, `glm53_exl3_routes.Prompt.wave`): the gate and
/// up outputs (f32, every mini's intermediate each), the middle (f16), the wave's down output and the waves' join (bf16,
/// `hidden` each), the wave's row table (the x row and the slot, u32) and the call's order (u32). Sushi's EXL3 MoE: per
/// tensor-parallel mini the prepared gate and up inputs (f16, `hidden` each), the gate and up outputs (f16, `mini`
/// each), the middle (f16, `mini`), the down output (f16, `hidden`), and the routing table's eight words; the output's
/// reshape. The affine bank (`sdk_ext.quant.FromGatherMatmul`): the gathered input, the gate and up outputs and the
/// SwiGLU's two arrays, the down output, the order arrays.
pub fn prefillRowBytes(c: *const glm.Config, g: glm.Geometry, gemm: PromptGemm) u64 {
    const h: u64 = c.hidden_size;
    const inter: u64 = c.moe_intermediate_size;
    return switch (gemm) {
        .glm53 => 2 * inter * f32b + inter * bf16 + 2 * h * bf16 + 3 * 4,
        .sushi => blk: {
            const tp: u64 = @max(g.minis_per_id, 1);
            const mini = inter / tp;
            break :blk tp * (2 * h * bf16 + 3 * mini * bf16 + h * bf16 + 8 * 4) + h * bf16;
        },
        .affine => 2 * h * bf16 + 4 * inter * bf16 + 3 * 4,
    };
}

/// One prompt layer's wave (`PromptWave`) at `rows` rows over `positions` keys: `full` an indexer layer, `sparse` a
/// routed one, `dsa` the attention on the DSA kernels, `gemm_row` the routed call's bytes per routed row in flight.
pub fn promptWaveOf(c: *const glm.Config, rows: u64, positions: u64, full: bool, sparse: bool, dsa: bool, select: Select, gemm: PromptGemm, gemm_row: u64, gemm_rows_in_flight: u64, classes: bool) PromptWave {
    const t = rows;
    const p = positions;
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const ql: u64 = c.q_lora_rank;
    const kv: u64 = c.kv_lora_rank;
    const rope: u64 = c.qk_rope_head_dim;
    const nope: u64 = c.qk_nope_head_dim;
    const v: u64 = c.v_head_dim;
    const ih: u64 = c.index_n_heads;
    const id: u64 = c.index_head_dim;
    const topk: u64 = c.index_topk;
    const e: u64 = c.n_routed_experts;
    const k: u64 = c.n_experts_per_tok;
    const embed_bits: u64 = c.quant.bitsOf("model.embed_tokens");
    const a: PromptAlloc = .{ .classes = classes };
    var w: PromptWave = .{};
    // h; the embedding's ids, codes, scales and biases and its dequantized rows (the forward's scope); the selection.
    w.cross = a.of(t * h * bf16) + a.of(t * 4) + a.of(t * h * embed_bits / 8) + 2 * a.of(t * (h / glm.group_size) * bf16) + a.of(t * h * bf16) +
        if (p > topk) a.of(t * topk * 4) else 0;
    // The input norm, q_a and its norm, q_b, the roped query part, the latent projection, its norm, the roped key; on a
    // full layer the indexer's query (its projection and its rope) and key (projection, norm, rope); the attention's
    // output and its transposed copy, the output projection, the residual.
    w.attn = a.of(t * h * bf16) + 2 * a.of(t * ql * bf16) + a.of(t * heads * (nope + rope) * bf16) + a.of(t * heads * rope * bf16) +
        a.of(t * (kv + rope) * bf16) + a.of(t * kv * bf16) + a.of(t * rope * bf16) + 2 * a.of(t * heads * v * bf16) + 2 * a.of(t * h * bf16);
    if (full) w.attn += 2 * a.of(t * ih * id * bf16) + 3 * a.of(t * id * bf16);
    if (full and dsa) w.attn += a.of(t * ih * bf16) + 2 * a.of(t * ih * f32b);
    if (dsa) {
        // A span: the indexer's f32 score over the keys (and argpartition's u32 ids over them) and the kept ids (a full
        // layer), the query's contiguous nope rows, the absorbed query and the latent output; the spans' outputs before
        // their join.
        const sp = spansOf(c, t, p);
        const r = sp.rows;
        const score: u64 = (if (select == .kernel) @as(u64, 1) else 2) * a.of(r * p * 4);
        const span = (if (full) score + a.of(r * topk * 4) else 0) + a.of(r * heads * nope * bf16) + 2 * a.of(r * heads * kv * bf16);
        const inflight: u64 = glm.attentionSpansInFlight(full, sparse);
        const outs = if (sp.n > 1) a.of(t * heads * v * bf16) else 0;
        w.spans = @min(sp.n, inflight) * span + outs;
        w.all_spans = sp.n * span + outs;
    } else {
        // The reference's form: the latent expanded per head over every key, and the query blocks' score arrays.
        w.attn += a.of(heads * p * nope * bf16) + a.of(heads * p * v * bf16);
        w.spans = 8 * a.of(glm.score_budget_bytes) + a.of(t * heads * v * bf16);
        w.all_spans = w.spans;
    }
    if (sparse) {
        // The post-attention norm, the router's f32 input and logits, its sigmoid, biased and negated copies and
        // partition, the top-k's weights and ids, and the router's f32 weight; every routed row's output (bf16).
        w.router = a.of(t * h * bf16) + a.of(t * h * f32b) + 5 * a.of(t * e * 4) + 4 * a.of(t * k * 4) + a.of(t * 4) + a.of(e * h * f32b);
        w.routed = a.routed(t * k * h * bf16);
        w.gemm = a.routed(@min(gemm_rows_in_flight, t * k) * gemm_row);
        if (gemm == .glm53) w.routed_x = a.of(t * h * f16b);
        // The combine's output, the shared expert's four intermediates and output, the routed + shared sum, the residual.
        const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
        w.tail = a.of(t * h * bf16) + 4 * a.of(t * si * bf16) + a.of(t * h * bf16) + 2 * a.of(t * h * bf16);
    } else {
        w.router = a.of(t * h * bf16);
        // The dense MLP's four intermediates and its output (the layer's sum), the residual.
        w.tail = 4 * a.of(t * @as(u64, c.intermediate_size) * bf16) + 2 * a.of(t * h * bf16);
    }
    return w;
}

/// The prompt pass's wave term at `tokens` prompt rows (every call up to `glm.maxPromptCallRows`, over every key up to
/// the context): the most a layer's arrays add past MLX's cache at `cache_limit` (the phase's cache term, its limit and
/// one buffer). A routed layer starts with the cache full (the layers before it freed more than the limit), so its
/// wave adds whole (`PromptWave.total`, every layer kind of the model's). A dense layer (the first three) starts with
/// the cache the pass began with (`start_cache`) and the arrays of the dense layer before it: each dense layer builds
/// the same shapes, and under the limit MLX keeps what one frees for the next, which takes its buffers back. So a dense
/// layer's wave adds only what passes the limit.
pub fn promptWaveBytes(c: *const glm.Config, g: glm.Geometry, tokens: u64, dsa: bool, select: Select, gemm: PromptGemm, gemm_rows_in_flight: u64, classes: bool, cache_limit: u64, start_cache: u64) u64 {
    const t = glm.maxPromptCallRows(tokens);
    const row = prefillRowBytes(c, g, gemm);
    var worst: u64 = 0;
    var before: u64 = start_cache;
    for (0..c.n_layers) |li| {
        const l: u32 = @intCast(li);
        const w = promptWaveOf(c, t, tokens, c.isFull(l), c.isSparse(l), dsa, select, gemm, row, gemm_rows_in_flight, classes);
        if (c.isSparse(l)) {
            worst = @max(worst, w.total());
        } else {
            worst = @max(worst, (@min(cache_limit, before) + w.total()) -| cache_limit);
            // Every array the dense layer builds (all its spans) goes to the cache for the next.
            before = @max(before, start_cache + w.attn + w.all_spans + w.router + w.tail);
        }
    }
    return worst;
}

/// The prompt pass's largest single buffer (MLX's cache can end one over its limit): a routed slice's widest array
/// (`glm.group_slice_rows` rows: sushi's prepared input or down output, f16 over every mini; GLM-5.3's kernels' join or
/// the affine bank's gathered input, `hidden` wide), the query projection, the attention's output and its copy, the
/// router's f32 input, a dense layer's intermediate, a span's indexer score, the embedding's codes.
pub fn promptOvershoot(c: *const glm.Config, g: glm.Geometry, tokens: u64, gemm: PromptGemm, classes: bool) u64 {
    const t = glm.maxPromptCallRows(tokens);
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const slice = @min(glm.group_slice_rows, t * c.n_experts_per_tok);
    const sp = spansOf(c, t, tokens);
    const page = std.heap.pageSize();
    var b: u64 = @as(u64, if (gemm == .sushi) @max(g.minis_per_id, 1) else 1) * slice * h * bf16;
    b = @max(b, t * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * bf16);
    b = @max(b, t * heads * c.v_head_dim * bf16);
    b = @max(b, t * h * f32b);
    b = @max(b, t * @as(u64, c.intermediate_size) * bf16);
    b = @max(b, sp.rows * tokens * 4);
    b = @max(b, t * h * c.quant.bitsOf("model.embed_tokens") / 8);
    return if (classes) sizeClassBytes(b) else cache_mod.bufferBytes(b, page);
}

/// The draft lane's prompt part (GLM-only: the MTP layer's catch-up, `glm_moe_dsa_mtp.Lane.append`): one append's
/// arrays over at most `mtp.append_rows` rows (the embedding's ids, codes, scales and biases and its rows, both norms,
/// the hidden's cast, the concat, eh_proj, the layer's input norm, its KV and indexer keys), after the trunk's pass.
pub fn mtpPromptWaveBytes(c: *const glm.Config, tokens: u64) u64 {
    const h: u64 = c.hidden_size;
    const r = @min(tokens, mtp_mod.append_rows);
    const embed_bits: u64 = c.quant.bitsOf("model.embed_tokens");
    const embed = 4 + h * embed_bits / 8 + 2 * (h / glm.group_size) * bf16 + h * bf16;
    const keys = (@as(u64, c.kv_lora_rank) + c.qk_rope_head_dim) * bf16 + @as(u64, c.kv_lora_rank) * bf16 + @as(u64, c.qk_rope_head_dim) * bf16 + 3 * @as(u64, c.index_head_dim) * bf16;
    return r * (embed + 3 * h * bf16 + 2 * h * bf16 + h * bf16 + h * bf16 + keys);
}

/// The prompt's final-normed hidden rows the draft lane holds before its append (GLM-only: `Lane.pend` keeps the last
/// call's rows): the call's rows and the pending row before them.
pub fn mtpPendingRows(tokens: u64) u64 {
    const t = glm.maxPromptCallRows(tokens);
    return t + @intFromBool(tokens > t);
}

/// The prompt's pending hidden (`mtpPendingRows`).
pub fn mtpPendingBytes(c: *const glm.Config, tokens: u64) u64 {
    return cache_mod.bufferBytes(mtpPendingRows(tokens) * c.hidden_size * bf16, std.heap.pageSize());
}

/// Decode's pending hidden past a handover where the lane holds `rows` pending rows: those rows (a copy of them past
/// each append, `Lane.append`, so the buffer holds just them), or at most `mtp.max_pending` serial rows and a pend's
/// concat of the new rows beside them, whichever is more. The lane's pending only shrinks from `rows` or stays within
/// `max_pending`: a round drafts from all of it.
pub fn mtpDecodePendingBytes(c: *const glm.Config, rows: u64) u64 {
    const row = c.hidden_size * bf16;
    return @max(cache_mod.bufferBytes(rows * row, std.heap.pageSize()), 2 * cache_mod.bufferBytes(@as(u64, mtp_mod.max_pending) * row, std.heap.pageSize()));
}

// ── Decode (`glm_moe_dsa_graph.forwardRows` over a serial step or a verify of `rows` rows) ──

/// One decode forward's live set over `positions` keys at `rows` rows (the verify's per-row attention), by
/// construction: a routed call's barrier evaluates its layer through the router, so a forward holds one layer's arrays
/// beside the previous layer's tail (two layers); each row's selection (the f32 index keys once a full layer, each row's
/// score chain and its kept ids) and attention over its selected keys (the gathered latent and roped rows, the absorbed
/// query, the scores, the output); the router's two f32 weight copies (its own and the lookahead's next one); the
/// verify's logits on every row and the decision's f32 copies (`glm_moe_dsa_mtp.decide`); the shared expert.
pub fn decodeWaveBytes(c: *const glm.Config, rows: u64, positions: u64) u64 {
    const r = rows;
    const p = positions;
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const kv: u64 = c.kv_lora_rank;
    const rope: u64 = c.qk_rope_head_dim;
    const ih: u64 = c.index_n_heads;
    const id: u64 = c.index_head_dim;
    const topk: u64 = @min(p, c.index_topk);
    const e: u64 = c.n_routed_experts;
    const vocab: u64 = c.vocab_size;
    const proj = r * (h * bf16 + 2 * @as(u64, c.q_lora_rank) * bf16 + heads * c.qHeadDim() * bf16 + heads * rope * bf16 + 2 * ih * id * bf16 + 2 * (kv + rope) * bf16 + 3 * id * bf16);
    const select = p * id * f32b + r * (12 * ih * p + 13 * p + 4 * topk + ih * (bf16 + 2 * f32b));
    const attend = r * (topk * (kv + rope) * bf16 + heads * (rope + topk + 2 * kv + 3 * c.v_head_dim) * bf16 + 2 * h * bf16);
    const route = r * (h * bf16 + 2 * h * f32b + 10 * e * 4 + 16 * c.n_experts_per_tok * 4) + 2 * e * h * f32b;
    const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
    const moe = r * (c.n_experts_per_tok * h * (f32b + 2 * bf16) + 4 * si * bf16 + 4 * h * bf16);
    const layer = proj + select + attend + route + moe;
    const tail = r * (h * bf16 + vocab * bf16) + 8 * r * vocab * f32b;
    return 2 * layer + tail;
}

/// A draft round's MTP steps (GLM-only: the draft lane's, `glm_moe_dsa_mtp.Lane.draft`): every step's layer over one
/// row (`decodeWaveBytes` at one row) and its logits and sampling, the round's draft scope holding them all until it
/// ends; the verify runs after it.
pub fn mtpDecodeWaveBytes(c: *const glm.Config, d: u32, positions: u64) u64 {
    return @as(u64, d) * decodeWaveBytes(c, 1, positions);
}

/// Decode's MLX cache past its limit: one freed buffer, at most the widest a decode step frees (the indexer keys in
/// fp32 over every position, or a verify's logits in fp32).
pub fn decodeCacheOvershoot(c: *const glm.Config, positions: u64, verify_rows: u64) u64 {
    return @max(positions * c.index_head_dim * 4, verify_rows * c.vocab_size * 4);
}

/// The KV lanes' bytes at `positions` on every layer, each buffer as MLX allocates it (`glm_moe_dsa_cache.bytesAt`).
pub fn kvBytes(c: *const glm.Config, positions: u64) u64 {
    return cache_mod.bytesAt(c, positions, std.heap.pageSize());
}

/// The MTP layer's lanes at `positions` (its one full layer).
pub fn mtpKvBytes(c: *const glm.Config, positions: u64) u64 {
    const one = mtp_mod.layerConfig(c);
    return kvBytes(&one, positions);
}

/// The read pool's host bytes (`sdk_ext.expert.io.poolBytes`) for the pool the module's stream starts: each worker's
/// staging buffer (`StreamOf.stagingBytes`: the widest gate/up or down span, page-rounded, plus a page), the lookahead's
/// speculative slots (two per record of its budget, each a routed id's span at the widest layer), the tickets' words.
pub fn poolStagingBytes(g: glm.Geometry, s: StreamShape) u64 {
    const page = std.heap.pageSize();
    const staging = std.mem.alignForward(u64, g.widest_span, page) + page;
    return io_mod.poolBytes(.{ .workers = s.workers, .staging_bytes = staging, .tickets = pool_tickets_per_mini * g.minis_per_id, .spec = .{ .threads = 1, .slots = 2 * s.lookahead_budget, .record_bytes = g.spec_record, .chunk_bytes = page } }, page);
}

/// The bill of `in`, rows-free (`Bill`).
pub fn billOf(in: Inputs) Bill {
    const c = in.model;
    const g = in.bank;
    const s = in.stream;
    const pool = poolStagingBytes(g, s);
    const mtp_res: u64 = if (in.mtp) |m| m.resident_bytes + m.expert_bytes else 0;
    const verify_rows: u64 = s.decode_window_rows / glm.routed_top_k;
    // The prompt's cache term (its limit and one freed buffer), and the cache a later prompt starts with: decode's.
    const prompt_cache = in.prompt_cache + promptOvershoot(c, g, in.prompt_tokens, in.gemm, in.size_classes);
    const decode_cache = decode_cache_bytes + decodeCacheOvershoot(c, in.decode_positions, verify_rows);
    const prompt_waves = @max(promptWaveBytes(c, g, in.prompt_tokens, in.dsa, in.select, in.gemm, in.gemm_rows_in_flight, in.size_classes, prompt_cache, decode_cache), if (in.mtp != null) mtpPromptWaveBytes(c, in.prompt_tokens) else 0);
    const decode_waves = @max(decodeWaveBytes(c, verify_rows, in.decode_positions), if (in.mtp) |m| mtpDecodeWaveBytes(c, m.depth, in.decode_positions) else 0);
    const pending: u64 = if (in.mtp != null) mtpPendingBytes(c, in.prompt_tokens) else 0;
    const decode_pending: u64 = if (in.mtp != null) mtpDecodePendingBytes(c, in.decode_pending_rows orelse mtpPendingRows(in.prompt_tokens)) else 0;
    const scratch = scratchBytes(g, @as(u64, s.wide_depth) * s.max_route_ids * g.rows_per_id);
    return .{ .geometry = g, .phases = .{
        // Constructed: the slot rows and the scratch, the trunk's and the draft lane's residents, the pool; the KV at the
        // first prompt.
        .{ .transient_rows = scratch, .residents = in.resident_bytes, .mtp_residents = mtp_res, .pool_staging = pool, .host_side = host_side_construction_bytes },
        .{
            .transient_rows = scratch,
            .residents = in.resident_bytes,
            .mtp_residents = mtp_res,
            .kv = kvBytes(c, in.prompt_tokens),
            .mtp_kv = if (in.mtp != null) mtpKvBytes(c, in.prompt_tokens) else 0,
            .mtp_pending = pending,
            .waves = prompt_waves,
            .mlx_cache = in.prompt_cache,
            .mlx_cache_overshoot = promptOvershoot(c, g, in.prompt_tokens, in.gemm, in.size_classes),
            .pool_staging = pool,
            .host_side = host_side_prompt_bytes,
        },
        .{
            .transient_rows = scratchBytes(g, @as(u64, s.decode_window_rows) * g.rows_per_id),
            .residents = in.resident_bytes,
            .mtp_residents = mtp_res,
            .kv = kvBytes(c, in.decode_positions),
            .mtp_kv = if (in.mtp != null) mtpKvBytes(c, in.decode_positions) else 0,
            .mtp_pending = decode_pending,
            .waves = decode_waves,
            .mlx_cache = decode_cache_bytes,
            .mlx_cache_overshoot = decodeCacheOvershoot(c, in.decode_positions, verify_rows),
            .pool_staging = pool,
            .host_side = host_side_decode_bytes,
        },
    } };
}

/// The terms as the SDK's bill (`sdk.MemoryBill`): each `PhaseTerms` field one named term in the prompt and decode
/// phases, the construction's own bytes where they differ (`Term.construction`), the host side a declared bound, the
/// slot rows exact (`slot_table`: with `prompt`, decode's past those units) and the page tables following them
/// (`phase_memory.wiringAt`). The baseline stays out (the fill and the admission take it).
pub fn memoryBill(a: std.mem.Allocator, b: Bill, prompt: ?u32) !sdk.MemoryBill {
    const T = sdk.MemoryBill.Term;
    const c = b.phases[@intFromEnum(Phase.construction)];
    const p = b.phases[@intFromEnum(Phase.prompt)];
    const d = b.phases[@intFromEnum(Phase.decode)];
    const terms = try a.dupe(T, &[_]T{
        .{ .name = "slot banks (transient rows)", .bytes = .{ p.transient_rows, d.transient_rows }, .at_construction = true, .construction = c.transient_rows },
        .{ .name = "residents", .bytes = .{ p.residents, d.residents }, .at_construction = true, .construction = c.residents },
        .{ .name = "MTP residents and experts", .bytes = .{ p.mtp_residents, d.mtp_residents }, .at_construction = true, .construction = c.mtp_residents },
        .{ .name = "KV", .bytes = .{ p.kv, d.kv }, .at_construction = false },
        .{ .name = "MTP KV", .bytes = .{ p.mtp_kv, d.mtp_kv }, .at_construction = false },
        .{ .name = "MTP pending hidden", .bytes = .{ p.mtp_pending, d.mtp_pending }, .at_construction = false },
        .{ .name = "waves", .bytes = .{ p.waves, d.waves }, .at_construction = false },
        .{ .name = "MLX allocator cache", .bytes = .{ p.mlx_cache, d.mlx_cache }, .at_construction = false },
        .{ .name = "MLX cache overshoot", .bytes = .{ p.mlx_cache_overshoot, d.mlx_cache_overshoot }, .at_construction = false },
        .{ .name = "read pool staging", .bytes = .{ p.pool_staging, d.pool_staging }, .at_construction = true, .construction = c.pool_staging },
        .{ .name = "host side", .bytes = .{ p.host_side, d.host_side }, .at_construction = true, .measured = true, .construction = c.host_side },
        .{ .name = "wire tables", .bytes = .{ wireTables(p.wired()), wireTables(d.wired()) }, .at_construction = false, .with_rows = true },
    });
    errdefer a.free(terms);
    return .{ .terms = terms, .per_row = unitBytes(b.geometry), .slot_table = try slotTable(a, b.geometry, prompt), .row_terms = .{ .data = .{ p.wired(), d.wired(), 0, 0 }, .at = phase_mem.wiringAt } };
}

/// The context the module bills: the model's `ctx_size`, else the standard request's (`phase_memory.servedContext`).
pub fn servedContext(cfg: *const settings.Config) u64 {
    return phase_mem.servedContext(cfg.max_context_tokens);
}

/// Positions a decode round may append past the request's own (the verify of `mtp_depth` drafts and its token).
pub fn verifyScratch(cfg: *const settings.Config) u64 {
    return @as(u64, cfg.mtpDepth()) + 1;
}

/// Decode's KV positions for a request reserving `reserved` positions (its prompt and `max_tokens`).
pub fn decodeCap(cfg: *const settings.Config, reserved: u64) u64 {
    return reserved + verifyScratch(cfg);
}

/// The positions decode's KV lanes hold for the longest request the load takes: the billed context and `max_output`.
pub fn maxPositions(cfg: *const settings.Config) u64 {
    return decodeCap(cfg, servedContext(cfg) + cfg.maxOutput());
}

/// Decode's transient window: the routed ids of the widest decode call (a serial step's top-k, or a round's verify of
/// `mtp_depth` drafts and its token).
pub fn decodeWindowRows(cfg: *const settings.Config) u32 {
    return (cfg.mtpDepth() + 1) * glm.routed_top_k;
}

/// The stream's shape the module builds from `cfg`.
pub fn streamShape(cfg: *const settings.Config) StreamShape {
    return .{ .wide_depth = cfg.wideDepth(), .decode_window_rows = decodeWindowRows(cfg), .lookahead_budget = cfg.lookaheadBudget() };
}

/// The bill of `cfg`'s pack for prompts up to `prompt_tokens` (the SDK's view, `memoryBill`): the residents from the
/// shard headers (checked against the spec), the bank's geometry from its manifest, the prompt's attention on the DSA
/// kernels where the dims fit them (a device without the tensor units runs the reference's form, which the module's own
/// bill charges), and with `mtp_depth` set the draft lane's (the served EXL3 bank's MTP directory). Pure host.
pub fn sdkBillOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, diag: ?*glm.Diag) !sdk.MemoryBill {
    return memoryBill(a, billOf(try inputsOf(a, io, cfg, prompt_tokens, .exl3, diag)), null);
}

/// The bill's inputs of `cfg`'s pack (the construction's: decode at `maxPositions`), the EXL3 bank's prompt GEMMs the
/// served EXL3 module's (`glm_moe_dsa_module.Exl3Module`: GLM-5.3's kernels). Pure host.
pub fn inputsOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, kind: mtp_mod.BankKind, diag: ?*glm.Diag) !Inputs {
    const dir = cfg.model_dir orelse return error.GlmPackDir;
    const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
    const geo = if (exl3_bank.present(dir)) try exl3_bank.Bank.geometry(a, io, dir, model, diag) else try bank_mod.Bank.geometry(a, io, dir, model, diag);
    const classes = sizeClassSwitch() != null;
    const residents = try glm.residentBufferBytes(a, io, dir, model, std.heap.pageSize(), classes, diag);
    return .{ .model = model, .bank = geo, .resident_bytes = residents, .stream = streamShape(cfg), .prompt_tokens = prompt_tokens, .decode_positions = maxPositions(cfg), .mtp = try mtpOf(a, io, cfg, kind, diag), .dsa = glm.dsaKernelsFit(model), .gemm = if (exl3_bank.present(dir)) promptGemmOf(glm53_quant) else .affine, .select = selectOf(model), .prompt_cache = prefillCacheBytes(sdk.memory.totalMemBytes()), .size_classes = classes };
}

/// The draft lane's bill inputs when `cfg` turns it on (its settings and the pack's MTP directory checked), else null.
pub fn mtpOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, kind: mtp_mod.BankKind, diag: ?*glm.Diag) !?Mtp {
    try cfg.checkMtp();
    const d = cfg.mtpDepth();
    if (d == 0) return null;
    const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
    try mtp_mod.checkConfig(model, d, diag);
    const f = try mtp_mod.facts(a, io, cfg.model_dir orelse return error.GlmPackDir, model, kind, diag);
    return .{ .depth = d, .resident_bytes = f.resident_bytes, .expert_bytes = f.expert_bytes };
}

/// What the module needs free to load at all (the load preflight): the bill of the billed context at the fill's
/// floor rows, its process bound.
pub fn loadRequirementBytes(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, diag: ?*glm.Diag) !u64 {
    const mb = try sdkBillOf(a, io, cfg, servedContext(cfg), diag);
    defer mb.free(a);
    return mb.processBound(.{ .prompt = min_fill_rows, .decode = min_fill_rows });
}

/// The bill of `in` with decode at a request's `positions` (its own KV).
pub fn atRequest(in: Inputs, positions: u64, pending_rows: u64) Bill {
    var req = in;
    req.decode_positions = positions;
    req.decode_pending_rows = pending_rows;
    return billOf(req);
}

/// The most grown bytes `x` with `fixed + x + wireTables(wired + x)` within `target` (the page tables over the wired bytes
/// and the grown ones), null when even none fit. The page tables are monotone in `x`: a search over the bytes.
pub fn grownBudget(fixed: u64, wired: u64, target: u64) ?u64 {
    const at0 = fixed + wireTables(wired);
    if (at0 > target) return null;
    var lo: u64 = 0;
    var hi: u64 = target - at0;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (fixed + mid + wireTables(wired + mid) <= target) lo = mid else hi = mid - 1;
    }
    return lo;
}

/// The grown bytes the bill of `in` with decode at a request's `positions` admits under `target` past the `prompt` units'
/// arrays (its decode terms there, the page tables over them and the grown rows); 0 when even the prompt units are over.
pub fn requestBudget(in: Inputs, positions: u64, pending_rows: u64, prompt: u32, baseline: u64, target: u64) u64 {
    const t = atRequest(in, positions, pending_rows).termsAt(.decode, slotBytes(in.bank, prompt));
    return grownBudget(baseline + t.sum() - t.wire_tables, t.wired(), target) orelse 0;
}

/// What decode adds past MLX's active bytes at the handover of a request at `positions`, beyond its grown rows: decode's
/// window 0 (the grow allocates it), the request's decode waves, the MLX decode cache and its overshoot (MLX's); the
/// bill's decode host terms, the read pool's staging and the host side's measured bound (the footprint outside MLX).
pub fn decodeAfter(in: Inputs, positions: u64) struct { device: u64, host: u64 } {
    // None of these terms reads the lane's pending rows (MLX's active bytes hold them at the handover).
    const d = atRequest(in, positions, 0).phases[@intFromEnum(Phase.decode)];
    return .{ .device = d.transient_rows + d.waves + d.mlx_cache + d.mlx_cache_overshoot, .host = d.pool_staging + d.host_side };
}

/// The handover's live reading (MLX's active bytes, exact) and what decode adds past it (`Live.budget`).
pub const Live = struct {
    /// The box's target (`targetOf`) and MLX's (the host's ceiling less its wired margin; 0: none): the grow's, and past
    /// either even the prompt rows refuse the request.
    target: u64,
    gpu_target: u64 = 0,
    baseline: u64,
    /// MLX's active bytes after the prompt's frees settled, with the request's KV lanes at its decode cap (the wired
    /// bytes the page tables cover).
    mlx_bytes: u64,
    /// What decode adds past the reading (MLX's, wired) and the footprint outside MLX (the bill's decode host terms,
    /// `decodeAfter`).
    device_after: u64,
    host: u64,

    /// MLX's bytes at `grown` grown bytes: the reading's, the grown rows and decode's MLX additions.
    pub fn mlxAt(l: Live, grown: u64) u64 {
        return l.mlx_bytes + grown + l.device_after;
    }

    /// The box's bytes at `grown` grown bytes: the baseline, MLX's bytes (`mlxAt`), the host terms and the page tables of
    /// every MLX byte.
    pub fn total(l: Live, grown: u64) u64 {
        const m = l.mlxAt(grown);
        return l.baseline + m + l.host + wireTables(m);
    }

    /// The most grown bytes that keep the box within its target and MLX within the GPU's; refused by name when even the
    /// prompt rows do not fit.
    pub fn budget(l: Live) error{DecodeOverTarget}!u64 {
        const m0 = l.mlxAt(0);
        var b = grownBudget(l.baseline + m0 + l.host, m0, l.target) orelse return error.DecodeOverTarget;
        if (l.gpu_target != 0) b = @min(b, std.math.sub(u64, l.gpu_target, m0) catch return error.DecodeOverTarget);
        return b;
    }
};

/// The rest of the box the bill sits beside: the reading `rest`, at least `floor`
/// (`settings.Config.rest_of_box_floor_bytes`; null: the reading).
pub fn restOfBox(rest: u64, floor: ?u64) u64 {
    return @max(rest, floor orelse 0);
}

/// `rows` at most `cap` units in each phase (`settings.Config.expert_units_cap`; null: as they are).
pub fn capRows(rows: sdk.Rows, cap: ?u32) sdk.Rows {
    const c = cap orelse return rows;
    return .{ .prompt = @min(rows.prompt, c), .decode = @min(rows.decode, c) };
}

/// The bill's one construction line: every term's bytes per phase (the slot rows apart), a unit's linear bytes and the
/// units the fill chose, each phase's total at them (the slot rows exact) against the target.
pub const BillLine = struct {
    mb: *const sdk.MemoryBill,
    rows: sdk.Rows,
    baseline: u64,
    target: u64,
    context: u64,
    decode_positions: u64,

    pub fn format(b: BillLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: bill ({d}-token context, decode KV at {d} positions):", .{ b.context, b.decode_positions });
        for (b.mb.terms, 0..) |t, i| {
            const at = if (t.with_rows) [2]u64{ b.mb.total(.prompt, 0, b.rows.prompt) - b.mb.fixed(.prompt) - b.mb.slots(b.rows.prompt), b.mb.total(.decode, 0, b.rows.decode) - b.mb.fixed(.decode) - b.mb.slots(b.rows.decode) } else t.bytes;
            try w.print("{s} {s} {d} / {d} B", .{ if (i == 0) "" else ",", t.name, at[0], at[1] });
        }
        try w.print("; a row {d} B; prompt {d} rows {d} B, decode {d} rows {d} B, of the {d} B target, baseline {d} B", .{
            b.mb.per_row, b.rows.prompt, b.mb.total(.prompt, b.baseline, b.rows.prompt), b.rows.decode, b.mb.total(.decode, b.baseline, b.rows.decode), b.target, b.baseline,
        });
    }
};

const testing = std.testing;

fn glm53Config() !glm.Config {
    const text = try glm.testConfigJson(testing.allocator, ",\"model.embed_tokens\":{\"group_size\":64,\"bits\":8},\"lm_head\":{\"group_size\":64,\"bits\":8}");
    defer testing.allocator.free(text);
    return glm.Config.parse(testing.allocator, text, &glm.glm53, null);
}

/// GLM-5.3's affine bank at 4 bits: 75 routed layers of 256 records of 21,233,664 B, one row a unit.
const glm53_slots: [75]glm.LayerSlots = @splat(.{ .record_bytes = 21_233_664, .n_records = 256 });

fn glm53Geometry() bank_mod.Geometry {
    const seg = bank_mod.layerSegments(4, 6144, 2048).?;
    const record = bank_mod.logicalBytes(&seg);
    return .{ .layers = &glm53_slots, .widest_record = record, .widest_span = seg[bank_mod.gu_components].offset, .spec_record = record };
}

/// The box's EXL3 pack (davidsyoung GLM-5.3 EXL3 TR3 3.42 bpw, ~/glm53/pack-exl3-3.42bpw): per routed layer a K3 bank
/// layer of 148 experts and a K4 one of 108, four minis an expert (3,578,880 / 4,758,528 B, in records of 3,579,904 /
/// 4,759,552 B), every layer alike (its manifest's 150 stream layers).
pub const box_exl3 = struct {
    /// A slot row's arrays at K `k`: each of the nine segments over an expert's four minis (`exl3_bank.miniSegments`).
    pub fn arrays(k: u64) [9]u64 {
        const segs = exl3_bank.miniSegments(6144, 512, k).segs;
        var a: [9]u64 = undefined;
        for (&a, segs) |*x, sg| x.* = 4 * sg.length;
        return a;
    }
    pub const k3: glm.LayerSlots = .{ .record_bytes = 4 * 3_578_880, .n_records = 148, .share = 148.0 / 128.0, .array_bytes = arrays(3) };
    pub const k4: glm.LayerSlots = .{ .record_bytes = 4 * 4_758_528, .n_records = 108, .share = 108.0 / 128.0, .array_bytes = arrays(4) };
    pub const slots: [150]glm.LayerSlots = blk: {
        var s: [150]glm.LayerSlots = undefined;
        for (&s, 0..) |*l, i| l.* = if (i % 2 == 0) k3 else k4;
        break :blk s;
    };
    pub const geometry: glm.Geometry = .{ .layers = &slots, .widest_record = 4 * (3_578_880 + 4_758_528), .widest_span = 3_172_352, .minis_per_id = 4, .spec_record = 3 * 4_759_552 + 4_758_528, .scratch = .{ arrays(4), arrays(3) }, .k_mask = 1 << 3 | 1 << 4 };
    /// The trunk's residents as MLX allocates them (its 2,680 shard tensors' buffers; 20,060,118,528 B in the shards),
    /// and at their size classes (the first prompt's forward layers read them with the classes on: the embedding's and
    /// the head's codes 55,050,240 B over each, 127,205,376 B over in all).
    pub const resident_bytes: u64 = 20_060_757_504;
    pub const resident_class_bytes: u64 = 20_187_962_880;
    /// The MTP directory: 23 residents of 578,488,832 B, the experts' 18 arrays of 4,174,381,056 B and their slot table.
    pub const mtp: Mtp = .{ .depth = 3, .resident_bytes = 578_488_832, .expert_bytes = 4_174_381_056 + 1024 };
    /// The box's RAM and the fill arm's GPU target (the 245,760 MB wired limit less the 2 GiB margin).
    pub const ram: u64 = 274_877_906_944;
    pub const gpu_target: u64 = 255_550_554_112;
};

/// The uniform EXL3 pack (davidsyoung GLM-5.3 EXL3 TR3 3.0 bpw, ~/glm53/pack-exl3-3.0bpw): per routed layer one K3 bank
/// layer of all 256 experts (two slot rows a unit) and an empty second stream layer of its geometry
/// (`exl3_bank.Bank.closeLayer`).
pub const box_exl3_30 = struct {
    pub const k3: glm.LayerSlots = .{ .record_bytes = 4 * 3_578_880, .n_records = 256, .share = 2, .array_bytes = box_exl3.arrays(3) };
    pub const empty: glm.LayerSlots = .{ .record_bytes = 4 * 3_578_880, .n_records = 0, .share = 0, .array_bytes = box_exl3.arrays(3) };
    pub const slots: [150]glm.LayerSlots = blk: {
        var s: [150]glm.LayerSlots = undefined;
        for (&s, 0..) |*l, i| l.* = if (i % 2 == 0) k3 else empty;
        break :blk s;
    };
    pub const geometry: glm.Geometry = .{ .layers = &slots, .widest_record = 4 * 3_578_880, .widest_span = 2_385_920, .minis_per_id = 4, .spec_record = 3 * 3_579_904 + 3_578_880, .scratch = .{ box_exl3.arrays(3), @splat(0) }, .k_mask = 1 << 3 };
    /// The trunk's residents: the 3.42 pack's (the converter takes them from the same mixed 4/8-bit pack).
    pub const resident_bytes = box_exl3.resident_bytes;
    pub const resident_class_bytes = box_exl3.resident_class_bytes;
    /// The MTP directory: the 3.42 pack's 23 residents of 578,488,832 B, the experts' 1,024 K3 minis of 3,664,773,120 B
    /// over two bank layers (the closing runs' "MTP experts" line) and their slot table.
    pub const mtp: Mtp = .{ .depth = 3, .resident_bytes = 578_488_832, .expert_bytes = 3_664_773_120 + 1024 };
};

/// The box's served EXL3 module: GLM-5.3's prompt kernels, the spans' top-k kernel.
fn boxInputs(c: *const glm.Config, cfg: *const settings.Config) Inputs {
    return .{ .model = c, .bank = box_exl3.geometry, .resident_bytes = box_exl3.resident_bytes, .stream = streamShape(cfg), .prompt_tokens = servedContext(cfg), .decode_positions = maxPositions(cfg), .mtp = if (cfg.mtpDepth() > 0) box_exl3.mtp else null, .dsa = true, .gemm = .glm53, .select = selectOf(c), .prompt_cache = prefillCacheBytes(box_exl3.ram) };
}

test "glm bill: a fill unit is every stream layer's rows per expert slot at its share; the slot table holds the rows' exact bytes" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(u64, 75 * 21_233_664), unitBytes(glm53Geometry()));
    try testing.expectEqual(@as(u32, 256), maxUnits(glm53Geometry()));
    var out: [75]u32 = undefined;
    for (rowsAt(glm53Geometry(), 131, &out)) |r| try testing.expectEqual(@as(u32, 131), r);
    try testing.expectEqual(131 * unitBytes(glm53Geometry()), slotBytes(glm53Geometry(), 131));
    const g = box_exl3.geometry;
    // The screen's bill line: a row 2,445,926,400 B (148/128 of a K3 expert and 108/128 of a K4 one per routed layer).
    try testing.expectEqual(@as(u64, 2_445_926_400), unitBytes(g));
    try testing.expectEqual(@as(u32, 128), maxUnits(g));
    var o: [150]u32 = undefined;
    // 76 units (the 9d157b5 screen's prompt): 87.875 K3 and 64.125 K4 expert slots, rounded down.
    const r = rowsAt(g, 76, &o);
    try testing.expectEqual(@as(u32, 87), r[0]);
    try testing.expectEqual(@as(u32, 64), r[1]);
    // The screen's preload: 11,325 experts, 184.77 GB of records in arrays page-rounded as MLX allocates them: the
    // K3 layers' three 87 x 4 KiB svh / suh arrays round up a page each (75 x 3 x 4,096 B), the construction's MLX bytes
    // 197,530,209,800 B with the scratch and the draft lane (8 B more than the bill).
    try testing.expectEqual(@as(u64, 184_772_505_600 + 75 * 3 * 4096), slotBytes(g, 76));
    try testing.expectEqual(@as(u64, 197_530_209_792), slotBytes(g, 76) + scratchBytes(g, 240) + box_exl3.mtp.resident_bytes + box_exl3.mtp.expert_bytes);
    // The table is the slot arrays' bytes at every count; past the prompt's units the grow's rows are their own arrays.
    const table = try slotTable(testing.allocator, g, null);
    defer testing.allocator.free(table);
    try testing.expectEqual(@as(usize, 129), table.len);
    for (table, 0..) |x, u| try testing.expectEqual(slotBytes(g, @intCast(u)), x);
    const grown = try slotTable(testing.allocator, g, 76);
    defer testing.allocator.free(grown);
    try testing.expectEqual(slotBytes(g, 76) + grownBytes(g, 76, 86), grown[86]);
    try testing.expectEqual(table[76], grown[76]);
}

test "glm bill: the prompt pass's cache limit is a sixteenth of the RAM, at most 16 GiB, and the bill's prompt cache term is that limit" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    // The 256 GiB M5 Ultra and a 24 GiB laptop.
    for ([_][2]u64{ .{ 256 << 30, 16 << 30 }, .{ 24 << 30, 1536 << 20 } }) |box| {
        const limit = prefillCacheBytes(box[0]);
        try testing.expectEqual(box[1], limit);
        const b = billOf(.{ .model = &c, .bank = glm53Geometry(), .resident_bytes = 20_100_000_000, .prompt_tokens = 16384, .decode_positions = 16384 + 1024 + 4, .dsa = true, .gemm = .affine, .select = selectOf(&c), .prompt_cache = limit });
        try testing.expectEqual(limit, b.phases[@intFromEnum(Phase.prompt)].mlx_cache);
    }
    try testing.expectEqual(prefill_cache_max_bytes, prefillCacheBytes(0));
}

test "glm bill: GLM-5.3's terms at the 16,448-token context, by construction on the box's EXL3 pack" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    const b = billOf(boxInputs(&c, &cfg));
    const p = b.phases[@intFromEnum(Phase.prompt)];
    const d = b.phases[@intFromEnum(Phase.decode)];
    const k = b.phases[@intFromEnum(Phase.construction)];
    // The scratch: five windows of 48 ids in both geometries' rows; decode's window 0: depth 3's verify of 32 ids.
    try testing.expectEqual(@as(u64, 240 * 33_349_632), p.transient_rows);
    try testing.expectEqual(@as(u64, 32 * 33_349_632), d.transient_rows);
    // The KV lanes at the context (every buffer whole pages), the MTP layer's.
    try testing.expectEqual(@as(u64, 1_567_014_912), p.kv);
    try testing.expectEqual(@as(u64, 23_166_976), p.mtp_kv);
    // The lane's residents, experts and slot table: the open measured 4,752,870,912 B on the box.
    try testing.expectEqual(@as(u64, 4_752_870_912), p.mtp_residents);
    // The read pool: eight 3,194,880 B staging buffers, four speculative slots of a K4 expert's span, 4,096 tickets.
    try testing.expectEqual(@as(u64, 8 * 3_194_880 + 4 * 19_070_976 + 4096 * 73), p.pool_staging);
    // The prompt cache at its limit; one freed buffer past it at most: the attention's output over the 16,448 rows
    // (GLM-5.3's kernels' widest routed array, a slice's join, is 16,384 rows x hidden; sushi's every mini of it).
    try testing.expectEqual(@as(u64, 16 << 30), p.mlx_cache);
    try testing.expectEqual(@as(u64, 16448 * 64 * 256 * 2), p.mlx_cache_overshoot);
    try testing.expectEqual(@as(u64, 4 * 16384 * 6144 * 2), promptOvershoot(&c, box_exl3.geometry, 16448, .sushi, false));
    try testing.expectEqual(selectOf(&c), Select.kernel);
    // The lane's pending hidden: the prompt's 16,448 rows.
    try testing.expectEqual(@as(u64, 16448 * 6144 * 2), p.mtp_pending);
    // Decode's at construction: the prompt call's rows the lane holds before its append.
    try testing.expectEqual(@as(u64, 16448 * 6144 * 2), d.mtp_pending);
    // At a handover past the append: the lane's 64 pending rows at most, and a pend's concat beside them.
    try testing.expectEqual(@as(u64, 2 * 64 * 6144 * 2), mtpDecodePendingBytes(&c, 1));
    // The construction: the trunk's residents, no KV, no wave, no cache.
    try testing.expectEqual(p.residents, k.residents);
    try testing.expectEqual(@as(u64, 0), k.kv + k.waves + k.mlx_cache + k.mlx_cache_overshoot + k.mtp_pending);
    try testing.expectEqual(p.transient_rows, k.transient_rows);
    // The decode cache at its limit and one freed buffer: the index keys in f32 over every position.
    try testing.expectEqual(decode_cache_bytes, d.mlx_cache);
    try testing.expectEqual(@as(u64, maxPositions(&cfg) * 128 * 4), d.mlx_cache_overshoot);
    // The prompt wave's stages on a sparse full layer at 16,448 rows.
    const w = promptWaveOf(&c, 16448, 16448, true, true, true, .kernel, .glm53, prefillRowBytes(&c, box_exl3.geometry, .glm53), gemm_rows_in_flight_window, false);
    std.debug.print("\nGLM53_PROMPT_WAVE {{\"cross\": {d}, \"attn\": {d}, \"spans\": {d}, \"router\": {d}, \"routed\": {d}, \"gemm\": {d}, \"tail\": {d}, \"total\": {d}, \"billed\": {d}, \"decode\": {d}}}\n", .{ w.cross, w.attn, w.spans, w.router, w.routed, w.gemm, w.tail, w.total(), p.waves, d.waves });
    try testing.expectEqual(@as(u64, 16448 * 171_328), w.attn);
    try testing.expectEqual(@as(u64, 16448 * 8 * 6144 * 2), w.routed);
    // The GEMMs in flight: the wide lane's two slices of 16,384 rows (the schedule's bound, not a reading).
    try testing.expectEqual(@as(u64, 2 * 16384 * 45_068), w.gemm);
    // GLM-5.3's kernels' f16 copy of the call's input: its 16,448 rows x hidden, in the routed call's stage only (under
    // the attention's, so the billed waves stay); sushi's prompt reads x itself.
    try testing.expectEqual(@as(u64, 16448 * 6144 * 2), w.routed_x);
    try testing.expectEqual(@as(u64, 0), promptWaveOf(&c, 16448, 16448, true, true, true, .kernel, .sushi, prefillRowBytes(&c, box_exl3.geometry, .sushi), gemm_rows_in_flight_window, false).routed_x);
    try testing.expectEqual(@as(u64, 0), promptWaveOf(&c, 16448, 16448, true, false, true, .kernel, .glm53, prefillRowBytes(&c, box_exl3.geometry, .glm53), gemm_rows_in_flight_window, false).routed_x);
    try testing.expect(p.waves >= w.total());
    // A routed row's GEMM transients: GLM-5.3's kernels' 45,068 B, sushi's 172,160 B; a span's selection without
    // argpartition's ids over every key.
    try testing.expectEqual(@as(u64, 45_068), prefillRowBytes(&c, box_exl3.geometry, .glm53));
    try testing.expectEqual(@as(u64, 172_160), prefillRowBytes(&c, box_exl3.geometry, .sushi));
    const wa = promptWaveOf(&c, 16448, 16448, true, true, true, .argpartition, .glm53, prefillRowBytes(&c, box_exl3.geometry, .glm53), gemm_rows_in_flight_window, false);
    try testing.expectEqual(@as(u64, glm.attention_spans_in_flight.full * spansOf(&c, 16448, 16448).rows * 4 * 16448), wa.spans - w.spans);
    // The wave term is a routed layer's widest stage: the dense layers start with a cache short of its limit.
    try testing.expectEqual(p.waves, w.total());
}

test "glm bill: the allocator's size classes: a page under 1 MiB, an eighth of the power of two above; the prompt's transients at them" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(u64, 100), sizeClassBytes(100));
    try testing.expectEqual(@as(u64, 32768), sizeClassBytes(16385));
    try testing.expectEqual(@as(u64, 1 << 20), sizeClassBytes(1 << 20));
    // Just past 1 MiB: steps of 128 KiB.
    try testing.expectEqual(@as(u64, (1 << 20) + (128 << 10)), sizeClassBytes((1 << 20) + 1));
    // 16,448 rows x hidden in bf16 (202,113,024 B): past 128 MiB, steps of 16 MiB: 208 MiB.
    try testing.expectEqual(@as(u64, 13 * (16 << 20)), sizeClassBytes(16448 * 6144 * 2));
    for ([_]u64{ 1 << 20, 3 << 20, 202_113_024, 5_000_000_007, 1 << 33 }) |x| try testing.expect(sizeClassBytes(x) >= x and sizeClassBytes(x) <= x + x / 8 + 16384);
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    var in = boxInputs(&c, &cfg);
    const off = billOf(in).phases[@intFromEnum(Phase.prompt)];
    in.size_classes = true;
    const on = billOf(in).phases[@intFromEnum(Phase.prompt)];
    // Only the prompt's transients move: the waves and the cache's one buffer past its limit; at most 12.5 % more.
    try testing.expect(on.waves > off.waves and on.waves <= off.waves + off.waves / 8 + 64 * 16384);
    try testing.expect(on.mlx_cache_overshoot >= off.mlx_cache_overshoot);
    try testing.expectEqual(off.mlx_cache, on.mlx_cache);
    try testing.expectEqual(off.kv + off.residents + off.slot_rows + off.transient_rows + off.mtp_residents, on.kv + on.residents + on.slot_rows + on.transient_rows + on.mtp_residents);
    const d_off = billOf(boxInputs(&c, &cfg)).phases[@intFromEnum(Phase.decode)];
    try testing.expectEqualDeep(d_off, billOf(in).phases[@intFromEnum(Phase.decode)]);
    std.debug.print("glm bill with the size classes on: the prompt waves {d} B (off {d} B), the overshoot {d} B (off {d} B)\n", .{ on.waves, off.waves, on.mlx_cache_overshoot, off.mlx_cache_overshoot });
}

test "glm bill: the SDK's view fills the exact units, admits them, and refuses one more in either phase" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    const in = boxInputs(&c, &cfg);
    const b = billOf(in);
    const mb = try memoryBill(testing.allocator, b, null);
    defer mb.free(testing.allocator);
    const baseline: u64 = 18_144_149_504;
    const target = targetOf(box_exl3.gpu_target, box_exl3.ram, baseline);
    try testing.expectEqual(@as(u64, 257_698_037_760), target);
    const rows = try sdk.fill(mb, baseline, target, maxUnits(in.bank), min_fill_rows);
    // The phases' totals are their terms' sums at the units (the slot rows exact, the page tables over the wired bytes).
    for ([_]Phase{ .prompt, .decode }, [_]u32{ rows.prompt, rows.decode }, [_]sdk.MemoryBill.Phase{ .prompt, .decode }) |ph, u, sp| {
        // Construction's fill plans decode in the prompt's arrays (the handover bills the grow's own).
        const t = b.terms(ph, u, u);
        try testing.expectEqual(baseline + t.sum(), mb.total(sp, baseline, u));
        try testing.expect(baseline + t.sum() <= target);
        try testing.expect(baseline + b.terms(ph, u + 1, u + 1).sum() > target or u == maxUnits(in.bank) or (ph == .prompt and u == rows.decode));
    }
    try phase_mem.expectFillEdge(mb, baseline, target, maxUnits(in.bank), min_fill_rows, rows);
    // The construction terms at the prompt units: the slot rows, the scratch, the draft lane, the pool, the host side.
    try testing.expectEqual(b.terms(.construction, rows.prompt, rows.prompt).sum(), mb.constructionBytes(rows.prompt));
    try testing.expectEqual(@max(b.terms(.prompt, rows.prompt, rows.prompt).sum(), b.terms(.decode, rows.decode, rows.decode).sum()), mb.processBound(rows));
    std.debug.print("glm bill on the box's pack at 18.144 GB beside it: {d} prompt / {d} decode units (prompt {d} B, decode {d} B of the {d} B target)\n", .{ rows.prompt, rows.decode, mb.total(.prompt, baseline, rows.prompt), mb.total(.decode, baseline, rows.decode), target });
}

test "glm bill: per machine size, the units the fill takes and the bytes the request's bill grows grow with the RAM" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    // The box's other memory beside the server (19.4 GB on the M5 Ultra), the GPU's wired limit at the RAM less 16 GiB
    // less the fill arm's 2 GiB margin; the coding request (16,387 + 1,024 tokens) at its handover.
    const baseline: u64 = 19_400_000_000;
    const positions = decodeCap(&cfg, 16387 + 1024);
    var last: [3]u64 = .{ 0, 0, 0 };
    for ([_]u64{ 128, 192, 256, 512 }) |gib| {
        const ram = gib << 30;
        for ([_]bool{ false, true }) |classes| {
            var in = boxInputs(&c, &cfg);
            in.prompt_cache = prefillCacheBytes(ram);
            in.size_classes = classes;
            const mb = try memoryBill(testing.allocator, billOf(in), null);
            defer mb.free(testing.allocator);
            const target = targetOf(ram - (18 << 30), ram, baseline);
            const rows = try sdk.fill(mb, baseline, target, maxUnits(in.bank), min_fill_rows);
            const decode = @min(requestBudget(in, positions, 1, rows.prompt, baseline, target), grownBytes(in.bank, rows.prompt, maxUnits(in.bank)));
            std.debug.print("glm bill per machine: {d} GiB, target {d} B, size classes {}: {d} prompt / {d} decode units constructed, {d} B grown at the coding request\n", .{ gib, target, classes, rows.prompt, rows.decode, decode });
            // Decode's slot rows: the prompt's arrays and the grown ones.
            const slots = slotBytes(in.bank, rows.prompt) + decode;
            if (!classes) {
                try testing.expect(rows.prompt >= last[0] and rows.decode >= last[1] and slots >= last[2]);
                last = .{ rows.prompt, rows.decode, slots };
            }
        }
    }
    // The 512 GiB machine holds every expert in both phases.
    try testing.expectEqual([3]u64{ 128, 128, slotBytes(box_exl3.geometry, 128) }, last);
}

test "glm bill: the bill at a request counts its slot rows exactly; a request's own KV fills more rows than the longest one's" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    const in = boxInputs(&c, &cfg);
    const baseline: u64 = 17_900_000_000;
    const target = targetOf(box_exl3.gpu_target, box_exl3.ram, baseline);
    // The coding workload: 16,387 prompt + 1,024 generated tokens, the verify's 4 positions.
    const positions = decodeCap(&cfg, 16387 + 1024);
    // Grown from the 9d157b5 screen's 76 prompt units: decode's rows in the grow's own arrays, a byte more over the
    // target.
    // The lane holds one pending row past the prompt's append.
    const x = requestBudget(in, positions, 1, 76, baseline, target);
    const req = atRequest(in, positions, 1);
    const at76 = slotBytes(in.bank, 76);
    try testing.expect(baseline + req.termsAt(.decode, at76 + x).sum() <= target and baseline + req.termsAt(.decode, at76 + x + 1).sum() > target);
    // The whole units in it are the SDK's fill over the request's bill.
    const mb = try memoryBill(testing.allocator, req, 76);
    defer mb.free(testing.allocator);
    const units = unitsIn(in.bank, 76, x);
    try testing.expectEqual((try sdk.fill(mb, baseline, target, maxUnits(in.bank), 0)).decode, units);
    try testing.expectEqual(req.terms(.decode, units, 76), req.termsAt(.decode, at76 + grownBytes(in.bank, 76, units)));
    const worst = requestBudget(in, maxPositions(&cfg), 1, 76, baseline, target);
    try testing.expect(x > worst);
    try testing.expectEqual(@as(u64, 0), requestBudget(in, positions, 1, 76, baseline, baseline));
    // A lane holding the prompt call's rows at the handover bills them.
    try testing.expectEqual(@as(u64, 16388 * 6144 * 2), atRequest(in, positions, 16388).phases[@intFromEnum(Phase.decode)].mtp_pending);
    try testing.expect(requestBudget(in, positions, 16388, 76, baseline, target) < x);
    // What decode adds past MLX's active bytes at the handover: window 0, the waves, the cache and its overshoot; the
    // decode host terms.
    const after = decodeAfter(in, positions);
    const d = req.phases[@intFromEnum(Phase.decode)];
    try testing.expectEqual(d.transient_rows + d.waves + d.mlx_cache + d.mlx_cache_overshoot, after.device);
    try testing.expectEqual(d.pool_staging + d.host_side, after.host);
}

test "glm bill: the live budget is the most grown bytes the reading leaves under the target, and refuses below the prompt rows" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var l: Live = .{ .target = 255_550_554_112, .baseline = 12_400_000_000, .mlx_bytes = 219_000_000_000, .device_after = 1_000_000_000, .host = 1_100_000_000 };
    const x = try l.budget();
    try testing.expect(l.total(x) <= l.target and l.total(x + 1) > l.target);
    // 255.551 GB less 233.5 GB and the page tables of 220 GB of MLX bytes (24 B a page), the grown rows' own with them.
    try testing.expect(x > 21_600_000_000 and x < 21_800_000_000);
    l.mlx_bytes = 242_000_000_000;
    try testing.expectError(error.DecodeOverTarget, l.budget());
}

test "glm bill: the target is the RAM less the OS reserve, and MLX's bytes stay within the GPU's" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const gib: u64 = 1 << 30;
    const ram: u64 = 274_877_906_944;
    const gpu = 240 * gib - 2 * gib;
    try testing.expectEqual(ram - os_reserve_bytes, targetOf(gpu, ram, 17_600_000_000));
    try testing.expectEqual(@as(u64, 257_698_037_760), targetOf(gpu, ram, 17_600_000_000));
    try testing.expectEqual(17_600_000_000 + host_side_decode_bytes + 190 * gib, targetOf(190 * gib, ram, 17_600_000_000));
    try testing.expectEqual(17_600_000_000 + host_side_decode_bytes + gpu, targetOf(gpu, 0, 17_600_000_000));
    var l: Live = .{ .target = targetOf(gpu, ram, 17_600_000_000), .gpu_target = gpu, .baseline = 17_600_000_000, .mlx_bytes = 213_600_000_000, .device_after = 1_400_000_000, .host = 1_500_000_000 };
    const x = try l.budget();
    try testing.expect(l.total(x) <= l.target and l.total(x + 1) > l.target and l.mlxAt(x) <= gpu);
    // MLX's target binds below the box's.
    l.gpu_target = l.mlxAt(10_000_000_000);
    try testing.expectEqual(@as(u64, 10_000_000_000), try l.budget());
    try testing.expect(l.total(10_000_000_001) <= l.target);
    l.gpu_target = l.mlxAt(0) - 1;
    try testing.expectError(error.DecodeOverTarget, l.budget());
}

test "glm bill: the grown budget is the most bytes under the target with their page tables, and a unit count's bytes" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(?u64, null), grownBudget(100, 0, 99));
    // 1 GiB under the target beside 200 GiB wired: the grown bytes and their page tables fill it to the byte.
    const wired: u64 = 200 << 30;
    const fixed: u64 = 10 << 30;
    const target = fixed + wireTables(wired) + (1 << 30);
    const x = grownBudget(fixed, wired, target).?;
    try testing.expect(fixed + x + wireTables(wired + x) <= target and fixed + x + 1 + wireTables(wired + x + 1) > target);
    const g = box_exl3.geometry;
    try testing.expectEqual(@as(u32, 72), unitsIn(g, 72, grownBytes(g, 72, 73) - 1));
    try testing.expectEqual(@as(u32, 85), unitsIn(g, 72, grownBytes(g, 72, 85)));
    try testing.expectEqual(maxUnits(g), unitsIn(g, 72, std.math.maxInt(u64)));
    var p: [150]u32 = undefined;
    var r: [150]u32 = undefined;
    try testing.expectEqual(grownBytes(g, 72, 85), grownOf(g, rowsAt(g, 72, &p), rowsAt(g, 85, &r)));
}

test "glm bill: the MTP lane adds its residents, experts and slot table, its layer's KV, its pending hidden and its waves" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .mtp_depth = 3 };
    var in = boxInputs(&c, &cfg);
    in.mtp = null;
    const off = billOf(in);
    in.mtp = box_exl3.mtp;
    const on = billOf(in);
    for ([_]Phase{ .prompt, .decode }) |ph| {
        const a = off.phases[@intFromEnum(ph)];
        const x = on.phases[@intFromEnum(ph)];
        try testing.expectEqual(@as(u64, 0), a.mtp_residents + a.mtp_kv + a.mtp_pending);
        try testing.expectEqual(@as(u64, 4_752_870_912), x.mtp_residents);
        try testing.expect(x.mtp_kv > 0 and x.mtp_pending > 0 and x.waves >= a.waves);
    }
    // Deeper drafts bill at least as much decode wave.
    in.mtp.?.depth = 5;
    try testing.expect(billOf(in).phases[@intFromEnum(Phase.decode)].waves >= on.phases[@intFromEnum(Phase.decode)].waves);
    const mb_off = try memoryBill(testing.allocator, off, null);
    defer mb_off.free(testing.allocator);
    const mb_on = try memoryBill(testing.allocator, on, null);
    defer mb_on.free(testing.allocator);
    try testing.expect(mb_on.total(.decode, 0, 100) > mb_off.total(.decode, 0, 100) + 4_752_870_912);
}

test "glm bill: on the DSA kernels the prompt wave stops growing past one call; the reference's form grows with the keys" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const g = box_exl3.geometry;
    var prev: u64 = 0;
    for ([_]u64{ 1024, 4096, 16384 }) |n| {
        const w = promptWaveBytes(&c, g, n, true, .kernel, .glm53, gemm_rows_in_flight_window, false, 17 << 30, 0);
        try testing.expect(w > prev);
        prev = w;
    }
    try testing.expect(promptWaveBytes(&c, g, 1048576, false, .kernel, .glm53, gemm_rows_in_flight_window, false, 17 << 30, 0) > promptWaveBytes(&c, g, 32768, false, .kernel, .glm53, gemm_rows_in_flight_window, false, 17 << 30, 0) + 30_000_000_000);
    // A dense layer's wave adds only past the cache's limit: with the cache at 0 the routed layers' widest stage binds,
    // with a limit of 0 every layer's wave adds whole.
    try testing.expect(promptWaveBytes(&c, g, 16384, true, .kernel, .glm53, gemm_rows_in_flight_window, false, 0, 0) >= promptWaveBytes(&c, g, 16384, true, .kernel, .glm53, gemm_rows_in_flight_window, false, 17 << 30, 0));
    try testing.expect(decodeWaveBytes(&c, 4, 131072) > decodeWaveBytes(&c, 4, 16384));
}

test "glm bill: the rest-of-box floor only raises the rest the bill sits beside, so it only lowers the rows" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(u64, 15_400_000_000), restOfBox(15_400_000_000, null));
    try testing.expectEqual(@as(u64, 21_000_000_000), restOfBox(15_400_000_000, 21_000_000_000));
    try testing.expectEqual(@as(u64, 22_000_000_000), restOfBox(22_000_000_000, 21_000_000_000));
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    const in = boxInputs(&c, &cfg);
    const positions = decodeCap(&cfg, 16387 + 1024);
    const target = targetOf(box_exl3.gpu_target, box_exl3.ram, 15_400_000_000);
    const floored = targetOf(box_exl3.gpu_target, box_exl3.ram, restOfBox(15_400_000_000, 21_000_000_000));
    try testing.expect(requestBudget(in, positions, 1, 74, restOfBox(15_400_000_000, 21_000_000_000), floored) < requestBudget(in, positions, 1, 74, 15_400_000_000, target));
}

test "glm bill: the units cap only lowers the rows, in each phase" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const r: sdk.Rows = .{ .prompt = 72, .decode = 87 };
    try testing.expectEqual(r, capRows(r, null));
    try testing.expectEqual(sdk.Rows{ .prompt = 72, .decode = 85 }, capRows(r, 85));
    try testing.expectEqual(r, capRows(r, 90));
    try testing.expectEqual(sdk.Rows{ .prompt = 60, .decode = 60 }, capRows(r, 60));
}

test "glm bill: the bill line lists every term and each phase's total at the units chosen" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .mtp_depth = 3 };
    const mb = try memoryBill(testing.allocator, billOf(boxInputs(&c, &cfg)), null);
    defer mb.free(testing.allocator);
    const rows: sdk.Rows = .{ .prompt = 70, .decode = 80 };
    const s = try std.fmt.allocPrint(testing.allocator, "{f}", .{BillLine{ .mb = &mb, .rows = rows, .baseline = 1, .target = 3, .context = 16384, .decode_positions = maxPositions(&cfg) }});
    defer testing.allocator.free(s);
    for (mb.terms) |t| try testing.expect(std.mem.indexOf(u8, s, t.name) != null);
    var want: [128]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, s, try std.fmt.bufPrint(&want, "decode 80 rows {d} B, of the 3 B target,", .{mb.total(.decode, 1, 80)})) != null);
}

test "glm bill: every measured host side sits under its phase's bound" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const bounds = [4]u64{ host_side_construction_bytes, host_side_prompt_bytes, host_side_decode_bytes, host_side_decode_bytes };
    for (host_side_readings) |r| try testing.expect((r[0] -| r[1] -| r[2]) -| r[4] <= bounds[r[3]]);
    // Each bound: the largest reading past the staging (constructed 1.036 GB, the 3.0 pack's closing runs; the prompt
    // pass 0.806 GB, stack 42abec0's; the handover and decode 0.931 GB, the sampler's) plus 0.1 GB, rounded up to 50 MB.
    var top: [3]u64 = .{ 0, 0, 0 };
    for (host_side_readings) |r| {
        const i = @min(r[3], 2);
        top[i] = @max(top[i], (r[0] -| r[1] -| r[2]) -| r[4]);
    }
    for ([3]u64{ host_side_construction_bytes, host_side_prompt_bytes, host_side_decode_bytes }, top) |b, t| try testing.expectEqual(b, (std.math.divCeil(u64, t + 100_000_000, 50_000_000) catch unreachable) * 50_000_000);
}
