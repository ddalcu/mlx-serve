//! ANE prefill-MLP offload (perf-plan-aug-17 P5, `--ane-prefill`).
//!
//! Splits each full-width prefill chunk's dense SwiGLU MLP (and GDN input
//! projections) between the GPU (existing MLX chain) and the Apple Neural
//! Engine (private framework via lib/ane, int8 per-row weights, fp16
//! datapath). Scope: qwen3_5 family; opt-in, default OFF, LOSSY by design
//! (the decode-attn-quant precedent — quality is A/B'd per arch, bytes are
//! not expected to match).
//!
//! Programs are BANKS: every covered layer's slice is one `procedureNNN`
//! function inside one compiled program, because the private runtime
//! accepts only ~121 resident model handles (oMLX probe) and our 27B alone
//! wants 112 — twice that under dual ANE.
//!
//! Threading contract: the inference thread stays the sole MLX caller — it
//! does the plane memcpys and mlx array creation; ONLY `msv_ane_mlp_eval`
//! runs on a dedicated ANE thread. Within ONE unit evals are strictly
//! serial (one in-flight kick/wait), which is what makes that unit's shared
//! I/O planes legal (A9). Separate UNITS eval concurrently, so each owns
//! its own planes.
//!
//! Stage-A measured facts this file encodes (harness at
//! ~/claude-tmp/perf-aug17/p5-ane-mlp): full-MLP-in-one-program parity
//! cos 0.9999 vs fp32; 11.8 TFLOPS eval with the down conv K-chunked
//! (a single K=17408 conv is a 2.6x cliff — chunking lives in
//! lib/ane/ane_mlp.m); fp16 range safe to 16x with the always-on
//! (1/16 .. x16) down-conv wrap.

const std = @import("std");
const log = @import("log.zig");

// ── C ABI (lib/ane/ane_mlp.h) ──

pub const MsvAneMlp = opaque {};
pub const MsvAneBank = opaque {};
pub const MsvAnePlane = opaque {};
extern fn msv_ane_available() c_int;
extern fn msv_ane_internal_free_disk() u64;
extern fn msv_ane_plane_create(bytes: usize) ?*MsvAnePlane;
extern fn msv_ane_plane_free(p: ?*MsvAnePlane) void;
extern fn msv_ane_plane_base(p: ?*MsvAnePlane) ?[*]f16;
extern fn msv_ane_bank_create() ?*MsvAneBank;
extern fn msv_ane_bank_free(b: ?*MsvAneBank) void;
extern fn msv_ane_bank_count(b: ?*const MsvAneBank) u32;
extern fn msv_ane_bank_bytes(b: ?*const MsvAneBank) u64;
extern fn msv_ane_bank_add_mlp(
    b: ?*MsvAneBank,
    hidden: u32,
    ffn: u32,
    rows: u32,
    gate_q: [*]const i8,
    gate_s: [*]const f32,
    up_q: [*]const i8,
    up_s: [*]const f32,
    down_q: [*]const i8,
    down_s: [*]const f32,
    err: [*]u8,
    err_size: usize,
) c_int;
extern fn msv_ane_bank_add_gdn(
    b: ?*MsvAneBank,
    hidden: u32,
    qkv_out: u32,
    z_out: u32,
    rows: u32,
    qkv_q: [*]const i8,
    qkv_s: [*]const f32,
    z_q: [*]const i8,
    z_s: [*]const f32,
    err: [*]u8,
    err_size: usize,
) c_int;
extern fn msv_ane_bank_finish(
    b: ?*MsvAneBank,
    name: [*:0]const u8,
    ane_instance: c_int,
    input_plane: ?*MsvAnePlane,
    output_plane: ?*MsvAnePlane,
    err: [*]u8,
    err_size: usize,
) ?*MsvAneMlp;
extern fn msv_ane_mlp_free(m: ?*MsvAneMlp) void;
extern fn msv_ane_mlp_input(m: ?*MsvAneMlp) ?[*]f16;
extern fn msv_ane_mlp_output(m: ?*MsvAneMlp) ?[*]f16;
extern fn msv_ane_mlp_eval(m: ?*MsvAneMlp, procedure: u32, err: [*]u8, err_size: usize) c_int;
extern fn msv_ane_mlp_compile_seconds(m: ?*const MsvAneMlp) f64;
extern fn msv_ane_mlp_cache_hit(m: ?*const MsvAneMlp) c_int;

/// Whether the private AppleNeuralEngine framework is present and usable.
pub fn available() bool {
    return msv_ane_available() != 0;
}

/// Minimum useful ANE tile: below this the per-layer kick/join overhead
/// outweighs the offloaded work (the engagement floor, not a correctness
/// bound — Stage A measured flat TFLOPS down to 512 rows).
pub const ANE_MIN_ROWS: u32 = 256;

/// Rows the ANE takes out of a fixed `chunk_rows`-wide prefill chunk at
/// `share` (0..1): floored to a multiple of 32, clamped so the GPU keeps at
/// least 16 rows, 0 when the tile would be below ANE_MIN_ROWS or the share
/// is degenerate. The 32-row quantum is a measured plane-pitch contract,
/// not a preference: an fp16 plane's per-channel pitch is rows x 2 bytes,
/// and a pitch off the 64-byte grid compiles fine but fails EVERY eval
/// ("Program Inference error") — the v2 share-0.35 collapse was rows
/// 2864 = 16 mod 32 falling back to a full GPU recompute per chunk (A3
/// probe 2026-08-18: 2864/2896 dead, 2880/2912/3264/3680 all ~11.8 TFLOPS,
/// no rate cliff among legal tiles).
pub fn aneShareRows(chunk_rows: u32, share: f32) u32 {
    if (chunk_rows < ANE_MIN_ROWS + 16 or !(share > 0)) return 0;
    const raw: f32 = @as(f32, @floatFromInt(chunk_rows)) * share;
    if (!(raw > 0) or raw != raw) return 0;
    var rows: u32 = @intFromFloat(raw);
    rows -= rows % 32;
    if (rows > chunk_rows - 16) rows = (chunk_rows - 16) - (chunk_rows - 16) % 32;
    if (rows < ANE_MIN_ROWS) return 0;
    return rows;
}

/// Chip name via sysctl ("Apple M3 Ultra"); empty on failure — callers fall
/// to their default row. Canonical copy (dflash.zig delegates here); the GPU
/// arch string cannot tell Ultra from Max, hence the CPU brand.
var chip_brand_buf: [128]u8 = undefined;
var chip_brand_len: usize = 0;
/// 0 = unread, 1 = one thread is reading it, 2 = published.
var chip_brand_state = std.atomic.Value(u8).init(0);

/// Cached `machdep.cpu.brand_string` — THE accessor for every per-silicon
/// table (ANE share, MTP depth cap, DFlash block cap). The chip cannot change
/// under a running process and these are read on request-shaped paths
/// (Generator init), so the sysctl runs exactly once. Callers that need to
/// inject a chip string take it as a parameter; nobody re-wraps this.
pub fn chipBrand() []const u8 {
    if (chip_brand_state.load(.acquire) != 2) {
        if (chip_brand_state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) {
            chip_brand_len = chipBrandString(&chip_brand_buf).len;
            chip_brand_state.store(2, .release);
        } else {
            while (chip_brand_state.load(.acquire) != 2) std.atomic.spinLoopHint();
        }
    }
    return chip_brand_buf[0..chip_brand_len];
}

fn chipBrandString(buf: []u8) []const u8 {
    var len: usize = buf.len;
    if (std.c.sysctlbyname("machdep.cpu.brand_string", buf.ptr, &len, null, 0) != 0) return "";
    if (len > 0 and buf[len - 1] == 0) len -= 1;
    return buf[0..len];
}

/// Per-(mode, silicon) default share. Every row is a MEASURED optimum of its
/// own sweep — never interpolated (a share change needs its own A/B):
///   channel / M4 Max: 0.45 (2026-08-18, Qwen3.8-27B oQ4e: 0.40/0.45/0.50
///     ranked 305/311/297 at 16k — rollover at 0.50).
///   channel / M3 Ultra: 0.35 (PR #223 tester, 2026-08-19, same pack 4-bit:
///     0.45 ≈ +1% at 32k — nothing; sweep 0.45/0.40/0.35/0.30 ranked
///     398/421/440/438 at 32k; clean post-reboot 0.35 A/B +0.2%/+8.5%/+13.7%
///     at 8k/16k/32k, reproduced 3x incl. across a reboot — and the smaller
///     share also drops the int8 copy 9.47 → 7.30 GB).
///   row / M4 Max: 0.40 (2026-08-17: 0.30 +10/+14%, 0.40 +12/+18%, 0.50
///     regresses). Row is unmeasured elsewhere and keeps the M4 row.
/// The DUAL default is deliberately the same number: MLX_SERVE_ANE_SPLIT is
/// the TOTAL ANE share either way, halved across the units, so every
/// measurement above carries over unchanged — and the dual optimum is
/// EXPECTED to sit higher (halving the ANE critical path is the whole
/// point), which is a re-sweep, not an interpolation.
pub fn defaultShare(mode: Mode, chip: []const u8) f32 {
    return defaultShareFor(mode, chip, dualDefault(chip));
}

/// M3 Ultra: 0.35 on one ANE (PR #223 tester: 0.45 was nothing, 0.35
/// +8.5%/+13.7% at 16k/32k); with BOTH ANEs 0.50 (2026-08-22 sweep, 16k
/// prefill: single 0.35 465 tok/s, dual 0.35 471, 0.45 482-492, 0.50 498,
/// 0.55 490, 0.65 466 — the rollover is one notch past 0.50).
pub fn defaultShareFor(mode: Mode, chip: []const u8, dual: bool) f32 {
    if (mode == .row) return 0.40;
    if (std.mem.indexOf(u8, chip, "M3 Ultra") != null) return if (dual) 0.50 else 0.35;
    return 0.45;
}

/// Dual ANE is the default on the M3 Ultra: measured 2026-08-22 on a 512 GB
/// box, both instances at equal eval counts, zero failures, both IOReport
/// ANE0_ counters moving, +7.1% 16k prefill over the single-ANE best.
pub fn dualDefault(chip: []const u8) bool {
    return std.mem.indexOf(u8, chip, "M3 Ultra") != null;
}

/// The ANE's TOTAL share of each covered projection (channel mode: fraction
/// of output channels, split evenly across the units; row mode: fraction of
/// chunk token rows). MLX_SERVE_ANE_SPLIT overrides; the default is per
/// (mode, silicon) — see `defaultShare`.
pub fn splitShare() f32 {
    const def = defaultShare(splitMode(), chipBrand());
    const raw = std.c.getenv("MLX_SERVE_ANE_SPLIT") orelse return def;
    const v = std.fmt.parseFloat(f32, std.mem.sliceTo(raw, 0)) catch return def;
    if (!(v > 0) or v > 1) return def;
    return v;
}

/// ANE prefill is for M4-and-below: on NAX-class GPUs (M5+) the GPU prefill
/// already outruns the seam — measured a LOSS on M5 Max (channel 0.45 median
/// -11%/-7.5% at 16k/32k, PR #223, two testers). MLX_SERVE_ANE_FORCE=1 keeps
/// the build for future silicon measurement (M6 etc.). Pure so it is
/// hermetically testable; the scheduler passes the live NAX probe + env.
pub fn anePrefillAllowed(nax_available: bool, force_env: ?[]const u8) bool {
    if (!nax_available) return true;
    if (force_env) |v| return v.len > 0 and v[0] == '1';
    return false;
}

/// GDN input-projection offload beside the MLP one (v2). MLX_SERVE_ANE_GDN=0
/// keeps `--ane-prefill` MLP-only — the attribution lever for A/Bs.
pub fn gdnEnabled() bool {
    const raw = std.c.getenv("MLX_SERVE_ANE_GDN") orelse return true;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

/// How each projection splits between ANE and GPU (A1, ane-plan-aug-18).
/// `row`: the ANE takes the first `aneShareRows` token rows through the
/// FULL weights (v1/v2 shipped design; int8 copy = 100% of covered
/// weights). `channel`: both units see ALL chunk tokens through sliced
/// weights — the ANE holds output channels [0..k) of gate/up (and qkv/z),
/// the GPU the rest; the down projection contributes PARTIAL sums added at
/// the seam — so the resident int8 copy scales with the share. Channel is
/// the DEFAULT since the 2026-08-18 counterbalanced A/B (27B oQ4e, M4 Max,
/// medians): channel-0.45 306.4/301.1 vs row-0.40 300.7/294.3 vs off
/// 258.2/239.0 at 16k/32k — +18.7%/+26.0% over off at 9.3 GB ANE bytes
/// against row's 20.4. MLX_SERVE_ANE_MODE=row restores the row split.
pub const Mode = enum { row, channel };

pub fn splitMode() Mode {
    const raw = std.c.getenv("MLX_SERVE_ANE_MODE") orelse return .channel;
    if (std.mem.eql(u8, std.mem.sliceTo(raw, 0), "row")) return .row;
    return .channel;
}

/// Dual-ANE split: two pinned units computing disjoint channel slices
/// concurrently. Default ON where it was measured (M3 Ultra), off elsewhere
/// (unmeasurable on single-ANE silicon; private API on top of private API).
pub fn dualEnabled() bool {
    return dualEnabledFrom(if (std.c.getenv("MLX_SERVE_ANE_DUAL")) |p| std.mem.sliceTo(p, 0) else null, chipBrand());
}

/// MLX_SERVE_ANE_DUAL=1 forces it on, =0 off; unset = the chip default.
pub fn dualEnabledFrom(raw: ?[]const u8, chip: []const u8) bool {
    if (raw) |v| {
        if (std.mem.eql(u8, v, "1")) return true;
        if (std.mem.eql(u8, v, "0")) return false;
    }
    return dualDefault(chip);
}

/// Whether concurrent units share ONE input surface instead of getting a
/// memcpy'd copy each. Default OFF: a shared input across two LIVE evals is
/// an unproven read-concurrency assumption on private API whose failure
/// mode is silently wrong numbers, and the copy is ~1.7 ms on the 27B
/// against a ~20 ms eval. Flip it once dual is proven.
pub fn dualShareInput() bool {
    const raw = std.c.getenv("MLX_SERVE_ANE_DUAL_SHARE_INPUT") orelse return false;
    return std.mem.eql(u8, std.mem.sliceTo(raw, 0), "1");
}

/// Neural Engine instances a chip carries. Ultra parts fuse two dies and
/// expose two ANE services with two IOReport counters (H11ANE/H11ANE1,
/// `macpow --dump | grep ANE0_` — measured on an M3 Ultra, PR #223, where
/// all energy landed on ANE0_0 because nothing named a die); everything
/// else has one.
pub fn aneInstanceCount(chip: []const u8) u32 {
    return if (std.mem.indexOf(u8, chip, "Ultra") != null) 2 else 1;
}

/// Ceiling on units — the seams size their fixed part arrays from it, and
/// no shipping silicon exposes more than two Neural Engines.
pub const MAX_UNITS: usize = 2;

/// Units the engine builds. Dual is CHANNEL-mode only (row mode is not the
/// default and is unmeasured), and only on silicon that has a second
/// instance to pin to — a dual request on a single-ANE machine self-
/// disables rather than failing the boot.
pub fn unitCount(mode: Mode, chip: []const u8, dual: bool) u32 {
    if (!dual or mode != .channel) return 1;
    return aneInstanceCount(chip);
}

/// Channel-slice boundary alignment: a multiple of 128 keeps the GPU-side
/// complement slices legal for every quant geometry we serve (group sizes
/// 32/64/128; packed-word boundaries at any bits in {2,3,4,5,6,8}) and the
/// ANE-side weights nicely tiled.
pub const CHANNEL_ALIGN: u32 = 128;

/// PER-UNIT slice of a projection's output channels at a TOTAL `share`
/// spread over `units`: floored to CHANNEL_ALIGN, clamped so the GPU keeps
/// at least CHANNEL_ALIGN channels after every unit's slice, 0 when the
/// slice degenerates. Unit u takes [u*k, (u+1)*k); the GPU takes
/// [units*k, width). The down conv's K-chunkability is guaranteed by
/// construction for 128-aligned widths (a power-of-two divisor <= 16 always
/// lands under the K cliff), mirrored in the test rather than walked here.
pub fn channelSliceWidthUnits(width: u32, share: f32, units: u32) u32 {
    if (units == 0 or !(share > 0)) return 0;
    if (width < (units + 1) * CHANNEL_ALIGN) return 0;
    const raw: f32 = @as(f32, @floatFromInt(width)) * share / @as(f32, @floatFromInt(units));
    if (!(raw > 0) or raw != raw) return 0;
    var k: u32 = @intFromFloat(raw);
    k -= k % CHANNEL_ALIGN;
    const room = (width - CHANNEL_ALIGN) / units;
    const cap = room - room % CHANNEL_ALIGN;
    if (k > cap) k = cap;
    if (k < CHANNEL_ALIGN) return 0;
    return k;
}

pub fn channelSliceWidth(width: u32, share: f32) u32 {
    return channelSliceWidthUnits(width, share, 1);
}

/// What the offload will actually cost in bytes, computable from the config
/// BEFORE any dequant: the int8 weight copies (dense MLP: gate/up/down; GDN:
/// the fused qkv+z stack) plus the fp16 IOSurface planes. Widths are
/// PER-UNIT, so the int8 total is `units` x the per-unit slice — i.e. the
/// same channels either way, which is exactly why the dual bill is not
/// bigger than the single one. Planes are per UNIT (concurrent evals cannot
/// share an output surface) but shared per shape class WITHIN a unit (A9):
/// input, MLP output, GDN output — ~11 GB back on the 27B vs the old
/// per-program pairs. Per-row fp16 scales are noise next to these (out_dim
/// × 2 bytes per weight) and deliberately not billed.
pub fn engineBillBytes(dense_layers: u64, gdn_layers: u64, hidden: u64, ffn: u64, qkv_out: u64, z_out: u64, rows: u64, units: u64) u64 {
    const dense_int8 = dense_layers * 3 * hidden * ffn;
    const gdn_int8 = gdn_layers * (qkv_out + z_out) * hidden;
    var planes: u64 = 0;
    if (dense_layers > 0 or gdn_layers > 0) planes += hidden * rows * 2; // input
    if (dense_layers > 0) planes += hidden * rows * 2; // MLP output
    if (gdn_layers > 0) planes += (qkv_out + z_out) * rows * 2; // GDN output
    return (dense_int8 + gdn_int8 + planes) * units;
}

/// The non-model part of the gate's headroom: OS, other apps, MLX's own
/// reclaimable pool. Everything that scales with the checkpoint is computed
/// per model by `server.aneGateHeadroom`.
pub const GATE_BASELINE_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// Context the gate RESERVES KV for before admitting an offload.
///
/// This is what keeps the offload from eating the advertised context. The ANE
/// int8 copies come out of the same memory the KV cache is sized from, and
/// auto-context is pinned AFTER the build — so with no reserve, admitting the
/// offload silently shrank the number clients read once per session (measured
/// 2026-08-20, Qwen3.8-27B iQ on a 32 GB M1 Pro: 97,280 tokens off, 5,120 on).
/// Reserving the KV up front means an offload is admitted only if a usable
/// context still fits beside it, and the sizer then finds that memory free.
pub const MIN_CONTEXT_TOKENS: u32 = 32768;

/// The per-model admission gate (replaces the v1 flat 96 GB total-RAM
/// check, which refused a 1 GB bill on a 64 GB Mac and said nothing about
/// WHY): the bill is admitted when resident + bill + headroom fits total
/// RAM. An unknown total (sysctl failure) is no information — allow, the
/// server's other memory guards still stand.
pub fn gateAllows(total_mem: u64, resident: u64, bill: u64, headroom: u64) bool {
    if (total_mem == 0) return true;
    return resident +| bill +| headroom <= total_mem;
}

/// The hard floor for starting an ANE build at all: below this much free
/// internal disk even cache RESTORES and the framework's own model saves
/// start failing bare ("Write weightsFilePath failed" in the unified log),
/// and the build ships silent partial coverage — the 2026-08-18 class. The
/// build is SKIPPED with a named refusal instead.
pub const BUILD_DISK_FLOOR_BYTES: u64 = 1 << 30;

/// Ceiling on ONE bank's weight blob. Two things bound it: oMLX hit an
/// 0x20004 load failure past roughly a 4 GiB per-instance device address
/// window, and the builder holds a group's quantized payloads AND its
/// assembled blob at once, so the cap is also the build's transient host
/// peak (2x). Under the cap a model banks monolithically — which is what
/// oMLX measured bit-stable across five greedy runs, against split banks
/// that were ~1% faster but occasionally diverged at a tie.
/// MLX_SERVE_ANE_BANK_MAX_BYTES overrides (the split-ladder test lever).
pub const DEFAULT_BANK_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub fn bankMaxBytes() u64 {
    const raw = std.c.getenv("MLX_SERVE_ANE_BANK_MAX_BYTES") orelse return DEFAULT_BANK_MAX_BYTES;
    const v = std.fmt.parseInt(u64, std.mem.sliceTo(raw, 0), 10) catch return DEFAULT_BANK_MAX_BYTES;
    return if (v == 0) DEFAULT_BANK_MAX_BYTES else v;
}

/// Programs the next bank takes from `sizes[start..]`: at least one (a
/// single program over the cap still has to be its own bank — refusing it
/// would only move the failure), then as many more as fit under `cap`,
/// never more than `group`. `group` is the split ladder's current rung: a
/// failed bank retries at half this count, down to one, and only then does
/// the program get dropped to the GPU.
pub fn bankGroupLen(sizes: []const u64, start: usize, group: usize, cap: u64) usize {
    if (start >= sizes.len or group == 0) return 0;
    var n: usize = 1;
    var bytes: u64 = sizes[start];
    while (n < group and start + n < sizes.len) : (n += 1) {
        const next = bytes + sizes[start + n];
        if (next > cap) break;
        bytes = next;
    }
    return n;
}

/// Free bytes on the internal volume that bounds the ANE compile-session
/// budget (aned's per-compile scratch lives in root tmp for the client's
/// lifetime, wherever OUR staging is). 0 = probe failed, no information.
pub fn internalFreeDiskBytes() u64 {
    return msv_ane_internal_free_disk();
}

/// Live ANE totals across resident engines, for the `--metrics` gauges and
/// anything else that wants "what is the ANE holding right now" without
/// walking the registry. Published by `publishLive` after a successful
/// build, retired by deinit; zero whenever no engine is resident (the
/// zero-when-off metrics invariant).
pub var live_int8_bytes = std.atomic.Value(u64).init(0);
pub var live_layers = std.atomic.Value(u64).init(0);

pub const RowQuant = struct {
    q: []i8,
    s: []f32,

    pub fn deinit(self: *RowQuant, allocator: std.mem.Allocator) void {
        allocator.free(self.q);
        allocator.free(self.s);
    }
};

/// Per-output-row symmetric int8 quantization of a dense row-major [n, k]
/// f32 weight: s[i] = max|row i| / 127, q = round(w / s) clamped to ±127.
/// An all-zero row gets scale 0 and zero codes (dequantizes to zero).
pub fn quantizeRowsInt8(allocator: std.mem.Allocator, w: []const f32, n: usize, k: usize) !RowQuant {
    std.debug.assert(w.len == n * k);
    const q = try allocator.alloc(i8, n * k);
    errdefer allocator.free(q);
    const s = try allocator.alloc(f32, n);
    errdefer allocator.free(s);
    for (0..n) |i| {
        const row = w[i * k .. (i + 1) * k];
        var amax: f32 = 0;
        for (row) |v| {
            const a = @abs(v);
            if (a > amax) amax = a;
        }
        if (amax == 0) {
            s[i] = 0;
            @memset(q[i * k .. (i + 1) * k], 0);
            continue;
        }
        const scale = amax / 127.0;
        s[i] = scale;
        const inv = 1.0 / scale;
        for (row, 0..) |v, j| {
            const r = @round(v * inv);
            q[i * k + j] = @intFromFloat(std.math.clamp(r, -127.0, 127.0));
        }
    }
    return .{ .q = q, .s = s };
}

/// Where one layer's slice lives: which compiled bank, and which procedure
/// inside it.
pub const ProgramRef = struct { bank: *MsvAneMlp, proc: u32 };

/// One program awaiting banking: the quantized payloads stay owned here
/// until a bank is assembled, so a failed bank can be rebuilt at a smaller
/// rung without re-dequantizing on the inference thread.
const Pending = struct {
    layer: u32,
    gdn: bool,
    ffn: u32 = 0,
    qkv_out: u32 = 0,
    z_out: u32 = 0,
    a: RowQuant, // gate | qkv
    b: RowQuant, // up   | z
    c: ?RowQuant = null, // down (MLP only)

    fn bytes(self: *const Pending) u64 {
        var n: u64 = self.a.q.len + self.b.q.len;
        if (self.c) |c| n += c.q.len;
        return n;
    }

    fn deinit(self: *Pending, allocator: std.mem.Allocator) void {
        self.a.deinit(allocator);
        self.b.deinit(allocator);
        if (self.c) |*c| c.deinit(allocator);
    }
};

/// One Neural Engine's worth of the offload: its own compiled banks, its
/// own I/O planes, and its own eval thread. Single-ANE is exactly one unit
/// at instance 0 (no affinity hint — byte-identical to every build before
/// the dual round).
pub const Unit = struct {
    parent: *AnePrefill,
    /// 0 = no affinity hint; 1..N name a die (M3 Ultra's are 1 and 2).
    instance: c_int,
    /// Per-layer MLP procedure refs; null = not offloaded on this unit.
    layers: []?ProgramRef,
    /// Per-layer GDN input-projection refs (fused qkv+z).
    gdn_layers: []?ProgramRef,
    banks: std.ArrayList(*MsvAneMlp) = .empty,
    plane_in: ?*MsvAnePlane = null,
    /// False when this unit borrows unit 0's input plane
    /// (MLX_SERVE_ANE_DUAL_SHARE_INPUT=1).
    owns_plane_in: bool = true,
    plane_mlp_out: ?*MsvAnePlane = null,
    plane_gdn_out: ?*MsvAnePlane = null,

    pending: std.ArrayList(Pending) = .empty,
    pending_bytes: u64 = 0,
    /// A bank is homogeneous (one output surface), so MLP and GDN
    /// programs never share one — a kind switch flushes.
    pending_gdn: bool = false,
    dropped: usize = 0,

    thread: ?std.Thread = null,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    /// Protected by mu: .idle → .requested → .done.
    state: enum { idle, requested, done } = .idle,
    pending_bank: ?*MsvAneMlp = null,
    pending_proc: u32 = 0,
    eval_ok: bool = true,
    stop_flag: bool = false,
    eval_err: [512]u8 = @splat(0),
    /// Lifetime eval counts, read by /props from the server thread (the M3
    /// Ultra tester could not verify DISPATCH from /props — the one-shot
    /// engagement lines live in the log, but a props probe is what a bench
    /// harness reads, and under dual it is the ONLY in-process evidence
    /// that both dies were addressed). Written on the eval thread, hence
    /// atomics.
    evals_ok: std.atomic.Value(u64) = .init(0),
    evals_failed: std.atomic.Value(u64) = .init(0),

    fn allocator(self: *Unit) std.mem.Allocator {
        return self.parent.allocator;
    }

    /// Number of layers with a compiled MLP procedure on this unit.
    pub fn coveredLayers(self: *const Unit) usize {
        var n: usize = 0;
        for (self.layers) |h| {
            if (h != null) n += 1;
        }
        return n;
    }

    pub fn coveredGdnLayers(self: *const Unit) usize {
        var n: usize = 0;
        for (self.gdn_layers) |h| {
            if (h != null) n += 1;
        }
        return n;
    }

    pub fn inputBase(self: *Unit) ?[*]f16 {
        return msv_ane_plane_base(self.plane_in);
    }

    pub fn mlpOutputBase(self: *Unit) ?[*]f16 {
        return msv_ane_plane_base(self.plane_mlp_out);
    }

    pub fn gdnOutputBase(self: *Unit) ?[*]f16 {
        return msv_ane_plane_base(self.plane_gdn_out);
    }

    /// Hand a procedure to this unit's eval thread. One in-flight eval per
    /// unit; the caller must `wait()` before the next kick.
    pub fn kick(self: *Unit, ref: ProgramRef) void {
        self.mu.lockUncancelable(self.parent.io);
        std.debug.assert(self.state == .idle);
        self.pending_bank = ref.bank;
        self.pending_proc = ref.proc;
        self.state = .requested;
        self.cond.broadcast(self.parent.io);
        self.mu.unlock(self.parent.io);
    }

    /// Block until this unit's in-flight eval finishes. False = the eval
    /// failed (the caller recomputes on the GPU).
    pub fn wait(self: *Unit) bool {
        self.mu.lockUncancelable(self.parent.io);
        while (self.state != .done) self.cond.waitUncancelable(self.parent.io, &self.mu);
        const ok = self.eval_ok;
        self.state = .idle;
        self.mu.unlock(self.parent.io);
        return ok;
    }

    fn evalLoop(self: *Unit) void {
        while (true) {
            self.mu.lockUncancelable(self.parent.io);
            while (self.state != .requested and !self.stop_flag)
                self.cond.waitUncancelable(self.parent.io, &self.mu);
            if (self.stop_flag) {
                self.mu.unlock(self.parent.io);
                return;
            }
            const bank = self.pending_bank;
            const proc = self.pending_proc;
            self.mu.unlock(self.parent.io);

            const ok = msv_ane_mlp_eval(bank, proc, &self.eval_err, self.eval_err.len) != 0;
            if (ok) {
                _ = self.evals_ok.fetchAdd(1, .monotonic);
            } else {
                _ = self.evals_failed.fetchAdd(1, .monotonic);
                log.warn("[ane] unit {d} eval failed: {s}\n", .{ self.instance, std.mem.sliceTo(&self.eval_err, 0) });
            }

            self.mu.lockUncancelable(self.parent.io);
            self.eval_ok = ok;
            self.state = .done;
            self.cond.broadcast(self.parent.io);
            self.mu.unlock(self.parent.io);
        }
    }

    /// Queue one layer's MLP slice. Row-major [ffn, hidden] gate/up and
    /// [hidden, ffn] down f32 weights — already sliced to THIS unit's
    /// channel range by the caller.
    fn addMlp(self: *Unit, layer: usize, gate: []const f32, up: []const f32, down: []const f32) !void {
        const alloc = self.allocator();
        const hidden = self.parent.hidden;
        const ffn = self.parent.ffn;
        if (self.pending.items.len > 0 and self.pending_gdn) self.flushPending();
        var p = Pending{ .layer = @intCast(layer), .gdn = false, .ffn = ffn, .a = undefined, .b = undefined };
        p.a = try quantizeRowsInt8(alloc, gate, ffn, hidden);
        errdefer p.a.deinit(alloc);
        p.b = try quantizeRowsInt8(alloc, up, ffn, hidden);
        errdefer p.b.deinit(alloc);
        p.c = try quantizeRowsInt8(alloc, down, hidden, ffn);
        errdefer p.c.?.deinit(alloc);
        try self.enqueue(p);
    }

    /// Queue one layer's GDN input projections (fused qkv+z).
    fn addGdn(self: *Unit, layer: usize, qkv: []const f32, z: []const f32) !void {
        const alloc = self.allocator();
        const hidden = self.parent.hidden;
        const qkv_out = self.parent.gdn_qkv_out;
        const z_out = self.parent.gdn_z_out;
        if (qkv_out == 0 or z_out == 0) return error.AneGdnPlaneMismatch;
        if (self.pending.items.len > 0 and !self.pending_gdn) self.flushPending();
        var p = Pending{ .layer = @intCast(layer), .gdn = true, .qkv_out = qkv_out, .z_out = z_out, .a = undefined, .b = undefined };
        p.a = try quantizeRowsInt8(alloc, qkv, qkv_out, hidden);
        errdefer p.a.deinit(alloc);
        p.b = try quantizeRowsInt8(alloc, z, z_out, hidden);
        errdefer p.b.deinit(alloc);
        try self.enqueue(p);
    }

    /// Takes ownership of `p` ONLY on success — a failed append leaves the
    /// caller's errdefers to free the payloads (freeing here as well is the
    /// double-free).
    fn enqueue(self: *Unit, p: Pending) !void {
        // Bound the builder's transient host peak: the queue holds the
        // quantized payloads and the flush assembles a blob of the same
        // size beside them, so the cap governs both.
        const cap = bankMaxBytes();
        if (self.pending.items.len > 0 and self.pending_bytes + p.bytes() > cap) self.flushPending();
        try self.pending.append(self.allocator(), p);
        self.pending_gdn = p.gdn;
        self.pending_bytes += p.bytes();
    }

    /// Assemble one bank from `pending[start..start+n)`. Returns false when
    /// the compile or load was refused — the caller walks the split ladder.
    fn buildBank(self: *Unit, start: usize, n: usize) bool {
        const bank = msv_ane_bank_create() orelse return false;
        var err: [512]u8 = @splat(0);
        var ok = true;
        for (self.pending.items[start..][0..n]) |*p| {
            const rc = if (p.gdn)
                msv_ane_bank_add_gdn(bank, self.parent.hidden, p.qkv_out, p.z_out, self.parent.rows, p.a.q.ptr, p.a.s.ptr, p.b.q.ptr, p.b.s.ptr, &err, err.len)
            else
                msv_ane_bank_add_mlp(bank, self.parent.hidden, p.ffn, self.parent.rows, p.a.q.ptr, p.a.s.ptr, p.b.q.ptr, p.b.s.ptr, p.c.?.q.ptr, p.c.?.s.ptr, &err, err.len);
            if (rc < 0) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            log.warn("[ane] unit {d} bank assembly failed: {s}\n", .{ self.instance, std.mem.sliceTo(&err, 0) });
            msv_ane_bank_free(bank);
            return false;
        }
        const kind: []const u8 = if (self.pending.items[start].gdn) "gdn" else "mlp";
        var name_buf: [96]u8 = undefined;
        const name = std.fmt.bufPrintSentinel(&name_buf, "{s}_u{d}_b{d}_l{d}n{d}_r{d}", .{
            kind, self.instance, self.banks.items.len, self.pending.items[start].layer, n, self.parent.rows,
        }, 0) catch "ane_bank";
        const out_plane = if (self.pending.items[start].gdn) self.plane_gdn_out else self.plane_mlp_out;
        const compiled = msv_ane_bank_finish(bank, name.ptr, self.instance, self.plane_in, out_plane, &err, err.len) orelse {
            log.warn("[ane] unit {d} bank of {d} programs failed: {s}\n", .{ self.instance, n, std.mem.sliceTo(&err, 0) });
            return false;
        };
        self.banks.append(self.allocator(), compiled) catch {
            msv_ane_mlp_free(compiled);
            return false;
        };
        for (self.pending.items[start..][0..n], 0..) |*p, i| {
            const ref = ProgramRef{ .bank = compiled, .proc = @intCast(i) };
            if (p.gdn) self.gdn_layers[p.layer] = ref else self.layers[p.layer] = ref;
        }
        return true;
    }

    /// Compile everything queued, walking the split ladder: one bank for
    /// the whole group, then halves, then progressively smaller, and only a
    /// bank of ONE that still fails drops its layer to the GPU.
    fn flushPending(self: *Unit) void {
        defer {
            for (self.pending.items) |*p| p.deinit(self.allocator());
            self.pending.clearRetainingCapacity();
            self.pending_bytes = 0;
        }
        if (self.pending.items.len == 0) return;
        const alloc = self.allocator();
        const sizes = alloc.alloc(u64, self.pending.items.len) catch {
            log.warn("[ane] unit {d}: out of memory partitioning {d} queued programs — they stay on GPU\n", .{ self.instance, self.pending.items.len });
            self.dropped += self.pending.items.len;
            return;
        };
        defer alloc.free(sizes);
        for (self.pending.items, 0..) |*p, i| sizes[i] = p.bytes();

        const cap = bankMaxBytes();
        var start: usize = 0;
        var group: usize = self.pending.items.len;
        while (start < self.pending.items.len) {
            const n = bankGroupLen(sizes, start, group, cap);
            if (n == 0) break;
            if (self.buildBank(start, n)) {
                start += n;
                continue;
            }
            if (n == 1) {
                log.warn("[ane] unit {d} layer {d} program dropped — stays on GPU\n", .{ self.instance, self.pending.items[start].layer });
                self.dropped += 1;
                start += 1;
                continue;
            }
            group = n / 2;
            log.warn("[ane] unit {d} bank of {d} refused — retrying at {d} programs per bank\n", .{ self.instance, n, group });
        }
    }

    fn deinit(self: *Unit) void {
        if (self.thread) |t| {
            self.mu.lockUncancelable(self.parent.io);
            self.stop_flag = true;
            self.cond.broadcast(self.parent.io);
            self.mu.unlock(self.parent.io);
            t.join();
        }
        const alloc = self.allocator();
        for (self.pending.items) |*p| p.deinit(alloc);
        self.pending.deinit(alloc);
        for (self.banks.items) |b| msv_ane_mlp_free(b);
        self.banks.deinit(alloc);
        alloc.free(self.layers);
        alloc.free(self.gdn_layers);
        if (self.owns_plane_in) msv_ane_plane_free(self.plane_in);
        msv_ane_plane_free(self.plane_mlp_out);
        msv_ane_plane_free(self.plane_gdn_out);
    }
};

/// Per-model ANE prefill engine: one or more units (one Neural Engine
/// each), each holding compiled procedure banks at a FIXED row tile. Built
/// at model load (scheduler), owned by the Transformer, freed on unload.
pub const AnePrefill = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Constructed units. During init this is a growing prefix of the
    /// `units_cap`-long allocation so a failed plane alloc unwinds only
    /// what exists; afterwards it is the whole thing.
    units: []Unit,
    units_cap: usize,
    hidden: u32,
    /// PER-UNIT MLP output-channel slice width in channel mode; the full
    /// intermediate size in row mode.
    ffn: u32,
    /// PER-UNIT GDN projection output widths the compiled programs are
    /// built at (0 = no GDN offload). The seam reads these to slice the
    /// output plane.
    gdn_qkv_out: u32 = 0,
    gdn_z_out: u32 = 0,
    /// The fixed ANE row tile every compiled program expects (== chunk_rows
    /// in channel mode: every unit sees all chunk tokens there).
    rows: u32,
    /// The full chunk width the tile was derived from — the forward seam
    /// only engages on chunks of exactly this width.
    chunk_rows: u32,
    /// row = token-row split through full weights; channel = output-channel
    /// split through sliced weights (partial-sum down join).
    mode: Mode = .row,
    /// The TOTAL share the engine was built at and the int8 bytes it holds
    /// — set by the builder after the layer loop, read by /props and the
    /// metrics gauges. 0 until publishLive.
    share: f32 = 0,
    int8_bytes: u64 = 0,
    published: bool = false,
    engaged_logged: bool = false,
    gdn_engaged_logged: bool = false,

    /// `gdn_qkv_out`/`gdn_z_out` of 0 skips the GDN output plane (MLP-only
    /// engine); non-zero sizes it for the fused qkv+z programs. In channel
    /// mode `ffn`/`gdn_*_out` are the PER-UNIT sliced widths and
    /// rows == chunk_rows. `units` > 1 pins unit u to ANE instance u+1.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, num_layers: usize, hidden: u32, ffn: u32, rows: u32, chunk_rows: u32, gdn_qkv_out: u32, gdn_z_out: u32, mode: Mode, units: u32) !*AnePrefill {
        std.debug.assert(units >= 1 and units <= MAX_UNITS);
        const self = try allocator.create(AnePrefill);
        errdefer allocator.destroy(self);
        const unit_slice = try allocator.alloc(Unit, units);
        errdefer allocator.free(unit_slice);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .units = unit_slice[0..0],
            .units_cap = units,
            .hidden = hidden,
            .ffn = ffn,
            .gdn_qkv_out = gdn_qkv_out,
            .gdn_z_out = gdn_z_out,
            .rows = rows,
            .chunk_rows = chunk_rows,
            .mode = mode,
        };
        errdefer for (self.units) |*u| u.deinit();
        const io_bytes = @as(usize, hidden) * rows * 2;
        const share_input = units > 1 and dualShareInput();
        for (0..units) |u| {
            const layers = try allocator.alloc(?ProgramRef, num_layers);
            @memset(layers, null);
            const gdn_layers = allocator.alloc(?ProgramRef, num_layers) catch |e| {
                allocator.free(layers);
                return e;
            };
            @memset(gdn_layers, null);
            unit_slice[u] = .{
                .parent = self,
                .instance = if (units > 1) @intCast(u + 1) else 0,
                .layers = layers,
                .gdn_layers = gdn_layers,
            };
            // The unit is live in `units` before any fallible plane alloc,
            // so errdefer self.deinit() frees what it already owns.
            self.units = unit_slice[0 .. u + 1];
            if (share_input and u > 0) {
                unit_slice[u].plane_in = unit_slice[0].plane_in;
                unit_slice[u].owns_plane_in = false;
            } else {
                unit_slice[u].plane_in = msv_ane_plane_create(io_bytes) orelse return error.AnePlaneAlloc;
            }
            unit_slice[u].plane_mlp_out = msv_ane_plane_create(io_bytes) orelse return error.AnePlaneAlloc;
            if (gdn_qkv_out > 0 and gdn_z_out > 0)
                unit_slice[u].plane_gdn_out = msv_ane_plane_create(@as(usize, gdn_qkv_out + gdn_z_out) * rows * 2) orelse return error.AnePlaneAlloc;
            unit_slice[u].thread = try std.Thread.spawn(.{}, Unit.evalLoop, .{&unit_slice[u]});
        }
        if (self.units[0].plane_gdn_out == null) {
            self.gdn_qkv_out = 0;
            self.gdn_z_out = 0;
        }
        return self;
    }

    /// Record this engine's built totals into the live gauges (once).
    pub fn publishLive(self: *AnePrefill, share: f32, int8_bytes: u64) void {
        std.debug.assert(!self.published);
        self.share = share;
        self.int8_bytes = int8_bytes;
        self.published = true;
        _ = live_int8_bytes.fetchAdd(int8_bytes, .monotonic);
        _ = live_layers.fetchAdd(@intCast(self.coveredLayers() + self.coveredGdnLayers()), .monotonic);
    }

    pub fn deinit(self: *AnePrefill) void {
        if (self.published) {
            _ = live_int8_bytes.fetchSub(self.int8_bytes, .monotonic);
            _ = live_layers.fetchSub(@intCast(self.coveredLayers() + self.coveredGdnLayers()), .monotonic);
        }
        const allocator = self.allocator;
        const base = self.units.ptr;
        const cap = self.units_cap;
        for (self.units) |*u| u.deinit();
        allocator.free(base[0..cap]);
        allocator.destroy(self);
    }

    /// Layers whose MLP slice is covered on EVERY unit (a layer covered by
    /// only some units cannot be dispatched — the seam needs every partial).
    pub fn coveredLayers(self: *const AnePrefill) usize {
        var n: usize = 0;
        for (self.units[0].layers, 0..) |_, i| {
            if (self.mlpReady(i)) n += 1;
        }
        return n;
    }

    pub fn coveredGdnLayers(self: *const AnePrefill) usize {
        var n: usize = 0;
        for (self.units[0].gdn_layers, 0..) |_, i| {
            if (self.gdnReady(i)) n += 1;
        }
        return n;
    }

    pub fn numLayers(self: *const AnePrefill) usize {
        return self.units[0].layers.len;
    }

    pub fn mlpReady(self: *const AnePrefill, layer: usize) bool {
        if (layer >= self.units[0].layers.len) return false;
        for (self.units) |*u| {
            if (u.layers[layer] == null) return false;
        }
        return true;
    }

    pub fn gdnReady(self: *const AnePrefill, layer: usize) bool {
        if (layer >= self.units[0].gdn_layers.len) return false;
        for (self.units) |*u| {
            if (u.gdn_layers[layer] == null) return false;
        }
        return true;
    }

    /// Queue one unit's slice of a layer's MLP / GDN weights.
    pub fn addMlpLayer(self: *AnePrefill, unit: usize, layer: usize, gate: []const f32, up: []const f32, down: []const f32) !void {
        try self.units[unit].addMlp(layer, gate, up, down);
    }

    pub fn addGdnLayer(self: *AnePrefill, unit: usize, layer: usize, qkv: []const f32, z: []const f32) !void {
        try self.units[unit].addGdn(layer, qkv, z);
    }

    /// Compile every queued program. After this call each layer is either
    /// dispatchable on all units or null everywhere it failed.
    pub fn finishPending(self: *AnePrefill) void {
        for (self.units) |*u| u.flushPending();
    }

    /// Programs the ladder had to drop this build (each = one layer that
    /// stays on the GPU).
    pub fn droppedPrograms(self: *const AnePrefill) usize {
        var n: usize = 0;
        for (self.units) |*u| n += u.dropped;
        return n;
    }

    pub fn resetDropped(self: *AnePrefill) void {
        for (self.units) |*u| u.dropped = 0;
    }

    pub fn compiledBanks(self: *const AnePrefill) usize {
        var n: usize = 0;
        for (self.units) |*u| n += u.banks.items.len;
        return n;
    }

    /// Kick every unit's slice of a layer, then wait for all of them.
    /// Returns false when ANY unit's eval failed — the seam then recomputes
    /// the whole layer on the GPU (a subset of the partials is not an
    /// answer).
    pub fn kickMlp(self: *AnePrefill, layer: usize) void {
        for (self.units) |*u| u.kick(u.layers[layer].?);
    }

    pub fn kickGdn(self: *AnePrefill, layer: usize) void {
        for (self.units) |*u| u.kick(u.gdn_layers[layer].?);
    }

    pub fn waitAll(self: *AnePrefill) bool {
        var ok = true;
        for (self.units) |*u| {
            if (!u.wait()) ok = false;
        }
        return ok;
    }

    /// One-shot engagement lines, one PER SEAM — a built-but-never-
    /// dispatched program is exactly the dispatch-hole class, so each seam
    /// proves its own dispatch in the log (the expectNoSpec rule).
    pub fn logEngagedOnce(self: *AnePrefill) void {
        if (self.engaged_logged) return;
        self.engaged_logged = true;
        log.info("[ane] prefill offload engaged: mode={s} units={d} mlp={d} rows={d}/{d} (--ane-prefill; MLX_SERVE_ANE_SPLIT sets the share)\n", .{ @tagName(self.mode), self.units.len, self.coveredLayers(), self.rows, self.chunk_rows });
    }

    pub fn logGdnEngagedOnce(self: *AnePrefill) void {
        if (self.gdn_engaged_logged) return;
        self.gdn_engaged_logged = true;
        log.info("[ane] gdn offload engaged: mode={s} units={d} {d} layers, rows={d}/{d} (MLX_SERVE_ANE_GDN=0 restores MLP-only)\n", .{ @tagName(self.mode), self.units.len, self.coveredGdnLayers(), self.rows, self.chunk_rows });
    }

    /// The dual-ANE proof line: which instances the units were pinned to
    /// and how wide a slice each computes. A silently IGNORED affinity hint
    /// cannot be detected in-process — both units' evals succeed and both
    /// land on one die — so this line is the pointer to the out-of-process
    /// check (`macpow --dump | grep ANE0_` must move BOTH counters).
    pub fn logDualReady(self: *AnePrefill) void {
        if (self.units.len < 2) return;
        log.info("[ane] dual engaged: {d} units pinned to instances {d}..{d}, {d} channels each of mlp / {d}+{d} of gdn (verify BOTH IOReport ANE0_ counters move; MLX_SERVE_ANE_DUAL=0 restores single-ANE)\n", .{
            self.units.len,
            self.units[0].instance,
            self.units[self.units.len - 1].instance,
            self.ffn,
            self.gdn_qkv_out,
            self.gdn_z_out,
        });
    }
};

// ── Tests ──

const testing = std.testing;

test "defaultShare: per-silicon channel rows, row mode keeps its M4 optimum" {
    // M3 Ultra channel row (PR #223 tester): 0.45 measured ≈ nothing, 0.35
    // measured +8.5%/+13.7% at 16k/32k — the default must be the measured
    // optimum for the machine, never one machine's number everywhere.
    try std.testing.expectEqual(@as(f32, 0.50), defaultShare(.channel, "Apple M3 Ultra")); // dual by default
    try std.testing.expectEqual(@as(f32, 0.35), defaultShareFor(.channel, "Apple M3 Ultra", false));
    try std.testing.expect(dualEnabledFrom(null, "Apple M3 Ultra"));
    try std.testing.expect(!dualEnabledFrom("0", "Apple M3 Ultra"));
    try std.testing.expect(!dualEnabledFrom(null, "Apple M4 Max"));
    try std.testing.expect(dualEnabledFrom("1", "Apple M4 Max"));
    try std.testing.expectEqual(@as(f32, 0.45), defaultShare(.channel, "Apple M4 Max"));
    try std.testing.expectEqual(@as(f32, 0.45), defaultShare(.channel, "Apple M3 Max"));
    try std.testing.expectEqual(@as(f32, 0.45), defaultShare(.channel, ""));
    // Row mode is only measured on M4; every chip keeps that row.
    try std.testing.expectEqual(@as(f32, 0.40), defaultShare(.row, "Apple M3 Ultra"));
    try std.testing.expectEqual(@as(f32, 0.40), defaultShare(.row, "Apple M4 Max"));
}

test "anePrefillAllowed: NAX-class GPUs refuse the seam unless forced" {
    // No NAX (M1-M4): allowed regardless of the force env.
    try std.testing.expect(anePrefillAllowed(false, null));
    try std.testing.expect(anePrefillAllowed(false, "0"));
    // NAX present (M5+): refused — the GPU prefill measured faster (PR #223).
    try std.testing.expect(!anePrefillAllowed(true, null));
    try std.testing.expect(!anePrefillAllowed(true, "0"));
    try std.testing.expect(!anePrefillAllowed(true, ""));
    // MLX_SERVE_ANE_FORCE=1 keeps the build for future-silicon measurement.
    try std.testing.expect(anePrefillAllowed(true, "1"));
}

test "unitCount: dual is channel-mode on two-instance silicon, self-disabling elsewhere" {
    // Two ANE services (H11ANE/H11ANE1) exist on the Ultra parts only.
    try testing.expectEqual(@as(u32, 2), aneInstanceCount("Apple M3 Ultra"));
    try testing.expectEqual(@as(u32, 2), aneInstanceCount("Apple M2 Ultra"));
    try testing.expectEqual(@as(u32, 1), aneInstanceCount("Apple M4 Max"));
    try testing.expectEqual(@as(u32, 1), aneInstanceCount(""));
    // Off by default, whatever the machine.
    try testing.expectEqual(@as(u32, 1), unitCount(.channel, "Apple M3 Ultra", false));
    // On, and the machine has a second die: two units.
    try testing.expectEqual(@as(u32, 2), unitCount(.channel, "Apple M3 Ultra", true));
    // A dual request on a single-ANE machine must self-disable, never fail
    // the boot.
    try testing.expectEqual(@as(u32, 1), unitCount(.channel, "Apple M4 Max", true));
    // Row mode is not the default and is unmeasured under dual: refused.
    try testing.expectEqual(@as(u32, 1), unitCount(.row, "Apple M3 Ultra", true));
}

test "engineBillBytes: int8 weights + per-unit fp16 planes (A9 sharing within a unit)" {
    // Within a unit evals are strictly serial, so the planes are per SHAPE
    // CLASS, not per program: one input (hidden x rows), one MLP output
    // (hidden x rows), one GDN output ((qkv+z) x rows). Hand-computed small
    // case: dense 2 layers (3 weights of [ffn=8, hidden=4] -> 192 int8) +
    // one GDN layer ((6+2)*4 = 32 int8); planes 128 (in) + 128 (mlp out) +
    // 256 (gdn out).
    try testing.expectEqual(@as(u64, 736), engineBillBytes(2, 1, 4, 8, 6, 2, 16, 1));
    // The 27B at full coverage, rows 3264: int8 ~19.7 GiB + ~166 MB planes
    // (the per-program planes billed ~10.3 GiB here before A9).
    try testing.expectEqual(
        @as(u64, 21_313_093_632),
        engineBillBytes(64, 48, 5120, 17408, 10240, 6144, 3264, 1),
    );
    // No GDN coverage bills neither GDN int8 nor a GDN output plane.
    try testing.expectEqual(@as(u64, 448), engineBillBytes(2, 0, 4, 8, 6, 2, 16, 1));
    // No dense coverage bills no MLP output plane.
    try testing.expectEqual(@as(u64, 32 + 128 + 256), engineBillBytes(0, 1, 4, 8, 6, 2, 16, 1));
    // Dual: the widths passed are PER UNIT, so two units at half the slice
    // hold the SAME int8 as one unit at the full slice — only the planes
    // double (concurrent evals cannot share an output surface).
    const single = engineBillBytes(2, 1, 4, 8, 6, 2, 16, 1);
    const dual = engineBillBytes(2, 1, 4, 4, 4, 2, 16, 2);
    try testing.expectEqual(@as(u64, 2 * (2 * 3 * 4 * 4 + 1 * 6 * 4 + 128 + 128 + 192)), dual);
    try testing.expect(dual > single); // the extra planes, nothing else
}

test "gateAllows: per-model bill vs total RAM, unknown total allows" {
    const gib = 1024 * 1024 * 1024;
    const hr = 12 * gib;
    // Unknown total (sysctl failure) is no information — allow, the old
    // `total_mem > 0` behavior.
    try testing.expect(gateAllows(0, 16 * gib, 32 * gib, hr));
    // 64 GB Mac, 16 GB resident, ~32 GB bill: 16+32+12 headroom = 60 <= 64.
    try testing.expect(gateAllows(64 * gib, 16 * gib, 32 * gib, hr));
    // 36 GB Mac, same model: refused.
    try testing.expect(!gateAllows(36 * gib, 16 * gib, 32 * gib, hr));
    // A small model on a small Mac passes (the flat 96 GB gate refused it).
    try testing.expect(gateAllows(16 * gib, 1 * gib, 1 * gib, hr));
    // Exact fit allows.
    try testing.expect(gateAllows(60 * gib, 16 * gib, 32 * gib, hr));

    // The headroom is a PARAMETER, which is the fix: the measured M1 Pro case
    // (32 GB, 12.5 GB resident, 10.3 GB bill) is refused under the flat 12 GB
    // the constant used to be, and admitted under the ~7 GB this model needs
    // at its chunk-1024 envelope — an arm measured at +38% prefill.
    const resident = 12_500 * 1024 * 1024;
    const bill = 10_300 * 1024 * 1024;
    try testing.expect(!gateAllows(32 * gib, resident, bill, 12 * gib));
    try testing.expect(gateAllows(32 * gib, resident, bill, 7 * gib));

    // Saturating: an absurd headroom refuses rather than wrapping to allow.
    try testing.expect(!gateAllows(32 * gib, resident, bill, std.math.maxInt(u64)));
}

test "aneShareRows: 32-row floor, GPU remainder, engagement minimum" {
    // 32-row quantum: an fp16 plane's per-channel pitch is rows x 2 bytes,
    // and a pitch off the 64-byte grid fails EVERY eval of the compiled
    // program (the v2 share-0.35 collapse: rows 2864 = 16 mod 32).
    try testing.expectEqual(@as(u32, 2432), aneShareRows(8192, 0.30));
    try testing.expectEqual(@as(u32, 2848), aneShareRows(8192, 0.35)); // was 2864, the live cliff
    try testing.expectEqual(@as(u32, 3264), aneShareRows(8192, 0.40)); // default share unchanged
    try testing.expectEqual(@as(u32, 1216), aneShareRows(4096, 0.30));
    try testing.expectEqual(@as(u32, 0), aneShareRows(8192, 0.0)); // no share
    try testing.expectEqual(@as(u32, 0), aneShareRows(8192, -1.0));
    try testing.expectEqual(@as(u32, 0), aneShareRows(512, 0.30)); // 144 < ANE_MIN_ROWS
    try testing.expectEqual(@as(u32, 0), aneShareRows(64, 0.9)); // chunk too small
    // Oversized share clamps so the GPU keeps >= 16 rows.
    const clamped = aneShareRows(8192, 1.5);
    try testing.expect(clamped <= 8192 - 16 and clamped % 32 == 0 and clamped > 0);
}

test "channelSliceWidth: 128-aligned slice, GPU remainder, degenerate shares" {
    // The 27B geometries at the default share (the spike's probed widths).
    try testing.expectEqual(@as(u32, 6912), channelSliceWidth(17408, 0.40));
    try testing.expectEqual(@as(u32, 4096), channelSliceWidth(10240, 0.40));
    try testing.expectEqual(@as(u32, 2432), channelSliceWidth(6144, 0.40));
    // Degenerate: tiny width, zero/negative share, NaN-safe.
    try testing.expectEqual(@as(u32, 0), channelSliceWidth(128, 0.5));
    try testing.expectEqual(@as(u32, 0), channelSliceWidth(17408, 0.0));
    try testing.expectEqual(@as(u32, 0), channelSliceWidth(17408, -1.0));
    // Oversized share clamps so the GPU keeps >= 128 channels.
    const clamped = channelSliceWidth(17408, 1.5);
    try testing.expect(clamped <= 17408 - 128 and clamped % 128 == 0 and clamped > 0);
    // Every 128-aligned width has a power-of-two K-chunk divisor <= 16
    // landing under the ANE down-conv cliff (mirrors down_chunks_for).
    var k: u32 = 128;
    while (k <= 17408) : (k += 128) {
        var ok = false;
        var n: u32 = 1;
        while (n <= 16) : (n += 1) {
            if (k % n == 0 and k / n <= 4608) ok = true;
        }
        try testing.expect(ok);
    }
}

test "channelSliceWidthUnits: the share is TOTAL, halved across units, GPU keeps a slice" {
    // MLX_SERVE_ANE_SPLIT keeps meaning the total fraction taken off the
    // GPU, so every single-ANE measurement carries over: two units at 0.40
    // take the same 40% of channels one unit at 0.40 does.
    const k1 = channelSliceWidthUnits(17408, 0.40, 1);
    const k2 = channelSliceWidthUnits(17408, 0.40, 2);
    try testing.expectEqual(@as(u32, 6912), k1);
    try testing.expectEqual(@as(u32, 3456), k2);
    try testing.expectEqual(k1, 2 * k2);
    // Every unit's boundary stays 128-aligned and the GPU keeps at least
    // CHANNEL_ALIGN channels after ALL units.
    for ([_]u32{ 17408, 10240, 6144, 4096, 1024 }) |w| {
        for ([_]f32{ 0.2, 0.35, 0.5, 0.9, 1.5 }) |s| {
            const k = channelSliceWidthUnits(w, s, 2);
            if (k == 0) continue;
            try testing.expect(k % CHANNEL_ALIGN == 0);
            try testing.expect(2 * k <= w - CHANNEL_ALIGN);
        }
    }
    // Degenerate: a width that cannot seat two slices plus the GPU's.
    try testing.expectEqual(@as(u32, 0), channelSliceWidthUnits(256, 0.5, 2));
    try testing.expectEqual(@as(u32, 0), channelSliceWidthUnits(17408, 0.0, 2));
    try testing.expectEqual(@as(u32, 0), channelSliceWidthUnits(17408, 0.4, 0));
}

test "bankGroupLen: monolithic under the cap, the ladder's rungs, never zero programs" {
    const sizes = [_]u64{ 100, 100, 100, 100, 100 };
    // Under the cap the whole group banks monolithically (oMLX measured
    // that bit-stable across five greedy runs; split banks occasionally
    // diverged at a tie).
    try testing.expectEqual(@as(usize, 5), bankGroupLen(&sizes, 0, 5, 1000));
    // The cap partitions: 250 seats two.
    try testing.expectEqual(@as(usize, 2), bankGroupLen(&sizes, 0, 5, 250));
    try testing.expectEqual(@as(usize, 2), bankGroupLen(&sizes, 2, 5, 250));
    try testing.expectEqual(@as(usize, 1), bankGroupLen(&sizes, 4, 5, 250));
    // The ladder's rungs cap the count regardless of bytes.
    try testing.expectEqual(@as(usize, 2), bankGroupLen(&sizes, 0, 2, 1000));
    try testing.expectEqual(@as(usize, 1), bankGroupLen(&sizes, 0, 1, 1000));
    // A single program over the cap still gets its own bank — refusing it
    // would only move the failure, and the ladder's last rung IS one.
    try testing.expectEqual(@as(usize, 1), bankGroupLen(&sizes, 0, 5, 1));
    // Past the end / no rung left.
    try testing.expectEqual(@as(usize, 0), bankGroupLen(&sizes, 5, 5, 1000));
    try testing.expectEqual(@as(usize, 0), bankGroupLen(&sizes, 0, 0, 1000));
    // The ladder walk covers every program exactly once at any rung.
    for ([_]usize{ 5, 2, 1 }) |group| {
        var start: usize = 0;
        var seen: usize = 0;
        while (start < sizes.len) {
            const n = bankGroupLen(&sizes, start, group, 10_000);
            try testing.expect(n > 0);
            seen += n;
            start += n;
        }
        try testing.expectEqual(sizes.len, seen);
    }
}

test "quantizeRowsInt8: round-trip, per-row scales, zero row" {
    const w = [_]f32{
        1.0,  -2.0, 0.5,  0.25, // row 0: amax 2
        0.0,  0.0,  0.0,  0.0, // row 1: all zero
        -0.1, 0.05, 0.02, 0.1, // row 2: amax 0.1
    };
    var rq = try quantizeRowsInt8(testing.allocator, &w, 3, 4);
    defer rq.deinit(testing.allocator);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 127.0), rq.s[0], 1e-7);
    try testing.expectEqual(@as(f32, 0), rq.s[1]);
    try testing.expectApproxEqAbs(@as(f32, 0.1 / 127.0), rq.s[2], 1e-7);
    // Extremes hit exactly ±127; zero row stays zero codes.
    try testing.expectEqual(@as(i8, -127), rq.q[1]);
    try testing.expectEqual(@as(i8, 0), rq.q[4]);
    try testing.expectEqual(@as(i8, 127), rq.q[11]);
    // Dequantized max error <= half a step per row.
    for (0..3) |i| {
        for (0..4) |j| {
            const deq = @as(f32, @floatFromInt(rq.q[i * 4 + j])) * rq.s[i];
            try testing.expect(@abs(deq - w[i * 4 + j]) <= rq.s[i] * 0.5 + 1e-9);
        }
    }
}
