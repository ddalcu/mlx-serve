//! Measured spec round-cost table: what a round at draft width w costs on
//! THIS model, on THIS machine, at the live context length.
//!
//! Every width decision in spec decode is throughput = accepted tokens over
//! round wall time, and every earlier cost source (the hand-typed chip rows,
//! the fitted EV surfaces, the boot ladder) measured only part of that, in
//! one regime, and shipped it for all of them. This table measures the
//! whole fraction from the rounds the server actually runs. Pure data, no
//! MLX — the generator feeds it and reads it, the fitted surface is the
//! cold-start prior.
//!
//! Width = drafts per round (MTP depth m; DFlash block_size - 1; 0 = serial).
//! Buckets = KV length at the round: <2k, 2-4k, 4-8k, 8-16k, 16-32k, 32k+.
//! Each cell holds an EMA of round ms AND an EMA of emitted tokens — cost is
//! never stored without the tokens it bought.
//!
//! Beside the width grid sits ONE more row: `serial`, the measured ms of a
//! PLAIN decode token per bucket. It is deliberately not width 0. A width-0
//! round is a block decoder's serial fallback measured on ITS round clock,
//! and the width grid interpolates, extrapolates and anchors through its
//! cells — a serial tick has no head forward, no capture and no verify, so
//! letting it into the grid would re-anchor `MtpCostSource` and let
//! `roundMs` lerp a depth-1 round down toward a quantity that is not a
//! round at all. Kept apart, it answers exactly one question no width can:
//! is the whole speculation worth running here?
const std = @import("std");

/// Drafts per round the table covers (MTP depth <= 8, a DFlash block up to 16); index 0 is serial.
pub const MAX_WIDTH: u32 = 16;
/// KV buckets. The top bucket used to be an unbounded `32k+`, which put 62.7k
/// and 374k in ONE cell: serial ms/token roughly doubles across that span, the
/// EMA blended two regimes, and the adaptive switch's "one re-decision per
/// bucket crossing" could never fire inside it — a request that left
/// speculation at 40k stayed serial to 1M. The long-context rungs get their
/// own edges; below 32k the layout is unchanged.
pub const N_BUCKETS: usize = 9;
const BUCKET_EDGES = [_]u32{ 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 };
pub const BUCKET_NAMES = [N_BUCKETS][]const u8{
    "<2k",     "2-4k",    "4-8k",     "8-16k", "16-32k",
    "32-64k",  "64-128k", "128-256k", "256k+",
};

/// The label for bucket `b` UNDER `layout`. A name is a claim about the cell's
/// EDGES, and the two grids disagree about exactly one cell: on `.long` bucket
/// 5 is 32k-64k, on `.legacy` it is the unbounded top bucket `32k+`
/// (`nBuckets(.legacy) == 6`). `BUCKET_NAMES` alone spells the long grid, so
/// every `[spec-stats]`, `[mtp-trace]` and adaptive-switch line on a
/// sidecar-MTP pack (qwen3.5/3.6/3.8 — every arch but qwen4_exp) reported a
/// 374k request as sitting in "32-64k". The number is right and the label is a
/// lie, which is worse than no label: it is the string an operator reads when
/// deciding whether a bucket crossing should have re-planned.
///
/// Names only — no cell, edge or plan moves. `bucketFor` already stops at the
/// legacy top bucket.
pub fn bucketName(layout: Layout, b: usize) []const u8 {
    const n = nBuckets(layout);
    if (b + 1 == n and n < N_BUCKETS) return "32k+";
    return BUCKET_NAMES[b];
}

/// Which bucket grid a table speaks. The split above, the serial row and the
/// adaptive switch that reads it were all MEASURED on qwen4_exp; every other
/// arch keeps the six-bucket grid every release through 26.9.1 wrote, so a
/// sidecar-MTP pack (qwen3.5/3.6/3.8) reads its OWN persisted table and plans
/// from exactly the cells it planned from before.
///
/// This is not decoration. `MtpCostSource.fromTable()` is the single term
/// that decides whether the EV plan prices extension from measurements or
/// from the fitted prior, and the prior's extension valve is deliberately
/// always-open (it exists to observe `a[m_lo]`, which no other round can).
/// Bumping the store version therefore does not merely cost one re-explore:
/// until the new file matures, EVERY round is a two-chunk round. On the 27B
/// at 8k that measured 37 of 40 extension rounds at 77 ms against a warm
/// table's 1 of 52 at 51 ms — a 25% decode loss with acceptance unchanged.
/// A layout that keeps its old version keeps its old file, and with it the
/// old plan.
pub const Layout = enum {
    /// Six buckets, top one unbounded at 32k. Store version 1, no serial row.
    legacy,
    /// Nine buckets (64k/128k/256k edges) + the serial row. Store version 3.
    long,
};

/// THE resolver: which bucket grid this model's table speaks. Every
/// `layout =` assignment in the tree goes through here (scan-pinned in
/// scheduler.zig), because the layout decides the STORE VERSION and therefore
/// which persisted file a model reads — an arch that resolves `.long` on the
/// serve path and `.legacy` on the offline `--prompt` path plans MTP from two
/// different grids for the same checkpoint (audit addendum 3).
///
/// `anytype` rather than a `*const ModelConfig`: this module deliberately
/// imports nothing but std, and the arch question already has exactly one
/// answer elsewhere (`ModelConfig.isQwen4`) that must not be respelled here.
pub fn layoutFor(config: anytype) Layout {
    return if (config.isQwen4()) .long else .legacy;
}

/// Buckets the layout actually uses. Cells past it are never written and
/// never active, so every scan over `N_BUCKETS` reads the same as a scan
/// over this — the array dimension stays the long layout's.
pub fn nBuckets(layout: Layout) usize {
    return switch (layout) {
        .legacy => 6,
        .long => N_BUCKETS,
    };
}

/// EMA weight. MIN is wrong here (thermal soak makes the early rounds the
/// fast ones); an EMA tracks the live machine.
pub const BETA: f32 = 0.10;
/// A cell unsampled for this many offered rounds takes its next sample at
/// RESEED_WEIGHT instead of BETA: whatever it held was measured in another
/// regime (thermal, context). It keeps its sample count — a cell that is
/// only ever re-sampled by trials (period up to EXPLORE_PERIOD_MAX) would
/// otherwise lose trust on every trial, reopen the horizon, and cycle.
pub const RESEED_GAP: u32 = 64;
pub const RESEED_WEIGHT: f32 = 0.5;
/// Measured widths a bucket needs before the table replaces the prior: ONE
/// trusted width anchors the prior's shape (scale) and lets raw cells floor
/// it — two were required first, and with w5 settled as clearly worse
/// after one sample the bucket never reached two, so the prior kept
/// planning the 4 -> 5 extension the table had already priced.
pub const MIN_WIDTHS: u32 = 1;
/// Samples a cell needs before it COUNTS as measured. A one-sample cell is
/// the seed (a legacy-controller round at the warmup boundary seeded a w2
/// cell on the M4 base 9B, activated the table on {w2, w4} and anchored it
/// at 2 — the plan then read w3 as a bargain and lost 6.6%).
pub const MIN_SAMPLES: u32 = 3;
/// Serial probes a bucket may attempt per process before it gives up. An
/// attempt is bounded (`MTP_ADAPTIVE_PROBE_TOKENS` serial ticks), so this
/// caps the whole cost of teaching one bucket.
pub const MAX_SERIAL_PROBES: u8 = 3;
/// A width whose FIRST sample already reads this much worse per token than
/// a trusted reference is settled as worse: the plan only needs "not
/// better", and every further trial block of it is a 3-4% hit on the
/// request that carries it (M1 Pro 27B: w5 read 94.7 against w4's 71.2 on
/// sample 1 and never moved; at 3-samples-to-trust that cost -7.1%).
pub const CLEARLY_WORSE: f32 = 0.20;

/// The bucket `kv_len` falls in, under `layout`. The legacy grid stops at
/// the 32768 edge and folds everything above into its last bucket — bit for
/// bit what `bucketFor` did through 26.9.1.
pub fn bucketForLayout(kv_len: u32, layout: Layout) usize {
    const edges = nBuckets(layout) - 1;
    for (BUCKET_EDGES[0..edges], 0..) |edge, i| {
        if (kv_len < edge) return i;
    }
    return edges;
}

/// The long layout's grid. Kept as the free function because that is what
/// the qwen4-only call sites and the tests want; anything reading a TABLE
/// must go through `Table.bucketOf`, which honours the table's own layout.
pub fn bucketFor(kv_len: u32) usize {
    return bucketForLayout(kv_len, .long);
}

pub const Cell = struct {
    ms: f32 = 0,
    tok: f32 = 0,
    n: u32 = 0,
    /// `Table.seq` at the last fold — the reseed clock.
    last_seen: u32 = 0,

    pub fn msPerTok(self: Cell) ?f32 {
        if (self.n == 0 or self.tok <= 0) return null;
        return self.ms / self.tok;
    }
};

pub const Verdict = enum { folded, reseeded, contended, transition, bad_sample, out_of_range };

pub const Table = struct {
    /// Which bucket grid (and which store version) this table speaks. Set
    /// once at load from the ARCH, never from a request. The default is the
    /// legacy grid: a table that nobody told is a table on the arch the
    /// split was not measured on.
    layout: Layout = .legacy,
    cells: [MAX_WIDTH + 1][N_BUCKETS]Cell = @splat(@splat(.{})),
    /// Measured ms per PLAIN serial decode token, per bucket (`tok` is 1 per
    /// sample, so `msPerTok` and `ms` agree). Fed by solo serial ticks — a
    /// request that never armed MTP, one whose speculation turned itself
    /// off, and the bounded serial probe.
    serial: [N_BUCKETS]Cell = @splat(.{}),
    /// Bounded serial probes ATTEMPTED per bucket this process. A bucket
    /// whose serial cell is untrusted cannot be decided at all, and without a
    /// probe a workload that never decodes serially there would never learn
    /// one.
    ///
    /// A count, not a flag: the flag was set at ARMING, so a probe that was
    /// interrupted before folding `MIN_SAMPLES` — the request ended, the
    /// server got busy and the ticks were dropped as contended, a stop
    /// sequence landed — burned the bucket's only chance and left it
    /// permanently undecidable. Up to `MAX_SERIAL_PROBES` attempts, which
    /// bounds the cost at that many short serial blocks per bucket per
    /// process while letting a broken attempt be retried.
    ///
    /// Runtime only — never serialized, so a restored table re-probes.
    serial_probes: [N_BUCKETS]u8 = @splat(0),
    /// Samples OFFERED (accepted or not): the width grid's reseed clock.
    seq: u32 = 0,
    /// The serial row's OWN reseed clock. It cannot share `seq`: a serial
    /// tick is a token, a width sample is a round, and a mixed workload runs
    /// thousands of the former — sharing would take every width cell past
    /// RESEED_GAP within a second of serial decoding and leave the width
    /// planner permanently blending at RESEED_WEIGHT.
    serial_seq: u32 = 0,
    folded: u32 = 0,
    dropped_transition: u32 = 0,
    dropped_contended: u32 = 0,
    dropped_bad: u32 = 0,
    /// The serial row's OWN fold/drop counters. It cannot share the width
    /// grid's: a width sample is a ROUND and a serial sample is a TOKEN, and
    /// a mixed workload runs thousands of the latter per hundred of the
    /// former. Sharing made `[spec-stats] table_drops` unreadable (two clocks
    /// summed into one number) and made the persistence trigger fire on plain
    /// decode traffic that had taught the width grid nothing.
    serial_folded: u32 = 0,
    serial_dropped_transition: u32 = 0,
    serial_dropped_contended: u32 = 0,
    serial_dropped_bad: u32 = 0,
    /// One-shot: the first plan that read the table instead of the prior.
    first_use_logged: bool = false,
    /// `folded` at the last store (persistence writes only when it moved).
    stored_at: u32 = 0,
    /// WIDTH cells restored from disk at load (diagnostics). The serial row
    /// is counted separately and deliberately: "how many cells does the
    /// width planner have" is a different question from "does this bucket
    /// know what a plain token costs", and one number cannot answer both.
    restored: u32 = 0,
    /// Serial cells restored from disk at load (diagnostics).
    restored_serial: u32 = 0,

    /// Folded WIDTH cells. Never counts the serial row — every caller of
    /// this asks about the width grid.
    pub fn foldedCells(self: *const Table) u32 {
        var n: u32 = 0;
        for (self.cells) |row| {
            for (row) |c| {
                if (c.n > 0) n += 1;
            }
        }
        return n;
    }

    /// Folded serial cells (buckets that have measured a plain token).
    pub fn foldedSerialCells(self: *const Table) u32 {
        var n: u32 = 0;
        for (self.serial) |c| {
            if (c.n > 0) n += 1;
        }
        return n;
    }

    /// Feed one realized round. `solo` = this was the only decoding stream
    /// (contention only ever ADDS time, so a busy server stops teaching the
    /// table rather than teaching it a lie); `transition` = the width
    /// differs from the previous round (the width change is a one-off cost
    /// that read the minority shape 5-7% slow in Phase 1).
    pub fn observe(self: *Table, width: u32, kv_len: u32, ms: f32, tokens: f32, solo: bool, transition: bool) Verdict {
        self.seq +%= 1;
        if (width > MAX_WIDTH) return .out_of_range;
        if (!std.math.isFinite(ms) or ms <= 0 or !(tokens > 0)) {
            self.dropped_bad += 1;
            return .bad_sample;
        }
        if (!solo) {
            self.dropped_contended += 1;
            return .contended;
        }
        if (transition) {
            self.dropped_transition += 1;
            return .transition;
        }
        self.folded += 1;
        return foldInto(&self.cells[width][self.bucketOf(kv_len)], ms, tokens, self.seq);
    }

    /// Feed one realized PLAIN serial decode token into `serial[bucket]`.
    /// Same drop rules as `observe` — the caller passes `solo` from the same
    /// scheduler flag and `transition` for the ticks that follow a
    /// speculative round (the GPU still holds its tail, so the first ticks
    /// of a serial block read slow).
    pub fn observeSerial(self: *Table, kv_len: u32, ms: f32, solo: bool, transition: bool) Verdict {
        self.serial_seq +%= 1;
        if (!std.math.isFinite(ms) or ms <= 0) {
            self.serial_dropped_bad += 1;
            return .bad_sample;
        }
        if (!solo) {
            self.serial_dropped_contended += 1;
            return .contended;
        }
        if (transition) {
            self.serial_dropped_transition += 1;
            return .transition;
        }
        self.serial_folded += 1;
        return foldInto(&self.serial[self.bucketOf(kv_len)], ms, 1.0, self.serial_seq);
    }

    fn foldInto(cell: *Cell, ms: f32, tokens: f32, clock: u32) Verdict {
        defer cell.last_seen = clock;
        if (cell.n == 0) {
            cell.ms = ms;
            cell.tok = tokens;
            cell.n = 1;
            return .reseeded;
        }
        // The first MIN_SAMPLES are a running MEAN (an EMA seeded from
        // sample 1 is still sample 1 at n=3); the EMA takes over after.
        const stale = clock -% cell.last_seen > RESEED_GAP;
        const beta: f32 = if (stale) RESEED_WEIGHT else if (cell.n < MIN_SAMPLES) 1.0 / @as(f32, @floatFromInt(cell.n + 1)) else BETA;
        cell.ms += beta * (ms - cell.ms);
        cell.tok += beta * (tokens - cell.tok);
        cell.n += 1;
        return if (stale) .reseeded else .folded;
    }

    /// Measured ms of one plain serial token in `bucket` (MIN_SAMPLES
    /// folded), or null. Never interpolated across buckets and never
    /// extrapolated: an unmeasured bucket is a bucket with no answer.
    pub fn serialMsPerTok(self: *const Table, bucket: usize) ?f32 {
        if (bucket >= N_BUCKETS or self.serial[bucket].n < MIN_SAMPLES) return null;
        return self.serial[bucket].msPerTok();
    }

    /// THE bucket resolver for anything holding a table: the layout is the
    /// table's, never the caller's guess. A free `bucketFor` beside a table
    /// with the legacy grid is how a 40k request came to read a bucket the
    /// table never writes.
    pub fn bucketOf(self: *const Table, kv_len: u32) usize {
        return bucketForLayout(kv_len, self.layout);
    }

    fn trusted(self: *const Table, width: u32, bucket: usize) bool {
        return width <= MAX_WIDTH and self.cells[width][bucket].n >= MIN_SAMPLES;
    }

    /// Measured round ms at exactly this width (MIN_SAMPLES folded), or null.
    pub fn measuredMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        return if (self.trusted(width, bucket)) self.cells[width][bucket].ms else null;
    }

    /// Measured tokens per round at exactly this width, or null.
    pub fn measuredTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        return if (self.trusted(width, bucket)) self.cells[width][bucket].tok else null;
    }

    pub fn msPerTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        return if (self.trusted(width, bucket)) self.cells[width][bucket].msPerTok() else null;
    }

    /// Round ms / ms per token from ANY folded cell (n >= 1): evidence for
    /// "worse", never for "better".
    pub fn rawMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (width > MAX_WIDTH or self.cells[width][bucket].n == 0) return null;
        return self.cells[width][bucket].ms;
    }

    pub fn rawMsPerTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (width > MAX_WIDTH) return null;
        return self.cells[width][bucket].msPerTok();
    }

    /// `width` has at least one sample and reads CLEARLY_WORSE per token
    /// than trusted `ref`.
    pub fn clearlyWorse(self: *const Table, width: u32, ref: u32, bucket: usize) bool {
        const w = self.rawMsPerTok(width, bucket) orelse return false;
        const r = self.msPerTok(ref, bucket) orelse return false;
        return w >= r * (1.0 + CLEARLY_WORSE);
    }

    pub fn measuredCount(self: *const Table, bucket: usize) u32 {
        var n: u32 = 0;
        for (0..MAX_WIDTH + 1) |w| {
            if (self.trusted(@intCast(w), bucket)) n += 1;
        }
        return n;
    }

    pub fn active(self: *const Table, bucket: usize) bool {
        return self.measuredCount(bucket) >= MIN_WIDTHS;
    }

    /// The bucket a plan at `kv_len` reads: its own when active, else the
    /// nearest active one (lower side preferred — cost grows with KV, so a
    /// lower bucket under-bills rather than over-bills). Null = no active
    /// bucket, the prior applies. A bucket boundary crossed mid-generation
    /// must not snap the plan back to the prior.
    pub fn bucketToRead(self: *const Table, kv_len: u32) ?usize {
        const own = self.bucketOf(kv_len);
        if (self.active(own)) return own;
        var d: usize = 1;
        while (d < N_BUCKETS) : (d += 1) {
            if (own >= d and self.active(own - d)) return own - d;
            if (own + d < N_BUCKETS and self.active(own + d)) return own + d;
        }
        return null;
    }

    /// Narrowest measured width in the bucket (the normalization anchor).
    pub fn narrowestMeasured(self: *const Table, bucket: usize) ?u32 {
        for (0..MAX_WIDTH + 1) |w| {
            if (self.trusted(@intCast(w), bucket)) return @intCast(w);
        }
        return null;
    }

    /// Round ms at `width`: measured, else linear between the two nearest
    /// measured widths. OUTSIDE the measured span the answer is null and
    /// the caller's prior fills in — below, because the prior's extended
    /// rounds land on the cliff first and the cliff's slope run downward
    /// reads every narrower width as free; above, because a shallow slope
    /// run upward (w3 -> w4 +6 ms) priced widths 5..8 as nearly free and the
    /// plan raced there in consecutive transition rounds, measuring nothing
    /// (the caller takes max(last slope, prior marginal) per extra width).
    /// Null while the bucket has fewer than MIN_WIDTHS measured widths.
    pub fn roundMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (!self.active(bucket)) return null;
        if (self.measuredMs(width, bucket)) |m| return m;
        var lo: ?u32 = null;
        var hi: ?u32 = null;
        for (0..MAX_WIDTH + 1) |wi| {
            const w: u32 = @intCast(wi);
            if (!self.trusted(w, bucket)) continue;
            if (w < width) {
                lo = w;
            } else if (hi == null) {
                hi = w;
            }
        }
        if (lo != null and hi != null) {
            return lerp(lo.?, self.cells[lo.?][bucket].ms, hi.?, self.cells[hi.?][bucket].ms, width);
        }
        return null;
    }

    pub fn widestMeasured(self: *const Table, bucket: usize) ?u32 {
        var i: usize = MAX_WIDTH + 1;
        while (i > 0) {
            i -= 1;
            if (self.trusted(@intCast(i), bucket)) return @intCast(i);
        }
        return null;
    }

    /// ms per width between the two widest measured widths (a cliff's slope
    /// when one was measured), null with fewer than two.
    pub fn lastSlope(self: *const Table, bucket: usize) ?f32 {
        const hi = self.widestMeasured(bucket) orelse return null;
        var lo: ?u32 = null;
        for (0..hi) |wi| {
            if (self.trusted(@intCast(wi), bucket)) lo = @intCast(wi);
        }
        const l = lo orelse return null;
        return (self.cells[hi][bucket].ms - self.cells[l][bucket].ms) / @as(f32, @floatFromInt(hi - l));
    }

    fn lerp(w0: u32, m0: f32, w1: u32, m1: f32, w: u32) f32 {
        const t = (@as(f32, @floatFromInt(w)) - @as(f32, @floatFromInt(w0))) / (@as(f32, @floatFromInt(w1)) - @as(f32, @floatFromInt(w0)));
        return m0 + t * (m1 - m0);
    }

    /// `w3:12.10/5,w4:11.80/2` — ms per emitted token per width in the
    /// bucket, with the sample count (a cell under MIN_SAMPLES is shown but
    /// does not count), for `[spec-stats]`. Empty when nothing was folded.
    pub fn formatBucket(self: *const Table, bucket: usize, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (0..MAX_WIDTH + 1) |wi| {
            const c = self.cells[wi][bucket];
            const mpt = c.msPerTok() orelse continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("w{d}:{d:.2}/{d}", .{ wi, mpt, c.n }) catch break;
        }
        return w.buffered();
    }
};

// ── Trial schedule + per-round width chooser ─────────────────────────────

/// Rounds between trials while the gap is unknown, the drag the period is
/// sized to once it is, the cap, and the block length. A block is THREE
/// rounds: the transition, the round after it (still elevated — the regime
/// gate measured the majority shape's first round after a block 3-4% slow),
/// and the measurement. The period is twice the regime gate's: a trial's
/// cost is paid per request while its knowledge persists on the model.
pub const EXPLORE_PERIOD: u32 = 16;
/// Period while the trial's target is still untrusted: the table persists,
/// so this is paid once per (chip, model, quant, OS) — measured M4 base 9B
/// cap 6, w5 reached 1-2 samples per 66-round boot at period 16 and the
/// +5% it buys stayed out of reach.
pub const EXPLORE_PERIOD_COLD: u32 = 8;
/// Half the regime gate's: a re-trial of a known cliff (M1 Pro 27B w5 is
/// 34% dearer than w4) costs block * gap / period of throughput, and the
/// measured per-request cost of one such block was 4.3% on a 22-round
/// request. The knowledge persists on the model; the drag is paid per call.
pub const EXPLORE_DRAG: f32 = 0.005;
pub const EXPLORE_PERIOD_MAX: u32 = 256;
pub const EXPLORE_BLOCK: u32 = 3;

/// Explicit trial schedule: a BLOCK of consecutive rounds every period,
/// idempotent per round (a planner may ask twice for the same round).
/// `idx % period` was tried first and chained trials because the block's
/// own observation moved the period.
pub const TrialSchedule = struct {
    trial_end: u32 = 0,
    next_trial: u32 = 0,
    trials: u32 = 0,
    last_idx: ?u32 = null,
    last_force: bool = false,

    pub fn force(t: *TrialSchedule, round_idx: u32, period: u32) bool {
        if (t.last_idx == round_idx) return t.last_force;
        t.last_idx = round_idx;
        t.last_force = blk: {
            if (round_idx < t.trial_end) break :blk true;
            if (t.next_trial == 0) {
                t.next_trial = round_idx + period;
                break :blk false;
            }
            if (round_idx >= t.next_trial) {
                t.trials += 1;
                t.trial_end = round_idx + EXPLORE_BLOCK;
                t.next_trial = t.trial_end + period;
                break :blk true;
            }
            break :blk false;
        };
        return t.last_force;
    }

    /// Start trialling at `round_idx` instead of one period later (a block
    /// drafter with no serial measurement must not run eight rounds blind).
    pub fn startAt(t: *TrialSchedule, round_idx: u32) void {
        if (t.next_trial == 0) t.next_trial = @max(1, round_idx);
    }
};

/// Period from the measured ms/tok gap between two widths (a width G worse,
/// run once in G/DRAG rounds, costs ~DRAG of throughput); the default while
/// either is unmeasured.
pub fn trialPeriod(a: ?f32, b: ?f32) u32 {
    const x = a orelse return EXPLORE_PERIOD_COLD;
    const y = b orelse return EXPLORE_PERIOD_COLD;
    if (!(x > 0) or !(y > 0)) return EXPLORE_PERIOD;
    const gap = @abs(x - y) / @min(x, y);
    const block: f32 = @floatFromInt(EXPLORE_BLOCK);
    const p: u32 = @intFromFloat(@ceil(block * gap / EXPLORE_DRAG - 1e-3));
    return @min(EXPLORE_PERIOD_MAX, @max(EXPLORE_PERIOD, p));
}

/// A standing choice only moves when the challenger beats it by this
/// margin: a width is measured from rounds interleaved with transitions
/// and reads a few percent slow.
pub const SWITCH_MARGIN: f32 = 0.05;

/// Per-round draft width for a block drafter (DFlash/DSpark): argmax over
/// widths 0..max of measured tokens per ms, serial (0) a candidate like any
/// other — "serial wins" IS the yield gate. The one unmeasured candidate is
/// widest+1 (tokens from the per-position acceptance chain, cost from the
/// last measured slope, never below flat), so the next cliff gets found;
/// everything else unmeasured is reached by trials: the standing width,
/// width-1, width+1, in that order. Serial is NEVER trialled: a plain
/// decode round does not extend the assistant context, so a request cannot
/// come back from serial (it is sticky, as the calibrated gate's fallback
/// is) — the w0 cell is fed by those sticky-serial rounds, and the chooser
/// picks serial only where one is measured. An m=0 verify round (one trunk
/// forward with captures, no assistant) would make serial trialable.
pub const WidthChooser = struct {
    pub const PRIOR: f32 = 0.8;
    /// Conditional per-position acceptance EMA, a[i] = P(draft i lands |
    /// drafts 0..i-1 landed).
    accept: [MAX_WIDTH]f32 = @splat(PRIOR),
    /// Standing width (drafts); the sidecar's default until data says else.
    current: u32,
    max_width: u32,
    trial: TrialSchedule = .{},
    hist: [MAX_WIDTH + 1]u32 = @splat(0),
    rounds: u32 = 0,
    /// Last standing verdict that was logged (the caller logs on change).
    logged: ?u32 = null,

    pub const Decision = struct { width: u32, trial: bool };

    pub fn init(default_width: u32, max_width: u32) WidthChooser {
        const mx = @min(max_width, MAX_WIDTH);
        return .{ .current = @min(@max(default_width, 1), mx), .max_width = mx };
    }

    pub fn observe(self: *WidthChooser, drafted: u32, accepted: u32, beta: f32) void {
        var i: usize = 0;
        while (i < accepted and i < self.accept.len) : (i += 1) self.accept[i] += beta * (1.0 - self.accept[i]);
        if (accepted < drafted and accepted < self.accept.len) self.accept[accepted] += beta * (0.0 - self.accept[accepted]);
    }

    pub fn expectedTokens(self: *const WidthChooser, w: u32) f32 {
        var chain: f32 = 1.0;
        var tok: f32 = 1.0;
        var k: u32 = 0;
        while (k < w and k < self.accept.len) : (k += 1) {
            chain *= self.accept[k];
            tok += chain;
        }
        return tok;
    }

    /// Same chain with a uniform per-position probability (tests).
    pub fn expectedTokensWith(_: *const WidthChooser, p: f32, w: u32) f32 {
        var chain: f32 = 1.0;
        var tok: f32 = 1.0;
        var k: u32 = 0;
        while (k < w) : (k += 1) {
            chain *= p;
            tok += chain;
        }
        return tok;
    }

    /// Tokens per ms of width `w` in `bucket`: measured where a cell exists;
    /// widest+1 from the chain + slope; null otherwise (not a candidate).
    pub fn score(self: *const WidthChooser, t: *const Table, bucket: usize, w: u32) ?f32 {
        if (w > self.max_width) return null;
        if (t.measuredMs(w, bucket)) |ms| {
            const tok = t.measuredTok(w, bucket) orelse return null;
            return if (ms > 0) tok / ms else null;
        }
        const widest = t.widestMeasured(bucket) orelse return null;
        if (w != widest + 1) return null;
        const base = t.measuredMs(widest, bucket) orelse return null;
        const slope = @max(t.lastSlope(bucket) orelse 0.0, 0.0);
        const ms = base + slope;
        return if (ms > 0) self.expectedTokens(w) / ms else null;
    }

    /// Which width a trial measures next, or null when nothing is owed.
    pub fn trialTarget(self: *const WidthChooser, t: *const Table, bucket: usize) ?u32 {
        if (self.current == 0) return null;
        if (t.measuredMs(self.current, bucket) == null) return self.current;
        if (self.current > 1 and t.measuredMs(self.current - 1, bucket) == null and !t.clearlyWorse(self.current - 1, self.current, bucket)) return self.current - 1;
        if (self.current < self.max_width and t.measuredMs(self.current + 1, bucket) == null and !t.clearlyWorse(self.current + 1, self.current, bucket)) return self.current + 1;
        return null;
    }

    /// The width this round runs. `round_idx` = rounds so far (post-warmup).
    pub fn choose(self: *WidthChooser, t: *const Table, kv_len: u32, round_idx: u32) Decision {
        const bucket = t.bucketToRead(kv_len) orelse t.bucketOf(kv_len);
        // Standing choice: the best measured-or-widest+1 candidate, with
        // hysteresis against the current width — and never while the
        // current width is itself unmeasured (a measured w0 from an earlier
        // sticky-serial request would otherwise win round 0 of every later
        // request before the block was ever measured at this context).
        if (self.score(t, bucket, self.current)) |cur| {
            var best_w = self.current;
            var best_s: f32 = cur;
            var w: u32 = 0;
            while (w <= self.max_width) : (w += 1) {
                const sc = self.score(t, bucket, w) orelse continue;
                if (sc > best_s * (1.0 + SWITCH_MARGIN)) {
                    best_s = sc;
                    best_w = w;
                }
            }
            self.current = best_w;
        }
        // Trials measure what the argmax cannot see.
        if (self.trialTarget(t, bucket)) |target| {
            self.trial.startAt(round_idx);
            const period = trialPeriod(t.msPerTok(self.current, bucket), t.msPerTok(target, bucket));
            if (self.trial.force(round_idx, period)) return .{ .width = target, .trial = true };
        }
        return .{ .width = self.current, .trial = false };
    }

    pub fn note(self: *WidthChooser, width: u32) void {
        self.rounds += 1;
        if (width <= MAX_WIDTH) self.hist[width] += 1;
    }

    /// Drafts proposed across all rounds (sum of width x rounds at it).
    pub fn draftsProposed(self: *const WidthChooser) u64 {
        var sum: u64 = 0;
        for (self.hist, 0..) |n, w| sum += @as(u64, n) * w;
        return sum;
    }

    pub fn avgWidth(self: *const WidthChooser) f32 {
        if (self.rounds == 0) return 0;
        var sum: u64 = 0;
        for (self.hist, 0..) |n, w| sum += @as(u64, n) * w;
        return @as(f32, @floatFromInt(sum)) / @as(f32, @floatFromInt(self.rounds));
    }

    /// `w0:3,w4:120,w5:2` for `[spec-stats]`.
    pub fn formatHist(self: *const WidthChooser, buf: []u8) []const u8 {
        var wr = std.Io.Writer.fixed(buf);
        var first = true;
        for (self.hist, 0..) |n, w| {
            if (n == 0) continue;
            if (!first) wr.writeAll(",") catch break;
            first = false;
            wr.print("w{d}:{d}", .{ w, n }) catch break;
        }
        return wr.buffered();
    }
};

// ── Persistence ──────────────────────────────────────────────────────────
//
// Every fresh boot otherwise pays the exploration again (measured: 3-4% on
// whichever request carries a trial block, 22-round requests), while the
// knowledge is per (chip, model, quant, OS build) and does not change
// between boots. Stored under ~/.mlx-serve/round-cost/<key>.txt, restored
// at load, written at the end of any request that folded new samples.
// Stale version or unreadable content is a QUIET miss (the kv_disk_cache
// discipline). `MLX_SERVE_ROUND_COST_PERSIST=0` disables both directions.

/// v2 added the `serial` row. The bump was not decoration: v1 files were
/// written by a build in which the only "no speculation" evidence was a
/// width-0 CELL measured on a block decoder's round clock, and a v1 reader
/// handed a v2 file would silently take the `s` lines for width `s`.
///
/// v3 split the top bucket (edges 64k/128k/256k). Bucket INDICES are the
/// file's only spelling of "which context", so a v2 file read as v3 would
/// map its `32k+` cell — an EMA of everything from 32k to 1M — onto the new
/// `32-64k`, and the three long-context buckets would inherit numbers
/// measured at a different context. A stale version is a QUIET miss, so the
/// cost of the bump is one re-explore per (chip, model, quant, OS build).
pub const STORE_VERSION: u32 = 3;

/// The store version a layout writes and reads. The legacy grid keeps `rc1`
/// — the format 26.9.1 shipped — so a user upgrading onto a sidecar pack
/// finds the file they already have and boots WARM. `rc2` is nobody's: it
/// was written by a build whose bucket indices were the legacy ones but
/// whose body carried a serial row, and reading one as either layout would
/// mis-assign a row.
pub fn storeVersion(layout: Layout) u32 {
    return switch (layout) {
        .legacy => 1,
        .long => 3,
    };
}

/// Samples folded into EITHER row. The persistence trigger reads this, so a
/// boot that only ever learned serial cells still writes them, while the two
/// counters stay separate for reporting.
pub fn totalFolded(t: *const Table) u32 {
    return t.folded +% t.serial_folded;
}

pub fn persistEnabled() bool {
    const raw = std.c.getenv("MLX_SERVE_ROUND_COST_PERSIST") orelse return true;
    return !std.mem.eql(u8, std.mem.span(raw), "0");
}

/// Same identity rule as the spec-cost probe's key: every field the cost
/// depends on, hashed, so one machine's cliff is never served to another.
pub fn cacheKey(buf: []u8, chip: []const u8, model_dir: []const u8, quant: []const u8, os_build: []const u8, layout: Layout) []const u8 {
    var h = std.hash.Fnv1a_64.init();
    for ([_][]const u8{ chip, model_dir, quant, os_build }) |part| {
        h.update(part);
        h.update("\x00");
    }
    // The HASH is layout-blind on purpose: only the version prefix moves, so
    // a legacy table lands on the very file a 26.9.1 boot wrote for the same
    // (chip, model, quant, OS build).
    return std.fmt.bufPrint(buf, "rc{d}-{x:0>16}", .{ storeVersion(layout), h.final() }) catch buf[0..0];
}

/// `rc3\n`, then one `width bucket ms tok n` line per folded width cell and
/// one `s bucket ms tok n` line per folded serial cell.
pub fn serialize(buf: []u8, t: *const Table) ![]const u8 {
    var w = std.Io.Writer.fixed(buf);
    try w.print("rc{d}\n", .{storeVersion(t.layout)});
    for (t.cells, 0..) |row, wi| {
        for (row, 0..) |c, b| {
            if (c.n == 0) continue;
            try w.print("{d} {d} {d:.4} {d:.4} {d}\n", .{ wi, b, c.ms, c.tok, c.n });
        }
    }
    // The serial row is v3-only. A legacy table never folds one (the switch
    // that reads it is arch-gated), and writing an `s` line into an rc1 file
    // would hand an older build a cell for width `s`.
    if (t.layout == .long) {
        for (t.serial, 0..) |c, b| {
            if (c.n == 0) continue;
            try w.print("s {d} {d:.4} {d:.4} {d}\n", .{ b, c.ms, c.tok, c.n });
        }
    }
    return w.buffered();
}

/// Null on any version or shape mismatch. Restored cells keep their sample
/// counts (trust) but are marked STALE, so the first live sample of each
/// blends at RESEED_WEIGHT — another boot is another thermal/OS state.
pub fn parse(text: []const u8, layout: Layout) ?Table {
    var lines = std.mem.splitScalar(u8, text, '\n');
    const head = lines.next() orelse return null;
    var hb: [16]u8 = undefined;
    const want = std.fmt.bufPrint(&hb, "rc{d}", .{storeVersion(layout)}) catch return null;
    if (!std.mem.eql(u8, std.mem.trim(u8, head, " \r"), want)) return null;
    var t = Table{ .layout = layout };
    while (lines.next()) |line| {
        const l = std.mem.trim(u8, line, " \r");
        if (l.len == 0) continue;
        var f = std.mem.splitScalar(u8, l, ' ');
        const head_field = f.next() orelse return null;
        const is_serial = std.mem.eql(u8, head_field, "s");
        const wi: u32 = if (is_serial) 0 else std.fmt.parseInt(u32, head_field, 10) catch return null;
        const b = std.fmt.parseInt(usize, f.next() orelse return null, 10) catch return null;
        const ms = std.fmt.parseFloat(f32, f.next() orelse return null) catch return null;
        const tok = std.fmt.parseFloat(f32, f.next() orelse return null) catch return null;
        const n = std.fmt.parseInt(u32, f.next() orelse return null, 10) catch return null;
        // Range is the LAYOUT's, not the array's: an rc1 file can only carry
        // buckets 0..5, and a body claiming otherwise is not an rc1 file.
        if (wi > MAX_WIDTH or b >= nBuckets(layout) or n == 0) return null;
        if (is_serial and layout != .long) return null;
        if (!std.math.isFinite(ms) or ms <= 0 or !(tok > 0)) return null;
        const cell = Cell{ .ms = ms, .tok = tok, .n = n, .last_seen = 0 };
        if (is_serial) t.serial[b] = cell else t.cells[wi][b] = cell;
    }
    t.seq = RESEED_GAP + 1;
    t.serial_seq = RESEED_GAP + 1;
    t.restored = t.foldedCells();
    t.restored_serial = t.foldedSerialCells();
    return t;
}

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse return "/tmp");
}

fn cachePath(buf: []u8, key: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/round-cost/{s}.txt", .{ homeDir(), key }) catch null;
}

/// Read one table file at `key`, parsed under `layout`. Null on anything at
/// all — missing, unreadable, wrong version.
fn readCached(allocator: std.mem.Allocator, io: std.Io, key: []const u8, layout: Layout) ?Table {
    if (key.len == 0) return null;
    var path_buf: [512]u8 = undefined;
    const path = cachePath(&path_buf, key) orelse return null;
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const text = rs.interface.allocRemaining(allocator, .limited(16384)) catch return null;
    defer allocator.free(text);
    return parse(text, layout);
}

/// Lift a legacy (six-bucket) table onto the long grid. The two grids share
/// the edges 2048/4096/8192/16384/32768, so buckets 0..4 mean EXACTLY the
/// same context in both and carry over cell for cell.
///
/// Bucket 5 does not: on the legacy grid it is `32k+`, an EMA of everything
/// from 32k to 1M, and the long grid has three cells in that span. Handing
/// it to any one of them would serve numbers measured at another context —
/// the reason the version was bumped in the first place — so it is DROPPED.
/// The long grid re-measures its own long-context cells, and the short ones,
/// which is where a boot's first requests live, are warm from token one.
pub fn migrateLegacy(src: Table) Table {
    var t = Table{ .layout = .long };
    const shared = nBuckets(.legacy) - 1; // 0..4: identical edges
    for (src.cells, 0..) |row, wi| {
        for (row[0..shared], 0..) |c, b| t.cells[wi][b] = c;
    }
    t.seq = RESEED_GAP + 1;
    t.serial_seq = RESEED_GAP + 1;
    t.restored = t.foldedCells();
    t.restored_serial = 0;
    return t;
}

/// The table for `layout`, warm-started from the previous format when its
/// own file is absent.
///
/// A store-version bump is not "one re-explore": `MtpCostSource.fromTable()`
/// is the single term deciding whether the EV plan prices widths and
/// extension from MEASUREMENTS or from the fitted prior, and the prior plans
/// a different width and an always-open extension valve. A cold boot is
/// therefore a measurable regression on the first requests of every process,
/// not a one-off. Reading the old file is what makes the bump free.
pub fn loadCached(allocator: std.mem.Allocator, io: std.Io, key: []const u8, layout: Layout) ?Table {
    if (!persistEnabled()) return null;
    if (readCached(allocator, io, key, layout)) |t| return t;
    if (layout != .long) return null;
    // Fall back to the legacy file for the SAME (chip, model, quant, OS
    // build): only the version prefix differs, so it is one bufPrint away.
    if (key.len < 4) return null;
    var legacy_key_buf: [64]u8 = undefined;
    const legacy_key = std.fmt.bufPrint(&legacy_key_buf, "rc{d}-{s}", .{ storeVersion(.legacy), key[4..] }) catch return null;
    const old = readCached(allocator, io, legacy_key, .legacy) orelse return null;
    return migrateLegacy(old);
}

/// Best-effort: a machine that cannot write re-explores next boot.
pub fn storeCached(io: std.Io, key: []const u8, t: *const Table) void {
    if (!persistEnabled() or key.len == 0) return;
    var dir_buf: [512]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/.mlx-serve/round-cost", .{homeDir()}) catch return;
    std.Io.Dir.cwd().createDirPath(io, dir) catch return;
    var path_buf: [512]u8 = undefined;
    const path = cachePath(&path_buf, key) orelse return;
    var text: [8192]u8 = undefined;
    const body = serialize(&text, t) catch return;
    const f = std.Io.Dir.createFileAbsolute(io, path, .{}) catch return;
    defer f.close(io);
    var wb: [8192]u8 = undefined;
    var fw = f.writer(io, &wb);
    fw.interface.writeAll(body) catch return;
    fw.interface.flush() catch {};
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

/// Fold MIN_SAMPLES identical samples so the cell counts as measured.
fn feed(t: *Table, width: u32, kv: u32, ms: f32, tok: f32) void {
    var i: u32 = 0;
    while (i < MIN_SAMPLES) : (i += 1) _ = t.observe(width, kv, ms, tok, true, false);
}

test "round_cost: kv buckets" {
    try testing.expectEqual(@as(usize, 0), bucketFor(0));
    try testing.expectEqual(@as(usize, 0), bucketFor(2047));
    try testing.expectEqual(@as(usize, 1), bucketFor(2048));
    try testing.expectEqual(@as(usize, 3), bucketFor(8192));
    try testing.expectEqual(@as(usize, 4), bucketFor(16384));
    try testing.expectEqual(@as(usize, 5), bucketFor(32768));
    // Long context is no longer one cell: every new edge is exact, and the
    // last bucket is the only unbounded one.
    try testing.expectEqual(@as(usize, 5), bucketFor(65535));
    try testing.expectEqual(@as(usize, 6), bucketFor(65536));
    try testing.expectEqual(@as(usize, 6), bucketFor(131071));
    try testing.expectEqual(@as(usize, 7), bucketFor(131072));
    try testing.expectEqual(@as(usize, 7), bucketFor(262143));
    try testing.expectEqual(@as(usize, 8), bucketFor(262144));
    try testing.expectEqual(@as(usize, 8), bucketFor(1_000_000));
    // The two contexts the A/B actually ran are now DIFFERENT buckets — the
    // whole point of the split (62.7k prose vs a 374k ladder rung).
    try testing.expect(bucketFor(62_755) != bucketFor(374_000));
    try testing.expectEqual(N_BUCKETS, BUCKET_NAMES.len);
    try testing.expectEqual(N_BUCKETS - 1, BUCKET_EDGES.len);
}

test "round_cost: EMA folds, first sample seeds, a cell counts at MIN_SAMPLES, a long gap reseeds" {
    var t = Table{};
    try testing.expectEqual(Verdict.reseeded, t.observe(4, 1000, 50.0, 4.0, true, false));
    try testing.expectEqual(Verdict.folded, t.observe(4, 1000, 60.0, 4.0, true, false));
    try testing.expect(t.measuredMs(4, 0) == null); // two samples do not count
    try testing.expectApproxEqAbs(55.0, t.cells[4][0].ms, 1e-4); // mean while filling
    _ = t.observe(4, 1000, 52.0, 4.0, true, false);
    try testing.expectApproxEqAbs(54.0, t.measuredMs(4, 0).?, 1e-4);
    _ = t.observe(4, 1000, 64.0, 4.0, true, false);
    try testing.expectApproxEqAbs(54.0 + BETA * 10.0, t.measuredMs(4, 0).?, 1e-4); // EMA after
    try testing.expectApproxEqAbs(t.measuredMs(4, 0).? / 4.0, t.msPerTok(4, 0).?, 1e-4);
    // Other widths tick the clock; the width-4 cell goes stale: the next
    // sample weighs RESEED_WEIGHT and the cell stays trusted.
    const before = t.cells[4][0].ms;
    var i: u32 = 0;
    while (i <= RESEED_GAP) : (i += 1) _ = t.observe(3, 1000, 40.0, 3.0, true, false);
    try testing.expectEqual(Verdict.reseeded, t.observe(4, 1000, 90.0, 4.0, true, false));
    try testing.expectApproxEqAbs(before + RESEED_WEIGHT * (90.0 - before), t.cells[4][0].ms, 1e-3);
    try testing.expect(t.measuredMs(4, 0) != null);
}

test "round_cost: a contended, transition or bad sample never moves the estimate" {
    var t = Table{};
    feed(&t, 4, 1000, 50.0, 4.0);
    try testing.expectEqual(Verdict.contended, t.observe(4, 1000, 500.0, 4.0, false, false));
    try testing.expectEqual(Verdict.transition, t.observe(4, 1000, 500.0, 4.0, true, true));
    try testing.expectEqual(Verdict.bad_sample, t.observe(4, 1000, 0.0, 4.0, true, false));
    try testing.expectEqual(Verdict.bad_sample, t.observe(4, 1000, 50.0, 0.0, true, false));
    try testing.expectEqual(Verdict.out_of_range, t.observe(MAX_WIDTH + 1, 1000, 50.0, 4.0, true, false));
    try testing.expectApproxEqAbs(50.0, t.measuredMs(4, 0).?, 1e-4);
    try testing.expectEqual(@as(u32, 1), t.dropped_contended);
    try testing.expectEqual(@as(u32, 1), t.dropped_transition);
    try testing.expectEqual(@as(u32, 2), t.dropped_bad);
    try testing.expectEqual(MIN_SAMPLES, t.folded);
}

test "round_cost: one width anchors, two interpolate, nothing extrapolates" {
    var t = Table{};
    _ = t.observe(3, 1000, 30.0, 3.0, true, false);
    try testing.expect(!t.active(0)); // one untrusted sample is nothing
    try testing.expect(t.bucketToRead(1000) == null);
    feed(&t, 3, 1000, 30.0, 3.0);
    try testing.expect(t.active(0));
    try testing.expect(t.roundMs(4, 0) == null); // one point: the prior's shape fills
    try testing.expectEqual(@as(usize, 0), t.bucketToRead(1000).?);
    feed(&t, 5, 1000, 50.0, 5.0);
    try testing.expect(t.active(0));
    try testing.expectApproxEqAbs(40.0, t.roundMs(4, 0).?, 1e-4); // between
    try testing.expect(t.roundMs(8, 0) == null); // past the widest: the caller composes
    try testing.expect(t.roundMs(2, 0) == null); // below the anchor: the prior's job
    try testing.expectEqual(@as(u32, 3), t.narrowestMeasured(0).?);
    try testing.expectEqual(@as(u32, 5), t.widestMeasured(0).?);
    try testing.expectApproxEqAbs(10.0, t.lastSlope(0).?, 1e-4);
    // A measured cliff is read as measured, and the slope past it is the cliff's.
    feed(&t, 6, 1000, 200.0, 6.0);
    try testing.expectApproxEqAbs(200.0, t.roundMs(6, 0).?, 1e-4);
    try testing.expectApproxEqAbs(150.0, t.lastSlope(0).?, 1e-4);
}

test "round_cost: an unmeasured bucket reads the nearest active one, lower side first" {
    var t = Table{};
    feed(&t, 3, 3000, 30.0, 3.0);
    feed(&t, 4, 3000, 40.0, 4.0);
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(3000).?);
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(20000).?); // 16-32k falls back down to 2-4k
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(100).?); // <2k falls back up
    feed(&t, 3, 20000, 60.0, 3.0);
    feed(&t, 4, 20000, 80.0, 4.0);
    try testing.expectEqual(@as(usize, 4), t.bucketToRead(20000).?);
    try testing.expectEqual(@as(usize, 1), t.bucketToRead(6000).?); // 4-8k: lower (2-4k) beats upper (16-32k)
    try testing.expectEqual(@as(usize, 4), t.bucketToRead(10000).?); // 8-16k: 16-32k is nearer than 2-4k
}

test "round_cost: formatBucket lists folded widths as ms/tok with sample counts" {
    var t = Table{};
    feed(&t, 3, 1000, 30.0, 3.0);
    _ = t.observe(5, 1000, 60.0, 4.0, true, false);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("w3:10.00/3,w5:15.00/1", t.formatBucket(0, &buf));
    try testing.expectEqualStrings("", t.formatBucket(1, &buf));
}

test "round_cost: trialPeriod and TrialSchedule blocks" {
    try testing.expectEqual(EXPLORE_PERIOD_COLD, trialPeriod(null, 10.0));
    try testing.expectEqual(@as(u32, 60), trialPeriod(10.0, 11.0));
    try testing.expectEqual(EXPLORE_PERIOD_MAX, trialPeriod(10.0, 30.0));
    var t = TrialSchedule{};
    var forced: u32 = 0;
    var i: u32 = 3;
    while (i < 103) : (i += 1) {
        const f = t.force(i, 8);
        try testing.expectEqual(f, t.force(i, 8));
        if (f) forced += 1;
    }
    try testing.expectEqual(t.trials * EXPLORE_BLOCK, forced);
    var s = TrialSchedule{};
    s.startAt(3);
    try testing.expect(s.force(3, 8)); // starts at once, not a period later
}

/// Synthetic block drafter: per-position acceptance p, round ms linear in
/// width with a cliff past `cliff` — the M4 base 2.6B (best block 6) and
/// 8B-A1B (serial wins) shapes, driven as nextDflash will drive it.
fn simChooser(p: f32, serial_ms: f32, per_pos_ms: f32, cliff: u32, cliff_ms: f32, default_w: u32, max_w: u32, rounds: u32, serial_known: bool) WidthChooser {
    var t = Table{};
    // Serial is measured only by sticky-serial rounds of earlier requests.
    if (serial_known) feed(&t, 0, 1000, serial_ms, 1.0);
    var c = WidthChooser.init(default_w, max_w);
    var prev: ?u32 = null;
    var i: u32 = 0;
    while (i < rounds) : (i += 1) {
        const d = c.choose(&t, 1000, i);
        const w = d.width;
        // Realize the round: drafts land while chain holds (deterministic
        // expectation, so the sim has no RNG).
        const tok = c.expectedTokensWith(p, w);
        var ms = serial_ms + per_pos_ms * @as(f32, @floatFromInt(w));
        if (w > cliff) ms += cliff_ms * @as(f32, @floatFromInt(w - cliff));
        const acc: u32 = @intFromFloat(@floor(tok - 1.0));
        c.observe(w, acc, 0.15);
        _ = t.observe(w, 1000, ms, tok, true, if (prev) |pw| pw != w else true);
        c.note(w);
        prev = w;
    }
    return c;
}

test "round_cost: WidthChooser settles at the best block and re-tries its neighbours" {
    // 2.6B-like: high acceptance, cost flat-ish to 6 then a cliff.
    const c = simChooser(0.9, 20.0, 1.0, 6, 30.0, 4, 8, 400, true);
    try testing.expectEqual(@as(u32, 6), c.current);
    try testing.expect(c.hist[6] > 200);
    try testing.expect(c.hist[7] + c.hist[8] <= 2 * EXPLORE_BLOCK); // widest+1 found the cliff, re-tried rarely
    try testing.expectEqual(@as(u32, 0), c.hist[0]); // serial is never trialled
    // Without a serial measurement the same loop settles the same.
    const d = simChooser(0.9, 20.0, 1.0, 6, 30.0, 4, 8, 400, false);
    try testing.expectEqual(@as(u32, 6), d.current);
}

test "round_cost: WidthChooser picks serial when the block loses, and comes back when it wins" {
    // 8B-A1B-like: low acceptance, expensive verify.
    const lose = simChooser(0.3, 10.0, 6.0, 8, 0.0, 4, 8, 300, true);
    try testing.expectEqual(@as(u32, 0), lose.current);
    try testing.expect(lose.hist[0] > 240);
    // Same machine, echo-like acceptance: the block pays.
    const win = simChooser(0.95, 10.0, 6.0, 8, 0.0, 4, 8, 300, true);
    try testing.expect(win.current >= 4);
    try testing.expectEqual(@as(u32, 0), win.hist[0]);
    // Serial unmeasured: the losing block keeps running (the calibrated
    // sticky gate is the bootstrap that gets serial measured).
    const blind = simChooser(0.3, 10.0, 6.0, 8, 0.0, 4, 8, 300, false);
    try testing.expect(blind.current >= 1);
}

test "round_cost: persistence round-trips folded cells, marks them stale, rejects other versions" {
    var t = Table{ .layout = .long };
    feed(&t, 4, 1000, 50.0, 4.5);
    _ = t.observe(5, 20000, 80.0, 5.0, true, false);
    var buf: [1024]u8 = undefined;
    const text = try serialize(&buf, &t);
    const back = parse(text, .long) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 2), back.restored);
    try testing.expectEqual(@as(u32, 0), back.restored_serial); // no serial row in this table
    try testing.expectApproxEqAbs(50.0, back.measuredMs(4, 0).?, 1e-3);
    try testing.expectApproxEqAbs(4.5, back.measuredTok(4, 0).?, 1e-3);
    try testing.expectEqual(@as(u32, 1), back.cells[5][4].n);
    // First live sample of a restored cell blends at RESEED_WEIGHT.
    var live = back;
    try testing.expectEqual(Verdict.reseeded, live.observe(4, 1000, 70.0, 4.5, true, false));
    try testing.expectApproxEqAbs(60.0, live.measuredMs(4, 0).?, 1e-3);
    try testing.expect(parse("rc0\n4 0 50 4 3\n", .long) == null);
    try testing.expect(parse("rc1\n4 0 50 4 3\n", .long) == null); // the pre-serial format is a quiet miss ON THE LONG LAYOUT
    // v2 is a quiet miss on BOTH layouts: its bucket INDICES are the legacy
    // ones but its body can carry a serial row, so neither reader can trust it.
    try testing.expect(parse("rc2\n4 5 50 4 3\n", .long) == null);
    try testing.expect(parse("rc2\n4 5 50 4 3\n", .legacy) == null);
    try testing.expect(parse("rc3\n99 0 50 4 3\n", .long) == null);
    try testing.expect(parse("", .long) == null);
    var kb: [64]u8 = undefined;
    // The version rides the KEY as well as the body, and the two layouts are
    // two files: `rc3-<h>` for qwen4_exp, `rc1-<h>` — the very file 26.9.1
    // wrote — for everything else. Same HASH, so a legacy boot finds it.
    try testing.expect(std.mem.startsWith(u8, cacheKey(&kb, "M4", "/m", "q4g64", "26.4", .long), "rc3-"));
    var kb2: [64]u8 = undefined;
    const legacy_key = cacheKey(&kb2, "M4", "/m", "q4g64", "26.4", .legacy);
    try testing.expect(std.mem.startsWith(u8, legacy_key, "rc1-"));
    try testing.expectEqualStrings(
        cacheKey(&kb, "M4", "/m", "q4g64", "26.4", .long)[4..],
        legacy_key[4..],
    );
}

test "round_cost: the serial row keeps its OWN fold and drop counters (M19)" {
    var t = Table{};
    // A width sample is a ROUND, a serial sample is a TOKEN. Sharing the
    // counters made `table_drops` two clocks summed into one number, and made
    // the persistence trigger fire on plain decode traffic that had taught the
    // width grid nothing.
    _ = t.observe(4, 1000, 50.0, 4.0, true, false);
    try testing.expectEqual(@as(u32, 1), t.folded);
    try testing.expectEqual(@as(u32, 0), t.serial_folded);

    _ = t.observeSerial(1000, 16.0, true, false);
    _ = t.observeSerial(1000, 16.0, true, false);
    try testing.expectEqual(@as(u32, 1), t.folded); // width clock did not move
    try testing.expectEqual(@as(u32, 2), t.serial_folded);

    // Drops are separate too, per row.
    _ = t.observe(4, 1000, 50.0, 4.0, false, false); // contended round
    _ = t.observeSerial(1000, 16.0, false, false); // contended tick
    _ = t.observeSerial(1000, 0.0, true, false); // bad tick
    _ = t.observeSerial(1000, 16.0, true, true); // transition tick
    try testing.expectEqual(@as(u32, 1), t.dropped_contended);
    try testing.expectEqual(@as(u32, 0), t.dropped_bad);
    try testing.expectEqual(@as(u32, 0), t.dropped_transition);
    try testing.expectEqual(@as(u32, 1), t.serial_dropped_contended);
    try testing.expectEqual(@as(u32, 1), t.serial_dropped_bad);
    try testing.expectEqual(@as(u32, 1), t.serial_dropped_transition);

    // The persistence trigger reads BOTH, so a boot that only learned serial
    // cells still writes them.
    try testing.expectEqual(@as(u32, 3), totalFolded(&t));
    var only_serial = Table{};
    try testing.expectEqual(@as(u32, 0), totalFolded(&only_serial));
    _ = only_serial.observeSerial(1000, 16.0, true, false);
    try testing.expectEqual(@as(u32, 1), totalFolded(&only_serial));
}

test "round_cost: the serial row folds, trusts at MIN_SAMPLES and round-trips beside the widths" {
    var t = Table{ .layout = .long };
    try testing.expect(t.serialMsPerTok(0) == null);
    // Same drop rules as a round sample; none of them touch the width grid.
    try testing.expectEqual(Verdict.contended, t.observeSerial(1000, 15.0, false, false));
    try testing.expectEqual(Verdict.transition, t.observeSerial(1000, 15.0, true, true));
    try testing.expectEqual(Verdict.bad_sample, t.observeSerial(1000, 0.0, true, false));
    try testing.expectEqual(Verdict.reseeded, t.observeSerial(1000, 15.0, true, false));
    _ = t.observeSerial(1000, 17.0, true, false);
    try testing.expect(t.serialMsPerTok(0) == null); // two samples do not count
    _ = t.observeSerial(1000, 16.0, true, false);
    try testing.expectApproxEqAbs(16.0, t.serialMsPerTok(0).?, 1e-4);
    // A serial cell is never a width: it does not activate the bucket, does
    // not anchor `narrowestMeasured`, and cannot be lerped through.
    try testing.expect(!t.active(0));
    try testing.expect(t.bucketToRead(1000) == null);
    try testing.expect(t.measuredMs(0, 0) == null);
    try testing.expect(t.narrowestMeasured(0) == null);
    // Buckets are independent — an unmeasured one is never filled from a
    // neighbour (the decision would then compare two different contexts).
    try testing.expect(t.serialMsPerTok(1) == null);
    try testing.expect(t.serialMsPerTok(N_BUCKETS) == null);

    feed(&t, 4, 1000, 50.0, 4.0);
    var buf: [1024]u8 = undefined;
    const text = try serialize(&buf, &t);
    try testing.expect(std.mem.indexOf(u8, text, "\ns 0 16.") != null);
    const back = parse(text, .long) orelse return error.TestUnexpectedResult;
    // The two counts are SEPARATE: `restored` is what the width planner got
    // back, `restored_serial` what the adaptive switch got back. A single
    // number would have made "1 width cell" and "1 serial cell" indistinguishable
    // from "2 width cells" in the one place either is ever read (the boot log).
    try testing.expectEqual(@as(u32, 1), back.restored);
    try testing.expectEqual(@as(u32, 1), back.restored_serial);
    try testing.expectEqual(@as(u32, 1), back.foldedCells());
    try testing.expectEqual(@as(u32, 1), back.foldedSerialCells());
    try testing.expectApproxEqAbs(16.0, back.serialMsPerTok(0).?, 1e-3);
    try testing.expectApproxEqAbs(50.0, back.measuredMs(4, 0).?, 1e-3);
    // A restored table re-probes at most once per boot: the flag is runtime.
    try testing.expectEqual(@as(u8, 0), back.serial_probes[0]);
}

test "round_cost: the serial row and the width grid keep SEPARATE reseed clocks" {
    // A serial tick is a TOKEN and a width sample is a ROUND. Sharing one
    // clock would take every width cell past RESEED_GAP within a second of
    // serial decoding, and the width planner would blend every later sample
    // at RESEED_WEIGHT instead of BETA — forever, on any mixed workload.
    var t = Table{};
    feed(&t, 3, 1000, 30.0, 3.0);
    const w3 = t.measuredMs(3, 0).?;
    var k: u32 = 0;
    while (k <= RESEED_GAP * 4) : (k += 1) _ = t.observeSerial(1000, 16.0, true, false);
    _ = t.observe(3, 1000, 40.0, 3.0, true, false);
    try testing.expectApproxEqAbs(w3 + BETA * (40.0 - w3), t.measuredMs(3, 0).?, 1e-3);

    // And the other way: rounds do not age the serial cell either.
    var u = Table{};
    var i: u32 = 0;
    while (i < MIN_SAMPLES) : (i += 1) _ = u.observeSerial(1000, 16.0, true, false);
    const ser = u.serialMsPerTok(0).?;
    var j: u32 = 0;
    while (j <= RESEED_GAP * 4) : (j += 1) _ = u.observe(3, 1000, 30.0, 3.0, true, false);
    _ = u.observeSerial(1000, 20.0, true, false);
    try testing.expectApproxEqAbs(ser + BETA * (20.0 - ser), u.serialMsPerTok(0).?, 1e-3);
}

test "round_cost: a clearly worse first sample settles a width" {
    var t = Table{};
    feed(&t, 4, 1000, 70.0, 5.0);
    try testing.expect(!t.clearlyWorse(5, 4, 0)); // unsampled: unknown
    _ = t.observe(5, 1000, 110.0, 6.0, true, false); // 18.3 vs 14.0 ms/tok = +31%
    try testing.expect(t.clearlyWorse(5, 4, 0));
    try testing.expect(t.measuredMs(5, 0) == null); // still not trusted for the plan's cost
    try testing.expectApproxEqAbs(110.0, t.rawMs(5, 0).?, 1e-4);
    var u = Table{};
    feed(&u, 4, 1000, 70.0, 5.0);
    _ = u.observe(5, 1000, 86.0, 6.0, true, false); // 14.3 vs 14.0: noise, keep trialling
    try testing.expect(!u.clearlyWorse(5, 4, 0));
}

test "round_cost: the legacy layout is a93e2c0's grid, writes rc1 and reads the file 26.9.1 wrote (L27)" {
    // The six-bucket grid, bucket for bucket. These literals are a93e2c0's
    // `BUCKET_EDGES` — the grid every release through 26.9.1 wrote — and
    // they are what a sidecar-MTP pack must keep reading, because the store
    // version rides the FILE NAME: a bump is a cold table, a cold table is
    // `fromTable() == false`, and the prior's extension valve is always open.
    try testing.expectEqual(@as(usize, 0), bucketForLayout(0, .legacy));
    try testing.expectEqual(@as(usize, 0), bucketForLayout(2047, .legacy));
    try testing.expectEqual(@as(usize, 1), bucketForLayout(2048, .legacy));
    try testing.expectEqual(@as(usize, 2), bucketForLayout(4096, .legacy));
    try testing.expectEqual(@as(usize, 3), bucketForLayout(8192, .legacy));
    try testing.expectEqual(@as(usize, 4), bucketForLayout(16384, .legacy));
    try testing.expectEqual(@as(usize, 5), bucketForLayout(32768, .legacy));
    // The top bucket is UNBOUNDED again on the legacy grid — 62.7k and 374k
    // share a cell, exactly as they did before the split.
    try testing.expectEqual(@as(usize, 5), bucketForLayout(65536, .legacy));
    try testing.expectEqual(@as(usize, 5), bucketForLayout(1_000_000, .legacy));
    try testing.expectEqual(bucketForLayout(62_755, .legacy), bucketForLayout(374_000, .legacy));
    // …and the long grid still splits them. Below 32k the two agree, which
    // is why an 8k request reads the same cell either way.
    try testing.expect(bucketForLayout(62_755, .long) != bucketForLayout(374_000, .long));
    var kv: u32 = 0;
    while (kv < 32768) : (kv += 337) {
        try testing.expectEqual(bucketForLayout(kv, .long), bucketForLayout(kv, .legacy));
    }
    try testing.expectEqual(@as(usize, 6), nBuckets(.legacy));
    try testing.expectEqual(N_BUCKETS, nBuckets(.long));

    // A table's own resolver honours its layout; the free `bucketFor` is the
    // LONG grid and must never be used against a legacy table.
    const legacy = Table{ .layout = .legacy };
    try testing.expectEqual(@as(usize, 5), legacy.bucketOf(400_000));
    try testing.expectEqual(@as(usize, 8), bucketFor(400_000));

    // The file 26.9.1 wrote, byte for byte, loads with its cells intact.
    const rc1_body = "rc1\n2 3 44.0000 2.7000 3\n3 3 88.0000 3.0500 3\n4 3 140.0000 3.2000 3\n";
    const back = parse(rc1_body, .legacy) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(Layout.legacy, back.layout);
    try testing.expectEqual(@as(u32, 3), back.restored);
    try testing.expectEqual(@as(u32, 0), back.restored_serial);
    try testing.expectApproxEqAbs(44.0, back.measuredMs(2, 3).?, 1e-3);
    try testing.expectApproxEqAbs(3.05, back.measuredTok(3, 3).?, 1e-3);
    try testing.expectApproxEqAbs(140.0, back.measuredMs(4, 3).?, 1e-3);
    try testing.expect(back.active(3));
    // …and the long reader refuses it rather than mis-assigning its buckets.
    try testing.expect(parse(rc1_body, .long) == null);

    // A legacy table writes rc1 and never an `s` line: the serial row is the
    // adaptive switch's, the switch is arch-gated, and an `s` line in an rc1
    // file would read as width `s` to an older build.
    var t = Table{ .layout = .legacy };
    for (0..MIN_SAMPLES) |_| _ = t.observe(2, 8192, 44.0, 2.7, true, false);
    // Even if something did fold one, it never reaches the file.
    for (0..MIN_SAMPLES) |_| _ = t.observeSerial(8192, 16.0, true, false);
    var buf: [1024]u8 = undefined;
    const text = try serialize(&buf, &t);
    try testing.expect(std.mem.startsWith(u8, text, "rc1\n"));
    try testing.expect(std.mem.indexOf(u8, text, "\ns ") == null);
    const rt = parse(text, .legacy) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 1), rt.restored);
    try testing.expectEqual(@as(u32, 0), rt.restored_serial);
    // A body claiming a bucket the legacy grid cannot produce is not an rc1
    // file, and neither is one carrying a serial row.
    try testing.expect(parse("rc1\n2 6 44.0 2.7 3\n", .legacy) == null);
    try testing.expect(parse("rc1\ns 3 16.0 1.0 3\n", .legacy) == null);
}

test "round_cost: the long layout warm-starts from a legacy file — no user boots cold (L27)" {
    // A store-version bump is not "one re-explore". `fromTable()` is the one
    // term deciding whether the EV plan prices WIDTHS and extension from
    // measurements or from the fitted prior, and the prior plans a different
    // width and an always-open extension valve. So the long grid reads the
    // legacy file it can honestly reuse.
    var legacy = Table{ .layout = .legacy };
    for (0..MIN_SAMPLES) |_| {
        _ = legacy.observe(2, 1000, 20.0, 2.0, true, false); // bucket 0
        _ = legacy.observe(3, 8192, 51.0, 3.2, true, false); // bucket 3
        _ = legacy.observe(4, 20000, 70.0, 4.0, true, false); // bucket 4
        _ = legacy.observe(3, 400_000, 900.0, 3.0, true, false); // bucket 5: `32k+`
    }
    try testing.expectEqual(@as(usize, 5), legacy.bucketOf(400_000));

    const lifted = migrateLegacy(legacy);
    try testing.expectEqual(Layout.long, lifted.layout);
    // Buckets 0..4 share their edges with the long grid, cell for cell.
    try testing.expectApproxEqAbs(20.0, lifted.measuredMs(2, 0).?, 1e-3);
    try testing.expectApproxEqAbs(2.0, lifted.measuredTok(2, 0).?, 1e-3);
    try testing.expectApproxEqAbs(51.0, lifted.measuredMs(3, 3).?, 1e-3);
    try testing.expectApproxEqAbs(70.0, lifted.measuredMs(4, 4).?, 1e-3);
    try testing.expectEqual(@as(u32, 3), lifted.restored);
    // The legacy `32k+` cell is an EMA of everything from 32k to 1M and the
    // long grid has three cells in that span, so it is DROPPED rather than
    // served to one of them — that mis-assignment is what the bump was for.
    for (5..N_BUCKETS) |b| {
        try testing.expect(lifted.measuredMs(3, b) == null);
        try testing.expect(!lifted.active(b));
    }
    // Restored cells are stale, never fresh: another boot is another thermal
    // and OS state, so the first live sample blends at RESEED_WEIGHT.
    try testing.expect(lifted.seq > RESEED_GAP);
    // A legacy file carries no serial row, so the long grid has none to lift.
    try testing.expectEqual(@as(u32, 0), lifted.restored_serial);
    try testing.expectEqual(@as(u32, 0), lifted.foldedSerialCells());

    // The short buckets — where a boot's first requests live — are warm, and
    // that is what `fromTable()` reads.
    try testing.expect(lifted.active(3));
    try testing.expectEqual(@as(?usize, 3), lifted.bucketToRead(8192));
}

test "layoutFor is THE round-cost layout resolver" {
    // audit addendum 3 (non-blocker). `scheduler.doLoadOnInferenceThread`
    // resolved the layout inline and `main.zig`'s offline `--prompt` path
    // never resolved it at all, so a qwen4_exp checkpoint served from `serve`
    // planned on the nine-bucket grid with a serial row while the SAME
    // checkpoint run offline planned on the six-bucket one.
    const Stub = struct {
        qwen4: bool,
        fn isQwen4(self: *const @This()) bool {
            return self.qwen4;
        }
    };
    const long = Stub{ .qwen4 = true };
    const legacy = Stub{ .qwen4 = false };
    try testing.expectEqual(Layout.long, layoutFor(&long));
    try testing.expectEqual(Layout.legacy, layoutFor(&legacy));
    // The layout decides the store version, which is the whole reason ONE
    // resolver matters: two answers = two files, and the second one reads none
    // of the first one's measurements.
    try testing.expect(storeVersion(layoutFor(&long)) != storeVersion(layoutFor(&legacy)));
}

test "bucketName: the legacy grid's top bucket is 32k+, not 32-64k" {
    // The legacy layout has SIX buckets and its last one is unbounded; the long
    // grid splits that span at 64k/128k/256k. `BUCKET_NAMES` alone spells the
    // long grid, so every `[spec-stats]`, `[mtp-trace]` and adaptive-switch
    // line on a sidecar-MTP pack (qwen3.5/3.6/3.8 — every arch but qwen4_exp)
    // labelled a 374k request "32-64k". The number was right and the label was
    // a lie, which is worse than no label: it is the string an operator reads
    // when deciding whether a bucket crossing should have re-planned.
    const t = std.testing;
    try t.expectEqual(@as(usize, 6), nBuckets(.legacy));
    try t.expectEqual(@as(usize, 9), nBuckets(.long));

    // The shared edges are shared, so the names agree on 0..4.
    var b: usize = 0;
    while (b + 1 < nBuckets(.legacy)) : (b += 1) {
        try t.expectEqualStrings(BUCKET_NAMES[b], bucketName(.legacy, b));
        try t.expectEqualStrings(BUCKET_NAMES[b], bucketName(.long, b));
    }
    // ...and disagree on exactly one cell: the legacy top bucket.
    try t.expectEqualStrings("32k+", bucketName(.legacy, 5));
    try t.expectEqualStrings("32-64k", bucketName(.long, 5));

    // The long grid's own top bucket keeps its name — the rule is "the last
    // bucket of a SHORTER grid is unbounded", not "index 5 is special".
    try t.expectEqualStrings("256k+", bucketName(.long, 8));

    // And the label matches where `bucketFor` actually puts a long request on
    // each grid: on legacy everything past 32k lands in bucket 5.
    try t.expectEqual(@as(usize, 5), bucketForLayout(374_000, .legacy));
    try t.expectEqualStrings("32k+", bucketName(.legacy, bucketForLayout(374_000, .legacy)));
}

test "bucketName: every label site asks the TABLE's layout, never the bare array" {
    // Class pin. A label is a claim about the cell's EDGES, so a site that
    // indexes `BUCKET_NAMES` directly is right on qwen4_exp and wrong on every
    // other arch — and it reads as correct in review, which is why this is a
    // scan and not a comment.
    const t = std.testing;
    const gen = @embedFile("generate.zig");
    try t.expectEqual(@as(usize, 0), std.mem.count(u8, gen, "round_cost.BUCKET_" ++ "NAMES["));
    try t.expect(std.mem.count(u8, gen, "round_cost.bucket" ++ "Name(self.xfm.round_cost.layout, ") >= 7);
}
