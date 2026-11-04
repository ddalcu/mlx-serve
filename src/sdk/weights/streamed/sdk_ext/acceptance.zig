//! The draft lanes' acceptance: one rule set, one sampled path and one seeded draw for every arch's draft lane
//! (DeepSeek-V4.1's DSpark lane; MiMo-V2.6's MTP lane next). A lane drafts, verifies [t1, drafts] on its target, and
//! hands this module the verify rows; it gets back how many drafts are accepted and the token after them. Arch state
//! (KV rollback, draft windows, trims) stays in the lane.
//!
//! The rules (`Rule`, chosen once at the lane's construction):
//!
//! - `exact`: draft x at row r is accepted iff x == argmax(logits_r). The control: every emitted token is the target's
//!   greedy token.
//! - `typical` (#475's rule, Medusa's typical acceptance): with p = softmax(logits_r) at temperature 1, unfiltered,
//!   and H(p) = -sum_i p_i ln p_i (nats),
//!
//!       accept x  <=>  p(x) > min(eps, delta * exp(-H(p)))        (strict; eps = 1.0 unless set)
//!
//!   delta scales the floor: a smaller delta lowers it, so acceptance never falls as delta falls (delta -> 0 accepts
//!   every draft with mass; delta 0.3 accepts at least what 0.5 does, row for row). exp(-H) is 1 on a one-hot row and
//!   1 / m on a flat row of m tokens, so the floor follows the row's spread: a confident row (low H) demands a high p(x),
//!   a flat row accepts any of its m tokens once delta < 1. The correction at the first rejected row, and the bonus
//!   after an accepted run, stay the row's argmax. Lossy by design (a draft the target ranks second can be kept).
//!
//! Requests (`Params`, the host's `sdk.SamplingParams`): a greedy request (temperature < 0.01 or top_k == 1) takes the
//! lane's `Rule` as above. A sampled request takes SAMPLE-THEN-VERIFY whatever the rule, over the target row tempered
//! (logits / T) and filtered (top_k, then top_p over the top_k-renormalised set, then min_p against the top
//! probability; ties at the kept set's smallest tempered logit are kept): p. Every row's token is p's draw by inverse
//! CDF at `drawV` (the request's one position-keyed stream, `drawAt` index 0), and draft x is accepted iff it IS that
//! token. Every emitted token is p's draw at its position, so a seeded request emits the same ids on every path: the
//! serial step (`sampler.zig`, the plugin's draw handed to the host), a lane's verify (drafts accepted or not), the
//! prompt's first token. It is exact (each token distributed as p) for any drafter; for an argmax drafter (a point mass
//! at x) a draft is accepted with probability p(x), as under min(1, p / q) speculative sampling. The draws are keyed
//! by (seed, absolute position of the drawn token, draw index), never by round: a seeded request is reproducible however
//! its rounds fall.
//!
//! The rule on every request (`acceptGeneral`, `typicalRows`; a lane opts in at construction by calling them: MiMo's
//! MTP lane does, DeepSeek-V4.1's DSpark lane keeps `accept` / `sampledRows`): `exact` is as above (greedy: argmax
//! equality; sampled: sample-then-verify). `typical` on a sampled request is Medusa's typical acceptance (Cai
//! et al. 2024, section 2.3.1) over the request's own target: p = `distribution` (tempered, then top_k, top_p, min_p,
//! the sampled path's transform, unchanged), draft x at row r accepted iff p_r(x) > min(eps, delta * exp(-H(p_r))) with
//! H over p_r's kept set; every row's token is p_r's plain draw at `drawV(seed, pos)` (inverse CDF): the first
//! rejected row emits its draw (the correction, which may equal the rejected draft: no residual, no resampling
//! adjustment), an accepted run the next row's draw (the bonus). `drawU` is not consumed. Lossy by design: the emitted
//! tokens are not distributed as p. A greedy request has no distribution to temper (the host's greedy token is the
//! argmax), so `typical` on it stays the greedy round above (T = 1 unfiltered test, argmax tokens): at a literal
//! T -> 0 every row is one-hot, the floor is delta, and typical would collapse to `exact` (Medusa's "reverts to greedy
//! decoding"). Medusa's first token is the target's greedy token; here it is the previous round's draw (t1), so a
//! sampled request samples every token the target emits.
//!
//! Two forms of the same rules: the device graphs (`typicalFlags`, `sampledRows`) a lane folds into its verify's eval
//! (written once against any op backend with the ops they call: `shapeOf`, `astype`, `slice`, `logsumexp`, `sub`,
//! `exp`, `neg`, `sum`, `mul`, `minimum`, `scalar`, `hostArray`, `takeAlongAxis`, `reshape`, `greater`, `sort`,
//! `softmax`, `arange`, `less`, `where`, `div`, `cumsum`, `logicalAnd`, `max`, `greaterEqual`, `equal`), and the host
//! reference (`accept`, `distribution`, `specStep`, `typicalAccepts`): the parity tests' oracle, rows of at most
//! `ref_max_vocab` tokens. `prefix` is the walk both share: from the rows' tokens (argmax or drawn) and the per-draft
//! decisions to the accepted count and the next token.

const std = @import("std");
const sdk = @import("sdk");

/// Rows one verify can hand the device graphs (drafts + the bonus row).
pub const max_rows = 17;
/// The host reference's row bound (its scratch lives on the stack).
pub const ref_max_vocab = 4096;

/// The typical rule's parameters: the host's (`sdk.acceptance.Mode.typical`: delta, eps = 1.0).
pub const Typical = @FieldType(sdk.acceptance.Mode, "typical");

/// How a lane accepts drafts, chosen once at its construction. Under `accept` it shapes greedy requests only (sampled
/// requests take exact speculative sampling under either); under `acceptGeneral` `typical` shapes every request.
pub const Rule = union(enum) {
    exact,
    typical: Typical,
};

/// A request's sampling as a lane applies it to the target's verify rows: the host's `sdk.SamplingParams`, field for
/// field.
pub const Params = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0,
    min_p: ?f32 = null,
    seed: u64 = 0,

    pub fn of(s: sdk.SamplingParams) Params {
        return .{ .temperature = s.temperature, .top_p = s.top_p, .top_k = s.top_k, .min_p = s.min_p, .seed = s.seed };
    }

    /// The host's greedy rule (`sdk.SamplingParams.greedy`): the lane's `Rule` path.
    pub fn greedy(s: Params) bool {
        return s.temperature < 0.01 or s.top_k == 1;
    }

    /// The sampled parameters, or null when the request is greedy (the `Rule` path).
    pub fn active(s: ?Params) ?Params {
        const x = s orelse return null;
        return if (x.greedy()) null else x;
    }

    /// Whether the filter sorts each row (top_k or top_p set).
    pub fn sorts(s: Params) bool {
        return s.top_k > 0 or s.top_p < 1.0;
    }
};

/// mlx-stream's one random stream: the uniform in (0, 1) of draw `index` at absolute token position `pos` of a request
/// seeded `seed`. splitmix64 over (seed ^ the index's key) + pos x the golden gamma, its top 24 bits, centred
/// ((bits + 0.5) / 2^24, never 0 or 1). Index 0 (`token_draw`) is the token's draw (every path samples it: serial,
/// verify, first token); index 1 (`test_draw`) the typical tier's spare. The same (seed, pos, index) is the same
/// uniform on every arch and path, however the rounds fall.
pub fn drawAt(seed: u64, pos: u64, index: u32) f32 {
    var z = (seed ^ draw_keys[index]) +% pos *% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    z ^= z >> 31;
    const bits: u32 = @intCast(z >> 40);
    return (@as(f32, @floatFromInt(bits)) + 0.5) / 16777216.0;
}

pub const token_draw: u32 = 0;
pub const test_draw: u32 = 1;
const draw_keys = [2]u64{ 0xD1B54A32D192ED03, 0 };

/// The token draw at `pos` (`drawAt` index 0).
pub fn drawV(seed: u64, pos: u64) f32 {
    return drawAt(seed, pos, token_draw);
}

/// The spare uniform at `pos` (`drawAt` index 1).
pub fn drawU(seed: u64, pos: u64) f32 {
    return drawAt(seed, pos, test_draw);
}

/// The reference sampler: the token at absolute position `pos` of a request seeded `seed`, drawn from `dist` (the
/// row's `distribution`) at its token draw. Every path's token is this one.
pub fn sample(dist: []const f64, seed: u64, pos: u64) u32 {
    return draw(dist, drawV(seed, pos));
}

/// The walk from a verify's rows to the round's outcome: `targets[r]` is row r's token (the argmax on the `Rule` path,
/// the drawn token on the sampled path: p's draw at the row's position), `decisions`
/// (one per draft; null = the exact rule: target == draft) the per-draft test. `targets` may be a chunk of the verify
/// (fewer rows than drafts + 1): `next` is null when every draft in it is accepted and it holds no further row.
pub const Prefix = struct { accepted: u32, next: ?u32 };

pub fn prefix(drafts: []const u32, targets: []const u32, decisions: ?[]const bool) Prefix {
    const n = @min(drafts.len, targets.len);
    var acc: u32 = 0;
    while (acc < n) : (acc += 1) {
        const ok = if (decisions) |d| d[acc] else targets[acc] == drafts[acc];
        if (!ok) return .{ .accepted = acc, .next = targets[acc] };
    }
    return .{ .accepted = acc, .next = if (targets.len > n) targets[n] else null };
}

/// One round's counters.
pub const RoundStats = struct {
    drafted: u32,
    accepted: u32,
    /// The next token is a correction (a draft was rejected), else the bonus.
    corrected: bool,
};

pub const Round = struct { n_accepted: u32, next_token: u32, stats: RoundStats };

/// The host reference of one round (the device graphs' oracle and the rules' statement): `logits` = the verify's
/// `drafts.len + 1` rows (row r predicts the token at `base` + r + 1), `draft_q` (null: an argmax drafter, point-mass
/// proposals) the drafts' distributions, one row per draft. `params` null or greedy: `rule`; sampled: exact speculative
/// sampling, seeded per position.
pub fn accept(rule: Rule, params: ?Params, logits: []const f32, drafts: []const u32, draft_q: ?[]const f64, base: u64) Round {
    const rows = drafts.len + 1;
    const vocab = logits.len / rows;
    std.debug.assert(vocab * rows == logits.len and vocab <= ref_max_vocab and rows <= max_rows);
    var targets: [max_rows]u32 = undefined;
    var decisions: [max_rows]bool = undefined;
    var d: [ref_max_vocab]f64 = undefined;
    const dist = d[0..vocab];
    var p: Prefix = undefined;
    if (Params.active(params)) |sm| {
        // Row by row to the first rejection: a row's draws are needed only once it is reached.
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            distribution(logits[r * vocab ..][0..vocab], sm, dist);
            const x: ?u32 = if (r < drafts.len) drafts[r] else null;
            const q: ?[]const f64 = if (x != null and draft_q != null) draft_q.?[r * vocab ..][0..vocab] else null;
            const s = specStep(dist, x, q, drawU(sm.seed, base + r + 1), drawV(sm.seed, base + r + 1));
            targets[r] = s.tok;
            if (x != null) decisions[r] = s.accept;
            if (x == null or !s.accept) break;
        }
        p = prefix(drafts[0..@min(r, drafts.len)], targets[0 .. r + 1], decisions[0..@min(r + 1, drafts.len)]);
    } else {
        for (targets[0..rows], 0..) |*t, r| t.* = argmaxRow(logits[r * vocab ..][0..vocab]);
        const typ: ?[]const bool = switch (rule) {
            .exact => null,
            .typical => |t| blk: {
                for (decisions[0..drafts.len], drafts, 0..) |*f, x, r| {
                    distribution(logits[r * vocab ..][0..vocab], .{ .temperature = 1.0 }, dist);
                    f.* = typicalAccepts(dist, x, t);
                }
                break :blk decisions[0..drafts.len];
            },
        };
        p = prefix(drafts, targets[0..rows], typ);
    }
    return roundOf(p, drafts.len);
}

fn roundOf(p: Prefix, drafted: usize) Round {
    const corrected = p.accepted < drafted;
    return .{ .n_accepted = p.accepted, .next_token = p.next.?, .stats = .{ .drafted = p.accepted + @intFromBool(corrected), .accepted = p.accepted, .corrected = corrected } };
}

/// The host reference of one round with the lane's `Rule` on every request (`typicalRows`' oracle; see the module
/// doc): `exact`, or a greedy request, is `accept`; `typical` on a sampled request is Medusa's typical acceptance over
/// the request's filtered target (`typicalStep` per row, seeded per position). `draft_q` is unused by `typical`.
pub fn acceptGeneral(rule: Rule, params: ?Params, logits: []const f32, drafts: []const u32, draft_q: ?[]const f64, base: u64) Round {
    const t = switch (rule) {
        .exact => return accept(rule, params, logits, drafts, draft_q, base),
        .typical => |t| t,
    };
    const sm = Params.active(params) orelse return accept(rule, null, logits, drafts, draft_q, base);
    const rows = drafts.len + 1;
    const vocab = logits.len / rows;
    std.debug.assert(vocab * rows == logits.len and vocab <= ref_max_vocab and rows <= max_rows);
    var targets: [max_rows]u32 = undefined;
    var decisions: [max_rows]bool = undefined;
    var d: [ref_max_vocab]f64 = undefined;
    const dist = d[0..vocab];
    // Row by row to the first rejection, as `accept`'s sampled path.
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        distribution(logits[r * vocab ..][0..vocab], sm, dist);
        const x: ?u32 = if (r < drafts.len) drafts[r] else null;
        const s = typicalStep(dist, x, t, drawV(sm.seed, base + r + 1));
        targets[r] = s.tok;
        if (x != null) decisions[r] = s.accept;
        if (x == null or !s.accept) break;
    }
    return roundOf(prefix(drafts[0..@min(r, drafts.len)], targets[0 .. r + 1], decisions[0..@min(r + 1, drafts.len)]), drafts.len);
}

/// The first index of the row's largest logit (the device argmax's tie rule).
fn argmaxRow(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = @intCast(i);
    };
    return best;
}

/// The typical floor min(eps, delta x exp(-H(dist))) (the host's `typicalThreshold`).
pub fn typicalFloor(dist: []const f64, t: Typical) f64 {
    return sdk.acceptance.typicalThreshold(dist, t.delta, t.eps);
}

/// The typical test on a row's distribution: p(draft) > the floor.
pub fn typicalAccepts(dist: []const f64, draft: u32, t: Typical) bool {
    return dist[draft] > typicalFloor(dist, t);
}

/// The host reference of a sampled row: `out` (the row's length, at most `ref_max_vocab`) gets the tempered, filtered
/// distribution (0 off the kept set), in f64 over the f32 tempered logits.
pub fn distribution(logits: []const f32, s: Params, out: []f64) void {
    const n = logits.len;
    const inv_t: f32 = 1.0 / s.temperature;
    var lt_buf: [ref_max_vocab]f32 = undefined;
    const lt = lt_buf[0..n];
    for (lt, logits) |*d, v| d.* = v * inv_t;
    var thr: f32 = -std.math.inf(f32);
    if (s.sorts()) {
        var srt_buf: [ref_max_vocab]f32 = undefined;
        const srt = srt_buf[0..n];
        @memcpy(srt, lt);
        std.mem.sort(f32, srt, {}, std.sort.desc(f32));
        const smax: f64 = srt[0];
        var sz: f64 = 0;
        for (srt) |v| sz += @exp(@as(f64, v) - smax);
        const kk: usize = if (s.top_k > 0) @min(s.top_k, n) else n;
        var zk: f64 = 0;
        for (srt[0..kk]) |v| zk += @exp(@as(f64, v) - smax) / sz;
        var kept: usize = 0;
        var cum: f64 = 0;
        for (srt[0..kk]) |v| {
            const p = @exp(@as(f64, v) - smax) / sz / zk;
            if (s.top_p < 1.0 and cum >= s.top_p) break;
            cum += p;
            kept += 1;
        }
        thr = srt[kept - 1];
    }
    var mx: f64 = -std.math.inf(f64);
    for (lt) |v| if (v >= thr) {
        mx = @max(mx, v);
    };
    var z: f64 = 0;
    for (lt) |v| if (v >= thr) {
        z += @exp(@as(f64, v) - mx);
    };
    for (out, lt) |*o, v| o.* = if (v >= thr) @exp(@as(f64, v) - mx) / z else 0;
    if (s.min_p) |mp| {
        var pmax: f64 = 0;
        for (out) |p| pmax = @max(pmax, p);
        var z2: f64 = 0;
        for (out) |*p| {
            if (p.* < mp * pmax) p.* = 0;
            z2 += p.*;
        }
        for (out) |*p| p.* /= z2;
    }
}

/// The reference draw: the first token whose cumulative probability reaches `u` x the total.
pub fn draw(dist: []const f64, u: f32) u32 {
    var total: f64 = 0;
    for (dist) |p| total += p;
    var cum: f64 = 0;
    for (dist, 0..) |p, i| {
        cum += p;
        if (cum >= @as(f64, u) * total and p > 0) return @intCast(i);
    }
    return @intCast(dist.len - 1);
}

/// The reference step on one row of a sampled request (sample-then-verify): the row's token is p's draw at v; a draft
/// is accepted iff it is that token. `q` and `u` are not read (any drafter: the emitted token is p's draw).
pub fn specStep(dist: []const f64, draft: ?u32, q: ?[]const f64, u: f32, v: f32) struct { accept: bool, tok: u32 } {
    _ = .{ q, u };
    const tok = draw(dist, v);
    const x = draft orelse return .{ .accept = false, .tok = tok };
    return .{ .accept = x == tok, .tok = tok };
}

/// The reference typical step on one row of a sampled request (`acceptGeneral`): the row's token is p's draw at `v`;
/// with a draft x, accept iff p(x) > the floor (`typicalAccepts`).
pub fn typicalStep(dist: []const f64, draft: ?u32, t: Typical, v: f32) struct { accept: bool, tok: u32 } {
    const tok = draw(dist, v);
    const x = draft orelse return .{ .accept = false, .tok = tok };
    return .{ .accept = typicalAccepts(dist, x, t), .tok = tok };
}

/// `typicalFlags`'s graph over the verify's rows `row2` [W, V] (any float dtype): the typical test on the first
/// `drafts.len` rows (each against its draft), in f32 on the device: bool [drafts.len]. drafts.len >= 1.
pub fn typicalFlags(g: anytype, row2: anytype, drafts: []const u32, t: Typical) !@TypeOf(row2) {
    const vocab = g.shapeOf(row2).dim(-1);
    const drafted: c_int = @intCast(drafts.len);
    const lg = try g.astype(try g.slice(row2, &.{ 0, 0 }, &.{ drafted, vocab }, &.{ 1, 1 }), .float32);
    return typicalTest(g, lg, null, drafts, t);
}

/// The typical test's graph on f32 rows `lg` [drafts.len, V], the logits of the distribution it reads (p =
/// softmax(lg)): -inf off `keep` when one is given (a filtered row), where the entropy's p ln p terms are 0 (not 0 x
/// -inf). Without `keep` the ops are `typicalFlags`' of before (the greedy tier's graph, unchanged).
fn typicalTest(g: anytype, lg: anytype, keep: ?@TypeOf(lg), drafts: []const u32, t: Typical) !@TypeOf(lg) {
    const drafted: c_int = @intCast(drafts.len);
    const log_p = try g.sub(lg, try g.logsumexp(lg, -1, true));
    const p = try g.exp(log_p);
    const plogp = try g.mul(p, log_p);
    const terms = if (keep) |k| try g.where(k, plogp, try g.scalar(0, .float32)) else plogp;
    const entropy = try g.neg(try g.sum(terms, -1, false));
    const floor = try g.minimum(try g.scalar(t.eps, .float32), try g.mul(try g.scalar(t.delta, .float32), try g.exp(try g.neg(entropy))));
    var idb: [max_rows]i32 = undefined;
    for (idb[0..drafts.len], drafts) |*d, v| d.* = @intCast(v);
    const ids = try g.hostArray(std.mem.sliceAsBytes(idb[0..drafts.len]), &.{ drafted, 1 }, .int32);
    return g.greater(try g.reshape(try g.takeAlongAxis(p, ids, -1), &.{-1}), floor);
}

/// A sampled verify's arrays: `tok` uint32 [W] (the rows' tokens, as the argmax rows), `accept` bool [drafts] (null
/// without drafts), `p` the rows' filtered distribution [W, V].
pub fn SampledRows(comptime T: type) type {
    return struct { tok: T, accept: ?T, p: T };
}

/// A sampled request's filtered rows: `lf` the tempered logits (x 1 / T, f32), -inf off the keep set; `keep` the keep
/// mask (null: no filter set); `p` = softmax(lf), the target distribution every sampled path reads; `zero` the f32 0
/// scalar the graph built (reused by the caller's masks, so `sampledRows`' op trace is the one it had before the split).
pub fn Filtered(comptime T: type) type {
    return struct { lf: T, keep: ?T, p: T, zero: T };
}

/// The sampled paths' transform (the host reference `distribution`'s graph, one implementation for `sampledRows` and
/// `typicalRows`): the logits [W, V] tempered (x 1 / T, f32); top_k / top_p (when set: one descending sort per row,
/// the kept set's smallest tempered logit as its threshold, ties kept) and min_p (against the row's top probability)
/// as a keep mask; the kept set's probabilities p.
pub fn filteredRows(g: anytype, row2: anytype, sm: Params) !Filtered(@TypeOf(row2)) {
    const sh = g.shapeOf(row2);
    const vocab = sh.dim(1);
    const lt = try g.mul(try g.astype(row2, .float32), try g.scalar(1.0 / sm.temperature, .float32));
    const ninf = try g.scalar(-std.math.inf(f64), .float32);
    const zero = try g.scalar(0, .float32);
    var keep: ?@TypeOf(row2) = null;
    if (sm.sorts()) {
        const srt = try g.neg(try g.sort(try g.neg(lt), -1));
        const ps = try g.softmax(srt, -1);
        const rank = try g.reshape(try g.arange(0, @floatFromInt(vocab), 1, .int32), &.{ 1, vocab });
        var kk: ?@TypeOf(row2) = if (sm.top_k > 0) try g.less(rank, try g.scalar(@floatFromInt(sm.top_k), .int32)) else null;
        if (sm.top_p < 1.0) {
            var pk = ps;
            if (kk) |m| {
                pk = try g.where(m, ps, zero);
                pk = try g.div(pk, try g.sum(pk, -1, true));
            }
            const before = try g.sub(try g.cumsum(pk, -1), pk);
            const kp = try g.less(before, try g.scalar(sm.top_p, .float32));
            kk = if (kk) |m| try g.logicalAnd(m, kp) else kp;
        }
        const count = try g.sum(try g.astype(kk.?, .int32), -1, true);
        const thr = try g.takeAlongAxis(srt, try g.sub(count, try g.scalar(1, .int32)), -1);
        keep = try g.greaterEqual(lt, thr);
    }
    if (sm.min_p) |mp| {
        const base_l = if (keep) |k| try g.where(k, lt, ninf) else lt;
        const pm = try g.softmax(base_l, -1);
        const km = try g.greaterEqual(pm, try g.mul(try g.max(pm, -1, true), try g.scalar(mp, .float32)));
        keep = if (keep) |k| try g.logicalAnd(k, km) else km;
    }
    const lf = if (keep) |k| try g.where(k, lt, ninf) else lt;
    return .{ .lf = lf, .keep = keep, .p = try g.softmax(lf, -1), .zero = zero };
}

/// The draws' uniforms for `w` rows: v = `drawV(seed, base + row + 1)` (the drawn token's absolute position), f32 [w, 1].
fn drawUniforms(g: anytype, w: c_int, sm: Params, base: u64) !@TypeOf(g.*).T {
    const nw: usize = @intCast(w);
    var vb: [max_rows]f32 = undefined;
    for (vb[0..nw], 0..) |*v, i| v.* = drawV(sm.seed, base + i + 1);
    return g.hostArray(std.mem.sliceAsBytes(vb[0..nw]), &.{ w, 1 }, .float32);
}

/// Each row's draw from `src` [W, V] (unnormalised mass) at its uniform `vv` [W, 1] (`drawUniforms`): the first token
/// whose cumulative mass reaches v x the row's total: uint32 [W].
fn drawRows(g: anytype, src: anytype, vv: @TypeOf(src)) !@TypeOf(src) {
    const sh = g.shapeOf(src);
    const w = sh.dim(0);
    const vocab = sh.dim(1);
    const cum = try g.cumsum(src, -1);
    const total = try g.slice(cum, &.{ 0, vocab - 1 }, &.{ w, vocab }, &.{ 1, 1 });
    return g.astype(try g.sum(try g.astype(try g.less(cum, try g.mul(vv, total)), .int32), -1, false), .uint32);
}

/// A sampled request's rows (sample-then-verify; the host's reference `distribution` / `specStep`): p = `filteredRows`;
/// every row's token is p's draw (`drawRows`, v = `drawV(seed, pos)`, pos = `base` + row + 1, the drawn token's
/// absolute position): the token the serial step draws there. `accept` (bool [drafts.len], null without drafts) = the
/// row's draft is its token. One host array of W floats and one of the drafts.
pub fn sampledRows(g: anytype, row2: anytype, sm: Params, base: u64, drafts: []const u32) !SampledRows(@TypeOf(row2)) {
    const w = g.shapeOf(row2).dim(0);
    const f = try filteredRows(g, row2, sm);
    const tok = try drawRows(g, f.p, try drawUniforms(g, w, sm, base));
    var accept_: ?@TypeOf(row2) = null;
    if (drafts.len > 0) {
        const nd: c_int = @intCast(drafts.len);
        const xs = try g.hostArray(std.mem.sliceAsBytes(drafts), &.{nd}, .uint32);
        accept_ = try g.equal(try g.slice(tok, &.{0}, &.{nd}, &.{1}), xs);
    }
    return .{ .tok = tok, .accept = accept_, .p = f.p };
}

/// A sampled request's rows under the typical rule (`acceptGeneral`'s graph; Medusa's typical acceptance): p =
/// `filteredRows` (the same transform as `sampledRows`). `accept` (bool [drafts.len], null without drafts) = p(x) >
/// min(eps, delta x exp(-H(p))) on the first `drafts.len` rows (`typicalTest` on the filtered logits, H over the kept
/// set); every row's token is p's own draw (`drawRows`, v = `drawV(seed, base + row + 1)`): the correction at a
/// rejected row is a plain draw from the target, no residual. `drawU` is not consumed.
pub fn typicalRows(g: anytype, row2: anytype, sm: Params, base: u64, drafts: []const u32, t: Typical) !SampledRows(@TypeOf(row2)) {
    const f = try filteredRows(g, row2, sm);
    const vv = try drawUniforms(g, g.shapeOf(row2).dim(0), sm, base);
    var accept_: ?@TypeOf(row2) = null;
    if (drafts.len > 0) {
        const vocab = g.shapeOf(row2).dim(1);
        const nd: c_int = @intCast(drafts.len);
        const lf = try g.slice(f.lf, &.{ 0, 0 }, &.{ nd, vocab }, &.{ 1, 1 });
        const keep = if (f.keep) |k| try g.slice(k, &.{ 0, 0 }, &.{ nd, vocab }, &.{ 1, 1 }) else null;
        accept_ = try typicalTest(g, lf, keep, drafts, t);
    }
    return .{ .tok = try drawRows(g, f.p, vv), .accept = accept_, .p = f.p };
}

const testing = std.testing;

/// `rows` x `v` seeded logits, each row with a leader; drafts alternate leader / runner / random.
fn testRows(seed: u64, comptime rows: usize, comptime v: usize, lg: *[rows * v]f32, drafts: *[rows - 1]u32, spread: f32) void {
    var rng: std.Random.DefaultPrng = .init(seed);
    for (lg) |*x| x.* = rng.random().floatNorm(f32) * spread;
    for (0..rows) |r| {
        const lead = rng.random().uintLessThan(u32, @as(u32, v));
        lg[r * v + lead] += 1.0 + 7.0 * rng.random().float(f32);
        if (r < rows - 1) drafts[r] = switch (rng.random().uintLessThan(u32, 4)) {
            0, 1 => lead,
            2 => (lead + 1) % @as(u32, v),
            else => rng.random().uintLessThan(u32, @as(u32, v)),
        };
    }
}

test "sdk_ext acceptance: Params is the host's SamplingParams field for field, greedy where the host's is" {
    for ([_]sdk.SamplingParams{ .{}, .{ .temperature = 0.005 }, .{ .temperature = 0.7, .top_p = 0.9, .top_k = 40, .min_p = 0.05, .seed = 123 }, .{ .temperature = 1.2, .top_k = 1, .seed = 7 } }) |sp| {
        const s = Params.of(sp);
        try testing.expectEqual(sp.greedy(), s.greedy());
        try testing.expectEqual(sp.greedy(), Params.active(s) == null);
        try testing.expect(s.temperature == sp.temperature and s.top_p == sp.top_p and s.top_k == sp.top_k and s.seed == sp.seed and std.meta.eql(s.min_p, sp.min_p));
    }
}

test "sdk_ext acceptance: exact accepts the leading run of argmax agreements; the next token is the argmax after it" {
    const lg = [_]f32{
        0, 5, 1, 0, // argmax 1
        3, 0, 0, 0, // argmax 0
        0, 0, 0, 2, // argmax 3
        0, 0, 9, 0, // argmax 2 (the bonus row)
    };
    const all = accept(.exact, null, &lg, &.{ 1, 0, 3 }, null, 0);
    try testing.expectEqual(@as(u32, 3), all.n_accepted);
    try testing.expectEqual(@as(u32, 2), all.next_token);
    try testing.expect(!all.stats.corrected and all.stats.drafted == 3);
    const one = accept(.exact, null, &lg, &.{ 1, 2, 3 }, null, 0);
    try testing.expectEqual(@as(u32, 1), one.n_accepted);
    try testing.expectEqual(@as(u32, 0), one.next_token);
    try testing.expect(one.stats.corrected and one.stats.drafted == 2);
    // A greedy request (T < 0.01, top_k 1) is the rule's path, and ignores q.
    const q: [3 * 4]f64 = @splat(0.25);
    try testing.expectEqual(one, accept(.exact, .{ .temperature = 0.005, .seed = 3 }, &lg, &.{ 1, 2, 3 }, &q, 0));
    try testing.expectEqual(one, accept(.exact, .{ .temperature = 0.9, .top_k = 1, .seed = 3 }, &lg, &.{ 1, 2, 3 }, null, 0));
    // The walk on a chunk without its bonus row: every draft in it accepted, no next token yet.
    try testing.expectEqual(Prefix{ .accepted = 2, .next = null }, prefix(&.{ 1, 0, 3 }, &.{ 1, 0 }, null));
}

test "sdk_ext acceptance: typical accepts p(x) > min(eps, delta exp(-H)); the correction stays the argmax" {
    // A flat pair: H = ln 2, floor = delta / 2. Either token passes at delta 0.5 (0.5 > 0.25), and the floor caps at eps.
    const p = [_]f64{ 0.5, 0.5, 0.0 };
    try testing.expectApproxEqAbs(@as(f64, 0.25), typicalFloor(&p, .{ .delta = 0.5 }), 1e-12);
    try testing.expect(typicalAccepts(&p, 1, .{ .delta = 0.5 }));
    try testing.expect(!typicalAccepts(&p, 1, .{ .delta = 1.0, .eps = 0.5 }));
    try testing.expect(!typicalAccepts(&p, 2, .{ .delta = 0.5 }));
    // Row 0's argmax is 0, but its runner-up 1 clears the floor: typical keeps it, exact does not.
    const lg = [_]f32{ 1.0, 0.9, -4.0, 0.0, 0.0, 9.0 };
    const typ = accept(.{ .typical = .{ .delta = 0.5 } }, null, &lg, &.{1}, null, 0);
    try testing.expectEqual(@as(u32, 1), typ.n_accepted);
    try testing.expectEqual(@as(u32, 2), typ.next_token);
    const ex = accept(.exact, null, &lg, &.{1}, null, 0);
    try testing.expectEqual(@as(u32, 0), ex.n_accepted);
    try testing.expectEqual(@as(u32, 0), ex.next_token);
    // Rejected under typical: the correction is the argmax, not a draw.
    const rej = accept(.{ .typical = .{ .delta = 0.5 } }, null, &lg, &.{2}, null, 0);
    try testing.expectEqual(@as(u32, 0), rej.n_accepted);
    try testing.expectEqual(@as(u32, 0), rej.next_token);
}

test "sdk_ext acceptance: typical acceptance never falls as delta falls (row for row); delta -> 0 accepts every draft" {
    const R = 7;
    const V = 96;
    var lg: [R * V]f32 = undefined;
    var drafts: [R - 1]u32 = undefined;
    const deltas = [_]f32{ 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 1e-20 };
    var totals: [deltas.len]u32 = @splat(0);
    for (0..64) |s| {
        testRows(7000 + s, R, V, &lg, &drafts, 0.5 + 0.25 * @as(f32, @floatFromInt(s % 16)));
        var prev: u32 = 0;
        for (deltas, 0..) |dl, i| {
            const n = accept(.{ .typical = .{ .delta = dl } }, null, &lg, &drafts, null, 0).n_accepted;
            try testing.expect(n >= prev);
            prev = n;
            totals[i] += n;
        }
        try testing.expectEqual(@as(u32, R - 1), prev);
    }
    // The fixture separates the served deltas: 0.3 accepts strictly more than 0.5 over the 64 sets.
    try testing.expect(totals[3] > totals[2]);
}

test "sdk_ext acceptance: seeded rounds reproduce; the draws are keyed by position, not by round" {
    const R = 5;
    const V = 64;
    var lg: [R * V]f32 = undefined;
    var drafts: [R - 1]u32 = undefined;
    testRows(31, R, V, &lg, &drafts, 1.0);
    const sm: Params = .{ .temperature = 0.8, .top_p = 0.95, .seed = 1234 };
    for ([_]Rule{ .exact, .{ .typical = .{ .delta = 0.3 } } }) |rule| {
        const a = accept(rule, sm, &lg, &drafts, null, 500);
        try testing.expectEqual(a, accept(rule, sm, &lg, &drafts, null, 500));
        // A sampled request ignores the rule.
        try testing.expectEqual(a, accept(.exact, sm, &lg, &drafts, null, 500));
    }
    // Some seed in eight changes the round (draws differ by seed).
    const a0 = accept(.exact, sm, &lg, &drafts, null, 500);
    var differs = false;
    for (1..9) |k| differs = differs or !std.meta.eql(a0, accept(.exact, .{ .temperature = 0.8, .top_p = 0.95, .seed = 1234 + k }, &lg, &drafts, null, 500));
    try testing.expect(differs);
    // Position keying: the round's first row draws at base + 1 whatever round it falls in.
    var d: [V]f64 = undefined;
    distribution(lg[0..V], sm, &d);
    const first = specStep(&d, drafts[0], null, drawU(sm.seed, 501), drawV(sm.seed, 501));
    const r = accept(.exact, sm, &lg, &drafts, null, 500);
    if (!first.accept) {
        try testing.expectEqual(@as(u32, 0), r.n_accepted);
        try testing.expectEqual(first.tok, r.next_token);
    } else try testing.expect(r.n_accepted >= 1);
    try testing.expectEqual(drawU(42, 1000), drawU(42, 1000));
    try testing.expect(drawU(42, 1000) != drawU(42, 1001) and drawU(42, 1000) != drawU(43, 1000) and drawU(42, 1000) != drawV(42, 1000));
    var lo: f32 = 1;
    var hi: f32 = 0;
    for (0..10000) |i| {
        const u = drawU(7, i);
        lo = @min(lo, u);
        hi = @max(hi, u);
    }
    try testing.expect(lo > 0 and hi < 1 and lo < 0.01 and hi > 0.99);
}

test "sdk_ext acceptance: the reference filter keeps top_k, then top_p over the renormalised set, then min_p" {
    const lg = [_]f32{ 2.0, 1.0, 0.5, 0.0, -1.0 };
    var d: [5]f64 = undefined;
    distribution(&lg, .{ .temperature = 1.0, .top_k = 2, .seed = 0 }, &d);
    try testing.expect(d[0] > 0 and d[1] > 0 and d[2] == 0 and d[3] == 0 and d[4] == 0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), d[0] + d[1], 1e-12);
    // top_p 0.5: the top token alone is 0.53 of the mass.
    distribution(&lg, .{ .temperature = 1.0, .top_p = 0.5, .seed = 0 }, &d);
    try testing.expect(d[0] == 1.0 and d[1] == 0);
    // min_p 0.4: tokens under 0.4 x the top's probability are dropped (e^-1 = 0.37 for the second).
    distribution(&lg, .{ .temperature = 1.0, .min_p = 0.4, .seed = 0 }, &d);
    try testing.expect(d[0] > 0 and d[1] == 0);
    // T = 1, no filter: the softmax; every draw lands on a token with mass.
    distribution(&lg, .{ .temperature = 1.0, .seed = 0 }, &d);
    for (0..1000) |i| try testing.expect(d[draw(&d, drawU(3, i))] > 0);
}

test "sdk_ext acceptance: speculative sampling emits tokens distributed as the tempered, filtered target, point-mass or sampled drafts (20k rounds)" {
    // A 6-token row, 20,000 seeded rounds at T 0.7 / 1.0 / 1.6, top_p 0.9: the emitted token (the accepted draft or the
    // residual's draw) has the target's frequencies within 5 binomial standard deviations, and tokens the filter drops
    // are never emitted. Drafts: a fixed point mass (0 or 2), or drawn from a draft distribution q (q given).
    const lg = [_]f32{ 1.2, 0.9, 0.4, 0.1, -0.6, -1.5 };
    const q = [_]f64{ 0.1, 0.4, 0.2, 0.1, 0.1, 0.1 };
    const n = 20000;
    for ([_]f32{ 0.7, 1.0, 1.6 }) |t| for ([_]?u32{ 0, 2, null }) |fixed| {
        const sm: Params = .{ .temperature = t, .top_p = 0.9, .seed = 314 };
        var p: [6]f64 = undefined;
        distribution(&lg, sm, &p);
        var count: [6]u32 = @splat(0);
        for (0..n) |round| {
            const pos: u64 = 1000 + round;
            const r = if (fixed) |x|
                specStep(&p, x, null, drawU(sm.seed, pos), drawV(sm.seed, pos))
            else
                specStep(&p, draw(&q, drawU(sm.seed ^ 0x5eed, pos)), &q, drawU(sm.seed, pos), drawV(sm.seed, pos));
            count[r.tok] += 1;
        }
        for (p, count) |pi, ci| {
            const sd = @sqrt(@as(f64, n) * pi * (1 - pi));
            const diff = @abs(@as(f64, @floatFromInt(ci)) - @as(f64, n) * pi);
            try testing.expect(diff <= 5 * sd + 1e-9);
            if (pi == 0) try testing.expectEqual(@as(u32, 0), ci);
        }
    };
}

test "sdk_ext acceptance: a point-mass draft the target disfavours is accepted less (acceptance = p(draft))" {
    const lg = [_]f32{ 2.0, 1.0, 0.5, 0.0, -1.0, -2.0 };
    const sm: Params = .{ .temperature = 1.0, .seed = 5 };
    var p: [6]f64 = undefined;
    distribution(&lg, sm, &p);
    var rates: [6]f64 = undefined;
    for (&rates, 0..) |*rate, x| {
        var acc: u32 = 0;
        for (0..20000) |round| acc += @intFromBool(specStep(&p, @intCast(x), null, drawU(sm.seed, round), drawV(sm.seed, round)).accept);
        rate.* = @as(f64, @floatFromInt(acc)) / 20000.0;
        try testing.expectApproxEqAbs(p[x], rate.*, 0.015);
    }
    for (rates[0..5], rates[1..]) |a, b| try testing.expect(b < a);
}

test "sdk_ext acceptance: temperature lowers the draft's probability; the typical rule's acceptance does not fall (8239d75)" {
    // 64 rows of a 512-token vocab with one clear leader per row; two drafters: one proposes the row's leader, one its
    // runner-up. The typical test at delta 0.3 on the tempered row: the floor falls with the entropy as fast as the
    // leader's probability does (a flat row of m tokens accepts any of them at 0.3 / m), so the rule's acceptance rises
    // with T while the draft's own probability falls at every step.
    var rng: std.Random.DefaultPrng = .init(20261007);
    const V = 512;
    var rows: [64][V]f32 = undefined;
    var lead: [64]u32 = undefined;
    var second: [64]u32 = undefined;
    for (&rows, &lead, &second) |*row, *ld, *sc| {
        for (row) |*v| v.* = rng.random().floatNorm(f32);
        ld.* = rng.random().uintLessThan(u32, V);
        sc.* = (ld.* + 1 + rng.random().uintLessThan(u32, V - 1)) % V;
        row[ld.*] += 6.0 + 2.0 * rng.random().float(f32);
        row[sc.*] += 4.0 + 1.0 * rng.random().float(f32);
    }
    const temps = [_]f32{ 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0 };
    var rate_lead: [temps.len]f64 = undefined;
    var rate_second: [temps.len]f64 = undefined;
    var mass_lead: [temps.len]f64 = undefined;
    var d: [V]f64 = undefined;
    for (temps, 0..) |t, ti| {
        var al: f64 = 0;
        var as: f64 = 0;
        var ml: f64 = 0;
        for (&rows, lead, second) |*row, ld, sc| {
            distribution(row, .{ .temperature = t, .seed = 0 }, &d);
            al += @floatFromInt(@intFromBool(typicalAccepts(&d, ld, .{ .delta = 0.3 })));
            as += @floatFromInt(@intFromBool(typicalAccepts(&d, sc, .{ .delta = 0.3 })));
            ml += d[ld];
        }
        rate_lead[ti] = al / 64.0;
        rate_second[ti] = as / 64.0;
        mass_lead[ti] = ml / 64.0;
    }
    for (rate_lead[0 .. temps.len - 1], rate_lead[1..]) |x, y| try testing.expect(y >= x);
    try testing.expectEqual(@as(f64, 1.0), rate_lead[temps.len - 1]);
    for (rate_second[0 .. temps.len - 1], rate_second[1..]) |x, y| try testing.expect(y >= x);
    try testing.expect(rate_second[temps.len - 1] > rate_second[0]);
    for (mass_lead[0 .. temps.len - 1], mass_lead[1..]) |x, y| try testing.expect(y < x);
}

test "sdk_ext acceptance: typical on a sampled request reads the tempered, top_p 0.95 target: a flat row accepts any of its m tokens once delta < 1, a one-hot row only its token" {
    const V = 16;
    const sm: Params = .{ .temperature = 1.0, .top_p = 0.95, .seed = 9 };
    // Flat over m = 4 tokens (the rest ~e^-40 each, dropped by top_p): p = 1/4 on the four, H = ln 4, floor = delta / 4.
    var flat: [2 * V]f32 = @splat(-40.0);
    for (0..4) |i| flat[i] = 0.0;
    for (0..4) |i| flat[V + i] = 0.0;
    var d: [V]f64 = undefined;
    distribution(flat[0..V], sm, &d);
    for (d, 0..) |p, i| try testing.expectEqual(@as(f64, if (i < 4) 0.25 else 0.0), p);
    for ([_]f32{ 0.9, 0.5, 0.3, 0.2 }) |dl| {
        try testing.expectApproxEqAbs(@as(f64, dl) / 4.0, typicalFloor(&d, .{ .delta = dl }), 1e-6);
        for (0..4) |x| try testing.expectEqual(@as(u32, 1), acceptGeneral(.{ .typical = .{ .delta = dl } }, sm, &flat, &.{@intCast(x)}, null, 40).n_accepted);
        // A token the filter dropped never clears the floor.
        try testing.expectEqual(@as(u32, 0), acceptGeneral(.{ .typical = .{ .delta = dl } }, sm, &flat, &.{5}, null, 40).n_accepted);
    }
    // The floor caps at eps: delta 1 with eps 0.2 accepts (0.25 > 0.2).
    try testing.expectEqual(@as(u32, 1), acceptGeneral(.{ .typical = .{ .delta = 1.0, .eps = 0.2 } }, sm, &flat, &.{3}, null, 40).n_accepted);
    // One-hot: top_p keeps token 7 alone (p = 1, H = 0, floor = delta): 7 accepted at any delta < 1, every other token
    // rejected, and the correction is 7 at every seed and position.
    var hot: [2 * V]f32 = @splat(-40.0);
    hot[7] = 0.0;
    hot[V + 7] = 0.0;
    distribution(hot[0..V], sm, &d);
    try testing.expectEqual(@as(f64, 1.0), d[7]);
    for ([_]f32{ 0.9, 0.3, 0.05 }) |dl| {
        const rule: Rule = .{ .typical = .{ .delta = dl } };
        try testing.expectEqual(@as(u32, 1), acceptGeneral(rule, sm, &hot, &.{7}, null, 3).n_accepted);
        for (0..V) |x| if (x != 7) for (0..16) |pos| {
            const r = acceptGeneral(rule, .{ .temperature = 1.0, .top_p = 0.95, .seed = pos * 31 }, &hot, &.{@intCast(x)}, null, pos);
            try testing.expectEqual(@as(u32, 0), r.n_accepted);
            try testing.expectEqual(@as(u32, 7), r.next_token);
            try testing.expect(r.stats.corrected and r.stats.drafted == 1);
        };
    }
}

test "sdk_ext acceptance: typical on a sampled request draws the correction and the bonus from the target itself, no residual (20k seeded rounds)" {
    // Two identical rows (T 1.0, top_p 0.95). A draft the floor rejects at delta 0.9: the correction is p's draw (the
    // rejected draft itself included, at p(x)); the leader, accepted: the bonus is the next row's p draw. Frequencies
    // within 5 binomial standard deviations; dropped tokens never emitted.
    const V = 8;
    const row = [V]f32{ 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0 };
    const lg = row ++ row;
    const sm0: Params = .{ .temperature = 1.0, .top_p = 0.95, .seed = 271 };
    const t: Typical = .{ .delta = 0.9 };
    var p: [V]f64 = undefined;
    distribution(&row, sm0, &p);
    try testing.expect(p[7] == 0 and p[6] == 0 and p[5] > 0);
    // The smallest kept token: below the floor.
    var x: u32 = 0;
    for (p, 0..) |pi, i| if (pi > 0 and pi < p[x]) {
        x = @intCast(i);
    };
    try testing.expect(!typicalAccepts(&p, x, t) and typicalAccepts(&p, 0, t));
    const n = 20000;
    for ([_]u32{ x, 0 }) |draft| {
        var count: [V]u32 = @splat(0);
        for (0..n) |round| {
            const r = acceptGeneral(.{ .typical = t }, sm0, &lg, &.{draft}, null, 5000 + 3 * round);
            try testing.expectEqual(@as(u32, @intFromBool(draft == 0)), r.n_accepted);
            count[r.next_token] += 1;
        }
        for (p, count) |pi, ci| {
            const sd = @sqrt(@as(f64, n) * pi * (1 - pi));
            const diff = @abs(@as(f64, @floatFromInt(ci)) - @as(f64, n) * pi);
            try testing.expect(diff <= 5 * sd + 1e-9);
            if (pi == 0) try testing.expectEqual(@as(u32, 0), ci);
        }
        // No resampling adjustment: a rejected draft is still emitted as its own correction at about p(x).
        if (draft == x) try testing.expect(count[x] > 0);
    }
}

test "sdk_ext acceptance: acceptGeneral is accept on greedy requests (either rule) and on exact sampled requests; typical sampled differs, rises as delta falls, reproduces by seed" {
    const R = 5;
    const V = 64;
    var lg: [R * V]f32 = undefined;
    var drafts: [R - 1]u32 = undefined;
    const greedy = [_]?Params{ null, .{ .temperature = 0.005, .seed = 3 }, .{ .temperature = 0.9, .top_k = 1, .seed = 3 } };
    const sampled = [_]Params{ .{ .temperature = 1.0, .top_p = 0.95, .seed = 1 }, .{ .temperature = 0.7, .top_k = 20, .top_p = 0.9, .min_p = 0.02, .seed = 78 } };
    const deltas = [_]f32{ 0.8, 0.6, 0.5, 0.4, 0.3, 0.2 };
    var differs = false;
    var totals: [deltas.len]u32 = @splat(0);
    for (0..64) |set| {
        testRows(9100 + set, R, V, &lg, &drafts, 0.5 + 0.25 * @as(f32, @floatFromInt(set % 16)));
        for ([_]Rule{ .exact, .{ .typical = .{ .delta = 0.3 } }, .{ .typical = .{ .delta = 0.5 } } }) |rule| for (greedy) |req| {
            try testing.expectEqual(accept(rule, req, &lg, &drafts, null, 77), acceptGeneral(rule, req, &lg, &drafts, null, 77));
        };
        for (sampled) |sm| {
            try testing.expectEqual(accept(.exact, sm, &lg, &drafts, null, 77 + set), acceptGeneral(.exact, sm, &lg, &drafts, null, 77 + set));
            const typ = acceptGeneral(.{ .typical = .{ .delta = 0.3 } }, sm, &lg, &drafts, null, 77 + set);
            try testing.expectEqual(typ, acceptGeneral(.{ .typical = .{ .delta = 0.3 } }, sm, &lg, &drafts, null, 77 + set));
            differs = differs or !std.meta.eql(typ, accept(.{ .typical = .{ .delta = 0.3 } }, sm, &lg, &drafts, null, 77 + set));
            var prev: u32 = 0;
            for (deltas, 0..) |dl, i| {
                const r = acceptGeneral(.{ .typical = .{ .delta = dl } }, sm, &lg, &drafts, null, 77 + set);
                try testing.expect(r.n_accepted >= prev);
                try testing.expectEqual(r.n_accepted + @intFromBool(r.stats.corrected), r.stats.drafted);
                prev = r.n_accepted;
                totals[i] += r.n_accepted;
            }
        }
    }
    try testing.expect(differs);
    try testing.expect(totals[deltas.len - 1] > totals[0]);
}

/// The lane's loop on the host reference over a seeded toy target: `model(prev, pos)` is row logits for the token at
/// `pos` after `prev`; the drafter chains a perturbed argmax. Rounds at most `cap` drafts; `n` tokens.
fn simLane(model_seed: u64, rule: Rule, sm: Params, cap: u32, n: usize, out: []u32) void {
    const V = 64;
    const Model = struct {
        fn row(ms: u64, prev: u32, pos: u64, salt: u64, lg: *[V]f32) void {
            var rng: std.Random.DefaultPrng = .init(ms ^ (@as(u64, prev) *% 0x9E3779B97F4A7C15) ^ (pos *% 0xD1B54A32D192ED03) ^ salt);
            for (lg) |*x| x.* = rng.random().floatNorm(f32);
            lg[rng.random().uintLessThan(u32, V)] += 3.0;
        }
    };
    var emitted: usize = 0;
    var t1: u32 = 0;
    var pos: u64 = 100;
    while (emitted < n) {
        const k: u32 = @intCast(@min(cap, n - emitted - 1));
        var ids: [max_rows]u32 = undefined;
        ids[0] = t1;
        for (1..k + 1) |i| {
            var dl: [V]f32 = undefined;
            Model.row(model_seed, ids[i - 1], pos + i, 0x5eed, &dl);
            var tl: [V]f32 = undefined;
            Model.row(model_seed, ids[i - 1], pos + i, 0, &tl);
            for (&dl, tl) |*a, b| a.* = b + 0.6 * a.*;
            ids[i] = argmaxRow(&dl);
        }
        var lg: [(max_rows) * V]f32 = undefined;
        for (0..k + 1) |r| Model.row(model_seed, ids[r], pos + r + 1, 0, lg[r * V ..][0..V]);
        const r = acceptGeneral(rule, sm, lg[0 .. (k + 1) * V], ids[1 .. k + 1], null, pos);
        for (ids[0 .. r.n_accepted + 1]) |tok| {
            out[emitted] = tok;
            emitted += 1;
        }
        t1 = r.next_token;
        pos += r.n_accepted + 1;
    }
}

test "sdk_ext acceptance: a seeded sampled run of the lane loop reproduces itself (fixture battery: 8 targets x card / filtered sampling x typical 0.2..0.8 and exact)" {
    const n = 160;
    var a: [n]u32 = undefined;
    var b: [n]u32 = undefined;
    var c: [n]u32 = undefined;
    var seed_differs: u32 = 0;
    for (0..8) |ms| for ([_]Params{ .{ .temperature = 1.0, .top_p = 0.95, .seed = 1 }, .{ .temperature = 0.7, .top_k = 20, .top_p = 0.9, .seed = 42 } }) |sm| {
        for ([_]Rule{ .exact, .{ .typical = .{ .delta = 0.2 } }, .{ .typical = .{ .delta = 0.5 } }, .{ .typical = .{ .delta = 0.8 } } }) |rule| {
            simLane(ms, rule, sm, 3, n, &a);
            simLane(ms, rule, sm, 3, n, &b);
            try testing.expectEqualSlices(u32, &a, &b);
            var other = sm;
            other.seed +%= 1;
            simLane(ms, rule, other, 3, n, &c);
            seed_differs += @intFromBool(!std.mem.eql(u32, &a, &c));
        }
    };
    // Draws are seeded: another seed changes the run (all 64 here).
    try testing.expectEqual(@as(u32, 64), seed_differs);
}
