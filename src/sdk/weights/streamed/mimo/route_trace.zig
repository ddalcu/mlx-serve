//! `mimo-route-trace-v2` (design-hit-rate §4a; v2 adds the MTP lane's M-row decode calls): one request's per-token
//! per-layer top-8 routes for the prompt and decode, the plan decode returned for each call, the residency at the phase
//! change and at the end, and the next layer's predictor top-16. Armed only in profile builds
//! (`MIMO_ROUTE_TRACE=<prefix>`, read once at construction): every hook sits behind `profile.enabled`, so a served build
//! has no field, branch or call for it. The type itself is plain so CPU tests exercise it in any build.
//!
//! A decode call is one routed forward of M rows (M = 1: a serial step; M = 1 + drafts <= `max_call_rows`: the MTP
//! verify of one round). Files (little endian), for request k: `<prefix>-r<k>.json` (the header) and
//!   `.prompt.u8`       [layers][prompt_tokens][8]  ids as `seedPrefill` received them (router order);
//!   `.decode.u8`       per call c in order: [layers][M_c][8]  the call's ids per row (router order); with every
//!                      M_c = 1 this is v1's [decode_calls][layers][8], byte for byte;
//!   `.calls.u8`        [decode_calls]              M_c (v2);
//!   `.accepted.u8`     [decode_calls]              the round's accepted drafts for an MTP verify call, 0xFF for a
//!                      call that is no round's verify (a serial step) (v2);
//!   `.plan.u8`         [decode_calls][layers][3]   n_hits, n_misses, n_persistent of the returned plan (the union's);
//!   `.resident0.u16`   [layers][max(rows)]          slot_to_expert right after the phase change (0xFFFF empty);
//!   `.resident_end.u16`[layers][max(rows)]          slot_to_expert when the decode phase ends;
//!   `.pred.bin`        [decode_calls][layers-1] x (16 u8 ids + 16 f32 scores): the top-16 of the scores layer L's
//!                      call carried for layer L+1 (score descending, ties by id; 0xFF / NaN when the call had none;
//!                      an M-row call: each expert's max over the rows).
const std = @import("std");
const env = @import("mlx_host").env;
const kset = @import("../sdk_ext/kernel_set.zig");
const expert_policy = @import("../sdk_ext.zig").expert.policy;

pub const format = "mimo-route-trace-v2";
pub const top_k: u32 = 8;
pub const pred_k: u32 = 16;
pub const pred_bytes: u32 = predBytes(u8);
/// The decode calls a trace keeps (4 x the 1,024-token cell); later calls set `overflow`.
pub const max_decode_calls: u32 = 4096;
/// The widest decode call a trace keeps (the MTP verify: t1 + 3 drafts); a wider decode-phase call is counted in
/// `wide_decode_calls`, not kept.
pub const max_call_rows: u32 = 4;
/// `.accepted.u8` of a call that is no round's verify.
pub const not_a_round: u8 = 0xFF;

pub const Io = struct { hits: u64 = 0, misses: u64 = 0, bytes: u64 = 0, records: u64 = 0 };

/// MiMo's trace: u8 expert ids (at most 256 experts), `mimo-route-trace-v2`'s files.
pub const Trace = TraceOf(u8, format);

/// The trace over expert ids of type `Id`: u8 (at most 256 experts: the v2 files above, byte for byte) or u16 (at most
/// 512 experts: the same layout with every expert id two bytes little endian, `.prompt.u16` and `.decode.u16`, and the
/// predictor's top-16 ids u16 before their f32 scores; GLM-5.3-Flash's 288 experts), named `format_name` in the header.
pub fn TraceOf(comptime Id: type, comptime format_name: []const u8) type {
    if (Id != u8 and Id != u16) @compileError("route trace: expert ids are u8 or u16");
    return struct {
        const Self = @This();
        pub const trace_format = format_name;
        /// The id files' extension (`.prompt.<ext>`, `.decode.<ext>`).
        pub const id_ext = if (Id == u8) "u8" else "u16";
        pub const max_experts: u32 = if (Id == u8) 256 else 512;
        /// One call's predictor record: `pred_k` ids then `pred_k` f32 scores.
        pub const pred_len: u32 = predBytes(Id);
        const no_id: Id = std.math.maxInt(Id);
        a: std.mem.Allocator,
        io: std.Io,
        prefix: []const u8,
        layers: u32,
        experts: u32,
        first_moe_layer: u32,
        record_bytes: u64,
        request: u32 = 0,
        active: bool = false,
        // One request's buffers (sized at `begin`).
        prompt_tokens: u32 = 0,
        decode_cap: u32 = 0,
        prompt: []Id = &.{},
        prompt_ids: []u32 = &.{},
        decode: []Id = &.{},
        plan: []u8 = &.{},
        pred: []u8 = &.{},
        decode_n: []u32 = &.{},
        /// v2: per call its rows (M), its first row's index in `.decode.u8`'s row sequence, its round's accepted drafts.
        call_rows: []u8 = &.{},
        row_off: []u32 = &.{},
        accepted: []u8 = &.{},
        calls_seen: u32 = 0,
        row_cap: u32 = 0,
        rounds: u32 = 0,
        accepted_total: u64 = 0,
        /// A layer saw a call another layer saw with a different M (or before the first layer): the replay refuses it.
        row_mismatch: bool = false,
        rows0: []u32 = &.{},
        rows_end: []u32 = &.{},
        resident0: []u16 = &.{},
        resident_end: []u16 = &.{},
        decode_ids: std.ArrayList(u32) = .empty,
        prompt_sha256: [64]u8 = @splat('0'),
        // Integrity flags the replay checks before use.
        wide_decode_calls: u32 = 0,
        overflow: bool = false,
        extended_prompt: bool = false,
        io_prefill: Io = .{},
        io_decode: Io = .{},

        pub fn init(a: std.mem.Allocator, io: std.Io, prefix: []const u8, layers: u32, experts: u32, first_moe_layer: u32, record_bytes: u64) !Self {
            if (experts == 0 or experts > max_experts or layers < 2) return error.MimoRouteTraceShape;
            return .{ .a = a, .io = io, .prefix = try a.dupe(u8, prefix), .layers = layers, .experts = experts, .first_moe_layer = first_moe_layer, .record_bytes = record_bytes };
        }

        pub fn deinit(t: *Self) void {
            t.freeRequest();
            t.a.free(t.prefix);
            t.* = undefined;
        }

        fn freeRequest(t: *Self) void {
            const a = t.a;
            inline for (.{ "prompt", "decode", "plan", "pred", "call_rows", "accepted" }) |f| {
                a.free(@field(t, f));
                @field(t, f) = &.{};
            }
            inline for (.{ "prompt_ids", "decode_n", "rows0", "rows_end", "row_off" }) |f| {
                a.free(@field(t, f));
                @field(t, f) = &.{};
            }
            a.free(t.resident0);
            a.free(t.resident_end);
            t.resident0 = &.{};
            t.resident_end = &.{};
            t.decode_ids.deinit(a);
            t.decode_ids = .empty;
            t.active = false;
        }

        /// A request's prompt (`ids`) begins: buffers for its prompt and up to `max_decode` decode calls.
        pub fn begin(t: *Self, ids: []const u32, max_decode: u64) !void {
            t.freeRequest();
            const a = t.a;
            const L: usize = t.layers;
            const n: usize = ids.len;
            t.prompt_tokens = @intCast(n);
            t.decode_cap = @intCast(@min(max_decode, max_decode_calls));
            const T: usize = t.decode_cap;
            t.row_cap = t.decode_cap * max_call_rows;
            t.prompt = try a.alloc(Id, L * n * top_k);
            t.prompt_ids = try a.alloc(u32, L);
            t.decode = try a.alloc(Id, @as(usize, t.row_cap) * L * top_k);
            t.call_rows = try a.alloc(u8, T);
            t.row_off = try a.alloc(u32, T);
            t.accepted = try a.alloc(u8, T);
            t.plan = try a.alloc(u8, T * L * 3);
            t.pred = try a.alloc(u8, T * (L - 1) * pred_len);
            t.decode_n = try a.alloc(u32, L);
            t.rows0 = try a.alloc(u32, L);
            t.rows_end = try a.alloc(u32, L);
            t.resident0 = try a.alloc(u16, L * t.experts);
            t.resident_end = try a.alloc(u16, L * t.experts);
            @memset(t.prompt, no_id);
            @memset(t.prompt_ids, 0);
            @memset(t.decode, no_id);
            @memset(t.call_rows, 0);
            @memset(t.row_off, 0);
            @memset(t.accepted, not_a_round);
            t.calls_seen = 0;
            t.rounds = 0;
            t.accepted_total = 0;
            t.row_mismatch = false;
            @memset(t.plan, 0);
            for (0..T * (L - 1)) |i| emptyPredOf(Id, t.pred[i * pred_len ..][0..pred_len]);
            @memset(t.decode_n, 0);
            @memset(t.rows0, 0);
            @memset(t.rows_end, 0);
            @memset(t.resident0, expert_policy.no_expert);
            @memset(t.resident_end, expert_policy.no_expert);
            t.prompt_sha256 = kset.shaHex(std.mem.sliceAsBytes(ids));
            t.wide_decode_calls = 0;
            t.overflow = false;
            t.extended_prompt = false;
            t.io_prefill = .{};
            t.io_decode = .{};
            t.active = true;
        }

        /// One prompt call's ids of routed layer `rel` (as `seedPrefill` received them).
        pub fn seed(t: *Self, rel: u32, ids: []const u16) void {
            if (!t.active) return;
            const per_layer: usize = @as(usize, t.prompt_tokens) * top_k;
            const at: usize = t.prompt_ids[rel];
            const n = @min(ids.len, per_layer - @min(at, per_layer));
            if (n < ids.len) t.overflow = true;
            for (ids[0..n], t.prompt[rel * per_layer + at ..][0..n]) |e, *out| out.* = @intCast(e);
            t.prompt_ids[rel] += @intCast(ids.len);
        }

        /// One decode call of routed layer `rel`: its ids (M rows x 8, router order), the plan the stream returned, and the
        /// next layer's scores (M rows x experts).
        pub fn decodeRoute(t: *Self, rel: u32, ids: []const u16, plan: *const expert_policy.Plan, scores: []const f32) void {
            if (!t.active) return;
            const m: usize = ids.len / top_k;
            if (ids.len % top_k != 0 or m == 0 or m > max_call_rows) {
                t.wide_decode_calls += 1;
                return;
            }
            const L: usize = t.layers;
            const i: usize = t.decode_n[rel];
            t.decode_n[rel] += 1;
            if (i >= t.decode_cap) {
                t.overflow = true;
                return;
            }
            if (i == t.calls_seen) {
                // The call's first layer: its rows and its place in the row sequence.
                const off: u32 = if (i == 0) 0 else t.row_off[i - 1] + t.call_rows[i - 1];
                if (off + m > t.row_cap) {
                    t.overflow = true;
                    return;
                }
                t.row_off[i] = off;
                t.call_rows[i] = @intCast(m);
                t.calls_seen += 1;
            } else if (i > t.calls_seen or t.call_rows[i] != m) {
                t.row_mismatch = true;
                return;
            }
            const at = (@as(usize, t.row_off[i]) * L + rel * m) * top_k;
            for (ids, t.decode[at..][0..ids.len]) |e, *out| out.* = @intCast(e);
            const p = t.plan[(i * L + rel) * 3 ..][0..3];
            p.* = .{ @intCast(plan.n_hits), @intCast(plan.n_misses), @intCast(plan.n_persistent) };
            if (rel + 1 < L and scores.len >= m * t.experts) {
                const out = t.pred[(i * (L - 1) + rel) * pred_len ..][0..pred_len];
                if (m == 1) return topScoresOf(Id, scores[0..t.experts], out);
                var best: [max_experts]f32 = undefined;
                const b = best[0..t.experts];
                @memcpy(b, scores[0..t.experts]);
                for (1..m) |r| for (b, scores[r * t.experts ..][0..t.experts]) |*x, y| {
                    if (std.math.isNan(x.*) or y > x.*) x.* = y;
                };
                topScoresOf(Id, b, out);
            }
        }

        /// The MTP round whose verify was the last decode call: its accepted drafts (`.accepted.u8`).
        pub fn roundAccepted(t: *Self, accepted: u32) void {
            if (!t.active or t.calls_seen == 0) return;
            t.accepted[t.calls_seen - 1] = @intCast(@min(accepted, not_a_round - 1));
            t.rounds += 1;
            t.accepted_total += accepted;
        }

        /// `<prefix>-r<k>` of the request in progress (owned by the caller).
        pub fn basePath(t: *const Self, a: std.mem.Allocator) ![]u8 {
            return std.fmt.allocPrint(a, "{s}-r{d}", .{ t.prefix, t.request });
        }

        /// One decode input token (the host's next id): its sequence identifies the trace.
        pub fn decodeToken(t: *Self, id: u32) void {
            if (!t.active) return;
            t.decode_ids.append(t.a, id) catch {
                t.overflow = true;
            };
        }

        /// Routed layer `rel`'s persistent slots (`slot_to_expert[0..capacity]`) at the phase change (`end` false) or end.
        pub fn snapshot(t: *Self, end: bool, rel: u32, slots: []const u16) void {
            if (!t.active) return;
            const dst = if (end) t.resident_end else t.resident0;
            const rows = if (end) t.rows_end else t.rows0;
            const n = @min(slots.len, t.experts);
            rows[rel] = @intCast(n);
            @memcpy(dst[rel * t.experts ..][0..n], slots[0..n]);
        }

        /// Writes request k's files and the header; the trace stays armed for the next request.
        pub fn write(t: *Self) !void {
            if (!t.active) return;
            const io = t.io;
            defer {
                t.request += 1;
                t.active = false;
            }
            const a = t.a;
            const L: usize = t.layers;
            var calls: u32 = t.decode_n[0];
            var ragged = false;
            for (t.decode_n) |c| {
                if (c != t.decode_n[0]) ragged = true;
                calls = @min(calls, c);
            }
            calls = @min(@min(calls, t.decode_cap), t.calls_seen);
            const decode_rows: usize = if (calls == 0) 0 else @as(usize, t.row_off[calls - 1]) + t.call_rows[calls - 1];
            var width: usize = 0;
            for (t.rows0, t.rows_end) |r0, r1| width = @max(width, @max(r0, r1));
            const res0 = try packRows(a, t.resident0, t.experts, L, width);
            defer a.free(res0);
            const res1 = try packRows(a, t.resident_end, t.experts, L, width);
            defer a.free(res1);
            const decode_sha = kset.shaHex(std.mem.sliceAsBytes(t.decode_ids.items));
            const base = try t.basePath(a);
            defer a.free(base);
            const files = .{
                .{ ".prompt." ++ id_ext, std.mem.sliceAsBytes(t.prompt) },
                .{ ".decode." ++ id_ext, std.mem.sliceAsBytes(t.decode[0 .. decode_rows * L * top_k]) },
                .{ ".calls.u8", t.call_rows[0..calls] },
                .{ ".accepted.u8", t.accepted[0..calls] },
                .{ ".plan.u8", t.plan[0 .. @as(usize, calls) * L * 3] },
                .{ ".resident0.u16", std.mem.sliceAsBytes(res0) },
                .{ ".resident_end.u16", std.mem.sliceAsBytes(res1) },
                .{ ".pred.bin", t.pred[0 .. @as(usize, calls) * (L - 1) * pred_len] },
            };
            inline for (files) |f| {
                const path = try std.fmt.allocPrint(a, "{s}{s}", .{ base, f[0] });
                defer a.free(path);
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = f[1] });
            }
            const header = .{
                .format = trace_format,
                .layers = t.layers,
                .first_moe_layer = t.first_moe_layer,
                .experts = t.experts,
                .top_k = top_k,
                .rows = t.rows0,
                .rows_end = t.rows_end,
                .resident_width = width,
                .record_bytes = t.record_bytes,
                .prompt_tokens = t.prompt_tokens,
                .prompt_ids_per_layer = t.prompt_ids,
                .decode_calls = calls,
                .decode_calls_per_layer = t.decode_n,
                .max_call_rows = max_call_rows,
                .decode_rows = decode_rows,
                .rounds = t.rounds,
                .accepted_total = t.accepted_total,
                .row_mismatch = t.row_mismatch,
                .pred_k = pred_k,
                .wide_decode_calls = t.wide_decode_calls,
                .overflow = t.overflow,
                .ragged = ragged,
                .extended_prompt = t.extended_prompt,
                .mimo_io_prefill = t.io_prefill,
                .mimo_io_decode = t.io_decode,
                .identity = .{ .prompt_ids_sha256 = @as([]const u8, &t.prompt_sha256), .decode_input_ids_sha256 = @as([]const u8, &decode_sha), .decode_input_ids = t.decode_ids.items },
            };
            const json = try std.json.Stringify.valueAlloc(a, header, .{});
            defer a.free(json);
            const path = try std.fmt.allocPrint(a, "{s}.json", .{base});
            defer a.free(path);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
        }
    };
}

fn packRows(a: std.mem.Allocator, src: []const u16, stride: usize, layers: usize, width: usize) ![]u16 {
    const out = try a.alloc(u16, layers * width);
    for (0..layers) |l| @memcpy(out[l * width ..][0..width], src[l * stride ..][0..width]);
    return out;
}

/// The predictor bytes per call for ids of type `Id`: `pred_k` ids then `pred_k` f32 scores.
pub fn predBytes(comptime Id: type) u32 {
    return pred_k * (@sizeOf(Id) + 4);
}

fn emptyPredOf(comptime Id: type, out: []u8) void {
    const w = @sizeOf(Id);
    @memset(out[0 .. pred_k * w], 0xFF);
    for (0..pred_k) |j| std.mem.writeInt(u32, out[pred_k * w + 4 * j ..][0..4], @bitCast(std.math.nan(f32)), .little);
}

/// The top `pred_k` of `scores` (descending, ties by lower id) as u8 ids then little-endian f32 scores (MiMo's layout).
fn topScores(scores: []const f32, out: []u8) void {
    topScoresOf(u8, scores, out);
}

/// The top `pred_k` of `scores` (descending, ties by lower id) as `Id` ids (little endian) then little-endian f32 scores.
fn topScoresOf(comptime Id: type, scores: []const f32, out: []u8) void {
    var ids: [pred_k]u16 = undefined;
    var n: usize = 0;
    for (scores, 0..) |s, e| {
        if (std.math.isNan(s)) continue;
        // Insert e into the sorted prefix (strictly better = higher score; equal keeps the earlier, lower id).
        var j: usize = n;
        while (j > 0 and s > scores[ids[j - 1]]) : (j -= 1) {}
        if (j >= pred_k) continue;
        var k: usize = @min(n, pred_k - 1);
        while (k > j) : (k -= 1) ids[k] = ids[k - 1];
        ids[j] = @intCast(e);
        n = @min(n + 1, pred_k);
    }
    emptyPredOf(Id, out);
    const w = @sizeOf(Id);
    for (ids[0..n], 0..) |e, j| {
        std.mem.writeInt(Id, out[w * j ..][0..w], @intCast(e), .little);
        std.mem.writeInt(u32, out[pred_k * w + 4 * j ..][0..4], @bitCast(scores[e]), .little);
    }
}

test "mimo route trace: records, plans, residency and predictor top-16 round-trip through the files" {
    const testing = std.testing;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const prefix = try std.fmt.allocPrint(a, "{s}/t", .{dir});
    defer a.free(prefix);
    var t = try Trace.init(a, testing.io, prefix, 3, 64, 1, 100);
    defer t.deinit();
    // Two prompt tokens, three decode calls (one beyond a decode-phase call wider than `max_call_rows`: counted, not kept).
    try t.begin(&.{ 5, 6 }, 3);
    for (0..3) |l| t.seed(@intCast(l), &.{ 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, @intCast(17 + l) });
    var policy = try expert_policy.LayerPolicy.init(a, 64, 4);
    defer policy.deinit(a);
    for (0..3) |l| t.snapshot(false, @intCast(l), &.{ 1, 2, expert_policy.no_expert, 3 });
    var scores: [64]f32 = undefined;
    for (&scores, 0..) |*s, e| s.* = @floatFromInt((e * 37) % 64);
    scores[9] = scores[8]; // a tie: the lower id ranks first
    var plans: [3][3]expert_policy.Plan = undefined;
    for (0..3) |i| {
        t.decodeToken(@intCast(100 + i));
        for (0..3) |l| {
            var ids: [8]u16 = undefined;
            for (&ids, 0..) |*e, j| e.* = @intCast((i * 3 + l * 5 + j * 7) % 64);
            policy.plan(&ids, .decode, &plans[i][l]);
            t.decodeRoute(@intCast(l), &ids, &plans[i][l], &scores);
        }
    }
    const wide: [(max_call_rows + 1) * top_k]u16 = @splat(0);
    t.decodeRoute(0, &wide, &plans[0][0], &.{});
    for (0..3) |l| t.snapshot(true, @intCast(l), policy.slot_to_expert[0..policy.capacity]);
    t.io_decode = .{ .hits = 1, .misses = 2, .bytes = 200, .records = 2 };
    try t.write();
    try testing.expect(!t.active and t.request == 1);

    const read = struct {
        fn file(al: std.mem.Allocator, p: []const u8, ext: []const u8) ![]u8 {
            const path = try std.fmt.allocPrint(al, "{s}-r0{s}", .{ p, ext });
            defer al.free(path);
            return std.Io.Dir.cwd().readFileAlloc(testing.io, path, al, .limited(1 << 20));
        }
    };
    const prompt = try read.file(a, prefix, ".prompt.u8");
    defer a.free(prompt);
    try testing.expectEqual(@as(usize, 3 * 2 * 8), prompt.len);
    try testing.expectEqual(@as(u8, 19), prompt[2 * 16 + 15]);
    const decode = try read.file(a, prefix, ".decode.u8");
    defer a.free(decode);
    const plan = try read.file(a, prefix, ".plan.u8");
    defer a.free(plan);
    try testing.expectEqual(@as(usize, 3 * 3 * 8), decode.len);
    for (0..3) |i| for (0..3) |l| {
        for (0..8) |j| try testing.expectEqual(@as(u8, @intCast((i * 3 + l * 5 + j * 7) % 64)), decode[(i * 3 + l) * 8 + j]);
        const p = plans[i][l];
        try testing.expectEqualSlices(u8, &.{ @intCast(p.n_hits), @intCast(p.n_misses), @intCast(p.n_persistent) }, plan[(i * 3 + l) * 3 ..][0..3]);
    };
    const r0 = try read.file(a, prefix, ".resident0.u16");
    defer a.free(r0);
    try testing.expectEqual(@as(usize, 3 * 4 * 2), r0.len);
    try testing.expectEqual(@as(u16, 0xFFFF), std.mem.readInt(u16, r0[4..6], .little));
    const r1 = try read.file(a, prefix, ".resident_end.u16");
    defer a.free(r1);
    for (0..4) |s| try testing.expectEqual(policy.slot_to_expert[s], std.mem.readInt(u16, r1[2 * (8 + s) ..][0..2], .little));
    const pred = try read.file(a, prefix, ".pred.bin");
    defer a.free(pred);
    try testing.expectEqual(@as(usize, 3 * 2 * pred_bytes), pred.len);
    // Scores (e * 37) % 64 descending: 63 at e = 19, then 62 at e = 38, ...; e = 8 and 9 tie at 40, outside the top-16 (the next test checks ties).
    try testing.expectEqual(@as(u8, 19), pred[0]);
    try testing.expectEqual(@as(u8, 38), pred[1]);
    var prev: f32 = std.math.inf(f32);
    for (0..pred_k) |j| {
        const s: f32 = @bitCast(std.mem.readInt(u32, pred[pred_k + 4 * j ..][0..4], .little));
        try testing.expect(s <= prev);
        prev = s;
    }
    // v2 beside v1: a serial trace's calls are all one row and no round's.
    const calls = try read.file(a, prefix, ".calls.u8");
    defer a.free(calls);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1 }, calls);
    const acc = try read.file(a, prefix, ".accepted.u8");
    defer a.free(acc);
    try testing.expectEqualSlices(u8, &.{ not_a_round, not_a_round, not_a_round }, acc);
    const header = try read.file(a, prefix, ".json");
    defer a.free(header);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings(format, o.get("format").?.string);
    try testing.expectEqual(@as(i64, 3), o.get("decode_calls").?.integer);
    try testing.expectEqual(@as(i64, 3), o.get("decode_rows").?.integer);
    try testing.expectEqual(@as(i64, 0), o.get("rounds").?.integer);
    try testing.expectEqual(@as(i64, 1), o.get("wide_decode_calls").?.integer);
    try testing.expectEqual(@as(i64, 2), o.get("prompt_tokens").?.integer);
    try testing.expectEqual(false, o.get("overflow").?.bool);
    try testing.expectEqual(@as(i64, 2), o.get("mimo_io_decode").?.object.get("misses").?.integer);
}

test "mimo route trace v2: MTP verify calls keep M rows per layer, the union plan, the rows' max scores and the round's accepted" {
    const testing = std.testing;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const prefix = try std.fmt.allocPrint(a, "{s}/m", .{dir});
    defer a.free(prefix);
    const L = 3;
    var t = try Trace.init(a, testing.io, prefix, L, 64, 1, 100);
    defer t.deinit();
    try t.begin(&.{ 1, 2, 3 }, 8);
    var policy = try expert_policy.LayerPolicy.init(a, 64, 6);
    defer policy.deinit(a);
    // Calls: a serial step (M = 1), a round with 3 drafts (M = 4, 2 accepted), a round with none (M = 1, 0 accepted),
    // a round with 1 draft (M = 2, 1 accepted).
    const ms = [_]usize{ 1, 4, 1, 2 };
    const acc = [_]?u32{ null, 2, 0, 1 };
    var scores: [4 * 64]f32 = @splat(0);
    scores[64 + 33] = 9; // row 1's best at 33: an M-row call's top-1 is the max over its rows
    var want_plan: [4][L]expert_policy.Plan = undefined;
    for (ms, 0..) |m, c| {
        for (0..L) |l| {
            var ids: [4 * 8]u16 = undefined;
            for (ids[0 .. m * 8], 0..) |*e, j| e.* = @intCast((c * 11 + l * 7 + j * 3) % 64);
            policy.plan(ids[0 .. m * 8], .decode, &want_plan[c][l]);
            t.decodeRoute(@intCast(l), ids[0 .. m * 8], &want_plan[c][l], scores[0 .. m * 64]);
        }
        if (acc[c]) |n| t.roundAccepted(n);
    }
    try t.write();
    const read = struct {
        fn file(al: std.mem.Allocator, p: []const u8, ext: []const u8) ![]u8 {
            const path = try std.fmt.allocPrint(al, "{s}-r0{s}", .{ p, ext });
            defer al.free(path);
            return std.Io.Dir.cwd().readFileAlloc(testing.io, path, al, .limited(1 << 20));
        }
    };
    const calls = try read.file(a, prefix, ".calls.u8");
    defer a.free(calls);
    try testing.expectEqualSlices(u8, &.{ 1, 4, 1, 2 }, calls);
    const accepted = try read.file(a, prefix, ".accepted.u8");
    defer a.free(accepted);
    try testing.expectEqualSlices(u8, &.{ not_a_round, 2, 0, 1 }, accepted);
    const decode = try read.file(a, prefix, ".decode.u8");
    defer a.free(decode);
    try testing.expectEqual(@as(usize, (1 + 4 + 1 + 2) * L * 8), decode.len);
    var off: usize = 0;
    for (ms, 0..) |m, c| {
        for (0..L) |l| for (0..m * 8) |j| {
            try testing.expectEqual(@as(u8, @intCast((c * 11 + l * 7 + j * 3) % 64)), decode[(off * L + l * m) * 8 + j]);
        };
        off += m;
    }
    const plan = try read.file(a, prefix, ".plan.u8");
    defer a.free(plan);
    for (0..4) |c| for (0..L) |l| {
        const p = want_plan[c][l];
        try testing.expectEqualSlices(u8, &.{ @intCast(p.n_hits), @intCast(p.n_misses), @intCast(p.n_persistent) }, plan[(c * L + l) * 3 ..][0..3]);
    };
    const pred = try read.file(a, prefix, ".pred.bin");
    defer a.free(pred);
    try testing.expectEqual(@as(u8, 0), pred[0]); // the serial call: row 0 alone, all zero -> ties by id
    try testing.expectEqual(@as(u8, 33), pred[(1 * (L - 1)) * pred_bytes]); // the M = 4 call: row 1's 33
    const header = try read.file(a, prefix, ".json");
    defer a.free(header);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqual(@as(i64, 4), o.get("decode_calls").?.integer);
    try testing.expectEqual(@as(i64, 8), o.get("decode_rows").?.integer);
    try testing.expectEqual(@as(i64, 3), o.get("rounds").?.integer);
    try testing.expectEqual(@as(i64, 3), o.get("accepted_total").?.integer);
    try testing.expectEqual(@as(i64, 0), o.get("wide_decode_calls").?.integer);
    try testing.expectEqual(false, o.get("row_mismatch").?.bool);

    // Row mismatch: layer 1 sees call 0 with two rows after layer 0 saw one.
    try t.begin(&.{1}, 4);
    var p0: expert_policy.Plan = undefined;
    const one: [8]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const two: [16]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    policy.plan(&one, .decode, &p0);
    t.decodeRoute(0, &one, &p0, &.{});
    t.decodeRoute(1, &two, &p0, &.{});
    try testing.expect(t.row_mismatch);
}

test "route trace u16 (glm5_next, 288 experts): ids past 255 round-trip through the .u16 files and the predictor's u16 ids" {
    const testing = std.testing;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const prefix = try std.fmt.allocPrint(a, "{s}/g", .{dir});
    defer a.free(prefix);
    const W = TraceOf(u16, "glm5_next-route-trace-v2");
    try testing.expectError(error.MimoRouteTraceShape, W.init(a, testing.io, prefix, 2, 513, 3, 100));
    try testing.expectError(error.MimoRouteTraceShape, Trace.init(a, testing.io, prefix, 2, 288, 3, 100));
    var t = try W.init(a, testing.io, prefix, 2, 288, 3, 100);
    defer t.deinit();
    try t.begin(&.{ 7, 8 }, 2);
    for (0..2) |l| t.seed(@intCast(l), &.{ 280, 281, 282, 283, 284, 285, 286, 287, 0, 1, 2, 3, 4, 5, 6, @intCast(256 + l) });
    var policy = try expert_policy.LayerPolicy.init(a, 288, 4);
    defer policy.deinit(a);
    var scores: [288]f32 = @splat(0);
    scores[270] = 5;
    scores[3] = 4;
    var plan: expert_policy.Plan = undefined;
    const ids: [8]u16 = .{ 287, 270, 3, 4, 5, 6, 7, 286 };
    for (0..2) |l| {
        policy.plan(&ids, .decode, &plan);
        t.decodeRoute(@intCast(l), &ids, &plan, &scores);
    }
    try t.write();
    const read = struct {
        fn file(al: std.mem.Allocator, pre: []const u8, ext: []const u8) ![]u8 {
            const path = try std.fmt.allocPrint(al, "{s}-r0{s}", .{ pre, ext });
            defer al.free(path);
            return std.Io.Dir.cwd().readFileAlloc(testing.io, path, al, .limited(1 << 20));
        }
    };
    const prompt = try read.file(a, prefix, ".prompt.u16");
    defer a.free(prompt);
    try testing.expectEqual(@as(usize, 2 * 2 * 8 * 2), prompt.len);
    try testing.expectEqual(@as(u16, 287), std.mem.readInt(u16, prompt[2 * 7 ..][0..2], .little));
    try testing.expectEqual(@as(u16, 257), std.mem.readInt(u16, prompt[2 * (16 + 15) ..][0..2], .little));
    const decode = try read.file(a, prefix, ".decode.u16");
    defer a.free(decode);
    try testing.expectEqual(@as(u16, 287), std.mem.readInt(u16, decode[0..2], .little));
    const pred = try read.file(a, prefix, ".pred.bin");
    defer a.free(pred);
    try testing.expectEqual(@as(usize, W.pred_len), pred.len);
    try testing.expectEqual(@as(u16, 270), std.mem.readInt(u16, pred[0..2], .little));
    try testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, pred[2..4], .little));
    const s0: f32 = @bitCast(std.mem.readInt(u32, pred[pred_k * 2 ..][0..4], .little));
    try testing.expectEqual(@as(f32, 5), s0);
    const header = try read.file(a, prefix, ".json");
    defer a.free(header);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("glm5_next-route-trace-v2", parsed.value.object.get("format").?.string);
    try testing.expectEqual(@as(i64, 288), parsed.value.object.get("experts").?.integer);
}

test "mimo route trace: top-16 breaks score ties by the lower id" {
    var scores: [32]f32 = @splat(1);
    scores[20] = 2;
    var out: [pred_bytes]u8 = undefined;
    topScores(&scores, &out);
    try std.testing.expectEqual(@as(u8, 20), out[0]);
    for (1..pred_k) |j| try std.testing.expectEqual(@as(u8, @intCast(j - 1)), out[j]);
}

// The replay's faithfulness fixtures (hit-rate/fixtures/mtp-policy-r0.* and, mimo/layer-rows, mtp-policy-split-r0.*):
// v2 traces whose plans come from the real `LayerPolicy` (the prompt's seed and prefill plans, the phase change's
// growth, then 240 decode calls of 1..4 rows over skewed routes). `MIMO_ROUTE_TRACE_FIXTURE=<prefix>` (uniform rows)
// and `MIMO_ROUTE_TRACE_FIXTURE_SPLIT=<prefix>` (per-layer rows: an `expert_rows_by_layer` split, every layer grown by
// the same rows) write them there (how the committed fixtures were made); otherwise into a temporary directory.
// hit-rate/test_replay.py replays them: the current-policy port must reproduce every plan at each layer's own rows.
fn writePolicyFixture(comptime env_name: []const u8, name: []const u8, prompt_rows: [3]u32, decode_rows: [3]u32) !void {
    const testing = std.testing;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const tmp_prefix = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(tmp_prefix);
    const raw = env.named(env_name);
    const prefix: []const u8 = if (raw) |p| std.mem.span(p) else tmp_prefix;
    const L = 3;
    const E = 64;
    const n_prompt = 48;
    const calls = 240;
    var t = try Trace.init(a, testing.io, prefix, L, E, 1, 1000);
    defer t.deinit();
    var prompt_ids: [n_prompt]u32 = undefined;
    for (&prompt_ids, 0..) |*x, i| x.* = @intCast(i);
    try t.begin(&prompt_ids, calls);
    var rng: std.Random.DefaultPrng = .init(0x6d74_7032);
    const r = rng.random();
    var policies: [L]expert_policy.LayerPolicy = undefined;
    var made: usize = 0;
    defer for (policies[0..made]) |*p| p.deinit(a);
    for (&policies, prompt_rows) |*p, rows| {
        p.* = try expert_policy.LayerPolicy.init(a, E, rows);
        made += 1;
    }
    var hot: [L][20]u16 = undefined;
    for (&hot) |*h| for (h) |*e| {
        e.* = r.uintLessThan(u16, E);
    };
    const pick = struct {
        fn row(rr: std.Random, hs: []const u16, out: []u16, n_experts: u16) void {
            var n: usize = 0;
            while (n < top_k) {
                const e: u16 = if (rr.float(f32) < 0.75) hs[rr.uintLessThan(usize, hs.len)] else rr.uintLessThan(u16, n_experts);
                if (std.mem.indexOfScalar(u16, out[0..n], e) != null) continue;
                out[n] = e;
                n += 1;
            }
        }
    };
    var plan: expert_policy.Plan = undefined;
    // The prompt: per layer, two chunks of 24 tokens (the wide seed: counts, then the prefill plans of 48-id groups).
    for (&policies, decode_rows, 0..) |*p, grown, l| {
        var ids: [n_prompt * top_k]u16 = undefined;
        for (0..n_prompt) |i| pick.row(r, &hot[l], ids[i * top_k ..][0..top_k], E);
        var c: usize = 0;
        while (c < n_prompt) : (c += 24) {
            const chunk = ids[c * top_k ..][0 .. 24 * top_k];
            t.seed(@intCast(l), chunk);
            p.prepareSeed(chunk);
            var g: usize = 0;
            while (g < chunk.len) : (g += expert_policy.max_route_ids) p.plan(chunk[g..@min(chunk.len, g + expert_policy.max_route_ids)], .prefill, &plan);
        }
        try p.grow(grown);
        t.snapshot(false, @intCast(l), p.slot_to_expert[0..p.capacity]);
    }
    // Decode: one call per round, 1..4 rows; every layer plans the rows' union.
    for (0..calls) |_| {
        const m = 1 + r.uintLessThan(usize, max_call_rows);
        for (&policies, 0..) |*p, l| {
            var ids: [max_call_rows * top_k]u16 = undefined;
            for (0..m) |row| pick.row(r, &hot[l], ids[row * top_k ..][0..top_k], E);
            p.plan(ids[0 .. m * top_k], .decode, &plan);
            t.decodeRoute(@intCast(l), ids[0 .. m * top_k], &plan, &.{});
            t.io_decode.misses += plan.n_misses;
            t.io_decode.hits += plan.n_hits;
        }
        t.roundAccepted(@intCast(r.uintLessThan(usize, m)));
    }
    t.io_decode.bytes = t.io_decode.misses * 1000;
    t.io_decode.records = t.io_decode.misses;
    for (&policies, 0..) |*p, l| t.snapshot(true, @intCast(l), p.slot_to_expert[0..p.capacity]);
    // Each layer's own rows reach the header (`rows`, `rows_end`): the replay seeds and plans every layer at them.
    for (t.rows0, t.rows_end, decode_rows) |r0, r1, d| {
        try testing.expectEqual(d, r0);
        try testing.expectEqual(d, r1);
    }
    try t.write();
    try testing.expectEqual(@as(u32, calls), t.rounds);
    try testing.expect(!t.overflow and !t.row_mismatch and t.wide_decode_calls == 0);
}

test "mimo route trace v2: the replay fixture's plans come from the real decode policy" {
    try writePolicyFixture("MIMO_ROUTE_TRACE_FIXTURE", "mtp-policy", .{ 12, 12, 12 }, .{ 16, 16, 16 });
}

test "mimo route trace v2 (mimo/layer-rows): a per-layer rows split's fixture: every layer at its own rows, one growth" {
    // An `expert_rows_by_layer` split at 12/16 rows: offsets +6, 0, -6 (sum 0) in both phases.
    try writePolicyFixture("MIMO_ROUTE_TRACE_FIXTURE_SPLIT", "mtp-policy-split", .{ 18, 12, 6 }, .{ 22, 16, 10 });
}
