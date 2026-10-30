//! The route recorder (`sdk_ext.expert.routes`) over GLM-5.3's synthetic banks and the stream they serve (host rows,
//! the read pool, the lookahead): on the affine bank a prompt pass, the grow, decode steps of one row and of four, the
//! file read back as the stream served each route; on the EXL3 bank (a K3 and a K4 bank layer per routed layer) each
//! call one route whose ids carry their bank layer's flags.

const std = @import("std");
const host_caps = @import("../host_caps.zig");
const sdk_ext = @import("../sdk_ext.zig");
const policy = sdk_ext.expert.policy;
const routes = sdk_ext.expert.routes;
const Flag = routes.Flag;

const testing = std.testing;

comptime {
    _ = @import("../sdk_ext/expert/routes.zig");
}

/// `rows` rows of `k` distinct experts of `n_experts`, row-major.
fn pickRows(rnd: std.Random, out: []u16, rows: usize, k: usize, n_experts: u16) []u16 {
    for (0..rows) |r| {
        const row = out[r * k ..][0..k];
        var n: usize = 0;
        while (n < k) {
            const e = rnd.uintLessThan(u16, n_experts);
            if (std.mem.indexOfScalar(u16, row[0..n], e) == null) {
                row[n] = e;
                n += 1;
            }
        }
    }
    return out[0 .. rows * k];
}

// ── tests ──

const bank_mod = @import("../../../../plugins/glm_moe_dsa/bank.zig");
const exl3 = @import("../../../../plugins/glm_moe_dsa/exl3_bank.zig");
const glm = @import("../../../../plugins/glm_moe_dsa/glm_moe_dsa.zig");

test "glm routes: the recorder keeps every decode route as the stream served it, and its file reads back" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var sb = try bank_mod.SynthBank.open();
    defer sb.close();
    const b = &sb.bank;
    const n_layers: u32 = @intCast(b.layers.len);
    const ne = b.n_experts;
    const k = glm.routed_top_k;
    const s = try bank_mod.Stream.Stream.init(a, b, .{ .rows = &.{ 4, 6, 3, 5 }, .max_route_ids = 32, .transient_rows = 32, .staging_from_bank = true, .lookahead = .{ .budget = 2 }, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    const rec = try routes.Recorder.init(a, n_layers, ne, k, 1);
    defer rec.deinit();
    s.recorder = rec;
    rec.begin(3, 3);
    var rng = std.Random.DefaultPrng.init(1234);
    const rnd = rng.random();
    var ids_buf: [32]u16 = undefined;

    // The prompt pass: three rows per layer, seeded, one route; nothing of it is a decode route.
    const counts = try a.alloc(u32, n_layers * ne);
    defer a.free(counts);
    @memset(counts, 0);
    for (0..n_layers) |li| {
        const l: u32 = @intCast(li);
        const ids = pickRows(rnd, &ids_buf, 3, k, @intCast(ne));
        for (ids) |e| counts[l * ne + e] += 1;
        try s.seedPrefill(l, ids);
        const r = try s.route(l, ids, &.{});
        try testing.expectEqual(routes.none, r.rec);
        for (0..r.n_parts) |p| {
            try s.waitGu(r, @intCast(p));
            try s.waitDown(r, @intCast(p));
        }
        s.release(r);
    }
    try s.flush();
    try testing.expect(!rec.any());

    // The handover: each layer's prompt counts and residents, its prompt and decode rows.
    const decode_rows = [_]u32{ 6, 8, 5, 7 };
    try s.grow(&decode_rows);
    const residents = try a.alloc(u16, n_layers * ne);
    defer a.free(residents);
    @memset(residents, policy.no_expert);
    for (s.layers, 0..) |*ls, l| @memcpy(residents[l * ne ..][0..ls.policy.capacity], ls.policy.slot_to_expert[0..ls.policy.capacity]);
    try testing.expectEqualSlices(u32, counts, rec.counts);
    try testing.expectEqualSlices(u16, residents, rec.residents);
    try testing.expectEqualSlices(u32, &.{ 4, 6, 3, 5 }, rec.prompt_rows);
    try testing.expectEqualSlices(u32, &decode_rows, rec.decode_rows);

    // Decode: steps of one row and of four (a verify), each layer routed with the next layer's scores.
    const n_steps = 10;
    const Want = struct { ids: [32]u16, n: u8, hits: u32, reads: u32, persistent: u32 };
    var want: [n_steps * 4]Want = undefined;
    var scores: [4 * 16]f32 = undefined;
    const s0 = s.stats();
    for (0..n_steps) |step| {
        const rows: usize = if (step % 2 == 0) 1 else 4;
        for (0..n_layers) |li| {
            const l: u32 = @intCast(li);
            const ids = pickRows(rnd, &ids_buf, rows, k, @intCast(ne));
            for (scores[0 .. rows * ne]) |*x| x.* = rnd.float(f32);
            const r = try s.route(l, ids, scores[0 .. rows * ne]);
            const w = &want[step * n_layers + l];
            w.* = .{ .ids = undefined, .n = @intCast(ids.len), .hits = r.plan.n_hits, .reads = 0, .persistent = r.plan.n_persistent };
            @memcpy(w.ids[0..ids.len], ids);
            for (r.reads[0..r.plan.n_loads]) |rd| w.reads += @intFromBool(rd);
            try testing.expectEqual(@as(u32, @intCast(step * n_layers + l)), r.rec);
            for (0..r.n_parts) |p| {
                try s.waitGu(r, @intCast(p));
                try s.waitDown(r, @intCast(p));
            }
            s.release(r);
        }
        try s.flush();
        rec.endStep(@intCast(rows), @intCast(rows - 1), @intCast(rows / 2), @intCast(rows / 2 + 1), 1000 * step);
    }
    const s1 = s.stats();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [512]u8 = undefined;
    const root = try bank_mod.tmpRoot(&tmp, &root_buf);
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/routes.bin", .{root});
    try rec.write(a, testing.io, path);
    const img = try tmp.dir.readFileAlloc(testing.io, "routes.bin", a, .limited(64 << 20));
    defer a.free(img);

    // The file's parts (`Parts.of` refuses any other length).
    const im = try routes.Parts.of(img);
    try testing.expectEqualSlices(u8, &routes.magic, &im.header.magic);
    try testing.expectEqual(routes.Header{ .n_layers = 4, .n_experts = 16, .top_k = 8, .prompt_tokens = 3, .mtp_depth = 3, .n_routes = n_steps * 4, .n_steps = n_steps, .dropped = 0, .banks_per_layer = 1 }, im.header);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u32{ 4, 6, 3, 5 }), im.prompt_rows);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&decode_rows), im.decode_rows);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(counts), im.counts);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(residents), im.residents);
    var first: u32 = 0;
    for (0..n_steps) |step| {
        const rows: u16 = if (step % 2 == 0) 1 else 4;
        try testing.expectEqual(routes.Step{ .first_route = first, .n_routes = 4, .rows = rows, .drafted = rows - 1, .accepted = rows / 2, .emitted = rows / 2 + 1, .wall_us = @intCast(step) }, im.step(step));
        first += 4;
    }
    var adopted: u64 = 0;
    var any_spec = false;
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(a, ne);
    defer seen.deinit(a);
    for (want[0 .. n_steps * 4], 0..) |w, i| {
        const r = im.route(i);
        try testing.expectEqual(@as(u16, @intCast(i % 4)), r.layer);
        try testing.expectEqual(@as(u32, @intCast(i / 4)), r.step);
        try testing.expectEqual(w.n, r.n_ids);
        try testing.expectEqual(@as(u8, @intCast(w.n / k)), r.rows);
        try testing.expectEqualSlices(u16, w.ids[0..w.n], r.ids[0..r.n_ids]);
        // Per unique expert: resident, or admitted / transient; every record in its row; the reads as planned.
        var hits: u32 = 0;
        var reads: u32 = 0;
        var persistent: u32 = 0;
        seen.unsetAll();
        for (r.ids[0..r.n_ids], r.flags[0..r.n_ids]) |e, f| {
            try testing.expect(f & Flag.settled != 0);
            const kinds = @as(u8, @intFromBool(f & Flag.hit != 0)) + @intFromBool(f & Flag.persistent != 0) + @intFromBool(f & Flag.transient != 0);
            try testing.expectEqual(@as(u8, 1), kinds);
            if (f & Flag.adopted != 0) try testing.expect(f & Flag.read != 0);
            if (seen.isSet(e)) continue;
            seen.set(e);
            hits += @intFromBool(f & Flag.hit != 0);
            reads += @intFromBool(f & Flag.read != 0);
            persistent += @intFromBool(f & Flag.persistent != 0);
            adopted += @intFromBool(f & Flag.adopted != 0);
        }
        try testing.expectEqual(w.hits, hits);
        try testing.expectEqual(w.reads, reads);
        try testing.expectEqual(w.persistent, persistent);
        for (r.spec) |e| {
            if (e == policy.no_expert) continue;
            try testing.expect(e < ne and r.layer + 1 < n_layers);
            any_spec = true;
        }
    }
    try testing.expect(any_spec);
    // A lookahead record serves both ranges of its expert's read, so each adopted record is two adopted ranges.
    try testing.expectEqual(s1.adopt_ranges - s0.adopt_ranges, 2 * adopted);
    std.debug.print("glm routes: {d} decode routes, {d} adopted records, {d} claimed\n", .{ n_steps * 4, adopted, s1.claimed - s0.claimed });
}

test "glm routes: an EXL3 call routes its K3 and K4 bank layers, the recorder keeps one route of the call, each id flagged by its bank layer" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var c = try bank_mod.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{});
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try exl3.Bank.open(a, testing.io, try exl3.tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    const bpl = exl3.banks_per_layer;
    const n_stream: u32 = @intCast(b.layers.len);
    const n_routed = n_stream / bpl;
    var rows: [8]u32 = @splat(3);
    const s = try exl3.Stream.Stream.init(a, &b, .{ .rows = rows[0..n_stream], .max_route_ids = 32, .transient_rows = 32, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 512 }, .lookahead = .{ .k = 8, .budget = 2, .preread = false } });
    defer s.deinit();
    const rec = try routes.Recorder.init(a, n_stream, b.n_experts, glm.routed_top_k, bpl);
    defer rec.deinit();
    s.recorder = rec;
    rec.begin(0, 0);
    var grown: [8]u32 = @splat(5);
    try s.grow(grown[0..n_stream]);
    var rng = std.Random.DefaultPrng.init(42);
    const rnd = rng.random();
    var ids_buf: [32]u16 = undefined;
    var scores: [4 * 16]f32 = undefined;
    const n_steps = 6;
    var hits_want: [n_steps * 4]u32 = undefined;
    for (0..n_steps) |step| {
        const n_rows: usize = 1 + step % 2;
        for (0..n_routed) |li| {
            const l: u32 = @intCast(li);
            const ids = pickRows(rnd, &ids_buf, n_rows, glm.routed_top_k, @intCast(b.n_experts));
            for (scores[0 .. n_rows * b.n_experts]) |*x| x.* = rnd.float(f32);
            // As the decode lane splits a call: its ids by bank layer, every route but the last held.
            var side_ids: [bpl][32]u16 = undefined;
            var side_n: [bpl]usize = @splat(0);
            for (ids) |e| {
                const sd = b.streamLayer(l, e) - bpl * l;
                side_ids[sd][side_n[sd]] = e;
                side_n[sd] += 1;
            }
            var last: usize = 0;
            for (side_n, 0..) |n, i| if (n > 0) {
                last = i;
            };
            rec.call(l, ids);
            var rs: [bpl]?*exl3.Stream.Route = @splat(null);
            hits_want[step * n_routed + l] = 0;
            for (0..bpl) |i| {
                if (side_n[i] == 0) continue;
                const sl: u32 = bpl * l + @as(u32, @intCast(i));
                const r = if (i == last) try s.route(sl, side_ids[i][0..side_n[i]], scores[0 .. n_rows * b.n_experts]) else try s.routeHeld(sl, side_ids[i][0..side_n[i]]);
                try testing.expectEqual(@as(u32, @intCast(step * n_routed + l)), r.rec);
                hits_want[step * n_routed + l] += r.plan.n_hits;
                rs[i] = r;
            }
            rec.endCall();
            for (rs) |ro| if (ro) |r| for (0..r.n_parts) |p| {
                try s.waitGu(r, @intCast(p));
                try s.waitDown(r, @intCast(p));
            };
            for (rs) |ro| if (ro) |r| s.release(r);
        }
        try s.flush();
        rec.endStep(@intCast(n_rows), @intCast(n_rows - 1), 0, 1, 0);
    }
    try testing.expectEqual(@as(usize, n_steps * 4), rec.routes.items.len);
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(a, b.n_experts);
    defer seen.deinit(a);
    for (rec.routes.items, hits_want[0 .. n_steps * 4], 0..) |r, want, i| {
        try testing.expectEqual(@as(u16, @intCast(i % 4)), r.layer);
        try testing.expectEqual(@as(u8, @intCast(r.n_ids / glm.routed_top_k)), r.rows);
        var hits: u32 = 0;
        seen.unsetAll();
        for (r.ids[0..r.n_ids], r.flags[0..r.n_ids]) |e, f| {
            try testing.expect(f & Flag.settled != 0);
            try testing.expectEqual(b.streamLayer(r.layer, e) % bpl == 1, f & Flag.bank != 0);
            if (seen.isSet(e)) continue;
            seen.set(e);
            hits += @intFromBool(f & Flag.hit != 0);
        }
        try testing.expectEqual(want, hits);
    }
    const im = try rec.image(a);
    defer a.free(im);
    try testing.expectEqual(@as(u32, bpl), std.mem.bytesToValue(routes.Header, im[0..@sizeOf(routes.Header)]).banks_per_layer);
}
