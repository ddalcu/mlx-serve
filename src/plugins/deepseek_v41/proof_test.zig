//! Parity proof for the V4.1 plugin: its forward on a tiny random pack equals the deleted legacy seam at every step,
//! its verify block the module's own all-rows extend, and a resumed prompt the legacy resume, as recorded in the tapes
//! in `fixtures/` (`../deepseek_v4/proof_test.zig`).

const std = @import("std");
const env = @import("../../env.zig");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const proof = @import("../deepseek_v4/proof_test.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const v41 = @import("plugin.zig");
const sdk = @import("../../sdk/root.zig");

const testing = std.testing;
const Twin = proof.Twin(v41);
test "deepseek_v41 plugin: prefill, GPU decode, a verify block and a later chunk equal the legacy seam bit for bit" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/steps.tape"));
    defer t.close();
    try proof.expectSteps(v41, t);
}

test "deepseek_v41 plugin: a paired instance keeps the legacy resume and continues bit-equal" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const first = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    const second = first ++ [_]u32{ 11, 2, 90 };
    const paired: dsv41.Pair = .{ .lo = 2, .hi = 4, .worker = false };
    for ([_]bool{ false, true }) |release_parked| {
        const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/proof-resume.tape"));
        defer t.close();
        const model: *v41.Model = @ptrCast(@alignCast(t.m));
        for ([_][]const u32{ &first, &second }, 0..) |prompt, turn| {
            model.mdl.pair = paired;
            const n: u32 = @intCast(v41.resumePrompt(t.m, 7, .of(prompt), .{}));
            model.mdl.pair = null;
            if (turn == 1) try testing.expect(n > 0);
            try t.check(mlx.mlx_array_new_data(&n, &[_]c_int{1}, 1, .uint32));
            try t.check(try t.plugin(.{ .mode = .prefill, .ids = prompt[n..] }, n));
            if (turn == 0 and release_parked)
                testing.allocator.free(try slotForward(t, .prefill, &.{8}, &.{ 4, 12, 6 }, &.{0}));
            model.mdl.pair = paired;
            v41.table.release_slot.?(t.m, 7);
            model.mdl.pair = null;
            if (turn == 0 and !release_parked)
                testing.allocator.free(try slotForward(t, .prefill, &.{8}, &.{ 4, 12, 6 }, &.{0}));
        }
        try t.tape.expectDone();
        testing.allocator.free(try slotForward(t, .decode, &.{8}, &.{13}, &.{3}));
    }
}

test "deepseek_v41 plugin: a decode token on the device equals the legacy decode" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/device-decode.tape"));
    defer t.close();
    try proof.expectDeviceDecode(v41, t);
}

test "deepseek_v41 plugin: trunk_weight hands out the loaded embedding and head" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    var w: @import("../../abi/quant.zig").Weight = .{};
    try testing.expect(v41.table.spec.?.trunk_weight.?(t.m, .embed, &w).isOk());
    try testing.expectEqual(m.mdl.embed.w.ctx, w.w.ctx);
    try testing.expect(v41.table.spec.?.trunk_weight.?(t.m, .head, &w).isOk());
    try testing.expectEqual(m.mdl.head.w.ctx, w.w.ctx);
    try testing.expectEqual(m.mdl.head.qp.bits, w.bits);
}

fn slotForwardRaw(t: *Twin, mode: abi.Mode, logits: abi.Logits, slots: []const u32, toks: []const u32, positions: []const u32) !abi.ForwardOut {
    var out: abi.ForwardOut = .{};
    const rows: u32 = if (mode == .batched_rows) 1 else @intCast(toks.len);
    const v = v41.table.forward(t.m, &.{ .mode = mode, .logits = logits, .rows = rows, .tokens = .of(toks), .slots = .of(slots), .positions = .of(positions) }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    return out;
}

/// Runs a forward requesting no logits and verifies the plugin returns a null logits handle.
fn slotForwardNoLogits(t: *Twin, mode: abi.Mode, slots: []const u32, toks: []const u32, positions: []const u32) !void {
    const out = try slotForwardRaw(t, mode, .none, slots, toks, positions);
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
    try testing.expect(out.logits.ctx == null);
}

/// The plugin's logits `[rows, V]` (host f32, caller frees) for `toks` of `slots` at `positions`.
fn slotForwardLogits(t: *Twin, mode: abi.Mode, logits: abi.Logits, slots: []const u32, toks: []const u32, positions: []const u32) ![]f32 {
    std.debug.assert(logits != .none);
    const out = try slotForwardRaw(t, mode, logits, slots, toks, positions);
    const l: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(l);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, l, .float32, t.s));
    try mlx.check(mlx.mlx_array_eval(f));
    return try testing.allocator.dupe(f32, (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..mlx.mlx_array_size(f)]);
}

/// The plugin's logits `[rows, V]` (host f32, caller frees) for `toks` of `slots` at `positions`.
fn slotForward(t: *Twin, mode: abi.Mode, slots: []const u32, toks: []const u32, positions: []const u32) ![]f32 {
    return slotForwardLogits(t, mode, .last_row, slots, toks, positions);
}

/// Three batched ticks of two slots (prompts `a` and `b`, decode tokens `da` and `db`), each tick's `[2, V]` logits.
fn batchTicks(files: []const sdk.fixture.File, a: []const u32, b: []const u32, da: [3]u32, db: [3]u32) ![3][]f32 {
    const t = try Twin.open(files, "");
    defer t.close();
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, a, &.{0}));
    testing.allocator.free(try slotForward(t, .prefill, &.{2}, b, &.{0}));
    var out: [3][]f32 = undefined;
    for (0..3) |i| out[i] = try slotForward(t, .batched_rows, &.{ 1, 2 }, &.{ da[i], db[i] }, &.{ @intCast(a.len + i), @intCast(b.len + i) });
    return out;
}

test "deepseek_v41 plugin: a slot's row in a batch depends on its own state only, past the window, a pooling group and the indexer's budget" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const p = [2][]const u32{ &.{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 }, &.{ 5, 8, 13, 21, 34, 55, 89, 4, 6, 10, 12, 14, 16, 18 } };
    const d = [2][3]u32{ .{ 44, 2, 71 }, .{ 9, 30, 63 } };
    // Two rows run other kernels than one (the MoE's gather arm), so a row is compared at the same width: beside
    // another slot it equals itself beside a copy of its own conversation.
    const mixed = try batchTicks(files, p[0], p[1], d[0], d[1]);
    defer for (mixed) |x| testing.allocator.free(x);
    const own0 = try batchTicks(files, p[0], p[0], d[0], d[0]);
    defer for (own0) |x| testing.allocator.free(x);
    const own1 = try batchTicks(files, p[1], p[1], d[1], d[1]);
    defer for (own1) |x| testing.allocator.free(x);
    for (0..3) |i| {
        const V = mixed[i].len / 2;
        try testing.expectEqualSlices(f32, own0[i][0..V], mixed[i][0..V]);
        try testing.expectEqualSlices(f32, own1[i][V..], mixed[i][V..]);
    }
}

test "deepseek_v41 plugin: a one-slot batch is the slot's own decode step, bit for bit" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 };
    const steps = [_]u32{ 44, 2, 71 };
    var ts: [2]*Twin = undefined;
    for (&ts) |*t| t.* = try Twin.open(files, "");
    defer for (ts) |t| t.close();
    for (ts) |t| testing.allocator.free(try slotForward(t, .prefill, &.{1}, &prompt, &.{0}));
    for (steps, 0..) |tok, i| {
        const pos: u32 = @intCast(prompt.len + i);
        const want = try slotForward(ts[0], .decode, &.{1}, &.{tok}, &.{pos});
        defer testing.allocator.free(want);
        const got = try slotForward(ts[1], .batched_rows, &.{1}, &.{tok}, &.{pos});
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, want, got);
    }
}

test "deepseek_v41 aux: overlapping captures survive either release order and restore independently" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const prompts = [2][]const u32{ &.{ 3, 17, 42, 9, 7, 25, 1, 60 }, &.{ 5, 8, 13, 21, 34, 55, 89, 4, 6 } };
    for ([_][2]u32{ .{ 1, 2 }, .{ 2, 1 } }) |order| {
        const t = try Twin.open(files, "");
        defer t.close();
        const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
        try testing.expect(aux.enabled(t.m));
        for (prompts, 1..) |p, slot| {
            try testing.expect(aux.begin_prompt(t.m, @intCast(slot), .of(p), .{}, 0, true).isOk());
            testing.allocator.free(try slotForward(t, .prefill, &.{@intCast(slot)}, p[0..2], &.{0}));
        }
        for (prompts, 1..) |p, slot| testing.allocator.free(try slotForward(t, .prefill, &.{@intCast(slot)}, p[2..], &.{2}));
        var snaps: [2]abi.Handle = .{ null, null };
        defer for (snaps) |sn| aux.free(t.m, sn);
        for (order) |slot| {
            const i = slot - 1;
            try testing.expect(aux.snapshot(t.m, slot, &snaps[i]).isOk());
            try testing.expect(snaps[i] != null);
            var empty: abi.Handle = null;
            try testing.expect(aux.snapshot(t.m, slot, &empty).isOk());
            try testing.expect(empty == null);
            v41.table.release_slot.?(t.m, slot);
            v41.table.release_slot.?(t.m, slot);
        }
        try testing.expectEqual(@as(u64, 0), v41.stateBytes(t.m));
        for (snaps, prompts) |sn, p| {
            var info: abi.AuxSnapshotInfo = .{};
            try testing.expect(aux.describe(t.m, sn, &info).isOk());
            try testing.expectEqual(p.len - 1, info.n);
            try testing.expectEqualSlices(u32, p[0 .. p.len - 1], info.ids.slice());
            try testing.expect(aux.bytes(t.m, sn) > 0);
            try testing.expectEqual(abi.Verdict.Code.unsupported, aux.restore_at(t.m, 3, sn, info.n - 1).code);
            var clone: abi.Handle = null;
            try testing.expectEqual(abi.Verdict.Code.unsupported, aux.slice(t.m, sn, info.n + 1, &clone).code);
            try testing.expect(clone == null);
            try testing.expect(aux.slice(t.m, sn, info.n, &clone).isOk());
            defer aux.free(t.m, clone);
            var warm: [2][]f32 = undefined;
            for ([_]u32{ 3, 4 }, 0..) |slot, k| {
                try testing.expect(aux.restore_at(t.m, slot, if (k == 0) sn else clone, info.n).isOk());
                try testing.expectEqual(abi.Verdict.Code.unsupported, aux.restore_at(t.m, slot, sn, info.n + 1).code);
                try testing.expectEqual(abi.Verdict.Code.failed, aux.begin_prompt(t.m, slot, .of(p), .{}, info.n - 1, false).code);
                try testing.expect(aux.begin_prompt(t.m, slot, .of(p), .{}, info.n, false).isOk());
                warm[k] = try slotForward(t, .prefill, &.{slot}, p[info.n..], &.{@intCast(info.n)});
            }
            defer for (warm) |row| testing.allocator.free(row);
            try testing.expectEqualSlices(f32, warm[0], warm[1]);
            try testing.expectEqual(abi.Verdict.Code.busy, aux.handoff(t.m, 3, 4).code);
            try testing.expect(aux.handoff(t.m, 3, 5).isOk());
            v41.table.release_slot.?(t.m, 3);
            v41.table.release_slot.?(t.m, 5);
            try testing.expect(aux.begin_prompt(t.m, 4, .of(p), .{}, 0, false).isOk());
            try testing.expect(aux.commit(t.m, 4, 0).isOk());
            testing.allocator.free(try slotForward(t, .prefill, &.{4}, p[0..2], &.{0}));
            testing.allocator.free(try slotForward(t, .prefill, &.{4}, p[2 .. p.len - 1], &.{2}));
            const cold = try slotForward(t, .prefill, &.{4}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
            defer testing.allocator.free(cold);
            try testing.expectEqualSlices(f32, cold, warm[0]);
            const reference = try Twin.open(files, "");
            defer reference.close();
            testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[0..2], &.{0}));
            testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[2 .. p.len - 1], &.{2}));
            const fresh = try slotForward(reference, .prefill, &.{1}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
            defer testing.allocator.free(fresh);
            for (fresh) |value| try testing.expect(std.math.isFinite(value));
            try testing.expectEqualSlices(f32, fresh, warm[0]);
            var no_capture: abi.Handle = null;
            try testing.expect(aux.snapshot(t.m, 4, &no_capture).isOk());
            try testing.expect(no_capture == null);
            v41.table.release_slot.?(t.m, 4);
        }
    }
}

test "deepseek_v41 aux: a snapshot sliced short of its end restores there by replaying its window" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var long: [41]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const tail = [_]u32{ 11, 29, 5 };
    // The run the snapshot comes from and the cold runs it is judged against are chunked alike and kept on the wide
    // path, so a row is the same bits wherever it was made.
    const chunks = [_]usize{ 0, 8, 24, 40 };
    for ([_]bool{ false, true }) |dspark| {
        const files = if (dspark) try fixture.v41DsparkFiles(arena.allocator(), 7) else try fixture.v41Files(arena.allocator(), 7);
        const wide = try WideTwin.open(files);
        defer wide.close();
        const t = wide.t;
        const model = wide.model();
        const w: usize = model.mdl.cfg.sliding_window;
        const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&long), .{}, 0, true).isOk());
        for (chunks[0..3], chunks[1..]) |lo, hi| testing.allocator.free(try slotForward(t, .prefill, &.{1}, long[lo..hi], &.{@intCast(lo)}));
        var sn: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
        try testing.expect(sn != null);
        defer aux.free(t.m, sn);
        const src: *const dsv41.Snap = @ptrCast(@alignCast(sn.?));
        v41.table.release_slot.?(t.m, 1);
        var ends: [2][]f32 = .{ &.{}, &.{} };
        defer for (ends) |e| testing.allocator.free(e);
        ends[0] = try restoredTail(t, aux, sn, &long, long[40..]);

        // A cut at one window replays the whole prefix (nothing is truncated); one at 31 stands at the pooling
        // group below it.
        for ([_][2]usize{ .{ 8, 8 }, .{ 24, 24 }, .{ 31, 30 } }) |c| {
            const n = c[1];
            var cut: abi.Handle = null;
            try testing.expect(aux.slice(t.m, sn, c[0], &cut).isOk());
            defer aux.free(t.m, cut);
            var info: abi.AuxSnapshotInfo = .{};
            try testing.expect(aux.describe(t.m, cut, &info).isOk());
            try testing.expectEqual(n, @as(usize, @intCast(info.n)));
            try testing.expectEqualSlices(u32, long[0..n], info.ids.slice());
            var out: abi.AuxExport = .{};
            try testing.expectEqual(abi.Verdict.Code.failed, aux.export_arrays(t.m, cut, &out).code);
            try testing.expectEqual(abi.Verdict.Code.unsupported, aux.restore_at(t.m, 2, cut, info.n + 2).code);
            try testing.expect(aux.restore_at(t.m, 2, cut, info.n).isOk());
            const ws = &model.mdl.dec_state.?;
            try testing.expectEqual(n, ws.n);

            const cold_twin = try WideTwin.open(files);
            defer cold_twin.close();
            const r = cold_twin.t;
            const cold_model = cold_twin.model();
            for (chunks[0..3], chunks[1..]) |lo, hi| {
                if (lo < n) testing.allocator.free(try slotForward(r, .prefill, &.{1}, long[lo..@min(hi, n)], &.{@intCast(lo)}));
            }
            const cs = &cold_model.mdl.dec_state.?;
            // Bars: the global rows and index keys are the snapshot's own, cut; the first layer's window is the cold
            // prompt's (nothing below it was truncated), and so is every layer's when the replay is the whole prefix.
            for (ws.layers, cs.layers, src.layers) |*wl, *cl, sl| {
                try testing.expect(wl.pend_kv.ctx == null and wl.pend_gate.ctx == null);
                inline for (.{ "comp", "idx_k" }) |f| if (@field(wl, f)) |rows| {
                    try testing.expectEqual(@field(cl, f).?.used, rows.used);
                    try expectRowsEqual(rows.buf, @field(sl, f).?.buf, 0, rows.used, rows.used);
                    if (n <= 24) try expectRowsEqual(rows.buf, @field(cl, f).?.buf, 0, rows.used, rows.used);
                };
                if (n == w) try expectRowsEqual(wl.win.buf, cl.win.buf, n - w, n, wl.win.cap) else try expectFinite(wl.win.buf);
            }
            if (n > w) {
                // The replay makes the first layer's window in one call of `w` rows; a row's bits can depend on its
                // call's width, so the cold prompt it is judged against ends with that same call.
                const split = try WideTwin.open(files);
                defer split.close();
                const r2 = split.t;
                const split_model = split.model();
                for (chunks[0..3], chunks[1..]) |lo, hi| {
                    if (lo < n - w) testing.allocator.free(try slotForward(r2, .prefill, &.{1}, long[lo..@min(hi, n - w)], &.{@intCast(lo)}));
                }
                testing.allocator.free(try slotForward(r2, .prefill, &.{1}, long[n - w .. n], &.{@intCast(n - w)}));
                const first = split_model.mdl.dec_state.?.layers[0].win;
                try expectRowsEqual(ws.layers[0].win.buf, first.buf, n - w, n, ws.layers[0].win.cap);
            }
            if (n == w) {
                for (ws.ds, cs.ds) |wr, cr| try expectRowsEqual(wr.buf, cr.buf, 0, n, wr.cap);
                if (ws.mh.ctx != null) try expectRowsEqual(ws.mh, cs.mh, 0, 1, 1);
            }
            try testing.expectEqualSlices(u32, cs.hist.items, ws.hist.items);

            var prompt: [44]u32 = undefined;
            @memcpy(prompt[0..n], long[0..n]);
            @memcpy(prompt[n..][0..tail.len], &tail);
            try testing.expect(aux.begin_prompt(t.m, 2, .of(prompt[0 .. n + tail.len]), .{}, info.n, false).isOk());
            const warm = try slotForward(t, .prefill, &.{2}, &tail, &.{@intCast(n)});
            defer testing.allocator.free(warm);
            v41.table.release_slot.?(t.m, 2);
            const cold = try slotForward(r, .prefill, &.{1}, &tail, &.{@intCast(n)});
            defer testing.allocator.free(cold);
            for (warm) |x| try testing.expect(std.math.isFinite(x));
            if (n == w) try testing.expectEqualSlices(f32, cold, warm);
        }
        // The snapshot the slices were cut from is untouched: it restores at its end as before them.
        ends[1] = try restoredTail(t, aux, sn, &long, long[40..]);
        try testing.expectEqualSlices(f32, ends[0], ends[1]);
    }
}

test "deepseek_v41 CED: a lead chunk that skips the decoder leaves what the chunk after it reads" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    {
        const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
        defer t.close();
        const model: *v41.Model = @ptrCast(@alignCast(t.m));
        const mdl = model.mdl;
        try testing.expect(mdl.ced_split != null);
        const w: usize = mdl.cfg.sliding_window;
        const skip = mdl.ced_min_skip;
        defer mdl.ced_min_skip = skip;
        mdl.ced_min_skip = w; // a chunk of two windows replays its last one
        // The same two chunks, the first with its decoder replay and without.
        var sts: [2]dsv41.State = undefined;
        var logits: [2][]f32 = undefined;
        for (&sts, &logits, [_]dsv41.Want{ .last, .encode }) |*st, *out, lead| {
            st.* = try dsv41.initState(mdl, testing.allocator);
            const head = try dsv41.extend(mdl, testing.allocator, st, ids[0..16], lead);
            if (head.ctx != null) _ = mlx.mlx_array_free(head);
            const rest = try dsv41.extend(mdl, testing.allocator, st, ids[16..], .last);
            defer _ = mlx.mlx_array_free(rest);
            out.* = try hostRows(rest);
        }
        defer for (&sts, logits) |*st, l| {
            st.deinit();
            testing.allocator.free(l);
        };
        try testing.expectEqual(@as(u64, 2 + 1), mdl.ced_stats.chunks);
        try testing.expectEqualSlices(f32, logits[0], logits[1]);
        const n = ids.len;
        for (sts[0].layers, sts[1].layers) |*a, *b| {
            inline for (.{ "comp", "idx_k" }) |f| if (@field(a, f)) |rows| {
                try testing.expectEqual(rows.used, @field(b, f).?.used);
                try expectRowsEqual(@field(b, f).?.buf, rows.buf, 0, rows.used, rows.used);
            };
            try expectRowsEqual(b.win.buf, a.win.buf, n - w, n, a.win.cap);
        }
    }
}

test "deepseek_v41 CED encode interior: multi-chunk prefill skips decoder on interior chunks and matches bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [65]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    const cold_t = try Twin.open(files, "");
    defer cold_t.close();
    const cold_m: *v41.Model = @ptrCast(@alignCast(cold_t.m));
    const w: usize = cold_m.mdl.cfg.sliding_window;
    cold_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(cold_t.m, 1, .of(&p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 }, .{ 48, 64 } }) |r| {
        testing.allocator.free(try slotForward(cold_t, .prefill, &.{1}, p[r[0]..r[1]], &.{@intCast(r[0])}));
    }
    const cold_logits = try slotForward(cold_t, .prefill, &.{1}, p[64..65], &.{64});
    defer testing.allocator.free(cold_logits);
    try testing.expectEqual(@as(u64, 0), cold_m.mdl.ced_stats.encode_chunks);
    try testing.expectEqual(@as(u64, 4), cold_m.mdl.ced_stats.chunks);

    const enc_t = try Twin.open(files, "");
    defer enc_t.close();
    const enc_m: *v41.Model = @ptrCast(@alignCast(enc_t.m));
    enc_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(enc_t.m, 1, .of(&p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 } }) |r| {
        try slotForwardNoLogits(enc_t, .prefill, &.{1}, p[r[0]..r[1]], &.{@intCast(r[0])});
        try testing.expect(enc_m.mdl.dec_state.?.decoder_stale);
    }
    try testing.expectEqual(@as(u64, 3), enc_m.mdl.ced_stats.encode_chunks);
    try testing.expectEqual(@as(u64, 48), enc_m.mdl.ced_stats.encode_rows);
    try testing.expectEqual(@as(u64, 0), enc_m.mdl.ced_stats.chunks);
    testing.allocator.free(try slotForward(enc_t, .prefill, &.{1}, p[48..64], &.{48}));
    try testing.expect(!enc_m.mdl.dec_state.?.decoder_stale);
    const enc_logits = try slotForward(enc_t, .prefill, &.{1}, p[64..65], &.{64});
    defer testing.allocator.free(enc_logits);
    try testing.expectEqual(@as(u64, 1), enc_m.mdl.ced_stats.chunks);
    try testing.expectEqualSlices(f32, cold_logits, enc_logits);

    // Every encoder/decoder KV and window ring at [p.len - w, p.len) matches bit for bit.
    const st_a = &cold_m.mdl.dec_state.?;
    const st_b = &enc_m.mdl.dec_state.?;
    for (st_a.layers, st_b.layers) |*a, *b| {
        inline for (.{ "comp", "idx_k" }) |f| if (@field(a, f)) |rows| {
            try testing.expectEqual(rows.used, @field(b, f).?.used);
            try expectRowsEqual(@field(b, f).?.buf, rows.buf, 0, rows.used, rows.used);
        };
        try expectRowsEqual(b.win.buf, a.win.buf, p.len - w, p.len, a.win.cap);
    }

    // Short-tail boundary: when remaining tokens <= w + ced_min_skip, the chunk before the short tail runs `.last` (cedTail) instead of `.encode`.
    const short_p = p[0..55];
    const short_cold_t = try Twin.open(files, "");
    defer short_cold_t.close();
    const short_cold_m: *v41.Model = @ptrCast(@alignCast(short_cold_t.m));
    short_cold_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(short_cold_t.m, 1, .of(short_p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 }, .{ 48, 54 } }) |r| {
        testing.allocator.free(try slotForward(short_cold_t, .prefill, &.{1}, short_p[r[0]..r[1]], &.{@intCast(r[0])}));
    }
    const short_cold = try slotForward(short_cold_t, .prefill, &.{1}, short_p[54..55], &.{54});
    defer testing.allocator.free(short_cold);

    const short_enc_t = try Twin.open(files, "");
    defer short_enc_t.close();
    const short_enc_m: *v41.Model = @ptrCast(@alignCast(short_enc_t.m));
    short_enc_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(short_enc_t.m, 1, .of(short_p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 } }) |r| {
        try slotForwardNoLogits(short_enc_t, .prefill, &.{1}, short_p[r[0]..r[1]], &.{@intCast(r[0])});
    }
    // Chunks [0..16) and [16..32) encoded; [32..48) declined `.encode` because remaining=7 <= 2*w=16.
    try testing.expectEqual(@as(u64, 2), short_enc_m.mdl.ced_stats.encode_chunks);
    try testing.expect(!short_enc_m.mdl.dec_state.?.decoder_stale);
    testing.allocator.free(try slotForward(short_enc_t, .prefill, &.{1}, short_p[48..54], &.{48}));
    const short_enc = try slotForward(short_enc_t, .prefill, &.{1}, short_p[54..55], &.{54});
    defer testing.allocator.free(short_enc);
    try testing.expectEqualSlices(f32, short_cold, short_enc);
}

test "deepseek_v41 CED encode interior: mid-prefill snap_at and cancelled prefill resume (long and short suffix)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [65]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    // Reference cold run for 65 tokens.
    const ref_t = try Twin.open(files, "");
    defer ref_t.close();
    const ref_m: *v41.Model = @ptrCast(@alignCast(ref_t.m));
    const w: usize = ref_m.mdl.cfg.sliding_window;
    ref_m.mdl.ced_min_skip = w;
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 }, .{ 48, 64 } }) |r| {
        testing.allocator.free(try slotForward(ref_t, .prefill, &.{1}, p[r[0]..r[1]], &.{@intCast(r[0])}));
    }
    const cold = try slotForward(ref_t, .prefill, &.{1}, p[64..65], &.{64});
    defer testing.allocator.free(cold);

    // 1) Mid-prefill snap_at: chunks within `min_chunk` of snap_at must NOT leave `decoder_stale == true`,
    // and the split chunk containing snap_at runs `.last` + captures an exact snapshot.
    {
        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        // Override snap_at to land inside chunk [16..32) at position 24 (within min_chunk=16 of chunk 0 end 16).
        m.mdl.dec_state.?.snap_at = 24;
        m.mdl.dec_state.?.gpa.free(m.mdl.dec_state.?.snap_ids);
        m.mdl.dec_state.?.snap_ids = try m.mdl.dec_state.?.gpa.dupe(u32, p[0..24]);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try testing.expect(m.mdl.dec_state.?.snap != null);
        try testing.expectEqual(@as(usize, 24), m.mdl.dec_state.?.snap.?.n);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[32..48], &.{32});
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[48..64], &.{48}));
        const got = try slotForward(t, .prefill, &.{1}, p[64..65], &.{64});
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, cold, got);
    }
    {
        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        // Override snap_at to 40 (inside chunk [32..48)): chunk 0 encodes, chunk 1 repairs via cedTail before split at 40.
        m.mdl.dec_state.?.snap_at = 40;
        m.mdl.dec_state.?.gpa.free(m.mdl.dec_state.?.snap_ids);
        m.mdl.dec_state.?.snap_ids = try m.mdl.dec_state.?.gpa.dupe(u32, p[0..40]);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try testing.expectEqual(@as(u64, 1), m.mdl.ced_stats.encode_chunks);
        try testing.expect(m.mdl.dec_state.?.decoder_stale);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expectEqual(@as(u64, 1), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[32..48], &.{32});
        try testing.expectEqual(@as(u64, 1), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try testing.expect(m.mdl.dec_state.?.snap != null);
        try testing.expectEqual(@as(usize, 40), m.mdl.dec_state.?.snap.?.n);
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[48..64], &.{48}));
        const got = try slotForward(t, .prefill, &.{1}, p[64..65], &.{64});
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, cold, got);
    }

    // 2) Cancelled prefill after `.encode` chunk (`decoder_stale == true`): snapshot is skipped (`snap == null`),
    // and direct decode on stale state without a `cedTail` / `.replay` chunk is refused (`error.Dsv41DecodeOnStaleState`).
    {
        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expect(m.mdl.dec_state.?.decoder_stale);

        var snap_stale: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &snap_stale).isOk());
        try testing.expectEqual(@as(abi.Handle, null), snap_stale);
        try testing.expect(m.mdl.dec_state.?.decoder_stale);
        try testing.expectError(
            error.Dsv41DecodeOnStaleState,
            dsv41.extend(m.mdl, testing.allocator, &m.mdl.dec_state.?, p[32..33], .last),
        );
        v41.table.release_slot.?(t.m, 1);
    }

    // 3) Cancelled prefill after `.encode` chunks followed by a `.last` (`cedTail`) chunk (`decoder_stale == false`):
    // snapshot captures an exact `whole` state, and resuming with both >= w + ced_min_skip and < w + ced_min_skip
    // remaining rows produces bit-identical logits to cold prefill.
    {
        const short_prompt = p[0..55]; // suffix 48..55 has 7 rows (< w + ced_min_skip = 16)
        const cold_short_t = try Twin.open(files, "");
        defer cold_short_t.close();
        const cold_short_m: *v41.Model = @ptrCast(@alignCast(cold_short_t.m));
        cold_short_m.mdl.ced_min_skip = w;
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[0..16], &.{0}));
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[16..32], &.{16}));
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[32..48], &.{32}));
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[48..54], &.{48}));
        const cold_short = try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[54..55], &.{54});
        defer testing.allocator.free(cold_short);

        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expect(m.mdl.dec_state.?.decoder_stale);
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[32..48], &.{32}));
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);

        var snap_48: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &snap_48).isOk());
        try testing.expect(snap_48 != null);
        defer aux.free(t.m, snap_48);

        var info: abi.AuxSnapshotInfo = .{};
        try testing.expect(aux.describe(t.m, snap_48, &info).isOk());
        try testing.expectEqual(@as(u64, 48), info.n);
        v41.table.release_slot.?(t.m, 1);

        // Long suffix (>= w + ced_min_skip: 17 rows remaining).
        try testing.expect(aux.restore_at(t.m, 2, snap_48, 48).isOk());
        try testing.expect(aux.begin_prompt(t.m, 2, .of(&p), .{}, 48, true).isOk());
        testing.allocator.free(try slotForward(t, .prefill, &.{2}, p[48..64], &.{48}));
        const warm_long = try slotForward(t, .prefill, &.{2}, p[64..65], &.{64});
        defer testing.allocator.free(warm_long);
        try testing.expectEqualSlices(f32, cold, warm_long);
        v41.table.release_slot.?(t.m, 2);

        // Short suffix (< w + ced_min_skip: 7 rows remaining).
        try testing.expect(aux.restore_at(t.m, 3, snap_48, 48).isOk());
        try testing.expect(aux.begin_prompt(t.m, 3, .of(short_prompt), .{}, 48, true).isOk());
        testing.allocator.free(try slotForward(t, .prefill, &.{3}, short_prompt[48..54], &.{48}));
        const warm_short = try slotForward(t, .prefill, &.{3}, short_prompt[54..55], &.{54});
        defer testing.allocator.free(warm_short);
        try testing.expectEqualSlices(f32, cold_short, warm_short);
    }
}

test "deepseek_v41 CED encode interior: all_rows, pair, TP and disabled CED never encode" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [40]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    const t = try Twin.open(files, "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const w: usize = m.mdl.cfg.sliding_window;
    m.mdl.ced_min_skip = w;

    // 1) `.all_rows` (prompt logprobs path) never encodes and returns [L, V].
    try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
    const all_logits = try slotForwardLogits(t, .prefill, .all_rows, &.{1}, p[0..16], &.{0});
    defer testing.allocator.free(all_logits);
    try testing.expectEqual(16 * m.mdl.vocab, all_logits.len);
    try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
    v41.table.release_slot.?(t.m, 1);

    // 2) `m.pair != null` and `m.tp != null` refuse `Want.encode`.
    var st = try dsv41.initState(m.mdl, testing.allocator);
    defer st.deinit();
    m.mdl.pair = .{ .lo = 1, .hi = 2, .worker = false };
    try testing.expectError(error.Dsv41EncodeUnsupported, dsv41.extend(m.mdl, testing.allocator, &st, p[0..16], .encode));
    m.mdl.pair = null;
    m.mdl.tp = .{ .rank = 0, .size = 4 };
    try testing.expectError(error.Dsv41EncodeUnsupported, dsv41.extend(m.mdl, testing.allocator, &st, p[0..16], .encode));
    m.mdl.tp = null;

    // 3) Disabled CED (`MLX_SERVE_DSV41_CED=0`) never encodes even on `.none` forwards.
    const prev_ced = env.get(.MLX_SERVE_DSV41_CED);
    env.set(.MLX_SERVE_DSV41_CED, "0");
    dsv41.resetCedEnvForTest();
    defer {
        if (prev_ced) |v| env.set(.MLX_SERVE_DSV41_CED, v) else env.unset(.MLX_SERVE_DSV41_CED);
        dsv41.resetCedEnvForTest();
    }
    try testing.expect(aux.begin_prompt(t.m, 2, .of(&p), .{}, 0, true).isOk());
    try slotForwardNoLogits(t, .prefill, &.{2}, p[0..16], &.{0});
    try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
    try testing.expect(!m.mdl.dec_state.?.decoder_stale);
}

fn isPrime(n: u64) bool {
    if (n < 2) return false;
    var i: u64 = 2;
    while (i * i <= n) : (i += 1) if (n % i == 0) return false;
    return true;
}

fn engramConfigForTest(cfg: *@import("config.zig").Config, layers: [2]u8) void {
    cfg.dsv41_n_engram_layers = 2;
    cfg.dsv41_engram_layers[0] = layers[0];
    cfg.dsv41_engram_layers[1] = layers[1];
    cfg.dsv41_engram_max_ngram = 4;
    cfg.dsv41_engram_n_heads = 8;
    cfg.dsv41_engram_head_dim = 256;
    cfg.dsv41_engram_vocab_size = 1000;
    cfg.dsv41_engram_pad_id = 2;
    cfg.dsv41_engram_compressed_vocab = cfg.vocab_size;
    var p = cfg.dsv41_engram_vocab_size - 1;
    const cols = (cfg.dsv41_engram_max_ngram - 1) * cfg.dsv41_engram_n_heads;
    for (0..2) |l| {
        var rows: u64 = 0;
        for (0..cols) |_| {
            p += 1;
            while (!isPrime(p)) p += 1;
            rows += p;
        }
        cfg.dsv41_engram_rows[l] = rows;
    }
}

fn writeEngramPackForTest(gpa: std.mem.Allocator, dir: std.Io.Dir, vocab: u32, layers: [2]u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg: @import("config.zig").Config = .{};
    cfg.vocab_size = vocab;
    engramConfigForTest(&cfg, layers);
    try dir.createDirPath(io, "engram");
    const hd = cfg.dsv41_engram_head_dim;
    const rec = hd + hd / 32;
    var man: std.ArrayList(u8) = .empty;
    defer man.deinit(gpa);
    try man.appendSlice(gpa, "{\"layers\":[");
    for (layers, 0..) |l, k| {
        const rows = cfg.dsv41_engram_rows[k];
        const name = try std.fmt.allocPrint(gpa, "engram/layer{d}.bin", .{l});
        defer gpa.free(name);
        const buf = try gpa.alloc(u8, @intCast(rows * rec));
        defer gpa.free(buf);
        var rng = std.Random.DefaultPrng.init(l);
        rng.random().bytes(buf);
        for (0..rows) |r| {
            const row = buf[r * rec ..][0..rec];
            for (row[0..hd]) |*b| b.* &= 0xb7;
            for (row[hd..]) |*b| b.* = 120 + (b.* & 7);
        }
        try dir.writeFile(io, .{ .sub_path = name, .data = buf });
        if (k > 0) try man.append(gpa, ',');
        try man.print(gpa, "{{\"layer_id\":{d},\"file\":\"layer{d}.bin\",\"rows\":{d},\"record_bytes\":{d},\"quant\":{{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\",\"head_dim\":{d}}}}}", .{ l, l, rows, rec, hd });
    }
    try man.appendSlice(gpa, "]}");
    try dir.writeFile(io, .{ .sub_path = "engram/engram-manifest.json", .data = man.items });
    const map = try gpa.alloc(u8, vocab * 4);
    defer gpa.free(map);
    for (0..vocab) |i| std.mem.writeInt(u32, map[i * 4 ..][0..4], @intCast(i), .little);
    try dir.writeFile(io, .{ .sub_path = "engram-token-map.u32", .data = map });
}

fn attachTinyEngramForTest(mdl: *dsv41.Dsv41Model, dir: []const u8) !void {
    const gpa = mdl.gpa;
    engramConfigForTest(&mdl.cfg, .{ 1, 2 });
    mdl.token_map = try engram.loadTokenMap(gpa, dir, @intCast(mdl.vocab), mdl.cfg.dsv41_engram_compressed_vocab);
    mdl.hash = try engram.Hash.init(&mdl.cfg, mdl.token_map[mdl.cfg.dsv41_engram_pad_id]);
    const cols = mdl.hash.?.cols();
    const d: c_int = @intCast(mdl.cfg.hidden_size);
    const hc: c_int = @intCast(mdl.cfg.dsv4_hc_mult);
    const in_dim: c_int = @intCast(cols * mdl.cfg.dsv41_engram_head_dim);
    mdl.engrams = try gpa.alloc(dsv41.EngramLayer, 2);
    var opened: usize = 0;
    errdefer {
        for (mdl.engrams[0..opened]) |*e| e.table.close();
        gpa.free(mdl.engrams);
        mdl.engrams = &.{};
    }
    const w_scale_f32 = mlx.mlx_array_new_float(0.01);
    defer _ = mlx.mlx_array_free(w_scale_f32);
    var w_scale = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_scale);
    try mlx.check(mlx.mlx_astype(&w_scale, w_scale_f32, mdl.act, mdl.s));
    const qk_scale = mlx.mlx_array_new_float(0.05);
    defer _ = mlx.mlx_array_free(qk_scale);
    for (0..2) |k| {
        const l: usize = mdl.cfg.dsv41_engram_layers[k];
        var w_ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_ones);
        try mlx.check(mlx.mlx_ones(&w_ones, &[_]c_int{ (hc + 1) * d, in_dim }, 2, mdl.act, mdl.s));
        var w_arr = mlx.mlx_array_new();
        {
            errdefer _ = mlx.mlx_array_free(w_arr);
            try mlx.check(mlx.mlx_multiply(&w_arr, w_ones, w_scale, mdl.s));
            try mdl.owned.append(gpa, w_arr);
        }

        var qk_ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qk_ones);
        try mlx.check(mlx.mlx_ones(&qk_ones, &[_]c_int{ hc, d }, 2, .float32, mdl.s));
        var qk_arr = mlx.mlx_array_new();
        {
            errdefer _ = mlx.mlx_array_free(qk_arr);
            try mlx.check(mlx.mlx_multiply(&qk_arr, qk_ones, qk_scale, mdl.s));
            try mdl.owned.append(gpa, qk_arr);
        }

        try mlx.check(mlx.mlx_array_eval(w_arr));
        try mlx.check(mlx.mlx_array_eval(qk_arr));
        mdl.engrams[k] = .{
            .layer = l,
            .wkv = .{ .w = w_arr },
            .qk = qk_arr,
            .table = try engram.Table.open(gpa, dir, k, l, mdl.cfg.dsv41_engram_rows[k], mdl.cfg.dsv41_engram_head_dim),
        };
        mdl.engrams[k].table.openCodes();
        opened += 1;
    }
}

test "deepseek_v41 CED encode interior: cold Engram chunk splits leadRows and matches cold bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try writeEngramPackForTest(testing.allocator, td.dir, 96, .{ 1, 2 });
    var path: [512]u8 = undefined;
    const dir = path[0..try td.dir.realPath(io, &path)];

    var p: [8705]u32 = undefined; // 4096 (cold lead) + 4096 (warm read-ahead) + 512 (cedTail) + 1 (final)
    for (&p, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    const cold_t = try Twin.open(files, "");
    defer cold_t.close();
    const cold_m: *v41.Model = @ptrCast(@alignCast(cold_t.m));
    try attachTinyEngramForTest(cold_m.mdl, dir);
    const w: usize = cold_m.mdl.cfg.sliding_window;
    cold_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(cold_t.m, 1, .of(&p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 4096 }, .{ 4096, 8192 }, .{ 8192, 8704 } }) |r| {
        testing.allocator.free(try slotForward(cold_t, .prefill, &.{1}, p[r[0]..r[1]], &.{@intCast(r[0])}));
    }
    const cold_logits = try slotForward(cold_t, .prefill, &.{1}, p[8704..8705], &.{8704});
    defer testing.allocator.free(cold_logits);
    // Chunk 0 splits at leadRows (1 .encode + 1 .last); chunks 1..2 use read-ahead (1 .last each).
    try testing.expectEqual(@as(u64, 1), cold_m.mdl.ced_stats.encode_chunks);
    try testing.expectEqual(@as(u64, 3), cold_m.mdl.ced_stats.chunks);

    const enc_t = try Twin.open(files, "");
    defer enc_t.close();
    const enc_m: *v41.Model = @ptrCast(@alignCast(enc_t.m));
    try attachTinyEngramForTest(enc_m.mdl, dir);
    enc_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(enc_t.m, 1, .of(&p), .{}, 0, true).isOk());
    try testing.expect(enc_m.mdl.ahead == null);

    // Cold chunk 0 (4096 >= LEAD_MIN, !read_ahead): extendEncodeResumable must split at leadRows -> 2 .encode chunks + ahead for [4096..8192).
    try slotForwardNoLogits(enc_t, .prefill, &.{1}, p[0..4096], &.{0});
    try testing.expectEqual(@as(u64, 2), enc_m.mdl.ced_stats.encode_chunks);
    try testing.expectEqual(@as(u64, 4096), enc_m.mdl.ced_stats.encode_rows);
    try testing.expectEqual(@as(u64, 0), enc_m.mdl.ced_stats.chunks);
    try testing.expect(enc_m.mdl.dec_state.?.decoder_stale);
    try testing.expect(enc_m.mdl.ahead != null);
    try testing.expect(enc_m.mdl.ahead.?.covers(4096, 4096));

    // Warm chunk 1 (4096 covered by read-ahead): runs in 1 .encode chunk and prefetches [8192..8704).
    try slotForwardNoLogits(enc_t, .prefill, &.{1}, p[4096..8192], &.{4096});
    try testing.expectEqual(@as(u64, 3), enc_m.mdl.ced_stats.encode_chunks);
    try testing.expectEqual(@as(u64, 8192), enc_m.mdl.ced_stats.encode_rows);
    try testing.expectEqual(@as(u64, 0), enc_m.mdl.ced_stats.chunks);
    try testing.expect(enc_m.mdl.dec_state.?.decoder_stale);
    try testing.expect(enc_m.mdl.ahead != null);
    try testing.expect(enc_m.mdl.ahead.?.covers(8192, 512));

    // Tail repair chunk 2 + final row 3: bit-identical to cold reference.
    testing.allocator.free(try slotForward(enc_t, .prefill, &.{1}, p[8192..8704], &.{8192}));
    try testing.expect(!enc_m.mdl.dec_state.?.decoder_stale);
    try testing.expectEqual(@as(u64, 1), enc_m.mdl.ced_stats.chunks);
    const enc_logits = try slotForward(enc_t, .prefill, &.{1}, p[8704..8705], &.{8704});
    defer testing.allocator.free(enc_logits);
    try testing.expectEqualSlices(f32, cold_logits, enc_logits);
}

/// The logits of `tail` on a slot restored from `sn` at its end (the prompt `ids`).
fn restoredTail(t: *Twin, aux: *const abi.AuxHooks, sn: abi.Handle, ids: []const u32, tail: []const u32) ![]f32 {
    const n = ids.len - tail.len;
    try testing.expect(aux.restore_at(t.m, 3, sn, n).isOk());
    try testing.expect(aux.begin_prompt(t.m, 3, .of(ids), .{}, n, false).isOk());
    defer v41.table.release_slot.?(t.m, 3);
    return slotForward(t, .prefill, &.{3}, tail, &.{@intCast(n)});
}

fn hostRows(a: mlx.mlx_array) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    return testing.allocator.dupe(f32, (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..mlx.mlx_array_size(f)]);
}

/// A twin kept on the wide prompt path (its compiled regions set aside until close).
const WideTwin = struct {
    t: *Twin,
    regions: @FieldType(std.meta.Child(@FieldType(v41.Model, "mdl")), "regions"),

    fn open(files: anytype) !WideTwin {
        const t = try Twin.open(files, "");
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        const regions = m.mdl.regions;
        m.mdl.regions = regions[0..0];
        return .{ .t = t, .regions = regions };
    }

    fn model(self: WideTwin) *v41.Model {
        return @ptrCast(@alignCast(self.t.m));
    }

    fn close(self: WideTwin) void {
        self.model().mdl.regions = self.regions;
        self.t.close();
    }
};

/// Rows of positions [lo, hi) of two `[cap, width]` arrays (a ring's slots, or rows in order when `cap` covers them).
fn expectRowsEqual(a: mlx.mlx_array, b: mlx.mlx_array, lo: usize, hi: usize, cap: usize) !void {
    const x = try hostRows(a);
    defer testing.allocator.free(x);
    const y = try hostRows(b);
    defer testing.allocator.free(y);
    const width = x.len / @as(usize, @intCast(mlx.getShape(a)[0]));
    for (lo..hi) |p| {
        const s = if (cap == 0) p else p % @max(cap, 1);
        try testing.expectEqualSlices(f32, y[s * width ..][0..width], x[s * width ..][0..width]);
    }
}

fn expectFinite(a: mlx.mlx_array) !void {
    const x = try hostRows(a);
    defer testing.allocator.free(x);
    for (x) |v| try testing.expect(std.math.isFinite(v));
}

test "deepseek_v41 aux: a prompt abandoned mid-prefill leaves a snapshot at its last chunk, and a restore there continues the cold prefill bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [40]u32 = undefined;
    for (&p, 0..) |*id, i| id.* = @intCast((i * 7 + 3) % 90 + 2);
    const cut = 23; // inside a pooling group and past the window: any chunk end is a restore point
    const t = try Twin.open(files, "");
    defer t.close();
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    // Armed for its end, forwarded up to `cut`, then abandoned.
    try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
    var none: abi.Handle = null;
    try testing.expect(aux.snapshot(t.m, 1, &none).isOk());
    try testing.expect(none == null); // nothing forwarded yet
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[0..9], &.{0}));
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[9..cut], &.{9}));
    var sn: abi.Handle = null;
    defer aux.free(t.m, sn);
    try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
    try testing.expect(sn != null);
    var info: abi.AuxSnapshotInfo = .{};
    try testing.expect(aux.describe(t.m, sn, &info).isOk());
    try testing.expectEqual(@as(u64, cut), info.n);
    try testing.expectEqualSlices(u32, p[0..cut], info.ids.slice());
    try testing.expect(aux.snapshot(t.m, 1, &none).isOk());
    try testing.expect(none == null); // handed over once
    v41.table.release_slot.?(t.m, 1);

    // The same prompt resumes from it and still captures its own end.
    try testing.expect(aux.restore_at(t.m, 2, sn, cut).isOk());
    try testing.expect(aux.begin_prompt(t.m, 2, .of(&p), .{}, cut, true).isOk());
    testing.allocator.free(try slotForward(t, .prefill, &.{2}, p[cut .. p.len - 1], &.{cut}));
    const warm = try slotForward(t, .prefill, &.{2}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
    defer testing.allocator.free(warm);
    var end: abi.Handle = null;
    defer aux.free(t.m, end);
    try testing.expect(aux.snapshot(t.m, 2, &end).isOk());
    try testing.expect(aux.describe(t.m, end, &info).isOk());
    try testing.expectEqual(@as(u64, p.len - 1), info.n);

    const reference = try Twin.open(files, "");
    defer reference.close();
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[0..9], &.{0}));
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[9..cut], &.{9}));
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[cut .. p.len - 1], &.{cut}));
    const cold = try slotForward(reference, .prefill, &.{1}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
    defer testing.allocator.free(cold);
    for (cold) |value| try testing.expect(std.math.isFinite(value));
    try testing.expectEqualSlices(f32, cold, warm);
}

test "deepseek_v41 aux: disabled capture and short prompts retain no checkpoint" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
    for ([_][]const u32{ &.{}, &.{3}, &.{ 3, 4, 5 } }) |p| {
        try testing.expect(aux.begin_prompt(t.m, 1, .of(p), .{}, 0, p.len < 2).isOk());
        try testing.expect(aux.reserve(t.m, 1, p.len).isOk());
        if (p.len > 0) testing.allocator.free(try slotForward(t, .prefill, &.{1}, p, &.{0}));
        try testing.expect(aux.commit(t.m, 1, p.len).isOk());
        try testing.expectEqual(abi.Verdict.Code.unsupported, aux.commit(t.m, 1, p.len + 1).code);
        var sn: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
        try testing.expect(sn == null);
        v41.table.release_slot.?(t.m, 1);
    }
}

test "deepseek_v41 aux: candidate image key ignores later rows and eligibility excludes streaming banks and paired instances" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
    m.mdl.cfg.image_token_id = 17;
    const a = mlx.mlx_array_new_data(&[_]f32{ 1, 2, 3, 4 }, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(a);
    const b = mlx.mlx_array_new_data(&[_]f32{ 1, 2, 9, 8 }, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(b);
    var key: [4]u64 = undefined;
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4 }), .{ .ctx = a.ctx }, &key[0]).isOk());
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4 }), .{ .ctx = b.ctx }, &key[1]).isOk());
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4, 17 }), .{ .ctx = a.ctx }, &key[2]).isOk());
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4, 17 }), .{ .ctx = b.ctx }, &key[3]).isOk());
    try testing.expectEqual(key[0], key[1]);
    try testing.expect(key[2] != key[3]);
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4 }), .{}, &key[1]).isOk());
    try testing.expect(key[0] != key[1]);
    try testing.expect(aux.enabled(t.m));
    m.cfg.dsv41_streamed = true;
    try testing.expect(aux.enabled(t.m));
    m.cfg.dsv41_streamed = false;
    m.bank_streams = true;
    try testing.expect(!aux.enabled(t.m));
    m.bank_streams = false;
    try testing.expect(v41.bankStreams(146, 183, 384));
    try testing.expect(v41.bankStreams(384, 200, 384));
    try testing.expect(!v41.bankStreams(384, 384, 384));
    m.mdl.pair = .{ .lo = 2, .hi = 4, .worker = false };
    defer m.mdl.pair = null;
    try testing.expect(!aux.enabled(t.m));
    try testing.expectEqual(abi.Verdict.Code.unsupported, aux.begin_prompt(t.m, 1, .of(&.{ 3, 4 }), .{}, 0, true).code);
}

const codec = @import("cache_codec.zig");
const dsv41 = @import("deepseek_v41.zig");

const Exported = struct {
    tensors: [codec.max_tensors]abi.AuxTensor = undefined,
    names: [16 * 1024]u8 = undefined,
    header: [codec.max_header]u8 = undefined,
    out: abi.AuxExport = .{},

    fn capture(self: *Exported, t: *Twin, p: []const u32) !void {
        const aux = v41.table.cache.aux.?;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(p), .{}, 0, true).isOk());
        const first = @min(2, p.len - 1);
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[0..first], &.{0}));
        if (first < p.len - 1) testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[first .. p.len - 1], &.{@intCast(first)}));
        var sn: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
        defer aux.free(t.m, sn);
        try testing.expect(sn != null);
        self.out = .{ .tensors = &self.tensors, .tensor_cap = self.tensors.len, .names = &self.names, .names_cap = self.names.len, .header = &self.header, .header_cap = self.header.len };
        const v = aux.export_arrays(t.m, sn, &self.out);
        if (!v.isOk()) std.debug.print("export: {s}\n", .{v.why.slice()});
        try testing.expect(v.isOk());
        v41.table.release_slot.?(t.m, 1);
    }

    fn deinit(self: *Exported) void {
        for (self.arrays()) |t| _ = mlx.mlx_array_free(.{ .ctx = t.array.ctx });
        self.out.tensor_len = 0;
    }

    fn arrays(self: *Exported) []abi.AuxTensor {
        return self.tensors[0..self.out.tensor_len];
    }

    fn bytes(self: *Exported) abi.Str {
        return .of(self.header[0..self.out.header_len]);
    }
};

fn expectImportRefused(t: *Twin, e: *Exported, ids: []const u32, arrays: []const abi.AuxTensor) !void {
    var sn: abi.Handle = null;
    const v = v41.table.cache.aux.?.import_arrays(t.m, .of(ids), e.bytes(), .of(arrays), &sn);
    defer v41.table.cache.aux.?.free(t.m, sn);
    try testing.expectEqual(abi.Verdict.Code.failed, v.code);
    try testing.expect(v.why.len > 0 and sn == null);
}

fn expectInvalidImport(t: *Twin, e: *Exported, ids: []const u32, arrays: []const abi.AuxTensor) !void {
    try expectImportRefused(t, e, ids, arrays);
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    const unexpected = dsv41.importSnapshot(m.mdl, failing.allocator(), ids, e.bytes().slice(), arrays) catch {
        try testing.expect(!failing.has_induced_failure);
        return;
    };
    unexpected.deinit(failing.allocator());
    failing.allocator().destroy(unexpected);
    return error.InvalidTensorAccepted;
}

test "codec rejects duplicate, missing, unexpected and malformed tensors before construction" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41DsparkFiles(arena.allocator(), 7), "");
    defer t.close();
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const ids = p[0 .. p.len - 1];
    try expectInvalidImport(t, &e, ids, e.arrays()[0 .. e.arrays().len - 1]);
    const last = e.arrays().len - 1;
    const saved = e.tensors[last];
    e.tensors[last] = e.tensors[0];
    try expectInvalidImport(t, &e, ids, e.arrays());
    e.tensors[last] = saved;
    for (e.arrays()) |*tensor| {
        const original = tensor.*;
        tensor.name = .of("unexpected");
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        tensor.array = .{};
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        tensor.size = 0;
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        const x: mlx.mlx_array = .{ .ctx = tensor.array.ctx };
        const shape = mlx.getShape(x);
        var wrong = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wrong);
        try mlx.check(mlx.mlx_zeros(&wrong, shape.ptr, shape.len, .int32, t.s));
        tensor.array = .{ .ctx = wrong.ctx };
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        var dims: [2]c_int = .{ 1, 1 };
        @memcpy(dims[0..shape.len], shape);
        dims[0] += 1;
        try mlx.check(mlx.mlx_zeros(&wrong, &dims, 2, mlx.mlx_array_dtype(x), t.s));
        tensor.array = .{ .ctx = wrong.ctx };
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        try mlx.check(mlx.mlx_zeros(&wrong, &.{1}, 1, mlx.mlx_array_dtype(x), t.s));
        tensor.array = .{ .ctx = wrong.ctx };
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
    }
    try expectInvalidImport(t, &e, ids[1..], e.arrays());
    try testing.expectEqual(@as(u64, 0), v41.stateBytes(t.m));
}

test "codec import unwinds every allocation and checked MLX failure" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41DsparkFiles(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const ids = p[0 .. p.len - 1];
    for (e.arrays()) |x| try mlx.check(mlx.mlx_array_eval(.{ .ctx = x.array.ctx }));
    try mlx.check(mlx.mlx_synchronize(t.s));
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    for (0..8) |at| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = at });
        const sn = dsv41.importSnapshot(m.mdl, failing.allocator(), ids, e.bytes().slice(), e.arrays()) catch |err| {
            try testing.expectEqual(error.OutOfMemory, err);
            try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            continue;
        };
        sn.deinit(failing.allocator());
        failing.allocator().destroy(sn);
        try testing.expect(!failing.has_induced_failure);
    }
    var completed = false;
    for (1..512) |at| {
        mlx.fault.arm(at);
        var sn: abi.Handle = null;
        const v = v41.table.cache.aux.?.import_arrays(t.m, .of(ids), e.bytes(), .of(e.arrays()), &sn);
        const fired = mlx.fault.didFire();
        mlx.fault.disarm();
        defer v41.table.cache.aux.?.free(t.m, sn);
        if (!fired) {
            try testing.expect(v.isOk() and sn != null);
            completed = true;
            break;
        }
        try testing.expectEqual(abi.Verdict.Code.failed, v.code);
        try testing.expect(sn == null);
    }
    try testing.expect(completed);
    try mlx.check(mlx.mlx_synchronize(t.s));
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    try testing.expectEqual(before, after);
    try testing.expectEqual(@as(u64, 0), v41.stateBytes(t.m));
}

test "optional codec failure preserves the original MLX error latch" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const original = "[matmul] original codec-test failure";
    mlx.latchErrorForTest(original);
    defer {
        var buf: [512]u8 = undefined;
        _ = mlx.takeError(&buf);
    }
    mlx.fault.arm(3);
    defer mlx.fault.disarm();
    try expectImportRefused(t, &e, p[0 .. p.len - 1], e.arrays());
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(original, mlx.takeError(&buf).?);
    mlx.fault.disarm();
    mlx.armLatchingFaultForTest(3);
    try expectImportRefused(t, &e, p[0 .. p.len - 1], e.arrays());
    try testing.expect(mlx.latchingFaultFiredForTest());
    try testing.expect(!mlx.errorPending());
}

fn codecRoundTrip(files: []const sdk.fixture.File, p: []const u32, dspark: bool) !void {
    const aux = v41.table.cache.aux.?;
    var e: Exported = .{};
    defer e.deinit();
    {
        const origin = try Twin.open(files, "");
        defer origin.close();
        try e.capture(origin, p);
    }
    const t = try Twin.open(files, "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    try testing.expectEqual(@as(usize, if (dspark) 2 else 0), m.mdl.n_mtp);
    var peak: u64 = 0;
    try testing.expect(aux.import_plan(t.m, e.bytes(), &peak).isOk());
    try testing.expect(peak > 0);
    std.mem.reverse(abi.AuxTensor, e.arrays());
    var sn: abi.Handle = null;
    const input_ids = try testing.allocator.dupe(u32, p[0 .. p.len - 1]);
    defer testing.allocator.free(input_ids);
    try testing.expect(aux.import_arrays(t.m, .of(input_ids), e.bytes(), .of(e.arrays()), &sn).isOk());
    defer aux.free(t.m, sn);
    e.deinit();
    @memset(input_ids, 0);
    @memset(&e.header, 0);
    @memset(&e.names, 0);
    var info: abi.AuxSnapshotInfo = .{};
    try testing.expect(aux.describe(t.m, sn, &info).isOk());
    try testing.expectEqual(p.len - 1, info.n);
    try testing.expectEqualSlices(u32, p[0 .. p.len - 1], info.ids.slice());
    try testing.expect(peak > aux.bytes(t.m, sn));
    try testing.expect(aux.restore_at(t.m, 1, sn, info.n).isOk());
    var caps: [64][2]usize = @splat(.{ 0, 0 });
    for (m.mdl.dec_state.?.layers, 0..) |l, i| caps[i] = .{ if (l.comp) |r| r.cap else 0, if (l.idx_k) |r| r.cap else 0 };
    try testing.expect(aux.begin_prompt(t.m, 1, .of(p), .{}, info.n, false).isOk());
    const warm = try slotForward(t, .prefill, &.{1}, p[info.n..], &.{@intCast(info.n)});
    defer testing.allocator.free(warm);
    for (m.mdl.dec_state.?.layers, 0..) |l, i| {
        if (l.comp) |r| try testing.expectEqual(caps[i][0], r.cap);
        if (l.idx_k) |r| try testing.expectEqual(caps[i][1], r.cap);
    }
    const reference = try Twin.open(files, "");
    defer reference.close();
    const first = @min(2, p.len - 1);
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[0..first], &.{0}));
    if (first < p.len - 1) testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[first .. p.len - 1], &.{@intCast(first)}));
    const cold = try slotForward(reference, .prefill, &.{1}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
    defer testing.allocator.free(cold);
    for (cold) |x| try testing.expect(std.math.isFinite(x));
    try testing.expectEqualSlices(f32, cold, warm);
}

test "codec round trip survives origin destruction at odd and even compressor boundaries with DSpark on and off" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var p: [27]u32 = undefined;
    for (&p, 0..) |*id, i| id.* = @intCast((i * 13 + 3) % 96);
    for ([_]bool{ false, true }) |dspark| {
        const files = if (dspark) try fixture.v41DsparkFiles(arena.allocator(), 7) else try fixture.v41Files(arena.allocator(), 7);
        for ([_]usize{ 2, 3, 8, 9, 26, 27 }) |len| try codecRoundTrip(files, p[0..len], dspark);
    }
}

test "codec refuses changed loaded settings and invalid IDs before allocating" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41DsparkFiles(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const ids = p[0 .. p.len - 1];
    const cfg = m.mdl.cfg;
    defer m.mdl.cfg = cfg;
    for (0..5) |change| {
        switch (change) {
            0 => m.mdl.cfg.sliding_window += 1,
            1 => m.mdl.cfg.dsv4_dspark_target_layers[0] += 1,
            2 => m.mdl.cfg.quant_bits += 1,
            3 => m.mdl.cfg.dsv41_engram_compressed_vocab += 1,
            4 => m.mdl.cfg.dsv4_compress_ratios[2] += 1,
            else => unreachable,
        }
        var peak: u64 = 123;
        const verdict = v41.table.cache.aux.?.import_plan(t.m, e.bytes(), &peak);
        try testing.expectEqual(abi.Verdict.Code.failed, verdict.code);
        try testing.expectEqual(@as(u64, 0), peak);
        try expectInvalidImport(t, &e, ids, e.arrays());
        m.mdl.cfg = cfg;
    }
    const stages = m.mdl.n_mtp;
    m.mdl.n_mtp = 0;
    try expectInvalidImport(t, &e, ids, e.arrays());
    m.mdl.n_mtp = stages;
    var invalid = p;
    invalid[0] = @intCast(m.mdl.vocab);
    try expectInvalidImport(t, &e, invalid[0..ids.len], e.arrays());
    try expectInvalidImport(t, &e, ids[1..], e.arrays());
}

test "codec export refuses short buffers and unwinds partially created references" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const aux = v41.table.cache.aux.?;
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, &p, &.{0}));
    var sn: abi.Handle = null;
    try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
    defer aux.free(t.m, sn);
    var e: Exported = .{};
    const storage: abi.AuxExport = .{ .tensors = &e.tensors, .tensor_cap = e.tensors.len, .names = &e.names, .names_cap = e.names.len, .header = &e.header, .header_cap = e.header.len };
    for (0..3) |small| {
        var out = storage;
        switch (small) {
            0 => out.tensor_cap = 1,
            1 => out.names_cap = 1,
            2 => out.header_cap = 1,
            else => unreachable,
        }
        try testing.expectEqual(abi.Verdict.Code.failed, aux.export_arrays(t.m, sn, &out).code);
        try testing.expectEqual(@as(usize, 0), out.tensor_len + out.names_len + out.header_len);
    }
    var completed = false;
    for (1..512) |at| {
        var out = storage;
        mlx.fault.arm(at);
        const v = aux.export_arrays(t.m, sn, &out);
        const fired = mlx.fault.didFire();
        mlx.fault.disarm();
        if (!fired) {
            try testing.expect(v.isOk());
            e.out = out;
            e.deinit();
            completed = true;
            break;
        }
        try testing.expectEqual(abi.Verdict.Code.failed, v.code);
        try testing.expectEqual(@as(usize, 0), out.tensor_len + out.names_len + out.header_len);
    }
    try testing.expect(completed);
}

const engram = @import("engram.zig");

test "codec preserves compressed Engram history including dead sentinels through export, import and restore" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const pm: *v41.Model = @ptrCast(@alignCast(t.m));
    const m = pm.mdl;
    const ids = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    try dsv41.reset(m, gpa);
    try dsv41.armPrompt(m, &ids, null, true);
    _ = mlx.mlx_array_free(try dsv41.extendResumable(m, gpa, &ids, null));
    const sn = &m.dec_state.?.snap.?;
    const history = [_]u32{ 9, engram.Hash.DEAD, 2, 4, engram.Hash.DEAD, 7, 1 };
    // The tiny pack has no Engram projections; supply its already-compressed history at the codec boundary.
    m.hash = .{ .n_layers = 0, .max_ngram = 2, .heads = 1, .pad = 0 };
    sn.hist = try gpa.dupe(u32, &history);
    var tensors: [codec.max_tensors]abi.AuxTensor = undefined;
    var names: [16 * 1024]u8 = undefined;
    var header: [codec.max_header]u8 = undefined;
    var out: abi.AuxExport = .{ .tensors = &tensors, .tensor_cap = tensors.len, .names = &names, .names_cap = names.len, .header = &header, .header_cap = header.len };
    try dsv41.exportSnapshot(m, sn, &out);
    defer for (tensors[0..out.tensor_len]) |x| {
        _ = mlx.mlx_array_free(.{ .ctx = x.array.ctx });
    };
    m.dec_state.?.deinit();
    m.dec_state = null;
    try testing.checkAllAllocationFailures(gpa, importForAllocationTest, .{ m, ids[0..history.len], header[0..out.header_len], tensors[0..out.tensor_len] });
    const imported = try dsv41.importSnapshot(m, gpa, ids[0..history.len], header[0..out.header_len], tensors[0..out.tensor_len]);
    defer {
        imported.deinit(gpa);
        gpa.destroy(imported);
    }
    try testing.expectEqualSlices(u32, &history, imported.hist);
    var restored = try dsv41.stateFromSnapshot(m, gpa, imported);
    defer restored.deinit();
    try testing.expectEqualSlices(u32, &history, restored.hist.items);
}

fn importForAllocationTest(gpa: std.mem.Allocator, m: *const dsv41.Dsv41Model, ids: []const u32, header: []const u8, arrays: []const abi.AuxTensor) !void {
    const sn = try dsv41.importSnapshot(m, gpa, ids, header, arrays);
    defer gpa.destroy(sn);
    defer sn.deinit(gpa);
}
