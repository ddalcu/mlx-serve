//! Parity proof for the V4.1 plugin: its forward on a tiny random pack equals the deleted legacy seam at every step,
//! its verify block the module's own all-rows extend, and a resumed prompt the legacy resume, as recorded in the tapes
//! in `fixtures/` (`../deepseek_v4/proof_test.zig`).

const std = @import("std");
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
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/steps.tape"));
    defer t.close();
    try proof.expectSteps(v41, t);
}

test "deepseek_v41 plugin: excluded configuration keeps the legacy resume and continues bit-equal" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const first = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    const second = first ++ [_]u32{ 11, 2, 90 };
    for ([_]bool{ false, true }) |release_parked| {
        const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/proof-resume.tape"));
        defer t.close();
        const model: *v41.Model = @ptrCast(@alignCast(t.m));
        model.cfg.dsv41_streamed = true;
        for ([_][]const u32{ &first, &second }, 0..) |prompt, turn| {
            const n: u32 = @intCast(v41.resumePrompt(t.m, 7, .of(prompt), .{}));
            if (turn == 1) try testing.expect(n > 0);
            try t.check(mlx.mlx_array_new_data(&n, &[_]c_int{1}, 1, .uint32));
            try t.check(try t.plugin(.{ .mode = .prefill, .ids = prompt[n..] }, n));
            if (turn == 0 and release_parked)
                testing.allocator.free(try slotForward(t, .prefill, &.{8}, &.{ 4, 12, 6 }, &.{0}));
            v41.table.release_slot.?(t.m, 7);
            if (turn == 0 and !release_parked)
                testing.allocator.free(try slotForward(t, .prefill, &.{8}, &.{ 4, 12, 6 }, &.{0}));
        }
        try t.tape.expectDone();
        testing.allocator.free(try slotForward(t, .decode, &.{8}, &.{13}, &.{3}));
    }
}

test "deepseek_v41 plugin: a decode token on the device equals the legacy decode" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/device-decode.tape"));
    defer t.close();
    try proof.expectDeviceDecode(v41, t);
}

test "deepseek_v41 plugin: trunk_weight hands out the loaded embedding and head" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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

/// The plugin's logits `[rows, V]` (host f32, caller frees) for `toks` of `slots` at `positions`.
fn slotForward(t: *Twin, mode: abi.Mode, slots: []const u32, toks: []const u32, positions: []const u32) ![]f32 {
    var out: abi.ForwardOut = .{};
    const rows: u32 = if (mode == .batched_rows) 1 else @intCast(toks.len);
    const v = v41.table.forward(t.m, &.{ .mode = mode, .rows = rows, .tokens = .of(toks), .slots = .of(slots), .positions = .of(positions) }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    const l: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(l);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, l, .float32, t.s));
    try mlx.check(mlx.mlx_array_eval(f));
    return testing.allocator.dupe(f32, (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..mlx.mlx_array_size(f)]);
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
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
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
            try testing.expectEqual(abi.Verdict.Code.unsupported, aux.slice(t.m, sn, info.n - 1, &clone).code);
            try testing.expect(clone == null);
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

test "deepseek_v41 aux: disabled capture and short prompts retain no checkpoint, codecs refuse by name" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
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
    var buf: [1]u8 = undefined;
    var len: usize = 99;
    const encoded = aux.serialize(t.m, null, &buf, buf.len, &len);
    try testing.expectEqual(abi.Verdict.Code.unsupported, encoded.code);
    try testing.expect(encoded.why.len > 0);
    try testing.expectEqual(@as(usize, 0), len);
    var sn: abi.Handle = null;
    const decoded = aux.deserialize(t.m, .of(&.{}), &sn);
    try testing.expectEqual(abi.Verdict.Code.unsupported, decoded.code);
    try testing.expect(decoded.why.len > 0 and sn == null);
}

test "deepseek_v41 aux: candidate image key ignores later rows and eligibility excludes streamed and paired instances" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
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
    try testing.expect(!aux.enabled(t.m));
    try testing.expectEqual(abi.Verdict.Code.unsupported, aux.begin_prompt(t.m, 1, .of(&.{ 3, 4 }), .{}, 0, true).code);
    m.cfg.dsv41_streamed = false;
    m.mdl.pair = .{ .lo = 2, .hi = 4, .worker = false };
    defer m.mdl.pair = null;
    try testing.expect(!aux.enabled(t.m));
}

test "deepseek_v41 slot request: Engram prompt and confidence follow the State across slots" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const a = [_]u32{ 3, 17, 42, 9 };
    const b = [_]u32{ 5, 8, 13, 21 };
    for ([_]bool{ false, true }) |legacy| {
        const t = try Twin.open(files, "");
        defer t.close();
        const model: *v41.Model = @ptrCast(@alignCast(t.m));
        model.cfg.dsv41_streamed = legacy;
        const aux = v41.table.cache.aux.?;
        if (legacy) {
            try testing.expectEqual(@as(u64, 0), v41.resumePrompt(t.m, 1, .of(&a), .{}));
        } else try testing.expect(aux.begin_prompt(t.m, 1, .of(&a), .{}, 0, true).isOk());
        model.mdl.dec_state.?.ds_want_conf = true;
        if (legacy) {
            try testing.expectEqual(@as(u64, 0), v41.resumePrompt(t.m, 2, .of(&b), .{}));
        } else try testing.expect(aux.begin_prompt(t.m, 2, .of(&b), .{}, 0, true).isOk());
        model.mdl.dec_state.?.ds_want_conf = false;
        try testing.expect(v41.table.activate_slot.?(t.m, 1).isOk());
        try testing.expectEqualSlices(u32, &a, model.mdl.dec_state.?.prompt);
        try testing.expect(model.mdl.dec_state.?.ds_want_conf);
        try testing.expect(v41.table.activate_slot.?(t.m, 2).isOk());
        try testing.expectEqualSlices(u32, &b, model.mdl.dec_state.?.prompt);
        try testing.expect(!model.mdl.dec_state.?.ds_want_conf);
    }
}

test "deepseek_v41 slot request: a fresh prompt clears confidence requested by the previous turn" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const model: *v41.Model = @ptrCast(@alignCast(t.m));
    model.cfg.dsv41_streamed = true;
    _ = v41.resumePrompt(t.m, 1, .of(&.{ 3, 17, 42 }), .{});
    model.mdl.dec_state.?.ds_want_conf = true;
    _ = v41.resumePrompt(t.m, 1, .of(&.{ 5, 8, 13 }), .{});
    try testing.expect(!model.mdl.dec_state.?.ds_want_conf);
}
