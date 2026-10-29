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
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/steps.tape"));
    defer t.close();
    try proof.expectSteps(v41, t);
}

test "deepseek_v41 plugin: excluded configuration keeps the legacy resume and continues bit-equal" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
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
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
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

test "deepseek_v41 aux: candidate image key ignores later rows and eligibility excludes streamed and paired instances" {
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
    try testing.expect(!aux.enabled(t.m));
    try testing.expectEqual(abi.Verdict.Code.unsupported, aux.begin_prompt(t.m, 1, .of(&.{ 3, 4 }), .{}, 0, true).code);
    m.cfg.dsv41_streamed = false;
    m.mdl.pair = .{ .lo = 2, .hi = 4, .worker = false };
    defer m.mdl.pair = null;
    try testing.expect(!aux.enabled(t.m));
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
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
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
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
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
