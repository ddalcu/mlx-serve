//! The Qwen family's parity harness: one tiny random pack on disk under a plugin's table, each forward step's logits
//! bit-equal to the legacy `Transformer`'s, recorded to a tape (`tape.zig`) beside a second load of the same tensors
//! with every recurrent state compared too. Shared by the qwen3 and qwen3_5 proofs.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const sdk = @import("../../sdk/root.zig");
const model = @import("../../sdk/config/model.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const tape_mod = @import("../../engines/mlx/plugin/tape.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

pub const Step = struct { mode: abi.Mode, ids: []const i32 };

/// A pack loaded through plugin `P`'s table, against its tape.
pub fn Twin(comptime P: type) type {
    return struct {
        tmp: testing.TmpDir,
        tape: tape_mod.Tape,
        res: Resident,
        parsed: abi.Parsed = .{},
        m: *anyopaque,

        /// Writes `p` with `config_json` (and `files` beside it) and loads it; `recorded` is its tape.
        pub fn open(p: *fixture.Pack, config_json: []const u8, files: []const sdk.fixture.File, recorded: []const u8) !*@This() {
            const a = testing.allocator;
            const io = std.Io.Threaded.global_single_threaded.io();
            const t = try a.create(@This());
            errdefer a.destroy(t);
            t.* = .{ .tmp = testing.tmpDir(.{}), .tape = try tape_mod.Tape.player(recorded), .res = undefined, .m = undefined };
            errdefer t.tmp.cleanup();
            var buf: [512]u8 = undefined;
            const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
            try p.save(dir);
            for (files) |f| try t.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
            var cfg = try model.parseConfigFromJson(a, config_json);
            defer cfg.deinit(a);
            t.res = Resident.adopt(try model.loadModelWeights(io, a, dir, &cfg, false));
            errdefer t.res.deinit();
            var all: [8]sdk.fixture.File = undefined;
            all[0] = .{ .name = "config.json", .bytes = config_json };
            @memcpy(all[1 .. 1 + files.len], files);
            const mem: sdk.fixture.MemDir = .{ .files = all[0 .. 1 + files.len] };
            const pd = mem.packDir();
            const v = P.table.parse(&pd, .of(config_json), .{}, null, false, &t.parsed);
            if (!v.isOk()) {
                std.debug.print("parse: {s}\n", .{v.why.slice()});
                return error.PluginParse;
            }
            errdefer P.table.free_parsed(&t.parsed);
            const src = t.res.source();
            var why: abi.Verdict = .{};
            t.m = P.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = fixture.stream().ctx }, .weights = &src }, &why) orelse {
                std.debug.print("load: {s}\n", .{why.why.slice()});
                return error.PluginLoad;
            };
            return t;
        }

        pub fn close(t: *@This()) void {
            P.table.unload(t.m);
            P.table.free_parsed(&t.parsed);
            t.res.deinit();
            t.tmp.cleanup();
            testing.allocator.destroy(t);
        }

        /// A playback read every recorded value (a proof's last call).
        pub fn expectDone(t: *const @This()) !void {
            try t.tape.expectDone();
        }

        /// One request.
        pub fn lane(t: *@This()) !*Lane {
            return Lane.init(t.parsed.shell.num_hidden_layers);
        }

        /// One step on `l`: the table's logits against the tape.
        pub fn step(t: *@This(), l: *Lane, st: Step) !void {
            var g = ops.Graph.init(testing.allocator, fixture.stream());
            defer g.deinit();
            const host = try testing.allocator.alloc(u32, st.ids.len);
            defer testing.allocator.free(host);
            for (st.ids, host) |x, *y| y.* = @intCast(x);
            const handles = [_]abi.Handle{l.slot.handle()};
            const pos = [_]u32{l.pos};
            l.pos += @intCast(st.ids.len);
            var out: abi.ForwardOut = .{};
            const v = P.table.forward(t.m, &.{ .mode = st.mode, .logits = .all_rows, .rows = @intCast(st.ids.len), .tokens = .of(host), .positions = .of(&pos), .caches = .of(&handles), .cache_ops = &cache_ops.ops }, &out);
            if (!v.isOk()) {
                std.debug.print("forward: {s}\n", .{v.why.slice()});
                return error.PluginForward;
            }
            try t.tape.expectNext(try g.adopt(.{ .ctx = out.logits.ctx }), false);
        }

        /// Two lanes stepped together (`rows` tokens each): the table's batched rows against the tape.
        pub fn batched(t: *@This(), lanes: [2]*Lane, tick: [2][]const i32) !void {
            const a = testing.allocator;
            var g = ops.Graph.init(a, fixture.stream());
            defer g.deinit();
            const rows = tick[0].len;
            const positions = [_]u32{ lanes[0].pos, lanes[1].pos };
            var flat: [16]i32 = undefined;
            var host: [16]u32 = undefined;
            @memcpy(flat[0..rows], tick[0]);
            @memcpy(flat[rows .. 2 * rows], tick[1]);
            for (flat[0 .. 2 * rows], host[0 .. 2 * rows]) |x, *y| y.* = @intCast(x);
            const handles = [_]abi.Handle{ lanes[0].slot.handle(), lanes[1].slot.handle() };
            const slots = [_]u32{ 0, 1 };
            var out: abi.ForwardOut = .{};
            const v = P.table.forward(t.m, &.{ .mode = .batched_rows, .logits = .all_rows, .rows = @intCast(rows), .tokens = .of(host[0 .. 2 * rows]), .slots = .of(&slots), .positions = .of(&positions), .caches = .of(&handles), .cache_ops = &cache_ops.ops }, &out);
            if (!v.isOk()) {
                std.debug.print("batched forward: {s}\n", .{v.why.slice()});
                return error.PluginForward;
            }
            try t.tape.expectNext(try g.adopt(.{ .ctx = out.logits.ctx }), false);
            for (lanes) |l| l.pos += @intCast(rows);
        }
    };
}

/// A request: the plugin's cache and recurrent entries.
pub const Lane = struct {
    /// The request's next position.
    pos: u32 = 0,
    kv: KVCache,
    ssm: []SSMCacheEntry,
    slot: cache_ops.Slot = undefined,

    fn init(n: u32) !*Lane {
        const l = try testing.allocator.create(Lane);
        l.* = .{ .kv = try KVCache.init(testing.allocator, n), .ssm = try fixture.newEntries(n) };
        l.slot = .{ .kv = &l.kv, .ssm = l.ssm };
        return l;
    }

    pub fn deinit(l: *Lane) void {
        l.kv.deinit();
        fixture.freeEntries(l.ssm);
        testing.allocator.destroy(l);
    }
};

/// `n` deterministic ids below 256.
pub fn seq(comptime n: usize, seed: u32) [n]i32 {
    var out: [n]i32 = undefined;
    for (&out, 0..) |*v, i| v.* = @intCast((seed + 37 * @as(u32, @intCast(i))) % 256);
    return out;
}
