//! The kernel routes' host test backend (`Trace`) and the helpers their tests share, over a pinned registry `R`
//! (G5): every launch, eval, join and graph op recorded with its inputs, no MLX. Test code only.

const std = @import("std");
const mlx = @import("sdk").mlx;
const kernel_routes = @import("kernel_routes.zig");
const QuantMode = @import("sdk").QuantMode;

/// The trace backend and the test helpers over registry `R` (`kernel_routes.Routes(R)`'s, plus `Sample`,
/// `TemplateArg`, `max_outputs`, `Registry`, `embedded` and `manifest_sha256`).
pub fn KernelTrace(comptime R: type) type {
    return struct {
        const xk = R;
        const kr = kernel_routes.Routes(R);

        const testing = std.testing;
        const Allocator = std.mem.Allocator;
        const Kernel = xk.Kernel;
        const Entry = xk.Entry;
        const Vars = xk.Vars;
        const LaunchConfig = xk.LaunchConfig;
        const Dtype = mlx.mlx_dtype;
        const Shape = kr.Shape;
        const argOf = kr.argOf;
        const RowPlans = kr.RowPlans;
        const rowsVars = kr.rowsVars;

        /// Host-only backend: nodes of shape + dtype (host arrays keep their bytes), every launch
        /// recorded with its inputs and outputs, every node with its origin, and the launches, evals
        /// and joins in one ordered log. Nothing reaches MLX.
        pub const Trace = struct {
            pub const T = u32;
            pub const Origin = union(enum) { none, host, ext: []const u8, out: [2]u32, view: T, cat: u32, take: u32, op: u32 };
            pub const Node = struct { shape: Shape, dtype: Dtype, bytes: []u8, origin: Origin = .none };
            pub const Launch = struct { k: Kernel, cfg: LaunchConfig, inputs: [xk.max_inputs]T = undefined, n_in: usize, outs: [xk.max_outputs]T = undefined, prepared: bool = false };
            pub const Ev = union(enum) { launch: u32, eval: []T, async_eval: []T, concat: []T, take: [2]T, op: u32 };
            /// A graph op outside the kernel launches (the C2 adapter's gathers and activation): its
            /// kind, inputs and parameters, in the log as `op`.
            pub const Op = struct { kind: []const u8, ins: [5]T = undefined, n_in: u8 = 0, out: T = 0, f: f64 = 0, bits: u32 = 0, group: u32 = 0, mode: []const u8 = "", sorted: bool = false };
            pub const Freed = struct { from: u32, to: u32, at: u32 };

            a: Allocator,
            nodes: std.ArrayList(Node) = .empty,
            launches: std.ArrayList(Launch) = .empty,
            log: std.ArrayList(Ev) = .empty,
            cats: u32 = 0,
            takes: u32 = 0,
            /// the graph ops (`Op`), in call order
            ops: std.ArrayList(Op) = .empty,
            keeps: isize = 0,
            /// node ranges a `resetTo` freed, [from, to), with the log length when it ran, in call order
            freed: std.ArrayList(Freed) = .empty,
            /// nodes with a kept handle (one entry per keep)
            held: std.ArrayList(T) = .empty,
            /// every keep, in order, with the number of resets before it (append-only)
            kept: std.ArrayList(struct { node: T, resets: u32 }) = .empty,
            /// the model backend's kernel launcher (`MlxOps.launcher: ?*const xk.Bound`), set by `kernel_set.Set.install`
            launcher: ?*const xk.Bound = null,
            /// prepared launches not yet released; launches made from a prepared config
            prepared_live: isize = 0,
            prepared_launches: usize = 0,

            pub fn deinit(t: *Trace) void {
                for (t.nodes.items) |n| t.a.free(n.bytes);
                t.nodes.deinit(t.a);
                t.launches.deinit(t.a);
                for (t.log.items) |e| switch (e) {
                    .eval, .async_eval, .concat => |xs| t.a.free(xs),
                    else => {},
                };
                t.log.deinit(t.a);
                t.freed.deinit(t.a);
                t.held.deinit(t.a);
                t.kept.deinit(t.a);
                t.ops.deinit(t.a);
            }

            pub fn mark(t: *const Trace) u32 {
                return @intCast(t.nodes.items.len);
            }

            pub fn resetTo(t: *Trace, m: u32) void {
                t.freed.append(t.a, .{ .from = m, .to = @intCast(t.nodes.items.len), .at = @intCast(t.log.items.len) }) catch @panic("trace: out of memory");
            }

            /// A node no `resetTo` freed and no kept handle holds (a leak past the call).
            pub fn leaked(t: *const Trace, x: T) bool {
                for (t.freed.items) |r| if (x >= r.from and x < r.to) return false;
                return std.mem.indexOfScalar(T, t.held.items, x) == null;
            }

            /// The node a view resolves to (reshape chains to their source).
            pub fn root(t: *const Trace, x: T) T {
                var n = x;
                while (t.nodes.items[n].origin == .view) n = t.nodes.items[n].origin.view;
                return n;
            }

            pub fn node(t: *Trace, shape: []const c_int, dt: Dtype, bytes: []const u8) !T {
                const copy = try t.a.dupe(u8, bytes);
                errdefer t.a.free(copy);
                try t.nodes.append(t.a, .{ .shape = Shape.of(shape), .dtype = dt, .bytes = copy });
                return @intCast(t.nodes.items.len - 1);
            }

            pub fn with(t: *Trace, x: T, origin: Origin) T {
                t.nodes.items[x].origin = origin;
                return x;
            }

            /// A caller array named `name` (a lane's ext: reference).
            pub fn ext(t: *Trace, name: []const u8, shape: []const c_int, dt: Dtype) !T {
                return t.with(try t.node(shape, dt, &.{}), .{ .ext = name });
            }

            pub fn shapeOf(t: *Trace, x: T) Shape {
                return t.nodes.items[x].shape;
            }

            pub fn dtypeOf(t: *Trace, x: T) Dtype {
                return t.nodes.items[x].dtype;
            }

            pub fn hostArray(t: *Trace, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
                return t.with(try t.node(shape, dt, bytes), .host);
            }

            pub fn keep(t: *Trace, x: T) T {
                t.keeps += 1;
                t.held.append(t.a, x) catch @panic("trace: out of memory");
                t.kept.append(t.a, .{ .node = x, .resets = @intCast(t.freed.items.len) }) catch @panic("trace: out of memory");
                return x;
            }

            pub fn release(t: *Trace, x: T) void {
                t.keeps -= 1;
                if (std.mem.indexOfScalar(T, t.held.items, x)) |i| _ = t.held.swapRemove(i);
            }

            pub fn reshape(t: *Trace, x: T, shape: []const c_int) !T {
                return t.with(try t.node(shape, t.nodes.items[x].dtype, &.{}), .{ .view = x });
            }

            pub fn astype(t: *Trace, x: T, dt: Dtype) !T {
                return t.node(t.nodes.items[x].shape.slice(), dt, &.{});
            }

            /// A prepared launch (the model backend's `xk.Prepared`): the config, kept by value.
            pub const Prepared = struct { k: Kernel, cfg: LaunchConfig };

            pub fn prepareLaunch(t: *Trace, k: Kernel, cfg: *const LaunchConfig) !Prepared {
                t.prepared_live += 1;
                return .{ .k = k, .cfg = cfg.* };
            }

            /// Records exactly what `launch(p.k, inputs, &p.cfg, out)` records, marked prepared.
            pub fn launchPrepared(t: *Trace, p: *const Prepared, inputs: []const T, out: []T) !void {
                t.prepared_launches += 1;
                try t.launch(p.k, inputs, &p.cfg, out);
                t.launches.items[t.launches.items.len - 1].prepared = true;
            }

            pub fn releasePrepared(t: *Trace, _: *Prepared) void {
                t.prepared_live -= 1;
            }

            pub fn launch(t: *Trace, k: Kernel, inputs: []const T, cfg: *const LaunchConfig, out: []T) !void {
                var l: Launch = .{ .k = k, .cfg = cfg.*, .n_in = inputs.len };
                @memcpy(l.inputs[0..inputs.len], inputs);
                const li: u32 = @intCast(t.launches.items.len);
                for (out, 0..) |*o, i| {
                    o.* = t.with(try t.node(cfg.out_shapes[i][0..cfg.out_ranks[i]], cfg.out_dtypes[i], &.{}), .{ .out = .{ li, @intCast(i) } });
                    l.outs[i] = o.*;
                }
                try t.launches.append(t.a, l);
                try t.log.append(t.a, .{ .launch = li });
            }

            pub fn evalAll(t: *Trace, xs: []const T) !void {
                const d = try t.a.dupe(T, xs);
                errdefer t.a.free(d);
                try t.log.append(t.a, .{ .eval = d });
            }

            pub fn asyncEval(t: *Trace, xs: []const T) !void {
                const d = try t.a.dupe(T, xs);
                errdefer t.a.free(d);
                try t.log.append(t.a, .{ .async_eval = d });
            }

            pub fn concat(t: *Trace, xs: []const T, axis: c_int) !T {
                std.debug.assert(axis == 0);
                var s = t.nodes.items[xs[0]].shape;
                s.d[0] = 0;
                for (xs) |x| s.d[0] += t.nodes.items[x].shape.d[0];
                const d = try t.a.dupe(T, xs);
                errdefer t.a.free(d);
                try t.log.append(t.a, .{ .concat = d });
                t.cats += 1;
                return t.with(try t.node(s.slice(), t.nodes.items[xs[0]].dtype, &.{}), .{ .cat = t.cats - 1 });
            }

            pub fn take(t: *Trace, x: T, idx: T, axis: c_int) !T {
                std.debug.assert(axis == 0);
                var s = t.nodes.items[x].shape;
                s.d[0] = t.nodes.items[idx].shape.d[0];
                try t.log.append(t.a, .{ .take = .{ x, idx } });
                t.takes += 1;
                return t.with(try t.node(s.slice(), t.nodes.items[x].dtype, &.{}), .{ .take = t.takes - 1 });
            }

            /// A caller array: `e`'s input `name` at `vars`.
            pub fn arg(t: *Trace, e: *const Entry, comptime name: []const u8, vars: *const Vars) !T {
                const a = argOf(e, name);
                var shape: [xk.max_rank]c_int = undefined;
                for (a.shape, 0..) |d, i| shape[i] = @intCast(d.eval(vars));
                return t.node(shape[0..a.shape.len], a.dtype, &.{});
            }

            pub fn back(t: *const Trace, n: usize) *const Launch {
                return &t.launches.items[t.launches.items.len - n];
            }

            // ── graph ops (the C2 adapter's gathers and activation), recorded in the log ──

            fn record(t: *Trace, op: Op, ins: []const T, shape: []const c_int, dt: Dtype) !T {
                var o = op;
                @memcpy(o.ins[0..ins.len], ins);
                o.n_in = @intCast(ins.len);
                const i: u32 = @intCast(t.ops.items.len);
                o.out = t.with(try t.node(shape, dt, &.{}), .{ .op = i });
                try t.ops.append(t.a, o);
                try t.log.append(t.a, .{ .op = i });
                return o.out;
            }

            pub fn mul(t: *Trace, x: T, y: T) !T {
                const s = if (t.nodes.items[x].shape.n >= t.nodes.items[y].shape.n) x else y;
                return t.record(.{ .kind = "mul" }, &.{ x, y }, t.nodes.items[s].shape.slice(), t.nodes.items[x].dtype);
            }

            pub fn silu(t: *Trace, x: T) !T {
                return t.record(.{ .kind = "silu" }, &.{x}, t.nodes.items[x].shape.slice(), t.nodes.items[x].dtype);
            }

            pub fn clip(t: *Trace, x: T, lo: T, hi: T) !T {
                return t.record(.{ .kind = "clip" }, &.{ x, lo, hi }, t.nodes.items[x].shape.slice(), t.nodes.items[x].dtype);
            }

            pub fn minimum(t: *Trace, x: T, y: T) !T {
                return t.record(.{ .kind = "minimum" }, &.{ x, y }, t.nodes.items[x].shape.slice(), t.nodes.items[x].dtype);
            }

            pub fn scalar(t: *Trace, v: f64, dt: Dtype) !T {
                return t.record(.{ .kind = "scalar", .f = v }, &.{}, &.{}, dt);
            }

            /// The stock quant's backend gather (`quant.GatherQmm`): x [rows, 1, in], w [slots, out, *],
            /// rhs [rows] -> [rows, 1, out] in x's dtype.
            pub fn gatherMatmul(t: *Trace, x: T, w: T, scales: ?T, biases: ?T, rhs: T, bits: u32, group: u32, mode: ?QuantMode, sorted: bool) !T {
                var ins: [5]T = undefined;
                var n: usize = 0;
                for ([_]?T{ x, w, scales, biases, rhs }) |v| if (v) |u| {
                    ins[n] = u;
                    n += 1;
                };
                const sx = t.nodes.items[x].shape;
                const sw = t.nodes.items[w].shape;
                const op: Op = .{ .kind = if (mode == null) "gather_mm" else "gather_qmm", .bits = bits, .group = group, .mode = if (mode) |m| @tagName(m) else "", .sorted = sorted };
                return t.record(op, ins[0..n], &.{ sx.d[0], 1, sw.d[1] }, t.nodes.items[x].dtype);
            }
        };

        /// The decode batch 2 families (their routes' own test covers them).
        pub fn isDecode2(e: *const Entry) bool {
            inline for (.{ "woa_ring_transpose", "index_topk", "attn_fuse", "minv_mxfp8_rows", "minv_woa_head", "minv_smallm" }) |f| {
                if (std.mem.eql(u8, e.family, f)) return true;
            }
            return false;
        }

        /// The prefill batch 2 families (their routes' own test covers them).
        pub fn isPrefill2(e: *const Entry) bool {
            inline for (.{ "pf_idxscore", "pf_attn_core", "pf_hcnorm", "pf_smallk", "pf_joinless", "pf_hcpost" }) |f| {
                if (std.mem.eql(u8, e.family, f)) return true;
            }
            return false;
        }

        pub fn testRegistry() !xk.Registry {
            var diag: xk.Diag = .{};
            return xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
                std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
                return e;
            };
        }

        /// A kernel a route tables per M (rows 1..8 / 32 / 48: the decode kernels); the prefill kernels
        /// (rows up to 2^20, or no row bound) launch per call.
        pub fn tabled(e: *const Entry) bool {
            const b = e.bounds.get(.rows) orelse return false;
            return b[1] <= 48;
        }

        /// Every decode launch came from a prepared config, every prefill launch from a per-call one.
        pub fn expectPreparedDecode(t: *const Trace, reg: *const xk.Registry) !void {
            for (t.launches.items) |l| if (tabled(reg.get(l.k)) != l.prepared) {
                std.debug.print("launch of {t}: prepared {}\n", .{ l.k, l.prepared });
                return error.TestUnexpectedResult;
            };
        }

        pub fn sampleAt(e: *const Entry, site: ?[]const u8, rows: u64) *const xk.Sample {
            for (e.samples) |*s| {
                const same_site = if (site) |a| (if (s.site) |b| std.mem.eql(u8, a, b) else false) else s.site == null;
                if (same_site and s.vars.get(.rows) == rows) return s;
            }
            unreachable;
        }

        /// `l` launched `e` with `inputs` (in the kernel's order) at the lane's own launch `s`.
        pub fn expectLaunch(l: *const Trace.Launch, e: *const Entry, s: *const xk.Sample, inputs: []const Trace.T) !void {
            try testing.expectEqual(e.kernel, l.k);
            try testing.expectEqualSlices(Trace.T, inputs, l.inputs[0..l.n_in]);
            const cfg = &l.cfg;
            try testing.expectEqual(s.grid, cfg.grid);
            try testing.expectEqual(s.threadgroup, cfg.threadgroup);
            try testing.expectEqual(s.output_shapes.len, cfg.n_out);
            for (s.output_shapes, s.output_dtypes, 0..) |shape, dt, i| {
                try testing.expectEqual(shape.len, cfg.out_ranks[i]);
                for (shape, 0..) |d, j| try testing.expectEqual(@as(c_int, @intCast(d)), cfg.out_shapes[i][j]);
                try testing.expectEqual(dt, cfg.out_dtypes[i]);
            }
            try testing.expectEqual(s.template.len, cfg.template.len);
            for (s.template, cfg.template) |want, got| {
                try testing.expectEqualStrings(want.name, got.name);
                try testing.expectEqual(want.value, got.value);
            }
        }

        pub fn templateInt(tmpl: []const xk.TemplateArg, name: []const u8) i32 {
            for (tmpl) |a| if (std.mem.eql(u8, a.name, name)) return a.value.int;
            unreachable;
        }

        pub fn shapeStr(out: *std.ArrayList(u8), a: Allocator, s: []const c_int) !void {
            try out.append(a, '[');
            for (s, 0..) |d, i| try out.print(a, "{s}{d}", .{ if (i == 0) "" else ",", d });
            try out.append(a, ']');
        }

        /// The lane samples' canonical reference of a node: ext:<name>, host:<dtype>:<shape>:<sha256[0..16]>,
        /// L<launch>.<output>, C<concat>, T<take>; a view appends @<its shape>.
        pub fn traceRef(t: *const Trace, x: Trace.T, out: *std.ArrayList(u8)) !void {
            const a = t.a;
            var n = x;
            var view: ?Shape = null;
            while (t.nodes.items[n].origin == .view) {
                if (view == null) view = t.nodes.items[n].shape;
                n = t.nodes.items[n].origin.view;
            }
            const nd = &t.nodes.items[n];
            switch (nd.origin) {
                .host => {
                    const hex = @import("kernel_set.zig").shaHex(nd.bytes);
                    try out.print(a, "host:{t}:", .{nd.dtype});
                    try shapeStr(out, a, nd.shape.slice());
                    try out.print(a, ":{s}", .{hex[0..16]});
                },
                .ext => |name| try out.print(a, "ext:{s}", .{name}),
                .out => |o| try out.print(a, "L{d}.{d}", .{ o[0], o[1] }),
                .cat => |i| try out.print(a, "C{d}", .{i}),
                .take => |i| try out.print(a, "T{d}", .{i}),
                .op => |i| try out.print(a, "O{d}", .{i}),
                .none, .view => return error.UntracedNode,
            }
            if (view) |s| {
                try out.append(a, '@');
                try shapeStr(out, a, s.slice());
            }
        }

        pub fn traceRefs(t: *const Trace, xs: []const Trace.T, out: *std.ArrayList(u8)) !void {
            for (xs, 0..) |x, i| {
                if (i > 0) try out.append(t.a, ';');
                try traceRef(t, x, out);
            }
        }

        pub fn traceEvent(t: *const Trace, e: Trace.Ev, out: *std.ArrayList(u8)) !void {
            const a = t.a;
            switch (e) {
                .launch => |li| {
                    const l = &t.launches.items[li];
                    const c = &l.cfg;
                    try out.print(a, "launch {t} g={d},{d},{d} t={d},{d},{d} in=", .{ l.k, c.grid[0], c.grid[1], c.grid[2], c.threadgroup[0], c.threadgroup[1], c.threadgroup[2] });
                    try traceRefs(t, l.inputs[0..l.n_in], out);
                    try out.appendSlice(a, " out=");
                    for (0..c.n_out) |i| {
                        if (i > 0) try out.append(a, ';');
                        try out.print(a, "{t}", .{c.out_dtypes[i]});
                        try shapeStr(out, a, c.out_shapes[i][0..c.out_ranks[i]]);
                    }
                    if (c.template.len > 0) return error.TemplateOnPrefillRoute;
                },
                .eval => |xs| {
                    try out.appendSlice(a, "eval ");
                    try traceRefs(t, xs, out);
                },
                .async_eval => |xs| {
                    try out.appendSlice(a, "async ");
                    try traceRefs(t, xs, out);
                },
                .concat => |xs| {
                    try out.appendSlice(a, "concat ");
                    try traceRefs(t, xs, out);
                },
                .take => |xi| {
                    try out.appendSlice(a, "take ");
                    try traceRef(t, xi[0], out);
                    try out.append(a, ' ');
                    try traceRef(t, xi[1], out);
                },
                .op => |i| {
                    const o = &t.ops.items[i];
                    try out.print(a, "op {s}", .{o.kind});
                    if (std.mem.eql(u8, o.kind, "scalar")) try out.print(a, " {d}", .{o.f});
                    if (o.mode.len > 0) try out.print(a, " {s} bits={d} group={d}", .{ o.mode, o.bits, o.group });
                    if (std.mem.startsWith(u8, o.kind, "gather")) try out.print(a, " sorted={}", .{o.sorted});
                    try out.appendSlice(a, " in=");
                    try traceRefs(t, o.ins[0..o.n_in], out);
                    try out.appendSlice(a, " out=");
                    try shapeStr(out, a, t.nodes.items[o.out].shape.slice());
                },
            }
        }

        /// A backend without prepared launches (the per-call path), recording through a Trace.
        pub const TracePerCall = struct {
            pub const T = u32;
            t: *Trace,

            pub fn launch(p: *TracePerCall, k: Kernel, inputs: []const T, cfg: *const LaunchConfig, out: []T) !void {
                return p.t.launch(k, inputs, cfg, out);
            }
        };

        /// Every M of `e`'s table: the prebuilt config == the per-call `launchFor`, and the prepared
        /// launch records exactly the launch the per-call path records.
        pub fn expectRowPlans(comptime n: usize, t: *Trace, e: *const Entry, site: ?[]const u8) !usize {
            var pc: TracePerCall = .{ .t = t };
            var prep = try RowPlans(Trace, n).init(t, e, site, null);
            defer prep.deinit(t);
            var per = try RowPlans(TracePerCall, n).init(&pc, e, site, null);
            defer per.deinit(&pc);
            const ins: [16]Trace.T = @splat(0);
            var o1: [xk.max_outputs]Trace.T = undefined;
            var o2: [xk.max_outputs]Trace.T = undefined;
            for (1..n + 1) |m| {
                const want = try xk.launchFor(e, &rowsVars(m), site);
                try testing.expect(std.meta.eql(want, (try prep.at(m)).*));
                const before = t.prepared_launches;
                try prep.launch(t, m, ins[0..e.inputs.len], &o1);
                try testing.expectEqual(before + 1, t.prepared_launches);
                const a = t.back(1).*;
                try per.launch(&pc, m, ins[0..e.inputs.len], &o2);
                try testing.expectEqual(before + 1, t.prepared_launches);
                const b = t.back(1).*;
                try testing.expect(a.prepared and !b.prepared);
                try testing.expect(a.k == b.k and a.n_in == b.n_in and std.mem.eql(Trace.T, a.inputs[0..a.n_in], b.inputs[0..b.n_in]));
                try testing.expect(std.meta.eql(a.cfg, b.cfg) and std.meta.eql(a.cfg, want));
            }
            try testing.expectError(error.RowsOutOfPlan, prep.at(0));
            return n;
        }

        /// The lane's own launch of `e` at exactly `vars` (the samples of an entry several routes share).
        pub fn sampleWith(e: *const Entry, vars: *const Vars) *const xk.Sample {
            for (e.samples) |*s| if (std.meta.eql(s.vars, vars.*)) return s;
            std.debug.print("{t}: no lane sample at {any}\n", .{ e.kernel, vars.values });
            unreachable;
        }
    };
}
