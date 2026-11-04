//! Kernel routes over a pinned registry `R` (G5; mlx-stream's is `exl3_kernels`): the machinery every consumer's
//! routes share. A route is built once (static inputs, per-M launch tables, the weights it binds checked
//! against the kernel signature and refused by name), then called with the per-call arrays only.
//! A call launches exactly what the Python lane's kernel call launches (inputs in the kernel's
//! order, the lane's grid / threadgroup / template) and returns the same outputs.
//!
//! mlx-stream's consumers: the EXL3 routed-expert quant (`exl3_quant`, C2) and the V4.1 trunk
//! (`dsv41_kernel_routes`). Both run on the one kernel set the load context owns
//! (`KernelSet(R).Set`).
//!
//! Routes are generic over the model's graph backend `G`, which provides `T`, `shapeOf`
//! (a value with `slice()`), `dtypeOf`, `hostArray`, `keep`, `release`, `reshape`, `astype`
//! and one launch:
//!     launch(g: *G, k: Kernel, inputs: []const G.T, cfg: *const LaunchConfig, out: []G.T) !void
//! (a real backend: `Bound.apply`, the outputs owned by the backend's scope; a trace backend:
//! nodes of `cfg`'s output shapes and dtypes).

const std = @import("std");
const mlx = @import("sdk").mlx;

/// The routes over registry `R`, which supplies `Kernel`, `Entry`, `Arg`, `Vars`, `LaunchConfig`, `Diag`,
/// `launchFor`, `max_rank` and `max_inputs`.
pub fn Routes(comptime R: type) type {
    return struct {
        const xk = R;

        const Kernel = xk.Kernel;
        const Entry = xk.Entry;
        const Vars = xk.Vars;
        const LaunchConfig = xk.LaunchConfig;
        const Dtype = mlx.mlx_dtype;

        pub const Refusal = error{
            /// A bound array (weight, bias, bank) whose dtype or shape is not the kernel's signature.
            RouteInput,
            /// A template the registry does not carry (its self-check never ran that instantiation).
            TemplateNotRegistered,
            /// A row count outside a plan kernel's table (the caller's phase route owns those rows).
            RowsOutOfPlan,
            /// A routed row naming a slot outside its call's bank.
            SlotOutOfBank,
            /// An attention key count outside the fused softmax's range (64 < k <= 1024): the stock chain's.
            KeysOutOfPlan,
        };

        pub fn refuse(diag: ?*xk.Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
            if (diag) |d| d.set(fmt, args);
            return err;
        }

        pub fn argOf(e: *const Entry, comptime name: []const u8) *const xk.Arg {
            for (e.inputs) |*a| if (std.mem.eql(u8, a.name, name)) return a;
            unreachable;
        }

        pub fn dims(comptime G: type, g: *G, x: G.T) Shape {
            const sh = g.shapeOf(x);
            return Shape.of(sh.slice());
        }

        /// A shape copied out of the backend (the backend's own shape value may be a temporary).
        pub const Shape = struct {
            n: u8 = 0,
            d: [xk.max_rank + 2]c_int = @splat(0),

            pub fn of(s: []const c_int) Shape {
                var r: Shape = .{ .n = @intCast(s.len) };
                @memcpy(r.d[0..s.len], s);
                return r;
            }

            pub fn slice(self: *const Shape) []const c_int {
                return self.d[0..self.n];
            }
        };

        /// `x` has input `name`'s dtype and its shape at `vars` (construction time, once).
        pub fn expectInput(comptime G: type, g: *G, e: *const Entry, comptime name: []const u8, x: G.T, vars: *const Vars, diag: ?*xk.Diag) Refusal!void {
            const a = argOf(e, name);
            const got = dims(G, g, x);
            const dt = g.dtypeOf(x);
            var ok = dt == a.dtype and got.n == a.shape.len;
            if (ok) for (a.shape, got.slice()) |dim, v| {
                ok = ok and dim.eval(vars) == @as(u64, @intCast(v));
            };
            if (!ok) return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} input {s} is {t} {any}, the kernel reads {t} [{f}]", .{ e.kernel, name, dt, got.slice(), a.dtype, shapeFmt(a, vars) });
        }

        fn shapeFmt(a: *const xk.Arg, vars: *const Vars) std.fmt.Alt(ShapeAt, ShapeAt.format) {
            return .{ .data = .{ .a = a, .vars = vars } };
        }

        const ShapeAt = struct {
            a: *const xk.Arg,
            vars: *const Vars,

            fn format(s: ShapeAt, w: *std.Io.Writer) std.Io.Writer.Error!void {
                for (s.a.shape, 0..) |dim, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", dim.eval(s.vars) });
            }
        };

        /// One launch of a rule kernel at `vars`.
        pub fn launchRule(comptime G: type, g: *G, e: *const Entry, vars: *const Vars, inputs: []const G.T, out: []G.T) !void {
            const cfg = xk.launchFor(e, vars, null) catch unreachable;
            try g.launch(e.kernel, inputs, &cfg, out[0..cfg.n_out]);
        }

        pub fn rowsOf(comptime G: type, g: *G, x: G.T, axis: usize) u64 {
            return @intCast(dims(G, g, x).slice()[axis]);
        }

        /// The static inputs of `e` (role static: the lane's own constants, from the manifest's values),
        /// built once and kept; `at(i)` is input i's array (undefined for non-static inputs).
        pub fn Statics(comptime G: type) type {
            return struct {
                const Self = @This();
                arrays: [xk.max_inputs]G.T = undefined,
                mask: u32 = 0,

                pub fn init(g: *G, e: *const Entry) !Self {
                    var s: Self = .{};
                    errdefer s.deinit(g);
                    for (e.inputs, 0..) |*a, i| {
                        if (a.role != .static) continue;
                        var buf: [1024]u8 = undefined;
                        const shape, const bytes = staticBytes(a, &buf);
                        s.arrays[i] = g.keep(try g.hostArray(bytes, shape.slice(), a.dtype));
                        s.mask |= @as(u32, 1) << @intCast(i);
                    }
                    return s;
                }

                pub fn deinit(s: *Self, g: *G) void {
                    for (0..xk.max_inputs) |i| {
                        if (s.mask & (@as(u32, 1) << @intCast(i)) != 0) g.release(s.arrays[i]);
                    }
                    s.mask = 0;
                }
            };
        }

        /// A static input's shape and bytes (little endian): the manifest's `values`, or zeros.
        pub fn staticBytes(a: *const xk.Arg, buf: *[1024]u8) struct { Shape, []const u8 } {
            const vars: Vars = .initFill(0);
            var shape: Shape = .{ .n = @intCast(a.shape.len) };
            var n: usize = 1;
            for (a.shape, 0..) |dim, i| {
                shape.d[i] = @intCast(dim.eval(&vars));
                n *= @intCast(shape.d[i]);
            }
            const size = dtypeSize(a.dtype);
            const bytes = buf[0 .. n * size];
            @memset(bytes, 0);
            switch (a.domain.kind) {
                .zeros => {},
                .values => for (0..n) |i| {
                    if (a.domain.ints.len > 0) {
                        const v = a.domain.ints[i % a.domain.ints.len];
                        switch (a.dtype) {
                            .uint32 => std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @intCast(v), .little),
                            .int32 => std.mem.writeInt(i32, bytes[i * 4 ..][0..4], @intCast(v), .little),
                            else => unreachable,
                        }
                    } else {
                        const v = a.domain.floats[i % a.domain.floats.len];
                        switch (a.dtype) {
                            .float32 => std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little),
                            else => unreachable,
                        }
                    }
                },
                else => unreachable,
            }
            return .{ shape, bytes };
        }

        pub fn dtypeSize(dt: Dtype) usize {
            return switch (dt) {
                .bool_, .uint8, .int8 => 1,
                .uint16, .int16, .float16, .bfloat16 => 2,
                .uint32, .int32, .float32 => 4,
                .uint64, .int64, .float64, .complex64 => 8,
            };
        }

        pub const no_vars: Vars = .initFill(0);

        pub fn rowsVars(rows: u64) Vars {
            var v: Vars = .initFill(0);
            v.set(.rows, rows);
            return v;
        }

        /// `G` declares `name` as a function (a wrapper backend declares an absent capability as `{}`).
        pub fn declaresFn(comptime G: type, comptime name: []const u8) bool {
            return @hasDecl(G, name) and @typeInfo(@TypeOf(@field(G, name))) == .@"fn";
        }

        /// A decode kernel's launch at every row count 1..n (the kernel's `rows` bound), built once at a
        /// route's construction; a call indexes it by M, the one launch value that varies per call. A
        /// backend that prepares launches (`Prepared`, `prepareLaunch`, `launchPrepared`,
        /// `releasePrepared`; chosen at compile time) gets each launch's mlx config built here too, so a
        /// call builds none; any other backend takes the per-call `launch` (deprecated for decode).
        pub fn RowPlans(comptime G: type, comptime n: usize) type {
            const prepared = declaresFn(G, "prepareLaunch");
            return struct {
                const Self = @This();
                e: *const Entry,
                cfg: [n]LaunchConfig,
                prep: if (prepared) [n]G.Prepared else void,

                /// `site`: a plan kernel's site (rcproj sites; "" for sinkhorn), null for a rule kernel.
                pub fn init(g: *G, e: *const Entry, site: ?[]const u8, diag: ?*xk.Diag) !Self {
                    return initAt(g, e, site, &no_vars, diag);
                }

                /// A table of explicit launches (`cfgs[m - 1]` at M = m; the template slices must outlive the plans).
                pub fn initCfgs(g: *G, e: *const Entry, cfgs: *const [n]LaunchConfig) !Self {
                    var p: Self = .{ .e = e, .cfg = cfgs.*, .prep = undefined };
                    if (prepared) {
                        var built: usize = 0;
                        errdefer for (p.prep[0..built]) |*x| g.releasePrepared(x);
                        for (&p.prep, &p.cfg) |*x, *c| {
                            x.* = try g.prepareLaunch(e.kernel, c);
                            built += 1;
                        }
                    }
                    return p;
                }

                /// As `init` at the fixed values `base` of the kernel's other vars (the attention's key
                /// count): the table varies M only. The kernel's rows bound covers the table (a kernel the
                /// prefill routes also launch, at more rows per call, has a wider bound).
                pub fn initAt(g: *G, e: *const Entry, site: ?[]const u8, base: *const Vars, diag: ?*xk.Diag) !Self {
                    const b = e.bounds.get(.rows) orelse return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} has no row bound", .{e.kernel});
                    if (b[0] != 1 or b[1] < n) return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} takes rows {d}..{d}, the route's table 1..{d}", .{ e.kernel, b[0], b[1], n });
                    var p: Self = .{ .e = e, .cfg = undefined, .prep = undefined };
                    for (&p.cfg, 1..) |*c, m| {
                        var vars = base.*;
                        vars.set(.rows, m);
                        c.* = xk.launchFor(e, &vars, site) catch return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} has no plan at site {?s} rows {d}", .{ e.kernel, site, m });
                    }
                    if (prepared) {
                        var built: usize = 0;
                        errdefer for (p.prep[0..built]) |*x| g.releasePrepared(x);
                        for (&p.prep, &p.cfg) |*x, *c| {
                            x.* = try g.prepareLaunch(e.kernel, c);
                            built += 1;
                        }
                    }
                    return p;
                }

                pub fn deinit(p: *Self, g: *G) void {
                    if (prepared) for (&p.prep) |*x| g.releasePrepared(x);
                }

                /// The launch at M rows; RowsOutOfPlan outside 1..n (the caller's phase route owns those).
                pub fn at(p: *const Self, m: u64) Refusal!*const LaunchConfig {
                    if (m < 1 or m > n) return error.RowsOutOfPlan;
                    return &p.cfg[m - 1];
                }

                pub fn launch(p: *const Self, g: *G, m: u64, inputs: []const G.T, out: []G.T) !void {
                    std.debug.assert(!(m < 1 or m > n));
                    const k = m - 1;
                    if (prepared) {
                        try g.launchPrepared(&p.prep[k], inputs, out[0..p.cfg[k].n_out]);
                    } else {
                        try g.launch(p.e.kernel, inputs, &p.cfg[k], out[0..p.cfg[k].n_out]);
                    }
                }
            };
        }
    };
}
