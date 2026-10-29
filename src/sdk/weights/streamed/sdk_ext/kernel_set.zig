//! The kernel set over a pinned registry `R` (G5): one manifest, its texts, its pin, built and bound once per
//! model load. The host's load context owns it. Its consumers take it at their accept (mlx-stream's EXL3 registry:
//! the routed-expert quant, `exl3_quant`, and the V4.1 trunk, `dsv41_kernel_routes`). Each consumer self-checks only
//! its own subset, and the set is freed after the last consumer's `deinit`.
//!
//! The consumers' subsets partition the registry (`checkPartition`, at comptime; `partitionError`
//! names a violator at run time). So the manifest sha, the self-check pins and every device receipt
//! stay the set's, whichever consumers load. With one consumer loaded, the whole set is still pinned
//! (every text verified) and only that consumer's subset is self-checked.

const std = @import("std");
const mlx = @import("sdk").mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("metal_launch.zig");
const profile = @import("profile.zig");

/// A load's kernel set as a quant's accept receives it (`sdk.quant.Context.kernels`): the set, erased, and the pin
/// of the registry it is built over. A quant takes it back as its own registry's set (`KernelSet(R).Set.of`); a set
/// over another registry is refused by name there, never cast.
pub const SetRef = struct {
    set: *const anyopaque,
    manifest_sha256: []const u8,
};

/// The kernel set over registry `R`, which supplies `Kernel`, `Check`, `Texts`, `embedded`, `manifest_sha256`,
/// `Registry`, `Bound` (with `observe`), `Diag`, `n_kernels`, `n_headers` and its self-check plan (`R.selfcheck`:
/// `Report`, `runSubset`, `judge`).
pub fn KernelSet(comptime R: type) type {
    return struct {
        const xk = R;
        const selfcheck = R.selfcheck;

        const Allocator = std.mem.Allocator;
        const Kernel = xk.Kernel;

        /// What the kernel set is built on.
        pub const Device = union(enum) {
            /// the arm's GPU stream: every kernel is built on it and the self-check plan runs there
            stream: mlx.mlx_stream,
            /// host tests: no kernel object and no MLX array; the plan's results are scripted
            stub: StubDevice,
        };

        /// A scripted device: every (kernel, check) of the plan passes, except `fail` when set.
        pub const StubDevice = struct {
            fail: ?struct { kernel: Kernel, check: xk.Check } = null,
        };

        pub const StartupOptions = struct {
            device: Device,
            /// the texts the registry checks against the pin (production: the embedded ones)
            texts: *const xk.Texts = &xk.embedded,
            pin: []const u8 = xk.manifest_sha256,
        };

        /// The backend methods every route uses (the phase-3 contract); the prefill route adds its
        /// own (evalAll, asyncEval, concat, take, mark, resetTo) and checks them itself.
        const backend_methods = [_][]const u8{ "launch", "shapeOf", "dtypeOf", "hostArray", "keep", "release", "reshape", "astype" };

        /// Where `Set.install` puts the launcher: the backend's `launcher: ?*const xk.Bound`
        /// field, or its base backend's under a wrapper that declares `Inner` and `base()`
        /// (a profiling wrapper); null when neither has one.
        fn launcherSlot(comptime G: type, g: *G) ?*?*const xk.Bound {
            if (@hasField(G, "launcher")) return &g.launcher;
            if (@hasDecl(G, "Inner")) return launcherSlot(G.Inner, g.base());
            return null;
        }

        /// The first route method `G` lacks, or null.
        pub fn missingBackendMethod(comptime G: type) ?[]const u8 {
            inline for (backend_methods) |m| if (!@hasDecl(G, m)) return m;
            return null;
        }

        /// The kernel set of one model load.
        pub const Set = struct {
            a: Allocator,
            reg: xk.Registry,
            bound: xk.Bound,
            device: Device,

            /// Builds the registry (every text against the pinned manifest) and, on a stream, every
            /// kernel. Refused by name: a registry refusal (xk.Refusal: TextSha256Mismatch,
            /// ManifestNotPinned, LanePinMismatch, ...; `diag` names the text), NotGpuStream /
            /// KernelCreateFailed at bind. Heap-allocated: the bound kernels point at the registry.
            pub fn init(a: Allocator, opts: StartupOptions, diag: *xk.Diag) !*Set {
                const s = try a.create(Set);
                errdefer a.destroy(s);
                s.* = .{ .a = a, .reg = undefined, .bound = undefined, .device = opts.device };
                s.reg = try xk.Registry.init(a, opts.texts, opts.pin, diag);
                errdefer s.reg.deinit();
                s.bound = switch (opts.device) {
                    .stream => |st| try s.reg.bind(st, diag),
                    .stub => .{ .reg = &s.reg, .stream = .{}, .kernels = @splat(.{}) },
                };
                return s;
            }

            /// The set as a quant's accept takes it (`sdk.quant.Context.kernels`), pinned to `R`.
            pub fn ref(self: *const Set) SetRef {
                return .{ .set = self, .manifest_sha256 = xk.manifest_sha256 };
            }

            /// `r` as a set over `R`; null when it is built over another registry (another pin).
            pub fn of(r: SetRef) ?*const Set {
                if (!std.mem.eql(u8, r.manifest_sha256, xk.manifest_sha256)) return null;
                return @ptrCast(@alignCast(r.set));
            }

            /// After every consumer's `deinit` and `uninstall`.
            pub fn deinit(self: *Set) void {
                self.bound.deinit();
                self.reg.deinit();
                self.a.destroy(self);
            }

            /// Points the backend's launcher at the bound kernels (its `launcher: ?*const xk.Bound`
            /// field, or its base backend's under a wrapper), before any consumer builds a route: a
            /// backend that prepares launches prepares them through it. The backend's profile hook
            /// (G7, `profile.of`) observes the set's launches from here on. A backend without a
            /// route method does not compile (the method is named).
            pub fn install(self: *Set, comptime G: type, g: *G) void {
                comptime {
                    if (missingBackendMethod(G)) |m| @compileError("kernel set: the backend " ++ @typeName(G) ++ " lacks " ++ m);
                }
                if (launcherSlot(G, g)) |s| s.* = &self.bound;
                self.bound.observe(profile.of(G).launch);
            }

            /// Clears the backend's launcher (after the last consumer's `deinit`).
            pub fn uninstall(comptime G: type, g: *G) void {
                if (launcherSlot(G, g)) |s| s.* = null;
            }

            /// One consumer's acceptance: the self-check plan of the entries of `subset` at `depth` on the set's
            /// device (the stub device's scripted plan in host tests), recorded in `report` and judged. The
            /// construction runs `.compile` (no numeric probe); the device tests `.full` (`selfcheck.Depth`).
            /// Refused: error.SelfCheckFailed, `diag` naming the first failing kernel / check / site.
            pub fn selfCheck(self: *const Set, a: Allocator, subset: []const Kernel, depth: selfcheck.Depth, report: *selfcheck.Report, diag: *xk.Diag) !void {
                switch (self.device) {
                    .stream => try selfcheck.runSubset(a, &self.reg, &self.bound, subset, depth, report),
                    .stub => |st| try stubPlan(a, &self.reg, st, subset, depth, report),
                }
                try selfcheck.judge(report, diag);
            }
        };

        /// The kernels of `subset` as a set.
        pub fn subsetOf(subset: []const Kernel) std.EnumSet(Kernel) {
            var s: std.EnumSet(Kernel) = .empty;
            for (subset) |k| s.insert(k);
            return s;
        }

        /// The stub device's plan over `subset`: one scripted result per (kernel, check), in registry order.
        fn stubPlan(a: Allocator, reg: *const xk.Registry, stub: StubDevice, subset: []const Kernel, depth: selfcheck.Depth, report: *selfcheck.Report) !void {
            const want = subsetOf(subset);
            const checks = selfcheck.plan(reg, want, depth);
            for (&reg.entries) |*e| {
                if (!want.contains(e.kernel)) continue;
                var it = checks.get(e.kernel).iterator();
                while (it.next()) |c| {
                    const fails = if (stub.fail) |f| f.kernel == e.kernel and f.check == c else false;
                    try report.results.append(a, .{ .kernel = e.kernel, .check = c, .words = 1, .ok = !fails, .err = if (fails) "stub device: scripted failure" else "" });
                }
            }
        }

        /// Comptime: the consumers' `kernels` lists partition the registry's kernels (every kernel in
        /// exactly one list); a compile error names the first kernel that is not.
        pub fn checkPartition(comptime lists: []const []const Kernel) void {
            comptime {
                @setEvalBranchQuota(100_000);
                var owners: [xk.n_kernels]u32 = @splat(0);
                for (lists) |l| for (l) |k| {
                    owners[@backingInt(k)] += 1;
                };
                for (owners, 0..) |n, i| if (n != 1)
                    @compileError(std.fmt.comptimePrint("kernel set: {s} is in {d} consumer subsets (exactly 1 wanted)", .{ @tagName(@as(Kernel, @fromBackingInt(@intCast(i)))), n }));
            }
        }

        /// The run-time partition check over a registry: every kernel in exactly one of `lists`, and no
        /// header read by kernels of two lists. Null when clean, else the first violation, in `buf`.
        pub fn partitionError(reg: *const xk.Registry, lists: []const []const Kernel, buf: []u8) ?[]const u8 {
            var owner: [xk.n_kernels]?usize = @splat(null);
            for (lists, 0..) |l, li| for (l) |k| {
                if (owner[@backingInt(k)]) |prev| return std.fmt.bufPrint(buf, "{t} is in subsets {d} and {d}", .{ k, prev, li }) catch buf;
                owner[@backingInt(k)] = li;
            };
            for (owner, 0..) |o, i| if (o == null) return std.fmt.bufPrint(buf, "{t} is in no subset", .{@as(Kernel, @fromBackingInt(@intCast(i)))}) catch buf;
            var header_owner: [xk.n_headers]?usize = @splat(null);
            for (&reg.entries) |*e| {
                const h = e.header orelse continue;
                const li = owner[@backingInt(e.kernel)].?;
                if (header_owner[@backingInt(h)]) |prev| if (prev != li)
                    return std.fmt.bufPrint(buf, "header {t} is read by subsets {d} and {d} ({t})", .{ h, prev, li, e.kernel }) catch buf;
                header_owner[@backingInt(h)] = li;
            }
            return null;
        }
    };
}

// ── What every pinned registry shares (`exl3_kernels`, `glm53_exl3_kernels`) ──
// The texts' sha256 pins against a manifest, the manifest's headers and kernels by name, the mlx fast-kernel objects,
// launch configs and launches over them with the decode-timers build's launch probe, and the self-check's results,
// report, depth, verdict and array scope. A registry keeps its own manifest schema, checks and refusal set; these take
// its message prefix.

const KDiag = @import("sdk").Diag;
const bo = @import("../build_flags.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The lowercase hex sha256 of `bytes`.
pub fn shaHex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// A 64-hex sha256 as bytes; null for any other string.
pub fn hexSha(s: []const u8) ?[32]u8 {
    if (s.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch return null;
    return out;
}

pub const TextError = error{ SchemaInvalid, TextSha256Mismatch };

/// `text` against its manifest record (`sha256` hex, `bytes`), refused by name: "<prefix>: <what> ...".
pub fn checkText(text: []const u8, sha256: []const u8, bytes: u64, comptime prefix: []const u8, what: []const u8, diag: ?*KDiag) TextError!void {
    const want = hexSha(sha256) orelse {
        if (diag) |d| d.set(prefix ++ ": {s}: sha256 \"{s}\" is not 64 hex", .{ what, sha256 });
        return error.SchemaInvalid;
    };
    var d: [32]u8 = undefined;
    Sha256.hash(text, &d, .{});
    if (!std.mem.eql(u8, &d, &want) or text.len != bytes) {
        if (diag) |dg| dg.set(prefix ++ ": {s} text ({d} B) differs from the manifest ({d} B, sha256 {s})", .{ what, text.len, bytes, sha256 });
        return error.TextSha256Mismatch;
    }
}

/// A template argument of a launch: an int or a dtype.
pub const TemplateValue = union(enum) { int: i32, dtype: mlx.mlx_dtype };
pub const TemplateArg = struct { name: [:0]const u8, value: TemplateValue };

/// The mlx fast-kernel object of one text (its inputs' and outputs' names in the signature's order, `header` before
/// `source`); null when mlx refuses it.
pub fn newKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool) ?mlx.mlx_fast_metal_kernel {
    const vin = mlx.mlx_vector_string_new_data(@constCast(inputs.ptr), inputs.len);
    defer _ = mlx.mlx_vector_string_free(vin);
    const vout = mlx.mlx_vector_string_new_data(@constCast(outputs.ptr), outputs.len);
    defer _ = mlx.mlx_vector_string_free(vout);
    const k = mtl.mlx_fast_metal_kernel_new(name, vin, vout, source, header, ensure_row_contiguous, false);
    return if (k.ctx == null) null else k;
}

/// One launch's outputs (shapes and dtypes), grid (threads), threadgroup and template arguments.
pub const Launch = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    out_shapes: []const []const c_int,
    out_dtypes: []const mlx.mlx_dtype,
    template: []const TemplateArg,
};

/// The mlx config of one launch, built once and handed to every launch at that geometry (`apply`); the caller frees it
/// (`mlx_fast_metal_kernel_config_free`) once its last launch has returned.
pub fn buildConfig(l: Launch) error{MlxError}!mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.out_shapes, l.out_dtypes) |sh, dt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @intCast(l.grid[0]), @intCast(l.grid[1]), @intCast(l.grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, @intCast(l.threadgroup[0]), @intCast(l.threadgroup[1]), @intCast(l.threadgroup[2])));
    for (l.template) |t| switch (t.value) {
        .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name.ptr, v)),
        .dtype => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, t.name.ptr, v)),
    };
    return c;
}

/// One launch of `kernel` with `config` on `stream`; `outs` receives the launch's outputs (new arrays, caller frees).
pub fn applyConfig(kernel: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array, stream: mlx.mlx_stream) error{MlxError}!void {
    const vin = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(vin);
    var vout = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vout);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&vout, kernel, vin, config, stream));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, vout, i));
    }
}

/// G7: the package's decode-timers build observes each launch of a bound set (its first dispatches per phase, the
/// observer `Bound.observe` installs); every other build has no observer field, launch key or call.
pub const launch_observed: bool = if (@hasDecl(bo, "dsv41_decode_timers")) bo.dsv41_decode_timers else false;
/// A launch config's template key (`launchKey`), kept with its prepared config.
pub const LaunchKey = if (launch_observed) u64 else void;
/// The observer a profile hook installs: the kernel's name, the launch key and the inputs.
pub const Observer = if (launch_observed) ?*const fn ([]const u8, u64, []const mlx.mlx_array) void else void;

/// The template key of a launch config's template arguments (name, then the value's bytes).
pub fn launchKey(names: []const []const u8, values: []const i64) u64 {
    var h = std.hash.Wyhash.init(0x5eed);
    for (names, values) |n, v| {
        h.update(n);
        h.update(std.mem.asBytes(&v));
    }
    return h.final();
}

/// `launchKey` of a launch's template arguments (an int's value, a dtype's tag); nothing outside the decode-timers build.
pub fn templateKey(template: []const TemplateArg) LaunchKey {
    if (comptime !launch_observed) return {};
    var names: [16][]const u8 = undefined;
    var values: [16]i64 = undefined;
    const nt = @min(template.len, names.len);
    for (template[0..nt], names[0..nt], values[0..nt]) |t, *n, *v| {
        n.* = t.name;
        v.* = switch (t.value) {
            .int => |x| x,
            .dtype => |x| @backingInt(x),
        };
    }
    return launchKey(names[0..nt], values[0..nt]);
}

/// The manifest against its pin (hex sha256), refused by name ("<prefix>: manifest sha256 ... is not the pinned ...");
/// its hex.
pub fn checkPin(manifest: []const u8, pin: []const u8, comptime prefix: []const u8, diag: ?*KDiag) error{ManifestNotPinned}![64]u8 {
    const got = shaHex(manifest);
    if (!std.mem.eql(u8, &got, pin)) {
        if (diag) |d| d.set(prefix ++ ": manifest sha256 {s} is not the pinned {s}", .{ &got, pin });
        return error.ManifestNotPinned;
    }
    return got;
}

/// A text pinned in code by its sha256 (hex), under the file name its manifest lists (a family without a registry:
/// MiMo's decode attention, fp8res and glue texts).
pub const TextPin = struct { file: []const u8, text: []const u8, sha256: []const u8 };

/// Once, at construction: the manifest is the pinned one (`checkPin`) and every text matches its pin (`checkText`, at
/// the text's own length), else the family's named refusal (`manifest_err`, `text_err`).
pub fn checkTextPins(manifest: []const u8, manifest_sha256: []const u8, pins: []const TextPin, comptime manifest_err: anytype, comptime text_err: anytype) (@TypeOf(manifest_err) || @TypeOf(text_err))!void {
    _ = checkPin(manifest, manifest_sha256, "", null) catch return manifest_err;
    for (pins) |p| checkText(p.text, p.sha256, p.text.len, "", p.file, null) catch return text_err;
}

/// The CPU test of a family's manifest (`headers` and `kernels[].source`, each with `file`, `sha256`, `bytes`): every
/// pinned text matches its pin and is listed once, its record matching the text (`checkText`); the manifest lists
/// `n_kernels` kernels (test code only).
pub fn expectManifestPins(manifest: []const u8, pins: []const TextPin, n_kernels: usize) !void {
    const J = struct {
        headers: []const struct { file: []const u8, sha256: []const u8, bytes: usize },
        kernels: []const struct { name: []const u8, source: struct { file: []const u8, sha256: []const u8, bytes: usize } },
    };
    const m = try std.json.parseFromSlice(J, testing.allocator, manifest, .{ .ignore_unknown_fields = true });
    defer m.deinit();
    var seen: usize = 0;
    for (pins) |p| {
        try testing.expectEqualStrings(&shaHex(p.text), p.sha256);
        for (m.value.headers) |h| if (std.mem.eql(u8, h.file, p.file)) {
            try checkText(p.text, h.sha256, h.bytes, "manifest pins", h.file, null);
            seen += 1;
        };
        for (m.value.kernels) |k| if (std.mem.eql(u8, k.source.file, p.file)) {
            try checkText(p.text, k.source.sha256, k.source.bytes, "manifest pins", k.source.file, null);
            seen += 1;
        };
    }
    try testing.expectEqual(pins.len, seen);
    try testing.expectEqual(n_kernels, m.value.kernels.len);
}

/// The manifest's header records (`id`, `sha256`, `bytes`), each once and naming a `Header` this build embeds, each
/// against its embedded text (`texts`, in `Header` order); every header listed.
pub fn adoptHeaders(comptime Header: type, texts: []const [:0]const u8, records: anytype, comptime prefix: []const u8, diag: ?*KDiag) (TextError || error{ UnknownHeader, MissingHeader })!void {
    var seen: std.EnumSet(Header) = .empty;
    for (records) |h| {
        const name: []const u8 = if (@TypeOf(h.id) == ?[]const u8) h.id orelse "" else h.id;
        const id = std.meta.stringToEnum(Header, name) orelse {
            if (diag) |d| d.set(prefix ++ ": manifest header \"{s}\" is not one this build embeds", .{name});
            return error.UnknownHeader;
        };
        if (seen.contains(id)) {
            if (diag) |d| d.set(prefix ++ ": header {t} listed twice", .{id});
            return error.SchemaInvalid;
        }
        seen.insert(id);
        try checkText(texts[@backingInt(id)], h.sha256, h.bytes, prefix, @tagName(id), diag);
    }
    const n = std.meta.fieldNames(Header).len;
    if (seen.count() != n) {
        if (diag) |d| d.set(prefix ++ ": the manifest lists {d} of {d} headers", .{ seen.count(), n });
        return error.MissingHeader;
    }
}

/// The manifest's kernels by name: each once (`admit`) and every one listed (`done`).
pub fn Tally(comptime Kernel: type) type {
    return struct {
        const Self = @This();
        seen: std.EnumSet(Kernel) = .empty,

        pub fn admit(t: *Self, name: []const u8, comptime prefix: []const u8, diag: ?*KDiag) error{ UnknownKernel, DuplicateKernel }!Kernel {
            const k = std.meta.stringToEnum(Kernel, name) orelse {
                if (diag) |d| d.set(prefix ++ ": manifest kernel \"{s}\" is not one this build implements", .{name});
                return error.UnknownKernel;
            };
            if (t.seen.contains(k)) {
                if (diag) |d| d.set(prefix ++ ": {t} listed twice", .{k});
                return error.DuplicateKernel;
            }
            t.seen.insert(k);
            return k;
        }

        pub fn done(t: *const Self, comptime prefix: []const u8, diag: ?*KDiag) error{MissingKernel}!void {
            const n = std.meta.fieldNames(Kernel).len;
            if (t.seen.count() == n) return;
            var it = t.seen.complement().iterator();
            if (diag) |d| d.set(prefix ++ ": the manifest does not list {t} ({d} of {d})", .{ it.next().?, t.seen.count(), n });
            return error.MissingKernel;
        }
    };
}

/// One launch's mlx config (output shapes and dtypes, grid, threadgroup, template arguments), built once
/// (`Bound.prepare`) and handed to every launch at that geometry (`applyPrepared`). mlx copies what a launch needs when
/// it is applied, so `deinit` is safe once the last `applyPrepared` using it has returned.
pub fn Prepared(comptime Kernel: type) type {
    return struct {
        kernel: Kernel,
        config: mlx.mlx_fast_metal_kernel_config,
        n_out: usize,
        /// its template arguments' key (the observer's; profile builds only)
        tkey: LaunchKey = if (launch_observed) 0 else {},

        pub fn deinit(self: *@This()) void {
            _ = mlx.mlx_fast_metal_kernel_config_free(self.config);
            self.* = undefined;
        }
    };
}

/// The kernels of registry `R` built on one GPU stream (`R.Registry.bind`); freed after that stream has drained. `R`
/// supplies `Registry`, `Kernel`, `n_kernels`, `max_outputs`, `LaunchConfig` and
///     launchOf(reg, k, cfg, shapes: *[max_outputs][]const c_int, template: *[16]TemplateArg) Launch
/// (the registry's config as the mlx launch; the slices may point into `shapes` and `template`).
pub fn BoundOf(comptime R: type) type {
    return struct {
        const Self = @This();
        reg: *const R.Registry,
        stream: mlx.mlx_stream,
        kernels: [R.n_kernels]mlx.mlx_fast_metal_kernel,
        /// the profile hook's launch probe (`observe`); void outside the decode-timers build
        observer: Observer = if (launch_observed) null else {},

        pub fn deinit(self: *Self) void {
            for (self.kernels) |k| {
                if (k.ctx != null) _ = mtl.mlx_fast_metal_kernel_free(k);
            }
            self.* = undefined;
        }

        /// Every later launch reaches the hook's launch probe `L` (`sdk_ext.profile.Hook.launch`), at construction,
        /// before the first: a no-op unless the build observes launches and `L` is enabled.
        pub fn observe(self: *Self, comptime L: type) void {
            if (comptime launch_observed and L.enabled) self.observer = &L.kernel;
        }

        /// A launch's mlx config built once (a route's construction): `applyPrepared` hands it to the kernel with no
        /// per-launch config work. The caller owns it (`Prepared.deinit`).
        pub fn prepare(self: *const Self, k: R.Kernel, cfg: *const R.LaunchConfig) error{MlxError}!Prepared(R.Kernel) {
            var shapes: [R.max_outputs][]const c_int = undefined;
            var template: [16]TemplateArg = undefined;
            const l = R.launchOf(self.reg, k, cfg, &shapes, &template);
            const c = try buildConfig(l);
            return .{ .kernel = k, .config = c, .n_out = l.out_shapes.len, .tkey = templateKey(l.template) };
        }

        /// One launch of a prepared config; `outs` receives `p.n_out` new arrays (caller frees).
        pub fn applyPrepared(self: *const Self, p: *const Prepared(R.Kernel), inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) error{MlxError}!void {
            if (comptime launch_observed) if (self.observer) |f| f(@tagName(p.kernel), p.tkey, inputs);
            try applyConfig(self.kernels[@backingInt(p.kernel)], p.config, inputs, outs[0..p.n_out], self.stream);
        }

        /// One launch of `k` with `cfg`, its config built for this launch only; `outs` receives the launch's outputs
        /// (new arrays, caller frees): the per-call path of routes whose geometry varies per call.
        pub fn apply(self: *const Self, k: R.Kernel, inputs: []const mlx.mlx_array, cfg: *const R.LaunchConfig, outs: []mlx.mlx_array) error{MlxError}!void {
            var p = try self.prepare(k, cfg);
            defer p.deinit();
            try self.applyPrepared(&p, inputs, outs);
        }
    };
}

/// The self-check's depth:
/// - `compile` (the construction's acceptance): every kernel compiles, binds and completes one launch; no numeric
///   comparison (the server's startup runs no numeric probes). A kernel that does not refuses the load by name.
/// - `full` (the device test and the window's smoke tests): every check the manifest lists.
pub const CheckDepth = enum { compile, full };

/// The self-check's results, report, plan and verdict over a registry's `Kernel` and `Check` enums.
pub fn Checks(comptime Kernel: type, comptime Check: type) type {
    return struct {
        pub const Result = struct {
            kernel: Kernel,
            check: Check,
            site: []const u8 = "",
            words: u64 = 0,
            bad: u64 = 0,
            metric: f64 = 0,
            limit: f64 = 0,
            ok: bool = false,
            err: []const u8 = "",
            /// the latched MLX message of a raised check (owned by the Report); "" otherwise
            msg: []const u8 = "",
        };

        pub const Report = struct {
            results: std.ArrayList(Result) = .empty,
            /// the copies of the latched MLX messages the results point at
            msgs: std.ArrayList([]u8) = .empty,

            pub fn deinit(self: *Report, a: std.mem.Allocator) void {
                for (self.msgs.items) |m| a.free(m);
                self.msgs.deinit(a);
                self.results.deinit(a);
            }

            /// Records a check that raised: its error name and the latched MLX message (copied).
            pub fn appendRaised(self: *Report, a: std.mem.Allocator, k: Kernel, c: Check, err: anyerror, msg: []const u8) !void {
                try self.msgs.ensureUnusedCapacity(a, 1);
                try self.results.ensureUnusedCapacity(a, 1);
                const copy = try a.dupe(u8, msg);
                self.msgs.appendAssumeCapacity(copy);
                self.results.appendAssumeCapacity(.{ .kernel = k, .check = c, .ok = false, .err = @errorName(err), .msg = copy });
            }

            pub fn failures(self: *const Report) usize {
                var n: usize = 0;
                for (self.results.items) |r| n += @intFromBool(!r.ok);
                return n;
            }

            /// One JSON object per result, one per line (the window keeps it as its receipt).
            pub fn writeJsonLines(self: *const Report, a: std.mem.Allocator) ![]u8 {
                var j: std.ArrayList(u8) = .empty;
                errdefer j.deinit(a);
                for (self.results.items) |r| {
                    try j.print(a, "{{\"kernel\":\"{t}\",\"check\":\"{t}\",\"site\":\"{s}\",\"words\":{d},\"bad\":{d},\"metric\":{e},\"limit\":{e},\"ok\":{},\"err\":\"{s}\"", .{ r.kernel, r.check, r.site, r.words, r.bad, r.metric, r.limit, r.ok, r.err });
                    // a raised check carries the MLX message (a line without one is byte-identical to before)
                    if (r.msg.len > 0) {
                        try j.appendSlice(a, ",\"msg\":\"");
                        try appendJsonEscaped(&j, a, r.msg);
                        try j.append(a, '"');
                    }
                    try j.appendSlice(a, "}\n");
                }
                return j.toOwnedSlice(a);
            }
        };

        /// The checks of `want`'s entries (anything with `kernel` and `checks`) at `depth`, per kernel, in registry
        /// order: at `compile` only `Check.compile`.
        pub fn plan(entries: anytype, want: std.EnumSet(Kernel), depth: CheckDepth) std.EnumArray(Kernel, std.EnumSet(Check)) {
            var out: std.EnumArray(Kernel, std.EnumSet(Check)) = .initFill(.empty);
            for (entries) |*e| {
                if (!want.contains(e.kernel)) continue;
                out.set(e.kernel, if (depth == .compile) e.checks.intersectWith(comptime std.EnumSet(Check).initOne(.compile)) else e.checks);
            }
            return out;
        }

        /// The acceptance verdict over a finished plan: refused (error.SelfCheckFailed, `diag` naming the first failing
        /// kernel / check / site, "<prefix>: self-check ...") unless every result passed and the plan ran.
        pub fn judge(report: *const Report, comptime prefix: []const u8, diag: ?*KDiag) error{SelfCheckFailed}!void {
            if (report.results.items.len == 0) {
                if (diag) |d| d.set(prefix ++ ": self-check produced no result", .{});
                return error.SelfCheckFailed;
            }
            for (report.results.items) |r| {
                if (r.ok) continue;
                if (diag) |d| d.set(prefix ++ ": self-check {t} {t} {s} failed ({d} of {d} words, metric {e} limit {e}, {s})", .{ r.kernel, r.check, r.site, r.bad, r.words, r.metric, r.limit, r.err });
                return error.SelfCheckFailed;
            }
        }
    };
}

/// `s` as the body of a JSON string: quote, backslash and control bytes escaped.
pub fn appendJsonEscaped(j: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    for (s) |c| switch (c) {
        '"' => try j.appendSlice(a, "\\\""),
        '\\' => try j.appendSlice(a, "\\\\"),
        '\n' => try j.appendSlice(a, "\\n"),
        '\t' => try j.appendSlice(a, "\\t"),
        '\r' => try j.appendSlice(a, "\\r"),
        0...8, 11, 12, 14...0x1f, 0x7f => try j.print(a, "\\u{x:0>4}", .{c}),
        else => try j.append(a, c),
    };
}

/// MLX arrays one check creates, freed together.
pub const Scope = struct {
    a: std.mem.Allocator,
    arrays: std.ArrayList(mlx.mlx_array) = .empty,

    /// `arr` kept until `deinit` (freed now when it cannot be kept).
    pub fn keep(sc: *Scope, arr: mlx.mlx_array) !mlx.mlx_array {
        sc.arrays.append(sc.a, arr) catch |e| {
            _ = mlx.mlx_array_free(arr);
            return e;
        };
        return arr;
    }

    /// A new array of `bytes` (copied), kept.
    pub fn host(sc: *Scope, bytes: []const u8, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
        const x = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (x.ctx == null) return error.MlxError;
        return sc.keep(x);
    }

    pub fn deinit(sc: *Scope) void {
        for (sc.arrays.items) |x| _ = mlx.mlx_array_free(x);
        sc.arrays.deinit(sc.a);
    }
};

const testing = std.testing;

/// A three-kernel registry without Metal: texts are names, the pin is their join, binding counts, the self-check plan
/// is the stub device's. Enough of `R` for the set's own machinery.
const FakeReg = struct {
    pub const Kernel = enum { k0, k1, k2 };
    pub const Header = enum { h0, h1 };
    pub const Check = enum { parity, bounds };
    pub const n_kernels = 3;
    pub const n_headers = 2;
    pub const Texts = [n_kernels][]const u8;
    pub const embedded: Texts = .{ "k0", "k1", "k2" };
    pub const manifest_sha256 = "fake-manifest";
    pub const Diag = struct { msg: []const u8 = "" };
    pub const Entry = struct { kernel: Kernel, checks: std.EnumSet(Check), header: ?Header };

    pub const Registry = struct {
        entries: [n_kernels]Entry,
        texts: *const Texts,
        live: *u32,

        var live_count: u32 = 0;
        var refuse_bind = false;

        pub fn init(a: std.mem.Allocator, texts: *const Texts, pin: []const u8, diag: *Diag) !Registry {
            _ = a;
            if (!std.mem.eql(u8, pin, manifest_sha256)) {
                diag.msg = "manifest not pinned";
                return error.ManifestNotPinned;
            }
            live_count += 1;
            return .{
                .entries = .{
                    .{ .kernel = .k0, .checks = .initMany(&.{ .parity, .bounds }), .header = .h0 },
                    .{ .kernel = .k1, .checks = .initOne(.parity), .header = .h0 },
                    .{ .kernel = .k2, .checks = .initOne(.bounds), .header = null },
                },
                .texts = texts,
                .live = &live_count,
            };
        }
        pub fn deinit(r: *Registry) void {
            r.live.* -= 1;
        }
        /// The stream arm's bind: here it builds nothing, it marks every kernel built.
        pub fn bind(self: *const Registry, stream: mlx.mlx_stream, diag: *Diag) !Bound {
            if (refuse_bind) {
                diag.msg = "kernel create failed";
                return error.KernelCreateFailed;
            }
            return .{ .reg = self, .stream = stream, .kernels = @splat(.{ .built = true }) };
        }
    };

    pub const Bound = struct {
        reg: *const Registry,
        stream: mlx.mlx_stream,
        kernels: [n_kernels]struct { built: bool = false },
        observed: bool = false,
        pub fn observe(self: *Bound, comptime L: type) void {
            self.observed = L.enabled;
        }
        pub fn deinit(_: *Bound) void {}
    };

    pub const selfcheck = struct {
        pub const Result = struct { kernel: Kernel, check: Check, words: u32, ok: bool, err: []const u8 };
        pub const Report = struct {
            results: std.ArrayList(Result) = .empty,
            fn deinit(r: *Report, a: std.mem.Allocator) void {
                r.results.deinit(a);
            }
        };
        pub const Depth = enum { compile, full };
        /// The fake's plan at either depth: every check of every wanted kernel.
        pub fn plan(reg: *const Registry, want: std.EnumSet(Kernel), _: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
            var out: std.EnumArray(Kernel, std.EnumSet(Check)) = .initFill(.empty);
            for (&reg.entries) |*e| if (want.contains(e.kernel)) out.set(e.kernel, e.checks);
            return out;
        }
        /// The stream arm's plan: one passing result per subset kernel, marked as the device's (words 2).
        pub fn runSubset(a: std.mem.Allocator, _: *const Registry, bound: *const Bound, subset: []const Kernel, _: Depth, report: *Report) !void {
            for (subset) |k| try report.results.append(a, .{ .kernel = k, .check = .parity, .words = 2, .ok = bound.kernels[@backingInt(k)].built, .err = "" });
        }
        pub fn judge(report: *const Report, diag: *Diag) !void {
            for (report.results.items) |r| if (!r.ok) {
                diag.msg = @tagName(r.kernel);
                return error.SelfCheckFailed;
            };
        }
    };
};

const FS = KernelSet(FakeReg);

/// A backend with every route method (bodies never run here) and the launcher slot `install` fills.
const RouteBackend = struct {
    launcher: ?*const FakeReg.Bound = null,
    pub fn launch() void {}
    pub fn shapeOf() void {}
    pub fn dtypeOf() void {}
    pub fn hostArray() void {}
    pub fn keep() void {}
    pub fn release() void {}
    pub fn reshape() void {}
    pub fn astype() void {}
};

test "sdk kernel set: a stub-device set is built over its pinned registry, erased and taken back by pin only" {
    var diag: FakeReg.Diag = .{};
    try testing.expectError(error.ManifestNotPinned, FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} }, .pin = "another" }, &diag));
    try testing.expectEqualStrings("manifest not pinned", diag.msg);
    try testing.expectEqual(@as(u32, 0), FakeReg.Registry.live_count);
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    try testing.expectEqual(@as(u32, 1), FakeReg.Registry.live_count);
    try testing.expect(s.bound.reg == &s.reg and s.reg.texts == &FakeReg.embedded);
    const r = s.ref();
    try testing.expectEqualStrings("fake-manifest", r.manifest_sha256);
    try testing.expect(FS.Set.of(r).? == s);
    try testing.expect(FS.Set.of(.{ .set = s, .manifest_sha256 = "another" }) == null);
    s.deinit();
    try testing.expectEqual(@as(u32, 0), FakeReg.Registry.live_count);
}

test "sdk kernel set: install points the backend's launcher (or its wrapped base's) at the bound set; uninstall clears it" {
    var diag: FakeReg.Diag = .{};
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer s.deinit();
    var g: RouteBackend = .{};
    s.install(RouteBackend, &g);
    try testing.expect(g.launcher.? == &s.bound and !s.bound.observed);
    FS.Set.uninstall(RouteBackend, &g);
    try testing.expect(g.launcher == null);
    // a profiling wrapper: the launcher lives on its base, and the wrapper's hook observes the launches
    const Wrapper = struct {
        inner: *RouteBackend,
        pub const Inner = RouteBackend;
        pub const profile_hook: profile.Hook = .{ .launch = struct {
            pub const enabled = true;
            pub fn kernel(_: []const u8, _: u64, _: []const mlx.mlx_array) void {}
        } };
        pub fn base(w: *@This()) *RouteBackend {
            return w.inner;
        }
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
    };
    var w: Wrapper = .{ .inner = &g };
    s.install(Wrapper, &w);
    try testing.expect(g.launcher.? == &s.bound and s.bound.observed);
    FS.Set.uninstall(Wrapper, &w);
    try testing.expect(g.launcher == null);
    // a backend without a slot installs nothing (the host trace without launches)
    const NoSlot = struct {
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
    };
    var n: NoSlot = .{};
    s.install(NoSlot, &n);
    FS.Set.uninstall(NoSlot, &n);
    try testing.expectEqual(@as(?[]const u8, null), FS.missingBackendMethod(RouteBackend));
    try testing.expectEqualStrings("launch", FS.missingBackendMethod(struct {}).?);
    try testing.expectEqualStrings("astype", FS.missingBackendMethod(struct {
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
    }).?);
}

test "sdk kernel set: a consumer self-checks its own subset in registry order; a scripted failure is refused by name" {
    var diag: FakeReg.Diag = .{};
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer s.deinit();
    var report: FakeReg.selfcheck.Report = .{};
    defer report.deinit(testing.allocator);
    try s.selfCheck(testing.allocator, &.{ .k2, .k0 }, .compile, &report, &diag);
    try testing.expectEqual(@as(usize, 3), report.results.items.len);
    const r = report.results.items;
    try testing.expect(r[0].kernel == .k0 and r[0].check == .parity and r[1].kernel == .k0 and r[1].check == .bounds and r[2].kernel == .k2);
    for (r) |x| try testing.expect(x.ok and x.err.len == 0);

    const f = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{ .fail = .{ .kernel = .k1, .check = .parity } } } }, &diag);
    defer f.deinit();
    var report2: FakeReg.selfcheck.Report = .{};
    defer report2.deinit(testing.allocator);
    // the failing kernel outside the subset is never planned
    try f.selfCheck(testing.allocator, &.{.k0}, .compile, &report2, &diag);
    try testing.expectError(error.SelfCheckFailed, f.selfCheck(testing.allocator, &.{.k1}, .compile, &report2, &diag));
    try testing.expectEqualStrings("k1", diag.msg);
    try testing.expectEqualStrings("stub device: scripted failure", report2.results.items[report2.results.items.len - 1].err);
    try testing.expect(FS.subsetOf(&.{ .k2, .k2 }).count() == 1 and FS.subsetOf(&.{}).count() == 0);

    // a stream device binds through the registry and runs the registry's own plan (this fake's touches no device)
    const d = try FS.Set.init(testing.allocator, .{ .device = .{ .stream = .{} } }, &diag);
    defer d.deinit();
    try testing.expect(d.device == .stream and d.bound.kernels[2].built);
    var report3: FakeReg.selfcheck.Report = .{};
    defer report3.deinit(testing.allocator);
    try d.selfCheck(testing.allocator, &.{.k1}, .compile, &report3, &diag);
    try testing.expect(report3.results.items.len == 1 and report3.results.items[0].words == 2);
    // a bind refusal frees the registry it built
    FakeReg.Registry.refuse_bind = true;
    defer FakeReg.Registry.refuse_bind = false;
    const live = FakeReg.Registry.live_count;
    try testing.expectError(error.KernelCreateFailed, FS.Set.init(testing.allocator, .{ .device = .{ .stream = .{} } }, &diag));
    try testing.expectEqualStrings("kernel create failed", diag.msg);
    try testing.expectEqual(live, FakeReg.Registry.live_count);
}

test "sdk kernel set: the consumers partition the kernels and the headers, or the first violation is named" {
    comptime FS.checkPartition(&.{ &.{ .k0, .k1 }, &.{.k2} });
    var diag: FakeReg.Diag = .{};
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer s.deinit();
    var buf: [128]u8 = undefined;
    try testing.expectEqual(@as(?[]const u8, null), FS.partitionError(&s.reg, &.{ &.{ .k0, .k1 }, &.{.k2} }, &buf));
    try testing.expectEqualStrings("k1 is in subsets 0 and 1", FS.partitionError(&s.reg, &.{ &.{ .k0, .k1 }, &.{ .k1, .k2 } }, &buf).?);
    try testing.expectEqualStrings("k2 is in no subset", FS.partitionError(&s.reg, &.{ &.{.k0}, &.{.k1} }, &buf).?);
    // k0 and k1 read h0: splitting them across consumers shares a header
    try testing.expectEqualStrings("header h0 is read by subsets 0 and 1 (k1)", FS.partitionError(&s.reg, &.{ &.{ .k0, .k2 }, &.{.k1} }, &buf).?);
    // a buffer too small for the message returns the buffer itself
    var tiny: [4]u8 = undefined;
    try testing.expectEqual(@as(usize, 4), FS.partitionError(&s.reg, &.{ &.{.k0}, &.{.k1} }, &tiny).?.len);
}

test "sdk kernel set: the shared registry pieces: a text matches its record only at its own sha256 and size, refused by name otherwise" {
    const text = "kernel text";
    const hex = shaHex(text);
    try checkText(text, &hex, text.len, "test kernels", "k0", null);
    var diag: KDiag = .{};
    try testing.expectError(error.TextSha256Mismatch, checkText("kernel texT", &hex, text.len, "test kernels", "k0", &diag));
    try testing.expect(std.mem.startsWith(u8, diag.message(), "test kernels: k0 text (11 B) differs from the manifest"));
    try testing.expectError(error.TextSha256Mismatch, checkText(text, &hex, text.len + 1, "test kernels", "k0", null));
    try testing.expectError(error.SchemaInvalid, checkText(text, "abc", text.len, "test kernels", "k0", &diag));
    try testing.expect(hexSha(&hex) != null and hexSha("zz") == null);
}

test "sdk kernel set: the shared self-check verdict refuses an empty plan and names the first failing kernel, check and site" {
    const K = enum { k0, k1 };
    const C = enum { compile, parity };
    const S = Checks(K, C);
    var r: S.Report = .{};
    defer r.deinit(testing.allocator);
    var diag: KDiag = .{};
    try testing.expectError(error.SelfCheckFailed, S.judge(&r, "test kernels", &diag));
    try testing.expectEqualStrings("test kernels: self-check produced no result", diag.message());
    try r.results.append(testing.allocator, .{ .kernel = .k0, .check = .compile, .ok = true });
    try S.judge(&r, "test kernels", &diag);
    try r.results.append(testing.allocator, .{ .kernel = .k1, .check = .parity, .site = "s1", .words = 8, .bad = 2, .err = "words differ" });
    try testing.expectError(error.SelfCheckFailed, S.judge(&r, "test kernels", &diag));
    try testing.expect(std.mem.startsWith(u8, diag.message(), "test kernels: self-check k1 parity s1 failed (2 of 8 words"));
    try testing.expectEqual(@as(usize, 1), r.failures());
}
