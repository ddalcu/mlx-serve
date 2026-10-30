//! The pinned Metal kernel registry of GLM-5.3's trunk routes: the indexer's top-k (`topk.zig`: one buffer of rows, or
//! up to eight rows of their own lengths in one launch) and the decode lane's routed combine (`experts.zig`). The texts
//! under kernels/glm_moe_dsa/ with a manifest of their signatures, template parameters, threadgroups, registered
//! instantiations (GLM-5.3's and the tiny fixture's) and checks, pinned by its sha256. `Registry.init` checks the manifest against `manifest_sha256` and every text against the manifest, once, and
//! refuses by name; `Registry.bind` builds the mlx fast kernels on a GPU stream, once. The set is
//! `sdk_ext.kernels.KernelSet`'s: the module builds it at its construction, where a route takes a kernel, and runs each
//! route's kernels once at the model's instantiation there (`selfcheck`), never inside a request.

const std = @import("std");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const mlx = @import("sdk").mlx;
const kset = @import("../../sdk/weights/streamed/sdk_ext/kernel_set.zig");
/// The registry's self-check plan (`sdk_ext.kernels.KernelSet(R)` runs it; the module at its construction).
pub const selfcheck = @import("trunk_selfcheck.zig");

const Allocator = std.mem.Allocator;

/// sha256 of kernels/glm_moe_dsa/manifest.json: pins every text.
pub const manifest_sha256 = "b88bc31a210fda60daa31e068cdc1e16f1da9a2147e06409470597f2084310f8";
pub const format = "mlx-stream-glm-trunk-kernels-v1";
const dir = "kernels/";

/// Every kernel of the registry; the tag is the kernel's MLX name and its file name.
pub const Kernel = enum {
    glm_moe_dsa_topk,
    glm_moe_dsa_topk_rows1,
    glm_moe_dsa_topk_rows2,
    glm_moe_dsa_topk_rows3,
    glm_moe_dsa_topk_rows4,
    glm_moe_dsa_topk_rows5,
    glm_moe_dsa_topk_rows6,
    glm_moe_dsa_topk_rows7,
    glm_moe_dsa_topk_rows8,
    glm_moe_dsa_combine,
};

/// The top-k's kernels: the one-buffer form, then the forms of 1 to `max_topk_rows` rows.
pub const topk_kernels = [_]Kernel{ .glm_moe_dsa_topk, .glm_moe_dsa_topk_rows1, .glm_moe_dsa_topk_rows2, .glm_moe_dsa_topk_rows3, .glm_moe_dsa_topk_rows4, .glm_moe_dsa_topk_rows5, .glm_moe_dsa_topk_rows6, .glm_moe_dsa_topk_rows7, .glm_moe_dsa_topk_rows8 };

/// The most rows of their own lengths one top-k launch takes (its lengths ride in `params`' eight slots).
pub const max_topk_rows = 8;

/// The top-k kernel of `rows` rows of their own lengths (1 to `max_topk_rows`).
pub fn topkRowsKernel(rows: usize) Kernel {
    std.debug.assert(rows >= 1 and rows <= max_topk_rows);
    return topk_kernels[rows];
}

/// The headers a kernel's text follows: `fp_contract_off` keeps the combine's products and sums rounded apart.
pub const Header = enum { fp_contract_off };

pub fn headerFile(h: Header) []const u8 {
    return switch (h) {
        inline else => |t| "header_" ++ @tagName(t) ++ ".metal",
    };
}

/// `compile`: the kernel builds, binds and completes one launch over zeros at an instantiation (the construction's
/// acceptance). Its numerics are the device tests' (topk.zig: argpartition's ids and order; experts.zig: the op
/// chain's bits).
pub const Check = enum { compile };
pub const Family = enum { topk, combine };

pub const n_kernels = std.meta.fieldNames(Kernel).len;
pub const n_headers = std.meta.fieldNames(Header).len;
/// Metal binds at most 31 buffers to a kernel; the combine's other three are `loc`, `w` and `out`.
pub const max_combine_parts = 31 - 3;
pub const max_inputs = max_combine_parts + 2;
pub const max_outputs = 1;
pub const max_template = 3;
pub const max_headers = 1;

/// One registered instantiation: the template's `K` (the top-k's k; the combine's routed rows per token) and the
/// combine's hidden size `H` (0 for the top-k).
pub const Instance = struct { k: u32, hidden: u32 = 0 };

/// The instantiations this build serves, which the manifest must register as they are: the top-k's (GLM-5.3's
/// index_topk 2048, the DSA kernels' fixture's 64, the tiny fixture's 8) and the combine's (GLM-5.3's hidden 6144 at
/// the routed top-8, the tiny fixture's 128). A route at any other is not installed (argpartition, the op chain's
/// combine).
pub const topk_instances = [_]Instance{ .{ .k = 2048 }, .{ .k = 64 }, .{ .k = 8 } };
pub const combine_instances = [_]Instance{ .{ .k = 8, .hidden = 6144 }, .{ .k = 8, .hidden = 128 } };

/// Whether the top-k route serves a top `k`: an instantiation the registry registers.
pub fn registersTopk(k: u32) bool {
    for (topk_instances) |i| if (i.k == k) return true;
    return false;
}

/// Whether the combine route serves `k` routed rows per token at hidden size `hidden`.
pub fn registersCombine(k: u32, hidden: u32) bool {
    for (combine_instances) |i| if (i.k == k and i.hidden == hidden) return true;
    return false;
}

pub fn familyOf(k: Kernel) Family {
    return if (k == .glm_moe_dsa_combine) .combine else .topk;
}

pub const Texts = struct {
    manifest: []const u8,
    sources: [n_kernels][:0]const u8,
    headers: [n_headers][:0]const u8,
};

pub const embedded: Texts = .{
    .manifest = @embedFile(dir ++ "manifest.json"),
    .sources = embedSources(),
    .headers = embedHeaders(),
};

fn embedSources() [n_kernels][:0]const u8 {
    var out: [n_kernels][:0]const u8 = undefined;
    inline for (std.meta.fieldNames(Kernel), 0..) |name, i| out[i] = @embedFile(dir ++ name ++ ".metal");
    return out;
}

fn embedHeaders() [n_headers][:0]const u8 {
    var out: [n_headers][:0]const u8 = undefined;
    inline for (comptime std.enums.values(Header), 0..) |h, i| out[i] = @embedFile(dir ++ comptime headerFile(h));
    return out;
}

pub const Refusal = error{
    ManifestNotPinned,
    ManifestSyntax,
    ManifestFormat,
    UnknownKernel,
    MissingKernel,
    DuplicateKernel,
    UnknownHeader,
    MissingHeader,
    TextSha256Mismatch,
    SchemaInvalid,
    InstancesNotServed,
    InstanceNotRegistered,
    NotGpuStream,
    KernelCreateFailed,
};

pub const Diag = @import("sdk").Diag;

const refuse = @import("../../sdk/weights/streamed/sdk_ext/expert/bank_file.zig").refuse;

/// An argument's name and dtype; an argument of dtype "T" takes the launch's template dtype T.
pub const Arg = struct { name: [:0]const u8, dtype: ?mlx.mlx_dtype };

pub const Entry = struct {
    kernel: Kernel,
    family: Family,
    source: [:0]const u8,
    headers: []const Header,
    /// the headers' texts in order, concatenated: the one header mlx compiles before `source` ("" without one)
    header_text: [:0]const u8,
    text_sha256: [32]u8,
    inputs: []const Arg,
    outputs: []const Arg,
    template: []const [:0]const u8,
    threadgroup: [3]u32,
    instances: []const Instance,
    checks: std.EnumSet(Check),

    pub fn registered(e: *const Entry, inst: Instance) bool {
        for (e.instances) |i| if (std.meta.eql(i, inst)) return true;
        return false;
    }
};

pub const Registry = struct {
    arena: std.heap.ArenaAllocator,
    entries: [n_kernels]Entry,
    headers: [n_headers][:0]const u8,
    manifest_hex: [64]u8,

    /// Parses the manifest, checks it against `pin` (hex sha256) and every text against it, once.
    pub fn init(gpa: Allocator, texts: *const Texts, pin: []const u8, diag: ?*Diag) (Refusal || Allocator.Error)!Registry {
        const got = try kset.checkPin(texts.manifest, pin, "glm trunk kernels", diag);
        var reg: Registry = .{ .arena = .init(gpa), .entries = undefined, .headers = texts.headers, .manifest_hex = got };
        errdefer reg.arena.deinit();
        const a = reg.arena.allocator();
        const m = std.json.parseFromSliceLeaky(JManifest, a, texts.manifest, .{ .allocate = .alloc_always }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ManifestSyntax, "glm trunk kernels: manifest does not parse ({t})", .{e}),
        };
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, "glm trunk kernels: format \"{s}\" is not " ++ format, .{m.format});
        try kset.adoptHeaders(Header, &texts.headers, m.headers, "glm trunk kernels", diag);
        for (m.headers) |h| {
            const id = std.meta.stringToEnum(Header, h.id).?;
            if (!std.mem.eql(u8, h.file, headerFile(id))) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: header {t} is file \"{s}\", not {s}", .{ id, h.file, headerFile(id) });
        }
        var tally: kset.Tally(Kernel) = .{};
        for (m.kernels) |jk| {
            const k = try tally.admit(jk.name, "glm trunk kernels", diag);
            reg.entries[@backingInt(k)] = try adoptKernel(a, texts, k, jk, diag);
        }
        try tally.done("glm trunk kernels", diag);
        return reg;
    }

    pub fn deinit(self: *Registry) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn get(self: *const Registry, k: Kernel) *const Entry {
        return &self.entries[@backingInt(k)];
    }

    /// `inst` of `k`, refused by name when the manifest does not register it.
    pub fn instance(self: *const Registry, k: Kernel, inst: Instance, diag: ?*Diag) Refusal!Instance {
        if (!self.get(k).registered(inst)) return refuse(diag, error.InstanceNotRegistered, "glm trunk kernels: {t} is not registered at K {d}, H {d}", .{ k, inst.k, inst.hidden });
        return inst;
    }

    /// Builds every kernel's mlx object on `stream`, once; the objects reach Metal at their first launch.
    pub fn bind(self: *const Registry, stream: mlx.mlx_stream, diag: ?*Diag) Refusal!Bound {
        if (!mlx.streamIsGpu(stream)) return refuse(diag, error.NotGpuStream, "glm trunk kernels: bind needs a GPU stream", .{});
        var b: Bound = .{ .reg = self, .stream = stream, .kernels = @splat(.{}) };
        errdefer b.deinit();
        for (&self.entries, 0..) |*e, i| {
            var in_names: [max_inputs][*:0]const u8 = undefined;
            var out_names: [max_outputs][*:0]const u8 = undefined;
            for (e.inputs, 0..) |arg, n| in_names[n] = arg.name.ptr;
            for (e.outputs, 0..) |arg, n| out_names[n] = arg.name.ptr;
            b.kernels[i] = kset.newKernel(@tagName(e.kernel), in_names[0..e.inputs.len], out_names[0..e.outputs.len], e.source.ptr, e.header_text.ptr, true) orelse
                return refuse(diag, error.KernelCreateFailed, "glm trunk kernels: mlx_fast_metal_kernel_new({t}) failed", .{e.kernel});
        }
        return b;
    }
};

/// One launch's geometry: the grid (threads), the threadgroup, the template values in the entry's order (ints; `T` a
/// dtype), the output's shape and dtype.
pub const LaunchConfig = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    ints: [max_template]i32 = @splat(0),
    dtype: mlx.mlx_dtype = .float32,
    n_out: usize = 1,
    out_ranks: [max_outputs]usize = @splat(0),
    out_shapes: [max_outputs][4]c_int = @splat(@splat(0)),
    out_dtypes: [max_outputs]mlx.mlx_dtype = @splat(.float32),
};

/// The top-k's launch: one threadgroup of 1024 threads per row, `K` = `k`, the int32 ids `[1, rows, k]` (the
/// one-buffer form, `rows_form` false) or `[rows, k]` (the rows form).
pub fn topkLaunch(rows: u32, k: u32, rows_form: bool) LaunchConfig {
    var c: LaunchConfig = .{ .grid = .{ 1024, rows, 1 }, .threadgroup = .{ 1024, 1, 1 } };
    c.ints[0] = @intCast(k);
    c.out_dtypes[0] = .int32;
    if (rows_form) {
        c.out_ranks[0] = 2;
        c.out_shapes[0] = .{ @intCast(rows), @intCast(k), 0, 0 };
    } else {
        c.out_ranks[0] = 3;
        c.out_shapes[0] = .{ 1, @intCast(rows), @intCast(k), 0 };
    }
    return c;
}

/// The combine's launch: one thread per token and 4 hidden values (`hidden` a multiple of 4), `T` = `dt`, `H` =
/// `hidden`, `K` = `k`; the output `[n, hidden]` in `dt`.
pub fn combineLaunch(n: u32, k: u32, hidden: u32, dt: mlx.mlx_dtype) LaunchConfig {
    std.debug.assert(hidden % 4 == 0);
    const h4 = hidden / 4;
    var c: LaunchConfig = .{ .grid = .{ h4, n, 1 }, .threadgroup = .{ @min(h4, 256), 1, 1 }, .dtype = dt };
    c.ints[1] = @intCast(hidden);
    c.ints[2] = @intCast(k);
    c.out_dtypes[0] = dt;
    c.out_ranks[0] = 2;
    c.out_shapes[0] = .{ @intCast(n), @intCast(hidden), 0, 0 };
    return c;
}

pub const Prepared = kset.Prepared(Kernel);

/// The kernels built on one GPU stream (`Registry.bind`); freed after that stream has drained: `apply` builds a
/// launch's mlx config for one launch (the routes' geometry varies per call), `observe` installs the profile hook's
/// launch probe.
pub const Bound = kset.BoundOf(@This());

/// `cfg` as the mlx launch (`Bound.prepare`): the entry's template parameters in its order, `T` the config's dtype.
pub fn launchOf(reg: *const Registry, k: Kernel, cfg: *const LaunchConfig, shapes: *[max_outputs][]const c_int, template: *[16]kset.TemplateArg) kset.Launch {
    const e = reg.get(k);
    for (e.template, template[0..e.template.len], 0..) |name, *t, i| t.* = .{ .name = name, .value = if (std.mem.eql(u8, name, "T")) .{ .dtype = cfg.dtype } else .{ .int = cfg.ints[i] } };
    for (0..cfg.n_out) |i| shapes[i] = cfg.out_shapes[i][0..cfg.out_ranks[i]];
    return .{ .grid = cfg.grid, .threadgroup = cfg.threadgroup, .out_shapes = shapes[0..cfg.n_out], .out_dtypes = cfg.out_dtypes[0..cfg.n_out], .template = template[0..e.template.len] };
}

/// The load's kernel set over this registry (`sdk_ext.kernels.KernelSet`).
pub const Set = kset.KernelSet(@This()).Set;

/// Device tests: the set on GPU stream `s` (a refusal printed by name).
pub fn openForTest(a: Allocator, s: mlx.mlx_stream) !*Set {
    var diag: Diag = .{};
    return Set.init(a, .{ .device = .{ .stream = s } }, &diag) catch |e| {
        std.debug.print("glm trunk kernels refused: {s}\n", .{diag.message()});
        return e;
    };
}

// ── Manifest adoption (once, at init) ──

const JText = struct { id: []const u8 = "", file: []const u8, sha256: []const u8, bytes: u64 };
const JArg = struct { name: []const u8, dtype: []const u8 };
const JInstance = struct { K: u32, H: u32 = 0 };
const JKernel = struct {
    name: []const u8,
    family: []const u8,
    source: JText,
    headers: []const []const u8,
    inputs: []const JArg,
    outputs: []const JArg,
    template: []const []const u8,
    threadgroup: [3]u32,
    instances: []const JInstance,
    checks: []const []const u8,
};
const JManifest = struct { format: []const u8, headers: []const JText, kernels: []const JKernel };

fn parseDtype(s: []const u8, k: Kernel, diag: ?*Diag) Refusal!?mlx.mlx_dtype {
    if (std.mem.eql(u8, s, "T")) return null;
    return std.meta.stringToEnum(mlx.mlx_dtype, s) orelse refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: unknown dtype \"{s}\"", .{ k, s });
}

fn adoptArgs(a: Allocator, js: []const JArg, max: usize, k: Kernel, diag: ?*Diag) (Refusal || Allocator.Error)![]const Arg {
    if (js.len == 0 or js.len > max) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: {d} arguments", .{ k, js.len });
    const out = try a.alloc(Arg, js.len);
    for (js, out, 0..) |j, *o, i| {
        for (js[0..i]) |prev| if (std.mem.eql(u8, prev.name, j.name)) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: argument {s} twice", .{ k, j.name });
        o.* = .{ .name = try a.dupeSentinel(u8, j.name, 0), .dtype = try parseDtype(j.dtype, k, diag) };
    }
    return out;
}

fn adoptKernel(a: Allocator, texts: *const Texts, k: Kernel, j: JKernel, diag: ?*Diag) (Refusal || Allocator.Error)!Entry {
    const source = texts.sources[@backingInt(k)];
    try kset.checkText(source, j.source.sha256, j.source.bytes, "glm trunk kernels", @tagName(k), diag);
    if (j.headers.len > max_headers) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: {d} headers", .{ k, j.headers.len });
    const headers = try a.alloc(Header, j.headers.len);
    var hbytes: usize = 0;
    for (j.headers, headers) |name, *o| {
        o.* = std.meta.stringToEnum(Header, name) orelse return refuse(diag, error.UnknownHeader, "glm trunk kernels: {t}: header \"{s}\"", .{ k, name });
        hbytes += texts.headers[@backingInt(o.*)].len;
    }
    const htext = try a.allocSentinel(u8, hbytes, 0);
    var at: usize = 0;
    for (headers) |h| {
        const t = texts.headers[@backingInt(h)];
        @memcpy(htext[at..][0..t.len], t);
        at += t.len;
    }
    const family = std.meta.stringToEnum(Family, j.family) orelse return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: family \"{s}\"", .{ k, j.family });
    if (family != familyOf(k)) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: family {t}, not {t}", .{ k, family, familyOf(k) });
    const tg = j.threadgroup;
    if (tg[0] == 0 or tg[1] == 0 or tg[2] == 0 or @as(u64, tg[0]) * tg[1] * tg[2] > 1024) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: threadgroup {d}x{d}x{d}", .{ k, tg[0], tg[1], tg[2] });
    if (j.template.len == 0 or j.template.len > max_template) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: {d} template parameters", .{ k, j.template.len });
    const template = try a.alloc([:0]const u8, j.template.len);
    for (j.template, template) |t, *o| o.* = try a.dupeSentinel(u8, t, 0);
    // The instantiations are the ones this build serves, as they are (`topk_instances`, `combine_instances`).
    const served: []const Instance = switch (family) {
        .topk => &topk_instances,
        .combine => &combine_instances,
    };
    const inst = try a.alloc(Instance, j.instances.len);
    for (j.instances, inst) |ji, *o| o.* = .{ .k = ji.K, .hidden = ji.H };
    if (inst.len != served.len or !for (inst, served) |x, y| {
        if (!std.meta.eql(x, y)) break false;
    } else true) return refuse(diag, error.InstancesNotServed, "glm trunk kernels: {t}: the manifest registers {d} instantiations, not the {d} this build serves", .{ k, inst.len, served.len });
    var checks: std.EnumSet(Check) = .empty;
    for (j.checks) |c| checks.insert(std.meta.stringToEnum(Check, c) orelse return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: check \"{s}\"", .{ k, c }));
    if (!checks.contains(.compile)) return refuse(diag, error.SchemaInvalid, "glm trunk kernels: {t}: no compile check", .{k});
    return .{
        .kernel = k,
        .family = family,
        .source = source,
        .headers = headers,
        .header_text = htext,
        .text_sha256 = kset.hexSha(j.source.sha256).?,
        .inputs = try adoptArgs(a, j.inputs, max_inputs, k, diag),
        .outputs = try adoptArgs(a, j.outputs, max_outputs, k, diag),
        .template = template,
        .threadgroup = tg,
        .instances = inst,
        .checks = checks,
    };
}

// ── Tests (host only: nothing here creates an MLX array or kernel) ──

const testing = std.testing;
const shaHex = kset.shaHex;

fn initOrPrint(texts: *const Texts, pin: []const u8) !Registry {
    var diag: Diag = .{};
    return Registry.init(testing.allocator, texts, pin, &diag) catch |e| {
        std.debug.print("glm trunk kernels refused: {s}\n", .{diag.message()});
        return e;
    };
}

test "glm trunk kernels: the embedded manifest is the pinned one; the top-k and the combine at the instantiations this build serves" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    for (reg.entries, 0..) |e, i| {
        try testing.expectEqual(@as(Kernel, @fromBackingInt(@intCast(i))), e.kernel);
        try testing.expect(e.checks.contains(.compile));
    }
    for (topk_kernels, 0..) |k, rows| {
        const e = reg.get(k);
        try testing.expectEqual(Family.topk, e.family);
        // the one-buffer form reads `scores`; the rows form one input per row; both `params` last
        try testing.expectEqual(@as(usize, @max(rows, 1) + 1), e.inputs.len);
        try testing.expectEqualStrings("params", e.inputs[e.inputs.len - 1].name);
        try testing.expectEqualStrings("", e.header_text);
        for (topk_instances) |inst| _ = try reg.instance(k, inst, null);
    }
    try testing.expectEqual(Kernel.glm_moe_dsa_topk_rows3, topkRowsKernel(3));
    const c = reg.get(.glm_moe_dsa_combine);
    try testing.expectEqual(@as(usize, max_combine_parts + 2), c.inputs.len);
    try testing.expect(c.inputs[0].dtype == null and c.outputs[0].dtype == null);
    try testing.expectEqualStrings("#pragma METAL fp contract(off)\n", c.header_text);
    try testing.expectEqual(@as(usize, 3), c.template.len);
    try testing.expect(registersTopk(2048) and registersTopk(8) and !registersTopk(1024));
    try testing.expect(registersCombine(8, 6144) and registersCombine(8, 128) and !registersCombine(6, 6144));
    var diag: Diag = .{};
    try testing.expectError(error.InstanceNotRegistered, reg.instance(.glm_moe_dsa_combine, .{ .k = 8, .hidden = 5120 }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "glm_moe_dsa_combine is not registered at K 8, H 5120") != null);
}

test "glm trunk kernels: a tampered text or header, another pin, an unknown kernel or other instantiations are refused by name" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var diag: Diag = .{};
    {
        const bad = try a.dupeSentinel(u8, embedded.sources[@backingInt(Kernel.glm_moe_dsa_topk_rows5)], 0);
        defer a.free(bad);
        bad[bad.len / 2] ^= 0x20;
        var texts = embedded;
        texts.sources[@backingInt(Kernel.glm_moe_dsa_topk_rows5)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "glm_moe_dsa_topk_rows5") != null);
    }
    {
        const bad = try a.dupeSentinel(u8, embedded.headers[@backingInt(Header.fp_contract_off)], 0);
        defer a.free(bad);
        bad[0] ^= 0x01;
        var texts = embedded;
        texts.headers[@backingInt(Header.fp_contract_off)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "fp_contract_off") != null);
    }
    try testing.expectError(error.ManifestNotPinned, Registry.init(a, &embedded, &(@as([64]u8, @splat('0'))), &diag));
    const Case = struct { needle: []const u8, repl: []const u8, err: Refusal, names: []const u8 };
    for ([_]Case{
        .{ .needle = "\"name\":\"glm_moe_dsa_combine\"", .repl = "\"name\":\"glm_moe_dsa_combine2\"", .err = error.UnknownKernel, .names = "glm_moe_dsa_combine2" },
        .{ .needle = "\"name\":\"glm_moe_dsa_topk_rows8\"", .repl = "\"name\":\"glm_moe_dsa_topk_rows7\"", .err = error.DuplicateKernel, .names = "glm_moe_dsa_topk_rows7 listed twice" },
        .{ .needle = "\"format\":\"mlx-stream-glm-trunk-kernels-v1\"", .repl = "\"format\":\"mlx-stream-glm-trunk-kernels-v2\"", .err = error.ManifestFormat, .names = "v2" },
        .{ .needle = "\"instances\":[{\"H\":6144,\"K\":8},{\"H\":128,\"K\":8}]", .repl = "\"instances\":[{\"H\":6144,\"K\":8}]", .err = error.InstancesNotServed, .names = "glm_moe_dsa_combine: the manifest registers 1" },
        .{ .needle = "\"threadgroup\":[256,1,1]", .repl = "\"threadgroup\":[2048,1,1]", .err = error.SchemaInvalid, .names = "threadgroup 2048x1x1" },
    }) |c| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, embedded.manifest, c.needle));
        const m = try std.mem.replaceOwned(u8, a, embedded.manifest, c.needle, c.repl);
        defer a.free(m);
        var texts = embedded;
        texts.manifest = m;
        try testing.expectError(c.err, Registry.init(a, &texts, &shaHex(m), &diag));
        testing.expect(std.mem.indexOf(u8, diag.message(), c.names) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ c.names, diag.message() });
            return e;
        };
    }
}

test "glm trunk kernels: the launches the routes and the self-check share" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const t = topkLaunch(3, 2048, true);
    try testing.expectEqual([3]u32{ 1024, 3, 1 }, t.grid);
    try testing.expectEqualSlices(c_int, &.{ 3, 2048 }, t.out_shapes[0][0..t.out_ranks[0]]);
    const one = topkLaunch(5, 8, false);
    try testing.expectEqualSlices(c_int, &.{ 1, 5, 8 }, one.out_shapes[0][0..one.out_ranks[0]]);
    const c = combineLaunch(3, 8, 6144, .bfloat16);
    try testing.expectEqual([3]u32{ 1536, 3, 1 }, c.grid);
    try testing.expectEqual([3]u32{ 256, 1, 1 }, c.threadgroup);
    try testing.expectEqualSlices(c_int, &.{ 3, 6144 }, c.out_shapes[0][0..c.out_ranks[0]]);
    try testing.expectEqual([3]u32{ 32, 1, 1 }, combineLaunch(1, 8, 128, .bfloat16).threadgroup);
    // `T`, `H`, `K` in the entry's order: the template the combine's text names
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    var shapes: [max_outputs][]const c_int = undefined;
    var template: [16]kset.TemplateArg = undefined;
    const l = launchOf(&reg, .glm_moe_dsa_combine, &c, &shapes, &template);
    try testing.expectEqualStrings("T", l.template[0].name);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, l.template[0].value.dtype);
    try testing.expectEqual(@as(i32, 6144), l.template[1].value.int);
    try testing.expectEqual(@as(i32, 8), l.template[2].value.int);
}
