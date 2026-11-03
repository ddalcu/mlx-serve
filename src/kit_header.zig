//! Writes the plugin kit's C header (`include/mlx_serve_plugin.h`) from the ABI's own types, for plugins written in C,
//! C++ or Rust (bindgen): every extern struct, enum and packed flag word a plugin table reaches, with a static assert
//! on each struct's size and field offsets, so a compiler that lays a table out differently fails at its build.
//! Run by `zig build kit` and `zig build test` (build/kit.zig), which also compile the header as C11 and C++17.

const std = @import("std");
const common = @import("abi/common.zig");
const dylib = @import("abi/dylib.zig");
const model = @import("abi/plugin.zig");
const media = @import("abi/media.zig");
const quant = @import("abi/quant.zig");
const host = @import("abi/host.zig");

const roots = [_]type{ dylib.Manifest, model.ModelTable, media.MediaTable, quant.QuantTable, host.Host, host.HostGpu };

pub fn main(init: std.process.Init) !void {
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer it.deinit();
    _ = it.skip();
    const path = it.next() orelse std.process.fatal("usage: kit_header <out.h>", .{});
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(init.gpa);
    try write(init.gpa, &out);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = out.items });
}

/// Every named struct, union, enum and packed word reachable from `roots`, dependencies of a by-value field first.
const types: []const type = blk: {
    @setEvalBranchQuota(1_000_000);
    var acc: []const type = &.{};
    for (roots) |T| acc = collect(T, acc);
    break :blk acc;
};

fn has(comptime list: []const type, comptime T: type) bool {
    for (list) |x| if (x == T) return true;
    return false;
}

fn collect(comptime T: type, comptime acc: []const type) []const type {
    if (has(acc, T)) return acc;
    var a = acc;
    switch (@typeInfo(T)) {
        inline .@"struct", .@"union" => |s| {
            if (s.layout == .@"packed") return a ++ .{T};
            if (s.layout != .@"extern") @compileError("kit header: " ++ @typeName(T) ++ " crosses the ABI but is not extern");
            // Pointer targets may come later (forward-declared); by-value fields first.
            for (s.field_types) |F| if (byValue(F)) |V| {
                a = collect(V, a);
            };
            a = a ++ .{T};
            for (s.field_types) |F| a = collectRefs(F, a);
            return a;
        },
        .@"enum" => return a ++ .{T},
        else => return collectRefs(T, a),
    }
}

/// The named type a by-value field stores (through arrays), or null.
fn byValue(comptime T: type) ?type {
    return switch (@typeInfo(T)) {
        .@"struct", .@"union", .@"enum" => T,
        .array => |x| byValue(x.child),
        else => null,
    };
}

/// The named types a field's pointers and function signatures reach.
fn collectRefs(comptime T: type, comptime acc: []const type) []const type {
    return switch (@typeInfo(T)) {
        .@"struct", .@"union", .@"enum" => collect(T, acc),
        .pointer => |p| if (p.child == anyopaque or @typeInfo(p.child) == .@"opaque") acc else collectRefs(p.child, acc),
        .optional => |o| collectRefs(o.child, acc),
        .array => |x| collectRefs(x.child, acc),
        .@"fn" => |f| blk: {
            var a = acc;
            for (f.param_types) |P| a = collectRefs(P.?, a);
            break :blk collectRefs(f.return_type.?, a);
        },
        else => acc,
    };
}

/// `abi.<file>.Name` → `Name` (generic arguments too), `<file>_Name` where two files share a name.
fn cName(comptime T: type) []const u8 {
    return comptime blk: {
        @setEvalBranchQuota(10_000_000);
        const short = strip(@typeName(T), false);
        for (types) |U| if (U != T and std.mem.eql(u8, strip(@typeName(U), false), short)) break :blk "mlxs_" ++ strip(@typeName(T), true);
        break :blk "mlxs_" ++ short;
    };
}

fn strip(comptime name: []const u8, comptime keep_file: bool) []const u8 {
    return comptime blk: {
        @setEvalBranchQuota(10_000_000);
        var out: []const u8 = "";
        var i: usize = 0;
        while (i < name.len) {
            if (std.mem.startsWith(u8, name[i..], "?*anyopaque")) {
                out = out ++ "ptr";
                i += "?*anyopaque".len;
                continue;
            }
            if (std.mem.startsWith(u8, name[i..], "abi.")) {
                const file_end = std.mem.indexOfScalarPos(u8, name, i + 4, '.').?;
                if (keep_file) out = out ++ name[i + 4 .. file_end] ++ "_";
                i = file_end + 1;
                continue;
            }
            const c = name[i];
            out = out ++ if (std.ascii.isAlphanumeric(c) or c == '_') &[_]u8{c} else if (c == ')') "" else "_";
            i += 1;
        }
        break :blk out;
    };
}

fn intName(comptime T: type) ?[]const u8 {
    if (T == c_int) return "int";
    if (T == usize) return "size_t";
    if (T == isize) return "ptrdiff_t";
    if (T == bool) return "bool";
    if (T == f32) return "float";
    if (T == f64) return "double";
    return switch (@typeInfo(T)) {
        .int => |i| std.fmt.comptimePrint("{s}int{d}_t", .{ if (i.signedness == .unsigned) "u" else "", i.bits }),
        else => null,
    };
}

/// The C type of `T` as a prefix to a declarator (`R` of `R name`); `fn_name` names a function pointer's typedef.
fn cType(comptime T: type, comptime fn_name: []const u8) []const u8 {
    return comptime if (intName(T)) |n| n else switch (@typeInfo(T)) {
        .void => "void",
        .@"struct", .@"union", .@"enum" => cName(T),
        .optional => |o| cType(o.child, fn_name),
        .pointer => |p| if (@typeInfo(p.child) == .@"fn") fn_name else ptrTo(p),
        else => @compileError("kit header: no C spelling for " ++ @typeName(T)),
    };
}

fn ptrTo(comptime p: anytype) []const u8 {
    const target = if (p.child == anyopaque or @typeInfo(p.child) == .@"opaque") "void" else cType(p.child, "");
    return comptime (if (p.attrs.@"const") "const " else "") ++ target ++ "*";
}

/// The function type behind a (possibly optional) function pointer, or null.
fn fnOf(comptime T: type) ?type {
    return switch (@typeInfo(T)) {
        .optional => |o| fnOf(o.child),
        .pointer => |p| if (@typeInfo(p.child) == .@"fn") p.child else null,
        else => null,
    };
}

fn writeFnTypedef(a: std.mem.Allocator, out: *std.ArrayList(u8), comptime F: type, comptime name: []const u8) !void {
    const f = @typeInfo(F).@"fn";
    inline for (f.param_types, 0..) |P, i| if (comptime fnOf(P.?)) |inner| try writeFnTypedef(a, out, inner, std.fmt.comptimePrint("{s}_p{d}", .{ name, i }));
    try out.print(a, "typedef {s} (*{s})(", .{ cType(f.return_type.?, ""), name });
    if (f.param_types.len == 0) try out.appendSlice(a, "void");
    inline for (f.param_types, 0..) |P, i| {
        if (i > 0) try out.appendSlice(a, ", ");
        try out.appendSlice(a, cType(P.?, std.fmt.comptimePrint("{s}_p{d}", .{ name, i })));
    }
    try out.appendSlice(a, ");\n");
}

/// C and C++ keywords a Zig field may be named; the header spells such a field with a trailing `_`.
const keywords = [_][]const u8{
    "auto",      "bool",     "case",    "catch",    "char",     "class",     "const",    "continue", "default",
    "delete",    "do",       "double",  "else",     "enum",     "explicit",  "export",   "extern",   "false",
    "float",     "for",      "friend",  "goto",     "if",       "inline",    "int",      "long",     "mutable",
    "namespace", "new",      "nullptr", "operator", "private",  "protected", "public",   "register", "restrict",
    "return",    "short",    "signed",  "sizeof",   "static",   "struct",    "switch",   "template", "this",
    "throw",     "true",     "try",     "typedef",  "typename", "union",     "unsigned", "using",    "virtual",
    "void",      "volatile", "while",
};

fn fieldName(comptime name: []const u8) []const u8 {
    return comptime for (keywords) |k| {
        if (std.mem.eql(u8, k, name)) break name ++ "_";
    } else name;
}

fn writeField(a: std.mem.Allocator, out: *std.ArrayList(u8), comptime owner: []const u8, comptime name: []const u8, comptime F: type) !void {
    const fn_name = owner ++ "_" ++ name ++ "_fn";
    switch (@typeInfo(F)) {
        .array => |x| try out.print(a, "    {s} {s}[{d}];\n", .{ cType(x.child, fn_name), fieldName(name), x.len }),
        else => try out.print(a, "    {s} {s};\n", .{ cType(F, fn_name), fieldName(name) }),
    }
}

const Version = struct { []const u8, u32 };
const versions = [_]Version{
    .{ "MLXS_MANIFEST_VERSION", dylib.version },
    .{ "MLXS_MODEL_TABLE_VERSION", model.version },
    .{ "MLXS_MEDIA_TABLE_VERSION", media.version },
    .{ "MLXS_QUANT_TABLE_VERSION", quant.version },
    .{ "MLXS_HOST_VERSION", host.version },
    .{ "MLXS_HOST_GPU_VERSION", host.gpu_version },
};

pub fn write(a: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
    try out.appendSlice(a,
        \\/* mlx_serve_plugin.h: the mlx-serve plugin ABI for C, C++ and Rust (bindgen). Generated from src/abi by the
        \\ * plugin kit; do not edit. A plugin is a dylib exporting one symbol, `
    ++ dylib.symbol ++
        \\`, a const mlxs_Manifest whose `table` points
        \\ * at its kind's table (mlxs_ModelTable, mlxs_MediaTable or mlxs_QuantTable). Strings and slices cross as pointer
        \\ * + length (never NUL-terminated); every struct passed by pointer opens with `size` and `version` and grows at its
        \\ * end; memory is freed by whoever allocated it. mlx-c's arrays and streams cross as their `ctx` handles. */
        \\#ifndef MLX_SERVE_PLUGIN_H
        \\#define MLX_SERVE_PLUGIN_H
        \\#include <stdbool.h>
        \\#include <stddef.h>
        \\#include <stdint.h>
        \\#ifdef __cplusplus
        \\#define MLXS_ASSERT(c, m) static_assert(c, m)
        \\extern "C" {
        \\#else
        \\#define MLXS_ASSERT(c, m) _Static_assert(c, m)
        \\#endif
        \\
        \\
    );
    inline for (versions) |v| try out.print(a, "#define {s} {d}u\n", .{ v[0], v[1] });
    try out.print(a, "#define MLXS_SYMBOL \"{s}\"\n\n", .{dylib.symbol});

    // Forward declarations, then enums and flag words, then function pointers, then the structs.
    inline for (types) |T| switch (@typeInfo(T)) {
        .@"struct" => |s| if (s.layout == .@"extern") try out.print(a, "typedef struct {0s} {0s};\n", .{cName(T)}),
        .@"union" => try out.print(a, "typedef union {0s} {0s};\n", .{cName(T)}),
        else => {},
    };
    try out.append(a, '\n');
    inline for (types) |T| switch (@typeInfo(T)) {
        .@"enum" => |e| {
            try out.print(a, "typedef {s} {s};\nenum {{", .{ intName(e.tag_type).?, cName(T) });
            inline for (e.field_names, e.field_values, 0..) |n, v, i| try out.print(a, "{s} {s}_{s} = {d}", .{ if (i == 0) "" else ",", cName(T), n, v });
            try out.appendSlice(a, " };\n");
        },
        .@"struct" => |s| if (s.layout == .@"packed") {
            try out.print(a, "typedef {s} {s};\n", .{ intName(s.backing_integer.?).?, cName(T) });
            inline for (s.field_names, s.field_types) |n, F| {
                const bit = @bitOffsetOf(T, n);
                if (F == bool) {
                    try out.print(a, "#define {s}_{s} ((({s})1) << {d})\n", .{ cName(T), n, cName(T), bit });
                } else try out.print(a, "#define {s}_{s}_shift {d}\n#define {s}_{s}_bits {d}\n", .{ cName(T), n, bit, cName(T), n, @bitSizeOf(F) });
            }
        },
        else => {},
    };
    try out.append(a, '\n');
    inline for (types) |T| switch (@typeInfo(T)) {
        inline .@"struct", .@"union" => |s| if (s.layout == .@"extern") inline for (s.field_names, s.field_types) |n, F| if (comptime fnOf(F)) |Fn| try writeFnTypedef(a, out, Fn, cName(T) ++ "_" ++ n ++ "_fn"),
        else => {},
    };
    try out.append(a, '\n');
    inline for (types) |T| switch (@typeInfo(T)) {
        inline .@"struct", .@"union" => |s, tag| if (s.layout == .@"extern") {
            try out.print(a, "{s} {s} {{\n", .{ if (tag == .@"union") "union" else "struct", cName(T) });
            inline for (s.field_names, s.field_types) |n, F| try writeField(a, out, cName(T), n, F);
            try out.appendSlice(a, "};\n");
            try out.print(a, "MLXS_ASSERT(sizeof({0s}) == {1d}, \"{0s} size\");\n", .{ cName(T), @sizeOf(T) });
            if (tag == .@"struct") inline for (s.field_names) |n| try out.print(a, "MLXS_ASSERT(offsetof({0s}, {1s}) == {2d}, \"{0s}.{1s} offset\");\n", .{ cName(T), fieldName(n), @offsetOf(T, n) });
            try out.append(a, '\n');
        },
        else => {},
    };
    try out.print(a,
        \\/* The plugin's one export: `const mlxs_Manifest {s} = {{ sizeof(mlxs_Manifest), MLXS_MANIFEST_VERSION, ... }};` */
        \\extern const mlxs_Manifest {s};
        \\
        \\#ifdef __cplusplus
        \\}}
        \\#endif
        \\#endif
        \\
    , .{ dylib.symbol, dylib.symbol });
}
