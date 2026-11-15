//! Directional ablation (Arditi et al. 2024) in the built-in DeepSeek-V4.1 module: a request's direction read from
//! `abi.Forward.ablation`, the ops that take it out of every write to the residual stream, and the calibration tap
//! that records the stream entering each layer. Stubs: the spec in `ablation_test.zig` fails against them.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Scope = dsv41.Scope;

pub const Refusal = error{ AblationStrength, AblationSlots };

/// Whether this tree's ABI has `Forward.ablation`; without it the module serves none.
pub const supported = @hasField(abi.Forward, "ablation");

/// One slot's ablation as the forward hands it: the directions table `[slots, hidden]` f32, the slot's row of it, its
/// strength and the host's key (equal keys ablate identically; 0 = none given).
pub const Spec = struct { directions: abi.Array, row: usize = 0, strength: f32, key: u64 = 0 };

pub fn ofSlot(f: *const abi.Forward, k: usize) Refusal!?Spec {
    _ = f;
    _ = k;
    return null;
}

pub fn of(f: *const abi.Forward) Refusal!?Spec {
    return ofSlot(f, 0);
}

/// An armed ablation: the unit direction `r` `[dim]`, the same as a column `col` `[dim, 1]` and the strength `s`, all
/// f32, with the key of the table it came from.
pub const Ablation = struct {
    r: A,
    col: A,
    s: A,
    key: u64 = 0,

    pub fn deinit(self: Ablation) void {
        for ([_]A{ self.r, self.col, self.s }) |x| _ = mlx.mlx_array_free(x);
    }
};

pub fn armed(a: Spec, dim: usize, s: S) !Ablation {
    _ = a;
    _ = dim;
    _ = s;
    return error.AblationShape;
}

pub fn same(a: ?Ablation, b: ?Ablation) bool {
    _ = a;
    _ = b;
    return false;
}

/// The variant of an ablated slot whose host gave no key.
pub const unkeyed: u64 = std.math.maxInt(u64);

pub fn variantOf(a: ?Spec) u64 {
    _ = a;
    return 0;
}

pub fn fits(n: usize, variant: ?u64, key: u64) bool {
    _ = n;
    _ = variant;
    _ = key;
    return false;
}

pub fn ablateWith(sc: *Scope, ab: Ablation, y: A) !A {
    _ = sc;
    _ = ab;
    return y;
}

pub fn ablateExpandedWith(sc: *Scope, ab: Ablation, x1: A, x2: A, comb: A) !A {
    _ = sc;
    _ = ab;
    _ = x1;
    _ = comb;
    return x2;
}

pub fn ablateValueWith(sc: *Scope, ab: Ablation, kv: A, split: usize) !A {
    _ = sc;
    _ = ab;
    _ = split;
    return kv;
}

/// Calibration: per layer, the stream entering it (and, last, leaving the final layer).
pub const StreamTap = struct {
    gpa: std.mem.Allocator,
    keep: usize,
    sites: []std.ArrayList(A),
    on: bool = false,

    pub fn init(gpa: std.mem.Allocator, layers: usize, keep: usize) !StreamTap {
        const sites = try gpa.alloc(std.ArrayList(A), layers + 1);
        for (sites) |*s| s.* = .empty;
        return .{ .gpa = gpa, .keep = keep, .sites = sites };
    }

    pub fn record(self: *StreamTap, s: S, site: usize, xs: []const A) !void {
        _ = self;
        _ = s;
        _ = site;
        _ = xs;
    }

    pub fn clear(self: *StreamTap) void {
        for (self.sites) |*s| {
            for (s.items) |a| _ = mlx.mlx_array_free(a);
            s.clearRetainingCapacity();
        }
    }

    pub fn deinit(self: *StreamTap) void {
        self.clear();
        for (self.sites) |*s| s.deinit(self.gpa);
        self.gpa.free(self.sites);
    }
};

pub fn npyHeader(buf: []u8, a: usize, b: usize, c: usize) ![]const u8 {
    _ = a;
    _ = b;
    _ = c;
    return buf[0..0];
}
