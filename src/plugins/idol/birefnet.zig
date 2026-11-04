//! BiRefNet (ZhengPeng7/BiRefNet, Swin-L backbone, as Idol ships it): the background matte of a photo with no alpha.
//! The image runs through the Swin-L twice, at 1024 and at 512, the two feature pyramids are concatenated, and a
//! decoder of deformable-ASPP blocks (the sampling is a gather, the offsets and masks come from convs) turns them
//! into one alpha plane. f16 as the reference runs it on a Mac, channels-last; the deformable sampling positions in f32.
//! The released tensors load as they are (OIHW convs are turned to OHWI here, batch norms folded to a scale and shift
//! in f32 first).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const scope_mod = @import("../../mlx/scope.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const glue = @import("glue.zig");

const A = mlx.mlx_array;
const Scope = scope_mod.Scope;
const Weights = model_mod.Weights;
const NONE = A{ .ctx = null };

const WINDOW: c_int = 12;
const SHIFT: c_int = 6;
const DEPTHS = [4]usize{ 2, 2, 18, 2 };
const HEADS = [4]c_int{ 6, 12, 24, 48 };
const EMBED: c_int = 192;
const SIZE: usize = 1024;
const BN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-5;
const MEAN = [3]f32{ 0.485, 0.456, 0.406 };
const STD = [3]f32{ 0.229, 0.224, 0.225 };
const ACT: mlx.mlx_dtype = .float16;
/// The padded planes of a 1024 image's two passes (1024 and 512) through the four stages; a shift mask per size.
const PLANES = [_]c_int{ 264, 132, 72, 36, 24 };

pub const Net = struct {
    gpa: std.mem.Allocator,
    s: mlx.mlx_stream,
    w: Weights,
    /// The relative-position bias of every Swin block, `[1, heads, 144, 144]`, built once.
    bias: [4][]A,
    /// The shifted windows' additive mask per padded plane in `PLANES`, built at first use.
    masks: [PLANES.len]?A = @splat(null),

    /// Takes ownership of `w` (a `birefnet.safetensors`).
    pub fn load(gpa: std.mem.Allocator, s: mlx.mlx_stream, w_in: Weights) !Net {
        var net: Net = .{ .gpa = gpa, .s = s, .w = w_in, .bias = undefined };
        errdefer net.w.deinit();
        try net.prepare();
        var made: usize = 0;
        errdefer for (0..made) |i| for (net.bias[i]) |b| {
            _ = mlx.mlx_array_free(b);
        };
        for (0..4) |stage| {
            net.bias[stage] = try gpa.alloc(A, DEPTHS[stage]);
            errdefer gpa.free(net.bias[stage]);
            for (net.bias[stage], 0..) |*b, blk| b.* = try net.relativeBias(stage, blk);
            made += 1;
        }
        return net;
    }

    pub fn deinit(self: *Net) void {
        for (self.bias) |bs| {
            for (bs) |b| _ = mlx.mlx_array_free(b);
            self.gpa.free(bs);
        }
        for (self.masks) |m| if (m) |a| {
            _ = mlx.mlx_array_free(a);
        };
        self.w.deinit();
    }

    fn setW(self: *Net, key: []const u8, v: A) !void {
        const e = self.w.map.getPtr(key) orelse return error.MissingWeight;
        _ = mlx.mlx_array_free(e.*);
        e.* = v;
    }

    /// OIHW convs to OHWI, each batch norm's folded `scale` / `shift` beside it (in f32), then every float in `ACT`.
    fn prepare(self: *Net) !void {
        const s = self.s;
        var keys: std.ArrayList([]const u8) = .empty;
        defer keys.deinit(self.gpa);
        var it = self.w.map.keyIterator();
        while (it.next()) |k| try keys.append(self.gpa, k.*);
        for (keys.items) |key| {
            const a = self.w.get(key).?;
            const dt = mlx.mlx_array_dtype(a);
            if (dt != .float16 and dt != .float32 and dt != .bfloat16) continue;
            var o = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(o);
            if (std.mem.endsWith(u8, key, ".weight") and mlx.mlx_array_ndim(a) == 4) {
                var t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(t);
                try mlx.check(mlx.mlx_transpose_axes(&t, a, &[_]c_int{ 0, 2, 3, 1 }, 4, s));
                try mlx.check(mlx.mlx_astype(&o, t, .float32, s));
            } else try mlx.check(mlx.mlx_astype(&o, a, .float32, s));
            try mlx.check(mlx.mlx_array_eval(o));
            try self.setW(key, o);
        }
        for (keys.items) |key| {
            if (!std.mem.endsWith(u8, key, ".running_var")) continue;
            const base = key[0 .. key.len - ".running_var".len];
            var sc = Scope.init(self.gpa, s);
            defer sc.deinit();
            const w = try self.get("{s}.weight", .{base});
            const b = try self.get("{s}.bias", .{base});
            const rm = try self.get("{s}.running_mean", .{base});
            const rv = self.w.get(key).?;
            var inv = mlx.mlx_array_new();
            const rs = try sc.res(mlx.mlx_rsqrt(&inv, try sc.addS(rv, BN_EPS), s), &inv);
            const scale = try sc.mul(w, rs);
            const shift = try sc.sub(b, try sc.mul(rm, scale));
            try mlx.check(mlx.mlx_array_eval(scale));
            try mlx.check(mlx.mlx_array_eval(shift));
            const k1 = try std.fmt.allocPrint(self.gpa, "{s}.scale", .{base});
            const k2 = try std.fmt.allocPrint(self.gpa, "{s}.shift", .{base});
            try self.w.map.put(k1, sc.out(scale));
            try self.w.map.put(k2, sc.out(shift));
        }
        var all = self.w.map.valueIterator();
        while (all.next()) |v| {
            if (mlx.mlx_array_dtype(v.*) != .float32) continue;
            var o = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_astype(&o, v.*, ACT, s));
            try mlx.check(mlx.mlx_array_eval(o));
            _ = mlx.mlx_array_free(v.*);
            v.* = o;
        }
    }

    fn get(self: *const Net, comptime fmt: []const u8, args: anytype) !A {
        return scope_mod.getW(&self.w, fmt, args);
    }

    /// `relative_position_bias_table[relative_position_index]`, `[1, heads, 144, 144]`.
    fn relativeBias(self: *Net, stage: usize, blk: usize) !A {
        var sc = Scope.init(self.gpa, self.s);
        defer sc.deinit();
        const table = try self.get("bb.layers.{d}.blocks.{d}.attn.relative_position_bias_table", .{ stage, blk });
        const idx = try self.get("bb.layers.{d}.blocks.{d}.attn.relative_position_index", .{ stage, blk });
        const n: c_int = WINDOW * WINDOW;
        const flat = try sc.reshape(try sc.astype(idx, .int32), &.{n * n});
        const g = try sc.reshape(try sc.take(table, flat), &.{ n, n, HEADS[stage] });
        const out = try sc.reshape(try sc.transpose(g, &.{ 2, 0, 1 }), &.{ 1, HEADS[stage], n, n });
        try mlx.check(mlx.mlx_array_eval(out));
        return sc.out(out);
    }

    // ── small ops ──

    fn ln(self: *const Net, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype) !A {
        return sc.layerNorm(x, try self.get(fmt ++ ".weight", args), try self.get(fmt ++ ".bias", args), LN_EPS);
    }

    fn linear(self: *const Net, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype) !A {
        const w = try self.get(fmt ++ ".weight", args);
        var buf: [200]u8 = undefined;
        const bkey = try std.fmt.bufPrint(&buf, fmt ++ ".bias", args);
        return sc.lin(x, w, self.w.get(bkey));
    }

    /// `conv2d` (channels-last, OHWI) with the bias when the checkpoint has one.
    fn conv(self: *const Net, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype, stride: c_int, pad: c_int) !A {
        const y = try sc.conv2d(x, try self.get(fmt ++ ".weight", args), stride, pad);
        var buf: [200]u8 = undefined;
        const bkey = try std.fmt.bufPrint(&buf, fmt ++ ".bias", args);
        return if (self.w.get(bkey)) |b| sc.add(y, b) else y;
    }

    /// Inference batch norm as the folded `x * scale + shift`.
    fn bn(self: *const Net, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype) !A {
        return sc.add(try sc.mul(x, try self.get(fmt ++ ".scale", args)), try self.get(fmt ++ ".shift", args));
    }

    /// `x[:, ::2, ::2]`-style strided slice of `[1, H, W, C]` starting at (`oy`, `ox`).
    fn every2(sc: *Scope, x: A, oy: c_int, ox: c_int) !A {
        const sh = mlx.getShape(x);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &[_]c_int{ 0, oy, ox, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &[_]c_int{ 1, 2, 2, 1 }, 4, sc.s), &o);
    }

    /// `torch.roll` by `n` along `axis`.
    fn roll(sc: *Scope, x: A, n: c_int, axis: usize) !A {
        const len = mlx.getShape(x)[axis];
        const k = @mod(n, len);
        if (k == 0) return x;
        return sc.concat(&.{ try sc.slice(x, axis, len - k, len), try sc.slice(x, axis, 0, len - k) }, @intCast(axis));
    }

    /// Bilinear resize with `align_corners=True` of `[1, H, W, C]` to `oh` x `ow`, as two dense interpolation matmuls.
    fn resize(self: *const Net, sc: *Scope, x: A, oh: c_int, ow: c_int) !A {
        const sh = mlx.getShape(x);
        if (sh[1] == oh and sh[2] == ow) return x;
        const h = sh[1];
        const w = sh[2];
        const c = sh[3];
        const dt = mlx.mlx_array_dtype(x);
        const wy = try sc.astype(try self.interp(sc, oh, h), dt);
        const wx = try sc.astype(try self.interp(sc, ow, w), dt);
        const rows = try sc.matmul(wy, try sc.reshape(x, &.{ h, w * c })); // [oh, W*C]
        const t = try sc.transpose(try sc.reshape(rows, &.{ oh, w, c }), &.{ 1, 0, 2 }); // [W, oh, C]
        const cols = try sc.matmul(wx, try sc.reshape(t, &.{ w, oh * c })); // [ow, oh*C]
        return sc.reshape(try sc.transpose(try sc.reshape(cols, &.{ ow, oh, c }), &.{ 1, 0, 2 }), &.{ 1, oh, ow, c });
    }

    fn interp(self: *const Net, sc: *Scope, out: c_int, in: c_int) !A {
        const m = try self.gpa.alloc(f32, @as(usize, @intCast(out)) * @as(usize, @intCast(in)));
        defer self.gpa.free(m);
        @memset(m, 0);
        for (0..@intCast(out)) |o| {
            const src: f64 = if (out == 1) 0 else @as(f64, @floatFromInt(o)) * @as(f64, @floatFromInt(in - 1)) / @as(f64, @floatFromInt(out - 1));
            const lo: usize = @intFromFloat(@floor(src));
            const hi_i = @min(lo + 1, @as(usize, @intCast(in - 1)));
            const f: f32 = @floatCast(src - @floor(src));
            m[o * @as(usize, @intCast(in)) + lo] += 1 - f;
            m[o * @as(usize, @intCast(in)) + hi_i] += f;
        }
        return sc.fromF32(m, &.{ out, in });
    }

    // ── Swin-L ──

    /// `shiftMask` of a padded square plane, made once (`ACT`).
    fn cachedMask(self: *Net, hp: c_int) !A {
        const i = std.mem.indexOfScalar(c_int, &PLANES, hp) orelse return error.UnexpectedPlane;
        if (self.masks[i]) |m| return m;
        var sc = Scope.init(self.gpa, self.s);
        defer sc.deinit();
        const m = try sc.astype(try self.shiftMask(&sc, hp, hp), ACT);
        try mlx.check(mlx.mlx_array_eval(m));
        self.masks[i] = sc.out(m);
        return self.masks[i].?;
    }

    /// The additive attention mask of the shifted windows over a padded `hp` x `wp` plane, `[nW, 1, 144, 144]`.
    fn shiftMask(self: *const Net, sc: *Scope, hp: c_int, wp: c_int) !A {
        const gpa = self.gpa;
        const n: usize = @intCast(WINDOW * WINDOW);
        const nw: usize = @intCast(@divExact(hp, WINDOW) * @divExact(wp, WINDOW));
        const label = try gpa.alloc(f32, @intCast(hp * wp));
        defer gpa.free(label);
        const bounds = [_]c_int{ 0, hp - WINDOW, hp - SHIFT, hp };
        const wbounds = [_]c_int{ 0, wp - WINDOW, wp - SHIFT, wp };
        var cnt: f32 = 0;
        for (0..3) |hi| for (0..3) |wi| {
            for (@intCast(bounds[hi])..@intCast(bounds[hi + 1])) |y| for (@intCast(wbounds[wi])..@intCast(wbounds[wi + 1])) |x| {
                label[y * @as(usize, @intCast(wp)) + x] = cnt;
            };
            cnt += 1;
        };
        const mask = try gpa.alloc(f32, nw * n * n);
        defer gpa.free(mask);
        const wins_w: usize = @intCast(@divExact(wp, WINDOW));
        for (0..nw) |win| {
            const wy = win / wins_w;
            const wx = win % wins_w;
            for (0..n) |a| for (0..n) |b| {
                const la = label[(wy * 12 + a / 12) * @as(usize, @intCast(wp)) + wx * 12 + a % 12];
                const lb = label[(wy * 12 + b / 12) * @as(usize, @intCast(wp)) + wx * 12 + b % 12];
                mask[(win * n + a) * n + b] = if (la != lb) -100 else 0;
            };
        }
        return sc.fromF32(mask, &.{ @intCast(nw), 1, @intCast(n), @intCast(n) });
    }

    fn block(self: *const Net, sc: *Scope, x_in: A, stage: usize, blk: usize, h: c_int, w: c_int, mask: A) !A {
        const c: c_int = EMBED << @intCast(stage);
        const heads = HEADS[stage];
        const hd = @divExact(c, heads);
        const shift: c_int = if (blk % 2 == 1) SHIFT else 0;
        const hp = @divTrunc(h + WINDOW - 1, WINDOW) * WINDOW;
        const wp = @divTrunc(w + WINDOW - 1, WINDOW) * WINDOW;
        var x = try self.ln(sc, x_in, "bb.layers.{d}.blocks.{d}.norm1", .{ stage, blk });
        x = try sc.reshape(x, &.{ 1, h, w, c });
        if (hp > h) x = try sc.padAxis(x, 1, 0, hp - h);
        if (wp > w) x = try sc.padAxis(x, 2, 0, wp - w);
        if (shift > 0) x = try roll(sc, try roll(sc, x, -shift, 1), -shift, 2);
        const nw = @divExact(hp, WINDOW) * @divExact(wp, WINDOW);
        const win = try sc.reshape(try sc.transpose(try sc.reshape(x, &.{ 1, @divExact(hp, WINDOW), WINDOW, @divExact(wp, WINDOW), WINDOW, c }), &.{ 0, 1, 3, 2, 4, 5 }), &.{ nw, WINDOW * WINDOW, c });
        const qkv = try sc.reshape(try self.linear(sc, win, "bb.layers.{d}.blocks.{d}.attn.qkv", .{ stage, blk }), &.{ nw, WINDOW * WINDOW, 3, heads, hd });
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var parts: [3]A = undefined;
        for (&parts, 0..) |*p, i| p.* = try sc.transpose(try sc.reshape(try sc.slice(qkv, 2, @intCast(i), @intCast(i + 1)), &.{ nw, WINDOW * WINDOW, heads, hd }), &perm);
        const bias = self.bias[stage][blk];
        const m = if (shift > 0) try sc.add(bias, mask) else bias;
        const att = try sc.sdpaMasked(parts[0], parts[1], parts[2], 1.0 / @sqrt(@as(f32, @floatFromInt(hd))), m);
        const merged = try sc.reshape(try sc.transpose(att, &perm), &.{ nw, WINDOW * WINDOW, c });
        const proj = try self.linear(sc, merged, "bb.layers.{d}.blocks.{d}.attn.proj", .{ stage, blk });
        var y = try sc.reshape(try sc.transpose(try sc.reshape(proj, &.{ 1, @divExact(hp, WINDOW), @divExact(wp, WINDOW), WINDOW, WINDOW, c }), &.{ 0, 1, 3, 2, 4, 5 }), &.{ 1, hp, wp, c });
        if (shift > 0) y = try roll(sc, try roll(sc, y, shift, 1), shift, 2);
        if (hp > h or wp > w) y = try sc.slice(try sc.slice(y, 1, 0, h), 2, 0, w);
        const res = try sc.add(x_in, try sc.reshape(y, &.{ 1, h * w, c }));
        const n2 = try self.ln(sc, res, "bb.layers.{d}.blocks.{d}.norm2", .{ stage, blk });
        const f = try self.linear(sc, try sc.geluErf(try self.linear(sc, n2, "bb.layers.{d}.blocks.{d}.mlp.fc1", .{ stage, blk })), "bb.layers.{d}.blocks.{d}.mlp.fc2", .{ stage, blk });
        return sc.add(res, f);
    }

    /// The four feature maps `[1, H/4.., W/4.., 192..1536]` of the Swin-L over `x` `[1, H, W, 3]`.
    fn backbone(self: *Net, sc: *Scope, x: A) ![4]A {
        const sh = mlx.getShape(x);
        var h = @divExact(sh[1], 4);
        var w = @divExact(sh[2], 4);
        const pe = try self.conv(sc, x, "bb.patch_embed.proj", .{}, 4, 0);
        var t = try sc.reshape(try self.ln(sc, pe, "bb.patch_embed.norm", .{}), &.{ 1, h * w, EMBED });
        var outs: [4]A = undefined;
        for (0..4) |stage| {
            const c: c_int = EMBED << @intCast(stage);
            const hp = @divTrunc(h + WINDOW - 1, WINDOW) * WINDOW;
            const wp = @divTrunc(w + WINDOW - 1, WINDOW) * WINDOW;
            if (hp != wp) return error.UnexpectedPlane;
            const mask = try self.cachedMask(hp);
            for (0..DEPTHS[stage]) |blk| {
                var bs = Scope.init(sc.a, sc.s);
                defer bs.deinit();
                const y = try self.block(&bs, t, stage, blk, h, w, mask);
                try mlx.check(mlx.mlx_array_eval(y));
                t = try sc.keep(bs.out(y));
            }
            outs[stage] = try sc.reshape(try self.ln(sc, t, "bb.norm{d}", .{stage}), &.{ 1, h, w, c });
            if (stage < 3) {
                const grid = try sc.reshape(t, &.{ 1, h, w, c });
                const cat = try sc.concat(&.{ try every2(sc, grid, 0, 0), try every2(sc, grid, 1, 0), try every2(sc, grid, 0, 1), try every2(sc, grid, 1, 1) }, 3);
                const flat = try sc.reshape(cat, &.{ 1, @divExact(h, 2) * @divExact(w, 2), 4 * c });
                const nrm = try self.ln(sc, flat, "bb.layers.{d}.downsample.norm", .{stage});
                t = try sc.lin(nrm, try self.get("bb.layers.{d}.downsample.reduction.weight", .{stage}), null);
                h = @divExact(h, 2);
                w = @divExact(w, 2);
            }
        }
        return outs;
    }

    // ── deformable ASPP decoder ──

    /// Bilinear samples of `x` `[H*W, C]` at the float positions `py`, `px` (`[H*W]`), zero outside the plane.
    fn sample(sc: *Scope, x2: A, py: A, px: A, h: c_int, w: c_int) !A {
        var fl = mlx.mlx_array_new();
        const y0 = try sc.res(mlx.mlx_floor(&fl, py, sc.s), &fl);
        var fl2 = mlx.mlx_array_new();
        const x0 = try sc.res(mlx.mlx_floor(&fl2, px, sc.s), &fl2);
        const ly = try sc.sub(py, y0);
        const lx = try sc.sub(px, x0);
        const one = try sc.scalar(1);
        var acc: ?A = null;
        for ([_]c_int{ 0, 1 }) |dy| for ([_]c_int{ 0, 1 }) |dx| {
            const yy = if (dy == 0) y0 else try sc.add(y0, one);
            const xx = if (dx == 0) x0 else try sc.add(x0, one);
            const wy = if (dy == 0) try sc.sub(one, ly) else ly;
            const wx = if (dx == 0) try sc.sub(one, lx) else lx;
            const inside = try sc.mul(try sc.mul(try inRange(sc, yy, h), try inRange(sc, xx, w)), try sc.mul(wy, wx));
            const yc = try clampIdx(sc, yy, h);
            const xc = try clampIdx(sc, xx, w);
            const idx = try sc.astype(try sc.add(try sc.mulS(yc, @floatFromInt(w)), xc), .int32);
            const rows = try sc.mul(try sc.take(x2, idx), try sc.astype(try sc.reshape(inside, &.{ h * w, 1 }), mlx.mlx_array_dtype(x2)));
            acc = if (acc) |a| try sc.add(a, rows) else rows;
        };
        return acc.?;
    }

    /// 1 where `0 <= v < n`, else 0.
    fn inRange(sc: *Scope, v: A, n: c_int) !A {
        const ge = try sc.astype(try boolOp(sc, mlx.mlx_greater_equal, v, try sc.scalar(0)), .float32);
        const lt = try sc.astype(try sc.less(v, try sc.scalar(@floatFromInt(n))), .float32);
        return sc.mul(ge, lt);
    }

    fn boolOp(sc: *Scope, comptime f: anytype, a: A, b: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(f(&o, a, b, sc.s), &o);
    }

    fn clampIdx(sc: *Scope, v: A, n: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_clip(&o, v, try sc.scalar(0), try sc.scalar(@floatFromInt(n - 1)), sc.s), &o);
    }

    /// `DeformableConv2d` + the `torchvision.ops.deform_conv2d` it calls: every output samples its `k` x `k` taps at the
    /// regular grid plus a predicted offset (rows first), weighted by `2 sigmoid(modulator)`.
    fn deform(self: *const Net, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype, k: c_int) !A {
        const sh = mlx.getShape(x);
        const h = sh[1];
        const w = sh[2];
        const c = sh[3];
        const pad = @divTrunc(k, 2);
        const off = try sc.astype(try self.conv(sc, x, fmt ++ ".offset_conv", args, 1, pad), .float32); // [1, H, W, 2 K]
        const msk = try sc.mulLike(try sc.sigmoid(try self.conv(sc, x, fmt ++ ".modulator_conv", args, 1, pad)), 2);
        const x2 = try sc.reshape(x, &.{ h * w, c });
        const stacked = if (try glue.deformCols(x2, off, msk, k, h, w, sc.s)) |cols| try sc.keep(cols) else try tapColumns(sc, x2, off, msk, k, h, w); // [HW, K*C]
        // weight [O, k, k, C] -> [K*C, O] in tap-major order
        const wt = try self.get(fmt ++ ".regular_conv.weight", args);
        const o = mlx.getShape(wt)[0];
        const w2 = try sc.transpose(try sc.reshape(wt, &.{ o, k * k * c }), &.{ 1, 0 });
        return sc.reshape(try sc.matmul(stacked, w2), &.{ 1, h, w, o });
    }

    /// `deformCols` as an op chain, tap by tap.
    fn tapColumns(sc: *Scope, x2: A, off: A, msk: A, k: c_int, h: c_int, w: c_int) !A {
        const pad = @divTrunc(k, 2);
        var ys = mlx.mlx_array_new();
        var xs = mlx.mlx_array_new();
        const ay = try sc.res(mlx.mlx_arange(&ys, 0, @floatFromInt(h), 1, .float32, sc.s), &ys);
        const ax = try sc.res(mlx.mlx_arange(&xs, 0, @floatFromInt(w), 1, .float32, sc.s), &xs);
        const gy = try sc.reshape(try sc.broadcast(try sc.reshape(ay, &.{ h, 1 }), &.{ h, w }), &.{ h * w });
        const gx = try sc.reshape(try sc.broadcast(try sc.reshape(ax, &.{ 1, w }), &.{ h, w }), &.{ h * w });
        const taps: usize = @intCast(k * k);
        var cols_buf: [49]A = undefined;
        const cols = cols_buf[0..taps];
        for (0..taps) |p| {
            const i: c_int = @intCast(p / @as(usize, @intCast(k)));
            const j: c_int = @intCast(p % @as(usize, @intCast(k)));
            const dy = try sc.reshape(try sc.slice(off, 3, @intCast(2 * p), @intCast(2 * p + 1)), &.{h * w});
            const dx = try sc.reshape(try sc.slice(off, 3, @intCast(2 * p + 1), @intCast(2 * p + 2)), &.{h * w});
            const py = try sc.add(try sc.addS(gy, @floatFromInt(i - pad)), dy);
            const px = try sc.add(try sc.addS(gx, @floatFromInt(j - pad)), dx);
            const m = try sc.reshape(try sc.slice(msk, 3, @intCast(p), @intCast(p + 1)), &.{ h * w, 1 });
            cols[p] = try sc.mul(try sample(sc, x2, py, px, h, w), m);
        }
        return if (taps == 1) cols[0] else sc.concat(cols, 1);
    }

    fn relu(sc: *Scope, x: A) !A {
        return sc.relu(x);
    }

    fn asppModule(self: *const Net, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype, k: c_int) !A {
        const d = try self.deform(sc, x, fmt ++ ".atrous_conv", args, k);
        return relu(sc, try self.bn(sc, d, fmt ++ ".bn", args));
    }

    /// `ASPPDeformable`: three deformable branches, a k=1 one and a global-pool one, concatenated and fused.
    fn aspp(self: *const Net, sc: *Scope, x: A, comptime pfx: []const u8, args: anytype) !A {
        const sh = mlx.getShape(x);
        const x1 = try self.asppModule(sc, x, pfx ++ ".aspp1", args, 1);
        var branches: [5]A = undefined;
        branches[0] = x1;
        inline for (.{ 1, 3, 7 }, 0..) |k, i| {
            branches[1 + i] = try self.asppModule(sc, x, pfx ++ ".aspp_deforms.{d}", args ++ .{i}, k);
        }
        const pooled = try sc.reshape(try meanHw(sc, x), &.{ 1, 1, 1, sh[3] });
        const g = try self.conv(sc, pooled, pfx ++ ".global_avg_pool.1", args, 1, 0);
        const g2 = try relu(sc, try self.bn(sc, g, pfx ++ ".global_avg_pool.2", args));
        branches[4] = try sc.broadcast(g2, &.{ 1, sh[1], sh[2], 256 });
        const y = try self.conv(sc, try sc.concat(&branches, 3), pfx ++ ".conv1", args, 1, 0);
        return relu(sc, try self.bn(sc, y, pfx ++ ".bn1", args));
    }

    /// The mean over the plane, summed in f32.
    fn meanHw(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        const m = try sc.res(mlx.mlx_mean_axes(&o, try sc.astype(x, .float32), &[_]c_int{ 1, 2 }, 2, false, sc.s), &o);
        return sc.astype(m, mlx.mlx_array_dtype(x));
    }

    /// `BasicDecBlk`: conv, bn, relu, deformable ASPP, conv, bn.
    fn decBlock(self: *const Net, sc: *Scope, x: A, comptime pfx: []const u8, args: anytype) !A {
        var y = try relu(sc, try self.bn(sc, try self.conv(sc, x, pfx ++ ".conv_in", args, 1, 1), pfx ++ ".bn_in", args));
        y = try self.aspp(sc, y, pfx ++ ".dec_att", args);
        return self.bn(sc, try self.conv(sc, y, pfx ++ ".conv_out", args, 1, 1), pfx ++ ".bn_out", args);
    }

    fn gdt(self: *const Net, sc: *Scope, p: A, comptime n: []const u8) !A {
        const f = try relu(sc, try self.bn(sc, try self.conv(sc, p, "decoder.gdt_convs_" ++ n ++ ".0", .{}, 1, 1), "decoder.gdt_convs_" ++ n ++ ".1", .{}));
        const a = try self.conv(sc, f, "decoder.gdt_convs_attn_" ++ n ++ ".0", .{}, 1, 0);
        return sc.mul(p, try sc.sigmoid(a));
    }

    /// `_image2patches(x, ref)` for `x` `[1, H, W, 3]`: the image cut into `ref`'s grid, each cell's pixels as channels.
    fn patches(sc: *Scope, x: A, rh: c_int, rw: c_int) !A {
        const sh = mlx.getShape(x);
        const hg = @divExact(sh[1], rh);
        const wg = @divExact(sh[2], rw);
        const r = try sc.reshape(x, &.{ 1, hg, rh, wg, rw, sh[3] });
        return sc.reshape(try sc.transpose(r, &.{ 0, 2, 4, 5, 1, 3 }), &.{ 1, rh, rw, sh[3] * hg * wg });
    }

    fn ipt(self: *const Net, sc: *Scope, x: A, ref: A, comptime n: []const u8) !A {
        const rs = mlx.getShape(ref);
        const p = try patches(sc, x, rs[1], rs[2]);
        const c1 = try self.conv(sc, p, "decoder.ipt_blk" ++ n ++ ".conv1", .{}, 1, 1);
        return self.conv(sc, c1, "decoder.ipt_blk" ++ n ++ ".conv_out", .{}, 1, 1);
    }

    /// The alpha plane `[1, 1024, 1024, 1]` in [0, 1] for the normalized image `x` `[1, 1024, 1024, 3]`.
    fn forward(self: *Net, sc: *Scope, x: A) !A {
        const f = try self.backbone(sc, x);
        const half = try self.resize(sc, x, SIZE / 2, SIZE / 2);
        const g = try self.backbone(sc, half);
        var enc: [4]A = undefined;
        for (0..4) |i| {
            const sh = mlx.getShape(f[i]);
            enc[i] = try sc.concat(&.{ f[i], try self.resize(sc, g[i], sh[1], sh[2]) }, 3);
        }
        const s4 = mlx.getShape(enc[3]);
        const x4 = try sc.concat(&.{ try self.resize(sc, enc[0], s4[1], s4[2]), try self.resize(sc, enc[1], s4[1], s4[2]), try self.resize(sc, enc[2], s4[1], s4[2]), enc[3] }, 3);
        const sq = try self.decBlock(sc, x4, "squeeze_module.{d}", .{0});
        try mlx.check(mlx.mlx_array_eval(sq));

        var p4 = try self.decBlock(sc, try sc.concat(&.{ sq, try self.ipt(sc, x, sq, "5") }, 3), "decoder.decoder_block4", .{});
        p4 = try self.gdt(sc, p4, "4");
        const s3 = mlx.getShape(enc[2]);
        var p3in = try sc.add(try self.resize(sc, p4, s3[1], s3[2]), try self.conv(sc, enc[2], "decoder.lateral_block4.conv", .{}, 1, 0));
        p3in = try sc.concat(&.{ p3in, try self.ipt(sc, x, p3in, "4") }, 3);
        var p3 = try self.decBlock(sc, p3in, "decoder.decoder_block3", .{});
        p3 = try self.gdt(sc, p3, "3");
        const s2 = mlx.getShape(enc[1]);
        var p2in = try sc.add(try self.resize(sc, p3, s2[1], s2[2]), try self.conv(sc, enc[1], "decoder.lateral_block3.conv", .{}, 1, 0));
        p2in = try sc.concat(&.{ p2in, try self.ipt(sc, x, p2in, "3") }, 3);
        var p2 = try self.decBlock(sc, p2in, "decoder.decoder_block2", .{});
        p2 = try self.gdt(sc, p2, "2");
        const s1 = mlx.getShape(enc[0]);
        var p1in = try sc.add(try self.resize(sc, p2, s1[1], s1[2]), try self.conv(sc, enc[0], "decoder.lateral_block2.conv", .{}, 1, 0));
        p1in = try sc.concat(&.{ p1in, try self.ipt(sc, x, p1in, "2") }, 3);
        var p1 = try self.decBlock(sc, p1in, "decoder.decoder_block1", .{});
        p1 = try self.resize(sc, p1, SIZE, SIZE);
        p1 = try sc.concat(&.{ p1, try self.ipt(sc, x, p1, "1") }, 3);
        return sc.sigmoid(try self.conv(sc, p1, "decoder.conv_out1.0", .{}, 1, 0));
    }

    /// The alpha matte of straight RGB8 `rgb` (`w` x `h`), one byte per pixel (caller frees): the image bilinearly to
    /// 1024 x 1024, normalized, through the network, and the plane back to `w` x `h`.
    pub fn matte(self: *Net, rgb: []const u8, w: u32, h: u32) ![]u8 {
        var sc = Scope.init(self.gpa, self.s);
        defer sc.deinit();
        const px = try self.gpa.alloc(f32, @as(usize, w) * h * 3);
        defer self.gpa.free(px);
        for (px, rgb) |*o, v| o.* = @as(f32, @floatFromInt(v)) / 255.0;
        const img = try sc.fromF32(px, &.{ 1, @intCast(h), @intCast(w), 3 });
        const small = try self.resize(&sc, img, SIZE, SIZE);
        const mean = try sc.fromF32(&MEAN, &.{ 1, 1, 1, 3 });
        const std_ = try sc.fromF32(&STD, &.{ 1, 1, 1, 3 });
        const x = try sc.astype(try sc.div(try sc.sub(small, mean), std_), ACT);
        const alpha = try sc.astype(try self.forward(&sc, x), .float32);
        const back = try self.resize(&sc, alpha, @intCast(h), @intCast(w));
        var cl = mlx.mlx_array_new();
        const clipped = try sc.res(mlx.mlx_clip(&cl, back, try sc.scalar(0), try sc.scalar(1), sc.s), &cl);
        const flat = try sc.reshape(try sc.mulS(clipped, 255), &.{ @as(c_int, @intCast(h)) * @as(c_int, @intCast(w)) });
        try mlx.check(mlx.mlx_array_eval(flat));
        const data = mlx.mlx_array_data_float32(flat) orelse return error.NoData;
        const out = try self.gpa.alloc(u8, @as(usize, w) * h);
        for (out, data[0..out.len]) |*o, v| o.* = @intFromFloat(@min(v, 255)); // `.to(torch.uint8)` truncates
        return out;
    }
};
