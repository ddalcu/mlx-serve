//! CAM++ speaker encoder (3D-Speaker campplus, 192-d) on MLX — the network
//! `campplus.onnx` runs in the vendor pipeline, ported from mlx-audio's
//! `StepAudio2CAMPPlus` (BatchNorms before convolutions are folded into the
//! convolution biases, as in the ONNX export).
//!
//! Weights load from the checkpoint's `campplus.onnx` initializers, or from a
//! `campplus.safetensors` holding them verbatim (mlx-audio's conversion): the folded convolutions are anonymous
//! `onnx::Conv_<n>` pairs in graph order, everything else keeps its
//! `xvector.*` name. Torch layouts are transposed to MLX at load.
//! Input: mono 16 kHz audio → Kaldi fbank (`ming_audio.fbank`), mean-normalized.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const mm = @import("model.zig");
const audio = @import("audio.zig");

const Arr = mm.Arr;
const G = mm.G;

const Conv = struct { w: Arr, b: ?Arr };
const Bn = struct { w: ?Arr, b: ?Arr, mean: Arr, inv_std: Arr };
const CamLayer = struct {
    bn: Bn,
    lin1: Conv, // 1x1 (folded BN), [out, in]
    local: Arr, // k3, MLX [out, k, in]
    cam1: Conv,
    cam2: Conv,
    dilation: c_int,
};
const Transit = struct { bn: Bn, lin: Conv };

pub const Campplus = struct {
    t: mm.Tensors,
    a: std.mem.Allocator,
    s: mm.S,
    owned: std.ArrayList(Arr) = .empty,
    head: [12]Conv, // conv1, layer1.{0.c1,0.c2,0.sc,1.c1,1.c2}, layer2.{...}, conv2
    tdnn: Conv,
    layers: [52]CamLayer,
    transits: [3]Transit,
    dense_w: Arr,
    dense_bn: Bn,

    pub fn load(a: std.mem.Allocator, s: mm.S, path: []const u8) !*Campplus {
        return fromTensors(a, s, try mm.Tensors.load(a, path, .float32, s));
    }

    /// Load straight from the checkpoint's `campplus.onnx` (same tensors).
    pub fn loadOnnx(io: std.Io, a: std.mem.Allocator, s: mm.S, path: []const u8) !*Campplus {
        return fromTensors(a, s, try mm.Tensors.loadOnnx(a, io, path, .float32, s));
    }

    fn fromTensors(a: std.mem.Allocator, s: mm.S, tensors: mm.Tensors) !*Campplus {
        var t = tensors;
        const self = a.create(Campplus) catch |e| {
            t.deinit();
            return e;
        };
        errdefer a.destroy(self);
        self.* = .{ .t = t, .a = a, .s = s, .head = undefined, .tdnn = undefined, .layers = undefined, .transits = undefined, .dense_w = undefined, .dense_bn = undefined };
        errdefer self.deinit();

        // Anonymous folded convolutions, in graph order.
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(a);
        var kit = self.t.map.keyIterator();
        while (kit.next()) |k| {
            if (std.mem.startsWith(u8, k.*, "onnx::Conv_"))
                try ids.append(a, try std.fmt.parseInt(u32, k.*["onnx::Conv_".len..], 10));
        }
        std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
        if (ids.items.len != 132) return error.UnexpectedCampplusLayout;
        var pair: usize = 0;
        for (&self.head) |*c| {
            c.* = try self.conv2d(ids.items[pair * 2], ids.items[pair * 2 + 1]);
            pair += 1;
        }
        self.tdnn = try self.conv1d(ids.items[pair * 2], ids.items[pair * 2 + 1]);
        pair += 1;

        const specs = [_]struct { n: usize, dil: c_int }{ .{ .n = 12, .dil = 1 }, .{ .n = 24, .dil = 2 }, .{ .n = 16, .dil = 2 } };
        var li: usize = 0;
        for (specs, 0..) |sp, bi| {
            for (0..sp.n) |j| {
                const L = &self.layers[li];
                L.lin1 = try self.mat(ids.items[pair * 2], ids.items[pair * 2 + 1]);
                pair += 1;
                L.bn = try self.bn("xvector.block{d}.tdnnd{d}.nonlinear1.batchnorm", .{ bi + 1, j + 1 }, true);
                L.local = try self.convT(try self.t.get("xvector.block{d}.tdnnd{d}.cam_layer.linear_local.weight", .{ bi + 1, j + 1 }));
                L.cam1 = .{ .w = try self.flat(try self.t.get("xvector.block{d}.tdnnd{d}.cam_layer.linear1.weight", .{ bi + 1, j + 1 })), .b = try self.t.get("xvector.block{d}.tdnnd{d}.cam_layer.linear1.bias", .{ bi + 1, j + 1 }) };
                L.cam2 = .{ .w = try self.flat(try self.t.get("xvector.block{d}.tdnnd{d}.cam_layer.linear2.weight", .{ bi + 1, j + 1 })), .b = try self.t.get("xvector.block{d}.tdnnd{d}.cam_layer.linear2.bias", .{ bi + 1, j + 1 }) };
                L.dilation = sp.dil;
                li += 1;
            }
        }
        for (0..2) |i| {
            self.transits[i] = .{
                .bn = try self.bn("xvector.transit{d}.nonlinear.batchnorm", .{i + 1}, true),
                .lin = .{ .w = try self.flat(try self.t.get("xvector.transit{d}.linear.weight", .{i + 1})), .b = null },
            };
        }
        self.transits[2] = .{ .bn = try self.bn("xvector.transit{d}.nonlinear.batchnorm", .{3}, true), .lin = try self.mat(ids.items[pair * 2], ids.items[pair * 2 + 1]) };
        pair += 1;
        std.debug.assert(pair == 66);
        self.dense_w = try self.flat(try self.t.get("xvector.dense.linear.weight", .{}));
        self.dense_bn = try self.bn("xvector.dense.nonlinear.batchnorm", .{}, false);
        try mm.evalAll(self.owned.items);
        return self;
    }

    pub fn deinit(self: *Campplus) void {
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.a);
        self.t.deinit();
        self.a.destroy(self);
    }

    fn keep(self: *Campplus, x: Arr) !Arr {
        try self.owned.append(self.a, x);
        return x;
    }

    fn onnx(self: *Campplus, id: u32) !Arr {
        return self.t.get("onnx::Conv_{d}", .{id});
    }

    /// torch [O, I, kh, kw] → MLX [O, kh, kw, I]
    fn conv2d(self: *Campplus, w_id: u32, b_id: u32) !Conv {
        var o = mlx.mlx_array_new();
        const perm = [_]c_int{ 0, 2, 3, 1 };
        try mlx.check(mlx.mlx_transpose_axes(&o, try self.onnx(w_id), &perm, 4, self.s));
        return .{ .w = try self.keep(o), .b = try self.onnx(b_id) };
    }

    /// torch [O, I, k] → MLX [O, k, I]
    fn convT(self: *Campplus, w: Arr) !Arr {
        var o = mlx.mlx_array_new();
        const perm = [_]c_int{ 0, 2, 1 };
        try mlx.check(mlx.mlx_transpose_axes(&o, w, &perm, 3, self.s));
        return self.keep(o);
    }

    fn conv1d(self: *Campplus, w_id: u32, b_id: u32) !Conv {
        return .{ .w = try self.convT(try self.onnx(w_id)), .b = try self.onnx(b_id) };
    }

    /// 1x1 convolution as a matmul: [O, I, 1] → transposed [I, O].
    fn flat(self: *Campplus, w: Arr) !Arr {
        const sh = mlx.getShape(w);
        var r = mlx.mlx_array_new();
        const s2 = [_]c_int{ sh[0], sh[1] };
        try mlx.check(mlx.mlx_reshape(&r, w, &s2, 2, self.s));
        _ = try self.keep(r);
        var o = mlx.mlx_array_new();
        const perm = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&o, r, &perm, 2, self.s));
        return self.keep(o);
    }

    fn mat(self: *Campplus, w_id: u32, b_id: u32) !Conv {
        return .{ .w = try self.flat(try self.onnx(w_id)), .b = try self.onnx(b_id) };
    }

    fn bn(self: *Campplus, comptime prefix: []const u8, args: anytype, affine: bool) !Bn {
        const mean = try self.t.get(prefix ++ ".running_mean", args);
        const v = try self.t.get(prefix ++ ".running_var", args);
        var g = G.init(self.a, self.s);
        defer g.deinit();
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_rsqrt(&r, try g.add(v, try g.scalar(1e-5, .float32)), self.s));
        return .{
            .w = if (affine) try self.t.get(prefix ++ ".weight", args) else null,
            .b = if (affine) try self.t.get(prefix ++ ".bias", args) else null,
            .mean = mean,
            .inv_std = try self.keep(r),
        };
    }

    fn applyBn(g: *G, x: Arr, p: Bn) !Arr {
        var y = try g.mul(try g.sub(x, p.mean), p.inv_std);
        if (p.w) |w| y = try g.add(try g.mul(w, y), p.b.?);
        return y;
    }

    fn relu(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        const z = try g.scalar(0, mlx.mlx_array_dtype(x));
        return g.keep(blk: {
            try mlx.check(mlx.mlx_maximum(&o, x, z, g.s));
            break :blk o;
        });
    }

    fn c2d(g: *G, x: Arr, c: Conv, sh: c_int, pad: c_int) !Arr {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv2d(&o, x, c.w, sh, 1, pad, pad, 1, 1, 1, g.s));
        return g.add(try g.keep(o), c.b.?);
    }

    fn c1d(g: *G, x: Arr, w: Arr, stride: c_int, pad: c_int, dil: c_int) !Arr {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv1d(&o, x, w, stride, pad, dil, 1, g.s));
        return g.keep(o);
    }

    fn lin(g: *G, x: Arr, c: Conv) !Arr {
        if (c.b) |b| return g.addmm(b, x, c.w);
        return g.matmul(x, c.w);
    }

    /// Ceil-mode 100-frame average pooling broadcast back to T (channel-last).
    fn segPool(g: *G, x: Arr) !Arr {
        const sh = mlx.getShape(x);
        const T: usize = @intCast(sh[1]);
        const C = sh[2];
        const segs = (T + 99) / 100;
        var xp = x;
        if (segs * 100 > T) xp = try g.concat(&.{ x, try g.zeros(&.{ 1, @intCast(segs * 100 - T), C }, .float32) }, 1);
        var sums = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&sums, try g.reshape(xp, &.{ 1, @intCast(segs), 100, C }), 2, false, g.s));
        var counts: [64]f32 = undefined;
        if (segs > counts.len) return error.SpeakerClipTooLong;
        for (0..segs) |i| counts[i] = @floatFromInt(@min(100, T - i * 100));
        const cnt = try g.fromF32(counts[0..segs], &.{ 1, @intCast(segs), 1 });
        const pooled = try g.div(try g.keep(sums), cnt);
        try mlx.check(mlx.mlx_array_eval(pooled)); // `counts` is stack memory
        var rep = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&rep, pooled, 100, 1, g.s));
        return g.sliceAxis(try g.keep(rep), 1, 0, @intCast(T), 1);
    }

    /// 16 kHz mono → 192-d embedding (caller frees).
    pub fn embed(self: *Campplus, a: std.mem.Allocator, wav16k: []const f32) ![]f32 {
        if (wav16k.len < 720) return error.SpeakerClipTooShort;
        const fb = try audio.fbank(a, wav16k);
        defer a.free(fb);
        const T = fb.len / audio.NUM_MELS;
        var mean: [audio.NUM_MELS]f64 = @splat(0);
        for (0..T) |t| for (0..audio.NUM_MELS) |b| {
            mean[b] += fb[t * audio.NUM_MELS + b];
        };
        for (0..T) |t| for (0..audio.NUM_MELS) |b| {
            fb[t * audio.NUM_MELS + b] -= @floatCast(mean[b] / @as(f64, @floatFromInt(T)));
        };
        var g = G.init(a, self.s);
        defer g.deinit();
        const feats = try g.fromF32(fb, &.{ 1, @intCast(T), audio.NUM_MELS });
        const out = try self.forward(&g, feats);
        return mm.toF32(a, out, self.s);
    }

    /// feats [1, T, 80] → [1, 192]
    pub fn forward(self: *Campplus, g: *G, feats: Arr) !Arr {
        // Head (FCM): NHWC with H = mel bins, W = time.
        var x = try g.reshape(try g.transpose(feats, &.{ 0, 2, 1 }), &.{ 1, audio.NUM_MELS, -1, 1 });
        x = try relu(g, try c2d(g, x, self.head[0], 1, 1));
        var hi: usize = 1;
        for (0..2) |_| { // layer1, layer2: [stride-2 block with shortcut, plain block]
            const c1 = self.head[hi];
            const c2 = self.head[hi + 1];
            const sc = self.head[hi + 2];
            var o = try relu(g, try c2d(g, x, c1, 2, 1));
            o = try c2d(g, o, c2, 1, 1);
            x = try relu(g, try g.add(o, try c2d(g, x, sc, 2, 0)));
            o = try relu(g, try c2d(g, x, self.head[hi + 3], 1, 1));
            o = try c2d(g, o, self.head[hi + 4], 1, 1);
            x = try relu(g, try g.add(o, x));
            hi += 5;
        }
        x = try relu(g, try c2d(g, x, self.head[11], 2, 1)); // [1, 10, T, 32]
        x = try g.reshape(try g.transpose(x, &.{ 0, 2, 3, 1 }), &.{ 1, -1, 320 });
        // TDNN k5 stride 2.
        x = try relu(g, try g.add(try c1d(g, x, self.tdnn.w, 2, 2, 1), self.tdnn.b.?));
        var li: usize = 0;
        for ([_]usize{ 12, 24, 16 }, 0..) |n, bi| {
            for (0..n) |_| {
                const L = &self.layers[li];
                var h = try relu(g, try applyBn(g, x, L.bn));
                h = try relu(g, try lin(g, h, L.lin1));
                const y = try c1d(g, h, L.local, 1, L.dilation, L.dilation);
                var mean_t = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_mean_axis(&mean_t, h, 1, true, g.s));
                var ctx = try g.add(try g.keep(mean_t), try segPool(g, h));
                ctx = try relu(g, try lin(g, ctx, L.cam1));
                const gate = try g.sigmoid(try lin(g, ctx, L.cam2));
                x = try g.concat(&.{ x, try g.mul(y, gate) }, 2);
                li += 1;
            }
            const tr = self.transits[bi];
            x = try lin(g, try relu(g, try applyBn(g, x, tr.bn)), tr.lin);
        }
        x = try relu(g, x);
        // Unbiased statistics pooling over time.
        const T = mlx.getShape(x)[1];
        if (T < 2) return error.SpeakerClipTooShort;
        var mu = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_mean_axis(&mu, x, 1, false, g.s));
        var vr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_var_axis(&vr, x, 1, false, 0, g.s));
        const tf: f32 = @floatFromInt(T);
        const var_u = try g.div(try g.mul(try g.keep(vr), try g.scalar(tf, .float32)), try g.scalar(tf - 1, .float32));
        var sd = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sqrt(&sd, try relu(g, var_u), g.s));
        const stats = try g.concat(&.{ try g.keep(mu), try g.keep(sd) }, -1); // [1, 1024]
        return applyBn(g, try g.matmul(stats, self.dense_w), self.dense_bn);
    }
};
