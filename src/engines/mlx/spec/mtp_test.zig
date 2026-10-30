const std = @import("std");
const mtp = @import("../../../sdk/spec/mtp/mtp.zig");
const QuantMode = @import("../../../sdk/quant/params.zig").QuantMode;
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const params_mod = @import("../../../sdk/quant/params.zig");
const Transformer = @import("../transformer.zig").Transformer;
const KVCache = @import("../kv/cache.zig").KVCache;
const testing = std.testing;

test "mtp: headQuantParams never mis-resolves a non-affine lm_head as affine (issue #81)" {
    // An mxfp8 8-bit/gs32 lm_head's GEOMETRY coincidentally solves as a valid
    // affine 8-bit/gs32 (w [V, H/4] u32, scales [V, H/32]) — but the scales
    // are fp8-encoded uint8 and no biases tensor exists, so a mode="affine"
    // matmul/dequantize throws "Biases must be provided for affine
    // quantization" and kills the first MTP forward. The scales-dtype gate
    // (uint8 → the config's non-affine mode) must win over the geometry
    // shortcut, exactly as computeQuantParams resolves the trunk.
    const s = mlx.gpuStream();
    const H = 512;
    const V = 8;

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    // mxfp8 head: w [V, H*8/32] u32, scales [V, H/32] u8.
    const w8 = try mk.arr(&.{ V, H * 8 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w8);
    const s8 = try mk.arr(&.{ V, H / 32 }, .uint8, s);
    defer _ = mlx.mlx_array_free(s8);

    var cfg = model_mod.ModelConfig{};
    cfg.hidden_size = H;
    cfg.quant_bits = 8;
    cfg.quant_group_size = 32;
    cfg.quant_mode = .mxfp8;

    const qp8 = mtp.headQuantParams(&cfg, w8, s8);
    try testing.expectEqual(QuantMode.mxfp8, qp8.mode);
    try testing.expectEqual(@as(u32, 8), qp8.bits);
    try testing.expectEqual(@as(u32, 32), qp8.group_size);

    // mxfp4 (same class, 4-bit/gs32 — also a false-positive affine geometry).
    const w4 = try mk.arr(&.{ V, H * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w4);
    cfg.quant_bits = 4;
    cfg.quant_mode = .mxfp4;
    const qp4 = mtp.headQuantParams(&cfg, w4, s8);
    try testing.expectEqual(QuantMode.mxfp4, qp4.mode);

    // Characterization (green before AND after): the mixed-AFFINE shape this
    // function exists for — hy_v3's 8-bit/gs32 head (bf16 scales) over a
    // 2-bit/gs64 trunk still resolves per-geometry, never per-config.
    const sb = try mk.arr(&.{ V, H / 32 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sb);
    cfg.quant_bits = 2;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    const qpa = mtp.headQuantParams(&cfg, w8, sb);
    try testing.expectEqual(QuantMode.affine, qpa.mode);
    try testing.expectEqual(@as(u32, 8), qpa.bits);
    try testing.expectEqual(@as(u32, 32), qpa.group_size);
}


/// Random dense bf16 sidecar (hidden 8, head_dim 4, 2 q / 2 kv heads, mlp 16, vocab 16)
/// plus the target fields `forward` reads.
const TinyHead = struct {
    tmp: std.testing.TmpDir,
    m: mtp.MtpModel,
    xfm: Transformer,

    fn init(self: *TinyHead) !void {
        const io = testing.io;
        const allocator = testing.allocator;
        const s = mlx.gpuStream();
        self.tmp = std.testing.tmpDir(.{ .iterate = true });
        errdefer self.tmp.cleanup();

        // ── synthetic DENSE sidecar (random bf16; zeros would make every rms-norm
        // output zero and the equivalence trivially true) ──
        var prng = std.Random.DefaultPrng.init(7);
        const save_map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(save_map);
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (owned.items) |a| _ = mlx.mlx_array_free(a);
            owned.deinit(allocator);
        }
        const putRand = struct {
            fn f(map: mlx.mlx_map_string_to_array, list: *std.ArrayList(mlx.mlx_array), alloc: std.mem.Allocator, rng: *std.Random.DefaultPrng, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
                var n: usize = 1;
                for (shape) |d| n *= @intCast(d);
                const buf = try alloc.alloc(f32, n);
                defer alloc.free(buf);
                for (buf) |*x| x.* = rng.random().floatNorm(f32) * 0.5;
                const f32_arr = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(&a, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(a));
                _ = mlx.mlx_map_string_to_array_insert(map, key, a);
                try list.append(alloc, a);
                return a;
            }
        }.f;

        // hidden 8, head_dim 4, 2 q heads (x2 for the q/gate split), 2 kv heads,
        // mlp inter 16, vocab 16.
        // Disk orientation is torch [out, in]: fc maps concat(2H) -> H.
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.fc.weight", &.{ 8, 16 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.pre_fc_norm_embedding.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.pre_fc_norm_hidden.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.norm.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.input_layernorm.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.post_attention_layernorm.weight", &.{8}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.q_norm.weight", &.{4}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.k_norm.weight", &.{4}, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.q_proj.weight", &.{ 16, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.k_proj.weight", &.{ 8, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.v_proj.weight", &.{ 8, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.o_proj.weight", &.{ 8, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.gate_proj.weight", &.{ 16, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.up_proj.weight", &.{ 16, 8 }, s);
        _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.down_proj.weight", &.{ 8, 16 }, s);

        var dir_buf: [512]u8 = undefined;
        const dir_n = try self.tmp.dir.realPath(io, &dir_buf);
        const dir_abs = dir_buf[0..dir_n];
        const file_path = try std.fs.path.joinZ(allocator, &.{ dir_abs, "model-mtp.safetensors" });
        defer allocator.free(file_path);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_save_safetensors(file_path.ptr, save_map, meta));

        self.m = try mtp.loadMtp(io, allocator, s, dir_abs);
        errdefer self.m.deinit();

        // ── toy target: only the fields forward() reads (config scalars, dense
        // bf16 embed table, dense bf16 lm_head) ──
        var emb_prng = std.Random.DefaultPrng.init(11);
        const mk2d = struct {
            fn f(alloc: std.mem.Allocator, rng: *std.Random.DefaultPrng, rows: usize, cols: usize, st: mlx.mlx_stream) !mlx.mlx_array {
                const buf = try alloc.alloc(f32, rows * cols);
                defer alloc.free(buf);
                for (buf) |*x| x.* = rng.random().floatNorm(f32) * 0.5;
                const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
                const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(&a, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(a));
                return a;
            }
        }.f;
        const emb_w = try mk2d(allocator, &emb_prng, 16, 8, s);
        errdefer _ = mlx.mlx_array_free(emb_w);
        const lm_w = try mk2d(allocator, &emb_prng, 16, 8, s);

        const xfm = &self.xfm;
        xfm.rht = null;
        xfm.allocator = allocator;
        xfm.s = s;
        xfm.config = .{};
        xfm.config.hidden_size = 8;
        xfm.config.num_attention_heads = 2;
        xfm.config.num_key_value_heads = 2;
        xfm.config.head_dim = 4;
        xfm.config.query_pre_attn_scalar = 4;
        xfm.config.partial_rotary_factor = 0.5;
        xfm.config.attn_output_gate = true;
        xfm.emb_w = emb_w;
        xfm.emb_s = .{ .ctx = null };
        xfm.emb_b = .{ .ctx = null };
        xfm.lm_head_w = lm_w;
        xfm.lm_head_s = .{ .ctx = null };
        xfm.lm_head_b = .{ .ctx = null };
    }

    fn deinit(self: *TinyHead) void {
        self.m.deinit();
        _ = mlx.mlx_array_free(self.xfm.emb_w);
        _ = mlx.mlx_array_free(self.xfm.lm_head_w);
        self.tmp.cleanup();
    }
};

test "mtp: multi-row forward projects the LAST row only and equals appendHistory + stepArr" {
    // The deferred-history round shape (Generator.nextMtp) folds the old
    // appendHistory head forward into the next round's first draft step: ONE
    // (n+1)-row forward over [committed..., t1] must append the same cache
    // entries AND produce the same last-row logits/hidden as the two-call
    // sequence appendHistory([committed], hist_hidden) + stepArr(t1, h_prev).
    // Logits must be [1, 1, V]: projecting every row through the vocab head
    // is pure waste, and the caller (draft chain) only consumes the last row.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    // Pin the KV-only history path ON regardless of the env, so the merged
    // and appendHistory arms below exercise it; the last arm flips it OFF
    // for the full-path cross-check.
    mtp.mtp_kv_only_override = true;
    defer mtp.mtp_kv_only_override = null;

    var fx: TinyHead = undefined;
    try fx.init();
    defer fx.deinit();
    const m = &fx.m;
    const xfm = &fx.xfm;

    // ── shared inputs: 3 hidden rows, tokens [5, 7] committed + t1 = 9 ──
    var hid_prng = std.Random.DefaultPrng.init(23);
    const hid_buf = try allocator.alloc(f32, 3 * 8);
    defer allocator.free(hid_buf);
    for (hid_buf) |*x| x.* = hid_prng.random().floatNorm(f32) * 0.5;
    const hid_shape = [_]c_int{ 1, 3, 8 };
    const hid_f32 = mlx.mlx_array_new_data(hid_buf.ptr, &hid_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hid_f32);
    var hidden3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden3);
    try mlx.check(mlx.mlx_astype(&hidden3, hid_f32, .bfloat16, s));

    const strides = [_]c_int{ 1, 1, 1 };
    var hid01 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hid01);
    try mlx.check(mlx.mlx_slice(&hid01, hidden3, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, 2, 8 }, 3, &strides, 3, s));
    var hid2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hid2);
    try mlx.check(mlx.mlx_slice(&hid2, hidden3, &[_]c_int{ 0, 2, 0 }, 3, &[_]c_int{ 1, 3, 8 }, 3, &strides, 3, s));

    // ── reference: appendHistory([5,7]) then stepArr(9) ──
    var cache_a = try m.makeCache(allocator);
    defer cache_a.deinit();
    try mtp.appendHistory(m, xfm, &cache_a, &[_]u32{ 5, 7 }, hid01, 0);
    const t9 = [_]i32{9};
    const t9_shape = [_]c_int{1};
    const t9_arr = mlx.mlx_array_new_data(&t9, &t9_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(t9_arr);
    const ref = try mtp.stepArr(m, xfm, &cache_a, t9_arr, hid2, 2);
    defer {
        _ = mlx.mlx_array_free(ref.logits);
        _ = mlx.mlx_array_free(ref.hidden_next);
    }

    // ── merged: one 3-row forward over [5, 7, 9] ──
    var cache_b = try m.makeCache(allocator);
    defer cache_b.deinit();
    const ids3 = [_]i32{ 5, 7, 9 };
    const ids3_shape = [_]c_int{3};
    const ids3_arr = mlx.mlx_array_new_data(&ids3, &ids3_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(ids3_arr);
    const merged = try mtp.forward(m, xfm, &cache_b, ids3_arr, hidden3, 0, true);
    defer {
        _ = mlx.mlx_array_free(merged.logits);
        _ = mlx.mlx_array_free(merged.hidden_next);
    }

    // Same cache length; logits/hidden are LAST-row-only.
    try testing.expectEqual(cache_a.step, cache_b.step);
    const ml_shape = mlx.getShape(merged.logits);
    try testing.expectEqual(@as(c_int, 1), ml_shape[1]);
    try testing.expectEqual(@as(c_int, 16), ml_shape[2]);
    const mh_shape = mlx.getShape(merged.hidden_next);
    try testing.expectEqual(@as(c_int, 1), mh_shape[1]);

    // Value equivalence vs the two-call reference (bf16 reduction-order
    // tolerance at toy scale).
    const close = struct {
        fn f(a: mlx.mlx_array, b: mlx.mlx_array, n: usize, st: mlx.mlx_stream) !void {
            var af = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(af);
            var bf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bf);
            try mlx.check(mlx.mlx_astype(&af, a, .float32, st));
            try mlx.check(mlx.mlx_astype(&bf, b, .float32, st));
            try mlx.check(mlx.mlx_array_eval(af));
            try mlx.check(mlx.mlx_array_eval(bf));
            const ad = mlx.mlx_array_data_float32(af).?;
            const bd = mlx.mlx_array_data_float32(bf).?;
            for (0..n) |i| {
                const denom = @max(1.0, @max(@abs(ad[i]), @abs(bd[i])));
                if (@abs(ad[i] - bd[i]) / denom > 0.05) {
                    std.debug.print("mismatch at {d}: {d} vs {d}\n", .{ i, ad[i], bd[i] });
                    return error.TestExpectedApproxEq;
                }
            }
        }
    }.f;
    try close(merged.logits, ref.logits, 16, s);
    try close(merged.hidden_next, ref.hidden_next, 8, s);

    // A sequential three-axis table must be equivalent to ordinary RoPE.
    // Keep only two explicit positions so the final row also exercises the
    // prompt-table → generated-text (`absolute + delta`) boundary.
    var cache_c = try m.makeCache(allocator);
    defer cache_c.deinit();
    const sequential_pos = [_]i32{
        0, 1,
        0, 1,
        0, 1,
    };
    const positioned = try mtp.forwardWithMrope(
        m,
        xfm,
        &cache_c,
        ids3_arr,
        hidden3,
        0,
        true,
        .{
            .pos = &sequential_pos,
            .total = 2,
            .delta = 0,
        },
    );
    defer {
        _ = mlx.mlx_array_free(positioned.logits);
        _ = mlx.mlx_array_free(positioned.hidden_next);
    }
    try testing.expectEqual(cache_b.step, cache_c.step);
    try close(positioned.logits, merged.logits, 16, s);
    try close(positioned.hidden_next, merged.hidden_next, 8, s);

    // ── KV-only vs FULL history path: the committed rows' layer outputs are
    // dead, so the KV-only append must leave the same cache length and a
    // last-row result within reduction-order tolerance of the full forward
    // (a different GEMM M reorders reductions — byte parity is not the bar).
    mtp.mtp_kv_only_override = false;
    var cache_d = try m.makeCache(allocator);
    defer cache_d.deinit();
    const full = try mtp.forward(m, xfm, &cache_d, ids3_arr, hidden3, 0, true);
    defer {
        _ = mlx.mlx_array_free(full.logits);
        _ = mlx.mlx_array_free(full.hidden_next);
    }
    try testing.expectEqual(cache_b.step, cache_d.step);
    try close(full.logits, merged.logits, 16, s);
    try close(full.hidden_next, merged.hidden_next, 8, s);

    // Same cross-check through the mrope table (appendKvOnly's explicit-table
    // branch vs the full path's).
    var cache_e = try m.makeCache(allocator);
    defer cache_e.deinit();
    const full_pos = try mtp.forwardWithMrope(m, xfm, &cache_e, ids3_arr, hidden3, 0, true, .{
        .pos = &sequential_pos,
        .total = 2,
        .delta = 0,
    });
    defer {
        _ = mlx.mlx_array_free(full_pos.logits);
        _ = mlx.mlx_array_free(full_pos.hidden_next);
    }
    try testing.expectEqual(cache_c.step, cache_e.step);
    try close(full_pos.logits, positioned.logits, 16, s);
    try close(full_pos.hidden_next, positioned.hidden_next, 8, s);
}

test "mtp: forwardLanes equals N solo steps" {
    // Lanes at different cache lengths: a shared rope offset or a crossed cache cannot pass.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    mtp.mtp_kv_only_override = true;
    defer mtp.mtp_kv_only_override = null;
    var fx: TinyHead = undefined;
    try fx.init();
    defer fx.deinit();
    const m = &fx.m;
    const xfm = &fx.xfm;

    const T = struct {
        fn hidden(a: std.mem.Allocator, seed: u64, rows: usize, st: mlx.mlx_stream) !mlx.mlx_array {
            var prng = std.Random.DefaultPrng.init(seed);
            const buf = try a.alloc(f32, rows * 8);
            defer a.free(buf);
            for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
            const shape = [_]c_int{ 1, @intCast(rows), 8 };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, st));
            return out;
        }
        /// `[1, 1, V]` logits -> `[1]` int32 argmax.
        fn pick(logits: mlx.mlx_array, st: mlx.mlx_stream) !mlx.mlx_array {
            var am = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(am);
            try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, st));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, am, &[_]c_int{1}, 1, st));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, flat, .int32, st));
            return out;
        }
        fn row(x: mlx.mlx_array, k: usize, st: mlx.mlx_stream) !mlx.mlx_array {
            const sh = mlx.getShape(x);
            const k_c: c_int = @intCast(k);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ 0, k_c, 0 }, 3, &[_]c_int{ 1, k_c + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            return out;
        }
        fn cos(a: mlx.mlx_array, b: mlx.mlx_array, st: mlx.mlx_stream) !f32 {
            var af = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(af);
            var bf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bf);
            try mlx.check(mlx.mlx_astype(&af, a, .float32, st));
            try mlx.check(mlx.mlx_astype(&bf, b, .float32, st));
            try mlx.check(mlx.mlx_array_eval(af));
            try mlx.check(mlx.mlx_array_eval(bf));
            const ad = mlx.mlx_array_data_float32(af).?;
            const bd = mlx.mlx_array_data_float32(bf).?;
            var dot: f64 = 0;
            var na: f64 = 0;
            var nb: f64 = 0;
            for (0..8) |i| {
                dot += @as(f64, ad[i]) * bd[i];
                na += @as(f64, ad[i]) * ad[i];
                nb += @as(f64, bd[i]) * bd[i];
            }
            return @floatCast(dot / @sqrt(na * nb));
        }
    };

    const hist = [_]usize{ 0, 2, 5 };
    const N = hist.len;
    const STEPS = 3;
    var solo: [N]KVCache = undefined;
    var grouped: [N]KVCache = undefined;
    var made: usize = 0;
    defer for (solo[0..made], grouped[0..made]) |*a, *b| {
        a.deinit();
        b.deinit();
    };
    var ids: [2][N]mlx.mlx_array = undefined;
    var hid: [2][N]mlx.mlx_array = undefined;
    for (0..N) |k| {
        solo[k] = try m.makeCache(allocator);
        grouped[k] = try m.makeCache(allocator);
        made = k + 1;
        if (hist[k] > 0) {
            const h = try T.hidden(allocator, 100 + k, hist[k], s);
            defer _ = mlx.mlx_array_free(h);
            const toks = [_]u32{ 3, 4, 5, 6, 7 };
            try mtp.appendHistory(m, xfm, &solo[k], toks[0..hist[k]], h, 0);
            try mtp.appendHistory(m, xfm, &grouped[k], toks[0..hist[k]], h, 0);
        }
        for (0..2) |arm| {
            const t: i32 = @intCast(k + 1);
            ids[arm][k] = mlx.mlx_array_new_data(&t, &[_]c_int{1}, 1, .int32);
            hid[arm][k] = try T.hidden(allocator, 200 + k, 1, s);
        }
    }
    defer for (0..2) |arm| for (0..N) |k| {
        _ = mlx.mlx_array_free(ids[arm][k]);
        _ = mlx.mlx_array_free(hid[arm][k]);
    };

    for (0..STEPS) |step| {
        for (0..N) |k| {
            const out = try mtp.stepArr(m, xfm, &solo[k], ids[0][k], hid[0][k], @intCast(hist[k] + step));
            defer _ = mlx.mlx_array_free(out.logits);
            _ = mlx.mlx_array_free(ids[0][k]);
            ids[0][k] = try T.pick(out.logits, s);
            _ = mlx.mlx_array_free(hid[0][k]);
            hid[0][k] = out.hidden_next;
        }
        var lanes: [N]mtp.Lane = undefined;
        for (0..N) |k| lanes[k] = .{ .cache = &grouped[k], .id = ids[1][k], .hidden = hid[1][k], .rope_offset = @intCast(hist[k] + step) };
        const post = try mtp.forwardLanes(m, xfm, &lanes);
        defer _ = mlx.mlx_array_free(post);
        const logits = try mtp.targetLmHead(m, xfm, post, s);
        defer _ = mlx.mlx_array_free(logits);
        for (0..N) |k| {
            const lrow = try T.row(logits, k, s);
            defer _ = mlx.mlx_array_free(lrow);
            _ = mlx.mlx_array_free(ids[1][k]);
            ids[1][k] = try T.pick(lrow, s);
            _ = mlx.mlx_array_free(hid[1][k]);
            hid[1][k] = try T.row(post, k, s);

            try testing.expectEqual(solo[k].step, grouped[k].step);
            try testing.expectEqual(try readIdScalar(ids[0][k]), try readIdScalar(ids[1][k]));
            try testing.expect(try T.cos(hid[0][k], hid[1][k], s) >= 0.9999);
        }
    }
}
test "mtp: dense bf16 head trunk is requantized at load (4b/g64); indivisible widths stay dense" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];

    // Dense bf16 sidecar with a 64-divisible hidden (64): q/k/v/o + gate/up
    // quantize; down's contraction dim is 96 (not 64-divisible) — per-weight
    // skip, stays dense pre-transposed. fc stays bf16 by contract (the m5Nax
    // cost profile demands it).
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/mtp.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const H = struct {
            fn put(m2: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                const data = try std.testing.allocator.alloc(f32, total);
                defer std.testing.allocator.free(data);
                for (data, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
                const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                _ = mlx.mlx_map_string_to_array_insert(m2, key, bf);
            }
        };
        try H.put(map, "mtp.fc.weight", &.{ 64, 128 }, s);
        try H.put(map, "mtp.pre_fc_norm_embedding.weight", &.{64}, s);
        try H.put(map, "mtp.pre_fc_norm_hidden.weight", &.{64}, s);
        try H.put(map, "mtp.norm.weight", &.{64}, s);
        try H.put(map, "mtp.layers.0.input_layernorm.weight", &.{64}, s);
        try H.put(map, "mtp.layers.0.post_attention_layernorm.weight", &.{64}, s);
        try H.put(map, "mtp.layers.0.self_attn.q_norm.weight", &.{32}, s);
        try H.put(map, "mtp.layers.0.self_attn.k_norm.weight", &.{32}, s);
        try H.put(map, "mtp.layers.0.self_attn.q_proj.weight", &.{ 128, 64 }, s);
        try H.put(map, "mtp.layers.0.self_attn.k_proj.weight", &.{ 32, 64 }, s);
        try H.put(map, "mtp.layers.0.self_attn.v_proj.weight", &.{ 32, 64 }, s);
        try H.put(map, "mtp.layers.0.self_attn.o_proj.weight", &.{ 64, 64 }, s);
        try H.put(map, "mtp.layers.0.mlp.gate_proj.weight", &.{ 96, 64 }, s);
        try H.put(map, "mtp.layers.0.mlp.up_proj.weight", &.{ 96, 64 }, s);
        try H.put(map, "mtp.layers.0.mlp.down_proj.weight", &.{ 64, 96 }, s);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    {
        var m = try mtp.loadMtp(io, allocator, s, dir_path);
        defer m.deinit();

        // q packed to 4-bit g64: uint32 codes, geometry re-solves.
        try testing.expect(m.q.s.ctx != null);
        try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(m.q.w));
        const qp = params_mod.affineParamsFromGeometry(m.q.w, m.q.s, 64) orelse return error.NoGeometry;
        try testing.expectEqual(@as(u32, 4), qp.bits);
        try testing.expectEqual(@as(u32, 64), qp.group_size);
        try testing.expect(m.mlp == .dense);
        try testing.expect(m.mlp.dense.gate.s.ctx != null);
        // down: contraction 96 not 64-divisible — stays dense pre-transposed.
        try testing.expect(m.mlp.dense.down.s.ctx == null);
        const down_shape = mlx.getShape(m.mlp.dense.down.w);
        try testing.expectEqual(@as(c_int, 96), down_shape[0]);
        // fc bf16 by contract.
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(m.fc.w));
    }

    // Lever off (0): everything stays dense.
    mtp.head_quant_override = 0;
    defer mtp.head_quant_override = null;
    {
        var m = try mtp.loadMtp(io, allocator, s, dir_path);
        defer m.deinit();
        try testing.expect(m.q.s.ctx == null);
        try testing.expect(m.mlp.dense.gate.s.ctx == null);
    }
}

/// A bare Transformer carrying nothing but a quantized lm_head + the config
/// fields `headQuantParams` reads. Every function under test touches exactly
/// those, which is the property that let the qwen4_exp head reuse the scheme.
/// A minimal target for the rerank scheme: a Transformer carrying nothing
/// but a quantized `lm_head`, which is all the scheme ever reads. `pub` so
/// the qwen4 arm's tests in generate.zig build the same target.
pub const RerankFixture = struct {
    xfm: Transformer,
    dense: mlx.mlx_array,
    vocab: c_int,
    hidden: c_int,

    pub fn init(s: mlx.mlx_stream, vocab: c_int, hidden: c_int, bits: u32, gs: u32, seed: u64) !RerankFixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const n: usize = @intCast(vocab * hidden);
        const buf = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(buf);
        // bf16-exact levels: the dense reference and the requantized source
        // must not differ by a rounding the test would read as a rerank miss.
        for (buf) |*v| v.* = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -128, 127))) * 0.0078125;
        const shape = [_]c_int{ vocab, hidden };
        const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var dense = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&dense, f32_arr, .bfloat16, s));

        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_quantize(
            &qv,
            dense,
            mlx.mlx_optional_int.some(@intCast(gs)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            "affine",
            .{ .ctx = null },
            s,
        ));
        var xfm: Transformer = undefined;
        xfm.rht = null;
        xfm.config = .{};
        xfm.config.hidden_size = @intCast(hidden);
        xfm.config.quant_mode = .affine;
        xfm.config.quant_bits = bits;
        xfm.config.quant_group_size = gs;
        xfm.s = s;
        xfm.lm_head_w = mlx.mlx_array_new();
        xfm.lm_head_s = mlx.mlx_array_new();
        xfm.lm_head_b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_w, qv, 0));
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_s, qv, 1));
        try mlx.check(mlx.mlx_vector_array_get(&xfm.lm_head_b, qv, 2));
        return .{ .xfm = xfm, .dense = dense, .vocab = vocab, .hidden = hidden };
    }

    /// The same head left dense: `lm_head_w` the bf16 [V, H] rows, no scales.
    pub fn initDense(s: mlx.mlx_stream, vocab: c_int, hidden: c_int, seed: u64) !RerankFixture {
        var fx = try init(s, vocab, hidden, 8, 64, seed);
        _ = mlx.mlx_array_free(fx.xfm.lm_head_s);
        _ = mlx.mlx_array_free(fx.xfm.lm_head_b);
        try mlx.check(mlx.mlx_array_set(&fx.xfm.lm_head_w, fx.dense));
        fx.xfm.lm_head_s = mlx.mlx_array_new();
        fx.xfm.lm_head_b = mlx.mlx_array_new();
        return fx;
    }

    pub fn deinit(self: *RerankFixture) void {
        _ = mlx.mlx_array_free(self.dense);
        _ = mlx.mlx_array_free(self.xfm.lm_head_w);
        _ = mlx.mlx_array_free(self.xfm.lm_head_s);
        _ = mlx.mlx_array_free(self.xfm.lm_head_b);
    }

    /// `[1, 1, hidden]` bf16 activation.
    fn randomX(self: *const RerankFixture, s: mlx.mlx_stream, seed: u64) !mlx.mlx_array {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        const buf = try testing.allocator.alloc(f32, @intCast(self.hidden));
        defer testing.allocator.free(buf);
        for (buf) |*v| v.* = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -64, 63))) * 0.015625;
        const shape = [_]c_int{ 1, 1, self.hidden };
        const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, s));
        return out;
    }

    /// The full-vocab logits the real head would produce — the exact answer
    /// the rerank is judged against.
    fn exactLogits(self: *const RerankFixture, s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
        const qp = mtp.headQuantParams(&self.xfm.config, self.xfm.lm_head_w, self.xfm.lm_head_s);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &out,
            x,
            self.xfm.lm_head_w,
            self.xfm.lm_head_s,
            self.xfm.lm_head_b,
            true,
            mlx.mlx_optional_int.some(@intCast(qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(qp.bits)),
            qp.mode.cstr(),
            s,
        ));
        return out;
    }
};

fn readIdScalar(arr: mlx.mlx_array) !u32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, arr));
    return @intCast(v);
}

fn readRow(allocator: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array, n: usize) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    const out = try allocator.alloc(f32, n);
    @memcpy(out, ptr[0..n]);
    return out;
}

test "mtp: draft rerank returns the exact argmax whenever it is inside the coarse shortlist" {
    // The scheme's whole contract. The coarse head only SHORTLISTS; the 32
    // survivors are re-scored through the real head's own rows, so the pick is
    // exact given the shortlist — and matches the full-vocab argmax exactly
    // when the coarse pass kept it. A miss costs ACCEPTANCE (the trunk verify
    // rejects a bad draft), never output, which is why the bar is conditional
    // rather than an equality on every draw.
    //
    // The bar is on the drafted token's exact LOGIT, not its id. A random
    // small head ties constantly — 256-wide dot products of bf16-exact levels
    // through an 8-bit quantizer collide, and the tie rate is scale-invariant
    // — and under a tie the host's first-max and a GPU argmax over the
    // shortlist's own ordering disagree on the id while agreeing on the value.
    // Two ids with the same logit are the same draft as far as acceptance is
    // concerned; convicting on the id would be judging our output against our
    // output's ordering. Id equality is still demanded wherever the reference's
    // OWN margin says the winner is unique.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    // Vocab must clear TOP32_MIN_ROWS with a ragged tail, like the top32 test.
    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0x5EED);
    defer fx.deinit();

    const coarse = mtp.buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?mtp.RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    var logged = false;

    // Two different fp paths read the same rows (a full-width qmm vs a 32-row
    // gathered one), so equality between them is approximate by construction.
    const eps: f32 = 1e-3;
    var kept: u32 = 0;
    var decisive: u32 = 0;
    var draws: u32 = 0;
    var seed: u64 = 1;
    while (seed <= 8) : (seed += 1) {
        const x = try fx.randomX(s, seed);
        defer _ = mlx.mlx_array_free(x);

        const exact = try fx.exactLogits(s, x);
        defer _ = mlx.mlx_array_free(exact);
        const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
        defer allocator.free(ex);
        var want: u32 = 0;
        for (ex, 0..) |v, i| if (v > ex[want]) {
            want = @intCast(i);
        };

        const picked_arr = (try mtp.rerankSelect(s, &fx.xfm, &slot, &logged, x, null)) orelse
            return error.RerankDeclined;
        defer _ = mlx.mlx_array_free(picked_arr);
        const picked = try readIdScalar(picked_arr);
        draws += 1;

        // What the coarse pass actually shortlisted, read the same way the
        // scheme reads it.
        var coarse_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(coarse_logits);
        try mlx.check(mlx.mlx_quantized_matmul(
            &coarse_logits,
            x,
            slot.?.q.w,
            slot.?.q.s,
            slot.?.q.b,
            true,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(3),
            "affine",
            s,
        ));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{fx.vocab};
        try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
        const cands = try mtp.draftTop32(s, flat, fx.vocab);
        defer _ = mlx.mlx_array_free(cands);
        try mlx.check(mlx.mlx_array_eval(cands));
        const cand_ptr = mlx.mlx_array_data_uint32(cands) orelse return error.NoData;

        var picked_in_list = false;
        var best_in_list: u32 = cand_ptr[0];
        var runner_up: f32 = -std.math.inf(f32);
        for (cand_ptr[0..32]) |c| {
            if (c == picked) picked_in_list = true;
            if (ex[c] > ex[best_in_list]) best_in_list = c;
        }
        // The reference's OWN margin: the gap between the shortlist's best and
        // the best OTHER value in it. Zero means the winner is not unique and
        // no id assertion is decidable.
        for (cand_ptr[0..32]) |c| {
            if (c == best_in_list) continue;
            if (ex[c] > runner_up) runner_up = ex[c];
        }
        const unique = (ex[best_in_list] - runner_up) > eps;

        // Unconditional: the pick is always FROM the shortlist and always
        // carries the shortlist's best exact logit. That is the re-scoring
        // step's own bar and it holds whether or not the coarse pass kept the
        // true argmax.
        try testing.expect(picked_in_list);
        try testing.expectApproxEqAbs(ex[best_in_list], ex[picked], eps);
        // Under a unique winner the id is decidable, so demand it.
        if (unique) {
            decisive += 1;
            try testing.expectEqual(best_in_list, picked);
        }

        // Conditional: the shortlist kept a row scoring the global max => the
        // drafted token scores the global max. This is the property acceptance
        // actually depends on — the trunk verify compares TOKENS, and two rows
        // with the same logit are the same draft.
        if (@abs(ex[best_in_list] - ex[want]) <= eps) {
            try testing.expectApproxEqAbs(ex[want], ex[picked], eps);
            kept += 1;
        }
    }
    // A coarse head that never keeps the argmax would satisfy every assertion
    // above while proposing garbage every step — the acceptance collapse this
    // change would otherwise ship silently.
    try testing.expect(kept * 2 >= draws);
    // And a fixture that ties on every draw would make the id bar vacuous.
    try testing.expect(decisive > 0);
}

test "mtp: the coarse rerank head drafts at its BUILT width" {
    // The width a coarse head is READ at is a property of the BUILT weights: callers pass `buildRerankCoarse` an
    // explicit width (the sidecar and the qwen4 arm do), and packed geometry implies K, so a mismatched width reads
    // a different vocab entirely.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0x5EED);
    defer fx.deinit();

    // Built at 3 bits...
    const coarse = mtp.buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?mtp.RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    // ...and the head SAYS so, without asking the environment.
    try testing.expectEqual(@as(u32, 3), slot.?.bits);
    try testing.expectEqual(@as(u32, 64), slot.?.group_size);
    try testing.expectEqual(fx.vocab, slot.?.rows);

    var logged = false;
    const eps: f32 = 1e-3;
    var kept: u32 = 0;
    var draws: u32 = 0;
    var seed: u64 = 1;
    while (seed <= 8) : (seed += 1) {
        const x = try fx.randomX(s, seed);
        defer _ = mlx.mlx_array_free(x);

        const exact = try fx.exactLogits(s, x);
        defer _ = mlx.mlx_array_free(exact);
        const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
        defer allocator.free(ex);
        var want: u32 = 0;
        for (ex, 0..) |v, i| if (v > ex[want]) {
            want = @intCast(i);
        };

        const picked_arr = (try mtp.rerankSelect(s, &fx.xfm, &slot, &logged, x, null)) orelse
            return error.RerankDeclined;
        defer _ = mlx.mlx_array_free(picked_arr);
        const picked = try readIdScalar(picked_arr);
        draws += 1;

        // The shortlist the 3-bit head actually produces — read at 3 bits,
        // which is the whole point.
        var coarse_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(coarse_logits);
        try mlx.check(mlx.mlx_quantized_matmul(
            &coarse_logits,
            x,
            slot.?.q.w,
            slot.?.q.s,
            slot.?.q.b,
            true,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(3),
            "affine",
            s,
        ));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{fx.vocab};
        try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
        const cands = try mtp.draftTop32(s, flat, fx.vocab);
        defer _ = mlx.mlx_array_free(cands);
        try mlx.check(mlx.mlx_array_eval(cands));
        const cand_ptr = mlx.mlx_array_data_uint32(cands) orelse return error.NoData;

        var picked_in_list = false;
        var best_in_list: u32 = cand_ptr[0];
        for (cand_ptr[0..32]) |c| {
            if (c == picked) picked_in_list = true;
            if (ex[c] > ex[best_in_list]) best_in_list = c;
        }
        // The draft came from the BUILT head's shortlist and carries its best
        // exact logit. Dispatched at any other width it comes from somewhere
        // else entirely.
        try testing.expect(picked_in_list);
        try testing.expectApproxEqAbs(ex[best_in_list], ex[picked], eps);
        if (@abs(ex[best_in_list] - ex[want]) <= eps) kept += 1;
    }
    // And the 3-bit head is still a USEFUL shortlister — a mis-dispatched read
    // that happened to stay in range would not be.
    try testing.expect(kept * 2 >= draws);
}

test "mtp: batched rerank ids equal N solo rerankSelect" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0xBA7C);
    defer fx.deinit();
    const coarse = mtp.buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?mtp.RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    var logged = false;
    const xa = try fx.randomX(s, 11);
    defer _ = mlx.mlx_array_free(xa);
    const xb = try fx.randomX(s, 23);
    defer _ = mlx.mlx_array_free(xb);
    const sa = (try mtp.rerankSelect(s, &fx.xfm, &slot, &logged, xa, null)) orelse return error.SoloRerankDeclined;
    defer _ = mlx.mlx_array_free(sa);
    const sb = (try mtp.rerankSelect(s, &fx.xfm, &slot, &logged, xb, null)) orelse return error.SoloRerankDeclined;
    defer _ = mlx.mlx_array_free(sb);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, xa);
    _ = mlx.mlx_vector_array_append_value(vec, xb);
    var stacked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    const bat = (try mtp.rerankSelectBatched(s, &fx.xfm, &slot, &logged, stacked, null)) orelse return error.BatchedRerankDeclined;
    defer _ = mlx.mlx_array_free(bat);
    var a0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a0);
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    try mlx.check(mlx.mlx_slice(&a0, bat, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_slice(&a1, bat, &[_]c_int{ 1, 0 }, 2, &[_]c_int{ 2, 1 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try testing.expectEqual(try readIdScalar(sa), try readIdScalar(a0));
    try testing.expectEqual(try readIdScalar(sb), try readIdScalar(a1));
}

test "mtp: exact batched shortlists hold each row's own head argmax" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0xBA7C);
    defer fx.deinit();
    var xs: [3]mlx.mlx_array = undefined;
    for (&xs, 0..) |*x, i| x.* = try fx.randomX(s, 31 + i);
    defer for (xs) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const vec = mlx.mlx_vector_array_new_data(&xs, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    var lists: [3]mtp.Shortlist = undefined;
    try testing.expect(try mtp.exactShortlistsBatched(s, &fx.xfm, stacked, null, &lists));
    defer for (&lists) |*sl| sl.deinit();
    for (xs, lists) |x, sl| {
        const want = try mtp.fullReadoutArgmax(s, &fx.xfm, x, null);
        defer _ = mlx.mlx_array_free(want);
        const got = try mtp.shortlistArgmax(s, sl);
        defer _ = mlx.mlx_array_free(got);
        try testing.expectEqual(try readIdScalar(want), try readIdScalar(got));
    }
}

test "mtp: rerankSelect declines without a coarse head, and the full readout is the exact argmax" {
    // The fallback is not decoration: `canRerankDrafts` false, a build refusal
    // and a mid-chain shortlist failure all land here, and the answer must
    // still be a legal draft — the full-vocab argmax, which is what the qwen4
    // head did before this change.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0xA11CE);
    defer fx.deinit();

    var none: ?mtp.RerankCoarse = null;
    var logged = false;
    const x = try fx.randomX(s, 7);
    defer _ = mlx.mlx_array_free(x);
    try testing.expect((try mtp.rerankSelect(s, &fx.xfm, &none, &logged, x, null)) == null);
    try testing.expect(!logged);

    const exact = try fx.exactLogits(s, x);
    defer _ = mlx.mlx_array_free(exact);
    const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
    defer allocator.free(ex);
    var want: u32 = 0;
    for (ex, 0..) |v, i| if (v > ex[want]) {
        want = @intCast(i);
    };

    const got = try mtp.fullReadoutArgmax(s, &fx.xfm, x, null);
    defer _ = mlx.mlx_array_free(got);
    try testing.expectEqual(want, try readIdScalar(got));
}

test "mtp: a suppressed id is never drafted, through the coarse shortlist or the full readout" {
    // The mask goes on the COARSE row, not the re-scored 32: a reserved id
    // that never enters the shortlist cannot be re-scored back into the draft.
    // Both paths are checked because a rerank build refusal silently routes
    // every draft through the other one.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0xBEEF);
    defer fx.deinit();
    const x = try fx.randomX(s, 3);
    defer _ = mlx.mlx_array_free(x);

    const exact = try fx.exactLogits(s, x);
    defer _ = mlx.mlx_array_free(exact);
    const ex = try readRow(allocator, s, exact, @intCast(fx.vocab));
    defer allocator.free(ex);
    var want: u32 = 0;
    for (ex, 0..) |v, i| if (v > ex[want]) {
        want = @intCast(i);
    };

    // Suppress exactly the winner.
    const mask_buf = try allocator.alloc(bool, @intCast(fx.vocab));
    defer allocator.free(mask_buf);
    @memset(mask_buf, false);
    mask_buf[want] = true;
    const mask_shape = [_]c_int{fx.vocab};
    const mask = mlx.mlx_array_new_data(mask_buf.ptr, &mask_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask);

    const coarse = mtp.buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    var slot: ?mtp.RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    var logged = false;
    const picked_arr = (try mtp.rerankSelect(s, &fx.xfm, &slot, &logged, x, mask)) orelse
        return error.RerankDeclined;
    defer _ = mlx.mlx_array_free(picked_arr);
    try testing.expect((try readIdScalar(picked_arr)) != want);

    const full = try mtp.fullReadoutArgmax(s, &fx.xfm, x, mask);
    defer _ = mlx.mlx_array_free(full);
    try testing.expect((try readIdScalar(full)) != want);
}

test "mtp: row-axis coarse logits equal each solo readout" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 2560, 8, 64, 0x3B17);
    defer fx.deinit();
    var rc = mtp.buildRerankCoarse(s, &fx.xfm, 3) orelse return error.CoarseBuildDeclined;
    defer rc.deinit();
    const tx = @import("../../../sdk/blocks/linear/qmv_rows.zig");
    {
        for ([_]usize{ 2, 4, 8, 17 }) |n| {
            var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
            defer for (parts[0..n]) |x| {
                if (x.ctx != null) _ = mlx.mlx_array_free(x);
            };
            for (0..n) |i| parts[i] = try fx.randomX(s, 17 + i);
            const vec = mlx.mlx_vector_array_new_data(parts[0..n].ptr, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_concatenate_axis(&x, vec, 0, s));
            const before = tx.mtp_head_row_dispatches;
            const got = try mtp.rerankCoarseRows(s, x, &rc);
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(usize, if (n <= 16) 1 else 0), tx.mtp_head_row_dispatches - before);
            for (parts[0..n], 0..) |input, i| {
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                try mlx.check(mlx.mlx_quantized_matmul(&want, input, rc.q.w, rc.q.s, rc.q.b, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(3), "affine", s));
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, got, &[_]c_int{ @intCast(i), 0, 0 }, 3, &[_]c_int{ @intCast(i + 1), 1, rc.rows }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                var same = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(same);
                try mlx.check(mlx.mlx_array_equal(&same, row, want, false, s));
                try mlx.check(mlx.mlx_array_eval(same));
                var equal: bool = false;
                try mlx.check(mlx.mlx_array_item_bool(&equal, same));
                if (!equal) std.debug.print("coarse logits differ: N={d} row={d}\n", .{ n, i });
                try testing.expect(equal);
            }
        }
    }
}

test "draft rerank on a dense bf16 head: shortlisted rows re-score to the full readout's logits" {
    if (mlx.noGpuBackend()) return;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var fx = try RerankFixture.initDense(s, mtp.TOP32_MIN_ROWS + 96, 256, 0xDE45E);
    defer fx.deinit();
    const coarse = mtp.buildRerankCoarse(s, &fx.xfm, 2) orelse return error.CoarseBuildDeclined;
    var slot: ?mtp.RerankCoarse = coarse;
    defer if (slot) |*q| q.deinit();
    var logged = false;
    const x = try fx.randomX(s, 3);
    defer _ = mlx.mlx_array_free(x);
    var full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(full);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, fx.dense, s));
    try mlx.check(mlx.mlx_matmul(&full, x, wt, s));
    const ex = try readRow(allocator, s, full, @intCast(fx.vocab));
    defer allocator.free(ex);
    const sl = (try mtp.rerankShortlist(s, &fx.xfm, &slot, &logged, x, null)) orelse return error.RerankDeclined;
    defer {
        _ = mlx.mlx_array_free(sl.cands);
        _ = mlx.mlx_array_free(sl.exact);
    }
    const got = try readRow(allocator, s, sl.exact, 32);
    defer allocator.free(got);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, sl.cands, .int32, s));
    try mlx.check(mlx.mlx_array_eval(ids));
    const pid = mlx.mlx_array_data_int32(ids).?;
    for (0..32) |i| try testing.expectApproxEqAbs(ex[@intCast(pid[i])], got[i], 1e-2);
    var eb: [512]u8 = undefined;
    if (mlx.takeError(&eb)) |m| {
        std.debug.print("latched: {s}\n", .{m});
        return error.MlxErrorLatched;
    }
}

test "mtp: a row-scaled FP8 trunk head reads through fp8Linear and latches nothing at bind" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var fx = try RerankFixture.init(s, mtp.TOP32_MIN_ROWS + 96, 256, 8, 64, 0x5EED);
    defer fx.deinit();

    // e4m3 0x38 is 1.0, so row r of the head is its two 128-wide block scales.
    const V: usize = @intCast(fx.vocab);
    const H: usize = @intCast(fx.hidden);
    const codes = try allocator.alloc(u8, V * H);
    defer allocator.free(codes);
    @memset(codes, 0x38);
    const sc = try allocator.alloc(f32, V * (H / 128));
    defer allocator.free(sc);
    for (sc, 0..) |*v, i| v.* = 0.125 * @as(f32, @floatFromInt(1 + (i / 2 + i % 2) % 4));
    _ = mlx.mlx_array_free(fx.xfm.lm_head_w);
    fx.xfm.lm_head_w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ fx.vocab, fx.hidden }, 2, .uint8);
    _ = mlx.mlx_array_free(fx.xfm.lm_head_s);
    fx.xfm.lm_head_s = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{ fx.vocab, @intCast(H / 128) }, 2, .float32);

    try testing.expect(mtp.buildRerankCoarse(s, &fx.xfm, 2) == null);

    const x = try fx.randomX(s, 3);
    defer _ = mlx.mlx_array_free(x);
    const logits = try fx.xfm.lmHeadForeign(x);
    defer _ = mlx.mlx_array_free(logits);
    const got = try readRow(allocator, s, logits, V);
    defer allocator.free(got);
    const xr = try readRow(allocator, s, x, H);
    defer allocator.free(xr);
    var half: [2]f32 = .{ 0, 0 };
    for (xr, 0..) |v, k| half[k / 128] += v;
    for (got, 0..) |g, r| try testing.expectApproxEqAbs(sc[2 * r] * half[0] + sc[2 * r + 1] * half[1], g, 0.0625);

    var eb: [512]u8 = undefined;
    if (mlx.takeError(&eb)) |m| {
        std.debug.print("latched: {s}\n", .{m});
        return error.MlxErrorLatched;
    }
}
