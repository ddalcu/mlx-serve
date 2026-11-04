//! engines/mlx/spec/trunk_hooks.
//! Owns lmHeadForDraft, tileLaneWeights, specTreeSupported, lmHeadRowsForDraft.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const lane_qmm = @import("../../../sdk/blocks/qmm/lane_qmm.zig");
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const gdnDecodeRecurEnabled = @import("../../../sdk/blocks/gdn/decode_fused.zig").gdnDecodeRecurEnabled;
const FusedRows = @import("../../../sdk/blocks/linear/fused_rows.zig").FusedRows;
const row_exact = @import("../../../sdk/bases/blocks/row_exact.zig");
const log = @import("../../../sdk/util/log.zig");
const Transformer = @import("../transformer.zig").Transformer;

/// Project a hidden state through the trunk's lm_head, dense (no argmax
/// prune). DFlash draft logits come from the assistant's hidden through
/// THIS head — the reference applies the bare Linear (no softcap /
/// output_multiplier; both are monotone, so draft argmax is unaffected).
pub fn lmHeadForDraft(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
    return self.lmHeadProject(x);
}

/// Re-orders every trunk projection the lane matmul serves into its tiled
/// layout, in the weight's own buffer (no copy stays resident): the exact
/// path reads them there at every width, the prompt path through the same
/// layout. The lm_head and embeddings keep MLX's layout (other readers).
/// A served model tiles its own projections (`SpecHooks.tile_lanes`) and logs what it tiled; 0 is returned.
pub fn tileLaneWeights(self: *Transformer) !u64 {
    if (!naxAvailable() or !self.config.rowExactDecode() or self.rht != null or self.actDtype() != .bfloat16) return 0;
    if (self.plugin) |b| {
        const tile = (b.table.spec orelse return 0).tile_lanes orelse return 0;
        try mlx.check(mlx.mlx_synchronize(self.s));
        const v = tile(b.model, @intFromEnum(row_exact.Lane.mma));
        if (!v.isOk()) {
            log.err("[plugin] {s}: tiling the trunk's lane weights failed: {s}\n", .{ b.table.name.slice(), v.why.slice() });
            return error.PluginTileFailed;
        }
        return 0;
    }
    const layers = self.moe_layers orelse return 0;
    try mlx.check(mlx.mlx_synchronize(self.s));
    const owner = @intFromPtr(self);
    var bytes: u64 = 0;
    const Tile = struct {
        fn one(x: *const Transformer, own: usize, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !u64 {
            if (sc.ctx == null) return 0;
            const qp = x.quantParamsFor(w, sc);
            if (qp.mode != .affine) return 0;
            return lane_qmm.tileInPlace(own, w, sc, bi, qp.bits, qp.group_size, widths, x.s);
        }
        fn rows(x: *const Transformer, own: usize, f: *const FusedRows, parts: []const [3]mlx.mlx_array) !u64 {
            if (f.w.ctx != null) return one(x, own, f.w, f.s, f.b, f.widths[0..f.count]);
            var sum: u64 = 0;
            for (parts) |p| if (p[0].ctx != null) {
                sum += try one(x, own, p[0], p[1], p[2], &.{});
            };
            return sum;
        }
    };
    for (layers) |*l| {
        switch (l.attn) {
            .full => |*fa| {
                bytes += try Tile.rows(self, owner, &fa.qkv, &.{ .{ fa.q_w, fa.q_s, fa.q_b }, .{ fa.k_w, fa.k_s, fa.k_b }, .{ fa.v_w, fa.v_s, fa.v_b } });
                bytes += try Tile.one(self, owner, fa.o_w, fa.o_s, fa.o_b, &.{});
            },
            .linear => |*la| {
                if (!la.combined_proj) bytes += try Tile.rows(self, owner, &la.in, &.{ .{ la.qkv_w, la.qkv_s, la.qkv_b }, .{ la.z_w, la.z_s, la.z_b }, .{ la.a_w, la.a_s, la.a_b }, .{ la.b_w, la.b_s, la.b_b } });
                bytes += try Tile.one(self, owner, la.out_w, la.out_s, la.out_b, &.{});
            },
        }
        switch (l.mlp) {
            .dense => |*dm| {
                bytes += try Tile.rows(self, owner, &dm.gu, &.{ .{ dm.gate_w, dm.gate_s, dm.gate_b }, .{ dm.up_w, dm.up_s, dm.up_b } });
                bytes += try Tile.one(self, owner, dm.down_w, dm.down_s, dm.down_b, &.{});
            },
            .moe => {},
        }
    }
    return bytes;
}

/// Verify rows can form a draft tree: every per-row piece of this forward
/// follows a row's ancestors (row-exact kernels, the GDN fold).
pub fn specTreeSupported(self: *const Transformer) bool {
    // The tree verify's GDN kernels are Metal-only; elsewhere a tree round fails, so draft chains.
    const forward = if (self.config.plugin) |pb| self.plugin != null and pb.parsed.caps.spec_tree else self.moe_layers != null;
    return self.config.rowExactDecode() and forward and self.rht == null and gdnDecodeRecurEnabled() and mlx.metalKernelsAvailable();
}

/// The quantized head's first `rows` vocab rows only (a draft vocabulary).
pub fn lmHeadRowsForDraft(self: *const Transformer, x: mlx.mlx_array, rows: c_int) !mlx.mlx_array {
    if (self.lm_head_s.ctx == null) return self.lmHeadProject(x);
    var parts: [3]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |p| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
    };
    for ([_]mlx.mlx_array{ self.lm_head_w, self.lm_head_s, self.lm_head_b }, &parts) |a, *out| {
        if (a.ctx == null) continue;
        const cols = mlx.getShape(a)[1];
        out.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(out, a, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ rows, cols }, 2, &[_]c_int{ 1, 1 }, 2, self.s));
    }
    return self.qmatmul(x, parts[0], parts[1], parts[2]);
}

// ── tests ──

test "specTreeSupported follows the backend that runs the tree kernels" {
    var xfm: Transformer = undefined;
    xfm.config = .{ .model_type = "qwen3_5", .row_exact_covered = true, .dflash_bound = true };
    xfm.moe_layers = &.{};
    xfm.rht = null;
    try std.testing.expect(xfm.config.rowExactDecode());
    try std.testing.expectEqual(gdnDecodeRecurEnabled() and mlx.metalKernelsAvailable(), xfm.specTreeSupported());
}
