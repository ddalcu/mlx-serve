//! GLM-5.3-Flash's MTP head (`mtp.0`) behind `SpecHooks.head`. Row p pairs the trunk's final-normed hidden at p
//! (`Forward.hidden_in`) with token p+1 (`Forward.ids`): `eh_proj(cat[enorm(embed(x)), hnorm(h)])` through one plain
//! DSA + MoE block appended to the head's one-layer cache at its own length, and the last row's `norm` is the output
//! (the next depth's hidden, which the engine projects through the trunk head). The weights load with the trunk
//! (`trunk.zig`); `load` makes them resident.
//! copied_from: plugins/glm5_next/mtp.zig (`Head.rows`, `Head.forward`), the cache read through `Slots`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const host = @import("host.zig");
const trunk = @import("trunk.zig");
const table_forward = @import("table_forward.zig");
const glm_moe = @import("moe.zig");
const Slots = @import("slots.zig").Slots;
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");

const Array = mlx.mlx_array;
const Verdict = abi.Verdict;

fn load(mp: *anyopaque, _: *const abi.LoadCtx) callconv(.c) Verdict {
    const m = host.of(mp);
    if (m.mtp == null) return Verdict.refuse(.unsupported, "the pack ships no MTP head");
    trunk.readyHead(m) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = host.of(mp);
    if (!m.mtp_ready) return Verdict.refuse(.failed, "MTP head not loaded");
    const h = &m.mtp.?;
    const ops = f.cache_ops orelse return Verdict.refuse(.unsupported, "a head forward without cache_ops");
    if (f.caches.len != 1 or f.hidden_in.ctx == null or f.head != 0) return Verdict.refuse(.unsupported, "one head, one cache, the trunk's hidden");
    const normed = step(m, h, ops, f) catch |e| return Verdict.named(.failed, @errorName(e));
    if (f.want_hidden) out.hidden = .{ .ctx = normed.ctx } else _ = mlx.mlx_array_free(normed);
    return .ok;
}

/// The block over the forward's rows, appended to the head's cache; the last row's normed output.
fn step(m: *host.Model, h: *const trunk.Head, ops: *const abi.CacheOps, f: *const abi.Forward) !Array {
    const s = m.s;
    const ids = try host.tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);
    const n: c_int = mlx.getShape(ids)[1];
    // The legacy head attends at its cache's own length, never the trunk's position.
    const pos = [_]u32{@intCast(ops.len(f.caches.slice()[0], 0))};
    const sl: Slots = .{ .ops = ops, .caches = f.caches.slice(), .positions = &pos };

    const emb = try m.embedding(ids);
    defer _ = mlx.mlx_array_free(emb);
    const en = try m.rmsNorm(emb, h.enorm);
    defer _ = mlx.mlx_array_free(en);
    var hid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hid);
    try mlx.check(mlx.mlx_astype(&hid, .{ .ctx = f.hidden_in.ctx }, mlx.mlx_array_dtype(emb), s));
    const hn = try m.rmsNorm(hid, h.hnorm);
    defer _ = mlx.mlx_array_free(hn);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    {
        const vec = mlx.mlx_vector_array_new_data(&[_]Array{ en, hn }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
    }
    const x = try m.mm(&h.eh_proj, cat);
    defer _ = mlx.mlx_array_free(x);
    const a_in = try m.rmsNorm(x, h.input_norm);
    defer _ = mlx.mlx_array_free(a_in);
    const attn = try table_forward.dsaAttn(m, sl, 0, a_in, &h.dsa, 0, 1, n);
    defer _ = mlx.mlx_array_free(attn);
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_add(&x1, x, attn, s));
    const m_in = try m.rmsNorm(x1, h.post_norm);
    defer _ = mlx.mlx_array_free(m_in);
    // Only the last row is returned (a history append's output is never read: it runs whole).
    if (n > 1 and f.want_hidden and lastRowOn()) if (try glm_moe.forwardLastRow(&h.moe, s, m_in)) |mlp1| {
        defer _ = mlx.mlx_array_free(mlp1);
        lastRowEngaged();
        return tailLast(m, h, x1, mlp1);
    };
    const mlp = try glm_moe.forward(&h.moe, s, m_in);
    defer _ = mlx.mlx_array_free(mlp);
    const y = try addOf(s, x1, mlp);
    defer _ = mlx.mlx_array_free(y);
    const last = try lastRow(s, y);
    defer _ = mlx.mlx_array_free(last);
    return m.rmsNorm(last, h.norm);
}

var last_row_switch: ?bool = null;
var last_row_said = false;

/// MLX_SERVE_GLM_HEAD_LAST_ROW, read once.
fn lastRowOn() bool {
    if (last_row_switch == null) last_row_switch = if (env.get(.MLX_SERVE_GLM_HEAD_LAST_ROW)) |v| v[0] != '0' else true;
    return last_row_switch.?;
}

fn lastRowEngaged() void {
    if (last_row_said) return;
    last_row_said = true;
    log.info("[glm5_next] MTP head last-row experts engaged\n", .{});
}

fn addOf(s: mlx.mlx_stream, a: Array, b: Array) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, a, b, s));
    return out;
}

/// The step's last row of `x1 + mlp` under the final norm, over the MoE's last row alone (`mlp1` [1, 1, D]): the bytes
/// the whole step makes for that row (the add is elementwise, the norm a row's).
fn tailLast(m: *host.Model, h: *const trunk.Head, x1: Array, mlp1: Array) !Array {
    const last = try lastRow(m.s, x1);
    defer _ = mlx.mlx_array_free(last);
    const y = try addOf(m.s, last, mlp1);
    defer _ = mlx.mlx_array_free(y);
    return m.rmsNorm(y, h.norm);
}

/// Row S - 1 of `y` [1, S, D]: [1, 1, D].
fn lastRow(s: mlx.mlx_stream, y: Array) !Array {
    const sh = mlx.getShape(y);
    var last = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(last);
    try mlx.check(mlx.mlx_slice(&last, y, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return last;
}

fn noop(_: *anyopaque, _: u32) callconv(.c) Verdict {
    return .ok;
}
fn release(_: *anyopaque, _: u32) callconv(.c) void {}
fn truncate(_: *anyopaque, _: u32, _: u64) callconv(.c) void {}
fn adopt(_: *anyopaque, _: u32, _: abi.Handle) callconv(.c) Verdict {
    return .ok;
}
fn select(_: *anyopaque, _: abi.Array, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "the engine selects drafts");
}
/// The head's tensors ride the trunk's shards, billed with them.
fn bill(_: *const abi.Parsed) callconv(.c) u64 {
    return 0;
}

pub const hooks: abi.HeadHooks = .{ .load = &load, .activate = &noop, .release = &release, .forward = &forward, .truncate = &truncate, .adopt = &adopt, .mark = &release, .draft_select = &select, .bill = &bill };
