//! One switch for the LOSSY int8 prefill routes on NAX GPUs: `--int8-prefill` or
//! `MLX_SERVE_INT8_PREFILL=1` turns on every route that applies to the loaded model,
//! and a model's `int8_prefill` setting overrides it either way. Each route keeps an
//! env variable that only turns it OFF (`=0`), for A/Bs of one route against the rest.
const std = @import("std");

pub const Route = enum {
    /// 4-bit affine projections through the rotated int8 GEMM (`w8a8.prefillQmm`).
    affine_4bit,
    /// int8 Q/K in the causal hd-256 prefill attention (`nax_attention.attention256Int8`).
    attn_qk,
    /// 2-bit Prism projections (`qmm_int8`).
    prism_2bit,

    pub fn killEnv(r: Route) [*:0]const u8 {
        return switch (r) {
            .affine_4bit => "MLX_SERVE_W8A8_PREFILL",
            .attn_qk => "MLX_SERVE_INT8_QK",
            .prism_2bit => "MLX_SERVE_BONSAI_INT8_PREFILL",
        };
    }
};

/// Set by `--int8-prefill`.
pub var cli: bool = false;

fn set(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != 0 and v[0] != '0';
}

fn killed(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] == '0';
}

/// Pure form of `on`: the model setting beats the process switch; a route's `=0` beats both.
pub fn onFrom(override: ?bool, cli_on: bool, master: ?[*:0]const u8, kill: ?[*:0]const u8) bool {
    if (killed(kill)) return false;
    return override orelse (cli_on or set(master));
}

/// Whether `route` runs for a model whose `int8_prefill` setting is `override`. Read at load.
pub fn on(override: ?bool, route: Route) bool {
    return onFrom(override, cli, std.c.getenv("MLX_SERVE_INT8_PREFILL"), std.c.getenv(route.killEnv()));
}

test "int8 prefill: off by default; --int8-prefill or MLX_SERVE_INT8_PREFILL=1 turns it on" {
    try std.testing.expect(!onFrom(null, false, null, null));
    try std.testing.expect(!onFrom(null, false, "0", null));
    try std.testing.expect(!onFrom(null, false, "", null));
    try std.testing.expect(onFrom(null, false, "1", null));
    try std.testing.expect(onFrom(null, true, null, null));
}

test "int8 prefill: a route variable only turns its route off" {
    try std.testing.expect(!onFrom(null, true, null, "0"));
    try std.testing.expect(!onFrom(null, false, "1", "0"));
    try std.testing.expect(!onFrom(null, false, null, "1"));
    try std.testing.expect(onFrom(null, true, null, "1"));
    try std.testing.expect(!onFrom(true, false, null, "0"));
}

test "int8 prefill: the model setting wins over the switch both ways" {
    try std.testing.expect(onFrom(true, false, null, null));
    try std.testing.expect(onFrom(true, false, "0", null));
    try std.testing.expect(!onFrom(false, true, null, null));
    try std.testing.expect(!onFrom(false, false, "1", null));
}

test "int8 prefill: every route has its own kill variable" {
    try std.testing.expectEqualStrings("MLX_SERVE_W8A8_PREFILL", std.mem.span(Route.affine_4bit.killEnv()));
    try std.testing.expectEqualStrings("MLX_SERVE_INT8_QK", std.mem.span(Route.attn_qk.killEnv()));
    try std.testing.expectEqualStrings("MLX_SERVE_BONSAI_INT8_PREFILL", std.mem.span(Route.prism_2bit.killEnv()));
}
