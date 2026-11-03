//! The compile-time refusals of the contracts this plugin keeps beside the host's SDK (sdk_ext.zig): `zig build
//! refusals` compiles this root once per case (`refusal_case.name`), each with one bad declaration, and expects the
//! compile error that names it. Never imported by the plugin.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const case = @import("refusal_case").name;

fn is(comptime name: []const u8) bool {
    return std.mem.eql(u8, case, name);
}

const Claims = struct {
    pub const name = "fixture-kind";
    pub fn claims(_: *const sdk.ConfigPeek) ?sdk.Priority {
        return null;
    }
};
const WrongCaps = struct {
    pub const name = "fixture-kind";
    pub const claims = Claims.claims;
    pub const caps: u32 = 1;
};
const HalfQuant = struct {
    pub const name = "fixture-quant";
    pub fn Arrays(comptime T: type) type {
        return T;
    }
    pub fn claims(_: *const sdk.GroupPeek, _: ?*sdk.Diag) ?sdk.Priority {
        return null;
    }
};
const QuantWrongClaims = struct {
    pub const name = "fixture-quant";
    pub fn Arrays(comptime T: type) type {
        return T;
    }
    pub fn claims(_: *const sdk.GroupPeek) ?sdk.Priority {
        return null;
    }
    pub fn Accepted(comptime G: type) type {
        return G;
    }
    pub fn accept() void {}
};
/// A backend whose profile hook lacks a prefill probe.
const HalfHook = struct {
    pub const profile_hook: sdk_ext.profile.Hook = .{ .prefill = struct {
        pub const enabled = true;
        pub const Stamp = u64;
    } };
};

comptime {
    if (is("expert_source_wrong_caps")) {
        _ = sdk_ext.ExpertSource.of(WrongCaps);
    } else if (is("quant_half_contract")) {
        _ = sdk_ext.Quant.of(HalfQuant);
    } else if (is("quant_claims_mistyped")) {
        _ = sdk_ext.Quant.of(QuantWrongClaims);
    } else if (is("quant_check_direct")) {
        // the C2 check of its own (a quant's tests call it outside the table, which checks claims first)
        sdk_ext.quant.check(QuantWrongClaims);
    } else if (is("profile_hook_missing_probe")) {
        _ = sdk_ext.profile.of(HalfHook);
    } else @compileError("unknown refusal case " ++ case);
}
