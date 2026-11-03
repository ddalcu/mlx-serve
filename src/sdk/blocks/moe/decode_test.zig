//! `decode.zig`'s tests, out of its import graph: a plugin unit that reaches `decode.zig` never compiles them.

const decode = @import("decode.zig");
const std = @import("std");
const moe_fp4 = @import("moe_fp4.zig");
const testing = std.testing;
const swigluClampLimit = @import("../mlp/activations.zig").swigluClampLimit;
const moeFp4DecodeEligible = decode.moeFp4DecodeEligible;
const ModelConfig = @import("../../config/model.zig").ModelConfig;

test "moe fp4 decode: expert biases or a clamp no kernel serves keep the composed chain" {
    var cfg: ModelConfig = undefined;
    cfg.swiglu_limit = 0;
    cfg.swiglu_clamp = false;
    try testing.expectEqual(@as(?u32, 0), swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp));
    cfg.swiglu_limit = 10;
    cfg.swiglu_clamp = true;
    try testing.expectEqual(@as(?u32, 10), swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, true, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 10, .bfloat16, .nvfp4, .nvfp4, .nvfp4));
    // An f32 stream (an f32 norm weight upstream) has no sigmoid table: decline, never throw.
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .float32, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .float16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, null, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .bfloat16, .affine, .mxfp4, .mxfp4));
    try testing.expect(moeFp4DecodeEligible(1, false, 0, .bfloat16, .nvfp4, .nvfp4, .nvfp4));
    try testing.expect(!moeFp4DecodeEligible(1, false, 0, .bfloat16, .nvfp4, .mxfp4, .nvfp4));
    try testing.expect(!moeFp4DecodeEligible(moe_fp4.MAX_ROWS + 1, false, 0, .bfloat16, .mxfp4, .mxfp4, .mxfp4));
}
