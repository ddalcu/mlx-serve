//! The legacy engine's EXL3 bank check: a loaded `MoeMlpWeights` against its config, through the EXL3 format.

const exl3 = @import("../../../sdk/quant/exl3/format.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;

/// An EXL3 bank's suh/svh bind optionally beside the affine names, so their absence, a trellis the kernels cannot
/// decode at the config's rate and a rate the pack's codebook table has no row for refuse here.
pub fn checkExl3Bank(mw: *const MoeMlpWeights, config: ModelConfig, spec: exl3.Spec) !void {
    const h = config.hidden_size;
    const inter = config.moe_intermediate_size;
    const projs = [_]struct { p: exl3.Proj, in: u32, out: u32 }{
        .{ .p = .{ .trellis = mw.switch_gate_w, .suh = mw.switch_gate_s, .svh = mw.switch_gate_b }, .in = h, .out = inter },
        .{ .p = .{ .trellis = mw.switch_up_w, .suh = mw.switch_up_s, .svh = mw.switch_up_b }, .in = h, .out = inter },
        .{ .p = .{ .trellis = mw.switch_down_w, .suh = mw.switch_down_s, .svh = mw.switch_down_b }, .in = inter, .out = h },
    };
    for (projs) |p| try exl3.admit(p.p, config.num_experts, p.in, p.out, spec);
}
