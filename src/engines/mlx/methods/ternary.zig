//! The legacy `Transformer`'s methods over `sdk/quant/ternary.zig` (what reads the engine's model state).

const qmv2 = @import("../../../sdk/blocks/qmm/qmv2.zig");
const Transformer = @import("../transformer.zig").Transformer;

/// A 2-bit ternary trunk whose verify rows ride qmv2's MMA lane: an
/// 8-row window costs about what 5 rows do, like the NAX lane.
pub fn ternaryWideVerify(self: *const Transformer) bool {
    return (self.rht != null or self.ternary_2bit) and qmv2.deviceMmaVerifiesWide();
}
