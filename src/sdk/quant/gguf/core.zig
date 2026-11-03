//! The MLX-free part: GGUF reader + reference dequant.
pub const gguf = @import("gguf.zig");
pub const quants = @import("quants.zig");
pub const meta = @import("meta.zig");
pub const arch = @import("arch.zig");

test {
    _ = gguf;
    _ = quants;
    _ = meta;
    _ = arch;
}
