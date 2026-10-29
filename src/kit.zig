//! The plugin kit's one Zig module (`mlx_serve`): the ABI tables and the MLX-free SDK an outside plugin builds
//! against. Its import closure is the kit's source (`zig build kit` copies exactly that and refuses one that reaches
//! the engine or the host); MLX itself reaches a plugin as the kit's `mlx_c` module (mlx-c's headers).

pub const abi = struct {
    pub const common = @import("abi/common.zig");
    pub const plugin = @import("abi/plugin.zig");
    pub const media = @import("abi/media.zig");
    pub const quant = @import("abi/quant.zig");
    pub const host = @import("abi/host.zig");
    pub const dylib = @import("abi/dylib.zig");
};

/// `sdk.Plugin`, `claimsByType`, `fixture`, `tensors`, `byte_tokenizer`, as built-in plugins see them.
pub const sdk = @import("sdk/root.zig");
/// The `Host` table's plugin-side accessor (`Host.of(ctx.host)`).
pub const host = @import("sdk/host.zig");
/// `exportManifest`: what the kit's build calls on the plugin root.
pub const dylib = @import("sdk/dylib.zig");
/// `generation_config.json` into `Shell.gen`.
pub const generation = @import("sdk/config/generation.zig");
/// WAV encoding for audio backends.
pub const wav = @import("sdk/media/wav.zig");

test {
    _ = abi.dylib;
    _ = dylib;
}
