//! The host surface this plugin reads beyond the SDK's, as one comptime capability table, read by the archs that need
//! it. Each entry is a PUBLIC declaration of the host's `mlx_host` root: `@hasDecl` sees only pub
//! declarations, and an absent entry is never a silent route: GLM-5.3 refuses by name at construction
//! (`error.GlmHostSurfaceMissing`, plugin.zig) and its tests skip. A host older than the GLM-5.3 stack (ddalcu main,
//! pr/mimo-host) exports neither entry; DeepSeek-V4.1 and MiMo need neither.

const mlx_host = @import("mlx_host");

/// sushi's EXL3 MoE (`mlx_host.exl3`): GLM-5.3's EXL3 routed experts and its MTP layer's.
pub const exl3: bool = @hasDecl(mlx_host, "exl3");
/// The host's DSA kernels (`mlx_host.dsa_nax`): GLM-5.3's sparse latent attention and its indexer scores.
pub const dsa_nax: bool = @hasDecl(mlx_host, "dsa_nax");
/// GLM-5.3 (`glm_moe_dsa`) is constructible on this host: both entries.
pub const glm: bool = exl3 and dsa_nax;
