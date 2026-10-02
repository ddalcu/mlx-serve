//! Capability flags derived from build options, so shared modules can swap MLX /
//! media engines for compile-time stubs on a llama.cpp-only build without an
//! `if (build_options.…)` scattered at every use site. Import sites alias, e.g.:
//!   const mlx = if (@import("build_cfg.zig").mlx_enabled)
//!       @import("mlx.zig") else @import("mlx_stub.zig");
//! On macOS the defaults keep every engine real; `-Dgguf-only` (default on Linux)
//! turns MLX + media generation off and serves GGUF through the embedded
//! llama.cpp engine only.
const opts = @import("build_options");

/// GGUF-only build: no MLX runtime, no MLX-arch safetensors, no media generation.
pub const gguf_only: bool = opts.gguf_only;

/// The MLX runtime + MLX-arch transformer/safetensors path is compiled in.
pub const mlx_enabled: bool = !gguf_only;

/// Image/audio/video/3D generation (all MLX-backed today).
pub const media_gen_enabled: bool = mlx_enabled;

/// The embedded llama.cpp engine (GGUF). Real on every non-iOS build.
pub const llama_enabled: bool = !opts.ios;

/// The DSV4-Flash native (ds4) engine — macOS Metal only.
pub const ds4_enabled: bool = opts.macos_engines and !gguf_only;

/// Apple Neural Engine offload — macOS only.
pub const ane_enabled: bool = opts.macos_engines and !gguf_only;
