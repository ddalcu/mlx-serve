# SD 3.5 port — interface contract (working doc)

One backend serves **SD 3.5 Large**, **Large Turbo** and **Medium**. Three lanes
are built in parallel against the interfaces below; nothing in a lane may edit a
file another lane owns.

## Verified geometry (read from the checkpoints' own configs, not from docs)

| | Large / Large-Turbo | Medium |
|---|---|---|
| `num_layers` | 38 | 24 |
| `num_attention_heads` x `attention_head_dim` | 38 x 64 = **2432** | 24 x 64 = **1536** |
| `caption_projection_dim` | 2432 | 1536 |
| `pos_embed_max_size` | 192 | **384** |
| `dual_attention_layers` | absent | **[0..12]** (MMDiT-**X**) |
| `qk_norm` | `rms_norm` | `rms_norm` |
| `patch_size` / `in_channels` / `out_channels` | 2 / 16 / 16 | same |
| `joint_attention_dim` (T5 width) | 4096 | same |
| `pooled_projection_dim` (CLIP-L 768 + CLIP-G 1280) | 2048 | same |
| scheduler | FlowMatchEuler, `shift: 3.0` | same |

VAE (all three): `AutoencoderKL`, `latent_channels: 16`,
`block_out_channels: [128,256,512,512]`, `layers_per_block: 2`,
`norm_num_groups: 32`, silu, mid-block attention **on**, `force_upcast: true`,
`scaling_factor: 1.5305`, `shift_factor: 0.0609`, and
**`use_quant_conv: false` / `use_post_quant_conv: false`** — the one place it
parts company with SDXL's VAE.

T5 (`text_encoder_3`): `T5EncoderModel`, 24 layers, `d_model` 4096, `d_ff` 10240,
64 heads x `d_kv` 64, `feed_forward_proj: gated-gelu` (`gelu_new` = tanh
approximation), `layer_norm_epsilon` 1e-6, `relative_attention_num_buckets` 32,
`relative_attention_max_distance` 128, `vocab_size` 32128, `tie_word_embeddings`
false. Tokenizer is SentencePiece **Unigram** (`tokenizer_3/tokenizer.json`),
NOT the CLIP BPE.

CLIP-L and CLIP-G are geometrically identical to SDXL's
(`sdxl.CLIP_L_CONFIG` 768/12/12 quick_gelu, `sdxl.CLIP_BIG_G_CONFIG`
1280/32/20 gelu) and are bound by `sdxl_clip.zig` unchanged. Both are
`CLIPTextModelWithProjection` here, so BOTH contribute a projected pooled vector.

Ungated config/weight mirrors (the stability repos are gated):
`adamo1139/stable-diffusion-3.5-{large,large-turbo,medium}-ungated`.

## Lanes and file ownership

Create ONLY the files your lane owns. Do not touch `build.zig`, `src/tests.zig`,
`src/gen.zig`, `src/cli.zig`, `src/model_discovery.zig` or anything under `app/`
— the pipeline lane owns every shared-file edit and will register your module.

| Lane | Owns |
|---|---|
| A — text | `src/t5_encoder.zig`, `src/t5_tokenizer.zig`, `tests/dump_t5_fixtures.py` |
| B — vae | `src/sd3_vae.zig`, `tests/dump_sd3_vae_fixtures.py` |
| C — mmdit | `src/sd3_mmdit.zig`, `tests/dump_sd3_mmdit_fixtures.py` |
| D — pipeline (owner) | `src/sd3.zig`, `src/sd3_pipeline.zig`, all shared-file wiring |

## Interfaces each lane must expose

Mirror `sdxl_unet.zig` / `sdxl_vae.zig`: a `load` that reads a component dir,
and a `loadFromWeights` half so a test can drive it from an in-memory map.

```zig
// Lane A
pub const T5Config = struct { num_layers: u32 = 24, d_model: u32 = 4096, d_ff: u32 = 10240,
    num_heads: u32 = 64, d_kv: u32 = 64, eps: f32 = 1e-6,
    rel_buckets: u32 = 32, rel_max_distance: u32 = 128, vocab_size: u32 = 32128 };
pub const T5Encoder = struct {
    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, sub: []const u8, dtype: mlx.mlx_dtype) !T5Encoder;
    pub fn loadFromWeights(a: std.mem.Allocator, s: S, w: *const Weights, cfg: T5Config, dtype: mlx.mlx_dtype) !T5Encoder;
    pub fn deinit(self: *T5Encoder) void;
    /// ids [1, T] int32 -> hidden [1, T, d_model]. No pooling, no final lm head.
    /// Padding is handled by the CALLER passing a full-length id buffer padded
    /// with pad_token_id 0 — SD 3 pads to a fixed length and does NOT mask.
    pub fn forward(self: *T5Encoder, ids: []const i32, s: S) !mlx.mlx_array;
};
pub const T5Tokenizer = struct {
    pub fn load(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, sub: []const u8) !T5Tokenizer;
    pub fn deinit(self: *T5Tokenizer) void;
    /// Unigram encode + EOS(1), then pad with 0 to exactly `max_len`. Truncates
    /// at `max_len - 1` so the EOS always survives.
    pub fn encodePadded(self: *T5Tokenizer, a: std.mem.Allocator, text: []const u8, max_len: usize) ![]i32;
};

// Lane B
pub const DEFAULT_DTYPE: mlx.mlx_dtype = .float32; // force_upcast
pub const Vae = struct {
    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, dtype: mlx.mlx_dtype) !Vae;
    pub fn deinit(self: *Vae) void;
    /// latent [1, 16, h, w] (already `z/scaling + shift`ed by the CALLER via
    /// sd3.decodeScale) -> pixels [1, 3, h*8, w*8] in [-1, 1].
    pub fn decode(self: *Vae, z: mlx.mlx_array, s: S) !mlx.mlx_array;
};
pub const Encoder = struct { // img2img; optional, load may fail softly
    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, dtype: mlx.mlx_dtype) !Encoder;
    pub fn deinit(self: *Encoder) void;
    /// pixels [1,3,H,W] in [-1,1] -> latent MEAN [1,16,H/8,W/8], UNSCALED.
    pub fn encodeMean(self: *Encoder, x: mlx.mlx_array, s: S) !mlx.mlx_array;
};

// Lane C
pub const Mmdit = struct {
    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, dtype: mlx.mlx_dtype) !*Mmdit;
    pub fn deinit(self: *Mmdit) void;
    pub fn config(self: *const Mmdit) sd3.MmditConfig;
    /// latent [B,16,h,w], encoder_hidden [B,T,4096], pooled [B,2048],
    /// timestep = sigma*1000 (sd3.timestepForSigma). Returns velocity [B,16,h,w].
    /// B is 1 or 2 — the caller batches CFG, so nothing here loops over guidance.
    pub fn forward(self: *Mmdit, latent: mlx.mlx_array, enc: mlx.mlx_array,
                   pooled: mlx.mlx_array, timestep: f32, s: S) !mlx.mlx_array;
};
```

`sd3.zig` (lane D, already written) owns the schedule, `MmditConfig`,
`VaeConfig`, the pos-embed crop rule and the latent scale/shift. Read constants
from there rather than restating them.

## Bars

- TDD: failing test first, tests at the BOTTOM of each source file.
- Every lane ships `tests/dump_*_fixtures.py` that runs the **reference class**
  (diffusers / transformers) and saves INPUTS + outputs to a safetensors file.
  A venv with torch 2.14 / diffusers 0.40 / transformers 5.16 is at
  `/tmp/claude-501/sd3venv/bin/python`.
- Fixtures are built from a **tiny random-weight model of the real class** (the
  `tests/dump_qwen4_exp_fixtures.py build` precedent), so the architecture is
  pinned without a 20 GB download. Add a `--model <dir>` mode for real-checkpoint
  parity the user can run later.
- The Zig parity test is **env-gated** on the fixture path and skips when unset,
  exactly like the existing `sdxl unet fixture` tests.
- Reference dumps run on **CPU in float32**. A reference computed less precisely
  than the thing it checks is not a reference.
- Build with `PATH=/Users/justinluque/Developer/mlx-serve/.zig-toolchain:$PATH`
  and `zig build test -Doptimize=ReleaseFast -Dtest-filter="<yours>"`.
  If `lib/ds4` etc. are empty dirs, run `/tmp/claude-501/libs.sh link` first.
  Your module is not in `src/tests.zig` yet — the pipeline lane adds it; until
  then test via a temporary `_ = @import(...)` you REVERT before finishing.
