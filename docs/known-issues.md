# Known issues

What is open today: undeclared exactness contracts, bugs found and not yet fixed, and gaps a reader would otherwise
rediscover. A fixed item leaves this file in the commit that fixes it; the story goes to `docs/gotchas/` if it is a
class. Plugin-specific debts also sit in each `src/plugins/<name>/AGENTS.md` under "Known issues".

## Exactness contracts

A plugin declares what its served output promises in `Caps` (`src/abi/plugin.zig`), checked by
`src/abi/exactness_test.zig`:

- `exact_drafted`: speculative decode (the engine's spec mode) gives the same bytes as `Request.no_drafts`.
- `exact_resumed`: the same prompt served cold (`CacheHint.cold`) and restored from its own cache entry (RAM, then
  SSD across a reopen) gives the same bytes.
- `exact_concurrent`: 2..8 streams batched give each stream's solo bytes.

A longer prompt over a shared cached prefix is in no contract on any arch: chunk width changes GEMM shapes (AGENTS.md
"INT4 long-greedy divergence is legit", "A prefix-cache HIT is not bit-identical on a HYBRID").

Declared today:

| Plugin | Drafted | Resumed | Conditions |
|---|---|---|---|
| `qwen3_5` (also parses `qwen3_next` and Hadamard packs) | dense affine packs only | yes | drafted: a DFlash drafter bound, every trunk projection inside the row kernels (`src/sdk/bases/blocks/row_exact.zig`), not MoE, not `qwen3_next`, not Hadamard, not bf16 |
| `nemotron_h` | yes | yes | drafted: as `qwen3_5`; a pack outside the row kernels serves plain kernels with the flag set and is not exact (`src/plugins/nemotron_h/AGENTS.md`) |
| `gemma4` | no | yes | resumed: greedy, through an SSD restart |
| `lfm2` | no | yes | resumed: text prompts |

`exact_concurrent` is declared by no plugin. Everything else is undeclared:

- **Drafted, a known divergence** (verify rows run qmm or sorted MoE kernels where serial runs qmv, so near-tie
  argmaxes
  flip): `qwen3_5` MoE and bf16 packs, `prism_hadamard_qwen35`, `qwen3_next`, `qwen4_exp` (MTP on by default; auto
  depth is not byte-reproducible), `qwen3`, `llama`, `gemma3`, `gemma4` (Gemma drafter, PLD), `laguna`, `flint`,
  `kolibri1`, `deepseek_v4` / `deepseek_v41` (DSpark), `glm5_next`, `mimo_v2`, `lfm2` (DSpark/PLD: greedy equality is
  asserted on an echo prompt only), `bailing_hybrid`, `hy_v3`, `muse_glimmer` (DFlash is byte-identical at 8 bits and
  diverges at 4, `tests/test_dflash.sh` [5]), `spark2_5`, `k2_horizon`, `gpt_oss`.
- **Resumed, unverified** (no known divergence, no guard runs a same-prompt resume): `prism_hadamard_qwen35`,
  `qwen4_exp` (QSA/PLE state; `tests/test_prefix_inherit_qsa.sh` compares only the first greedy tokens), `qwen3`,
  `qwen3_asr`, `llama`, `gemma3`, `laguna`, `flint`, `kolibri1`, `deepseek_v41` (module-owned state resumes one
  conversation), `glm5_next`, `mimo_v2`, `bailing_hybrid`, `hy_v3`, `inkling`, `muse_glimmer`, `spark2_5`,
  `k2_horizon`, `gpt_oss`. `deepseek_v4` has no prefix cache.
- **Concurrent**: archs that batch decode (B > 1) are not row-exact across batch widths; serial archs (`qwen3_next`,
  `laguna`, `kolibri1`, `bailing_hybrid`, `hy_v3`, `inkling`, `gpt_oss`, `nemotron_h`, `lfm2`) and the single-flight
  `deepseek_v4`/`deepseek_v41` are equal to solo by construction but no guard checks it.
- **Embedded engines** (`ds4`, `llama`, `ane`): every contract unverified. The adapters ignore `Request.no_drafts` and
  `CacheHint.cold`, and ds4 reports `Stats.drafted = 0`, so a drafted check could not see a draft.
- Embedding, decision and media plugins have no token decode loop; their bar is seeded outputs identical.

## Correctness

- **Special-token injection on GGUF and ds4**: a literal control token in user, system or tool content becomes a real
  special token. Fixed for the MLX chat path; libllama tokenizes the rendered string with `parse_special=true`, ds4
  renders its own prompt, and control tokens inside the `tools` JSON are not marked.
- **Gemma-3 doubled BOS**: `add_bos_token: true` prepends BOS while the template already writes `{{ bos_token }}`; HF
  gives one.
- **Mistral-7B-Instruct-v0.3**: the streamed reply starts with a space the non-streamed one trims (stream and
  non-stream must be the same bytes); a system message plus tools renders Gemma turn markers and runs to `max_tokens`
  (the silent generic fallback template).
- **EmbeddingGemma 1**: no floor under the band + key mask, so a padded batch whose length gap exceeds
  `sliding_window` (512) returns NaN for the short row. EmbeddingGemma 2 floors at -1e9.
- **GLM-5.3 advertises video** for packs with `video_token_id` beside `vision_config`, and the engine never reads it
  for GLM: `/v1/models` claims a capability it cannot serve.
- **`hunyuan3d` `steps: 1`** passes the request check and fails in `buildSigmas` (needs 2): a 500 instead of a named
  400.
- **A bare `Infinity` in config.json** (`mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits`, `time_step_limit`) is a JSON
  syntax error, so the pack is refused; sanitize bare `Infinity`/`NaN` at the config read.
- **`gpt_oss` YaRN `truncate: false`** is ignored; check against mlx-lm.
- **Media image backends do not poll cancel**: a cancelled image job runs to its end.
- **A loop cut inside a buffered tool call delivers nothing** on a tools request (the cut lands before the call
  resolves).
- **Streaming gate gaps**: a bare Hermes `<function=` opener with no `<tool_call>` can leak through the stream gate,
  and a re-opened `<|channel>thought` inside reasoning is not re-latched.
- **Ollama sink**: SSE keepalive comments are dropped, and an Ollama non-stream reply (and the LAN tunnel) is not
  marked `length_framed`, so its close waits for the peer.
- **K2-Horizon** with a JSON schema keeps thinking off.

## Kernels

Kernel-contract findings (gate output in each plugin's `contract.zig`; kernels unchanged, the findings are input
classes the composed chain treats differently):

- `deepseek_v4` / `deepseek_v41`: a NaN in a quantized group comes out a finite code; a NaN gate sum becomes the clip
  limit; `idx_score`/`idx_tile` score a NaN q·k as 0.
- `deepseek_v41` `engram` launches 1024 threads at 5120-wide rows where M1/M2 compile it for 928, unprobed: decode
  Engram writes fail on M1/M2.
- `ltx` `ltx_na3d_*`: a +Inf score returns that key's value where a float64 softmax gives NaN.
- No registered contract (`src/sdk/testing/kernel_contract/REGISTERING.md`) for the 20 `glm5_next` kernels, the
  absorbed EXL3 kernels (`src/sdk/quant/exl3/`), MiMo's streamer and GLM-5.3-Flash's EXL3 lane
  (`src/sdk/weights/streamed/`) and GLM-5.3's (`src/plugins/glm_moe_dsa/`): about half the kernel files.
- `msv_simd_qmm` 3-row 4-bit launches 256 threads where M1/M2 build 224 at K 4096, unprobed.
- `add_norm` / `gate_norm_mod` looped-width probe hole and the Gemma 4 MoE seam's unprobed 1024-thread launch: confirm
  on M1/M2, then probe per arm and width.
- `msv_qmm_int8_prefill` accepts bf16 but the bf16 build fails when N % 128 == 0 (unreachable today: Hadamard packs
  are f16).
- `msv_nax_attn_256_q8` (`--int8-qk`) fails to compile when exactly one scale input has fewer than 8 elements.
- `mxfp8_qmv` and the NVFP4 kernels read e4m3 `0x7F`/`0xFF` (NaN) as 480; converters never write it.
- `msv_keyed_sample` ignores the nucleus when a logit is NaN and `top_k` is 0.
- The GELU tables (`gelu_erf_tables`, `gelu_tanh_tables` in `src/sdk/blocks/mlp/fused_activations.zig`) and the
  hc_prefill sigmoid table are cached per dtype only: a CPU caller's table would reach GPU kernels (the SwiGLU table
  was fixed per device).
- Kernel-contract gates 1-4 have no CUDA rig, and gate 5 no nvcc register records.

## Memory and loading

- A ModelOpt NVFP4 linear dequantized to dense (a vision tower, GLM `kv_b_proj`) loads as f32 where bf16 is asked:
  MLX's CPU NVFP4 dequantize with a global scale ignores the dtype.
- YuE2 `Kv.init` keeps a second reference to the zero K/V buffers in the caller, so the first in-place write per layer
  cannot reuse them: about one extra KV cache per phase (x2 with CFG).
- MLX's CPU `gather_mm` hangs on f16 inputs: any CPU-stream test over dense f16 MoE experts deadlocks.
- The 3D load gate can admit a third model beside two residents it should have counted.
- The hybrid prefix cache's auto `--prefix-cache-mem` default is still a fixed cap, not sized from the machine (#330).
- A media DiT with a runtime LoRA pays `deltaSum` per adapted linear on FLUX, Krea and LTX (H3 moved to `lora.addTo`).
- `muse_glimmer` loads with a transient peak ~4.8 GB above the pre-refactor engine's (35.8 vs 31.0 GB on the 8-bit
  30B; the steady state is equal).

## Not served yet

- The MiMo FP8 release (refused by name; `src/sdk/quant/fp8_block128.zig` exists but is not a registered format).
- Quant blocks we ignore are not refused by name (bitsandbytes, compressed-tensors FP8, a modelopt_fp4 drafter).
- ds4 reasoning budget on the plain path (refused by name).
- EmbeddingGemma 2 audio; Muse's MoVA; the EigenLabs Nemotron-3.5-Bolt IQ2 pack (`iq2_xxs` is refused).
- Tokenizer parity: NFC normalization of U+2000-range spaces, DeepSeek's zero-width handling and Spark's blank-line
  shapes differ from HF.

## Platform and build

- `zig build ios-lib` / `ios-lib-sim` fail (std `Io.Threaded` NullFile has no `fd`; two modules are not wired into the
  iOS root).
- CUDA: the FLUX fault that made chat-beside-media unsafe has no root cause yet (chat does not interleave with media
  on CUDA).
- `mlx-stream-test` crashed once under heavy shared-GPU load and passed on rerun: on a crash, rerun the step alone.
- Build times miss their targets: a one-line plugin edit rebuilds in ~55 s Debug under `-Dunits` (three units of
  ~40 s in parallel plus an 8 s link; target 30 s), a clean ReleaseFast takes ~3:40 at full parallelism (target 1:30),
  and the host unit is still the largest compile (~10 GB peak). `-Dunits` stays opt-in.
- `clef` and `qwen4_exp` still import engine internals (`zig build unit-closure` lists them as exceptions): an edit
  there rebuilds the host.
