# Laya port — working notes

Extension points in mlx-serve used by the `laya` model_type (branch `laya`).

## How a model dir becomes a served model

- `src/model_discovery.zig` `peekConfig` reads `<dir>/config.json` `model_type`
  and classifies it (`supported_model_types`, `isMediaModelType`, fallbacks for
  dirs without config.json: MageFlow `model_index.json`, mflux FLUX.2).
  `modelKindFromType` maps it to a `ModelKind` (chat/image/audio/video/mesh/
  embed/...) that drives `/v1/models` capability rows and refusal messages.
- `src/gen.zig` `peekModelType` is the ROUTING-side twin (must agree with
  discovery). `modalityFromType` maps `model_type` -> `Modality`; `detectModality`
  is what `main.zig` (`--model-dir` primary load, `runGenServe`) and
  `scheduler.zig` (`preloadCpuState`, `doLoadGenOnInferenceThread`) call to
  decide "media engine, not the MLX transformer".
- `src/model_registry.zig` `LoadedModel` has one engine slot per modality
  (`image_engine`, `audio_engine`, ...). Slots are freed in `deinit` and
  `unloadResident`.
- `src/scheduler.zig` `doLoadGenOnInferenceThread` builds the engine on the
  inference thread (the sole MLX caller) and installs a stub config/tokenizer
  (`gen.buildStubCpuState`).
- `src/server.zig`: `ROUTE_PATHS` (404-vs-503 gate), the dispatch chain in
  `handleConnection`, `handleGen` -> `GenJob` -> `genJobRun` (runs the handler
  on the inference thread via `scheduler.runGeneration`), `ReadyCaps` /
  `readyCapsJson` (+ the stub-caps branch keyed on `arch_hint`) for
  `/v1/models`, `textGenRejectReason` for chat routes hitting a non-LM.

## The BERT embedding path (not reused)

`model_type: "bert"` is parsed into the generic `ModelConfig`
(`model.zig` ~L3461) and served by `transformer.zig` + `handleEmbeddings`.
That path is entangled with the chat/KV machinery; a self-contained module in
the style of `kokoro.zig` (own config, `ltx.loadComponent` safetensors map,
own forward over `mlx.zig` externs) is smaller and mirrors laya_mlx 1:1.

## Reusable pieces

- `ltx_video.loadComponent(allocator, path, cpu_stream)` -> name -> mlx_array
  map (safetensors load must run on the CPU stream).
- `tokenizer.loadTokenizer(io, allocator, dir)` reads `dir/tokenizer.json`;
  `encode` adds no specials for sentencepiece BPE. Laya's tokenizer.json is
  Gemma-style BPE with a `Metaspace` pre_tokenizer (`prepend_scheme: always`,
  `split: true`) which `encodeSentencePiece` did not implement — added, gated
  on the parsed pre_tokenizer type (Gemma 4 uses `Split`, unaffected).
- `mlx.zig` already declares every op needed: matmul, fast_layer_norm,
  fast_rope, fast_scaled_dot_product_attention (mask_mode "array" with a bool
  mask), split, take, erf (exact GELU), softmax.
- `chat.appendJsonString` for response escaping.

## Laya checkpoint layout (aac6fef/laya-multilingual-mlx)

No top-level config.json. `encoder/config.json` (ModernBERT: 22 layers, 768
hidden, 12 heads, GeGLU 1152, local_attention 128, rope theta 160000 for
both layer kinds, layer_types list), `rl_agent_config.json` (head_layers 2,
max_len 1024, head_max_len 256, temperature [1,1,1], act_costs {escalate}),
`tokenizer/tokenizer.json` (+config: cls=<bos>=2, sep=<eos>=1, pad=0,
mask=<mask>=4), `model.safetensors` (170 fp16 tensors, MLX names, e.g.
`encoder.layers.N.attn.Wqkv.weight`, `head.layers.N.self_attn.in_proj.weight`,
`scorer.layers.{0,1,3}`, `act_head.layers.{0,2}`, `type_emb.weight`).
Discovery keys on `rl_agent_config.json` + `encoder/config.json` when there
is no config.json; a config.json with `model_type: "laya"` also works.

## Decisions

- New modality `.decision` in `gen.Modality` (capability "decisions", stub
  model_type "laya") rather than a `ModelKind.embed` sibling: it gets the
  media load/unload/serialization plumbing for free.
- Questions in one request are batched into one forward (padded to the longest
  sequence, chunks of 16) exactly like `laya_mlx.Agent.system_one`.
- The act head runs in MLX too (second tiny graph after the logits are read
  back for the entropy features), so numerics stay fp16 like the reference.
- `model` in the response is the registry id, not the constant "laya-rl-agent".
