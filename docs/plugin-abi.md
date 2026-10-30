# Plugin ABI reference

Generated from `src/abi/*.zig` by `scripts/plugin_abi_doc.py`; do not edit by hand (`tests/test_plugin_abi_doc.sh`
fails when it is stale). The same sources ship in the plugin kit (`src/abi/`), and `include/mlx_serve_plugin.h`
mirrors them for C, C++ and Rust. Guides: [plugins.md](plugins.md), [model](plugin-model.md),
[media](plugin-media.md), [quant](plugin-quant.md).

Every struct opens with `size`: one passed by pointer grows at its end (an older plugin's shorter struct still
reads), one held by value or as an array element is frozen per `version`. A field added later defaults to the
behaviour before it, so a plugin that never sets it is unchanged.

## `plugin.zig`

The model plugin table (`ModelTable`) and every struct that crosses it. Extern layouts, C calling convention, no globals. Every struct opens with `size`: one passed by pointer grows at its end; one held by value or as an array element is frozen per `version` (its `size` is a check). Per-model values ride `Parsed`; hook tables hold functions.

| Name | Value | |
|---|---|---|
| `Span` | `common.Span` |  |
| `Str` | `common.Str` |  |
| `Verdict` | `common.Verdict` |  |
| `Array` | `common.Array` |  |
| `Stream` | `common.Stream` |  |
| `Claim` | `common.Claim` |  |
| `PackDir` | `common.PackDir` |  |
| `version` | `u32 = 1` |  |
| `Handle` | `?*anyopaque` | An object the engine owns and the plugin only passes back (a slot's KV cache): opaque until P3 types it. |
| `max_eos` | `16` |  |
| `max_round_tokens` | `32` | The most tokens a native drafter's round commits (`Round.tokens`). |

### `LayerSpan` (extern struct)

The layers this process serves when a model is split across machines; null = all of them.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(LayerSpan)` |  |
| `first` | `u32` | `0` |  |
| `count` | `u32` | `0` |  |

### `DType` (enum(u8))

| Member | Value | |
|---|---|---|
| `unset` | |  |
| `f16` | |  |
| `bf16` | |  |
| `f32` | |  |

### `Tri` (enum(u8))

| Member | Value | |
|---|---|---|
| `unset` | |  |
| `off` | |  |
| `on` | |  |

### `QuantMode` (enum(u8))

| Member | Value | |
|---|---|---|
| `none` | |  |
| `affine` | |  |
| `nvfp4` | |  |
| `mxfp4` | |  |
| `mxfp8` | |  |
| `gguf` | |  |

### `Quant` (extern struct)

The config's `quantization` block.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Quant)` |  |
| `bits` | `u8` | `0` |  |
| `mode` | `QuantMode` | `.none` |  |
| `group_size` | `u32` | `0` |  |

### `GenDefaults` (extern struct)

Sampling defaults from the pack (`generation_config.json`); NaN or -1 = unset.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(GenDefaults)` |  |
| `temperature` | `f32` | `nan` |  |
| `top_p` | `f32` | `nan` |  |
| `min_p` | `f32` | `nan` |  |
| `top_k` | `i32` | `-1` |  |
| `enable_thinking` | `Tri` | `.unset` |  |

### `Shell` (extern struct)

The generic shell: the only config the engine reads. Per-layer facts are `CacheHooks.layout`, not here.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Shell)` |  |
| `vocab_size` | `u32` | `0` |  |
| `hidden_size` | `u32` | `0` |  |
| `intermediate_size` | `u32` | `0` | Leaves once every plugin bills `Bill.prefill_streams`. |
| `num_hidden_layers` | `u32` | `0` |  |
| `num_attention_heads` | `u32` | `0` |  |
| `num_key_value_heads` | `u32` | `0` |  |
| `head_dim` | `u32` | `0` |  |
| `num_experts_per_tok` | `u32` | `0` | Temporary (decisions.md): both leave once every plugin's prefill bill lands. |
| `moe_intermediate_size` | `u32` | `0` |  |
| `max_position_embeddings` | `u64` | `0` |  |
| `context_cap` | `u64` | `0` | The derived context the model serves. |
| `quant` | `Quant` | `.{}` |  |
| `gen` | `GenDefaults` | `.{}` |  |
| `bos` | `i64` | `-1` | -1 = none |
| `eos` | `[max_eos]u32` | `@splat(0)` |  |
| `eos_count` | `u32` | `0` |  |
| `act_dtype` | `DType` | `.bf16` |  |
| `state_dtype` | `DType` | `.unset` |  |
| `model_type` | `Str` | `.{}` | For display only: the host never compares it (the plugin is the identity). |

### `Caps` (packed struct(u64))

Model facts the host keeps once at load. Defaults are the generic model's.

| Field | Type | Default | |
|---|---|---|---|
| `batches_decode` | `bool` | `false` |  |
| `batch_per_slot_attn` | `bool` | `true` |  |
| `batch_pads_kv` | `bool` | `true` |  |
| `owns_decode_state` | `bool` | `false` | Per-request decode state on the module, not the KV cache: single-flight, prefix cache through restore only. |
| `shared_readonly_state` | `bool` | `false` |  |
| `long_ctx` | `bool` | `false` |  |
| `moe` | `bool` | `false` |  |
| `prefix_cache` | `bool` | `true` |  |
| `exact_drafted` | `bool` | `false` | The exactness contracts (P2 step 7): drafted == plain, restored == cold, batched == solo. |
| `exact_resumed` | `bool` | `false` |  |
| `exact_concurrent` | `bool` | `false` |  |
| `dflash_capture` | `bool` | `false` | The forward fills `Forward.capture_layers`. |
| `dflash_media` | `bool` | `false` |  |
| `spec_rollback` | `bool` | `false` |  |
| `spec_tree` | `bool` | `false` |  |
| `whole_prompt_prefill` | `bool` | `false` |  |
| `ane_prefill` | `bool` | `false` | The forward honours `Forward.offload` and `WeightHooks.layer_weight` lends the offloaded projections. |
| `streamed_weights` | `bool` | `false` |  |
| `embed` | `bool` | `false` |  |
| `images` | `bool` | `false` |  |
| `video` | `bool` | `false` |  |
| `audio` | `bool` | `false` |  |
| `companion` | `bool` | `false` | A pack another pack loads beside it (the ASR forced aligner). |
| `drafter` | `bool` | `false` |  |
| `prefill_mask` | `bool` | `false` | The forward reads `Forward.mask_add`. |
| `cluster_plan` | `bool` | `false` | The model reads a Thunderbolt cluster's plan itself (`distributed.planFor`); a planned pack whose plugin does not is loaded by the legacy path. |
| `dflash_policy` | `bool` | `false` | DFlash rounds follow the round policy (`dflash_policy.zig`): a sparse target whose verify rows cost. |
| `verify_truncates` | `bool` | `false` | A rejected draft rolls back by truncating the KV cache (an attention trunk): no snapshot, no re-forward. |
| `merged_state` | `bool` | `false` | A batch's recurrent state is read and written only through `CacheOps.state_rows` / `set_state_rows`, so the engine may keep it merged across batched ticks instead of re-stacking every layer each tick. |
| `slot_state` | `bool` | `false` | Module-owned state (`owns_decode_state`) kept per slot (`Forward.slots`, `release_slot`, `activate_slot`): requests interleave and batch instead of running single-flight. |
| `_` | `u34` | `0` |  |

### `ThinkingDefault` (enum(u8))

| Member | Value | |
|---|---|---|
| `unset` | |  |
| `off` | |  |
| `on` | |  |
| `on_with_tools` | |  |

### `ReasoningDialect` (enum(u8))

| Member | Value | |
|---|---|---|
| `unset` | |  |
| `think_tags` | |  |
| `gemma_channel` | |  |
| `harmony` | |  |
| `muse` | |  |
| `inkling` | |  |
| `content_thinking` | |  |

### `EffortVocab` (enum(u8))

The effort words a template understands; the tables stay in the host.

| Member | Value | |
|---|---|---|
| `none` | |  |
| `generic` | |  |
| `qwen38` | |  |
| `kolibri` | |  |
| `dsv4` | |  |
| `dsv41` | |  |
| `glm5` | |  |
| `k2` | |  |
| `inkling` | |  |

### `RoundTrip` (enum(u8))

| Member | Value | |
|---|---|---|
| `keep` | |  |
| `drop_prior_turns` | |  |
| `required_field` | |  |

### `ToolDialect` (enum(u8))

The dialect `tool_choice` enforcement commits an opener in; parsing stays dialect-agnostic in the host.

| Member | Value | |
|---|---|---|
| `unset` | |  |
| `qwen_xml` | |  |

### `NoThink` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(NoThink)` |  |
| `template_refuses` | `bool` | `false` |  |
| `tail_suffix` | `Str` | `.{}` |  |

### `SamplingDefaults` (extern struct)

Family sampling defaults when the pack has no generation_config; NaN or -1 = none.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SamplingDefaults)` |  |
| `top_k` | `i32` | `-1` |  |
| `top_p` | `f32` | `nan` |  |
| `wild_temp_override` | `f32` | `nan` |  |

### `ChatFacts` (extern struct)

Chat facts. A fact left `unset`/empty means the host falls back to its template probe.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(ChatFacts)` |  |
| `thinking_default` | `ThinkingDefault` | `.unset` |  |
| `reasoning_dialect` | `ReasoningDialect` | `.unset` |  |
| `effort_vocab` | `EffortVocab` | `.none` |  |
| `reasoning_round_trip` | `RoundTrip` | `.keep` |  |
| `tool_dialect` | `ToolDialect` | `.unset` |  |
| `nothink` | `NoThink` | `.{}` |  |
| `template_family` | `Str` | `.{}` |  |
| `sampling_defaults` | `SamplingDefaults` | `.{}` |  |
| `video_placeholder` | `Str` | `.{}` |  |

### `MarkerAlias` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(MarkerAlias)` |  |
| `from` | `Str` | `.{}` |  |
| `to` | `Str` | `.{}` |  |

### `TokenizerFacts` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(TokenizerFacts)` |  |
| `extra_eos` | `[max_eos]u32` | `@splat(0)` |  |
| `extra_eos_count` | `u32` | `0` |  |
| `defined_vocab` | `u32` | `0` | The vocabulary rows that are real tokens; 0 = `vocab_size`. |
| `marker_aliases` | `Span(MarkerAlias)` | `.{}` |  |

### `SnapshotPolicy` (enum(u8))

How the engine keeps a recurrent layer's state across a pipelined step.

| Member | Value | |
|---|---|---|
| `lagged` | |  |
| `in_place` | |  |

### `SsmState` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SsmState)` |  |
| `entries` | `u32` | `0` | Per-request recurrent entries; 0 = none. |
| `dtype` | `DType` | `.unset` |  |
| `snapshot_policy` | `SnapshotPolicy` | `.lagged` |  |
| `checkpoint_bytes` | `u64` | `0` |  |

### `CacheFacts` (extern struct)

`CacheHooks`' per-model values.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(CacheFacts)` |  |
| `layout_namespace` | `Str` | `.{}` | A fresh SSD-cache root when stored K/V change meaning; empty = the default root. |
| `sliding_ring` | `u32` | `0` | Rows a sliding layer keeps (window + slack); 0 = the full buffer. |
| `ssm_state` | `SsmState` | `.{}` |  |

### `MtpLayout` (enum(u8))

| Member | Value | |
|---|---|---|
| `none` | |  |
| `sidecar` | |  |
| `in_checkpoint` | |  |
| `nemotron` | |  |

### `SpecPolicy` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SpecPolicy)` |  |
| `drafter_ok` | `bool` | `true` |  |
| `mtp_group_solo` | `bool` | `false` |  |
| `seeded_sampling` | `bool` | `false` |  |
| `dflash_block` | `u32` | `0` | 0 = the machine's block. |
| `dflash_floor_x10` | `u32` | `0` | The DFlash yield gate's floor in accepted tokens per round x10; 0 = the calibrated gate. |
| `draft_temperature` | `f32` | `nan` |  |

### `SpecFacts` (extern struct)

`SpecHooks`' per-model values.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SpecFacts)` |  |
| `mtp_default_depth` | `u8` | `0` |  |
| `mtp_layout` | `MtpLayout` | `.none` |  |
| `mtp_heads` | `u32` | `0` | MTP heads, one per draft depth (MiMo: 3), each with its own history and each reading the trunk's hidden rows (`hidden_in` stays the trunk's along the round); 0 or 1 = one head chained on its own output. A round's draft k runs head k (`Forward.head`); a draft past the last head re-runs it and is never accepted. |
| `head_module_owned` | `bool` | `false` |  |
| `head_reads_final_norm` | `bool` | `false` |  |
| `draft_vocab` | `u32` | `0` |  |
| `cost_profile` | `u32` | `0` | An id into the engine's measured round-cost tables; 0 = generic. |
| `native_stages` | `u32` | `0` |  |
| `native_block` | `u32` | `0` |  |
| `policy` | `SpecPolicy` | `.{}` |  |
| `embed_scale` | `f32` | `0` | The trunk multiplies its embedding rows by this before the first layer (Gemma: sqrt(hidden)); 0 = unscaled. A drafter that embeds through the trunk's table (`trunk_weight`) applies it itself. |

### `PlaceholderIds` (extern struct)

Placeholder token ids a media item's rows replace; 0 = unset, never token 0.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(PlaceholderIds)` |  |
| `image` | `u32` | `0` |  |
| `video` | `u32` | `0` |  |
| `audio` | `u32` | `0` |  |
| `boi` | `u32` | `0` |  |
| `eoi` | `u32` | `0` |  |
| `boa` | `u32` | `0` |  |
| `eoa` | `u32` | `0` |  |
| `vision_start` | `u32` | `0` |  |
| `vision_end` | `u32` | `0` |  |
| `speech` | `u32` | `0` | A spoken frame of the model's own (`SpeechFacts.frame_token`): a replayed turn's frame rows splice here. |

### `SpeechFacts` (extern struct)

Speech output: each `frame_token` slot is one frame of `codebooks` codes the model's `SpeechHooks.frame` draws from the trunk's last hidden row; `codebooks` 0 = the model does not speak. A generation starts in text, or in speech after `audio_start`, or interleaved after `mixed_start`: `interleave_text` text slots, then `interleave_audio` frames, until `text_end` ends the block's text; a frame of `eos_code` ends the speech.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SpeechFacts)` |  |
| `codebooks` | `u32` | `0` |  |
| `codebook_size` | `u32` | `0` |  |
| `eos_code` | `u32` | `0` |  |
| `frame_token` | `u32` | `0` |  |
| `audio_start` | `u32` | `0` |  |
| `mixed_start` | `u32` | `0` |  |
| `text_end` | `u32` | `0` |  |
| `interleave_text` | `u32` | `0` |  |
| `interleave_audio` | `u32` | `0` |  |
| `sample_rate` | `u32` | `0` | The codec's output: mono samples per second, samples per frame. |
| `frame_samples` | `u32` | `0` |  |
| `temperature` | `f32` | `1.0` | The speech sampling a request leaves unset (`top_k` 0 = every code); the penalty reads the last `repetition_window` frames. |
| `top_k` | `u32` | `0` |  |
| `repetition_penalty` | `f32` | `1.0` |  |
| `repetition_window` | `u32` | `0` |  |
| `tools` | `bool` | `false` | The model speaks its reply to a request with tools (its card's tools recipe: no prefill, a call or speech). false: such a request answers in text. |

### `MropeFacts` (extern struct)

The model's interleaved M-RoPE spectrum, from which the engine builds a prompt chunk's angles off the prompt's position table (`Forward.mrope`).

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(MropeFacts)` |  |
| `sections` | `[3]u32` | `.{ 0, 0, 0 }` | The t/h/w frequency split; all zero = the model takes no M-RoPE table. |
| `rope_dims` | `u32` | `0` | Rotated dims per head. |
| `inv_freq` | `Span(f64)` | `.{}` | `rope_dims / 2` inverse frequencies (YaRN-scaled when the model scales), plugin-owned until `free_parsed`. |
| `mscale` | `f64` | `1` | YaRN's attention factor on the angles; 1 = none. |

### `Pooling` (enum(u8))

| Member | Value | |
|---|---|---|
| `unset` | |  |
| `mean` | |  |
| `cls` | |  |
| `last_token` | |  |

### `EmbedFacts` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(EmbedFacts)` |  |
| `pooling` | `Pooling` | `.unset` |  |
| `has_projection` | `bool` | `false` |  |

### `WeightFacts` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(WeightFacts)` |  |
| `prefix` | `Str` | `.{}` | The weight-name prefix (probed for packs that differ); empty = none. |
| `quant_layout` | `Str` | `.{}` | The quant plugin that reads this pack's tensors (exl3, modelopt, hadamard, ...); empty = MLX's own. |

### `Parsed` (extern struct)

Every `Str` here points at static text or at memory the plugin owns until `ModelTable.free_parsed`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Parsed)` |  |
| `caps` | `Caps` | `.{}` |  |
| `shell` | `Shell` | `.{}` |  |
| `chat` | `ChatFacts` | `.{}` |  |
| `tokenizer` | `TokenizerFacts` | `.{}` |  |
| `cache` | `CacheFacts` | `.{}` |  |
| `spec` | `SpecFacts` | `.{}` |  |
| `placeholder_ids` | `PlaceholderIds` | `.{}` |  |
| `embed` | `EmbedFacts` | `.{}` |  |
| `weights` | `WeightFacts` | `.{}` |  |
| `mrope` | `MropeFacts` | `.{}` |  |
| `private` | `?*anyopaque` | `null` | The plugin's own parsed config, read by its `bill` and `load`. |
| `speech` | `SpeechFacts` | `.{}` |  |

### `Shape` (extern struct)

The load-time shape `bill` prices.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Shape)` |  |
| `context` | `u64` | `0` |  |
| `slots` | `u32` | `1` |  |
| `kv_bits` | `u8` | `0` | KV precision: 0 = the activation dtype, else 8 or 4. |

### `Bill` (extern struct)

The plugin's memory terms. A zero or null term is the engine's default arithmetic over the shell and layout.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Bill)` |  |
| `kv_per_token` | `u64` | `0` |  |
| `fixed_slot_bytes` | `u64` | `0` |  |
| `state_per_token` | `u64` | `0` |  |
| `prefill_streams` | `u64` | `0` |  |
| `load_bytes` | `u64` | `0` |  |
| `ssm_checkpoint` | `u64` | `0` |  |
| `prefill_score_width` | `u32` | `0` |  |
| `prefill_attn_keys` | `?*const fn (p: *const Parsed, seq: u64) callconv(.c) u64` | `null` |  |
| `prefill_bytes` | `?*const fn (p: *const Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64` | `null` |  |
| `batched_kv_len` | `?*const fn (p: *const Parsed, kv: u32, gather_on: bool, min: u32) callconv(.c) u32` | `null` |  |

### `LoadCtx` (extern struct)

Valid during `load` only: the model keeps none of its pointers.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(LoadCtx)` |  |
| `model_dir` | `Str` |  |  |
| `dir` | `*const PackDir` |  |  |
| `parsed` | `*const Parsed` |  |  |
| `settings_json` | `Str` | `.{}` |  |
| `span` | `?*const LayerSpan` | `null` |  |
| `stream` | `Stream` | `.{}` |  |
| `host` | `?*const Host` | `null` | The process's host table (`host/plugin_host_impl.zig`); null outside a server. |
| `weights` | `?*const Source` | `null` | The pack's tensors as the engine loaded them (`sdk/weights/resident.zig`); null = the plugin reads its own. |
| `stream_room` | `u64` | `0` | GPU bytes a streamed expert bank may fill (with what is resident): the GPU ceiling less one prompt's bill at 16K + 1K and the wired margin; 0 = none. |
| `pair_remote` | `LayerSpan` | `.{}` | A two-Mac pair's layers on the other Mac (DeepSeek-V4.1's worker run); `count` 0 = no pair. |
| `wired_margin_bytes` | `u64` | `8 << 30` | The host's wired-limit margin (`--wired-margin-gib`): the GPU ceiling's stop a streamed bank leaves unplanned. |

### `Mode` (enum(u8))

| Member | Value | |
|---|---|---|
| `prefill` | |  |
| `decode` | |  |
| `verify` | |  |
| `batched_rows` | |  |

### `Logits` (enum(u8))

| Member | Value | |
|---|---|---|
| `last_row` | |  |
| `all_rows` | |  |
| `none` | |  |

### `Forward` (extern struct)

One forward. `tokens` holds `rows` tokens per slot, slot after slot; `slots`, `positions` and `caches` hold one entry per slot (its id, its first position, its KV cache).

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Forward)` |  |
| `mode` | `Mode` |  |  |
| `logits` | `Logits` | `.last_row` |  |
| `want_hidden` | `bool` | `false` |  |
| `rows` | `u32` |  |  |
| `tokens` | `Span(u32)` |  |  |
| `slots` | `Span(u32)` | `.{}` |  |
| `positions` | `Span(u32)` | `.{}` |  |
| `caches` | `Span(Handle)` | `.{}` |  |
| `capture_layers` | `Span(u32)` | `.{}` | Layers whose hidden rows the forward publishes (DFlash); filled into `ForwardOut.captured`. |
| `media` | `Array` | `.{}` | Media rows `[1, n, hidden]` to splice at the placeholder ids (`Parsed.placeholder_ids`) in token order, from row `media_offset` on (a chunked prefill resumes there). |
| `media_offset` | `u32` | `0` |  |
| `mrope` | `Array` | `.{}` | A prompt chunk's M-RoPE angles, cos and sin stacked `[2, 1, 1, rows, rope_dims]` (built from `Parsed.mrope`); null = 1-D positions, which then carry the slot's M-RoPE delta. |
| `cache_ops` | `?*const CacheOps` | `null` | How the forward reads and writes `caches`; null = a forward that keeps no state (an encoder). |
| `ids` | `Array` | `.{}` | The token ids as a device array `[slots, rows]` (a still-lazy sampled token); when set, `tokens` is empty. |
| `eval_every` | `u32` | `0` | A wide forward evaluates its residual after every this many layers and after the last (0 = never): the engine's bound on a prompt chunk's transient memory. |
| `hidden_in` | `Array` | `.{}` | An MTP head's input: the trunk's hidden rows `[1, rows, hidden]` beside its token rows. |
| `defer_host` | `bool` | `false` | The engine reads nothing off the result before the next `forward_settle`: a host-side input the forward needs (an n-gram lookup of a still-lazy token) may be deferred to it rather than synced inside the forward. |
| `key_mask` | `Array` | `.{}` | A padded encoder batch's additive key mask `[slots, rows]` (0 = a real token, a large negative = padding); empty = none. An encoder batch (`cache_ops` null) carries one `positions` entry per sequence. |
| `head` | `u32` | `0` | An MTP head forward's head (`SpecFacts.mtp_heads`): its history is layer `head` of `caches[0]`. A history append (`.none` without `want_hidden`) appends to every head. |
| `row_exact` | `bool` | `false` | A DFlash drafter is bound: projections and attention read the row-exact kernels wherever the model's weights fit them, so a verify row reproduces serial decode bit for bit (`Caps.exact_drafted`). |
| `offload` | `?*const Offload` | `null` | An engine unit computing a prompt chunk's projections beside the GPU (`Caps.ane_prefill`); null = none. |
| `tree` | `Tree` | `.{}` | A verify whose rows form a draft tree (`Caps.spec_tree`); null `table` = a chain. |
| `pipeline_every` | `u32` | `0` | A verify hands the residual built so far to the GPU every this many layers (0 = never), so the GPU runs while the rest of the graph is built. |
| `mask_add` | `Array` | `.{}` | An additive attention mask `[1, 1, rows, rows]` over a prompt forwarded whole from position 0 (causal itself, finite: a left-padded prompt's pads masked), summed into every layer's scores (`Caps.prefill_mask`); empty = none. |
| `accept_cap` | `u32` | `0` | A native drafter's round (`NativeDrafter.step`): the most drafts it may keep, the tokens still allowed less one. |
| `suppress_mask` | `Array` | `.{}` | The sampler's suppress mask `[vocab]` (true = never drawn), which a round's greedy accept applies; empty = none. |
| `embeds` | `Array` | `.{}` | Input rows `[slots, rows, hidden]` in place of the ids' embedding (a spoken frame's `SpeechHooks.frame_rows`); empty = embed the ids. |

### `Offload` (extern struct)

The engine's calls a forward makes at a layer's projections: each returns true with new references in its outputs when it computed them from input `x` (`[1, rows, hidden]`), false to leave them to the model.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Offload)` |  |
| `ctx` | `*anyopaque` |  |  |
| `mlp` | `*const fn (ctx: *anyopaque, layer: u32, x: Array, out: *Array) callconv(.c) bool` |  | The dense MLP's output. |
| `gdn_proj` | `*const fn (ctx: *anyopaque, layer: u32, x: Array, qkv: *Array, z: *Array) callconv(.c) bool` |  | A GatedDeltaNet layer's joined q\|k\|v and z projections. |

### `Tree` (extern struct)

Draft-tree verify rows: row w sits at its slot's position + `depth[w]`, its keys and recurrence follow its ancestors. A recurrent layer leaves the round's input state and per-row prework (`StateSlot.state_in`..) for the engine to replay the accepted path from.

| Field | Type | Default | |
|---|---|---|---|
| `table` | `Array` | `.{}` | `[6 W]` int32: each row's parent (-1 for row 0), then its conv-input sources (`gdn_decode.treeTable`). |
| `depth` | `Array` | `.{}` | `[W]` int32 depths, and `[W, max_depth + 1]` int32: each row's ancestor at every depth. |
| `path` | `Array` | `.{}` |  |
| `max_depth` | `i32` | `0` |  |

### `Owed` (enum(u8))

What a `defer_host` forward left owed (`ForwardOut.owed`), settled by `forward_settle`.

| Member | Value | |
|---|---|---|
| `none` | |  |
| `input` | | A host-filled input of this forward's graph: settled before anything of the forward is evaluated. |
| `history` | | Host bookkeeping off the forward's own tokens only: the engine may settle it later (with the tokens it read). |

### `Settle` (enum(u8))

| Member | Value | |
|---|---|---|
| `flush` | |  |
| `discard` | |  |

### `KvView` (extern struct)

One layer's keys and values as attention reads them: dense `[B, H, T, D]`, owned by the view only when `owned` (`CacheOps.release` frees them), plus the stored quantized triples (borrowed) when the cache keeps them.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(KvView)` |  |
| `k` | `Array` | `.{}` |  |
| `v` | `Array` | `.{}` |  |
| `owned` | `bool` | `false` |  |
| `packed_reads` | `bool` | `false` | The request reads the triples directly where a packed attention kernel serves the shape. |
| `bits` | `u8` | `0` | 0 = dense storage, else the quantized width of the triples below. |
| `group_size` | `u32` | `0` |  |
| `capacity` | `u32` | `0` | Axis-2 capacity of the triples' buffers (the packed kernels stride by it); 0 = unknown. |
| `k_q` | `Array` | `.{}` |  |
| `k_scales` | `Array` | `.{}` |  |
| `k_biases` | `Array` | `.{}` |  |
| `v_q` | `Array` | `.{}` |  |
| `v_scales` | `Array` | `.{}` |  |
| `v_biases` | `Array` | `.{}` |  |

### `StateSlot` (enum(u8))

A recurrent layer's states: the convolution window and the scan state, and what a verify forward leaves for the partial-accept rollback: the conv input behind the window (`conv_input` [B, K-1+S, C]) and the state after every row (`state_seq` [S, B, Hv, Dv, Dk]); a tree verify leaves the round's input state (`state_in`) and its rows' prework (`prework_k`, `prework_v`, `prework_gb`) in place of `state_seq`. Writing a null array clears a slot.

| Member | Value | |
|---|---|---|
| `conv` | |  |
| `ssm` | |  |
| `conv_input` | |  |
| `state_seq` | |  |
| `state_in` | |  |
| `prework_k` | |  |
| `prework_v` | |  |
| `prework_gb` | |  |

### `CacheOps` (extern struct)

The engine's calls on a slot's cache (`Handle`), handed in `Forward.cache_ops`: blocks append and read KV and recurrent state through it and never see the engine's types. Every `Array` passed in stays the caller's.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(CacheOps)` |  |
| `len` | `*const fn (cache: Handle, layer: u32) callconv(.c) u64` |  | Rows `layer` holds, counted from position 0 (a ring counts the rows it dropped); 0 before its first append. |
| `append` | `*const fn (cache: Handle, layer: u32, k: Array, v: Array, keep: u32, s: Stream, out: *KvView) callconv(.c) Verdict` |  | Appends `k`/`v` `[1, H, T, D]` at `layer`, then writes the view attention reads: the newest `keep` rows, 0 = all. |
| `append_ring` | `*const fn (cache: Handle, layer: u32, k: Array, v: Array, keep: u32, window: u32, ring: u32, s: Stream, out: *KvView) callconv(.c) Verdict` |  | `append` on a sliding layer whose buffer holds only its newest `ring` rows (`ring >= window`). |
| `view` | `*const fn (cache: Handle, layer: u32, s: Stream, out: *KvView) callconv(.c) Verdict` |  | The rows `layer` already holds (a layer that reads another layer's KV). |
| `release` | `*const fn (view: *KvView) callconv(.c) void` |  | Frees what `out` of `append`/`view` owns. |
| `append_rows` | `*const fn (caches: [*]const Handle, n: u32, layer: u32, k: Array, v: Array, keep: u32, s: Stream) callconv(.c) Verdict` |  | Appends row `i` of the stacked `k`/`v` `[N, H, T, D]` to `caches[i]` (batched rows). |
| `state` | `*const fn (cache: Handle, layer: u32, slot: StateSlot) callconv(.c) Array` |  | The state, borrowed; a null array before its first write. |
| `set_state` | `*const fn (cache: Handle, layer: u32, slot: StateSlot, value: Array) callconv(.c) Verdict` |  | Replaces the state with a new reference on `value`. |
| `state_rows` | `*const fn (caches: [*]const Handle, n: u32, layer: u32, slot: StateSlot, s: Stream) callconv(.c) Array` |  | Layer `layer`'s `slot` over `caches[0..n]`, stacked on the batch axis (axis 1 for `.state_seq`): a new reference the caller frees; a null array when any cache holds none yet. An engine that keeps a batch's states merged across ticks hands its group's array, with no concat. |
| `set_state_rows` | `*const fn (caches: [*]const Handle, n: u32, layer: u32, slot: StateSlot, value: Array, s: Stream) callconv(.c) Verdict` |  | Replaces layer `layer`'s `slot` of `caches[0..n]` with the rows of `value` (the engine takes its own reference). A merged group keeps `value` whole and splits a row out only when that cache is read alone or leaves the group. |
| `sdk_slots` | `bool` | `false` | Each handle is an SDK `cache_ops.Slot` (`sdk/kv/cache_ops.zig`), which an in-tree model may read directly. A flag, not the table's address: every compile unit holds its own copy of the table. |

### `Round` (extern struct)

What a native drafter's round committed.

| Field | Type | Default | |
|---|---|---|---|
| `n` | `u32` | `0` | Tokens committed, `t1` first: `tokens[0..n]`. |
| `accepted` | `u32` | `0` | Drafts kept of those verified (`n - 1`). |
| `next_token` | `u32` | `0` | The trunk's token after them, the next round's `t1`. |
| `rows` | `u32` | `0` | Verify rows run, `t1` included, for the engine's stats. |
| `position` | `u64` | `0` | The model's position after the round. |
| `tokens` | `[max_round_tokens]u32` | `@splat(0)` |  |

### `ForwardOut` (extern struct)

The forward's results: every array is a new reference the host owns. `logits` is [R, vocab] with R the number of slots (`.last_row`) or of tokens (`.all_rows`).

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(ForwardOut)` |  |
| `logits` | `Array` | `.{}` |  |
| `hidden` | `Array` | `.{}` |  |
| `captured` | `?[*]Array` | `null` | Host-allocated, `capture_layers.len` entries. |
| `owed` | `Owed` | `.none` | What each slot of a `defer_host` forward left owed (one value for every slot of the forward). |
| `round` | `Round` | `.{}` | `NativeDrafter.step`'s result. |

### `LayerKind` (enum(u8))

| Member | Value | |
|---|---|---|
| `none` | |  |
| `full` | |  |
| `sliding` | |  |
| `linear` | |  |
| `conv` | |  |
| `ssm` | |  |

### `LayerCache` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(LayerCache)` |  |
| `kind` | `LayerKind` | `.full` |  |
| `nope` | `bool` | `false` | No rotary position on this layer. |
| `kv_source` | `i16` | `-1` | The layer whose KV this one reads; -1 = its own. |
| `window` | `u32` | `0` |  |
| `kv_heads` | `u32` | `0` |  |
| `k_dim` | `u32` | `0` |  |
| `v_dim` | `u32` | `0` |  |

### `AuxSnapshotInfo` (extern struct)

Snapshot metadata; `ids` is borrowed until the snapshot's `free`, and `n == ids.len`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(AuxSnapshotInfo)` |  |
| `n` | `u64` | `0` |  |
| `ids` | `Span(u32)` | `.{}` |  |
| `prefix_key` | `u64` | `0` |  |

### `AuxTensor` (extern struct)

A named array reference. Export transfers the reference to the caller; import borrows it and the name. As an array element, this layout is frozen for ABI version 1.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(AuxTensor)` |  |
| `name` | `Str` | `.{}` |  |
| `array` | `Array` | `.{}` |  |

### `AuxExport` (extern struct)

Caller-owned bounded storage. Names and header bytes live in the supplied buffers; each returned tensor owns a new array reference. Failure releases partial references and leaves the output lengths zero.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(AuxExport)` |  |
| `tensors` | `?[*]AuxTensor` | `null` |  |
| `tensor_cap` | `usize` | `0` |  |
| `tensor_len` | `usize` | `0` |  |
| `names` | `?[*]u8` | `null` |  |
| `names_cap` | `usize` | `0` |  |
| `names_len` | `usize` | `0` |  |
| `header` | `?[*]u8` | `null` |  |
| `header_cap` | `usize` | `0` |  |
| `header_len` | `usize` | `0` |  |

### `AuxHooks` (extern struct)

A plugin's opaque per-slot state beside the KV cache, under the RAM prefix cache, the SSD tier and restores. Every call runs on the inference thread, except `free` at process teardown, which runs on the thread tearing the model down after the inference thread has exited. A snapshot's caller owns it until `free`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(AuxHooks)` |  |
| `reserve` | `*const fn (m: *anyopaque, slot: u32, tokens: u64) callconv(.c) Verdict` |  |  |
| `commit` | `*const fn (m: *anyopaque, slot: u32, len: u64) callconv(.c) Verdict` |  |  |
| `snapshot` | `*const fn (m: *anyopaque, slot: u32, out: *Handle) callconv(.c) Verdict` |  | Transfers the slot's completed capture to the caller; null means no checkpoint was reached. |
| `restore_at` | `*const fn (m: *anyopaque, slot: u32, snap: Handle, pos: u64) callconv(.c) Verdict` |  | Accepts only `pos == snap.n`; builds the replacement before installing it. The caller keeps the snapshot. |
| `slice` | `*const fn (m: *anyopaque, snap: Handle, len: u64, out: *Handle) callconv(.c) Verdict` |  | Clones references when `len == snap.n`; any other length returns named `.unsupported`. |
| `handoff` | `*const fn (m: *anyopaque, from_slot: u32, to_slot: u32) callconv(.c) Verdict` |  |  |
| `free` | `*const fn (m: *anyopaque, snap: Handle) callconv(.c) void` |  | Releases the complete snapshot and its host allocations exactly once. |
| `bytes` | `*const fn (m: *anyopaque, snap: Handle) callconv(.c) u64` |  | Resident bytes, including retained buffer capacities; encoded size is a separate quantity. |
| `enabled` | `*const fn (m: *anyopaque) callconv(.c) bool` |  | Whether this loaded instance supports auxiliary caching, including its runtime configuration. |
| `describe` | `*const fn (m: *anyopaque, snap: Handle, out: *AuxSnapshotInfo) callconv(.c) Verdict` |  | Describes the snapshot's own boundary and identity; `ids` remains borrowed until its `free`. |
| `prefix_key` | `*const fn (m: *anyopaque, ids: Span(u32), media_rows: Array, out: *u64) callconv(.c) Verdict` |  | Computes identity for `ids` and only the media rows belonging to that prefix. |
| `begin_prompt` | `*const fn (m: *anyopaque, slot: u32, prompt: Span(u32), media_rows: Array, restored_at: u64, capture: bool) callconv(.c) Verdict` |  | Activates `slot`; zero `restored_at` resets it, otherwise verifies its position. Arms that slot's capture at `prompt.len - 1` only when `capture` is true; media rows are borrowed for this call. |
| `import_plan` | `*const fn (m: *anyopaque, header: Str, out_peak_bytes: *u64) callconv(.c) Verdict` |  | Allocation-free validation of the complete header/inventory against the loaded instance. Reports the additional simultaneous load/construction/restore peak, including MLX copy scratch. Disk hooks may return named `.unsupported` while the instance retains its RAM snapshot capability. |
| `export_arrays` | `*const fn (m: *anyopaque, snap: Handle, out: *AuxExport) callconv(.c) Verdict` |  | Exports used rows, not spare capacity, into caller-owned storage; insufficient capacity is named. |
| `import_arrays` | `*const fn (m: *anyopaque, ids: Span(u32), header: Str, arrays: Span(AuxTensor), out: *Handle) callconv(.c) Verdict` |  | Borrows all inputs; validates every name/shape/dtype before allocating destination state. Success returns an evaluated, independent snapshot with growth headroom; failure frees partial state and leaves out null. |

### `CacheHooks` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(CacheHooks)` |  |
| `layout` | `?*const fn (p: *const Parsed, span: ?*const LayerSpan, out: [*]LayerCache, cap: u32) callconv(.c) u32` | `null` | Writes one entry per served layer into `out` (`cap` entries) and returns the count; null = every layer full attention at the shell's geometry. |
| `aux` | `?*const AuxHooks` | `null` |  |
| `resume_prompt` | `?*const fn (m: *anyopaque, slot: u32, prompt: Span(u32), media_rows: Array) callconv(.c) u64` | `null` | Module-owned conversation state (`Caps.owns_decode_state`): called before slot `slot`'s prefill with the whole prompt and its media rows (empty = none); returns the prompt tokens the slot's state already holds, from which the engine prefills and which it reports as cached. 0 = none held: the plugin has reset that state for a cold prefill. |
| `state_bytes` | `?*const fn (m: *anyopaque) callconv(.c) u64` | `null` | Bytes that module-owned state holds now (the cache budget's resident figures). |

### `HeadHooks` (extern struct)

An MTP head's calls. `forward` takes the head's token rows (`ids` or `tokens`), the trunk's hidden rows in `hidden_in`, the head's own cache in `caches[0]` (through `cache_ops`) at `positions[0]`, and writes the head's normed output for the LAST row (`[1, 1, hidden]`, the vector the trunk's head reads) to `out.hidden`; the engine projects it. `logits = .last_row`: the engine reads that row (leading rows may append K/V only); `.none` with `want_hidden`: every row runs, the last one's output is read; `.none` without it: history only (K/V, no output).

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(HeadHooks)` |  |
| `load` | `*const fn (m: *anyopaque, ctx: *const LoadCtx) callconv(.c) Verdict` |  |  |
| `activate` | `*const fn (m: *anyopaque, slot: u32) callconv(.c) Verdict` |  |  |
| `release` | `*const fn (m: *anyopaque, slot: u32) callconv(.c) void` |  |  |
| `forward` | `*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict` |  |  |
| `truncate` | `*const fn (m: *anyopaque, slot: u32, len: u64) callconv(.c) void` |  |  |
| `adopt` | `*const fn (m: *anyopaque, slot: u32, snap: Handle) callconv(.c) Verdict` |  |  |
| `mark` | `*const fn (m: *anyopaque, slot: u32) callconv(.c) void` |  |  |
| `draft_select` | `*const fn (m: *anyopaque, logits: Array, out: *Array) callconv(.c) Verdict` |  |  |
| `shortlist` | `?*const fn (m: *anyopaque, hidden: Array, k: u32, out: *Array) callconv(.c) Verdict` | `null` |  |
| `rerank` | `?*const fn (m: *anyopaque, hidden: Array, shortlist: Array, out: *Array) callconv(.c) Verdict` | `null` |  |
| `bill` | `*const fn (p: *const Parsed) callconv(.c) u64` |  |  |
| `forward_lanes` | `?*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict` | `null` | One draft step for several requests as the rows of one head forward: row k of `ids` `[n, 1]` and of `hidden_in` `[1, n, hidden]` runs at `positions[k]` over `caches[k]`; `out.hidden` = the normed rows `[1, n, hidden]`. Null = the engine drafts each request alone. |
| `cost_fingerprint` | `?*const fn (m: *anyopaque) callconv(.c) u32` | `null` | The head's half of the engine's measured cost fingerprint: `bits << 16 \| group` when the loaded head is the measured dense layout (bf16 fc `[2H, H]` and norms, every projection affine at that one width); 0 = none. |

### `NativeDrafter` (extern struct)

A drafter the model carries in its own forward (DSpark); stages and block are `SpecFacts`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(NativeDrafter)` |  |
| `step` | `*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict` |  | One greedy round on the model's own state: `f.tokens` is the committed token `t1` (not yet in the state, which stands at `f.positions[0]`), `f.accept_cap` the most drafts to keep. The model drafts, verifies, keeps the drafts its own argmax agrees with (under `f.suppress_mask`), rewinds its state to them and writes `out.round`. |
| `verify` | `?*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict` | `null` | The first half of a round the engine accepts itself (a sampled request): the model drafts from `t1` and verifies the block, leaving the verify open. A `verify` while one is open discards the open one. `out.round.tokens[0..n]` are the verified rows' ids (`t1` first) and `out.logits` their logits `[n, vocab]` (f32). Needs `finish`. |
| `finish` | `?*const fn (m: *anyopaque, accepted: u32, out: *ForwardOut) callconv(.c) Verdict` | `null` | The second half: keeps `t1` and the first `accepted` drafts of the open verify, rewinds the state to them and writes the model's position into `out.round.position`. |

### `TrunkWeight` (enum(u8))

A trunk tensor a drafter reads beside the trunk's forward.

| Member | Value | |
|---|---|---|
| `embed` | |  |
| `head` | |  |

### `SpecHooks` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SpecHooks)` |  |
| `head` | `?*const HeadHooks` | `null` |  |
| `trunk_weight` | `?*const fn (m: *anyopaque, which: TrunkWeight, out: *quant.Weight) callconv(.c) Verdict` | `null` | The trunk's embedding table or LM head as loaded (borrowed until `unload`; a tied head is the table), for drafters that read them (DFlash, the Gemma assistant, MTP heads). Null = the model serves no drafter. |
| `native_drafter` | `?*const NativeDrafter` | `null` |  |
| `warmup_shapes` | `?*const fn (p: *const Parsed, out: [*]u32, cap: u32) callconv(.c) u32` | `null` | Writes the verify widths to warm into `out` (`cap` entries); returns the count. |
| `tile_lanes` | `?*const fn (m: *anyopaque, lane: u32) callconv(.c) Verdict` | `null` | Once a drafter binds, re-orders the trunk's projections in place for row-exact lane `lane` (the bases' `row_exact.Lane`), the GPU synchronized; the model releases them at `unload`. |

### `MediaKind` (enum(u8))

| Member | Value | |
|---|---|---|
| `image` | |  |
| `video` | |  |
| `audio` | |  |

### `MediaInput` (extern struct)

One request's media item as the host decoded it: RGB8 frames (an image is one frame) or mono f32 samples.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(MediaInput)` |  |
| `kind` | `MediaKind` | `.image` |  |
| `rgb` | `Span(u8)` | `.{}` | Packed RGB8, `frames` frames of `width x height`, frame after frame. |
| `width` | `u32` | `0` |  |
| `height` | `u32` | `0` |  |
| `frames` | `u32` | `0` |  |
| `fps` | `f32` | `0` | The rate a video's frames were sampled at; 0 = the client did not say. |
| `samples` | `Span(f32)` | `.{}` |  |
| `sample_rate` | `u32` | `0` |  |

### `MediaItem` (extern struct)

One prepared media item: its pixels (or samples) as a new reference the host owns, and its row plan.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(MediaItem)` |  |
| `kind` | `MediaKind` | `.image` |  |
| `rows` | `u32` | `0` | Rows the item adds to the prompt. |
| `grid` | `[3]u32` | `.{ 0, 0, 0 }` | Grid in patches: time, height, width. |
| `tiles` | `u32` | `1` | Encoder calls the item takes (crops plus a thumbnail); 1 = a single pass. |
| `pixels` | `Array` | `.{}` |  |
| `bytes` | `Span(u8)` | `.{}` | The pixels as host memory instead of `pixels`: float32 little-endian, `grid[1] * grid[2]` rows of equal width. Plugin-owned until the host has copied it and called `free`. `prepare` runs on a connection thread, which never calls MLX, so an outside plugin returns its pixels this way. |
| `free` | `?*const fn (item: *MediaItem) callconv(.c) void` | `null` |  |
| `fps` | `f32` | `0` | A video's frame rate as prepared and its frames before any padding (its run's timestamps); 0 elsewhere. |
| `frames` | `u32` | `0` |  |

### `TextTokens` (extern struct)

The model's tokenizer, for text inside a placeholder run (a video's timestamps): writes `text`'s ids into `out` (`cap` entries) and returns their count.

| Field | Type | Default | |
|---|---|---|---|
| `ctx` | `?*anyopaque` |  |  |
| `encode` | `*const fn (ctx: ?*anyopaque, text: Str, out: [*]u32, cap: u32) callconv(.c) u32` |  |  |

### `MediaHooks` (extern struct)

Every hook runs on the inference thread, `tower` in the same encode as the `prepare` that made its item.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(MediaHooks)` |  |
| `tower` | `*const fn (m: *anyopaque, item: *const MediaItem, out: *Array) callconv(.c) Verdict` |  | The tower forward: the item's rows `[rows, hidden]`. |
| `prepare` | `*const fn (p: *const Parsed, in: *const MediaInput, budget: u64, out: *MediaItem) callconv(.c) Verdict` |  | Resizes and lays out one decoded input as the model's processor does, within `budget` pixels a frame. |
| `placeholder_rows` | `?*const fn (p: *const Parsed, item: *const MediaItem, tok: *const TextTokens, out: [*]u32, cap: u32) callconv(.c) u32` | `null` | Writes the token run replacing the item's one placeholder into `out` (`cap` entries); returns its length, 0 when it does not fit. Null: `rows` of the kind's placeholder id. |
| `positions` | `?*const fn (p: *const Parsed, ids: Span(u32), items: Span(MediaItem), out: *Array) callconv(.c) Verdict` | `null` | The M-RoPE table over the expanded prompt `ids`. |
| `video_prune` | `?*const fn (m: *anyopaque, item: *const MediaItem, rows: Array, out: *Array) callconv(.c) Verdict` | `null` |  |

### `SpeechSampling` (extern struct)

How one frame samples (`SpeechFacts` holds the defaults). `order` empty = MaskGIT reveals codebooks in a random order drawn from `key`; else the codebook permutation it reveals in.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SpeechSampling)` |  |
| `temperature` | `f32` | `1.0` |  |
| `top_k` | `u32` | `0` |  |
| `repetition_penalty` | `f32` | `1.0` |  |
| `key` | `u64` | `0` |  |
| `order` | `Span(u32)` | `.{}` |  |

### `SpeechHooks` (extern struct)

Speech output (`Parsed.speech`); every hook runs on the inference thread.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(SpeechHooks)` |  |
| `frame` | `*const fn (m: *anyopaque, hidden: Array, how: *const SpeechSampling, history: Span(u32), out: [*]u32) callconv(.c) Verdict` |  | One frame's `codebooks` codes into `out` from the trunk's final-normed last row `hidden` `[1, hidden]`; `history` = earlier frames' codes, frame after frame, for the penalty. A frame holding `eos_code` is all of it. |
| `frame_rows` | `*const fn (m: *anyopaque, codes: Span(u32), out: *Array) callconv(.c) Verdict` |  | The input rows `[1, frames, hidden]` of whole frames of `codes` (`Forward.embeds`, a replayed turn's media). |
| `codec_open` | `*const fn (m: *anyopaque, out: *Handle) callconv(.c) Verdict` |  | A streaming decode of frames into samples: `push` appends whole frames and writes the samples now final into `out` (`cap` at least `frame_samples` per frame pushed, plus one frame), `finish` true also the tail. |
| `codec_push` | `*const fn (m: *anyopaque, h: Handle, codes: Span(u32), finish: bool, out: [*]f32, cap: usize, len: *usize) callconv(.c) Verdict` |  |  |
| `codec_close` | `*const fn (m: *anyopaque, h: Handle) callconv(.c) void` |  |  |

### `EmbedHooks` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(EmbedHooks)` |  |
| `project` | `?*const fn (m: *anyopaque, pooled: Array, out: *Array) callconv(.c) Verdict` | `null` | The projection head over pooled rows; pooling itself is `EmbedFacts.pooling`. |

### `DistHooks` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(DistHooks)` |  |
| `plan` | `*const fn (p: *const Parsed, hosts: u32, out: [*]LayerSpan, cap: u32) callconv(.c) u32` |  | Splits the model across `hosts` machines: one span per host into `out`; returns the count. |
| `serve` | `*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict` |  | This machine's part of a forward. |

### `LayerWeight` (enum(u8))

A layer projection the engine reads beside the forward (`WeightHooks.layer_weight`).

| Member | Value | |
|---|---|---|
| `mlp_gate` | |  |
| `mlp_up` | |  |
| `mlp_down` | |  |
| `gdn_qkv` | |  |
| `gdn_z` | |  |

### `WeightHooks` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(WeightHooks)` |  |
| `transform` | `?*const fn (m: *anyopaque) callconv(.c) Verdict` | `null` | After the weights load: folds, narrows, residents. |
| `keep_key` | `?*const fn (p: *const Parsed, key: Str) callconv(.c) bool` | `null` | Whether the loader keeps tensor `key`; null = all. |
| `layer_weight` | `?*const fn (m: *anyopaque, layer: u32, which: LayerWeight, out: *quant.Weight) callconv(.c) Verdict` | `null` | Layer `layer`'s projection `which` as loaded (borrowed until `unload`); refused where the layer has none. |
| `resident_bytes` | `?*const fn (p: *const Parsed, model_dir: Str, settings_json: Str, shards: u64, out: *u64) callconv(.c) Verdict` | `null` | The bytes a load of the pack at `model_dir` holds resident, from `shards` (the safetensors bytes its index names) and the load's `settings_json` (`LoadCtx.settings_json`); null = `shards`. |

### `ModelTable` (extern struct)

A model plugin's table, exported as `exportName(name, version)`. `claims` and `parse` never reach MLX.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(ModelTable)` |  |
| `version` | `u32` | `version` |  |
| `name` | `Str` |  |  |
| `claims` | `*const fn (model_type: Str, config_json: Str, dir: *const PackDir) callconv(.c) Claim` |  |  |
| `parse` | `*const fn (dir: *const PackDir, config_json: Str, settings_json: Str, span: ?*const LayerSpan, stub: bool, out: *Parsed) callconv(.c) Verdict` |  | `stub`: discovery's parse (no sidecar beyond what the shell needs). A refusal leaves nothing to free. |
| `free_parsed` | `*const fn (p: *Parsed) callconv(.c) void` |  | Frees what `parse` allocated (`Parsed.private` and the text it points at). |
| `bill` | `*const fn (p: *const Parsed, shape: *const Shape, out: *Bill) callconv(.c) void` |  |  |
| `load` | `*const fn (ctx: *const LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque` |  |  |
| `unload` | `*const fn (m: *anyopaque) callconv(.c) void` |  |  |
| `forward` | `*const fn (m: *anyopaque, f: *const Forward, out: *ForwardOut) callconv(.c) Verdict` |  | A mode the model does not declare is refused `unsupported`. |
| `forward_settle` | `?*const fn (m: *anyopaque, slots: Span(u32), how: Settle, tokens: *const Span(u32)) callconv(.c) Verdict` | `null` | Settles what `slots` were left owed: `.flush` reads the tokens off the device once for every slot, or takes `tokens` (one slot's rows, already read); `.discard` drops it (the forward is abandoned). `tokens` by pointer: a trailing by-value span past the argument registers miscompiles on x86_64 (LLVM byval). |
| `release_slot` | `?*const fn (m: *anyopaque, slot: u32) callconv(.c) void` | `null` | A slot id the engine handed in `Forward.slots` is gone: per-slot state the plugin keeps for it is freed. |
| `activate_slot` | `?*const fn (m: *anyopaque, slot: u32) callconv(.c) Verdict` | `null` | Engine code is about to drive the model's module directly for `slot` (a native draft round, `Caps.slot_state`): its state becomes the module's current one. |
| `cache` | `*const CacheHooks` |  |  |
| `spec` | `?*const SpecHooks` | `null` |  |
| `media` | `?*const MediaHooks` | `null` |  |
| `embed` | `?*const EmbedHooks` | `null` |  |
| `dist` | `?*const DistHooks` | `null` |  |
| `weights` | `*const WeightHooks` |  |  |
| `meters` | `?*const fn (m: *anyopaque, slot: u32, out: *Str) callconv(.c) void` | `null` | `slot`'s arm counters for its request's summary lines (each a full log line), in memory the plugin owns until the next call on `m`; the counters reset. |
| `speech` | `?*const SpeechHooks` | `null` |  |

### `declaresMode(caps: Caps, mode: Mode) bool`

The modes a model serves: prefill, decode and verify always; the rest as its caps declare. An encoder (`embed`) serves prefill only, its result the hidden rows.

## `media.zig`

The media plugin table (`MediaTable`): image, audio, video and mesh backends, and the per-request decision and diarization jobs. Same rules as `plugin.zig`: extern layouts, C calling convention, `size` first, no globals.

| Name | Value | |
|---|---|---|
| `Span` | `common.Span` |  |
| `Str` | `common.Str` |  |
| `Verdict` | `common.Verdict` |  |
| `Stream` | `common.Stream` |  |
| `Claim` | `common.Claim` |  |
| `PackDir` | `common.PackDir` |  |
| `version` | `u32 = 1` |  |

### `Modality` (enum(u8))

| Member | Value | |
|---|---|---|
| `image` | |  |
| `audio` | |  |
| `video` | |  |
| `mesh` | |  |
| `decision` | |  |
| `diarization` | |  |

### `Routes` (extern struct)

The audio routes a pack serves.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Routes)` |  |
| `speech` | `bool` | `false` |  |
| `music` | `bool` | `false` |  |
| `sound` | `bool` | `false` |  |

### `MediaCaps` (extern struct)

What one pack supports. Read from the pack (a distilled and a base checkpoint of one family differ), no MLX.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(MediaCaps)` |  |
| `img2img` | `bool` | `false` |  |
| `edit` | `bool` | `false` |  |
| `edit_raw_bytes` | `bool` | `false` | The edit takes the upload's bytes as sent (no decode by the host). |
| `guidance` | `bool` | `false` | Real CFG: `guidance_scale` and `negative_prompt`. |
| `cond_weights` | `bool` | `false` |  |
| `multiview` | `bool` | `false` |  |
| `default_guidance` | `f32` | `0` |  |
| `max_dim` | `u32` | `0` | The largest side in pixels; 0 = no cap. |
| `edit_ref_cap` | `u32` | `0` | Reference images an edit takes; 0 = edits take none. |
| `lora_arch` | `Str` | `.{}` | The LoRA key family the pack's adapters are written for; empty = no LoRA. |
| `routes` | `Routes` | `.{}` |  |

### `Machine` (extern struct)

The machine and launch facts a bill reads (the host fills them; a bill never reaches MLX).

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Machine)` |  |
| `gpu_working_set` | `u64` | `0` | Metal's recommended working set in bytes; 0 = unknown. |
| `w8a8` | `bool` | `false` | `--w8a8`: media DiT linears load as int8 copies that replace the packed weights. |
| `ane_image` | `bool` | `false` | `--ane-image` / `--ane-video` / `--ane-audio`: that modality's DiT stays packed for the ANE seam. |
| `ane_video` | `bool` | `false` |  |
| `ane_audio` | `bool` | `false` |  |
| `low_mem` | `bool` | `false` | iOS, `MLXSERVE_LOWMEM=1`, or RAM at most 17 GiB: stage the large encoders per request. |
| `ram_total` | `u64` | `0` | Host RAM in bytes, total and what the OS will grant now (free + file-backed + purgeable); 0 = unknown. |
| `ram_available` | `u64` | `0` |  |

### `LoadCtx` (extern struct)

Valid during `load` only.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(LoadCtx)` |  |
| `model_dir` | `Str` |  |  |
| `dir` | `*const PackDir` |  |  |
| `settings_json` | `Str` | `.{}` |  |
| `stream` | `Stream` | `.{}` |  |
| `host` | `?*const Host` | `null` | The host table (P2 step 8); opaque until it lands. |
| `machine` | `?*const Machine` | `null` |  |

### `Lora` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Lora)` |  |
| `path` | `Str` |  |  |
| `scale` | `f32` | `1` |  |

### `Message` (extern struct)

Host memory a plugin writes a refusal text into when the text is composed at run time (a limit's number, the field value it names). The host answers with it instead of `Verdict.why` when `len.*` is non-zero.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Message)` |  |
| `ptr` | `?[*]u8` | `null` |  |
| `cap` | `usize` | `0` |  |
| `len` | `?*usize` | `null` |  |

### `Request` (extern struct)

One job. `inputs` by modality, in body order: image = `image` then each `ref_images` entry, mesh = `image` or each `images` entry (base64-decoded; a WebP arrives as a PNG of the same pixels), diarization = the clip as f32 mono PCM at 16 kHz (native byte order); audio, video and decision jobs read their uploads from `body`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Request)` |  |
| `body` | `Str` |  |  |
| `inputs` | `Span(Str)` | `.{}` |  |
| `model_id` | `Str` | `.{}` | The id the request resolved to, for responses that echo it. |
| `message` | `Message` | `.{}` |  |
| `machine` | `?*const Machine` | `null` | The machine facts read when this step began (live RAM included); null outside the host. |
| `dir` | `?*const PackDir` | `null` | The pack, valid for the engine's lifetime (the job reads files by name instead of re-opening the path). |

### `PreviewOpts` (extern struct)

The client's per-step preview request (#208): off unless it asked and the response streams.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(PreviewOpts)` |  |
| `enabled` | `bool` | `false` |  |
| `frames` | `u32` | `1` | Temporal slices in one filmstrip (1..8). |
| `max_side` | `u32` | `256` | Longest side in pixels; 0 = the latent's own size. |

### `PreviewFrame` (extern struct)

One encoded preview, valid during the `preview` call.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(PreviewFrame)` |  |
| `bytes` | `Str` |  |  |
| `mime` | `Str` | `Str.of("image/jpeg")` |  |
| `w` | `u32` |  |  |
| `h` | `u32` |  |  |

### `Progress` (extern struct)

The host's callbacks into a running job, on the GPU thread.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Progress)` |  |
| `ctx` | `?*anyopaque` |  |  |
| `report` | `*const fn (ctx: ?*anyopaque, stage: Str, step: u32, total: u32) callconv(.c) void` |  | A step's progress; an empty `stage` only begins the response (a streamed one reports later failures as events). |
| `boundary` | `*const fn (ctx: ?*anyopaque) callconv(.c) bool` |  | Between steps: the host serves chat for its share of the step, then answers whether the client left (stop). |
| `preview_opts` | `PreviewOpts` | `.{}` |  |
| `preview` | `?*const fn (ctx: ?*anyopaque, stage: Str, step: u32, total: u32, frame: *const PreviewFrame) callconv(.c) void` | `null` | A step's progress with its preview; null (or `preview_opts.enabled` false) = previews are not wanted. |

### `MediaTable` (extern struct)

A media plugin's table, exported as `exportName(name, version)`. `claims`, `caps` and the bills never reach MLX.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(MediaTable)` |  |
| `version` | `u32` | `version` |  |
| `name` | `Str` |  |  |
| `claims` | `*const fn (model_type: Str, dir: *const PackDir) callconv(.c) Claim` |  | `model_type` is empty for a configless pack (classified from its index files). |
| `required_file` | `Str` |  | The file whose presence makes a claimed pack complete; until it lands the pack is invisible. Empty: no marker, the pack is complete as discovered. |
| `modality` | `Modality` |  |  |
| `caps` | `*const fn (dir: *const PackDir, out: *MediaCaps) callconv(.c) void` |  |  |
| `bill_load` | `*const fn (dir: *const PackDir, m: *const Machine) callconv(.c) u64` |  | The load peak in bytes. |
| `bill_request` | `*const fn (dir: *const PackDir, m: *const Machine, body: Str) callconv(.c) u64` |  | One request's whole peak from its body, as the scheduler reserves it (`GenRequest.peak_bytes`); 0 = the load bill covers it. A request past what the GPU can take is the plugin's to refuse by name, inside `step`. |
| `load` | `*const fn (ctx: *const LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque` |  |  |
| `unload` | `*const fn (e: *anyopaque) callconv(.c) void` |  |  |
| `set_loras` | `?*const fn (e: *anyopaque, loras: Span(Lora)) callconv(.c) Verdict` | `null` | Installs the stack for the next job (summed, never merged); an empty span clears it. |
| `step` | `*const fn (e: *anyopaque, req: *const Request, progress: *const Progress) callconv(.c) Verdict` |  | Runs one job to its result, calling `progress` at every step boundary. |
| `encode` | `*const fn (e: *anyopaque, out: *Str) callconv(.c) Verdict` |  | The last job's result, in memory the plugin owns until its next `step` or `unload`: image = PNG, audio = WAV, mesh = GLB, video = the response's JSON members (`"frames":…,"data":"…"`, no braces), decision = the response JSON, diarization = the speaker count (u32 little-endian) then f32 probabilities [frames][speakers]. |
| `step_batch` | `?*const fn (e: *anyopaque, reqs: Span(Request), out: [*]Verdict, results: [*]Str, progress: *const Progress) callconv(.c) void` | `null` | Several queued jobs in one pass (cross-request merging): `out[i]` is job i's verdict, `results[i]` its result (as `encode`'s). Null: the host runs `step` and `encode` per job. |
| `extras` | `?*const fn (e: *anyopaque, out: *Str) callconv(.c) void` | `null` | The last job's extra response members (`,"key":value`, valid as `encode`'s result) that a streamed audio result's `complete` event carries beside the WAV; null or empty = none. |

## `quant.zig`

The quant plugin table (`QuantTable`): a format claims a config's quantization block, validates and bills it, loads its tensors and banks from a weight source (`Source`), runs its matmul and MoE dispatch, registers its kernels. The built-in formats (`src/sdk/quant/`) implement the same table.

| Name | Value | |
|---|---|---|
| `Span` | `common.Span` |  |
| `Str` | `common.Str` |  |
| `Verdict` | `common.Verdict` |  |
| `Array` | `common.Array` |  |
| `Stream` | `common.Stream` |  |
| `Claim` | `common.Claim` |  |
| `PackDir` | `common.PackDir` |  |
| `version` | `u32 = 1` |  |

### `Source` (extern struct)

Where a format reads a pack's tensors: the resident source (`sdk/weights/resident.zig`) or the streamed one.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Source)` |  |
| `ctx` | `?*anyopaque` |  |  |
| `get` | `*const fn (ctx: ?*anyopaque, name: Str) callconv(.c) Array` |  | The tensor `name` (the full key, `<base>.weight`) as the source holds it, borrowed for the source's life; an empty array when the pack has none. |
| `resident` | `bool` | `false` | `ctx` is an SDK `Resident` whose map an in-tree model may take (`Resident.take`). A flag, not `get`'s address: every compile unit holds its own copy of the function. |

### `Weight` (extern struct)

A loaded weight in the format's own layout. It owns every array in it; `QuantTable.release` frees them.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Weight)` |  |
| `w` | `Array` | `.{}` |  |
| `scales` | `Array` | `.{}` |  |
| `biases` | `Array` | `.{}` |  |
| `side` | `Array` | `.{}` | A side tensor the matmul reads (Hadamard: the sign vector). |
| `bits` | `u8` | `0` | Resolved per weight: a tensor may differ from its block's declaration (mixed-precision packs). |
| `group_size` | `u32` | `0` |  |
| `layout` | `u8` | `0` | The format's own layout tag (the MLX formats: dense, an MLX mode, row-scaled e4m3). |
| `flags` | `u8` | `0` | The format's own flags (Hadamard: the biases are exactly -scales). |
| `block` | `u32` | `0` | A width the format reads beside the arrays (Hadamard: the rotation block). |
| `state` | `?*anyopaque` | `null` | The format's own state for this weight (a codebook, a bank's rate groups). |

### `LoadCtx` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(LoadCtx)` |  |
| `model_dir` | `Str` | `.{}` |  |
| `dir` | `?*const PackDir` | `null` |  |
| `block_json` | `Str` |  | The claimed block (`quantization`, `quantization_config`, `expert_quant`) as JSON. |
| `config_json` | `Str` | `.{}` | The pack's whole config.json: a format may read a marker beside its block. |
| `source` | `*const Source` |  |  |
| `stream` | `Stream` | `.{}` |  |
| `host` | `?*const Host` | `null` |  |
| `state` | `?*anyopaque` | `null` | The format's per-load state: set by `open`, read by the loads, freed by `close`. |

### `QuantTable` (extern struct)

A quant plugin's table, exported as `exportName(name, version)`. `claims`, `validate` and `bill` never reach MLX. A load runs `open`, then any number of `load_tensor`/`load_bank`, then `close`; the weights outlive `close`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(QuantTable)` |  |
| `version` | `u32` | `version` |  |
| `name` | `Str` |  |  |
| `claims` | `*const fn (block_json: Str, config_json: Str) callconv(.c) Claim` |  | Claims one quantization block of `config_json`; a marker beside the block outranks the block's own mode. |
| `validate` | `*const fn (block_json: Str, config_json: Str, dir: *const PackDir) callconv(.c) Verdict` |  | Refuses by name a block the format cannot serve (an unknown mode, a width MLX has no kernels for). |
| `bill` | `*const fn (block_json: Str, model_dir: Str, stored: u64) callconv(.c) u64` |  | Resident bytes of the pack once loaded, given the `stored` bytes of the shards it loads. |
| `open` | `*const fn (ctx: *LoadCtx) callconv(.c) Verdict` |  |  |
| `close` | `*const fn (ctx: *LoadCtx) callconv(.c) void` |  |  |
| `load_tensor` | `*const fn (ctx: *const LoadCtx, name: Str, in_dim: u32, out: *Weight) callconv(.c) Verdict` |  | The projection `<name>.*`; `in_dim` (the activation's inner dim, 0 = unknown) pins its quant params exactly. |
| `load_bank` | `*const fn (ctx: *const LoadCtx, name: Str, in_dim: u32, out: *Weight) callconv(.c) Verdict` |  | Routed experts stacked under `<name>.*` (mlx-lm's `switch_mlp` layout), one bank. |
| `release` | `*const fn (w: *Weight) callconv(.c) void` |  |  |
| `matmul` | `*const fn (w: *const Weight, x: Array, s: Stream, out: *Array) callconv(.c) Verdict` |  | `x @ W^T`; `out` receives a new array. |
| `moe` | `*const fn (bank: *const Weight, x: Array, lhs: Array, rhs: Array, sorted: bool, s: Stream, out: *Array) callconv(.c) Verdict` |  | Routed experts: `x` through the bank rows `rhs` selects (`lhs` the gathered input rows, or empty; `sorted` when `rhs` is sorted). `out` receives a new array. |
| `register_kernels` | `?*const fn (host: ?*const Host) callconv(.c) Verdict` | `null` | Registers the format's kernels with the kernel contract (P2 step 6). |
| `swiglu` | `?*const fn (gate: *const Weight, up: *const Weight, down: *const Weight, x: Array, inds: Array, scores: Array, opts: *const Swiglu, s: Stream, out: *Array) callconv(.c) Verdict` | `null` | Routed SwiGLU experts in one dispatch, for a format whose kernels fuse the three projections (EXL3): `x` [B, S, D] through the banks' experts `inds` [B, S, K], weighted by `scores`. Null: compose `moe` per projection. |

### `Swiglu` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Swiglu)` |  |
| `limit` | `i32` | `0` | > 0: the SwiGLU's clamp (GLM's `swiglu_limit`). |
| `verify_rows` | `bool` | `false` | Spec-verify rows: the decode chain at any width. |

## `host.zig`

The `Host` table: everything process-wide a plugin reaches (the allocator, the log, the MLX error latch, the kernel cache, levers, metrics, the memory ceiling) and `HostGpu`, the C API below MLX's kernel API. A plugin receives it at load (`LoadCtx.host`) and keeps the pointer in its own per-load state: sdk holds no copy. Extern layouts, the C calling convention, `size` first, grows at its end. Called only from the GPU thread, unless a member says otherwise.

| Name | Value | |
|---|---|---|
| `Span` | `common.Span` |  |
| `Str` | `common.Str` |  |
| `Verdict` | `common.Verdict` |  |
| `Array` | `common.Array` |  |
| `Stream` | `common.Stream` |  |
| `version` | `u32 = 1` |  |
| `gpu_version` | `u32 = 1` |  |

### `LogLevel` (enum(u8))

`log.Level`'s order.

| Member | Value | |
|---|---|---|
| `err` | |  |
| `warn` | |  |
| `info` | |  |
| `debug` | |  |

### `CheckSite` (enum(u8))

Which latch checkpoint a check is: a prefill chunk or a decode step (their fault injectors count apart).

| Member | Value | |
|---|---|---|
| `prefill_chunk` | |  |
| `decode_step` | |  |

### `Latched` (enum(u8))

A consumed latch: nothing, a memory-class failure (a 503), or any other MLX failure.

| Member | Value | |
|---|---|---|
| `none` | |  |
| `out_of_memory` | |  |
| `failure` | |  |

### `KernelBackend` (enum(u8))

| Member | Value | |
|---|---|---|
| `metal` | |  |
| `cuda` | |  |

### `KernelDesc` (extern struct)

One custom kernel as mlx-c's `mlx_fast_{metal,cuda}_kernel_new` takes it. The cache key is (backend, `name`, `digest`); `digest` names the source and header (the caller's hash, usually comptime).

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(KernelDesc)` |  |
| `backend` | `KernelBackend` | `.metal` |  |
| `ensure_row_contiguous` | `bool` | `true` |  |
| `atomic_outputs` | `bool` | `false` | Metal only. |
| `shared_memory` | `i32` | `0` | CUDA only: dynamic shared memory bytes. |
| `digest` | `u64` |  |  |
| `name` | `Str` |  |  |
| `source` | `Str` |  |  |
| `header` | `Str` | `.{}` |  |
| `inputs` | `Span(Str)` |  |  |
| `outputs` | `Span(Str)` |  |  |

### `Host` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Host)` |  |
| `version` | `u32` | `version` |  |
| `ctx` | `?*anyopaque` |  |  |
| `alloc` | `*const fn (ctx: ?*anyopaque, len: usize, align_log2: u8) callconv(.c) ?[*]u8` |  |  |
| `resize` | `*const fn (ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8, new_len: usize) callconv(.c) bool` |  |  |
| `remap` | `*const fn (ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8, new_len: usize) callconv(.c) ?[*]u8` |  |  |
| `free` | `*const fn (ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8) callconv(.c) void` |  |  |
| `log_enabled` | `*const fn (ctx: ?*anyopaque, level: LogLevel) callconv(.c) bool` |  |  |
| `log` | `*const fn (ctx: ?*anyopaque, level: LogLevel, line: Str) callconv(.c) void` |  |  |
| `latch_pending` | `*const fn (ctx: ?*anyopaque) callconv(.c) bool` |  | An MLX call failed since the last consume (the `had_error` a best-effort op snapshots first). |
| `latch_check` | `*const fn (ctx: ?*anyopaque, site: CheckSite) callconv(.c) Latched` |  | Consume the latch at a checkpoint. |
| `latch_drop_unless` | `*const fn (ctx: ?*anyopaque, had_error: bool) callconv(.c) void` |  | Drop a latch the op just raised unless one was pending before it (`had_error`). |
| `gpu_context_lost` | `*const fn (ctx: ?*anyopaque) callconv(.c) bool` |  | A sticky CUDA fault: the process serves nothing until it restarts. Any thread. |
| `kernel` | `*const fn (ctx: ?*anyopaque, desc: *const KernelDesc) callconv(.c) ?*anyopaque` |  | The kernel cache: the process's one kernel object per (backend, name, digest), made on the first ask and kept for the process; null when the backend is not this build's or mlx-c refused. The handle is the mlx-c kernel's `ctx` (`mlx_fast_metal_kernel` / `mlx_fast_cuda_kernel`); a Metal one launches through `HostGpu.launch` too. |
| `lever` | `*const fn (ctx: ?*anyopaque, name: Str) callconv(.c) Str` |  | A lever's raw value; `ptr` null = unset. Read once and keep it: a lookup scans the environment. |
| `counter` | `*const fn (ctx: ?*anyopaque, name: Str) callconv(.c) ?*u64` |  | A named counter's word, added to with a relaxed atomic add; null when metrics are off or the name is unknown, so a plugin resolves it once and pays nothing per add when off. Any thread. |
| `memory_ceiling` | `*const fn (ctx: ?*anyopaque) callconv(.c) u64` |  | Bytes the plugin may still allocate on the GPU right now: the host's ceiling less what is resident. |
| `gpu` | `*const HostGpu` |  |  |

### `EventKind` (enum(u8))

| Member | Value | |
|---|---|---|
| `metal` | |  |
| `host` | |  |

### `Event` (extern struct)

An event across rounds: `id` names it to `wait`/`signal`; `object` is what an off-thread signaller holds (the id<MTLSharedEvent>, or the address of the host event's int64 word).

| Field | Type | Default | |
|---|---|---|---|
| `id` | `i32` | `0` |  |
| `kind` | `EventKind` | `.metal` |  |
| `object` | `u64` | `0` |  |

### `EventStats` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(EventStats)` |  |
| `gpu_waits` | `i64` | `0` |  |
| `gpu_signals` | `i64` | `0` |  |
| `host_ready` | `i64` | `0` | Host waits already satisfied when evaluated. |
| `host_blocked` | `i64` | `0` |  |
| `host_timeouts` | `i64` | `0` |  |
| `host_wait_ns` | `i64` | `0` |  |
| `cpu_passthrough` | `i64` | `0` | Waits on a CPU stream over a Metal event (pass-through). |
| `events` | `i64` | `0` |  |

### `VectorArray` (extern struct)

mlx-c handles the GPU calls pass through (by value, one context pointer each).

| Field | Type | Default | |
|---|---|---|---|
| `ctx` | `?*anyopaque` | `null` |  |

### `KernelConfig` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `ctx` | `?*anyopaque` | `null` |  |

### `HostGpu` (extern struct)

The C API below MLX's kernel API: one implementation in the host against MLX's internals, re-verified on each MLX bump. A graph without it (Linux, Windows) refuses every call `unsupported`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(HostGpu)` |  |
| `version` | `u32` | `gpu_version` |  |
| `ctx` | `?*anyopaque` |  |  |
| `event_metal` | `*const fn (ctx: ?*anyopaque, start: u64, out: *Event) callconv(.c) Verdict` |  | An MTLSharedEvent on MLX's device at `start`, kept for the process. |
| `event_host` | `*const fn (ctx: ?*anyopaque, word: *i64, timeout_ns: i64, out: *Event) callconv(.c) Verdict` |  | A host event over an 8-aligned int64 word, for CPU streams; a wait gives up after `timeout_ns`. |
| `wait` | `*const fn (ctx: ?*anyopaque, xs: [*]const Array, n: usize, event: i32, value: u64, deps: ?[*]const Array, n_deps: usize, track_inputs: bool, stream: Stream, outs: [*]Array) callconv(.c) Verdict` |  | `outs[i]` (fresh handles) alias `xs[i]` and are read only once `event` reaches `value` (>= 1). On a GPU stream the wait is encoded into the stream's current command buffer. The barrier choice: `deps` order the wait after them; `track_inputs` also orders it after the producers of `xs` (only for GPU-produced inputs). |
| `signal` | `*const fn (ctx: ?*anyopaque, xs: [*]const Array, n: usize, event: i32, value: u64, stream: Stream, outs: [*]Array) callconv(.c) Verdict` |  | `outs` alias `xs`; the GPU hands `value` to a Metal event after every pass encoded before it. |
| `event_set` | `*const fn (ctx: ?*anyopaque, event: Event, value: u64) callconv(.c) void` |  | Hand `value` to an event from any thread (not an MLX call): the read pool's completion. |
| `event_value` | `*const fn (ctx: ?*anyopaque, event: i32) callconv(.c) u64` |  | The event's current value (0 for an unknown id). Any thread. |
| `event_stats` | `*const fn (ctx: ?*anyopaque, out: *EventStats) callconv(.c) void` |  |  |
| `alloc_uninit` | `*const fn (ctx: ?*anyopaque, shape: [*]const i32, ndim: usize, dtype: i32, out: *Array) callconv(.c) Verdict` |  | `out` (a fresh handle) becomes an evaluated [shape] array of `dtype` (mlx-c's `mlx_dtype` value) over one buffer from MLX's allocator, its bytes unspecified until written. |
| `launch` | `*const fn (ctx: ?*anyopaque, outputs: *VectorArray, kernel: ?*anyopaque, inputs: VectorArray, config: KernelConfig, stream: Stream) callconv(.c) Verdict` |  | `mlx_fast_metal_kernel_apply`'s contract; a kernel from the host's cache launches MLX's built pipeline directly once MLX has built it (no per-launch source regeneration). Bit-identical to MLX's apply. |
| `buffer_addresses` | `*const fn (ctx: ?*anyopaque, arrays: [*]const Array, n: usize, out: [*]u64) callconv(.c) Verdict` |  | Each evaluated, row-contiguous GPU array's buffer address + offset, for kernels reading sources from a table. |

## `engine.zig`

The one interface between the host and every engine (ours included). Extern layouts, C calling convention.  Rules: slices cross as pointer and length (UTF-8, never NUL-terminated); memory is freed by whoever allocated it; a struct passed on its own starts with `size` and grows only at the end; callbacks run on the engine's thread, return at once and never call back into the engine; every fallible call returns a `Verdict` whose `why` is static text; `Caps` describe the opened model, never a static table.

| Name | Value | |
|---|---|---|
| `version` | `u32 = 1` |  |
| `Span` | `common.Span` |  |
| `Str` | `common.Str` |  |
| `Verdict` | `common.Verdict` |  |
| `no_token` | `u32 = 0xffff_ffff` |  |
| `entry_symbol` | `"mlx_serve_engine_v1"` | Each engine exports exactly this symbol. |
| `kv_configured` | `u8 = 0xff` |  |

### `Caps` (packed struct(u64))

What this opened model does beyond the required core (sampling, seed, EOS, cancel). Returned by `open`; the host refuses by name a request that needs what is not declared.

| Field | Type | Default | |
|---|---|---|---|
| `penalties` | `bool` | `false` |  |
| `logit_bias` | `bool` | `false` |  |
| `logprobs` | `bool` | `false` |  |
| `grammar` | `bool` | `false` | JSON schema, JSON object, regex, choice, EBNF |
| `think_budget` | `bool` | `false` | closes a thought at its budget, at decode |
| `tool_gate` | `bool` | `false` | tool_choice "required" or a named function, enforced at decode |
| `images` | `bool` | `false` |  |
| `video` | `bool` | `false` |  |
| `audio` | `bool` | `false` |  |
| `embed` | `bool` | `false` |  |
| `decide` | `bool` | `false` | typed decisions (laya, kev) |
| `owns_tokenizer` | `bool` | `false` | a GGUF engine that carries its own vocabulary |
| `owns_template` | `bool` | `false` | ... and renders its own chat prompt |
| `batches` | `bool` | `false` | reported in /props and /v1/models |
| `drafts` | `bool` | `false` |  |
| `prefix_cache` | `bool` | `false` |  |
| `distributed` | `bool` | `false` | reserved: a model served across several Macs |
| `prompt_logprobs` | `bool` | `false` | logprobs of the prompt's own tokens |
| `_` | `u46` | `0` |  |

### `Threading` (enum(u8))

MLX engines run on the host's one GPU thread (one MLX caller per process); engines with their own GPU runtime run their own thread.

| Member | Value | |
|---|---|---|
| `own_thread` | |  |
| `host_gpu_thread` | |  |

### `Host` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Host)` |  |
| `ctx` | `*anyopaque` |  |  |
| `log` | `*const fn (ctx: *anyopaque, level: u8, msg: Str) callconv(.c) void` |  |  |
| `wake` | `*const fn (ctx: *anyopaque) callconv(.c) void` |  | a host_gpu_thread engine has work without a submit |
| `progress` | `?*const fn (ctx: *anyopaque, what: Progress, n: i64) callconv(.c) void` | `null` |  |

### `Progress` (enum(u8))

What the host shows while it waits on an engine, reported as deltas from any of the engine's threads.

| Member | Value | |
|---|---|---|
| `load_total` | | `open`: units of work the load does (a Neural Engine build: its programs) |
| `load_done` | |  |
| `load_slow` | | a unit missed its cache: this load takes minutes (a first compile) |
| `prefill_todo` | | prompt tokens the engine's requests will feed; taken back when a prefill ends |
| `prefill_done` | |  |

### `Open` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Open)` |  |
| `model_dir` | `Str` |  |  |
| `model_type` | `Str` |  |  |
| `config_json` | `Str` |  | config.json, read once by the host |
| `context` | `u32` |  | the model's ctx_size; 0 = the engine's own |
| `max_concurrent` | `u32` |  |  |
| `memory_budget` | `u64` |  | what the engine may hold: GPU ceiling less memory in use, OS reserve out |
| `settings_json` | `Str` |  | launch flags + Model Settings that shape a load (kv_quant, mtp, drafter, caches, cluster) |
| `options_json` | `Str` |  | Model Settings `engine_options`, verbatim |
| `host` | `*const Host` |  | lives until `close` |

### `Info` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Info)` |  |
| `name` | `Str` |  |  |
| `version` | `Str` |  |  |
| `caps` | `Caps` |  |  |
| `threading` | `Threading` |  |  |
| `context_window` | `u32` |  | what the host advertises: prompt + reply |
| `max_concurrent` | `u32` |  |  |
| `embedding_dims` | `u32` |  |  |
| `eos` | `Span(u32)` |  | the engine's stop set; the host merges generation_config's |
| `startup` | `Str` |  | one line the host logs at load (the memory plan, the cluster) |
| `vocab_size` | `u32` | `0` | Caps.owns_tokenizer: the ids a prompt may carry |

### `Sampling` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `temperature` | `f32` | `1` |  |
| `top_p` | `f32` | `1` |  |
| `top_k` | `u32` | `0` |  |
| `min_p` | `f32` | `0` |  |
| `seed` | `u64` | `0` |  |
| `seeded` | `bool` | `false` |  |
| `repeat_penalty` | `f32` | `1` | Caps.penalties |
| `frequency_penalty` | `f32` | `0` |  |
| `presence_penalty` | `f32` | `0` |  |
| `logit_bias` | `Span(Bias)` | `.{}` | Caps.logit_bias |

### `Bias` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `token` | `u32` |  |  |
| `bias` | `f32` |  |  |

### `Grammar` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Grammar)` |  |
| `kind` | `enum(u8) { json_object, json_schema, regex, choice, ebnf }` |  |  |
| `text` | `Str` |  |  |
| `after` | `u32` | `no_token` | with thinking on, the constraint starts after this token |

### `Think` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Think)` |  |
| `budget` | `u32` |  | reply tokens allowed inside the thought |
| `open` | `Span(u32)` |  | the model's own opener (`<think>`, `<\|channel>thought`, ` to=self<\|message\|>`) |
| `opened_in_prompt` | `bool` |  | the rendered prompt already ended inside a thought: count from token 0 |
| `close` | `Span(u32)` |  | the early-stop line + closer, committed as one forward |
| `end` | `u32` |  | the token that ends a thought |

### `ToolGate` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(ToolGate)` |  |
| `opener` | `Span(u32)` |  | the dialect's call opener, committed after the thought |
| `names` | `Span(Str)` |  | offered names; empty = any |
| `think_end` | `u32` | `no_token` |  |

### `Media` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `kind` | `enum(u8) { image, video_frame, audio }` |  |  |
| `at` | `u32` |  | where its placeholder run starts in the prompt |
| `width` | `u32` |  |  |
| `height` | `u32` |  |  |
| `data` | `Span(u8)` |  | RGB8 pixels or PCM f32, decoded by the host |
| `hash` | `u64` |  | pixel hash: the encoder cache and the prefix cache key on it |

### `CacheHint` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `history_len` | `u32` | `0` | where the rendered history ends: worth keeping for the next turn |
| `key` | `u64` | `0` | the workload key for fair eviction |
| `keep` | `bool` | `true` | false: do not retain this request's state (the host sets it after a tool-call reply) |
| `cold` | `bool` | `false` | true: read no kept state (the exactness contracts' cold reference) |

### `Request` (extern struct)

Valid only during `submit`: the engine copies what it keeps.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Request)` |  |
| `prompt` | `Span(u32)` |  |  |
| `media` | `Span(Media)` | `.{}` | Caps.images / video / audio |
| `max_tokens` | `u32` |  |  |
| `sampling` | `Sampling` | `.{}` |  |
| `eos` | `Span(u32)` | `.{}` | empty: the request ignores EOS |
| `logprobs` | `u8` | `0` | top-n per token, 0 = off (Caps.logprobs) |
| `grammar` | `?*const Grammar` | `null` |  |
| `think` | `?*const Think` | `null` |  |
| `tool_gate` | `?*const ToolGate` | `null` |  |
| `cache` | `CacheHint` | `.{}` |  |
| `background` | `bool` | `false` |  |
| `options_json` | `Str` | `.{}` | enable_mtp, enable_drafter, enable_batch_mtp, kv_quant, draft depths... |
| `no_drafts` | `bool` | `false` | no speculative round of any kind: the plain reference of the exactness contracts |
| `prompt_logprobs` | `bool` | `false` | Caps.prompt_logprobs |

### `Logprob` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `token` | `u32` |  |  |
| `logprob` | `f32` |  |  |
| `top` | `Span(Bias)` |  |  |

### `Stats` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Stats)` |  |
| `rounds` | `u64` | `0` |  |
| `drafted` | `u64` | `0` |  |
| `accepted` | `u64` | `0` |  |
| `prefill_ns` | `u64` | `0` |  |
| `decode_ns` | `u64` | `0` |  |
| `spec_mode` | `Str` | `.{}` |  |
| `telemetry_json` | `Str` | `.{}` |  |

### `Event` (extern struct)

Per request: at most one `prefilled`, then `tokens`*, then exactly one `finished`, last. A request that ends before its prefill completes (cancelled, failed) gets `finished` alone.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Event)` |  |
| `kind` | `enum(u8) { prefilled, tokens, finished, audio }` |  |  |
| `cached` | `u32` | `0` | prefilled: prompt tokens restored from a kept prefix |
| `tokens` | `Span(u32)` | `.{}` | tokens: committed ids in order, valid during the call |
| `logprobs` | `Span(Logprob)` | `.{}` | tokens: one per id when asked for |
| `reason` | `enum(u8) { stop, length, cancelled, failed }` | `.stop` | finished |
| `stats` | `?*const Stats` | `null` | finished |
| `verdict` | `Verdict` | `.{}` | finished(failed): the named error |
| `prompt_tokens` | `u32` | `0` | prefilled: the prompt as the engine counted it |
| `prompt_logprobs` | `Span(Logprob)` | `.{}` | prefilled: the prompt's own logprobs when asked for |
| `constraint_at` | `u32` | `std.math.maxInt(u32)` | tokens: the token index where a constrained payload begins |
| `constraint_offset` | `u32` | `0` | tokens: its byte offset inside that token |
| `finish_details` | `Str` | `.{}` | finished: a static cause beside the reason (`repetition_loop`) |
| `loop_trim_start` | `u64` | `std.math.maxInt(u64)` | finished: where a loop-stopped span begins in the generated tokens |
| `refused_needed` | `u64` | `0` | finished(failed): bytes the engine's admission needed ... |
| `refused_available` | `u64` | `0` | ... and had, when it refused the prefill |
| `codes` | `Span(u32)` | `.{}` | audio: spoken frames' codes, frame after frame, valid during the call |
| `samples` | `Span(f32)` | `.{}` | audio: the mono samples those frames made final |
| `sample_rate` | `u32` | `0` | audio |

### `Action` (enum(u8))

Runs on the engine's thread; returns at once. `stop` ends the request as `finished(stop)` after this round: the host decodes the round's tail there and judges stop strings, loop-stop and disconnect.

| Member | Value | |
|---|---|---|
| `go` | |  |
| `stop` | |  |

### `Sink` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `ctx` | `*anyopaque` |  |  |
| `event` | `*const fn (ctx: *anyopaque, id: u64, ev: *const Event) callconv(.c) Action` |  |  |

### `Status` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Status)` |  |
| `running` | `u32` | `0` |  |
| `waiting` | `u32` | `0` |  |
| `decode_tps` | `f64` | `0` |  |
| `prefill_tps` | `f64` | `0` |  |

### `Memory` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Memory)` |  |
| `active` | `u64` | `0` |  |
| `cache` | `u64` | `0` |  |
| `peak` | `u64` | `0` |  |

### `Engine` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Engine)` |  |
| `version` | `u32` | `version` |  |
| `probe` | `*const fn (model_dir: Str, config_json: Str, out: *Verdict) callconv(.c) bool` |  |  |
| `load_bytes` | `*const fn (o: *const Open, bytes: *u64) callconv(.c) Verdict` |  | the preflight's bill |
| `open` | `*const fn (o: *const Open, info: *Info, out: *Verdict) callconv(.c) ?*anyopaque` |  |  |
| `close` | `*const fn (e: *anyopaque) callconv(.c) void` |  | cancels every request; each still gets `finished` |
| `submit` | `*const fn (e: *anyopaque, id: u64, r: *const Request, sink: Sink) callconv(.c) Verdict` |  |  |
| `cancel` | `*const fn (e: *anyopaque, id: u64) callconv(.c) void` |  |  |
| `step` | `?*const fn (e: *anyopaque, until_ns: u64) callconv(.c) bool` |  | host_gpu_thread only: true = did work |
| `status` | `*const fn (e: *anyopaque, out: *Status) callconv(.c) void` |  |  |
| `memory` | `*const fn (e: *anyopaque, out: *Memory) callconv(.c) void` |  |  |
| `report_json` | `?*const fn (e: *anyopaque, what: enum(u8) { props, metrics }, out: *Str) callconv(.c) void` |  |  |
| `trim` | `?*const fn (e: *anyopaque, want_bytes: u64) callconv(.c) u64` |  | drop caches under memory pressure |
| `media_rows` | `?*const fn (e: *anyopaque, m: *const Media) callconv(.c) u32` |  | placeholder run length |
| `embed` | `?*const fn (e: *anyopaque, inputs: Span(Span(u32)), out: [*]f32) callconv(.c) Verdict` |  |  |
| `decide` | `?*const fn (e: *anyopaque, request_json: Str, out_json: *Str) callconv(.c) Verdict` |  |  |
| `tokenize` | `?*const fn (e: *anyopaque, text: Str, specials: bool, out: [*]u32, cap: usize) callconv(.c) usize` |  |  |
| `token_bytes` | `?*const fn (e: *anyopaque, token: u32) callconv(.c) Str` |  | engine-owned, stable |
| `chat_template` | `?*const fn (e: *anyopaque) callconv(.c) Str` |  |  |
| `render` | `?*const fn (e: *anyopaque, chat_json: Str, out: [*]u32, cap: usize) callconv(.c) usize` |  |  |
| `admission` | `?*const Admission` | `null` |  |

### `Admission` (extern struct)

What an engine on the host's GPU thread prices against memory: the context it advertises and whether a prompt fits. `model` is the engine's own per-model object (ours: its `ModelConfig`).

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Admission)` |  |
| `context` | `*const fn (model: *const anyopaque) callconv(.c) u32` |  | prompt + reply, as advertised |
| `pin_context` | `*const fn (model: *anyopaque, announce: bool) callconv(.c) u32` |  | freezes it; `announce` logs how |
| `max_safe_context` | `*const fn (model: *const anyopaque) callconv(.c) u32` |  | the largest that fits now, no headroom |
| `cache_budget` | `*const fn () callconv(.c) u64` |  | the resolved prefix-cache byte budget; 0 = none |
| `admit` | `*const fn (model: *const anyopaque, keep: Str, prompt: Span(u32), max_tokens: u32, kv_bits: u8, mtp: bool, media_bytes: u64, out: *Bill) callconv(.c) bool` |  | true admits; false refuses, `out` holding what was compared. `keep` names the model no idle unload may take; `kv_bits`: 0 dense, 4 or 8, `kv_configured` = the model's own scheme. |

### `Bill` (extern struct)

A prompt's memory bill and the headroom it was compared against, in bytes.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Bill)` |  |
| `needed` | `u64` | `0` |  |
| `available` | `u64` | `0` |  |
| `evictable` | `u64` | `0` | Bytes the hot prefix cache is holding. Reported in a refusal, never credited (see `reclaimable`). |
| `reclaimable` | `u64` | `0` | The part of `evictable` an eviction can prove it will return: residency minus the largest entry (a restore pins at most one). |
| `chunk` | `u64` | `0` | The prefill width this bill was taken at, so a refusal can name it. |

### `span(comptime T: type, s: []const T) Span(T)`

### `items(comptime T: type, s: Span(T)) []const T`

### `needs(r: *const Request) Caps`

The capabilities a request asks for. A field at its neutral value asks for nothing.

### `missing(have: Caps, need: Caps) ?[]const u8`

The first capability `need` asks that `have` lacks, by its `Caps` field name; null when served.

### `refusal(have: Caps, need: Caps) ?Verdict`

The refusal for the first capability `need` asks that `have` lacks, naming its `Caps` field; null when served.

## `common.zig`

The ABI primitives every table shares (engine, model, media, quant): slices, verdicts, MLX handles, claims and the pack directory discovery reads. Extern layouts and the C calling convention; nothing here holds state.

| Name | Value | |
|---|---|---|
| `Str` | `Span(u8)` | UTF-8 text, never NUL-terminated. |

### `Span(comptime T: type) type`

A borrowed slice: pointer and length. Freed, if ever, by whoever allocated it.

### `Verdict` (extern struct)

The answer of every fallible call: a code the host maps to HTTP, and the reason by name.

| Field | Type | Default | |
|---|---|---|---|
| `code` | `Code` | `.ok` |  |
| `why` | `Str` | `.{}` | names the reason; static text, safe to read from any thread at any time |

### `Array` (extern struct)

An mlx-c `mlx_array` by value (one context pointer, the same layout). Who owns the reference is stated at each use.

| Field | Type | Default | |
|---|---|---|---|
| `ctx` | `?*anyopaque` | `null` |  |

### `Stream` (extern struct)

An mlx-c `mlx_stream` by value.

| Field | Type | Default | |
|---|---|---|---|
| `ctx` | `?*anyopaque` | `null` |  |

### `Claim` (enum(u8))

How a plugin claims a pack, in rank order: a marker file or field beats a `model_type` match.

| Member | Value | |
|---|---|---|
| `no` | `0` |  |
| `by_type` | `1` |  |
| `by_marker` | `2` |  |

### `pick(claims: []const Claim) error{Ambiguous}!?usize`

The one claimant of a pack: the highest rank wins, two claims at that rank are ambiguous, null = nobody claims it.

### `PackDir` (extern struct)

A pack directory as discovery sees it: existence and bounded reads, paths relative to the pack root, no MLX.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(PackDir)` |  |
| `ctx` | `?*anyopaque` |  |  |
| `exists` | `*const fn (ctx: ?*anyopaque, rel: Str) callconv(.c) bool` |  | Whether `rel` exists (a file or a folder). |
| `read` | `*const fn (ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict` |  | Reads `rel` whole into `buf`, setting `len`; a file longer than `cap` is `too_large`, a missing one `failed`. |
| `safetensors_header` | `*const fn (ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict` |  | Reads the JSON header of the safetensors file `rel` into `buf` (the same bounds as `read`). |
| `list` | `*const fn (ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict` |  | The entries of folder `rel` ("" = the root) as NUL-separated names, a folder's name ending in '/'; symlinks followed. The same bounds as `read`. |
| `file_bytes` | `*const fn (ctx: ?*anyopaque, rel: Str) callconv(.c) u64` |  | `rel`'s size in bytes through symlinks; 0 when it is missing or not a file. |

### `Entries` (struct)

A `PackDir.list` result, one name at a time (a folder's name ends in '/').

| Field | Type | Default | |
|---|---|---|---|
| `rest` | `[]const u8` |  |  |

### `exportName(comptime name: []const u8, comptime version: u32) []const u8`

The symbol a plugin's table is exported under: `mlx_serve_plugin_<name>_v<version>`.

## `dylib.zig`

An outside plugin's dylib: the one symbol it exports (`Manifest`, under `symbol`) and the negotiation the host runs before it reads anything else. Extern layouts, the C calling convention, `size` first, grows at its end.

| Name | Value | |
|---|---|---|
| `Span` | `common.Span` |  |
| `Str` | `common.Str` |  |
| `Verdict` | `common.Verdict` |  |
| `version` | `u32 = 2` |  |
| `symbol` | `"mlx_serve_plugin"` | The manifest's export name. Unversioned: a dylib built for another manifest version is refused by name, never reported as "no plugin here". |

### `Kind` (enum(u8))

| Member | Value | |
|---|---|---|
| `model` | |  |
| `media` | |  |
| `quant` | |  |

### `File` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `name` | `Str` |  |  |
| `bytes` | `Str` |  |  |

### `Fixture` (extern struct)

The plugin's conformance fixture (`sdk.fixture.Fixture` across the boundary), in memory the plugin owns until `Manifest.fixture_free`.

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Fixture)` |  |
| `files` | `Span(File)` | `.{}` |  |
| `body` | `Str` | `.{}` |  |
| `block` | `Str` | `.{}` |  |
| `ctx` | `?*anyopaque` | `null` |  |

### `Manifest` (extern struct)

| Field | Type | Default | |
|---|---|---|---|
| `size` | `u32` | `@sizeOf(Manifest)` |  |
| `version` | `u32` | `version` |  |
| `name` | `Str` |  |  |
| `kind` | `Kind` |  |  |
| `mlx` | `Str` |  | The MLX build the plugin was compiled against (the kit's `mlx.version`): one MLX per process. |
| `table` | `*const anyopaque` |  | The kind's table (`ModelTable`, `MediaTable` or `QuantTable`), `size` and `version` first. |
| `claims` | `Span(Str)` | `.{}` | The `model_type`s it declares (`sdk.Plugin.claims`), for the conformance suite. |
| `fixture` | `*const fn (out: *Fixture) callconv(.c) Verdict` |  |  |
| `fixture_free` | `*const fn (f: *Fixture) callconv(.c) void` |  |  |
| `plugin_version` | `Str` | `.{}` | The plugin's own version (its root's `version`), shown at install and in Settings; "" when it declares none. |

### `negotiate(m: *const Manifest, host_mlx: []const u8) Verdict`

The host's checks on a manifest, before any other member is read: refuses by name a manifest, table or hook table of another version or a smaller size, another MLX, or a name a plugin cannot have.

### `validName(name: []const u8) bool`

A plugin name: `[a-z0-9_]`, 1 to 64 bytes (it names a file, a log line and a `/v1/models` badge).
