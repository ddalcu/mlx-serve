# tests/ — index of the integration scripts (auto-loaded when working here)

TDD policy and the class-guard rule live in the root AGENTS.md. This file is the map: one line per script, by area,
plus the rules for running them. What a script checks, what it needs and how to run it is in its own header.

## Running rules

- **Binary**: scripts run `./zig-out/bin/mlx-serve` (most take `BINARY` or `MLX_SERVE_BIN`). Build
  `zig build -Doptimize=ReleaseFast` first; `zig build test` does not refresh it, and Debug is 2-4x slower.
- **Boot and wait through `tests/lib/server.sh`** (`srv_wait_health`, `srv_wait_log`, `srv_stop`; Python twin
  `tests/lib/server.py`, `from lib.server import wait_health`). Readiness is the STATUS: `/health` answers 503
  `loading` while a model loads, so a bare socket or `curl -s` check lies.
- **Kill only the server you started** (`srv_stop "$PID"` or `pkill -f "mlx-serve.*--port $PORT"`), never by name or
  binary path: that kills the app's server and other sessions' runs. The app's server is 11234 (11240/11250 are
  taken too); every script has its own default port, pass one when running several.
- **Scratch HOME** for anything that lists, picks or configures models: a real `~/.mlx-serve/model-settings.json`
  attaches drafters and pins KV schemes, `providers.json` adds `@provider` rows. A test asserting one spec mode passes
  `--no-drafter` / `--no-mtp` for the arms it does not test.
- **Models**: name checkpoints by `<org>/<repo>` through `tests/_lib_models.sh` (`find_model`, `model_fits`), never a
  volume path; `./tests/fetch_test_models.sh` downloads what this box can load. A missing model SKIPs (exit 0).
- **One model at a time**: check free RAM and resident models before a big load; a SIGKILLed load is a stop, not a
  retry. Scratch output goes under `~/claude-tmp/`, never `/tmp`.
- **Assert engagement, not just output**: a spec / kernel / cache path that silently falls back is output-identical,
  so scripts count the server's own log lines. Assert invariants, branch on the model's choices.
- **Env-gated Zig tests** (`FOO_MODEL=<dir> zig build test -Dtest-filter=...`): `zig build test` caches a passing run
  (env is not in the cache key), so rerun the built binary (`zig-out/tests/test` after `zig build test-build`, or
  `.zig-cache/o/<hash>/test`) to re-execute.

## Hermetic suites (no model)

| Command | What |
|---|---|
| `zig build check` / `zig build test` | Sema over both roots / unit tests (Debug). |
| `zig build test -Dslow-tests` | Adds the corpus replays and benchmark tests (`tool traffic`, stream gate); a bare `-Dtest-filter="tool traffic"` compiles and reports success without running them. |
| `zig build test -Dtest-filter="format corpus"` | Format corpus: captured outputs per family + no-tag-leak / valid-JSON-args invariants. |
| `zig build preview-test` | JPEG / latent->RGB preview tests, no MLX (Linux too). |
| `bash app/test.sh` | Swift unit tests. `cd app-web && npm ci && npm test` the console. |
| `node --test tests/*.test.mjs`, `node tests/website_*_logic.mjs` | film skill, llmprobe smoke planner, website logic. |
| `bash tests/test_lib_server.sh`, `bash tests/test_lib_models.sh` | The harness libs. |

## Script index

Marks: **H** no model needed; **E** env-gated on a pack (SKIP without); **R** needs a running server you started.

**Harness**
- `lib/server.sh`, `lib/server.py` boot-and-poll helpers · `test_lib_server.sh` **H** their tests
- `_lib_models.sh` model lookup + GPU-budget gate · `test_lib_models.sh` **H** · `_lib_supported_models.sh` model enumeration
- `fetch_test_models.sh` download the release-test checkpoints this box can load

**Matrices and smoke**
- `test_smoke_matrix.sh` every text arch x configs x ~28 API checks (~1 h)
- `test_format_matrix.sh` 7 format families x thinking/tools/arg fidelity
- `test_tool_matrix_small.sh` tool calls across the sub-4B fleet
- `test_validator_matrix.sh` llmprobe + pi html case per arch (pkills every mlx-serve)
- `llmprobe_smoke_test.sh` conformance + speed per family x spec cell · `llmprobe_smoke_lib.mjs` + `.test.mjs` **H**
- `sweep_all_archs.sh` every arch vs mlx-lm · `sweep_agent_memory.sh` the 11-turn memory test per arch
- `integration_test.sh` 36 API checks · `pi_integration_run.sh` pi x streaming x thinking driver
- `test_mlxcore.sh` MLX Core app UI automation · `test_swift_agent.sh` app agent tools via TestServer

**API surfaces**
- `test_api_edges.sh` 60 request-validation contracts · `test_anthropic_api.sh` **R** /v1/messages · `test_anthropic_vision.sh` image blocks
- `test_responses_api.sh`, `test_responses_streaming.sh` /v1/responses · `test_output_config.sh` Anthropic `output_config`
- `test_ollama_api.sh` /api/* · `test_continuation.sh` continue a partial reply · `test_timings.sh` `timings` block
- `test_models_capabilities.sh` /v1/models capabilities · `test_metrics.sh`, `test_metrics_outcomes.sh` --metrics
- `test_index_page.sh` **H** console at GET / · `test_route_404_no_load.sh` unknown path loads nothing
- `test_headless_spec_flags.sh` **H** headless honours spec flags, 404 before resolution
- `test_providers.sh` **H** upstream providers via stubs · `test_launch_cmd.sh` `mlx-serve launch <agent>`
- `test_launch_script.sh` **H** GET /launch · `test_website_agent_install.sh` **H** website/agent.sh · `test_opencode_probe.sh` **H**
- `test_run_quiet.sh` **H** REPL greeting has no discovery noise · `test_partial_download.sh` **H** interrupted pull resumes
- `test_pull_registers_org_id.sh` **H** `/api/pull org/name` id · `test_lan_share.sh`, `test_lan_peer_key.sh` two servers over LAN

**Tool calling, thinking, sampling, formats**
- `test_tool_args.sh`, `test_tool_parsing.sh`, `test_tool_call_parse.sh` argument parsing · `test_tool_response.sh` Gemma 4 round trip
- `test_tool_choice_required.sh` forced calls · `test_tools_concurrent.sh` per-stream calls under batching
- `test_tools_long_context.sh` calls at 16-64k · `test_cancel_mid_tool_buffer.sh` disconnect while buffering
- `test_no_tools_markup_passthrough.sh` no `tools` = raw text · `test_minicpm5_tool_call.sh` MiniCPM5 XML calls
- `test_thinking_split.sh`, `test_thinking_streaming.sh`, `test_thinking_tools.sh` reasoning/content split
- `test_messages_stream_thinking_tools.sh` Claude Code's stream shape · `test_reasoning_budget_stream.sh` budget ends the thought
- `test_stop_in_reasoning.sh` stop strings end the answer only · `test_loop_stop_signal.sh` loop-stop wire contract
- `test_json_schema.sh`, `test_json_schema_enforcement.sh`, `test_json_schema_thinking.sh` grammar-enforced schemas
- `test_json_schema_protocol_routing.py` protocol-looking JSON survives · `test_json_mode_multi_model.sh` per-model grammar table
- `test_hot_switch_state.sh` per-model tokenizer/template/grammar · `test_tokenizer_hf_parity.sh` /tokenize == HF
- `test_logprobs.sh` logprobs contract · `test_repeat_penalty.sh` penalties on every path · `test_sampling_defaults.py` min_p
- `test_generation_defaults.py` defaults files + `ignore_client` · `glm_exactness.py` **R** greedy/seed/logprob exactness
- `harvest_tool_traffic.py` capture real tool traffic into the replay fixture

**Agents**
- `test_agent_loop.sh` / `.py`, `test_deep_agent_loop.sh`, `test_agent_thinking.py` multi-turn tool loops
- `test_agent_loop_spec.sh`, `test_long_conversation_spec.sh` agent loops under spec decode
- `test_long_agent_memory.sh` (+ `.py`) 11-turn recall · `test_agent_stop_repro.py` mid-task stop probe
- `test_coding_agent.py`, `test_coding_stress.py`, `test_research_agent.py` long agent stress · `cli_agent.py` **R** agent loop CLI
- `glm_pi_matrix.sh` pi on GLM-5.3 per cell · `analyze_pi_sessions.py` scores pi sessions · `agent_eval/` (README) judged agent tasks

**Speculative decoding**
- `test_pld_equivalence.sh`, `test_streaming_pld.sh`, `test_pld_tools.sh` PLD byte equivalence
- `test_drafter_equivalence.sh`, `test_streaming_drafter.sh`, `test_drafter_tools.sh` Gemma 4 drafter
- `test_completions_spec.sh` spec on /v1/completions · `test_tool_call_streaming_spec.sh` call boundaries under PLD
- `test_mtp_equivalence.sh` MTP engagement + tie-aware equivalence · `test_mtp_adaptive.sh` adaptive serial switch
- `test_mtp_batched.sh` concurrent MTP at fixed depth · `test_mtp_batch_stops.py` **R** grouped-MTP EOS/cancel
- `test_dflash.sh` **E**, `test_dflash_company.sh` **E**, `test_dspark_lfm2.sh` **E** DFlash / DSpark · `test_dsv41_speculation.py` **R**
- `build_mtp_sidecar.py` MTP sidecar from Qwen weights · `record_mtp_accept_traces.py` **R** the MTP replay fixture

**Scheduler, batching, streams, cancellation**
- `test_batched_equivalence.sh` batched == serial (incl. kv8) · `test_batched_transition.sh` solo -> batched join
- `test_batched_features.sh` mixed features in one group · `test_batched_past_block_width.sh` 20 streams beside a drafter
- `test_concurrent_throughput.sh`, `test_concurrent_cancellation.sh` --max-concurrent
- `test_prefill_interleave.sh` decode during a cold prefill · `test_boundary_admission.sh` small requests at chunk boundaries
- `test_first_token_early.sh` prefill token streams first · `test_stream_keepalive.sh` (+ `stream_keepalive_probe.py`) socket liveness
- `test_disconnect_cancel.sh`, `test_stop_latency.sh`, `test_cancel_commit.sh`, `test_cancel_mid_tick.sh` cancellation
- `test_shutdown_midstream.sh` SIGTERM mid-stream · `test_connection_thread_reaping.sh` thread stacks reclaimed
- `test_mlx_error_recovery.sh` an MLX fault costs one request · `test_sleep_inhibit.sh` power assertions
- `test_multiclient_stress.sh` agents + chats + TTS + image on one server · `test_soak_24h.sh` 4-way soak

**Memory, loading, registry**
- `test_memory_pressure_4way.sh` low guard ceiling · `test_oom_preflight.sh` **H** refusal exits cleanly
- `test_kv_cache_growth_memory.sh` long decode does not ratchet · `test_evict_returns_memory.sh` make-room frees memory
- `test_idle_evict.sh` --idle-evict-secs · `test_load_progress.sh` /v1/load-progress · `test_cold_load_launch_flags.sh` cold loads keep flags
- `test_load_failure_no_fallback.sh` a broken pack never falls back · `test_load_model_api.sh`, `test_load_model_default.sh` /v1/load-model
- `test_model_alias.sh` Model Settings alias · `test_model_settings.sh` per-model settings · `test_model_discovery.sh` discovery
- `test_model_rescan.sh` **H** rescan · `test_multi_model_dir.sh` **H** repeatable --model-dir
- `test_multi_model_{basic,concurrent,lru,load_failure,eviction_with_active_request}.sh` multi-model routing and eviction

**KV and prefix cache**
- `test_kv_cache_poison.sh` pad-only gens · `test_kv_cache_sliding_window.sh` sliding reuse
- `test_kv_quant_equivalence.sh`, `test_kv_quant_per_request.sh`, `test_kv_quant_fused_equivalence.sh` quantized KV · `measure_fused_kv.sh`
- `test_prefix_cache_{hot,mem,disk,disabled,workloads,budget_revisit}.sh` RAM + SSD tiers · `test_prefix_inherit_qsa.sh`
- `test_cache_reuses_generated_tokens.sh`, `test_hybrid_reuse_equivalence.sh`, `test_mid_system_cache.sh` reuse across turns
- `test_tokenize_cache.sh`, `test_tokenize_ms_present.sh` tokenize cache · `diff_restore_dump.py` diff restore dumps

**Engines and runtime**
- `test_llama_gguf.sh` **E**, `test_llama_concurrent.sh` **E**, `test_llama_multi_session.sh`, `test_llama_persistent_session.sh` llama.cpp
- `test_ds4_serve.sh`, `test_ds4_smoke.sh`, `test_ds4_ssd_streaming.sh` ds4 · `test_mlx_gguf.sh` MLX GGUF · `test_qwen4exp_gguf_route.sh`
- `test_ane_prefill.sh`, `test_ane_engine.sh` Neural Engine · `test_mlx_staged_nax.sh` **H** NAX-enabled mlx stage
- `test_phase2_forward_equivalence.sh` forward pinned vs a baseline · `test_decode_graph_dump.sh` graph dump guard
- `test_plugin_kit.sh` **H** the kit builds `_example` from itself; `plugin new|check|fixture|install`; a kit-built dylib serves its pack

**Architectures (live)**
- `test_qwen4_exp.sh` **E** Flash Next (external PLE or embedded `shards.N` / `shard_N`) · `test_qwen4_mtp_{padded_head,head_persist}.sh` **E** its MTP head
- `test_qsa_nax_prefill.py` **R** QSA NAX prefill · `qsa_nax_precision.py` float64 QSA kernel oracle
- `test_dsv4.sh` **E**, `test_dsv41.sh` **E** DeepSeek V4/V4.1 · `test_inkling.sh` **E** · `test_bailing_hybrid.sh` **E** Ling 3.0
- `test_muse_glimmer.sh` **E**, `test_muse_repetition.sh` **E** Muse · `test_mimo_v2.sh` **E** · `test_glm5_next.sh` **E** · `test_glm53.sh` **E**
- `test_flint.sh` **E** · `test_kolibri1.py` **E** · `test_minicpmv4_7.sh` **E** · `test_nemotron_bolt.py` Nemotron 3.5
- `test_ud_moe.sh` · `test_nvfp4.sh` · `test_dense_bf16.sh` · `test_hadamard_fidelity.sh` (+ `hadamard_fidelity.py`)
- `test_kev_decisions.sh`, `test_laya_decisions.sh`, `test_d1_decisions.sh` POST /v1/decisions

**Vision, audio input, embeddings**
- `test_vision_all.sh` arch-dispatch sweep · `test_vision.sh` **R** · `test_qwen_vision.sh` · `test_qwen_video_input.sh`
- `test_qwen_vision_parity.sh`, `test_qwen_preprocessing_parity.sh` Qwen3-VL vs mlx-vlm · `test_muse_vision.sh` · `test_lfm2_vision.sh`
- `test_dense_vision.sh` · `test_vision_moe_regression.sh` · `test_gemma4_unified_vision.sh`, `test_gemma4_unified_audio.sh`
- `test_vision_chunked_prefill.sh` · `test_vision_prefix_cache.sh` · `test_multi_image_history.sh` · `test_decode_end_media_reuse.sh`
- `test_embeddings.sh` /v1/embeddings · `test_transcriptions.sh` ASR + aligner + diarization
- `test_lfm3.sh` **E** LFM3 text/VL/Audio/Omni: tools, images, speech out (chat audio, stream, TTS, ASR round trip)

**Media generation**
- `test_unified_gen.sh` load/gen/unload · `test_gen_chat_interleave.sh` chat during media · `test_gen_chat_mix.sh` release battery
- `test_image_gen.sh` FLUX.2 · `test_image_cancel.sh` · `test_flux_lowmem.sh` · `test_krea_gen.sh` · `test_mageflow_edit.sh`
- `test_qwen_image_gen.sh` · `test_qwen_image_lora.sh` (+ `qwen_image_lora_metrics.py`) · `test_moss_image_gen.sh`
- `test_multi_lora.sh` stacked LoRAs · `test_real_loras.sh` Hub LoRAs (+ `lora_noise.py`) · `test_media_gen_memory.sh` · `test_media_eviction_gate.sh` **H**
- `test_video_gen.sh` LTX · `test_video_preview.sh` (part **H**) · `test_minimax_h3.sh`, `test_h3_resident.sh`, `test_fasth3_omni.sh` H3
- `test_tts.sh`, `test_tts_quant.sh`, `test_tts_clone_cache.sh`, `test_tts_language.sh` Qwen3-TTS · `test_kokoro_tts.sh`
- `test_music_gen.sh` ACE-Step · `test_music3_gen.sh` · `test_yue2_gen.sh` · `test_sound_gen.sh` Stable Audio 3
- `test_3d_gen.sh`, `test_3d_paint.sh` Hunyuan3D · `test_trellis2_3d.sh` · `test_pixal3d_3d.sh` · `film_skill.test.mjs` **H**

**Static and release guards** (all **H**)
- `test_release_workflow_gates.sh` release.yml + app/build.sh · `test_release.sh` release.sh · `test_resource_xattrs.sh`
- `test_model_card_frontmatter.sh` converter model cards · `test_website_pages.sh` · `website_benchmarks_logic.mjs`, `website_tier_list_logic.mjs`

**Benchmarks and diagnostics** (measurements, not pass/fail; method in the `/bench` skill)
- `bench.sh` THE bench · `bench_versions.sh` (+ `bench_csv.py`, `plot_version_ab.py`) two builds · `bench_chart.py` PR charts
- `bench_concurrency_ladder.sh`, `bench_batched_spec.sh`, `bench_laguna_xs.sh`
- `fwd_ubench.sh` ms/forward · `greedy_ab.sh` output-preservation bar · `prefill_ab.sh` · `qwen4_ab.sh` · `qwen4_engagement.sh`
- `MLX_SERVE_DIAG=vqmm-fn-ubench zig build test -Doptimize=ReleaseFast -Dtest-filter="Flash Next shapes µbench"` verify-lane prices

**Converters, pack tools, calibration** (user-run; the header names inputs and the bar)
- `convert_{acestep_weights,dsv4_weights,hunyuan3d_weights,hunyuan3d_paint_weights,kev_weights,kokoro_weights,mageflow_weights,mimo_v2,minicpmv4_7,minimax_h3_weights,moss_image_weights,nemotron_lilicorr,qwen38_flash_next,qwen38_iq,qwen_image21_weights,trellis2_pixal3d,yue2}.py`
- `fetch_convert_dsv4_0731.py`, `fetch_convert_minimax_h3_ref2va.py` disk-safe fetch+convert · `graft_qwen_image21_vision.py`
- `requant_qwen4_pack.py`, `restandardize_qwen38_pack.py` repack in place · `verify_nemotron_lilicorr.py`
- `test_convert_{minicpmv4_7,minimax_h3_omni,nemotron_lilicorr,qwen38_flash_next_bf16}.py` **H** converter checks (mlx or torch)
- `dsv4_imatrix.py` weighted affine quant · `qwen38_imatrix_collect.py`, `qwen38_iq_allocate.py`, `qwen38_iq_battery.py` 27B iQ
- `qwen38_flash_next_imatrix_collect.py`, `qwen38_flash_next_iq_allocate.py`, `qwen38_flash_next_score.py` Flash Next iQ

**Reference oracles** (write fixtures for env-gated Zig parity tests; each header names its test)
- `dump_*_fixtures.py` / `dump_*_fixture.py`, one per port: acestep, csf32, d1, dflash, dflash2, dsv41 (+ token map, vision),
  embeddinggemma (+ 2, 2 media), flint, glm5_next (+ vision), h3_vsa, hunyuan3d (+ paint), inkling, kev, kokoro, krea,
  laguna, latent_rgb_factors, laya (+ numeric), lfm2_vision, lilicorr, ltx (audio, audio encoder, diffvae, upsampler, vae
  encoder), mageflow (dit, e2e, edit e2e, te, te edit, vae, vae encoder, vit), mimo_v2, minicpmv4_7_vision, minimax_h3
  (+ layout, vision, vision tower, vae encoder, audio encoder), moss_image, muse_vision, music3, nemotron_diarization,
  pixal3d, qwen3_aligner, qwen3_asr, qwen3_embedding, qwen4_exp, qwen4_yarn, qwen_image21, radio (full, vision),
  stable_audio, tokenizer_parity, trellis2, yue2
- `build_qwen_preprocess_fixture.py`, `build_qwen_vision_fixture.py` Qwen3-VL fixtures · `dsv4_mlx_ref.py` MLX reference
- `dsv4_template_ab.py`, `dsv41_template_ab.py` byte-exact template A/B · `dsv4_task_compare.py` **R** native vs GGUF tasks

**Other files**: `fixtures/` (prompts, images, captured requests) · `hy3d_paint_unet_dossier.md`,
`hy3d_paint_weights_contract.md` Hunyuan3D paint port notes · `agent_eval/README.md`.
