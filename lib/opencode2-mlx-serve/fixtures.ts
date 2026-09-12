/**
 * Fixtures captured verbatim from a live mlx-serve run on this machine
 * (Qwen3.8-Flash-Next, M5 Max, `serve --metrics --mtp`), so the parsers and the
 * rows are tested against the bytes the server really produces rather than an
 * idealisation of them. Shared by stats.test.ts and rows.test.ts.
 */

import { parseFeed, type MetricsFeed, type RawMetricsJson, type WireCounters, type WireGauges } from "./stats.ts"

// Captured from a live `mlx-serve serve --metrics` run (Qwen3.8-Flash-Next, M5 Max).
export const LIVE_FEED = {
  counters: {
    prompt_tokens_total: 347251,
    prefill_tokens_total: 132503,
    prefix_cache_tokens_total: 214748,
    generation_tokens_total: 3271,
    requests_success_total: 15,
    requests_cancelled_total: 0,
    prefix_cache_queries_total: 15,
    prefix_cache_hits_total: 9,
  },
  gauges: {
    requests_running: 1,
    requests_waiting: 0,
    gpu_utilization_pct: 63,
    memory_mb: 75063,
    generation_tokens_live: 3094,
    prefill_tokens_live: 0,
    requests_prefilling: 0,
    mlx_active_bytes: 77386308302,
    mlx_cache_bytes: 766692711,
    ane_int8_bytes: 0,
    ane_layers: 0,
    ngram_warm_bytes: 32000153976,
  },
  histograms: {
    time_to_first_token_seconds: {
      count: 15,
      sum: 109.987431123,
      bounds: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
      bucket_counts: [0, 0, 0, 1, 2, 2, 2, 6, 11, 13, 15],
    },
    e2e_request_latency_seconds: {
      count: 15,
      sum: 157.459021048,
      bounds: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
      bucket_counts: [0, 0, 0, 1, 2, 2, 2, 4, 6, 10, 15],
    },
    prefill_time_seconds: {
      count: 15,
      sum: 101.082668209,
      bounds: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
      bucket_counts: [0, 0, 0, 1, 3, 3, 3, 7, 11, 14, 15],
    },
    decode_time_seconds: {
      count: 15,
      sum: 47.471589925,
      bounds: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
      bucket_counts: [1, 1, 3, 3, 3, 4, 5, 7, 12, 14, 15],
    },
    prompt_tokens: {
      count: 15,
      sum: 347251,
      bounds: [32, 128, 256, 512, 1024, 2048, 4096, 8192],
      bucket_counts: [2, 3, 3, 3, 4, 4, 4, 4, 15],
    },
    output_tokens: {
      count: 15,
      sum: 3271,
      bounds: [32, 128, 256, 512, 1024, 2048, 4096, 8192],
      bucket_counts: [4, 7, 10, 13, 15, 15, 15, 15, 15],
    },
  },
}

export const LIVE_PROPS = {
  default_generation_settings: { model: "qwen4_exp", n_ctx: 1048576 },
  total_slots: 1,
  model_info: { vocab_size: 248320, num_hidden_layers: 48 },
  memory: {
    active_bytes: 76641529386,
    peak_bytes: 81257077642,
    available_bytes: 47785148416,
    max_safe_context: 1048576,
    cache_bytes: 411423012,
  },
  ngram_warm: { bytes: 32000153976, total: 32000153976 },
}

export const LIVE_MODELS = {
  object: "list",
  data: [
    {
      id: "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit",
      object: "model",
      created: 1788848181,
      owned_by: "mlx-serve",
      loaded: true,
      state: "ready",
      bytes_resident: 1966080,
      context_length: 1048576,
      max_model_len: 1048576,
      capabilities: ["chat", "tool_use", "streaming", "vision", "reasoning"],
      input_modalities: ["text", "image", "video"],
      meta: {
        architecture: "qwen4_exp",
        engine: "mlx",
        vocab_size: 248320,
        hidden_size: 2560,
        num_layers: 48,
        quantization: "4-bit",
        context_length: 1048576,
        is_moe: true,
        drafter_loaded: false,
        drafter_path: null,
        mtp_loaded: true,
        kv_quant: "8",
        gen_temperature: 1,
        gen_top_p: 0.95,
        gen_top_k: 20,
      },
    },
  ],
}

/** Verbatim from ~/.mlx-serve/logs/mlx-serve-11234.log, trimmed to the fields we read. */
export const MTP_LINE =
  "  [spec-stats] mode=mtp attempts=168 accepts=191 avg_per_round=1.14 per_draft_pct=67.7% depth=6 drafted=282 ext_rounds=21 partial_rounds=54 runtime_disabled=false reason=none adaptive=mtp serial_cell=21.74 sync_ms=2.93 round_ms=47.31 two_ms_tok=13.24 one_ms_tok=11.00 verdict_round=15 trials=3 width_trials=4 table=128-256k:w1:22.73/390,w2:20.11/851 table_drops=t2029/c0/b0/i22 serial_drops=t22/c0/b0"

/** A real request where the adaptive controller gave up on speculation. */
export const GATED_LINE =
  "  [spec-stats] mode=mtp attempts=55 accepts=89 avg_per_round=1.62 per_draft_pct=38.4% depth=6 drafted=232 ext_rounds=1 partial_rounds=44 runtime_disabled=true reason=adaptive adaptive=serial serial_cell=25.63 sync_ms=3.60 round_ms=35.17 two_ms_tok=16.82 one_ms_tok=11.98 verdict_round=15 trials=2 width_trials=6 table=32-64k:w1:19.26/611 table_drops=t2006/c0/b0/i22 serial_drops=t22/c0/b0"

export const PLD_LINE = "  [spec-stats] mode=pld attempts=410 accepts=612 avg_per_round=1.49 runtime_disabled=false"

export const DFLASH_LINE =
  "  [spec-stats] mode=dflash attempts=90 accepts=201 avg_per_round=2.23 gate_min=1.50 per_draft_pct=74.4% block_size=4 partial_rounds=12 runtime_disabled=false table=128-256k:w1:22.73/390 table_drops=t1/c0/b0/i0 block_avg=3.10 block_hist=1:20,2:30 chooser_trials=4"

export const CHAT_LINE =
  "POST /v1/chat/completions (127 msgs, max_tokens=64000 (launch default), temp=1.00, top_p=0.95, top_k=20, stream=true, thinking=true, sys=18571b, user=760b, tools=13061b, tool_msgs=68) "

export const RESPONSES_LINE =
  "POST /v1/responses (3 msgs, max_out=4096, temp=0.70, stream=true, thinking=false, prev=null)"

/** Loose on purpose: the parsers must survive whatever the server sent. */
export function raw(value: unknown): RawMetricsJson {
  return value as RawMetricsJson
}

/** The live feed with wire-level overrides, parsed the way the TUI parses it. */
export function feed(
  overrides: { counters?: WireCounters; gauges?: WireGauges } = {},
): MetricsFeed {
  return parseFeed(
    raw({
      counters: { ...LIVE_FEED.counters, ...overrides.counters },
      gauges: { ...LIVE_FEED.gauges, ...overrides.gauges },
      histograms: LIVE_FEED.histograms as unknown as Record<string, unknown>,
    }),
  )
}

/** Verbatim KV-cache tier lines from ~/.mlx-serve/logs/mlx-serve-11234.log. */
export const HOT_TIER_LINE = "  [hot-cache] resident=9481.06 / 28672.00 MB (1/1 entries)"
export const SSD_TIER_LINE =
  "  [disk-cache] persisted 643681/643681 tokens (+1 chunks, 3 ssm-cp, 7.6 MB, 12ms); resident=10933.6 MB (12 entries)"
/** A partial persist with no ssm-cp copies: the same line, a different parenthetical. */
export const SSD_PARTIAL_LINE =
  "  [disk-cache] persisted 322560/524241 tokens (+1 chunks, 0 ssm-cp, 12.8 MB, 67ms); resident=38009.6 MB (48 entries)"
