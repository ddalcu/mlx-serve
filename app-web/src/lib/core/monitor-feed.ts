import { t } from "../i18n/i18n";
import { record, StudioError } from "./client";
import { fetchMetrics, metricValue } from "./metrics";
import { HISTORY_KEYS, number } from "./monitor-history";
import type { Client } from "./client";
import type { Feed } from "./monitor-history";
const GAUGES = [
  "requests_running",
  "requests_waiting",
  "gpu_utilization_pct",
  "memory_mb",
  "process_start_time_seconds",
  "generation_tokens_live",
  "prefill_tokens_live",
  "prefill_tokens_expected",
  "requests_prefilling",
  "mlx_active_bytes",
  "mlx_cache_bytes",
  "batched_group_size",
];
const HISTOGRAMS = [
  "time_to_first_token_seconds",
  "e2e_request_latency_seconds",
  "prefill_time_seconds",
  "decode_time_seconds",
];
const names: Record<string, string> = {
  prompt_tokens_total: "vllm:prompt_tokens_total",
  generation_tokens_total: "vllm:generation_tokens_total",
  requests_success_total: "vllm:request_success_total",
  requests_cancelled_total: "vllm:request_cancelled_total",
  requests_failed_total: "mlx_serve:request_failed_total",
  requests_rejected_total: "mlx_serve:request_rejected_total",
  prefix_cache_queries_total: "vllm:prefix_cache_queries_total",
  prefix_cache_hits_total: "vllm:prefix_cache_hits_total",
  requests_running: "vllm:num_requests_running",
  requests_waiting: "vllm:num_requests_waiting",
  prefill_time_seconds: "vllm:request_prefill_time_seconds",
  decode_time_seconds: "vllm:request_decode_time_seconds",
};
/**
 * JSON is authoritative; text may only supplement absent fields. Access failures
 * on the JSON route are not permission to try another route.
 */
async function fetchMonitorFeed(client: Client, signal?: AbortSignal): Promise<Feed> {
  const bytes = await client.bytes(
    "/metrics.json",
    { cache: "no-store", headers: { Accept: "application/json" } },
    { signal, timeoutMs: 5000 },
    2 * 1024 * 1024,
  );
  let raw;
  try {
    raw = record(JSON.parse(new TextDecoder().decode(bytes)));
  } catch {
    throw new StudioError(
      "unsupported",
      t("Metrics are disabled or unavailable on this server."),
    );
  }
  const counters = Object.fromEntries(
      HISTORY_KEYS.map((k) => [k, number(record(raw.counters)[k])]),
    ),
    gauges = Object.fromEntries(
      GAUGES.map((k) => [k, number(record(raw.gauges)[k])]),
    );
  const histograms = Object.fromEntries(
    HISTOGRAMS.map((k) => {
      const h = record(record(raw.histograms)[k]);
      return [k, { sum: number(h.sum), count: number(h.count) }];
    }),
  );
  if (
    [
      ...Object.values(counters),
      ...Object.values(gauges),
      ...Object.values(histograms).flatMap((h) => [h.sum, h.count]),
    ].some((v) => v === null)
  ) {
    try {
      const text = await fetchMetrics(client, signal);
      for (const group of [counters, gauges])
        for (const k of Object.keys(group))
          if (group[k] === null)
            group[k] = number(metricValue(text, names[k] ?? "mlx_serve:" + k));
      for (const k of HISTOGRAMS)
        for (const part of (["sum", "count"] as const))
          if (histograms[k][part] === null)
            histograms[k][part] = number(
              metricValue(text, (names[k] ?? "vllm:" + k) + "_" + part),
            );
    } catch (error) {
      if (signal?.aborted) throw error;
    }
  }
  if (
    ![...Object.values(counters), ...Object.values(gauges)].some(
      (v) => v !== null,
    )
  )
    throw new StudioError(
      "unsupported",
      t("Metrics are disabled or unavailable on this server."),
    );
  const sessions = Array.isArray(raw.sessions)
    ? raw.sessions.flatMap((s) => {
        const r = record(s);
        if (
          typeof r.model !== "string" ||
          typeof r.phase !== "string" ||
          number(r.request_id) === null
        )
          return [];
        return [
          {
            request_id: Number(r.request_id),
            model: r.model,
            phase: r.phase,
            client: typeof r.client === "string" ? r.client : undefined,
            ...Object.fromEntries(
              [
                "context_tokens",
                "context_length",
                "cached_tokens",
                "generated_tokens",
                "state_bytes",
              ].map((k) => [k, number(r[k]) ?? undefined]),
            ),
          },
        ];
      })
    : null;
  return { counters, gauges, histograms, sessions };
}

export { fetchMonitorFeed };
