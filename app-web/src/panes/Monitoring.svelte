<script lang="ts">
  import ChartCard from "../components/ChartCard.svelte";
  import DataTable, { type Cell } from "../components/DataTable.svelte";
  import Meter from "../components/Meter.svelte";
  import type { App } from "../lib/app.svelte";
  import { number, modelTotals, rangeRows, rateSeries, windowTotals, type Numbers } from "../lib/core/monitor-history";
  import { bytes, fmt, level } from "../lib/format";
  import { displayName, N, t } from "../lib/i18n/i18n";
  import { RANGES } from "../lib/state/monitor-workspace.svelte";

  let { app }: { app: App } = $props();
  const ws = $derived(app.monitor);
  const connection = $derived(app.connection);

  const cards = [
    ["decode", N("Decode"), "tok/s"],
    ["prefill", N("Prefill"), "tok/s"],
    ["running", N("Running requests"), ""],
    ["ttft", N("Time to first token"), N("ms avg")],
    ["cache", N("Cache hit rate"), "%"],
    ["gpu", N("GPU utilization"), "%"],
    ["memory", N("Process memory"), "MB"],
    ["generated", N("Generated total"), "tokens"],
  ] as const;

  const c = $derived(ws.feed?.counters ?? {});
  const g = $derived(ws.feed?.gauges ?? {});
  const h = $derived(ws.feed?.histograms ?? {});
  const rates = $derived(ws.rates);
  const avg = (key: string) => {
    const v = h[key];
    return v?.sum != null && v.count != null && v.count > 0 ? (v.sum / v.count) * 1000 : null;
  };
  const cache = $derived(c.prefix_cache_queries_total != null && c.prefix_cache_queries_total > 0 && c.prefix_cache_hits_total != null ? (100 * c.prefix_cache_hits_total) / c.prefix_cache_queries_total : null);
  const reuse = $derived(c.prompt_tokens_total != null && c.prompt_tokens_total > 0 && c.prefix_cache_tokens_total != null ? (100 * c.prefix_cache_tokens_total) / c.prompt_tokens_total : null);
  const values = $derived<Record<string, number | null | undefined>>({
    decode: rates?.decode,
    prefill: rates?.prefill,
    running: g.requests_running,
    ttft: avg("time_to_first_token_seconds"),
    cache,
    gpu: g.gpu_utilization_pct,
    memory: g.memory_mb,
    generated: rates?.live,
  });
  const details = $derived<Record<string, string>>({
    decode: t("%@ ms avg · %@", [fmt(avg("decode_time_seconds")), g.generation_tokens_live == null ? t("completed-token fallback") : t("includes in-flight tokens")]),
    prefill: t("%@%@ tok/s avg · %@ ms avg", [
      (g.requests_prefilling ?? 0) > 0 ? t("Prefilling · ") + fmt(g.prefill_tokens_live) + t(" forwarded · ") : "",
      fmt(rates?.averagePrefill),
      fmt(avg("prefill_time_seconds")),
    ]),
    running: t("%@ waiting · %@ req/s%@ · %@ cancelled", [fmt(g.requests_waiting), fmt(rates?.requests), (g.batched_group_size ?? 0) > 1 ? t(" · batch of ") + g.batched_group_size : "", fmt(c.requests_cancelled_total)]),
    ttft: t("%@ ms end-to-end avg · since startup", [fmt(avg("e2e_request_latency_seconds"))]),
    cache: t("%@ / %@ queries · %@% tokens reused", [fmt(c.prefix_cache_hits_total), fmt(c.prefix_cache_queries_total), fmt(reuse)]),
    gpu: values.gpu == null ? t("Current utilization unavailable") : level(values.gpu) === "normal" ? t("Current utilization") : level(values.gpu) === "critical" ? t("Critical (≥90%)") : t("High (≥70%)"),
    memory: t("Physical footprint · MLX %@ active · %@ pool", [bytes(g.mlx_active_bytes), bytes(g.mlx_cache_bytes)]),
    generated: t("%@ successful requests", [fmt(c.requests_success_total)]),
  });
  const prefillBar = $derived((g.requests_prefilling ?? 0) > 0 && g.prefill_tokens_expected != null && g.prefill_tokens_expected > 0 && g.prefill_tokens_live != null ? (100 * g.prefill_tokens_live) / g.prefill_tokens_expected : null);
  const badge = $derived(ws.feed ? t("Live") : !ws.status || ws.status === "Connecting to metrics…" ? t("Connecting") : t("Unavailable"));
  const phase = (name: string) => ({ prefill: t("Prefilling"), decode: t("Decoding"), cached: t("In cache") })[name] ?? name;

  const now = $derived(Math.floor(ws.now / 1000) * 1000);
  const from = $derived(now - ws.range);
  const samples = $derived(ws.doc.samples);
  const live = $derived(ws.range === 60000);
  const series = (key: string) => rateSeries(samples, key, from, now, 60);
  const liveSeries = (key: "decode" | "prefill") => Array.from({ length: 60 }, (_, i) => ws.liveChart.find((s) => s.t === from + (i + 1) * 1000)?.[key] ?? null);
  const failed = $derived(["requests_failed_total", "requests_rejected_total", "requests_cancelled_total"].map(series));
  const errors = $derived(failed[0]!.map((_, i) => (failed.some((s) => s[i] === null) ? null : failed.reduce((a, s) => a + (s[i] as number), 0) * 60)));
  const hint = (id: string) =>
    t("%@ · point at a chart or use ← / → to inspect", [ws.range === 60000 && (id === "decode" || id === "prefill") ? t("Live phase rates, one-second samples") : t("Completed-counter rates over wall time, 60 buckets")]);
  const summary = (totals: Numbers) => t("%@ ok · %@ failed · %@ rejected · %@ cancelled", [fmt(totals.requests_success_total), fmt(totals.requests_failed_total), fmt(totals.requests_rejected_total), fmt(totals.requests_cancelled_total)]);
  const sessionRows = $derived(
    (ws.feed?.sessions ?? [])
      .slice()
      .sort((a, b) => a.model.localeCompare(b.model) || Number(a.phase === "cached") - Number(b.phase === "cached"))
      .map((s): Cell[] => [
        s.model,
        phase(s.phase),
        {
          text: `${fmt(s.context_tokens)}${s.context_length ? ` / ${fmt(s.context_length)} · ${fmt(Math.min(100, (100 * (s.context_tokens ?? 0)) / s.context_length))}%` : ""}`,
          ...(s.context_length && s.context_tokens != null ? { meter: (100 * s.context_tokens) / s.context_length } : {}),
        },
        fmt(s.cached_tokens),
        s.phase === "cached" ? "—" : fmt(s.generated_tokens),
        bytes(s.state_bytes),
      ]),
  );
  const modelRows = $derived.by(() => {
    const byModel = modelTotals(samples, from, now);
    return Object.keys(byModel)
      .sort()
      .map((k) => [k, fmt(byModel[k]!.requests_success_total), fmt(byModel[k]!.generation_tokens_total), fmt(byModel[k]!.prefill_tokens_total)]);
  });
  const requestRows = $derived(
    rangeRows(ws.doc.rows, from, now).map((r) => [
      new Date(r.startT).toLocaleString(),
      r.client,
      r.model,
      r.endT === null ? (now - r.lastT > 2000 ? t("Last seen ") + phase(r.phase) : phase(r.phase)) : t("No longer observed"),
      fmt(((r.endT ?? r.lastT) - r.startT) / 1000) + " s",
      fmt(r.ctx),
      fmt(r.cached),
      fmt(r.generated),
    ]),
  );
  const discovered = $derived(
    (ws.models ?? [])
      .slice()
      .sort((a, b) => Number(b.state === "ready" || b.loaded === true) - Number(a.state === "ready" || a.loaded === true) || String(a.id).localeCompare(String(b.id)))
      .map((m) => [
        String(m.id),
        Array.isArray(m.capabilities) ? m.capabilities.filter((x) => typeof x === "string").map(displayName).join(", ") : "—",
        bytes(number(m.bytes_resident) || number(m.bytes_on_disk)),
        typeof m.state === "string" ? displayName(m.state) : m.loaded === true ? t("Loaded") : m.loaded === false ? t("Unloaded") : t("Not reported"),
      ]),
  );
</script>

<section class="monitoring-screen" aria-label={t("Monitoring")}>
  <div class="monitoring-heading">
    <div>
      <h1>{t("Monitoring")}</h1>
      <p class="section-description" id="monitor-server">{connection.serverName(ws.server?.name)} · {ws.server?.url}</p>
    </div>
    <span class="monitor-badge" id="monitor-badge">{badge}</span>
  </div>
  <p id="monitor-status" role="status">{t(ws.status)}</p>
  <div id="monitor-dashboard">
    <div class="monitor-grid">
      {#each cards as [id, label, unit]}
        <section class="metric-card">
          <h2>{t(label)}</h2>
          <p class="metric-reading"><strong id="metric-{id}">{fmt(values[id])}</strong> <span>{t(unit)}</span></p>
          <p class="metric-detail" id="detail-{id}">{details[id]}</p>
          <div id="bar-{id}">
            {#if id === "gpu"}<Meter value={g.gpu_utilization_pct} />{:else if id === "prefill"}<Meter value={prefillBar} />{/if}
          </div>
        </section>
      {/each}
    </div>
    <div class="monitor-range" role="group" aria-label={t("Monitoring range")}>
      {#each RANGES as [label, value]}<button aria-pressed={value === ws.range} onclick={() => (ws.range = value)}>{t(label)}</button>{/each}
    </div>
    <p id="monitor-chart-note" class="section-description">{new Date(from).toLocaleTimeString()} – {new Date(now).toLocaleTimeString()}</p>
    <div class="monitor-charts">
      <ChartCard id="decode" label={live ? t("Decode tok/s") : t("Generated tok/s")} values={live ? liveSeries("decode") : series("generation_tokens_total")} {from} to={now} description={hint("decode")} />
      <ChartCard id="prefill" label={t("Prefill tok/s")} values={live ? liveSeries("prefill") : series("prefill_tokens_total")} {from} to={now} description={hint("prefill")} />
      <ChartCard id="requests" label={t("Successful requests / min")} values={series("requests_success_total").map((n) => (n === null ? null : n * 60))} {from} to={now} description={hint("requests")} />
      <ChartCard id="errors" label={t("Failed + rejected + cancelled / min")} values={errors} {from} to={now} description={hint("errors")} />
    </div>
    <p id="monitor-totals" class="monitor-totals">{t("In this window (observed): ") + (samples.length > 1 ? summary(windowTotals(samples, from, now)) : "—")}</p>
    <p id="monitor-since" class="monitor-totals">{t("Since server startup: ") + summary(ws.feed?.counters ?? {})}</p>
    <section class="monitor-section">
      <h2>{t("Sessions")} <small>{t("Live")}</small></h2>
      <DataTable
        id="monitor-sessions"
        label={t("Live sessions")}
        heads={[t("Model"), t("Phase"), t("Context"), t("Cached"), t("Generated"), t("KV + state")]}
        rows={sessionRows}
        empty={ws.feed?.sessions == null ? t("Live sessions unavailable.") : t("No sessions.")}
      />
    </section>
    <section class="monitor-section">
      <h2>{t("By model")}</h2>
      <DataTable id="monitor-by-model" label={t("By model")} heads={[t("Model"), t("Requests"), t("Generated"), t("Prefill")]} rows={modelRows} empty={t("No observed requests in this range.")} />
    </section>
    <section class="monitor-section">
      <h2>{t("Request history")}</h2>
      <DataTable
        id="monitor-requests"
        label={t("Request history")}
        heads={[t("Started"), t("Client"), t("Model"), t("Phase / observation"), t("Duration"), t("Context"), t("Cached"), t("Generated")]}
        rows={requestRows}
        empty={t("No observed requests in this range.")}
      />
    </section>
  </div>
  <details class="monitor-notes">
    <summary>{t("History is collected in this browser while the page is open.")}</summary>
    <p class="monitor-note">{t("Collects every second while this page is open, including other panes. Hidden tabs may be throttled. History covers observed data only; the server stores no history. Short outages may be averaged; gaps over 5 minutes and restarts are not bridged. Samples are kept for 24 hours, thinned to minutes after 1 hour. Requests shorter than a poll appear only in totals. Model attribution is approximate under concurrency. “No longer observed” does not identify a request’s outcome; token counts are the last observation. Latest 50 matching requests shown (up to 200 closed rows retained).")}</p>
    <p id="monitor-storage" class="section-description">
      {ws.where === "memory" ? t("History is in memory only and will be lost on reload.") : ws.where === "Checking storage…" ? t(ws.where) : t("History is stored in this browser (%@), separately for each server URL and mount.", [ws.where])}
    </p>
  </details>
  <section class="monitor-section">
    <h2>{t("Models")}</h2>
    <DataTable
      id="monitor-models"
      label={t("Discovered models")}
      heads={[t("Model"), t("Capabilities"), t("Size"), t("State")]}
      rows={discovered}
      empty={ws.models === null ? t("Models unavailable. Retrying…") : t("No models discovered.")}
    />
  </section>
</section>
