import { t } from "../i18n/i18n";
import { StudioError } from "./client";
import type { Client } from "./client";
export type Sample = { labels: Record<string, string>; value: number; };
export type Metrics = { types: Map<string, string>; samples: Map<string, Sample[]>; };
/** Prometheus text format; ignore comments and malformed individual samples. */
function parseMetrics(text: string): Metrics {
  const types = new Map(),
    samples = new Map();
  for (const line of text.split(/\r?\n/)) {
    const type = line.match(
      /^# TYPE ([\w:]+) (counter|gauge|histogram|untyped)\s*$/,
    );
    if (type) {
      types.set(type[1], type[2]);
      continue;
    }
    const match = line.match(
      /^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)(?:\s+\d+)?\s*$/,
    );
    if (!match) continue;
    const labels: Record<string, string> = Object.create(null);
    let rest = match[2]?.trim() ?? "",
      valid = true;
    while (rest) {
      const label = rest.match(
        /^([a-zA-Z_]\w*)\s*=\s*"((?:[^"\\]|\\[\\"n])*)"\s*(,\s*|$)/,
      );
      if (!label || Object.hasOwn(labels, label[1])) {
        valid = false;
        break;
      }
      labels[label[1]] = label[2].replace(/\\([\\"n])/g, (_, c) =>
        c === "n" ? "\n" : c,
      );
      rest = rest.slice(label[0].length);
    }
    if (!valid) continue;
    const value = match[3].endsWith("Inf")
      ? match[3][0] === "-"
        ? -Infinity
        : Infinity
      : Number(match[3]);
    const group = samples.get(match[1]) ?? [];
    group.push({ labels, value });
    samples.set(match[1], group);
  }
  return { types, samples };
}
function metricValue(m: Metrics, name: string): number | null {
  const samples = m.samples.get(name);
  return samples?.length && samples.every((s) => Number.isFinite(s.value))
    ? samples.reduce((n, s) => n + s.value, 0)
    : null;
}
const seriesKey = (sample: Sample) =>
  JSON.stringify(
    Object.entries(sample.labels).sort(([a], [b]) => a.localeCompare(b)),
  );
/** No extrapolation across resets, missing series, pauses or failed scrapes. */
function counterRate(previous: Metrics | null, current: Metrics, name: string, elapsedMs: number) {
  if (!previous || !Number.isFinite(elapsedMs) || elapsedMs <= 0) return null;
  if (
    metricValue(previous, "mlx_serve:process_start_time_seconds") !==
    metricValue(current, "mlx_serve:process_start_time_seconds")
  )
    return null;
  const before = previous.samples.get(name),
    after = current.samples.get(name);
  if (!before?.length || before.length !== after?.length) return null;
  const old = new Map(before.map((s) => [seriesKey(s), s.value]));
  let delta = 0;
  for (const sample of after) {
    const value = old.get(seriesKey(sample));
    if (
      value === undefined ||
      !Number.isFinite(value) ||
      !Number.isFinite(sample.value) ||
      value < 0 ||
      sample.value < value
    )
      return null;
    delta += sample.value - value;
  }
  return delta / (elapsedMs / 1000);
}
/**
 * Linear estimate from cumulative, nonnegative histogram buckets. Infinite tail
 * is bounded by the last finite bucket. Values describe the server lifetime.
 */
function histogramQuantile(m: Metrics, name: string, q: number): number | null {
  const samples = m.samples.get(name + "_bucket");
  if (!samples?.length || q < 0 || q > 1 || !Number.isFinite(q)) return null;
  const buckets = new Map();
  for (const { labels, value } of samples) {
    const bound = labels.le === "+Inf" ? Infinity : Number(labels.le);
    if (
      Number.isNaN(bound) ||
      bound < 0 ||
      !Number.isFinite(value) ||
      value < 0
    )
      return null;
    buckets.set(bound, (buckets.get(bound) ?? 0) + value);
  }
  const sorted = [...buckets].sort(([a], [b]) => a - b),
    total = buckets.get(Infinity);
  if (!total || sorted.length < 2) return null;
  let lastCount = 0;
  for (const [, count] of sorted) {
    if (count < lastCount) return null;
    lastCount = count;
  }
  const rank = q * total;
  let lower = 0,
    countBefore = 0;
  for (const [upper, count] of sorted) {
    if (count >= rank) {
      if (upper === Infinity) return lower;
      if (count === countBefore) return upper;
      return (
        lower + ((upper - lower) * (rank - countBefore)) / (count - countBefore)
      );
    }
    lower = upper;
    countBefore = count;
  }
  return null;
}
async function fetchMetrics(client: Client, signal?: AbortSignal) {
  const bytes = await client.bytes(
    "/metrics",
    { headers: { Accept: "text/plain" } },
    { signal, timeoutMs: 5000 },
    2 * 1024 * 1024,
  );
  const metrics = parseMetrics(new TextDecoder().decode(bytes));
  if (
    ![...metrics.samples.keys()].some(
      (n) => n.startsWith("mlx_serve:") || n.startsWith("vllm:"),
    )
  )
    throw new StudioError(
      "unsupported",
      t("Metrics are disabled or unavailable on this server."),
    );
  return metrics;
}
/**
 * Sequential polling; the owner starts/stops collection, independent of pane visibility.
 */
class MetricsPoller<T> {
  visible = false;
  stopped = false;
  pending: AbortController | undefined;
  timer: ReturnType<typeof setTimeout> | undefined;
  onError: (error: unknown) => void;
  onValue: (value: T) => void;
  load: (signal: AbortSignal) => Promise<T>;
  interval: number;
  constructor(load: (signal: AbortSignal) => Promise<T>, onValue: (value: T) => void, onError: (error: unknown) => void, interval: number = 1000) {
    this.interval = interval;
    this.load = load;
    this.onValue = onValue;
    this.onError = onError;
  }
  setVisible(visible: boolean) {
    if (this.stopped || visible === this.visible) return;
    this.visible = visible;
    clearTimeout(this.timer);
    this.pending?.abort();
    this.pending = undefined;
    if (visible) void this.poll();
  }
  async poll() {
    const started = Date.now();
    const controller = new AbortController();
    this.pending = controller;
    try {
      const value = await this.load(controller.signal);
      if (!controller.signal.aborted) this.onValue(value);
    } catch (error) {
      if (!controller.signal.aborted) this.onError(error);
    } finally {
      if (this.pending === controller) {
        this.pending = undefined;
        if (this.visible && !this.stopped)
          this.timer = setTimeout(
            () => void this.poll(),
            Math.max(0, this.interval - (Date.now() - started)),
          );
      }
    }
  }
  stop() {
    this.setVisible(false);
    this.stopped = true;
  }
}

export { parseMetrics, metricValue, counterRate, histogramQuantile, fetchMetrics, MetricsPoller };
