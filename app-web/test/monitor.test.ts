import { describe, expect, it } from "vitest";
import { liveRates, makeSample, mergeDocs, modelTotals, rangeRows, rateSeries, trackRequests, windowTotals } from "../src/lib/core/monitor-history";

const feed = (n: number, gauges: Record<string, number> = {}) =>
  ({
    counters: { generation_tokens_total: n, requests_success_total: n, prefill_tokens_total: n },
    gauges: { process_start_time_seconds: 1, generation_tokens_live: n, prefill_tokens_live: 0, requests_running: 1, ...gauges },
    histograms: { prefill_time_seconds: { sum: 2, count: 1 } },
    sessions: [],
  }) as never;
const sample = (t: number, n: number, g: Record<string, number> = {}) => makeSample(t, feed(n, g));

describe("live rates", () => {
  it("use elapsed seconds, active counters and completed prefill time", () => {
    const current = feed(100, { prefill_tokens_live: 300 });
    expect(liveRates([{ t: 0, live: 20, pre: 100, req: 60 }], current, 4000)).toEqual({
      decode: 20,
      prefill: 50,
      requests: 10,
      averagePrefill: 50,
      live: 100,
      pre: 300,
    });
  });

  it("a page refresh and a live tick agree on idle and prefill completion", () => {
    const old = [{ t: 0, live: 20, pre: 100, req: 1 }];
    liveRates(old, feed(100, { prefill_tokens_live: 300 }), 4000);
    const idle = feed(100, { requests_running: 0, prefill_tokens_live: 0 });
    for (const samples of [old, []]) {
      const rates = liveRates(samples, idle, 5000);
      expect(rates.decode).toBe(0);
      expect(rates.prefill).toBe(0);
      expect(rates.averagePrefill).toBe(50);
    }
    // A fresh page has no delta baseline for an active decode; unknown is not a remembered rate.
    expect(liveRates([], feed(100), 5000).decode).toBeNull();
    expect(liveRates(old, feed(100), 5000).prefill).toBe(0);
    const history = [sample(0, 0), sample(1000, 10), sample(4000, 40)];
    expect(rateSeries(JSON.parse(JSON.stringify(history)), "generation_tokens_total", 0, 4000, 4)).toEqual(
      rateSeries(history, "generation_tokens_total", 0, 4000, 4),
    );
  });

  it("zero work yields zero; missing samples, zero elapsed and counter reset stay unknown", () => {
    const before = [{ t: 1000, live: 20, pre: 10, req: 5 }];
    expect(liveRates(before, feed(20), 2000).decode).toBe(0);
    expect(liveRates(before, feed(10), 2000).decode).toBeNull();
    expect(liveRates(before, feed(30), 1000).decode).toBeNull();
    expect(liveRates(before, feed(30), 302000).decode).toBeNull();
    expect(liveRates([], feed(30), 2000).decode).toBeNull();
  });
});

describe("history", () => {
  it("buckets never interpolate through gaps, restarts or decreases", () => {
    const before = sample(0, 10);
    for (const after of [sample(1000, 20, { process_start_time_seconds: 2 }), sample(1000, 1), sample(301000, 20)]) {
      expect(rateSeries([before, after], "generation_tokens_total", 0, after.t, 1)).toEqual([null]);
      expect(windowTotals([before, after], 0, after.t).generation_tokens_total).toBe(0);
    }
    expect(rateSeries([before, sample(300000, 20)], "generation_tokens_total", 0, 300000, 1)[0]).toBe(10 / 300);
    const missing = sample(1000, 20);
    (missing.c as Record<string, number | null>).generation_tokens_total = null;
    expect(rateSeries([before, missing], "generation_tokens_total", 0, 1000, 1)).toEqual([null]);
  });

  it("rates use counter deltas and measured time, not poll counts", () => {
    const samples = [sample(0, 0), sample(1000, 10), sample(4000, 40)];
    expect(rateSeries(samples, "generation_tokens_total", 0, 4000, 4)).toEqual([10, null, null, 10]);
    expect(windowTotals(samples, 1000, 4000).generation_tokens_total).toBe(30);
  });

  it("merged history deduplicates tabs, retains 24 hours and preserves gap/reset edges", () => {
    const now = 86400000;
    const samples = [sample(-1000, 0), sample(1000, 1), sample(2000, 2), sample(3000, 0), sample(4000, 1), sample(5000, 2), sample(600000, 3), sample(now - 1000, 4), sample(now, 5)];
    const merged = mergeDocs({ samples, rows: [] } as never, { samples: [sample(now, 5)], rows: [] } as never, now);
    expect(merged.samples.map((s) => s.t)).toEqual([1000, 2000, 3000, 5000, 600000, now - 1000, now]);
  });

  it("sessions track observed request ids, close departures, and exclude cached slots", () => {
    const s = { model: "m", phase: "prefill", request_id: 9, client: "codex", context_tokens: 10, context_length: 100, cached_tokens: 0, generated_tokens: 0 };
    let rows = trackRequests([], [s, { ...s, request_id: 0, phase: "cached" }] as never, 1000);
    rows = trackRequests(rows, [{ ...s, phase: "decode", generated_tokens: 5 }] as never, 2000);
    expect(rows.length).toBe(1);
    expect(rows[0]!.generated).toBe(5);
    expect(rows[0]!.endT).toBeNull();
    rows = trackRequests(rows, [], 3000);
    expect(rows[0]!.endT).toBe(3000);
    expect(rangeRows(rows, 1500, 2500).length).toBe(1);
    expect(rangeRows(rows, 3001, 4000).length).toBe(0);
  });

  it("per-model totals attribute completed deltas to the departing model", () => {
    const a = sample(0, 0), b = sample(1000, 10), c = sample(2000, 20);
    a.m = "alpha";
    c.m = "beta";
    expect(modelTotals([a, b, c], 0, 2000).alpha!.generation_tokens_total).toBe(10);
    expect(modelTotals([a, b, c], 0, 2000).beta!.generation_tokens_total).toBe(10);
  });
});
