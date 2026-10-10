import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioError } from "../src/lib/core/client";
import type { Feed } from "../src/lib/core/monitor-history";
import { mockApi } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => {
  ui?.unmount();
  vi.useRealTimers();
});

const START = Date.UTC(2026, 9, 8, 12, 0, 0);
/** A feed `seconds` into a run that decodes 50 tok/s with one request open. */
function feed(seconds: number, extra: Partial<Feed> = {}): Feed {
  return {
    counters: { requests_success_total: seconds, requests_failed_total: 0, requests_rejected_total: 0, requests_cancelled_total: 0, generation_tokens_total: seconds * 50, prefill_tokens_total: 0, prompt_tokens_total: 0, prefix_cache_queries_total: 0, prefix_cache_hits_total: 0, prefix_cache_tokens_total: 0 },
    gauges: { requests_running: 1, requests_waiting: 0, process_start_time_seconds: 1000, generation_tokens_live: seconds * 50, prefill_tokens_live: 0, requests_prefilling: 0 },
    histograms: {},
    sessions: [{ request_id: 1, model: "m/chat", phase: "decode", context_tokens: 100, context_length: 400, generated_tokens: seconds * 50 }],
    ...extra,
  };
}
const home = () => ui!.app.monitor;

describe("MonitorWorkspace", () => {
  it("turns a run of feeds into live rates, a chart and history", async () => {
    ui = await mountApp();
    vi.useFakeTimers({ toFake: ["Date"] });
    for (const s of [0, 1, 2, 3]) {
      vi.setSystemTime(START + s * 1000);
      home().accept(feed(s));
    }
    expect(home().status).toBe("");
    expect(home().rates?.decode).toBeCloseTo(50, 5);
    expect(home().doc.samples.length).toBe(4);
    expect(home().liveChart.map((p) => p.decode).slice(-1)).toEqual([expect.closeTo(50, 5)]);
    expect(home().doc.rows.map((r) => [r.model, r.endT]).flat()).toEqual(["m/chat", null]);
  });

  it("closes a request row when it stops being reported", async () => {
    ui = await mountApp();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(START);
    home().accept(feed(0));
    vi.setSystemTime(START + 1000);
    home().accept(feed(1, { sessions: [] }));
    expect(home().doc.rows[0]!.endT).toBe(START + 1000);
  });

  it("forgets live numbers when the server restarts under it", async () => {
    ui = await mountApp();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(START);
    home().accept(feed(0));
    vi.setSystemTime(START + 1000);
    const restarted = feed(1);
    restarted.gauges.process_start_time_seconds = 2000;
    restarted.counters.generation_tokens_total = 0;
    restarted.gauges.generation_tokens_live = 0;
    home().accept(restarted);
    expect(home().rates?.decode).toBe(null);
  });

  describe("why metrics are unavailable", () => {
    const failure = (status: number | undefined, kind = "http") => new StudioError(kind as never, "x", status);
    it.each([
      [403, "", "Monitoring needs an API key on this server (Settings → Server → API key)."],
      [403, "k", "Monitoring access was denied. Check the API key and server access policy in Settings."],
      [401, "", "Monitoring authentication failed. Check the API key in Settings."],
      [undefined, "", "Cannot reach server metrics. Check the server connection. Retrying every second."],
    ])("%s with key %j says %s", async (status, key, message) => {
      ui = await mountApp();
      home().server = { ...home().server!, apiKey: key };
      home().failed(failure(status));
      expect(home().status).toBe(message);
      expect(home().feed).toBe(null);
    });

    it("tells the page's own server to be started with --metrics, and stops asking", async () => {
      const api = mockApi();
      ui = await mountApp({ api });
      await vi.waitFor(() => expect(home().status).toMatch(/Start mlx-serve with --metrics/));
      const asked = () => api.requests.filter((r) => r.path === "/metrics.json").length;
      const before = asked();
      await new Promise((r) => setTimeout(r, 1300));
      expect(asked()).toBe(before);
    });

    it("keeps asking a server that is not the page's own", async () => {
      ui = await mountApp();
      home().server = { ...home().server!, url: "http://elsewhere:9" };
      home().failed(failure(503, "unsupported"));
      expect(home().status).toBe("Metrics are disabled or unavailable on this server.");
    });
  });

  it("polls a server that has metrics and shows its models", async () => {
    const api = mockApi();
    api.state.metrics = { counters: { requests_success_total: 3 }, gauges: { requests_running: 0 }, sessions: [] };
    ui = await mountApp({ api });
    await vi.waitFor(() => expect(home().feed?.counters.requests_success_total).toBe(3));
    expect(home().status).toBe("");
    await vi.waitFor(() => expect(home().models?.map((m) => m.id)).toEqual(["m/chat", "m/other"]));
  });
});
