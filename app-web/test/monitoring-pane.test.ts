import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Feed } from "../src/lib/core/monitor-history";
import { mockApi } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => {
  ui?.unmount();
  vi.useRealTimers();
});
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";
const START = Date.UTC(2026, 9, 8, 12, 0, 0);

function feed(seconds: number, extra: Partial<Feed> = {}): Feed {
  return {
    counters: { requests_success_total: seconds, requests_failed_total: 1, requests_rejected_total: 0, requests_cancelled_total: 2, generation_tokens_total: seconds * 50, prefill_tokens_total: 800, prompt_tokens_total: 1000, prefix_cache_queries_total: 10, prefix_cache_hits_total: 4, prefix_cache_tokens_total: 250 },
    gauges: { requests_running: 1, requests_waiting: 2, gpu_utilization_pct: 95, memory_mb: 2048, process_start_time_seconds: 1000, generation_tokens_live: seconds * 50, prefill_tokens_live: 0, requests_prefilling: 0, mlx_active_bytes: 3 * 1024 ** 3, mlx_cache_bytes: 512 * 1024 ** 2 },
    histograms: { time_to_first_token_seconds: { sum: 2, count: 4 }, decode_time_seconds: { sum: 1, count: 2 } },
    sessions: [
      { request_id: 1, model: "m/chat", phase: "decode", context_tokens: 100, context_length: 400, generated_tokens: 12, state_bytes: 1024 ** 3 },
      { request_id: 2, model: "<img src=x onerror=alert(1)>", phase: "cached", context_tokens: 50, cached_tokens: 50 },
    ],
    ...extra,
  };
}

async function monitoring(api = mockApi()) {
  ui = await mountApp({ hash: "#monitoring", api });
  vi.useFakeTimers({ toFake: ["Date"] });
  for (const s of [0, 1, 2, 3]) {
    vi.setSystemTime(START + s * 1000);
    ui.app.monitor.accept(feed(s));
  }
  flushSync();
  return ui;
}

describe("Monitoring pane", () => {
  it("shows the live cards", async () => {
    const ui = await monitoring();
    expect(text(ui.q("#monitor-badge"))).toBe("Live");
    expect(text(ui.q("#monitor-status"))).toBe("");
    expect(text(ui.q("#metric-decode"))).toBe("50");
    expect(text(ui.q("#metric-running"))).toBe("1");
    expect(text(ui.q("#metric-ttft"))).toBe("500");
    expect(text(ui.q("#metric-cache"))).toBe("40");
    expect(text(ui.q("#metric-memory"))).toBe("2,048");
    expect(text(ui.q("#detail-decode"))).toBe("500 ms avg · includes in-flight tokens");
    expect(text(ui.q("#detail-running"))).toBe("2 waiting · 1 req/s · 2 cancelled");
    expect(text(ui.q("#detail-cache"))).toBe("4 / 10 queries · 25% tokens reused");
    expect(text(ui.q("#detail-memory"))).toBe("Physical footprint · MLX 3 GiB active · 512 MiB pool");
  });

  it("colors the GPU meter by how loaded it is", async () => {
    const ui = await monitoring();
    expect(text(ui.q("#detail-gpu"))).toBe("Critical (≥90%)");
    const meter = ui.q<HTMLMeterElement>("#bar-gpu meter")!;
    expect([meter.getAttribute("data-level"), meter.getAttribute("value")]).toEqual(["critical", "95"]);
    expect(ui.q("#bar-prefill meter")!.hasAttribute("hidden")).toBe(true);
  });

  it("lists sessions with a context meter and shows hostile model names as text", async () => {
    const ui = await monitoring();
    const rows = ui.qa("#monitor-sessions tbody tr");
    expect(rows.map((r) => [...r.children].map(text))).toEqual([
      ["<img src=x onerror=alert(1)>", "In cache", "50", "50", "—", "—"],
      ["m/chat", "Decoding", "100 / 400 · 25%", "—", "12", "1 GiB"],
    ]);
    expect(ui.q("#monitor-sessions img")).toBeNull();
    expect(rows[1]!.querySelector("meter")?.getAttribute("value")).toBe("25");
  });

  it("charts the last minute and reads a point with the arrow keys", async () => {
    const ui = await monitoring();
    const chart = ui.q("#chart-decode")!;
    expect(chart.getAttribute("role")).toBe("img");
    expect(text(ui.q("#chart-label-decode"))).toBe("Decode tok/s");
    const before = text(ui.q("#chart-value-decode"));
    expect(before).toMatch(/· 50$/);
    chart.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
    flushSync();
    expect(text(ui.q("#chart-value-decode"))).not.toBe(before);
    expect(chart.querySelectorAll("path").length).toBe(2);
  });

  it("switches the window and relabels the charts", async () => {
    const ui = await monitoring();
    const buttons = ui.qa<HTMLButtonElement>(".monitor-range button");
    expect(buttons.map((b) => [text(b), b.getAttribute("aria-pressed")])).toEqual([["Live", "true"], ["10m", "false"], ["30m", "false"], ["1h", "false"], ["24h", "false"]]);
    await ui.click(buttons[1]!);
    expect(buttons.map((b) => b.getAttribute("aria-pressed"))).toEqual(["false", "true", "false", "false", "false"]);
    expect(text(ui.q("#chart-label-decode"))).toBe("Generated tok/s");
    expect(text(ui.q("#monitor-totals"))).toContain("ok");
  });

  it("totals what it observed and what the server has done since startup", async () => {
    const ui = await monitoring();
    expect(text(ui.q("#monitor-since"))).toBe("Since server startup: 3 ok · 1 failed · 0 rejected · 2 cancelled");
    expect(text(ui.q("#monitor-totals"))).toMatch(/^In this window \(observed\): 3 ok/);
    expect(ui.qa("#monitor-by-model tbody tr").length).toBeGreaterThan(0);
  });

  it("keeps a table for requests it saw, with the model as plain text", async () => {
    const ui = await monitoring();
    const hostile = { request_id: 7, model: "<img src=x onerror=alert(1)>", client: "<b>cli</b>", phase: "prefill", context_tokens: 9 };
    vi.setSystemTime(START + 4000);
    ui.app.monitor.accept(feed(4, { sessions: [hostile] }));
    flushSync();
    const rows = ui.qa("#monitor-requests tbody tr").map((r) => [...r.children].map(text));
    expect(rows.map((r) => [r[1], r[2], r[3]])).toEqual([
      ["<b>cli</b>", "<img src=x onerror=alert(1)>", "Prefilling"],
      ["—", "m/chat", "No longer observed"],
    ]);
    expect(ui.q("#monitor-requests img, #monitor-requests b")).toBeNull();
  });

  it("explains an unavailable feed and clears the live numbers", async () => {
    const ui = await monitoring();
    ui.app.monitor.failed(new Error("down"));
    flushSync();
    expect(text(ui.q("#monitor-badge"))).toBe("Unavailable");
    expect(text(ui.q("#monitor-status"))).toBe("Cannot reach server metrics. Check the server connection. Retrying every second.");
    expect(text(ui.q("#metric-decode"))).toBe("—");
    expect(text(ui.q("#monitor-sessions"))).toBe("Live sessions unavailable.");
  });

  it("lists the discovered models, loaded first, with their capabilities", async () => {
    const ui = await monitoring();
    await vi.waitFor(() => expect(ui.qa("#monitor-models tbody tr").length).toBe(2), { timeout: 3000 });
    expect(ui.qa("#monitor-models tbody tr").map((r) => [...r.children].map(text))).toEqual([
      ["m/chat", "Chat, Thinking, Vision", "—", "Ready"],
      ["m/other", "Chat", "—", "Unloaded"],
    ]);
  });

  it("says where history is kept", async () => {
    const ui = await monitoring();
    expect(text(ui.q("#monitor-storage"))).toMatch(/^(Checking storage…|History is )/);
  });
});
