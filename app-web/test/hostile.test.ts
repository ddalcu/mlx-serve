import { flushSync, mount, unmount } from "svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ToolCards from "../src/components/ToolCards.svelte";
import type { Feed } from "../src/lib/core/monitor-history";
import { assertInert, HOSTILE, HOSTILE_LINE } from "./support/inert";
import { chatModel, mockApi } from "./support/mock-api";
import { MemoryLibrary } from "./support/memory-library";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
beforeEach(() => localStorage.clear());
afterEach(() => {
  ui?.unmount();
  ui = undefined;
  vi.useRealTimers();
});
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";

describe("hostile text on every surface stays text", () => {
  it("the inspector itself notices markup, handlers and bad links", () => {
    const bad = (html: string) => {
      const root = document.createElement("div");
      root.innerHTML = html;
      return () => assertInert(root);
    };
    expect(bad("<p>fine</p><a href='https://x.test'>ok</a>")).not.toThrow();
    expect(bad("<img src=x onerror=1>")).toThrow();
    expect(bad("<a href='javascript:1'>x</a>")).toThrow();
    expect(bad("<div><iframe></iframe></div>")).toThrow();
    expect(bad("<img src='data:text/html;base64,AAAA'>")).toThrow();
  });

  it("chat: what you type, the reply, its reasoning, the title and a server error", async () => {
    const api = mockApi();
    api.replies.push({ reasoning: HOSTILE.join("\n"), content: HOSTILE.join("\n\n") });
    api.replies.push({ status: 500, error: HOSTILE_LINE });
    ui = await mountApp({ api });
    for (const prompt of [HOSTILE.join("\n"), "again"]) {
      await ui.type("#chat-input", prompt);
      await ui.click("#chat-send");
      await vi.waitFor(() => expect(ui!.app.chat.c.run).toBeUndefined());
    }
    flushSync();
    assertInert(ui.root, "chat");
    expect(text(ui.q(".message-text"))).toContain("<script>window.__pwned = 1</script>");
    expect(text(ui.q(".session-entry .nav-row"))).toContain("<script>");
    expect(text(ui.q(".message.assistant .error-message"))).toContain(HOSTILE[1]);
    expect(document.title).toContain("<script>");
  });

  it("tool cards: names, arguments and results", () => {
    const target = document.createElement("div");
    document.body.append(target);
    const call = (name: string) => ({ id: name, name, args: { [HOSTILE[0]!]: HOSTILE[1], prompt: HOSTILE_LINE, nested: { x: HOSTILE[2] } }, status: "complete", result: HOSTILE.join("\n"), durationMs: 5 });
    const rounds = [
      { text: HOSTILE_LINE, calls: [call(HOSTILE[1]!)] },
      { text: "", calls: HOSTILE.slice(0, 5).map(call) },
    ];
    const app = mount(ToolCards, { target, props: { rounds: rounds as never } });
    flushSync();
    assertInert(target, "tool cards");
    expect(text(target)).toContain(HOSTILE[1]);
    unmount(app);
    target.remove();
  });

  it("models: names, capabilities and architectures in the lists, the chooser, the quick start and the monitor", async () => {
    const models = HOSTILE.map((id, i) => ({ id: `org/${id}`, capabilities: ["chat", HOSTILE[(i + 1) % HOSTILE.length]], loaded: i === 0, state: i === 0 ? "ready" : HOSTILE[2], context_length: 4096, meta: { architecture: HOSTILE[i] } }));
    const api = mockApi(models);
    ui = await mountApp({ hash: "#models", api });
    assertInert(ui.root, "models");
    expect(ui.qa("#catalogue tbody tr").length).toBe(HOSTILE.length);
    ui.app.go("api");
    flushSync();
    assertInert(ui.root, "api");
    ui.app.go("chat");
    flushSync();
    await ui.click("#chat-model-name");
    assertInert(document.body, "model palette");
    ui.app.go("monitoring");
    flushSync();
    await vi.waitFor(() => expect(ui!.qa("#monitor-models tbody tr").length).toBe(HOSTILE.length));
    assertInert(ui.root, "monitoring models");
  });

  it("servers: a name and an address in Settings, the sidebar and Monitoring", async () => {
    ui = await mountApp({ hash: "#settings" });
    ui.app.connection.select(ui.app.connection.add({ name: HOSTILE_LINE, url: "http://hostile.test:11234" }));
    flushSync();
    assertInert(ui.root, "settings");
    expect(text(ui.q("#server-list"))).toContain(HOSTILE[0]);
    ui.app.go("monitoring");
    flushSync();
    assertInert(ui.root, "monitoring");
    expect(text(ui.q("#monitor-server"))).toContain(HOSTILE[0]);
  });

  it("monitoring: session, client and model names, and the status line", async () => {
    ui = await mountApp({ hash: "#monitoring" });
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.UTC(2026, 9, 8, 12, 0, 0));
    const feed = (n: number): Feed => ({
      counters: { requests_success_total: n },
      gauges: { requests_running: 1, process_start_time_seconds: 1 },
      histograms: {},
      sessions: HOSTILE.map((model, i) => ({ request_id: i + 1, model, client: HOSTILE[(i + 3) % HOSTILE.length], phase: i % 2 ? "decode" : HOSTILE[0], context_tokens: 5, context_length: 10 })),
    });
    ui.app.monitor.accept(feed(1));
    vi.setSystemTime(Date.UTC(2026, 9, 8, 12, 0, 1));
    ui.app.monitor.accept(feed(2));
    flushSync();
    assertInert(ui.root, "monitoring sessions");
    expect(ui.qa("#monitor-sessions tbody tr").length).toBe(HOSTILE.length);
    expect(ui.qa("#monitor-requests tbody tr").length).toBeGreaterThan(0);
    ui.app.monitor.failed(new Error(HOSTILE_LINE));
    flushSync();
    assertInert(ui.root, "monitoring error");
  });

  it("library: titles, prompts and a saved conversation, listed and opened", async () => {
    const library = new MemoryLibrary();
    const server = "http://localhost:3000";
    for (const [i, prompt] of HOSTILE.entries()) await library.add({ type: i % 2 ? "image" : "speech", server, model: `org/${prompt}`, prompt, blob: new Blob(["x"], { type: i % 2 ? "image/png" : "audio/wav" }) });
    const session = { version: 1, id: "hostile", title: HOSTILE_LINE, server, model: "m/chat", createdAt: 1, updatedAt: 2, settings: { system: HOSTILE_LINE, temperature: 1, maxTokens: null, thinking: false, mtp: null }, draft: "", messages: [{ id: "u", role: "user", text: HOSTILE.join("\n"), createdAt: 1 }, { id: "a", role: "assistant", text: HOSTILE.join("\n\n"), thinking: HOSTILE_LINE, createdAt: 2, status: "complete" }] };
    await library.put("hostile", { type: "chat", server, model: "m/chat", prompt: HOSTILE_LINE, blob: new Blob([JSON.stringify(session)], { type: "application/json" }) });
    ui = await mountApp({ hash: "#library", library });
    await vi.waitFor(() => expect(ui!.qa(".library-item").length).toBeGreaterThan(HOSTILE.length));
    flushSync();
    assertInert(ui.root, "library list");
    const open = ui.qa<HTMLButtonElement>(".library-title").find((b) => text(b).startsWith("<script>window.__pwned = 1</script> <img"));
    expect(open).toBeDefined();
    await ui.click(open!);
    await vi.waitFor(() => expect(ui!.q("dialog[open] .library-transcript")).not.toBeNull());
    assertInert(document.body, "library transcript");
  });

  it("media panes: a model failure is shown as text", async () => {
    const api = mockApi([chatModel]);
    ui = await mountApp({ hash: "#chat", api });
    api.replies.push({ status: 400, error: HOSTILE.join(" ") });
    await ui.type("#chat-input", "x");
    await ui.click("#chat-send");
    await vi.waitFor(() => expect(ui!.app.chat.c.run).toBeUndefined());
    flushSync();
    assertInert(ui.root, "failure");
  });
});
