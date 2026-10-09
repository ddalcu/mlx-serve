import { flushSync } from "svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mockApi } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
beforeEach(() => localStorage.clear());
afterEach(() => ui?.unmount());
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";
const html = () => document.documentElement;
const stored = (key: string) => JSON.parse(localStorage.getItem(key) ?? "null");

async function settings() {
  ui = await mountApp({ hash: "#settings" });
  return ui;
}
function choose(select: HTMLSelectElement, value: string) {
  [...select.options].forEach((o) => (o.selected = o.value === value));
  select.dispatchEvent(new Event("change", { bubbles: true }));
  flushSync();
}
const button = (label: string) => ui!.qa<HTMLButtonElement>("button").find((b) => text(b) === label)!;

describe("Settings: interface", () => {
  it("appearance, accent, text size, column and compact mode apply at once and are remembered", async () => {
    const ui = await settings();
    await ui.click(button("Dark"));
    choose(ui.q<HTMLSelectElement>("#accent")!, "purple");
    choose(ui.q<HTMLSelectElement>("#textSize")!, "large");
    await ui.click(button("Narrow"));
    await ui.click("#compact");
    expect([html().dataset.theme, html().dataset.accent, html().dataset.textSize, html().dataset.column, html().dataset.compact]).toEqual(["dark", "purple", "large", "narrow", "true"]);
    expect(stored("studio.interface")).toEqual({ theme: "dark", accent: "purple", textSize: "large", column: "narrow", compact: true });
    expect(ui.qa(".segmented")[0]!.querySelector("[aria-pressed=true]")?.textContent).toBe("Dark");
  });

  it("starts from what was remembered", async () => {
    localStorage.setItem("studio.interface", JSON.stringify({ theme: "light", accent: "green", textSize: "small", column: "medium", compact: true }));
    const ui = await settings();
    expect([html().dataset.theme, html().dataset.accent, html().dataset.textSize, html().dataset.column]).toEqual(["light", "green", "small", "medium"]);
    expect(ui.q<HTMLInputElement>("#compact")!.checked).toBe(true);
    expect(ui.q<HTMLSelectElement>("#accent")!.value).toBe("green");
  });

  it("switches language for the whole page and remembers it", async () => {
    const ui = await settings();
    choose(ui.q<HTMLSelectElement>("#language")!, "zh-Hans");
    expect(html().lang).toBe("zh-Hans");
    expect(localStorage.getItem("mlx-serve-language") ?? localStorage.getItem("mlx-serve-lang")).toBe("zh-Hans");
    expect(text(ui.q("#pane-title"))).toBe("设置");
    expect(text(ui.q(".settings-section[data-section=interface] h1"))).toBe("界面");
    choose(ui.q<HTMLSelectElement>("#language")!, "en");
    expect(text(ui.q("#pane-title"))).toBe("Settings");
  });
});

describe("Settings: search and categories", () => {
  const visible = () => ui!.qa(".settings-section").filter((s) => !s.hidden).map((s) => s.dataset.section);

  it("shows every section, or only the chosen category", async () => {
    const ui = await settings();
    expect(visible()).toEqual(["interface", "servers", "about"]);
    await ui.click(button("Servers"));
    expect(visible()).toEqual(["servers"]);
    await ui.click(button("All Settings"));
    expect(visible()).toEqual(["interface", "servers", "about"]);
  });

  it("filters sections by what they say, and admits when nothing matches", async () => {
    const ui = await settings();
    await ui.type("#settings-search", "accent");
    expect(visible()).toEqual(["interface"]);
    await ui.type("#settings-search", "zzzz");
    expect(visible()).toEqual([]);
    expect(ui.q<HTMLElement>("#settings-empty")!.hidden).toBe(false);
  });
});

describe("Settings: servers", () => {
  // happy-dom's native URL validation is stricter than a browser's, so submit the event and let the pane check.
  const submit = () => {
    ui!.q<HTMLFormElement>("#server-form")!.dispatchEvent(new SubmitEvent("submit", { cancelable: true, bubbles: true }));
    flushSync();
  };
  const form = (name: string, url: string, key = "", remember = false) => async () => {
    await ui!.type("#server-name", name);
    await ui!.type("#server-url", url);
    await ui!.type("#server-key", key);
    const box = ui!.q<HTMLInputElement>("#remember-key")!;
    if (box.checked !== remember) await ui!.click(box);
    submit();
  };
  const rows = () => ui!.qa("#server-list .server-row").map((r) => text(r.querySelector(".server-choice")));

  it("lists this page's server first, selected", async () => {
    const ui = await settings();
    expect(rows().length).toBe(1);
    expect(rows()[0]).toContain("This server");
    expect(ui.q(".server-choice")!.getAttribute("aria-pressed")).toBe("true");
  });

  it("adds a server, selects it and checks it", async () => {
    const ui = await settings();
    await form("Studio Mac", "http://studio.local:11234")();
    await vi.waitFor(() => expect(rows().length).toBe(2));
    expect(rows()[1]).toContain("Studio Mac");
    expect(rows()[1]).toContain("http://studio.local:11234");
    expect(ui.qa(".server-choice")[1]!.getAttribute("aria-pressed")).toBe("true");
    expect(ui.app.connection.active.url).toBe("http://studio.local:11234");
    expect(stored("studio.servers").map((s: { name: string }) => s.name)).toEqual(["This server", "Studio Mac"]);
  });

  it("refuses an address that is not a plain http(s) URL, saying why", async () => {
    const ui = await settings();
    await form("Bad", "http://user:pw@host:1")();
    expect(text(ui.q("#form-error"))).toBe("Server URLs must use HTTP(S) without credentials, query strings or fragments.");
    expect(rows().length).toBe(1);
    await form("Bad", "ftp://host")();
    expect(text(ui.q("#form-error"))).toMatch(/must use HTTP\(S\)/);
  });

  it("keeps an API key for the session only, unless asked to remember it", async () => {
    await settings();
    await form("Keyed", "http://keyed.local:1", "sk-secret")();
    expect(localStorage.getItem("studio.servers")).not.toContain("sk-secret");
    expect(ui!.app.connection.active.apiKey).toBe("sk-secret");
    await form("Kept", "http://kept.local:1", "sk-kept", true)();
    expect(localStorage.getItem("studio.servers")).toContain("sk-kept");
    expect(localStorage.getItem("studio.servers")).not.toContain("sk-secret");
  });

  it("edits the selected server in place", async () => {
    const ui = await settings();
    await form("Mac", "http://mac.local:1")();
    await ui.click("#edit-server");
    expect(ui.q<HTMLInputElement>("#server-name")!.value).toBe("Mac");
    expect(text(ui.q("#server-form-title"))).toBe("Edit Server");
    await ui.type("#server-name", "Studio");
    submit();
    expect(rows().filter((r) => r.includes("Studio")).length).toBe(1);
    expect(rows().length).toBe(2);
  });

  it("removes a server and falls back to another", async () => {
    const ui = await settings();
    await form("Mac", "http://mac.local:1")();
    await ui.click(ui.qa<HTMLButtonElement>(".remove-server")[1]!);
    expect(rows().length).toBe(1);
    expect(ui.app.connection.active.name).toBe("This server");
  });

  it("removing the last server leaves this page's own", async () => {
    const ui = await settings();
    await ui.click(ui.qa<HTMLButtonElement>(".remove-server")[0]!);
    expect(rows().length).toBe(1);
    expect(rows()[0]).toContain("This server");
  });
});

describe("Settings: about", () => {
  it("shows the server's version and the licence links", async () => {
    const api = mockApi();
    ui = await mountApp({ hash: "#settings", api });
    await vi.waitFor(() => expect(text(ui!.q("[data-section=about] h2"))).toContain("9.9.9"));
    const links = ui!.qa<HTMLAnchorElement>("[data-section=about] a").map((a) => [text(a), a.target, a.rel]);
    expect(links).toEqual([
      ["mlx-serve", "_blank", "noopener noreferrer"],
      ["License (MIT)", "_blank", "noopener noreferrer"],
      ["Third-party notices", "_blank", "noopener noreferrer"],
    ]);
  });
});
