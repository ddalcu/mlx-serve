import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryLibrary } from "./support/memory-library";
import { mockApi } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";
const idle = () => vi.waitFor(() => expect(ui!.q<HTMLButtonElement>("#library-export")!.disabled).toBe(false));
function choose(select: HTMLSelectElement, value: string) {
  [...select.options].forEach((o) => (o.selected = o.value === value));
  select.dispatchEvent(new Event("change", { bubbles: true }));
  flushSync();
}

async function library() {
  const lib = new MemoryLibrary();
  const server = "http://localhost:3000";
  const png = new Blob([Uint8Array.of(137, 80, 78, 71)], { type: "image/png" });
  await lib.add({ type: "image", server, model: "m/img", prompt: "a red fox", blob: png, createdAt: 3000 });
  await lib.add({ type: "speech", server, model: "m/tts", prompt: "hello there", blob: new Blob(["RIFF"], { type: "audio/wav" }), createdAt: 2000 });
  await lib.add({ type: "image", server: "http://other:2", model: "m/img", prompt: "elsewhere", blob: png, createdAt: 1000 });
  return lib;
}

describe("Library pane", () => {
  // The console saves its own empty "New Chat" next to whatever was seeded.
  const titles = () => ui!.qa(".library-item .library-title").map(text).filter((t) => t !== "New Chat");

  it("lists every saved item, newest first, with a storage summary", async () => {
    ui = await mountApp({ hash: "#library", library: await library() });
    await vi.waitFor(() => expect(titles().length).toBe(3));
    expect(titles()).toEqual(["a red fox", "hello there", "elsewhere"]);
    expect(text(ui.q("#library-storage"))).toMatch(/\d+ items/);
    expect([...ui.q<HTMLSelectElement>("#library-server")!.options].map((o) => o.value)).toEqual(expect.arrayContaining(["", "http://localhost:3000", "http://other:2"]));
  });

  it("filters by search text, type and server", async () => {
    ui = await mountApp({ hash: "#library", library: await library() });
    await vi.waitFor(() => expect(titles().length).toBe(3));
    await ui.type("#library-search", "FOX");
    expect(titles()).toEqual(["a red fox"]);
    await ui.type("#library-search", "");
    choose(ui.q<HTMLSelectElement>("#library-type")!, "speech");
    expect(titles()).toEqual(["hello there"]);
    choose(ui.q<HTMLSelectElement>("#library-type")!, "");
    choose(ui.q<HTMLSelectElement>("#library-server")!, "http://other:2");
    expect(titles()).toEqual(["elsewhere"]);
  });

  it("deleting asks first, then removes the item", async () => {
    const lib = await library();
    ui = await mountApp({ hash: "#library", library: lib });
    await vi.waitFor(() => expect(titles().length).toBe(3));
    await idle();
    const row = ui.qa(".library-item").find((a) => text(a.querySelector(".library-title")) === "hello there")!;
    await ui.click([...row.querySelectorAll("button")].find((b) => text(b) === "Delete…")!);
    expect(text(ui.q("dialog[open] h2"))).toBe("Delete this item?");
    await ui.click("#library-confirm-delete");
    await vi.waitFor(() => expect(titles()).toEqual(["a red fox", "elsewhere"]));
    expect([...lib.items.values()].some((i) => i.prompt === "hello there")).toBe(false);
    expect(text(ui.q("#library-status"))).toBe("Item deleted.");
  });

  it("shows a saved chat as a transcript with escaped Markdown, and continues it in Chat", async () => {
    const api = mockApi();
    api.replies.push({ content: "<b>raw</b> and **bold**" });
    ui = await mountApp({ api });
    await ui.type("#chat-input", "question");
    await ui.click("#chat-send");
    await vi.waitFor(() => expect(ui!.app.chat.c.run).toBeUndefined());
    await ui.app.chat.c.flush();
    await ui.click("button[aria-label='New Chat']");
    ui.app.go("library");
    flushSync();
    await vi.waitFor(() => expect(ui!.qa(".library-item").length).toBeGreaterThan(0));
    await idle();
    await ui.click(ui.qa<HTMLButtonElement>(".library-title").find((b) => text(b) === "question")!);
    await vi.waitFor(() => expect(ui!.q("dialog[open] .library-transcript")).not.toBeNull());
    const transcript = ui.q(".library-transcript")!;
    expect(transcript.querySelector("b")).toBeNull();
    expect(text(transcript)).toContain("<b>raw</b>");
    expect(transcript.querySelector("strong")?.textContent).toBe("bold");
    await ui.click("#library-open-chat");
    flushSync();
    expect(ui.app.router.view).toBe("chat");
    expect(text(ui.q(".message.user .message-text"))).toBe("question");
  });
});
