import { describe, expect, it } from "vitest";
import { archiveLibrary, chatMarkdown, cleanSession, importArchive } from "../src/lib/core/library-transfer";
import type { Session } from "../src/lib/state/chat-state.svelte";
import { MemoryLibrary } from "./support/memory-library";

const server = "http://host:1";
const session = (id: string): Session => ({
  version: 1,
  id,
  title: "Plans",
  server,
  model: "m/chat",
  createdAt: 1,
  updatedAt: 2,
  settings: { system: "be brief", temperature: 0.5, maxTokens: null, thinking: false, mtp: null },
  messages: [
    { id: "u1", role: "user", text: "hello", createdAt: 1 },
    { id: "a1", role: "assistant", text: "hi **there**", thinking: "hmm", createdAt: 2, status: "complete", tokensPerSecond: 30 },
  ],
  draft: "",
});
const png = new Blob([Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3)], { type: "image/png" });

async function filled() {
  const library = new MemoryLibrary();
  await library.put("chat-1", { type: "chat", server, model: "m/chat", prompt: "Plans", blob: new Blob([JSON.stringify(session("chat-1"))], { type: "application/json" }) });
  const imageId = await library.add({ type: "image", server, model: "m/img", prompt: "a fox", blob: png });
  return { library, imageId };
}

describe("Library archive", () => {
  it("round-trips chats and media into a fresh library, with new media ids", async () => {
    const { library, imageId } = await filled();
    const archive = await archiveLibrary(library.asLibrary);
    const copy = new MemoryLibrary();
    const ids = await importArchive(copy.asLibrary, archive);
    expect(ids.length).toBe(2);
    const rows = await copy.list();
    expect(rows.map((r) => [r.type, r.model, r.prompt]).sort()).toEqual([["chat", "m/chat", "Plans"], ["image", "m/img", "a fox"]]);
    const image = [...copy.items.values()].find((i) => i.type === "image")!;
    expect(image.id).not.toBe(imageId);
    expect(new Uint8Array(await image.blob.arrayBuffer())).toEqual(new Uint8Array(await png.arrayBuffer()));
    const chat = [...copy.items.values()].find((i) => i.type === "chat")!;
    expect(JSON.parse(await chat.blob.text()).messages.map((m: { text: string }) => m.text)).toEqual(["hello", "hi **there**"]);
  });

  it("refuses archives that are not Studio's, in one piece: nothing is imported", async () => {
    const copy = new MemoryLibrary();
    const wrapped = (extra: object) => JSON.stringify({ format: "mlx-serve-studio", version: 1, items: [], ...extra });
    for (const text of ['{"format":"other"}', wrapped({ format: "someone-elses" }), wrapped({ version: 2 }), "not json", JSON.stringify({ format: "mlx-serve-studio", version: 1, items: [{ type: "image", server, model: "m", createdAt: 1, mime: "text/html", data: "AAAA", sourceId: "x" }] })])
      await expect(importArchive(copy.asLibrary, new Blob([text]))).rejects.toThrow();
    expect(copy.items.size).toBe(0);
  });

  it("an imported chat takes the archive's server and model, and drops tool-call media ids it cannot resolve", async () => {
    const copy = new MemoryLibrary();
    const dirty = { ...session("old"), messages: [{ id: "a", role: "assistant", text: "x", createdAt: 1, toolRounds: [{ text: "", calls: [{ id: "c", name: "generate_image", args: {}, status: "complete", result: "ok", mediaId: "gone", mediaType: "image" }] }] }] };
    const archive = new Blob([JSON.stringify({ format: "mlx-serve-studio", version: 1, items: [{ sourceId: "old", type: "chat", model: "m/other", server: "http://elsewhere:2", createdAt: 5, prompt: "P", session: dirty }] })]);
    await importArchive(copy.asLibrary, archive);
    const chat = JSON.parse(await [...copy.items.values()][0]!.blob.text());
    expect([chat.server, chat.model]).toEqual(["http://elsewhere:2", "m/other"]);
    expect(chat.messages[0].toolRounds[0].calls[0].mediaId).toBeUndefined();
  });
});

describe("chat export", () => {
  it("writes Markdown with speaker headings, reasoning in details and attachments as images", () => {
    const s = cleanSession(session("c"), "c");
    s.messages[0]!.images = ["data:image/png;base64,AAAA"];
    const md = chatMarkdown(s);
    expect(md).toContain("# Plans");
    expect(md).toContain("## You");
    expect(md).toContain("## Assistant");
    expect(md).toMatch(/<details><summary>Thinking<\/summary>\s+hmm\s+<\/details>/);
    expect(md).toContain("![Attachment](data:image/png;base64,AAAA)");
  });

  it("escapes the reasoning label so translated text cannot add markup", () => {
    const s = cleanSession(session("c"), "c");
    expect(chatMarkdown(s)).not.toMatch(/<summary[^>]*data-/);
  });
});
