import { flushSync } from "svelte";
import { describe, expect, it } from "vitest";
import type { Client } from "../src/lib/core/client";
import type { ChatEvent } from "../src/lib/core/chat";
import type { Model } from "../src/lib/core/models";
import { ChatController, SessionStore } from "../src/lib/state/chat-state.svelte";
import { MemoryLibrary } from "./support/memory-library";

const model = { id: "m", capabilities: ["chat"], meta: {} } as Model;
const client = { baseUrl: "http://host:1" } as Client;

/** A stream the test steps by hand: `push` emits one event and waits for the consumer to take it. */
function manualStream() {
  const queue: ChatEvent[] = [];
  let wake: (() => void) | undefined;
  let closed = false;
  const stream = async function* () {
    for (;;) {
      while (queue.length) yield queue.shift()!;
      if (closed) return;
      await new Promise<void>((resolve) => (wake = resolve));
    }
  };
  return {
    stream: stream as never,
    push(event: ChatEvent) {
      queue.push(event);
      wake?.();
    },
    end() {
      closed = true;
      wake?.();
    },
  };
}
const done = (): ChatEvent => ({ type: "done", metrics: { ttftMs: 1, elapsedMs: 2, tokensPerSecond: 42, rateSource: "wall" }, usage: { total_tokens: 7 } });
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function setup(stream = manualStream()) {
  const library = new MemoryLibrary();
  const chat = new ChatController(new SessionStore(library.asLibrary), stream.stream);
  chat.create("http://host:1", "m");
  return { chat, library, stream };
}

describe("ChatController", () => {
  it("streams a reply into reactive state: an effect sees every growth of the message text", async () => {
    const { chat, stream } = setup();
    const seen: string[] = [];
    const stop = $effect.root(() => {
      $effect(() => {
        seen.push(chat.active.messages.at(-1)?.text ?? "");
      });
    });
    flushSync();
    const sending = chat.send("hi", [], model, client);
    await tick();
    flushSync();
    stream.push({ type: "content", text: "Hel" });
    await tick();
    flushSync();
    stream.push({ type: "content", text: "lo" });
    await tick();
    flushSync();
    expect(chat.run).toBeDefined();
    stream.push({ type: "finish", reason: "stop" });
    stream.push(done());
    stream.end();
    await sending;
    flushSync();
    stop();
    expect(seen).toContain("Hel");
    expect(seen.at(-1)).toBe("Hello");
    const reply = chat.active.messages.at(-1)!;
    expect(reply.status).toBe("complete");
    expect(reply.tokensPerSecond).toBe(42);
    expect(chat.run).toBeUndefined();
    expect(chat.active.title).toBe("hi");
  });

  it("saves every session through the library and reloads it", async () => {
    const { chat, library, stream } = setup();
    stream.push({ type: "content", text: "ok" });
    stream.push({ type: "finish", reason: "stop" });
    stream.push(done());
    stream.end();
    await chat.send("hello", [], model, client);
    await chat.flush();
    expect(library.items.size).toBe(1);
    const again = new ChatController(new SessionStore(library.asLibrary));
    await again.load();
    expect(again.sessions.length).toBe(1);
    expect(again.sessions[0]!.messages.map((m) => m.text)).toEqual(["hello", "ok"]);
  });

  it("stop marks the reply stopped and keeps what arrived", async () => {
    const { chat, stream } = setup();
    const sending = chat.send("hi", [], model, client);
    await tick();
    stream.push({ type: "content", text: "partial" });
    await tick();
    chat.stop();
    stream.end();
    await sending;
    const reply = chat.active.messages.at(-1)!;
    expect(reply.status).toBe("stopped");
    expect(reply.text).toBe("partial");
    expect(chat.run).toBeUndefined();
  });

  it("splits a leading <think> envelope from the answer, across chunk boundaries", async () => {
    const { chat, stream } = setup();
    for (const text of ["<thi", "nk>pond", "er</think>", "answer"]) stream.push({ type: "content", text });
    stream.push({ type: "finish", reason: "stop" });
    stream.push(done());
    stream.end();
    await chat.send("q", [], model, client);
    const reply = chat.active.messages.at(-1)!;
    expect(reply.thinking).toBe("ponder");
    expect(reply.text).toBe("answer");
  });

  it("refuses to send to another server, and validates before touching the transcript", async () => {
    const { chat } = setup();
    await expect(chat.send("x", [], model, { baseUrl: "http://other:2" } as Client)).rejects.toThrow(/server/i);
    chat.active.settings.temperature = 9;
    await expect(chat.send("x", [], model, client)).rejects.toThrow(/Temperature/);
    expect(chat.active.messages.length).toBe(0);
    expect(chat.active.draft).toBe("");
  });

  it("deleting an assistant reply removes its turn; edit truncates and regenerates", async () => {
    const { chat } = setup();
    const replies = ["first", "second"];
    chat.stream = (async function* () {
      yield { type: "content", text: replies.shift() ?? "late" } as ChatEvent;
      yield { type: "finish", reason: "stop" } as ChatEvent;
      yield done();
    }) as never;
    await chat.send("a", [], model, client);
    await chat.send("b", [], model, client);
    expect(chat.active.messages.map((m) => m.text)).toEqual(["a", "first", "b", "second"]);
    const first = chat.active.messages[0]!.id;
    await chat.edit(first, "a2", model, client);
    expect(chat.active.messages.map((m) => m.text)).toEqual(["a2", "late"]);
    chat.deleteMessage(chat.active.messages[1]!.id);
    expect(chat.active.messages.length).toBe(0);
  });

  it("surfaces a failed save without losing the chat", async () => {
    const { chat, library, stream } = setup();
    library.failWrites = true;
    stream.push({ type: "content", text: "x" });
    stream.push({ type: "finish", reason: "stop" });
    stream.push(done());
    stream.end();
    await chat.send("q", [], model, client);
    await chat.flush();
    expect(chat.persistenceError).not.toBe("");
    expect(chat.active.messages.length).toBe(2);
    library.failWrites = false;
    await chat.save();
    expect(chat.persistenceError).toBe("");
  });

  it("an unsent chat retargets to a new server, a used one keeps its own", () => {
    const { chat } = setup();
    chat.setServer("http://elsewhere:3");
    expect(chat.active.server).toBe("http://elsewhere:3");
    expect(chat.active.model).toBe("");
    chat.active.messages.push({ id: "1", role: "user", text: "kept", createdAt: 1 });
    chat.setServer("http://third:4");
    expect(chat.active.server).toBe("http://elsewhere:3");
  });
});
