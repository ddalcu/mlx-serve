import { describe, expect, it } from "vitest";
import { Client } from "../src/lib/core/client";
import { chat } from "../src/lib/core/chat";

describe("Client", () => {
  it("sends the page's own cookies and HTTP auth, never another origin's", async () => {
    const seen: RequestInit[] = [];
    const fetch = async (_url: string | URL | Request, init?: RequestInit) => {
      seen.push(init ?? {});
      return new Response("{}", { headers: { "content-type": "application/json" } });
    };
    const client = new Client("http://host:1", { fetch: fetch as typeof globalThis.fetch });
    await client.json("/health");
    await client.json("/v1/chat/completions", {}, { model: "m" });
    expect(seen.map((init) => init.credentials)).toEqual(["same-origin", "same-origin"]);
  });
});

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function trickle(parts: [number, string][]): ReadableStream<Uint8Array> {
  let stopped = false;
  return new ReadableStream({
    async start(controller) {
      for (const [delay, text] of parts) {
        await sleep(delay);
        if (stopped) return;
        controller.enqueue(new TextEncoder().encode(text));
      }
      controller.close();
    },
    cancel() {
      stopped = true;
    },
  });
}

function serve(body: () => ReadableStream<Uint8Array>, headDelayMs = 0) {
  return (async (_url: string | URL | Request, init?: RequestInit) => {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, headDelayMs);
      init?.signal?.addEventListener("abort", () => {
        clearTimeout(timer);
        reject(init.signal?.reason);
      });
    });
    return new Response(body(), { headers: { "content-type": "text/event-stream" } });
  }) as typeof globalThis.fetch;
}

const done = 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';

describe("Client timeout", () => {
  it("is a stall timeout once the body flows: a stream that keeps sending outlives it", async () => {
    const client = new Client("http://host:1", {
      fetch: serve(() =>
        trickle([
          [40, ": keepalive\n\n"],
          [40, ": keepalive\n\n"],
          [40, 'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'],
          [40, ": keepalive\n\n"],
          [40, done],
        ]),
      ),
    });
    const events = [];
    for await (const event of chat(client, { messages: [] }, { timeoutMs: 100 })) events.push(event.type);
    expect(events).toEqual(["content", "finish", "done"]);
  });

  it("still ends a body that goes silent for the whole timeout", async () => {
    const client = new Client("http://host:1", { fetch: serve(() => trickle([[20, ": keepalive\n\n"], [400, done]])) });
    const run = async () => {
      for await (const _ of chat(client, { messages: [] }, { timeoutMs: 100 }));
    };
    await expect(run()).rejects.toMatchObject({ kind: "timeout" });
  });

  it("still ends a request whose headers never arrive", async () => {
    const client = new Client("http://host:1", { fetch: serve(() => trickle([[0, done]]), 400) });
    await expect(client.open("/v1/chat/completions", {}, { timeoutMs: 100 })).rejects.toMatchObject({ kind: "timeout" });
  });

  it("keeps the response's status and headers", async () => {
    const client = new Client("http://host:1", { fetch: serve(() => trickle([[0, done]])) });
    const opened = await client.open("/v1/chat/completions");
    expect(opened.response.status).toBe(200);
    expect(opened.response.headers.get("content-type")).toBe("text/event-stream");
    opened.close();
  });
});
