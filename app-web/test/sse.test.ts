import { describe, expect, it } from "vitest";
import { readSSE } from "../src/lib/core/sse";

const bytes = new TextEncoder().encode(
  ": keepalive\r\nevent: progress\r\nid: 7\r\ndata: hé\r\ndata: two\r\n\r\ndata: [DONE]\n\n",
);
const expected = [
  { event: "progress", data: "hé\ntwo", id: "7" },
  { event: "message", data: "[DONE]", id: "7" },
];

async function collect(chunks: Uint8Array[]) {
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      for (const chunk of chunks) c.enqueue(chunk);
      c.close();
    },
  });
  const out = [];
  for await (const event of readSSE(stream)) out.push(event);
  return out;
}

describe("readSSE", () => {
  it("survives every two-chunk byte split, CRLF, comments, UTF-8 and DONE", async () => {
    for (let split = 1; split < bytes.length; split++)
      expect(await collect([bytes.slice(0, split), bytes.slice(split)])).toEqual(expected);
  });

  it("survives one byte per chunk", async () => {
    expect(await collect([...bytes].map((b) => Uint8Array.of(b)))).toEqual(expected);
  });
});
