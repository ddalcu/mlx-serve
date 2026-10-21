import { describe, expect, it } from "vitest";
import { Client } from "../src/lib/core/client";

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
