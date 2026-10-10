import { describe, expect, it } from "vitest";
import { apiPrefix, pageApiKey, pageServer } from "../src/lib/core/console";
import { Connection } from "../src/lib/state/connection.svelte";

describe("page API key", () => {
  it("reads ?api_key= from the query string", () => {
    expect(pageApiKey("?api_key=123")).toBe("123");
    expect(pageApiKey("?x=1&api_key=a%20b")).toBe("a b");
    expect(pageApiKey("")).toBeUndefined();
    expect(pageApiKey("?api_key=")).toBeUndefined();
  });

  it("authenticates its own server, never another, and stays off disk", () => {
    const saved = new Map<string, string>();
    const storage = { getItem: (k: string) => saved.get(k) ?? null, setItem: (k: string, v: string) => void saved.set(k, v) };
    const conn = new Connection("http://lan:11234", storage, {}, "123");
    expect(conn.active.apiKey).toBe("123");
    const other = conn.add({ url: "http://other:1" });
    expect(conn.store.get(other)?.apiKey).toBeUndefined();
    expect(saved.get("studio.servers")).not.toMatch(/apiKey/);
  });
});

describe("mount prefix", () => {
  it("console and Monitoring share the same resolution, without query keys", () => {
    for (const [path, prefix] of [
      ["/", ""],
      ["/mount", "/mount"],
      ["/mount/index.html", "/mount"],
    ] as const) {
      expect(apiPrefix(path)).toBe(prefix);
      expect(pageServer(new URL("https://host" + path + "?api_key=ignored"))).toBe("https://host" + prefix);
    }
  });
});

const memory = () => {
  const saved = new Map<string, string>();
  return { saved, getItem: (k: string) => saved.get(k) ?? null, setItem: (k: string, v: string) => void saved.set(k, v) };
};
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("Connection", () => {
  it("lists models and picks a loaded chat model when the server answers", async () => {
    const fetch = async (url: string | URL | Request) => {
      const path = new URL(String(url)).pathname;
      if (path === "/health") return json({ status: "ok" });
      if (path === "/v1/models")
        return json({ data: [{ id: "img", capabilities: ["image"] }, { id: "cold", capabilities: ["chat"] }, { id: "hot", capabilities: ["chat"], loaded: true }] });
      return json({}, 404);
    };
    const conn = new Connection("http://host:1", memory(), { fetch: fetch as typeof globalThis.fetch });
    expect(conn.status).toBe("idle");
    const done = conn.refresh();
    expect(conn.status).toBe("checking");
    await done;
    expect(conn.status).toBe("online");
    expect(conn.models.map((m) => m.id)).toEqual(["img", "cold", "hot"]);
    expect(conn.selectedModel).toBe("hot");
    expect(conn.checks[conn.activeId]?.status).toBe("online");
  });

  it("reports a failing server by its message and keeps the selection empty", async () => {
    const fetch = async () => json({ error: { message: "nope" } }, 500);
    const conn = new Connection("http://host:1", memory(), { fetch: fetch as typeof globalThis.fetch });
    await conn.refresh();
    expect(conn.status).toBe("error");
    expect(conn.models).toEqual([]);
    expect(conn.selectedModel).toBe("");
    expect(conn.message.length).toBeGreaterThan(0);
  });

  it("switching servers resets state, and removing the active server falls back to another", () => {
    const storage = memory();
    const conn = new Connection("http://host:1", storage);
    const other = conn.add({ url: "http://other:2", name: "Other" });
    conn.select(other);
    expect(conn.activeId).toBe(other);
    expect(storage.saved.get("studio.activeServer")).toBe(other);
    expect(conn.active.url).toBe("http://other:2");
    conn.remove(other);
    expect(conn.servers.map((s) => s.url)).toEqual(["http://host:1"]);
    expect(conn.active.url).toBe("http://host:1");
    expect(conn.checks[other]).toBeUndefined();
  });

  it("restores the saved active server", () => {
    const storage = memory();
    const first = new Connection("http://host:1", storage);
    const other = first.add({ url: "http://other:2" });
    first.select(other);
    expect(new Connection("http://host:1", storage).activeId).toBe(other);
  });
});
