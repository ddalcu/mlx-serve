import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { apiReference } from "../src/lib/core/console";

// The routes the server answers, read out of the Zig source: the reference below must not drift from them.
const zig = readFileSync(`${import.meta.dirname}/../../src/server.zig`, "utf8");
const block = /const ROUTE_PATHS = \[_\]\[\]const u8\{([^}]*)\};/s.exec(zig)?.[1] ?? "";
const routes = [...block.matchAll(/"([^"]+)"/g)].map((m) => m[1]!);
const documented = new Set(apiReference.map((r) => r.path));

describe("API reference vs the server's routes", () => {
  it("found the route table", () => {
    expect(routes.length).toBeGreaterThan(30);
    expect(routes).toContain("/v1/chat/completions");
  });

  it("documents every endpoint the server serves", () => {
    // "/" is the page itself; the reference still lists it as the console's row.
    const missing = routes.filter((p) => p !== "/" && !documented.has(p));
    expect(missing).toEqual([]);
  });

  it("documents nothing the server does not serve", () => {
    // `/v1/responses/{id}` is matched by prefix in the server, not listed in the table.
    const served = (path: string) => routes.includes(path) || (path.startsWith("/v1/responses/") && path.includes("{"));
    expect(apiReference.map((r) => r.path).filter((p) => !served(p))).toEqual([]);
  });

  it("describes every row and never repeats a method and path", () => {
    expect(apiReference.every((r) => r.description.trim() && /^(GET|POST|DELETE|WS)$/.test(r.method))).toBe(true);
    const keys = apiReference.map((r) => `${r.method} ${r.path}`);
    expect(new Set(keys).size).toBe(keys.length);
  });
});
