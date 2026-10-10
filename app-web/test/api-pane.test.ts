import { afterEach, describe, expect, it } from "vitest";
import { apiReference } from "../src/lib/core/console";
import { chatModel, mockApi } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";

describe("API reference pane", () => {
  it("opens with a runnable curl example for this server and its chat model, and no key", async () => {
    ui = await mountApp({ hash: "#api", api: mockApi([chatModel]) });
    const example = ui.q("pre")!.textContent!;
    expect(example).toContain(`curl '${ui.app.connection.active.url}/v1/chat/completions'`);
    expect(JSON.parse(/-d '(.*)'$/s.exec(example)![1]!).model).toBe("m/chat");
    expect(example).not.toMatch(/api[_-]?key|Authorization|Bearer/i);
  });

  it("lists every documented endpoint; plain GETs link to this server", async () => {
    ui = await mountApp({ hash: "#api" });
    const rows = ui.qa(".api-table tbody tr");
    expect(rows.length).toBe(apiReference.length);
    const base = ui.app.connection.active.url;
    for (const [i, row] of apiReference.entries()) {
      const cells = [...rows[i]!.children];
      expect(text(cells[0])).toBe(row.method);
      expect(text(cells[1])).toBe(row.path);
      const link = cells[1]!.querySelector("a");
      if (row.method === "GET" && !row.path.includes("{")) expect([link?.getAttribute("href"), link?.rel]).toEqual([base + row.path, "noopener noreferrer"]);
      else expect(link).toBeNull();
    }
  });
});
