import { afterEach, describe, expect, it } from "vitest";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());

describe("Sidebar", () => {
  it("names the address of the server it talks to", async () => {
    ui = await mountApp();
    const host = new URL(ui.app.connection.active.url).host;
    expect(host).not.toBe("");
    expect(ui.q("#server-address")?.textContent).toBe(host);
  });
});
