import { afterEach, describe, expect, it, vi } from "vitest";
import { chatModel, imageModel, mockApi, otherModel } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";
const rows = () => ui!.qa("#catalogue tbody tr").map((r) => [...r.children].slice(0, 2).map(text));

describe("Models pane", () => {
  it("lists the server's models with what each can do", async () => {
    ui = await mountApp({ hash: "#models", api: mockApi([chatModel, otherModel, imageModel]) });
    expect(rows()).toEqual([["m/chat", "Chat, Thinking, Vision"], ["m/other", "Chat"], ["m/flux", "Image"]]);
    expect(text(ui.q("#catalogue-server"))).toContain("This server");
  });

  it("filters by name and says when nothing matches", async () => {
    ui = await mountApp({ hash: "#models", api: mockApi([chatModel, otherModel, imageModel]) });
    await ui.type("#model-search", "FLUX");
    expect(rows()).toEqual([["m/flux", "Image"]]);
    await ui.type("#model-search", "nope");
    expect(text(ui.q("#catalogue h2"))).toBe("No models match your search");
  });

  it("Use selects a model for chat, and marks it", async () => {
    ui = await mountApp({ hash: "#models", api: mockApi([chatModel, otherModel]) });
    const use = () => ui!.qa<HTMLButtonElement>("#catalogue tbody button").map(text);
    expect(use()).toEqual(["Selected", "Use"]);
    await ui.click(ui.qa<HTMLButtonElement>("#catalogue tbody button")[1]!);
    expect(ui.app.connection.selectedModel).toBe("m/other");
    expect(use()).toEqual(["Use", "Selected"]);
  });

  it("Refresh asks the server again", async () => {
    const api = mockApi([chatModel]);
    ui = await mountApp({ hash: "#models", api });
    const asked = () => api.requests.filter((r) => r.path === "/v1/models").length;
    const before = asked();
    await ui.click("#refresh-models");
    await vi.waitFor(() => expect(asked()).toBeGreaterThan(before));
    await vi.waitFor(() => expect(rows().length).toBe(1));
  });

  it("an empty server says so", async () => {
    ui = await mountApp({ hash: "#models", api: mockApi([]) });
    expect(text(ui.q("#catalogue h2"))).toBe("No models on this server");
  });
});
