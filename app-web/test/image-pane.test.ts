import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ImageWorkspace } from "../src/lib/state/image-workspace.svelte";
import { chatModel, imageModel, kreaModel, mockApi } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());
const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";

async function imagePane(models: object[] = [imageModel, chatModel]) {
  const api = mockApi(models);
  ui = await mountApp({ hash: "#image", api });
  await vi.waitFor(() => expect(ui!.app.image.ready && ui!.app.image.d.model !== "").toBe(true));
  flushSync();
  return { ui, api };
}
const generate = () => ui!.q<HTMLButtonElement>("#image-generate")!;
/** What the pane saved to the Library (the chat pane keeps its own empty "New Chat" there too). */
const saved = () => [...ui!.library.items.values()].filter((i) => i.type === "image");
const posts = (api: ReturnType<typeof mockApi>) => api.requests.filter((r) => r.path === "/v1/images/generations");

describe("Image pane", () => {
  it("opens on the loaded-or-first image model with its own defaults, and Generate waits for a prompt", async () => {
    const { ui } = await imagePane();
    expect(ui.app.image.d.model).toBe("m/flux");
    expect(text(ui.q(".image-model-card strong"))).toBe("flux");
    expect(generate().disabled).toBe(true);
    expect(text(ui.q("#image-validation"))).toBe("");
    await ui.type("#image-prompt", "a red fox");
    expect(generate().disabled).toBe(false);
  });

  it("generates, shows the picture and saves it to the Library", async () => {
    const { ui, api } = await imagePane();
    await ui.type("#image-prompt", "a red fox");
    await ui.type("input[name=seed]", "7");
    await ui.click(generate());
    await vi.waitFor(() => expect(ui.q("#image-result")).not.toBeNull());
    expect(posts(api)[0]!.body).toMatchObject({ model: "m/flux", prompt: "a red fox", seed: 7, steps: 8, size: "1024x1024", response_format: "b64_json", stream: true });
    expect(ui.q<HTMLImageElement>("#image-result")!.alt).toBe("a red fox");
    expect(text(ui.q(".image-result-actions"))).toMatch(/^image-.*\.png Save PNG/);
    await vi.waitFor(() => expect(saved().map((i) => [i.model, i.prompt])).toEqual([["m/flux", "a red fox"]]));
    expect(ui.q("#image-retry-save")).toBeNull();
  });

  it("shows a failure as text, keeps the prompt, and offers another go", async () => {
    const { ui, api } = await imagePane();
    api.failures.push({ status: 500, error: "out of <b>memory</b>" });
    await ui.type("#image-prompt", "a red fox");
    await ui.click(generate());
    await vi.waitFor(() => expect(text(ui.q("#image-preview h2"))).toBe("Failed"));
    expect(text(ui.q("#image-preview .empty-state p"))).toContain("out of <b>memory</b>");
    expect(ui.q("#image-preview b")).toBeNull();
    expect(ui.q<HTMLTextAreaElement>("#image-prompt")!.value).toBe("a red fox");
    expect(generate().disabled).toBe(false);
  });

  it("reports why Generate is unavailable instead of sending a bad request", async () => {
    const { ui, api } = await imagePane();
    await ui.type("#image-prompt", "a red fox");
    await ui.type("input[name=seed]", "-3");
    expect(generate().disabled).toBe(true);
    expect(text(ui.q("#image-validation"))).toBe("Seed must be a non-negative whole number, or empty for random.");
    await ui.type("input[name=seed]", "");
    await ui.type("input[name=width]", "100");
    expect(text(ui.q("#image-validation"))).toBe("This model samples between 256 and 1536 px per side.");
    expect(posts(api)).toEqual([]);
  });

  it("rounds an off-grid canvas and tells the user", async () => {
    const { ui } = await imagePane();
    await ui.type("#image-prompt", "a red fox");
    await ui.type("input[name=width]", "1000");
    await ui.type("input[name=height]", "1000");
    expect(text(ui.q("#image-size-hint"))).toContain("Rounded to 1024 × 1024");
  });

  it("quality buttons set the step count the model recommends", async () => {
    const { ui } = await imagePane();
    const buttons = ui.qa<HTMLButtonElement>("#image-quality-control .segmented button");
    expect(buttons.map(text)).toEqual(["Fast", "Good", "Quality", "Super Quality"]);
    expect(buttons.map((b) => b.getAttribute("aria-pressed"))).toEqual(["false", "true", "false", "false"]);
    await ui.click(buttons[2]!);
    expect(ui.app.image.d.steps).toBe(12);
    expect(text(ui.q("#image-quality-hint"))).toBe("12 steps");
  });

  it("only shows the controls the chosen model has", async () => {
    const { ui } = await imagePane([imageModel, kreaModel]);
    await ui.click("#image-advanced summary");
    expect(ui.q("input[name=weights]")).not.toBeNull();
    expect(ui.q("input[name=negative]")).toBeNull();
    expect(ui.q("#image-sources")).not.toBeNull();
    ui.app.image.chooseModel("m/krea");
    flushSync();
    expect(ui.app.image.d.model).toBe("m/krea");
    expect(ui.q("input[name=weights]")).not.toBeNull();
    expect(text(ui.q(".image-model-card strong"))).toBe("krea");
  });

  it("cancelling a run stops it without a result", async () => {
    const { ui, api } = await imagePane();
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const original = api.fetch;
    vi.stubGlobal("fetch", async (input: string | URL | Request, init?: RequestInit) => {
      if (new URL(String(input)).pathname === "/v1/images/generations") await gate;
      return original(input, init);
    });
    await ui.type("#image-prompt", "a red fox");
    await ui.click(generate());
    expect(text(generate())).toBe("Cancel");
    await ui.click(generate());
    release();
    await vi.waitFor(() => expect(ui.app.image.c.run).toBeNull());
    expect(ui.q("#image-result")).toBeNull();
    expect(saved()).toEqual([]);
  });

  it("keeps the draft for this server and restores it in the next session", async () => {
    const { ui } = await imagePane();
    await ui.type("#image-prompt", "remember me");
    await ui.type("input[name=seed]", "99");
    ui.app.image.flush();
    const later = new ImageWorkspace(ui.app.connection, ui.library.asLibrary, ui.app.image.drafts);
    await vi.waitFor(async () => {
      await later.init();
      expect(later.d).toMatchObject({ prompt: "remember me", seed: "99" });
    });
    expect(later.ready).toBe(true);
    expect(later.draftError).toBe("");
  });
});
