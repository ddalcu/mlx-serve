import { flushSync, mount, unmount } from "svelte";
import { vi } from "vitest";
import AppView from "../../src/App.svelte";
import { App } from "../../src/lib/app.svelte";
import { MemoryLibrary } from "./memory-library";
import { mockApi } from "./mock-api";

let mounted: ReturnType<typeof mount> | undefined;

/** Mount the whole console against a mock server and an in-memory library. */
export async function mountApp({ hash = "#chat", api = mockApi(), library = new MemoryLibrary() }: { hash?: string; api?: ReturnType<typeof mockApi>; library?: MemoryLibrary } = {}) {
  vi.stubGlobal("fetch", api.fetch);
  history.replaceState(null, "", hash || "#");
  document.body.innerHTML = '<div id="app"></div>';
  const app = new App(library.asLibrary, () => new MemoryLibrary().asLibrary);
  mounted = mount(AppView, { target: document.getElementById("app")!, props: { app } });
  flushSync();
  await vi.waitFor(() => {
    if (app.connection.status !== "online") throw new Error("connecting");
  });
  await app.chat.init();
  flushSync();
  const root = document.getElementById("app")!;
  return {
    app,
    api,
    library,
    root,
    q: <T extends Element = HTMLElement>(selector: string) => root.querySelector<T>(selector),
    qa: <T extends Element = HTMLElement>(selector: string) => [...root.querySelectorAll<T>(selector)],
    async type(selector: string, value: string) {
      const el = root.querySelector<HTMLInputElement | HTMLTextAreaElement>(selector)!;
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      flushSync();
    },
    async click(selector: string | Element) {
      const el = typeof selector === "string" ? root.querySelector<HTMLElement>(selector)! : (selector as HTMLElement);
      el.click();
      flushSync();
      await Promise.resolve();
    },
    unmount() {
      if (mounted) unmount(mounted);
      mounted = undefined;
      vi.unstubAllGlobals();
      document.body.innerHTML = "";
    },
  };
}
