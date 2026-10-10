export const views = ["chat", "image", "audio", "video", "library", "monitoring", "models", "settings", "api"] as const;
export type View = (typeof views)[number];

/** `#monitor` is the public name of the Monitoring view; every other view uses its own name. */
const hashOf = (view: View) => (view === "monitoring" ? "monitor" : view);
export function viewFromHash(hash: string): View | undefined {
  const name = hash.replace(/^#/, "") === "monitor" ? "monitoring" : hash.replace(/^#/, "");
  return views.find((v) => v === name);
}

export class Router {
  view = $state<View>("chat");

  constructor(hash = location.hash) {
    this.view = viewFromHash(hash) ?? "chat";
    window.addEventListener("hashchange", () => {
      const next = viewFromHash(location.hash);
      if (next) this.view = next;
    });
  }

  go(next: View) {
    this.view = next;
    if (location.hash !== `#${hashOf(next)}`) history.pushState(null, "", `#${hashOf(next)}`);
  }
}
