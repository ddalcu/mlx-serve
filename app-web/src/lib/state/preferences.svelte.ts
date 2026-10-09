import { record } from "../core/client";
import type { StorageAdapter } from "../core/servers";

export const choices = {
  theme: ["system", "light", "dark"],
  accent: ["system", "blue", "purple", "pink", "red", "orange", "yellow", "green", "graphite"],
  textSize: ["small", "medium", "large", "xlarge"],
  column: ["narrow", "medium", "wide"],
} as const;

export type Preferences = {
  theme: (typeof choices.theme)[number];
  accent: (typeof choices.accent)[number];
  textSize: (typeof choices.textSize)[number];
  column: (typeof choices.column)[number];
  compact: boolean;
};

const KEY = "studio.interface";

export function readPreferences(storage: StorageAdapter): Preferences {
  let value: Record<string, unknown> = {};
  try {
    value = record(JSON.parse(storage.getItem(KEY) ?? "{}"));
  } catch {
    /* defaults */
  }
  const pick = <K extends "theme" | "accent" | "textSize" | "column">(key: K, fallback: Preferences[K]): Preferences[K] =>
    (choices[key] as readonly unknown[]).includes(value[key]) ? (value[key] as Preferences[K]) : fallback;
  return {
    theme: pick("theme", "system"),
    accent: pick("accent", "system"),
    textSize: pick("textSize", "medium"),
    column: pick("column", "wide"),
    compact: value.compact === true,
  };
}

/** Interface preferences, persisted on every change. `resolvedTheme` is what the page actually shows. */
export class InterfacePreferences implements Preferences {
  theme = $state<Preferences["theme"]>("system");
  accent = $state<Preferences["accent"]>("system");
  textSize = $state<Preferences["textSize"]>("medium");
  column = $state<Preferences["column"]>("wide");
  compact = $state(false);
  systemDark = $state(false);
  resolvedTheme = $derived(this.theme === "system" ? (this.systemDark ? "dark" : "light") : this.theme);
  #storage: StorageAdapter;

  constructor(storage: StorageAdapter) {
    this.#storage = storage;
    Object.assign(this, readPreferences(storage));
    if (typeof matchMedia === "function") {
      const query = matchMedia("(prefers-color-scheme: dark)");
      this.systemDark = query.matches;
      query.addEventListener("change", () => (this.systemDark = query.matches));
    }
  }

  set<K extends keyof Preferences>(key: K, value: Preferences[K]) {
    (this as Preferences)[key] = value;
    const { theme, accent, textSize, column, compact } = this;
    this.#storage.setItem(KEY, JSON.stringify({ theme, accent, textSize, column, compact }));
  }
}
