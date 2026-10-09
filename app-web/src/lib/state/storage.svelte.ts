import { t } from "../i18n/i18n";
import type { StorageAdapter } from "../core/servers";

/** localStorage with an in-memory fallback, so a blocked browser store degrades to a session-only console. */
export class SafeStorage implements StorageAdapter {
  unavailable = $state(false);
  warning = $derived(this.unavailable ? t("Browser storage is unavailable. Changes last for this session only.") : "");
  #memory = new Map<string, string>();

  getItem(key: string): string | null {
    try {
      return localStorage.getItem(key) ?? this.#memory.get(key) ?? null;
    } catch {
      this.unavailable = true;
      return this.#memory.get(key) ?? null;
    }
  }

  setItem(key: string, value: string) {
    this.#memory.set(key, value);
    try {
      localStorage.setItem(key, value);
    } catch {
      this.unavailable = true;
    }
  }
}
