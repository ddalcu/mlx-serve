import { describe, expect, it } from "vitest";
import { InterfacePreferences, readPreferences } from "../src/lib/state/preferences.svelte";
import { Router, viewFromHash } from "../src/lib/state/router.svelte";
import { SafeStorage } from "../src/lib/state/storage.svelte";

const memory = (initial: Record<string, string> = {}) => {
  const saved = new Map(Object.entries(initial));
  return { saved, getItem: (k: string) => saved.get(k) ?? null, setItem: (k: string, v: string) => void saved.set(k, v) };
};

describe("preferences", () => {
  it("fall back to defaults for anything stored that is not a known choice", () => {
    expect(readPreferences(memory())).toEqual({ theme: "system", accent: "system", textSize: "medium", column: "wide", compact: false });
    expect(readPreferences(memory({ "studio.interface": '{"theme":"dark","accent":"nope","textSize":"large","column":7,"compact":"yes"}' }))).toEqual({
      theme: "dark", accent: "system", textSize: "large", column: "wide", compact: false,
    });
    expect(readPreferences(memory({ "studio.interface": "[" })).theme).toBe("system");
  });

  it("persist every change under the existing key", () => {
    const storage = memory();
    const prefs = new InterfacePreferences(storage);
    prefs.set("accent", "green");
    prefs.set("compact", true);
    expect(JSON.parse(storage.saved.get("studio.interface")!)).toEqual({ theme: "system", accent: "green", textSize: "medium", column: "wide", compact: true });
    expect(new InterfacePreferences(storage).accent).toBe("green");
  });

  it("resolve the system theme from the OS setting", () => {
    const prefs = new InterfacePreferences(memory());
    prefs.systemDark = true;
    expect(prefs.resolvedTheme).toBe("dark");
    prefs.set("theme", "light");
    expect(prefs.resolvedTheme).toBe("light");
  });
});

describe("SafeStorage", () => {
  it("keeps working in memory when the browser store throws, and says so", () => {
    const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage")!;
    Object.defineProperty(globalThis, "localStorage", { configurable: true, get: () => ({ getItem: () => { throw Error("blocked"); }, setItem: () => { throw Error("blocked"); } }) });
    try {
      const storage = new SafeStorage();
      expect(storage.warning).toBe("");
      storage.setItem("k", "v");
      expect(storage.getItem("k")).toBe("v");
      expect(storage.warning).not.toBe("");
    } finally {
      Object.defineProperty(globalThis, "localStorage", original);
    }
  });
});

describe("router", () => {
  it("maps hashes to views, with #monitor for Monitoring", () => {
    expect(viewFromHash("#chat")).toBe("chat");
    expect(viewFromHash("#monitor")).toBe("monitoring");
    expect(viewFromHash("#monitoring")).toBe("monitoring");
    expect(viewFromHash("#bogus")).toBeUndefined();
    expect(viewFromHash("")).toBeUndefined();
  });

  it("starts on the hash's view and writes the hash when navigating", () => {
    const router = new Router("#models");
    expect(router.view).toBe("models");
    router.go("monitoring");
    expect(router.view).toBe("monitoring");
    expect(location.hash).toBe("#monitor");
    expect(new Router("#nope").view).toBe("chat");
  });
});
