import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { translate, ZH } from "../src/lib/i18n/i18n";
import { resolveLanguage } from "../src/lib/i18n/language.svelte";
import { appKeys, svelteKeys, typescriptKeys } from "./support/keys";

const src = join(import.meta.dirname, "../src");

async function boot(stored: string | null, languages: string[], blocked = false) {
  vi.resetModules();
  vi.stubGlobal("localStorage", {
    getItem: () => {
      if (blocked) throw Error("blocked");
      return stored;
    },
    setItem() {},
  });
  vi.stubGlobal("navigator", { languages, language: languages[0] ?? "en" });
  return (await import("../src/lib/i18n/language.svelte")).language.current;
}
afterEach(() => vi.unstubAllGlobals());

describe("language choice", () => {
  it("respects the stored choice, browser zh variants and blocked storage", async () => {
    expect(await boot(null, ["zh-CN"])).toBe("zh-Hans");
    expect(await boot("en", ["zh-TW"])).toBe("en");
    expect(await boot("zh-Hans", ["en"])).toBe("zh-Hans");
    expect(await boot("system", ["en", "zh-SG"])).toBe("zh-Hans");
    expect(await boot("bad", ["fr"])).toBe("en");
    expect(await boot("en", ["zh-CN"], true)).toBe("zh-Hans");
  });

  it("only English, zh-Hans or the browser can be chosen", () => {
    expect(resolveLanguage("fr", ["zh"])).toBe("zh-Hans");
    expect(resolveLanguage(null, [])).toBe("en");
  });
});

describe("translate", () => {
  it("keeps placeholders and the English fallback literal", () => {
    expect(translate("zh-Hans", "Models")).toBe("模型");
    expect(translate("zh-Hans", "unknown %@ / %@", ["$&", "<b>"])).toBe("unknown $& / <b>");
    expect(translate("zh-Hans", "unknown %@ %@", ["one"])).toBe("unknown one %@");
    expect(translate("en", "Models")).toBe("Models");
  });
});

describe("key collection", () => {
  it("TypeScript: finds t/th/N literals and ignores comments, strings and dynamic keys", () => {
    const source = '// t("comment")\n/* N("comment2") */\nconst a = t("real"); const b = `x ${th("nested")}`; const c = N(`tpl`); t(dynamic); const s = \'t("in string")\';';
    expect(typescriptKeys(source)).toEqual(["real", "nested", "tpl"]);
  });

  it("Svelte: finds keys in the script and in every template position", () => {
    const source = '<script lang="ts">const a = t("script");</script>\n<p title={t("attr")}>{t("text")} {#if x}{N("block")}{/if}</p>';
    expect(svelteKeys(source).sort()).toEqual(["attr", "block", "script", "text"]);
  });
});

describe("zh-Hans coverage", () => {
  it("has an entry, with matching placeholders, for every marked key in the app", () => {
    const keys = appKeys(src);
    const missing = [...keys].filter(([k]) => !Object.hasOwn(ZH, k) || !ZH[k]).map(([k, f]) => `${k}  (${f.replace(src, "src")})`);
    expect(missing).toEqual([]);
    for (const key of keys.keys()) expect((ZH[key] ?? "").match(/%@/g)?.length ?? 0, key).toBe(key.match(/%@/g)?.length ?? 0);
  });

  it("adds no markup the English does not have: a translation is text, never HTML", () => {
    const tags = (text: string) => (text.match(/<\/?[a-zA-Z][^>]*>/g) ?? []).sort();
    const added = Object.entries(ZH).filter(([key, value]) => JSON.stringify(tags(value)) !== JSON.stringify(tags(key))).map(([key]) => key);
    expect(added).toEqual([]);
  });

  it("keeps no entry the app no longer asks for", () => {
    const keys = appKeys(src);
    expect(Object.keys(ZH).filter((k) => !keys.has(k))).toEqual([]);
  });
});
