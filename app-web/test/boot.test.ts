import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { resolveLanguage } from "../src/lib/i18n/language.svelte";

const html = readFileSync(join(import.meta.dirname, "../index.html"), "utf8");
const script = /<script>([\s\S]*?)<\/script>/.exec(html)![1]!;

type Env = { interface?: string | "throws"; lang?: string | null; dark?: boolean; languages?: string[] };

function boot({ interface: stored, lang = null, dark = false, languages = ["en-US"] }: Env) {
  const root = { dataset: {} as Record<string, string>, lang: "" };
  const env = {
    document: { documentElement: root },
    localStorage: {
      getItem(key: string) {
        if (stored === "throws") throw Error("blocked");
        return key === "studio.interface" ? (stored ?? null) : lang;
      },
    },
    matchMedia: () => ({ matches: dark }),
    navigator: { languages, language: languages[0] },
  };
  new Function(...Object.keys(env), script)(...Object.values(env));
  return root;
}

describe("pre-paint boot script", () => {
  it("runs in the head before the app module", () => {
    expect(html.indexOf("<script>")).toBeGreaterThan(html.indexOf("<head>"));
    expect(html.indexOf("<script>")).toBeLessThan(html.indexOf("</head>"));
    expect(html.indexOf("<script>")).toBeLessThan(html.indexOf('type="module"'));
  });

  it("honors the stored theme and falls back to the OS setting", () => {
    for (const theme of [null, "light", "dark", "system", "bogus", "broken", "throws"])
      for (const dark of [false, true]) {
        const stored = theme === "broken" ? "[" : theme === "throws" ? "throws" : theme === null ? undefined : JSON.stringify({ theme });
        expect(boot({ interface: stored, dark }).dataset.theme, `${theme} dark=${dark}`).toBe(theme === "light" || theme === "dark" ? theme : dark ? "dark" : "light");
      }
  });

  it("applies the stored accent, defaulting to system", () => {
    expect(boot({ interface: JSON.stringify({ accent: "green" }) }).dataset.accent).toBe("green");
    expect(boot({ interface: JSON.stringify({ accent: 7 }) }).dataset.accent).toBe("system");
    expect(boot({ interface: "throws" }).dataset.accent).toBe("system");
  });

  it("picks the language exactly as the app does", () => {
    for (const lang of [null, "en", "zh-Hans", "system", "bad"])
      for (const languages of [["en-US"], ["zh-CN"], ["fr", "zh-TW"], ["zh"]])
        expect(boot({ lang, languages }).lang, `${lang} ${languages}`).toBe(resolveLanguage(lang, languages));
  });
});
