import { flushSync } from "svelte";
import { expect, it } from "vitest";
import { setLanguage, t } from "../src/lib/i18n/i18n";

it("t() re-runs its dependents when the language changes", () => {
  setLanguage("en");
  const seen: string[] = [];
  const stop = $effect.root(() => {
    $effect(() => {
      seen.push(t("Models"));
    });
  });
  flushSync();
  setLanguage("zh-Hans");
  flushSync();
  setLanguage("system");
  flushSync();
  stop();
  expect(seen).toEqual(["Models", "模型", "Models"]);
});
