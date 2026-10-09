export const languageKey = "mlx-serve-lang";

export type Locale = "en" | "zh-Hans";
export type LanguageChoice = Locale | "system";

export function resolveLanguage(override: string | null, languages: readonly string[]): Locale {
  if (override === "en" || override === "zh-Hans") return override;
  return languages.some((language) => /^zh\b|^zh-/i.test(language)) ? "zh-Hans" : "en";
}

const choiceOf = (value: string | null): LanguageChoice => (value === "en" || value === "zh-Hans" ? value : "system");

const browserLanguages = (): readonly string[] =>
  typeof navigator === "undefined" ? [] : navigator.languages?.length ? navigator.languages : [navigator.language];

function storedChoice(): string | null {
  try {
    return localStorage.getItem(languageKey);
  } catch {
    return null;
  }
}

/** The UI language: a stored choice, else the browser's. Reading `current` inside a component tracks it. */
class Language {
  choice = $state<LanguageChoice>(choiceOf(storedChoice()));
  browser = $state(browserLanguages());
  current = $derived(resolveLanguage(this.choice, this.browser));

  constructor() {
    if (typeof window === "undefined") return;
    window.addEventListener("languagechange", () => (this.browser = browserLanguages()));
  }

  set(next: string) {
    this.choice = choiceOf(next);
    try {
      localStorage.setItem(languageKey, this.choice);
    } catch {
      /* The session choice still works. */
    }
  }
}

export const language = new Language();
