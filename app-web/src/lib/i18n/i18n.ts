// English is the key; source translations are shared with the native app.
import { language, type Locale } from "./language.svelte";
import { ZH } from "./zh-hans";

/** Mark a stored English label without translating data/protocol values. */
export const N = (key: string) => key;

export function translate(locale: Locale, key: string, args: unknown[] = []): string {
  const value = locale === "zh-Hans" && Object.hasOwn(ZH, key) ? ZH[key]! : key;
  let i = 0;
  return value.replace(/%@/g, () => (i < args.length ? String(args[i++]) : "%@"));
}

/** Reads the language rune, so a template or `$derived` that calls it re-runs when the language changes. */
export const t = (key: string, args?: unknown[]) => translate(language.current, key, args);

const escapes: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

/** Escape translated literal text before inserting already-escaped template arguments. */
export function th(key: string, args: unknown[] = []): string {
  let i = 0;
  return t(key)
    .replace(/[&<>"']/g, (c) => escapes[c] ?? c)
    .replace(/%@/g, () => (i < args.length ? String(args[i++]) : "%@"));
}

// Translate protocol enums only where they are displayed, never in request data.
const displayNames: Record<string, string> = {
  chat: N("Chat"),
  image: N("Image"),
  video: N("Video"),
  audio: N("Audio"),
  speech: N("Voice"),
  music: N("Music"),
  sound: N("Sound Effects"),
  reasoning: N("Thinking"),
  vision: N("Vision"),
  embedding: N("Embeddings"),
  ready: N("Ready"),
  idle: N("Idle"),
  loading: N("Loading"),
  running: N("Running"),
  complete: N("Completed"),
  completed: N("Completed"),
  error: N("Failed"),
  stopped: N("Stopped"),
  listening: N("Listening"),
  thinking: N("Thinking"),
  speaking: N("Speaking"),
  off: N("Off"),
  woodwinds: N("Woodwinds"),
  brass: N("Brass"),
  fx: N("Effects"),
  synth: N("Synth"),
  strings: N("Strings"),
  percussion: N("Percussion"),
  keyboard: N("Keyboard"),
  guitar: N("Guitar"),
  bass: N("Bass"),
  drums: N("Drums"),
  backing_vocals: N("Backing vocals"),
  vocals: N("Vocals"),
};
export const displayName = (value: string) => t(displayNames[value] ?? value);

export const setLanguage = (next: string) => language.set(next);
export const languageChoice = () => language.choice;
export { ZH };
