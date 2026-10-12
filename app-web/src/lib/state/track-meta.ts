import { t } from "../i18n/i18n";

// What the Modern player says about a saved clip: a title a person would give
// it, the model that made it, and its day in the history.

export const TITLE_LIMIT = 60;

const words = (line: string) => line.split(" ").filter(Boolean).length;
const isHeader = (line: string) => words(line) <= 3 && !/\p{P}/u.test(line);
const capitalized = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

function clip(text: string) {
  if (text.length <= TITLE_LIMIT) return text;
  const head = text.slice(0, TITLE_LIMIT),
    space = head.lastIndexOf(" ");
  return (space >= 0 ? head.slice(0, space) : head).replace(/^[\p{P}\s]+|[\p{P}\s]+$/gu, "") + "…";
}

/**
 * The prompt's first real line, cut to its first sentence: "Global Metadata"
 * style headers and "Label:" prefixes are skipped. Music 3's caption format
 * ends its "Basic Attributes" line with the genre, which names a track better
 * than its bpm. Null for an empty prompt.
 */
export function trackTitle(prompt: string) {
  const lines = prompt
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const body = lines.length > 1 ? lines.filter((l) => !isHeader(l)) : lines;
  let line = body[0] ?? lines[0];
  if (line === undefined) return null;
  let label = "";
  const colon = line.indexOf(":");
  if (colon >= 0 && words(line.slice(0, colon)) <= 3 && line.slice(colon + 1).trim()) {
    label = line.slice(0, colon);
    line = line.slice(colon + 1).trim();
  }
  const sentences = line
    .split(/[.!?;]/)
    .map((s) => s.trim())
    .filter(Boolean);
  line = (label === "Basic Attributes" ? sentences.at(-1) : sentences[0]) ?? "";
  return line ? capitalized(clip(line)) : null;
}

/** "org/Repo-Name-8bit" → "Repo-Name-8bit": the folder is noise in a row. */
export const modelName = (raw: string) => raw.split("/").at(-1) ?? raw;

/** HSB, the way the app picks cover hues, as CSS. */
function hsb(h: number, s: number, v: number) {
  const l = v * (1 - s / 2),
    sl = l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l);
  return `hsl(${Math.round(h * 360)} ${Math.round(sl * 100)}% ${Math.round(l * 100)}%)`;
}

/** Two colours picked from `key`: the same clip always gets the same cover, and neighbours rarely share one. */
export function coverColors(key: string): [string, string] {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(key)) hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
  const first = (hash % 360) / 360,
    spread = 0.08 + (((hash >>> 16) % 100) / 100) * 0.14;
  return [hsb(first, 0.62, 0.78), hsb((first + spread) % 1, 0.72, 0.46)];
}

const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();

function dayLabel(day: Date | null, now: Date, locale: string) {
  if (!day) return t("Earlier");
  const today = startOfDay(now),
    at = startOfDay(day);
  if (at === today) return t("Today");
  if (at === startOfDay(new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1))) return t("Yesterday");
  const year = day.getFullYear() === now.getFullYear() ? {} : { year: "numeric" as const };
  return new Intl.DateTimeFormat(locale, { weekday: "long", month: "short", day: "numeric", ...year }).format(day);
}

/** The history in day groups, newest first as given: consecutive items that share a day form one group; no date joins "Earlier". */
export function dayGroups<T>(items: T[], date: (item: T) => Date | null, now: Date, locale: string) {
  const days: { id: string; label: string; items: T[] }[] = [];
  for (const item of items) {
    const day = date(item),
      id = day ? String(startOfDay(day)) : "earlier",
      last = days.at(-1);
    if (last?.id === id) last.items.push(item);
    else days.push({ id, label: dayLabel(day, now, locale), items: [item] });
  }
  return days;
}
