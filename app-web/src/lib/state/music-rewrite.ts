import { languages, musicLyrics, musicStyles, music3Styles, sectionTags, yue2Lyrics, yue2SectionTags, yue2Styles } from "./audio-presets";

/** Prompts that have a chat model write a music style or lyrics shaped like the family's own examples. */
export type Rewrite = { system: string; user: string };

const styleExamples = (family: string) => (family === "acestep" ? musicStyles : family === "yue2" ? yue2Styles : music3Styles);
const lyricsExamples = (family: string) => (family === "yue2" ? yue2Lyrics : musicLyrics);
const tagsFor = (family: string) => (family === "yue2" ? yue2SectionTags : sectionTags);
const languageName = (code: string) => languages.find(([, id]) => id === code)?.[0] ?? code;
const examples = (list: { body: string }[]) => list.map((e) => e.body).join("\n\n---\n\n");
/** How many earlier prompts a radio request shows the model. */
const RECENT_SHOWN = 6;

function lyricsSystem(family: string) {
  return `You write song lyrics for a text-to-music model. Use section tags on their own lines, exactly like the examples: ${tagsFor(family).join(" ")}. Keep the user's theme and any lines they wrote; tighten rhythm and rhyme. Reply with ONLY the lyrics, no title, no preamble, no quotes, no markdown.\n\nExamples of the expected format:\n\n${examples(lyricsExamples(family))}`;
}

function styleSystem(family: string) {
  const shape =
    family === "minimax_music3"
      ? "Write the caption in the exact three-block format of the examples (Global Metadata / Vocal Details / Arrangement) with the same labelled lines, including bpm, key and scale."
      : family === "yue2"
        ? "Write ONE line of comma-separated tags like the examples: language or genre first, then instruments, mood and the lead vocal. No sentences, no tempo or key (the score carries those)."
        : "Write ONE paragraph of plain prose like the examples: genre, mood, instruments, production. No tempo, key or time signature (those are separate controls). No headings, no lists.";
  return `You rewrite music style prompts for a text-to-music model. ${shape} Keep the user's intent; make it more specific and evocative. Reply with ONLY the rewritten prompt, no preamble, no quotes, no markdown.\n\nExamples of the expected format:\n\n${examples(styleExamples(family))}`;
}

/** The Enhance… dialog: rewrite the style prompt or the lyrics. `other` is the other field, for context. */
function rewriteRequest(field: "prompt" | "lyrics", text: string, family: string, other: string, instrumental: boolean, language: string): Rewrite {
  const context = other.trim();
  if (field === "prompt") {
    let user = `Rewrite this style prompt:\n\n${text}`;
    if (instrumental) user += "\n\nThe track is instrumental: no vocals.";
    else if (context) user += `\n\nIt will sing these lyrics (for mood and language):\n${context}`;
    return { system: styleSystem(family), user };
  }
  let user = `Rewrite these lyrics in ${languageName(language)}:\n\n${text}`;
  if (context) user += `\n\nThe music style is:\n${context}`;
  return { system: lyricsSystem(family), user };
}

/** AI Radio: the style prompt for the next track. The theme never changes, so the request names recent tracks and a nudge. */
function radioRequest(theme: string, recent: string[], nudge: string, family: string, instrumental: boolean): Rewrite {
  const system =
    styleSystem(family) +
    `\n\nYou are the programme director of an endless radio station with ONE fixed theme. Each request asks for the style prompt of the NEXT track. Stay inside the theme, but make this track clearly different from the recent ones: change at least two of tempo, mood, lead instrument, sub-genre, era or production, rhythm, energy. Never reuse the instrument and mood combination of a recent track. Keep any language, culture or genre the theme names (for example "Romanian") in the prompt. Treat the "lean toward" line as a hard requirement and fold it into the prompt naturally. Write a fresh prompt, do not edit an earlier one.`;
  let user = `Station theme: ${theme}`;
  const shown = recent.slice(-RECENT_SHOWN);
  if (shown.length) user += `\n\nRecent tracks, oldest first (this one must not sound like them):\n${shown.map((p) => `- ${p}`).join("\n")}`;
  user += `\n\nThis track should lean toward: ${nudge}.`;
  if (instrumental) user += "\n\nThe track is instrumental: no vocals.";
  return { system, user };
}

/** AI Radio, for a model that always sings: lyrics in the language the theme names, the pane's language as the fallback. */
function radioLyricsRequest(theme: string, style: string, nudge: string, recent: string[], family: string, fallbackLanguage: string): Rewrite {
  let user = `Write original lyrics for the next song on a radio station.\n\nStation theme: ${theme}\n\nWrite the lyrics in the language the station theme names or implies (a Romanian theme gets Romanian lyrics); if it names none, write them in ${languageName(fallbackLanguage)}.\n\nThe music style is:\n${style}\n\nMood and feel: ${nudge}.\n\nThe song ends by itself after the last line, so keep it to two verses and a chorus: at most 16 sung lines.`;
  const shown = recent.slice(-RECENT_SHOWN);
  if (shown.length) user += `\n\nEarlier songs on the station had these styles; choose a different subject and imagery than they would suggest:\n${shown.map((p) => `- ${p}`).join("\n")}`;
  return { system: lyricsSystem(family), user };
}

export { styleExamples, lyricsExamples, tagsFor, rewriteRequest, radioRequest, radioLyricsRequest };
