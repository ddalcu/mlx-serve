import { chat } from "../core/chat";
import type { Client } from "../core/client";
import type { Model } from "../core/models";
import { cleanReply } from "../core/rewrite";
import { t } from "../i18n/i18n";
import { audioDefaults, audioProfile, buildAudioRequest, type AudioController } from "./audio-state.svelte";
import { defaultModel } from "./connection.svelte";
import { radioLyricsRequest, radioRequest } from "./music-rewrite";

type Draft = ReturnType<typeof audioDefaults>;

/** Every track names two aspects to change, rotating through the lists so neighbours never share one. */
const ASPECTS = [
  ["a slower tempo", "a laid-back mid tempo", "an upbeat tempo", "a fast, driving tempo"],
  ["a warm, nostalgic mood", "a bright, hopeful mood", "a darker, moodier feel", "a dreamy, floating atmosphere", "a tense, cinematic mood", "a playful, light feel"],
  ["a piano lead", "a warm electric guitar lead", "soft synth pads with a plucked lead", "strings and cello", "a saxophone or flute lead", "mallets and marimba", "an analog bass-led groove", "an acoustic guitar lead"],
  ["a different sub-genre of the theme", "an unexpected sub-genre blend", "a more traditional take", "a more experimental take"],
  ["lo-fi, tape-saturated production", "clean, spacious production", "gritty, punchy drums", "wide ambient reverb", "a minimal, sparse arrangement", "a lush, layered arrangement"],
  ["a swung groove", "a straight four-on-the-floor pulse", "a syncopated rhythm", "no drums, a beatless texture"],
  ["calm, steady energy", "slowly building energy", "high, peak energy", "gentle, winding-down energy"],
  ["a 1970s flavour", "a 1980s flavour", "a 1990s flavour", "a modern, contemporary sound"],
];

/** Two aspects to change for track `index`; `salt` makes each station walk the lists differently. */
function nudge(index: number, salt: number) {
  return [0, 1]
    .map((k) => {
      const aspect = (index * 2 + k + salt) % ASPECTS.length,
        values = ASPECTS[aspect]!;
      // A small integer hash, so the value does not cycle with the aspect.
      let h = (Math.imul(index, 2654435761) + Math.imul(aspect, 40503) + Math.imul(salt, 97)) >>> 0;
      h = (h ^ (h >>> 15)) >>> 0;
      h = Math.imul(h, 2246822519) >>> 0;
      h = (h ^ (h >>> 13)) >>> 0;
      return values[h % values.length];
    })
    .join("; ");
}

/** The prompt when no chat model is there to write one. */
const fallbackPrompt = (theme: string, direction: string) => `${theme}, ${direction.replaceAll("; ", ", ")}`;

/** Tracks generated ahead of playback (one is enough: generation shares the GPU), and the prompts that made them. */
class RadioQueue<T = string> {
  ready: T[] = [];
  recentPrompts: string[] = [];
  get needsTrack() {
    return !this.ready.length;
  }
  push(prompt: string, track: T) {
    this.ready.push(track);
    this.recentPrompts = [...this.recentPrompts, prompt].slice(-8);
  }
  pop() {
    return this.ready.shift();
  }
}

/** The strip is a progress bar: a fill grows behind the status, and with no number a block sweeps to and fro. */
const STRIP_BLOCK = 24;
/** Skin pixels per second. */
const STRIP_SPEED = 40;
const fillWidth = (progress: number | null, width: number) => (progress === null ? 0 : Math.floor(width * Math.min(1, Math.max(0, progress))));
function blockOffset(time: number, width: number) {
  const travel = Math.trunc(Math.max(1, width - STRIP_BLOCK)),
    position = Math.trunc(time * STRIP_SPEED) % (2 * travel);
  return position <= travel ? position : 2 * travel - position;
}

/** A station's track waits behind a GPU that also plays the last one: short tracks, fewer refinement steps. */
const MAX_TRACK_SECONDS = 120;
const MAX_STEPS = 16;

/** The pane's settings, bent to what an endless station needs. */
function stationDraft(model: Model | undefined, d: Draft): Draft {
  const p = audioProfile(model);
  return {
    ...d,
    task: "text2music",
    bpm: "",
    keyscale: "",
    timesignature: "",
    duration: Math.min(Number(d.duration), MAX_TRACK_SECONDS),
    // A model that ends its own song is bounded by its lyrics, not by a cap that would cut them.
    autoLength: !!p?.autoLength,
    steps: p?.steps ? Math.min(Number(d.steps), MAX_STEPS) : d.steps,
  };
}

/** What the radio plays: a saved track, or the blob when saving failed. */
export type RadioTrack = { id?: string; prompt?: string; blob: Blob };
type Host = {
  readonly model: Model | undefined;
  readonly d: Draft;
  readonly c: AudioController;
  readonly server: string;
  client(): Client;
  chatModels(): Model[];
  play(track: RadioTrack): void;
  isPlaying(): boolean;
};
const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(timer), resolve()), { once: true });
  });

/** AI Radio: writes a fresh prompt per track from one theme, generates it while the last one plays, until stopped. */
class AIRadio {
  on = $state(false);
  status = $state("");
  #queue = new RadioQueue<RadioTrack>();
  #run: AbortController | null = null;
  #composing = $state(false);
  #ready = $state(false);
  readonly #host: Host;

  constructor(host: Host) {
    this.#host = host;
  }

  /** The strip's line: the generation's own progress while a track is composed. */
  get line() {
    const c = this.#host.c;
    return this.#composing && c.phase === "running" ? c.message : this.status;
  }
  /** 0...1, or null while there is no number (the strip sweeps a block). */
  get progress() {
    const c = this.#host.c;
    if (!this.#composing) return this.#ready ? 1 : null;
    return c.phase === "running" && c.total > 0 ? c.step / c.total : null;
  }

  start(theme: string) {
    theme = theme.trim();
    const model = this.#host.model;
    if (this.on || !theme || !model) return;
    this.on = true;
    this.status = t("Starting…");
    this.#queue = new RadioQueue();
    const run = new AbortController();
    this.#run = run;
    void this.#loop(theme, model, stationDraft(model, this.#host.d), run);
  }

  /** The user's Stop and giving up after failures alike: the track being composed is discarded. */
  stop(status = "") {
    if (!this.on) return;
    this.on = false;
    this.status = status;
    this.#composing = false;
    this.#run?.abort();
    this.#run = null;
    this.#queue = new RadioQueue();
    if (this.#host.c.run) this.#host.c.cancel();
  }

  /** Next: jump to the buffered track now. Also what a track ending on its own does. */
  skip() {
    if (!this.on) return false;
    const next = this.#queue.pop();
    if (next) this.#host.play(next);
    else this.status = t("Waiting for the next track…");
    return true;
  }

  async #loop(theme: string, model: Model, template: Draft, run: AbortController) {
    const salt = Math.floor(Math.random() * 1000),
      live = () => this.#run === run && !run.signal.aborted;
    let index = 0,
      failures = 0;
    while (live()) {
      // One track ahead of what plays; nothing playing after the first (playback refused) means wait, not compose more.
      if (!this.#queue.needsTrack || (index > 0 && !this.#host.isPlaying())) {
        await sleep(500, run.signal);
        continue;
      }
      this.status = t("Writing the next track…");
      this.#ready = false;
      const direction = nudge(index, salt);
      const { prompt, lyrics } = await this.#compose(theme, direction, model, template, run.signal);
      if (!live()) return;
      const p = audioProfile(model)!;
      const draft = { ...template, prompt, lyrics, instrumental: p.instrumental, seed: "" };
      this.status = t("Composing…");
      this.#composing = true;
      const c = this.#host.c;
      try {
        await c.generate(this.#host.client(), buildAudioRequest(model, draft, {}), this.#host.server);
      } catch (e) {
        c.phase = "failed";
        c.message = e instanceof Error ? e.message : t("Music generation failed.");
      }
      this.#composing = false;
      if (!live()) return;
      if (c.phase === "completed" && c.result) {
        failures = 0;
        index++;
        this.#queue.push(prompt, { id: c.result.id, prompt, blob: c.result.blob });
        if (!this.#host.isPlaying()) this.skip();
        this.status = t("Next track ready");
        this.#ready = true;
      } else {
        const why = c.message || t("Music generation failed.");
        if (++failures >= 3) return this.stop(t("AI Radio stopped: %@", [why]));
        this.status = t("Retrying: %@", [why]);
        await sleep(3000, run.signal);
      }
    }
  }

  /** The next track's style prompt (and lyrics, for a model that always sings), written by a chat model when there is one. */
  async #compose(theme: string, direction: string, model: Model, template: Draft, signal: AbortSignal) {
    const p = audioProfile(model)!,
      chatModel = defaultModel(this.#host.chatModels())?.id;
    const ask = async (request: { system: string; user: string }, maxTokens: number) => {
      if (!chatModel) return "";
      let text = "";
      try {
        const events = chat(this.#host.client(), { model: chatModel, messages: [{ role: "system", content: request.system }, { role: "user", content: request.user }], max_tokens: maxTokens }, { signal });
        for await (const e of events) if (e.type === "content") text += e.text;
      } catch {
        return "";
      }
      return cleanReply(text);
    };
    const recent = this.#queue.recentPrompts;
    const prompt = (await ask(radioRequest(theme, recent, direction, p.family, p.instrumental), 512)) || fallbackPrompt(theme, direction);
    if (p.instrumental) return { prompt, lyrics: "" };
    return { prompt, lyrics: await ask(radioLyricsRequest(theme, prompt, direction, recent, p.family, template.language), 1024) };
  }
}

export { nudge, fallbackPrompt, RadioQueue, fillWidth, blockOffset, stationDraft, AIRadio, STRIP_BLOCK, STRIP_SPEED, MAX_TRACK_SECONDS, MAX_STEPS };
