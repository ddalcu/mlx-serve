import { t, N } from "../i18n/i18n";
import { audioRequest } from "../core/media";
import { buildMusicRequest } from "../core/music";
import { randomSeed } from "./image-state.svelte";
import { kokoroVoices, trackClasses, keys, languages } from "./audio-presets";
import type { Model } from "../core/models";
import type { LibraryInput } from "../core/library";
import type { Client } from "../core/client";
import type { Library } from "../core/library";
import type { MusicRequest } from "../core/music";
import { MediaRun } from "./media-run.svelte";
export type AudioTab = 'voice' | 'music' | 'sound';
function audioProfile(model: Model | undefined) {
  if (
    !model ||
    !model.capabilities.some((c) =>
      ["audio", "speech", "music", "sound"].includes(c),
    )
  )
    return;
  const family = String(model.architecture || model.meta.architecture || "");
  if (
    ![
      "kokoro",
      "qwen3_tts",
      "acestep",
      "minimax_music3",
      "stable_audio3",
      "yue2",
    ].includes(family)
  )
    return;
  const voice = ["kokoro", "qwen3_tts"].includes(family),
    ace = family === "acestep",
    music = ace || ["minimax_music3", "yue2"].includes(family),
    yue2 = family === "yue2";
  return {
    family,
    tab: ((voice ? "voice" : music ? "music" : "sound") as AudioTab),
    voices: family === "kokoro",
    speed: family === "kokoro",
    reference: family === "qwen3_tts" || ace,
    source: ace,
    meta: ace,
    instrumental: !yue2,
    tempo: !yue2,
    /** YuE2 plans an ABC score first (`cot`) and renders one you paste (`abc`). */
    score: yue2,
    /** YuE2 ends its own song; its duration is only a cap. */
    autoLength: yue2,
    steps: !voice && !ace,
    stepsRange: music ? (yue2 ? [1, 100] : [4, 100]) : [1, 50],
    duration: music ? (ace ? [10, 600] : [5, 360]) : [0.5, 120],
  };
}
/** Seconds a YuE2 length cap is stretched by, so the last line can finish rather than be cut. */
/** YuE2's `cot` values: plan melody and chords, melody only, or no score. */
const musicPlans = ["full", "melody", "off"];
const lengthGrace = (seconds: number) => Math.max(10, Math.floor(seconds / 4));
/** YuE2's own stages; nil leaves the shared "stage…" message. */
function musicStageLabel(e: Record<string, unknown>) {
  const step = Number(e.step) || 0,
    total = Number(e.total) || 0;
  switch (e.stage) {
    case "abc":
      return t("Writing the score…");
    case "semantic":
      return t("Composing (%@ s so far)…", [Math.floor(step / 25)]);
    case "nar":
      return t("Rendering (step %@/%@)…", [step, total]);
  }
}
function audioDefaults(model?: Model | undefined) {
  const p = audioProfile(model);
  return {
    model: model?.id || "",
    prompt: "",
    lyrics: "",
    voice: "af_heart",
    speed: 1,
    duration: p?.tab === "music" ? 60 : 10,
    autoLength: true,
    plan: "full",
    score: "",
    instrumental: false,
    steps: p?.family === "yue2" ? 32 : p?.tab === "music" ? 30 : 8,
    bpm: "",
    keyscale: "",
    timesignature: "",
    language: "en",
    seed: "",
    task: "text2music",
    coverStrength: 1,
    coverNoise: 0,
    trackClasses: ([] as string[]),
    advanced: true,
  };
}
export type AudioReference = { base64: string; name?: string; duration?: number; };
function range(value: unknown, lo: number, hi: number, name: string) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < lo || n > hi)
    throw Error(t("%@ must be %@–%@.", [name, lo, hi]));
  return n;
}
function buildAudioRequest(model: Model | undefined, d: ReturnType<typeof audioDefaults>, clips: { reference?: AudioReference; source?: AudioReference; } = {}, random: () => number = randomSeed): { path: string; type: "speech" | "music" | "sound"; body: Record<string, unknown>; } {
  const p = audioProfile(model);
  if (!model || !p) throw Error(t("Choose a supported model on this server."));
  if (!d.prompt.trim())
    throw Error(
      p.tab === "voice"
        ? t("Enter text to be generated.")
        : t("Enter a prompt."),
    );
   const body: Record<string, unknown> = { model: model.id };
  if (p.tab === "voice") {
    body.input = d.prompt.trim();
    if (p.voices) {
      const voices = d.voice
        .split(",")
        .map((v) => v.trim())
        .filter(Boolean);
      if (!voices.length || voices.some((v) => !kokoroVoices.includes(v)))
        throw Error(t("Choose a valid Kokoro voice or comma-separated blend."));
      body.voice = [...new Set(voices)].join(",");
      body.speed = range(d.speed, 0.5, 2, t("Speed"));
    }
    if (p.reference && clips.reference) body.ref_audio = clips.reference.base64;
    return {
      path: "/v1/audio/speech",
      type: ("speech" as 'speech'),
      body,
    };
  }
  body.prompt = d.prompt.trim();
  if (!(p.autoLength && d.autoLength)) {
    const duration = range(d.duration, p.duration[0], p.duration[1], t("Duration"));
    body.duration_seconds = p.autoLength ? Math.min(Math.round(duration) + lengthGrace(Math.round(duration)), p.duration[1]) : duration;
  }
  const seed = d.seed.trim();
  body.seed =
    !seed || seed === "-1" ? random() : range(seed, 0, 4294967295, t("Seed"));
  if (!Number.isInteger(body.seed))
    throw Error(t("Seed must be a whole number."));
  if (p.steps) {
    body.steps = range(d.steps, p.stepsRange[0], p.stepsRange[1], t("Steps"));
    if (!Number.isInteger(body.steps))
      throw Error(t("Steps must be a whole number."));
  }
  if (p.tab === "sound")
    return {
      path: "/v1/audio/sound-generations",
      type: ("sound" as 'sound'),
      body,
    };
  if (p.instrumental && d.instrumental) body.instrumental = true;
  else if (d.lyrics.trim()) body.lyrics = d.lyrics.trim();
  else if (!p.instrumental) throw Error(t("This model requires lyrics."));
  else if (!p.meta)
    throw Error(t("This model requires lyrics or Instrumental."));
  if (p.tempo && d.bpm.trim()) {
    body.bpm = range(d.bpm, 30, 300, t("Tempo"));
    if (!Number.isInteger(body.bpm))
      throw Error(t("Tempo must be a whole number."));
  }
  if (p.tempo && d.keyscale) {
    if (!keys.includes(d.keyscale)) throw Error(t("Choose a valid key."));
    body.keyscale = d.keyscale;
  }
  if (p.meta) {
    if (d.language) {
      if (!languages.some(([, id]) => id === d.language))
        throw Error(t("Choose a vocal language."));
      body.vocal_language = d.language;
    }
    if (d.timesignature) {
      if (!["2", "3", "4", "6"].includes(d.timesignature))
        throw Error(t("Choose a time signature."));
      body.timesignature = d.timesignature;
    }
  }
  if (p.reference && clips.reference) body.ref_audio = clips.reference.base64;
  if (p.score) {
    if (!musicPlans.includes(d.plan)) throw Error(t("Choose a score plan."));
    body.cot = d.plan;
    const abc = d.score.trim();
    if (d.plan !== "off" && abc) body.abc = abc;
  }
  if (p.source && d.task !== "text2music") {
    if (!["cover", "complete"].includes(d.task))
      throw Error(t("Choose a music mode."));
    if (!clips.source) throw Error(t("Choose source audio for this mode."));
    body.task = d.task;
    body.src_audio = clips.source.base64;
    if (d.task === "cover") {
      body.cover_strength = range(d.coverStrength, 0, 1, t("Cover strength"));
      body.cover_noise_strength = range(
        d.coverNoise,
        0,
        1,
        t("Noise strength"),
      );
    } else if (d.trackClasses.length) {
      if (d.trackClasses.some((t) => !trackClasses.includes(t)))
        throw Error(t("Choose valid instruments."));
      body.track_classes = d.trackClasses;
    }
  }
  return {
    path: "/v1/audio/music-generations",
    type: ("music" as 'music'),
    body: buildMusicRequest(
      (body as MusicRequest),
    ),
  };
}
class AudioController extends MediaRun<LibraryInput & { id?: string; elapsedMs: number }> {
  protected readonly busy = t("An audio request is already running.");
  protected readonly saveFailed = N("Not saved in this browser: %@. Download the audio or retry saving.");
  request: typeof audioRequest;
  /** The ABC score the last YuE2 song was rendered from, for "Use last score". */
  lastScore = $state("");

  constructor(library: Pick<Library, "add">, generate: typeof audioRequest = audioRequest) {
    super(library);
    this.request = generate;
  }

  async generate(client: Client, request: ReturnType<typeof buildAudioRequest>, server: string) {
    const { body, path, type } = request;
    const run = this.begin(Number(body.steps) || 0);
    try {
      const progress = this.progress(run);
      const onProgress = (e: Record<string, unknown>) => {
        progress(e);
        const label = musicStageLabel(e);
        if (label && this.run === run) this.message = label;
      };
      const result = await this.request(client, path, body, { stream: true, signal: run.signal, onProgress });
      if (this.run !== run || run.signal.aborted) return;
      if (result.abc) this.lastScore = result.abc;
      await this.complete({
        type,
        model: String(body.model ?? ""),
        server,
        prompt: String(body.input ?? body.prompt ?? ""),
        blob: result.blob,
        createdAt: Date.now(),
        elapsedMs: result.elapsedMs,
      });
    } catch (e) {
      this.fail(run, e);
    } finally {
      this.end(run);
    }
  }
}

export { audioProfile, audioDefaults, buildAudioRequest, AudioController, lengthGrace, musicPlans, musicStageLabel };
