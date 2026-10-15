import { Library, type LibraryItem } from "../core/library";
import { audioRequest } from "../core/media";
import type { Model } from "../core/models";
import { t } from "../i18n/i18n";
import { audioReference } from "./audio-reference";
import type { Connection } from "./connection.svelte";
import { imageReference } from "../core/image-reference";
import {
  buildVideoRequest,
  frameOptions,
  qualities,
  videoDefaults,
  videoProfile,
  videoQuality,
  videoSize,
  VideoController,
  type VideoInputs,
} from "./video-state.svelte";
import { videoReference } from "./video-reference";
import type { VideoRequest } from "../core/video";
import { newId } from "../core/id";
import { storyboardRewrite, videoRewrite, type RewriteRequest } from "../core/rewrite";
import { deliveredFrames, lengthLabel, LONGEST_PLANNED_STORY, parseStoryboard, plannedRange, secondsRange, shotFrames, type SecondsRange } from "../core/storyboard";

export type MediaKey = "first" | "last" | "audio" | "images" | "videos" | "audios";
const LISTS = ["images", "videos", "audios"] as const;
const PRESET_KEYS = ["mode", "steps", "cfg", "stg", "frames", "turbo"] as const;

/** The video pane's draft (kept per server), media inputs, recording and generation, without any drawing. */
export class VideoWorkspace {
  readonly c: VideoController;
  readonly library: Library;
  // A separate draft database keeps reference media out of the results gallery.
  readonly drafts: Pick<Library, "get" | "put">;
  d = $state(videoDefaults());
  inputs = $state<VideoInputs>({});
  ready = $state(false);
  busy = $state(false);
  recording = $state(false);
  silentClip = $state(false);
  error = $state("");
  storageError = $state("");
  server: string;
  #connection: Connection;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #saveChain: Promise<void> = Promise.resolve();
  #epoch = 0;
  #hydration = 0;
  #signature = "";
  #activity: AbortController | null = null;
  #recorder: MediaRecorder | null = null;
  #mic: MediaStream | null = null;
  #recordingTimer: ReturnType<typeof setTimeout> | undefined;

  constructor(connection: Connection, library: Library, drafts: Pick<Library, "get" | "put">) {
    this.#connection = connection;
    this.library = library;
    this.drafts = drafts;
    this.c = new VideoController(library);
    this.server = connection.active.url;
  }

  get models() {
    return this.#connection.models.filter((m) => videoProfile(m));
  }
  get model() {
    return this.models.find((m) => m.id === this.d.model);
  }
  get profile() {
    return videoProfile(this.model);
  }
  get frames() {
    return frameOptions(this.model, this.d, this.inputs);
  }
  /** Which quality preset the current settings match, or "Custom". */
  get quality() {
    for (const q of [this.d.quality, ...qualities]) {
      const preset = videoQuality(this.model, q) as Record<string, unknown>;
      if (PRESET_KEYS.every((k) => preset[k] === (this.d as Record<string, unknown>)[k])) return q;
    }
    return "Custom";
  }
  get references() {
    return (this.inputs.images?.length || 0) + (this.inputs.videos?.length || 0) + (this.inputs.audios?.length || 0);
  }
  /** Whether work is in flight (generating, converting or recording). */
  get working() {
    return !!this.c.run || this.busy;
  }

  /** Storyboard mode, on a model whose shots can open on a first frame (REF2VA has no first-frame anchor). */
  get storyboardOn() {
    return this.d.storyboard && !!this.profile && !this.profile.references;
  }
  /** The rungs a shot can use: this canvas's ladder, one window, within one response's raw-frame budget. */
  get shotLadder() {
    return frameOptions(this.model, { ...this.d, windows: 1 }, { ...this.inputs, audio: undefined });
  }
  get shotRange(): SecondsRange {
    return secondsRange(this.shotLadder, 24, this.profile?.statedFrames ?? 0);
  }
  get shotFrames() {
    const [lo, hi] = this.shotRange;
    return this.d.shots.map((s) => shotFrames(Math.min(hi, Math.max(lo, s.seconds)), 24, this.shotLadder));
  }
  get storyboardSeconds() {
    return Math.round(deliveredFrames(this.shotFrames) / 24);
  }
  /** One line under Quality: the pipeline, the steps and the length the request will run. */
  get qualityNote() {
    const p = this.profile;
    if (!p) return "";
    const mode = p.audio && this.inputs.audio && this.d.mode === "one_stage" && !this.storyboardOn
        ? t("2-stage (audio-to-video)")
        : { one_stage: t("1-stage"), two_stage: t("2-stage"), two_stage_hq: t("2-stage HQ") }[this.d.mode] ?? this.d.mode,
      turbo = p.turbo && this.d.turbo ? t(" (Turbo)") : "";
    return this.storyboardOn
      ? t("%@, %@ steps%@, %@ shots · %@", [mode, this.d.steps, turbo, this.d.shots.length, lengthLabel(this.storyboardSeconds)])
      : t("%@, %@ steps%@, %@ frames (~%@ s)", [mode, this.d.steps, turbo, this.d.frames, (this.d.frames / 24).toFixed(1)]);
  }

  /** A storyboard's base request: the pane's settings with the first shot's length and no soundtrack. */
  #storyboardBase(): VideoRequest {
    if (!this.d.shots.length) throw Error(t("Add a shot."));
    if (this.d.shots.some((s) => !s.prompt.trim())) throw Error(t("Every shot needs a prompt."));
    const d = { ...this.d, prompt: this.d.prompt.trim() || this.d.shots[0]!.prompt, frames: this.shotFrames[0]!, windows: 1 };
    return buildVideoRequest(this.model, d, { ...this.inputs, audio: undefined });
  }

  addShot() {
    this.d.shots.push({ id: newId(), prompt: "", seconds: plannedRange(this.shotRange)[1] });
  }
  removeShot(id: string) {
    this.d.shots = this.d.shots.filter((s) => s.id !== id);
  }

  /** Past one shot, or with the storyboard already on, Enhance plans shots. */
  plansStoryboard(seconds: number) {
    return !!this.profile && !this.profile.references && (this.storyboardOn || seconds > this.shotRange[1]);
  }
  /** Enhance's length slider: a storyboard-capable model runs to two minutes, anything else to its longest clip. */
  get enhanceLength() {
    const p = this.profile;
    if (!p || p.references) {
      const max = Math.max(1, Math.floor((this.frames.at(-1) ?? this.d.frames) / 24));
      return { initial: Math.min(max, Math.max(1, Math.round(this.d.frames / 24))), min: 1, max };
    }
    const initial = this.storyboardOn ? (this.d.shots.length ? this.storyboardSeconds : 60) : Math.round(this.d.frames / 24);
    return { initial: Math.min(LONGEST_PLANNED_STORY, Math.max(1, initial)), min: 1, max: LONGEST_PLANNED_STORY };
  }
  /** What Enhance asks the chat model for at `seconds`; the first frame rides along for a model that can see it. */
  rewriteRequest(seconds: number): RewriteRequest {
    const p = this.profile,
      format = p?.h3 ? (p.references ? "h3Reference" : "h3Base") : "ltx",
      first = this.inputs.first?.base64;
    return this.plansStoryboard(seconds)
      ? storyboardRewrite(this.d.prompt, format, seconds, plannedRange(this.shotRange), first)
      : videoRewrite(this.d.prompt, format, seconds, first);
  }
  /** A planned storyboard fills the shots and turns the mode on; anything else replaces the prompt. */
  applyRewrite(text: string, seconds: number) {
    const shots = parseStoryboard(text, this.shotRange);
    if (shots.length) {
      this.d.shots = shots;
      this.d.storyboard = true;
      return;
    }
    this.d.prompt = text;
    if (!this.plansStoryboard(seconds)) this.d.frames = this.frames.find((f) => f >= seconds * 24) ?? this.frames.at(-1) ?? this.d.frames;
  }
  /** Why `text` written at `seconds` cannot be applied, or "". */
  rewriteError(text: string, seconds: number) {
    return this.plansStoryboard(seconds) && !parseStoryboard(text, this.shotRange).length ? t("No shots found. Each shot starts with a line like === SHOT 1 | 8s ===.") : "";
  }
  /** A line under Enhance's length slider when that length plans a storyboard. */
  enhanceNote(seconds: number) {
    const longest = plannedRange(this.shotRange)[1];
    return this.plansStoryboard(seconds) ? t("Enhance writes a storyboard: about %@ shots of up to %@ s each.", [Math.ceil(seconds / longest), longest]) : "";
  }

  /** The reason Generate is unavailable, and the pipeline's size rounding hint. */
  get validation() {
    let error = this.error;
    try {
      if (this.storyboardOn) this.#storyboardBase();
      else buildVideoRequest(this.model, this.d, this.inputs);
    } catch (e) {
      error ||= e instanceof Error ? e.message : t("Invalid settings.");
    }
    let hint = "";
    try {
      const s = videoSize(this.model, this.d, this.inputs);
      hint = s.width !== this.d.width || s.height !== this.d.height ? t("Rounded to %@ × %@ for this pipeline.", [s.width, s.height]) : "";
    } catch (e) {
      hint = e instanceof Error ? e.message : "";
    }
    return { error, hint };
  }

  async init() {
    const epoch = ++this.#hydration;
    this.ready = false;
    try {
      await this.#saveChain;
      const row = await this.drafts.get(this.server);
      if (epoch !== this.#hydration) return;
      if (row) {
        const value = JSON.parse(await row.blob.text());
        if (epoch !== this.#hydration) return;
        const d = videoDefaults();
        for (const key of Object.keys(d) as (keyof typeof d)[]) if (typeof value.d?.[key] === typeof d[key]) Object.assign(d, { [key]: value.d[key] });
        // A stored draft is untrusted: keep only well-formed shots.
        d.shots = (Array.isArray(d.shots) ? d.shots : []).filter((s) => typeof s?.prompt === "string" && Number.isFinite(s?.seconds)).map((s) => ({ id: typeof s.id === "string" ? s.id : newId(), prompt: s.prompt, seconds: s.seconds }));
        this.d = d;
        this.inputs = value.inputs || {};
      }
    } catch {
      this.storageError = t("Could not restore the video draft. Changes remain in this tab.");
    }
    if (epoch !== this.#hydration) return;
    this.ready = true;
    this.#signature = this.signature();
    this.connectionChanged();
  }

  /** Cheap change detector for the draft: every field, but only the inputs' names and sizes. */
  signature() {
    const names = (v: unknown): unknown => (Array.isArray(v) ? v.map(names) : v && typeof v === "object" ? [(v as { name?: string }).name, JSON.stringify(v).length] : v);
    return JSON.stringify([this.d, Object.entries(this.inputs).map(([k, v]) => [k, names(v)])]);
  }

  /** Save the draft shortly after it stops changing. */
  touch() {
    if (!this.ready) return;
    const signature = this.signature();
    if (signature === this.#signature) return;
    this.#signature = signature;
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => this.flush(), 250);
  }

  flush() {
    clearTimeout(this.#timer);
    if (!this.ready) return;
    const server = this.server,
      snapshot = JSON.stringify({ d: this.d, inputs: this.inputs });
    this.#saveChain = this.#saveChain.then(async () => {
      try {
        await this.drafts.put(server, { type: "video", model: "", server, blob: new Blob([snapshot], { type: "application/json" }) });
        if (server === this.server) this.storageError = "";
      } catch {
        if (server === this.server) this.storageError = t("Video draft could not be saved. Keep this tab open and download results.");
      }
    });
  }

  connectionChanged() {
    const url = this.#connection.active.url;
    if (this.server !== url) {
      this.flush();
      this.stopActivity();
      this.c.cancel();
      this.c.result = null;
      this.c.phase = "idle";
      this.d = videoDefaults();
      this.inputs = {};
      this.server = url;
      this.error = "";
      this.storageError = "";
      void this.init();
      return;
    }
    if (!this.ready) return;
    if (!this.d.model && this.models.length) this.d = { ...videoDefaults(this.models.find((m) => m.loaded) || this.models[0]), prompt: this.d.prompt };
  }

  chooseModel(id: string) {
    const model = this.models.find((m) => m.id === id);
    this.stopActivity();
    this.d = { ...videoDefaults(model), prompt: this.d.prompt, storyboard: this.d.storyboard, shots: this.d.shots };
    if (!videoProfile(model)?.audio) delete this.inputs.audio;
    this.clamp();
  }

  setQuality(name: string) {
    Object.assign(this.d, videoQuality(this.model, name));
    this.clamp();
  }

  /** `value` is `WxH`, or "source" to pick the pipeline size closest to the starting frame's shape. */
  setSize(value: string) {
    if (value === "source") {
      const input = this.inputs.first;
      if (input?.width && input.height) {
        const ratio = input.width / input.height;
        const size = [...(this.profile?.sizes || [])].sort((a, b) => Math.abs(a[0]! / a[1]! - ratio) - Math.abs(b[0]! / b[1]! - ratio))[0];
        if (size) [this.d.width, this.d.height] = [size[0], size[1]];
      }
    } else [this.d.width, this.d.height] = value.split("x").map(Number) as [number, number];
    this.clamp();
  }

  /** Keep the frame count on a value this model and canvas allow. */
  clamp() {
    const options = frameOptions(this.model, this.d, this.inputs);
    if (options.length && !options.includes(this.d.frames)) this.d.frames = options.filter((f) => f <= this.d.frames).at(-1) || options[0]!;
  }

  /** Lengthen the clip to cover an attached soundtrack. */
  coverAudio() {
    const options = frameOptions(this.model, this.d, this.inputs),
      needed = (this.inputs.audio?.duration || 0) * 24;
    this.d.frames = options.find((f) => f >= needed) || options.at(-1) || this.d.frames;
  }

  stopActivity() {
    this.#epoch++;
    this.#activity?.abort();
    this.#activity = null;
    this.busy = false;
    clearTimeout(this.#recordingTimer);
    if (this.#recorder?.state === "recording") this.#recorder.stop();
    this.#mic?.getTracks().forEach((track) => track.stop());
    this.#mic = null;
    this.#recorder = null;
    this.recording = false;
  }

  /** Cancel a generation, a conversion or a recording. */
  cancel() {
    if (this.#recorder) {
      this.#recorder.stop();
      return;
    }
    this.stopActivity();
    this.c.cancel();
  }

  remove(key: MediaKey, index?: number) {
    if (index !== undefined) (this.inputs[key as (typeof LISTS)[number]] as unknown[]).splice(index, 1);
    else delete this.inputs[key as "first" | "last" | "audio"];
  }

  async attach(key: MediaKey, files: File[]) {
    if (!files.length || this.busy || this.c.run) return;
    this.error = "";
    this.busy = true;
    const epoch = this.#epoch,
      run = new AbortController();
    this.#activity = run;
    try {
      for (const file of files) {
        const value =
          key === "first" || key === "last" || key === "images"
            ? await imageReference(file, file.name)
            : key === "videos"
              ? await videoReference(file, this.d.frames, !this.silentClip, run.signal)
              : await audioReference(file, file.name, "music", 30);
        if (epoch !== this.#epoch || run.signal.aborted) return;
        if ((LISTS as readonly string[]).includes(key)) {
          const list = ((this.inputs as Record<string, unknown[]>)[key] ??= []);
          const total = this.references;
          if (list.length >= (key === "images" ? 9 : 3) || total >= 12) throw Error(t("Reference limit reached: 9 images, 3 clips, 3 audio, 12 total."));
          list.push(value);
        } else Object.assign(this.inputs, { [key]: value });
        if (key === "audio") this.coverAudio();
      }
    } catch (e) {
      if (epoch === this.#epoch) this.error = e instanceof Error ? e.message : t("Could not read media.");
    } finally {
      if (epoch === this.#epoch) {
        this.busy = false;
        this.#activity = null;
      }
    }
  }

  /** Make speech with the server's Qwen3-TTS model and attach it as the soundtrack. */
  async createSpeech(model: Model, text: string) {
    this.d.speech = text;
    const epoch = this.#epoch,
      run = new AbortController();
    this.#activity = run;
    this.busy = true;
    try {
      const result = await audioRequest(this.#connection.client(), "/v1/audio/speech", { model: model.id, input: text }, { stream: true, signal: run.signal });
      const reference = await audioReference(result.blob, t("Generated speech.wav"), "music", 30);
      if (epoch !== this.#epoch || run.signal.aborted) return;
      this.inputs.audio = reference;
      this.coverAudio();
    } finally {
      if (epoch === this.#epoch) {
        this.busy = false;
        this.#activity = null;
      }
    }
  }

  /** Record the soundtrack from the microphone (up to 8 seconds); a second call stops it. */
  async record() {
    if (this.#recorder) {
      this.#recorder.stop();
      return;
    }
    this.error = "";
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !globalThis.MediaRecorder) {
      this.error = !window.isSecureContext ? t("Recording needs HTTPS or localhost") : t("Recording is unavailable in this browser. Choose an audio file instead.");
      return;
    }
    const epoch = this.#epoch;
    this.busy = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (epoch !== this.#epoch) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      this.#mic = stream;
      const recorder = new MediaRecorder(stream),
        parts: Blob[] = [];
      this.#recorder = recorder;
      this.recording = true;
      recorder.ondataavailable = (e) => {
        if (e.data.size) parts.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        clearTimeout(this.#recordingTimer);
        if (epoch !== this.#epoch) return;
        this.#recorder = null;
        this.#mic = null;
        this.recording = false;
        try {
          const ref = await audioReference(new Blob(parts, { type: recorder.mimeType }), "Recording.wav", "music", 30);
          if (epoch !== this.#epoch) return;
          this.inputs.audio = ref;
          this.coverAudio();
        } catch (e) {
          this.error = e instanceof Error ? e.message : t("Recording failed.");
        }
        this.busy = false;
      };
      recorder.start();
      this.#recordingTimer = setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, 8000);
    } catch (e) {
      this.busy = false;
      this.error = e instanceof Error ? e.message : t("Microphone unavailable. Choose an audio file.");
    }
  }

  async generate() {
    this.error = "";
    try {
      if (this.storyboardOn) {
        const base = this.#storyboardBase(),
          frames = this.shotFrames;
        this.flush();
        await this.c.generateStoryboard(this.#connection.client(), base, this.d.shots.map((s, i) => ({ prompt: s.prompt, frames: frames[i]! })), this.server);
        return;
      }
      const request = buildVideoRequest(this.model, this.d, this.inputs);
      this.flush();
      await this.c.generate(this.#connection.client(), request, this.server);
    } catch (e) {
      this.error = e instanceof Error ? e.message : t("Generation failed.");
    }
  }

  /** Show a saved gallery clip in the preview. */
  show(item: LibraryItem) {
    if (this.c.run) return;
    this.c.result = { ...item, elapsedMs: 0 };
    this.c.phase = "completed";
    this.c.saveError = "";
  }
}
