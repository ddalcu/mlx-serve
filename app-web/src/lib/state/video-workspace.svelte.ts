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

  /** The reason Generate is unavailable, and the pipeline's size rounding hint. */
  get validation() {
    let error = this.error;
    try {
      buildVideoRequest(this.model, this.d, this.inputs);
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
    this.d = { ...videoDefaults(model), prompt: this.d.prompt };
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
        if (size) [this.d.width, this.d.height] = size as [number, number];
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
