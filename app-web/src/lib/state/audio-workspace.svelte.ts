import { Library, type LibraryItem } from "../core/library";
import { decodeBase64 } from "../core/media";
import { t } from "../i18n/i18n";
import type { StorageAdapter } from "../core/servers";
import { AmpPlayer } from "./amp-player.svelte";
import { soundPrompts } from "./audio-presets";
import { audioReference } from "./audio-reference";
import {
  audioDefaults,
  audioProfile,
  AudioController,
  buildAudioRequest,
  type AudioReference,
  type AudioTab,
} from "./audio-state.svelte";
import type { Connection } from "./connection.svelte";
import { lyricsExamples, styleExamples } from "./music-rewrite";
import { AIRadio, type RadioTrack } from "./radio.svelte";
import { SafeStorage } from "./storage.svelte";

type Draft = ReturnType<typeof audioDefaults>;
export type Clip = "reference" | "source";
export type Pane = { d: Draft } & Partial<Record<Clip, AudioReference>>;
export type Field = "prompt" | "lyrics";
export type Saved = { title: string; body: string }[];

export const tabs = ["voice", "music", "sound"] as const;
const fresh = (): Record<AudioTab, Pane> => ({ voice: { d: audioDefaults() }, music: { d: audioDefaults() }, sound: { d: audioDefaults() } });
const clipOk = (clip: unknown): clip is AudioReference =>
  typeof (clip as AudioReference)?.base64 === "string" &&
  (clip as AudioReference).base64.length <= 160 * 1024 * 1024 &&
  typeof (clip as AudioReference).name === "string" &&
  /^[A-Za-z0-9+/]+=*$/.test((clip as AudioReference).base64);

/** The audio pane's drafts (one per tab, kept per server), reference clips, recording and generation, without any drawing. */
export class AudioWorkspace {
  readonly c: AudioController;
  readonly library: Library;
  // A separate draft database keeps reference clips out of the results history.
  readonly drafts: Pick<Library, "get" | "put">;
  panes = $state<Record<AudioTab, Pane>>(fresh());
  tab = $state<AudioTab>("voice");
  saved = $state<Record<Field, Saved>>({ prompt: [], lyrics: [] });
  ready = $state(false);
  busy = $state(false);
  recording = $state(false);
  error = $state("");
  storageError = $state("");
  /** Object URL of a reference clip being previewed. */
  clipUrl = $state("");
  server: string;
  #connection: Connection;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #saveChain: Promise<void> = Promise.resolve();
  #epoch = 0;
  #hydration = 0;
  #signature = "";
  #recorder: MediaRecorder | null = null;
  #mic: MediaStream | null = null;
  #recordingTimer: ReturnType<typeof setTimeout> | undefined;
  /** The Music tab's deck: this server's saved music, played back. */
  readonly player: AmpPlayer;
  readonly radio: AIRadio;
  readonly #storage: StorageAdapter;
  #theme = $state("");

  constructor(connection: Connection, library: Library, drafts: Pick<Library, "get" | "put">, storage: StorageAdapter = new SafeStorage()) {
    this.#connection = connection;
    this.library = library;
    this.drafts = drafts;
    this.c = new AudioController(library);
    this.server = connection.active.url;
    this.#storage = storage;
    this.#theme = storage.getItem("aiRadio.theme") ?? "";
    this.player = new AmpPlayer(library, storage);
    const ws = this;
    this.radio = new AIRadio({
      get model() {
        return ws.model;
      },
      get d() {
        return ws.d;
      },
      get server() {
        return ws.server;
      },
      c: this.c,
      client: () => connection.client(),
      chatModels: () => connection.models.filter((m) => m.capabilities.includes("chat")),
      play: (track: RadioTrack) => void this.#playNew(track),
      isPlaying: () => this.player.active,
    });
    // A track that ends while the radio is on is the radio's to follow.
    this.player.onFinish = () => this.radio.skip();
  }

  /** The AI Radio theme typed into the deck, kept for the next visit. */
  get radioTheme() {
    return this.#theme;
  }
  set radioTheme(value: string) {
    this.#theme = value;
    this.#storage.setItem("aiRadio.theme", value);
  }

  get pane() {
    return this.panes[this.tab];
  }
  get d() {
    return this.pane.d;
  }
  get models() {
    return this.#connection.models.filter((m) => audioProfile(m)?.tab === this.tab);
  }
  get model() {
    return this.models.find((m) => m.id === this.d.model);
  }
  get profile() {
    return audioProfile(this.model);
  }
  /** Library type this tab saves under. */
  get type() {
    return this.tab === "voice" ? "speech" : this.tab;
  }

  /** The reason Generate is unavailable; empty while the form is valid. */
  get validation() {
    try {
      if (!this.ready) throw Error(t("Restoring audio draft…"));
      buildAudioRequest(this.model, this.d, this.pane, () => 0);
      return "";
    } catch (e) {
      return e instanceof Error ? e.message : t("Invalid audio settings.");
    }
  }

  examples(field: Field): { title: string; body: string }[] {
    const family = this.profile?.family ?? "acestep";
    return this.tab === "sound" ? soundPrompts.map((body) => ({ title: body, body })) : field === "lyrics" ? lyricsExamples(family) : styleExamples(family);
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
        for (const tab of tabs) {
          const input = value.panes?.[tab];
          if (!input) continue;
          const d = audioDefaults();
          for (const key of Object.keys(d) as (keyof Draft)[]) if (typeof input.d?.[key] === typeof d[key]) Object.assign(d, { [key]: input.d[key] });
          d.trackClasses = Array.isArray(d.trackClasses) ? d.trackClasses.filter((v) => typeof v === "string") : [];
          const pane: Pane = { d };
          for (const key of ["reference", "source"] as const) if (clipOk(input[key])) pane[key] = input[key];
          this.panes[tab] = pane;
        }
        if ((tabs as readonly string[]).includes(value.tab)) this.tab = value.tab;
        for (const key of ["prompt", "lyrics"] as const)
          if (Array.isArray(value.saved?.[key])) this.saved[key] = value.saved[key].filter((v: { title: unknown; body: unknown }) => typeof v.title === "string" && typeof v.body === "string");
      }
    } catch {
      this.storageError = t("Could not restore the audio draft. Changes remain available in this tab.");
    }
    if (epoch !== this.#hydration) return;
    this.ready = true;
    void this.player.refresh(this.server);
    this.#signature = this.signature();
    this.connectionChanged();
  }

  /** Cheap change detector for the draft: every field, but only the clips' names and sizes. */
  signature() {
    const clip = (c?: AudioReference) => (c ? [c.name, c.base64.length] : null);
    return JSON.stringify([tabs.map((tab) => [this.panes[tab].d, clip(this.panes[tab].reference), clip(this.panes[tab].source)]), this.tab, this.saved]);
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
      snapshot = JSON.stringify({ panes: this.panes, tab: this.tab, saved: this.saved });
    this.#saveChain = this.#saveChain.then(async () => {
      try {
        await this.drafts.put(server, { type: "speech", server, model: "", blob: new Blob([snapshot], { type: "application/json" }) });
        if (server === this.server) this.storageError = "";
      } catch {
        if (server === this.server) this.storageError = t("Audio draft could not be saved. Keep this tab open and download results.");
      }
    });
  }

  connectionChanged() {
    const url = this.#connection.active.url;
    if (this.server !== url) {
      this.flush();
      this.stopActivity();
      this.server = url;
      this.panes = fresh();
      this.saved = { prompt: [], lyrics: [] };
      this.c.result = null;
      this.c.phase = "idle";
      this.error = "";
      this.storageError = "";
      void this.init();
    }
    if (!this.ready) return;
    if (!this.d.model && this.models.length) this.pane.d = { ...audioDefaults(this.models.find((m) => m.loaded) || this.models[0]), prompt: this.d.prompt, lyrics: this.d.lyrics };
  }

  setTab(tab: AudioTab) {
    this.stopActivity();
    this.tab = tab;
    this.c.result = null;
    this.c.phase = "idle";
    this.c.message = "";
    this.error = "";
    this.connectionChanged();
  }

  chooseModel(id: string) {
    const model = this.models.find((m) => m.id === id);
    this.pane.d = { ...audioDefaults(model), prompt: this.d.prompt, lyrics: this.d.lyrics };
    this.stopRecording(true);
  }

  /** Everything in flight stops: generation, the radio, recording, playback. */
  stopActivity() {
    this.#epoch++;
    this.radio.stop();
    this.busy = false;
    this.error = "";
    if (this.c.run) this.c.cancel();
    this.leave();
  }

  /** The pane closed: the microphone, the radio and playback stop, a generation keeps running. */
  leave() {
    this.stopRecording(true);
    this.radio.stop();
    this.stopPlayback();
  }

  stopPlayback() {
    this.player.stop();
    document.querySelectorAll<HTMLAudioElement>(".audio-screen audio").forEach((a) => {
      a.pause();
      a.currentTime = 0;
    });
  }

  clear(key: Clip) {
    delete this.pane[key];
    this.stopPlayback();
  }

  async attach(file: File, key: Clip) {
    if (this.busy || this.c.run) return;
    this.stopRecording(true);
    const epoch = this.#epoch,
      pane = this.pane;
    this.busy = true;
    this.error = t("Converting…");
    try {
      pane[key] = await audioReference(file, file.name, this.tab === "voice" ? "voice" : "music", key === "source" ? 600 : 30);
      if (epoch === this.#epoch) this.error = "";
    } catch (e) {
      if (epoch === this.#epoch) this.error = e instanceof Error ? e.message : t("Audio conversion failed.");
    } finally {
      if (epoch === this.#epoch) this.busy = false;
    }
  }

  async record() {
    if (this.#recorder || this.busy || this.c.run) return;
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
      this.busy = false;
      this.#mic = stream;
      const recorder = new MediaRecorder(stream);
      this.#recorder = recorder;
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        if (epoch === this.#epoch && chunks.length) void this.attach(new File(chunks, t("Recorded voice"), { type: recorder.mimeType }), "reference");
      };
      recorder.onerror = () => {
        this.stopRecording(true);
        this.error = t("Recording failed. Choose an audio file instead.");
      };
      recorder.start();
      this.recording = true;
      this.#recordingTimer = setTimeout(() => this.stopRecording(), 8000);
    } catch (e) {
      this.#mic?.getTracks().forEach((track) => track.stop());
      this.busy = false;
      this.#mic = null;
      this.#recorder = null;
      this.error = e instanceof Error ? e.message : t("Microphone unavailable. Choose an audio file instead.");
    }
  }

  stopRecording(discard = false) {
    clearTimeout(this.#recordingTimer);
    const recorder = this.#recorder;
    this.#recorder = null;
    this.recording = false;
    if (recorder) {
      if (discard) recorder.onstop = null;
      if (recorder.state !== "inactive") recorder.stop();
    }
    this.#mic?.getTracks().forEach((track) => track.stop());
    this.#mic = null;
  }

  /** Preview a reference clip; the pane plays it from `clipUrl`. */
  previewClip(clip: AudioReference) {
    this.stopPlayback();
    if (this.clipUrl) URL.revokeObjectURL(this.clipUrl);
    this.clipUrl = URL.createObjectURL(new Blob([decodeBase64(clip.base64)], { type: "audio/wav" }));
  }

  async generate() {
    if (this.c.run) {
      this.c.cancel();
      return;
    }
    this.error = "";
    try {
      const request = buildAudioRequest(this.model, this.d, this.pane);
      this.stopPlayback();
      this.flush();
      await this.c.generate(this.#connection.client(), request, this.server);
      // A finished song takes over the deck; one already playing kept playing while it was made.
      if (request.type === "music" && this.c.phase === "completed" && this.c.result) await this.#playNew(this.c.result);
    } catch (e) {
      this.error = e instanceof Error ? e.message : t("Generation failed.");
    }
  }

  /** A new track joins the playlist and plays. */
  async #playNew(track: RadioTrack) {
    await this.player.refresh(this.server);
    await this.player.play({ ...track, model: "", createdAt: Date.now(), ...this.player.tracks.find((t) => t.id === track.id) });
  }

  /** Show a saved history item in the preview. */
  show(item: LibraryItem) {
    if (this.c.run) return;
    this.c.result = { ...item, elapsedMs: 0 };
    this.c.phase = "completed";
    this.c.saveError = "";
  }

  /** A template menu choice other than "save": delete a saved one, or load a saved or built-in one into the field. */
  chooseTemplate(field: Field, value: string) {
    const [action, index] = value.split(":");
    if (action === "delete") {
      this.saved[field].splice(Number(index), 1);
      return;
    }
    const entry = (action === "saved" ? this.saved[field] : this.examples(field))[Number(index)];
    if (entry) this.d[field] = entry.body;
  }

  saveTemplate(field: Field, title: string, body: string) {
    this.saved[field] = [{ title, body }, ...this.saved[field].filter((v) => v.title !== title)];
  }
}
