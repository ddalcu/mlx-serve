import type { Library, LibraryItem } from "../core/library";
import type { StorageAdapter } from "../core/servers";
import { t } from "../i18n/i18n";

export type Track = Pick<LibraryItem, "prompt" | "createdAt" | "model"> & { id?: string; blob?: Blob };
export type WavInfo = { seconds: number; sampleRate: number; channels: number; kbps: number };

/** Length and format from a WAV header (the first few KB of the file); null when it is not one. */
export function wavInfo(header: ArrayBuffer, totalBytes: number): WavInfo | null {
  const view = new DataView(header),
    tag = (at: number) => String.fromCharCode(...new Uint8Array(header, at, 4));
  if (header.byteLength < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  let at = 12,
    channels = 0,
    sampleRate = 0,
    byteRate = 0;
  while (at + 8 <= header.byteLength) {
    const kind = tag(at),
      size = view.getUint32(at + 4, true);
    if (kind === "fmt " && at + 24 <= header.byteLength) {
      channels = view.getUint16(at + 10, true);
      sampleRate = view.getUint32(at + 12, true);
      byteRate = view.getUint32(at + 16, true);
    }
    if (kind === "data") {
      if (!byteRate) return null;
      const seconds = size / byteRate;
      return { seconds, sampleRate, channels, kbps: seconds > 0 ? Math.trunc((totalBytes * 8) / seconds / 1000) : 0 };
    }
    at += 8 + size + (size % 2);
  }
  return null;
}

/** m:ss, the way a playlist writes a length. */
export const clock = (seconds: number) => {
  const t = Math.trunc(Math.max(0, seconds));
  return `${Math.trunc(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

export const BANDS = 19;
const BINS = 512;
/** Log-spaced bin edges: low bands are single bins, high ones span many. */
const EDGES = (() => {
  const e = [1];
  for (let k = 1; k <= BANDS; k++) e.push(Math.max(e[k - 1]! + 1, Math.round(Math.pow(BINS - 1, k / BANDS))));
  return e;
})();

/** 0...1 per band from an analyser's dBFS bins; the tilt lifts the highs, which carry far less energy than the lows. */
export function bandLevels(db: Float32Array) {
  return Array.from({ length: BANDS }, (_, k) => {
    let peak = -Infinity;
    for (let i = EDGES[k]!; i < Math.max(EDGES[k]! + 1, EDGES[k + 1]!) && i < db.length; i++) peak = Math.max(peak, db[i]!);
    return Math.min(1, Math.max(0, (peak + k * 0.6 + 50) / 46));
  });
}

/** Loudness (RMS) per slice, the loudest scaled to 1: a mastered track peaks near full scale everywhere, so peaks would draw a flat bar code. */
export function envelope(samples: ArrayLike<number>, slices = 1024) {
  if (!samples.length || slices <= 0) return [];
  const levels = Array.from({ length: slices }, (_, i) => {
    const lo = Math.floor((i * samples.length) / slices),
      hi = Math.max(lo + 1, Math.floor(((i + 1) * samples.length) / slices));
    if (hi > samples.length) return 0;
    let sum = 0;
    for (let j = lo; j < hi; j++) sum += samples[j]! * samples[j]!;
    return Math.sqrt(sum / (hi - lo));
  });
  const top = Math.max(...levels);
  return top > 0 ? levels.map((v) => v / top) : levels;
}

/** `envelope` at `count` bars: each bar the loudest slice it covers. */
export function bars(outline: number[], count: number) {
  if (!outline.length || count <= 0) return [];
  return Array.from({ length: count }, (_, i) => {
    const lo = Math.floor((i * outline.length) / count),
      hi = Math.max(lo + 1, Math.floor(((i + 1) * outline.length) / count));
    return Math.max(...outline.slice(lo, Math.min(hi, outline.length)));
  });
}

/** Bars rise to the new level and fall at a fixed speed; peak caps hang a little longer. */
export class Spectrum {
  levels = new Array<number>(BANDS).fill(0);
  peaks = new Array<number>(BANDS).fill(0);
  /** Nothing is moving: the deck stops redrawing. */
  settled = true;
  step(target: number[] | null) {
    for (let i = 0; i < BANDS; i++) {
      this.levels[i] = Math.max(target?.[i] ?? 0, this.levels[i]! - 0.08);
      this.peaks[i] = Math.max(this.levels[i]!, this.peaks[i]! - 0.015);
    }
    this.settled = !target && this.levels.every((v) => v <= 0) && this.peaks.every((v) => v <= 0);
  }
}

/**
 * The deck's playback: one audio element, the playlist (this server's saved music, newest first),
 * shuffle and repeat. AI Radio takes over what happens when a track ends (`onFinish`).
 */
export class AmpPlayer {
  tracks = $state<Track[]>([]);
  /** The track on the deck, kept after it stops so Play has something to start. */
  loaded = $state<Track | null>(null);
  /** A track is playing or paused. */
  active = $state(false);
  paused = $state(false);
  volume = $state(0.8);
  shuffle = $state(false);
  autoNext = $state(false);
  error = $state("");
  /** Answers true when it takes over from a track that ended on its own. */
  onFinish: (() => boolean) | null = null;
  readonly #library: Pick<Library, "list" | "get">;
  readonly #storage: StorageAdapter;
  #audio: HTMLAudioElement | null = null;
  #url = "";
  #analyser: AnalyserNode | null = null;
  #bins: Float32Array<ArrayBuffer> | null = null;
  #info = new Map<string, WavInfo | null>();
  #outlines = new Map<string, number[]>();

  constructor(library: Pick<Library, "list" | "get">, storage: StorageAdapter) {
    this.#library = library;
    this.#storage = storage;
    this.shuffle = storage.getItem("mlxamp.shuffle") === "1";
    this.autoNext = storage.getItem("mlxamp.autoNext") === "1";
  }

  setShuffle(on: boolean) {
    this.shuffle = on;
    this.#storage.setItem("mlxamp.shuffle", on ? "1" : "0");
  }
  setAutoNext(on: boolean) {
    this.autoNext = on;
    this.#storage.setItem("mlxamp.autoNext", on ? "1" : "0");
  }
  setVolume(v: number) {
    this.volume = Math.min(1, Math.max(0, v));
    if (this.#audio) this.#audio.volume = this.volume;
  }

  /** Re-read the playlist for `server`. */
  async refresh(server: string) {
    try {
      this.tracks = await this.#library.list({ server, type: "music" });
    } catch {
      this.tracks = [];
    }
  }

  get elapsed() {
    return this.active ? (this.#audio?.currentTime ?? 0) : 0;
  }

  /** Length and format of a track, read once from its header. */
  async info(track: Track) {
    const key = track.id ?? String(track.createdAt);
    if (this.#info.has(key)) return this.#info.get(key)!;
    const blob = track.blob ?? (track.id ? (await this.#library.get(track.id))?.blob : undefined);
    const info = blob ? wavInfo(await blob.slice(0, 4096).arrayBuffer(), blob.size) : null;
    this.#info.set(key, info);
    return info;
  }
  /** The track's loudness outline for a waveform seek bar, decoded once; empty where the browser cannot decode. */
  async outline(track: Track) {
    const key = track.id ?? String(track.createdAt);
    const hit = this.#outlines.get(key);
    if (hit) return hit;
    let outline: number[] = [];
    try {
      const blob = await this.blob(track);
      if (blob && typeof OfflineAudioContext !== "undefined") {
        const audio = await new OfflineAudioContext(1, 1, 44_100).decodeAudioData(await blob.arrayBuffer());
        const mono = audio.getChannelData(0).slice();
        if (audio.numberOfChannels > 1) audio.getChannelData(1).forEach((v, i) => (mono[i] = (mono[i]! + v) / 2));
        outline = envelope(mono);
      }
    } catch {
      outline = [];
    }
    this.#outlines.set(key, outline);
    return outline;
  }
  /** The cached info, without reading. */
  known(track: Track | null) {
    return track ? this.#info.get(track.id ?? String(track.createdAt)) : undefined;
  }

  async blob(track: Track) {
    return track.blob ?? (track.id ? (await this.#library.get(track.id))?.blob : undefined);
  }

  /** Play `track` from the start, replacing whatever is playing. */
  async play(track: Track) {
    this.stop();
    this.loaded = track;
    this.error = "";
    const blob = await this.blob(track);
    if (!blob || this.loaded !== track) return;
    void this.info(track);
    const audio = (this.#audio ??= this.#element());
    URL.revokeObjectURL(this.#url);
    this.#url = URL.createObjectURL(blob);
    audio.src = this.#url;
    audio.volume = this.volume;
    this.active = true;
    this.paused = false;
    this.#connectSpectrum(audio);
    try {
      await audio.play();
    } catch {
      if (this.loaded === track) {
        this.active = false;
        this.error = t("Playback could not start. Try Play again or download the WAV.");
      }
    }
  }

  stop() {
    this.#audio?.pause();
    if (this.#audio) this.#audio.currentTime = 0;
    this.active = false;
    this.paused = false;
  }

  togglePause() {
    const audio = this.#audio;
    if (!audio || !this.active) return;
    if (this.paused) void audio.play().catch(() => {});
    else audio.pause();
    this.paused = !this.paused;
  }

  /** Jump within the loaded track, starting it if it is stopped. */
  async seek(fraction: number) {
    const track = this.loaded,
      info = track && (await this.info(track));
    if (!track || !info) return;
    if (!this.active) await this.play(track);
    if (this.#audio) this.#audio.currentTime = Math.max(0, fraction * info.seconds);
  }

  /** The track `delta` rows from `from` (default: the loaded one); a random other one when shuffling. Past either end it stops, unless `wrap`. */
  advance(delta: number, from: Track | null = this.loaded, wrap = false) {
    const ids = this.tracks.map((t) => t.id),
      i = from?.id ? ids.indexOf(from.id) : -1;
    if (i < 0) return;
    let next: Track | undefined;
    if (this.shuffle && this.tracks.length > 1) {
      const others = this.tracks.filter((t) => t.id !== from!.id);
      next = others[Math.floor(Math.random() * others.length)];
    } else next = this.tracks[i + delta] ?? (wrap ? this.tracks[(((i + delta) % ids.length) + ids.length) % ids.length] : undefined);
    if (next) void this.play(next);
  }

  /** The current band levels, or null while nothing plays. */
  levels() {
    if (!this.active || this.paused || !this.#analyser || !this.#bins) return null;
    this.#analyser.getFloatFrequencyData(this.#bins);
    return bandLevels(this.#bins);
  }

  /** Everything stops when the pane goes. */
  dispose() {
    this.stop();
    URL.revokeObjectURL(this.#url);
    this.#url = "";
  }

  #element() {
    const audio = new Audio();
    audio.addEventListener("ended", () => {
      this.active = false;
      this.paused = false;
      if (this.onFinish?.()) return;
      if (this.autoNext) this.advance(1, this.loaded, true);
    });
    return audio;
  }

  /** A spectrum needs Web Audio, made on the first play (a user gesture); without it the bars stay down. */
  #connectSpectrum(audio: HTMLAudioElement) {
    if (this.#analyser || typeof AudioContext === "undefined") return;
    try {
      const context = new AudioContext(),
        analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0;
      context.createMediaElementSource(audio).connect(analyser);
      analyser.connect(context.destination);
      this.#analyser = analyser;
      this.#bins = new Float32Array(analyser.frequencyBinCount);
    } catch {
      this.#analyser = null;
    }
  }
}
