import { download } from "../download";
import { t } from "../i18n/i18n";
import type { IconName } from "../icons";
import { clock, type AmpPlayer, type Track, type WavInfo } from "./amp-player.svelte";
import type { AIRadio } from "./radio.svelte";

// What the player deck's keys do and what the Modern deck shows. Both looks
// (MLX-Amp, Modern) bind their keys here, so a change to the radio's locks or
// redirects lands in both.

export const DECK_KEYS = ["previous", "playPause", "stop", "next", "shuffle", "repeatAll", "download", "radio", "historyRow"] as const;
export type DeckKey = (typeof DECK_KEYS)[number];
/** `own`: the key acts on the player or the file; `radio`: the station takes the press. */
export type DeckRoute = "own" | "radio" | "locked";

/** A running AI Radio owns playback: it locks the keys that would pick the next track and takes Stop (asked first) and Next. */
export function deckRoute(key: DeckKey, radioOn: boolean): DeckRoute {
  switch (key) {
    case "radio":
      return "radio";
    case "playPause":
    case "download":
      return "own";
    case "stop":
    case "next":
      return radioOn ? "radio" : "own";
    default:
      return radioOn ? "locked" : "own";
  }
}

export type DeckStatus = { kind: "busy"; message: string; progress: number | null } | { kind: "failed"; message: string } | null;
type Deck = {
  player: AmpPlayer;
  radio: AIRadio;
  /** The track on the deck. */
  track: Track | null;
  onradiostart: () => void;
  /** Asked first: stopping a station discards the track being composed. */
  onradiostop: () => void;
};

export const isLocked = (key: DeckKey, radio: AIRadio) => deckRoute(key, radio.on) === "locked";

export async function saveTrack(player: AmpPlayer, track: Track | null) {
  const blob = track && (await player.blob(track));
  if (blob) download(blob, `music-${track!.createdAt}.wav`);
}

export function press(key: DeckKey, { player, radio, track, onradiostart, onradiostop }: Deck) {
  const route = deckRoute(key, radio.on);
  if (route === "locked") return;
  if (key === "radio") return radio.on ? onradiostop() : onradiostart();
  if (route === "radio") return key === "stop" ? onradiostop() : void radio.skip();
  switch (key) {
    case "previous":
      return player.advance(-1, track);
    case "playPause":
      if (player.active) player.togglePause();
      else if (track) void player.play(track);
      return;
    case "stop":
      return player.stop();
    case "next":
      return player.advance(1, track);
    case "shuffle":
      return player.setShuffle(!player.shuffle);
    case "repeatAll":
      return player.setAutoNext(!player.autoNext);
    case "download":
      return void saveTrack(player, track);
  }
}

/** "48 kHz · Stereo · 1536 kbps"; empty when the file cannot be read. */
export function infoLine(info: WavInfo | null | undefined) {
  if (!info) return "";
  const khz = info.sampleRate / 1000;
  return `${Number.isInteger(khz) ? khz : khz.toFixed(1)} kHz · ${t(info.channels > 1 ? "Stereo" : "Mono")} · ${info.kbps} kbps`;
}

export const volumeIcon = (volume: number): IconName => (volume <= 0 ? "volume-x" : volume < 0.34 ? "volume-1" : "volume-2");

/** "-0:19": the right-hand clock counts what is left. */
export const remaining = (elapsed: number, total: number) => "-" + clock(Math.max(0, total - elapsed));

/** What a history row's cover shows: what a click would do while hovered, else that it plays. Only a playing track moves. */
export function rowGlyph(playing: boolean, paused: boolean, hovered: boolean, locked: boolean): { icon: IconName | null; moving: boolean } {
  if (hovered && !locked) return { icon: playing ? "stop" : "play", moving: false };
  if (playing) return paused ? { icon: "pause", moving: false } : { icon: "audio-lines", moving: true };
  return { icon: null, moving: false };
}
