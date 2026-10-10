// A long clip as shots the CONSOLE chains (port of the app's Storyboard.swift):
// each shot is one ordinary request opening on the last frame of the shot
// before it, so no single response has to carry the whole video.
import { t } from "../i18n/i18n";
import { newId } from "./id";
import { cleanReply } from "./rewrite";

export type Shot = { id: string; prompt: string; seconds: number };
export type SecondsRange = [number, number];

/** The longest story Enhance plans; longer ones are built with Add shot. */
const LONGEST_PLANNED_STORY = 120;

/** The shortest rung of `ladder` that covers `seconds`, the longest when none does. */
function shotFrames(seconds: number, fps: number, ladder: number[]) {
  const needed = seconds * Math.max(1, fps);
  return ladder.find((f) => f >= needed) ?? ladder.at(-1) ?? needed;
}

/** Frames of the joined clip: every join shares one frame. */
function deliveredFrames(frames: number[]) {
  return frames.length ? frames.reduce((a, b) => a + b, 0) - (frames.length - 1) : 0;
}

/** Whole seconds a shot can ask for: the model's tested floor (or the first rung) to its longest rung. */
function secondsRange(ladder: number[], fps: number, floorFrames: number): SecondsRange {
  const f = Math.max(1, fps),
    hi = Math.max(1, Math.floor((ladder.at(-1) ?? f) / f)),
    lo = Math.max(1, Math.floor(Math.max(floorFrames, ladder[0] ?? 1) / f));
  return [Math.min(lo, hi), hi];
}

/** Attention grows with the square of a shot's length, so planned shots stop at 10 s; one can still be dragged to the ceiling. */
function plannedRange([lo, hi]: SecondsRange): SecondsRange {
  return [lo, Math.max(lo, Math.min(hi, 10))];
}

/** "45 s" under a minute, "5:00" past it. */
function lengthLabel(seconds: number) {
  return seconds < 60 ? t("%@ s", [seconds]) : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

/** The header the planner writes above each shot: `=== SHOT 2 | 8s ===`. */
const HEADER = /^[#*\s]*=+\s*SHOT\s+\d+\s*[|:\-–—]\s*(\d+)\s*s[a-z]*\s*=+[*\s]*$/gim;

/** The planner's reply as shots, seconds clamped into `range`; an empty shot is dropped, and no headers is no plan. */
function parseStoryboard(reply: string, [lo, hi]: SecondsRange): Shot[] {
  const matches = [...reply.matchAll(HEADER)];
  return matches.flatMap((m, i) => {
    const start = m.index + m[0].length,
      end = matches[i + 1]?.index ?? reply.length,
      prompt = cleanReply(reply.slice(start, end));
    return prompt ? [{ id: newId(), prompt, seconds: Math.min(hi, Math.max(lo, Number(m[1]))) }] : [];
  });
}

/**
 * Shot `index` of `count`: the pane's request with this shot's prompt and length, opening on the previous
 * shot's last frame (shot 0 keeps the user's first frame) and landing on the user's last frame only at the end.
 */
function shotRequest<R extends Record<string, unknown>>(base: R, prompt: string, frames: number, index: number, count: number, previousLastFrame: string | undefined) {
  const r: Record<string, unknown> = { ...base, prompt, num_frames: frames, seed: Number(base.seed) + 2 * index };
  delete r.chain_windows;
  // One clip cannot condition every shot; each shot makes its own sound.
  delete r.audio;
  if (index > 0) {
    if (previousLastFrame) r.first_frame_image = previousLastFrame;
    else delete r.first_frame_image;
  }
  if (index < count - 1) delete r.last_frame_image;
  return r as R;
}

export { LONGEST_PLANNED_STORY, shotFrames, deliveredFrames, secondsRange, plannedRange, lengthLabel, parseStoryboard, shotRequest };
