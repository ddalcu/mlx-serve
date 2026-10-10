import { describe, expect, it } from "vitest";
import { deliveredFrames, lengthLabel, parseStoryboard, plannedRange, secondsRange, shotFrames, shotRequest } from "../src/lib/core/storyboard";

const h3Ladder = Array.from({ length: 22 }, (_, i) => 5 + 17 * i);
const ltxLadder = Array.from({ length: 24 }, (_, i) => 9 + 8 * i);

describe("storyboard lengths", () => {
  it("a shot's seconds pick the shortest rung that covers them, the longest when none does", () => {
    expect(shotFrames(8, 24, h3Ladder)).toBe(192);
    expect(shotFrames(15, 24, h3Ladder)).toBe(362);
    expect(shotFrames(20, 24, h3Ladder)).toBe(362);
    expect(shotFrames(8, 24, ltxLadder)).toBe(193);
  });

  it("every join shares one frame", () => {
    expect(deliveredFrames([192, 192, 192])).toBe(574);
    expect(deliveredFrames([124])).toBe(124);
    expect(deliveredFrames([])).toBe(0);
  });

  it("shot seconds run from the tested floor to the longest rung, collapsing onto the ceiling", () => {
    expect(secondsRange(h3Ladder, 24, 107)).toEqual([4, 15]);
    expect(secondsRange(ltxLadder, 24, 0)).toEqual([1, 8]);
    expect(secondsRange([5, 22, 39], 24, 107)).toEqual([1, 1]);
  });

  it("planned shots stop at ten seconds inside the shot range", () => {
    expect(plannedRange([5, 15])).toEqual([5, 10]);
    expect(plannedRange([1, 8])).toEqual([1, 8]);
    expect(plannedRange([12, 15])).toEqual([12, 12]);
  });

  it("labels minutes past one minute", () => {
    expect(lengthLabel(45)).toBe("45 s");
    expect(lengthLabel(300)).toBe("5:00");
    expect(lengthLabel(75)).toBe("1:15");
  });
});

describe("parseStoryboard", () => {
  it("reads the planner's shots, clamping their seconds", () => {
    const shots = parseStoryboard("Here you go:\n=== SHOT 1 | 8s ===\nA fox trots.\n\n**=== Shot 2 — 40 seconds ===**\nIt leaps.\n=== SHOT 3 | 9s ===\n   \n", [5, 15]);
    expect(shots.map((s) => [s.prompt, s.seconds])).toEqual([["A fox trots.", 8], ["It leaps.", 15]]);
    expect(new Set(shots.map((s) => s.id)).size).toBe(2);
  });

  it("a reply with no shot headers is no plan", () => {
    expect(parseStoryboard("A fox in the snow, eight seconds.", [5, 15])).toEqual([]);
  });
});

describe("shotRequest", () => {
  const base = { model: "m", prompt: "story", seed: 10, num_frames: 124, first_frame_image: "FIRST", last_frame_image: "LAST", audio: "WAV", chain_windows: 3 };

  it("each shot is one window with its own prompt, length and seed", () => {
    const r = shotRequest(base, "a leap", 192, 1, 3, "PREV");
    expect(r).toMatchObject({ prompt: "a leap", num_frames: 192, seed: 12, first_frame_image: "PREV" });
    expect(r).not.toHaveProperty("chain_windows");
    expect(r).not.toHaveProperty("audio");
  });

  it("the first shot keeps the user's first frame, only the last lands on the user's last frame", () => {
    expect(shotRequest(base, "p", 124, 0, 3, undefined)).toMatchObject({ first_frame_image: "FIRST" });
    expect(shotRequest(base, "p", 124, 0, 3, undefined)).not.toHaveProperty("last_frame_image");
    expect(shotRequest(base, "p", 124, 2, 3, "PREV")).toMatchObject({ first_frame_image: "PREV", last_frame_image: "LAST" });
  });
});
