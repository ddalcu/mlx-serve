import { describe, expect, it } from "vitest";
import type { Model } from "../src/lib/core/models";
import { audioDefaults } from "../src/lib/state/audio-state.svelte";
import { musicStyles } from "../src/lib/state/audio-presets";
import { radioLyricsRequest, radioRequest, rewriteRequest } from "../src/lib/state/music-rewrite";
import { blockOffset, fallbackPrompt, fillWidth, MAX_STEPS, MAX_TRACK_SECONDS, nudge, RadioQueue, stationDraft, STRIP_BLOCK, STRIP_SPEED } from "../src/lib/state/radio.svelte";

const model = (architecture: string): Model => ({ id: `org/${architecture}`, capabilities: ["audio", "music"], architecture, meta: {} });
const ace = model("acestep"), music3 = model("minimax_music3"), yue2 = model("yue2");

describe("the variation nudge", () => {
  it("gives consecutive tracks different directions", () => {
    const directions = Array.from({ length: 12 }, (_, i) => nudge(i, 3));
    for (let i = 1; i < directions.length; i++) expect(directions[i]).not.toBe(directions[i - 1]);
    expect(new Set(directions).size).toBeGreaterThan(8);
  });

  it("is deterministic, salted per station, and names two aspects", () => {
    expect(nudge(4, 9)).toBe(nudge(4, 9));
    expect(Array.from({ length: 6 }, (_, i) => nudge(i, 1))).not.toEqual(Array.from({ length: 6 }, (_, i) => nudge(i, 2)));
    expect(nudge(0, 0).split("; ")).toHaveLength(2);
  });

  it("keeps the theme and the nudge without a chat model", () => {
    const p = fallbackPrompt("synthwave night drive", "slower tempo; piano lead");
    expect(p).toContain("synthwave night drive");
    expect(p).toContain("slower tempo, piano lead");
  });
});

describe("the LLM requests", () => {
  it("carry theme, history and nudge, in the family's own format", () => {
    const r = radioRequest("mellow lo-fi for coding", ["slow piano lo-fi", "dusty vinyl beat"], "faster tempo; brighter mood", "acestep", true);
    for (const s of ["mellow lo-fi for coding", "slow piano lo-fi", "dusty vinyl beat", "faster tempo; brighter mood"]) expect(r.user).toContain(s);
    expect(r.user.toLowerCase()).toContain("instrumental");
    expect(r.system).toContain("different");
    expect(r.system).toContain(musicStyles[0]!.body);
    expect(r.system).not.toContain("Global Metadata");
    expect(radioRequest("x", [], "n", "minimax_music3", true).system).toContain("Global Metadata");
  });

  it("shows only the last few prompts", () => {
    const r = radioRequest("x", Array.from({ length: 20 }, (_, i) => `track number ${i + 1}`), "n", "acestep", true);
    expect(r.user).toContain("track number 20");
    expect(r.user).not.toContain("track number 1\n");
  });

  it("writes lyrics in the language the theme names, bounded for a song that ends itself", () => {
    const r = radioLyricsRequest("romanian manele about summer", "manele, accordion", "a faster tempo", ["old style"], "yue2", "en");
    expect(r.user).toContain("romanian manele about summer");
    expect(r.user).toContain("language the station theme names");
    expect(r.user).toContain("English");
    expect(r.user).toContain("manele, accordion");
    expect(r.user).toContain("16");
    expect(r.system).toContain("[Verse]");
  });

  it("asks YuE2 for a tag line, not prose", () => {
    const r = rewriteRequest("prompt", "pop", "yue2", "", false, "en");
    expect(r.system).toContain("comma-separated tags");
    expect(rewriteRequest("lyrics", "la", "yue2", "pop", false, "en").system).toContain("[Pre-Chorus]");
  });
});

describe("the queue", () => {
  it("asks for a track only when nothing is buffered, first in first out", () => {
    const q = new RadioQueue();
    expect(q.needsTrack).toBe(true);
    q.push("a", "1");
    q.push("b", "2");
    expect(q.needsTrack).toBe(false);
    expect(q.pop()).toBe("1");
    expect(q.pop()).toBe("2");
    expect(q.pop()).toBeUndefined();
  });

  it("keeps the last prompts only", () => {
    const q = new RadioQueue();
    for (let i = 1; i <= 10; i++) q.push(`p${i}`, String(i));
    expect(q.recentPrompts).toHaveLength(8);
    expect(q.recentPrompts.at(-1)).toBe("p10");
  });
});

describe("the station's settings", () => {
  it("caps the length, lowers steps where the model takes them, and lets each prompt pick tempo and key", () => {
    const long = stationDraft(music3, { ...audioDefaults(music3), duration: 300, steps: 60, bpm: "90", keyscale: "C major", timesignature: "4" });
    expect(long).toMatchObject({ duration: MAX_TRACK_SECONDS, steps: MAX_STEPS, bpm: "", keyscale: "", timesignature: "", autoLength: false });
    expect(MAX_TRACK_SECONDS).toBe(120);
    expect(stationDraft(music3, { ...audioDefaults(music3), duration: 45, steps: 8 })).toMatchObject({ duration: 45, steps: 8 });
    expect(stationDraft(ace, { ...audioDefaults(ace), task: "cover" }).task).toBe("text2music");
    expect(stationDraft(yue2, { ...audioDefaults(yue2), autoLength: false }).autoLength).toBe(true);
  });
});

describe("the radio strip", () => {
  it("fills in whole pixels and clamps", () => {
    expect(fillWidth(0.5, 100)).toBe(50);
    expect(fillWidth(0.333, 100)).toBe(33);
    expect(fillWidth(1.7, 100)).toBe(100);
    expect(fillWidth(-1, 100)).toBe(0);
    expect(fillWidth(null, 100)).toBe(0);
  });

  it("bounces the waiting block inside the strip", () => {
    const travel = 100 - STRIP_BLOCK;
    for (let time = 0; time <= 20; time += 0.37) {
      const x = blockOffset(time, 100);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(travel);
      expect(Number.isInteger(x)).toBe(true);
    }
    expect(blockOffset(0, 100)).toBe(0);
    const atEnd = blockOffset(travel / STRIP_SPEED, 100);
    expect(Math.abs(atEnd - travel)).toBeLessThanOrEqual(1);
    expect(blockOffset(travel / STRIP_SPEED + 0.5, 100)).toBeLessThan(atEnd);
  });
});
