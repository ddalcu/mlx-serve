import { describe, expect, it } from "vitest";
import { bars, envelope } from "../src/lib/state/amp-player.svelte";
import { DECK_KEYS, deckRoute, infoLine, remaining, rowGlyph, volumeIcon, type DeckKey, type DeckRoute } from "../src/lib/state/deck";
import { coverColors, dayGroups, modelName, TITLE_LIMIT, trackTitle } from "../src/lib/state/track-meta";

describe("a track's title", () => {
  it("is the prompt's first sentence", () => {
    expect(trackTitle("funk meets orchestra in a groovy pleasing drive. Brass stabs.")).toBe("Funk meets orchestra in a groovy pleasing drive");
    expect(trackTitle("upbeat synthwave with driving bass\nand dreamy pads")).toBe("Upbeat synthwave with driving bass");
  });

  it("skips a template's header and labels, and takes Music 3's genre", () => {
    const prompt = "Global Metadata\nBasic Attributes: bpm is 118. key is F, and scale is minor. Retro Synthwave / Outrun.\nGlobal Emotional Progression: Opens cool and mechanical.";
    expect(trackTitle(prompt)).toBe("Retro Synthwave / Outrun");
    expect(trackTitle("Mood: dark ambient drone. Slow.")).toBe("Dark ambient drone");
    expect(trackTitle("Global Metadata")).toBe("Global Metadata");
  });

  it("is cut at a word, and an empty prompt has none", () => {
    const title = trackTitle("lush orchestral strings ".repeat(6))!;
    expect(title.length).toBeLessThanOrEqual(TITLE_LIMIT + 1);
    expect(title).toMatch(/(strings|lush|orchestral)…$/);
    expect(trackTitle("  \n ")).toBeNull();
  });

  it("names the model without its folder", () => {
    expect(modelName("ddalcu/YuE2-3B-MLX-Serve-8bit")).toBe("YuE2-3B-MLX-Serve-8bit");
    expect(modelName("Qwen3-TTS")).toBe("Qwen3-TTS");
  });
});

describe("the history", () => {
  it("gives the same clip the same cover, and another clip another", () => {
    expect(coverColors("one")).toEqual(coverColors("one"));
    expect(coverColors("one")[0]).not.toBe(coverColors("two")[0]);
  });

  it("groups consecutive clips by day: Today, Yesterday, a date, and Earlier for no date", () => {
    const now = new Date(2026, 9, 11, 18);
    const dates: Record<string, Date> = { a: new Date(2026, 9, 11, 15), b: new Date(2026, 9, 11, 9), c: new Date(2026, 9, 10, 23), d: new Date(2026, 8, 2, 23) };
    const days = dayGroups(["a", "b", "c", "d", "e"], (k) => dates[k] ?? null, now, "en");
    expect(days.map((d) => d.items)).toEqual([["a", "b"], ["c"], ["d"], ["e"]]);
    expect(days.map((d) => d.label)).toEqual(["Today", "Yesterday", expect.stringContaining("2"), "Earlier"]);
    expect(days[2]!.label).not.toContain("2026");
    expect(dayGroups(["x"], () => new Date(2025, 0, 5), now, "en")[0]!.label).toContain("2025");
  });

  it("shows on a row's cover what a click does while hovered, else that it plays", () => {
    expect(rowGlyph(true, false, false, false)).toEqual({ icon: "audio-lines", moving: true });
    expect(rowGlyph(true, true, false, false)).toEqual({ icon: "pause", moving: false });
    expect(rowGlyph(true, false, true, false).icon).toBe("stop");
    expect(rowGlyph(false, false, true, false).icon).toBe("play");
    expect(rowGlyph(false, false, true, true).icon).toBeNull();
    expect(rowGlyph(false, false, false, false).icon).toBeNull();
  });
});

describe("the Modern deck", () => {
  it("names rate, channels and bitrate", () => {
    expect(infoLine({ seconds: 120, sampleRate: 48_000, channels: 2, kbps: 1536 })).toBe("48 kHz · Stereo · 1536 kbps");
    expect(infoLine({ seconds: 4, sampleRate: 44_100, channels: 1, kbps: 705 })).toBe("44.1 kHz · Mono · 705 kbps");
    expect(infoLine(null)).toBe("");
  });

  it("draws the volume glyph from the level", () => {
    expect([0, 0.2, 0.5, 0.9].map(volumeIcon)).toEqual(["volume-x", "volume-1", "volume-2", "volume-2"]);
  });

  it("counts what is left on the right clock", () => {
    expect(remaining(13, 32)).toBe("-0:19");
    expect(remaining(40, 32)).toBe("-0:00");
  });
});

describe("the waveform", () => {
  it("scales the loudest slice to one, by loudness not sign", () => {
    expect(envelope([0, 0.25, -1, 0.5], 4)).toEqual([0, 0.25, 1, 0.5]);
    expect(envelope([3, -3, 0, 0], 2)).toEqual([1, 0]);
    expect(envelope([0, 0], 2)).toEqual([0, 0]);
    expect(envelope([], 8)).toEqual([]);
  });

  it("keeps the loudest slice in each bar, so a hit never vanishes", () => {
    expect(bars([0.1, 0.9, 0.2, 0.3], 2)).toEqual([0.9, 0.3]);
    expect(bars([0.4], 3)).toEqual([0.4, 0.4, 0.4]);
    expect(bars([], 3)).toEqual([]);
  });
});

describe("the deck's keys, in either look", () => {
  it("route the same way: a running AI Radio locks the keys that pick a track and takes Stop and Next", () => {
    const off: Record<DeckKey, DeckRoute> = { previous: "own", playPause: "own", stop: "own", next: "own", shuffle: "own", repeatAll: "own", download: "own", radio: "radio", historyRow: "own" };
    const on: Record<DeckKey, DeckRoute> = { previous: "locked", playPause: "own", stop: "radio", next: "radio", shuffle: "locked", repeatAll: "locked", download: "own", radio: "radio", historyRow: "locked" };
    expect([...DECK_KEYS].sort()).toEqual(Object.keys(off).sort());
    for (const key of DECK_KEYS) {
      expect(deckRoute(key, false), key).toBe(off[key]);
      expect(deckRoute(key, true), key).toBe(on[key]);
    }
  });
});
