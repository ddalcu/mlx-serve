import { describe, expect, it } from "vitest";
import { motionRates, printable, sliderFraction, textWidth, titleOffset } from "../src/lib/amp/skin";
import { AmpPlayer, bandLevels, BANDS, clock, Spectrum, wavInfo } from "../src/lib/state/amp-player.svelte";

function wav(seconds: number, rate = 48000, channels = 2) {
  const data = seconds * rate * channels * 2,
    bytes = new ArrayBuffer(44 + data),
    v = new DataView(bytes),
    tag = (at: number, s: string) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
  tag(0, "RIFF");
  v.setUint32(4, 36 + data, true);
  tag(8, "WAVE");
  tag(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, channels, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * channels * 2, true);
  v.setUint16(32, channels * 2, true);
  v.setUint16(34, 16, true);
  tag(36, "data");
  v.setUint32(40, data, true);
  return bytes;
}

describe("the pixel font", () => {
  it("folds what it lacks instead of drawing question marks", () => {
    expect(printable("Composing…")).toBe("COMPOSING...");
    expect(printable("Mănăstire șlagăr")).toBe("MANASTIRE SLAGAR");
    expect(printable("a – b — c")).toBe("A - B - C");
    expect(printable("it’s “live”")).toBe("IT'S \"LIVE\"");
    expect(textWidth("Composing…")).toBe(textWidth("COMPOSING..."));
  });
});

describe("motion", () => {
  it("is modest, and reduced motion slows the spectrum and stills the title", () => {
    expect(motionRates(false)).toEqual({ spectrum: 12, title: 6, seek: 4 });
    expect(motionRates(true)).toEqual({ spectrum: 6, title: 0, seek: 2 });
  });

  it("scrolls a long title in whole pixels and wraps; a title that fits never moves", () => {
    expect(titleOffset(50, 60, 10)).toBe(0);
    expect(titleOffset(100, 60, 1)).toBe(14);
    expect(titleOffset(100, 60, 10)).toBe(140 % 136);
  });

  it("centres the slider thumb on the pointer and clamps", () => {
    expect(sliderFraction(7, 58, 14)).toBe(0);
    expect(sliderFraction(51, 58, 14)).toBe(1);
    expect(sliderFraction(29, 58, 14)).toBe(0.5);
    expect(sliderFraction(-10, 58, 14)).toBe(0);
  });
});

describe("track info", () => {
  it("reads length, rate, channels and bitrate from the WAV header", () => {
    const bytes = wav(2);
    expect(wavInfo(bytes.slice(0, 4096), bytes.byteLength)).toEqual({ seconds: 2, sampleRate: 48000, channels: 2, kbps: Math.trunc((bytes.byteLength * 8) / 2 / 1000) });
    expect(wavInfo(new ArrayBuffer(8), 8)).toBeNull();
    expect(clock(75.9)).toBe("1:15");
  });
});

describe("the spectrum", () => {
  it("maps loud bins high, silence to zero, and settles once the bars fall", () => {
    const loud = bandLevels(new Float32Array(512).fill(0)),
      quiet = bandLevels(new Float32Array(512).fill(-120));
    expect(loud).toHaveLength(BANDS);
    expect(loud.every((v) => v === 1)).toBe(true);
    expect(quiet.every((v) => v === 0)).toBe(true);
    const s = new Spectrum();
    s.step(loud);
    expect(s.settled).toBe(false);
    for (let i = 0; i < 100; i++) s.step(null);
    expect(s.settled).toBe(true);
  });
});

describe("the player", () => {
  const storage = () => {
    const m = new Map<string, string>();
    return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) };
  };
  const tracks = ["a", "b", "c"].map((id, i) => ({ id, model: "m", createdAt: 3 - i, blob: new Blob([wav(1)]) }));

  it("steps through the playlist, stops past the end unless wrapping, and remembers shuffle and repeat", () => {
    const store = storage(),
      player = new AmpPlayer({ list: async () => [], get: async () => undefined }, store);
    player.tracks = tracks;
    player.advance(1, tracks[0]!);
    expect(player.loaded?.id).toBe("b");
    player.advance(1, tracks[2]!);
    expect(player.loaded?.id).toBe("b");
    player.advance(1, tracks[2]!, true);
    expect(player.loaded?.id).toBe("a");
    player.setShuffle(true);
    player.advance(1, tracks[0]!);
    expect(player.loaded?.id).not.toBe("a");
    player.setAutoNext(true);
    const later = new AmpPlayer({ list: async () => [], get: async () => undefined }, store);
    expect([later.shuffle, later.autoNext]).toEqual([true, true]);
  });
});
