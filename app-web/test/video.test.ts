import { describe, expect, it } from "vitest";
import { decodeVideo } from "../src/lib/core/video";

describe("decodeVideo", () => {
  it("decodes real-size LTX video without per-byte intermediate arrays", () => {
    const length = 768 * 512 * 97 * 3;
    const data = "AQID".repeat(length / 3);
    const input = { format: "rgb8", width: 768, height: 512, frames: 97, fps: 24, data };
    const raw = decodeVideo(input);
    expect(raw.rgb.length).toBe(length);
    for (const i of [0, 1, 2, 49151, 49152, length - 3, length - 2, length - 1]) expect(raw.rgb[i]).toBe((i % 3) + 1);
    expect(() => decodeVideo({ ...input, data: data.slice(0, -1) + "!" })).toThrow(/Invalid/);
    expect(() => decodeVideo({ ...input, frames: 96 })).toThrow(/large/);
  });
});
