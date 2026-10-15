import { afterEach, describe, expect, it, vi } from "vitest";
import { openShotEncoder } from "../src/lib/core/encode-video";
import type { RawVideo } from "../src/lib/core/video";

/** WebCodecs stand-ins that record what they are fed: happy-dom has none. */
function fakeWebCodecs() {
  const frames: number[] = [],
    audio: { timestamp: number; frames: number }[] = [];
  class Encoder {
    state = "configured";
    encodeQueueSize = 0;
    #output: (chunk: object, meta?: object) => void;
    constructor(init: { output: (chunk: object, meta?: object) => void }) {
      this.#output = init.output;
    }
    static isConfigSupported = async () => ({ supported: true });
    configure() {}
    encode(input: { timestamp: number; numberOfFrames?: number }) {
      if (input.numberOfFrames === undefined) frames.push(input.timestamp);
      else audio.push({ timestamp: input.timestamp, frames: input.numberOfFrames });
      const chunk = { byteLength: 1, timestamp: input.timestamp, duration: 1, type: "key", copyTo: (d: Uint8Array) => d.set([1]) };
      // An avcC with one SPS and one PPS for the video side; the audio side lets the muxer write its own config.
      this.#output(chunk, input.numberOfFrames === undefined ? { decoderConfig: { description: Uint8Array.of(1, 0x42, 0, 0x1f, 0xff, 0xe1, 0, 4, 0x67, 0x42, 0, 0x1f, 1, 0, 2, 0x68, 0xce) } } : undefined);
    }
    async flush() {}
    close() {
      this.state = "closed";
    }
  }
  class Media {
    timestamp: number;
    numberOfFrames?: number;
    constructor(_data: unknown, init: { timestamp: number; numberOfFrames?: number }) {
      this.timestamp = init.timestamp;
      this.numberOfFrames = init.numberOfFrames;
    }
    close() {}
  }
  class AudioMedia extends Media {
    constructor(init: { timestamp: number; numberOfFrames: number }) {
      super(null, init);
    }
  }
  vi.stubGlobal("VideoEncoder", Encoder);
  vi.stubGlobal("AudioEncoder", Encoder);
  vi.stubGlobal("VideoFrame", Media);
  vi.stubGlobal("AudioData", AudioMedia);
  return { frames, audio };
}
afterEach(() => vi.unstubAllGlobals());

/** `frames` 2x2 frames at 4 fps with as many quarter-seconds of 8 kHz mono sound (2000 samples a frame). */
const clip = (frames: number): RawVideo => ({
  rgb: new Uint8Array(frames * 2 * 2 * 3),
  width: 2,
  height: 2,
  frames,
  fps: 4,
  durationSeconds: frames / 4,
  audio: { pcm: new Uint8Array(frames * 2000 * 2), sampleRate: 8000, channels: 1 },
});

describe("openShotEncoder", () => {
  it("lays shots end to end, each later one dropping the frame and the sound it shares with the shot before", async () => {
    const seen = fakeWebCodecs();
    const encoder = await openShotEncoder(clip(3));
    await encoder.add(clip(3), 0);
    await encoder.add(clip(3), 1);
    const result = await encoder.finish();
    expect(seen.frames).toEqual([0, 250_000, 500_000, 750_000, 1_000_000]);
    // 6000 samples from the first shot, then the second's minus its first frame's 2000, straight after.
    expect(seen.audio.reduce((n, a) => n + a.frames, 0)).toBe(10_000);
    expect(seen.audio.find((a) => a.timestamp >= 750_000)).toEqual({ timestamp: 750_000, frames: 1024 });
    expect(result).toMatchObject({ container: "mp4", codec: "H.264 + AAC" });
    expect(result.blob.size).toBeGreaterThan(0);
  });

  it("refuses a shot whose canvas differs", async () => {
    fakeWebCodecs();
    const encoder = await openShotEncoder(clip(3));
    await expect(encoder.add({ ...clip(3), width: 4, rgb: new Uint8Array(3 * 4 * 2 * 3) }, 0)).rejects.toThrow(/share one canvas/);
  });
});
