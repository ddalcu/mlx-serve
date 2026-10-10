import { describe, expect, it } from "vitest";
import type { Model } from "../src/lib/core/models";
import { audioDefaults, audioProfile, buildAudioRequest } from "../src/lib/state/audio-state.svelte";

const model = (id: string, architecture: string, ...capabilities: string[]): Model => ({ id, capabilities: (capabilities.length ? capabilities : ["audio"]) as Model["capabilities"], architecture, meta: {} });
const kokoro = model("org/kokoro", "kokoro");
const qwenTts = model("org/qwen3-tts", "qwen3_tts");
const ace = model("org/ace-step", "acestep", "audio", "music");
const music3 = model("org/minimax-music3", "minimax_music3", "audio", "music");
const sound = model("org/stable-audio", "stable_audio3");
const draft = (m: Model, patch: Record<string, unknown> = {}) => ({ ...audioDefaults(m), prompt: "hello there", seed: "5", ...patch });
const clip = { base64: "AAAA", name: "ref.wav" };

describe("audioProfile", () => {
  it("assigns each family to its tab with its own controls", () => {
    expect(audioProfile(kokoro)).toMatchObject({ tab: "voice", voices: true, speed: true, reference: false });
    expect(audioProfile(qwenTts)).toMatchObject({ tab: "voice", voices: false, reference: true });
    expect(audioProfile(ace)).toMatchObject({ tab: "music", source: true, meta: true, steps: false, duration: [10, 600] });
    expect(audioProfile(music3)).toMatchObject({ tab: "music", source: false, meta: false, steps: true, duration: [5, 360] });
    expect(audioProfile(sound)).toMatchObject({ tab: "sound", steps: true, duration: [0.5, 120] });
  });

  it("offers nothing for unknown architectures or non-audio models", () => {
    expect(audioProfile(undefined)).toBeUndefined();
    expect(audioProfile(model("x", "mystery"))).toBeUndefined();
    expect(audioProfile({ ...kokoro, capabilities: ["chat"] })).toBeUndefined();
  });
});

describe("buildAudioRequest: speech", () => {
  it("speaks the text with a Kokoro voice and speed", () => {
    expect(buildAudioRequest(kokoro, draft(kokoro), {})).toEqual({ path: "/v1/audio/speech", type: "speech", body: { model: "org/kokoro", input: "hello there", voice: "af_heart", speed: 1 } });
  });

  it("accepts a blend of known voices, once each, and refuses unknown ones", () => {
    expect((buildAudioRequest(kokoro, draft(kokoro, { voice: "af_heart, af_bella,af_heart" }), {}).body as { voice: string }).voice).toBe("af_heart,af_bella");
    expect(() => buildAudioRequest(kokoro, draft(kokoro, { voice: "nobody" }), {})).toThrow(/valid Kokoro voice/);
    expect(() => buildAudioRequest(kokoro, draft(kokoro, { voice: "" }), {})).toThrow(/valid Kokoro voice/);
  });

  it("bounds the speed", () => {
    expect(() => buildAudioRequest(kokoro, draft(kokoro, { speed: 3 }), {})).toThrow(/Speed must be 0.5–2/);
  });

  it("a cloning model sends its reference clip and no Kokoro fields", () => {
    const { body } = buildAudioRequest(qwenTts, draft(qwenTts), { reference: clip });
    expect(body).toEqual({ model: "org/qwen3-tts", input: "hello there", ref_audio: "AAAA" });
  });

  it("asks for text first", () => {
    expect(() => buildAudioRequest(kokoro, draft(kokoro, { prompt: " " }), {})).toThrow("Enter text to be generated.");
    expect(() => buildAudioRequest(undefined, draft(kokoro), {})).toThrow("Choose a supported model on this server.");
  });
});

describe("buildAudioRequest: sound", () => {
  it("sends duration, a seed and steps", () => {
    expect(buildAudioRequest(sound, draft(sound), {})).toEqual({ path: "/v1/audio/sound-generations", type: "sound", body: { model: "org/stable-audio", prompt: "hello there", duration_seconds: 10, seed: 5, steps: 8 } });
  });

  it("draws a random seed for empty or -1, and checks the ranges", () => {
    expect((buildAudioRequest(sound, draft(sound, { seed: "" }), {}, () => 77).body as { seed: number }).seed).toBe(77);
    expect((buildAudioRequest(sound, draft(sound, { seed: "-1" }), {}, () => 78).body as { seed: number }).seed).toBe(78);
    expect(() => buildAudioRequest(sound, draft(sound, { seed: "5000000000" }), {})).toThrow(/Seed must be/);
    expect(() => buildAudioRequest(sound, draft(sound, { seed: "1.5" }), {})).toThrow(/whole number/);
    expect(() => buildAudioRequest(sound, draft(sound, { duration: 500 }), {})).toThrow(/Duration must be 0.5–120/);
    expect(() => buildAudioRequest(sound, draft(sound, { steps: 60 }), {})).toThrow(/Steps must be 1–50/);
  });
});

describe("buildAudioRequest: music", () => {
  const lyrics = "[verse]\nla la";
  it("needs lyrics or Instrumental where the model sings its lyrics", () => {
    expect(() => buildAudioRequest(music3, draft(music3), {})).toThrow("This model requires lyrics or Instrumental.");
    expect(buildAudioRequest(music3, draft(music3, { lyrics }), {}).body).toMatchObject({ lyrics, response_format: "wav" });
  });

  it("Instrumental wins over lyrics and drops them", () => {
    const { body } = buildAudioRequest(music3, draft(music3, { lyrics, instrumental: true }), {});
    expect(body).toMatchObject({ instrumental: true });
    expect(body).not.toHaveProperty("lyrics");
  });

  it("ACE-Step can run without lyrics and carries its metadata fields", () => {
    const { path, body } = buildAudioRequest(ace, draft(ace, { bpm: "120", keyscale: "C major", language: "en", timesignature: "4" }), {});
    expect(path).toBe("/v1/audio/music-generations");
    expect(body).toMatchObject({ bpm: 120, keyscale: "C major", vocal_language: "en", timesignature: "4", duration_seconds: 60 });
    expect(body).not.toHaveProperty("steps");
  });

  it("checks tempo, key, language and time signature", () => {
    expect(() => buildAudioRequest(ace, draft(ace, { bpm: "20" }), {})).toThrow(/Tempo must be 30–300/);
    expect(() => buildAudioRequest(ace, draft(ace, { bpm: "100.5" }), {})).toThrow(/whole number/);
    expect(() => buildAudioRequest(ace, draft(ace, { keyscale: "H# minor" }), {})).toThrow("Choose a valid key.");
    expect(() => buildAudioRequest(ace, draft(ace, { language: "xx" }), {})).toThrow("Choose a vocal language.");
    expect(() => buildAudioRequest(ace, draft(ace, { timesignature: "5" }), {})).toThrow("Choose a time signature.");
  });

  it("cover and complete need a source clip; cover sends its strengths, complete its instruments", () => {
    expect(() => buildAudioRequest(ace, draft(ace, { task: "cover" }), {})).toThrow("Choose source audio for this mode.");
    expect(buildAudioRequest(ace, draft(ace, { task: "cover", coverStrength: 0.5, coverNoise: 0.2 }), { source: clip }).body).toMatchObject({ task: "cover", src_audio: "AAAA", cover_strength: 0.5, cover_noise_strength: 0.2 });
    const complete = buildAudioRequest(ace, draft(ace, { task: "complete", trackClasses: ["drums"] }), { source: clip }).body;
    expect(complete).toMatchObject({ task: "complete", src_audio: "AAAA", track_classes: ["drums"] });
    expect(complete).not.toHaveProperty("cover_strength");
    expect(() => buildAudioRequest(ace, draft(ace, { task: "complete", trackClasses: ["kazoo"] }), { source: clip })).toThrow("Choose valid instruments.");
    expect(() => buildAudioRequest(ace, draft(ace, { task: "remix" }), { source: clip })).toThrow("Choose a music mode.");
  });

  it("a model without source support ignores the source", () => {
    expect(buildAudioRequest(music3, draft(music3, { lyrics, task: "cover" }), { source: clip }).body).not.toHaveProperty("src_audio");
  });
});
