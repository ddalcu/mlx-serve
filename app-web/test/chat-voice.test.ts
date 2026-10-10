import { describe, expect, it } from "vitest";
import { speechChunks, VoiceLoop, type Recognition } from "../src/lib/state/chat-voice.svelte";

describe("speechChunks", () => {
  it("omits fences and URLs; every chunk is at most 300 characters", () => {
    for (const text of ["Yes indeed! Hello **world**. ```js\nsecret()", "word ".repeat(250), "x".repeat(901)]) {
      const chunks = speechChunks(text);
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks.every((s) => s.length <= 300)).toBe(true);
      expect(chunks.join(" ")).not.toMatch(/secret|```|\*\*/);
    }
    expect(speechChunks("Read [this](https://example.test). https://example.test")).toEqual(["Read this."]);
  });
});

describe("VoiceLoop", () => {
  it("waits for recognition end, never listens while speaking, and Stop cancels playback", async () => {
    let mic = false;
    let recognition!: Recognition;
    let finishPlayback: (() => void) | undefined;
    const phases: string[] = [];
    const loop: VoiceLoop = new VoiceLoop({
      recognition: () =>
        (recognition = {
          continuous: false,
          interimResults: false,
          lang: "",
          onresult: null,
          onerror: null,
          onend: null,
          start() {
            mic = true;
          },
          abort() {},
        }),
      send: async () => {
        expect(mic).toBe(false);
        return "One. Two.";
      },
      synthesize: async (text) => {
        expect(mic).toBe(false);
        return new Blob([text]);
      },
      play: async (_blob, signal) => {
        expect(mic).toBe(false);
        await new Promise<void>((resolve) => {
          finishPlayback = resolve;
          signal.addEventListener("abort", () => resolve(), { once: true });
        });
      },
      stopChat() {},
      changed() {
        phases.push(loop.phase);
      },
    });
    const done = loop.start();
    recognition.onresult!({ results: [{ isFinal: true, 0: { transcript: "hello" } }] });
    expect(loop.phase).toBe("listening"); // abort requested; hardware has not ended yet
    mic = false;
    recognition.onend!();
    for (let i = 0; i < 20 && !finishPlayback; i++) await new Promise((resolve) => setTimeout(resolve, 0));
    expect(loop.phase).toBe("speaking");
    expect(mic).toBe(false);
    loop.stop();
    await done;
    expect(loop.phase).toBe("off");
    expect(phases).toEqual(["listening", "thinking", "speaking", "off"]);
  });
});
