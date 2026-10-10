import { t } from "../i18n/i18n";
import { throwIfAborted } from "../core/client";
export type Recognition = { continuous: boolean; interimResults: boolean; lang: string; onresult: ((e: any) => void) | null; onerror: ((e: any) => void) | null; onend: (() => void) | null; start(): void; abort(): void; };
const voiceAvailable = (secure: boolean, recognition: unknown) =>
  secure && typeof recognition === "function";
/**
 * Plain prose only; sentence chunks are bounded even for unpunctuated replies.
 */
function speechChunks(markdown: string) {
  const prose = markdown
    .replace(/```[\s\S]*?(?:```|$)/g, " ")
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/g, " ")
    .split("\n")
    .filter((line) => !line.includes("|"))
    .join("\n")
    .replace(/<[^>]*>/g, " ")
    .replace(/[*_#>`~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const chunks = [];
  for (let sentence of prose.match(/[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$|[^.!?]+/g) ??
    []) {
    sentence = sentence.trim();
    while (sentence.length > 300) {
      const space = sentence.lastIndexOf(" ", 300),
        at = space > 0 ? space : 300;
      chunks.push(sentence.slice(0, at));
      sentence = sentence.slice(at).trim();
    }
    if (sentence) chunks.push(sentence);
  }
  return chunks;
}
function listen(factory: () => Recognition, signal: AbortSignal) {
  return new Promise((resolve, reject) => {
    const r = factory();
    let text = "";
    let settled = false;
    const cleanup = () => {
      r.onresult = null;
      r.onerror = null;
      r.onend = null;
      signal.removeEventListener("abort", abort);
    };
    const finish = ( error: Error | undefined) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (error) reject(error);
      else if (text.trim()) resolve(text.trim());
      else
        reject(Error(t("No speech detected. Start voice chat to try again.")));
    };
    const abort = () => {
      try {
        r.abort();
      } finally {
        finish(Error(t("Voice stopped.")));
      }
    };
    r.continuous = false;
    r.interimResults = false;
    r.lang = "en-US";
    r.onresult = (e) => {
      if (signal.aborted) return;
      for (let i = e.resultIndex ?? 0; i < e.results.length; i++)
        if (e.results[i].isFinal) text += e.results[i][0].transcript + " ";
      // Wait for onend before sending, so the microphone is stopped before speech.
      if (text.trim()) r.abort();
    };
    r.onend = () =>
      finish(signal.aborted ? Error(t("Voice stopped.")) : undefined);
    r.onerror = (e) => {
      finish(
        Error(t("Speech recognition: %@.", [String(e.error || "unavailable")])),
      );
      r.abort();
    };
    signal.addEventListener("abort", abort, { once: true });
    try {
      throwIfAborted(signal);
      r.start();
    } catch (e) {
      finish(e instanceof Error ? e : Error(t("Could not start microphone.")));
    }
  });
}
export type VoiceOptions = { recognition: () => Recognition; send: (text: string, signal: AbortSignal) => Promise<string>; synthesize: (text: string, signal: AbortSignal) => Promise<Blob>; play: (blob: Blob, signal: AbortSignal) => Promise<void>; stopChat: () => void; changed: () => void; };
class VoiceLoop {
  phase = $state("off");
  error = $state("");
  run = $state.raw<AbortController | null>(null);
  options: VoiceOptions;
  constructor(options: VoiceOptions) {
    this.options = options;
  }
  state(phase: string) {
    this.phase = phase;
    this.options.changed();
  }
  stop() {
    const run = this.run;
    this.run = null;
    run?.abort();
    if (run) this.options.stopChat();
    this.state("off");
  }
  async start() {
    if (this.run) return;
    const run = new AbortController();
    this.run = run;
    this.error = "";
    try {
      while (this.run === run) {
        this.state("listening");
        const text = String(await listen(this.options.recognition, run.signal));
        throwIfAborted(run.signal);
        this.state("thinking");
        const reply = await this.options.send(text, run.signal);
        throwIfAborted(run.signal);
        const chunks = speechChunks(reply);
        // Wrap failures immediately, including the prefetched chunk, to avoid unhandled rejections.
        const synthesize = ( chunk: string) =>
          this.options.synthesize(chunk, run.signal).then(
            (blob) => ({ blob, error: null }),
            (error) => ({ blob: null, error }),
          );
        if (chunks.length) {
          this.state("speaking");
          let next = synthesize(chunks[0]);
          for (let i = 0; i < chunks.length; i++) {
            const result = await next;
            throwIfAborted(run.signal);
            if (result.error || !result.blob)
              throw result.error || Error(t("Speech synthesis failed."));
            if (i + 1 < chunks.length) next = synthesize(chunks[i + 1]);
            await this.options.play(result.blob, run.signal);
            throwIfAborted(run.signal);
          }
        }
      }
    } catch (e) {
      if (this.run === run && !run.signal.aborted)
        this.error = e instanceof Error ? e.message : t("Voice chat failed.");
    } finally {
      if (this.run === run) {
        this.run = null;
        run.abort();
        this.state("off");
      }
    }
  }
}
function playSpeech(blob: Blob, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob),
      audio = new Audio(url);
    let settled = false;
    const finish = ( error: Error | undefined) => {
      if (settled) return;
      settled = true;
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      URL.revokeObjectURL(url);
      signal.removeEventListener("abort", abort);
      audio.onended = null;
      audio.onerror = null;
      if (error) reject(error);
      else resolve();
    };
    const abort = () => finish(Error(t("Voice stopped.")));
    audio.onended = () => finish(undefined);
    audio.onerror = () => finish(Error(t("Speech playback failed.")));
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) {
      abort();
      return;
    }
    audio
      .play()
      .catch(() =>
        finish(
          Error(
            t(
              "Browser blocked speech playback. Allow audio and start voice chat again.",
            ),
          ),
        ),
      );
  });
}

export { voiceAvailable, speechChunks, VoiceLoop, playSpeech };
