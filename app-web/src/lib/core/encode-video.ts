import { t } from "../i18n/i18n";
import { abortError, StudioError, throwIfAborted } from "./client";
import { muxMp4 } from "../mp4/mp4";
import type { EncodedSample } from "../mp4/mp4";
import type { RawVideo } from "./video";

export type EncodeOptions = { signal?: AbortSignal; timeoutMs?: number; preferWebM?: boolean; };

export type EncodeResult = { blob: Blob; encodeMs: number; container: "mp4" | "webm"; codec: string; heapBeforeBytes: number | null; peakHeapBytes: number | null; workingBufferBytes: number; };

const heap = () =>
  (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize ?? null;
function rgba(raw: RawVideo, index: number, target: Uint8ClampedArray<ArrayBuffer>) {
  for (
    let pixel = 0, source = index * raw.width * raw.height * 3;
    pixel < target.length;
    pixel += 4, source += 3
  ) {
    target[pixel] = raw.rgb[source];
    target[pixel + 1] = raw.rgb[source + 1];
    target[pixel + 2] = raw.rgb[source + 2];
    target[pixel + 3] = 255;
  }
}
function pcmFloats(raw: RawVideo) {
  const audio =
    (raw.audio as { pcm: Uint8Array<ArrayBuffer>; sampleRate: number; channels: number; });
  const view = new DataView(
      audio.pcm.buffer,
      audio.pcm.byteOffset,
      audio.pcm.byteLength,
    ),
    samples = audio.pcm.length / 2;
  const out = new Float32Array(samples);
  for (let i = 0; i < samples; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}
async function support(raw: RawVideo): Promise<"aac" | "opus" | "silent" | null> {
  if (typeof VideoEncoder === "undefined") return null;
  try {
    const video = await VideoEncoder.isConfigSupported({
      codec: "avc1.42001f",
      width: raw.width,
      height: raw.height,
      bitrate: 1_000_000,
      framerate: raw.fps,
      avc: { format: "avc" },
      latencyMode: "realtime",
    });
    if (!video.supported) return null;
    if (!raw.audio) return "silent";
    if (typeof AudioEncoder === "undefined") return null;
    for (const codec of (["aac", "opus"] as const)) {
      if (codec === "opus" && raw.audio.channels > 2) continue;
      if (
        (
          await AudioEncoder.isConfigSupported({
            codec: codec === "aac" ? "mp4a.40.2" : "opus",
            sampleRate: raw.audio.sampleRate,
            numberOfChannels: raw.audio.channels,
            bitrate: 128_000,
          })
        ).supported
      )
        return codec;
    }
    return null;
  } catch {
    return null;
  }
}
async function mp4(raw: RawVideo, signal: AbortSignal, audioCodec: "aac" | "opus" | "silent"): Promise<Blob> {
  const video: EncodedSample[] = [],
    audio: EncodedSample[] = [];
  let description: Uint8Array | undefined,
    audioDescription: Uint8Array | undefined,
    failure: Error | undefined;
  const onError = (e: DOMException) => {
    failure = e;
  };
  const ve = new VideoEncoder({
    output: (c, m) => {
      const data = new Uint8Array(c.byteLength);
      c.copyTo(data);
      video.push({
        data,
        timestamp: c.timestamp,
        duration: c.duration ?? Math.round(1e6 / raw.fps),
        key: c.type === "key",
      });
      if (m?.decoderConfig?.description) {
        const d = m.decoderConfig.description;
        description = new Uint8Array(
          ArrayBuffer.isView(d)
            ? d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength)
            : d.slice(0),
        );
      }
    },
    error: onError,
  });
  let ae: AudioEncoder | undefined;
  const cancel = () => {
    if (ve.state !== "closed") ve.close();
    if (ae && ae.state !== "closed") ae.close();
  };
  signal.addEventListener("abort", cancel, { once: true });
  const check = () => {
    throwIfAborted(signal);
    if (failure) throw failure;
  };
  try {
    check();
    ve.configure({
      codec: "avc1.42001f",
      width: raw.width,
      height: raw.height,
      bitrate: 1_000_000,
      framerate: raw.fps,
      avc: { format: "avc" },
      latencyMode: "realtime",
    });
    const pixels = new Uint8ClampedArray(raw.width * raw.height * 4);
    for (let i = 0; i < raw.frames; i++) {
      check();
      rgba(raw, i, pixels);
      const frame = new VideoFrame(pixels, {
        format: "RGBA",
        codedWidth: raw.width,
        codedHeight: raw.height,
        timestamp: Math.round((i * 1e6) / raw.fps),
        duration: Math.round(1e6 / raw.fps),
      });
      try {
        ve.encode(frame, { keyFrame: i % Math.max(1, raw.fps * 2) === 0 });
      } finally {
        frame.close();
      }
      if (ve.encodeQueueSize >= 4) await ve.flush();
    }
    await ve.flush();
    check();
    if (raw.audio) {
      const a = raw.audio,
        pcm = pcmFloats(raw);
      ae = new AudioEncoder({
        output: (c, m) => {
          if (m?.decoderConfig?.description) {
            const d = m.decoderConfig.description;
            audioDescription = new Uint8Array(
              ArrayBuffer.isView(d)
                ? d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength)
                : d.slice(0),
            );
          }
          const data = new Uint8Array(c.byteLength);
          c.copyTo(data);
          audio.push({
            data,
            timestamp: c.timestamp,
            duration: c.duration ?? Math.round((1024 * 1e6) / a.sampleRate),
            key: true,
          });
        },
        error: onError,
      });
      ae.configure({
        codec: audioCodec === "opus" ? "opus" : "mp4a.40.2",
        sampleRate: a.sampleRate,
        numberOfChannels: a.channels,
        bitrate: 128_000,
      });
      const count = pcm.length / a.channels;
      for (let offset = 0; offset < count; offset += 1024) {
        check();
        const frames = Math.min(1024, count - offset);
        const sample = new AudioData({
          format: "f32",
          sampleRate: a.sampleRate,
          numberOfChannels: a.channels,
          numberOfFrames: frames,
          timestamp: Math.round((offset * 1e6) / a.sampleRate),
          data: pcm.subarray(
            offset * a.channels,
            (offset + frames) * a.channels,
          ),
        });
        try {
          ae.encode(sample);
        } finally {
          sample.close();
        }
        if (ae.encodeQueueSize >= 8) await delay(0, signal);
      }
      await ae.flush();
      check();
    }
    if (!description)
      throw new StudioError(
        "protocol",
        t("H.264 encoder returned no decoder configuration."),
      );
    return muxMp4(
      {
        width: raw.width,
        height: raw.height,
        fps: raw.fps,
        description,
        chunks: video,
      },
      raw.audio
        ? {
            sampleRate: audioCodec === "opus" ? 48000 : raw.audio.sampleRate,
            channels: raw.audio.channels,
            chunks: audio,
            codec: audioCodec === "opus" ? "opus" : "aac",
            description: audioDescription,
            durationSeconds:
              raw.audio.pcm.byteLength /
              (2 * raw.audio.channels * raw.audio.sampleRate),
          }
        : undefined,
    );
  } catch (e) {
    throwIfAborted(signal);
    throw e;
  } finally {
    signal.removeEventListener("abort", cancel);
    cancel();
  }
}
function abortable<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const abort = () => reject(abortError(signal));
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    promise
      .then(resolve, reject)
      .finally(() => signal.removeEventListener("abort", abort));
  });
}
const delay = (ms: number, signal: AbortSignal) =>
  (new Promise((resolve, reject) => {
      throwIfAborted(signal);
      const abort = () => {
        clearTimeout(timer);
        reject(abortError(signal));
      };
      const timer = setTimeout(() => {
        signal.removeEventListener("abort", abort);
        resolve();
      }, ms);
      signal.addEventListener("abort", abort, { once: true });
    }) as Promise<void>);
/**
 * Native WebM fallback: real-time canvas capture + PCM audio; no downloadable codecs.
 */
async function webm(raw: RawVideo, signal: AbortSignal): Promise<Blob> {
  if (
    typeof MediaRecorder === "undefined" ||
    typeof document === "undefined" ||
    raw.durationSeconds > 30
  )
    throw new StudioError(
      "unsupported",
      t(
        "H.264/AAC encoding is unavailable. WebM fallback needs MediaRecorder and a clip of at most 30 seconds.",
      ),
    );
  // Firefox waits for an audio encoder that never starts if a silent canvas
  // stream is declared as VP9+Opus. Advertise only tracks present in the stream.
  const audioSuffix = raw.audio ? ",opus" : "";
  const mime = [
    `video/webm;codecs=vp9${audioSuffix}`,
    `video/webm;codecs=vp8${audioSuffix}`,
    "video/webm",
  ].find((m) => MediaRecorder.isTypeSupported(m));
  if (!mime)
    throw new StudioError(
      "unsupported",
      t("No supported browser video encoder."),
    );
  const canvas = document.createElement("canvas");
  canvas.width = raw.width;
  canvas.height = raw.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || !canvas.captureStream)
    throw new StudioError(
      "unsupported",
      t("Canvas video capture is unavailable."),
    );
  const stream = canvas.captureStream(raw.fps),
    pixels = new Uint8ClampedArray(raw.width * raw.height * 4),
    frame = new ImageData(pixels, raw.width, raw.height);
  let audioContext: AudioContext | undefined,
    source: AudioBufferSourceNode | undefined,
    recorder: MediaRecorder | undefined;
  try {
    throwIfAborted(signal);
    if (raw.audio) {
      audioContext = new AudioContext({ sampleRate: raw.audio.sampleRate });
      const floats = pcmFloats(raw),
        count = floats.length / raw.audio.channels,
        buffer = audioContext.createBuffer(
          raw.audio.channels,
          count,
          raw.audio.sampleRate,
        );
      for (let ch = 0; ch < raw.audio.channels; ch++) {
        const data = buffer.getChannelData(ch);
        for (let i = 0; i < count; i++)
          data[i] = floats[i * raw.audio.channels + ch];
      }
      const destination = audioContext.createMediaStreamDestination();
      source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(destination);
      destination.stream.getAudioTracks().forEach((t) => stream.addTrack(t));
      await abortable(audioContext.resume(), signal);
    }
    const parts: Blob[] = [];
    recorder = new MediaRecorder(stream, {
      mimeType: mime,
      videoBitsPerSecond: 1_000_000,
    });
    const r = recorder;
    const stopped: Promise<void> = new Promise((resolve, reject) => {
      r.ondataavailable = (e) => {
        if (e.data.size) parts.push(e.data);
      };
      r.onstop = () => resolve();
      r.onerror = () =>
        reject(new StudioError("unsupported", t("WebM recording failed.")));
    });
    // Attach a handler now so an asynchronous recorder error cannot become unhandled.
    void stopped.catch(() => {});
    rgba(raw, 0, pixels);
    ctx.putImageData(frame, 0, 0);
    r.start();
    source?.start();
    const start = performance.now();
    for (let i = 0; i < raw.frames; i++) {
      throwIfAborted(signal);
      rgba(raw, i, pixels);
      ctx.putImageData(frame, 0, 0);
      await delay(
        Math.max(0, start + ((i + 1) * 1000) / raw.fps - performance.now()),
        signal,
      );
    }
    source?.stop();
    r.stop();
    await abortable(stopped, signal);
    throwIfAborted(signal);
    if (!parts.length)
      throw new StudioError("protocol", t("WebM encoder returned no data."));
    return new Blob(parts, { type: "video/webm" });
  } finally {
    if (recorder && recorder.state !== "inactive") recorder.stop();
    stream.getTracks().forEach((t) => t.stop());
    await audioContext?.close();
  }
}
async function encodeVideo(raw: RawVideo, options: EncodeOptions = {}): Promise<EncodeResult> {
  const controller = new AbortController(),
    abort = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener("abort", abort, { once: true });
  if (options.signal?.aborted) abort();
  const timer = setTimeout(
    () =>
      controller.abort(
        new StudioError("timeout", t("Video encoding timed out.")),
      ),
    options.timeoutMs ?? 120_000,
  );
  const before = heap();
  let peak = before;
  const sample = () => {
    const h = heap();
    if (h !== null) peak = Math.max(peak ?? 0, h);
  };
  const sampler = setInterval(sample, 25),
    start = performance.now();
  try {
    throwIfAborted(controller.signal);
    const selected = options.preferWebM ? null : await support(raw),
      useMp4 = selected !== null;
    const blob = await (selected
      ? mp4(raw, controller.signal, selected)
      : webm(raw, controller.signal));
    sample();
    return {
      blob,
      encodeMs: performance.now() - start,
      container: useMp4 ? "mp4" : "webm",
      codec: useMp4
        ? "H.264" +
          (raw.audio ? (selected === "opus" ? " + Opus" : " + AAC") : "")
        : `browser WebM (VP8/VP9${raw.audio ? " + Opus" : ""})`,
      heapBeforeBytes: before,
      peakHeapBytes: peak,
      workingBufferBytes:
        raw.rgb.byteLength +
        (raw.audio?.pcm.byteLength ?? 0) * 3 +
        raw.width * raw.height * 4 +
        blob.size,
    };
  } finally {
    clearTimeout(timer);
    clearInterval(sampler);
    options.signal?.removeEventListener("abort", abort);
  }
}

export { encodeVideo };
