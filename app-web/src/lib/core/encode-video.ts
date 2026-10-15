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
/** An MP4 encoder that takes clips one after another on one timeline: a storyboard's shots arrive one at a time. */
function mp4Writer(shape: RawVideo, signal: AbortSignal, audioCodec: "aac" | "opus" | "silent") {
  const video: EncodedSample[] = [],
    audio: EncodedSample[] = [];
  let description: Uint8Array | undefined,
    audioDescription: Uint8Array | undefined,
    failure: Error | undefined,
    frameIndex = 0,
    audioFrames = 0;
  const onError = (e: DOMException) => {
    failure = e;
  };
  const copyDescription = (d: AllowSharedBufferSource) =>
    new Uint8Array(ArrayBuffer.isView(d) ? d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength) : d.slice(0));
  const ve = new VideoEncoder({
    output: (c, m) => {
      const data = new Uint8Array(c.byteLength);
      c.copyTo(data);
      video.push({
        data,
        timestamp: c.timestamp,
        duration: c.duration ?? Math.round(1e6 / shape.fps),
        key: c.type === "key",
      });
      if (m?.decoderConfig?.description) description = copyDescription(m.decoderConfig.description);
    },
    error: onError,
  });
  const a = shape.audio;
  const ae = a
    ? new AudioEncoder({
        output: (c, m) => {
          if (m?.decoderConfig?.description) audioDescription = copyDescription(m.decoderConfig.description);
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
      })
    : undefined;
  const close = () => {
    signal.removeEventListener("abort", close);
    if (ve.state !== "closed") ve.close();
    if (ae && ae.state !== "closed") ae.close();
  };
  signal.addEventListener("abort", close, { once: true });
  const check = () => {
    throwIfAborted(signal);
    if (failure) throw failure;
  };
  check();
  ve.configure({
    codec: "avc1.42001f",
    width: shape.width,
    height: shape.height,
    bitrate: 1_000_000,
    framerate: shape.fps,
    avc: { format: "avc" },
    latencyMode: "realtime",
  });
  if (a && ae)
    ae.configure({
      codec: audioCodec === "opus" ? "opus" : "mp4a.40.2",
      sampleRate: a.sampleRate,
      numberOfChannels: a.channels,
      bitrate: 128_000,
    });
  const pixels = new Uint8ClampedArray(shape.width * shape.height * 4);
  return {
    /** Append `raw` after what is already encoded, dropping its first `skip` frames (and their sound). */
    async add(raw: RawVideo, skip: number) {
      if (raw.width !== shape.width || raw.height !== shape.height || raw.fps !== shape.fps || !raw.audio !== !a)
        throw new StudioError("protocol", t("Every shot must share one canvas, frame rate and soundtrack."));
      for (let i = skip; i < raw.frames; i++, frameIndex++) {
        check();
        rgba(raw, i, pixels);
        const frame = new VideoFrame(pixels, {
          format: "RGBA",
          codedWidth: raw.width,
          codedHeight: raw.height,
          timestamp: Math.round((frameIndex * 1e6) / raw.fps),
          duration: Math.round(1e6 / raw.fps),
        });
        try {
          ve.encode(frame, { keyFrame: frameIndex % Math.max(1, raw.fps * 2) === 0 });
        } finally {
          frame.close();
        }
        if (ve.encodeQueueSize >= 4) await ve.flush();
      }
      if (!raw.audio || !a || !ae) return;
      const pcm = pcmFloats(raw),
        count = pcm.length / a.channels,
        from = Math.min(count, Math.round((skip * a.sampleRate) / raw.fps));
      for (let offset = from; offset < count; offset += 1024) {
        check();
        const frames = Math.min(1024, count - offset);
        const sample = new AudioData({
          format: "f32",
          sampleRate: a.sampleRate,
          numberOfChannels: a.channels,
          numberOfFrames: frames,
          timestamp: Math.round(((audioFrames + offset - from) * 1e6) / a.sampleRate),
          data: pcm.subarray(offset * a.channels, (offset + frames) * a.channels),
        });
        try {
          ae.encode(sample);
        } finally {
          sample.close();
        }
        if (ae.encodeQueueSize >= 8) await delay(0, signal);
      }
      audioFrames += count - from;
    },
    async finish(): Promise<Blob> {
      await ve.flush();
      if (ae) await ae.flush();
      check();
      if (!description)
        throw new StudioError(
          "protocol",
          t("H.264 encoder returned no decoder configuration."),
        );
      return muxMp4(
        {
          width: shape.width,
          height: shape.height,
          fps: shape.fps,
          description,
          chunks: video,
        },
        a
          ? {
              sampleRate: audioCodec === "opus" ? 48000 : a.sampleRate,
              channels: a.channels,
              chunks: audio,
              codec: audioCodec === "opus" ? "opus" : "aac",
              description: audioDescription,
              durationSeconds: audioFrames / a.sampleRate,
            }
          : undefined,
      );
    },
    close,
  };
}
async function mp4(raw: RawVideo, signal: AbortSignal, audioCodec: "aac" | "opus" | "silent"): Promise<Blob> {
  let writer: ReturnType<typeof mp4Writer> | undefined;
  try {
    writer = mp4Writer(raw, signal, audioCodec);
    await writer.add(raw, 0);
    return await writer.finish();
  } catch (e) {
    throwIfAborted(signal);
    throw e;
  } finally {
    writer?.close();
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
      codec: useMp4 ? h264Label(raw, selected) : `browser WebM (VP8/VP9${raw.audio ? " + Opus" : ""})`,
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

const h264Label = (raw: RawVideo, audio: "aac" | "opus" | "silent") =>
  "H.264" + (raw.audio ? (audio === "opus" ? " + Opus" : " + AAC") : "");

export type ShotEncoder = { add(raw: RawVideo, skip: number): Promise<void>; finish(): Promise<EncodeResult>; close(): void };

/**
 * A storyboard's encoder: shots are appended as they arrive, each dropping the frame it shares with the
 * shot before, so only one shot's raw frames are ever held. Needs WebCodecs: the WebM fallback records in
 * real time from one buffer.
 */
async function openShotEncoder(first: RawVideo, options: EncodeOptions = {}): Promise<ShotEncoder> {
  const signal = options.signal ?? new AbortController().signal;
  const selected = await support(first);
  if (!selected)
    throw new StudioError("unsupported", t("Storyboards need this browser's H.264 encoder (WebCodecs). Try Chrome, Edge or Safari."));
  const writer = mp4Writer(first, signal, selected);
  let encodeMs = 0;
  return {
    async add(raw, skip) {
      const start = performance.now();
      await writer.add(raw, skip);
      encodeMs += performance.now() - start;
    },
    async finish() {
      const start = performance.now(),
        blob = await writer.finish();
      encodeMs += performance.now() - start;
      writer.close();
      return { blob, encodeMs, container: "mp4", codec: h264Label(first, selected), heapBeforeBytes: null, peakHeapBytes: null, workingBufferBytes: blob.size };
    },
    close: writer.close,
  };
}

/** The clip's last frame as a lossless PNG (base64): the next shot opens on it. */
async function lastFramePng(raw: RawVideo): Promise<string> {
  const pixels = new Uint8ClampedArray(raw.width * raw.height * 4);
  rgba(raw, raw.frames - 1, pixels);
  const image = new ImageData(pixels, raw.width, raw.height);
  let blob: Blob | null;
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(raw.width, raw.height);
    canvas.getContext("2d")!.putImageData(image, 0, 0);
    blob = await canvas.convertToBlob({ type: "image/png" });
  } else {
    const canvas = document.createElement("canvas");
    canvas.width = raw.width;
    canvas.height = raw.height;
    canvas.getContext("2d")!.putImageData(image, 0, 0);
    blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  }
  if (!blob) throw new StudioError("unsupported", t("Couldn't save the shot's last frame."));
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

export { encodeVideo, openShotEncoder, lastFramePng };
