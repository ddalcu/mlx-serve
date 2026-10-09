import { t } from "../i18n/i18n";
import { Client, StudioError } from "./client";
import { decodeBase64, mediaPayload } from "./media";
import type { MediaOptions } from "./media";
export type RawVideo = {
  rgb: Uint8Array<ArrayBuffer>; width: number; height: number; frames: number; fps: number; durationSeconds: number; audio?: {
    pcm: Uint8Array<ArrayBuffer>;
    sampleRate: number;
    channels: number;
  };
};

function decodeVideo(value: Record<string, unknown>): RawVideo {
  const integer = (key: string, max: number, defaultValue?: number) => {
    const n = value[key] ?? defaultValue;
    if (typeof n !== "number" || !Number.isSafeInteger(n) || n <= 0 || n > max)
      throw new StudioError("protocol", t("Invalid video %@.", [key]));
    return n;
  };
  const width = integer("width", 4096),
    height = integer("height", 4096),
    frames = integer("frames", 10000),
    fps = integer("fps", 120, 24),
    expected = width * height * frames * 3;
  if (value.format !== "rgb8" || expected > 256 * 1024 * 1024)
    throw new StudioError("protocol", t("Unsupported or oversized video."));
  const rgb = decodeBase64(value.data, expected);
  if (rgb.length !== expected)
    throw new StudioError(
      "protocol",
      t("Video frame length does not match dimensions."),
    );
  let audio: RawVideo["audio"];
  if (value.audio_data) {
    if (value.audio_format !== "pcm_s16le")
      throw new StudioError("protocol", t("Unsupported video audio format."));
    const sampleRate = integer("audio_sample_rate", 192000),
      channels = integer("audio_channels", 8),
      pcm = decodeBase64(value.audio_data, 64 * 1024 * 1024);
    if (pcm.length % (channels * 2))
      throw new StudioError("protocol", t("Invalid PCM audio length."));
    audio = { pcm, sampleRate, channels };
  }
  return {
    rgb,
    width,
    height,
    frames,
    fps,
    durationSeconds: frames / fps,
    audio,
  };
}
export type VideoRequest = Record<string, unknown> & { model?: string; prompt: string; width?: number; height?: number; num_frames?: number; steps?: number; preview?: boolean; };

async function generateVideo(client: Client, request: VideoRequest, options: MediaOptions = {}): Promise<{ raw: RawVideo; elapsedMs: number; wireBytes: number; }> {
  const result = await mediaPayload(
    client,
    "/v1/video/generations",
    { ...request, stream: true, preview: request.preview ?? false },
    { ...options, stream: true },
    384 * 1024 * 1024,
  );
  return {
    raw: decodeVideo(result.payload),
    elapsedMs: result.elapsedMs,
    wireBytes: result.wireBytes,
  };
}

export { decodeVideo, generateVideo };
