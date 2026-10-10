import { t } from "../i18n/i18n";
import { Client, jsonPost, record, readBody, serverMessage, StudioError } from "./client";
import { parseEvent, readSSE } from "./sse";
import type { RequestOptions } from "./client";
export type MediaOptions = RequestOptions & { stream?: boolean; onProgress?: (event: Record<string, unknown>) => void; };

export type MediaResult = { blob: Blob; elapsedMs: number; wireBytes: number; };

function decodeBase64(value: unknown, maxBytes = 64 * 1024 * 1024): Uint8Array<ArrayBuffer> {
  if (typeof value !== "string" || value.length === 0)
    throw new StudioError("protocol", t("Missing base64 media."));
  if (value.length > Math.ceil(maxBytes / 3) * 4)
    throw new StudioError("protocol", t("Media is too large."));
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  const length = (value.length / 4) * 3 - padding;
  if (value.length % 4 !== 0 || length > maxBytes)
    throw new StudioError("protocol", t("Invalid base64 media."));
  // Validate bounded spans with a negated character class: no giant regex
  // backtracking stack, binary string, or iterable-to-array conversion.
  const end = value.length - padding;
  for (let at = 0; at < end; at += 65536)
    if (/[^A-Za-z0-9+/]/.test(value.slice(at, Math.min(at + 65536, end))))
      throw new StudioError("protocol", t("Invalid base64 media."));
  try {
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (let at = 0; at < value.length; at += 65536) {
      const binary = atob(value.slice(at, at + 65536));
      for (let i = 0; i < binary.length; i++)
        bytes[offset++] = binary.charCodeAt(i);
    }
    return bytes;
  } catch {
    throw new StudioError("protocol", t("Invalid base64 media."));
  }
}
function pngBlob(value: unknown): Blob {
  const bytes = decodeBase64(value, 32 * 1024 * 1024),
    view = new DataView(bytes.buffer);
  if (
    ![137, 80, 78, 71, 13, 10, 26, 10].every((v, i) => bytes[i] === v) ||
    bytes.length < 45
  )
    throw new StudioError("protocol", t("Invalid PNG."));
  const width = view.getUint32(16),
    height = view.getUint32(20);
  if (!width || !height || width * height > 16 * 1024 * 1024)
    throw new StudioError("protocol", t("PNG dimensions are too large."));
  let offset = 8,
    ended = false;
  while (offset + 12 <= bytes.length) {
    const length = view.getUint32(offset);
    if (offset + 12 + length > bytes.length) break;
    const kind = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    offset += length + 12;
    if (kind === "IEND") {
      ended = true;
      break;
    }
  }
  if (!ended) throw new StudioError("protocol", t("Truncated PNG."));
  return new Blob([bytes], { type: "image/png" });
}
function wavBlob(bytes: Uint8Array<ArrayBuffer>): Blob {
  const text = (a: number, b: number) => String.fromCharCode(...bytes.subarray(a, b));
  if (
    bytes.length < 44 ||
    text(0, 4) !== "RIFF" ||
    text(8, 12) !== "WAVE" ||
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
      4,
      true,
    ) +
      8 >
      bytes.length
  )
    throw new StudioError("protocol", t("Invalid or truncated WAV."));
  let offset = 12,
    fmt = false,
    data = false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  while (offset + 8 <= bytes.length) {
    const size = view.getUint32(offset + 4, true);
    if (offset + 8 + size > bytes.length)
      throw new StudioError("protocol", t("Truncated WAV chunk."));
    const kind = text(offset, offset + 4);
    if (kind === "fmt ") fmt = size >= 16;
    if (kind === "data") data = size > 0;
    offset += 8 + size + (size % 2);
  }
  if (!fmt || !data)
    throw new StudioError("protocol", t("WAV audio is missing."));
  return new Blob([bytes], { type: "audio/wav" });
}
async function mediaPayload(
  client: Client,
  path: string,
  body: unknown,
  options: MediaOptions = {},
  maxEventBytes = 64 * 1024 * 1024,
): Promise<{
  payload: Record<string, unknown>;
  wireBytes: number;
  elapsedMs: number;
}> {
  const start = performance.now(),
    init = body instanceof FormData ? { method: "POST", body } : jsonPost(body);
  const opened = await client.open(path, init, {
    timeoutMs: 15 * 60_000,
    ...options,
  });
  let wireBytes = 0;
  try {
    if (!opened.response.body)
      throw new StudioError("protocol", t("Media response has no body."));
    if (
      options.stream === false ||
      opened.response.headers.get("Content-Type")?.includes("application/json")
    ) {
      const bytes = await readBody(
        opened.response,
        opened.signal,
        maxEventBytes,
      );
      wireBytes = bytes.length;
      const payload = parseEvent(new TextDecoder().decode(bytes));
      if (payload.error) throw new StudioError("http", serverMessage(payload));
      return { payload, wireBytes, elapsedMs: performance.now() - start };
    }
    for await (const event of readSSE(opened.response.body, {
      signal: opened.signal,
      maxEventBytes,
      onBytes: (n) => {
        wireBytes += n;
      },
    })) {
      if (event.data === "[DONE]") break;
      const payload = parseEvent(event.data),
        type = payload.type ?? event.event;
      if (type === "error" || payload.error)
        throw new StudioError("http", serverMessage(payload));
      if (type === "complete")
        return { payload, wireBytes, elapsedMs: performance.now() - start };
      options.onProgress?.(payload);
    }
    throw new StudioError(
      "protocol",
      t("Media stream ended without a complete payload."),
    );
  } catch (error) {
    throw client.error(error, opened.signal);
  } finally {
    opened.close();
  }
}
async function audioRequest(client: Client, path: string, body: Record<string, unknown>, options: MediaOptions = {}): Promise<MediaResult> {
  // Swift permits 600s of 48 kHz stereo PCM16: 115.2 MB plus WAV header.
  const maxWavBytes = 128 * 1024 * 1024;
  const stream = options.stream ?? false,
    start = performance.now();
  if (stream) {
    const result = await mediaPayload(
      client,
      path,
      { ...body, stream: true },
      { ...options, stream: true },
      Math.ceil(maxWavBytes / 3) * 4 + 8192,
    );
    return {
      ...result,
      blob: wavBlob(decodeBase64(result.payload.data, maxWavBytes)),
    };
  }
  const bytes = await client.bytes(
    path,
    jsonPost({ ...body, stream: false }),
    {
      timeoutMs: 15 * 60_000,
      ...options,
    },
    maxWavBytes,
  );
  return {
    blob: wavBlob(bytes),
    elapsedMs: performance.now() - start,
    wireBytes: bytes.length,
  };
}
const imageData = (payload: Record<string, unknown>) =>
  Array.isArray(payload.data) ? record(payload.data[0]).b64_json : payload.data;

export { decodeBase64, pngBlob, wavBlob, mediaPayload, audioRequest, imageData };
