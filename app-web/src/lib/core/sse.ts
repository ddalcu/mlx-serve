import { t } from "../i18n/i18n";
import { StudioError, throwIfAborted } from "./client";
export type SSEEvent = { event: string; data: string; id?: string; };

/**
 * Generic SSE framing shared by chat and media; EOF does not dispatch an unfinished event.
 */
async function* readSSE(source: ReadableStream<Uint8Array>, options: {
  signal?: AbortSignal;
  maxEventBytes?: number;
  onBytes?: (n: number) => void;
} = {}): AsyncGenerator<SSEEvent> {
  const reader = source.getReader(),
    decoder = new TextDecoder("utf-8", { fatal: true });
  let line = ([] as string[]),
    data: string[] = [],
    event = "",
    id: string | undefined,
    size = 0,
    afterCR = false;
  const limit = options.maxEventBytes ?? 1024 * 1024;
  const abort = () => {
    void reader.cancel().catch(() => {});
  };
  options.signal?.addEventListener("abort", abort, { once: true });
  const consumeLine = (): SSEEvent | undefined => {
    const current = line.join("");
    line = [];
    if (current === "") {
      const result = data.length
        ? {
            event: event || "message",
            data: data.join("\n"),
            ...(id === undefined ? {} : { id }),
          }
        : undefined;
      data = [];
      event = "";
      size = 0;
      return result;
    }
    if (current.startsWith(":")) return;
    const colon = current.indexOf(":"),
      field = colon < 0 ? current : current.slice(0, colon);
    let value = colon < 0 ? "" : current.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") data.push(value);
    else if (field === "event") event = value;
    else if (field === "id" && !value.includes("\0")) id = value;
  };
  try {
    for (;;) {
      throwIfAborted(options.signal);
      const { value, done } = await reader.read();
      throwIfAborted(options.signal);
      if (value) options.onBytes?.(value.byteLength);
      let text: string;
      try {
        text = decoder.decode(value, { stream: !done });
      } catch {
        throw new StudioError("protocol", t("Invalid UTF-8 in SSE stream."));
      }
      // Scan spans rather than appending each character (media complete events can be large).
      let start = 0;
      for (let i = 0; i < text.length; i++) {
        const char = text[i];
        if (afterCR) {
          afterCR = false;
          if (char === "\n") {
            start = i + 1;
            continue;
          }
        }
        if (char !== "\r" && char !== "\n") continue;
        const span = text.slice(start, i);
        if (span) line.push(span);
        size += new TextEncoder().encode(span).length + 1;
        if (size > limit)
          throw new StudioError("protocol", t("SSE event is too large."));
        const result = consumeLine();
        if (result) yield result;
        afterCR = char === "\r";
        start = i + 1;
      }
      const rest = text.slice(start);
      if (rest) line.push(rest);
      size += new TextEncoder().encode(rest).length;
      if (size > limit)
        throw new StudioError("protocol", t("SSE event is too large."));
      if (done) break;
    }
  } finally {
    options.signal?.removeEventListener("abort", abort);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
function parseEvent(data: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    throw new StudioError("protocol", t("Invalid JSON in SSE event."));
  }
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new StudioError("protocol", t("Invalid SSE event shape."));
  return (value as Record<string, unknown>);
}

export { readSSE, parseEvent };
