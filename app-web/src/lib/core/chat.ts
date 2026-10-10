import { t } from "../i18n/i18n";
import { Client, jsonPost, record, serverMessage, StudioError } from "./client";
import { readSSE, parseEvent } from "./sse";
import type { RequestOptions } from "./client";
export type ChatMessage = {
  role: string; content:
    string | { type: string; text?: string; image_url?: { url: string; }; }[]; tool_calls?: { id: string; type: string; function: { name: string; arguments: string; }; }[]; tool_call_id?: string;
};

export type ChatRequest = Record<string, unknown> & { model?: string; messages: ChatMessage[]; };

function visionMessage(text: string, dataUrl: string): ChatMessage {
  if (!/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+=*$/.test(dataUrl))
    throw new StudioError(
      "protocol",
      t("Vision input must be an image data URL."),
    );
  return {
    role: "user",
    content: [
      { type: "text", text },
      { type: "image_url", image_url: { url: dataUrl } },
    ],
  };
}
function buildChatRequest(request: ChatRequest) {
  return {
    ...request,
    stream: true,
    stream_options: { ...record(request.stream_options), include_usage: true },
  };
}
export type ChatMetrics = { ttftMs: number | null; elapsedMs: number; tokensPerSecond: number | null; rateSource: "server" | "wall" | null; };

export type ChatEvent = { type: "content" | "reasoning"; text: string; } |
{ type: "tools"; delta: unknown[]; } |
{ type: "finish"; reason: string; } |
{
  type: "metadata";
  usage?: Record<string, unknown>;
  timings?: Record<string, unknown>;
} |
{
  type: "done";
  usage?: Record<string, unknown>;
  timings?: Record<string, unknown>;
  metrics: ChatMetrics;
};

async function* chat(client: Client, request: ChatRequest, options: RequestOptions = {}): AsyncGenerator<ChatEvent> {
  const start = performance.now();
  let first: number | undefined,
    last = start,
    finished = false;
  let usage: Record<string, unknown> | undefined,
    timings: Record<string, unknown> | undefined;
  const opened = await client.open(
    "/v1/chat/completions",
    jsonPost(buildChatRequest(request)),
    options,
  );
  try {
    if (!opened.response.body)
      throw new StudioError("protocol", t("Chat stream has no body."));
    for await (const event of readSSE(opened.response.body, {
      signal: opened.signal,
    })) {
      if (event.data === "[DONE]") {
        finished = true;
        break;
      }
      const r = parseEvent(event.data);
      if (r.error)
        throw new StudioError("http", client.redact(serverMessage(r)));
      if (!Array.isArray(r.choices))
        throw new StudioError("protocol", t("Chat event is missing choices."));
      if (r.usage || r.timings) {
        usage = r.usage ? record(r.usage) : usage;
        timings = r.timings ? record(r.timings) : timings;
        yield { type: "metadata", usage, timings };
      }
      for (const item of r.choices) {
        const choice = record(item),
          delta = record(choice.delta);
        if (typeof choice.index === "number" && choice.index !== 0) continue;
        for (const [key, type] of ([
          ["reasoning_content", "reasoning"],
          ["content", "content"],
        ] as const)) {
          if (delta[key] != null && typeof delta[key] !== "string")
            throw new StudioError("protocol", t("Invalid chat content delta."));
          if (typeof delta[key] === "string" && delta[key]) {
            last = performance.now();
            first ??= last;
            yield { type, text: delta[key] };
          }
        }
        if (delta.tool_calls != null && !Array.isArray(delta.tool_calls))
          throw new StudioError("protocol", t("Invalid tool call delta."));
        if (Array.isArray(delta.tool_calls))
          yield { type: "tools", delta: delta.tool_calls };
        if (typeof choice.finish_reason === "string") {
          finished = true;
          yield { type: "finish", reason: choice.finish_reason };
        }
      }
    }
    if (!finished)
      throw new StudioError(
        "protocol",
        t("Chat stream ended before a finish marker."),
      );
    const count =
      typeof usage?.completion_tokens === "number"
        ? usage.completion_tokens
        : null;
    const serverMs =
      typeof timings?.predicted_ms === "number" && timings.predicted_ms > 0
        ? timings.predicted_ms
        : null;
    const elapsed = first === undefined ? 0 : last - first;
    const rate =
      count === null
        ? null
        : serverMs
          ? count / (serverMs / 1000)
          : elapsed > 0 && count > 1
            ? (count - 1) / (elapsed / 1000)
            : null;
    yield {
      type: "done",
      usage,
      timings,
      metrics: {
        ttftMs: first === undefined ? null : first - start,
        elapsedMs: performance.now() - start,
        tokensPerSecond: rate,
        rateSource: rate === null ? null : serverMs ? "server" : "wall",
      },
    };
  } catch (error) {
    throw client.error(error, opened.signal);
  } finally {
    opened.close();
  }
}

export { visionMessage, buildChatRequest, chat };
