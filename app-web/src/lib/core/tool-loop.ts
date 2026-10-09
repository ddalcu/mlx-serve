import { t } from "../i18n/i18n";
import { newId } from "./id";
import { chat } from "./chat";
import { record, StudioError, throwIfAborted } from "./client";
import type { ChatMessage } from "./chat";
import type { Client } from "./client";
import type { ChatRequest } from "./chat";
import type { ChatEvent } from "./chat";
const MAX_ROUNDS = 3;
export type ToolCall = { id: string; name: string; args: Record<string, unknown>; status: 'pending' | 'running' | 'complete' | 'error' | 'stopped'; result?: string; durationMs?: number; mediaId?: string; mediaType?: 'image' | 'speech' | 'music' | 'sound' | 'video'; };
export type ToolRound = { text: string; calls: ToolCall[]; };
export type ToolOutput = { text: string; mediaId?: string; mediaType?: ToolCall['mediaType']; };
export type LoopOptions = { tools: Record<string, unknown>[]; execute: (call: ToolCall, signal: AbortSignal) => Promise<ToolOutput>; stream?: Function; systemPrompt?: string; signal?: AbortSignal; };
/** SSE fragments are data only. Text/XML tool markup is never executed. */
class ToolCalls {
  parts: Map<number, { id: string; name: string; args: string; }> = new Map();
  push(delta: unknown[]) {
    for (const [position, value] of delta.entries()) {
      const d = record(value),
        f = record(d.function ?? d),
        index = d.index ?? position;
      if (!Number.isInteger(index) || Number(index) < 0 || Number(index) >= 4)
        throw new StudioError(
          "protocol",
          t("Invalid tool call index (maximum 4 calls per round)."),
        );
      const part = this.parts.get(Number(index)) ?? {
        id: "",
        name: "",
        args: "",
      };
      if (d.id !== undefined) {
        if (typeof d.id !== "string" || (part.id && part.id !== d.id))
          throw new StudioError("protocol", t("Invalid tool call identity."));
        part.id = d.id;
      }
      if (f.name !== undefined) {
        if (typeof f.name !== "string")
          throw new StudioError("protocol", t("Invalid tool name."));
        part.name += f.name;
      }
      if (f.arguments !== undefined)
        part.args +=
          typeof f.arguments === "string"
            ? f.arguments
            : JSON.stringify(f.arguments);
      if (
        part.args.length > 65536 ||
        part.name.length > 128 ||
        part.id.length > 256
      )
        throw new StudioError("protocol", t("Tool call is too large."));
      this.parts.set(Number(index), part);
    }
  }
  finish(): ToolCall[] {
    const ids = new Set();
    return [...this.parts]
      .sort(([a], [b]) => a - b)
      .map(([, p]) => {
        if (!/^[a-zA-Z0-9_-]+$/.test(p.name))
          throw new StudioError("protocol", t("Invalid tool name."));
        let raw = p.args.trim() || "{}";
        // Some compatible servers send object args, encoded JSON strings or the
        // double-brace wrapper repaired by mlx-serve's chat.zig. No loose eval.
        if (raw.startsWith("{{") && raw.endsWith("}}")) raw = raw.slice(1, -1);
        let args;
        try {
          args = JSON.parse(raw);
          if (typeof args === "string") args = JSON.parse(args);
        } catch {
          throw new StudioError("protocol", t("Malformed tool arguments."));
        }
        if (!args || typeof args !== "object" || Array.isArray(args))
          throw new StudioError(
            "protocol",
            t("Tool arguments must be an object."),
          );
        const id = p.id || `call_${newId()}`;
        if (ids.has(id))
          throw new StudioError("protocol", t("Duplicate tool call identity."));
        ids.add(id);
        return { id, name: p.name, args, status: "pending" };
      });
  }
}
function roundMessages(round: ToolRound): ChatMessage[] {
  return [
    {
      role: "assistant",
      content: round.text,
      tool_calls: round.calls.map((c) => ({
        id: c.id,
        type: "function",
        function: { name: c.name, arguments: JSON.stringify(c.args) },
      })),
    },
    ...round.calls.map((c) => ({
      role: "tool",
      tool_call_id: c.id,
      content: c.result ?? "Tool stopped before a result was available.",
    })),
  ];
}
async function* toolLoop(client: Client, request: ChatRequest, options: LoopOptions): AsyncGenerator<ChatEvent | { type: 'tool-round' | 'tool-result'; round: ToolRound; }> {
  const signal = options.signal ?? new AbortController().signal;
  const messages = structuredClone(request.messages);
  if (options.systemPrompt)
    messages.unshift({ role: "system", content: options.systemPrompt });
  const names = new Set(options.tools.map((t) => record(t.function).name));
  let mediaUsed = false;
  for (let n = 0; n < MAX_ROUNDS; n++) {
    throwIfAborted(signal);
    const calls = new ToolCalls();
    let text = "",
      finish = "";
    for await (const event of (options.stream ?? chat)(
      client,
      {
        ...request,
        messages,
        tools: options.tools,
        parallel_tool_calls: false,
      },
      { signal },
    )) {
      throwIfAborted(signal);
      if (event.type === "tools") calls.push(event.delta);
      else {
        if (event.type === "content") text += event.text;
        if (event.type === "finish") finish = event.reason;
        yield event;
      }
    }
    throwIfAborted(signal);
    if (!calls.parts.size) {
      if (finish === "tool_calls")
        throw new StudioError(
          "protocol",
          t("Tool response is missing calls; no tools were run."),
        );
      return;
    }
    if (finish !== "tool_calls" && finish !== "stop")
      throw new StudioError(
        "protocol",
        t("Tool response is incomplete; no tools were run."),
      );
    const parsed = calls.finish();
    for (const c of parsed)
      if (!names.has(c.name))
        throw new StudioError(
          "unsupported",
          t("Tool %@ is not available in this web client.", [c.name]),
        );
    if (n === MAX_ROUNDS - 1)
      throw new StudioError(
        "protocol",
        t("Tools stopped at the %@-round limit.", [MAX_ROUNDS]),
      );
    const round = { text, calls: parsed };
    yield { type: "tool-round", round };
    for (const call of parsed) {
      throwIfAborted(signal);
      call.status = "running";
      yield { type: "tool-result", round };
      const start = performance.now();
      try {
        if (call.name !== "search_library") {
          if (mediaUsed)
            throw Error(
              t(
                "Only one media generation is allowed per turn. Ask the user to start another turn.",
              ),
            );
          // A failed request may already have used the GPU; do not retry this turn.
          mediaUsed = true;
        }
        const result = await options.execute(call, signal);
        throwIfAborted(signal);
        call.result = result.text.slice(0, 16000);
        call.mediaId = result.mediaId;
        call.mediaType = result.mediaType;
        call.status = "complete";
      } catch (error) {
        throwIfAborted(signal);
        call.status = "error";
        call.result = client
          .redact(error instanceof Error ? error.message : "Tool failed.")
          .slice(0, 16000);
      }
      call.durationMs = Math.round(performance.now() - start);
      yield { type: "tool-result", round };
    }
    messages.push(...roundMessages(round));
  }
}
/**
 * Restore display/history data only; pending work never resumes on reload/import.
 */
function cleanToolRounds(value: unknown): ToolRound[] {
  if (!Array.isArray(value) || value.length > MAX_ROUNDS)
    throw Error(t("Invalid saved tool rounds."));
  const text = ( v: unknown, limit = 16000) => {
    if (typeof v !== "string" || v.length > limit)
      throw Error(t("Invalid saved tool text."));
    return v;
  };
  return value.map((v) => {
    const round = record(v);
    if (!Array.isArray(round.calls) || round.calls.length > 4)
      throw Error(t("Invalid saved tool calls."));
    return {
      text: text(round.text ?? "", 2_000_000),
      calls: round.calls.map((v) => {
        const c = record(v),
          name = text(c.name, 128);
        if (
          ![
            "generate_image",
            "generate_speech",
            "generate_music",
            "generate_sound",
            "generate_video",
            "search_library",
          ].includes(name)
        )
          throw Error(t("Unknown saved tool."));
         const call: ToolCall = {
          id: text(c.id, 256),
          name,
          args: Object.fromEntries(
            Object.entries(record(c.args))
              .filter(([k]) => ["query", "model", "prompt"].includes(k))
              .map(([k, v]) => [k, text(v, 4096)]),
          ),
          status:
            c.status === "complete"
              ? "complete"
              : c.status === "error"
                ? "error"
                : "stopped",
        };
        if (c.result !== undefined) call.result = text(c.result);
        if (
          typeof c.durationMs === "number" &&
          Number.isFinite(c.durationMs) &&
          c.durationMs >= 0
        )
          call.durationMs = c.durationMs;
        if (
          typeof c.mediaId === "string" &&
          ["image", "speech", "music", "sound", "video"].includes(
            String(c.mediaType),
          )
        ) {
          call.mediaId = text(c.mediaId, 256);
          call.mediaType = (c.mediaType as ToolCall['mediaType']);
        }
        return call;
      }),
    };
  });
}

export { MAX_ROUNDS, ToolCalls, roundMessages, toolLoop, cleanToolRounds };
