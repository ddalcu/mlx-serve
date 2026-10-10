import type { Model } from "../../src/lib/core/models";

export type Metrics = { counters?: Record<string, number>; gauges?: Record<string, number>; histograms?: Record<string, { sum: number; count: number }>; sessions?: object[] };
export type Reply = { reasoning?: string; content?: string; status?: number; error?: string; hold?: Promise<void> };
export type Mock = ReturnType<typeof mockApi>;

/** A 1x1 PNG, and a 4-sample mono WAV: the smallest payloads the console's media checks accept. */
export const PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
export const WAV_BYTES = (() => {
  const data = new Uint8Array(8);
  const head = new Uint8Array(44);
  const view = new DataView(head.buffer);
  head.set([..."RIFF"].map((c) => c.charCodeAt(0)), 0);
  view.setUint32(4, 36 + data.length, true);
  head.set([..."WAVEfmt "].map((c) => c.charCodeAt(0)), 8);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 24000, true);
  view.setUint32(28, 48000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  head.set([..."data"].map((c) => c.charCodeAt(0)), 36);
  view.setUint32(40, data.length, true);
  return new Uint8Array([...head, ...data]);
})();
export const WAV_BASE64 = btoa(String.fromCharCode(...WAV_BYTES));

export const chatModel = { id: "m/chat", capabilities: ["chat", "reasoning", "vision"], loaded: true, state: "ready", context_length: 8192, meta: { supports_thinking: true } };
export const imageModel = { id: "m/flux", capabilities: ["image"], loaded: false, meta: { architecture: "flux2" } };
export const kreaModel = { id: "m/krea", capabilities: ["image"], loaded: false, meta: { architecture: "krea" } };
export const speechModel = { id: "m/kokoro", capabilities: ["audio"], loaded: false, meta: { architecture: "kokoro" } };
export const musicModel = { id: "m/ace", capabilities: ["audio", "music"], loaded: false, meta: { architecture: "acestep" } };
export const soundModel = { id: "m/sat", capabilities: ["audio"], loaded: false, meta: { architecture: "stable_audio3" } };
export const videoModel = { id: "m/ltx", capabilities: ["video"], loaded: false, meta: { architecture: "AudioVideo" } };
export const otherModel = { id: "m/other", capabilities: ["chat"], loaded: false, context_length: 4096, meta: {} };

/** A fetch that speaks the mlx-serve routes the console uses; replies to /v1/chat/completions come from the script. */
export function mockApi(models: object[] = [chatModel, otherModel]) {
  const requests: { path: string; method: string; body: any }[] = [];
  const replies: Reply[] = [];
  /** A scripted failure for the next media request (any media route), else it succeeds. */
  const failures: { status: number; error: string }[] = [];
  /** What GET /metrics.json answers: a feed, or null for a server started without --metrics. */
  const state: { metrics: Metrics | null } = { metrics: null };
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  const sse = (frames: object[]) => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode(`data: ${JSON.stringify(frame)}\n\n`));
        controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
  const progress = (steps: number) => Array.from({ length: steps }, (_, i) => ({ type: "progress", step: i + 1, total: steps, stage: "working" }));
  const fetch = async (input: string | URL | Request, init: RequestInit = {}) => {
    const path = new URL(String(input)).pathname;
    const body = typeof init.body === "string" ? JSON.parse(init.body) : init.body;
    requests.push({ path, method: init.method ?? "GET", body });
    if (path === "/health") return json({ status: "ok" });
    if (path === "/api/version") return json({ version: "9.9.9" });
    if (path === "/props") return json({ memory: { active_bytes: 2 * 1024 ** 3 } });
    if (path === "/v1/models") return json({ data: models });
    if (path === "/metrics.json") return state.metrics ? json(state.metrics) : json({ error: { message: "metrics disabled" } }, 503);
    if (path === "/v1/images/generations" || path.startsWith("/v1/audio/")) {
      const failure = failures.shift();
      if (failure) return json({ error: { message: failure.error } }, failure.status);
      if (path === "/v1/images/generations") return sse([...progress(2), { type: "complete", data: [{ b64_json: PNG_1X1 }] }]);
      if (body?.stream === false) return new Response(WAV_BYTES, { status: 200, headers: { "content-type": "audio/wav" } });
      return sse([...progress(2), { type: "complete", data: WAV_BASE64 }]);
    }
    if (path === "/v1/chat/completions") {
      const reply = replies.shift() ?? { content: "ok" };
      if (reply.status) return json({ error: { message: reply.error ?? "failed" } }, reply.status);
      const encoder = new TextEncoder();
      const frame = (delta: object, finish: string | null = null) => encoder.encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`);
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          if (reply.reasoning) controller.enqueue(frame({ reasoning_content: reply.reasoning }));
          for (const piece of (reply.content ?? "").match(/.{1,5}/gs) ?? []) controller.enqueue(frame({ content: piece }));
          await reply.hold;
          controller.enqueue(frame({}, "stop"));
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 }, timings: { predicted_ms: 500 } })}\n\ndata: [DONE]\n\n`));
          controller.close();
        },
      });
      return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
    }
    return json({ error: { message: "not found " + path } }, 404);
  };
  return { fetch: fetch as typeof globalThis.fetch, requests, replies, failures, state, models: models as unknown as Model[] };
}
