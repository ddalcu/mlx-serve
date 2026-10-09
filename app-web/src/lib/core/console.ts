import { N } from "../i18n/i18n";
import type { Model } from "./models";
function apiPrefix(pathname: string) {
  let p = String(pathname || "");
  if (!p.startsWith("/")) return "";
  while (p.length > 1 && p.endsWith("/")) p = p.slice(0, -1);
  const slash = p.lastIndexOf("/");
  if (p.indexOf(".", slash) >= 0) p = p.slice(0, slash);
  return p === "/" ? "" : p;
}
const pageServer = (location: Pick<Location, 'origin' | 'pathname'>) =>
  location.origin + apiPrefix(location.pathname);
/**
 * The `?api_key=` the page was opened with: the server accepts it for the page, the console sends it on.
 */
const pageApiKey = (search: string) =>
  new URL(search || "", "http://page").searchParams.get("api_key") || undefined;
const shellQuote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'";
function curlExample(base: string, models: Model[]) {
  const model =
    models.find((m) => m.capabilities.includes("chat"))?.id ??
    "YOUR_CHAT_MODEL";
  return `curl ${shellQuote(base + "/v1/chat/completions")} \\\n  -H 'Content-Type: application/json' \\\n  -d ${shellQuote(JSON.stringify({ model, messages: [{ role: "user", content: "Hello!" }], stream: true }))}`;
}

const apiReference = [
  { method: "GET", path: "/", description: N("MLX Serve Studio web console") },
  {
    method: "POST",
    path: "/v1/chat/completions",
    description: N(
      "Streaming and non-streaming \u00b7 tool calling \u00b7 JSON mode \u00b7 vision (when supported)",
    ),
  },
  {
    method: "POST",
    path: "/v1/completions",
    description: N("Legacy text completions"),
  },
  {
    method: "POST",
    path: "/v1/responses",
    description: N(
      "Stateful responses with tool calling \u00b7 stream/non-stream \u00b7 vision",
    ),
  },
  {
    method: "POST",
    path: "/v1/responses/compact",
    description: N("Compact a conversation into a round-trippable opaque blob"),
  },
  {
    method: "GET",
    path: "/v1/responses/{id}",
    description: N("Retrieve a stored response envelope"),
  },
  {
    method: "DELETE",
    path: "/v1/responses/{id}",
    description: N("Delete a stored response"),
  },
  {
    method: "WS",
    path: "/v1/responses",
    description: N(
      "WebSocket transport \u00b7 per-connection store-false cache \u00b7 sequential turns",
    ),
  },
  {
    method: "POST",
    path: "/v1/messages",
    description: N(
      "Claude SDK / Claude Code compatible \u00b7 stream & non-stream \u00b7 tool use \u00b7 thinking blocks",
    ),
  },
  {
    method: "POST",
    path: "/api/chat",
    description: N(
      "Ollama chat \u00b7 NDJSON stream (default on) \u00b7 tool calls \u00b7 images \u00b7 think",
    ),
  },
  {
    method: "POST",
    path: "/api/generate",
    description: N("Ollama completion \u00b7 templated or raw"),
  },
  {
    method: "GET",
    path: "/api/tags",
    description: N("List local models (ollama list)"),
  },
  {
    method: "POST",
    path: "/api/show",
    description: N("Model details, template and parameters"),
  },
  {
    method: "GET",
    path: "/api/ps",
    description: N("Models currently resident in memory"),
  },
  {
    method: "POST",
    path: "/api/pull",
    description: N("Download a model from Hugging Face \u00b7 NDJSON progress"),
  },
  {
    method: "GET",
    path: "/api/version",
    description: N(
      "Version string (clients probe this to detect an Ollama server)",
    ),
  },
  {
    method: "POST",
    path: "/api/embed",
    description: N("Embeddings, current shape (input string or array)"),
  },
  {
    method: "POST",
    path: "/api/embeddings",
    description: N("Embeddings, legacy shape (prompt)"),
  },
  {
    method: "POST",
    path: "/v1/embeddings",
    description: N("Vector embeddings (encoder-only models)"),
  },
  {
    method: "POST",
    path: "/v1/decisions",
    description: N(
      "Laya, Kev and Clef typed decisions \u00b7 choice / score / noul questions over a JSON state",
    ),
  },
  {
    method: "POST",
    path: "/v1/systemone",
    description: N("Alias for /v1/decisions"),
  },
  {
    method: "POST",
    path: "/tokenize",
    description: N("Tokenize a string"),
  },
  {
    method: "POST",
    path: "/detokenize",
    description: N("Detokenize an id sequence"),
  },
  {
    method: "POST",
    path: "/v1/images/generations",
    description: N(
      "FLUX.2, Krea & Mage-Flow text-to-image \u00b7 img2img + instruction edit \u00b7 runtime LoRA \u00b7 base64 PNG",
    ),
  },
  {
    method: "POST",
    path: "/v1/images/edits",
    description: N(
      "OpenAI-compatible image editing \u00b7 multipart form \u00b7 one or more reference images + an instruction",
    ),
  },
  {
    method: "POST",
    path: "/v1/audio/speech",
    description: N(
      "Qwen3-TTS (zero-shot voice cloning) or Kokoro (54 blendable voices) \u00b7 WAV",
    ),
  },
  {
    method: "POST",
    path: "/v1/audio/music-generations",
    description: N("ACE-Step text-to-music \u00b7 48 kHz stereo WAV"),
  },
  {
    method: "POST",
    path: "/v1/audio/sound-generations",
    description: N(
      "Stable Audio 3 text-to-audio \u00b7 sound effects \u00b7 44.1 kHz stereo WAV",
    ),
  },
  {
    method: "POST",
    path: "/v1/video/generations",
    description: N(
      "LTX-Video or MiniMax-H3 \u00b7 text / image / audio \u2192 video with its own soundtrack \u00b7 frames + PCM",
    ),
  },
  {
    method: "POST",
    path: "/v1/3d/generations",
    description: N(
      "Hunyuan3D-2.1 \u00b7 one photo \u2192 GLB mesh \u00b7 optional PBR texturing",
    ),
  },
  {
    method: "POST",
    path: "/v1/load-model",
    description: N(
      "Load a discovered model, or register + load one by absolute path",
    ),
  },
  {
    method: "POST",
    path: "/v1/unload-model",
    description: N("Free a model's memory now"),
  },
  {
    method: "POST",
    path: "/v1/models/rescan",
    description: N("Pick up models added to the model folders since startup"),
  },
  {
    method: "GET",
    path: "/v1/providers",
    description: N(
      "Configured upstream chat providers (~/.mlx-serve/providers.json) and whether each answered its last probe",
    ),
  },
  {
    method: "POST",
    path: "/v1/providers/reload",
    description: N("Re-read providers.json and re-probe now"),
  },
  {
    method: "GET",
    path: "/v1/models",
    description: N(
      "OpenAI models list (id, capabilities, context length) \u2014 this console's model picker",
    ),
  },
  {
    method: "GET",
    path: "/props",
    description: N("llama.cpp-style server props (chat template, memory)"),
  },
  {
    method: "GET",
    path: "/health",
    description: N("Liveness probe"),
  },
  {
    method: "GET",
    path: "/metrics",
    description: N(
      "Prometheus metrics, text exposition format (enable with --metrics)",
    ),
  },
  {
    method: "GET",
    path: "/metrics.json",
    description: N(
      "Metrics as JSON \u2014 drives the Monitor panel (enable with --metrics)",
    ),
  },
];

export { apiPrefix, pageServer, pageApiKey, curlExample, apiReference };
