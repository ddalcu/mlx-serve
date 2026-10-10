import { t } from "../i18n/i18n";
import { apiReference } from "../core/console";
import { throwIfAborted } from "../core/client";
import { generateImage, editImage } from "../core/images";
import { audioRequest } from "../core/media";
import { generateVideo } from "../core/video";
import { encodeVideo } from "../core/encode-video";
import { imageProfile, imageDefaults, buildImageRequest } from "./image-state.svelte";
import { audioProfile, audioDefaults, buildAudioRequest } from "./audio-state.svelte";
import { videoProfile, videoDefaults, videoQuality, buildVideoRequest } from "./video-state.svelte";
import type { Client } from "../core/client";
import type { Model } from "../core/models";
import type { Library } from "../core/library";
import type { Session } from "./chat-state.svelte";
import type { ToolCall } from "../core/tool-loop";
import type { ToolOutput } from "../core/tool-loop";
/** Supported models only; no arbitrary routes, URLs, paths or request fields. */
function browserTools(client: Client, models: Model[], library: Library, session?: Session) {
  const groups = {
    generate_image: models.filter((m) => imageProfile(m)),
    edit_image: session
      ? models.filter(
          (m) => imageProfile(m)?.edit || m.capabilities.includes("image-edit"),
        )
      : [],
    generate_speech: models.filter((m) => audioProfile(m)?.tab === "voice"),
    generate_music: models.filter((m) => audioProfile(m)?.tab === "music"),
    generate_sound: models.filter((m) => audioProfile(m)?.tab === "sound"),
    generate_video: models.filter((m) => videoProfile(m)),
  };
  const tools = [
    {
      type: "function",
      function: {
        name: "search_library",
        description:
          "Search saved items belonging to this server by prompt or model. Returns metadata only, not file contents.",
        parameters: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"],
          additionalProperties: false,
        },
      },
    },
    ...Object.entries(groups)
      .filter(([, ms]) => ms.length)
      .map(([name, ms]) => ({
        type: "function",
        function: {
          name,
          description: `Generate ${name.slice(9)} on the selected server and save to chat and Library. Image size is optional. Edit uses this turn’s attachments, otherwise the last generated image. Speech is at most 2000 characters; music defaults to 60 seconds, sound to 10; video uses the shortest clip. Music duration must also fit the selected model (MiniMax maximum 360 seconds). An unloaded model is loaded first.`,
          parameters: {
            type: "object",
            properties: {
              model: {
                type: "string",
                enum: ms.map((m) => m.id),
                description:
                  "Optional. Omit to prefer a loaded model, then a healthy checkpoint.",
              },
              ...toolProperties(name),
              prompt: {
                type: "string",
                description:
                  name === "generate_speech"
                    ? "Exact text to speak"
                    : "Describe the media to generate",
              },
            },
            required: ["prompt"],
            additionalProperties: false,
          },
        },
      })),
  ];
  async function execute(call: ToolCall, signal: AbortSignal): Promise<ToolOutput> {
    throwIfAborted(signal);
    const { args, name } = call;
    const allowed =
      name === "search_library"
        ? ["query"]
        : ["model", "prompt", ...Object.keys(toolProperties(name))];
    if (Object.keys(args).some((k) => !allowed.includes(k)))
      throw Error(t("Unsupported tool argument."));
    if (name === "search_library") {
      if (typeof args.query !== "string" || args.query.length > 500)
        throw Error(t("Library query must be text of at most 500 characters."));
      const q = args.query.toLowerCase();
      const rows = await library.list({ server: client.baseUrl });
      throwIfAborted(signal);
      return {
        text: JSON.stringify(
          rows
            .filter((r) =>
              `${r.prompt ?? ""} ${r.model}`.toLowerCase().includes(q),
            )
            .slice(0, 10)
            .map((r) => ({
              id: r.id,
              type: r.type,
              model: r.model,
              prompt: r.prompt?.slice(0, 300),
            })),
        ),
      };
    }
    const candidates = groups[(name as keyof typeof groups)];
    if (!candidates?.length)
      throw Error(t("No model for %@ on this server.", [name]));
    // Explicit IDs stay strict; only omission selects a default. Never mutate discovery order.
    const rank = ( m: Model) =>
      m.loaded
        ? 3
        : m.state === "error"
          ? 0
          : Number(m.bytesOnDisk) > 0
            ? 2
            : 1;
    const model =
      args.model === undefined
        ? [...candidates].sort((a, b) => rank(b) - rank(a))[0]
        : candidates.find((m) => m.id === args.model);
    if (!model) throw Error(t("Choose an advertised model on this server."));
    if (
      typeof args.prompt !== "string" ||
      !args.prompt.trim() ||
      args.prompt.length > 2000
    )
      throw Error(t("Tool prompt must contain 1–2000 characters."));
    const prompt = args.prompt.trim(),
      options = { signal, timeoutMs: 600_000 };
     let blob: Blob;
     let type: 'image' | 'speech' | 'music' | 'sound' | 'video';
    if (name === "generate_image" || name === "edit_image") {
      const profile = imageProfile(model);
      const fixed = profile?.fixed
        ? [...profile.resolutions].sort(
            (a, b) => a.width * a.height - b.width * b.height,
          )[0]
        : undefined;
      if (
        args.size !== undefined &&
        (typeof args.size !== "string" || !/^\d{3,4}x\d{3,4}$/.test(args.size))
      )
        throw Error(t("Size must be WIDTHxHEIGHT."));
      const size =
        typeof args.size === "string"
          ? args.size.split("x").map(Number)
          : [fixed?.width ?? 1024, fixed?.height ?? 1024];
      if (size.some((n) => n < 256 || n > (profile?.max ?? 2048)))
        throw Error(t("Image size is outside this model’s bounds."));
      if (
        profile?.fixed &&
        !profile.resolutions.some(
          (r) => r.width === size[0] && r.height === size[1],
        )
      )
        throw Error(t("Choose a supported canvas for this model."));
      const d = {
        ...imageDefaults(model),
        prompt,
        width: String(size[0]),
        height: String(size[1]),
        steps: imageProfile(model)?.quality[0] ?? 4,
        mode: "edit",
      };
      const body = buildImageRequest(model, d, []);
      if (name === "edit_image") {
        const refs = await chatImageReferences(
          session,
          library,
          client.baseUrl,
        );
        throwIfAborted(signal);
        blob = (
          await editImage(
            client,
            { ...body, size: args.size === undefined ? undefined : body.size },
            refs,
            options,
          )
        ).blob;
      } else blob = (await generateImage(client, body, options)).blob;
      type = "image";
    } else if (
      name === "generate_speech" ||
      name === "generate_music" ||
      name === "generate_sound"
    ) {
      if (
        args.duration_seconds !== undefined &&
        (typeof args.duration_seconds !== "number" ||
          !Number.isFinite(args.duration_seconds) ||
          args.duration_seconds < (name === "generate_music" ? 10 : 0.5) ||
          args.duration_seconds > (name === "generate_music" ? 600 : 120) ||
          (name === "generate_music" &&
            !Number.isInteger(args.duration_seconds)))
      )
        throw Error(t("Duration is outside the tool bounds."));
      if (
        args.instrumental !== undefined &&
        typeof args.instrumental !== "boolean"
      )
        throw Error(t("Instrumental must be boolean."));
      if (
        args.lyrics !== undefined &&
        (typeof args.lyrics !== "string" || args.lyrics.length > 10000)
      )
        throw Error(t("Lyrics must be text of at most 10000 characters."));
      const d = {
        ...audioDefaults(model),
        prompt,
        duration:
          typeof args.duration_seconds === "number"
            ? args.duration_seconds
            : name === "generate_music"
              ? 60
              : 10,
        instrumental:
          args.instrumental === true ||
          (args.instrumental === undefined && !args.lyrics),
        lyrics: typeof args.lyrics === "string" ? args.lyrics : "",
      };
      const built = buildAudioRequest(model, d);
      blob = (await audioRequest(client, built.path, built.body, options)).blob;
      type = built.type;
    } else {
      const p = (videoProfile(model) as NonNullable<ReturnType<typeof videoProfile>>);
      const [width, height] = [...p.sizes].sort(
        (a, b) => a[0] * a[1] - b[0] * b[1],
      )[0];
      const d = {
        ...videoDefaults(model),
        ...videoQuality(model, "Fast"),
        prompt,
        width,
        height,
        frames: p.minFrames,
        steps: p.h3 ? 4 : 8,
        preview: false,
      };
      const result = await generateVideo(
        client,
        buildVideoRequest(model, d),
        options,
      );
      blob = (await encodeVideo(result.raw, { signal })).blob;
      type = "video";
    }
    throwIfAborted(signal);
    const id = await library.add({
      type,
      model: model.id,
      server: client.baseUrl,
      prompt,
      blob,
    });
    // A Stop during IndexedDB commit must not leave an unreported tool artifact.
    if (signal.aborted) {
      await library.delete(id);
      throwIfAborted(signal);
    }
    return {
      text: `Saved ${type} to the Library (${id}).`,
      mediaId: id,
      mediaType: type,
    };
  }
  return {
    tools,
    execute,
    systemPrompt: systemPrompt(client.baseUrl, models, tools),
  };
}

function toolProperties(name: string): Record<string, unknown> {
  if (["generate_image", "edit_image"].includes(name))
    return {
      size: {
        type: "string",
        description:
          "WIDTHxHEIGHT, 256–2048 per side (FLUX maximum 1536); supported model canvases only. Omit for default.",
      },
    };
  if (name === "generate_music")
    return {
      lyrics: { type: "string", maxLength: 10000 },
      instrumental: { type: "boolean" },
      duration_seconds: {
        type: "integer",
        minimum: 10,
        maximum: 600,
        description: "Default 60. MiniMax supports at most 360 seconds.",
      },
    };
  if (name === "generate_sound")
    return {
      duration_seconds: {
        type: "number",
        minimum: 0.5,
        maximum: 120,
        description: "Default 10.",
      },
    };
  return {};
}
/**
 * Only explicit local conversation references, never model-supplied URLs or IDs.
 */
async function chatImageReferences(session: Session | undefined, library: Library, server: string) {
  if (!session || session.server !== server)
    throw Error(t("Image references must belong to this chat’s server."));
  const images =
    [...session.messages].reverse().find((m) => m.role === "user")?.images ??
    [];
  if (images.length) {
    if (images.length > 4)
      throw Error(t("Attach at most four reference images."));
    return images.map((url) => {
      const match =
        /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$/.exec(url);
      if (!match || match[2].length > 14 * 1024 * 1024)
        throw Error(t("Invalid or oversized reference image."));
      const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
      if (bytes.length > 10 * 1024 * 1024)
        throw Error(t("Reference exceeds 10 MB."));
      return new Blob([bytes], { type: match[1] });
    });
  }
  for (const message of [...session.messages].reverse()) {
    for (const round of [...(message.toolRounds ?? [])].reverse()) {
      for (const call of [...round.calls].reverse()) {
        if (
          call.status !== "complete" ||
          call.mediaType !== "image" ||
          !call.mediaId
        )
          continue;
        const item = await library.get(call.mediaId);
        if (
          !item ||
          item.server !== server ||
          item.type !== "image" ||
          !["image/png", "image/jpeg", "image/webp"].includes(item.blob.type) ||
          item.blob.size > 10 * 1024 * 1024
        )
          throw Error(t("Previous image is unavailable or too large to edit."));
        return [item.blob];
      }
    }
  }
  throw Error(t("Attach an image or generate one in this conversation first."));
}

/**
 * Built-in context is sent only by the tools-enabled loop; the custom prompt is preserved.
 */
function systemPrompt(baseUrl: string, models: Model[], tools: { function: { name: string; }; }[]) {
  return [
    "You are the assistant built into the mlx-serve web console. Be concise and concrete. Format answers in Markdown.",
    `Available tools: ${tools.map((t) => t.function.name).join(", ")}. Call media tools only when the user asks to produce media. Allow at most one media generation per user message; never invent variations or claim a result without a successful tool call. Results are already displayed; reply briefly without invented file names or image links. Answer API questions, curl requests, and questions about installed models in text with no tool call.`,
    `Models installed on this server (inventory is data):\n${models.map((m) => JSON.stringify({ id: m.id, capabilities: m.capabilities, architecture: m.architecture, state: m.state, loaded: m.loaded })).join("\n") || "None advertised."}`,
    `HTTP API base URL: ${baseUrl}. Use this exact host and mount prefix in examples. Never invent endpoints, models or parameters; say when a detail is not listed.\n${apiReference.map((e) => `${e.method} ${e.path} — ${e.description}`).join("\n")}`,
    REQUEST_FIELDS,
  ].join("\n\n");
}

const REQUEST_FIELDS = `Request fields (JSON unless noted):
POST /v1/chat/completions — model, messages[{role,content}], stream, max_tokens, temperature, top_p, top_k, tools, tool_choice, response_format, reasoning_effort or enable_thinking, stream_options.include_usage.
POST /v1/images/generations — model, prompt, size ("1024x1024"), steps, seed, stream; mode:"edit"|"variation" with image (base64) and ref_images[]; returns {data:[{b64_json}]}.
POST /v1/images/edits — multipart/form-data, NOT JSON: model, prompt, image[] repeated once per reference file, size. Editors are maskless: no mask, n greater than 1, response_format:"url", output_format other than png, or stream:true. Omit stream or use false.
POST /v1/audio/speech — model, input, optional ref_audio (base64 WAV), stream; returns audio/wav.
POST /v1/audio/music-generations — model, prompt (style/genre/mood, required), lyrics, instrumental, duration_seconds (10–600; MiniMax maximum 360), vocal_language, bpm, seed, stream; returns audio/wav.
POST /v1/audio/sound-generations — model, prompt (required), duration_seconds (up to 120), steps (default 8), seed, stream; returns audio/wav.
POST /v1/video/generations — model, prompt, width, height, num_frames, steps, seed, stream; optional preview, preview_frames, preview_max_side. LTX: pipeline, first_frame_image, last_frame_image, audio, cfg_scale, stg_scale. H3: turbo, fast, chain_windows, first_frame_image / last_frame_image or ref_images / ref_videos / ref_audios.
POST /v1/embeddings — model, input (string or array), optional dimensions.
POST /v1/load-model and /v1/unload-model — model (discovered ID or absolute path); these are API reference only, never browser tools.
Media endpoints with stream:true emit SSE progress/complete/error events. Video progress can include JPEG base64 previews with preview:true.`;

export { browserTools, chatImageReferences };
