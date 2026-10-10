import { t, th } from "../i18n/i18n";
import { decodeBase64 } from "./media";
import { newId } from "./id";
import { cleanToolRounds } from "./tool-loop";
import { normalizeBaseUrl } from "./client";
import type { LibraryItem } from "./library";
import type { Library } from "./library";
import type { Session } from "../state/chat-state.svelte";
const archiveLimit = 256 * 1024 * 1024;
const types = ["chat", "image", "speech", "music", "sound", "video"];
function object(value: unknown): Record<string, any> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error(t("Invalid archive object."));
  return value;
}
function text(value: unknown, limit: number = 2_000_000) {
  if (typeof value !== "string" || value.length > limit)
    throw new Error(t("Invalid or oversized archive text."));
  return value;
}
function number(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0)
    throw new Error(t("Invalid archive number."));
  return value;
}
/**
 * Only explicit, known fields survive. Never retain imported DOM identifiers or a pending run.
 */
function cleanSession(value: unknown, id: string, mediaIds?: Map<string, string>): Session {
  const s = object(value),
    settings = object(s.settings);
  if (
    s.version !== 1 ||
    !Array.isArray(s.messages) ||
    s.messages.length > 10000
  )
    throw new Error(t("Unsupported chat format."));
  const temperature =
    settings.temperature === null ? null : number(settings.temperature);
  const maxTokens =
    settings.maxTokens === null ? null : number(settings.maxTokens);
  if (
    (temperature !== null && temperature > 2) ||
    (maxTokens !== null && (!Number.isSafeInteger(maxTokens) || maxTokens < 1))
  )
    throw new Error(t("Invalid chat settings."));
  if (
    typeof settings.thinking !== "boolean" ||
    ![null, true, false].includes(settings.mtp)
  )
    throw new Error(t("Invalid chat settings."));
  return {
    version: 1,
    id,
    title: text(s.title, 10000),
    server: normalizeBaseUrl(text(s.server, 4096)),
    model: text(s.model, 4096),
    createdAt: number(s.createdAt),
    updatedAt: number(s.updatedAt),
    draft: text(s.draft),
    settings: {
      system: text(settings.system),
      temperature,
      maxTokens,
      thinking: settings.thinking,
      mtp: settings.mtp,
      toolsEnabled: settings.toolsEnabled === true,
    },
    messages: s.messages.map((value) => {
      const m = object(value);
      if (!["user", "assistant"].includes(m.role))
        throw new Error(t("Invalid chat role."));
      const images = m.images === undefined ? undefined : m.images;
      if (
        images !== undefined &&
        (!Array.isArray(images) ||
          images.length > 32 ||
          !images.every(
            (x) =>
              typeof x === "string" &&
              /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(
                x,
              ),
          ))
      )
        throw new Error(t("Chat images must be embedded PNG, JPEG or WebP."));
      const toolRounds =
        m.toolRounds === undefined ? undefined : cleanToolRounds(m.toolRounds);
      if (mediaIds)
        for (const round of toolRounds ?? [])
          for (const call of round.calls)
            if (call.mediaId) {
              call.mediaId = mediaIds.get(call.mediaId);
              if (!call.mediaId) delete call.mediaType;
            }
      return {
        ...(toolRounds ? { toolRounds } : {}),
        id: newId(),
        role: m.role,
        text: text(m.text),
        createdAt: number(m.createdAt),
        ...(images ? { images } : {}),
        ...(m.imagePurpose === "edit" ? { imagePurpose: "edit" } : {}),
        ...(m.thinking !== undefined ? { thinking: text(m.thinking) } : {}),
        ...(m.thinkingSeconds !== undefined
          ? { thinkingSeconds: number(m.thinkingSeconds) }
          : {}),
        ...(m.tokensPerSecond != null
          ? { tokensPerSecond: number(m.tokensPerSecond) }
          : {}),
        ...(m.error !== undefined ? { error: text(m.error) } : {}),
        ...(m.usage
          ? {
              usage: Object.fromEntries(
                Object.entries(object(m.usage))
                  .filter(([key]) =>
                    [
                      "prompt_tokens",
                      "completion_tokens",
                      "total_tokens",
                    ].includes(key),
                  )
                  .map(([key, value]) => [key, number(value)]),
              ),
            }
          : {}),
        status:
          m.status === "streaming"
            ? "stopped"
            : ["complete", "error", "stopped"].includes(m.status)
              ? m.status
              : "complete",
      };
    }),
  };
}
function chatMarkdown(session: Session) {
  return `# ${session.title}\n\n${session.messages
    .map(
      (m) =>
        `## ${m.role === "user" ? "You" : "Assistant"}\n\n${
          m.thinking
            ? `<details><summary>${th("Thinking")}</summary>

${m.thinking}

</details>

`
            : ""
        }${
          m.toolRounds
            ?.map((r) =>
              r.calls
                .map(
                  (c) =>
                    `\n\nTool: ${c.name}\n\n${(c.result || c.status)
                      .split("\n")
                      .map((line) => "> " + line)
                      .join(
                        "\n",
                      )}\n${c.mediaId ? "Media is saved separately in the Library." : ""}`,
                )
                .join("\n"),
            )
            .join("\n") || ""
        }${m.text}${m.images?.map((x) => `\n\n![Attachment](${x})`).join("") || ""}`,
    )
    .join("\n\n")}\n`;
}
async function archiveLibrary(library: Library) {
  const rows = await library.list(),
    items = [];
  let bytes = 0;
  for (const row of rows) {
    const item = await library.get(row.id);
    if (!item) continue;
    bytes += item.blob.size;
    if (bytes > archiveLimit * 0.7)
      throw new Error(
        t("Archive exceeds 256 MiB. Download large media separately."),
      );
    const base = {
      sourceId: item.id,
      type: item.type,
      model: item.model,
      server: item.server,
      createdAt: item.createdAt,
      prompt: item.prompt || "",
    };
    if (item.type === "chat")
      items.push({
        ...base,
        session: cleanSession(JSON.parse(await item.blob.text()), item.id),
      });
    else {
      const data = new Uint8Array(await item.blob.arrayBuffer());
      const parts = [];
      // Whole three-byte groups allow concatenating independently encoded chunks.
      for (let i = 0; i < data.length; i += 8190)
        parts.push(btoa(String.fromCharCode(...data.subarray(i, i + 8190))));
      items.push({ ...base, mime: item.blob.type, data: parts.join("") });
    }
  }
  const blob = new Blob(
    [JSON.stringify({ format: "mlx-serve-studio", version: 1, items })],
    { type: "application/json" },
  );
  if (blob.size > archiveLimit)
    throw new Error(
      t("Archive exceeds 256 MiB. Download large media separately."),
    );
  return blob;
}
/** Validate everything before one atomic append transaction. */
async function importArchive(library: Library, blob: Blob) {
  if (blob.size > archiveLimit) throw new Error(t("Archive exceeds 256 MiB."));
  const archive = object(JSON.parse(await blob.text()));
  if (
    archive.format !== "mlx-serve-studio" ||
    archive.version !== 1 ||
    !Array.isArray(archive.items) ||
    archive.items.length > 10000
  )
    throw new Error(t("Choose a Studio version 1 JSON archive."));
  const mediaIds = new Map(),
    sourceIds = new Set();
  for (const value of archive.items) {
    const row = object(value);
    if (row.sourceId !== undefined) {
      const old = text(row.sourceId, 256);
      if (sourceIds.has(old)) throw Error(t("Duplicate archive item ID."));
      sourceIds.add(old);
      if (row.type !== "chat") mediaIds.set(old, newId());
    }
  }
  const items: LibraryItem[] = archive.items.map((value) => {
    const row = object(value),
      id = mediaIds.get(row.sourceId) ?? newId();
    if (!types.includes(row.type)) throw new Error(t("Unknown library type."));
    const server = normalizeBaseUrl(text(row.server, 4096)),
      model = text(row.model, 4096),
      createdAt = number(row.createdAt),
      prompt = text(row.prompt ?? "", 100000);
    let data;
    if (row.type === "chat") {
      const session = cleanSession(row.session, id, mediaIds);
      session.server = server;
      session.model = model;
      data = new Blob([JSON.stringify(session)], { type: "application/json" });
    } else {
      const mime = text(row.mime, 100),
        encoded = text(row.data, archiveLimit);
      const allowed =
        row.type === "image"
          ? ["image/png", "image/jpeg", "image/webp"]
          : row.type === "video"
            ? ["video/mp4", "video/webm"]
            : ["audio/wav", "audio/mpeg", "audio/ogg", "audio/webm"];
      if (!allowed.includes(mime))
        throw new Error(t("Invalid media type or base64 data."));
      data = new Blob([decodeBase64(encoded, archiveLimit)], { type: mime });
    }
    return { id, type: row.type, server, model, createdAt, prompt, blob: data };
  });
  await library.append(items);
  return items.map((x) => x.id);
}

export { archiveLimit, cleanSession, chatMarkdown, archiveLibrary, importArchive };
