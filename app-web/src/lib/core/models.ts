import { record } from "./client";
export type Capability = "chat" |
  "vision" |
  "image" |
  "image-edit" |
  "video" |
  "speech" |
  "music" |
  "embeddings" |
  (string & {});

export type Model = { id: string; capabilities: Capability[]; contextLength?: number; engine?: string; architecture?: string; loaded?: boolean; state?: string; bytesOnDisk?: number; meta: Record<string, unknown>; };

function parseModels(value: unknown): Model[] {
  const data = record(value).data;
  if (!Array.isArray(data)) return [];
  return data.flatMap((item) => {
    const r = record(item);
    if (typeof r.id !== "string" || !r.id) return [];
    const meta = record(r.meta),
      capabilities = Array.isArray(r.capabilities)
        ? r.capabilities.filter(
            (v): v is string => typeof v === "string",
          )
        : [];
    const architecture =
      typeof meta.architecture === "string" ? meta.architecture : undefined;
    if (
      capabilities.includes("audio") &&
      ["kokoro", "qwen3_tts"].includes(architecture ?? "") &&
      !capabilities.includes("speech")
    )
      capabilities.push("speech");
    const length = ([r.context_length, r.max_model_len, meta.context_length].find(
        (v) => typeof v === "number" && Number.isSafeInteger(v) && v > 0,
      ) as number | undefined);
    return [
      {
        id: r.id,
        capabilities: [...new Set(capabilities)],
        contextLength: length,
        engine: typeof meta.engine === "string" ? meta.engine : undefined,
        architecture,
        bytesOnDisk:
          typeof r.bytes_on_disk === "number" ? r.bytes_on_disk : undefined,
        state: typeof r.state === "string" ? r.state : undefined,
        loaded:
          typeof r.loaded === "boolean"
            ? r.loaded
            : r.state === "ready"
              ? true
              : undefined,
        meta,
      },
    ];
  });
}

export { parseModels };
