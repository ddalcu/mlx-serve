import { t } from "../i18n/i18n";
import { Client, StudioError } from "./client";

import { imageData, mediaPayload, pngBlob } from "./media";
import type { MediaOptions } from "./media";
import type { MediaResult } from "./media";
export type ImageRequest = Record<string, unknown> & { model?: string; prompt: string; size?: string; steps?: number; transparent?: boolean; };

async function generateImage(client: Client, request: ImageRequest, options: MediaOptions = {}): Promise<MediaResult> {
  const stream = options.stream ?? true;
  const result = await mediaPayload(
    client,
    "/v1/images/generations",
    { ...request, response_format: "b64_json", stream },
    { ...options, stream },
  );
  return {
    blob: pngBlob(imageData(result.payload)),
    elapsedMs: result.elapsedMs,
    wireBytes: result.wireBytes,
  };
}
function buildEditRequest(request: ImageRequest, images: Blob[], stream = false): FormData {
  if (!images.length)
    throw new StudioError(
      "protocol",
      t("Choose at least one reference image."),
    );
  const form = new FormData();
  for (const [key, value] of Object.entries({
    ...request,
    response_format: "b64_json",
    stream,
  }))
    if (value !== undefined)
      form.append(
        key,
        typeof value === "string" ? value : JSON.stringify(value),
      );
  for (const [index, blob] of images.entries())
    form.append(
      "image[]",
      blob,
      `reference-${index}.${blob.type === "image/jpeg" ? "jpg" : "png"}`,
    );
  return form;
}
async function editImage(client: Client, request: ImageRequest, images: Blob[], options: MediaOptions = {}): Promise<MediaResult> {
  const stream = options.stream ?? false,
    result = await mediaPayload(
      client,
      "/v1/images/edits",
      buildEditRequest(request, images, stream),
      { ...options, stream },
    );
  return {
    blob: pngBlob(imageData(result.payload)),
    elapsedMs: result.elapsedMs,
    wireBytes: result.wireBytes,
  };
}

export { generateImage, buildEditRequest, editImage };
