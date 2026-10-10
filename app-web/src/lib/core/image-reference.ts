import { t } from "../i18n/i18n";

export type Reference = { name: string; base64: string; width: number; height: number };

/** Decode and normalize orientation/formats to PNG before sending raw base64 to the server. */
export async function imageReference(file: File | Blob, name = "image.png"): Promise<Reference> {
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw Error(t("Choose a PNG, JPEG or WebP image."));
  if (file.size > 10 * 1024 * 1024) throw Error(t("Each reference must be at most 10 MB."));
  const bitmap = await createImageBitmap(file);
  try {
    if (bitmap.width * bitmap.height > 40_000_000) throw Error(t("Each reference must be at most 40 megapixels."));
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    if (!context) throw Error(t("Image decoding unavailable."));
    context.drawImage(bitmap, 0, 0);
    const base64 = canvas.toDataURL("image/png").split(",")[1]!;
    if (base64.length > 14 * 1024 * 1024) throw Error(t("Decoded reference is too large. Choose a smaller image."));
    return { name, base64, width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}
