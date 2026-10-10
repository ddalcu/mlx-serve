import { t, N } from "../i18n/i18n";
import { generateImage } from "../core/images";
import { fluxResolutions, kreaResolutions, mageFlowResolutions, qwenImageResolutions } from "./image-presets";
import type { ImageRequest } from "../core/images";
import type { LibraryInput } from "../core/library";
import type { Client } from "../core/client";
import type { Library } from "../core/library";
import type { Model } from "../core/models";
import { MediaRun } from "./media-run.svelte";
export type Canvas = { width: number; height: number; label: string; };
export type Profile = { family: string; edit: boolean; variation: boolean; guidance: boolean; transparent: boolean; weights: number; fixed: boolean; quality: number[]; defaultSteps: number; alignment: number; min: number; max: number; resolutions: Canvas[]; };
/**
 * Pinned Swift presets win over broad family metadata (notably undistilled base Klein).
 * Unknown architectures retain basic generation only, never guessed controls.
 */
function imageProfile(model: Model | undefined): Profile | null {
  if (!model?.capabilities.includes("image")) return null;
  const arch = model.architecture ?? "";
  const id = model.id.split("@")[0].toLowerCase();
  const base = [
    "aitrader/flux2-klein-base-9b-mlx-4bit",
    "mflux/flux2-klein-9b-base-q4",
  ].includes(id);
  const family =
    base || arch.startsWith("flux2")
      ? "flux"
      : arch.startsWith("krea")
        ? "krea"
        : arch.startsWith("mage_flow") || arch === "mageflow"
          ? "mage"
          : arch.startsWith("qwen_image")
            ? "qwen"
            : "";
  if (!family) return null;
  const mage = family === "mage",
    qwen = family === "qwen",
    flux = family === "flux";
  const quality = mage
    ? [4, 4, 4, 4]
    : base || qwen
      ? [20, 30, 40, 50]
      : flux
        ? [4, 8, 12, 20]
        : [6, 8, 12, 16];
  return {
    family,
    edit: flux || (mage && /mage-?flow-edit/.test(id)),
    variation: !mage,
    guidance: base || qwen,
    transparent: qwen,
    weights: flux ? 3 : family === "krea" ? 12 : 0,
    fixed: mage,
    quality,
    defaultSteps: quality[qwen ? 2 : 1],
    alignment: flux ? 32 : 16,
    min: 256,
    max: flux ? 1536 : 2048,
    resolutions: flux
      ? fluxResolutions
      : mage
        ? mageFlowResolutions
        : qwen
          ? qwenImageResolutions
          : kreaResolutions,
  };
}
function imageDefaults(model: Model | undefined) {
  return {
    model: model?.id ?? "",
    prompt: "",
    width: "1024",
    height: "1024",
    steps: imageProfile(model)?.defaultSteps ?? 8,
    seed: "",
    mode: "edit",
    strength: 0.6,
    gain: 1,
    weights: "",
    guidance: 1,
    negative: "",
    transparent: false,
    advanced: false,
  };
}
export type ImageDraft = ReturnType<typeof imageDefaults>;
function resolveCanvas(p: Profile, width: string, height: string) {
  const w = Number(width),
    h = Number(height);
  if (![w, h].every((v) => Number.isInteger(v) && v > 0))
    throw Error(t("Width and height must be whole numbers above zero."));
  if (![w, h].every((v) => v >= p.min && v <= p.max))
    throw Error(
      t("This model samples between %@ and %@ px per side.", [p.min, p.max]),
    );
  const sw = Math.ceil(w / p.alignment) * p.alignment,
    sh = Math.ceil(h / p.alignment) * p.alignment;
  return {
    width: sw,
    height: sh,
    hint:
      sw !== w || sh !== h
        ? t("Rounded to %@ × %@ — this model samples in steps of %@ px.", [
            sw,
            sh,
            p.alignment,
          ])
        : "",
  };
}
/**
 * AspectCanvases.swift: nearest-grid faithful candidates, spread by area; source first.
 */
function sourceCanvases(p: Profile, width: number, height: number) {
  if (!(width > 0 && height > 0)) return [];
  const ratio = width / height;
  const candidates = [];
  for (let w = p.min; w <= p.max; w += p.alignment) {
    const h = Math.round(w / ratio / p.alignment) * p.alignment;
    if (h >= p.min && h <= p.max)
      candidates.push({
        width: w,
        height: h,
        name: "",
        deviation: Math.abs(w / h - ratio) / ratio,
      });
  }
  let faithful = candidates.filter((c) => c.deviation <= 0.03);
  if (!faithful.length)
    faithful = candidates.sort((a, b) => a.deviation - b.deviation).slice(0, 5);
  faithful.sort((a, b) => b.width * b.height - a.width * a.height);
  let picked = faithful;
  if (faithful.length > 5) {
    const first = faithful[0],
      last = faithful[faithful.length - 1];
    picked = [first, last];
    for (let i = 1; i < 4; i++) {
      const target =
        first.width * first.height +
        ((last.width * last.height - first.width * first.height) * i) / 4;
      picked.push(
        faithful
          .filter((c) => !picked.includes(c))
          .sort(
            (a, b) =>
              Math.abs(a.width * a.height - target) -
              Math.abs(b.width * b.height - target),
          )[0],
      );
    }
    picked.sort((a, b) => b.width * b.height - a.width * a.height);
  }
  const names = [
    [],
    [""],
    ["largest", "smallest"],
    ["largest", "medium", "smallest"],
    ["largest", "large", "small", "smallest"],
    ["largest", "large", "medium", "small", "smallest"],
  ][picked.length];
  picked = picked.map((c, i) => ({ ...c, name: names[i] }));
  const w = Math.round(width / p.alignment) * p.alignment,
    h = Math.round(height / p.alignment) * p.alignment;
  if ([w, h].every((v) => v >= p.min && v <= p.max))
    picked = [
      { width: w, height: h, name: "source size", deviation: 0 },
      ...picked.filter((c) => c.width !== w || c.height !== h),
    ];
  return picked;
}
const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
function bounded(value: number, min: number, max: number, label: string) {
  if (!Number.isFinite(value) || value < min || value > max)
    throw Error(t("%@ must be between %@ and %@.", [label, min, max]));
  return value;
}
/**
 * Swift ImageGenService.requestJson: all modes use generations JSON, not multipart edits.
 * Only active model controls reach the request; hidden stale draft values cannot leak.
 */
function buildImageRequest(model: Model, d: ImageDraft, refs: { base64: string; }[], random: () => number = randomSeed): ImageRequest {
  if (!model.capabilities.includes("image"))
    throw Error(t("Select an image model."));
  if (!d.prompt.trim()) throw Error(t("Prompt is empty."));
  const seed = d.seed.trim() === "" ? random() : Number(d.seed);
  if (!Number.isSafeInteger(seed) || seed < 0)
    throw Error(
      t("Seed must be a non-negative whole number, or empty for random."),
    );
  const body = { model: model.id, prompt: d.prompt, seed };
  const p = imageProfile(model);
  if (!p) return body;
  const editing = p.edit && (d.mode === "edit" || !p.variation);
  if (refs.length > 4)
    throw Error(t("Use at most four images (source plus three references)."));
   const request: ImageRequest = {
    ...body,
    steps: bounded(d.steps, 1, 50, t("Steps")),
  };
  if (!Number.isInteger(d.steps))
    throw Error(t("Steps must be a whole number."));
  if (!(editing && refs.length && d.width === "0" && d.height === "0")) {
    const size = resolveCanvas(p, d.width, d.height);
    request.size = `${size.width}x${size.height}`;
  }
  if (refs.length && (p.edit || p.variation)) {
    request.image = refs[0].base64;
    if (editing) {
      request.mode = "edit";
      if (refs.length > 1)
        request.ref_images = refs.slice(1).map((r) => r.base64);
    } else
      request.strength = bounded(d.strength, 0.1, 1, t("Variation strength"));
  }
  if (p.guidance) {
    const guidance = bounded(d.guidance, 1, 20, t("Guidance scale"));
    if (guidance !== 1) request.guidance_scale = guidance;
    if (d.negative.trim()) request.negative_prompt = d.negative.trim();
  }
  if (p.weights) {
    const gain = bounded(d.gain, 0, 4, t("Global gain"));
    if (gain !== 1) request.cond_gain = gain;
    if (d.weights.trim()) {
      const weights = d.weights
        .trim()
        .split(/[,\s]+/)
        .map(Number);
      if (weights.length !== p.weights || !weights.every(Number.isFinite))
        throw Error(
          t("Needs exactly %@ finite numbers — one per tapped encoder layer.", [
            p.weights,
          ]),
        );
      request.cond_weights = weights;
    }
  }
  if (p.transparent && d.transparent) request.transparent = true;
  return request;
}
class ImageController extends MediaRun<LibraryInput & { id?: string; elapsedMs: number }> {
  protected readonly busy = t("An image request is already running.");
  protected readonly saveFailed = N("Not saved in this browser: %@. Download the image or retry saving.");
  request: typeof generateImage;

  constructor(library: Pick<Library, "add">, generate: typeof generateImage = generateImage) {
    super(library);
    this.request = generate;
  }

  async generate(client: Client, body: ImageRequest, server: string) {
    const run = this.begin(Number(body.steps) || 0);
    try {
      const result = await this.request(client, body, { signal: run.signal, onProgress: this.progress(run) });
      if (this.run !== run || run.signal.aborted) return;
      await this.complete({ type: "image", model: body.model ?? "", server, prompt: body.prompt, blob: result.blob, createdAt: Date.now(), elapsedMs: result.elapsedMs });
    } catch (e) {
      this.fail(run, e);
    } finally {
      this.end(run);
    }
  }
}

/** PromptMarkerInsert.swift, using the browser's UTF-16 selection offsets. */
function insertImageMarker(marker: string, text: string, start: number = text.length, end: number = start) {
  const lo = Math.max(0, Math.min(start, text.length)),
    hi = Math.max(lo, Math.min(end, text.length));
  const before = text.slice(0, lo).replace(/[ \t]+$/, ""),
    after = text.slice(hi).replace(/^[ \t]+/, "");
  const prefix = before + (before && !before.endsWith("\n") ? " " : "") + marker + (after && !/^[\n,.;:!?\)\]\}]/.test(after) ? " " : "");
  return { text: prefix + after, cursor: prefix.length };
}

export { imageProfile, imageDefaults, resolveCanvas, sourceCanvases, randomSeed, buildImageRequest, ImageController, insertImageMarker };
