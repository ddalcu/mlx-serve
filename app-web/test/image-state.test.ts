import { describe, expect, it } from "vitest";
import type { Model } from "../src/lib/core/models";
import { buildImageRequest, imageDefaults, imageProfile, insertImageMarker, resolveCanvas, sourceCanvases } from "../src/lib/state/image-state.svelte";

const model = (id: string, architecture: string): Model => ({ id, capabilities: ["image"], architecture, meta: {} });
const chat: Model = { id: "c", capabilities: ["chat"], meta: {} };
const flux = model("org/flux2-klein-4b", "flux2");
const krea = model("org/krea-2", "krea");
const mage = model("org/Mage-Flow-Turbo", "mage_flow");
const mageEdit = model("org/Mage-Flow-Edit-Turbo", "mage_flow");
const qwen = model("org/qwen-image", "qwen_image21");
const draft = (m: Model, patch: Record<string, unknown> = {}) => ({ ...imageDefaults(m), prompt: "a fox", seed: "7", ...patch });
const ref = { base64: "AAAA" };

describe("imageProfile", () => {
  it("knows each family's controls", () => {
    expect(imageProfile(flux)).toMatchObject({ family: "flux", edit: true, variation: true, guidance: false, weights: 3, alignment: 32, max: 1536 });
    expect(imageProfile(krea)).toMatchObject({ family: "krea", edit: false, variation: true, weights: 12, alignment: 16, max: 2048 });
    expect(imageProfile(mage)).toMatchObject({ family: "mage", edit: false, variation: false, fixed: true, quality: [4, 4, 4, 4] });
    expect(imageProfile(mageEdit)).toMatchObject({ family: "mage", edit: true });
    expect(imageProfile(qwen)).toMatchObject({ family: "qwen", guidance: true, transparent: true, defaultSteps: 40 });
  });

  it("pins the undistilled base Klein to real guidance whatever its architecture says", () => {
    expect(imageProfile(model("mflux/flux2-klein-9b-base-q4", "flux2"))).toMatchObject({ family: "flux", guidance: true, quality: [20, 30, 40, 50] });
  });

  it("offers nothing for a model that is not an image model or has an unknown architecture", () => {
    expect(imageProfile(undefined)).toBe(null);
    expect(imageProfile({ ...chat, architecture: "flux2" })).toBe(null);
    expect(imageProfile(model("x", "mystery"))).toBe(null);
  });
});

describe("resolveCanvas", () => {
  const p = imageProfile(flux)!;
  it("rounds up to the model's step and says so", () => {
    expect(resolveCanvas(p, "1000", "1000")).toMatchObject({ width: 1024, height: 1024 });
    expect(resolveCanvas(p, "1000", "1000").hint).toContain("1024 × 1024");
    expect(resolveCanvas(p, "1024", "512").hint).toBe("");
  });

  it("refuses sizes the model cannot sample", () => {
    expect(() => resolveCanvas(p, "100", "512")).toThrow(/between 256 and 1536/);
    expect(() => resolveCanvas(p, "512", "2000")).toThrow(/between 256 and 1536/);
    expect(() => resolveCanvas(p, "12.5", "512")).toThrow(/whole numbers/);
    expect(() => resolveCanvas(p, "", "512")).toThrow(/whole numbers/);
  });
});

describe("buildImageRequest", () => {
  it("sends a plain generation with only the fields the model has", () => {
    expect(buildImageRequest(krea, draft(krea), [])).toEqual({ model: "org/krea-2", prompt: "a fox", seed: 7, steps: 8, size: "1024x1024" });
  });

  it("draws a random seed when none is given, and refuses a bad one", () => {
    expect(buildImageRequest(krea, draft(krea, { seed: "" }), [], () => 42).seed).toBe(42);
    for (const seed of ["-1", "1.5", "abc"]) expect(() => buildImageRequest(krea, draft(krea, { seed }), [])).toThrow(/Seed must be/);
  });

  it("explains an empty prompt, a non-image model and too many references", () => {
    expect(() => buildImageRequest(krea, draft(krea, { prompt: "  " }), [])).toThrow("Prompt is empty.");
    expect(() => buildImageRequest(chat, draft(krea), [])).toThrow("Select an image model.");
    expect(() => buildImageRequest(flux, draft(flux), [ref, ref, ref, ref, ref])).toThrow(/at most four/);
  });

  it("an edit sends the source first, then the references, and keeps the canvas it was given", () => {
    const body = buildImageRequest(flux, draft(flux, { mode: "edit", width: "512", height: "768" }), [{ base64: "SRC" }, { base64: "R1" }, { base64: "R2" }]);
    expect(body).toMatchObject({ mode: "edit", image: "SRC", ref_images: ["R1", "R2"], size: "512x768" });
    expect(body).not.toHaveProperty("strength");
  });

  it("an edit that asks for the source's own size sends no canvas", () => {
    expect(buildImageRequest(flux, draft(flux, { mode: "edit", width: "0", height: "0" }), [ref])).not.toHaveProperty("size");
  });

  it("a variation sends the source and a strength, never edit fields", () => {
    const body = buildImageRequest(krea, draft(krea, { strength: 0.4 }), [{ base64: "SRC" }]);
    expect(body).toMatchObject({ image: "SRC", strength: 0.4 });
    expect(body).not.toHaveProperty("mode");
    expect(() => buildImageRequest(krea, draft(krea, { strength: 1.5 }), [ref])).toThrow(/Variation strength must be between/);
  });

  it("guidance and a negative prompt reach only models that use them, and only when set", () => {
    const base = model("mflux/flux2-klein-9b-base-q4", "flux2");
    expect(buildImageRequest(base, draft(base, { guidance: 4, negative: " blur " }), [])).toMatchObject({ guidance_scale: 4, negative_prompt: "blur" });
    expect(buildImageRequest(base, draft(base, { guidance: 1 }), [])).not.toHaveProperty("guidance_scale");
    expect(buildImageRequest(flux, draft(flux, { guidance: 4, negative: "blur" }), [])).not.toHaveProperty("guidance_scale");
  });

  it("conditioning gain and layer weights are for the models that tap layers", () => {
    expect(buildImageRequest(flux, draft(flux, { gain: 2, weights: "1, 0.5 2" }), [])).toMatchObject({ cond_gain: 2, cond_weights: [1, 0.5, 2] });
    expect(() => buildImageRequest(flux, draft(flux, { weights: "1 2" }), [])).toThrow(/exactly 3 finite numbers/);
    expect(buildImageRequest(mage, draft(mage, { gain: 3, weights: "1 2 3" }), [])).not.toHaveProperty("cond_gain");
  });

  it("transparency is a Qwen-Image option", () => {
    expect(buildImageRequest(qwen, draft(qwen, { transparent: true }), [])).toMatchObject({ transparent: true });
    expect(buildImageRequest(krea, draft(krea, { transparent: true }), [])).not.toHaveProperty("transparent");
  });

  it("bounds the step count", () => {
    expect(() => buildImageRequest(krea, draft(krea, { steps: 51 }), [])).toThrow(/Steps must be between 1 and 50/);
    expect(() => buildImageRequest(krea, draft(krea, { steps: 2.5 }), [])).toThrow(/whole number/);
  });
});

describe("sourceCanvases", () => {
  const p = imageProfile(flux)!;
  it("puts the source's own size first and offers a spread of faithful sizes", () => {
    const list = sourceCanvases(p, 1536, 1024);
    expect(list[0]).toMatchObject({ width: 1536, height: 1024, name: "source size" });
    expect(list.length).toBeLessThanOrEqual(6);
    for (const c of list.slice(1)) expect(Math.abs(c.width / c.height - 1.5) / 1.5).toBeLessThan(0.05);
  });

  it("offers nothing for a source without a size", () => {
    expect(sourceCanvases(p, 0, 0)).toEqual([]);
  });
});

describe("insertImageMarker", () => {
  it("puts the marker where the cursor is, spaced like prose", () => {
    expect(insertImageMarker("image 1", "make it red", 11)).toEqual({ text: "make it red image 1", cursor: 19 });
    expect(insertImageMarker("image 1", "", 0)).toEqual({ text: "image 1", cursor: 7 });
    expect(insertImageMarker("image 1", "a b", 1, 2).text).toBe("a image 1 b");
    expect(insertImageMarker("image 1", "swap, now", 4).text).toBe("swap image 1, now");
  });
});
