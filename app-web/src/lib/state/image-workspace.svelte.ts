import { imageReference, type Reference } from "../core/image-reference";
import { Library, type LibraryItem } from "../core/library";
import { t } from "../i18n/i18n";
import type { Connection } from "./connection.svelte";
import { buildImageRequest, ImageController, imageDefaults, imageProfile, resolveCanvas, sourceCanvases } from "./image-state.svelte";

const finite = (v: unknown): v is Reference =>
  !!v &&
  typeof (v as Reference).name === "string" &&
  typeof (v as Reference).base64 === "string" &&
  (v as Reference).base64.length <= 14 * 1024 * 1024 &&
  /^[A-Za-z0-9+/]+=*$/.test((v as Reference).base64) &&
  (v as Reference).width > 0 &&
  (v as Reference).height > 0;

/** The image pane's draft (kept per server), references and generation, without any drawing. */
export class ImageWorkspace {
  readonly c: ImageController;
  readonly library: Library;
  // A separate draft database keeps source pictures out of the results gallery.
  readonly drafts: Pick<Library, "get" | "put">;
  d = $state(imageDefaults(undefined));
  refs = $state<Reference[]>([]);
  error = $state("");
  draftError = $state("");
  ready = $state(false);
  attaching = $state(false);
  server: string;
  #connection: Connection;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #saveChain: Promise<void> = Promise.resolve();
  #epoch = 0;
  #signature = "";

  constructor(connection: Connection, library: Library, drafts: Pick<Library, "get" | "put">) {
    this.#connection = connection;
    this.library = library;
    this.drafts = drafts;
    this.c = new ImageController(library);
    this.server = connection.active.url;
  }

  get models() {
    return this.#connection.models.filter((m) => m.capabilities.includes("image"));
  }
  get model() {
    return this.models.find((m) => m.id === this.d.model);
  }
  get profile() {
    return imageProfile(this.model);
  }
  get editing() {
    const p = this.profile;
    return !!p?.edit && (this.d.mode === "edit" || !p.variation);
  }

  /** The reason Generate is unavailable (empty while the form is valid) and the canvas rounding hint. */
  get validation() {
    const model = this.model,
      p = this.profile;
    if (!this.ready) return { error: t("Restoring image draft…"), hint: "" };
    if (!model) return { error: this.#connection.status === "checking" ? t("Loading models…") : t("No available image model selected. Choose one on this server, or check Settings."), hint: "" };
    try {
      buildImageRequest(model, this.d, this.refs, () => 0);
      return { error: "", hint: p ? resolveCanvas(p, this.d.width, this.d.height).hint : "" };
    } catch (e) {
      return { error: e instanceof Error ? e.message : t("Invalid settings."), hint: "" };
    }
  }

  async init() {
    const epoch = ++this.#epoch,
      server = this.server;
    this.ready = false;
    try {
      await this.#saveChain;
      const saved = await this.drafts.get(server);
      if (epoch !== this.#epoch) return;
      if (saved) {
        const value = JSON.parse(await saved.blob.text());
        if (epoch !== this.#epoch) return;
        const defaults = imageDefaults(undefined);
        for (const key of Object.keys(defaults) as (keyof typeof defaults)[])
          if (typeof value.d?.[key] === typeof defaults[key]) Object.assign(this.d, { [key]: value.d[key] });
        if (Array.isArray(value.refs)) this.refs = value.refs.filter(finite).slice(0, 4);
      }
    } catch {
      if (epoch === this.#epoch) this.draftError = t("Could not restore the image draft. Changes can still be used this session.");
    }
    if (epoch !== this.#epoch) return;
    this.ready = true;
    this.#signature = this.signature();
    this.connectionChanged();
  }

  /** Cheap change detector for the draft: every field, but only the references' names (not their pixels). */
  signature() {
    return JSON.stringify([this.d, this.refs.map((r) => r.name + r.base64.length)]);
  }

  /** Save the draft shortly after it stops changing. */
  touch() {
    if (!this.ready) return;
    const signature = this.signature();
    if (signature === this.#signature) return;
    this.#signature = signature;
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => this.flush(), 250);
  }

  flush() {
    clearTimeout(this.#timer);
    if (!this.ready) return;
    const server = this.server,
      snapshot = JSON.stringify({ d: this.d, refs: this.refs }),
      model = this.d.model;
    this.#saveChain = this.#saveChain.then(async () => {
      try {
        await this.drafts.put(server, { type: "image", server, model, blob: new Blob([snapshot], { type: "application/json" }) });
        if (server === this.server) this.draftError = "";
      } catch {
        if (server === this.server) this.draftError = t("Image draft was not saved. Keep this tab open and download any results.");
      }
    });
  }

  /** The selected server changed: drop this server's draft state and load the new one's. */
  connectionChanged() {
    const url = this.#connection.active.url;
    if (this.server !== url) {
      this.flush();
      this.c.cancel();
      this.server = url;
      this.d = imageDefaults(undefined);
      this.refs = [];
      this.c.result = null;
      this.c.phase = "idle";
      this.c.message = "";
      this.error = "";
      this.draftError = "";
      this.ready = false;
      void this.init();
      return;
    }
    if (!this.ready) return;
    if (!this.d.model && this.models.length) this.d = imageDefaults(this.models.find((m) => m.loaded) ?? this.models[0]);
  }

  chooseModel(id: string) {
    const m = this.models.find((m) => m.id === id);
    this.d = { ...imageDefaults(m), prompt: this.d.prompt, seed: this.d.seed, advanced: this.d.advanced };
    this.adoptSource();
    this.error = "";
  }

  /** A single edit source sets the canvas to its own size. */
  adoptSource() {
    const p = this.profile,
      r = this.refs[0];
    if (!p || !r || !this.editing) return;
    const own = sourceCanvases(p, r.width, r.height).find((c) => c.name === "source size");
    if (own) {
      this.d.width = String(own.width);
      this.d.height = String(own.height);
    }
  }
  removeRef(index: number) {
    if (!this.editing) this.refs = [];
    else this.refs.splice(index, 1);
    this.adoptSource();
  }

  async attach(files: FileList | File[] | null | undefined) {
    const p = this.profile;
    if (!files?.length || this.attaching || this.c.run || !p || !(p.edit || p.variation)) return;
    const epoch = this.#epoch,
      room = (this.editing ? 4 : 1) - this.refs.length;
    if (files.length > room) {
      this.error = t("Choose at most %@ more image(s).", [Math.max(room, 0)]);
      return;
    }
    this.attaching = true;
    try {
      const added: Reference[] = [];
      for (const f of files) added.push(await imageReference(f, f.name));
      if (epoch !== this.#epoch) return;
      this.refs.push(...added);
      this.adoptSource();
      this.error = "";
    } catch (e) {
      if (epoch === this.#epoch) this.error = e instanceof Error ? e.message : t("Could not read the image.");
    } finally {
      this.attaching = false;
    }
  }

  async generate() {
    if (this.c.run) {
      this.c.cancel();
      return;
    }
    const model = this.model;
    if (!model || this.attaching) return;
    try {
      const body = buildImageRequest(model, this.d, this.refs);
      this.error = "";
      await this.c.generate(this.#connection.client(), body, this.server);
    } catch (e) {
      this.error = e instanceof Error ? e.message : t("Generation failed.");
    }
  }

  get filename() {
    return `image-${new Date(this.c.result?.createdAt ?? Date.now()).toISOString().replace(/[:.]/g, "-")}.png`;
  }

  /** Use the finished picture as the next source image. */
  async reuse() {
    if (!this.c.result || this.c.run) return;
    const epoch = this.#epoch;
    try {
      const reference = await imageReference(this.c.result.blob, this.filename);
      if (epoch !== this.#epoch) return;
      this.refs = [reference];
      this.d.mode = this.profile?.edit ? "edit" : "variation";
      this.adoptSource();
      return true;
    } catch (e) {
      this.error = e instanceof Error ? e.message : t("Could not use image.");
    }
  }

  /** Show a saved gallery picture in the preview. */
  show(item: LibraryItem) {
    if (this.c.run) return;
    this.c.result = { ...item, elapsedMs: 0 };
    this.c.phase = "completed";
    this.c.saveError = "";
  }

}
