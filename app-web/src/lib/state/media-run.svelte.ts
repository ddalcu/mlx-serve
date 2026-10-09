import type { Client } from "../core/client";
import type { Library, LibraryInput } from "../core/library";
import { t } from "../i18n/i18n";

export type Saved = LibraryInput & { id?: string; elapsedMs: number };

/**
 * One generation at a time for an image, audio or video pane: progress, cancel and saving the result to the Library.
 * Navigation can discard a late reply even when the transport cannot stop remote computation.
 */
export abstract class MediaRun<R extends Saved = Saved> {
  phase = $state("idle");
  message = $state("");
  step = $state(0);
  total = $state(0);
  saveError = $state("");
  saving = $state(false);
  run = $state.raw<AbortController | null>(null);
  result = $state<R | null>(null);
  readonly library: Pick<Library, "add">;
  protected abstract readonly busy: string;
  protected abstract readonly saveFailed: string;
  protected readonly cancelledPhase: string = "idle";
  protected readonly cancelled: string = t("Cancelled. The server may still be finishing the request.");

  constructor(library: Pick<Library, "add">) {
    this.library = library;
  }

  /** Claim the single run slot and reset progress. */
  protected begin(total: number): AbortController {
    if (this.run) throw Error(this.busy);
    const run = new AbortController();
    this.run = run;
    this.phase = "running";
    this.step = 0;
    this.total = total;
    this.message = t("Loading model…");
    this.saveError = "";
    return run;
  }

  /** Progress events from the server stream, ignored once this run is no longer the current one. */
  protected progress(run: AbortController, extra?: (event: Record<string, unknown>) => void) {
    return (e: Record<string, unknown>) => {
      if (this.run !== run) return;
      extra?.(e);
      this.step = typeof e.step === "number" ? e.step : 0;
      this.total = typeof e.total === "number" ? e.total : this.total;
      this.message = typeof e.stage === "string" ? `${e.stage}…` : t("Generating…");
    };
  }

  protected fail(run: AbortController, e: unknown) {
    if (this.run !== run) return;
    this.phase = "failed";
    this.message = e instanceof Error ? e.message : t("Generation failed.");
  }

  protected end(run: AbortController) {
    if (this.run === run) this.run = null;
  }

  /** Record the finished result and keep it in the Library. */
  protected async complete(result: R) {
    this.result = result;
    this.phase = "completed";
    this.message = "";
    await this.save();
  }

  cancel() {
    if (!this.run) return;
    this.run.abort();
    this.run = null;
    this.phase = this.cancelledPhase;
    this.message = this.cancelled;
  }

  async save() {
    const result = this.result;
    if (!result || result.id || this.saving) return;
    this.saving = true;
    try {
      result.id = await this.library.add(result);
      this.saveError = "";
    } catch (e) {
      this.saveError = t(this.saveFailed, [e instanceof Error ? e.message : t("storage unavailable")]);
    }
    this.saving = false;
  }

  abstract generate(client: Client, ...args: never[]): Promise<void>;
}
