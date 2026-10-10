<script lang="ts">
  import { onMount } from "svelte";
  import Dialog from "./Dialog.svelte";
  import Menu from "./Menu.svelte";
  import { chat } from "../lib/core/chat";
  import { applyRewriteEvent, cleanReply, emptyReplyReason, newRewriteProgress, rewriteLabel, type RewriteRequest } from "../lib/core/rewrite";
  import { lengthLabel } from "../lib/core/storyboard";
  import { t } from "../lib/i18n/i18n";
  import { defaultModel, loadNotice, type Connection } from "../lib/state/connection.svelte";

  // Have a chat model rewrite a prompt: pick the model, run it, review the text, then apply it.
  type Props = {
    connection: Connection;
    title: string;
    class: string;
    /** The text being rewritten. */
    text: string;
    system?: string;
    request?: string;
    maxTokens?: number;
    /** Builds the request per length instead of `system`/`request`/`maxTokens` (the video pane). */
    compose?: (seconds: number) => RewriteRequest;
    /** A length slider, in seconds, read by `compose`. */
    clip?: { initial: number; min: number; max: number };
    /** A line under the slider for a length. */
    note?: (seconds: number) => string;
    /** Why the text written at a length cannot be applied, or "". */
    applyError?: (text: string, seconds: number) => string;
    textLabel: string;
    reviewed: string;
    noModel: string;
    onapply: (text: string, seconds: number) => void;
    onclose: () => void;
  };
  let { connection, title, class: className, text: initial, system = "", request = "", maxTokens = 2048, compose, clip, note, applyError, textLabel, reviewed, noModel, onapply, onclose }: Props = $props();
  // The dialog is a snapshot: it opens with what it was given and does not follow later changes.
  // svelte-ignore state_referenced_locally
  const models = connection.models.filter((m) => m.capabilities.includes("chat"));
  let model = $state(defaultModel(models)?.id ?? "");
  // svelte-ignore state_referenced_locally
  let text = $state(initial);
  // svelte-ignore state_referenced_locally
  let seconds = $state(clip?.initial ?? 0);
  /** The length the text in the box was written for. */
  // svelte-ignore state_referenced_locally
  let written = $state(seconds);
  let status = $state("");
  let failed = $state(false);
  let blind = $state(false);
  let progress = $state(newRewriteProgress());
  let started = $state(0);
  let now = $state(Date.now());
  let rewriting = $state.raw<AbortController | null>(null);
  const problem = $derived(rewriting ? "" : (applyError?.(text, written) ?? ""));
  const elapsed = $derived(new Date(Math.max(0, now - started)).toISOString().slice(14, 19).replace(/^0/, ""));

  async function rewrite() {
    if (rewriting) return;
    const run = new AbortController(),
      chosen = models.find((m) => m.id === model),
      req = compose?.(seconds) ?? { system, user: request, maxTokens, userWith: () => request },
      sees = !!req.firstFrame && !!chosen?.capabilities.includes("vision");
    rewriting = run;
    text = "";
    failed = false;
    written = seconds;
    blind = !!req.firstFrame && !sees;
    progress = newRewriteProgress(chosen && !chosen.loaded ? chosen.id : "");
    started = now = Date.now();
    const user = req.userWith(sees);
    const content = sees ? [{ type: "text", text: user }, { type: "image_url", image_url: { url: `data:image/png;base64,${req.firstFrame}` } }] : user;
    try {
      status = "";
      const events = chat(connection.client(), { model, messages: [{ role: "system", content: req.system }, { role: "user", content }], max_tokens: req.maxTokens }, { signal: run.signal });
      for await (const e of events) {
        progress = applyRewriteEvent(progress, e);
        if (e.type === "content") text = progress.text;
      }
      text = cleanReply(text);
      status = emptyReplyReason(progress);
      failed = !!status;
      status ||= reviewed;
    } catch (e) {
      status = e instanceof Error ? e.message : t("Rewrite failed.");
      failed = true;
    } finally {
      if (rewriting === run) rewriting = null;
    }
  }
  onMount(() => {
    const clock = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(clock);
  });
  $effect(() => () => rewriting?.abort());
</script>

<Dialog {title} class={className} {onclose}>
  {#snippet children(close)}
    {#if models.length}
      <p>{t("Model")}</p>
      <Menu id="rewrite-model" label={model} onchoose={(value) => (model = value)}>
        {#each models as m (m.id)}<button type="button" role="menuitemradio" tabindex="-1" aria-checked={m.id === model} data-value={m.id}>{m.id}</button>{/each}
      </Menu>
      <div class="rewrite-box">
        <textarea id="rewrite-text" aria-label={textLabel} bind:value={text}></textarea>
        {#if rewriting && !text && progress.thought}
          <!-- The think, until the answer starts: proof the model is working. -->
          <p id="rewrite-thought" class="field-note">{progress.thought.slice(-1200)}</p>
        {/if}
      </div>
      {#if clip}
        <label class="video-slider">
          {t("Clip length")}<output id="rewrite-seconds-value">{lengthLabel(seconds)}</output>
          <input id="rewrite-seconds" type="range" min={clip.min} max={Math.max(clip.min + 1, clip.max)} step="1" disabled={!!rewriting} bind:value={seconds} />
        </label>
        {#if note?.(seconds)}<p id="rewrite-clip-note" class="field-note">{note(seconds)}</p>{/if}
      {/if}
      <p id="rewrite-load-notice" class="field-note">{loadNotice(models.find((m) => m.id === model), connection.serverName())}</p>
      {#if rewriting}
        <p id="rewrite-status" role="status"><span class="spinner"></span> {rewriteLabel(progress)} {elapsed}</p>
      {:else if problem || failed}
        <p id="rewrite-status" role="status" class="rewrite-error">{problem || status}</p>
      {:else}
        <p id="rewrite-status" role="status">{status}</p>
      {/if}
      {#if blind && !rewriting}
        <p id="rewrite-blind" class="field-note rewrite-warning">{t("This chat model can't see images, so this was written without your first frame and the video may cut away from it. A vision chat model fixes that.")}</p>
      {/if}
      <div class="form-actions">
        <button id="rewrite-close" onclick={close}>{t("Cancel")}</button>
        <button id="rewrite-run" disabled={!!rewriting} onclick={() => void rewrite()}>{t("Rewrite")}</button>
        <button
          id="rewrite-apply"
          class="primary"
          disabled={!!rewriting || !text.trim() || !!problem}
          onclick={() => {
            onapply(text, written);
            close();
          }}>{t("Apply")}</button
        >
      </div>
    {:else}
      <p>{noModel}</p>
      <button id="rewrite-close" onclick={close}>{t("Close")}</button>
    {/if}
  {/snippet}
</Dialog>
