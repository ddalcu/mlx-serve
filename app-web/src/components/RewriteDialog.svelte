<script lang="ts">
  import Dialog from "./Dialog.svelte";
  import Menu from "./Menu.svelte";
  import { chat } from "../lib/core/chat";
  import { t } from "../lib/i18n/i18n";
  import { defaultModel, loadNotice, type Connection } from "../lib/state/connection.svelte";

  // Have a chat model rewrite a prompt: pick the model, run it, review the text, then apply it.
  type Props = {
    connection: Connection;
    title: string;
    class: string;
    /** The text being rewritten. */
    text: string;
    system: string;
    request: string;
    maxTokens: number;
    textLabel: string;
    reviewed: string;
    noModel: string;
    onapply: (text: string) => void;
    onclose: () => void;
  };
  let { connection, title, class: className, text: initial, system, request, maxTokens, textLabel, reviewed, noModel, onapply, onclose }: Props = $props();
  // The dialog is a snapshot: it opens with what it was given and does not follow later changes.
  // svelte-ignore state_referenced_locally
  const models = connection.models.filter((m) => m.capabilities.includes("chat"));
  let model = $state(defaultModel(models)?.id ?? "");
  // svelte-ignore state_referenced_locally
  let text = $state(initial);
  let status = $state("");
  let rewriting = $state.raw<AbortController | null>(null);

  async function rewrite() {
    if (rewriting) return;
    const run = new AbortController();
    rewriting = run;
    text = "";
    try {
      status = t("Rewriting…");
      const events = chat(
        connection.client(),
        { model, messages: [{ role: "system", content: system }, { role: "user", content: request }], max_tokens: maxTokens },
        { signal: run.signal },
      );
      for await (const e of events) if (e.type === "content") text += e.text;
      status = reviewed;
    } catch (e) {
      status = e instanceof Error ? e.message : t("Rewrite failed.");
    } finally {
      if (rewriting === run) rewriting = null;
    }
  }
  $effect(() => () => rewriting?.abort());
</script>

<Dialog {title} class={className} {onclose}>
  {#snippet children(close)}
    {#if models.length}
      <p>{t("Model")}</p>
      <Menu id="rewrite-model" label={model} onchoose={(value) => (model = value)}>
        {#each models as m (m.id)}<button type="button" role="menuitemradio" tabindex="-1" aria-checked={m.id === model} data-value={m.id}>{m.id}</button>{/each}
      </Menu>
      <textarea id="rewrite-text" aria-label={textLabel} bind:value={text}></textarea>
      <p id="rewrite-load-notice" class="field-note">{loadNotice(models.find((m) => m.id === model), connection.serverName())}</p>
      <p id="rewrite-status" role="status">{status}</p>
      <div class="form-actions">
        <button id="rewrite-close" onclick={close}>{t("Cancel")}</button>
        <button id="rewrite-run" disabled={!!rewriting} onclick={() => void rewrite()}>{t("Rewrite")}</button>
        <button
          id="rewrite-apply"
          class="primary"
          disabled={!!rewriting || !text.trim()}
          onclick={() => {
            onapply(text);
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
