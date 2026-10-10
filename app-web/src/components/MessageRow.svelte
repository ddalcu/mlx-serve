<script lang="ts">
  import { getApp } from "../lib/app.svelte";
  import { copyText } from "../lib/clipboard";
  import { t } from "../lib/i18n/i18n";
  import { thinkingLabel, type Message } from "../lib/state/chat-state.svelte";
  import Icon from "./Icon.svelte";
  import Markdown from "./Markdown.svelte";
  import ToolCards from "./ToolCards.svelte";

  let { message: m, last }: { message: Message; last: boolean } = $props();
  const chat = getApp().chat;
  const busy = $derived(!!chat.c.run || chat.voicing);
  const assistant = $derived(m.role === "assistant");
  const streaming = $derived(m.status === "streaming");
  let editing = $state(false);
  let draft = $state("");
  let copied = $state(false);

  // Only images this console produced or accepted are drawn, whatever an imported archive claims.
  const showable = (url: string) => /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+=*$/.test(url);

  async function copy() {
    if (await copyText(m.text)) {
      copied = true;
      setTimeout(() => (copied = false), 2000);
    } else chat.error = t("Clipboard unavailable. Select the text and copy it manually.");
  }
  function edit() {
    if (busy) return;
    draft = m.text;
    editing = true;
  }
  async function save() {
    editing = false;
    await chat.edit(m.id, draft);
  }
  const focus = (node: HTMLElement) => node.focus();
</script>

<div class="message-body">
  {#if editing}
    <textarea class="message-editor" aria-label={t("Edit message")} bind:value={draft} use:focus></textarea>
    <div class="edit-actions">
      <button onclick={() => (editing = false)}>{t("Cancel")}</button>
      <button onclick={save}>{t("Save")}</button>
    </div>
  {:else}
    {#if m.images?.length}
      <div class="message-images">
        {#each m.images as url}{#if showable(url)}<img src={url} alt={t("Attached image")} />{/if}{/each}
      </div>
    {/if}
    {#if m.thinking}
      <details class="thinking-block">
        <summary><Icon name="lightbulb" />{streaming && !m.text ? t("Thinking…") : thinkingLabel(m.thinkingSeconds)}<Icon name="chevron-down" /></summary>
        <div class="thinking-text">{m.thinking}</div>
      </details>
    {/if}
    <ToolCards rounds={m.toolRounds ?? []} />
    <div class="message-text" class:markdown={assistant}>
      {#if assistant}<Markdown source={m.text} />{:else}{m.text}{/if}
    </div>
    {#if streaming && !m.text && !m.thinking && !m.toolRounds?.length}<span class="stream-wait">{t("Thinking…")}</span>{/if}
    {#if m.error}<p class="error-message">{m.error}</p>{/if}
  {/if}
</div>
<footer>
  <time datetime={new Date(m.createdAt).toISOString()}>{new Date(m.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</time>
  {#if m.status === "stopped"}<span>{t("Stopped")}</span>{/if}
  <button title={t("Copy Message")} aria-label={t("Copy Message")} onclick={copy}>{#if copied}{t("Copied")}{:else}<Icon name="copy" />{/if}</button>
  <button disabled={busy} title={m.role === "user" ? t("Edit & Resend") : t("Edit Reply")} aria-label={m.role === "user" ? t("Edit & Resend") : t("Edit Reply")} onclick={edit}><Icon name="pencil" /></button>
  {#if assistant && last}
    <button disabled={busy} title={t("Regenerate")} aria-label={t("Regenerate")} onclick={() => void chat.regenerate()}><Icon name="refresh-cw" /></button>
  {/if}
  <button disabled={busy} title={m.role === "user" ? t("Delete Message") : t("Delete Turn")} aria-label={m.role === "user" ? t("Delete Message") : t("Delete Turn")} onclick={() => chat.c.deleteMessage(m.id)}><Icon name="trash" /></button>
  {#if m.tokensPerSecond}<span class="token-rate">{t("%@ tok/sec", [Math.round(m.tokensPerSecond)])}</span>{/if}
</footer>
