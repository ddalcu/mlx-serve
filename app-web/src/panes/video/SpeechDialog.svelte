<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import type { App } from "../../lib/app.svelte";
  import { t } from "../../lib/i18n/i18n";
  import { defaultModel, loadNotice } from "../../lib/state/connection.svelte";

  let { app, onclose }: { app: App; onclose: () => void } = $props();
  // svelte-ignore state_referenced_locally
  const { connection, video: ws } = app;
  const model = defaultModel(connection.models.filter((m) => m.architecture === "qwen3_tts"));
  let text = $state(ws.d.speech);
  let status = $state("");

  async function create() {
    const line = text.trim();
    if (!line || !model) return;
    status = "";
    try {
      await ws.createSpeech(model, line);
      return true;
    } catch (e) {
      status = e instanceof Error ? e.message : t("Speech failed.");
    }
  }
  $effect(() => () => ws.stopActivity());
</script>

<Dialog title={t("Speech & sound")} class="audio-rewrite" {onclose}>
  {#snippet children(close)}
    {#if model}
      <p>{model.id}</p>
      <p class="field-note">{loadNotice(model, connection.serverName())}</p>
      <label>{t("Text")}<textarea maxlength="600" bind:value={text}></textarea></label>
      <p>{t("Creates speech on this server, then attaches it to the video. Keep the line short.")}</p>
      <button type="button" id="video-create-speech" disabled={ws.busy} onclick={async () => (await create()) && close()}>{ws.inputs.audio ? t("Recreate speech") : t("Create speech")}</button>
    {:else}
      <p>{t("Choose an installed Qwen3-TTS model in Audio first, then come back.")}</p>
    {/if}
    <p role="status">{status}</p>
    <button onclick={close}>{t("Back")}</button>
  {/snippet}
</Dialog>
