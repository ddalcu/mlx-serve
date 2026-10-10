<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import { getApp } from "../../lib/app.svelte";
  import { t } from "../../lib/i18n/i18n";
  import { kokoroVoices } from "../../lib/state/audio-presets";
  import { audioProfile } from "../../lib/state/audio-state.svelte";

  let { onclose }: { onclose: () => void } = $props();
  const chat = getApp().chat;
  const models = chat.speechModels;
  let selected = $state(models.find((m) => m.id === chat.voiceModel)?.id ?? models[0]?.id ?? "");
  let voice = $state(chat.voiceName);
  const hasVoices = $derived(!!audioProfile(models.find((m) => m.id === selected))?.voices);
</script>

<Dialog title={t("Voice chat")} class="voice-dialog" {onclose}>
  {#snippet children(close)}
    <p>{t("Opt in to browser speech recognition. Your browser may send microphone audio to its speech service; recognition may need the internet. Replies are synthesized by the selected mlx-serve server. The microphone pauses while replies play.")}</p>
    <label>{t("Speech model")}<select id="voice-model" bind:value={selected}>{#each models as m}<option value={m.id}>{m.id}</option>{/each}</select></label>
    <label id="voice-name-label" hidden={!hasVoices}>{t("Kokoro voice")}<select id="voice-name" bind:value={voice}>{#each kokoroVoices as v}<option>{v}</option>{/each}</select></label>
    <p>{models.length ? t("An unloaded speech model will be loaded on the selected server.") : t("No supported speech model is advertised by this server.")}</p>
    <footer class="form-actions">
      <button id="voice-cancel" onclick={close}>{t("Cancel")}</button>
      <button
        class="primary"
        id="voice-start"
        disabled={!models.length || !chat.model}
        onclick={() => {
          chat.startVoice(selected, voice);
          close();
        }}>{t("Start voice chat")}</button
      >
    </footer>
  {/snippet}
</Dialog>
