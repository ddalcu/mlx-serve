<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import { getApp } from "../../lib/app.svelte";
  import { t } from "../../lib/i18n/i18n";
  import { canMTP } from "../../lib/state/chat-state.svelte";

  let { onclose }: { onclose: () => void } = $props();
  const chat = getApp().chat;
  const settings = chat.c.active.settings;
  const model = chat.model;
  let system = $state(settings.system);
  let temperature = $state(settings.temperature === null ? "" : String(settings.temperature));
  let maxTokens = $state(settings.maxTokens === null ? "" : String(settings.maxTokens));
  let mtp = $state(settings.mtp === null ? "" : String(settings.mtp));

  function submit(event: SubmitEvent, close: () => void) {
    event.preventDefault();
    settings.system = system;
    settings.temperature = temperature ? Number(temperature) : null;
    settings.maxTokens = maxTokens ? Number(maxTokens) : null;
    settings.mtp = mtp === "true" ? true : mtp === "false" ? false : null;
    void chat.c.save();
    close();
  }
</script>

<Dialog title={t("Chat settings")} {onclose}>
  {#snippet children(close)}
    <form id="chat-settings-form" onsubmit={(e) => submit(e, close)}>
      <label>{t("System prompt")}<textarea name="system" bind:value={system}></textarea></label>
      <div class="field-pair">
        <label>{t("Temperature")}<input name="temperature" type="number" min="0" max="2" step="0.01" placeholder={t("Model default")} bind:value={temperature} /></label>
        <label>{t("Max tokens")}<input name="maxTokens" type="number" min="1" step="1" placeholder={t("Model default")} bind:value={maxTokens} /></label>
      </div>
      {#if model && canMTP(model)}
        <label>
          {t("Multi-Token Prediction")}
          <select name="mtp" bind:value={mtp}>
            <option value="">{t("Model default")}</option>
            <option value="true">{t("On")}</option>
            <option value="false">{t("Off")}</option>
          </select>
        </label>
      {/if}
      <p>{t("Applies to this chat. Leave sampling fields empty to use model defaults.")}</p>
      <button type="submit">{t("Save")}</button>
    </form>
  {/snippet}
</Dialog>
