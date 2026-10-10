<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import { t } from "../../lib/i18n/i18n";
  import type { Field } from "../../lib/state/audio-workspace.svelte";

  let { field, text, onsave, onclose }: { field: Field; text: string; onsave: (title: string) => void; onclose: () => void } = $props();
  // svelte-ignore state_referenced_locally
  let title = $state(text.split("\n")[0]!.slice(0, 50));
</script>

<Dialog title={t("Save %@", [field === "prompt" ? t("style") : t("lyrics")])} class="audio-rewrite" {onclose}>
  {#snippet children(close)}
    <form
      onsubmit={(e) => {
        e.preventDefault();
        if (title.trim()) onsave(title.trim());
        close();
      }}
    >
      <label>{t("Name")}<input name="title" bind:value={title} required /></label>
      <div class="form-actions">
        <button type="button" onclick={close}>{t("Cancel")}</button>
        <button type="submit">{t("Save")}</button>
      </div>
    </form>
  {/snippet}
</Dialog>
