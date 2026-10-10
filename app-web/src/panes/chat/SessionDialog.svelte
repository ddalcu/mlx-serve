<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import { getApp } from "../../lib/app.svelte";
  import { t } from "../../lib/i18n/i18n";

  let { id, onclose }: { id: string; onclose: () => void } = $props();
  const chat = getApp().chat;
  const session = chat.c.sessions.find((s) => s.id === id);
  let title = $state(session ? (session.title === "New Chat" ? t("New Chat") : session.title) : "");
  let confirming = $state(false);
  let error = $state("");

  function rename(event: SubmitEvent, close: () => void) {
    event.preventDefault();
    if (!session) return;
    session.title = title.trim() || t("New Chat");
    void chat.c.save(session);
    close();
  }
  async function remove(close: () => void) {
    try {
      await chat.deleteSession(id);
      close();
    } catch {
      error = t("Could not delete chat. Try again.");
    }
  }
</script>

{#if session}
  <Dialog title={confirming ? t("Delete Chat") : t("Chat")} {onclose}>
    {#snippet children(close)}
      {#if confirming}
        <p>{t("Delete “%@”? This can't be undone.", [session.title])}</p>
        {#if error}<p class="error-message">{error}</p>{/if}
        <button id="confirm-delete" onclick={() => void remove(close)}>{t("Delete")}</button>
        <button id="cancel-delete" onclick={close}>{t("Cancel")}</button>
      {:else}
        <form onsubmit={(e) => rename(e, close)}>
          <label>{t("Rename")}<input name="title" bind:value={title} required maxlength="100" /></label>
          <button type="submit">{t("Save")}</button>
        </form>
        <hr />
        <button id="delete-chat" onclick={() => (confirming = true)}>{t("Delete Chat…")}</button>
      {/if}
    {/snippet}
  </Dialog>
{/if}
