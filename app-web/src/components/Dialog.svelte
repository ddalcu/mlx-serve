<script lang="ts">
  import type { Snippet } from "svelte";
  import { t } from "../lib/i18n/i18n";

  let { title, onclose, class: className = "", base = "chat-dialog", bare = false, children }: { title: string; onclose: () => void; class?: string; base?: string; bare?: boolean; children: Snippet<[() => void]> } = $props();
  let dialog: HTMLDialogElement;
  let finished = false;
  const opener = document.activeElement;

  $effect(() => dialog.showModal());

  // Every way of closing ends here once: Escape, the close button and the dialog's own buttons.
  function finish() {
    if (finished) return;
    finished = true;
    onclose();
    if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
  }
  function close() {
    dialog.close();
    finish();
  }
</script>

<dialog
  bind:this={dialog}
  class="{base} {className}"
  aria-label={title}
  oncancel={(event) => {
    event.preventDefault();
    close();
  }}
  onclose={finish}
>
  {#if !bare}
    <header>
      <h2>{title}</h2>
      <button aria-label={t("Close")} onclick={close}>×</button>
    </header>
  {/if}
  {@render children(close)}
</dialog>
