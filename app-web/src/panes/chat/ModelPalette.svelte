<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import { getApp } from "../../lib/app.svelte";
  import { t } from "../../lib/i18n/i18n";

  let { onclose }: { onclose: () => void } = $props();
  const chat = getApp().chat;
  let query = $state("");
  let index = $state(0);
  const rows = $derived(chat.models.filter((m) => m.id.toLowerCase().includes(query.toLowerCase())));
  const current = $derived(Math.max(0, Math.min(index, rows.length - 1)));

  $effect(() => {
    document.getElementById(`model-result-${current}`)?.scrollIntoView({ block: "nearest" });
  });

  function keys(event: KeyboardEvent, close: () => void) {
    if (!["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Enter") {
      const m = rows[current];
      if (m) {
        chat.pick(m.id);
        close();
      }
    } else index = current + (event.key === "ArrowDown" ? 1 : -1);
  }
  const focus = (node: HTMLElement) => node.focus();
</script>

<Dialog title={t("Switch Model")} class="model-palette" {onclose}>
  {#snippet children(close)}
    <input
      id="model-query"
      bind:value={query}
      oninput={() => (index = 0)}
      onkeydown={(e) => keys(e, close)}
      placeholder={t("Search models…")}
      aria-label={t("Search models")}
      autocomplete="off"
      role="combobox"
      aria-controls="model-results"
      aria-expanded="true"
      aria-activedescendant={rows.length ? `model-result-${current}` : ""}
      use:focus
    />
    <div id="model-results" role="listbox" aria-label={t("Chat models")}>
      {#each rows as m, i (m.id)}
        <button
          id="model-result-{i}"
          role="option"
          aria-selected={i === current}
          onclick={() => {
            chat.pick(m.id);
            close();
          }}>{m.id === chat.c.active.model ? "✓ " : ""}{m.id}</button
        >
      {:else}
        <p>{t("No models match your search")}</p>
      {/each}
    </div>
    <p class="palette-footer">{t("↑↓ move · ↩ switch · esc close")}</p>
  {/snippet}
</Dialog>
