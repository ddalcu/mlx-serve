<script lang="ts">
  import Icon from "../../components/Icon.svelte";
  import type { ItemType, Library, LibraryItem } from "../../lib/core/library";
  import { download } from "../../lib/download";
  import { t } from "../../lib/i18n/i18n";

  // The clips saved for this server and tab, newest first.
  let { library, server, type, onpick }: { library: Library; server: string; type: ItemType; onpick: (item: LibraryItem) => void } = $props();
  const STEP = 60;
  let limit = $state(STEP);
  let rows = $state<Omit<LibraryItem, "blob">[]>([]);
  let failed = $state(false);

  $effect(() => {
    let live = true;
    failed = false;
    library.list({ server, type }).then(
      (found) => live && (rows = found),
      () => live && (failed = true),
    );
    return () => (live = false);
  });

  async function open(id: string, save: boolean) {
    const item = await library.get(id);
    if (!item) return;
    if (save) download(item.blob, `${type}-${item.createdAt}.wav`);
    else onpick(item);
  }
</script>

{#if failed}
  {t("Could not read audio saved in this browser.")}
{:else}
  <h2>{t("History")}</h2>
  <div class="audio-history">
    {#each rows.slice(0, limit) as r (r.id)}
      <div class="audio-history-row">
        <button type="button" onclick={() => void open(r.id, false)}><Icon name="audio-lines" /><span>{r.prompt || r.model}</span><Icon name="play" /></button>
        <button type="button" aria-label={t("Download %@", [r.prompt || r.model])} onclick={() => void open(r.id, true)}><Icon name="download" /></button>
      </div>
    {:else}
      <p class="field-note">{t("No audio saved on this server yet.")}</p>
    {/each}
  </div>
  {#if rows.length > limit}<button id="audio-more" type="button" onclick={() => (limit += STEP)}>{t("Show more")}</button>{/if}
{/if}
