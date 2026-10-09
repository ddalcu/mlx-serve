<script lang="ts">
  import Icon from "../../components/Icon.svelte";
  import type { Library, LibraryItem } from "../../lib/core/library";
  import { t } from "../../lib/i18n/i18n";

  // The clips saved for this server, newest first.
  let { library, server, onpick }: { library: Library; server: string; onpick: (item: LibraryItem) => void } = $props();
  const STEP = 60;
  let limit = $state(STEP);
  let rows = $state<Omit<LibraryItem, "blob">[]>([]);
  let failed = $state(false);

  $effect(() => {
    let live = true;
    failed = false;
    library.list({ type: "video", server }).then(
      (found) => live && (rows = found),
      () => live && (failed = true),
    );
    return () => (live = false);
  });
</script>

{#if failed}
  {t("Could not read video saved in this browser.")}
{:else}
  <div class="audio-history">
    {#each rows.slice(0, limit) as r (r.id)}
      <div class="audio-history-row">
        <button
          type="button"
          onclick={async () => {
            const item = await library.get(r.id);
            if (item) onpick(item);
          }}><Icon name="film" /><span>{r.prompt || t("Video")}<small>{r.model} · {new Date(r.createdAt).toLocaleString()}</small></span><Icon name="play" /></button
        >
      </div>
    {:else}
      <p class="field-note">{t("No video saved on this server yet.")}</p>
    {/each}
  </div>
  {#if rows.length > limit}<button type="button" id="video-more" onclick={() => (limit += STEP)}>{t("Show more")}</button>{/if}
{/if}
