<script lang="ts">
  import type { Library, LibraryItem } from "../lib/core/library";
  import { t } from "../lib/i18n/i18n";

  // The pictures saved for this server, newest first, drawn lazily while the gallery is open.
  let { library, server, onpick }: { library: Library; server: string; onpick: (item: LibraryItem) => void } = $props();
  const STEP = 60;
  let limit = $state(STEP);
  let total = $state(0);
  let cards = $state<{ item: LibraryItem; url: string }[]>([]);
  let failed = $state(false);

  $effect(() => {
    const max = limit;
    let live = true;
    const urls: string[] = [];
    failed = false;
    (async () => {
      try {
        const rows = await library.list({ type: "image", server });
        const items = await Promise.all(rows.slice(0, max).map((r) => library.get(r.id)));
        if (!live) return;
        total = rows.length;
        cards = items
          .filter((i): i is LibraryItem => i?.blob.type === "image/png")
          .map((item) => {
            const url = URL.createObjectURL(item.blob);
            urls.push(url);
            return { item, url };
          });
      } catch {
        if (live) failed = true;
      }
    })();
    return () => {
      live = false;
      for (const url of urls) URL.revokeObjectURL(url);
    };
  });
</script>

{#if failed}
  {t("Could not read saved images in this browser.")}
{:else}
  <h2>{t("Images")}</h2>
  <p class="field-note">{t("Saved in this browser · %@ result(s)", [total])}</p>
  <div class="image-gallery-grid">
    {#each cards as { item, url } (item.id)}
      <button class="image-gallery-card" onclick={() => onpick(item)}><img src={url} alt={item.prompt} /><span>{item.prompt || item.model}</span></button>
    {:else}
      <p class="field-note">{t("No saved images yet.")}</p>
    {/each}
  </div>
  {#if total > limit}<button id="image-gallery-more" onclick={() => (limit += STEP)}>{t("Show more")}</button>{/if}
{/if}
