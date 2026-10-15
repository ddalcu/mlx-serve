<script lang="ts">
  import Icon from "../../components/Icon.svelte";
  import { t } from "../../lib/i18n/i18n";
  import { language } from "../../lib/i18n/language.svelte";
  import { clock, type Track, type WavInfo } from "../../lib/state/amp-player.svelte";
  import { rowGlyph } from "../../lib/state/deck";
  import { coverColors, dayGroups, modelName, trackTitle } from "../../lib/state/track-meta";

  // The history in the Modern look, for the Voice, Music and Sound tabs: clips grouped by day, each with its cover,
  // title, model, time and length. A row plays or stops; Download sits beside it, never inside the row's button.
  type Props = {
    id: string;
    rows: Track[];
    /** The id of the track that plays, if any. */
    current?: string | null;
    paused?: boolean;
    /** A running AI Radio owns playback: rows only offer Download. */
    locked?: boolean;
    info: (row: Track) => Promise<WavInfo | null>;
    onplay: (row: Track) => void;
    onstop?: () => void;
    ondownload: (row: Track) => void;
  };
  let { id, rows, current = null, paused = false, locked = false, info, onplay, onstop, ondownload }: Props = $props();

  let hovered = $state<string | null>(null);
  let lengths = $state<Record<string, string>>({});
  const days = $derived(dayGroups(rows, (r) => new Date(r.createdAt), new Date(), language.current));
  const titleOf = (r: Track) => trackTitle(r.prompt ?? "") ?? modelName(r.model);

  // Each row's length, read once from its header.
  $effect(() => {
    for (const r of rows.slice(0, 200))
      if (r.id && !(r.id in lengths)) void info(r).then((i) => (lengths = { ...lengths, [r.id!]: i ? clock(i.seconds) : "" }));
  });
</script>

<section {id} class="modern-history" aria-label={t("History")}>
  <h2>{t("History")} <span>{rows.length}</span></h2>
  <div class="modern-history-list">
    {#each days as day (day.id)}
      <h3>{day.label}</h3>
      {#each day.items as r (r.id)}
        {@const playing = current !== null && current === r.id}
        {@const glyph = rowGlyph(playing, paused, hovered === r.id, locked)}
        {@const [a, b] = coverColors(r.id ?? String(r.createdAt))}
        {@const time = new Date(r.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
        <div class="modern-row" class:playing role="group" aria-label={titleOf(r)} onpointerenter={() => (hovered = r.id ?? null)} onpointerleave={() => hovered === r.id && (hovered = null)}>
          <button type="button" disabled={locked} title={r.prompt} onclick={() => (playing && onstop ? onstop() : onplay(r))}>
            <span class="modern-tile" style="background:linear-gradient(135deg,{a},{b})">
              {#if glyph.icon}<span class:moving={glyph.moving}><Icon name={glyph.icon} filled={glyph.icon !== "audio-lines"} /></span>{/if}
            </span>
            <span class="modern-row-text"><strong>{titleOf(r)}</strong><small>{[modelName(r.model), time].filter(Boolean).join(" · ")}</small></span>
          </button>
          {#if hovered === r.id || playing}
            <button type="button" class="icon-button" aria-label={t("Download %@", [titleOf(r)])} title={t("Download WAV")} onclick={() => ondownload(r)}><Icon name="download" /></button>
          {/if}
          <span class="modern-length">{lengths[r.id!] ?? ""}</span>
        </div>
      {/each}
    {:else}
      <p class="field-note">{t("No audio saved on this server yet.")}</p>
    {/each}
  </div>
</section>
