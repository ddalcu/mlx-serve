<script lang="ts">
  import { loadNotice, type Status } from "../lib/state/connection.svelte";
  import type { Model } from "../lib/core/models";
  import { t } from "../lib/i18n/i18n";
  import Menu from "./Menu.svelte";

  // One current-model card, a switcher and a sentence about whether it is resident.
  let { id, models, selected, status, detail, server, onchoose }: { id: string; models: Model[]; selected: string; status: Status; detail: string; server: string; onchoose: (id: string) => void } = $props();
  const current = $derived(models.find((m) => m.id === selected));
  const loaded = $derived(status === "online" && !!current?.loaded);
  const state = $derived(status !== "online" ? (status === "checking" ? t("Checking server…") : t("Server unavailable")) : loaded ? t("Loaded on server") : selected ? t("Loads when generating") : t("Choose a model"));
</script>

<section class="image-section media-chooser">
  <h2>{t("Model")}</h2>
  <div class="image-model-card"><strong>{selected.split("/").at(-1) || t("Select a model")}</strong><span class="field-note">{detail}</span></div>
  <div class="media-model-actions">
    <Menu {id} label={t("Change Model")} {onchoose}>
      {#each models as m (m.id)}
        <button type="button" role="menuitemradio" tabindex="-1" aria-checked={m.id === selected} data-value={m.id}>{m.id}</button>
      {:else}
        <span class="field-note">{t("No supported models on this server")}</span>
      {/each}
    </Menu>
    <span class="field-note"><span class="status-dot" data-status={loaded ? "online" : "idle"}></span>{state}</span>
  </div>
  <p class="field-note model-load-notice">{loadNotice(current, server)}</p>
</section>
