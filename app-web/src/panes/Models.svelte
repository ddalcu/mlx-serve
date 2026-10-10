<script lang="ts">
  import { displayName, t } from "../lib/i18n/i18n";
  import type { App } from "../lib/app.svelte";
  import Icon from "../components/Icon.svelte";

  let { app }: { app: App } = $props();
  const connection = $derived(app.connection);
  let query = $state("");
  const shown = $derived(connection.models.filter((m) => m.id.toLowerCase().includes(query.toLowerCase())));
</script>

<section class="models-screen">
  <div class="section-bar">
    <span class="selected-chip"><Icon name="layers" />{t("My Models")}</span>
    <button id="refresh-models" onclick={() => void connection.refresh()}><Icon name="refresh-cw" />{t("Refresh")}</button>
  </div>
  <div class="catalogue-content">
    <h1>{t("Models")}</h1>
    <p id="catalogue-server" class="section-description">{connection.serverName()} · {connection.active.url}</p>
    <label class="search-field">
      <Icon name="search" />
      <input id="model-search" type="search" bind:value={query} placeholder={t("Filter your models…")} aria-label={t("Filter your models")} />
    </label>
    <div id="catalogue" aria-live="polite">
      {#if connection.status === "checking"}
        <p class="empty-message">{t("Loading models…")}</p>
      {:else if connection.status === "error"}
        <p class="error-message">{connection.message}</p>
      {:else if !shown.length}
        <div class="empty-state">
          <Icon name="layers" />
          <h2>{query ? t("No models match your search") : t("No models on this server")}</h2>
          <p>{query ? t("Try a different model name.") : t("Choose another server in Settings, or add models in the Mac app.")}</p>
        </div>
      {:else}
        <table>
          <thead>
            <tr>
              <th>{t("Model")}</th>
              <th>{t("Capability")}</th>
              <th><span class="sr-only">{t("Selection")}</span></th>
            </tr>
          </thead>
          <tbody>
            {#each shown as m (m.id)}
              <tr>
                <td>{m.id}</td>
                <td>{m.capabilities.map(displayName).join(", ") || t("Not advertised")}</td>
                <td>
                  <button aria-label={t("Select %@", [m.id])} onclick={() => connection.chooseModel(m.id)}>
                    {#if connection.selectedModel === m.id}<Icon name="check" />{t("Selected")}{:else}{t("Use")}{/if}
                  </button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
    <p class="native-note">{t("Downloading, loading and deleting models are available in the Mac app.")}</p>
  </div>
</section>
