<script lang="ts">
  import { apiReference, curlExample } from "../lib/core/console";
  import { t } from "../lib/i18n/i18n";
  import type { App } from "../lib/app.svelte";

  let { app }: { app: App } = $props();
  const connection = $derived(app.connection);
  const base = $derived(connection.active.url);
</script>

<section class="api-screen">
  <h1>{t("API Reference")}</h1>
  <p>{t("mlx-serve endpoints. Host management routes are documented here; Studio does not execute them.")}</p>
  <h2>{t("Quick start")}</h2>
  <pre>{curlExample(base, connection.models)}</pre>
  <p>{t("Add your bearer header privately if the server requires a key. Keys are never included in these examples or links.")}</p>
  <div class="api-table">
    <table>
      <thead>
        <tr><th>{t("Method")}</th><th>{t("Path")}</th><th>{t("Description")}</th></tr>
      </thead>
      <tbody>
        {#each apiReference as row}
          <tr>
            <td>{row.method}</td>
            <td>
              <code>
                {#if row.method === "GET" && !row.path.includes("{")}
                  <a href={base + row.path} target="_blank" rel="noopener noreferrer">{row.path}</a>
                {:else}{row.path}{/if}
              </code>
            </td>
            <td>{t(row.description)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>
