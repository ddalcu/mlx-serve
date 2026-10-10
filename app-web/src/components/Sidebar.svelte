<script lang="ts">
  import type { Snippet } from "svelte";
  import { N, t } from "../lib/i18n/i18n";
  import type { App } from "../lib/app.svelte";
  import type { IconName } from "../lib/icons";
  import type { View } from "../lib/state/router.svelte";
  import Icon from "./Icon.svelte";
  import NavRow from "./NavRow.svelte";

  let { app, go, newChat, sessions }: { app: App; go: (view: View) => void; newChat: () => void; sessions?: Snippet } = $props();
  const connection = $derived(app.connection);
  const router = $derived(app.router);

  const info = $derived(
    t("Server %@ · %@ models · %@ ready · Resident %@", [
      connection.serverVersion || t("version unavailable"),
      connection.models.length,
      connection.models.filter((m) => (m.state ? m.state === "ready" : m.loaded)).length,
      connection.residentBytes === null ? "unavailable" : (connection.residentBytes / 1024 ** 3).toFixed(2) + " GiB",
    ]),
  );
  const main: [View, string, IconName][] = [
    ["models", N("Models"), "layers"],
    ["monitoring", N("Monitoring"), "activity"],
    ["api", N("API"), "info"],
    ["settings", N("Settings"), "settings"],
  ];
  const create: [View, string, IconName][] = [
    ["image", N("Image Generation"), "image"],
    ["video", N("Video Generation"), "film"],
    ["audio", N("Audio & Music"), "audio-lines"],
  ];
</script>

<aside class="sidebar" id="sidebar" aria-label={t("Sidebar")} inert={!app.sidebarOpen}>
  <div class="sidebar-heading">
    <span>MLX Serve Studio</span>
    <button class="icon-button" id="close-sidebar" aria-label={t("Close sidebar")} onclick={() => (app.sidebarOpen = false)}><Icon name="x" /></button>
  </div>
  <nav aria-label={t("Main navigation")}>
    {#each main as [view, label, icon]}<NavRow {icon} {label} current={router.view === view} onclick={() => go(view)} />{/each}
    <h2 class="nav-heading">{t("Create")}</h2>
    {#each create as [view, label, icon]}<NavRow {icon} {label} current={router.view === view} onclick={() => go(view)} />{/each}
    <div class="library-nav"><NavRow icon="library" label={N("Library")} current={router.view === "library"} onclick={() => go("library")} /></div>
    <div class="nav-heading sessions-heading">
      <h2>{t("Sessions")}</h2>
      <button class="icon-button small" aria-label={t("New Chat")} onclick={newChat}><Icon name="plus" /></button>
    </div>
    <div id="session-list">{@render sessions?.()}</div>
  </nav>
  <button
    class="connection-pill"
    id="connection-link"
    title="{connection.active.url} — {connection.message}"
    aria-label={t("Server connection: %@, %@", [connection.serverName(), connection.message])}
    onclick={() => {
      app.settingsCategory = "servers";
      go("settings");
    }}
  >
    <span class="status-dot" id="header-dot" data-status={connection.status}></span>
    <span id="server-label">{connection.serverName()}</span>
  </button>
  <p class="sidebar-note" id="server-info">{info}</p>
  <p class="sidebar-note">{t("Saved in this browser")}</p>
</aside>
