<script lang="ts">
  import { onMount, setContext, tick, untrack } from "svelte";
  import Icon from "./components/Icon.svelte";
  import SessionList from "./components/SessionList.svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import { App, appKey, titles } from "./lib/app.svelte";
  import { language } from "./lib/i18n/language.svelte";
  import { setLanguage, t } from "./lib/i18n/i18n";
  import type { View } from "./lib/state/router.svelte";
  import Api from "./panes/Api.svelte";
  import Audio from "./panes/Audio.svelte";
  import Chat from "./panes/Chat.svelte";
  import Image from "./panes/Image.svelte";
  import Library from "./panes/Library.svelte";
  import Models from "./panes/Models.svelte";
  import Monitoring from "./panes/Monitoring.svelte";
  import Settings from "./panes/Settings.svelte";
  import Video from "./panes/Video.svelte";

  let { app }: { app: App } = $props();
  // The app object is fixed for the page's life, so capturing it once is intended.
  // svelte-ignore state_referenced_locally
  setContext(appKey, app);
  // svelte-ignore state_referenced_locally
  const { connection, prefs, router, chat, image, audio, video, monitor } = app;
  const paneTitle = $derived(router.view === "chat" ? (chat.c.active.title === "New Chat" ? t("New Chat") : chat.c.active.title) : t(titles[router.view]));
  let content = $state<HTMLElement>();

  function go(view: View) {
    app.go(view);
    void tick().then(() => content?.focus());
  }
  const closeSidebar = () => {
    app.sidebarOpen = false;
    document.getElementById("sidebar-toggle")?.focus();
  };

  $effect(() => {
    const root = document.documentElement;
    root.dataset.theme = prefs.resolvedTheme;
    root.dataset.accent = prefs.accent;
    root.dataset.textSize = prefs.textSize;
    root.dataset.column = prefs.column;
    root.dataset.compact = String(prefs.compact);
    root.lang = language.current;
  });
  $effect(() => {
    document.body.dataset.workspace = router.view;
    document.title = t("%@ — MLX Serve Studio", [paneTitle]);
  });
  $effect(() => {
    document.body.classList.toggle("sidebar-open", app.sidebarOpen);
    document.body.classList.toggle("sidebar-hidden", !app.sidebarOpen);
    if (app.sidebarOpen && app.phone) untrack(() => document.getElementById("close-sidebar")?.focus());
  });
  // The server summary follows whichever server is selected.
  $effect(() => {
    connection.activeId;
    untrack(() => void connection.refreshInfo());
  });
  // The chat follows the server: an unsent chat retargets and an empty model is filled in.
  $effect(() => {
    connection.active.url;
    connection.models;
    connection.selectedModel;
    untrack(() => {
      chat.connectionChanged();
      image.connectionChanged();
      audio.connectionChanged();
      video.connectionChanged();
      monitor.connectionChanged();
    });
  });
  $effect(() => app.storage.setItem("studio.activeChat", chat.c.active.id));

  onMount(() => {
    void chat.init();
    void image.init();
    void audio.init();
    void video.init();
    void connection.refresh();
    // Read-only summaries stop while hidden; requests never overlap.
    const info = setInterval(() => !document.hidden && void connection.refreshInfo(), 5000);
    const status = setInterval(() => !document.hidden && !chat.c.run && !connection.pending && void connection.refresh(), 15000);
    const visibility = () => {
      if (!document.hidden) {
        void connection.refreshInfo();
        void connection.refresh();
      } else {
        connection.infoPending?.abort();
        connection.infoPending = undefined;
      }
    };
    document.addEventListener("visibilitychange", visibility);
    return () => {
      clearInterval(info);
      clearInterval(status);
      document.removeEventListener("visibilitychange", visibility);
    };
  });

  function keydown(event: KeyboardEvent) {
    if (!app.phone) return;
    if (event.key === "Escape" && !document.querySelector("dialog[open]")) closeSidebar();
    if (event.key === "Tab" && app.sidebarOpen) {
      const items = [...document.querySelectorAll<HTMLElement>("#sidebar button")].filter((e) => e.getClientRects().length);
      const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
  }
</script>

<svelte:document onkeydown={keydown} />

<a
  class="skip-link"
  href="#content"
  onclick={(e) => {
    e.preventDefault();
    content?.focus();
  }}>{t("Skip to content")}</a
>
<Sidebar {app} {go} newChat={() => chat.newChat()}>
  {#snippet sessions()}<SessionList />{/snippet}
</Sidebar>
<button class="scrim" id="scrim" aria-label={t("Close sidebar")} hidden={!app.sidebarOpen || !app.phone} onclick={closeSidebar}></button>
<div class="workspace" inert={app.sidebarOpen && app.phone}>
  <header class="topbar">
    <button class="icon-button" id="sidebar-toggle" aria-label={t("Toggle sidebar")} aria-controls="sidebar" aria-expanded={app.sidebarOpen} onclick={() => (app.sidebarOpen = !app.sidebarOpen)}><Icon name="panel-left" /></button>
    <span id="pane-title">{paneTitle}</span>
    <button class="icon-button" id="language-toggle" aria-label={t("Language")} title={t("Language")} onclick={() => setLanguage(language.current === "en" ? "zh-Hans" : "en")}>{language.current === "en" ? "中" : "EN"}</button>
  </header>
  <p class="storage-warning" id="storage-warning" role="status" hidden={!app.storage.warning}>{app.storage.warning}</p>
  <main id="content" tabindex="-1" bind:this={content}>
    {#if router.view === "chat"}
      <Chat {app} />
    {:else if router.view === "image"}
      <Image {app} />
    {:else if router.view === "audio"}
      <Audio {app} />
    {:else if router.view === "video"}
      <Video {app} />
    {:else if router.view === "library"}
      <Library {app} />
    {:else if router.view === "monitoring"}
      <Monitoring {app} />
    {:else if router.view === "settings"}
      <Settings {app} />
    {:else if router.view === "models"}
      <Models {app} />
    {:else if router.view === "api"}
      <Api {app} />
    {:else}
      <section class="empty-state"><h2>{t(titles[router.view])}</h2></section>
    {/if}
  </main>
</div>
