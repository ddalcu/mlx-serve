<script lang="ts">
  import { onMount } from "svelte";
  import Dialog from "../components/Dialog.svelte";
  import Icon from "../components/Icon.svelte";
  import Markdown from "../components/Markdown.svelte";
  import type { App } from "../lib/app.svelte";
  import type { ItemType, LibraryItem } from "../lib/core/library";
  import { archiveLibrary, archiveLimit, chatMarkdown, cleanSession, importArchive } from "../lib/core/library-transfer";
  import { download } from "../lib/download";
  import { N, t } from "../lib/i18n/i18n";

  let { app }: { app: App } = $props();
  const library = $derived(app.library);
  const names: Record<ItemType, string> = { chat: N("Chat"), image: N("Image"), speech: N("Voice"), music: N("Music"), sound: N("Sound Effects"), video: N("Video") };
  const size = (value: number) => (value < 1024 * 1024 ? t("%@ KB", [(value / 1024).toFixed(1)]) : t("%@ MB", [(value / 1024 / 1024).toFixed(1)]));

  let rows = $state<Omit<LibraryItem, "blob">[]>([]);
  let query = $state("");
  let type = $state("");
  let server = $state("");
  let busy = $state(false);
  let status = $state("");
  let storage = $state("");
  let importing = $state<HTMLInputElement>();
  let opened = $state<{ id: string; title: string; kind: ItemType; session?: ReturnType<typeof cleanSession>; url?: string } | null>(null);
  let deleting = $state("");
  let live = true;

  const servers = $derived([...new Set(rows.map((x) => x.server))]);
  const shown = $derived(
    rows.filter((x) => (!type || x.type === type) && (!server || x.server === server) && `${x.prompt || ""} ${x.model}`.toLowerCase().includes(query.toLowerCase())),
  );

  onMount(() => {
    void work(async () => {
      await app.chat.c.flush();
      await refresh();
      return "";
    });
    return () => {
      live = false;
      if (opened?.url) URL.revokeObjectURL(opened.url);
    };
  });

  /** Run one Library action at a time, with a status line for how it ended. */
  async function work(run: () => Promise<string>) {
    if (busy) return;
    busy = true;
    status = t("Working…");
    try {
      const message = await run();
      if (live) status = message;
    } catch (error) {
      if (live)
        status = t("Could not complete Library action: %@ If storage is full, download a backup and delete unwanted items. Imports are limited to %@.", [
          error instanceof Error ? error.message : t("Browser storage unavailable."),
          size(archiveLimit),
        ]);
    } finally {
      busy = false;
    }
  }

  async function refresh() {
    const found = await library.list();
    if (!live) return;
    rows = found;
    if (!found.some((x) => x.server === server)) server = "";
    let bytes = 0;
    for (const row of found) bytes += (await library.get(row.id))?.blob.size || 0;
    const estimate = await navigator.storage?.estimate?.().catch(() => null);
    if (live)
      storage = t("%@ items · %@ in Library%@. Browser data can be cleared; export a backup.", [
        found.length,
        size(bytes),
        estimate?.quota ? t(" · %@ of %@ browser storage used", [size(estimate.usage || 0), size(estimate.quota)]) : t(" · Browser storage estimate unavailable"),
      ]);
  }

  /** The chats on screen reload after the library changed underneath them. */
  async function chatsChanged() {
    const chat = app.chat,
      id = chat.c.active.id;
    await chat.c.load();
    if (chat.c.sessions.some((s) => s.id === id)) chat.c.select(id);
    else chat.c.create(app.connection.active.url, "");
  }

  const exportAll = () =>
    work(async () => {
      await app.chat.c.flush();
      download(await archiveLibrary(library), "mlx-serve-studio-library.json");
      return t("Library exported. JSON includes all media; keep a backup outside this browser.");
    });

  const importFile = () => {
    const file = importing?.files?.[0];
    if (importing) importing.value = "";
    if (file)
      void work(async () => {
        await app.chat.c.flush();
        const ids = await importArchive(library, file);
        await chatsChanged();
        await refresh();
        return t("Imported %@ items. Existing items were kept.", [ids.length]);
      });
  };

  const downloadItem = (id: string) =>
    work(async () => {
      const item = await library.get(id);
      if (!item) throw new Error(t("Item no longer exists."));
      if (item.type === "chat") {
        const s = cleanSession(JSON.parse(await item.blob.text()), id);
        download(new Blob([chatMarkdown(s)], { type: "text/markdown" }), "chat.md");
      } else {
        const file = await library.export(id);
        download(file.blob, file.filename);
      }
      return t("Download ready.");
    });

  const open = (id: string) =>
    work(async () => {
      const item = await library.get(id);
      if (!item) throw new Error(t("Item no longer exists."));
      if (!live) return "";
      if (opened?.url) URL.revokeObjectURL(opened.url);
      opened =
        item.type === "chat"
          ? { id, title: item.prompt || t(names.chat), kind: item.type, session: cleanSession(JSON.parse(await item.blob.text()), id) }
          : { id, title: item.prompt || t(names[item.type]), kind: item.type, url: URL.createObjectURL(item.blob) };
      return "";
    });

  const closeItem = () => {
    if (opened?.url) URL.revokeObjectURL(opened.url);
    opened = null;
  };

  function continueChat(id: string) {
    const chat = app.chat,
      session = chat.c.sessions.find((s) => s.id === id);
    if (!session) return;
    const found = app.connection.servers.find((s) => s.url === session.server);
    if (!found) {
      status = t("Add this conversation’s server in Settings before continuing it.");
      return;
    }
    if (app.connection.activeId !== found.id) {
      app.connection.select(found.id);
      void app.connection.refresh();
    }
    chat.c.select(id);
    app.go("chat");
  }

  const remove = (id: string) =>
    work(async () => {
      await app.chat.c.flush();
      await library.delete(id);
      await chatsChanged();
      await refresh();
      return t("Item deleted.");
    });
</script>

<section class="library-screen">
  <div class="section-bar">
    <h1>{t("Library")}</h1>
    <button id="library-export" disabled={busy} onclick={() => void exportAll()}>{t("Export JSON")}</button>
    <button id="library-import" disabled={busy} onclick={() => importing?.click()}>{t("Import JSON…")}</button>
    <input id="library-file" type="file" accept="application/json,.json" hidden bind:this={importing} onchange={importFile} />
  </div>
  <p class="section-description">{t("Your conversations and creations, saved in this browser.")}</p>
  <div class="library-filters">
    <label class="search-field">
      <Icon name="search" />
      <input id="library-search" type="search" aria-label={t("Search library")} placeholder={t("Search prompts or models…")} bind:value={query} />
    </label>
    <label>
      {t("Type")}
      <select id="library-type" bind:value={type}>
        <option value="">{t("All types")}</option>
        {#each Object.entries(names) as [id, name]}<option value={id}>{t(name)}</option>{/each}
      </select>
    </label>
    <label>
      {t("Server")}
      <select id="library-server" bind:value={server}>
        <option value="">{t("All servers")}</option>
        {#each servers as s}<option value={s}>{s}</option>{/each}
      </select>
    </label>
  </div>
  <p id="library-storage" class="field-note">{storage}</p>
  <p id="library-status" role="status">{status}</p>
  <div id="library-items" class="library-items">
    {#each shown as x (x.id)}
      <article class="library-item">
        <span class="library-kind">{t(names[x.type])}</span>
        <button class="library-title" disabled={busy} onclick={() => void open(x.id)}>{(x.type === "chat" && x.prompt === "New Chat" ? t("New Chat") : x.prompt) || x.model || t(names[x.type])}</button>
        <p>{x.model}</p>
        <p>{x.server} · {new Date(x.createdAt).toLocaleString()}</p>
        <div class="library-actions">
          <button disabled={busy} onclick={() => void downloadItem(x.id)}>{t("Download%@", [x.type === "chat" ? t(" Markdown") : ""])}</button>
          <button disabled={busy} onclick={() => (deleting = x.id)}>{t("Delete…")}</button>
        </div>
      </article>
    {:else}
      <div class="empty-state">
        <Icon name="library" />
        <h2>{t("No items")}</h2>
        <p>{t("Your saved conversations and creations appear here.")}</p>
      </div>
    {/each}
  </div>
</section>

{#if deleting}
  <Dialog title={t("Library item")} base="library-dialog" bare onclose={() => (deleting = "")}>
    {#snippet children(close)}
      <h2>{t("Delete this item?")}</h2>
      <p>{t("This removes the saved copy from this browser. Download it first if you want to keep it.")}</p>
      <div class="form-actions">
        <button onclick={close}>{t("Cancel")}</button>
        <button
          id="library-confirm-delete"
          onclick={() => {
            const id = deleting;
            close();
            void remove(id);
          }}>{t("Delete")}</button
        >
      </div>
    {/snippet}
  </Dialog>
{/if}

{#if opened}
  {@const item = opened}
  <Dialog title={t("Library item")} base="library-dialog" bare onclose={closeItem}>
    {#snippet children(close)}
      <div class="section-bar"><h2>{item.title}</h2><button onclick={close}>{t("Close")}</button></div>
      {#if item.session}
        <button
          id="library-open-chat"
          onclick={() => {
            const id = item.id;
            close();
            continueChat(id);
          }}>{t("Open conversation")}</button
        >
        <div class="library-transcript">
          {#each item.session.messages as m}
            <article>
              <h3>{m.role === "user" ? t("You") : t("Assistant")}</h3>
              <div class="markdown"><Markdown source={m.text} /></div>
              {#each m.images ?? [] as image}{#if /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+=*$/.test(image)}<img src={image} alt={t("Attachment")} />{/if}{/each}
            </article>
          {/each}
        </div>
      {:else if item.kind === "image"}
        <img id="library-media" alt={t("Saved image")} src={item.url} />
      {:else if item.kind === "video"}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video id="library-media" controls preload="metadata" src={item.url}></video>
      {:else}
        <audio id="library-media" controls preload="metadata" src={item.url}></audio>
      {/if}
    {/snippet}
  </Dialog>
{/if}
