<script lang="ts">
  import { tick, type Snippet } from "svelte";
  import type { App, SettingsCategory } from "../lib/app.svelte";
  import { languageChoice, N, setLanguage, t } from "../lib/i18n/i18n";
  import type { IconName } from "../lib/icons";
  import { choices } from "../lib/state/preferences.svelte";
  import Icon from "../components/Icon.svelte";

  let { app }: { app: App } = $props();
  const connection = $derived(app.connection);
  const prefs = $derived(app.prefs);

  const categories: [SettingsCategory, string, IconName][] = [
    ["all", N("All Settings"), "settings"],
    ["interface", N("Interface"), "palette"],
    ["servers", N("Servers"), "server"],
    ["about", N("About"), "info"],
  ];
  const labels = $derived({
    theme: [t("System"), t("Light"), t("Dark")],
    accent: [t("System"), t("Blue"), t("Purple"), t("Pink"), t("Red"), t("Orange"), t("Yellow"), t("Green"), t("Graphite")],
    textSize: [t("Small"), t("Default"), t("Large"), t("Extra Large")],
    column: [t("Narrow"), t("Medium"), t("Wide")],
  });

  let query = $state("");
  let empty = $state(false);
  let screen: HTMLElement;

  // A section stays visible while its text matches the search, else only the chosen category shows.
  $effect(() => {
    const needle = query.trim().toLowerCase();
    const category = app.settingsCategory;
    let visible = 0;
    for (const section of screen.querySelectorAll<HTMLElement>(".settings-section")) {
      section.hidden = needle ? !section.textContent?.toLowerCase().includes(needle) : category !== "all" && section.dataset.section !== category;
      if (!section.hidden) visible++;
    }
    empty = visible === 0;
  });

  let editing = $state("");
  let name = $state("");
  let url = $state("");
  let key = $state("");
  let remember = $state(false);
  let error = $state("");

  function reset() {
    editing = name = url = key = error = "";
    remember = false;
  }

  function edit() {
    const s = connection.active;
    editing = s.id;
    name = s.name;
    url = s.url;
    key = s.apiKey ?? "";
    remember = s.rememberKey;
    void tick().then(() => document.getElementById("server-name")?.focus());
  }

  function save(event: SubmitEvent) {
    event.preventDefault();
    try {
      const input = { name: name.trim() || undefined, url, apiKey: key || undefined, rememberKey: remember };
      const id = editing ? (connection.update(editing, input), editing) : connection.add(input);
      connection.select(id);
      reset();
      void connection.refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : t("Could not save server.");
    }
  }

  const statusText = (status?: string) =>
    status === "online" ? t("Connected") : status === "error" ? t("Connection failed") : status === "checking" ? t("Checking…") : t("Not checked");
  const about = $derived(t("The web console built into %@.").split("%@"));
</script>

{#snippet setting(id: string, title: string, text: string, control: Snippet)}
  <div class="settings-row">
    <div>
      <label id="{id}-label" for={id}>{title}</label>
      <p>{text}</p>
    </div>
    {@render control()}
  </div>
{/snippet}

{#snippet segmented(field: "theme" | "column", label: string)}
  <div class="segmented" role="group" aria-label={label}>
    {#each choices[field] as value, i}
      <button aria-pressed={prefs[field] === value} onclick={() => prefs.set(field, value as never)}>{labels[field][i]}</button>
    {/each}
  </div>
{/snippet}

{#snippet pick(field: "accent" | "textSize")}
  <select id={field} value={prefs[field]} onchange={(e) => prefs.set(field, e.currentTarget.value as never)}>
    {#each choices[field] as value, i}<option {value}>{labels[field][i]}</option>{/each}
  </select>
{/snippet}

<section class="settings-screen" aria-label={t("Settings")} bind:this={screen}>
  <nav class="settings-categories" aria-label={t("Settings categories")}>
    {#each categories as [id, label, icon]}
      <button class="nav-row" aria-current={app.settingsCategory === id ? "page" : undefined} onclick={() => (app.settingsCategory = id)}><Icon name={icon} />{t(label)}</button>
    {/each}
  </nav>
  <div class="settings-main">
    <div class="settings-search">
      <label>
        <Icon name="search" />
        <input type="search" id="settings-search" bind:value={query} placeholder={t("Search settings…")} aria-label={t("Search settings")} />
      </label>
    </div>
    <div class="settings-scroll">
      <div class="settings-section" data-section="interface">
        <h1>{t("Interface")}</h1>
        {#snippet languagePicker()}
          <select id="language" aria-label={t("Language")} value={languageChoice()} onchange={(e) => setLanguage(e.currentTarget.value)}>
            <option value="system">{t("System")}</option>
            <option value="en">English</option>
            <option value="zh-Hans">简体中文</option>
          </select>
        {/snippet}
        {@render setting("language", t("Language"), t("Choose the language for this interface."), languagePicker)}
        {#snippet theme()}{@render segmented("theme", t("Appearance"))}{/snippet}
        {@render setting("theme", t("Appearance"), t("Follow the system setting, or force light/dark for this app only."), theme)}
        {#snippet accent()}{@render pick("accent")}{/snippet}
        {@render setting("accent", t("Accent Color"), t("Tint for buttons, links and the selected message bubble."), accent)}
        {#snippet textSize()}{@render pick("textSize")}{/snippet}
        {@render setting("textSize", t("Text Size"), t("Size of the chat transcript’s prose and code."), textSize)}
        {#snippet column()}{@render segmented("column", t("Chat Column"))}{/snippet}
        {@render setting("column", t("Chat Column"), t("Narrow and Medium use fixed widths; Wide follows the window."), column)}
        {#snippet compact()}
          <input id="compact" type="checkbox" role="switch" checked={prefs.compact} onchange={(e) => prefs.set("compact", e.currentTarget.checked)} />
        {/snippet}
        {@render setting("compact", t("Compact Mode"), t("Tighter spacing between messages — more of the conversation on screen."), compact)}
      </div>

      <div class="settings-section" data-section="servers">
        <h1>{t("Servers")}</h1>
        <p class="section-description">{t("Connect to an mlx-serve server. This page’s server is used by default.")}</p>
        <div id="server-list">
          {#each connection.servers as s (s.id)}
            {@const check = connection.checks[s.id]}
            <div class="server-row">
              <button
                class="server-choice"
                aria-pressed={s.id === connection.activeId}
                onclick={() => {
                  connection.select(s.id);
                  void connection.refresh();
                }}
              >
                <span class="status-dot" data-status={check?.status ?? "idle"}></span>
                <span><strong>{connection.serverName(s.name)}</strong><small>{s.url} · {statusText(check?.status)}</small></span>
                {#if s.id === connection.activeId}<Icon name="check" />{/if}
              </button>
              <button
                class="icon-button remove-server"
                aria-label={t("Remove %@", [connection.serverName(s.name)])}
                onclick={() => {
                  connection.remove(s.id);
                  void connection.refresh();
                }}><Icon name="trash" /></button
              >
            </div>
          {/each}
        </div>
        <div class="server-actions">
          <button id="refresh-server" onclick={() => void connection.refresh()}><Icon name="refresh-cw" />{t("Check connection")}</button>
          <button id="edit-server" onclick={edit}>{t("Edit selected server")}</button>
        </div>
        <p id="server-status" class="connection-detail" class:error-message={connection.status === "error"} role="status">{connection.message}</p>
        <form id="server-form" onsubmit={save}>
          <h2 id="server-form-title">{editing ? t("Edit Server") : t("Add Server")}</h2>
          <label>{t("Name")} <input id="server-name" autocomplete="off" bind:value={name} placeholder={t("My server")} /></label>
          <label>{t("Server URL")} <input id="server-url" type="url" required bind:value={url} placeholder="http://server:11234" autocomplete="url" /></label>
          <label>{t("API key")} <input id="server-key" type="password" autocomplete="off" spellcheck="false" bind:value={key} placeholder={t("Optional")} /></label>
          <label class="check-label"><input type="checkbox" id="remember-key" bind:checked={remember} />{t("Remember key on this device")}</label>
          <p class="field-note">{t("Keys last for this session by default. Browser storage is not a keyring.")}</p>
          <p id="form-error" role="alert">{error}</p>
          <div class="form-actions">
            <button type="submit" class="primary" id="save-server">{editing ? t("Save") : t("Add Server")}</button>
            <button type="button" id="cancel-edit" hidden={!editing} onclick={reset}>{t("Cancel")}</button>
          </div>
        </form>
      </div>

      <div class="settings-section" data-section="about">
        <h1>{t("About")}</h1>
        <div class="settings-row">
          <div>
            <h2>MLX Serve Studio {#if connection.serverVersion}<span class="field-note">{connection.serverVersion}</span>{/if}</h2>
            <p>{about[0]}<a href="https://github.com/ddalcu/mlx-serve" target="_blank" rel="noopener noreferrer">mlx-serve</a>{about[1]}</p>
            <p>
              <a href="https://github.com/ddalcu/mlx-serve/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">{t("License (MIT)")}</a> ·
              <a href="https://github.com/ddalcu/mlx-serve/blob/main/NOTICE" target="_blank" rel="noopener noreferrer">{t("Third-party notices")}</a>
            </p>
          </div>
        </div>
      </div>
      <p id="settings-empty" hidden={!empty}>{t("No settings match your search.")}</p>
      <p class="native-note">{t("Server launch, model management and other native settings are available in the Mac app.")}</p>
    </div>
  </div>
</section>
