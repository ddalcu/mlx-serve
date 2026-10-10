<script lang="ts">
  import { getApp } from "../lib/app.svelte";
  import { t } from "../lib/i18n/i18n";

  // A tool call's saved result, read back from the Library when it scrolls into the page.
  let { id }: { id: string } = $props();
  const library = getApp().library;
  type Loaded = { kind: "img" | "audio" | "video"; url: string; type: string; prompt?: string };
  let state = $state<"loading" | "missing" | "failed" | Loaded>("loading");

  $effect(() => {
    let url: string | undefined;
    let current = true;
    state = "loading";
    library
      .get(id)
      .then((item) => {
        if (!current) return;
        if (!item || !["image", "speech", "music", "sound", "video"].includes(item.type)) return void (state = "missing");
        url = URL.createObjectURL(item.blob);
        state = { kind: item.type === "image" ? "img" : item.type === "video" ? "video" : "audio", url, type: item.type, prompt: item.prompt };
      })
      .catch(() => current && (state = "failed"));
    return () => {
      current = false;
      if (url) URL.revokeObjectURL(url);
    };
  });
</script>

<div class="tool-media">
  {#if state === "loading"}
    <p>{t("Loading saved media…")}</p>
  {:else if state === "missing"}
    {t("Saved media is no longer in the Library.")}
  {:else if state === "failed"}
    {t("Could not read saved media.")}
  {:else}
    {#if state.kind === "img"}
      <img src={state.url} alt={state.prompt || t("Generated image")} />
    {:else if state.kind === "video"}
      <!-- svelte-ignore a11y_media_has_caption -->
      <video src={state.url} controls preload="metadata"></video>
    {:else}
      <audio src={state.url} controls preload="metadata"></audio>
    {/if}
    <a href={state.url} download="{state.type}-{id}.{state.kind === 'img' ? 'png' : state.kind === 'video' ? 'mp4' : 'wav'}">{t("Download")}</a>
  {/if}
</div>
