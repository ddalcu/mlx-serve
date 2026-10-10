<script lang="ts">
  import { onMount, untrack } from "svelte";
  import Icon from "../components/Icon.svelte";
  import Menu from "../components/Menu.svelte";
  import ModelChooser from "../components/ModelChooser.svelte";
  import RewriteDialog from "../components/RewriteDialog.svelte";
  import type { App } from "../lib/app.svelte";
  import { download } from "../lib/download";
  import { t } from "../lib/i18n/i18n";
  import { videoPrompts } from "../lib/state/video-presets";
  import { qualities } from "../lib/state/video-state.svelte";
  import type { MediaKey } from "../lib/state/video-workspace.svelte";
  import SpeechDialog from "./video/SpeechDialog.svelte";
  import TipsDialog from "./video/TipsDialog.svelte";
  import VideoGallery from "./video/VideoGallery.svelte";

  let { app }: { app: App } = $props();
  const ws = $derived(app.video);
  const connection = $derived(app.connection);
  const c = $derived(ws.c);
  const d = $derived(ws.d);
  const p = $derived(ws.profile);
  const working = $derived(ws.working);
  const examples = $derived(videoPrompts[p?.h3 ? (p.references ? "h3Reference" : "h3Base") : "ltx"]);
  const frames = $derived(ws.frames);
  const validation = $derived(ws.validation);
  const names = { images: "Picture", videos: "Video", audios: "Audio" } as const;
  const lockedAudio = $derived(!!ws.inputs.audio && d.mode === "one_stage");
  const modeLabel = $derived(ws.inputs.audio && d.mode === "one_stage" ? t("2-stage (audio)") : d.mode.replace("one_stage", "1-stage").replace("two_stage_hq", "2-stage HQ").replace("two_stage", "2-stage"));

  let prompt = $state<HTMLTextAreaElement>();
  let dialog = $state<"" | "tips" | "rewrite" | "speech">("");
  let gallery = $state(false);
  let url = $state("");
  let playbackFailed = $state(false);
  let audioUrl = $state("");
  let now = $state(Date.now());
  const mp4 = $derived(!!c.result?.blob.type.includes("mp4"));
  const live = $derived.by(() => {
    const preview = c.preview?.preview;
    return typeof preview === "string" && preview.length < 2 * 1024 * 1024 && /^[A-Za-z0-9+/]+=*$/.test(preview) ? "data:image/jpeg;base64," + preview : "";
  });
  const elapsed = $derived.by(() => {
    const seconds = (now - c.started) / 1000;
    return t("%@ s elapsed%@", [Math.floor(seconds), c.step > 0 && c.total > c.step ? t(" · ~%@ s remaining", [Math.ceil((seconds / c.step) * (c.total - c.step))]) : ""]);
  });

  $effect(() => {
    ws.signature();
    untrack(() => ws.touch());
  });
  $effect(() => () => {
    ws.flush();
    ws.stopActivity();
  });
  onMount(() => {
    const clock = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(clock);
  });
  // The finished clip plays from an object URL that lives as long as that result.
  $effect(() => {
    const blob = c.phase === "completed" ? c.result?.blob : undefined;
    playbackFailed = false;
    if (!blob) return void (url = "");
    const made = URL.createObjectURL(blob);
    url = made;
    return () => URL.revokeObjectURL(made);
  });
  // The attached soundtrack previews from its own object URL.
  $effect(() => {
    const clip = ws.inputs.audio;
    if (!clip) return void (audioUrl = "");
    const bytes = Uint8Array.from(atob(clip.base64), (ch) => ch.charCodeAt(0));
    const made = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    audioUrl = made;
    return () => URL.revokeObjectURL(made);
  });

  function marker(text: string) {
    const input = prompt!,
      at = input.selectionStart;
    ws.d.prompt = input.value.slice(0, at) + text + input.value.slice(input.selectionEnd);
    input.focus();
  }
  const fileInput = (key: MediaKey) => async (event: Event) => {
    const input = event.currentTarget as HTMLInputElement;
    await ws.attach(key, Array.from(input.files || []));
    input.value = "";
  };
  const drop = (key: MediaKey) => (event: DragEvent) => {
    event.preventDefault();
    if (!ws.working) void ws.attach(key, Array.from(event.dataTransfer?.files || []));
  };
  const height = (value: number) => Math.max(70, Math.min(600, value));
</script>

{#snippet slider(id: "steps" | "cfg" | "stg" | "audioGuidance" | "refine" | "windows", title: string, min: number, max: number, step = 1, disabled = false)}
  <label class="video-slider">{title}<output id="video-{id}-value">{ws.d[id]}</output><input id="video-{id}" type="range" {min} {max} {step} {disabled} bind:value={ws.d[id]} onchange={() => ws.clamp()} /></label>
{/snippet}

{#snippet toggle(id: "turbo" | "decoder" | "best" | "preview", label: string)}
  <label class="video-toggle">
    <input
      type="checkbox"
      checked={ws.d[id]}
      onchange={(e) => {
        ws.d[id] = e.currentTarget.checked;
        if (id === "turbo") ws.d.steps = e.currentTarget.checked ? 4 : 30;
        ws.clamp();
      }}
    />{label}
  </label>
{/snippet}

{#snippet well(key: "first" | "last" | "audio", title: string, accept: string)}
  {@const input = ws.inputs[key]}
  <section>
    <h3>{title} <small>{t("optional%@", [key === "audio" ? t(", audio-to-video") : key === "first" ? t(", starts here") : t(", ends here")])}</small></h3>
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="video-well" ondragover={(e) => e.preventDefault()} ondrop={drop(key)}>
      {#if input}
        {#if key === "audio"}
          <!-- svelte-ignore a11y_media_has_caption -->
          <audio controls preload="metadata" src={audioUrl}></audio>
        {:else}
          <img alt={title} src="data:image/png;base64,{input.base64}" />
        {/if}
        <span>{input.name}</span>
        <button type="button" onclick={() => ws.remove(key)}>{t("Remove")}</button>
      {:else}
        <Icon name={key === "audio" ? "audio-lines" : "image"} />
        <label class="chip">{t("Choose %@…", [key === "audio" ? t("audio") : t("image")])}<input type="file" {accept} onchange={fileInput(key)} /></label>
        <small>{t("or drag one here")}</small>
      {/if}
    </div>
  </section>
{/snippet}

<section class="image-screen video-screen" aria-label={t("Video Generation")}>
  <div class="image-controls" id="video-controls">
    <fieldset id="video-fieldset" disabled={!ws.ready || working}>
      <ModelChooser id="video-model" models={ws.models} selected={d.model} status={connection.status} detail={p?.h3 ? t("Video · audio · references") : t("Text to video · Image to video")} server={connection.serverName()} onchoose={(id) => ws.chooseModel(id)} />
      <section class="image-section">
        <div class="image-heading">
          <label for="video-prompt">{t("Prompt")}</label>
          <button type="button" id="video-enhance" class="chip" disabled={working || !d.prompt.trim()} onclick={() => (dialog = "rewrite")}><Icon name="sparkles" />{t("Enhance…")}</button>
          <Menu id="video-templates" label={t("Templates")} onchoose={(id) => (id === "tips" ? (dialog = "tips") : (ws.d.prompt = examples[Number(id)]?.body || ws.d.prompt))}>
            {#each examples as x, i}<button type="button" role="menuitem" tabindex="-1" data-value={String(i)}>{t(x.title)}</button>{/each}
            <button type="button" role="menuitem" tabindex="-1" data-value="tips">{t("Prompt tips…")}</button>
          </Menu>
        </div>
        <textarea
          id="video-prompt"
          bind:this={prompt}
          bind:value={ws.d.prompt}
          style:height="{height(d.promptHeight)}px"
          onpointerup={() => prompt && (ws.d.promptHeight = height(prompt.getBoundingClientRect().height))}
          oninput={() => (ws.error = "")}
          placeholder={p?.h3
            ? p.references
              ? t("MiniMax-H3 REF2VA expects six labelled sections in order — subject_definitions:, summary:, retention_analysis: (what to DO with each reference), detailed_description:, overall_soundscape:, non_diegetic_music:. Refer to attachments as") + " <Picture 1>, <Video 1>, <Audio 1>" + t(". Click Templates above.")
              : t("MiniMax-H3 expects three labelled fields — integrated_multimodal_description: (the shot, its style, action and camera movement), overall_soundscape: (ambience and physical sound), non_diegetic_music: (score only the audience hears, or N/A). Click Templates above for the exact shape.")
            : t("Describe your shot like a cinematographer — subject, action, camera movement, lighting, setting. 4–8 sentences. Put spoken dialogue in quotes to make characters talk. Click Templates above for a starting point.")}
        ></textarea>
      </section>
      <details id="video-media" bind:open={ws.d.media}>
        <summary>{t("Media inputs for generation")}</summary>
        <div class="video-media">
          <div class="video-frames">
            {@render well("first", t("First frame"), "image/png,image/jpeg,image/webp")}
            {#if p?.last}{@render well("last", t("Last frame"), "image/png,image/jpeg,image/webp")}{/if}
          </div>
          {#if p?.references}
            <section class="image-section">
              <h3>{t("References")} <small>{t("%@/12 · optional, the model follows them", [ws.references])}</small></h3>
              {#each ["images", "videos", "audios"] as const as key}
                {@const list = ws.inputs[key] || []}
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div class="video-well" ondragover={(e) => e.preventDefault()} ondrop={drop(key)}>
                  <h4>{key === "videos" ? t("Clips") : key === "audios" ? t("Audio") : t("Images")} ({list.length}/{key === "images" ? 9 : 3})</h4>
                  {#each list as v, i}
                    <span class="video-ref">
                      <button type="button" onclick={() => marker(`<${names[key]} ${i + 1}>`)}>{t("%@ · <%@ %@>", [v.name ?? "", names[key], i + 1])}</button>
                      <button type="button" aria-label={t("Remove %@", [v.name ?? ""])} onclick={() => ws.remove(key, i)}>×</button>
                    </span>
                  {/each}
                  <label class="chip">
                    {t("Choose %@…", [key === "images" ? t("image") : key === "videos" ? t("clip") : t("audio")])}<input type="file" multiple accept={key === "images" ? "image/png,image/jpeg,image/webp" : key === "videos" ? "video/*" : "audio/*"} onchange={fileInput(key)} />
                  </label>
                </div>
              {/each}
              <Menu id="video-ref-size" label={d.refSize === "max" ? t("Maximum detail") : t("Match generation size")} onchoose={(v) => (ws.d.refSize = v)}>
                <button type="button" role="menuitemradio" tabindex="-1" aria-checked={d.refSize === "match"} data-value="match">{t("Match generation size")}</button>
                <button type="button" role="menuitemradio" tabindex="-1" aria-checked={d.refSize === "max"} data-value="max">{t("Maximum detail (several times slower)")}</button>
              </Menu>
              <label class="video-toggle"><input id="video-silent" type="checkbox" bind:checked={ws.silentClip} />{t("Clip has no soundtrack")}</label>
              <p class="field-note">{t("Clips are sampled at 24 fps, up to the chosen length. Every reference adds work to each step.")}</p>
            </section>
          {/if}
          {#if p?.audio}
            {@render well("audio", t("Speech & sound"), "audio/*")}
            <div class="video-audio-tools">
              <button type="button" id="video-record" onclick={() => void ws.record()}>{ws.recording ? t("Stop recording") : t("Record")}</button>
              <button type="button" id="video-speech" onclick={() => (dialog = "speech")}>{t("Speech…")}</button>
            </div>
            <p class="field-note">{t("The model invents a soundtrack from your prompt. Attach speech to make characters say exact words. Uploads up to 30 seconds.")}</p>
          {:else if p?.h3}
            <section class="video-sound">
              <h3>{t("Sound")} <small>{t("generated")}</small></h3>
              <p class="field-note">{t("This model writes its own soundtrack. Describe it in the prompt after “overall_soundscape:” (and “non_diegetic_music:” for score).")}</p>
            </section>
          {/if}
        </div>
      </details>
      <div class="video-size">
        <label>{t("Clip width")}<input type="number" id="video-width" bind:value={ws.d.width} onchange={() => ws.clamp()} /></label>
        <span>×</span>
        <label>{t("Clip height")}<input type="number" id="video-height" bind:value={ws.d.height} onchange={() => ws.clamp()} /></label>
        <Menu id="video-presets" label={t("Presets")} onchoose={(value) => ws.setSize(value)}>
          {#each p?.sizes || [] as [w, h]}
            <button type="button" role="menuitem" tabindex="-1" data-value="{w}x{h}">{w} × {h} ({w === h ? "square" : w! > h! ? "landscape" : "portrait"})</button>
          {/each}
          {#if ws.inputs.first}<button type="button" role="menuitem" tabindex="-1" data-value="source">{t("Set by starting frame…")}</button>{/if}
        </Menu>
      </div>
      <p id="video-size-hint" class="field-note">{validation.hint}</p>
      <section class="image-quality-control">
        <h3>{t("Quality")}</h3>
        <div class="segmented image-quality-segments" role="group" aria-label={t("Quality")}>
          {#each qualities as q}<button type="button" aria-pressed={ws.quality === q} onclick={() => ws.setQuality(q)}>{t(q)}</button>{/each}
          {#if ws.quality === "Custom"}<button type="button" disabled aria-pressed="true">{t("Custom")}</button>{/if}
        </div>
        <Menu id="video-quality" label={t(ws.quality)} onchoose={(q) => ws.setQuality(q)}>
          {#each qualities as q}<button type="button" role="menuitemradio" tabindex="-1" aria-checked={q === ws.quality} data-value={q}>{t(q)}</button>{/each}
        </Menu>
        <p class="field-note" id="video-quality-note">{t("%@ steps · %@", [d.steps, p?.audio ? d.mode.replaceAll("_", "-") : p?.h3 ? t("native audio") : ""])}</p>
      </section>
      <label class="video-slider">
        {t("Frames")}<output id="video-frames-value">{t("%@ (~%@ s)", [d.frames, (d.frames / 24).toFixed(1)])}</output>
        <input
          type="range"
          id="video-frames"
          aria-label={t("Frames")}
          aria-valuetext={t("%@ (~%@ s)", [d.frames, (d.frames / 24).toFixed(1)])}
          min="0"
          max={Math.max(0, frames.length - 1)}
          value={Math.max(0, frames.indexOf(d.frames))}
          oninput={(e) => (ws.d.frames = frames[Number(e.currentTarget.value)] || ws.d.frames)}
        />
      </label>
      <p class="field-note">{p?.h3 && d.frames < 107 ? t("Below the model’s stated 4–15 s range; useful for short tests.") : t("24 frames per second.")}</p>
      <details id="video-advanced" bind:open={ws.d.advanced}>
        <summary>{t("Advanced options")}</summary>
        <div class="video-advanced">
          {@render slider("steps", t("Steps"), 4, p?.turbo && d.turbo ? 16 : 50)}
          {#if p?.audio}
            {@render slider("cfg", t("CFG scale"), 1, 10, 0.5, working || lockedAudio)}
            {@render slider("stg", t("STG scale"), 0, 4, 0.5, working || lockedAudio)}
            {#if ws.inputs.audio}{@render slider("audioGuidance", t("Audio guidance"), 1, 12, 0.5, working || lockedAudio)}{/if}
            <div class="video-mode">
              <label for="video-mode">{t("Mode")}</label>
              <Menu id="video-mode" label={modeLabel} disabled={working || lockedAudio} onchoose={(v) => ((ws.d.mode = v), ws.clamp())}>
                {#each ["one_stage", "two_stage", "two_stage_hq"] as v, i}
                  <button type="button" role="menuitemradio" tabindex="-1" aria-checked={d.mode === v} data-value={v}>{["1-stage", "2-stage", t("2-stage HQ")][i]}</button>
                {/each}
              </Menu>
            </div>
            {@render slider("refine", t("Refine steps"), 0, 6, 1, d.mode === "one_stage" && !ws.inputs.audio)}
            <p class="field-note">
              {ws.inputs.audio && d.mode === "one_stage"
                ? t("Audio uses two-stage with server guidance defaults (video 3, audio 7).")
                : d.mode === "one_stage"
                  ? t("Refine steps: Off (1-stage).")
                  : "Refine steps: 0 = Auto (3)."}
            </p>
          {/if}
          {#if p?.chain}{@render slider("windows", t("Chained windows"), 1, 6)}{/if}
          <label>{t("Seed")}<input type="number" id="video-seed" min="0" bind:value={ws.d.seed} /></label>
          {#if p?.turbo}
            {@render toggle("turbo", t("Turbo (distilled 4-step sampling)"))}
            <p class="field-note">{t("Requires the Turbo adapter on the server hosting this model.")}</p>
          {/if}
          {#if p?.decoder}{@render toggle("decoder", t("Diffusion decoder (sharper, slower)"))}{/if}
          {#if p?.h3}{@render toggle("best", t("Max quality (slower)"))}{/if}
          {@render toggle("preview", t("Show live preview while generating (~1% slower)"))}
        </div>
      </details>
    </fieldset>
    <div class="image-action-row">
      <button id="video-generate" class="primary generate" type="button" disabled={!ws.ready || working || !!validation.error} onclick={() => void ws.generate()}><Icon name="sparkles" />{t("Generate")}</button>
      <button id="video-cancel" type="button" hidden={!working} onclick={() => ws.cancel()}>{ws.recording ? t("Stop recording") : t("Cancel")}</button>
    </div>
    <p id="video-validation" class="field-note" role="status">{validation.error}</p>
    <p id="video-storage" class="field-note">{ws.storageError}</p>
  </div>

  <div class="image-output">
    <div class="image-preview" id="video-preview">
      {#if c.phase === "completed" && c.result}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video
          controls
          playsinline
          preload="metadata"
          aria-label={t("Generated video")}
          src={url}
          onerror={() => (playbackFailed = true)}
        ></video>
        {#if playbackFailed}<p class="field-note">{t("This browser cannot play the encoded file. Download it to open in a compatible player.")}</p>{/if}
      {:else if c.phase === "running" || c.phase === "encoding"}
        <div class="video-progress">
          <div id="video-live-preview">{#if live}<img alt={t("Live generation preview")} src={live} />{/if}</div>
          <span class="spinner"></span>
          <h2>{c.message}</h2>
          <progress max={c.total || 1} value={c.step}></progress>
          <p>{t("%@/%@ steps ·", [c.step, c.total])} <span id="video-elapsed">{elapsed}</span></p>
        </div>
      {:else}
        <div class="empty-state">
          <Icon name={c.phase === "failed" ? "circle-alert" : "film"} />
          <h2>{c.phase === "failed" ? t("Generation failed") : c.phase === "cancelled" ? t("Cancelled") : t("No generation yet")}</h2>
          <p>{c.message || t("Enter a prompt and press Generate.")}</p>
        </div>
      {/if}
    </div>
    <div id="video-actions">
      {#if c.phase === "completed" && c.result}
        {@const r = c.result}
        <div class="image-result-actions">
          <button id="video-download" type="button" onclick={() => download(r.blob, `video-${r.createdAt}.${mp4 ? "mp4" : "webm"}`)}><Icon name="download" />{t("Download")} {mp4 ? "MP4" : "WebM"}</button>
          <span class="field-note">{r.codec || r.blob.type}{r.elapsedMs ? t(" · %@ s", [(r.elapsedMs / 1000).toFixed(1)]) : ""}</span>
        </div>
        {#if !mp4}<p class="field-note">{t("MP4 encoding is unavailable in this browser. Saved as WebM.")}</p>{/if}
        {#if c.saveError}
          <p class="field-note">{c.saveError}</p>
          <button type="button" id="video-retry" onclick={() => void c.save()}>{t("Retry saving")}</button>
        {/if}
      {/if}
    </div>
    <details id="video-gallery" bind:open={gallery}>
      <summary>{t("Gallery")}</summary>
      <div id="video-gallery-list">{#if gallery}<VideoGallery library={ws.library} server={ws.server} onpick={(item) => ws.show(item)} />{/if}</div>
    </details>
  </div>
</section>

{#if dialog === "tips"}<TipsDialog h3={!!p?.h3} references={!!p?.references} onclose={() => (dialog = "")} />{/if}
{#if dialog === "speech"}<SpeechDialog {app} onclose={() => (dialog = "")} />{/if}
{#if dialog === "rewrite"}
  {@const format = p?.h3
    ? p.references
      ? "subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music"
      : "integrated_multimodal_description, overall_soundscape, non_diegetic_music"
    : "4–8 cinematographer sentences with camera, action, lighting and sound"}
  <RewriteDialog
    {connection}
    title={t("Rewrite video prompt")}
    class="audio-rewrite"
    text={d.prompt}
    system={`Rewrite this video prompt for a ${(d.frames / 24).toFixed(1)} second clip. Format: ${format}. Return only the prompt.`}
    request={d.prompt}
    maxTokens={1800}
    textLabel={t("Rewritten prompt")}
    reviewed={t("Review the prompt, then Apply.")}
    noModel={t("No chat model is available on this server.")}
    onapply={(text) => (ws.d.prompt = text)}
    onclose={() => (dialog = "")}
  />
{/if}
