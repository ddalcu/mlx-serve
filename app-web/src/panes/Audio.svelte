<script lang="ts">
  import { untrack } from "svelte";
  import Icon from "../components/Icon.svelte";
  import Menu from "../components/Menu.svelte";
  import ModelChooser from "../components/ModelChooser.svelte";
  import RewriteDialog from "../components/RewriteDialog.svelte";
  import type { App } from "../lib/app.svelte";
  import { download } from "../lib/download";
  import { displayName, N, t } from "../lib/i18n/i18n";
  import type { IconName } from "../lib/icons";
  import { commonTempos, kokoroVoices, keys, languages, musicLyrics, musicStyles, music3Styles, sectionTags, trackClasses, voiceLabel, voiceLanguages, voiceName } from "../lib/state/audio-presets";
  import { tabs, type Field } from "../lib/state/audio-workspace.svelte";
  import AudioHistory from "./audio/AudioHistory.svelte";
  import SaveTemplateDialog from "./audio/SaveTemplateDialog.svelte";

  let { app }: { app: App } = $props();
  const ws = $derived(app.audio);
  const connection = $derived(app.connection);
  const c = $derived(ws.c);
  const d = $derived(ws.d);
  const p = $derived(ws.profile);
  const tab = $derived(ws.tab);
  const voice = $derived(tab === "voice");
  const music = $derived(tab === "music");
  const source = $derived(!!p?.source && d.task !== "text2music");
  const busy = $derived(!!c.run);
  const labels = { voice: N("Voice"), music: N("Music"), sound: N("Sound Effects") };
  const glyphs: Record<string, IconName> = { voice: "audio-lines", music: "music", sound: "volume-2" };
  const glyph = $derived(glyphs[tab]!);
  const detail = $derived(p ? t(labels[p.tab]) + (p.reference ? " · " + (voice ? t("Voice cloning") : t("Reference audio")) : "") : t("No supported %@ model is available on this server.", [t(labels[tab])]));
  const error = $derived(ws.error || ws.validation);
  const clock = (seconds: number) => t("sec · %@:%@", [Math.floor(seconds / 60), String(Math.floor(seconds % 60)).padStart(2, "0")]);

  let gallery = $state(false);
  let rewrite = $state<"" | Field>("");
  let saving = $state<"" | Field>("");
  let url = $state("");
  let audio = $state<HTMLAudioElement>();
  let playing = $state(false);
  let playError = $state("");
  let length = $state("");

  // Draft saving follows every change to the forms, and everything stops with the pane.
  $effect(() => {
    ws.signature();
    untrack(() => ws.touch());
  });
  $effect(() => () => {
    ws.flush();
    ws.stopActivity();
    if (ws.clipUrl) URL.revokeObjectURL(ws.clipUrl);
    ws.clipUrl = "";
  });
  // The finished clip plays from an object URL that lives as long as that result.
  $effect(() => {
    const blob = c.phase === "completed" ? c.result?.blob : undefined;
    playing = false;
    playError = "";
    length = "";
    if (!blob) return void (url = "");
    const made = URL.createObjectURL(blob);
    url = made;
    return () => URL.revokeObjectURL(made);
  });

  const play = async () => {
    if (!audio) return;
    if (!audio.paused) {
      audio.pause();
      audio.currentTime = 0;
      return;
    }
    ws.stopPlayback();
    try {
      await audio.play();
    } catch {
      playError = t("Playback could not start. Try Play again or download the WAV.");
    }
  };
  const examples = (field: Field) => ws.examples(field);
  function chooseTemplate(field: Field, value: string) {
    if (value === "save") {
      if (ws.d[field].trim()) saving = field;
    } else ws.chooseTemplate(field, value);
  }
  const enhanceExamples = (field: Field) => (field === "lyrics" ? musicLyrics : p?.meta ? musicStyles : music3Styles);
  const fileHandler = (key: "reference" | "source") => async (event: Event) => {
    const input = event.currentTarget as HTMLInputElement;
    if (input.files?.[0]) await ws.attach(input.files[0], key);
    input.value = "";
  };
  const drop = (key: "reference" | "source") => (event: DragEvent) => {
    event.preventDefault();
    if (event.dataTransfer?.files[0]) void ws.attach(event.dataTransfer.files[0], key);
  };
  const inputs: Record<string, HTMLInputElement | undefined> = $state({});
  const autoplay = (node: HTMLAudioElement) => {
    node.play().catch(() => (ws.error = t("Playback could not start. Use the audio controls to try again.")));
  };
</script>

{#snippet slider(key: "duration" | "steps" | "speed" | "coverStrength" | "coverNoise", label: string, min: number, max: number, step: number)}
  {@const numeric = (tab === "music" && ["duration", "steps"].includes(key)) || (tab === "sound" && key === "steps")}
  {#if tab === "sound" && key === "steps"}
    <label>{label}<input class="audio-number" name={key} type="number" aria-label={label} {min} {max} {step} bind:value={ws.d[key]} /></label>
  {:else}
    <label class="image-slider {key === 'duration' ? 'audio-duration' : ''}">
      <span>
        {label}
        {#if numeric}
          <span class="audio-number-group">
            <input class="audio-number" name={key} type="number" aria-label={label} {min} {max} step={key === "duration" ? 1 : step} bind:value={ws.d[key]} />
            {#if key === "duration"}<span>{clock(Number(ws.d.duration))}</span>{/if}
          </span>
        {:else}
          <output>{key === "duration" ? Number(ws.d[key]).toFixed(1) + t(" sec") : ws.d[key]}</output>
        {/if}
      </span>
      <input name={key} aria-label={t("%@ slider", [label])} type="range" {min} {max} {step} bind:value={ws.d[key]} />
    </label>
  {/if}
{/snippet}

{#snippet entry(value: string, label: string, checked?: boolean)}
  <button type="button" tabindex="-1" role={checked === undefined ? "menuitem" : "menuitemradio"} aria-checked={checked} data-value={value}>{label}</button>
{/snippet}

{#snippet choice(key: "keyscale" | "language" | "timesignature", label: string, values: string[][])}
  <div class="audio-choice">
    <span>{label}</span>
    <Menu id="audio-{key}" label={t(values.find(([v]) => v === String(ws.d[key]))?.[1] || "Auto")} onchoose={(value) => (ws.d[key] = value)}>
      {#each values as [v, l]}{@render entry(v, t(l), String(ws.d[key]) === v)}{/each}
    </Menu>
  </div>
{/snippet}

{#snippet templates(field: Field)}
  <Menu id="audio-{field}-templates" label={t("Templates")} onchoose={(value) => chooseTemplate(field, value)}>
    {#if music}
      {@render entry("save", t("Save current…"))}
      {#each ws.saved[field] as saved, i}
        {@render entry("saved:" + i, saved.title)}
        {@render entry("delete:" + i, t("Delete saved: ") + saved.title)}
      {/each}
    {/if}
    {#each examples(field) as e, i}{@render entry("example:" + i, t(e.title))}{/each}
  </Menu>
{/snippet}

{#snippet clipWell(key: "reference" | "source", title: string)}
  {@const clip = ws.pane[key]}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <section class="image-section audio-clip" ondragover={(e) => e.preventDefault()} ondrop={drop(key)}>
    <h2>{title}</h2>
    <div class="image-drop-well">
      {#if clip}
        <div class="audio-clip-filled">
          <Icon name="audio-lines" />
          <span>{t("%@ · %@s", [clip.name ?? "", clip.duration?.toFixed(1) || "?"])}</span>
          <button type="button" aria-label={t("Preview %@", [title])} onclick={() => ws.previewClip(clip)}>{t("Play")}</button>
          <button type="button" aria-label={t("Clear %@", [title])} onclick={() => ws.clear(key)}><Icon name="x" /></button>
        </div>
      {:else}
        <button type="button" class="image-well-action" onclick={() => inputs[key]?.click()}>
          <Icon name="audio-lines" />
          <span>
            {voice ? t("Choose audio file…") : t("Choose file…")}
            <small>{voice ? t("or drag one here") : key === "source" ? t("10 seconds to 10 minutes. The new track will have the same length.") : t("A light touch: the model takes the overall feel and timbre from up to 30 seconds of the clip, it does not recreate the song.")}</small>
          </span>
        </button>
        {#if voice}
          <button id="audio-record" type="button" onclick={() => (ws.recording ? ws.stopRecording() : void ws.record())}>
            <Icon name="mic" />
            <span>{ws.recording ? t("Recording… Stop (max 8s)") : t("Record audio…")}<small>{t("~8s recommended")}</small></span>
          </button>
        {/if}
      {/if}
      <input type="file" accept="audio/*" hidden bind:this={inputs[key]} onchange={fileHandler(key)} />
    </div>
    <p class="field-note">{voice ? t("About 8 seconds recommended. Without a reference, the default voice is used.") : ""}</p>
  </section>
{/snippet}

<section class="audio-screen" aria-label={t("Audio & Music")}>
  <div class="audio-tabs segmented" role="group" aria-label={t("Audio type")}>
    {#each tabs as name}
      <button
        type="button"
        aria-pressed={name === tab}
        onclick={() => {
          gallery = false;
          ws.setTab(name);
        }}>{t(labels[name])}</button
      >
    {/each}
  </div>
  <div class="image-screen">
    <!-- The pane validates itself; native validation would silently block Generate over a control inside a closed section. -->
    <form
      id="audio-form"
      class="image-controls audio-controls"
      novalidate
      data-tab={tab}
      onsubmit={(e) => {
        e.preventDefault();
        void ws.generate();
      }}
      oninput={() => (ws.error = "")}
    >
      <fieldset id="audio-fields" disabled={!ws.ready || busy || ws.busy}>
        <ModelChooser id="audio-model" models={ws.models} selected={d.model} status={connection.status} {detail} server={connection.serverName()} onchoose={(id) => ws.chooseModel(id)} />
        {#if p?.source}
          <div class="segmented" role="group" aria-label={t("Music mode")}>
            {#each [["text2music", t("Text to music")], ["cover", t("Cover")], ["complete", t("Vocal to BGM")]] as [value, label]}
              <button type="button" aria-pressed={d.task === value} onclick={() => (ws.d.task = value!)}>{label}</button>
            {/each}
          </div>
          {#if d.task === "cover"}<p class="field-note">{t("Re-sings an existing track in the style you describe: melody and structure stay, the caption and lyrics decide the rest.")}</p>{/if}
        {/if}
        {#if source}
          {@render clipWell("source", t("Source track"))}
          {#if d.task === "cover"}
            <div class="image-section">
              {@render slider("coverStrength", t("Cover strength"), 0, 1, 0.05)}
              <p class="field-note">{t("How many of the steps follow the source. 1 keeps it all the way; lower lets the caption take over for the last steps.")}</p>
            </div>
            <div class="image-section">
              {@render slider("coverNoise", t("Noise strength"), 0, 1, 0.05)}
              <p class="field-note">{t("0 starts from pure noise (the default). Higher starts closer to the original audio, so more of it comes through.")}</p>
            </div>
          {:else}
            <div class="audio-tracks" role="group" aria-label={t("Instruments to add")}>
              {#each trackClasses as name}
                <label class="check-label"><input type="checkbox" bind:group={ws.d.trackClasses} value={name} />{t(displayName(name))}</label>
              {/each}
            </div>
          {/if}
        {/if}
        <section class="image-section">
          <div class="image-heading">
            <label for="audio-prompt">{voice ? t("Text to be generated") : music ? t("Style prompt") : t("Describe the sound")}</label>
            {#if music}
              <button type="button" class="chip" title={t("Rewrite with LLM")} aria-label={t("Rewrite style prompt")} onclick={() => (rewrite = "prompt")}><Icon name="sparkles" />{t("Enhance…")}</button>
            {/if}
            {#if !voice}{@render templates("prompt")}{/if}
          </div>
          <textarea id="audio-prompt" name="prompt" bind:value={ws.d.prompt} placeholder={voice ? t("Enter the text to speak…") : music ? t("Genre, mood, instruments…") : t("What makes it, the material, the space…")}></textarea>
          {#if music}<p class="field-note">{t('Genre, mood, instruments — e.g. "upbeat synthwave with driving bass and dreamy pads".')}</p>{/if}
        </section>
        {#if p?.voices}
          <section class="image-section">
            <h2>{t("Voice")}</h2>
            <Menu
              id="audio-voice"
              label={d.voice
                .split(",")
                .map((v) => voiceName(v.trim()))
                .join(" + ")}
              onchoose={(value) => (ws.d.voice = value)}
            >
              {#each voiceLanguages as [code, label]}
                <span role="group" aria-label={label}>
                  <span class="image-menu-heading">{t(label)}</span>
                  {#each kokoroVoices.filter((v) => v[0] === code) as v}{@render entry(v, voiceLabel(v), v === d.voice)}{/each}
                </span>
              {/each}
            </Menu>
            <label>{t("Blend voices")}<input name="voice" bind:value={ws.d.voice} placeholder={t("af_heart,af_bella")} /></label>
            <p class="field-note">{t("Select a voice or mix voices with comma-separated IDs.")}</p>
          </section>
        {/if}
        {#if music && p}
          <section class="image-section">
            <div class="image-heading">
              <label for="audio-lyrics">{t("Lyrics")}</label>
              <label class="check-label"><input name="instrumental" type="checkbox" role="switch" bind:checked={ws.d.instrumental} />{t("Instrumental")}</label>
              {#if !d.instrumental}
                <span class="lyrics-tools">
                  <button type="button" class="chip" onclick={() => (rewrite = "lyrics")}><Icon name="sparkles" />{t("Enhance…")}</button>
                  {@render templates("lyrics")}
                  <Menu id="audio-tags" label={t("Sections")} onchoose={(value) => (ws.d.lyrics += (ws.d.lyrics && !ws.d.lyrics.endsWith("\n") ? "\n" : "") + value + "\n")}>
                    {#each sectionTags as tag}{@render entry(tag, tag)}{/each}
                  </Menu>
                </span>
              {/if}
            </div>
            {#if !d.instrumental}
              <textarea id="audio-lyrics" name="lyrics" bind:value={ws.d.lyrics} placeholder="{p.family === 'minimax_music3' ? t('This model sings your lyrics. Section tags go on their own lines:') : t('Leave empty, or tick Instrumental, for a track with no vocals. Section tags:')} {sectionTags.join(' ')}"></textarea>
            {:else}
              <p class="field-note">{t("A wordless track. Lyrics are kept here for later, and omitted from this request.")}</p>
            {/if}
          </section>
        {/if}
        {#if p?.reference}{@render clipWell("reference", voice ? t("Reference voice") : t("Reference audio (optional)"))}{/if}
        {#if p && !voice && !source}
          {@render slider("duration", t("Duration"), p.duration[0]!, p.duration[1]!, music ? 5 : 0.5)}
          {#if p.family === "minimax_music3"}<p class="field-note">{t("An upper bound — the model may end the song earlier.")}</p>{/if}
        {/if}
        {#if p && (p.speed || !voice)}
          <details id="audio-advanced" bind:open={ws.d.advanced}>
            <summary>{t("Advanced options")}</summary>
            <div class="image-advanced-body">
              {#if p.speed}{@render slider("speed", t("Speed (×)"), 0.5, 2, 0.05)}{/if}
              {#if p.steps}{@render slider("steps", music ? t("Quality passes") : t("Steps"), music ? 4 : 1, music ? 100 : 50, 1)}{/if}
              {#if music}
                <div class="audio-knobs">
                  <label class="audio-tempo">
                    <span>{t("Tempo (BPM)")}</span>
                    <input name="bpm" inputmode="numeric" placeholder={t("Auto")} bind:value={ws.d.bpm} />
                    <Menu id="audio-bpm" label={t("Tempos")} onchoose={(value) => (ws.d.bpm = value)}>
                      {#each commonTempos as [v, l]}{@render entry(v, l, d.bpm === v)}{/each}
                    </Menu>
                  </label>
                  {@render choice("keyscale", t("Key"), [["", t("Auto")], ...keys.map((k) => [k, k])])}
                  {#if p.meta}
                    {@render choice("language", t("Vocal language"), languages.map(([label, id]) => [id, label]))}
                    {@render choice("timesignature", t("Time signature"), [["", t("Auto")], ["4", "4/4"], ["3", "3/4"], ["2", "2/4"], ["6", "6/8"]])}
                  {/if}
                </div>
              {/if}
              {#if !voice}
                <label>{t("Seed")}<input name="seed" inputmode="numeric" placeholder={t("Random")} bind:value={ws.d.seed} /></label>
                <p class="field-note">{t("Same seed + prompt%@ reproduces the %@.", [music ? "" : t(" + length"), music ? t("track") : t("sound")])}</p>
              {/if}
            </div>
          </details>
        {/if}
      </fieldset>
      <p id="audio-validation" class="field-note" role="status">{error}</p>
      <button id="audio-generate" type="submit" class="generate" class:primary={!busy} disabled={!busy && (!!ws.validation || ws.busy || ws.recording)}>
        {#if busy}{t("Cancel")}{:else}<Icon name={glyph} />{t("Generate")}{/if}
      </button>
      <p id="audio-storage" class="field-note" role="status">{ws.storageError}</p>
    </form>

    <div class="image-output">
      <div id="audio-preview" class="image-preview" class:playing>
        {#if c.phase === "completed" && c.result}
          {@const r = c.result}
          <div class="audio-completed">
            <Icon name={glyph} />
            <audio
              id="audio-result"
              preload="metadata"
              src={url}
              bind:this={audio}
              onloadedmetadata={() => (length = audio && Number.isFinite(audio.duration) ? audio.duration.toFixed(1) + t(" sec") : "")}
              onplay={() => (playing = true)}
              onpause={() => (playing = false)}
              onended={() => (playing = false)}
              onerror={() => (playError = t("This browser could not play the audio. Download the WAV to open it elsewhere."))}
            ></audio>
            <button id="audio-play" type="button" onclick={() => void play()}>{playing ? t("Stop") : t("Play")}</button>
            <span id="audio-duration" class="field-note">{length}</span>
            <div class="audio-file">
              <span>{t("%@.wav", [r.model.split("/").at(-1)])}</span>
              <button type="button" id="audio-download" onclick={() => download(r.blob, `${tab}-${r.createdAt}.wav`)}><Icon name="download" />{t("Download WAV")}</button>
            </div>
            <p class="field-note">{t("%@ seconds · %@", [(r.elapsedMs / 1000).toFixed(1), r.prompt])}</p>
            <p id="audio-save-error" class="field-note" role="status">{playError || c.saveError}</p>
            {#if c.saveError}<button type="button" id="audio-retry-save" onclick={() => void c.save()}>{t("Retry saving")}</button>{/if}
          </div>
        {:else if c.phase === "running"}
          <div class="image-progress">
            <progress max={c.total > 0 ? c.total : undefined} value={c.total > 0 ? c.step : undefined}></progress>
            <p>{c.message}</p>
            {#if c.total > 0}<span>{c.step} / {c.total}</span>{/if}
          </div>
        {:else}
          <div class="empty-state">
            <Icon name={glyph} />
            <h2>{c.phase === "failed" ? t("Failed") : voice ? t("No audio yet") : music ? t("No music yet") : t("No sounds yet")}</h2>
            <p>{c.message || (voice ? t("Enter text, add a reference voice, and press Generate.") : music ? t("Describe the music and press Generate.") : t("Describe a sound and press Generate."))}</p>
          </div>
        {/if}
      </div>
      <button id="audio-gallery-toggle" class="image-output-link" aria-expanded={gallery} onclick={() => (gallery = !gallery)}><Icon name="library" />{t("Audio saved in this browser")}</button>
      <section id="audio-gallery" class="audio-gallery" aria-label={t("Audio history")} hidden={!gallery}>
        {#if gallery}<AudioHistory library={ws.library} server={ws.server} type={ws.type} onpick={(item) => ws.show(item)} />{/if}
      </section>
    </div>
  </div>
  {#if ws.clipUrl}
    <!-- svelte-ignore a11y_media_has_caption -->
    <audio class="audio-reference-player" controls src={ws.clipUrl} use:autoplay onerror={() => (ws.error = t("Reference audio could not be played."))}></audio>
  {/if}
</section>

{#if saving}
  <SaveTemplateDialog field={saving} text={ws.d[saving]} onsave={(title) => saving && ws.saveTemplate(saving, title, ws.d[saving].trim())} onclose={() => (saving = "")} />
{/if}

{#if rewrite}
  {@const field = rewrite}
  {@const context = field === "lyrics" ? d.prompt : d.instrumental ? "" : d.lyrics}
  {@const shape =
    field === "lyrics"
      ? `Use section tags on their own lines: ${sectionTags.join(" ")}. Keep the user's theme and lines, tighten rhythm and rhyme.`
      : p?.meta
        ? "Write one paragraph of genre, mood, instruments and production. No tempo, key or time signature; those are separate controls."
        : "Use the examples’ Global Metadata / Vocal Details / Arrangement format, including bpm, key and scale."}
  <RewriteDialog
    {connection}
    title={t("Rewrite %@", [field === "prompt" ? t("style prompt") : t("lyrics")])}
    class="audio-rewrite"
    text={d[field]}
    system={`You rewrite music ${field === "lyrics" ? "lyrics" : "style prompts"}. ${shape} Reply only with the rewritten text, no preamble or markdown.\nExamples:\n${enhanceExamples(field).map((e) => e.body).join("\n---\n")}`}
    request={`Rewrite: ${d[field]}\n${d.instrumental ? "Instrumental, no vocals." : `Vocal language: ${d.language}.`}\nOther context: ${context}`}
    maxTokens={1024}
    textLabel={t("Rewritten text")}
    reviewed={t("Review the text, then Apply.")}
    noModel={t("No chat model is available on this server.")}
    onapply={(text) => (ws.d[field] = text)}
    onclose={() => (rewrite = "")}
  />
{/if}
