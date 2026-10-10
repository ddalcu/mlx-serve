<script lang="ts">
  import { untrack } from "svelte";
  import Icon from "../components/Icon.svelte";
  import ImageGallery from "../components/ImageGallery.svelte";
  import Menu from "../components/Menu.svelte";
  import ModelChooser from "../components/ModelChooser.svelte";
  import type { App } from "../lib/app.svelte";
  import { download } from "../lib/download";
  import { N, t } from "../lib/i18n/i18n";
  import { imageTemplates } from "../lib/state/image-presets";
  import { insertImageMarker, randomSeed, sourceCanvases } from "../lib/state/image-state.svelte";
  import RewriteDialog from "../components/RewriteDialog.svelte";

  let { app }: { app: App } = $props();
  const ws = $derived(app.image);
  const connection = $derived(app.connection);
  const p = $derived(ws.profile);
  const c = $derived(ws.c);
  const busy = $derived(!!c.run);
  const validation = $derived(ws.validation);
  const modeRefs = $derived(ws.editing ? ws.refs : ws.refs.slice(0, 1));
  const room = $derived((ws.editing ? 4 : 1) - modeRefs.length);
  const quality = $derived(p ? [t("Fast"), t("Good"), t("Quality"), t("Super Quality")] : []);
  const selected = $derived(p ? p.quality.indexOf(ws.d.steps) : -1);
  const groups = $derived(
    (ws.editing && ws.refs.length ? (p?.family === "mage" ? Object.keys(imageTemplates).slice(1) : [N("Content"), N("Appearance"), N("Scene & camera"), N("People")]) : [N("Starters")]).map((name) => ({
      name,
      examples: imageTemplates[name as keyof typeof imageTemplates],
    })),
  );
  const orientations = $derived(
    p
      ? [t("Landscape"), t("Square"), t("Portrait")].map((name, i) => ({
          name,
          items: p.resolutions
            .filter((r) => (i === 0 ? r.width > r.height : i === 1 ? r.width === r.height : r.width < r.height))
            .sort((a, b) => b.width * b.height - a.width * a.height),
        }))
      : [],
  );
  const detail = $derived(
    p ? [t("Text to image"), p.edit ? t("Edit") : "", p.variation ? t("Variation") : ""].filter(Boolean).join(" · ") : t("Capabilities not specified"),
  );

  let prompt = $state<HTMLTextAreaElement>();
  let files = $state<HTMLInputElement>();
  let dragging = $state(false);
  let enhancing = $state(false);
  let gallery = $state(false);
  let url = $state("");

  // Draft saving follows every change to the form, and stops with the pane.
  $effect(() => {
    ws.signature();
    untrack(() => ws.touch());
  });
  $effect(() => () => {
    ws.flush();
    if (ws.c.run) ws.c.cancel();
  });
  // The finished picture is shown from an object URL that lives as long as that result.
  $effect(() => {
    const blob = c.phase === "completed" ? c.result?.blob : undefined;
    if (!blob) return void (url = "");
    const made = URL.createObjectURL(blob);
    url = made;
    return () => URL.revokeObjectURL(made);
  });

  function marker(index: number) {
    const input = prompt!;
    const focused = document.activeElement === input;
    const result = insertImageMarker(t("image %@", [index + 1]), input.value, focused ? input.selectionStart : input.value.length, focused ? input.selectionEnd : input.value.length);
    ws.d.prompt = result.text;
    input.focus();
    queueMicrotask(() => input.setSelectionRange(result.cursor, result.cursor));
  }
  const size = (value: string) => {
    const [w, h] = value.split("x");
    if (w && h) {
      ws.d.width = w;
      ws.d.height = h;
    }
  };
</script>

{#snippet slider(key: "steps" | "strength" | "guidance" | "gain", label: string, min: number, max: number, step: number)}
  <label class="image-slider">
    <span>{label}<output>{key === "strength" ? Math.round(ws.d.strength * 100) + "%" : ws.d[key]}</output></span>
    <input name={key} type="range" {min} {max} {step} bind:value={ws.d[key]} />
  </label>
{/snippet}

{#snippet entry(value: string, label: string, checked = false)}
  <button type="button" role="menuitemradio" aria-checked={checked} tabindex="-1" data-value={value}>{label}</button>
{/snippet}

<section class="image-screen" aria-label={t("Image Generation")}>
  <!-- The pane validates itself; native validation would silently block Generate over a control inside the closed Advanced section. -->
  <form
    id="image-form"
    class="image-controls"
    novalidate
    onsubmit={(e) => {
      e.preventDefault();
      void ws.generate();
    }}
    oninput={() => (ws.error = "")}
  >
    <fieldset id="image-fields" disabled={!ws.ready || busy || ws.attaching}>
      <ModelChooser id="image-model" models={ws.models} selected={ws.d.model} status={connection.status} {detail} server={connection.serverName()} onchoose={(id) => ws.chooseModel(id)} />
      <section class="image-section">
        <div class="image-heading">
          <label for="image-prompt">{t("Prompt")}</label>
          <button type="button" id="image-enhance" class="chip" disabled={!ws.d.prompt.trim() || busy} onclick={() => (enhancing = true)}><Icon name="sparkles" />{t("Enhance…")}</button>
          <Menu id="image-templates" label={t("Templates")} onchoose={(value) => (ws.d.prompt = value)}>
            {#each groups as g}
              <span role="group" aria-label={t(g.name)}>
                <span class="image-menu-heading">{t(g.name)}</span>
                {#each g.examples as e}{@render entry(e.body, t(e.title))}{/each}
              </span>
            {/each}
          </Menu>
        </div>
        <textarea id="image-prompt" name="prompt" bind:this={prompt} bind:value={ws.d.prompt} placeholder={t("Describe what you want to create…")}></textarea>
      </section>

      {#if p && (p.edit || p.variation)}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <section
          id="image-sources"
          class="image-section"
          class:drag-over={dragging}
          ondragover={(e) => {
            e.preventDefault();
            dragging = true;
          }}
          ondragleave={() => (dragging = false)}
          ondrop={(e) => {
            e.preventDefault();
            dragging = false;
            void ws.attach(e.dataTransfer?.files);
          }}
        >
          <div class="image-heading">
            <h3>{t("Source image(s)")}</h3>
            <span class="field-note">{t("optional")}</span>
            {#if ws.refs.length && p.edit && p.variation}
              <div class="segmented" role="group" aria-label={t("Source mode")}>
                <button type="button" aria-pressed={ws.editing} onclick={() => { ws.d.mode = "edit"; ws.adoptSource(); }}>{t("Edit")}</button>
                <button type="button" aria-pressed={!ws.editing} onclick={() => { ws.d.mode = "variation"; ws.adoptSource(); }}>{t("Variation")}</button>
              </div>
            {/if}
          </div>
          <div class="image-drop-well">
            <div class="image-ref-grid">
              {#each modeRefs as r, i}
                <div class="image-ref">
                  <button type="button" title={t("Insert image %@ into prompt", [i + 1])} onmousedown={(e) => e.preventDefault()} onclick={() => marker(i)}>
                    <img src="data:image/png;base64,{r.base64}" alt={r.name} /><span>{t("image %@", [i + 1])}</span>
                  </button>
                  <button type="button" aria-label={t("Remove image %@", [i + 1])} onclick={() => ws.removeRef(i)}><Icon name="x" /></button>
                </div>
              {/each}
            </div>
            {#if room > 0}
              <button type="button" id="image-choose" class="image-well-action" onclick={() => files?.click()}>
                <Icon name="image" /><span>{p.variation ? t("Choose image…") : t("Choose image to edit…")}<small>{t("or drag one here")}</small></span>
              </button>
            {/if}
            <input
              id="image-files"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              multiple={ws.editing}
              hidden
              bind:this={files}
              onchange={async () => {
                await ws.attach(files?.files);
                if (files) files.value = "";
              }}
            />
          </div>
          <p class="field-note">
            {#if !ws.refs.length}
              {p.edit
                ? t("Edit an existing image with an instruction") + (p.variation ? t(", or generate a variation of it.") : t(" — say what to change and the rest stays put."))
                : t("Generate a variation of an existing image, guided by the prompt (image-to-image).")}
            {:else if ws.editing}
              {t("Describe the change in the prompt. The source is image 1; references follow in order. Click a picture to insert its name.")}
            {:else}
              {t("Low = stay close to the source; high = mostly the prompt.")}
            {/if}
          </p>
          {#if ws.refs.length && !ws.editing}{@render slider("strength", t("Variation strength"), 0.1, 1, 0.05)}{/if}
        </section>
      {/if}

      {#if p}
        <section class="image-section">
          <h2>{t("Quality")}</h2>
          {#if p.fixed}
            <p class="field-note">{t("Fixed at %@ steps — this model is distilled for a %@-step schedule, so more steps cost time without adding detail.", [p.quality[0], p.quality[0]])}</p>
          {:else}
            <div id="image-quality-control" class="image-quality-control">
              <div class="segmented image-quality-segments" role="group" aria-label={t("Quality")}>
                {#each quality as label, i}
                  <button type="button" aria-pressed={i === selected} onclick={() => (ws.d.steps = p.quality[i]!)}>{label}</button>
                {/each}
                {#if selected < 0}<button type="button" disabled aria-pressed="true">{t("Custom")}</button>{/if}
              </div>
              <Menu id="image-quality" label={quality[selected] || t("Custom")} onchoose={(value) => (ws.d.steps = Number(value))}>
                {#each quality as label, i}
                  <button type="button" role="menuitemradio" tabindex="-1" aria-checked={i === selected} data-value={p.quality[i]}>{label}</button>
                {/each}
                {#if selected < 0}<button type="button" role="menuitemradio" aria-checked="true" disabled>{t("Custom")}</button>{/if}
              </Menu>
            </div>
            <p id="image-quality-hint" class="field-note">{t("%@ steps", [ws.d.steps])}</p>
          {/if}
        </section>
        <section class="image-section">
          <div class="image-canvas">
            <label>{t("Width")}<input name="width" inputmode="numeric" bind:value={ws.d.width} /></label>
            <span>×</span>
            <label>{t("Height")}<input name="height" inputmode="numeric" bind:value={ws.d.height} /></label>
            <Menu id="image-presets" label={t("Presets")} onchoose={size}>
              {#each orientations as o}
                <span role="group" aria-label={o.name}>
                  <span class="image-menu-heading">{o.name}</span>
                  {#each o.items as r}{@render entry(`${r.width}x${r.height}`, t(r.label), String(r.width) === ws.d.width && String(r.height) === ws.d.height)}{/each}
                </span>
              {/each}
              {#if ws.refs[0]}
                <span role="group" aria-label={t("Set by source image…")}>
                  <span class="image-menu-heading">{t("Set by source image…")}</span>
                  {#each sourceCanvases(p, ws.refs[0].width, ws.refs[0].height) as r}{@render entry(`${r.width}x${r.height}`, `${r.width} × ${r.height} — ${r.name}`)}{/each}
                </span>
              {:else}
                <button type="button" role="menuitem" disabled>{t("Set by source image…")}</button>
              {/if}
            </Menu>
          </div>
          <p id="image-size-hint" class="field-note">{validation.hint}</p>
        </section>
      {/if}

      <details id="image-advanced" bind:open={ws.d.advanced}>
        <summary>{t("Advanced options")}</summary>
        <div class="image-advanced-body">
          {#if p}{@render slider("steps", t("Steps"), 1, 50, 1)}{/if}
          {#if p?.fixed}<p class="field-note">{t("This model is distilled for %@ steps; other values cost time without adding detail.", [p.quality[0]])}</p>{/if}
          {#if p?.guidance}
            <h3>{t("Classifier-free guidance")}</h3>
            {@render slider("guidance", t("Guidance scale"), 1, 20, 0.5)}
            <label>{t("Negative prompt")}<input name="negative" placeholder={t("what to steer away from (optional)")} bind:value={ws.d.negative} /></label>
          {/if}
          <label>
            {t("Seed")}
            <span class="image-seed">
              <input name="seed" placeholder={t("random")} inputmode="numeric" bind:value={ws.d.seed} />
              <button id="image-roll" type="button" aria-label={t("Roll seed")} onclick={() => (ws.d.seed = String(randomSeed()))}><Icon name="refresh-cw" /></button>
            </span>
          </label>
          {#if p?.weights}
            <hr />
            <h3>{t("Conditioning rebalance")}</h3>
            {@render slider("gain", t("Global gain"), 0, 4, 0.1)}
            <label>{t("Layer weights (%@ numbers, comma or space separated)", [p.weights])}<input name="weights" placeholder={Array(p.weights).fill("1").join(" ")} bind:value={ws.d.weights} /></label>
            <p class="field-note">{t("Scales each tapped text-encoder layer’s contribution (1 = neutral). Empty = off.")}</p>
          {/if}
          {#if p?.transparent}<label class="check-label"><input name="transparent" type="checkbox" bind:checked={ws.d.transparent} />{t("Transparent PNG")}</label>{/if}
        </div>
      </details>
    </fieldset>
    <p id="image-validation" class="field-note" role="status">{ws.error || (validation.error === t("Prompt is empty.") ? "" : validation.error)}</p>
    <button id="image-generate" class="generate" class:primary={!busy} type="submit" disabled={!busy && (!!validation.error || ws.attaching)}>
      {#if busy}{t("Cancel")}{:else}<Icon name="sparkles" />{t("Generate")}{/if}
    </button>
    <p id="image-draft-warning" class="field-note" role="status">{ws.draftError}</p>
  </form>

  <div class="image-output">
    <div id="image-preview" class="image-preview">
      {#if c.phase === "running"}
        <div class="image-progress">
          <progress value={c.total > 0 ? Math.min(c.step, c.total) : undefined} max={c.total > 0 ? c.total : undefined}></progress>
          <p>{c.message}</p>
          <span>{c.total > 0 ? `${c.step} / ${c.total}` : ""}</span>
        </div>
      {:else if c.phase === "completed" && c.result}
        <div class="image-completed">
          <img id="image-result" src={url} alt={c.result.prompt} />
          <div class="image-result-actions">
            <span>{ws.filename}</span>
            <button id="image-download" title={t("Download PNG")} onclick={() => c.result && download(c.result.blob, ws.filename)}>{t("Save PNG")}</button>
            {#if p?.edit || p?.variation}
              <button
                id="image-reuse"
                onclick={async () => {
                  if (await ws.reuse()) prompt?.focus();
                }}>{t("Use as source")}</button
              >
            {/if}
          </div>
          <p class="field-note">{c.result.elapsedMs > 0 ? (c.result.elapsedMs / 1000).toFixed(1) + t(" seconds") : t("Saved image")} · {c.result.model}</p>
          {#if c.saveError}
            <p class="error-message">{c.saveError}</p>
            <button id="image-retry-save" onclick={() => void c.save()}>{t("Retry saving")}</button>
          {/if}
        </div>
      {:else}
        <div class="empty-state">
          <Icon name="image" />
          <h2>{c.phase === "failed" ? t("Failed") : t("No generation yet")}</h2>
          <p>{c.message || t("Enter a prompt and press Generate.")}</p>
        </div>
      {/if}
    </div>
    <button id="image-gallery-toggle" class="image-output-link" aria-expanded={gallery} onclick={() => (gallery = !gallery)}><Icon name="library" />{t("Images saved in this browser")}</button>
    <section id="image-gallery" class="image-gallery" aria-label={t("Image gallery")} hidden={!gallery}>
      {#if gallery}<ImageGallery library={ws.library} server={ws.server} onpick={(item) => ws.show(item)} />{/if}
    </section>
  </div>
</section>

{#if enhancing}
  {@const editing = ws.editing && ws.refs.length > 0}
  {@const noun = editing ? "image edit instruction" : "image prompt"}
  {@const shape = editing
    ? "Write ONE imperative instruction about the attached picture, like the examples: what to change and what must stay the same."
    : "Write one or two sentences of plain prose like the examples: subject, setting, lighting, composition, medium. Quote any text that must appear in the picture."}
  <RewriteDialog
    {connection}
    title={t("Rewrite image prompt")}
    class="image-rewrite"
    text={ws.d.prompt}
    system={`You rewrite ${noun}s for a generative model. ${shape} Keep the user's intent; make it more specific and evocative. Reply with ONLY the rewritten ${noun}, no preamble, no quotes, no markdown.\n\nExamples of the expected format:\n\n${groups.flatMap((g) => g.examples.slice(0, 2).map((e) => e.body)).join("\n\n---\n\n")}`}
    request={`Rewrite this ${noun}:\n\n${ws.d.prompt}`}
    maxTokens={512}
    textLabel={t("Rewritten prompt")}
    reviewed={t("Review the prompt, then Apply.")}
    noModel={t("No chat model is available on this server to rewrite the prompt.")}
    onapply={(text) => (ws.d.prompt = text)}
    onclose={() => (enhancing = false)}
  />
{/if}
