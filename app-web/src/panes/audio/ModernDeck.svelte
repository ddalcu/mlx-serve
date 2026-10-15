<script lang="ts">
  import Icon from "../../components/Icon.svelte";
  import Menu from "../../components/Menu.svelte";
  import { t } from "../../lib/i18n/i18n";
  import type { IconName } from "../../lib/icons";
  import { motionRates } from "../../lib/amp/skin";
  import { bars, clock, Spectrum, type AmpPlayer } from "../../lib/state/amp-player.svelte";
  import { infoLine, isLocked, press, remaining, volumeIcon, type DeckKey, type DeckStatus } from "../../lib/state/deck";
  import type { AIRadio } from "../../lib/state/radio.svelte";
  import { coverColors, modelName, trackTitle } from "../../lib/state/track-meta";

  // The Music tab's deck in the Modern look: the cover and the title, the waveform seek bar, the transport and AI Radio.
  // Every key does what its MLX-Amp key does, through `deck.ts`.
  type Props = { player: AmpPlayer; radio: AIRadio; status: DeckStatus; theme: string; showPlaylist: boolean; onradiostart: () => void; onradiostop: () => void };
  let { player, radio, status, theme = $bindable(), showPlaylist = $bindable(), onradiostart, onradiostop }: Props = $props();

  const BAR_PITCH = 3;
  let cover = $state<HTMLCanvasElement>();
  let wave = $state<HTMLCanvasElement>();
  let outline = $state<number[]>([]);
  let now = $state(0);
  const spectrum = new Spectrum();
  const rates = motionRates(typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches);

  const track = $derived(player.loaded ?? player.tracks[0] ?? null);
  const info = $derived(player.known(track));
  const total = $derived(info?.seconds ?? 0);
  const playing = $derived(status === null && player.active && !player.paused);
  const moving = $derived(status === null && player.active && total > 0);
  const deck = $derived({ player, radio, track, onradiostart, onradiostop });
  const title = $derived(track ? (trackTitle(track.prompt ?? "") ?? modelName(track.model)) : t("No music yet"));
  const detail = $derived(
    track ? [modelName(track.model), new Date(track.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })].filter(Boolean).join(" · ") : "",
  );
  const elapsed = $derived((void now, moving ? player.elapsed : 0));

  $effect(() => {
    if (!track) return;
    void player.info(track);
    let live = true;
    outline = [];
    void player.outline(track).then((o) => live && (outline = o));
    return () => (live = false);
  });

  const ink = (el: Element, name: string) => getComputedStyle(el).getPropertyValue(name).trim() || "gray";

  function paintCover() {
    const ctx = cover?.getContext("2d");
    if (!cover || !ctx) return;
    const size = cover.clientWidth || 92,
      scale = globalThis.devicePixelRatio || 1;
    if (cover.width !== Math.round(size * scale)) cover.width = cover.height = Math.round(size * scale);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    const [a, b] = status === null && track ? coverColors(track.id ?? String(track.createdAt)) : [ink(cover, "--accent"), "#222"];
    const wash = ctx.createLinearGradient(0, 0, size, size);
    wash.addColorStop(0, a);
    wash.addColorStop(1, b);
    ctx.fillStyle = wash;
    ctx.fillRect(0, 0, size, size);
    const shine = ctx.createRadialGradient(size * 0.2, size * 0.1, 0, size * 0.2, size * 0.1, size);
    shine.addColorStop(0, "rgb(255 255 255 / 0.28)");
    shine.addColorStop(1, "rgb(255 255 255 / 0)");
    ctx.fillStyle = shine;
    ctx.fillRect(0, 0, size, size);
    spectrum.step(playing ? player.levels() : null);
    if (spectrum.settled && !playing) return;
    const inset = size * 0.12,
      slot = (size - 2 * inset) / spectrum.levels.length,
      width = Math.max(1.5, slot * 0.62),
      base = size - inset;
    ctx.fillStyle = "rgb(255 255 255 / 0.9)";
    spectrum.levels.forEach((level, i) => {
      const height = Math.max(2, Math.min(1, level) * size * 0.6);
      ctx.beginPath();
      ctx.roundRect(inset + i * slot + (slot - width) / 2, base - height, width, height, width / 2);
      ctx.fill();
    });
  }

  function paintWave() {
    const ctx = wave?.getContext("2d");
    if (!wave || !ctx) return;
    const w = wave.clientWidth,
      h = wave.clientHeight,
      scale = globalThis.devicePixelRatio || 1;
    if (!w || !h) return;
    if (wave.width !== Math.round(w * scale)) wave.width = Math.round(w * scale);
    if (wave.height !== Math.round(h * scale)) wave.height = Math.round(h * scale);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const fraction = moving ? Math.min(1, player.elapsed / total) : 0,
      mid = Math.round(h / 2),
      lit = ink(wave, "--accent"),
      dim = ink(wave, "--text-tertiary"),
      levels = bars(outline, Math.max(1, Math.floor((w + 1) / BAR_PITCH)));
    if (!levels.length) {
      ctx.fillStyle = dim;
      ctx.fillRect(0, mid - 1, w, 2);
      ctx.fillStyle = lit;
      ctx.fillRect(0, mid - 1, w * fraction, 2);
      return;
    }
    levels.forEach((level, i) => {
      const height = Math.max(2, level * h),
        played = (i + 0.5) / levels.length <= fraction;
      ctx.fillStyle = played ? lit : dim;
      ctx.globalAlpha = played ? 1 : 0.45;
      ctx.beginPath();
      ctx.roundRect(i * BAR_PITCH, mid - height / 2, BAR_PITCH - 1, height, (BAR_PITCH - 1) / 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }

  // Redraw when what the deck shows changes, and keep a timer only while something moves.
  $effect(() => {
    void [track, outline, status, player.active, player.paused, info];
    paintCover();
    paintWave();
    if (!playing && spectrum.settled) return;
    const timer = setInterval(() => {
      now = performance.now();
      paintCover();
      paintWave();
      if (!playing && spectrum.settled) clearInterval(timer);
    }, 1000 / rates.spectrum);
    return () => clearInterval(timer);
  });
</script>

{#snippet key(name: DeckKey, icon: IconName, label: string, lit = false, big = false)}
  <button type="button" class="modern-key" class:lit class:big aria-label={label} title={label} aria-pressed={name === "shuffle" || name === "repeatAll" ? lit : undefined} disabled={isLocked(name, radio) || (name === "playPause" && !track)} onclick={() => press(name, deck)}>
    <Icon name={icon} filled={big || name === "previous" || name === "next"} />
  </button>
{/snippet}

<div id="modern-deck" class="modern-deck">
  <div class="modern-main">
    <div class="modern-top">
      <canvas bind:this={cover} class="modern-cover" aria-hidden="true"></canvas>
      <div class="modern-head" title={status === null ? (track?.prompt ?? "") : ""}>
        {#if status?.kind === "failed"}
          <p class="modern-failed" role="status">{status.message}</p>
        {:else}
          <p class="modern-title" aria-live="polite">{status?.kind === "busy" ? status.message : title}</p>
          {#if status === null && detail}<p class="modern-detail">{detail}</p>{/if}
          {#if status === null && infoLine(info)}<p class="modern-format">{infoLine(info)}</p>{/if}
        {/if}
      </div>
      <Menu
        id="deck-more"
        icon="ellipsis"
        label={t("More")}
        onchoose={(value) => press(value === "download" ? "download" : "stop", deck)}
      >
        <button type="button" tabindex="-1" role="menuitem" data-value="download" disabled={!track}>{t("Download WAV")}</button>
        <button type="button" tabindex="-1" role="menuitem" data-value="stop" disabled={!radio.on && !player.active}>{radio.on ? t("Stop AI Radio") : t("Stop")}</button>
      </Menu>
    </div>
    {#if status?.kind === "busy"}
      <div class="modern-progress" class:sweep={status.progress === null} role="progressbar" aria-label={t("Progress")} aria-valuemin={0} aria-valuemax={1} aria-valuenow={status.progress ?? undefined}>
        <span style={status.progress === null ? "" : `width:${Math.min(1, Math.max(0, status.progress)) * 100}%`}></span>
      </div>
    {:else}
      <div class="modern-seek">
        <canvas bind:this={wave} class="modern-wave" aria-hidden="true"></canvas>
        <input class="modern-seek-input" type="range" min="0" max="1" step="0.001" aria-label={t("Seek")} disabled={!track || !total} value={total ? elapsed / total : 0} onchange={(e) => void player.seek(Number(e.currentTarget.value))} />
        {#if track}
          <div class="modern-clocks"><span>{clock(elapsed)}</span><span>{remaining(elapsed, total)}</span></div>
        {/if}
      </div>
    {/if}
    <div class="modern-controls">
      <label class="modern-volume">
        <Icon name={volumeIcon(player.volume)} />
        <input type="range" min="0" max="1" step="0.01" aria-label={t("Volume")} value={player.volume} oninput={(e) => player.setVolume(Number(e.currentTarget.value))} />
      </label>
      <div class="modern-transport">
        {@render key("shuffle", "shuffle", t("Shuffle"), player.shuffle)}
        {@render key("previous", "skip-back", t("Previous"))}
        {@render key("playPause", playing ? "pause" : "play", playing ? t("Pause") : t("Play"), false, true)}
        {@render key("next", "skip-forward", t("Next"))}
        {@render key("repeatAll", "repeat", t("Play the next track when one ends"), player.autoNext)}
      </div>
      <button type="button" class="modern-key" class:lit={showPlaylist} aria-label={t("Playlist")} title={t("Playlist")} aria-pressed={showPlaylist} onclick={() => (showPlaylist = !showPlaylist)}><Icon name="list" /></button>
    </div>
  </div>
  <div class="modern-radio">
    <span class="modern-radio-icon" class:on={radio.on}><Icon name="radio" /></span>
    {#if radio.on}
      <div id="deck-radio-strip" class="modern-strip" class:sweep={radio.progress === null}>
        <span style={radio.progress === null ? "" : `width:${radio.progress * 100}%`}></span>
        <p aria-live="polite">{radio.line}</p>
      </div>
    {:else}
      <input id="deck-radio-theme" aria-label={t("AI Radio theme")} placeholder={t("AI Radio theme, e.g. mellow lo-fi for coding")} bind:value={theme} onkeydown={(e) => e.key === "Enter" && onradiostart()} />
    {/if}
    <button type="button" id="deck-radio" aria-pressed={radio.on} onclick={() => press("radio", deck)}>{radio.on ? t("Stop AI Radio") : t("Start AI Radio")}</button>
  </div>
  {#if player.error}<p class="field-note" role="status">{player.error}</p>{/if}
</div>
