<script lang="ts">
  import { AMP, clockScreen, glyphKey, marquee, miniWidth, motionRates, Px, repeatKey, seekBar, textKey, textWidth, volumeSlider, type Glyph } from "../../lib/amp/skin";
  import { t } from "../../lib/i18n/i18n";
  import { clock, Spectrum, type AmpPlayer, type Track } from "../../lib/state/amp-player.svelte";
  import { isLocked, press as pressKey, saveTrack, type DeckKey, type DeckStatus } from "../../lib/state/deck";
  import { blockOffset, fillWidth, STRIP_BLOCK, type AIRadio } from "../../lib/state/radio.svelte";

  // The MLX-Amp deck and its playlist: a pixel-art skin on a canvas, with a real control over every key.
  type Props = { player: AmpPlayer; radio: AIRadio; status: DeckStatus; theme: string; onradiostart: () => void; onradiostop: () => void };
  let { player, radio, status, theme = $bindable(), onradiostart, onradiostop }: Props = $props();

  let width = $state(0);
  let deck = $state<HTMLCanvasElement>();
  let list = $state<HTMLCanvasElement>();
  let down = $state("");
  let showPlaylist = $state(true);
  let lengths = $state<Record<string, string>>({});
  const spectrum = new Spectrum();
  const reduced = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const rates = motionRates(reduced);

  /** The skin stretches with the pane; the controls need its narrowest width. */
  const W = $derived(Math.max(AMP.minWidth, Math.floor(width / 1.5) || AMP.minWidth));
  const rx = 115;
  const rw = $derived(W - rx - 9);
  const seekWidth = $derived(W - 27);
  const stripWidth = $derived(W - 77);
  const track = $derived(player.loaded ?? player.tracks[0] ?? null);
  const info = $derived(player.known(track));
  const name = (tr: Track) => (tr.prompt || tr.model).replace(/\s+/g, " ").trim();
  const number = $derived(track ? player.tracks.findIndex((x) => x.id === track.id) + 1 || 1 : 0);
  const title = $derived(
    status?.kind === "busy"
      ? { text: status.message, color: AMP.lcd }
      : status?.kind === "failed"
        ? { text: status.message, color: AMP.red }
        : track
          ? { text: `*** ${number}. ${name(track)} (${clock(info?.seconds ?? 0)}) ***`, color: AMP.lcd }
          : { text: t("No music yet"), color: AMP.lcd },
  );
  const playing = $derived(status === null && player.active && !player.paused);
  /** % of the skin, so a control sits on its key at any width. */
  const at = (x: number, y: number, w: number, h: number, height = AMP.height) => `left:${(x / W) * 100}%;top:${(y / height) * 100}%;width:${(w / W) * 100}%;height:${(h / height) * 100}%`;

  function paint() {
    if (!deck) return;
    const scale = Math.max(1, Math.round((width / W) * (globalThis.devicePixelRatio || 1))) || 1;
    if (deck.width !== W * scale) deck.width = W * scale;
    if (deck.height !== AMP.height * scale) deck.height = AMP.height * scale;
    const ctx = deck.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.imageSmoothingEnabled = false;
    const p = new Px(ctx),
      now = performance.now() / 1000;
    spectrum.step(status === null ? player.levels() : null);
    p.window("MLX-AMP", W, AMP.height);
    clockScreen(p, 11, 22, status === null && player.active ? (player.paused ? "paused" : "playing") : null, status === null ? player.elapsed : 0, spectrum.levels, spectrum.peaks);
    p.screen(rx, 24, rw, 13);
    const scrolls = rates.title > 0 && textWidth(title.text) > rw - 6 && (playing || status !== null);
    marquee(p, title.text, title.color, rx, 24, rw, 13, scrolls ? now : null);
    // Bitrate, sample rate, stereo.
    const kbps = String(info?.kbps ?? 0),
      khz = String(Math.round((info?.sampleRate ?? 0) / 1000)),
      right = W - 4,
      stereo = (info?.channels ?? 2) > 1,
      dim = "#77778f";
    p.screen(rx, 41, 25, 11);
    p.text(kbps, rx + 24 - textWidth(kbps), 43, AMP.lcd);
    p.miniText("KBPS", rx + 28, 44, AMP.white);
    p.screen(rx + 47, 41, 15, 11);
    p.text(khz, rx + 61 - textWidth(khz), 43, AMP.lcd);
    p.miniText("KHZ", rx + 65, 44, AMP.white);
    const monoX = right - miniWidth("STEREO") - 4 - miniWidth("MONO"),
      both = monoX - rx >= 79;
    if (both || stereo) p.miniText("STEREO", right - miniWidth("STEREO"), 44, stereo ? AMP.lcd : dim);
    if (both || !stereo) p.miniText("MONO", both ? monoX : right - miniWidth("MONO"), 44, stereo ? dim : AMP.lcd);
    volumeSlider(p, rx, 57, 58, player.volume);
    textKey(p, "PL", showPlaylist, W - 33, 58, 23, 12, down === "pl");
    const total = info?.seconds ?? 0,
      moving = status === null && player.active && total > 0,
      busy = status?.kind === "busy" ? (status.progress ?? 0) : 0;
    seekBar(p, 16, 72, seekWidth, moving ? Math.min(1, player.elapsed / total) : busy, moving || busy > 0);
    const keys: [Glyph, string, number, number, number, number][] = [
      ["prev", "prev", 16, 88, 23, 18],
      [playing ? "pause" : "play", "play", 39, 88, 23, 18],
      ["stop", "stop", 62, 88, 23, 18],
      ["next", "next", 85, 88, 22, 18],
      ["eject", "eject", 113, 89, 22, 16],
    ];
    for (const [glyph, id, x, y, w, h] of keys) glyphKey(p, glyph, x, y, w, h, down === id);
    if (radio.on) ctx.globalAlpha = 0.4;
    textKey(p, "SHUFFLE", player.shuffle, 138, 89, 56, 15, down === "shuffle");
    repeatKey(p, player.autoNext, 194, 89, down === "repeat");
    ctx.globalAlpha = 1;
    textKey(p, "RADIO", radio.on, 16, 111, 46, 15, down === "radio");
    p.screen(66, 111, stripWidth, 15);
    if (radio.on) {
      const inner = stripWidth - 2,
        progress = radio.progress;
      if (progress !== null) p.rect(67, 112, fillWidth(progress, inner), 13, "#15601c");
      else p.rect(67 + blockOffset(now, inner), 112, STRIP_BLOCK, 13, "#15601c");
      marquee(p, radio.line, AMP.lcd, 66, 111, stripWidth, 15, rates.title > 0 ? now : null);
    }
    paintPlaylist();
  }

  function paintPlaylist() {
    if (!list || !showPlaylist) return;
    const scale = Math.max(1, Math.round((width / W) * (globalThis.devicePixelRatio || 1))) || 1;
    if (list.width !== W * scale) list.width = W * scale;
    if (list.height !== AMP.playlistHeight * scale) list.height = AMP.playlistHeight * scale;
    const ctx = list.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    const p = new Px(ctx);
    p.window("PLAYLIST", W, AMP.playlistHeight);
    p.screen(8, 17, W - 16, AMP.playlistHeight - 25);
  }

  // Redraw when anything the skin shows changes, and keep a timer only while something moves.
  $effect(() => {
    void [W, width, title, info, player.active, player.paused, player.volume, player.shuffle, player.autoNext, player.tracks, radio.on, radio.line, radio.progress, status, down, showPlaylist, theme];
    paint();
    const moving = () => playing || !spectrum.settled || radio.on || (status !== null && rates.title > 0);
    if (!moving()) return;
    const fps = radio.on ? Math.max(rates.title, 4) : playing ? rates.spectrum : Math.max(rates.title, rates.seek);
    const timer = setInterval(() => {
      paint();
      if (!moving()) clearInterval(timer);
    }, 1000 / fps);
    return () => clearInterval(timer);
  });

  // Each row's length, read once from its header.
  $effect(() => {
    for (const tr of player.tracks.slice(0, 200))
      if (tr.id && !(tr.id in lengths)) void player.info(tr).then((i) => (lengths = { ...lengths, [tr.id!]: i ? clock(i.seconds) : "" }));
  });

  const press = (id: string) => ({ onpointerdown: () => (down = id), onpointerup: () => (down = ""), onpointerleave: () => (down = "") });
  const deckKey = (key: DeckKey) => pressKey(key, { player, radio, track, onradiostart, onradiostop });
</script>

<div class="amp" bind:clientWidth={width}>
  <div class="amp-window">
    <canvas bind:this={deck} class="amp-skin" style="aspect-ratio:{W}/{AMP.height}" aria-hidden="true"></canvas>
    <p class="sr-only" aria-live="polite">{title.text}</p>
    <input class="amp-hit" style={at(rx, 57, 58, 13)} type="range" min="0" max="1" step="0.01" aria-label={t("Volume")} value={player.volume} oninput={(e) => player.setVolume(Number(e.currentTarget.value))} />
    <button type="button" class="amp-hit" style={at(W - 33, 58, 23, 12)} aria-label={t("Playlist")} aria-pressed={showPlaylist} {...press("pl")} onclick={() => (showPlaylist = !showPlaylist)}></button>
    <input
      class="amp-hit"
      style={at(16, 72, seekWidth, 10)}
      type="range"
      min="0"
      max="1"
      step="0.001"
      aria-label={t("Position")}
      disabled={!track || status !== null}
      value={info?.seconds ? player.elapsed / info.seconds : 0}
      onchange={(e) => void player.seek(Number(e.currentTarget.value))}
    />
    <button type="button" class="amp-hit" style={at(16, 88, 23, 18)} aria-label={t("Previous")} disabled={isLocked("previous", radio)} {...press("prev")} onclick={() => deckKey("previous")}></button>
    <button type="button" class="amp-hit" style={at(39, 88, 23, 18)} aria-label={playing ? t("Pause") : t("Play")} disabled={!track} {...press("play")} onclick={() => deckKey("playPause")}></button>
    <button type="button" class="amp-hit" style={at(62, 88, 23, 18)} aria-label={t("Stop")} {...press("stop")} onclick={() => deckKey("stop")}></button>
    <button type="button" class="amp-hit" style={at(85, 88, 22, 18)} aria-label={t("Next")} {...press("next")} onclick={() => deckKey("next")}></button>
    <button type="button" class="amp-hit" style={at(113, 89, 22, 16)} aria-label={t("Download WAV")} disabled={!track} {...press("eject")} onclick={() => deckKey("download")}></button>
    <button type="button" class="amp-hit" style={at(138, 89, 56, 15)} aria-label={t("Shuffle")} aria-pressed={player.shuffle} disabled={isLocked("shuffle", radio)} {...press("shuffle")} onclick={() => deckKey("shuffle")}></button>
    <button type="button" class="amp-hit" style={at(194, 89, 28, 15)} aria-label={t("Play the next track when one ends")} aria-pressed={player.autoNext} disabled={isLocked("repeatAll", radio)} {...press("repeat")} onclick={() => deckKey("repeatAll")}></button>
    <button type="button" id="amp-radio" class="amp-hit" style={at(16, 111, 46, 15)} aria-label={radio.on ? t("Stop AI Radio") : t("Start AI Radio")} aria-pressed={radio.on} {...press("radio")} onclick={() => deckKey("radio")}></button>
    {#if radio.on}
      <p class="sr-only" aria-live="polite">{radio.line}</p>
    {:else}
      <input
        id="amp-theme"
        class="amp-theme"
        style={at(70, 113, stripWidth - 8, 11)}
        aria-label={t("AI Radio theme")}
        placeholder={t("AI Radio theme, e.g. mellow lo-fi for coding")}
        bind:value={theme}
        onkeydown={(e) => e.key === "Enter" && onradiostart()}
      />
    {/if}
  </div>
  {#if showPlaylist}
    <div class="amp-window">
      <canvas bind:this={list} class="amp-skin" style="aspect-ratio:{W}/{AMP.playlistHeight}" aria-hidden="true"></canvas>
      <ol class="amp-playlist" style={at(9, 18, W - 18, AMP.playlistHeight - 27, AMP.playlistHeight)} aria-label={t("Playlist")}>
        {#each player.tracks as tr, i (tr.id)}
          {@const current = player.active && player.loaded?.id === tr.id}
          <li class:current>
            <button type="button" disabled={isLocked("historyRow", radio)} title={name(tr)} onclick={() => (current ? player.stop() : void player.play(tr))}>{i + 1}. {name(tr)}</button>
            <span>{lengths[tr.id!] ?? ""}</span>
            <button type="button" aria-label={t("Download %@", [name(tr)])} onclick={() => void saveTrack(player, tr)}>⤓</button>
          </li>
        {:else}
          <li class="amp-empty">{t("No music saved on this server yet.")}</li>
        {/each}
      </ol>
    </div>
  {/if}
  {#if player.error}<p class="field-note" role="status">{player.error}</p>{/if}
</div>
