// film.tsx: the Remotion side of the mlx-serve film skill. `node film.mjs init <dir>` copies it to
// <dir>/src/index.tsx; it turns <dir>/timeline.json (+ song.json from `film.mjs song`) into a video.
// Every time in timeline.json is a SONG second; the video starts at song.start and ends at song.end.
import React from "react";
import { AbsoluteFill, Audio, Composition, Easing, Img, OffthreadVideo, Sequence, interpolate, random, registerRoot, staticFile, useCurrentFrame } from "remotion";
import { loadFont } from "@remotion/fonts";
import timeline from "../timeline.json";
import song from "../song.json";

type Spec = { [k: string]: any };
const T: Spec = timeline as Spec;
const SONG: Spec = song as Spec;
const FPS: number = T.fps || 30;
const W: number = T.width || 1920;
const H: number = T.height || 1080;
const START: number = T.song?.start || 0;
const END: number = T.song?.end || START + 10;
/** Song seconds -> frames of the video. */
const F = (t: number) => Math.round((t - START) * FPS);
const len = (b: Spec) => Math.max(1, F(b.to) - F(b.from));
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

const INK = T.colors?.ink || "#f6f2ff";
const ACCENT = T.colors?.accent || "#a78bfa";
const WARM = T.colors?.warm || "#ffcf9a";
const GLOW = T.colors?.glow || "124,58,237";
const SERIF = "Instrument Serif";
const SANS = "Inter";
const MONO = "JetBrains Mono";
for (const w of ["400", "600", "700", "800", "900"]) loadFont({ family: SANS, url: staticFile("_film/inter-" + w + ".woff2"), weight: w });
loadFont({ family: SERIF, url: staticFile("_film/instrument-serif-400.woff2"), weight: "400" });
loadFont({ family: SERIF, url: staticFile("_film/instrument-serif-400-italic.woff2"), weight: "400", style: "italic" });
loadFont({ family: MONO, url: staticFile("_film/jetbrains-mono-400.woff2"), weight: "400" });

const fade = (f: number, dur: number, fin = 0, fout = 0) =>
  Math.min(fin ? interpolate(f, [0, fin], [0, 1], clamp) : 1, fout ? interpolate(f, [dur - fout, dur], [1, 0], clamp) : 1);
const fades = (b: Spec): [number, number] => {
  const v = b.fade ?? [0.2, 0];
  return Array.isArray(v) ? [Math.round(v[0] * FPS), Math.round((v[1] ?? 0) * FPS)] : [Math.round(v * FPS), Math.round(v * FPS)];
};
const isVideo = (src: string) => /\.(mp4|mov|webm|m4v)$/i.test(src);
const pair = (v: any, d: [number, number]): [number, number] => (Array.isArray(v) ? [v[0], v[1] ?? v[0]] : v == null ? d : [v, v]);

// ---------- picture blocks ----------

/** An image with a slow camera move: zoom [from, to], optional pan [x0, y0, x1, y1] in % of the frame. */
const Still: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const k = interpolate(f, [0, dur], [0, 1], clamp);
  const [z0, z1] = pair(b.zoom, [1, 1.08]);
  const p = b.pan || [0, 0, 0, 0];
  const [fin, fout] = fades(b);
  const s = z0 + (z1 - z0) * k;
  return (
    <AbsoluteFill style={{ overflow: "hidden", opacity: fade(f, dur, fin, fout) }}>
      <Img src={staticFile(b.src)} style={{ width: "100%", height: "100%", objectFit: "cover", transformOrigin: b.origin || "50% 50%", filter: b.filter, transform: "translate(" + (p[0] + (p[2] - p[0]) * k) + "%, " + (p[1] + (p[3] - p[1]) * k) + "%) scale(" + s + ")" }} />
    </AbsoluteFill>
  );
};

/** A video clip, full frame; `trim` skips its first seconds, `volume` > 0 plays its own sound. */
const Clip: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const [z0, z1] = pair(b.zoom, [1, 1]);
  const [fin, fout] = fades(b);
  const s = interpolate(f, [0, dur], [z0, z1], clamp);
  return (
    <AbsoluteFill style={{ overflow: "hidden", opacity: fade(f, dur, fin, fout) }}>
      <OffthreadVideo src={staticFile(b.src)} trimBefore={Math.round((b.trim || 0) * FPS)} playbackRate={b.rate || 1} muted={!b.volume} volume={b.volume || 0} style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scale(" + s + ")", transformOrigin: b.origin || "50% 50%", filter: b.filter }} />
    </AbsoluteFill>
  );
};

/** Deep navy with a soft accent glow: the ground for screens, titles and cards. */
const Backdrop: React.FC<{ glow?: number }> = ({ glow = 1 }) => (
  <AbsoluteFill style={{ background: "radial-gradient(ellipse 70% 60% at 50% 42%, rgba(" + GLOW + "," + 0.42 * glow + ") 0%, rgba(30,20,70," + 0.5 * glow + ") 38%, #07060f 72%, #030307 100%)" }} />
);

/** A screen capture (image or video) in a window frame, zooming about `focus` ([x, y] in fractions, or [[x0, y0], [x1, y1]]). */
const Screen: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const k = Easing.inOut(Easing.cubic)(interpolate(f, [0, dur], [0, 1], clamp));
  const [z0, z1] = pair(b.zoom, [1, 1]);
  const fo = Array.isArray(b.focus?.[0]) ? b.focus : [b.focus || [0.5, 0.5], b.focus || [0.5, 0.5]];
  const fx = fo[0][0] + (fo[1][0] - fo[0][0]) * k, fy = fo[0][1] + (fo[1][1] - fo[0][1]) * k;
  const z = z0 + (z1 - z0) * k;
  const width = b.width || 1520, h = width * (b.aspect || 9 / 16);
  const [fin, fout] = fades(b);
  const tx = (0.5 - fx) * width * (z - 1), ty = (0.5 - fy) * h * (z - 1);
  return (
    <AbsoluteFill style={{ opacity: fade(f, dur, fin, fout) }}>
      <Backdrop glow={b.glow ?? 1} />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <div style={{ transform: "translate(" + tx + "px, " + ty + "px) scale(" + z + ")" }}>
          <div style={{ width, borderRadius: 16, overflow: "hidden", border: "1px solid rgba(255,255,255,0.14)", boxShadow: "0 50px 140px rgba(0,0,0,0.65), 0 0 120px rgba(" + GLOW + ",0.25)", background: "#1d1d20" }}>
            {b.url && (
              <div style={{ height: 46, background: "#2a2a2e", display: "flex", alignItems: "center", gap: 10, padding: "0 18px" }}>
                {["#ff5f57", "#febc2e", "#28c840"].map((c) => <div key={c} style={{ width: 13, height: 13, borderRadius: 7, background: c }} />)}
                <div style={{ marginLeft: 24, flex: 1, height: 28, borderRadius: 8, background: "#1c1c1f", color: "#c9c9d1", fontFamily: SANS, fontSize: 16, display: "flex", alignItems: "center", paddingLeft: 14 }}>{b.url}</div>
              </div>
            )}
            <div style={{ width, height: h }}>
              {isVideo(b.src) ? <OffthreadVideo src={staticFile(b.src)} trimBefore={Math.round((b.trim || 0) * FPS)} playbackRate={b.rate || 1} muted style={{ width: "100%", height: "100%" }} />
                : <Img src={staticFile(b.src)} style={{ width: "100%", height: "100%" }} />}
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/** Pictures popping into a grid on a stagger. */
const Wall: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const items: string[] = b.items || [];
  const cols = b.cols || 4, gap = 14, tile = (W - gap * (cols + 1)) / cols;
  const rows = Math.ceil(items.length / cols);
  const zoom = interpolate(f, [0, dur], [1.08, 1.0], { ...clamp, easing: Easing.out(Easing.cubic) });
  return (
    <AbsoluteFill style={{ backgroundColor: "#06050c" }}>
      <div style={{ position: "absolute", left: "50%", top: "50%", width: W, height: rows * (tile + gap) + gap, transform: "translate(-50%, -50%) scale(" + zoom + ")" }}>
        {items.map((src, i) => {
          const order = (i * 7) % items.length;
          const stagger = (b.stagger ?? 0.05) * FPS;
          const a = interpolate(f, [order * stagger, order * stagger + 7], [0, 1], { ...clamp, easing: Easing.out(Easing.back(1.4)) });
          return <Img key={src + i} src={staticFile(src)} style={{ position: "absolute", left: gap + (i % cols) * (tile + gap), top: gap + Math.floor(i / cols) * (tile + gap), width: tile, height: tile, objectFit: "cover", borderRadius: 10, opacity: a, transform: "scale(" + (0.6 + 0.4 * a) + ")" }} />;
        })}
      </div>
    </AbsoluteFill>
  );
};

/** Up to nine images or clips moving at once, with an optional title over them. */
const Grid: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const items: string[] = (b.items || []).slice(0, 9);
  const cols = items.length <= 4 ? 2 : 3, rows = Math.ceil(items.length / cols), gap = 10;
  const w = (W - gap * (cols + 1)) / cols, h = (H - gap * (rows + 1)) / rows;
  return (
    <AbsoluteFill style={{ backgroundColor: "#040309" }}>
      <AbsoluteFill style={{ transform: "scale(" + interpolate(f, [0, dur], [1.1, 1.0], clamp) + ")" }}>
        {items.map((src, k) => (
          <div key={src + k} style={{ position: "absolute", left: gap + (k % cols) * (w + gap), top: gap + Math.floor(k / cols) * (h + gap), width: w, height: h, overflow: "hidden", borderRadius: 8, opacity: interpolate(f, [k * 1.2, k * 1.2 + 6], [0, 1], clamp) }}>
            {isVideo(src) ? <OffthreadVideo src={staticFile(src)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : <Img src={staticFile(src)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />}
          </div>
        ))}
      </AbsoluteFill>
      {b.title && <AbsoluteFill style={{ background: "rgba(4,3,10,0.35)" }} />}
      {b.title && <Title b={{ text: b.title }} dur={dur} />}
    </AbsoluteFill>
  );
};

// ---------- text blocks ----------

const where = (position: string | undefined) => (position === "top" ? "12%" : position === "center" ? "50%" : "84%");

/** A line of text in the serif face: sung lyrics and quiet statements. */
const Caption: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const lift = interpolate(f, [0, 14], [14, 0], { ...clamp, easing: Easing.out(Easing.cubic) });
  const [fin, fout] = b.fade == null ? [9, 9] : fades(b);
  const center = b.position === "center";
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", top: where(b.position), width: "100%", textAlign: "center", transform: "translateY(-50%) translateY(" + lift + "px)", opacity: fade(f, dur, fin, fout), fontFamily: SERIF, fontStyle: "italic", fontSize: b.size || (center ? 84 : 60), color: b.color || INK, textShadow: "0 2px 22px rgba(0,0,0,0.9)", padding: "0 6%" }}>{b.text}</div>
    </AbsoluteFill>
  );
};

/** A big punch-in title (Inter Black), with an optional serif subtitle. */
const Title: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const size = b.size || 150;
  const pop = interpolate(f, [0, 8], [1.28, 1], { ...clamp, easing: Easing.out(Easing.exp) }) * interpolate(f, [8, dur], [1, 1.04], clamp);
  const split = interpolate(f, [0, 7], [14, 0], clamp);
  const top = b.position === "top" ? 30 : b.position === "bottom" ? 70 : 46;
  const base: React.CSSProperties = { position: "absolute", top: top + "%", width: "100%", textAlign: "center", fontFamily: SANS, fontWeight: 900, fontSize: size, letterSpacing: -3, lineHeight: 0.95, textTransform: b.upper === false ? "none" : "uppercase", transform: "translateY(-50%) scale(" + pop + ")", padding: "0 4%" };
  return (
    <AbsoluteFill style={{ opacity: fade(f, dur, 2, 5) }}>
      {split > 0.5 && <div style={{ ...base, color: "rgba(255,40,90,0.7)", marginLeft: -split, mixBlendMode: "screen" }}>{b.text}</div>}
      {split > 0.5 && <div style={{ ...base, color: "rgba(40,200,255,0.7)", marginLeft: split, mixBlendMode: "screen" }}>{b.text}</div>}
      <div style={{ ...base, color: INK, textShadow: "0 0 40px rgba(167,139,250,0.55), 0 6px 30px rgba(0,0,0,0.6)" }}>{b.text}</div>
      {b.sub && <div style={{ position: "absolute", top: "calc(" + top + "% + " + size * 0.62 + "px)", width: "100%", textAlign: "center", fontFamily: SERIF, fontStyle: "italic", fontSize: size * 0.32, color: WARM, opacity: interpolate(f, [4, 12], [0, 1], clamp) }}>{b.sub}</div>}
    </AbsoluteFill>
  );
};

/** Words that land one by one on their (song) times. */
const Words: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ opacity: fade(f, dur, 0, 5) }}>
      <div style={{ position: "absolute", top: "50%", width: "100%", textAlign: "center", transform: "translateY(-50%)", fontFamily: SANS, fontWeight: 900, fontSize: b.size || 170, letterSpacing: -4, color: INK, textTransform: "uppercase", lineHeight: 1, padding: "0 4%" }}>
        {(b.words || []).map((w: [string, number], i: number) => {
          const at = F(w[1]) - F(b.from);
          const k = interpolate(f, [at, at + 6], [0, 1], { ...clamp, easing: Easing.out(Easing.exp) });
          return <span key={i} style={{ display: "inline-block", margin: "0 0.12em", opacity: k, transform: "translateY(" + (1 - k) * 40 + "px) scale(" + (1.25 - 0.25 * k) + ")", textShadow: "0 0 50px rgba(255,214,150,0.6), 0 6px 30px rgba(0,0,0,0.6)" }}>{w[0]}</span>;
        })}
      </div>
    </AbsoluteFill>
  );
};

/** A freeze-frame introduction: name, role, one line, over a dimmed background block. */
const Card: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const wipe = interpolate(f, [2, 10], [0, 100], { ...clamp, easing: Easing.out(Easing.exp) });
  const roleK = interpolate(f, [6, 14], [0, 1], clamp), subK = interpolate(f, [10, 18], [0, 1], clamp);
  const name: string = b.name || "";
  return (
    <AbsoluteFill style={{ backgroundColor: "#05040c" }}>
      <AbsoluteFill style={{ transform: "scale(" + interpolate(f, [0, dur], [1.04, 1.12], clamp) + ")", filter: "saturate(0.55) brightness(0.62) contrast(1.08)" }}>{b.bg ? renderPicture(b.bg, dur) : <Backdrop />}</AbsoluteFill>
      <AbsoluteFill style={{ background: "linear-gradient(to right, rgba(5,4,12,0.94) 0%, rgba(5,4,12,0.8) 38%, transparent 70%)" }} />
      <div style={{ position: "absolute", left: 120, top: "50%", transform: "translateY(-50%)", width: 1100 }}>
        {b.label !== "" && <div style={{ fontFamily: SANS, fontWeight: 700, fontSize: 24, letterSpacing: 9, color: ACCENT, marginBottom: 18, opacity: roleK }}>{b.label || (b.n ? "CREW · " + String(b.n).padStart(2, "0") : "")}</div>}
        <div style={{ fontFamily: SANS, fontWeight: 900, fontSize: name.length > 14 ? 90 : 104, letterSpacing: -3, color: INK, lineHeight: 0.98, whiteSpace: "nowrap", clipPath: "inset(0 " + (100 - wipe) + "% 0 0)" }}>{name}</div>
        <div style={{ height: 6, width: interpolate(f, [4, 14], [0, 220], clamp), background: "linear-gradient(90deg, " + ACCENT + ", #60a5fa)", borderRadius: 3, margin: "26px 0 22px" }} />
        {b.role && <div style={{ fontFamily: SERIF, fontStyle: "italic", fontSize: 76, color: WARM, opacity: roleK }}>{b.role}</div>}
        {b.sub && <div style={{ fontFamily: SANS, fontWeight: 500, fontSize: 32, color: "rgba(246,242,255,0.78)", marginTop: 14, opacity: subK }}>{b.sub}</div>}
      </div>
    </AbsoluteFill>
  );
};

/** A terminal window replaying real output line by line. */
const Terminal: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const cmd: string = b.command || "";
  const per = b.perLine || 2;
  const typed = Math.min(cmd.length, Math.floor(f * 5));
  const shown = Math.max(0, Math.floor((f - Math.ceil(cmd.length / 5) - 1) / per));
  return (
    <AbsoluteFill>
      <Backdrop glow={0.7} />
      <div style={{ position: "absolute", left: b.x ?? 160, top: b.y ?? 200, width: b.w ?? 1600, height: b.h ?? 680, borderRadius: 14, overflow: "hidden", background: "rgba(13,12,20,0.94)", border: "1px solid rgba(255,255,255,0.12)", boxShadow: "0 40px 120px rgba(0,0,0,0.6)", opacity: fade(f, dur, 5, 5) }}>
        <div style={{ height: 40, background: "#1f1d29", display: "flex", alignItems: "center", gap: 9, padding: "0 16px" }}>
          {["#ff5f57", "#febc2e", "#28c840"].map((c) => <div key={c} style={{ width: 12, height: 12, borderRadius: 6, background: c }} />)}
          <div style={{ flex: 1, textAlign: "center", fontFamily: SANS, fontSize: 16, color: "#9a96ad", marginRight: 60 }}>{b.title || "zsh"}</div>
        </div>
        <div style={{ padding: "18px 22px", fontFamily: MONO, fontFeatureSettings: '"calt" 0, "liga" 0', fontSize: 22, lineHeight: 1.55, color: "#d9d6e8" }}>
          <div><span style={{ color: "#7ee787" }}>{b.cwd || "~"}</span> <span style={{ color: ACCENT }}>❯</span> {cmd.slice(0, typed)}</div>
          {(b.lines || []).slice(0, shown).slice(-16).map((l: string, i: number) => <div key={i} style={{ color: "#a9a4c0", whiteSpace: "pre" }}>{l}</div>)}
        </div>
      </div>
    </AbsoluteFill>
  );
};

/** A film-style credits roll: sections of [role, name] rows or plain name lists. */
const Credits: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const y = interpolate(f, [0, dur], [H, -(b.height || 1500)], clamp);
  return (
    <AbsoluteFill style={{ opacity: fade(f, dur, 10, 12) }}>
      {b.bg && <AbsoluteFill>{renderPicture(b.bg, dur)}</AbsoluteFill>}
      {b.bg && <AbsoluteFill style={{ background: "linear-gradient(to left, rgba(5,4,10,0.88) 0%, rgba(5,4,10,0.7) 40%, rgba(5,4,10,0.1) 75%)" }} />}
      <div style={{ position: "absolute", left: b.bg ? 1050 : 580, width: 760, top: y, textAlign: "center", color: INK }}>
        {(b.sections || []).map((s: Spec, i: number) => (
          <div key={i} style={{ marginBottom: 80 }}>
            <div style={{ fontFamily: SANS, fontWeight: 700, fontSize: 26, letterSpacing: 10, color: ACCENT, marginBottom: 30 }}>{s.heading}</div>
            {(s.rows || []).map((r: [string, string]) => (
              <div key={r[0]} style={{ display: "flex", justifyContent: "center", gap: 28, margin: "16px 0", alignItems: "baseline" }}>
                <div style={{ width: 330, textAlign: "right", fontFamily: SANS, fontWeight: 500, fontSize: 26, letterSpacing: 3, color: "rgba(246,242,255,0.6)", textTransform: "uppercase" }}>{r[0]}</div>
                <div style={{ width: 330, textAlign: "left", fontFamily: SERIF, fontStyle: "italic", fontSize: 44 }}>{r[1]}</div>
              </div>
            ))}
            {s.names && <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px 30px", fontFamily: SANS, fontWeight: 500, fontSize: 30 }}>{s.names.map((n: string) => <div key={n}>{n}</div>)}</div>}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

const Logo: React.FC<{ b: Spec; width: number }> = ({ b, width }) =>
  b.wordmark ? (
    <div style={{ width, height: (width * (b.wordmarkAspect || 118 / 960)), backgroundColor: INK, WebkitMaskImage: "url(" + staticFile(b.wordmark) + ")", WebkitMaskSize: "100% 100%", maskImage: "url(" + staticFile(b.wordmark) + ")", maskSize: "100% 100%" }} />
  ) : (
    <div style={{ fontFamily: SANS, fontWeight: 900, fontSize: width / 7, letterSpacing: -2, color: INK }}>{b.title}</div>
  );

/** A brand card (icon, wordmark or title, version): the thumbnail on frame 0, or a logo slam. */
const Brand: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const k = b.punch === false ? 1 : interpolate(f, [0, 9], [0, 1], { ...clamp, easing: Easing.out(Easing.exp) });
  const out = b.fadeOut === false ? 1 : interpolate(f, [dur - 6, dur], [1, 0], clamp);
  return (
    <AbsoluteFill style={{ opacity: out }}>
      <Backdrop glow={1.25} />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", flexDirection: "column", gap: 34, transform: "scale(" + (1.3 - 0.3 * k) + ")" }}>
        {b.icon && <div style={{ width: 200, height: 200, borderRadius: 46, overflow: "hidden", boxShadow: "0 0 90px rgba(" + GLOW + ",0.7)" }}><Img src={staticFile(b.icon)} style={{ width: "100%", height: "100%" }} /></div>}
        <Logo b={b} width={920} />
        {b.version && <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 60, letterSpacing: 16, color: ACCENT }}>{b.version}</div>}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/** The last card: icon, wordmark, version, one line, a URL and copyable commands (code ligatures off). */
const EndCard: React.FC<{ b: Spec; dur: number }> = ({ b, dur }) => {
  const f = useCurrentFrame();
  const pulse = 0.75 + 0.25 * Math.sin(f / 9);
  const k = interpolate(f, [0, 24], [0, 1], { ...clamp, easing: Easing.out(Easing.cubic) });
  const at = (a: number) => interpolate(f, [a, a + 18], [0, 1], clamp);
  return (
    <AbsoluteFill style={{ opacity: fade(f, dur, 0, Math.round(1.4 * FPS)) }}>
      <Backdrop glow={0.9 + 0.2 * pulse} />
      <AbsoluteFill style={{ alignItems: "center" }}>
        {b.icon && <div style={{ marginTop: 90, width: 230, height: 230, borderRadius: 52, overflow: "hidden", opacity: k, transform: "scale(" + (0.85 + 0.15 * k) + ")", boxShadow: "0 0 " + 110 * pulse + "px rgba(" + GLOW + ",0.75)" }}><Img src={staticFile(b.icon)} style={{ width: "100%", height: "100%" }} /></div>}
        <div style={{ marginTop: b.icon ? 56 : 260, opacity: at(8) }}><Logo b={b} width={760} /></div>
        {b.version && <div style={{ marginTop: 26, fontFamily: SANS, fontWeight: 700, fontSize: 30, letterSpacing: 12, color: ACCENT, opacity: at(14) }}>{b.version}</div>}
        {b.line && <div style={{ marginTop: 44, fontFamily: SERIF, fontStyle: "italic", fontSize: 44, color: WARM, opacity: at(26), padding: "0 6%", textAlign: "center" }}>{b.line}</div>}
        <div style={{ marginTop: 44, display: "flex", gap: 44, alignItems: "center", opacity: at(40) }}>
          {b.url && <div style={{ fontFamily: SANS, fontWeight: 800, fontSize: 46, color: INK }}>{b.url}</div>}
          {b.url && b.code && <div style={{ width: 2, height: 110, background: "rgba(255,255,255,0.25)" }} />}
          {b.code && (
            <div style={{ fontFamily: MONO, fontFeatureSettings: '"calt" 0, "liga" 0', fontSize: 25, lineHeight: 1.6, color: "#d7d2ea", padding: "14px 24px", borderRadius: 12, background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.14)" }}>
              {b.code.map((l: string) => <div key={l}><span style={{ color: ACCENT }}>$</span> {l}</div>)}
            </div>
          )}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// ---------- effects ----------

const Flash: React.FC<{ dur: number; color?: string; peak?: number }> = ({ dur, color = "#fff", peak = 1 }) => {
  const f = useCurrentFrame();
  return <AbsoluteFill style={{ backgroundColor: color, opacity: interpolate(f, [0, dur], [peak, 0], { ...clamp, easing: Easing.out(Easing.quad) }) }} />;
};

const Letterbox: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const h = interpolate(f, [0, 8], [0, H * 0.12], { ...clamp, easing: Easing.out(Easing.cubic) }) * fade(f, dur, 0, 8);
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: h, background: "#000" }} />
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: h, background: "#000" }} />
    </AbsoluteFill>
  );
};

/** Film grain and a vignette over everything. */
const Grain: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ pointerEvents: "none" }}>
      <AbsoluteFill style={{ backgroundImage: "url(" + staticFile("_film/noise.png") + ")", backgroundPosition: Math.floor(random("gx" + f) * 512) + "px " + Math.floor(random("gy" + f) * 512) + "px", opacity: 0.07, mixBlendMode: "overlay" }} />
      <AbsoluteFill style={{ background: "radial-gradient(ellipse 85% 80% at 50% 50%, transparent 55%, rgba(0,0,0,0.55) 100%)" }} />
    </AbsoluteFill>
  );
};

// ---------- timeline ----------

/** The picture behind a card or the credits: a still, a clip or a screen spec (times come from the parent). */
function renderPicture(p: Spec | string, dur: number): React.ReactNode {
  const spec: Spec = typeof p === "string" ? { src: p } : p;
  const type = spec.type || (isVideo(spec.src || "") ? "clip" : "still");
  if (type === "screen") return <Screen b={spec} dur={dur} />;
  if (type === "clip") return <Clip b={{ fade: 0, ...spec }} dur={dur} />;
  return <Still b={{ fade: 0, ...spec }} dur={dur} />;
}

/** Sung lines from song.json inside the block's range, each held until the next line starts. */
const Lyrics: React.FC<{ b: Spec }> = ({ b }) => {
  const lines: Spec[] = (SONG.lines || []).filter((l: Spec) => l.start >= b.from && l.start < b.to);
  return (
    <>
      {b.scrim !== false && <Sequence from={F(b.from)} durationInFrames={len(b)}><AbsoluteFill style={{ background: "linear-gradient(to top, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.24) 22%, transparent 42%)" }} /></Sequence>}
      {lines.map((l, i) => {
        const next = lines[i + 1]?.start ?? b.to;
        const until = Math.min(next, l.end + 1.2) - 0.05;
        return <Sequence key={i} from={F(l.start)} durationInFrames={Math.max(1, F(until) - F(l.start))}><Caption b={{ text: l.text, size: b.size, position: b.position }} dur={Math.max(1, F(until) - F(l.start))} /></Sequence>;
      })}
    </>
  );
};

function renderBlock(b: Spec, i: number): React.ReactNode {
  if (b.type === "sfx") return <Sequence key={i} from={F(b.at)}><Audio src={staticFile(b.src)} volume={b.volume ?? 0.5} /></Sequence>;
  if (b.type === "flash") {
    const d = Math.round((b.duration || 0.3) * FPS);
    return <Sequence key={i} from={F(b.at)} durationInFrames={d}><Flash dur={d} color={b.color} peak={b.peak ?? 0.8} /></Sequence>;
  }
  if (b.type === "lyrics") return <Lyrics key={i} b={b} />;
  const dur = len(b);
  const body: { [k: string]: React.ReactNode } = {
    black: <AbsoluteFill style={{ backgroundColor: "#000" }} />,
    backdrop: <Backdrop glow={b.glow ?? 1} />,
    still: <Still b={b} dur={dur} />,
    clip: <Clip b={b} dur={dur} />,
    screen: <Screen b={b} dur={dur} />,
    wall: <Wall b={b} dur={dur} />,
    grid: <Grid b={b} dur={dur} />,
    caption: <Caption b={b} dur={dur} />,
    title: <Title b={b} dur={dur} />,
    words: <Words b={b} dur={dur} />,
    card: <Card b={b} dur={dur} />,
    terminal: <Terminal b={b} dur={dur} />,
    credits: <Credits b={b} dur={dur} />,
    brand: <Brand b={b} dur={dur} />,
    endcard: <EndCard b={b} dur={dur} />,
    letterbox: <Letterbox dur={dur} />,
  };
  return <Sequence key={i} from={F(b.from)} durationInFrames={dur} name={b.type}>{body[b.type] ?? null}</Sequence>;
}

const Film: React.FC = () => {
  const song: Spec = T.song || {};
  const fadeOut = Math.round((song.fadeOut ?? 3) * FPS);
  const total = F(END);
  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {(T.blocks || []).map(renderBlock)}
      {T.grain !== false && <Grain />}
      {song.file && <Audio src={staticFile(song.file)} trimBefore={Math.round(START * FPS)} volume={(f) => (song.volume ?? 1) * interpolate(f, [total - fadeOut, total], [1, 0], clamp)} />}
    </AbsoluteFill>
  );
};

const Root: React.FC = () => <Composition id="Film" component={Film} durationInFrames={Math.max(1, F(END))} fps={FPS} width={W} height={H} />;
registerRoot(Root);
