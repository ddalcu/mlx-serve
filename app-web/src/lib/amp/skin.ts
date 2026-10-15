// The MLX-Amp skin as pixel art: every coordinate is a pixel of the classic main window,
// drawn as rectangles into a 2D canvas scaled to whole device pixels, so nothing is smoothed.

export const AMP = {
  /** Narrowest window, in skin pixels, the controls fit in. */
  minWidth: 236,
  height: 132,
  playlistHeight: 125,
  chassis: "#3a3760",
  chassisLight: "#8583b0",
  chassisDark: "#14132a",
  innerLight: "#514e80",
  innerDark: "#242245",
  lcd: "#00e000",
  lcdDim: "#0a3a0a",
  white: "#e8e8f4",
  goldLight: "#e6d08c",
  gold: "#c4a95a",
  goldDark: "#7a6430",
  keyFace: "#b7bfd0",
  keyHi: "#f4f6fc",
  keyHi2: "#dde2ee",
  keyLo: "#2a2c42",
  keyLo2: "#7c849e",
  ink: "#1c1d2e",
  red: "#f04030",
};

/** 5 x 7 dot-matrix capitals, rows top to bottom, bit 4 the left column. */
const GLYPHS: Record<string, number[]> = {
  A: [0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11], B: [0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e],
  C: [0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e], D: [0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e],
  E: [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f], F: [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
  G: [0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f], H: [0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
  I: [0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e], J: [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c],
  K: [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11], L: [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f],
  M: [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11], N: [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
  O: [0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e], P: [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
  Q: [0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d], R: [0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11],
  S: [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e], T: [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
  U: [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e], V: [0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04],
  W: [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a], X: [0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11],
  Y: [0x11, 0x11, 0x11, 0x0a, 0x04, 0x04, 0x04], Z: [0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f],
  "0": [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e], "1": [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
  "2": [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f], "3": [0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e],
  "4": [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02], "5": [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
  "6": [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e], "7": [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
  "8": [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e], "9": [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
  ".": [0, 0, 0, 0, 0, 0x0c, 0x0c], ",": [0, 0, 0, 0, 0x0c, 0x04, 0x08], "-": [0, 0, 0, 0x1f, 0, 0, 0],
  _: [0, 0, 0, 0, 0, 0, 0x1f], ":": [0, 0x0c, 0x0c, 0, 0x0c, 0x0c, 0], "(": [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
  ")": [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08], "*": [0, 0x04, 0x15, 0x0e, 0x15, 0x04, 0],
  "/": [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10], "'": [0x04, 0x04, 0x08, 0, 0, 0, 0], " ": [0, 0, 0, 0, 0, 0, 0],
  "+": [0, 0x04, 0x04, 0x1f, 0x04, 0x04, 0], "!": [0x04, 0x04, 0x04, 0x04, 0x04, 0, 0x04],
  "?": [0x0e, 0x11, 0x01, 0x02, 0x04, 0, 0x04], '"': [0x0a, 0x0a, 0x0a, 0, 0, 0, 0],
};
/** 3 x 5 capitals for the small labels and the vertical O-A-I-D-U-V strip. */
const MINI: Record<string, number[]> = {
  O: [7, 5, 5, 5, 7], A: [2, 5, 7, 5, 5], I: [7, 2, 2, 2, 7], D: [6, 5, 5, 5, 6], U: [5, 5, 5, 5, 7], V: [5, 5, 5, 5, 2],
  K: [5, 5, 6, 5, 5], B: [6, 5, 6, 5, 6], P: [6, 5, 6, 4, 4], S: [3, 4, 2, 1, 6], H: [5, 5, 7, 5, 5], Z: [7, 1, 2, 4, 7],
  M: [5, 7, 7, 5, 5], N: [6, 5, 5, 5, 5], T: [7, 2, 2, 2, 2], E: [7, 4, 6, 4, 7], R: [6, 5, 6, 5, 5],
};
/** Lit segments of a numeral: bit i = a b c d e f g. */
const SEGMENTS = [0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111, 0b1111111, 0b1101111];

/** What the font can draw: capitals, with what it lacks folded to its plain look so no stray "?" shows. */
export function printable(s: string) {
  const folded = s
    .replaceAll("…", "...")
    .replace(/[–—]/g, "-")
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .normalize("NFD")
    .replace(/\p{M}/gu, "");
  return folded.toUpperCase();
}
export const textWidth = (s: string, advance = 6) => [...printable(s)].length * advance - 1;
export const miniWidth = (s: string) => s.length * 4 - 1;

/** Rectangles in skin pixels over a canvas already scaled to the skin. A pixel is a rect, so nothing needs anti-aliasing. */
export class Px {
  constructor(readonly ctx: CanvasRenderingContext2D) {}
  rect(x: number, y: number, w: number, h: number, color: string) {
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, w, h);
  }
  /** 1-px frame, light on the top/left edges and dark on the bottom/right. */
  bevel(x: number, y: number, w: number, h: number, light: string, dark: string) {
    this.rect(x, y, w, 1, light);
    this.rect(x, y, 1, h, light);
    this.rect(x, y + h - 1, w, 1, dark);
    this.rect(x + w - 1, y, 1, h, dark);
  }
  /** A recessed black screen. */
  screen(x: number, y: number, w: number, h: number) {
    this.rect(x, y, w, h, "#000");
    this.bevel(x, y, w, h, AMP.chassisDark, AMP.chassisLight);
  }
  clipped(x: number, y: number, w: number, h: number, body: () => void) {
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(x, y, w, h);
    this.ctx.clip();
    body();
    this.ctx.restore();
  }
  text(s: string, x: number, y: number, color: string, advance = 6) {
    this.ctx.fillStyle = color;
    let at = x;
    for (const ch of printable(s)) {
      const rows = GLYPHS[ch] ?? GLYPHS["?"]!;
      rows.forEach((bits, row) => {
        for (let col = 0; col < 5; col++) if (bits & (0b10000 >> col)) this.ctx.fillRect(at + col, y + row, 1, 1);
      });
      at += advance;
    }
  }
  miniText(s: string, x: number, y: number, color: string) {
    this.ctx.fillStyle = color;
    [...s].forEach((c, i) =>
      (MINI[c] ?? []).forEach((bits, row) => {
        for (let col = 0; col < 3; col++) if (bits & (0b100 >> col)) this.ctx.fillRect(x + i * 4 + col, y + row, 1, 1);
      }),
    );
  }
  /** Seven-segment numeral in a 9 x 13 cell. */
  digit(d: number, x: number, y: number, color: string) {
    const segs = [[1, 0, 7, 2], [7, 1, 2, 5], [7, 7, 2, 5], [1, 11, 7, 2], [0, 7, 2, 5], [0, 1, 2, 5], [1, 5, 7, 2]];
    segs.forEach(([sx, sy, w, h], i) => SEGMENTS[d]! & (1 << i) && this.rect(x + sx!, y + sy!, w!, h!, color));
  }

  /** A window: bevelled chassis with the gold-railed title bar. */
  window(title: string, w: number, h: number) {
    this.rect(0, 0, w, h, AMP.chassis);
    this.bevel(0, 0, w, h, AMP.chassisLight, AMP.chassisDark);
    this.bevel(1, 1, w - 2, h - 2, AMP.innerLight, AMP.innerDark);
    [3, 7, 5, 7, 3].forEach((bar, i) => this.rect(5 + i * 2, 3 + (7 - bar) / 2, 1, bar, AMP.gold));
    const tw = textWidth(title),
      tx = Math.round((w - tw) / 2);
    for (const [x0, x1] of [[18, tx - 5], [tx + tw + 5, w - 6]] as const) {
      this.rect(x0, 4, x1 - x0, 2, AMP.goldLight);
      this.rect(x0, 6, x1 - x0, 1, AMP.gold);
      this.rect(x0, 8, x1 - x0, 2, AMP.gold);
      this.rect(x0, 10, x1 - x0, 1, AMP.goldDark);
    }
    this.text(title, tx, 4, AMP.white);
  }

  /** A key face; `down` swaps the bevel and shifts the ink one pixel. */
  key(x: number, y: number, w: number, h: number, down: boolean) {
    this.rect(x, y, w, h, AMP.keyFace);
    if (down) {
      this.bevel(x, y, w, h, AMP.keyLo, AMP.keyHi);
      this.bevel(x + 1, y + 1, w - 2, h - 2, AMP.keyLo2, AMP.keyHi2);
    } else {
      this.bevel(x, y, w, h, AMP.keyHi, AMP.keyLo);
      this.bevel(x + 1, y + 1, w - 2, h - 2, AMP.keyHi2, AMP.keyLo2);
    }
    return down ? 1 : 0;
  }
}

/** How fast the moving parts move; reduced motion gets a slower spectrum and a still title. */
export const motionRates = (reduced: boolean) => (reduced ? { spectrum: 6, title: 0, seek: 2 } : { spectrum: 12, title: 6, seek: 4 });
const TITLE_GAP = 36;
/** How far a title has scrolled `time` seconds in: whole pixels, 0 while it fits. */
export const titleOffset = (text: number, box: number, time: number) => (text > box ? Math.trunc(time * 14) % Math.trunc(text + TITLE_GAP) : 0);
/** Where a pointer at `x` (skin pixels from the slider's left edge) puts the thumb, 0...1. */
export const sliderFraction = (x: number, length: number, thumb: number) => Math.min(1, Math.max(0, (x - thumb / 2) / Math.max(1, length - thumb)));

/** A screen with one line of text that scrolls when it does not fit and `time` is given. */
export function marquee(p: Px, text: string, color: string, x: number, y: number, w: number, h: number, time: number | null) {
  const box = w - 6,
    width = textWidth(text),
    offset = time === null ? 0 : titleOffset(width, box, time),
    ty = y + Math.floor((h - 7) / 2);
  p.clipped(x + 3, y + 1, box, h - 2, () => {
    p.text(text, x + 3 - offset, ty, color);
    if (width > box) p.text(text, x + 3 + width + TITLE_GAP - offset, ty, color);
  });
}

const levelColor = (t: number) => (t < 0.45 ? "#20d020" : t < 0.7 ? "#c8d020" : t < 0.85 ? "#e09020" : "#e02010");

/** The clock + spectrum screen (98 x 38). `state` is null when stopped. */
export function clockScreen(p: Px, x: number, y: number, state: "playing" | "paused" | null, seconds: number, levels: number[], peaks: number[]) {
  p.screen(x, y, 98, 38);
  [..."OAIDUV"].forEach((c, i) => p.miniText(c, x + 3, y + 3 + i * 6, "#5a5a70"));
  if (state === "playing") for (let r = 0; r < 9; r++) p.rect(x + 15, y + 4 + r, Math.min(r, 8 - r) + 1, 1, AMP.lcd);
  else if (state === "paused") {
    p.rect(x + 15, y + 4, 3, 9, AMP.lcd);
    p.rect(x + 20, y + 4, 3, 9, AMP.lcd);
  } else p.rect(x + 15, y + 5, 7, 7, AMP.lcd);
  p.rect(x + 15, y + 14, 3, 2, state === "playing" ? "#c03020" : "#501810");
  const t = Math.trunc(Math.max(0, seconds));
  p.digit(Math.trunc(t / 600) % 10, x + 29, y + 4, AMP.lcd);
  p.digit(Math.trunc(t / 60) % 10, x + 41, y + 4, AMP.lcd);
  p.rect(x + 55, y + 7, 2, 2, AMP.lcd);
  p.rect(x + 55, y + 12, 2, 2, AMP.lcd);
  p.digit(Math.trunc((t % 60) / 10), x + 60, y + 4, AMP.lcd);
  p.digit(t % 10, x + 72, y + 4, AMP.lcd);
  // Spectrum: 19 bars, 3 px wide with 1 px between, a grey cap hanging above each.
  levels.forEach((level, b) => {
    const bx = x + 13 + b * 4;
    for (let r = 0; r < Math.trunc(16 * level); r++) p.rect(bx, y + 36 - r, 3, 1, levelColor(r / 16));
    const cap = Math.trunc(16 * (peaks[b] ?? 0));
    if (cap > 0) p.rect(bx, y + 36 - cap, 3, 1, "#b0b0b8");
  });
  for (let dx = 13; dx < 89; dx += 2) p.rect(x + dx, y + 36, 1, 1, "#2e48a0");
}

/** The seek bar (w x 10): a groove and, while something is loaded, the gold thumb. */
export function seekBar(p: Px, x: number, y: number, w: number, fraction: number, thumb: boolean) {
  p.rect(x, y, w, 10, "#000");
  p.bevel(x, y, w, 10, AMP.chassisDark, AMP.chassisLight);
  if (!thumb) return;
  const tx = x + Math.round((w - 29) * fraction);
  p.rect(tx, y, 29, 10, AMP.gold);
  p.bevel(tx, y, 29, 10, AMP.goldLight, AMP.goldDark);
  p.bevel(tx + 3, y + 2, 23, 6, AMP.goldDark, AMP.goldLight);
}

/** The volume slider: a groove filled green to red, and a thumb. */
export function volumeSlider(p: Px, x: number, y: number, length: number, fraction: number) {
  const thumb = 14,
    tx = Math.round((length - thumb) * fraction),
    filled = tx + thumb / 2;
  p.rect(x, y + 3, length, 7, "#000");
  p.bevel(x, y + 3, length, 7, AMP.chassisDark, AMP.chassisLight);
  for (let dx = 1; dx < Math.min(length - 1, filled); dx += 2) {
    const t = dx / length;
    p.rect(x + dx, y + 4, Math.min(2, filled - dx), 5, `rgb(${Math.round(Math.min(1, 0.1 + t * 1.6) * 255)},${Math.round(Math.min(0.85, 1.4 - t * 1.2) * 255)},13)`);
  }
  p.rect(x + tx, y + 1, thumb, 11, AMP.keyFace);
  p.bevel(x + tx, y + 1, thumb, 11, AMP.keyHi, AMP.keyLo);
  p.bevel(x + tx + 1, y + 2, thumb - 2, 9, AMP.keyHi2, AMP.keyLo2);
  for (let g = 0; g < 3; g++) p.rect(x + tx + 4 + g * 2, y + 4, 1, 5, AMP.keyLo2);
}

export type Glyph = "prev" | "play" | "pause" | "stop" | "next" | "eject";
/** A transport key's glyph, centred in its face. */
export function glyphKey(p: Px, glyph: Glyph, x: number, y: number, w: number, h: number, down: boolean) {
  const s = p.key(x, y, w, h, down),
    ink = AMP.ink,
    cx = x + Math.round(w / 2) + s,
    cy = y + Math.round(h / 2) + s;
  const triangle = (left: number, forward: boolean) => {
    for (let r = 0; r < 9; r++) {
      const tw = Math.round((Math.min(r, 8 - r) + 1) * 1.6);
      p.rect(forward ? left : left + 8 - tw, cy - 4 + r, tw, 1, ink);
    }
  };
  if (glyph === "play") triangle(cx - 4, true);
  else if (glyph === "pause") {
    p.rect(cx - 4, cy - 4, 3, 9, ink);
    p.rect(cx + 1, cy - 4, 3, 9, ink);
  } else if (glyph === "stop") p.rect(cx - 4, cy - 4, 8, 8, ink);
  else if (glyph === "prev") {
    p.rect(cx - 5, cy - 4, 2, 9, ink);
    triangle(cx - 3, false);
  } else if (glyph === "next") {
    triangle(cx - 5, true);
    p.rect(cx + 3, cy - 4, 2, 9, ink);
  } else {
    for (let r = 0; r < 5; r++) p.rect(cx - r - 1, cy - 5 + r, r * 2 + 2, 1, ink);
    p.rect(cx - 6, cy + 1, 12, 2, ink);
  }
}

/** A key with a label and, for toggles, an LED. */
export function textKey(p: Px, text: string, led: boolean | null, x: number, y: number, w: number, h: number, down: boolean) {
  const s = p.key(x, y, w, h, down),
    ledW = led === null ? 0 : 6,
    tx = x + Math.round((w - ledW - textWidth(text)) / 2) + ledW + s,
    ty = y + Math.round((h - 7) / 2) + s;
  if (led !== null) p.rect(tx - ledW + 1, ty + 2, 3, 3, led ? AMP.lcd : AMP.lcdDim);
  p.text(text, tx, ty, AMP.ink);
}

/** The repeat key: an LED and a loop arrow. */
export function repeatKey(p: Px, lit: boolean, x: number, y: number, down: boolean) {
  const s = p.key(x, y, 28, 15, down),
    ink = AMP.ink,
    lx = x + 10 + s,
    ly = y + 4 + s;
  p.rect(x + 4 + s, y + 6 + s, 3, 3, lit ? AMP.lcd : AMP.lcdDim);
  p.rect(lx, ly, 10, 1, ink);
  p.rect(lx, ly + 6, 10, 1, ink);
  p.rect(lx, ly + 1, 1, 5, ink);
  p.rect(lx + 9, ly + 1, 1, 2, ink);
  p.rect(lx + 7, ly + 3, 5, 1, ink);
  p.rect(lx + 8, ly + 4, 3, 1, ink);
  p.rect(lx + 9, ly + 5, 1, 1, ink);
}
