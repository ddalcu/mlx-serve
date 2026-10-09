// First-party stroke icons on a 24x24 grid, drawn for this console. Each is one path.
const circle = (cx: number, cy: number, r: number) => `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0z`;
const rect = (x: number, y: number, w: number, h: number, r = 0) =>
  `M${x + r} ${y}h${w - 2 * r}a${r} ${r} 0 0 1 ${r} ${r}v${h - 2 * r}a${r} ${r} 0 0 1 ${-r} ${r}h${-(w - 2 * r)}a${r} ${r} 0 0 1 ${-r} ${-r}v${-(h - 2 * r)}a${r} ${r} 0 0 1 ${r} ${-r}z`;
const dot = (x: number, y: number) => `M${x} ${y}h.01`;

export const icons = {
  activity: "M3 12h4l3-8 4 16 3-8h4",
  "arrow-up": "M12 19V5M6 11l6-6 6 6",
  "audio-lines": "M3 10v4M7 7v10M11 4v16M15 8v8M19 6v12M21.5 11v2",
  check: "M5 12.5l4.5 4.5L19 7",
  "chevron-down": "M6 9l6 6 6-6",
  "circle-alert": `${circle(12, 12, 9)}M12 7.5v5${dot(12, 16)}`,
  code: "M8 8l-4 4 4 4M16 8l4 4-4 4M14 5l-4 14",
  copy: `${rect(9, 9, 11, 11, 2)}M5 15V6a2 2 0 0 1 2-2h9`,
  cpu: `${rect(6, 6, 12, 12, 1.5)}${rect(9.5, 9.5, 5, 5)}M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3`,
  download: "M12 4v11M7 11l5 5 5-5M5 20h14",
  film: `${rect(3, 4, 18, 16, 2)}M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4`,
  image: `${rect(3, 4, 18, 16, 2)}${circle(9, 10, 2)}M21 17l-5-5L7 20`,
  info: `${circle(12, 12, 9)}M12 11v5${dot(12, 8)}`,
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5",
  library: `${rect(3, 4, 4, 16, 1)}${rect(9, 4, 4, 16, 1)}M16.5 5.5l3.5-1 3 15-3.5 1z`,
  lightbulb: "M9 18h6M10 21.5h4M12 2.5a6 6 0 0 0-3.6 10.8c.7.5 1.1 1.2 1.1 2.2h5c0-1 .4-1.7 1.1-2.2A6 6 0 0 0 12 2.5z",
  "message-square": "M4 5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H9l-5 4z",
  mic: `${rect(9, 3, 6, 11, 3)}M5 11a7 7 0 0 0 14 0M12 18v3`,
  monitor: `${rect(3, 4, 18, 12, 2)}M8 20h8M12 16v4`,
  music: `M9 18V6l10-2v12${circle(6.5, 18, 2.5)}${circle(16.5, 16, 2.5)}`,
  palette: `M12 3a9 9 0 1 0 0 18c1.6 0 2.2-1.1 1.7-2.3-.5-1.3.4-2.4 1.8-2.4H17a4 4 0 0 0 4-4c0-5.2-4-9.3-9-9.3z${circle(7.5, 11, 1)}${circle(10, 7, 1)}${circle(15, 7, 1)}`,
  "panel-left": `${rect(3, 4, 18, 16, 2)}M9 4v16`,
  paperclip: "M20 11l-8 8a5 5 0 0 1-7-7l9-9a3.5 3.5 0 0 1 5 5l-9 9a2 2 0 0 1-3-3l8-8",
  pencil: "M4 20l1-4L16 5a2.1 2.1 0 0 1 3 3L8 19zM14 7l3 3",
  play: "M7 5l12 7-12 7z",
  plus: "M5 12h14M12 5v14",
  "refresh-cw": "M20 11a8 8 0 0 0-14-4L4 9M4 4v5h5M4 13a8 8 0 0 0 14 4l2-2M20 20v-5h-5",
  search: `${circle(11, 11, 7)}M20 20l-4-4`,
  server: `${rect(3, 3, 18, 7, 2)}${rect(3, 14, 18, 7, 2)}${dot(7, 6.5)}${dot(7, 17.5)}`,
  settings: `${circle(12, 12, 3)}${circle(12, 12, 7)}M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9L7 7M17 17l2.1 2.1M19.1 4.9L17 7M7 17l-2.1 2.1`,
  sparkles: "M10 3l1.8 5.2L17 10l-5.2 1.8L10 17l-1.8-5.2L3 10l5.2-1.8zM19 14l.8 2.2 2.2.8-2.2.8L19 20l-.8-2.2-2.2-.8 2.2-.8z",
  stop: rect(6, 6, 12, 12, 2),
  table: `${rect(3, 4, 18, 16, 2)}M3 10h18M3 15h18M9 4v16`,
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  "volume-2": "M4 9v6h4l5 4V5L8 9zM16.5 9a4 4 0 0 1 0 6M19 6.5a8 8 0 0 1 0 11",
  wrench: "M20.5 6.5a5 5 0 0 1-7 6.5L6.5 20a2 2 0 0 1-3-3l7-7a5 5 0 0 1 6.5-7l-3 3 .5 2.5 2.5.5z",
  x: "M6 6l12 12M18 6L6 18",
} as const;

export type IconName = keyof typeof icons;
