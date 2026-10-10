export type Inline =
  | { type: "text"; text: string }
  | { type: "code"; text: string }
  | { type: "em"; children: Inline[] }
  | { type: "strong"; children: Inline[] }
  | { type: "del"; children: Inline[] }
  | { type: "link"; href: string; title?: string; children: Inline[] }
  | { type: "br" };

export type Align = "left" | "center" | "right" | null;

export type Block =
  | { type: "paragraph"; children: Inline[] }
  | { type: "heading"; level: 1 | 2 | 3 | 4 | 5 | 6; children: Inline[] }
  | { type: "code"; lang: string; text: string }
  | { type: "quote"; children: Block[] }
  | { type: "list"; ordered: boolean; start: number; tight: boolean; items: Block[][] }
  | { type: "hr" }
  | { type: "table"; align: Align[]; head: Inline[][]; rows: Inline[][][] };
