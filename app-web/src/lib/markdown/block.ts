import type { Align, Block } from "./ast";
import { decode, normalizeLabel, parseInline, type Refs } from "./inline";

type Raw =
  | { type: "paragraph"; text: string }
  | { type: "heading"; level: 1 | 2 | 3 | 4 | 5 | 6; text: string }
  | { type: "code"; lang: string; text: string }
  | { type: "quote"; children: Raw[] }
  | { type: "list"; ordered: boolean; start: number; tight: boolean; items: Raw[][] }
  | { type: "hr" }
  | { type: "table"; align: Align[]; head: string[]; rows: string[][] };

const MAX_DEPTH = 24;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const CLOSING_FENCE = /^ {0,3}(`{3,}|~{3,})[ \t]*$/;
const ATX = /^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$/;
const HR = /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const QUOTE = /^ {0,3}>/;
const LIST = /^( {0,3})([-+*]|\d{1,9}[.)])(?:([ \t]+)(.*)|$)/;
const SETEXT = /^ {0,3}(=+|-+)[ \t]*$/;
const DEFINITION = /^ {0,3}\[((?:[^\]\\]|\\.)+)\]:[ \t]*(<[^>\n]*>|\S+)(?:[ \t]+("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\((?:[^)\\]|\\.)*\)))?[ \t]*$/;

const isBlank = (line: string) => !line.trim();

/** Indentation in columns, tabs to the next multiple of four. */
function indent(line: string): number {
  let col = 0;
  for (const ch of line) {
    if (ch === " ") col++;
    else if (ch === "\t") col += 4 - (col % 4);
    else break;
  }
  return col;
}

/** Remove `n` columns of indentation, splitting a tab that straddles the edge. */
function strip(line: string, n: number): string {
  let col = 0, i = 0;
  while (i < line.length && col < n) {
    if (line[i] === " ") (col++, i++);
    else if (line[i] === "\t") {
      const width = 4 - (col % 4);
      if (col + width > n) return " ".repeat(col + width - n) + line.slice(i + 1);
      col += width;
      i++;
    } else break;
  }
  return line.slice(i);
}

type Marker = { ordered: boolean; start: number; delim: string; offset: number; empty: boolean; text: string };

function marker(line: string): Marker | null {
  const m = LIST.exec(line);
  if (!m || indent(line) > 3) return null;
  const mark = m[2]!, rest = m[4] ?? "", head = m[1]!.length + mark.length;
  let spaces = 0;
  for (const ch of m[3] ?? "") spaces += ch === "\t" ? 4 - ((head + spaces) % 4) : 1;
  const empty = !rest.trim();
  const ordered = /\d/.test(mark[0]!);
  return {
    ordered,
    start: ordered ? parseInt(mark, 10) : 1,
    delim: ordered ? mark.slice(-1) : mark,
    offset: empty || spaces >= 5 ? head + 1 : head + spaces,
    empty,
    text: empty ? "" : spaces >= 5 ? " ".repeat(spaces - 1) + rest : rest,
  };
}

const openFence = (line: string) => {
  const m = FENCE.exec(line);
  return m && !(m[1]![0] === "`" && m[2]!.includes("`")) ? m : null;
};

/** Does `lines[i]` end a paragraph that precedes it? */
function interrupts(lines: string[], i: number): boolean {
  const line = lines[i]!;
  if (indent(line) >= 4) return false;
  if (openFence(line) || ATX.test(line) || HR.test(line) || QUOTE.test(line)) return true;
  const mk = marker(line);
  if (mk && !mk.empty && (!mk.ordered || mk.start === 1)) return true;
  return !!table(lines, i);
}

/** `\|` keeps its pipe; cells split on the others. Optional leading and trailing pipes are dropped. */
function cells(line: string): string[] {
  const out: string[] = [];
  let cell = "";
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "\\" && line[i + 1] === "|") (cell += "|", i++);
    else if (line[i] === "|") (out.push(cell), (cell = ""));
    else cell += line[i];
  }
  out.push(cell);
  if (out.length && !out[0]!.trim()) out.shift();
  if (out.length && !out[out.length - 1]!.trim()) out.pop();
  return out.map((c) => c.trim());
}

function table(lines: string[], i: number): { block: Raw; next: number } | null {
  const head = lines[i]!, rule = lines[i + 1];
  if (rule === undefined || !head.includes("|") || indent(head) >= 4 || indent(rule) >= 4 || !/^[-:|][-:|\s]*$/.test(rule.trim())) return null;
  if (rule.trim()[0] === "-" && /\s/.test(rule.trim()[1] ?? "x") ) return null;
  const parts = rule.split("|");
  const align: Align[] = [];
  for (const [k, part] of parts.entries()) {
    const p = part.trim();
    if (!p) {
      if (k === 0 || k === parts.length - 1) continue;
      return null;
    }
    if (!/^:?-+:?$/.test(p)) return null;
    align.push(p.startsWith(":") ? (p.endsWith(":") ? "center" : "left") : p.endsWith(":") ? "right" : null);
  }
  const header = cells(head);
  if (!header.length || header.length !== align.length) return null;
  const rows: string[][] = [];
  let next = i + 2;
  while (next < lines.length && !isBlank(lines[next]!) && indent(lines[next]!) < 4 && !terminatesTable(lines[next]!)) {
    const row = cells(lines[next]!);
    rows.push(Array.from({ length: align.length }, (_, k) => row[k] ?? ""));
    next++;
  }
  return { block: { type: "table", align, head: header, rows }, next };
}

const terminatesTable = (line: string) => !!(openFence(line) || ATX.test(line) || HR.test(line) || QUOTE.test(line) || marker(line));

/** Inside an open code fence a line can never continue a paragraph. */
function insideFence(lines: string[]): boolean {
  let open: string | undefined;
  for (const line of lines) {
    if (open === undefined) {
      const m = openFence(line);
      if (m) open = m[1]!;
    } else {
      const m = CLOSING_FENCE.exec(line);
      if (m && m[1]![0] === open[0] && m[1]!.length >= open.length) open = undefined;
    }
  }
  return open !== undefined;
}

function parseBlocks(lines: string[], refs: Refs, depth: number): { blocks: Raw[]; loose: boolean } {
  const blocks: Raw[] = [];
  let loose = false, blank = false, i = 0;
  const add = (block: Raw) => {
    if (blocks.length && blank) loose = true;
    blank = false;
    blocks.push(block);
  };
  const nested = depth < MAX_DEPTH;
  while (i < lines.length) {
    const line = lines[i]!;
    if (isBlank(line)) {
      blank = true;
      i++;
      continue;
    }
    if (indent(line) >= 4) {
      const code: string[] = [];
      while (i < lines.length && (isBlank(lines[i]!) || indent(lines[i]!) >= 4)) code.push(strip(lines[i++]!, 4));
      let trailing = 0;
      while (code.length && !code[code.length - 1]!.trim()) (code.pop(), trailing++);
      add({ type: "code", lang: "", text: code.join("\n") + "\n" });
      blank = trailing > 0;
      continue;
    }
    const fence = openFence(line);
    if (fence) {
      const body: string[] = [], pad = line.length - line.trimStart().length, ch = fence[1]![0]!;
      let closed = false;
      for (i++; i < lines.length && !closed; i++) {
        const close = CLOSING_FENCE.exec(lines[i]!);
        if (close && close[1]![0] === ch && close[1]!.length >= fence[1]!.length) closed = true;
        else body.push(lines[i]!.replace(new RegExp(`^ {0,${pad}}`), ""));
      }
      // An unclosed fence (a reply still streaming) ends where the text does.
      add({ type: "code", lang: fence[2]!.trim().split(/\s+/)[0] ?? "", text: body.join("\n") + (closed && body.length ? "\n" : "") });
      continue;
    }
    const atx = ATX.exec(line);
    if (atx) {
      add({ type: "heading", level: atx[1]!.length as 1, text: atx[2]!.replace(/(?:^|[ \t]+)#+[ \t]*$/, "").trim() });
      i++;
      continue;
    }
    if (HR.test(line)) {
      add({ type: "hr" });
      i++;
      continue;
    }
    if (nested && QUOTE.test(line)) {
      const inner: string[] = [];
      for (; i < lines.length; i++) {
        const l = lines[i]!;
        if (QUOTE.test(l)) inner.push(l.replace(/^ {0,3}> ?/, ""));
        else {
          const prev = inner[inner.length - 1];
          const paragraphText = prev !== undefined && !isBlank(prev) && !openFence(prev) && !ATX.test(prev) && !HR.test(prev);
          if (isBlank(l) || !paragraphText || interrupts(lines, i)) break;
          inner.push(l);
        }
      }
      add({ type: "quote", children: parseBlocks(inner, refs, depth + 1).blocks });
      continue;
    }
    const first = marker(line);
    if (nested && first) {
      const items: Raw[][] = [];
      let tight = true;
      while (i < lines.length) {
        const mk = marker(lines[i]!);
        if (!mk || mk.ordered !== first.ordered || mk.delim !== first.delim || HR.test(lines[i]!)) break;
        const item = [mk.text];
        for (i++; i < lines.length; ) {
          const l = lines[i]!;
          if (isBlank(l)) {
            let j = i;
            while (j < lines.length && isBlank(lines[j]!)) j++;
            if (j < lines.length && indent(lines[j]!) >= mk.offset) {
              for (; i < j; i++) item.push("");
              continue;
            }
            break;
          }
          if (indent(l) >= mk.offset) item.push(strip(l, mk.offset));
          else {
            const prev = item[item.length - 1]!;
            if (marker(l) || isBlank(prev) || insideFence(item) || interrupts(lines, i)) break;
            item.push(l.trimStart());
          }
          i++;
        }
        const parsed = parseBlocks(item, refs, depth + 1);
        items.push(parsed.blocks);
        if (parsed.loose) tight = false;
        let j = i;
        while (j < lines.length && isBlank(lines[j]!)) j++;
        const next = j < lines.length ? marker(lines[j]!) : null;
        if (j > i && next && next.ordered === first.ordered && next.delim === first.delim && !HR.test(lines[j]!)) {
          tight = false;
          i = j;
        } else if (j > i) break;
      }
      add({ type: "list", ordered: first.ordered, start: first.start, tight, items });
      continue;
    }
    const grid = table(lines, i);
    if (grid) {
      add(grid.block);
      i = grid.next;
      continue;
    }
    const para = [line];
    let heading: 1 | 2 | undefined;
    for (i++; i < lines.length; i++) {
      const l = lines[i]!;
      if (isBlank(l)) break;
      const underline = SETEXT.exec(l);
      if (underline) {
        heading = underline[1]![0] === "=" ? 1 : 2;
        i++;
        break;
      }
      if (interrupts(lines, i)) break;
      para.push(l);
    }
    while (!heading && para.length) {
      const def = DEFINITION.exec(para[0]!);
      if (!def) break;
      const label = normalizeLabel(decode(def[1]!));
      const dest = def[2]!.startsWith("<") ? def[2]!.slice(1, -1) : def[2]!;
      if (!refs.has(label)) refs.set(label, { href: decode(dest), title: def[3] ? decode(def[3].slice(1, -1)) : undefined });
      para.shift();
    }
    if (!para.length) continue;
    const text = para.map((l) => l.replace(/^[ \t]+/, "")).join("\n").replace(/[ \t]+$/, "");
    add(heading ? { type: "heading", level: heading, text } : { type: "paragraph", text });
  }
  return { blocks, loose };
}

function finish(block: Raw, refs: Refs): Block {
  switch (block.type) {
    case "paragraph":
      return { type: "paragraph", children: parseInline(block.text, refs) };
    case "heading":
      return { type: "heading", level: block.level, children: parseInline(block.text, refs) };
    case "quote":
      return { type: "quote", children: block.children.map((b) => finish(b, refs)) };
    case "list":
      return { ...block, items: block.items.map((item) => item.map((b) => finish(b, refs))) };
    case "table":
      return {
        type: "table",
        align: block.align,
        head: block.head.map((c) => parseInline(c, refs)),
        rows: block.rows.map((row) => row.map((c) => parseInline(c, refs))),
      };
    default:
      return block;
  }
}

/** Markdown to an AST. Never throws: input the parser cannot handle shows up as plain text. */
export function parseMarkdown(source: string): Block[] {
  const refs: Refs = new Map();
  try {
    const { blocks } = parseBlocks(source.replace(/\r\n?/g, "\n").split("\n"), refs, 0);
    return blocks.map((b) => finish(b, refs));
  } catch {
    return source.trim() ? [{ type: "paragraph", children: [{ type: "text", text: source }] }] : [];
  }
}
