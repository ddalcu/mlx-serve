import type { Inline } from "./ast";

/** Link reference definitions by normalized label. */
export type Refs = Map<string, { href: string; title?: string }>;

type Delim = { delim: true; ch: "*" | "_" | "~"; count: number; orig: number; open: boolean; close: boolean };
type Item = Inline | Delim;

const PUNCT = /[\p{P}\p{S}]/u;
const ESCAPABLE = /[!-/:-@[-`{-~]/;
const SPECIAL = /[\\`<&!*_~[\n]/g;
const MAX_DEPTH = 8;
const MAX_TAIL = 4096; // longest destination or title: bounds the scan for text like `[a]([a]([a](`
const ENTITIES: Record<string, string> = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", copy: "©", reg: "®", trade: "™",
  hellip: "…", mdash: "—", ndash: "–", times: "×", rarr: "→", larr: "←",
};

export const normalizeLabel = (label: string) => label.trim().replace(/\s+/g, " ").toLowerCase();

/** The only links the console follows: absolute http(s). */
export function safeHref(dest: string): string | null {
  if (!/^https?:\/\//i.test(dest)) return null;
  try {
    const url = new URL(dest);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function entity(name: string): string | undefined {
  if (name[0] !== "#") return ENTITIES[name];
  const code = name[1] === "x" || name[1] === "X" ? parseInt(name.slice(2), 16) : parseInt(name.slice(1), 10);
  return code > 0 && code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff) ? String.fromCodePoint(code) : "�";
}

/** Backslash escapes and entities, as in link destinations and titles. */
export const decode = (s: string) =>
  s.replace(/\\([!-/:-@[-`{-~])|&(#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7}|[A-Za-z][A-Za-z0-9]{1,31});/g, (all, escaped, name) =>
    escaped ?? entity(name) ?? all,
  );

/** One scan over the source: closing backtick runs by length, and bracket pairs. Linear, so degenerate input cannot go quadratic. */
function index(src: string) {
  const ticks = new Map<number, number[]>();
  for (let i = 0; i < src.length; ) {
    if (src[i] === "`") {
      let n = 1;
      while (src[i + n] === "`") n++;
      (ticks.get(n) ?? ticks.set(n, []).get(n)!).push(i);
      i += n;
    } else i++;
  }
  /** Start of the next backtick run of exactly `n` at or after `from`. */
  const closer = (n: number, from: number) => {
    const list = ticks.get(n);
    if (!list) return -1;
    let lo = 0, hi = list.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid]! < from) lo = mid + 1;
      else hi = mid;
    }
    return list[lo] ?? -1;
  };
  const pairs = new Map<number, number>();
  const stack: number[] = [];
  for (let i = 0; i < src.length; ) {
    const c = src[i];
    if (c === "\\") i += 2;
    else if (c === "`") {
      let n = 1;
      while (src[i + n] === "`") n++;
      const end = closer(n, i + n);
      i = end < 0 ? i + n : end + n;
    } else {
      if (c === "[") stack.push(i);
      else if (c === "]" && stack.length) pairs.set(stack.pop()!, i);
      i++;
    }
  }
  return { closer, pairs };
}

/** `(destination "title")` after a link label; null when it is not one. */
function inlineTail(src: string, pos: number): { href: string; title?: string; end: number } | null {
  if (src[pos] !== "(") return null;
  let i = pos + 1;
  const skip = () => {
    const from = i;
    while (i < src.length && (src[i] === " " || src[i] === "\t" || src[i] === "\n")) i++;
    return i - from;
  };
  skip();
  let dest: string;
  if (src[i] === "<") {
    let j = i + 1;
    while (j < src.length && j - i < MAX_TAIL && src[j] !== ">" && src[j] !== "\n" && src[j] !== "<") j += src[j] === "\\" ? 2 : 1;
    if (src[j] !== ">") return null;
    dest = decode(src.slice(i + 1, j));
    i = j + 1;
  } else {
    let j = i, depth = 0;
    while (j < src.length && j - i < MAX_TAIL) {
      const ch = src[j]!;
      if (ch === "\\" && ESCAPABLE.test(src[j + 1] ?? "")) j += 2;
      else if (/[\s\u0000-\u001f\u007f]/.test(ch)) break;
      else if (ch === "(") {
        depth++;
        j++;
      } else if (ch === ")") {
        if (depth === 0) break;
        depth--;
        j++;
      } else j++;
    }
    if (depth !== 0 || j - i >= MAX_TAIL) return null;
    dest = decode(src.slice(i, j));
    i = j;
  }
  let title: string | undefined;
  if (skip() > 0 && (src[i] === '"' || src[i] === "'" || src[i] === "(")) {
    const close = src[i] === "(" ? ")" : src[i]!;
    let j = i + 1;
    while (j < src.length && j - i < MAX_TAIL && src[j] !== close) j += src[j] === "\\" ? 2 : 1;
    if (src[j] !== close) return null;
    title = decode(src.slice(i + 1, j));
    i = j + 1;
    skip();
  }
  return src[i] === ")" ? { href: dest, title, end: i + 1 } : null;
}

function merge(items: Inline[]): Inline[] {
  const out: Inline[] = [];
  for (const item of items) {
    const last = out[out.length - 1];
    if (item.type === "text" && last?.type === "text") last.text += item.text;
    else if (!(item.type === "text" && !item.text)) out.push(item);
  }
  return out;
}

const demote = (item: Item): Inline => ("delim" in item ? { type: "text", text: item.ch.repeat(item.count) } : item);

/** CommonMark's "process emphasis": match closers to the nearest compatible opener. */
function emphasis(items: Item[]): Inline[] {
  const bottom = new Map<string, number>();
  for (let c = 0; c < items.length; c++) {
    const closer = items[c]!;
    if (!("delim" in closer) || !closer.close) continue;
    const key = `${closer.ch}${closer.open ? 1 : 0}${closer.orig % 3}`;
    let o = c - 1;
    const floor = bottom.get(key) ?? -1;
    for (; o > floor; o--) {
      const op = items[o]!;
      if (!("delim" in op) || op.ch !== closer.ch || !op.open || !op.count) continue;
      if (op.ch === "~" && (op.count < 2 || closer.count < 2)) continue;
      const odd = (op.open && op.close) || (closer.open && closer.close);
      if (odd && (op.orig + closer.orig) % 3 === 0 && !(op.orig % 3 === 0 && closer.orig % 3 === 0)) continue;
      break;
    }
    if (o <= floor) {
      bottom.set(key, c - 1);
      continue;
    }
    const opener = items[o] as Delim;
    const use = closer.ch === "~" || (opener.count >= 2 && closer.count >= 2) ? 2 : 1;
    const children = merge(items.slice(o + 1, c).map(demote));
    const node: Inline = { type: closer.ch === "~" ? "del" : use === 2 ? "strong" : "em", children };
    opener.count -= use;
    closer.count -= use;
    items.splice(o + 1, c - o - 1, node);
    c = o + 2;
    if (!opener.count) {
      items.splice(o, 1);
      c--;
    }
    if (!closer.count) items.splice(c, 1);
    c--;
  }
  return merge(items.map(demote));
}

export function parseInline(src: string, refs: Refs, depth = 0): Inline[] {
  const items: Item[] = [];
  let text = "";
  const flush = () => {
    if (text) items.push({ type: "text", text });
    text = "";
  };
  let scan: ReturnType<typeof index> | undefined;
  let i = 0;
  while (i < src.length) {
    const c = src[i]!;
    SPECIAL.lastIndex = i;
    const special = SPECIAL.exec(src);
    if (!special || special.index > i) {
      const stop = special ? special.index : src.length;
      text += src.slice(i, stop);
      i = stop;
      continue;
    }
    if (c === "\\") {
      const next = src[i + 1];
      if (next === "\n") {
        flush();
        items.push({ type: "br" });
        i += 2;
      } else if (next !== undefined && ESCAPABLE.test(next)) {
        text += next;
        i += 2;
      } else text += src[i++];
    } else if (c === "`") {
      scan ??= index(src);
      let n = 1;
      while (src[i + n] === "`") n++;
      const end = scan.closer(n, i + n);
      if (end < 0) {
        text += "`".repeat(n);
        i += n;
      } else {
        let code = src.slice(i + n, end).replace(/\n/g, " ");
        if (code.length > 2 && code[0] === " " && code.endsWith(" ") && /[^ ]/.test(code)) code = code.slice(1, -1);
        flush();
        items.push({ type: "code", text: code });
        i = end + n;
      }
    } else if (c === "\n") {
      const hard = / {2,}$/.test(text);
      text = text.replace(/ +$/, "");
      if (hard) {
        flush();
        items.push({ type: "br" });
      } else text += "\n";
      i++;
      while (src[i] === " ") i++;
    } else if (c === "<") {
      const m = /^<([A-Za-z][A-Za-z0-9+.-]{1,31}:[^\s<>]*)>/.exec(src.slice(i, i + 2100));
      const href = m ? safeHref(m[1]!) : null;
      if (m && href) {
        flush();
        items.push({ type: "link", href, children: [{ type: "text", text: m[1]! }] });
        i += m[0].length;
      } else text += src[i++];
    } else if (c === "&") {
      const m = /^&(#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7}|[A-Za-z][A-Za-z0-9]{1,31});/.exec(src.slice(i, i + 40));
      const decoded = m && entity(m[1]!);
      if (m && decoded) {
        text += decoded;
        i += m[0].length;
      } else text += src[i++];
    } else if (c === "!" && src[i + 1] === "[") {
      scan ??= index(src);
      const close = depth < MAX_DEPTH ? scan.pairs.get(i + 1) : undefined;
      const tail = close === undefined ? null : inlineTail(src, close + 1);
      if (close !== undefined && tail && safeHref(tail.href)) {
        text += src.slice(i + 2, close); // images never load: the alt text stands in
        i = tail.end;
      } else text += src[i++];
    } else if (c === "[") {
      scan ??= index(src);
      const close = depth < MAX_DEPTH ? scan.pairs.get(i) : undefined;
      let found: { href: string; title?: string; end: number } | null = null;
      if (close !== undefined) {
        found = inlineTail(src, close + 1);
        if (!found) {
          let label = src.slice(i + 1, close), end = close + 1;
          if (src[end] === "[") {
            const closing = src.indexOf("]", end + 1);
            if (closing > 0) {
              const named = src.slice(end + 1, closing);
              if (named.trim()) label = named;
              end = closing + 1;
            }
          }
          const ref = refs.get(normalizeLabel(label));
          if (ref) found = { ...ref, end };
        }
      }
      const href = found && safeHref(found.href);
      if (close !== undefined && found && href) {
        flush();
        items.push({ type: "link", href, ...(found.title ? { title: found.title } : {}), children: parseInline(src.slice(i + 1, close), refs, depth + 1) });
        i = found.end;
      } else text += src[i++];
    } else {
      // * _ ~ runs
      let n = 1;
      while (src[i + n] === c) n++;
      if (c === "~" && n < 2) {
        text += c;
        i++;
        continue;
      }
      const before = src[i - 1] ?? " ", after = src[i + n] ?? " ";
      const spaceBefore = /\s/.test(before), spaceAfter = /\s/.test(after);
      const punctBefore = PUNCT.test(before), punctAfter = PUNCT.test(after);
      const left = !spaceAfter && (!punctAfter || spaceBefore || punctBefore);
      const right = !spaceBefore && (!punctBefore || spaceAfter || punctAfter);
      flush();
      items.push({
        delim: true,
        ch: c as Delim["ch"],
        count: n,
        orig: n,
        open: c === "_" ? left && (!right || punctBefore) : left,
        close: c === "_" ? right && (!left || punctAfter) : right,
      });
      i += n;
    }
  }
  flush();
  return emphasis(items);
}
