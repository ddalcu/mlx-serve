import type { Block, Inline } from "../../src/lib/markdown";

const escape = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function inline(nodes: Inline[]): string {
  return nodes
    .map((n) => {
      switch (n.type) {
        case "text":
          return escape(n.text);
        case "code":
          return `<code>${escape(n.text)}</code>`;
        case "br":
          return "<br>";
        case "link":
          return `<a href="${escape(n.href)}"${n.title ? ` title="${escape(n.title)}"` : ""} rel="noopener noreferrer" target="_blank">${inline(n.children)}</a>`;
        default:
          return `<${{ em: "em", strong: "strong", del: "s" }[n.type]}>${inline(n.children)}</${{ em: "em", strong: "strong", del: "s" }[n.type]}>`;
      }
    })
    .join("");
}

const copy = '<button type="button" data-copy-code><span class="icon i-copy" aria-hidden="true"></span><span data-i18n="Copy">Copy</span></button>';

/** markdown-it-shaped HTML for an AST (blocks are only compared after `canonical`). */
export function toHtml(blocks: Block[], tight = false): string {
  return blocks
    .map((b) => {
      switch (b.type) {
        case "paragraph":
          return tight ? inline(b.children) : `<p>${inline(b.children)}</p>`;
        case "heading":
          return `<h${b.level}>${inline(b.children)}</h${b.level}>`;
        case "code":
          return `<section class="code-block"><header><span>${escape(b.lang)}</span>${copy}</header><pre><code>${escape(b.text)}</code></pre></section>`;
        case "quote":
          return `<blockquote>${toHtml(b.children)}</blockquote>`;
        case "hr":
          return "<hr>";
        case "list": {
          const tag = b.ordered ? "ol" : "ul", start = b.ordered && b.start !== 1 ? ` start="${b.start}"` : "";
          return `<${tag}${start}>${b.items.map((item) => `<li>${toHtml(item, b.tight)}</li>`).join("")}</${tag}>`;
        }
        case "table": {
          const cell = (tag: string, i: number, c: Inline[]) => `<${tag}${b.align[i] ? ` class="align-${b.align[i]}"` : ""}>${inline(c)}</${tag}>`;
          const head = `<thead><tr>${b.head.map((c, i) => cell("th", i, c)).join("")}</tr></thead>`;
          const body = b.rows.length ? `<tbody>${b.rows.map((r) => `<tr>${r.map((c, i) => cell("td", i, c)).join("")}</tr>`).join("")}</tbody>` : "";
          return `<div class="markdown-table"><table>${head}${body}</table></div>`;
        }
      }
    })
    .join("");
}

const BLOCK = "p|h[1-6]|ul|ol|li|blockquote|table|thead|tbody|tr|th|td|div|section|header|pre|hr";

/** Strip whitespace around block tags, keep `<pre>` untouched, normalize hrefs and indented code to the console's block. */
export function canonical(html: string): string {
  html = html.replace(/(?<!<\/header>)<pre><code>([\s\S]*?)<\/code><\/pre>\n?/g, (_, code) => `<section class="code-block"><header><span></span>${copy}</header><pre><code>${code}</code></pre></section>`);
  html = html.replace(/<br>\n/g, "<br>").replace(/href="([^"]*)"/g, (_, h) => {
    try {
      return `href="${escape(new URL(h.replace(/&amp;/g, "&")).href)}"`;
    } catch {
      return `href="${h}"`;
    }
  });
  return html
    .split(/(<pre>[\s\S]*?<\/pre>)/)
    .map((part, i) =>
      i % 2
        ? part
        : part.replace(new RegExp(`\\s+(?=</?(?:${BLOCK})\\b)`, "g"), "").replace(new RegExp(`(</?(?:${BLOCK})\\b[^>]*>)\\s+`, "g"), "$1"),
    )
    .join("")
    .trim();
}
