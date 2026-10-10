import { describe, expect, it } from "vitest";
import { parseMarkdown } from "../src/lib/markdown";
import golden from "./fixtures/markdown-golden.json";
import { canonical, toHtml } from "./support/markdown-html";

describe("markdown parity with the previous renderer", () => {
  for (const [index, { md, html }] of golden.entries())
    it(`#${index} ${JSON.stringify(md.slice(0, 50))}`, () => {
      expect(canonical(toHtml(parseMarkdown(md)))).toBe(canonical(html));
    });
});

const walk = (value: unknown, visit: (node: { type: string; [k: string]: unknown }) => void) => {
  if (Array.isArray(value)) return value.forEach((v) => walk(v, visit));
  if (value && typeof value === "object") {
    const node = value as { type?: string };
    if (typeof node.type === "string") visit(node as { type: string });
    Object.values(node).forEach((v) => walk(v, visit));
  }
};

describe("markdown safety", () => {
  const TYPES = new Set(["text", "code", "em", "strong", "del", "link", "br", "paragraph", "heading", "quote", "list", "hr", "table"]);
  const hostile = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "[x](javascript:alert(1))",
    "[x](JaVaScRiPt:alert(1))",
    "[x](  javascript:alert(1))",
    "[x](data:text/html,<script>alert(1)</script>)",
    "[x](vbscript:msgbox(1))",
    "[x](https://ok.test\njavascript:alert(1))",
    "[x](<javascript:alert(1)>)",
    "![x](javascript:alert(1))",
    "![x](https://tracker.test/pixel.png)",
    "<javascript:alert(1)>",
    "[a]: javascript:alert(1)\n\n[a]",
    "[a][b]\n\n[b]: data:text/html,hi",
    "[x](https://ok.test \"title\" onclick=\"alert(1)\")",
    '[x](https://ok.test "a\\" onmouseover=\\"alert(1)")',
    "&lt;script&gt;alert(1)&lt;/script&gt;",
    "&#60;script&#62;",
    "`<script>`",
    "```html\n<script>alert(1)</script>\n```",
    "| <script> | b |\n|---|---|\n| <img src=x onerror=y> | c |",
  ];

  it("produces only whitelisted node types and absolute http(s) links", () => {
    for (const md of hostile) {
      walk(parseMarkdown(md), (node) => {
        expect(TYPES.has(node.type), `${node.type} from ${JSON.stringify(md)}`).toBe(true);
        if (node.type === "link") expect(String(node.href), md).toMatch(/^https?:\/\//);
      });
    }
  });

  it("keeps raw HTML as text and never links a non-http URL", () => {
    const html = toHtml(parseMarkdown("<script>alert(1)</script>\n\n```html\n<script>x</script>\n```"));
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toMatch(/<script[ >]/i);
    for (const url of ["javascript:alert(1)", "data:text/html,hi", "mailto:a@b.test", "/relative", "#anchor", "//host/path", "ftp://host/file"])
      expect(toHtml(parseMarkdown(`[click](${url})`)), url).not.toMatch(/href=/);
    for (const url of ["https://example.test/a", "http://example.test/b"]) expect(toHtml(parseMarkdown(`[click](${url})`))).toMatch(/href="https?:/);
    expect(toHtml(parseMarkdown("![tracking](https://example.test/pixel)"))).not.toMatch(/<img/);
  });
});

describe("markdown while streaming", () => {
  it("parses every prefix of every sample without throwing", () => {
    for (const { md } of golden) {
      for (let end = 0; end <= md.length; end++) parseMarkdown(md.slice(0, end));
      expect(parseMarkdown(md)).toEqual(parseMarkdown(md));
    }
  });
});

describe("markdown beyond the previous renderer", () => {
  const html = (md: string) => canonical(toHtml(parseMarkdown(md)));

  it("closes a code span after a trailing backslash (Windows paths)", () => {
    expect(html("`C:\\Users\\` and more")).toBe("<p><code>C:\\Users\\</code> and more</p>");
  });

  it("resolves reference links and hides their definitions", () => {
    expect(html("See [docs] and [the guide][g].\n\n[docs]: https://example.test/docs\n[g]: https://example.test/guide \"Guide\"")).toBe(
      '<p>See <a href="https://example.test/docs" rel="noopener noreferrer" target="_blank">docs</a> and <a href="https://example.test/guide" title="Guide" rel="noopener noreferrer" target="_blank">the guide</a>.</p>',
    );
    expect(html("[undefined] stays [text]")).toBe("<p>[undefined] stays [text]</p>");
  });
});

describe("markdown on degenerate input", () => {
  const slow = [
    "[".repeat(50_000),
    "[a](".repeat(10_000),
    "*".repeat(50_000),
    "* a ".repeat(20_000),
    "`".repeat(50_000),
    "`a ".repeat(20_000),
    "> ".repeat(5_000) + "x",
    "- ".repeat(5_000) + "x",
    "  ".repeat(2_000) + "- x\n".repeat(1_000),
    "| a ".repeat(5_000) + "|\n" + "|---".repeat(5_000) + "|\n" + "| b ".repeat(5_000),
    "![".repeat(20_000),
    "_a ".repeat(20_000),
    "word ".repeat(100_000),
  ];

  it("finishes quickly and never throws", () => {
    for (const md of slow) {
      const start = performance.now();
      parseMarkdown(md);
      expect(performance.now() - start, md.slice(0, 20)).toBeLessThan(1500);
    }
  });
});
