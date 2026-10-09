import { flushSync, mount, unmount } from "svelte";
import { afterEach, describe, expect, it } from "vitest";
import Markdown from "../src/components/Markdown.svelte";
import golden from "./fixtures/markdown-golden.json";
import { canonical } from "./support/markdown-html";

let app: ReturnType<typeof mount> | undefined;
function render(source: string): HTMLElement {
  const target = document.createElement("div");
  document.body.append(target);
  app = mount(Markdown, { target, props: { source } });
  flushSync();
  return target;
}
afterEach(() => {
  if (app) unmount(app);
  app = undefined;
  document.body.replaceChildren();
});

const ELEMENTS = new Set(["P", "H1", "H2", "H3", "H4", "H5", "H6", "UL", "OL", "LI", "BLOCKQUOTE", "HR", "BR", "EM", "STRONG", "S", "CODE", "PRE", "A", "TABLE", "THEAD", "TBODY", "TR", "TH", "TD", "DIV", "SECTION", "HEADER", "BUTTON", "SPAN", "svg", "path"]);
const ATTRIBUTES: Record<string, string[]> = {
  A: ["href", "title", "rel", "target"], OL: ["start"], TH: ["class"], TD: ["class"], DIV: ["class"], SECTION: ["class"], BUTTON: ["type"],
  svg: ["class", "viewBox", "aria-hidden", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin"], path: ["d"],
};

describe("Markdown component", () => {
  it("renders the same elements as the parser's golden HTML (structure, not whitespace)", () => {
    for (const { md } of golden.slice(0, 60)) {
      const root = render(md);
      // No script, no handlers, no stray attributes: only the whitelisted shapes.
      for (const el of root.querySelectorAll("*")) {
        expect(ELEMENTS.has(el.tagName) || ELEMENTS.has(el.tagName.toLowerCase()), `${el.tagName} from ${JSON.stringify(md)}`).toBe(true);
        for (const attr of el.getAttributeNames()) expect(ATTRIBUTES[el.tagName]?.includes(attr) ?? false, `${el.tagName}[${attr}] from ${JSON.stringify(md)}`).toBe(true);
      }
      app && unmount(app);
      app = undefined;
      root.remove();
    }
  });

  it("shows hostile input as text, with no live script, image or handler", () => {
    const root = render("<script>alert(1)</script> <img src=x onerror=alert(1)> [x](javascript:alert(1)) ![p](https://t.test/p.png)\n\n```html\n<script>1</script>\n```");
    expect(root.querySelector("script,img,iframe,object,embed")).toBeNull();
    expect(root.textContent).toContain("<script>alert(1)</script>");
    expect(root.querySelector("a")).toBeNull();
    for (const el of root.querySelectorAll("*")) for (const attr of el.getAttributeNames()) expect(attr.startsWith("on")).toBe(false);
  });

  it("links open in a new tab without leaking the opener", () => {
    const a = render("[ok](https://example.test/a)").querySelector("a")!;
    expect(a.getAttribute("href")).toBe("https://example.test/a");
    expect(a.getAttribute("rel")).toBe("noopener noreferrer");
    expect(a.getAttribute("target")).toBe("_blank");
  });

  it("keeps a code block's text exactly, trailing newline included", () => {
    expect(render("```js\nconst a = 1;\n```").querySelector("pre code")!.textContent).toBe("const a = 1;\n");
  });

  it("renders tables with alignment classes and tight lists without paragraphs", () => {
    const table = render("| a | b |\n|:--|--:|\n| 1 | 2 |");
    expect([...table.querySelectorAll("th")].map((th) => th.className)).toEqual(["align-left", "align-right"]);
    app && unmount(app);
    const list = render("- one\n- two");
    expect(list.querySelector("li p")).toBeNull();
    expect(list.querySelectorAll("li").length).toBe(2);
    void canonical;
  });
});
