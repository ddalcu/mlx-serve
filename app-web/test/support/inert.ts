import { expect } from "vitest";

/** Text that would do harm if any surface turned it into markup, a handler or a link. */
export const HOSTILE = [
  "<script>window.__pwned = 1</script>",
  '<img src=x onerror="window.__pwned = 2">',
  '"><svg onload=window.__pwned=3>',
  "'><iframe srcdoc='<script>1</script>'></iframe>",
  "[x](javascript:window.__pwned=4)",
  "![x](javascript:window.__pwned=5)",
  '<a href="javascript:window.__pwned=6">click</a>',
  "[x](data:text/html,<script>1</script>)",
  "[ x ]( JaVaScRiPt:1 )",
  "<style>body{display:none}</style>",
  "<form action=//evil.test><input name=x></form>",
  "&lt;script&gt;1&lt;/script&gt; &#60;img src=x onerror=1&#62;",
  "{@html '<b>x</b>'} ${1 + 1} {{7*7}} `${1}`",
  "‮<b>rtl</b>‬",
];
export const HOSTILE_LINE = HOSTILE.join(" ");

const FORBIDDEN = new Set(["script", "iframe", "frame", "frameset", "object", "embed", "applet", "base", "meta", "link", "style", "template", "noscript"]);
const URL_ATTRIBUTES = new Set(["href", "src", "action", "formaction", "xlink:href", "srcdoc", "data", "poster", "ping"]);
// What the app itself writes: its own http(s) links, in-page anchors, and media it built from bytes.
const SAFE_URL = /^(https?:\/\/|#|blob:|data:image\/(png|jpeg|webp|gif);base64,)/;

/** No script-bearing element, no event handler, no URL attribute that is not one the app itself writes. */
export function assertInert(root: ParentNode, where = "page") {
  for (const el of root.querySelectorAll("*")) {
    const tag = el.localName;
    expect(FORBIDDEN.has(tag), `<${tag}> in ${where}`).toBe(false);
    for (const { name, value } of el.attributes) {
      expect(name.toLowerCase().startsWith("on"), `${name} on <${tag}> in ${where}`).toBe(false);
      if (URL_ATTRIBUTES.has(name.toLowerCase())) expect(SAFE_URL.test(value.trim()), `${name}="${value}" on <${tag}> in ${where}`).toBe(true);
    }
  }
  expect((window as unknown as { __pwned?: number }).__pwned, `script ran in ${where}`).toBeUndefined();
}
