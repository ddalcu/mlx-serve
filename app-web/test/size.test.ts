import { build } from "vite";
import { describe, expect, it } from "vitest";

// The page ships uncompressed as one file: a ratchet, lowered whenever the page shrinks.
const BUDGET_BYTES = 540 * 1024;

/** The same build `npm run build` runs, kept in memory; vitest sets NODE_ENV=test, which would compile Svelte in development mode. */
async function built() {
  const env = process.env.NODE_ENV;
  process.env.NODE_ENV = "production";
  try {
    const result = await build({ configFile: `${import.meta.dirname}/../vite.config.ts`, mode: "production", logLevel: "silent", build: { write: false } });
    return (Array.isArray(result) ? result : [result as { output: { fileName: string; type: string; source?: string | Uint8Array }[] }]).flatMap((r) => ("output" in r ? r.output : []));
  } finally {
    process.env.NODE_ENV = env;
  }
}

describe("production build", () => {
  it("is one self-contained HTML file within the size budget", async () => {
    const outputs = await built();
    expect(outputs.map((o) => o.fileName)).toEqual(["index.html"]);
    const html = String((outputs[0] as { source: string | Uint8Array }).source);
    expect(Buffer.byteLength(html)).toBeLessThan(BUDGET_BYTES);
    // Nothing is fetched from outside the file, and nothing but the console's own script runs. (Links inside the
    // script are text for the user to follow, so only the page skeleton around it is checked for references.)
    const skeleton = html.replace(/(<(script|style)\b[^>]*>)[\s\S]*?<\/\2>/gi, "$1</$2>");
    expect(skeleton).not.toMatch(/<link\b[^>]*\brel=["']?(?:stylesheet|modulepreload|preload)/i);
    expect(skeleton).not.toMatch(/<script\b[^>]*\bsrc=/i);
    expect(skeleton).not.toMatch(/\b(?:src|href)=["']https?:/i);
    expect([...skeleton.matchAll(/<script\b[^>]*>/gi)].map((m) => m[0])).toEqual(["<script>", '<script type="module" crossorigin>']);
  }, 60_000);
});
