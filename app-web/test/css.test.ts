import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync(`${import.meta.dirname}/../src/app.css`, "utf8");

describe("stylesheet", () => {
  it("type scales with browser preferences: no pixel font sizes or fixed root", () => {
    expect(css).not.toMatch(/(?:font-size|--[\w-]*font[\w-]*size)\s*:[^;}]*\dpx/i);
    expect(css).not.toMatch(/(?:^|[;{])\s*font\s*:[^;}]*\dpx/m);
    expect(css).not.toMatch(/(?:^|})\s*(?:html|:root)\s*\{[^}]*(?:^|[;{])\s*font-size\s*:/m);
    expect(css).toMatch(/font-size:\s*[.\d]+rem/);
  });

  it("carries the brand wordmark once, as an unprefixed mask", () => {
    expect(css.match(/data:image\/png;base64,/g)?.length).toBe(1);
    expect(css).not.toContain("-webkit-mask");
  });
});
