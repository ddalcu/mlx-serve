import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "svelte/compiler";
import ts from "typescript";

const MARKERS = new Set(["t", "th", "N"]);

/** Literal first arguments of t()/th()/N() calls: the only places English UI text becomes a translation key. */
export function typescriptKeys(source: string, fileName = "file.ts"): string[] {
  const keys: string[] = [];
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.ES2022, true);
  const visit = (node: ts.Node) => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && MARKERS.has(node.expression.text)) {
      const arg = node.arguments[0];
      if (arg && (ts.isStringLiteral(arg) || ts.isNoSubstitutionTemplateLiteral(arg))) keys.push(arg.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return keys;
}

type Node = { type?: string; [key: string]: unknown };

/** The same for a Svelte component: its script and every template expression. */
export function svelteKeys(source: string): string[] {
  const keys: string[] = [];
  const walk = (value: unknown) => {
    if (Array.isArray(value)) return value.forEach(walk);
    if (!value || typeof value !== "object") return;
    const node = value as Node;
    if (node.type === "CallExpression") {
      const callee = node.callee as Node, arg = (node.arguments as Node[])[0];
      if (callee?.type === "Identifier" && MARKERS.has(callee.name as string) && arg) {
        if (arg.type === "Literal" && typeof arg.value === "string") keys.push(arg.value);
        else if (arg.type === "TemplateLiteral" && (arg.expressions as unknown[]).length === 0)
          keys.push(((arg.quasis as Node[])[0]!.value as { cooked: string }).cooked);
      }
    }
    for (const child of Object.values(node)) walk(child);
  };
  walk(parse(source, { modern: true }));
  return keys;
}

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const path = join(dir, e.name);
    return e.isDirectory() ? sourceFiles(path) : [path];
  });
}

/** Every marked key in app source (the dictionary itself and tests excluded). */
export function appKeys(root: string): Map<string, string> {
  const keys = new Map<string, string>();
  for (const file of sourceFiles(root)) {
    if (file.endsWith("zh-hans.ts")) continue;
    const found = file.endsWith(".svelte") ? svelteKeys(readFileSync(file, "utf8")) : file.endsWith(".ts") ? typescriptKeys(readFileSync(file, "utf8"), file) : [];
    for (const key of found) if (!keys.has(key)) keys.set(key, file);
  }
  return keys;
}
