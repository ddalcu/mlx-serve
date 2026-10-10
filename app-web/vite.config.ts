import { svelte } from "@sveltejs/vite-plugin-svelte";
import { viteSingleFile } from "vite-plugin-singlefile";
import { defineConfig } from "vitest/config";

// `npm run dev` / `npm run preview` forward the API to a running mlx-serve.
const target = process.env.MLX_SERVE_URL ?? "http://127.0.0.1:11234";
const proxy = Object.fromEntries(
  ["/v1", "/api", "/props", "/health", "/metrics", "/tokenize", "/detokenize"].map((p) => [p, target]),
);

export default defineConfig({
  plugins: [svelte(), viteSingleFile()],
  // Svelte ships separate server/browser runtimes; tests need the browser one for effects.
  resolve: process.env.VITEST ? { conditions: ["browser"] } : undefined,
  server: { proxy },
  build: { outDir: "../src/html", emptyOutDir: true, cssTarget: ["chrome120", "edge120", "firefox114", "safari16.4"], sourcemap: false, modulePreload: false },
  test: { environment: "happy-dom", setupFiles: ["./test/setup.ts"] },
});
