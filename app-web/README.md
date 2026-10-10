# app-web — MLX Serve Studio

The source of the web console `mlx-serve` serves at `GET /`: a Svelte 5 + TypeScript app that builds to ONE self-contained, minified `src/html/index.html` (script and style inlined by `vite-plugin-singlefile`). That file is committed, so `zig build` never needs Node.

```sh
cd app-web
npm ci
npm run dev        # http://localhost:5173, API proxied to a running mlx-serve
npm run build      # -> ../src/html/index.html, the file the server embeds
npm run preview    # serve the built file, same proxy
npm run check      # svelte-check (strict TypeScript)
npm test           # Vitest + happy-dom
```

`MLX_SERVE_URL` points dev and preview at another server (default `http://127.0.0.1:11234`). Node `^22.12 || ^24 || >=26`.

The page takes everything it needs from the server it is served by: version from `GET /api/version`, the API mount from `location.pathname`, models from `/v1/models`, metrics availability from `/metrics.json`. Nothing is injected by the Zig side.
