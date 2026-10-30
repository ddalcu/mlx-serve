# Write a media plugin

A media plugin adds an image, audio, video, 3D, decision or diarization backend. The host runs the job queue, progress,
previews and the HTTP endpoints; you supply the claim, the bills and the job itself. Contract: `src/abi/media.zig`.
Overview: [plugins.md](plugins.md). The model-plugin walkthrough ([plugin-model.md](plugin-model.md)) covers the kit,
install and test flow; this page covers what differs.

```
mlx-serve plugin new tone --media --kit <kit dir> --dir ~/work
```

The scaffold is a sine-tone generator: one `gain` weight, a WAV out. Replace the tone with your model.

## The table

`abi.MediaTable` in `plugin.zig`:

| Field | What it does |
|---|---|
| `claims` | `kit.sdk.mediaClaimsByType(&model_types)`, or by marker for a configless pack (`model_index.json`, `pipeline.json`) |
| `required_file` | A pack with this file missing is invisible until it lands (incomplete downloads) |
| `modality` | `image`, `audio`, `video`, `mesh`, `decision` or `diarization`; decides the endpoint and the app flow |
| `caps` | What this pack supports, read from the pack with no MLX (img2img, edit, CFG, LoRA, audio `routes`) |
| `bill_load`, `bill_request` | Bytes the load and each request will allocate, as upper bounds you measured |
| `load` / `unload` | Build and free the engine; return a `Verdict` that refuses by name |
| `step` | Run one job. Read `req.body` (the request JSON), report progress, honour the boundary |
| `encode` | Hand back the finished bytes (WAV, PNG, MP4, GLB...) |

Audio packs say which route they serve in `caps.routes` (`speech`, `music`, `sound`), which map to
`/v1/audio/speech`, `/v1/audio/music-generations` and `/v1/audio/sound-generations`.

## The job loop

```zig
progress.report(progress.ctx, .of("tone"), i + 1, config.chunks);   // stage, step, total
if (progress.boundary(progress.ctx)) return error.ClientLeft;       // chat runs here; true = the client left
```

- Call `boundary` between steps. That is where chat requests get the GPU and where a cancel lands.
- Run MLX on the stream the host handed you in `LoadCtx`. The inference thread is the only MLX caller, so never
  spin up your own GPU thread.
- Load weights on a CPU stream (`mlx_default_cpu_stream_new()`), then use the host's stream for compute. MLX has no
  GPU kernel for file loads, so `mlx_load_safetensors` on `LoadCtx.stream` fails with `Load::eval_gpu Not implemented`.
- Parse the body defensively and refuse a bad one by name (`req.refuseFmt`). Unhonored fields are a named 400, never
  silently ignored.
- Bill honestly. Admission uses `bill_request` to decide whether a job fits next to a loaded chat model.

## Check and try it

```
cd ~/work/tone && zig build test && zig build
mlx-serve plugin check ./tone
mlx-serve plugin fixture ./tone ~/work/tone-pack      # writes the plugin's tiny pack so you can serve it
mlx-serve plugin install tone/zig-out/lib/libtone.dylib
mlx-serve --serve --port 18082 --model ~/work/tone-pack
curl -s localhost:18082/v1/audio/sound-generations -H 'Content-Type: application/json' \
  -d '{"seconds":1,"frequency":440}' -o tone.wav
```

Enable plugins in `~/.mlx-serve/plugins.json`, same as for model plugins (see the gotchas in
[plugin-model.md](plugin-model.md)).

In-tree backends with full SDK media blocks, previews, LoRA and ANE offload:
[recipes/add-a-media-backend.md](recipes/add-a-media-backend.md).

## Read these in the kit

- `examples/_example_media/plugin.zig` and its `config.zig`: the scaffold `plugin new` copies.
- `src/abi/media.zig`: `MediaTable`, `Request`, `Progress`, `MediaCaps`, `Machine`. Field comments are the reference.
- `src/sdk/root.zig`, `include/mlx_serve_plugin.h`: the SDK exports and the C ABI.
