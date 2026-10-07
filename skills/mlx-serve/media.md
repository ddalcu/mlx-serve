# mlx-serve media endpoints

All take a JSON body with `"model": "<id>"` (pick it from `/v1/models` by
capability) and answer JSON, except speech/music/sound which answer raw WAV bytes
when not streaming. Binary inputs and outputs are base64 strings. Omit a field
to get the model's default. Unknown fields are ignored; a field the loaded model
cannot honor is a 400 that names it.

## Streaming (every media endpoint)

Add `"stream": true` for long jobs. The response is SSE, one JSON object per
`data:` line:

```
data: {"type":"progress","stage":"denoise","step":3,"total":8}
data: {"type":"complete", ...same fields as the non-stream JSON body...}
data: {"type":"error","message":"..."}
```

`total: 0` means indeterminate. Speech, music and sound `complete` events carry
`"format":"wav","data":"<base64 WAV>"`. Closing the connection cancels the job.

```js
// Browser or Node 18+: POST + read SSE progress
async function generate(base, path, body, onProgress) {
  const res = await fetch(base + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok) throw new Error((await res.json()).error?.message ?? res.statusText);
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) throw new Error("stream ended without a result");
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
      if (!line.startsWith("data:")) continue;
      const ev = JSON.parse(line.slice(5));
      if (ev.type === "progress") onProgress?.(ev.step, ev.total, ev.stage);
      else if (ev.type === "error") throw new Error(ev.message);
      else if (ev.type === "complete") return ev;
    }
  }
}
```

## Image: `POST /v1/images/generations`

| field | notes |
|---|---|
| `prompt` | required |
| `size` | `"WxH"`, default `"1024x1024"`; the backend snaps to what it supports (FLUX is 1024x1024) |
| `seed` | default 42 |
| `steps` | model default (distilled models: 4-8) |
| `image` | base64 PNG/JPEG source; with `mode` |
| `mode` | `"variation"` (default when `image` is set: re-noise the source, `strength` 0-1, default 0.6) or `"edit"` (instruction edit: "make the sword glow") |
| `ref_images` | edit mode: array of extra base64 references ("put the hat from image 2 on image 1") |
| `transparent` | `true` for an RGBA PNG with alpha; Qwen-Image models only |
| `guidance_scale`, `negative_prompt` | undistilled FLUX base and Qwen-Image only |
| `lora_paths`, `lora_scales` | absolute paths to `.safetensors` LoRAs (max 8), scales parallel |

Response: `{"created":0,"data":[{"b64_json":"<base64 PNG>"}]}`

`POST /v1/images/edits` is the OpenAI multipart form (`image`, `prompt`,
`model`), so the OpenAI SDK's `images.edit` works as is. From your own code,
prefer the JSON `mode:"edit"` body above.

Sprite and texture tips: ask for a plain or solid background and cut it out
yourself unless the model supports `transparent`. Generate at the model's native
size and downscale; for pixel art, downscale with nearest-neighbour.

## Speech (TTS): `POST /v1/audio/speech`

For models with `audio` but not `music` capability.

| field | notes |
|---|---|
| `input` | required text (`text` also accepted) |
| `voice` | Kokoro: a voice id like `af_heart` (default), or a blend `"af_bella,af_sky"` |
| `speed` | Kokoro, (0, 5], default 1 |
| `ref_audio` | Qwen3-TTS: base64 WAV of a few seconds of speech to clone that voice |

Response: raw `audio/wav` bytes (or SSE `complete` with base64 `data`). Kokoro is
fast enough for on-demand lines; still cache every line you can.

## Music: `POST /v1/audio/music-generations`

| field | notes |
|---|---|
| `prompt` | required: genre, mood, instruments, tempo feel |
| `lyrics` | section-tagged lines (`[verse]`, `[chorus]`, `[bridge]` on their own line) |
| `instrumental` | `true` = no vocals; do not also send `lyrics` |
| `duration_seconds` | ACE-Step 10-600 (default 60); MiniMax Music 3 1-360 |
| `seed` | reproducible track |
| `bpm`, `keyscale`, `timesignature`, `vocal_language` | ACE-Step only |
| `task`, `src_audio`, `ref_audio` | ACE-Step only: `cover` / `complete` a base64 WAV, or match a reference's timbre |

MiniMax Music 3 needs `lyrics` unless `instrumental` is true, and rejects the
ACE-Step-only fields by name. Response: raw `audio/wav` bytes or SSE.
For game loops, generate a track then crossfade the tail yourself; the model does
not guarantee a seamless loop point.

## Sound effects: `POST /v1/audio/sound-generations`

Stable Audio 3: effects and ambiences from a description (footsteps, rain, a
door, an engine, impacts, UI blips).

| field | notes |
|---|---|
| `prompt` | required: the source, the action and how it ends, the space ("heavy wooden door slamming shut in a stone hall, fast decay"); adding `TrackType: SFX` steers it toward a clean effect |
| `duration_seconds` | up to 120 (default 10, an ambience length): set it to the sound's real length, 0.3-2 for a click, hit or footstep, 2-5 for a door or explosion, 10+ for ambiences; the WAV is exactly this long |
| `steps` | 1-50 (default 8, what the model is distilled for) |
| `seed` | reproducible sound: same seed + prompt + length = same bytes |

Response: raw `audio/wav` bytes (44.1 kHz stereo) or SSE. About a second per
clip, so a game can generate variations up front (several seeds of one prompt)
and pick at random at play time.

## Video: `POST /v1/video/generations`

| field | notes |
|---|---|
| `prompt` | required; describe motion and camera |
| `width`, `height` | set them: LTX defaults to a 384x256 preview; multiples of 32 |
| `num_frames` | LTX default 9 (a preview); MiniMax-H3 default 56 |
| `seed`, `steps` | |
| `first_frame_image`, `last_frame_image` | LTX: base64 PNG/JPEG keyframes (image-to-video) |
| `audio` | LTX: base64 WAV to drive the video (audio-to-video) |

Response (and SSE `complete`):

```json
{"created":0,"frames":49,"height":512,"width":768,"fps":24,"format":"rgb8","data":"<base64>",
 "audio_sample_rate":48000,"audio_channels":2,"audio_format":"pcm_s16le","audio_data":"<base64>"}
```

`data` is RAW RGB bytes, `frames x height x width x 3`, not an MP4. Encode it
yourself, e.g. write `data` to `frames.rgb` and:

```sh
ffmpeg -f rawvideo -pix_fmt rgb24 -s 768x512 -r 24 -i frames.rgb \
  [-f s16le -ar 48000 -ac 2 -i audio.pcm] -c:v libx264 -pix_fmt yuv420p out.mp4
```

The `audio_*` fields appear only when the model made sound. Video renders take
minutes and hold a lot of memory: one at a time, unload the model after.

## 3D: `POST /v1/3d/generations`

| field | notes |
|---|---|
| `image` | required base64 PNG/JPEG of ONE object; transparent or plain background works best |
| `texture` | `true` = paint PBR textures too (slower; 400 if the pack has no paint weights) |
| `octree_resolution` | 64-512, default 256; higher = finer mesh, slower |
| `steps`, `guidance_scale`, `seed`, `texture_steps` | |

Response: `{"created":0,"format":"glb","data":"<base64 GLB>"}`. Loads directly
in three.js (`GLTFLoader`), Godot, Unity (glTFast) and Blender. Text-to-3D is
two calls: generate an image of the object first, then send it here.
