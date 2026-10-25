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
| `guidance_scale`, `negative_prompt` | undistilled FLUX base, Qwen-Image and MOSS-Image only (MOSS-Image defaults to 3; its authors recommend the negative `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`) |
| `lora_paths`, `lora_scales` | absolute paths to `.safetensors` LoRAs (max 8), scales parallel |

MOSS-Image (`moss_image`) also takes a layout as the prompt: one line of JSON,
`{"summary":{"meta":["poster"],"canvas":"single","caption":"..."},"items":[{"label":"text","caption":"...","text":"EXPLORE","bbox":[y1,x1,y2,x2]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}`,
where each `bbox` is pixels from the CENTRE of the canvas (subtract W/2 from x
and H/2 from y), y first; `label` is `object`, `character` or `text`. Keep the
request's `size` the canvas the boxes were drawn on. It is text-to-image only.

Response: `{"created":0,"data":[{"b64_json":"<base64 PNG>"}]}`

`POST /v1/images/edits` is the OpenAI multipart form (`image`, `prompt`,
`model`), so the OpenAI SDK's `images.edit` works as is. From your own code,
prefer the JSON `mode:"edit"` body above.

Sprite and texture tips: ask for a plain or solid background and cut it out
(`node gen.mjs cutout`) unless the model supports `transparent`; `gen.mjs tile`
and `gen.mjs maps` make a texture seamless and derive its normal, roughness and
AO maps. Generate at the model's native
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
| `instrumental` | `true` = no vocals; do not also send `lyrics` (not YuE2) |
| `duration_seconds` | ACE-Step 10-600 (default 60); MiniMax Music 3 1-360; YuE2 5-360, an upper bound (the song ends itself) |
| `seed` | reproducible track |
| `bpm`, `keyscale`, `timesignature`, `vocal_language` | ACE-Step only |
| `task`, `src_audio`, `ref_audio` | ACE-Step only: `cover` / `complete` a base64 WAV, or match a reference's timbre |

MiniMax Music 3 needs `lyrics` unless `instrumental` is true, and rejects the
ACE-Step-only fields by name. YuE2 always needs `lyrics` (`[verse]`-style tags on
their own lines; for a calm, mostly instrumental feel keep them short and soft) and
refuses `instrumental` and the ACE-Step fields by name; its own fields are `cot`
(`full` default, `melody` or `off`: whether it writes a melody score first), `abc`
(your own or an edited score), `steps` 1-100 (default 32) and `cfg_scale` 0-20. Its
streaming `complete` event also returns the score as `abc`. Response: raw `audio/wav` bytes or SSE.
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

## Speech to text: `POST /v1/audio/transcriptions`

Qwen3-ASR, in OpenAI's multipart form (the OpenAI SDK's transcription call works):
what is said in a WAV (any rate, mono or stereo, any length: a long one is cut into pieces), in 30 languages.

| field | notes |
|---|---|
| `file` | required: the WAV |
| `model` | a model with the `transcription` capability |
| `language` | ISO code or name; omit it and the model detects it |
| `prompt` | names and terms the transcript should spell your way |
| `response_format` | `json` (default, `{"text"}`), `text`, `verbose_json`, `srt`, `vtt`, `diarized_json` |
| `timestamp_granularities[]` | `word` adds `words` to `verbose_json`; segments are always there |

Times come from the forced aligner, a model with the `alignment` capability (the
app downloads `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` with the ASR):
`verbose_json` segments end at sentences and pauses, each word has `start`/`end`
seconds, and `srt`/`vtt` are those segments as subtitles. It times Chinese,
Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean
and Japanese (a Chinese or Japanese word is one character); speech in another
language is one segment. Without the aligner, `verbose_json` is one segment for
the whole clip and `srt`/`vtt` answer 400.

`diarized_json` says who said what: `{"task","duration","text","segments":[{"type",
"id","start","end","text","speaker"}]}`, the speakers `"A"`, `"B"`, … in the order
they first speak (up to eight), a segment closing where the speaker changes. It
needs the diarizer, a model with the `diarization` capability (the app downloads
`mlx-community/Nemotron-3-Diarization-8bit` with the ASR); without it, a 400.

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
| `image` | base64 PNG/JPEG of ONE object; a cutout with real transparency works best (TRELLIS.2 and Pixal3D have no background remover: an opaque photo's whole frame becomes the subject) |
| `images` | Pixal3D only, instead of `image`: 2-8 views of the same object, the first its front. Each is used as framed (never cropped), so the framing must match the cameras. 400 beside `image`, with one view, or on another model |
| `cameras` | Pixal3D with `images`: one 4x4 camera-to-world matrix per view (Blender axes: Z up, the camera looks down its own -Z). Omitted = the reference's rig: 0/90/180/270 degrees around the object, level, so at most 4 views |
| `fov` | Pixal3D: horizontal field of view of the picture(s) in degrees (1-120). For `image` that is the cutout as cropped to its subject; the reference estimates it with MoGe-2, which is not shipped, so the default is 49.1. With `images` the default is the rig's 20 |
| `texture` | Hunyuan3D only: `true` = paint PBR textures too (slower; 400 if the pack has no paint weights). TRELLIS.2 and Pixal3D are always textured |
| `octree_resolution` | Hunyuan3D: 64-512, default 256; higher = finer mesh, slower |
| `resolution` | TRELLIS.2 and Pixal3D: 1024 (default, about 23 GB of free GPU memory) or 1536 (about 36 GB; a 400 names the shortfall); 512 only on packs that ship the 512 texture model (about 6 GB; `gen.mjs 3d --resolution 512`) |
| `decimation_target`, `texture_size` | TRELLIS.2 and Pixal3D: faces kept (10000-2000000, default 300000) and texture side (512/1024/2048/4096, default 2048) |
| `roughness_scale`, `clearcoat` | TRELLIS.2 and Pixal3D, the GLB's finish: `roughness_scale` (0.05-2) multiplies the model's roughness map, below 1 is glossier; `clearcoat` (0-1) adds a clear layer. Omitted = what the model predicted. Hunyuan3D refuses both by name |
| `steps`, `guidance_scale`, `seed`, `texture_steps` | Hunyuan3D defaults 30 / 5.0 / 42 / 30; TRELLIS.2 and Pixal3D `steps` omitted = the published 12 |

What each model needs: Hunyuan3D-2.1 one photo of the object; TRELLIS.2
(`microsoft/TRELLIS.2-4B`) one cutout, its alpha the subject; Pixal3D
(`TencentARC/Pixal3D`) one cutout the same way, or several posed views (the
pack's `pipeline_mv.json` models). TRELLIS.2 and Pixal3D return a textured PBR
GLB (base colour and metallic-roughness). Both are Y up, but TRELLIS.2's object
stands axis-aligned facing +Z while Pixal3D's keeps the picture's viewing angle
(a car from a three-quarter photo comes out turned about 30 degrees, nose near -Z):
`gen.mjs 3d ... --vehicle` (or `"vehicle": true` in a batch job) turns a car so its
length runs along Z with the nose at +Z (the nose is judged by the lower end, so check the first car).

One car cutout, 1024, 2048 texture, 150000 faces, M5 Ultra (MLX peak = GPU memory):

| model | GPU stages | whole request | MLX peak |
|---|---|---|---|
| TRELLIS.2 | 58 s | 91 s | 21.9 GiB |
| Pixal3D, one view | 49 s | 79 s | 21.9 GiB |
| Pixal3D at 1536 | 152 s | 186 s | 31.4 GiB |
| Pixal3D, four views (the reference's example head, not the car) | 62 s | 81 s | 22.0 GiB |

TRELLIS.2 at 1536 peaks at 34.1 GiB. On the same red sports-car photo Pixal3D kept
what the picture shows (rear wing, exhausts, diffuser, glass that reads as glass,
wheel spokes) with a mottled body texture and lumpy wheels; TRELLIS.2 gave a smoother,
cleaner body and wheels but turned glass into black patches with white speckles and
the rear window into a black blob. That red coupe was at 1024.

Which to use: for a car or other hard-surface object from a MOSS photo, use TRELLIS.2 at `"resolution": 1536`
(about 36 GB of free GPU memory; 1024 where less is free) and `"texture_size": 2048`. On a blue rally hatchback at those settings (same photo, same seed, M5 Ultra)
TRELLIS.2 kept the spoked rims, round headlights, bumper vents, rear lights, mudflaps and roof scoop and took
250-270 s; Pixal3D came out soft and mottled with flat wheels and took 360-420 s. Two cars is a small sample
(the red coupe above went Pixal3D's way on glass and rear detail at 1024), so treat TRELLIS.2 as the default.
Pixal3D is the choice for several posed views (`images`), the only mode TRELLIS.2 does not have.

A model's roughness map is nearly flat (a car's paint, trim and wheels all come out near 0.58), so the finish is
one setting for the whole object: a car wants `"roughness_scale": 0.4, "clearcoat": 1.0` (car paint), a satin
`0.7, 0`. A glossy finish reflects every bump of the surface, and how lumpy a car comes out depends on the seed:
try two or three. Tyres and glass share the paint's gloss; a viewer's environment map (three.js `RoomEnvironment`)
is what makes either look real.

For a game, cut the cost where the mesh is made. `decimation_target` (150000 for a hero car) decimates before
the unwrap, so no UV seam stops it, and `texture_size` 2048 costs about 22 MB of GPU memory per texture where 4096
costs 89 MB. Afterwards `gen.mjs glb --max-texture 2048 --quantize` costs nothing visible: textures shrink by area
(colour in linear light), and `--quantize` stores positions and UVs as 16-bit and normals as 8-bit integers
(`KHR_mesh_quantization`; three.js, Blender, Godot 4 and glTFast read it), halving the vertex data. Do not use
`--max-tris` to cut a car: its UV seams stay fixed, so a 2M-face car asked down to 200k stopped at 766k and the
hood came out crumpled. It is for small props.

A studio photo's soft ground shadow is part of the subject to the 3D model: it comes out as a grey slab under
the object. The `gen.mjs` cutout keeps it (it is darker than the white floor), and both models build it.

Response: `{"created":0,"format":"glb","data":"<base64 GLB>"}`. Loads directly
in three.js (`GLTFLoader`), Godot, Unity (glTFast) and Blender. Text-to-3D is
two calls: generate an image of the object first, then send it here.
