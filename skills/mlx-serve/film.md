# Films on mlx-serve

Make a whole film on this Mac: a song (YuE2) is the spine, pictures (Krea) and short clips (MiniMax-H3)
fill it, sound effects (Stable Audio) punch the cuts, and every cut lands on a sung word or a beat. Good for
a music video, a launch or release trailer, a story short.

Tools next to this file (Node 18+):
- `film.mjs`: project setup, song + timing, checks, stills, render. `node film.mjs help`.
- `gen.mjs`: pictures, clips, sound effects. `node gen.mjs help`.
- `film.tsx`: the renderer that `film.mjs init` copies into the project. Edit that copy only when a shot
  needs something no block below does.

Needs the server with a `music` model (YuE2) and an `image` model (Krea); `video` (MiniMax-H3) and `sound`
(Stable Audio 3) models for clips and effects; a `transcription` model to hear the takes. ffmpeg
(`brew install ffmpeg`) for clips and the final master. About 1 GB of disk for Remotion and its Chrome.

## The process

Do the phases in order: each one writes files the next reads. Show the user what each phase made (the
lyrics, the chosen take's line times, the contact sheets): they direct, you produce.

0. **Brief.** Ask what the film is for, the feeling, and where it plays (X: 2:20 at most, muted until
   tapped, frame 0 is the thumbnail). Write a one-paragraph concept and the arc: hook, setup, lift,
   payoff, call to action.
1. **Project.** `node film.mjs init film` (installs Remotion the first time: a few minutes).
2. **Song.** Write the lyrics (below), then
   `node film.mjs song film --lyrics lyrics.txt --style "<tags>" --takes 3`.
   It sings the takes, transcribes them, scores each (how well it sang the lyrics, loud choruses over
   quiet verses, an ending of its own), copies the best to `film/public/song.wav` and writes
   `film/song.json`: the tempo, the beats and, when the server gives word times (see "Timing comes
   from the server"), every sung line and word with its time. It prints the
   `song` range to put in the timeline. You cannot listen: tell the user the takes are in
   `film/public/takes/` and switch on their pick with `--file film/public/takes/<take>.wav --lyrics
   lyrics.txt`. A song made in the app's Music tab works the same way.
3. **Shot list.** `node film.mjs words film` prints each sung line with its time (`--words` adds every
   word), or says there are none, and then every shot cuts on beats. One shot per line, two per line
   in a chorus; instrumental stretches cut on beats
   (`node film.mjs beats film --from 0 --to 14 --div 2` prints half-beats). Write the list: time range,
   what we see, which file.
4. **Assets.** One `assets.json` for every picture, clip and effect (below), run from the project:
   `cd film && node <this folder>/gen.mjs batch assets.json`. Pictures first: show them to the user,
   then make the clips from the approved ones.
5. **Timeline.** Write `film/timeline.json` (below).
6. **Look.** `node film.mjs check film`, then `node film.mjs stills film <one song second inside every
   shot...>` and READ the contact sheets it prints (`film/stills/sheet-*.jpg`). Fix what is off: text
   over a face, a black frame, a clip that starts late, a title that wraps. Repeat until clean.
7. **Render.** `node film.mjs render film` writes `film/out/film.mp4`: H.264 + AAC, -14 LUFS, peaks
   under -1.5 dBFS, cut at 140 s. Look at frame 0 once more (`stills film <song.start>`).

## Lyrics and style (YuE2)

- Section tags on their own lines, capitalized: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`,
  `[Bridge]`, `[Outro]`. One sung line per text line, 6 to 10 syllables. About 26 to 30 sung lines make a
  2:20 song at a ballad tempo. Write the chorus out in full every time it repeats.
- The lyrics tell the film's story in the order the pictures will show it. The chorus's first line is
  the hook: it becomes the title card.
- `--style` is comma-separated tags: language, genre, mood, instruments, lead vocal. Example:
  `English, cinematic pop ballad, emotional, uplifting, hopeful, piano intro, strings, big drums,
  anthemic chorus, warm female lead vocal`.
- `--max` (240 s) is an upper bound, not a target: a take that runs into it was cut off mid-song and
  scores low.

## Assets (gen.mjs)

Put `assets.json` in the project with `"out": "public"`; paths inside jobs are relative to the project.
Jobs run one at a time in order, unchanged jobs are skipped, and the models unload at the end.

```json
{"out": "public", "jobs": [
  {"type": "image", "id": "room", "prompt": "Wide shot of ... , cinematic film still, anamorphic lens, shallow depth of field, soft film grain, rich shadows, highly detailed", "size": "1920x1088", "seed": 11},
  {"type": "image", "id": "room_dawn", "prompt": "The same small bedroom at dawn, ...", "source": "public/room.png", "source_strength": 0.6, "size": "1920x1088", "seed": 21},
  {"type": "image", "id": "wall1", "prompt": "...", "size": "1024x1024", "seed": 3},
  {"type": "sound", "id": "boom", "prompt": "deep cinematic boom impact with a long tail, TrackType: SFX", "duration_seconds": 3, "seed": 1},
  {"type": "video", "id": "room_clip", "prompt": "<the labelled H3 prompt below>", "first_frame": "public/room.png", "turbo": true, "seed": 42}
]}
```

Pictures (Krea, `image`):
- `1920x1088` for anything full frame (and for a clip's first frame), `1024x1024` for wall tiles.
- One style suffix on every story shot so they match (the one in the example).
- Continuity comes from variations: make the hero shot once, then the same place at another time with
  `source` + `source_strength` 0.55 to 0.65. A variation will not switch a lit screen off; darken in the
  timeline (`"filter": "brightness(0.4)"`).
- Krea paints what you name: "a computer" gets a 1990s CRT. Name modern hardware exactly ("a modern
  Apple Mac Studio beside a thin bezel-less Studio Display"). Ask for "no logo, no text" on screens and
  signs; generated lettering is gibberish.

Clips (MiniMax-H3, `video`), the slow part (minutes each), so make them last:
- Always from a first frame: the 16:9 still of that shot. The clip opens exactly on it, so a shot can
  hold the still and cut into the clip without a jump. gen.mjs defaults to 1312x736 and 124 frames
  (5.2 s at 24 fps); `"turbo": true` (4 steps) when the pack has its Turbo LoRA.
- Free prose drifts off the first frame: write H3's labelled format.

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] Cinematic, wide shot, slow push-in, opening on the exact composition, subject and lighting of <Picture 1> and holding them throughout. <What moves and how the light changes, in 4 to 6 sentences.> The camera keeps moving without cutting.

overall_soundscape:
<physical sounds, near to far>

non_diegetic_music:
N/A
```

- A clip brings its own sound (quiet, except storms): play it at volume 0.2 to 0.5 under the song.

Sound effects (Stable Audio 3, `sound`): one to three seconds each, prompt ending in `TrackType: SFX`:
a boom (title slams), hits (cards), whooshes (into a new section), a riser (before the chorus).

Screens: a capture of an app or web page is a PNG or MP4 in `public/`, shown with a `screen` block. Crop
out numbers and private data nobody should read.

## timeline.json

```json
{
  "fps": 30, "width": 1920, "height": 1080,
  "song": {"file": "song.wav", "start": 5.08, "end": 139.9, "fadeOut": 3},
  "colors": {"ink": "#f6f2ff", "accent": "#a78bfa", "warm": "#ffcf9a", "glow": "124,58,237"},
  "blocks": [
    {"type": "brand", "from": 5.08, "to": 5.6, "icon": "brand/icon.png", "title": "My App", "version": "2.0", "punch": false},
    {"type": "still", "src": "room.png", "from": 5.6, "to": 14.08, "zoom": [1.0, 1.15], "origin": "70% 40%"},
    {"type": "lyrics", "from": 14.0, "to": 41.0},
    {"type": "sfx", "src": "boom.wav", "at": 42.16, "volume": 0.6}
  ]
}
```

- The video runs from `song.start` to `song.end` on the song's clock: every `from`, `to` and `at` is a
  song second, so the times from `words` and `beats` go in unchanged. `song.volume` scales the song (1).
- Blocks draw in list order, later on top. Overlap them to layer (a `still` under a `caption`, a `clip`
  under a `letterbox`). Paths are relative to `public/`. `"grain": false` turns the film grain off.

| type | fields besides `from`, `to` | for |
|---|---|---|
| `still` | `src`, `zoom` [start, end] (1 to 1.08), `pan` [x0, y0, x1, y1] in %, `origin` "x% y%", `fade` seconds or [in, out], `filter` (CSS) | a picture with a slow camera move |
| `clip` | `src`, `trim` (seconds skipped at its start), `volume` (0 = muted), `rate`, `zoom`, `origin`, `fade`, `filter` | a video clip |
| `screen` | `src` (image or video), `zoom` [start, end], `focus` [x, y] or [[x0, y0], [x1, y1]] (fractions of the window: the point that stays in place while zooming), `url` (adds a browser bar), `width` (1520), `aspect` (9/16), `trim`, `glow` | an app capture in a window |
| `wall` | `items` [src...], `cols` (4), `stagger` (0.05 s) | pictures popping into a grid |
| `grid` | `items` (up to 9 images or clips), `title` | many things at once |
| `title` | `text`, `sub`, `size` (150), `upper` (true), `position` (top, center, bottom) | a punch-in title |
| `caption` | `text`, `position` (bottom), `size`, `color`, `fade` | a quiet serif line |
| `words` | `words` [[text, song second], ...], `size` | words landing on their sung times |
| `lyrics` | `position`, `size`, `scrim` (true) | song.json's lines as captions, each held until the next |
| `card` | `name`, `role`, `sub`, `n` (shows CREW · 01) or `label`, `bg` (a src, or {"type": "still"/"clip"/"screen", ...}) | introduce a character, a model, a feature |
| `terminal` | `command`, `lines`, `title`, `cwd`, `x`, `y`, `w`, `h`, `perLine` (frames per line, 2) | replay real terminal output |
| `credits` | `sections` [{"heading", "rows": [[role, name]]} or {"heading", "names": [...]}], `bg`, `height` | a credits roll |
| `brand` | `icon`, `wordmark` (a PNG used as a mask; `wordmarkAspect` height/width) or `title`, `version`, `punch` (true), `fadeOut` (true) | frame 0, or a logo slam |
| `endcard` | `icon`, `wordmark` or `title`, `version`, `line`, `url`, `code` [command lines] | the call to action |
| `black`, `backdrop` | `glow` (backdrop) | plain grounds |
| `letterbox` | | cinema bars |
| `sfx` | `src`, `at`, `volume` (0.5) | a sound effect |
| `flash` | `at`, `duration` (0.3), `color`, `peak` (0.8) | a white hit on a cut |

## Craft rules

- **The song plays unbroken.** Never splice it: a loud chorus cut against a quiet intro sounds broken.
  Choose `song.start` so the hook comes early, and lay a cold-open montage over the song's own intro.
- **Frame 0 is the thumbnail.** Start on a `brand` block (`punch: false`, half a second to a second) or
  the strongest picture under a title.
- **The first five seconds decide** whether anyone keeps watching: motion and the best pictures, fast
  cuts on half-beats, then the title. No slow fade up from black.
- **Cut on sung words.** A shot starts where a line starts (or on a strong word inside it); montages
  cut on beats. Three frames late already feels late.
- **One idea per card.** Titles of six words or fewer, a one-sentence subtitle, every text on screen
  for at least 1.5 s. Lyrics sit at the bottom, never over a face.
- **Escalate.** Verse: story stills with slow moves. Pre-chorus: faster, more motion. Chorus: clips,
  cards, the title slam with a `flash` and a boom. Bridge: a breath. Last chorus: everything (`grid`,
  `wall`). Outro: credits and the end card.
- **Sound design.** A boom under each title slam, hits on cards, whooshes into new sections, at 0.3 to
  0.6. The song carries the rest.
- **True claims only.** Everything on screen is something this Mac made or the product really does. No
  benchmark numbers unless the user asks for them.
- **Look before you render.** A contact sheet takes seconds, a wrong render minutes.

## Timing comes from the server

`film.mjs song` transcribes with `POST /v1/audio/transcriptions`, OpenAI's multipart shape: `file`
(WAV), `model` (the first model listing the `transcription` capability, when there is one),
`response_format=verbose_json`, `timestamp_granularities[]=word` and `=segment`. It reads
`words: [{word, start, end}]` in seconds, else `segments: [{start, end, text}]` when there are several,
each short enough to place a word. mlx-serve times words with its forced aligner (an `alignment`
model; the app downloads it with the ASR). Without one, or for a language it does not time, the
transcript is one segment for the whole song: it still picks the take that sang the lyrics, but
song.json gets no line times; on a 404 the takes are scored by sound alone. Either way the `lyrics`
blocks stay empty and cuts go on `beats`.

## Gotchas

- The first `stills` or `render` downloads Remotion's headless Chrome (about 170 MB).
- A pack on disk but missing from `/v1/models`: register it, `POST /v1/load-model {"model":
  "/absolute/path/to/pack"}`. Ids follow how the server found a pack (`org/name` or the folder name), so
  pick by capability (gen.mjs does) and never hardcode one.
- The server runs one generation at a time: while a clip renders, chat replies wait. Make clips when
  nobody is chatting with the server.
- Remotion is free for individuals and companies of up to three people; larger companies need its
  license (remotion.dev/license).
