# Changelog

## v26.8.10 — Neural Engine prefill offload, batched decode, DFlash 2

### Highlights

- **Faster replies, zero setup.** The server now learns the fastest speculation settings for *your* Mac while it runs and remembers them. Up to +24% reply speed over 26.8.9.
- **Long prompts, much faster.** Turn on Neural Engine prefill (`--ane-prefill` / Settings) and a 16k-token prompt loads up to 35% faster. Works on M1 through M4.
- **DFlash 2 works**, including Muse-Glimmer 30B with its bundled draft head (86 tok/s on an M4 Max).
- **LFM 2.5 gets DSpark speed.** Liquid's draft heads run on the 1.2B, 2.6B and the new 8B-A1B MoE — the 2.6B replies 1.4x faster.
- **Several chats at once, no slowdown.** Concurrent requests on Qwen 3.5/3.6/3.8 decode together: 2.8x total throughput at 4 streams.
- **Tables render as tables** in chat. Thanks @slava-kudzinau (#216).
- **Cover a song, or build a track around a vocal.** ACE-Step gained Cover and Vocal to BGM modes plus reference audio for style, all in the Music tab.
- **Videos that start and end where you say.** LTX takes a last frame next to the first one.
- **Rewrite with LLM** in the Music tab: the chat model reshapes your style prompt or lyrics to match the loaded model's format, and you edit before applying.
- **Image content filter removed.** The Safe mode toggle, the `--no-safety` flag and the classifier download are gone. Both are still accepted (`"safety"` in a request, `--no-safety` on the command line) and ignored.

Same models, same Macs, 26.8.9 vs 26.8.10:

| Mac | Model | 26.8.9 | 26.8.10 | |
|---|---|---|---|---|
| M1 Pro 32 GB | Qwen3.8 27B | 9.3 tok/s | 11.5 tok/s | **+24%** |
| M1 Pro 32 GB | Qwen3.5 9B, 8k chat | 26.3 tok/s | 30.8 tok/s | **+17%** |
| M4 Max | Qwen3.8 27B | 66 tok/s | 73 tok/s | **+10%** |
| M4 16 GB | Qwen3.5 9B, 8k chat | 30.5 tok/s | 32.5 tok/s | **+7%** |

Neural Engine prefill, 16k-token prompt (new, off by default):

| Mac | Model | GPU only | + Neural Engine | |
|---|---|---|---|---|
| M1 Pro 32 GB | Qwen3.8 27B | 41 tok/s | 56 tok/s | **+35%** |
| M4 16 GB | Qwen3.5 4B | 368 tok/s | 487 tok/s | **+32%** |
| M4 Max | Qwen3.8 27B | 247 tok/s | 293 tok/s | **+19%** |
| M3 Ultra 512 GB | Qwen3.8 27B | 414 tok/s | 498 tok/s (both Neural Engines) | **+20%** |

### Neural Engine prefill

- **Your Mac's Neural Engine now helps read long prompts.** Turn on `--ane-prefill` (Settings ▸ "Neural Engine prefill boost") and the GPU and Neural Engine split the work: a 16k-token prompt loads 19-35% faster depending on the Mac (table above). Reply speed is unchanged.
- Works with Qwen 3.5/3.6/3.8 models on M1 through M4 Macs. The first load of a model compiles the Neural Engine programs once (1-2 minutes), then it is instant.
- It needs extra memory next to the model (~11 GB for a 27B, ~1 GB for a small model). On a 16 GB Mac use a small model (a 4B fits and gets +32%); if it does not fit, the log says so and the model runs on the GPU alone.
- Off on M5-class Macs, where the GPU is already faster on its own.
- **M3 Ultra uses both of its Neural Engines** by default: +7% on a 16k prompt over one (498 vs 465 tok/s, Qwen3.8 27B). `MLX_SERVE_ANE_DUAL=0` turns it off.

### Concurrency

- **Concurrent requests on a Qwen 3.5/3.6/3.8 trunk now decode as one batch.** They used to run strictly serially, each stream re-reading the whole model. Measured 2.76x aggregate at 4 streams. Two separate gates were cancelling it: the server clamped concurrency to 1 on these architectures, and a prompt that merely armed predictive decoding lost batching permanently even after that speculation turned itself back off (9.6 → 12.4 tok/s per stream on 4 concurrent 5.5k-token prompts on a Mac Mini).
- A batch mixing one very long conversation with short ones no longer builds a padded attention tensor sized by the longest stream — the long ones fall out to serial for that step and everything still advances. Left unchecked this was several GB per step that nothing accounted for, ending in an unrecoverable GPU out-of-memory.

### Speculative decoding

- **DFlash 2 draft heads load and run** (the nested `dflash_config` sidecars), including the trained path selector and its convolutions. On an M4 the built-in MTP head is still the faster option; the selector's win needs the wider draft blocks only larger machines serve.
- Draft-head history now survives the on-disk prefix cache, not just the in-memory one. Resuming a conversation after a restart used to draft blind.
- **The speculation width tunes itself.** The server measures how much each draft width actually costs and yields on your Mac, per model and context size, and keeps that in `~/.mlx-serve/round-cost/` so the next boot starts tuned. The first request on a new model may be a few percent slower while it learns; after that it is free. Reply speed +7% to +24% over 26.8.9 depending on the Mac (table above). `MLX_SERVE_MTP_COST_TABLE=0` restores the old behaviour.
- Experimental, off by default: DFlash/DSpark drafters can pick their block per round the same way (`MLX_SERVE_DFLASH_CHOOSER=1`) — +53% on LFM2.5-2.6B on a base M4 in our test, not yet stable on MoE models.
- +3% decode from removing a redundant copy in the draft gate, plus a re-scored draft shortlist: 62 → 64 tok/s on an M4 Max, and +5% on novel content.
- A model that ships a draft head but fails to load it now says so instead of quietly falling back to the slower predictive path.

### Models

- **Huge images no longer crash Qwen vision.** A 5100x3300 photo on Qwen3.8 27B used to kill the server (the pack allows 16.7 Mpx, which is 65k patches of attention the GPU cannot hold). Images are now capped at 1536x1536 worth of pixels before the vision tower; screenshots are untouched.

- **Ling 3.0 flash-line checkpoints** that ship a direct query projection (`"q_lora_rank": null`) are no longer refused at load, and Ling 3.0 tiny loads under both converter layouts. Thanks @Fe2-O3 (#232). The published flash quants need a further layout change and still do not load.
- **Alis (avlp12) Qwen packs are served**: their quantized draft-head projection binds, the vision tower's prefix and convolution layout are probed instead of assumed, and a false "broken norm" repair that was halving draft acceptance is fixed (33% → 70%).
- Embedded DeepSeek and llama.cpp engines updated to their latest upstream versions.

### Releases

- The release workflow gained a **Pre-release** checkbox: it tags `v<version>-pre-release.1`, `.2`, `.3` instead of `v<version>`, so a build can go out for testing without spending the version number that the real release will use. Still created as a draft, still kept out of Homebrew.

### Media

- **ACE-Step Cover**: drop in a track, describe a new style, and it re-sings the song. Melody and structure stay, the caption and lyrics decide the rest. Cover strength picks how much of the render follows the source; noise strength blends a fresh start in. Needs the new `fsq.safetensors` in the pack; the app fetches it into packs downloaded before this release.
- **ACE-Step Vocal to BGM**: arrange around a vocal stem (or any single part). Pick the instruments to add, or leave them all off and let the model decide. The new track is exactly as long as the clip, 10 seconds to 10 minutes.
- **Reference audio for music** (#259): a clip whose feel and timbre the track follows. Up to 30 seconds is used; it is a style hint, not a copy. Thanks @Fe2-O3.
- **LTX first and last frame** (#260): `last_frame_image` pins the final frame the same way `first_frame_image` pins the first, on every LTX pipeline including two-stage. A request the model cannot honour (no VAE encoder, fewer than 9 frames) is a named 400 rather than a plausible video of something else.
- `instrumental: true` beside non-empty lyrics is a named 400 instead of a silent choice.

### App

- **Send a video to Qwen3-VL models** and chat about it. Thanks @justinluque (#246).
- **Rewrite with LLM**: wand buttons next to the style prompt and lyrics. The chat model rewrites the text in the loaded music model's own format (one-line ACE-Step caption, three-block Music 3 caption, tagged lyrics) and streams it into a sheet you can edit before applying.
- Dropping a long audio file on the Music tab no longer freezes the window: the conversion runs in the background with a "Converting" indicator, and the WAV writer is a single pass instead of one append per sample.
- The Music tab's Advanced controls line up in one grid, and the Voice / Music switch is larger with room above the pane.
- The Video pane now exposes all five generation settings it used to hide. Thanks @Fe2-O3 (#244).

- **Music mode gained an instrumental switch** plus tempo and key controls, and remembers your settings between generations. Instrumental is marked experimental on MiniMax Music 3, where the open weights have no real switch and the tag alone still leaves vocal texture in; ACE-Step's is documented and works. Closes #225. Thanks @Fe2-O3 (#226).
- **Media checkpoints in My Models stopped reading as "Unsupported"** and gained a Use button. MiniMax Music 3, MageFlow and Kokoro were missing from the app's copy of the architecture list, so a model the app itself offers to download came back with a red badge. Thanks @Fe2-O3 (#229).
- **The tray shows live prefill and decode tokens/s** Metrics are on by default now so those rows always have something to read — the cost is a few counters per request, never per token.
- **The memory meter is one bar** — model, everything else in use, free — instead of two bars measured against the same total, which invited reading them as if they added up.


## v26.8.9 — Launch your coding agent, richer chat, faster decode

### Highlights

- **`mlx-serve launch <agent>`.** One command (or one click in the app) configures and starts Claude Code, pi, oh-my-pi, OpenCode, Codex, hermes, or aider against the local server — each agent gets its own config folder and the model's real context window, and the app starts itself first if the server isn't running.
- **Chat picked up the moves of a real editor.** Continue a cut-off reply instead of restarting it, edit or regenerate any message with a version history you can page through, branch a conversation from any point (right click your own chat bubble), and bulk-select or ⌘+digit-jump between chats.
- **Decode got faster on several fronts**: quantized draft heads, a deeper speculative-depth planner, faster verify passes, and — specifically on Qwen 3.8 27B — automatic depth-8 speculation on the calibrated hardware profile, up to 31% faster than the previous depth-6 cap.
- **Long chats with images stopped erroring out** (#197). Vision prompts now prefill in the same memory-bounded chunks text does, so a screenshot 51k tokens into a conversation runs instead of getting refused.
- **Concurrent chats no longer stall behind a big prefill.** A cold prefill used to block every other stream until it finished (7.6s on a 53k-token prompt); it now yields at chunk boundaries, capping the stall at about one chunk (~150ms), byte-identical output either way. Thanks @sf-jin-ku (#205).

### Coding agents

- `mlx-serve launch` supports `claude`, `pi`, `omp` (oh-my-pi), `opencode`, `codex`, `hermes`, and `aider`. `--model` picks a model, `--print` shows the launch script instead of running it, and anything after `--` passes through to the agent (`mlx-serve launch codex -- resume`).
- oh-my-pi and other OpenAI-style clients that read a model's context length from the top level of `/v1/models` instead of `meta.*` now see the real advertised context instead of a hardcoded 128k.
- The context size and max-tokens sliders in Settings gained finer steps between 32K/64K/128K, so a memory-limited Mac isn't stuck jumping straight from too small to too large.

### Chat

- A reply that hit the token limit gets a **Continue** action that extends it instead of starting over.
- Double-click your own message to edit it, ⌘R to regenerate a reply — both keep every prior version behind a pager.
- Right-click any message to branch the conversation from that point into a new chat.
- ⌘-click / ⇧-click select chats in the sidebar and ⌘⌫ deletes the selection; hold ⌘ to badge the visible chats 1-9 and jump with ⌘+digit; ⌘L opens a model switcher without leaving the transcript; ↑ in an empty composer recalls your last message, ⎋ stops a reply mid-stream. Thanks @justinluque (#180).
- Pick any resolution you like for images and video. The Image and Video panes gained a **Custom…** option with width and height fields, alongside a new 512 × 512 preset for FLUX — the server always accepted it, it was just missing from the menu. A size the model can nearly do is nudged onto its grid with a note saying what it used and in what steps, so the next guess lands; a size it cannot do at all is refused with the range it enforces, instead of silently coming back a third of the size you asked for. Video is where this matters most: an off-grid canvas there is refused outright by the server, and a two-stage LTX tier tightens the rule further, so the fields follow the tier you picked. Thanks @justinluque (#218).
- Assistant responses render inline and display LaTeX natively with SwaTex, including `$...$`, `$$...$$`, `\(...\)`, `\[...\]`, and common equation environments. Incomplete or invalid streamed TeX stays readable as source, fenced code and user prompts remain literal, and copying inline math restores its original delimiters. SwaTex is MIT-licensed; its bundled KaTeX fonts retain the SIL Open Font License 1.1.
- Video generations now write a `.txt` settings sidecar next to the clip — model, preset, seed, resolution, frames, fps, steps — matching what audio and music generations already do. Thanks @Morac2 (#199).

### Performance

- Speculative decoding got faster without KV quantization: verify steps 6-9 tokens wide at head-dim 256 ran MLX's slow attention fallback on every machine. They now split into two fast passes, +4-9% decode with PLD or a deep draft head on Qwen-class models.
- Draft heads that ship in bf16 (MTPLX packs, the stock Qwen MTP release) are now quantized to 4-bit at load. The head only proposes tokens and verification corrects them, so output quality is decided by the main model either way: measured +10% decode at equal acceptance on Qwen3.8-27B.
- The speculative depth planner was re-measured against the faster verify steps: it now drafts one position deeper on predictable content, +3% decode on Qwen-class models with the draft head.
- Warm requests kept the fast prefill but lost the draft head: reusing a cached prefix left the head's history empty, so follow-up turns decoded at almost half speed (38 vs 72 tok/s measured). The history is now saved and restored with the prefix, so warm turns decode as fast as cold ones.
- Qwen 3.8 27B can now auto-speculate 8 tokens deep instead of capping at 6, once its checkpoint matches a calibrated quantization profile — the 4-, 6- and 8-bit MLX-Serve builds all qualify. Measured up to 31% faster decode at depth 8 vs. 6 on the 4-bit build. Thanks @CerebralCoding (#194).
- Images stopped failing in long conversations (#197). A prompt with an image ran as one whole-prompt forward, so the memory check billed the full width and past roughly 40k tokens on Qwen 3.8 27B every screenshot got a 400 no flag could fix. Vision prompts now prefill in the same memory-bounded chunks text uses, with the image splice resuming exactly across chunk boundaries: a 51k-token conversation with a screenshot that used to be refused (82 GB billed against 67 available) now runs, output is unchanged on LFM2.5-VL, Gemma 4 and Muse, and time to first token stays within 3% either way. Cancelling mid-prefill now also stops a vision prompt within one chunk instead of running it to the end.
- Concurrent chat streams no longer stall for an entire cold prefill: a 9k-token prefill used to freeze every other active stream for 0.8s, a 53k-token one for 7.6s. Prefill now yields at chunk boundaries, capping the stall at about one chunk-forward (~150ms), with byte-identical greedy output. Thanks @sf-jin-ku (#205).

### Fixes

- Claude Code's newer `output_config.effort` and `output_config.format: json_schema` fields are now honored on `/v1/messages`. Previously ignored, so every Claude Code request ran with an unbounded thinking budget, and JSON-schema replies came back as plain markdown the client rejected.
- `POST /v1/unload-model` now rejects a body that names the wrong key (`model_id`, `id`) with a 400, instead of quietly unloading the default model and reporting success. Thanks @Fe2-O3 (#211).
- Fixed a memory/CPU leak in the app: a slow or failing MCP server connection kept running in the background forever instead of being torn down, measured at ~160% sustained CPU over a 23-hour session. Thanks @slava-kudzinau (#201).
- Fixed a memory leak from long-running servers accumulating one thread stack per HTTP connection instead of reaping it. Thanks @sf-jin-ku (#203).
- The command-line binary only started when launched from the repo root; it now resolves its library path relative to itself, so it runs from anywhere it's installed. Thanks @Fe2-O3 (#193).

## v26.8.8 — Faster 6-bit models, Better memory checks, UI Bug fixes

### Highlights

- **6-bit builds decode faster with speculation.** Our verify kernels only served 4-bit weights, so every 6-bit model fell back to stock kernels on M1-M4 Macs. They now serve 5, 6 and 8-bit too: 10-22% faster decode with the draft head on the Qwen 3.8 27B 6-bit build, same output.
- **Memory checks are honest in both directions.** The admission guard was measured against real prefill peaks on five checkpoints and came up short on 5 of 8 shapes, worst 42% under, which is the difference between a clean "prompt too large" and the whole server dying in a Metal abort. Every measured peak is billed now, hybrids and MoE included, and image prompts are billed at the width they actually run.
- **Downloads stopped looking like they restart.** Every progress bar drew the current file, so a four-shard model filled 0-100% four times. Bars now show the whole transfer, resumed bytes included.
- **Switching models shows a spinner.** A hot switch to a big checkpoint used to sit for a minute under the old model's name and a green dot. The pill now names the model it is loading and spins until it answers.

### Fixes

- Speculative decoding with a quantized KV cache collapsed past 8k context: 28 tok/s where the dense read does 60, because spec verify steps read the packed cache through a chain of small quantized matmuls. Verify steps now have their own packed-read kernel on the shapes it was measured to win on, and fall back to a plain dense read everywhere else. Measured on Qwen3.6-27B and Qwen3.8-27B with the draft head on: 28 to 61 tok/s at 11k context, and at 32k the quantized cache now decodes within 2% of running with no KV quantization at all, at half the memory.
- The same pass found the quantized-KV fused read was also losing on Gemma 4 without speculation: its full-attention shape fell to the slow composed chain on every token and its sliding windows sat exactly at the engagement floor, together a 1.45x decode loss at 11k. Both read dense now, 21 to 30 tok/s.
- LFM2 and Nemotron-H were billed a KV cache for every layer when only their attention layers keep one, which charged LFM2 3.75x the real bytes and shrank its auto-context for nothing.
- A model loaded while a bigger one was resident kept its narrowed prefill width forever, even after the big one was evicted. It re-resolves on the next load.
- A model served out of the Hugging Face cache showed its commit hash in the model pill instead of its name.
- Remote MCP servers can send auth headers now: a `headers` block on a `url` entry in mcp.json (an `Authorization` token, an API version) was silently ignored, so the server got an unauthenticated connect, and saving any MCP change deleted the block from the file. Headers now ride every request and survive edits, and unknown fields like `type` are kept too.
- `--prefill-chunk` with a typo in the value silently became 8192 and turned the machine sizing off. A bad value now keeps the defaults.

## v26.8.7 — Qwen 3.8 27B, Ling 3.0, thinking that knows when to stop

### Highlights

- **Qwen 3.8 27B runs on your Mac**, hours after Qwen released it. Text, vision and tools, in an 18.2 GB 4-bit build with the draft head baked in: about **75 tok/s on code** and 40 on prose on an M4 Max.
- **Ling 3.0 runs.** inclusionAI's hybrid model, a new attention design we hadn't served before. The 4-bit tiny build is 4.2 GB and does 96 tok/s.
- **Thinking models stop thinking sooner.** Qwen 3.8 asked to think as hard as it can unless told otherwise, which on an agent turn meant 16k tokens of reasoning before the client gave up. It now thinks briefly by default, and a thinking cap shortens the thought instead of hiding it.
- **The app finds your Hugging Face cache wherever you put it.** If you moved it with `HF_HOME` or `HF_HUB_CACHE`, the app never saw the variable and listed nothing. It asks your shell now.
- **Drag files onto any Create pane.** Image, Video, Audio and 3D all take a dropped file, including the reference lists.

### Qwen 3.8 27B

- It is the app's default recommendation on any Mac with more than 32 GB, replacing Qwen 3.6 27B in the welcome card, the Model Browser and the menu-bar tray. `ddalcu/Qwen3.8-27B-MLX-Serve-4bit`, 18.2 GB, wants 24 GB+ of RAM.
- Images work. Tools work. Thinking works, and the model ships its own draft head, so it is fast out of the box: 75.3 tok/s on code and 39.7 on prose against 26.3 / 26.7 with the draft head off, same Mac and same prompts.
- Qwen 3.8 has its own words for how hard to think (`xhigh`, `medium`, `low`) and its template refuses anything else, so `reasoning_effort: "high"` from an OpenAI client is translated instead of rejected. Sending nothing gets `low`: the model's own default is unbounded, and asking for a short answer is not the same as cutting a long one off.
- The bigger 2.4T-A95B sibling renders correctly too, including its refusal to turn thinking off.

### Ling 3.0

- Pick any `bailing_hybrid` build, such as `rapid-mlx/Ling-3.0-tiny-MLX-4bit` (4.2 GB). Measured on that build: 2320 tok/s on a prompt, 96 tok/s decoding, and a fact 30k tokens back recalled exactly.
- Thinking is on by default, the way the model's own template asks for it. Tool calls, multi-turn and prompt reuse all work.
- Thanks @justinluque (#166).

### Create panes

- Drop a file on any pane and it lands in the right slot: the Image source and its references, the Video first frame, the 3D photo, the reference voice clip, and the ref2va reference lists, which sort a mixed drop by type. A file the pane can't use is refused while it's still in the air instead of being swallowed. Thanks @justinluque (#139).
- The Image pane's pictures are one numbered list, so the numbers match what your prompt refers to.
- New **544 x 960 portrait** preset for MiniMax-H3, the fastest one for long clips. Thanks @Morac2 (#177).

### App

- New **Only use tools when I ask** setting in Settings, Agent Workspace. With it on, the composer stops offering to turn Tools or MCP on when your message looks like a task, and tools are only ever on because you turned them on. Thanks @justinluque (#174).
- Models in your Hugging Face cache are listed wherever that cache lives. `HF_HUB_CACHE`, `HF_HOME` and `XDG_CACHE_HOME` are read from your login shell, since an app launched from Finder inherits none of them, and a cache you pointed elsewhere that isn't there is empty rather than quietly falling back to the default folder.

### Fixes

- A capped thought is now streamed as it is written. With tools in play, asking for a reasoning budget showed nothing at all until the model finished and then dumped the whole cut-down thought at once, so a capped agent session looked frozen.
- Asking a model for `medium` effort could make it think longer, not shorter: the word went to the model and a token cap was derived from the same word, so the reply was cut where you could see it while generation ran on invisibly. On models whose template reads the word, the word is now the only lever, and an explicit `reasoning_budget_tokens` still caps.
- A tool declared without a description, or without parameters, could silently drop the entire tool list from the prompt. Both are legal, and the model was left with no idea the tools existed.
- LFM 2.5 VL read numbers in your prompt one digit at a time, which put every number off distribution, and its image tiles were resized with the wrong filter.
- pi's own thinking-level picker reaches the server now. It was a label the client kept to itself, so every request arrived with no level set.

## v26.8.6 — LTX-Video 2.5, MiniMax Music 3, faster long chats

### Highlights

- **LTX-Video 2.5 runs on your Mac.** Lightricks' newest video model, joint audio+video like 2.3, in a 4-bit build (36 GB) and an 8-bit quality build (59 GB). Text-to-video, image-to-video, audio-to-video and the two-stage pipelines all work.
- **MiniMax Music 3 writes full songs.** An 8B language model composes the track frame by frame from your style caption and your lyrics, then a diffusion decoder renders it at 44.1 kHz. Strongest vocals we ship.
- **LFM 2.5 VL reads images.** Liquid's small vision model runs now. Big pictures are split into tiles instead of shrunk to fit, so fine print stays readable.
- **Video renders at the size your Mac can actually hold.** The default canvas used to be 768x512 on every machine, which is a quarter of what LTX's own pipeline denoises. It is now picked from your memory, so a big Mac gets a big picture without touching a setting.
- **Long chats got faster on sliding-window models.** Muse-Glimmer 30B decodes 63% faster at 16k and 144% faster at 64k; Laguna XS chews through a 64k prompt at more than twice the speed.

### LTX-Video 2.5

- Pick **LTX-Video 2.5** in the Video window. It brings its own text encoder, so unlike 2.3 there is no separate 8 GB download on first use.
- Two packs. 4-bit for Macs that can't hold more, 8-bit when they can: 4-bit quantization injects about 10% noise into every layer of the model against 0.6% at 8-bit, and the same clip at the same seed keeps faces, legs and fur that the 4-bit render loses. It costs 3.5% more time, because the model is compute-bound at this size.
- **Diffusion decoder** toggle (8-bit pack only): LTX's own decoder, the one their published clips use. It denoises the frames instead of interpolating them, so texture and edges come out sharper. Adds about 21 s on a 97-frame 768x512 clip, which end to end sits inside run-to-run variance. Over the API it is `"decoder": "diffusion"`.
- Resolutions and frame counts are now per-Mac and per-canvas: bigger machines default to a bigger canvas, and the frame ladder follows it. Two-stage tiers denoise at half the chosen size and upscale, so on a small canvas "Quality" is softer than the tier above it.
- LTX prompts were padded to a quarter of the length the model expects. Both versions now use the full length, so prompts are followed more closely and long ones are no longer cut short.
- 2.3 keeps working exactly as before, and both can sit side by side.

### MiniMax Music 3

- New music model in the Music window, alongside ACE-Step. 8-bit pack, 13.6 GB to download, about 20 GB of memory to run.
- Give it a style caption and lyrics and it sings them. Songs up to six minutes; the duration is an upper bound and the model may end earlier.
- Lyrics are required. Structure tags like `[verse]` and `[chorus]` go on their own lines.
- ACE-Step's tempo, key, meter and language controls don't exist on this model, so they disappear when you pick it. Put those facts in the caption instead.
- The app ships a **music3** skill that writes the three-block caption format the model was trained on, so asking the chat for a song gets you a proper caption and original lyrics rather than a one-liner.
- ACE-Step is unchanged, and stays the fast option (8 steps).
- Over the API: `POST /v1/audio/music-generations` with `prompt` and `lyrics`.

### LFM 2.5 VL

- Pick **LFM 2.5 VL 3B** and attach an image. It is small and quick: 2.2 GB in 4-bit, and it answers at over 200 tokens a second.
- A large picture is cut into tiles and sent with a thumbnail of the whole thing, rather than being shrunk down to fit. On a 1800x1400 screenshot that is seven times the detail, which is the difference between reading the fine print and guessing at it.
- The smaller 1.6B build works too.

### Skills from the composer

- Type `/` in the chat box to see your skills and pick one; `/name` runs that skill in any chat, agent mode or not.
- Skills that ship with the app are now seeded one at a time, so a skill added in a later version reaches existing installs. Deleting one still sticks.

### Long-context speedup

A sliding-window model looks back over a fixed window, not the whole conversation. The server trimmed its read to that window on single-token steps only, so anything wider read everything: every speculative step, every chunk of a long prompt. On a model where 3 layers in 4 slide, that was full attention on most of the network, every round. Nothing to turn on, nothing to tune.

Measured against v26.8.5 on an M4 Max, median of three runs per point, same models and settings on both:

| Model | | 16k context | 64k context |
|---|---|---|---|
| Muse-Glimmer 30B 4-bit | decode | 24.7 -> **40.2 tok/s** | 8.6 -> **21.0 tok/s** |
| Laguna XS 2.1 NVFP4 | prompt | 609 -> **774 tok/s** | 236 -> **540 tok/s** |
| Laguna XS 2.1 NVFP4 | decode | 62.7 -> **77.7 tok/s** | 34.9 -> **46.8 tok/s** |

The gain grows with the conversation, so short prompts are unchanged and below roughly 8k there is nothing to trim yet. Inkling Small gains the same way on prompt processing. Gemma 4 gains less: its prompt processing already applied the window itself, so only its speculative steps get quicker.

### Fixes

- Laguna models converted by mlx-community (oQ4e, oQ5e) load now. Their converter nests the router weight where we weren't looking (#169).
- Models whose chat template lives in a separate file next to the config are rendered correctly. Newer conversion tools write a pointer into the config instead of the template, which we read as the template itself and quietly fell back to a generic format, breaking tool calls (#169).
- Sending Muse-Glimmer an image quietly switched its draft companion off, so vision chats ran at serial speed. Images keep the speedup now (thanks @cerebralcoding, #160).
- A speculative round could push a reply past the token limit you asked for. `max_tokens` is now respected exactly (thanks @cerebralcoding, #160).
- Picking up an earlier conversation kept the drafter speedup instead of quietly drafting blind, and the draft size adapts correctly on Macs without the widest verify path.
- Streaming and non-streaming replies could differ by a couple of blank lines at the start of an answer. The same question now gives the same text either way, on chat completions and on the Anthropic endpoint.
- Turning thinking on for a model that has no thinking mode put the whole reply inside the Thinking box and left the answer blank. Only streaming replies were affected.
- A tool call could be named after the wrong tool when one of its arguments contained something that looked like a tool call, such as a package.json being written to a file. Qwen 3.5 and 3.6 allow two ways of writing a call and the server read the wrong one first, so a file's contents could decide which tool ran.

## v26.8.5 — Muse-Glimmer 

### Highlights

- **Meta's Muse-Glimmer-30B runs on your Mac.** Chat, tools and thinking all work, with 4-bit and 8-bit ddalcu builds on the Hub with DFlash built in (up to **75 tok/s on M4Max**).
- **Speculative decoding you don't have to set up.** A model can carry its own draft companion, and the server picks it up whenever that model loads. Muse-Glimmer decodes about twice as fast with it.
- **Thinking got quicker and more visible.** Thinking off no longer waits on a hidden reasoning pass, thinking that does happen is shown instead of thrown away, and you can pick how hard the model thinks right in the composer.

### Muse-Glimmer

- The 30B model runs natively, its chat and tool format handled end to end. Vision isn't served yet.
- Ships with its own draft companion, so it's fast out of the box.
- Expect between 30-75 tok/s on 4-bit, and 16-52 tok/s on 8-bit on M4Max

### Drafters that ship with the model

- A model folder can include a `drafter/` companion; the server loads it with the model, and switching models keeps the speedup. `--no-drafter` turns it off.
- The draft size adapts to your Mac, and reused conversations keep their speedup instead of restarting it cold.
- https://huggingface.co/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit
- https://huggingface.co/ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit

### Thinking

- Thinking off means no thinking: the reply starts right away instead of after an invisible reasoning pass — on Muse-Glimmer that pass was half a minute of silence.
- Right-click the thinking icon to pick the effort; click still toggles it. Tool chats keep thinking by default, plain chats skip it, and the "thinking with Tools" warning is gone.

### Fixes

- The "stopped repeating itself" notice showed twice on a cut reply and could be sent back to the model as chat text. It now shows once, under the reply, and never reaches the model (#147) thanks @justinluque for your PR.
- GGUF models downloaded with the Hugging Face CLI load again (#158).
- Big video renders no longer get cancelled after 15 minutes of quiet work (#152, #157).
- Fix bugs related to model hot swap / changing models.
- Homebrew now learns about a release when it's published, not while it's still a draft, so `brew upgrade` can't offer a version whose download isn't up yet.

## v26.8.4 — One window, your own media models, hot model switching

### Highlights

- **The app is one window now.** Models, Tasks, Settings and the media generators live inside the chat window as modes instead of scattered windows, behind a three-column layout with a proper Agents section in the sidebar.
- **Switch chat models without restarting.** Picking a model in the app loads it into the running server and makes it the default, instead of tearing the server down and booting it again.
- **Bring your own media models.** The image, video, voice, music and 3D panes list checkpoints you added yourself, and the Model Browser downloads community packs of those families.
- **The Agent Sandbox is a normal Linux.** `apt-get install` works, and the agent CLIs that failed to install now install.
- **Agents can pin their own sampling.** Top-p, top-k, repeat penalty, presence penalty and reasoning budget join temperature and max tokens per agent.

### One-window app

- One window, three columns: sidebar, content, detail. Models, Tasks, Settings and the media generators became chat modes rather than separate windows, and the persistent toolbar is gone — titles and create buttons moved into the native navigation bar.
- The sidebar has a dedicated **Agents** section, and the agent editor was rebuilt: real cards instead of a cramped form, proper naming, threading and capabilities.
- The welcome screen is a sheet on the chat window instead of a floating window, and creating a model-backed chat has its own Create pane with a single-row model picker.

### Model switching without a restart

- Changing the chat model used to restart the server. It now loads into the running one and takes over as the default, so requests that omit a model — the Claude Code launcher, plain `curl` — reach the model you just picked.
- Over the API this is `POST /v1/load-model` with `"default": true`. Without the flag a model loads alongside the current one and the default is untouched, which is what media generation uses so it can never steal the chat model.
- Known gap: Gemma 4 hot-loaded this way runs without its drafter speedup companion until the next restart.

### Use your own media models

- The media panes list anything in your model folders with a family the server can run, under **On This Mac** — with that family's settings and controls.
- The Model Browser offers community packs of those families. A repo's layout is checked against the family's converted shape before the Download button appears, so only packs that will actually load are offered, and they download as a full bundle.
- Models downloaded while the server runs appear without a restart (`POST /v1/models/rescan`), and the app calls it after every download.
- Mage-Flow moved to the `mage-flow-community` org; the bf16 build left the built-in list and shows under On This Mac if you have it.

### Agent Sandbox

- `apt-get install` works. Apple's file sharing made any file created without an owner-read bit unreachable from inside the guest, which broke every package install and the Node extract in the agent CLI installers. The sandbox now ships a patched kernel that fixes it, plus `xz-utils` and a newer npm baked in.
- Agents no longer crash on launch on an M4. The old guest kernel advertised a CPU feature the chip does not have, and anything probing it — OpenSSL, Go binaries — died instantly.
- The sandbox gets up to 4 GB of RAM instead of 1 GB, which is what killed new version of Hermes. It is committed lazily, so idle sandboxes stay small.

### Agents pin their own sampling

- Top-p, top-k, repeat penalty, presence penalty and reasoning budget join temperature and max tokens in the agent editor (#135).
- Each is an override: App default follows Settings, a set value wins for that agent's turns, and an off value (top-k 0, repeat penalty 1.0) clears your global default for that agent.

### Fixes

- Generating a video from reference clips no longer fails with "Request body too large" (#151). Reference media rides as base64, so one clip alone is around 100 MB; the limit is per endpoint now — 512 MB for media, 64 MB elsewhere — and a refusal names both numbers.
- "Model load failed" now says why (#144). A load refused by the memory check comes back as a clear not-enough-free-memory error you can retry after closing other apps; any other failure names its reason.
- Picking a download folder no longer hides the models you already have, and a media pack in an extra model folder no longer shows a Download button for a copy already on disk. Every folder the server scans is checked; deleting stays limited to the app's own folders.
- Models served from the HuggingFace cache no longer slip past the memory check. Those folders store weights as symlinks and every size scan skipped them, so a 121 GB model measured as 0 bytes and swapped the machine.
- The Gemma 4 QAT speed-up companion loads again instead of crashing the model (#109). Quantized companion checkpoints are unpacked at load; the forward pass expects plain weights.
- The MiniMax-H3 time estimate stopped swinging and the live "time left" stopped under-promising: cheap cached steps were pricing the expensive closing ones, and the video decode after the last step was not counted at all, so "2 min left" could take 4.
- Long code blocks no longer make the chat stutter while a reply streams in.
- GGUF repos that ship the same quant for several releases label each file with its build (`0731` and so on), so the new DeepSeek files are tellable from the old ones in the quant picker.
- `--model-dir` is repeatable, and `~/.mlx-serve/models` is always served even when you set a custom download location.
- Starting the server without `--host` now warns that it is reachable from the network you are on, since the default bind is still `0.0.0.0`. Pass `--host 127.0.0.1` to keep it local; a future version will make that the default.
- Hybrid models (LFM2.5, Nemotron-H) leaked a little memory on every load and unload.
- Internal docs reorganized and compacted.

## v26.8.3 — MiniMax-H3 references and Turbo, stacked LoRAs, model folders

### MiniMax-H3: references, Turbo, longer clips

- Attach pictures, clips or audio and the video is built around them. Refer to them in the prompt by number: `<Picture 1>`, `<Video 1>`, `<Audio 1>`. This needs the REF2VA build, so the app only offers it when that build is loaded and the server refuses references on a build that would silently ignore them.
- **Turbo** renders in 4 steps instead of 30, about twice as fast end to end, with slightly softer detail. It ships with the model now, and packs downloaded before it existed fetch it once when you tick the box. Thanks to [larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) for the adapter.
- Clips reach 15 seconds. Under the Generate button there is a live time estimate that updates as you change size, length and steps, and the memory warning now uses H3's own numbers instead of LTX's.
- Video models no longer refuse to load on a 48 GB Mac (#126). The memory gate billed the sum of every file in the folder, 37.55 GB, when the parts are never all in memory at once. Thanks to funk80rus for the report and the diagnosis.
- Over the API, long clips can be generated as chained windows (`chain_windows`), each continuing from the last frame of the one before.

### LoRAs

- Attach several style adapters at once, on any image or video model. Their effects add up, so the order does not matter. Thanks to Justin ([@justinluque](https://github.com/justinluque)) for the multi-adapter work (#118).
- Adapters now run at the strength their own file declares. Anything exported by PEFT or diffusers was running up to 8x too strong, which renders as static rather than a stronger style. If you dialled one down to compensate, put it back up.
- MiniMax-H3 takes them too, stacked on top of Turbo.

### Model folders: pick where downloads go

- **Settings ▸ Model Folders ▸ Default folder** chooses where new downloads land. Everything already downloaded keeps working, because the old folder stays in the scan list.
- Custom folders are actually served now. `--model-dir` is repeatable, so every folder you list shows up in `/v1/models` instead of only in the app's own picker. If the same model sits in two folders the download folder wins, and a folder that is not reachable is skipped with a warning instead of stopping the server.

### Logprobs. Fixed.

- Three things were wrong at once, on every model: the numbers moved when you changed the temperature instead of being the model's own, ties handed back an arbitrary token, and the whole list was off by one so each token came back with the *next* token's alternatives.
- None of this changed the text any model produced. It matters if you score outputs, route on confidence, or build evals.
- `/v1/completions` ignored its `logprobs` field entirely. It works now, in the shape the OpenAI legacy API defines.
- Streaming ignored it too, on both chat and `/v1/completions` — and it cost you speed to do it, because asking for logprobs turns off speculative decoding whether or not you get anything back. Streaming now returns the same numbers as non-streaming, token for token.
- On a model that thinks, the numbers described the thinking, not the reply. `logprobs.content` is the tokens of the answer, but it was built from the whole generation, so the first entry was the opening token of the model's reasoning and the list lined up with nothing you could see — 186 entries against an 8-character answer on Qwen3.6. It now covers exactly what comes back in `content`, streaming and not.
- A reply could come back unreadable. Tokens are fragments, so one can hold half an emoji, and those raw bytes went into the JSON — the whole response then failed to parse, not just the logprobs. The token text now shows the standard replacement character and the `bytes` field beside it still carries the exact bytes.

### Also new

- **Pick your quant from a multi-variant repo.** Publishers who ship every quantization in one repo as subfolders used to read as a single model; each variant now shows its own size and downloads, loads and deletes independently, like GGUF repos already did.
- **The seed is a text field you can paste into** in the image and video panes, with a dice button beside it that gives you a concrete number to read off and paste back.
- **The app says why a reply stopped.** A cut caused by the model going in circles is labeled as one instead of claiming you hit the output limit. On the API that is `finish_details: {"type": "repetition_loop"}` beside `finish_reason`.
- **A quant playground on the website**: pick a Mac, a model, a quantization and a context length, and see the speed you can expect and whether it fits in RAM.
- LICENSE, NOTICE and the Apache-2.0 text now ship with the CLI tarball, the app bundle and the Homebrew install, crediting the kernels and libraries mlx-serve builds on.

### Fixes

- Quitting the app left `mlx-serve` running with the model still loaded (#133). ⌘Q, the Quit menu and Dock ▸ Quit all went straight to termination without stopping it, so it kept holding the memory until you killed it by hand. The power button in the menu bar was the only route that worked. Stopping the server is part of quitting now, whichever way you quit. Thanks to freppair for the report.
- The 8-bit video models still would not load on a 48 GB Mac. The gate had stopped billing the whole folder but was still adding up parts that never exist at the same time, and still charging the transformer its file size when nearly 40% of it is released the moment a generation starts. It bills the largest stage on its own now: 28 GB instead of 43 for the 8-bit MiniMax-H3 build, against a measured peak of 26. The largest sizes can still run out of memory mid-render on a 48 GB Mac, but the model loads and works.
- **Settings ▸ Server ▸ Model memory cap**, a slider, Auto by default. The setting that decides whether a model is allowed to load at all was reachable only from the command line, so when the automatic cap refused a model you knew would fit there was nothing to do about it in the app. "Skip memory pre-flight check" does not help here, because the cap is checked before that.
- LTX video was billed for a transformer it never loads (packs ship two, only one is ever in memory) and not billed at all for its text encoder, which lives in a separate folder.
- A video or music generation kept running after the client hung up, with everything else queued behind it for the rest of the run. Requests that do not stream progress had no way to notice at all, which is most API clients. Note that long renders need a generous client timeout either way: 1344x768 can take well over 45 minutes.
- Reserved tokens such as `<|fim_hole|>` can no longer be sampled into a reply. The list is built per model from its own tokenizer and chat template, so thinking tags and tool markers are untouched.
- Tool calls no longer vanish when an argument mentions `</think>`. Coding agents write files about prompts, so this was reachable on every thinking model.
- The transcript follows a streaming reply and stops the moment you scroll up, on every input rather than only the mouse wheel.
- The MiniMax-H3 fast recipe silently did nothing on runs of about 8 steps or fewer while still reporting that it was on.
- Added a repetition guard for loops that reword themselves; the existing guards only caught exact repeats.
- Streaming a long thinking block no longer costs CPU that grows with the length of the thought.
- `/detokenize` returned bodies that no JSON parser would accept if any token contained a control byte.

Thanks [@h9q2cyxvgm-ui](https://github.com/h9q2cyxvgm-ui), [@funk80rus](https://github.com/funk80rus) and [@AideYu](https://github.com/AideYu) for testing and finding a bunch of problems! It really helps ! 

---

## v26.8.2 — MiniMax-H3 video with its own soundtrack, LFM2.5

### MiniMax-H3 (Hailuo 3.0) generates video and audio together

- One prompt gives you a clip and a matching stereo soundtrack, produced in the same pass rather than dubbed on afterwards. Describe the scene, then what you want to hear after `overall_soundscape:`.
- Two builds on Hugging Face: 8-bit at a 69 GB download (44 GB while running) and 4-bit at 40 GB (26 GB), which fits a 32 GB Mac. Same generation speed either way, the 4-bit is a little softer.
- It is genuinely slow. The recommended 1344x768 at 124 frames takes about 50 minutes on an M4 Max, and 209 frames closer to two hours. That is with the fast recipe on, which is the default and about 2.8x quicker than turning it off in Settings.

### LFM2.5 support

- Liquid AI's LFM2.5 runs end to end: chat, thinking and tool calls.
- It writes tool calls in Python syntax rather than JSON, so numbers, lists and true/false now arrive as the types your tool actually declared instead of strings.
- The mlx-community 8-bit and nvfp4 builds load now. They used to stop at startup with a missing-weight error.

### Thinking no longer shows up as your answer

- Gemma could reply to a short question with an empty message and file the actual answer as its private reasoning. "What is 23 times 17" came back blank.
- LFM2.5 streamed its entire chain of thought as the visible reply, with and without tools.
- Both were streaming only, so the same question answered correctly if you turned streaming off. Fixed in both directions.

### An empty chat shows what else the app does

- Media generation, the Model Browser, Tasks and the coding-agent launcher only lived in the menu-bar tray, and people told us they never found them.
- They are chips under the greeting now, and they disappear as soon as the conversation starts.

### Fixes

- Checkpoints kept on an external drive and linked back show up in `mlx-serve list` again. The server was already serving them.
- Picking a LAN-shared video model now uses that model's own resolutions and frame counts instead of whatever was selected locally, which could produce bad clips that looked like a model problem.
- Video generation shows a live progress bar again instead of a dead card for the whole run, and cancelling actually stops the GPU.
- Loading MiniMax-H3 in the app read it as a chat model and failed on a missing tensor.

---

## v26.8.1 - New DeepseekV4 Flash optimizations, Qwen3 embeddings

- **DeepSeek V4 got a lot faster on our engine.** Serial decode went from 22.6 to about 30 tokens per second on an M4 Max (31.8 with fast decode on), and prefill roughly doubled to about 270 tokens per second at 8K. DSpark was retuned and now also works on sampled requests, which is what agent CLIs actually send: with it on, code, lists and editing decode at **54-68 tokens per second, about twice serial.**
- **A better DeepSeek V4 conversion.** The published model on HF is rebuilt with imatrix calibration collected on the 0731 weights themselves, and the last few expert layers moved to 4-bit: agent sessions no longer get stuck repeating themselves the way uniform 2-bit did. Also found and fixed a tokenizer bug that split numbers digit by digit before the model ever saw them, which read as quant damage ("1o" instead of "10") and wasn't.
- **Two chats at once no longer corrupt each other on DeepSeek V4.** Its decode state lives on the model, not the request, and a second concurrent request used to reset it mid-generation: replies leaked between conversations with every word doubled. A second request now waits its turn, and streaming clients get keepalives while they queue. Other models are unaffected and still run concurrently.
- **Qwen3-Embedding models work now (#116).** Serve the mlx-community Qwen3-Embedding conversions on `/v1/embeddings` and `/api/embed` like any other model. The server reads the checkpoint's sentence-transformers pooling metadata (or recognizes the family by name, since the MLX conversions strip it) and pools the last token the way the model card says, instead of returning wrong-recipe vectors. Verified against the official reference: our vectors match to within quantization on the 4-bit build, and bit-for-bit against mlx-lm on the same weights. A checkpoint that declares a pooling recipe we don't implement is refused with a clear error instead of being served with the wrong one. Also Embedding input limits are explicit (#117) `--embedding-max-length`
- **bge and mxbai embeddings are now correct (#116).** Their model cards ask for CLS pooling and we were mean-pooling them; bge-small now matches the sentence-transformers reference. Heads up if you built a vector index against an older release: re-index it, old vectors and new queries no longer line up. The app's own attach-a-folder search rebuilds per session and needs nothing.
- **Other fixes.**
  - Voice mode left on no longer re-reads the previous reply before every new one (#119).
  - DSML tool markup no longer leaks into DeepSeek V4 replies.
  - The chat window's minimum width is a bit wider.
  - Fix Prefill Guard. Adjusted OutOfMemory detector to be more accurate, and allow more of a ceiling.


## v26.7.12 — Deepseek V4 0731 + DSpark, Inkling Small support, Laguna 5x faster, Agents, media generation in chat

- **The big speed release.** poolside Laguna XS generates almost 5x faster than the last release, from 25 to 121 tokens per second on an M4 Max. Qwen3.6 35B jumps 20%, from 129 to 155 tokens per second, and reaches 237 with its speed helper on. Most other models pick up a little too. One piece of this trades a tiny amount of wording variation for speed on certain models; it is on by default and can be switched off in Settings ("Fast decode for bf16-attention models"). The full version-by-version history lives in benchmarks.md.
- **Inkling Small support.** Thinking Machines' 276B model (12B active) runs end to end: chat, thinking, tool calls and streaming. It's an unusual architecture. The pruned REAP-25 4-bit build (112 GB) fits a 128 GB Mac and our output matches the reference implementation token for token. It's also fast here: 47 tokens per second on an M4 Max, where the reference engine that ships with the checkpoint manages about 3 on the same machine. Text only for now, image and audio input come later.
  - On a 128 GB Mac, raise the GPU memory ceiling once per boot or the model barely fits: `sudo sysctl iogpu.wired_limit_mb=120000`. That took decode from 29 to 47 tokens per second and usable context from 1.5K to 17K tokens in our runs. The setting resets on reboot. Close all your apps/browsers, it can crash your Mac if you OOM.
  - Tool calling auto-corrected
- **DeepSeek V4 Flash runs on our own engine.** The 284B model (13B active, 1M context) is now a full native port instead of going through the embedded GGUF engine: chat, thinking, tool calls and streaming, no Python. Our mixed 2/3/8-bit conversion is 117.8 GB and lives on Hugging Face. On an M4 Max it decodes at 22.6 tokens per second, or 35.3 with DSpark on (below), against 26.7 for the GGUF engine it replaces. It needs the 0731 release of the checkpoint; the earlier preview is turned away at load with a message telling you which build to get.
  - The tool format the model was trained on (DSML) is parsed and repaired like every other family, and its chat template is byte-for-byte identical to the one shipped with the model, checked on every test run.
  - `reasoning_effort` reaches the model's own low/high/max setting rather than just picking a thinking budget.
- **DSpark, DeepSeek's own way of drafting ahead.** The checkpoint ships three draft stages that guess a whole block of tokens in one pass, and the server now uses them: 1.56x faster decode. Every token is still chosen by the full model, the draft stages only propose. It is opt-in because the stages want about 11 GB on top of the model, so turn it on in Settings under Speculative Decoding, or with `--dspark`. On a 128 GB Mac, raise the GPU ceiling first or it will decline to load them: `sudo sysctl iogpu.wired_limit_mb=124000`. It extremely tight on 128 with our model, not very useable.
- **Agents.** Create named assistants with their own personality, voice, model, tools and wake phrase. The app writes the persona prompt for you, and every way of starting a conversation can run as that agent: chat, voice, tasks, Telegram, Quick Launcher.
- **Kokoro voices.** 54 built-in voices that run fully on your Mac, about 17x faster than realtime, in a 330 MB download. Voices can be blended together. Qwen3-TTS remains the one that can clone your own voice.
- **Media generation in chat.** Ask for an image, a spoken line, a music track or a short clip right in the conversation and it appears inline with a progress bar. The bigger FLUX.2-klein 9B image model is also in the model list now (10 GB), with the same fast generation and picture editing as the small one and better prompt following.
- **A nicer chat window.** Code blocks get colors, line numbers and a copy button. There is a model picker in the toolbar, web sources on answers, failed turns show as a tidy card instead of raw errors, and tools and MCP servers flip on and off with one click.
- **Speed helpers set themselves up.** Download a dense Gemma 4 model and its speed-up companion model now comes along and just works, 27 to 40% faster on code and agent tasks. Nothing to pair up by hand.
- **Faster model downloads.** Up to 16 connections per file, and model search stops getting rate limited when you have a Hugging Face token set.
- **KV cache quantization is now fast.** With `--kv-quant 4` or `8`, decode now reads the compressed cache in place instead of unpacking it every token: 10% faster at 10K context and 56% faster at 42K on Laguna XS, 26% faster on Qwen3.6 27B at 37K. 
- **Fixed runaway memory on long chats.** A long session could quietly grow the server to tens of GB until you quit it (#110). Fixed, and the memory panel now shows exactly where the memory goes.
- **Fixed thinking for some models.** Laguna needed a bit special handling for thinking tags, and potentially other models based on their chat template.
---

## v26.7.11 — Mage-Flow image editing, a built-in console, faster MoE decode

- **Mage-Flow runs natively.** Microsoft's Mage-Flow Turbo generates images in 4 steps, and Mage-Flow Edit Turbo edits them from reference images: no masks, no fine-tuning, just the picture and what you want changed. Point it at several references and it composes them ("put the object from image 2 into image 1"). Full native port like everything else here, no Python. 8-bit conversions are on Hugging Face at 8.5 GB for generation and 9.1 GB for editing, against 16 GB for the originals.
- **Image editing in the app.** The Image tab takes a source image now, tells you the edit comes back at your picture's own size, and hides the knobs a distilled model ignores (Mage-Flow is fixed at 4 steps, so raising it only costs you time).
- **Edit images with the OpenAI SDK.** `client.images.edit(image=..., prompt=...)` now works against mlx-serve, including repeated `image[]` for multi-reference edits. It reaches the same engine as everything else, so it works on any edit-capable model you have loaded (FLUX.2 or Mage-Flow Edit). Anything OpenAI accepts that we can't honor gets a named 400 rather than being quietly ignored: masks, `n` > 1, URL response format, non-PNG output, streaming.
- **The server has a real console now.** Open its address in a browser (http://localhost:11234 by default) and you get a working chat against any model on the box, with history in the sidebar, a Monitor page showing what's loaded and the live metrics, and the full API reference. You can also just ask for things: "generate an image of a fox", attach a photo and say "make it winter", "write me a lo-fi track". It picks the right model for the job and shows the result inline, and it can answer questions about the server's own API.
- **MoE coding models decode a lot faster.** A new in-place gather kernel for MoE decode reads the expert bank where it sits instead of paying for its size on every token. Laguna at 4-bit went from 37.0 to 55.5 tok/s on an M4 Max, and 2-bit is up 24%, both against the previous release.
- **Bug fixes.**
  - Image edits keep the source photo's shape instead of squashing it.
  - Image generation no longer leaks memory (about 2.2 GB per megapixel).
  - Qwen3-VL image preprocessing is correct, and image chats keep speculative decoding (#102).
  - `/v1/images/edits` uses the model named in the form, not the default one.
  - Unknown endpoints return 404 instead of "no model configured".
  - A mistyped flag like `--model=/path` exits instead of being ignored.
  - F16 checkpoints no longer crash on long prompts.
  - `--pld` and its tuning flags are honored in headless mode (#95).
  - `--ssm-checkpoint-stride` and `--ssm-checkpoint-max` are documented in `--help` (#96).
  - Multipart uploads parse in any field order.
  - Debug logging no longer dumps raw image bytes into the log file.
  - Published model conversions are labelled quantized on Hugging Face, not fine-tunes.

---

## v26.7.10 — Insanely fast, LAN Sharing, Sandbox Pi & Hermes with 1 click, Laguna support

- **LAN model sharing.** Turn it on and every Mac on your network can use the models this Mac hosts, chat and image / speech / music / video / 3D generation alike. Macs find each other over Bonjour, shared models show up in every picker as "model · peer", requests stream to the Mac that has the model. Off by default, you pick what to share, and only inference is exposed (management & metrics stay private).
- **Run coding agents inside the sandbox.** The Sandbox window now has a real terminal, so you can run pi, hermes or a plain shell fully inside the isolated Linux VM talking to your local model, nothing runs on your Mac. First run installs the CLI in the guest, and there's a copyable ssh command if you'd rather use your own terminal app. The sandbox image is now pinned (the custom base-image setting is gone), which also fixes upgrades being stuck on the old pre-ssh image with an "image out of date" popup that re-pull and reset couldn't clear. You can also switch models mid-session with `/model`, and sandboxed agents can now use a LAN-shared model too. MCP servers work inside the sandbox now as well; the guest agent they connect through was missing from released builds, so they could never start before. (#89)
- **Laguna support.** poolside's Laguna S 2.1, a 117.6B-A8.5B MoE coding model, runs end to end: chat, thinking, tool calls, and long context.
- **oQ checkpoints get native MTP too.** mlx-serve now reads MTP heads straight out of oMLX's own oQ-format checkpoints, no sidecar conversion needed. Also fixed a head-norm bug that was quietly tanking the acceptance rate on these. Net result: mlx-serve now beats oMLX's own MTP on oMLX's own checkpoint.
- **M5 neural accelerator support.** We now build MLX ourselves with Apple's NAX kernels enabled (the stock package ships them disabled) and added our own NAX kernels for speculative verification, 1.2-2.2x faster than the stock path, drafting up to depth 8 on M5. M1-M4 Macs keep their existing tuned path. Prefill also got noticeably faster across the board from new fused attention and quantized matmul kernels, independent of the M5 work. Heads up: all builds now require macOS 26.2+.
- **EmbeddingGemma support.** Google's EmbeddingGemma models now work on `/v1/embeddings`, validated at 0.98+ parity vs the reference implementation. (#79)
- **Speculative decoding fixes.** Fixed a Gemma BF16 failure with the assistant drafter (#84), a crash loading mxfp8-quantized MTP heads (#81), and cross-request acceptance seeding being stuck off.
- **Ready for macOS 27.** Builds and runs on the macOS 27 beta (moved to a pinned Zig 0.17 nightly, the build stages it for you), released binaries still run on macOS 26.2+.
- **Closer to the OpenAI spec.** `reasoning_effort` now turns thinking on for chat completions (any effort except "none", budget scales with the level), so standard clients don't need the `enable_thinking` extension anymore. `parallel_tool_calls: false` is now honored with at most one tool call per response (the OpenAI SDK sends it in strict structured-output mode), same for Anthropic's `disable_parallel_tool_use`. Asking for `n` > 1 choices now gets a clear 400 instead of silently returning one choice. The Anthropic API now reports which stop sequence fired (`stop_reason: "stop_sequence"` plus the matched string, non-streaming & streaming). Chat completions usage now reports prompt cache hits in `prompt_tokens_details.cached_tokens` (the cache was always there, it just wasn't visible in the standard field, so cost dashboards and conformance tools read it as "no caching"). Also fixed a small per-request memory leak on responses with no visible content.
- **A few small additions.** Dictation in the Voice Clone tab, and a community model tier list on the website.
- **Bug fixes.** A KV cache crash, a clearer error when a model's quantization isn't supported, a skill-trigger matching bug (#92), and ds4 bumped to its latest upstream for another speed bump.

---

## v26.7.9 — DeepSeek drafts ahead, and sturdier tool calls

- **DeepSeek V4 Flash learns to draft ahead.** The 284B flagship now uses its published draft head for speculative decoding: the app downloads the small companion file automatically beside the model, and the server drafts and verifies several tokens per pass with identical output. On by default; `--no-ds4-mtp` turns it off.
- **The community builds of Hunyuan 3 now load.** Popular conversions published with a different internal weight layout — including the smaller pruned variants that fit more comfortably on a 128 GB Mac — used to fail at startup with a "missing weight" error. The server now handles both layouts; measured on an M4 Max, both the full 2-bit build and a 4-bit pruned one decode at ~26–28 tok/s.
- **Giant split GGUFs are now first-class.** Quants that Hugging Face ships as multiple shard files — like Hunyuan 3's ~89 GB 1-bit build — now download as one unit, show their real size, and load as a single model. Also fixed: models cached by other Hugging Face tools could show up as 0 MB and vanish from the picker.
- **Agents survive sloppier tool calls.** A six-hour agent soak caught three more ways a small or heavily-pruned model can mangle a tool call — dropped separators, mismatched closing tags, and a stray close marker leaking into the visible reply. All three are now repaired into clean calls (or stripped) instead of derailing the agent into retry loops. The full benchmark matrix was re-run afterwards: no speed regressions anywhere.
- **No more false "not enough memory" rejections on DeepSeek and GGUF models.** Requests to the embedded engines were being screened by a memory estimate built for the MLX engine, so a prompt the same server had just handled fine on another model could bounce with a bogus "requires ~25 GB of GPU memory" error.
- **Know exactly what you're running.** `mlx-serve --version` — and the app's Settings — now report the version of every embedded engine (MLX, llama.cpp, ds4), and the menu-bar tray gains a one-click copy button for full model names.
- **A new front door, and an iPhone in the family.** The website home page is redesigned around what the app does for you — with a friendly getting-started guide for people new to local AI — and the same open-source engine now powers **MLX Chat — Local AI** on the iPhone App Store: chat, voice cloning, image and music generation, fully on-device.

---

## v26.7.8 — Hunyuan 3: a 295B flagship on your Mac

- **Tencent's Hunyuan 3 (295B-A21B) runs natively.** The strongest open model mlx-serve has ever served: `mlx-serve run hy3` pulls the 2-bit mixed-precision build (~105 GB) and serves it at ~26 tok/s decode with ~235 tok/s prefill on an M4 Max — thinking, tool calling, streaming, and all four API surfaces included. Recommended for Macs with more than 128 GB of memory; on a 128 GB Mac it runs with a minimal context window. Its native multi-token-prediction head works too: opt in per request with `enable_mtp: true` (+11% on code-edit workloads with `--mtp-depth 1`).
- **Checkpoints with float16 quantization scales now load at full speed.** Mixed-precision conversions that store scales as f16 beside bf16 compute silently hit a 4× slower matmul path — on the 295B that was the difference between 1.2 and 23 tok/s. The server now normalizes them once at load, with no measurable quality change.
- **Hy3 GGUFs work too — including Tencent's official 1-bit build.** The embedded llama.cpp engine is updated to b9999, which carries the freshly-merged hy_v3 support, so AngelSlim's IQ1_M (~89 GB) and the community IQ2/Q4 quants load through the same auto-routing every other GGUF uses.
- **Streaming reasoning can no longer split a character in half.** Multibyte characters (², °, emoji) that straddle two tokens used to occasionally ship as broken bytes in `reasoning_content` deltas, tripping strict SSE clients; reasoning streams are now always valid UTF-8, across both the OpenAI and Anthropic surfaces.
- **A cut-off tool call can never ship half a file anymore.** When a tool call is truncated mid-argument — the model bails, hits a limit, or gets cut by the repetition guard — the server now recovers the tool name and the arguments that fully arrived, and drops the fragment. Previously a Gemma-format write call cut inside its content could hand an agent a partial page as if it were the real thing; with the file path already present, the agent would have written a corrupt file and reported success.
- **Runaway repetition loops now report what actually happened.** When the server cuts a model stuck repeating the same phrase, the request finishes as an honest truncation (`finish_reason: "length"`) and logs the cut — so agent clients fire their retry-with-smaller-chunks recovery instead of validating a garbage call, and a post-mortem no longer needs log archaeology to see the guard fired.
- **Mixture-of-experts models can finally use their MTP head.** MoE checkpoints that ship a multi-token-prediction head kept it switched off for every client that doesn't ask for it by name — which is all of them (Claude Code, curl, benchmark tools). The new `--mtp` flag, and a matching Settings toggle, turns it on: on the Qwen3.6 35B-A3B distill that is a measured 2.1× on predictable, echo-heavy work like code edits and agent loops. It stays off by default because the same head is worth almost nothing on novel prose — so measure your checkpoint before flipping it.
- **`/v1/models` tells the truth about models it hasn't loaded yet.** Multimodal checkpoints — most Gemma 3/4 and Qwen-VL builds — advertised a context window of 0, no dimensions, and "not a mixture-of-experts model" until they were loaded, so a client couldn't size a request or even tell a 128-expert model from a dense one without first pulling 16 GB of weights into memory.

---

## v26.7.7 — The fastest MTP runtime on Apple Silicon

- **New verify-tuned Metal kernels make speculative decoding decisively faster.** Quantized matmuls in the 2–7-row shapes that speculative verification actually uses, now run through custom split-K kernels held near the memory-bandwidth floor. On Qwen3.6-27B the verify round dropped from 76 ms to 54 ms at depth 3 and from 119 ms to 90 ms at depth 6 — and the same kernels accelerate PLD and drafter verification on every 4-bit model, Gemma included, with byte-identical output pinned by tests.
- **Head-to-head vs the reference MTP runtime on its own published checkpoint: faster warm AND 2× faster cold.** Same machine, same prompts, same weights, back-to-back runs: warm decode 75–79 tok/s vs 69–74, and first-request decode 70 vs 35 tok/s. Every claim measured in paired same-session runs.
- **Dynamic draft depth on Qwen MTP models.** The speculative controller plans every round from live per-position acceptance estimates — drafting deeper on easy stretches and pulling back instantly on hard ones — and now remembers what it learned across requests, so a warm server hits full speed from the first tokens of a new request instead of re-calibrating each time. Creative writing still falls back to plain decoding cleanly when drafting stops paying.
- **Agent chats no longer lose their MTP speedup.** An old routing rule sent repetitive-looking prompts (agent histories, tool results) to prompt-lookahead drafting instead of the MTP head; when lookahead acceptance then collapsed mid-request, decoding fell all the way back to plain speed — measured live at 28 tok/s on a session the MTP head decodes at 55–75. The rule is retired: the MTP head now runs whenever it's loaded.
- **Agents stop burning turns on misplaced tool arguments.** When a model puts a required argument in the wrong place — burying the file path *inside* the edit object instead of beside it — the server now puts it back where the tool's own schema says it belongs, so the call lands on the first try. One live agent session lost three full multi-thousand-token generations to exactly this before the model gave up on editing and rewrote the entire file from scratch. Correct calls are passed through untouched, and `--no-tool-autocorrect` turns it off.
- **Tighter OpenAI API compliance.** `/v1/responses` streams now end with the standard `data: [DONE]` sentinel that proxies and SSE middleware key stream-end off; `/v1/embeddings` honors the `dimensions` parameter (truncate + renormalize, OpenAI text-embedding-3 semantics) instead of silently returning full-width vectors; and unsupported `background` requests get a clean error instead of quietly running synchronously.
- **New settings panel.** Rewrote the settings pannel to be more UX friendly, now has a sidebar to filter down options.
- **Fix server crash** If you sent some un-expected payloads on some endpoints it would crash the server.
- **Support for multiple different GGUF quant** Fixes issue #76, once you downloaded a GGUF quant you were stuck with that, now you can download & select others also.
- **Swift window focus issue** Sometimes the windows would not properly focus on click, and put them in a weird state, now fixed.
---

## v26.7.6 — Long contexts that actually fit, and fly

- **Gemma long-context prefill more than doubled.** A custom flash-attention Metal kernel now handles Gemma's sliding-window attention during prefill, skipping everything outside each layer's attention window instead of computing and masking it — and a second round of kernel tuning (wider tiles, half the memory traffic per query row) added another ~10% on top. Measured on gemma-4-26B-A4B with a ~100K-token prompt: 299 → 715 tok/s prefill (2.4×) at identical peak memory and byte-identical output; Gemma E4B prefill up 4% too.
- **Multi-token prediction rebuilt — three drafts per round by default, and long-context decode is up ~50%.** Speculative rounds no longer copy gigabytes of KV cache per round at long context, a rejected draft costs milliseconds instead of a full re-forward, and drafts project through a compact 3-bit head. On Qwen3.6-27B: code generation +25%, coding-agent sessions +15–26%, and 64K-context decode 21.7 → 33.0 tok/s vs the previous release — while creative writing holds steady, with the adaptive controller backing off automatically on hard content. In a fresh head-to-head against the reference MTP runtime on the identical checkpoint and prompts, mlx-serve wins all 8 decode speeds (by 11–30%) AND all 8 prefill speeds, from 0.5K to 64K.
- **Qwen3.6-35B-A3B MoE models can now use their MTP head.** Sidecars with mixture-of-experts drafting layers (and mixed per-tensor quantization) load and draft correctly — including artifacts published for other MTP runtimes, loaded unmodified. Opt in per request with `enable_mtp: true`; measured 73% per-draft acceptance with byte-identical greedy output.
- **Qwen long-prompt prefill: faster and ~9 GB lighter.** Prompts on Qwen 3.5/3.6 models now prefill in chunks tuned to the architecture: an 8K-token prompt on the 27B runs ~5% faster with peak process memory down from 29 GB to 20 GB — while decoding 1.5–1.9× faster via the native MTP head.
- **Your launch settings now stick everywhere.** Models that load on demand (multi-model serving, generation-first launches) previously ignored parts of the server configuration: MTP and embedded-llama.cpp flags on cold loads, `--kv-quant` and the hot prefix cache in headless mode. All of them now apply no matter how a model gets loaded.
- **Stability fixes from a long agentic soak.** Fixed a crash on Gemma 26B MoE during long multi-turn agent sessions (a KV-cache restore could drift out of alignment around 16K context) and unbounded memory growth on DiffusionGemma with long prompts (a 14 GB model could balloon past 90 GB of process footprint; now bounded).
- **Beginner-friendly model onboarding.** The Model Browser gained a Recommended pane — curated Gemma 4 and Qwen 3.5/3.6 picks explained in plain English (what each is good at, the trade-offs, and the real on-disk size) — and a fresh install now walks you from the welcome screen to your first model download instead of dropping you into an empty chat.
- **MLX Core app improvements.** Customizable voice wake word (make it "Hey Jarvis"), the Claude Code / pi / opencode launchers no longer overwrite your existing CLI settings, the agent gained built-in file-search, archive, and system-info tools instead of shelling out for them, and assorted chat, model-browser, and HuggingFace-search UI fixes.
- **Re-benchmarked against current LM Studio: +48% geomean on identical MLX weights.** The comparison matrix was re-run against LM Studio 0.4.15 across six models (Gemma 4 E2B/E4B/31B/26B-A4B, Qwen 3.6 27B/35B-A3B) — up from +35% at the last measurement, driven by speculative decoding, faster prefill, and the new native-MTP cells. README charts and tables refreshed, including a new MTP context-ladder chart.
- **Long prompts no longer OOM-crash the server, and the memory check tells the truth.** On every Gemma-4 and Qwen 3.5/3.6 model, a long prompt could kill the whole server with a Metal out-of-memory abort (a 255K prompt died around 100K) because prefill scratch memory silently grew with prompt length. Prefill now bounds that scratch automatically — a 102K-token prompt's peak dropped from 51.6 GB to 27.0 GB while getting 14% *faster* — and the admission check now accounts for KV-cache quantization (including the per-request `kv_quant` override), so quantized long-context requests that comfortably fit are no longer spuriously rejected with a "requires ~100 GB" error.

---

## v26.7.5 — Tool calls that don't fight your agent

- **Local agents stop looping on tool calls they already got right.** Point a coding agent (Claude Code, pi, opencode) at a local model and weaker or reasoning-distilled models routinely mistype a tool argument — sending Python's `False` where JSON wants `false`, or a list of edits as a quoted string — so strict clients reject the call with "expected boolean, provided string." The model can't see its own serialized request, so it "fixes" a value that was already correct and burns turn after turn (one captured session failed six times in a row, then gave up on editing and rewrote whole files). The server now reads the tool's own schema and corrects the argument's type before the client ever sees it, so the call goes through the first time.
- **Malformed tool calls from small models are recovered instead of dropped.** A 1–4B model writing a large file in one shot mangles its own JSON a dozen ways — a dropped opening tag, a repeated parameter, an invalid escape in a Windows path. Any of these used to drop the entire call (the file leaked into the chat as text) or ship broken JSON the client couldn't parse at all. The server now repairs these at the source and guarantees every tool call it emits is well-formed JSON — recovering the call where it can, never handing a client garbage.
- **No more stray thinking markers in replies.** Large reasoning models under load occasionally spray their internal channel markers into the answer; those could surface in the visible reply. They're now stripped before anything reaches you.
- **An off switch, if you want the model's raw output.** A new *Tool-call auto-correct* toggle in Settings (and `--no-tool-autocorrect` on the command line) disables the type coercion and passes arguments through exactly as the model wrote them — for debugging a model, or if you'd rather see the unaltered call. On by default; the safety net that keeps output well-formed stays on regardless.
- **Shaken out by an eight-hour soak.** All of the above was found by driving Claude Code, pi, and opencode through heavy tool-call variations — with thinking on and off — against every supported model family (Qwen 3.5/3.6, Gemma 3/4, the GGUF and DeepSeek-V4 engines, and block-diffusion), replaying a growing corpus of real captured traffic on every build. Ten distinct issues fixed, zero regressions.

---

## v26.7.4 — Agent sessions that survive, and a Model Browser that makes sense

- **Long agent turns no longer die at five minutes.** When a model spends minutes writing a large file into one tool call, the server buffers every token and sends nothing — so Node-based agents (pi, opencode, anything on `fetch`) hit their 300-second idle timeout, killed the connection, and threw the work away. Every streaming surface now keeps the socket alive while it thinks. Measured: a ten-minute generation that previously died after 301 seconds having delivered a single chunk now streams to completion, with a five-second worst-case gap between bytes.
- **Agent CLIs finally see your real context window.** pi, opencode, and Claude Code were launched with a hardcoded 32K context and 8K output budget no matter which model was serving — so a session on a 92K-context model watched its own budget collapse and began asking for a single output token. Their configs are now written from the context the server actually advertises, and that number stops drifting: it's pinned when the model loads instead of being recomputed from free memory on every request (it wandered between 92,387 and 94,883 in one measured session). A client that omits `max_tokens` now gets the remaining context instead of a silent 4,096-token cap that truncated large tool calls. Settings ▸ Context size also stops conflating the three numbers it shows — the model's architectural maximum, what this Mac's memory could hold, and what the server has actually pinned and hands to agent CLIs.
- **The Model Browser is rebuilt around what you already have.** A model you finished downloading used to disappear from the search results at the exact moment it succeeded. Now it stays listed, marked, with a one-click **Use** that loads it, and an "In use" badge on whichever model the server is really serving. The pane is organized as Discover / My Models / Downloads / Drafters instead of one list behind a "Downloaded" toggle that quietly swapped the data underneath you. My Models lists everything the app can load — including models discovered from LM Studio and your custom folder, grouped by source — rather than only what we fetched ourselves. Downloads is its own destination with a live badge, so a transfer in progress is visible from anywhere. And the Model Browser is now one click from the menu bar, without expanding the download list first.
- **Find any setting by typing.** Settings gains a filter field: type "prefix cache" or "api key" and everything else folds away, matching on both the setting's name and its description. Searching a section by name — "telegram", "voice" — opens that whole section.
- **Server logs survive a crash.** Every serving session now writes to `~/.mlx-serve/logs/mlx-serve-<port>.log` — 32 MB, rotating, one file per port, `--log-file` to relocate or disable. Until now the only history lived in the app's in-memory buffer and died with the process, which is precisely when a post-mortem needs it. The in-app Server Log view holds 16× more history too (1 MB instead of 64 KB), so a model-load dump no longer scrolls an entire agent session out of view.

---

## v26.7.3 — Edit with reference images, metrics, auth, and restart-survivable chats

- **Multi-reference image editing**: instruction edits on FLUX.2 Klein can now see extra reference pictures. Add up to 3 reference images beside the source in the Image pane's edit mode and refer to them by number — "replace the face of the man in image 1 with the face from image 2" — . API users: `ref_images` (base64 array) beside `image` with `mode:"edit"` on `/v1/images/generations`.
- **Long conversations survive server restarts — no more re-reading from scratch.** The prefix cache gains an SSD tier: long prompts the server has already processed are persisted to disk in chunks and restored instead of recomputed, across model switches, app relaunches, and reboots. Re-opening a long chat that used to sit through a 40-second re-read of the whole history now answers in under 3 seconds.
- **Fixed: some models froze for 10+ seconds before answering long prompts.** Tokenizers that ship thousands of special tokens hit a quadratic scan on every uncached prompt — ~12 seconds of pure CPU before the GPU even started on a 30 KB prompt, and again on every follow-up turn.
- **Switching models no longer silently degrades the response cache.** Models loaded on demand (model switches, `/v1/load-model`) used to get a minimal single-slot prefix cache regardless of your settings; they now inherit the full configured cache — including warm multi-turn reuse on Qwen 3.5/3.6-class hybrid models.
- **Watch your server live, right on its homepage.** Start with `--metrics` (or flip *Metrics panel* on in Settings) and the server's index page grows a real-time dashboard: decode and prefill tokens/sec with hover-readable sparklines, requests in flight, time-to-first-token, prefix-cache hit rate, GPU utilization and memory — updating as you generate. The same figures are exported at a standard Prometheus `/metrics` endpoint under vLLM-compatible names, so existing Grafana dashboards work with zero configuration. Fully opt-in, with no measurable effect on tokens/sec.
- **Optional API key for network deployments.** Exposing the server on your network? Set `--api-key <key>` (or the field in Settings) and every request from another machine — the OpenAI, Anthropic, and Ollama APIs plus the metrics page — must present it (Authorization Bearer, `x-api-key`, HTTP Basic, or `?api_key=`). Your own Mac stays trusted and key-free, so the app and local tools are unaffected; the key guards only what's reachable off the box.

---

## v26.7.2 — Type a vibe, get a song. Turn a photo into a 3D model. And the app updates itself

- **Type a vibe, get a song — music generation lands.** The Audio pane grows a second tab: describe a style ("upbeat synthwave with driving bass"), optionally paste lyrics, pick a length from 10 seconds to 10 minutes, and ACE-Step 1.5 XL Turbo — a 4-billion-parameter music diffusion model ported natively to Apple Silicon — composes an original 48 kHz stereo track in just 8 diffusion steps, entirely on-device inside the same no-Python binary. BPM, key, and time signature are steerable; instrumental or vocal. The existing text-to-speech pane lives on as the Voice tab, and both tabs gain a persistent history list — every track and voice clip you've ever generated stays one click away (play, stop, reveal in Finder), and starting a new generation stops whatever is still playing. The Music tab ships genre style-prompt starters and original lyric templates, and you can save your own style prompts and lyrics to reuse from the Examples menus. Every generation drops a matching `.txt` next to the audio with the exact prompt, lyrics, and settings used, so any track is reproducible. `POST /v1/audio/music-generations` for API users.
- **Photo → 3D model, fully on-device.** A new 3D pane in the menu-bar tray turns a single photo into a 3D mesh using Hunyuan3D-2.1, ported natively to Apple Silicon — the diffusion shape model, SDF decoding, marching cubes, and a glTF writer all live inside the one no-Python binary. Drop in a picture, the subject is cut out automatically, and the finished model spins in a built-in 3D viewer with a gentle turntable idle; the GLB file opens anywhere glTF does. One click downloads the whole 3D stack (shape + texture in a single package); `POST /v1/3d/generations` for API users.
- **Full PBR texturing for 3D models — the photo now paints the mesh.** Turn on "Texture (PBR)" in the 3D pane and the same photo that shaped the model now paints it: a native port of Hunyuan3D-2.1's multiview paint stage generates albedo AND metallic-roughness maps across six views, bakes them into a 2K texture atlas, and ships a standard glTF PBR model that renders correctly in any viewer. The entire stack — UV unwrapping, a 2-billion-parameter multiview diffusion UNet, differentiable-renderer-grade baking, and texture inpainting — runs inside the same no-Python binary, validated against the PyTorch reference at cosine 1.000 on the full denoiser step. Included in the one-click 3D download; `"texture": true` on `/v1/3d/generations` for API users.
- **3D meshes extract 6× faster, at a finer default.** The surface-extraction stage now samples the field coarse-to-fine instead of sweeping every point in the volume — measured 158s vs 952s at the highest mesh resolution, with byte-identical geometry. That win funds bumping the default mesh resolution from "balanced" (256) to the reference's "fine" (384), so models come out noticeably more detailed AND faster than before. A generation-history shelf with clickable thumbnails also lands under the 3D preview — every past model one click away.
- **Your voice, cloned once, spoken everywhere.** Record or pick a few seconds of your voice under Settings → Voice, and the hands-free voice assistant answers in it — every spoken reply is synthesized locally by Qwen3-TTS from your clip, sentence-by-sentence while the model is still thinking. No clip set (or TTS unavailable)? Answers fall back to the macOS system voice, so voice mode never goes silent. The voice picker in the tray now treats your clone as a first-class voice: it shows your clip by name when it's the voice that's actually speaking (no more misleading "Jamie"), lets you pick a new audio file to clone right from the menu, switches between your voice and any Apple voice with one click — and tells you when the Qwen3-TTS model still needs downloading from the Audio tile.
- **Voice mode works in noisy rooms — and the mic permission asks at the right time.** The assistant now tracks your room's ambient noise level and detects the end of your sentence relative to it, so fans, AC, or a humming GPU no longer leave it listening forever without answering (a stalled-transcript backstop catches anything else). And the app no longer asks for microphone access at launch — the permission prompt appears when you actually enable voice mode. Playing a generated track or voice clip doesn't trigger a microphone prompt anymore either (a macOS 26 quirk where the standard playback API consults the mic permission on the way out).
- **Chat works even when the server was started for media generation.** Generating an image/video/3D model first boots the server without a chat model; typing into Chat then failed with "No default model configured" even though a model was selected. Chat surfaces now load your selected model on the spot, and the server adopts the first chat model it loads as the default for API clients — so media-first sessions flow straight into chat.
- **Automatic updates**: MLX Core now checks the project's GitHub releases page once a day and shows an update banner in the menu-bar tray when a new version ships. One click downloads the notarized installer, swaps the app in place, and relaunches — your models, chats, and settings are untouched. A manual "Check Now" button and an opt-out toggle live in Settings → Updates.
- **Fixed: deleting a chat now stops its generation.** Previously, deleting a chat while it was still answering left that generation running invisibly — the model stayed busy, every other chat reported "answering another chat", and even restarting the server couldn't clear it. Deleting the chat now cancels its turn on the spot.
- **Long answers no longer get cut off at 5 minutes.** The request timeout now measures stalls (no new tokens), not total time — a model that's actively writing can run as long as it needs, while a genuinely hung request still gets reaped. Previously a big agent file-write on a large model was silently guillotined mid-tool-call at 300 seconds, then retried from scratch: verified live, a 7¾-minute 50KB write now completes in one shot.
- **Sub-4-bit FLUX for small devices.** The native FLUX.2-klein image engine now loads any of the mlx-community 3/4/5/6/8-bit quantizations — each weight's precision is inferred from its stored geometry, no configuration needed. The 3-bit build cuts the download from ~5 GB to ~3.7 GB, and a new low-memory mode halves the resident footprint on top: the text encoder loads per request and is freed the moment the prompt is encoded, with byte-identical images and a measured cost of ~0.3 s per image. It's automatic on iPhone and on Macs with 16 GB or less; bigger machines keep everything resident.
- **The engine now runs on iPhone.** The whole no-Python engine — chat, streaming, and Qwen3-TTS voice cloning included — now cross-compiles to an iOS static library with the full Metal GPU backend, booting headless in-process and loading models on demand. It powers MLX Chat, a new minimal iPhone app for latest-generation iPhones (chat + on-device voice clone), developed in its own repository; the macOS product is byte-for-byte unchanged.
- **8-bit voice models are the new default.** Text-to-speech and voice cloning now run on the 8-bit Qwen3-TTS builds out of the box, on both Mac and iPhone: 20-30% smaller downloads (the 1.7B quality model drops from 4.5 GB to 3.1 GB) and a lighter memory footprint, with speech that tracks full precision nearly exactly — the codec and speaker encoder stay unquantized, so cloning fidelity is unchanged. The bf16 builds remain in the Mac picker as full-precision fallbacks.
- **Type `mlx-serve` in Terminal.** The welcome screen — which now greets you on every launch, not just the first — gains a one-click Install button that puts the `mlx-serve` command on your PATH. If you already have a `~/.local/bin` or `~/bin` on your PATH it links there with no password; otherwise it creates the standard `/usr/local/bin` link after a single admin prompt. Your shell config files are never touched.
- **⌘Tab now finds MLX Core whenever a window is open.** Previously the app only appeared in the app switcher after you'd opened the Chat window at least once — Audio, Video, 3D, the intro screen, and other windows left it invisible to ⌘Tab (and it never returned to menu-bar-only mode afterwards). The app now shows in ⌘Tab and the Dock exactly while any window is open — intro window included — and goes back to a clean menu-bar-only presence when the last one closes. And when a Dock-icon click has no window to restore, it opens the menu-bar tray instead of doing nothing.
- **Small polish across the app.** The model browser now shows MLX models by default (GGUF and Both stay one click away in the format picker), the tray's generation tiles read Image / Video / Audio / 3D, and the intro window doubles as a quick-start screen shown on every launch.
- **Pulled GGUF models now serve headlessly.** `mlx-serve pull` a GGUF repo from Hugging Face, then `mlx-serve serve` — the models show up and load on demand, exactly like MLX ones. GGUF repos ship no `config.json`, so the headless server used to report "Discovered 0 models" for them even though `mlx-serve list` showed them (issue #59); discovery now recognizes any folder of `.gguf` weights, loads it through the embedded llama.cpp (or DeepSeek) engine on first request, and unloads/reloads it like any other model. Works for Ollama clients too — pulled GGUFs appear in `/api/tags` and resolve by name.
- **One bad request can no longer crash the server.** A chat request aimed at an image, audio, video, 3D, or embedding model — from any client, local or remote — used to segfault the whole server; it now gets a clear 400 that names the model's kind and the endpoint that does serve it, without loading gigabytes of weights first. `mlx-serve run` applies the same sense check up front: pointing it at a non-chat model prints what the model is and the serve command to use instead of booting a REPL that could never answer.
- **`mlx-serve list` tells you what each model actually is.** A new TYPE column labels every entry — chat, image, audio, video, 3d, embed, drafter — so it's obvious which rows `run` can talk to, and sizes now include weights stored in subfolders (media bundles previously showed as a few KB). DiffusionGemma checkpoints are also discoverable by the headless server now, matching what `--model` could already load.
- **Agents recover from truncated tool calls instead of looping.** When a generation is cut off mid-tool-call (token cap), the truncation is now reported honestly to the client and the agent immediately switches to writing the file in chunks — before, the model was blamed for "forgetting" content it had actually written, and retried the same failing call for 15+ wasted minutes. The system prompt also stops advertising six-figure output budgets on big-memory machines — the very invitation that pushed models into those five-minute one-shot writes; the budget warning now appears only when the budget is actually tight.

---

## v26.7.1 — Edit photos, animate them, sandbox your agent, drop in for Ollama

- **Agent Sandbox, built on Apple's own virtualization** The isolated Linux VM that runs the agent's shell commands is now powered directly by Apple's Virtualization framework: it boots in under a second, and the same design is Mac App Store-compatible. The agent is also told which environment it's in — Linux sandbox or your Mac — so it stops reaching for `brew` inside the VM (and vice versa), a green shield in the chat toolbar shows when commands run isolated, and the `/workspace` mount follows your working-folder switch automatically. The working-folder chip now shows just the folder's name (full path in the tooltip).
- **Quick Launcher: ⌃Space, ask, done.** A new Spotlight-style prompt panel summons over any app — hit ⌃Space, type a question, and the answer streams in right there from your local model, no window shuffling. Follow-ups keep their context, ⌘↩ hands the conversation off to the full chat window, and Esc dismisses while the answer keeps generating into your chat sidebar. Opt in with the new toggle under Voice in the menu-bar tray; no permissions prompt, works from any Space or full-screen app.
- **Two-stage video quality is back, native and actually looks good now.** The Quality and Super-Quality video presets now run the full reference two-stage pipeline on the native engine: a guided half-resolution pass on the dev model (CFG + modality guidance, with the second-order res_2s sampler for Super-Quality), a learned 2× latent upscale, then a distilled refine at full resolution. 
- **Make your characters speak.** Put the spoken words in quotes in your video prompt — short phrases with acting directions between them — and LTX generates the voice, timed to the picture. A new "Talking character" example in the Video pane shows the format, and audio guidance on the Quality presets now steers harder toward clean speech: clearer voices, less stray background noise. Attach a real speech or music clip in the Video pane's new Speech & sound section — or type a line and have the local Qwen3-TTS voice speak it — and the video is generated *against* that soundtrack: voices, lip sync, and performance follow the clip, and the original audio (not a lossy re-synthesis) lands in the mp4. Any WAV/MP3/M4A works, the frame count auto-fits the clip length, and everything runs on-device through the same one-click LTX download (`audio` field on `/v1/video/generations` for API users).
- **Image-to-video: animate your own photo.** Drop a picture into the Video pane's First frame slot and the clip begins from it — the image is VAE-encoded and locked as the clean opening frame on your Mac, and the model animates forward from there. It works on the standard one-stage pipeline at any resolution, and if you don't attach an image (or haven't downloaded the encoder) it simply generates from the prompt as before.
- **Edit your own photos with instructions.** Attach a picture in the Image pane, type what should change — "make the hair blue", "remove the monitor in the background" — and FLUX.2-klein edits the image while keeping the subject, pose, and scene intact: the source rides through the model as a clean in-context reference (the mechanism klein was trained on), not a noisy remix. Your photo keeps its proportions too — the reference is passed to the model at its own aspect ratio, so a portrait or landscape source is recomposed into the output size instead of being squished. Verified live: a "make the fox blue" edit kept 97% structural correlation with the original photo. Runs fully on-device (`mode:"edit"` + `image` on `/v1/images/generations`).
- **Image-to-image variations too.** The same source-image slot also offers a Variation mode on every image model (including Krea-2-Turbo): the picture is VAE-encoded and partially renoised, with a strength slider from subtle remix to full re-imagination — sources with a different shape than the output are center-cropped, never stretched. The needed encoders ship inside the model downloads you already have (both ports validated by encode→decode round-trips at pixel correlation 0.999+), so there's nothing extra to fetch (`image` + `strength` for API users).
- **Style LoRAs for image & video models.** Attach any diffusers-format LoRA `.safetensors` under the Image & Video pane's Advanced options to restyle LTX, FLUX or Krea generations. Adapters apply at runtime — no re-quantization, zero quality loss on the base weights — and detach cleanly between requests (`lora_path` / `lora_scale` on the API).
- **Conditioning rebalance (Advanced).** A new power-user control reweights how the prompt drives the image: a global conditioning gain plus per-text-encoder-layer weights — 12 numbers for Krea's stacked encoder, 3 for FLUX's — typed comma- or space-separated, with live count validation.
- **Video generation is about 2× faster.** The one-stage LTX path now runs without classifier-free guidance by default — the setting it's actually designed for — which halves the work per step (one model pass instead of two) and tends to give a more natural, less over-saturated look. Want the punchier, higher-contrast style? Pass a guidance scale per request to turn it back on.
- **Drop-in Ollama replacement.** mlx-serve now speaks the Ollama wire protocol (`/api/chat`, `/api/generate`, `/api/tags`, `/api/embed`, `/api/show`, `/api/ps`, `/api/pull`) alongside its OpenAI and Anthropic APIs — point Raycast, Obsidian, Enchanted, Open WebUI, ollama-python/js, or anything else that expects Ollama at your mlx-serve port and it just works: streaming, tool calling, thinking, images, JSON-schema formats, and tagged model names like `qwen3.6:latest` all translate natively. Same GGUF or MLX weights, the faster engine underneath.
- **Improve command line: `mlx-serve run gemma4`.** One command downloads the model (resumable, straight from Hugging Face), starts the server, and drops you into a streaming chat REPL with live tok/s. `mlx-serve pull` and `mlx-serve list` round it out — short names like `qwen3.6:27b`, `gemma4:12b`, or any Hugging Face `org/repo` work everywhere, and `mlx-serve serve` exposes everything you've pulled for on-demand loading by name (models stored in `org/repo` folders are now discovered too, listed under that full name).


---

## v26.6.13 — Create images, voices, and video locally, all Zig Native

- **Image generation.** Generate images from a text prompt right on your Mac — pick **FLUX.2** for fast results or **Krea-2-Turbo**, a 12.9B photorealistic model, then type a prompt and get a PNG with a live progress bar as it denoises. The whole pipeline (text encoder, diffusion transformer, VAE) runs natively on Apple Silicon: no venv, no setup step. Krea is a one-click ~15 GB download and was validated numerically faithful to the reference (end-to-end pixel cosine 0.9996); any size from 256² to 2048² works.
- **On-device safety filter for images.** Every generated image is screened by an NSFW classifier that runs natively on your Mac — nothing is uploaded anywhere — and explicit results are blocked before they reach you. On by default, with a Safe-mode toggle in the Image tab (and a `--no-safety` server flag) to turn it off.
- **Text-to-speech with zero-shot voice cloning.** Type text and hear it spoken by Qwen3-TTS — and record or pick a few seconds of any voice to have the model speak your text *in that voice*. Cloning runs entirely on device (validated bit-for-bit against the reference) and needs only the reference audio — no transcript.
- **Text-to-video with audio.** Turn a prompt into a short LTX-Video 2.3 clip with synchronized audio, muxed straight to an mp4 — the full diffusion + 3D-VAE pipeline ported natively and validated tensor-by-tensor against the reference.
- **One app, one server, one memory budget.** Chat and every media type now share a single local server instead of separate background processes. A model loads on demand when you generate and unloads when it's done to free GPU memory — flip "Keep loaded" for instant repeat runs — and a chat model and a media model can stay resident together without stepping on each other.
- **Download media models right where you use them.** When a model you pick isn't on disk, the generation pane offers a one-click download with progress and only enables Generate once it's ready. Downloads pull just the files the engine actually reads — LTX grabs ~26 GB (model + its text encoder) instead of the repo's ~70 GB of unused weights — and that LTX text encoder doubles as a selectable chat model.
- **Live progress everywhere.** Image, audio, and video all stream per-step progress as they generate, so you watch the work happen instead of staring at a spinner.
- **Generate images right in chat.** In Agent mode, just ask for an image — "draw a red fox in the snow" — and it renders inline in the conversation using your saved Image settings (model, quality, resolution, seed, safe mode), no need to leave chat or restate the model. Double-click any image in a chat to open it full-size in Preview. (Audio and video generation stay in their tray windows for this release.)
- **Your generation settings stick.** The Image, Audio, and Video panels now remember your last-used model, quality, resolution, steps, seed, and toggles — between opening the window and across app restarts — so you stop re-picking the same setup every time.

---

## v26.6.12 — Big writes finish, agents run servers

- **Your Qwen models can see now.** Qwen 3.5 and 3.6 vision checkpoints read images out of the box — attach a photo in chat, or send one through the OpenAI or Anthropic API, and the model describes the scene, reads text in the image, and answers questions about what's there. Validated from the tiny 0.8B up to the 27B; vision-capable models keep their multi-token-prediction speedup on text turns and switch it off automatically for image turns, so picture questions stay correct.
- **Large file writes are reliable now.** Ask a local model to write a whole HTML page, a long script, or a multi-page document and it lands as a real file instead of spilling into the chat as raw text. The app now quietly repairs the small mistakes smaller models make when emitting a big file in one shot — stray quotes, unescaped characters, literal line breaks — and recognizes when a write was simply cut off for being too long, telling the model to finish the job in chunks and append each part to the same file. The "write me a big file" requests that used to silently fail now succeed, even on 1–4B models.
- **Let a response run as long as it needs.** A new "Auto" option for maximum output length lets a single reply run until it's genuinely done — bounded only by the model's context window — so a long file or detailed answer isn't clipped at an arbitrary limit. On smaller-memory Macs the agent is also told its real output budget up front, so it paces a big file instead of starting one it can't finish.
- **Your agent can run servers and long jobs.** Agent mode can now start a web app, a dev server, or any long-lived command in the background: it returns instantly with a handle and keeps running while the agent continues working. The model can read the process's output, stop it, or list everything it has running — and when it launches something you can open, it binds to your network and hands back a ready-to-click URL for your other devices.
- **DeepSeek-V4-Flash respects your context setting.** The context-size control in Settings now applies to the DeepSeek-V4-Flash engine too — previously it always ran at a fixed window. Dial it to match your prompts and memory, or leave it on Auto for the sensible default.
- **Qwen 3.6 decodes faster on agent turns.** A new speculative-decoding path for Qwen's GatedDeltaNet models skips redundant work when the model echoes back existing content — exactly what happens during file edits — for about 22% faster decode on echo-heavy turns (Qwen3.6-27B), with byte-identical output.
- **Every chat tab keeps its own setup.** Think, Agent, and MCP toggles — and a tool's "always allow" approval — now belong to the individual conversation you set them in, instead of bleeding across tabs or being forgotten when you switch. The Stop button is per-chat too: only the conversation that's actually replying shows it, so another tab stays free for a new message. Voice Mode opens with the same Think/Agent/MCP settings as the chat you launched it from, and stays focused on that one conversation.
- **Watch replies as they're written.** The chat now shows a live token count for the response in progress, and the context-usage bar fills as the model streams — so you can see length and remaining room in real time instead of waiting for the reply to finish.

---

## v26.6.11 — Message your model from your phone

- **Telegram bot — your model in your pocket.** Make a bot in Telegram, paste its token, flip a switch, and message your local model from anywhere — no public URL, port-forwarding, or cloud relay; it works behind home Wi-Fi over your normal connection. Turn on Agent mode and it can run tools, read and write files (confined to a workspace folder), and even schedule tasks for you, all from your phone. The bot locks to the first chat that messages it, so no one else can drive your Mac.
- **Paste anything straight into chat.** Drop or paste an image, a PDF, or a whole folder into the message box — the same as the attach button. Folders get indexed for question-answering, PDFs have their text pulled in, and images go to vision models.
- **Cleaner agent conversations.** Tool calls and their results now fold into a compact, expandable summary, so a long agent run reads like a clear narrative instead of screens of raw output.
- **Memory you can see — and that stops surprising you.** A new memory readout in the menu-bar tray shows what your model and context are using, and a pre-flight check turns a "model too big for free RAM" crash into a clear, upfront message. On 16 GB Macs the context window and cross-request cache now size themselves to your RAM, so long agent sessions stay stable — and if a prompt genuinely won't fit, you get a plain "prompt too long" notice instead of an out-of-memory crash. The server log also shows the exact launch command at the top for easy troubleshooting.
- **Run DeepSeek-V4-Flash even when it's bigger than your RAM.** A new SSD weight-streaming option lets the DeepSeek-V4-Flash engine stream expert weights from disk instead of holding the whole model in memory — so the 80 GB checkpoint that used to crash at startup ("insufficient memory") now loads and serves. Flip it on in Settings when the model is larger than available RAM; it trades a little decode speed for the disk reads, and is ignored by every other model.
- **More Gemma 3 models supported.** Flat text-only Gemma 3 checkpoints — including the popular abliterated builds — now load and run out of the box.
- **Smoother Voice Mode setup, cleaner Gemma replies.** Turning on Voice Mode now shows a friendly card naming exactly what's missing — the on-device dictation model, microphone access — instead of quietly failing. And a Gemma quirk that occasionally leaked a raw thinking tag into the end of a reply is fixed, so answers stay clean.

---

## v26.6.10 — Text diffusion lands on Apple Silicon

- **DiffusionGemma runs natively.** Google's block-diffusion model ([diffusiongemma-26B-A4B-it](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit)) writes whole 256-token blocks in parallel instead of one token at a time: the full canvas-denoising loop — entropy-bound sampling, self-conditioning, adaptive early stopping — validated tensor-by-tensor against the reference implementation. Up to 25 tokens land per forward pass, and decode runs ~30% faster than the mlx-vlm reference on the same M-series hardware (31.8 vs 24.6 tok/s on a story prompt).
- **Diffusion on every API surface, day one.** Chat completions, Anthropic messages, Responses, and FIM completions all serve it — streaming arrives block-by-block as each canvas commits, thinking mode separates reasoning cleanly, and tool calls come out with exact JSON arguments, ready for agent loops.
- **NVFP4 quantized models load and serve.** Checkpoints converted with MLX's NVIDIA-FP4 mode (`gemma-4-31b-it-nvfp4`, `Qwen3.6-27B-nvfp4`, `Qwen3-Next-80B-A3B-Thinking-mlx-nvfp4`, and the rest of the growing nvfp4 catalog) now run out of the box instead of crashing at load — output verified token-identical to the reference implementation at temperature 0. mxfp4 and mxfp8 checkpoints ride the same path.
- **Mixed-precision QAT checkpoints resolve per weight.** NVFP4 QAT conversions that keep sensitive layers at affine 8-bit (the gemma-4 QAT series overrides the shared MLP and MoE router) dispatch each tensor to its own scheme automatically — dense, MoE expert gather, embeddings, and vision projections included.
- **Discovery picks them up.** `--model-dir` folders now list nvfp4/mxfp4/mxfp8 models in `/v1/models` and the app's model picker instead of skipping them, and the startup banner reports the quantization mode. The Model Browser offers these repos for download too — they were still stamped "Unsupported quantization" by a stale client-side gate — and badges them with their format (NVFP4/MXFP4/MXFP8) in the quant column.
- **Ask your documents.** Attach a folder of mixed files — chat transcripts, notes, PDFs, JSON/YAML exports — from the chat's paperclip menu and ask questions about them in plain language. The app indexes the folder in memory (nothing leaves your Mac, nothing written to disk) and the model pulls in the relevant passages automatically, citing source filenames. Works in plain chat or alongside Agent and MCP tools.
- **Document indexing runs on the GPU — about 5× faster, zero setup.** The first time you attach a folder, the app quietly fetches a 35 MB embedding model (one-time, resumable) and registers it with the running server; from then on indexing rides the GPU — a 500-file folder indexes in ~7 s instead of ~33 s, with your CPU left free. Everything stays local: the model downloads once from Hugging Face, your documents never leave the Mac. The `/v1/embeddings` API got the same treatment for everyone: input arrays embed in single batched GPU passes (~1.4 ms per 1200-char passage), results identical to one-at-a-time calls, encoders hot-load beside your chat model, and `/v1/load-model` now accepts an absolute model path. Encoder repos (BGE, MiniLM …) are downloadable from the Model Browser too.
- **Agents that run colorful CLIs no longer derail the model.** A tool result carrying raw terminal control codes (an interactive npm prompt, a spinner, anything ANSI) could silently break prompt construction from that turn on — Gemma 4 models would respond by hallucinating entire conversations, inventing tool calls and their results. Any byte a tool emits now round-trips safely into the conversation history, and a prompt-format downgrade is logged loudly instead of passing silently.

---

## v26.6.9 — Qwen 3.6 predicts its own future

- **Native multi-token prediction for Qwen 3.6.** Models that ship Qwen's trained MTP head as an `mtp/` sidecar (like [ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve](https://huggingface.co/ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve)) now use it automatically: the model drafts its own next tokens and verifies them in one pass, with exact rejection sampling — same output distribution, measurably faster. Agent-style edit and echo workloads decode **up to 1.8× faster** (29 → 51.6 tok/s on Qwen3.6-27B 4-bit, M4 Max), code generation 1.43×, creative writing ~1.1× — beating the reference MTP runtime on every workload measured, at identical output quality.
- **Zero setup.** Drop a model with an `mtp/weights.safetensors` sidecar into your model folder and every API surface — chat completions, Anthropic messages, Responses, and FIM completions, streaming and non-streaming — speculates by default. `--no-mtp` or per-request `enable_mtp: false` opts out; `--mtp-depth` goes deeper.
- **Self-tuning speculation.** The MTP controller watches its own acceptance rate per request and adapts draft depth on the fly, so echo-heavy agent turns speculate aggressively while novel prose stays at the safe depth — no manual tuning pass required.

---

## v26.6.8 — Smarter speculation, honest reasoning, agent-ready defaults

- **Local coding agents work out of the box.** Requests that omit `max_tokens` now generate until done (bounded by the context window) instead of stopping at 256 tokens — the old cap silently cut agent clients like pi off mid-thought on every turn. Verified end to end: pi completes multi-step build-test-fix coding tasks against both Qwen 3.6 and Gemma 4 models.
- **Reasoning never leaks — or swallows the answer.** Thinking output is now cleanly separated from the visible reply on every API surface, fixing the two cases that went wrong: a thought truncated by the token budget used to land in the visible content, and with tools active the final answer could be misfiled as reasoning, leaving agent clients with an empty reply. Usage now reports `reasoning_tokens` so token accounting is honest.
- **Speculative decoding manages itself mid-request.** PLD now watches its own payoff: on novel content it steps aside and recovers the full pipelined decode rate, then re-engages the moment output turns repetitive — exactly when an agent starts echoing file content. Novel-preamble-then-echo turns decode ~25% faster; pure echo keeps its ~2× win; output stays byte-identical.
- **Gemma 4 12B agent turns are clean.** Four 12B-specific bugs found via live agent testing: a trailing thought-channel opener no longer leaks the raw `<|channel>thought` tag into the visible reply; a thought channel re-opened mid-answer (seen live via Claude Code) is folded into the reasoning block instead of leaking raw tags; a malformed tool-call string no longer swallows the argument block’s closing brace (which once created a file literally named ``mlx_pi1.html`}``); and a tool call emitted as a bare JSON arguments object — no tool name at all — is now matched against the request’s tool schemas and executed when exactly one tool fits. File-edit tool calls land with exact filenames.
- **Qwen 3.6 decodes faster.** The GatedDeltaNet decay gate now runs as a single fused kernel instead of ten separate ops per layer per token: 27B hybrid +1.3%, 35B MoE now +3.9% ahead of mlx-lm. Against a fresh mlx-lm 0.31.3, mlx-serve leads or ties on every supported model.
- **Gemma 3 answers correctly — and can call tools.** A sliding-window layer-pattern mismatch had every Gemma 3 layer attending with the wrong RoPE base and scope, degrading arithmetic and digit handling (17×23 came back as 21); output now matches the reference implementation. Gemma 3 also gained tool calling: the JSON-in-a-markdown-fence calls it naturally emits are now parsed, with exact argument fidelity.
- **Format correctness is now pinned for every model family.** A new three-layer test suite — a hermetic corpus of real captured model outputs that runs in CI without weights, a live seven-family matrix (Qwen, Gemma 4, Gemma 3, Qwen MoE Coder, GGUF, DeepSeek-V4-Flash), and agent-transcript audits — guards against thinking-tag leaks, mangled tool arguments, and misrouted answers across all of them.
- **Parallel tool calls work on every model.** Models without a trained tool format (Gemma 3) emit parallel calls as a JSON array — previously only the first call executed and the rest silently dropped, on all three API surfaces. Tool calls that arrive mangled — a dropped closing brace, a hallucinated `</tool_action>` close tag, or DeepSeek-V4-Flash's JSON-free XML form (`<tool_name>shell</tool_name><command>…</command>`), all captured live — are now repaired and executed instead of leaking into the visible reply. DeepSeek-V4-Flash's name-in-the-tag variants (`<tool_read><path>…</path></tool_read>` and `<tool_write>{…}</tool_write>`) are recognized too, while hallucinated result tags like `<tool_output>` correctly stay prose.
- **The prompt cache engages where it silently couldn't.** Dense Qwen 3.6 and pure-attention MoE models (Qwen3-Coder, Gemma 4 MoE) never got a warm prefix hit — every agent turn paid a full cold prefill. GGUF models served via llama.cpp had the same blind spot: the engine defaulted to a single KV session, so even back-to-back requests sharing a long prefix re-prefilled from scratch — it now keeps 4 sessions resident (`--llama-cache-entries`, sessions created lazily). All classes now reuse cached prefixes; a full API-compliance sweep (112 checks across all three API surfaces) passes on every architecture.
- **Long prompts no longer wedge the server.** Huge MCP-laden Claude Code prompts (40K+ tokens) take minutes of prefill on bigger models; clients used to time out, retry, and silently stack abandoned prefills behind each other while the server looked dead. Streaming responses now send keepalive pings every 5 seconds during prefill (no more client timeouts), and a disconnected client cancels its request within seconds — the prefill aborts at the next chunk and the GPU moves on.
- **Models now sample the way their authors intended.** Requests that omit sampling parameters used to run at temperature 1.0 over the full untruncated vocabulary — far outside any model card's envelope (Qwen 3.6 wants top-k 20, Gemma 4 wants top-k 64). The server now reads each model's `generation_config.json` recommendations and applies them to omitted fields, and the app's Settings temperature/top-p reach external clients like Claude Code via new `--temp`/`--top-p`/`--top-k` launch flags. Explicit request values always win; rambling and premature turn-endings in Claude Code drop noticeably.
- **Claude Code works cleanly with thinking models.** Streaming Anthropic-API turns with thinking and tools together — Claude Code's exact request shape — used to leak raw `</think>` markers into the visible transcript on Qwen 3.5/3.6 and could abort tool-call turns with a protocol error ("Content block not found"). Thinking now arrives as proper thinking blocks, the visible answer and tool calls stream in valid order, and both streaming surfaces share one gate that is hermetically tested against every recorded model family.
- **Strict JSON mode holds on `/v1/responses`.** `json_object` requests on the Responses API now grammar-constrain decoding the same way chat completions do, so fence-happy models can't return markdown-wrapped output to structured-output clients.
- **The menu bar shows which engine your model runs on.** The status menu now displays the selected model's engine (MLX-Serve, GGUF · llama.cpp, or GGUF · DS4) next to the auto-start toggle, and the model picker disambiguates same-named MLX/GGUF entries — which also fixes both rows showing as selected.

---

## v26.6.7 — Agent-grade speed and rock-solid concurrent streaming

- **MacOS 26 Is now required** (issue #21), 
- **Network GUI Settings** Expose port & bind ip (issue #22)
- **Support qwen3_moe architecture** (Qwen3-30B-A3B / Coder) (issue #20)
- **Big prompts tokenize faster.** A rewrite of the BPE tokenizer's merge loop takes a Claude Code-sized system prompt (~30 KB) from 3.9 seconds to 8 milliseconds — every request used to pay that cost, even on a full KV-cache hit. Warm agent turns now round-trip in ~0.1s end to end.
- **Concurrent streams no longer garble.** When a second request arrived mid-generation, the first stream could emit a duplicated token, drop its tail, and silently corrupt its KV cache — breaking tool-call parsing for agent clients. Fixed, with a byte-equivalence regression test covering mid-stream joins and simultaneous bursts.
- **The prompt cache survives agent traffic.** Interleaved requests (subagents, title generation, parallel tools) used to evict the long system-prompt prefix on every turn, forcing a full re-prefill. The cache now retains up to 32 conversation roots by default, still bounded by the 2 GB memory budget.
- **Anthropic API reports cache hits.** `/v1/messages` responses now include `usage.cache_read_input_tokens` (non-streaming and streaming), so Claude Code and Anthropic SDK clients see real prompt-cache savings.
- **Speculative decoding now works with tools — 2× faster agent edits.** Both PLD and the Gemma 4 assistant drafter used to switch off whenever a request defined tools, which is every Claude Code request. With the gates lifted, file-edit tool calls that echo code back decode at ~2.1× (72 → 150 tok/s measured on Gemma 4 E4B, both modes), with byte-identical output and tool calls. Coverage is now uniform: both modes run on every API surface — Chat Completions, Anthropic Messages, OpenAI Responses, and legacy completions — streaming and non-streaming alike.
- **Code-completion clients get speculative decoding too.** The legacy `/v1/completions` endpoint (used by FIM / autocomplete tooling) silently ignored `--pld`/`--drafter`; repetitive-code completions now decode at ~1.9× (73 → 139 tok/s). A companion fix keeps the first line's leading indentation intact in non-streaming responses, matching streaming output exactly.
- **Full API-compliance sweep.** All 112 llmprobe checks across OpenAI Responses, Chat Completions, and Anthropic Messages now pass, including WebSocket transport, streaming parity, and truncation semantics.

---

## v26.6.6 — Scheduled Tasks: your private, always-on agent

- **Set it and forget it.** A new Tasks window lets you hand your local model a goal — "every weekday at 8am, check my watched sites and write me a briefing" — and it runs unattended in the background, on a schedule or on demand. Everything stays on your Mac: no cloud, no per-run fees, your logged-in browser sessions never leave the machine.

- **Just say when.** Schedule tasks in plain English ("every 15 minutes", "weekdays at 8am", "weekly on Mondays") with one-tap presets, or drop in a cron expression for full control. What you typed is echoed back as a plain confirmation so there's no guesswork.

- **You decide how much it can do.** Pick an autonomy level per task — Read-only, Workspace, Full auto, or YOLO. If a task wants to do something beyond its level, it pauses and sends you a notification you can Approve or Deny right from Notification Center; approve it and the run picks up where it left off, even after a restart.

- **Stay in the loop.** Every run finishes with a notification and a full transcript plus any files it produced, all kept in a per-task history you can scroll back through. Edit a task, run it on demand to test it, or pin it to a specific model — the server starts and switches models automatically when the task fires.

- **Tidier menu bar.** The tray's quick actions are now clearly labelled — Chat, Tasks, and Code — and the browser moved into the app's top menu bar (⇧⌘B) to keep the tray focused.

---

## v26.6.5 — Voice Mode reliability, agents that finish, dense bf16 Gemma

- **Voice Mode can hear you again.** A code-signing fix grants the app the microphone entitlement it needs under macOS's hardened runtime, so the permission prompt actually appears the first time you say "Hey Loki" — previously mic access was silently denied and Voice Mode never picked up a word. Applied to both local and released/notarized builds.

- **The menu-bar tray stays responsive during an answer.** While the model streamed a reply into the open tray popover, the **Stop** button — and the rest of the tray — could go dead even though the dropdown menus still worked. Streaming updates are now batched to a steady cadence, the tray status dot is a solid color instead of a constant animation, and the microphone is released cleanly before it reopens on barge-in. Together these keep the tray clickable from the first token to the last.

- **The agent stops quitting mid-task.** A long, multi-step agent run could halt early after a single stray bad tool call — even after lots of successful work. Recoverable failures (malformed tool calls, truncated arguments, empty replies) are now counted consecutively and reset on every real tool round, so one isolated hiccup no longer ends a productive turn.

- **No more infinite tool loops.** MCP tool calls now pass through the same repetition guard as the built-in tools, so a model can't get wedged firing the same database query — or any MCP call — over and over. Genuinely different calls stay independent and aren't over-blocked.

- **Gemma 4 12B won't spin forever after a big tool result.** The 12B model occasionally collapsed into repeating its thinking opener endlessly until it burned the entire token budget; the server now detects a stuck repetition loop mid-generation and ends the turn cleanly. A companion fix keeps a raw control tag from leaking into a reply that was cut off mid-thought.

- **Longer answers get cut off far less often.** The default max-tokens rises from 4,096 to 16,384, so a reasoning trace plus a real code or agent answer no longer trips the truncation cap in the middle of a reply (the server still clamps to your context window, so it can't overflow). When output is genuinely truncated, the "output truncated" notice now appears exactly once per turn instead of stacking on every step.

- **Full-precision bf16 Gemma 4, no repack required.** Checkpoints that ship in plain bf16 with no quantization key now load and generate — including Google's quantization-aware-trained `gemma-4-E2B-it-qat-bf16` and dense bf16 Qwen 3.5/3.6. Gemma's E-series layout is handled natively, including its memory-saving attention blocks that share key/value tensors across layers, so you get full-fidelity output without converting the model to a quantized format first.

- **bf16 Gemma 4 sees images too.** The QAT bf16 Gemma 4 vision tower now works out of the box — no need to launch with vision disabled. Attach a photo and the model recognizes colors, shapes, and objects, exactly like the quantized Gemma 4 vision models.

---

## v26.6.2 — Hey Loki ! Voice Mode

- **Hands-free Voice Mode.** Say "Hey Loki" and just talk to your local model — no typing, no buttons. Speech is transcribed **entirely on-device**, so your audio never leaves the Mac and it works with no internet and runs straight from the menu-bar tray with no window open — a chime confirms it heard you, a soft cue plays while it's thinking — or as a full-screen animated orb over the chat. Agent tools, thinking, and MCP all work in voice exactly as they do in text, because both now run through one shared engine.

- **Gemma 4 12B now sees and hears.** The 12B "unified" checkpoint (`gemma-4-12b-it-4bit`) understands **both images and spoken audio** with no separate vision or audio tower — send a photo or raw microphone audio and it reasons over them directly. Voice mode and the chat window can hand the model what you say and show, not just what you type.

- **Neural text-to-speech with voice cloning.** A new Audio generation window speaks any text aloud — and can **clone a voice** from a few seconds of reference audio you record in-app or drop in as a file. Three on-device models from lightest to highest fidelity: MOSS-TTS Nano (100M, ~0.5 GB), Qwen3-TTS 0.6B (~1.5 GB), and Qwen3-TTS 1.7B (~3.5 GB) — all MLX-native, no PyTorch. Reference clips are normalized in-app, so there's nothing extra to install.

- **Video generation & setup fix.** A breaking rename in the upstream `ltx-2-mlx` pipelines — plus a newly mandatory frame-rate setting — had been leaving on-device video generation broken even after a clean install. MLX Core now drives the current pipeline API across all three quality tiers (one-stage, two-stage, two-stage HQ), and the fast one-stage path picked up first-frame image-to-video support along the way.


---

## v26.6.1 — Gemma 4 12b Support
- **Gemma 4 12B.** Run `gemma-4-12b-it-4bit` — the dense 12B slots between E4B and the 26B-A4B MoE for a quality-vs-speed middle ground.
- **Agent mode that actually codes.** The built-in agent now completes real multi-step coding tasks instead of stalling. Tool calls whose name carries a stray trailing colon (some Gemma 4 builds emit `shell:`) resolve correctly instead of dead-looping on "unknown tool"; the shell tool closes stdin so interactive scaffolders like `npm create svelte` / `npx sv create` fail fast instead of freezing the agent, backed by a timeout that can't hang on a runaway command; and the agent is steered toward non-interactive setup (`npm install` + writing files directly) over interactive wizards. A local model can now `npm install`, initialize Prisma, and create a SQLite database end-to-end.
- **Reliable Gemma 4 tool calls with nested arguments.** Tool calls whose arguments contain nested objects or arrays — a metadata object, a list of recipients — now come back as valid JSON instead of malformed output that broke the call.
- **Improved GGUF DS4 routing between llama.cpp & ds4**
- **Broader GGUF model support.** Refreshed the embedded llama.cpp engine, adding native support for more model families out of the box — including GGUF Gemma 4, DeepSeek V3.2, LFM2.5, EXAONE 4.5, and MiniCPM5.
- **DeepSeek-V4-Flash engine refresh.** Updated to the latest ds4 engine with generation-correctness and Metal kernel fixes, plus Metal 4 acceleration that kicks in on M5-class hardware.
- **Fix Brew release**

## v26.5.7 — Run any GGUF model, faster than LM Studio on the same file

- **Any GGUF model, natively.** mlx-serve now embeds llama.cpp's inference library, so the whole GGUF world — Qwen, Llama, Mistral, Gemma, and thousands more — runs on Apple Silicon alongside MLX models. Pick a `.gguf` in the menu-bar app and it just works: the server auto-detects the format and routes to the right engine (DeepSeek-V4-Flash still uses the dedicated ds4 engine; everything else uses llama.cpp). No new app to trust — the engine ships inside the same signed, notarized bundle, so there's no "unidentified developer" dialog.

- **Faster than LM Studio on the same `.gguf`.** Head-to-head on Gemma 4 E4B Q4_K_M (identical file, Apple M4 16GB): free-form decode +15%, echo +13%, code +12%, prefill +5%. Warm TTFT 15–26% better than LM Studio across both MLX and GGUF backends. Side-by-side chart and CSV ship under `docs/`.

- **Warm chats 7.7× faster.** A new chat-template + tokenize cache turns the second hit on a long conversation into a memcpy: on a 1813-token prompt, the wall between "send" and "first token" drops from 271 ms to 35 ms. Applies to every engine — MLX, llama.cpp, and ds4 — and pairs with the existing prefix cache so multi-turn agent loops feel near-instant.

- **Multi-doc agents stay warm.** llama.cpp now keeps an LRU of KV sessions, so alternating between two long prompts no longer pays the cold prefill twice. On a Qwen3.5-4B Q4_NL workload with two long-doc QA prompts, second-time A reuses 71/72 tokens (was 3/72). New `--llama-cache-entries N` knob; defaults to 1 for backwards compatibility, the menu-bar Settings panel exposes it.

- **Engine-aware Settings.** The Settings window now shows the right knobs for the model you've loaded: MLX targets see the MLX KV-quant + speculative-decode controls; GGUF targets see llama.cpp's own quant and session-cache controls instead of MLX toggles that silently no-op. New rows for `--llama-kv-quant`, `--llama-cache-entries`, and `--tokenize-cache-entries`; restart banner fires when launch flags change.

- **Smarter Model Browser for GGUF.** GGUF repos now show a "X–Y GB" RAM-estimate range covering the smallest and largest quants in the repo, the previous "Unsupported architecture" false-flag on LM Studio's community GGUF repacks (`lmstudio-community/gemma-4-E4B-it-GGUF` and friends) is fixed, mmproj sidecars are auto-skipped when picking a `.gguf` from a folder, and the MLX-only drafter pairing chip no longer appears on rows where it can't apply. Downloads + Download action columns widened so headers and the GGUF "Download ▾" menu render on one line.


---

## v26.5.6 — DeepSeek-V4 done right, faster than LM Studio, continuous batching

- **DeepSeek-V4-Flash, the right way.** The 284B-parameter beast now runs through Salvatore Sanfilippo's [`antirez/ds4`](https://github.com/antirez/ds4) engine — native Metal kernels, byte-validated against the reference forward, single self-contained binary (kernel sources are embedded and staged at first launch). Available on 96 GB+ Macs straight from the MLX Core Model Browser: one-click download of the GGUF, served alongside MLX models from the same picker. Agent mode and MCP tool calling work on DSV4 too — the chat-template fallback inlines the tool catalog so the model sees the full toolset. We retired our previous 7,000-line in-house implementation in favor of the upstream engine; the result is faster, more memory-stable, and a lot less code to maintain.

- **Faster than LM Studio (MLX) on every model we test.** Refreshed cross-engine charts across Gemma 4 (E2B / E4B / 31B / 26B-A4B-MoE) and Qwen 3.6 (27B / 35B-A3B) put MLX-serve ahead on echo, code completion, and free-form writing — every cell, every model. `--pld` takes the top bar on echo-heavy workloads (up to 1.5× on MoE); `--drafter` wins Gemma 4 code completion. Side-by-side charts and CSVs ship under `docs/`.

- **Continuous batching.** A new `--max-concurrent N` flag batches up to N decode requests through a single forward pass — about 1.6× throughput at 4-way parallel on dense models (Gemma 4, Qwen 3, Llama, Mistral). Hybrid SSM and MoE models route through the same scheduler queue but stay single-stream. A 24-hour soak across four mixed workloads holds RSS drift under 5%.

- **Smaller KV cache, bigger context.** `--kv-quant {4, 8, turbo2, turbo4}` (plus a per-request override on every chat endpoint) shrinks KV memory by ~4× at 4-bit and ~2× at 8-bit. 16K contexts now fit on hardware that couldn't hold them dense, or you double your parallel-request budget at the same context length. The TurboQuant variants add a per-layer Hadamard rotation that handles heavy-tailed activations more gracefully.

- **One server, every model on disk.** `--model-dir <path>` discovers and serves every model in a folder; clients route by name in the request's `"model"` field. LRU eviction keeps the resident set within configurable byte/count caps. MLX Core's menu-bar picker now hot-switches models in place — no chat-session interruption.

- **3.57× faster first request, smarter multi-turn.** Eager warmup at boot page-faults the weights and pre-compiles the decode kernels (1097 → 307 ms wall on Gemma 4 E4B 4-bit). A new shared-prefix cache (`--prefix-cache-entries`, `--prefix-cache-mem`) skips re-prefilling system prompts across turns; agent loops feel tighter. `/v1/embeddings` now runs on the same thread-local-stream-safe path as generation, so encoder-only models go parallel too. Verified by a new 11-turn agent memory harness (plant facts → tools → thinking → recall under mode transitions) that passes 15/15 on every supported arch including DSV4 via ds4.

- **MLX Core, more in-app control.** A new Settings → Performance section exposes continuous batching, KV-cache quantization, and the prefix cache as menu-bar tunables instead of CLI-only flags. A "Reset to Defaults" footer restores every Settings field with one click + confirmation. The chat toolbar's Agent button hover now enumerates all 10 built-in tools so you can see exactly what Agent mode activates; every other toolbar button (Workspace, Folder, Settings, Think, MCP) gained a substantive tooltip too. New tool-approval dialog in Agent mode — **Allow** / **Deny** / **Always allow this session** — pops before each tool runs, so you can shape-check shell commands and file edits before the model touches your machine. The Model Browser gained a custom-folder picker so models that live outside `~/.mlx-serve/models` and `~/.lmstudio/models` show up in the picker without re-downloading. The GPU-memory indicator now reports correctly when the ds4 engine is loaded, and the picker only surfaces DeepSeek-V4-Flash GGUFs (not arbitrary LM Studio GGUFs the server can't load).

---

## v26.5.5 — Multi-turn agent speed-ups, MoE forward, +39% vs LM Studio

- **+39% faster than LM Studio overall** (geomean across 18 cells, identical 4-bit MLX weights, ctx=4096, temp=0). Echo +60–122%, code +47–53% on dense Gemma 4, free-form +20–35%. New apples-to-apples benchmark at `tests/bench_vs_lmstudio.sh`.
- **Multi-turn agent loops dramatically faster**: KV cache now reuses the previous turn's generated tokens, so turn N+1 skips re-prefilling its own assistant reply. Cache hit jumps from ~15% to ~97% on the second turn; savings compound across long conversations. Side-benefit: no per-turn K/V drift from re-running the same tokens through different reduction orders at INT4/FP16.
- **Smarter speculative decoding**: per-target tuned block sizes and a per-draft runtime acceptance gate keep PLD/drafter on where they pay off (echo, RAG, code) and step aside on creative content. Drafter auto-disables on Mixture-of-Experts targets where verify-forward dominates; PLD stays on and wins. One-click drafter toggle in MLX Core (Settings → Speculative Decoding) with auto-discovery and a contextual "pair with this drafter for +30-50% on code" chip in the Model Browser.
- **Faster Mixture-of-Experts**: multi-position MoE inference (prefill, PLD verify, drafter verify) now uses sorted-expert HBM streaming as soon as there's more than one position. PLD on Gemma 4 26B-A4B and Qwen 3.6 35B-A3B picked up another +13–18% on echo. Unsloth UD MoE checkpoints (Qwen 3.6 35B-A3B-UD-MLX and friends — router/shared-expert in bf16, experts 4-bit) now load and run cleanly.
- **KV cache + image-cache fixes**: pure-attention models no longer hard-reset on mid-conversation prompt divergence (truncates to shared prefix instead — fixes a long-running cache regression). Anthropic Messages API now invalidates cache on image requests, fixing a red→blue PNG round-trip bug where vision embeddings could leak across turns.
- **Per-request speculative telemetry + agent memory test**: every speculative request logs acceptance rate, per-round average, and runtime-gate state. New `test_long_agent_memory.sh` plants three facts in turn 1 and asserts they survive a 10-turn conversation across tool / thinking / mode transitions — guards against the "model acts like first-time-seen" class of bug.
- **Removed Multi-Token Prediction**: cross-model bench showed MTP at parity or slower than regular generation on every workload. PLD covers the same ground with bigger wins. Existing MTP-bearing checkpoints (Qwen 3.5 / 3.6 with MTP heads) continue to load and run as regular models.

---

## v26.5.4 — Speculative decoding (MTP / PLD / Gemma 4 drafter), Settings window, tokenizer fix

- **MTP (Multi-Token Prediction)**: native self-speculative decoding for Qwen3.5/3.6/Qwen3-Next checkpoints that ship MTP weights. `--mtp` flag and per-request `enable_mtp`. Snapshot/restore handles hybrid GatedDeltaNet rollback; tools/logprobs/grammar auto-disable.
- **PLD (Prompt Lookup Decoding) on by default**: model-agnostic n-gram speculative decoding works on every supported architecture (Gemma, Qwen, Llama, Mistral, Nemotron-H, LFM2.5). Up to 1.82× on heavy-echo Gemma-4-E4B, 1.16× on RAG-style retrieval. `--no-pld` to disable.
- **Gemma 4 assistant drafter**: cross-attention drafter using Google's `gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant-bf16` checkpoints. `--drafter <dir>` activates it; 1.98× decode on echo-heavy E4B-4bit (3.0/3 max acceptance). Streaming supported across chat / Anthropic / Responses paths.
- **Adaptive prompt-time gate**: per-request 3-gram repetition score on the prompt disables PLD/drafter on novel content (`spec_gate_threshold = 0.01`). Validated 9/9 on a tuning corpus. Bypass with explicit `enable_pld:true` / `enable_drafter:true` in the request body.
- **Runtime acceptance gate**: mid-decode fallback when actual draft acceptance is below break-even — < 0.30 after 5 attempts for PLD/drafter, < 0.70 after 8 attempts for MTP (binary outcome → separate threshold). Sticky per-request; protects against workloads the prompt-time gate misjudged.
- **Settings window** (MLX Core, Cmd+,): single source-of-truth for server-launch flags (port, ctx-size, log-level, vision, MTP/PLD/drafter, draft lengths) and per-request defaults (max-tokens, temperature, top-p/top-k, repeat/presence penalty, reasoning budget, thinking, per-request spec-decode overrides). Restart banner appears when launch flags change; per-request fields apply on the next chat.
- **Tokenizer correctness fix**: GPT-2 pre-tokenizer rewritten as a priority-ordered state machine matching the reference regex. Four classes of splits now correct — leading-space + letters as one pre-token (` total`), leading-space + punct (` +=`), multi-space runs preceding identifiers (`    total`), and digits as single codepoints (`100` → 1, 0, 0). Old impl perturbed BPE merges on every subsequent word.
- **Markdown rendering**: assistant messages render in a single NSTextView so drag-select spans paragraphs / lists / code blocks / tables. Adds GFM table parsing with column alignment; small in-prompt nudge steers smaller models toward GFM table syntax for plain-chat tabular output.
- **`/v1/models` meta additions**: `model_max_tokens` (architectural cap, independent of `--ctx-size`) and `supports_mtp` (config declares MTP layers).
- **Build**: Swift 5 language mode globally (`-Xswiftc -swift-version -Xswiftc 5`) — required under Swift 6.3 / Xcode 26+ because the pinned `swift-sdk` 0.10.x trips new `SendingRisksDataRace` diagnostics. No-op on the Swift 6.1 CI runner.
- **Tests**: PLD / MTP / drafter byte-equivalence suites (greedy temp=0); streaming-vs-non-streaming byte-equivalence; long-greedy memorized-prompt test that asserts byte-identical first 30 tokens (INT4 float-noise tail documented in CLAUDE.md). New `bench_spec.sh` with `--corpus` and `--gated` modes.

---

## v26.5.3 — Real Sonoma compatibility, CI test gate, dependency pinning

- **Bundled dylibs are now actually Sonoma-compatible.** Switched the release runner from `macos-26` to `macos-14`; Homebrew bottles for `mlx`, `mlx-c`, `webp`, and `libsharpyuv` come out stamped `minos 14.0` instead of `minos 26.0`. v26.5.2 fixed the Zig binary's minOS but the bundled libs still required Tahoe — dyld would refuse them on Sonoma at first launch, surfacing as "Server failed to start" in MLX Core.
- **CI test gate**: `zig build test` and `swift test` now run between build and packaging. A regression that breaks the suite no longer ships.
- **Post-build smoke tests**: `mlx-serve --version` runs against both the freshly built binary and the install_name_tool-rewired CLI artifact, so missing-dylib failures surface before the notarize step burns a submission slot.
- **Homebrew dependencies pinned in `build.zig`**: builds now hard-fail with a clear message if `mlx`, `mlx-c`, or `webp` are below the minimum versions the codebase expects (mlx >= 0.31.2 — the version the v26.4.33 thread-local-stream hotfix targeted).
- **Zig 0.16+ enforced**: `comptime` check at the top of `build.zig` produces "needs Zig 0.16, run brew upgrade zig" instead of a cryptic `StdIo.inherit` enum error on older Zig. Belt-and-suspenders to `build.zig.zon`'s `minimum_zig_version`, which Zig 0.15 doesn't enforce for root projects.
- **`Brewfile`**: declarative dep manifest. `brew bundle install` from a fresh checkout (or in CI) covers `zig`, `mlx-c`, `webp`, `create-dmg`.
- **`workflow_dispatch` version scheme fixed**: now reads the latest `vYY.M.N` release tag and increments N (matching the documented CalVer scheme). Was using `github.run_number`, a global counter, which would have produced versions like v26.5.1234.

---

## v26.5.2 — Sonoma compatibility for CLI binary

- **Fix `mlx-serve` failing to launch on macOS 14 (Sonoma)**: pin `LC_BUILD_VERSION minos` to 14.0 in `build.zig` so binaries built on the `macos-26` (Tahoe) CI runner still load on Sonoma. dyld refuses any image whose minOS is newer than the running OS. MLX Core (Swift) was already fine via `Package.swift`'s `.macOS(.v14)`; only the Zig binary was affected.
- **SDK auto-detection workaround**: setting any non-default target field in Zig disables native macOS framework discovery, so `build.zig` now resolves the SDK with `xcrun --sdk macosx --show-sdk-path` and adds its `Frameworks` dir as a search path. No workflow change needed.

---

## v26.5.1 — OpenAI Responses API + WebSockets, tokenizer arena fix, LM Studio discovery

- **Tokenizer ~30× faster load**: `loadTokenizer` keeps the parsed `tokenizer.json` arena alive and borrows vocab/merge string pointers from it instead of duping per entry; hashmaps pre-sized to skip rehashing. Headline downstream effect: **Qwen3.5-4B prefill 144 → 383 tok/s** (+165%, now ~93% of mlx-lm 0.31.2 reference) on 844-token prompts. Gemma-4-E4B and LFM2.5-350M within run-variance of prior numbers.

- **OpenAI Responses API (`POST /v1/responses`, `GET`/`DELETE /v1/responses/{id}`)**: stateful chains via `previous_response_id`, in-memory `ResponseStore`, streaming SSE with per-event `sequence_number`, schema-conformant envelope (`tools` / `tool_choice` / `text` / `reasoning` / `usage` echo). `experiments/openresponses` compliance suite passes 17/17. Plus `POST /v1/responses/compact` — opaque base64 history blob (`{v:1, msgs:[…]}`) that round-trips back as a `compaction` input item without an LLM call.

- **WebSocket transport on `/v1/responses`**: standard `Upgrade: websocket` handshake, each text frame is a `response.create` JSON message and each SSE event becomes one outbound text frame. New `src/ws.zig` (RFC 6455 framing, server-side). Per-connection `WsLocalCache` for `store: false` responses; no `[DONE]` on success — `response.completed` is the per-response terminator.

- **PDF chat attachments** (MLX Core): drag-drop or paperclip-pick a PDF; PDFKit extracts the text into the message preamble. Encrypted or scan-only PDFs surface a clear error alert instead of silently dropping.

- **LM Studio model auto-discovery** (MLX Core): reads LM Studio's `downloadsFolder` from `~/.lmstudio/settings.json` (falls back to `~/.lmstudio/models`), scans two levels deep for valid MLX models, groups them in the picker under "Other Discovered Models" alongside "MLX-Serve Models". GGUF folders skipped automatically via the existing `.safetensors` check. The Model Browser's "Downloaded" tab still shows only mlx-serve-managed models.

- **Server auto-restarts on model-dropdown change** (MLX Core): switching model while the server is running stops and relaunches with the new model. Fixed `ServerManager.stop()` to detach the dying process's `terminationHandler` + stderr handler so its trailing "Shutting down gracefully…" can't bleed into the new server's log or hijack `status = .starting` into `.error("Failed to start")`.

- **Native NSAlert on download failure** (MLX Core): "Not enough disk space. Need 8.4 GB but only 4.6 GB available." now pops as a modal alert in addition to the inline red text — doesn't get missed when the menu bar popover closes.

---

## v26.4.33 — Hotfix: thread-local streams in mlx 0.31.2

- **Inference now runs on the listener thread.** mlx 0.31.2 made GPU streams thread-local — model weights loaded on the main thread couldn't be evaluated from connection threads, so any chat completion crashed with `MLX error: There is no Stream(gpu, 1) in current thread.`. Removed the thread-per-connection spawn in `server.zig` and handle connections inline. The `inference_mutex` was already serializing the slow path, so this doesn't reduce real concurrency — only quick endpoints (`/health`, `/v1/models`, `/props`) get briefly delayed during generation, which is fine.
- **Transformer uses the current thread's default GPU stream** (`mlx.gpuStream()`) instead of a dedicated stream created at init time. Adds `useCurrentThreadStream()` for any future call sites that need to rebind.
- v26.4.32 fixed the `libjaccl.dylib` bundling issue but still hit this stream issue at the first inference. v26.4.33 is the actual working build.

---

## v26.4.32 — Hotfix: `libjaccl.dylib` not found at startup

- **Bundle all sibling dylibs from `/opt/homebrew/opt/mlx/lib/`**, not just `libmlx.dylib`. mlx 0.31.2 (the version on the macOS-26 GitHub runner) added a new `@rpath/libjaccl.dylib` dependency that we weren't copying — caused the v26.4.31 binary to fail at startup with `Library not loaded: @rpath/libjaccl.dylib`.
- **Add `@loader_path` to `libmlx.dylib`'s rpath** so future `@rpath` sibling deps from mlx resolve cleanly to the bundled Frameworks dir without further workflow changes.
- v26.4.31 had the same MCP + Zig 0.16 changes — this is purely a packaging fix. If you already grabbed v26.4.31 and got the dyld error, just download v26.4.32.

---

## v26.4.31 — MCP Client + Marketplace, Zig 0.16

- **MCP toggle pill**: Purple **MCP** capsule next to Think and Agent in the chat toolbar with an embedded gear icon that opens a marketplace sheet. Works with or without Agent mode.
- **swift-sdk integration**: `MCPManager` spawns each enabled stdio server via `/bin/zsh -lc 'exec npx …'`, wires stdio into `StdioTransport`, and namespaces tools as `<server>__<tool>` so cross-server collisions are impossible.
- **HTTP transport too**: URL-based MCP entries (just `"url": "https://…"`) connect via `HTTPClientTransport` with SSE streaming — no subprocess. Marketplace shows them with a blue HTTP pill.
- **10-server curated catalog**: GitHub, Azure DevOps, DBHub (universal SQL via dbhub.ai), Docker, Kubernetes, Playwright, Slack (Zencoder fork), Notion, Filesystem, Shell — each with inline `SecureField`s for required env vars / args.
- **Claude Desktop config format**: `~/.mlx-serve/mcp.json` follows the `{"mcpServers": {...}}` shape so configs paste straight across. **Source order preserved** through save/load via `OrderedDictionary` + manual outer-object emit + raw-text key-order recovery on load (Foundation's JSON encoder/decoder both shuffle keys via a hash store).
- **Auto-encoded secrets**: New `envEncoded` input kind base64-encodes ADO PATs as `base64("x:<pat>")`. Conditional `argsWhenPresent` lets ADO default to interactive browser auth and switch to PAT mode when the optional field is filled.
- **Live status per row**: Toggle a server in the marketplace and you get instant feedback — yellow "starting" → green dot + "N tools" on success, red dot + tooltip with stderr on failure. Auto-spawns on toggle so the indicator is meaningful without leaving the sheet.
- **Auto-reload on app activate**: Edit `mcp.json` in your editor, switch back to the app, and the marketplace re-hydrates from disk. No close/reopen needed.
- **Pre-flight runtime check**: `command -v <command>` runs in a login zsh before spawn — if `npx` / `docker` / etc. is missing, throws `MCPSpawnError.commandNotFound` with an install hint instead of a 30s dead-wait.
- **Fast-fail on subprocess crash**: `Process.terminationHandler` resumes a one-shot continuation the moment the child exits — docker-mcp dies in 0.6s when the daemon is down, k8s-mcp similar with broken kubeconfig, etc. We surface the captured stderr in the chat warning instead of timing out.
- **Stale errors purge on disable**: Toggling a server off clears its old `startErrors` entry instead of letting it linger in the inline chat warning.
- **Inline chat warnings**: Failed MCP startups show as a warning bubble in chat, not just hidden behind the marketplace gear.
- **Default cwd `~/.mlx-serve/workspace`**: Spawned MCP servers (filesystem, shell, etc.) anchor at the same workspace dir the agent uses by default, with per-entry `cwd` override via mcp.json. New chat sessions inherit it; old sessions saved before this default existed get backfilled on load.
- **Session cwd → MCP cwd**: When MCP servers spawn, they pick up the active chat session's `workingDirectory`. Per-entry `cwd` in mcp.json still wins.
- **Empty-arg fix**: `convertArguments` always returns a (possibly empty) dict so `"arguments": {}` lands on the wire — fixes ADO and other strict-Zod servers rejecting empty calls before auth could fire.
- **Friendly context-overflow error**: Typed `APIError.badStatus` replaces the cryptic `NSURLErrorDomain -1011`; suggests context bump / smaller toolset when the model context is exceeded.
- **Spinner cleared on agent error**: Orphaned streaming bubble no longer keeps `GeneratingIndicator` running forever.
- **Tool-call watchdog**: GCD timer (immune to Swift cooperative-pool saturation from the SDK's hot-spinning message loop) caps tool calls at 90s, terminates the child, and detaches a `client.disconnect()` to resume the pending continuation.
- **mcp.json no longer escapes slashes**: `JSONEncoder.outputFormatting.withoutEscapingSlashes` drops the `\/` legacy HTML-safety escapes, so the file matches what Claude Desktop emits.
- **Zig 0.16 migration**: `minimum_zig_version` 0.15.2 → 0.16.0, new `main(init: std.process.Init)`, `Conn` wrapper bundling `std.Io.net.Stream` + Reader/Writer state, `std.Thread.Mutex/Condition` → `std.Io.Mutex/Condition` with explicit `io` parameter, `mod.linkFramework` for IOKit/CoreFoundation, new `src/io_util.zig` for shared timing helpers.
- **Tests**: 162 Swift unit tests (incl. real `npx -y docker-mcp` integration covering missing-command / missing-package / daemon-down / fast-fail timing, plus key-order round-trip), 210 Zig server tests.

---

## v26.4.30 — Gemma 4 Vision Fix, /v1/models Capabilities, Responses Streaming

- **Gemma 4 vision fix**: `populateUserTurnMarker` encodes the user-turn prefix from each model's `chat_template` at boot, replacing hardcoded Gemma 3 token IDs. Image tokens now insert at the right position; Gemma 4 actually sees attached images.
- **`/v1/models` capabilities**: New `capabilities` array (`chat`, `tool_use`, `streaming`, `vision`, `reasoning`, `json_schema`, `embeddings`), `input_modalities` array, and `meta.architecture`. Model id is now the directory basename so quantization variants are distinguishable.
- **Anthropic `/v1/messages` vision**: Base64 and URL image blocks accepted and routed through the SigLIP pipeline; same-message text + image bundling.
- **`/v1/responses` live streaming**: Reasoning, message, and function-call output items now stream incrementally with proper `delta` / `done` lifecycle events instead of buffering server-side.
- **Browse `extractText`**: New action runs `querySelectorAll(selector)` and returns up to 50 elements joined by `\n---\n`. `readText` now picks `<main>` / `<article>` and strips combobox menus.
- **Schema enforcement repair**: `parseTextFormat` and `parseResponseFormatAlias` accept both flat and nested-`json_schema` shapes on both `text.format` and `response_format` fields — no more silently-dropped schemas.
- **Default port 8080 → 11234**: Avoids conflict with common dev tools.
- **Orphan-process reaper**: `ServerManager` SIGTERMs leftover `mlx-serve` processes holding the target port before launching its own child.

---

## v26.4.28 — Grammar-Constrained JSON Schema Decoding

- **Token-level mask**: `response_format: json_schema` now filters every sampled token against a streaming JSON grammar derived from the schema. Non-conforming output is structurally unreachable, replacing the prior soft prompt-side instruction.
- **Supported subset**: type, properties, required, additionalProperties (defaults false), items, enum, const, min/maxLength, min/maximum, exclusive variants, regex patterns. `anyOf` / `oneOf` relaxed to "any JSON" at branch points.
- **EOS gating**: End-of-sequence masked off until the grammar reports the root value as fully parsed — eliminates premature truncation.
- **Graceful fallback**: Dead grammar states flip the mask to "everything allowed" and log a warning — request still completes.
- **Token-byte cache**: Per-id byte sequences computed once at first use (~50ms for 100k vocab), reused across requests; per-token mask building runs in 1–5ms.
- New modules: `json_schema.zig`, `regex.zig` (Thompson NFA), `json_grammar.zig`, `token_mask.zig`. New integration script `tests/test_json_schema.sh`.

---

## v26.4.27 — Multi-CLI Launcher (Claude Code / pi / OpenCode)

- **Menu-bar dropdown**: Replaces the single Launch button. Detects installed CLIs via login `zsh -l` and shows one entry per installed agent (Claude Code, pi, OpenCode).
- **Smart visibility**: Single button when one CLI is installed, dropdown for 2+, hidden when none.
- **Per-CLI config staging**: pi gets `~/.pi/agent/models.json`, OpenCode gets a dedicated `OPENCODE_CONFIG` in `$TMPDIR` so the user's main config is left untouched.
- **Real model id**: All three launches use the served model id from `/v1/models` instead of a hardcoded alias.

---

## v26.4.26 — Qwen 3.5/3.6 Tool-Call Reliability, Thinking Streaming, Swift Agent Robustness

- **Qwen 3.5/3.6 tool-call repairs**: Walks down nested-name wrappers (`{"name":{"name":{…}}}`), fixes missing `"arguments":` quote/colon, fixes unquoted-key variants. KV cache reset on identical-prompt replay.
- **Thinking-tag streaming**: Handles template-pre-injected `<think>\n` openers via 9-byte look-behind buffer; dual close-tag scan (`</think>` and `<channel|>`).
- **Swift agent watchdog**: 90s SSE inactivity watchdog around the agent-loop consumer, surfaces a clear stall error instead of hanging forever.
- **`failedRetry` flag**: Pad-retry and truncation recovery flag the streamed message instead of removing it — reasoning stays visible in the UI but excluded from API history.
- **Per-tool 30s timeout**: Browse and webSearch capped via task group; BrowserManager `evaluateJavaScript` capped at 25s.
- **Anthropic streaming parity**: Same think-tag handling applied to `/v1/messages` for Claude Code clients.

---

## v26.4.25 — Nemotron-H, LFM2, Qwen3.5 GatedDeltaNet Fixes

- **Nemotron-H Mamba2 SSM**: `A_neg` cast to float32 (BF16 broke decay precision across 42 layers); `time_step_limit` defaults to `(0.0, inf)` matching Python — no more dt clipping with stale config values.
- **Qwen 3.5 GatedDeltaNet**: Pass `ones([dk], bf16)` for parameter-free RMS norm (mlx-c rejects null); SSM state init checks `ssm_state.ctx == null` instead of the prematurely-set `initialized` flag.
- **Qwen 3.6 compatibility**: `qwen3_5_moe` model_type with both GatedDeltaNet and MoE works after the fixes.
- **Bench suite**: `bench.sh` rewrite with deterministic prompts, warmup exclusion, mlx-lm side-by-side reference, `BenchmarkLog.md` for tracking across releases.
- **CalVer auto-increment**: `build.sh` uses `YY.M.N` versioning where N is auto-incremented from the last GitHub release for the current month.

---

## v26.4.22 — Model Browser, Menu Bar Status Icon

- **HuggingFace search**: New Model Browser window with sortable columns (downloads, likes, RAM estimate, last updated), capability badges, RAM-fit indicator, architecture detection.
- **Resume support**: Downloads track `.partial` files and active downloads appear in the Downloaded tab with progress bars.
- **Vision crash fix**: Models with `vision_config` but no vision weights (e.g. text-only quantized Qwen 3.5) return `MissingVisionWeights` instead of crashing.
- **Status-tinted tray icon**: Menu-bar icon turns red when stopped, orange when starting, normal tint when running. `AppState` forwards `ServerManager.objectWillChange` so MenuBarExtra reacts.

---

## v26.4.21 — Vision Pipeline, Prefill Speedup, AgentEngine

- **Gemma 4 SigLIP vision**: Full pipeline — patch embedding, 2D RoPE, clipped linears, position pooling, embedding projection. JPEG/PNG/WebP decode via stb_image + libwebp. KV cache invalidation on image requests so vision features don't get reused.
- **3× prefill speedup (split prefill)**: Prefix pass builds the lazy graph but only KV cache entries are evaluated — MLX skips the `lm_head` matmul over the whole prompt. Last-token pass produces the logits for sampling. Matches mlx-lm; ~1,266 tok/s prefill on long prompts.
- **AgentEngine refactor**: Extracted ~350 lines of duplicated agent logic from ChatView and TestServer into a shared module — history building, tool execution, repetition tracking, overflow management.
- **Tool blocking overhaul**: Arg-aware repetition keys (`listFiles:src` and `listFiles:lib` are different), three-phase warn → soft-block → escalate, write tools exempt.
- **Image attachment UI**: Drag-drop / paste, thumbnails, `ImagePreprocessor` for vision encoder input.
- **Generating indicator**: Animated dual-arc GPU/Memory visualization with live stats and rotating whimsy text.
- **JPEG orientation fix**: `CGImageSource` with `kCGImageSourceCreateThumbnailWithTransform` so camera JPEGs aren't sideways.
- **Welcome window**: First-launch onboarding via direct NSWindow (MenuBarExtra apps don't auto-open SwiftUI scenes).

---

## v26.4.20 — Tool Reliability, Thinking+Tools, Truncation Recovery

- **Tool parameter key order**: Pre-serialized `toolDefinitionsJSON` with guaranteed `path` before `content`; request body splicing bypasses Swift's non-deterministic key ordering.
- **Truncated JSON recovery**: `extractPathFromTruncatedJSON` finds `"path":"..."` even when JSON parsing fails. Improved repair tracks unmatched `{` / `[` openers respecting quoted regions.
- **Thinking + tools fix**: Streaming and non-streaming paths both emit `reasoning_content` for tool-using turns instead of stripping silently.
- **Gemma 4 tool args**: Depth-tracked brace matching for nested objects (`{config:{...}}`) and arrays — was previously falling through to bare-value parsing.
- **Default max_tokens 8192 → 32768**: Prevents tool-call argument truncation for large file writes.
- **Max tokens warning**: `SSEEvent.maxTokensReached` surfaces a clear "Output truncated" message in chat.

---

## 2026.4.12 — MLX Core Rename, Agent Overhaul

- **Rename**: MLX Claw → MLX Core across all source, scripts, CI, docs, and bundle id (`com.dalcu.mlx-core`).
- **`listFiles` tool**: Dedicated file listing with glob and recursive traversal — system prompt steers the model toward dedicated tools instead of shell equivalents.
- **150 max iterations** (up from 30); token-aware history fitting; per-tool context caps; tool result overflow saved to `~/.mlx-serve/tool-output/` with truncated preview.
- **System prompt redesign**: Hardcoded base + additive user customization; explicit readFile → editFile workflow; structured error-recovery section.
- **Tool enhancements**: `readFile` shows `N| text` line numbers; `searchFiles` uses ripgrep with `include` / `context` / `maxResults`; `writeFile` unescapes double-escaping from smaller models.
- **API client**: Retry with exponential backoff on network errors (was single-retry).
- **Workspace context injection**: Working directory listing auto-injected each iteration so the model knows what files exist without calling `listFiles`.

---

## 2026.4.11 — Anthropic API, Claude Code, KV Cache Fix

- **`/v1/messages` Anthropic compat**: Full conversion of Anthropic content blocks (text, tool_use, tool_result, thinking), `input_schema` → `parameters`, named SSE events, stop_reason mapping.
- **Claude Code launcher**: "Launch Claude Code" button opens Terminal with `ANTHROPIC_BASE_URL` configured; binary detection via login shell PATH.
- **GPU memory preflight**: Estimates peak attention + KV memory with 20% margin, rejects with HTTP 400 instead of crashing on Metal C++ exceptions. Dynamic Metal limit from `sysctl hw.memsize`.
- **Context size auto-detection**: Default context computed from GPU memory at startup; new Auto / 16K / 32K / 64K / 128K UI presets.
- **KV cache sliding window fix**: Removed incorrect cache reset for prompts > sliding window. 3–4× faster Claude Code agent loops with shared 24K-token prefix.

---

## 2026.4.10 — Deep Agent Loop Reliability

- **KV cache reuse after tool calls**: Removed unnecessary full invalidation — `cache.truncate()` already discards stale generated-token entries. Major perf win in deep loops.
- **History windowing**: First user message pinned even when `.suffix(28)` would drop it. Progressive truncation: older tool results to 500 chars, last 2 to 2000.
- **Generation budget warning**: Logs when remaining tokens fall below 25% of `max_tokens` — flags potential argument truncation.
- **Pre-validation of required params**: Detailed error with example JSON instead of forcing the model to retry blind.
- **Browse URL auto-fix**: `BrowserManager.navigate()` prepends `https://` when scheme is missing.
- **`sampleTokenLazy` refactor**: Replaced 3 boolean ownership flags with a `current` variable pattern — fixes a memory leak when `temperature=1.0` with top-k/top-p applied.

---

## 2026.4.9 — Inference Performance Optimization

- **Submit-first pipeline**: Build and `async_eval` next step BEFORE eval'ing current token — `eval()` returns instantly. Matches mlx-lm's `_step → async_eval → y.item()` pattern.
- **Fully-lazy token pipeline**: Sampled tokens stay as lazy MLX arrays into the next forward pass — no GPU↔CPU roundtrip between decode steps.
- **JIT-compiled activations**: `mlx_compile(shapeless=true)` fuses GELU (8 ops → 1 kernel), GeGLU, and softcap.
- **GPU memory wiring**: `mlx_set_wired_limit` set to `max_recommended_working_set_size` to prevent weight paging.
- **Periodic cache clearing**: `mlx_clear_cache()` every 256 tokens reduces fragmentation.
- **Results**: Decode ~33 tok/s on Gemma-4 E4B 4-bit (M4 16GB), matching mlx-lm. Memory 4.0 GB (7% less than mlx-lm). Startup 3× faster — no Python runtime.

---

## 2026.4.6 — Gemma 4 MoE, Jinja Upgrade, Tool Calling Overhaul

- **Gemma 4 MoE (26B-A4B)**: Sigma-MoE routing, separate shared/routed expert branches, 5 feedforward norms, GeGLU activation.
- **Gemma 4 E2B/E4B**: Per-Layer Embeddings (PLE) with gated projection and per-layer input scaling. ProportionalRoPE for global attention. K=V attention. Sliding window with full prefill / windowed decode views.
- **Per-weight quantization detection**: Auto-detects quant bits per weight instead of using a global default — fixes 8-bit shared expert in a 4-bit model.
- **Jinja upgrade**: Replaced jinja.hpp with llama.cpp's Jinja engine. Fixes empty tool-call args (`{command:{}}`), missing parameter types, and broken tool-message transformation.
- **Tool calling reliability**: Gemma 4 double-brace unwrapping; full SSE arg deltas in single chunk; KV cache invalidated after tool-calling requests; user nudge after tool results for models that need it.
- **Thinking with tools**: `<|channel>thought` no longer streamed as visible content; `<|channel>` and `<channel|>` tags stripped; partial-tag detection prevents premature flushing.
- **MLX Core test API**: Port 8090 with REST endpoints (`/test/start`, `/test/chat`, `/test/agent`, etc.) for automated testing.

---

## 2026.4.5 — Prompt-Based Skills, Resumable Downloads

- **Prompt-based skills**: User-defined agent capabilities via `~/.mlx-serve/skills/*.md` with YAML frontmatter (name, description, trigger keywords).
- **Resumable downloads**: Streaming writes to `.partial` files, Range header support for resume, 3 automatic retries with backoff.
- **Disk space safety**: Pre-check available space before large downloads.
- **SkillManager**: Scans skills directory on each agent loop, re-reads when directory modification date changes.

## 2026.4.4 — KV Cache & Tool Calling Fixes

- **KV cache corruption fix**: Invalid suffix cache invalidation, SSM state reset.
- **Tool calling reliability**: Improved parsing, agent harness stability.
- **App bundle packaging**: Removed Bundle.module dependency, fixed codesigning.

## 2026.4.3 — MLX Core Major Update

- **Native tool calling UI**: 7 built-in tools (shell, readFile, writeFile, editFile, searchFiles, browse, webSearch).
- **Agent mode**: Automatic ReAct loop with tool execution and result feeding.
- **Browser integration**: WKWebView-based browsing, headless operation for background tool use.
- **Streaming chat**: SSE parsing with delta reconstruction.
- **Multi-session chat**: Persistent history with session management.

## 2026.4.2 — MLX Core Initial Release

- **Swift macOS menu bar app**: Server management, model selection, chat interface.
- **Server lifecycle**: Subprocess launch/termination with stderr capture.
- **Model discovery**: Local model scanning from `~/.mlx-serve/models/`.

## 2026.3 — Embeddings, Reasoning, Jinja

- **Embedding support**: BERT and encoder-only models via `/v1/embeddings`.
- **Reasoning budget**: `--reasoning-budget` CLI flag to limit thinking tokens.
- **Jinja_cpp integration**: Replaced vibe-based Jinja (macros caused infinite loops).
- **Qwen3.5 MoE support**: GatedDeltaNet linear attention, shared expert routing.
- **TUI status bar**: Live CPU, memory, GPU metrics.

## 2026.2 — Initial Release

- **Zig native server**: OpenAI-compatible HTTP API on Apple Silicon.
- **MLX-c FFI**: GPU-accelerated tensor operations via Apple's MLX C API.
- **Model support**: Llama 3, Mistral, Qwen 3.
- **BPE tokenizer**: SentencePiece and byte-level BPE.
- **Streaming generation**: SSE-based real-time token delivery.
- **KV cache reuse**: Prompt prefix matching across requests.
- **Sampling**: Temperature, top-p, top-k, repeat penalty.
