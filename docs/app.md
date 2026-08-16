# MLX Core (macOS app)

Menu-bar app that wraps the server with a full UI. [Download the latest release](https://github.com/ddalcu/mlx-serve/releases/latest) or `brew install --cask mlx-core`. Since v26.8.4 it is one window with a sidebar, content and detail column: Models, Tasks, Settings and the media generators are modes of the chat window rather than separate windows.

- **Model browser** — download from HuggingFace with resumable multi-connection transfers (up to 16 connections per file), auto-discovers LM Studio's existing model folder (`~/.lmstudio/settings.json`) and your Hugging Face cache (wherever `HF_HOME` / `HF_HUB_CACHE` point) so you don't re-download what's on disk, GGUF rows show a min–max RAM-estimate range. Multi-variant repos let you pick your quant, and **Settings ▸ Model Folders** chooses where downloads land.
- **Model switching without a restart** — picking a chat model loads it into the running server and makes it the default (`POST /v1/load-model` with `"default": true` over the API).
- **Chat interface** — multi-session chat with markdown rendering. Drop in PDFs (PDFKit-extracted) or images alongside text.
- **Agent mode** — 10 built-in tools (shell, cwd, readFile, writeFile, editFile, searchFiles, listFiles, browse, webSearch, saveMemory) with automatic tool calling loop and a per-tool approval dialog (**Allow** / **Deny** / **Always allow this session**).
- **Agents** — named assistants with their own personality, voice, model, tools, workspace and wake phrase, plus their own pinned sampling (temperature, top-p, top-k, penalties, reasoning budget). The app writes the persona prompt for you, and every way of starting a conversation can run as one: chat tabs, hands-free voice, scheduled tasks, Telegram, the Quick Launcher.
- **MCP client** — curated marketplace of stdio + HTTP MCP servers (GitHub, Azure DevOps, DBHub, Docker, Kubernetes, Playwright, Slack, Notion, Filesystem, Shell) plus your own from `~/.mlx-serve/mcp.json`.
- **Agent Sandbox** — flip one toggle and every agent shell command runs inside an isolated Linux VM built on Apple's Virtualization framework: boots in under a second, guest servers mirror to `localhost` live (an Express app on guest port 8080 is `http://localhost:8080` on your Mac), and a green shield in the toolbar shows when commands run isolated. It's a normal Linux where `apt-get install` works, with up to 4 GB of RAM committed lazily. Let the agent go wild — your Mac stays untouched.
- **⌃Space Quick Launcher** — a Spotlight-style prompt panel over any app: hit ⌃Space, ask, and the answer streams in from your local model. Follow-ups keep context; ⌘↩ hands the conversation off to the full chat window.
- **Hands-free Voice Mode** — say "Hey Loki" and just talk: on-device speech recognition (audio never leaves the Mac), spoken replies with barge-in interruption, and voice-driven agent tools — all from the menu bar with no window open. Replies speak in your pick of 54 built-in Kokoro voices (fully local, ~17× realtime, a 345 MB download, blendable) or a Qwen3-TTS clone of your own voice.
- **LAN Sharing** — share chosen models with the other Macs in your house and use theirs: peers appear automatically, shared models show up in the tray and every generation pane as "model · peer", and requests stream to the Mac that hosts the weights — chat, image, speech, music, video, and 3D alike, with models loading on demand on the host. Per-model share checkboxes in Settings; prompts sent to a shared model run on (and are visible to) the hosting Mac.
- **Telegram bridge** — message your local model from your phone: no public URL, no port-forwarding, no cloud relay. Agent tools and scheduled tasks work remotely; the bot locks to the first chat that messages it.
- **Scheduled tasks** — hand the agent a goal and a schedule in plain English ("weekdays at 8am, check my watched sites and write a briefing") and it runs unattended, with saved transcripts.
- **Document folder RAG** — attach a folder of mixed files and ask questions about them; GPU-batched embeddings index ~500 files in ~7 s, everything in memory, nothing leaves the Mac.
- **Editable system prompt + persistent memory** — `~/.mlx-serve/system-prompt.md` and `~/.mlx-serve/memory.md`.
- **Prompt-based skills** — drop `.md` files into `~/.mlx-serve/skills/` with YAML frontmatter to teach the agent custom capabilities triggered by keywords, or type `/` in the chat box to pick one and run it by name in any chat, agent mode or not.
- **Engine-aware Settings window** (Cmd+,) — every server-launch flag and per-request default, with sections that show only the knobs relevant to the engine you've loaded (MLX vs GGUF vs ds4).
- **Server management** — start/stop, live log buffer, restart-on-flag-change banner.
- **Image / Video / Music / Speech / 3D generation** — FLUX.2, Krea-2, Mage-Flow, LTX-Video 2.3 / 2.5, MiniMax-H3, ACE-Step, MiniMax Music 3, Qwen3-TTS, Kokoro and Hunyuan3D, all native via the mlx-serve zig server.

## Image / Video / Music / Speech / 3D generation

One server, five modalities — the **Image**, **Video**, **Audio** (speech + music) and **3D** Create panes run [FLUX.2](https://huggingface.co/black-forest-labs) / Krea-2 / Microsoft Mage-Flow, [LTX-Video 2.3 and 2.5](https://github.com/dgrauet/ltx-2-mlx) / [MiniMax-H3](https://huggingface.co/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit), [ACE-Step 1.5](https://huggingface.co/ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit) / MiniMax Music 3, [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) / [Kokoro-82M](https://huggingface.co/ddalcu/Kokoro-82M-MLX-Serve), and [Hunyuan3D-2.1](https://huggingface.co/ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit) natively on MLX. Click a pane, hit **Download**, generate. Drag a file onto any pane and it lands in the right slot, reference lists included. Each pane remembers your last-used model, quality, resolution, steps and seed between sessions.

The panes also list checkpoints you added yourself: anything in your model folders with a family the server can run shows under **On This Mac** with that family's controls, and the Model Browser offers community packs of those families, layout-checked before the Download button appears.

You can also **generate media straight from chat**: ask for an image, a spoken line, a music track or a short clip and it renders inline in the conversation with a progress bar, using your saved settings for that modality. Attach a photo and say "make it winter" and the edit comes back in the same thread. Double-click any chat image to open it full-size in Preview.

And it goes well beyond text-to-X:

- **Edit photos with instructions** — attach a picture, type *"make the hair blue"* or *"remove the monitor in the background"*, and FLUX.2-klein or Mage-Flow Edit changes it while keeping subject, pose, and scene intact (in-context reference conditioning — 0.97 structural correlation measured live). Mage-Flow Edit also composes multiple references ("put the object from image 2 into image 1") and does control maps, restoration and relighting from plain instructions. The source keeps its own aspect ratio, never squished. Works from the OpenAI SDK too: `client.images.edit(image=…, prompt=…)` against `POST /v1/images/edits`.
- **Image-to-image variations** — every image model (Krea-2 included) takes a source image plus a strength slider, from subtle remix to full re-imagination.
- **Animate your photos** — drop a picture into the Video pane's First-frame slot and LTX animates forward from it, starting exactly on your image.
- **Talking characters** — put spoken lines in quotes in the video prompt, attach a real speech or music clip, or type a line for Qwen3-TTS to voice — the video is generated *against* that soundtrack, performance synced, and the original audio (not a re-synthesis) lands in the mp4.
- **Clone a voice from seconds of audio** — record or pick a clip in Settings ▸ Voice and Qwen3-TTS speaks in that voice — in the AudioGen panel, in hands-free Voice Mode, everywhere.
- **Compose full music tracks** — ACE-Step 1.5 turns a style prompt (and optional lyrics) into a 48 kHz stereo track: a 30-second song renders in about 4 seconds.
- **Turn a photo into a 3D model** — Hunyuan3D-2.1 converts an image into a watertight GLB mesh, optionally with full PBR textures — drops straight into a game engine or slicer.
- **Video with its own soundtrack** — MiniMax-H3 (Hailuo 3.0) denoises the clip and a stereo soundtrack together in one pass, so the sound is produced with the video rather than dubbed on after. Describe the scene, then what you want to hear after `overall_soundscape:`. The REF2VA build builds the clip around pictures, clips or audio you attach (`<Picture 1>`, `<Video 1>`, `<Audio 1>` in the prompt); the FL2VA build takes first/last keyframes and chains windows into longer clips. **Turbo** renders in 4 steps instead of 30. It is slow either way: 1344×768 at 124 frames is about 50 minutes on an M4 Max.
- **Style LoRAs** — attach diffusers, kohya or PEFT `.safetensors` adapters to restyle FLUX, Krea, Mage-Flow, LTX or MiniMax-H3 generations at runtime. Up to 8 at once, summed rather than merged, so nothing is re-quantized and there is zero quality loss on the base weights. Each adapter runs at the strength its own file declares.

### Models

| Feature | Default | Other options | Approx. RAM |
|---|---|---|---|
| Image | FLUX.2-klein 4B 4-bit (mflux, ~5 GB pre-quantized) | FLUX.2-klein 9B (10 GB), Krea-2-Turbo, Mage-Flow Turbo / Edit 8-bit (8.5 / 9.1 GB) | 8 / 12 / 16 GB |
| Video | LTX-Video 2.5 4-bit (36 GB, bundled text encoder) | LTX-Video 2.5 8-bit (59 GB, sharper + diffusion decoder), LTX-Video 2.3 Q4 (~50 GB), MiniMax-H3 (Hailuo 3.0) 4-bit / 8-bit, video **and** matching soundtrack in one pass | LTX 24 GB RAM; H3 26 GB (40 GB) or 44 GB (69 GB) |
| Speech | Qwen3-TTS 1.7b (voice cloning) | Qwen3-TTS 0.6b, Kokoro-82M (54 voices, ~345 MB) | 8 GB RAM, ~3.5 GB first-run download |
| Music | ACE-Step 1.5 XL Turbo 8-bit (fast, 8 steps) | MiniMax Music 3 8-bit (sings your lyrics, songs up to 6 min, strongest vocals) | ACE 8 GB RAM, ~6.2 GB download; Music 3 ~20 GB RAM, 13.6 GB download |
| 3D | Hunyuan3D-2.1 8-bit (shape + PBR texture) | — | 16 GB RAM |

> The 41 GB LTX 2.3 snapshot ships **both** transformer variants (1-stage distilled + 2-stage dev, ~11 GB each) plus a 7.6 GB distillation LoRA, so you can switch between Fast/Good/Quality/Super offline without re-downloading.

> LTX-Video 2.5 brings its own text encoder, so there is no separate 8 GB download on first use. The 8-bit pack keeps detail the 4-bit one loses and adds a **Diffusion decoder** toggle (the decoder Lightricks' own published clips use, `"decoder": "diffusion"` over the API) for sharper texture and edges. The default canvas and frame ladder are sized per Mac; two-stage tiers denoise at half the chosen size and upscale.

> MiniMax Music 3 requires lyrics; structure tags like `[verse]` and `[chorus]` go on their own lines. ACE-Step's tempo, key, meter and language controls don't exist on it, so put those facts in the caption instead. The bundled **music3** skill writes the caption format the model was trained on when you ask the chat for a song.

Outputs go to `~/.mlx-serve/generations/` under per-modality, per-date folders.

> The app won't let you start a generation if there isn't enough free RAM. If the mlx-serve server is running and competing for memory, you'll be prompted to stop it first.
