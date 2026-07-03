# Changelog

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
