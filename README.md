![MLX-Serve, a local LLM and media server for Apple Silicon](website/assets/mlx-serve-header.jpg)

# MLX-Serve

Run open models on your Mac: chat, coding agents, images, video, music, speech and 3D. One native app and one local server that speaks the OpenAI, Anthropic and Ollama APIs. It runs offline, needs no Python and no account, and it's MIT licensed.

[![Release](https://img.shields.io/github/v/release/ddalcu/mlx-serve?style=flat-square&color=0071e3)](https://github.com/ddalcu/mlx-serve/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/ddalcu/mlx-serve/total?style=flat-square&color=30d158)](https://github.com/ddalcu/mlx-serve/releases)
[![Stars](https://img.shields.io/github/stars/ddalcu/mlx-serve?style=flat-square&color=f7a41d)](https://github.com/ddalcu/mlx-serve/stargazers)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

**[Download for Mac](https://github.com/ddalcu/mlx-serve/releases/latest/download/MLX-Serve.dmg)** · [Homebrew](#install) · [iPhone](https://apps.apple.com/us/app/mlx-chat-local-ai/id6787603518) · [Linux](#linux-and-windows) · [mlxserve.com](https://mlxserve.com/) · [简体中文](README.zh-CN.md)

![Chatting with a local model, making a picture and coding with Claude Code, all on one Mac](website/assets/demo.gif)

## Install

**The app.** [Download MLX-Serve](https://github.com/ddalcu/mlx-serve/releases/latest/download/MLX-Serve.dmg), drag it to Applications and open it. It needs macOS 26.2 or later on Apple Silicon, suggests a model that fits your Mac, and updates itself.

**Homebrew.**

```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew trust ddalcu/mlx-serve
brew install --cask mlx-serve   # the app
brew install mlx-serve          # or only the command-line server
```

**The terminal**, if you know Ollama:

```bash
mlx-serve run gemma4          # downloads Gemma 4 E4B, serves it and chats right here
mlx-serve pull gemma4:26b     # download only, resumable
mlx-serve serve               # serve everything you downloaded; models load when asked for
```

The server listens on `http://localhost:11234`. Point any OpenAI, Anthropic or Ollama client at it, or open that address in a browser for the built-in web console.

### iPhone

[MLX-Serve](https://apps.apple.com/us/app/mlx-chat-local-ai/id6787603518) puts the same engine on your iPhone: chat with Gemma 4, clone your voice, and make pictures and music, all on the phone and offline. It needs an A17 Pro-class iPhone or newer with iOS 26.

### Linux and Windows

Every release has Linux downloads for NVIDIA GPUs: x64 for RTX 20 to 50 cards, arm64 for DGX Spark, GH200 and Jetson Thor. On Windows with an NVIDIA card, one script builds it. Details are in [docs/building.md](docs/building.md#linux-and-windows-nvidia).

## What it does

- **Coding agents on a local model.** `mlx-serve launch claude` starts Claude Code against your Mac, set up with the model's real context window. Codex, OpenCode, pi, Grok and others work the same way. [Integrations](docs/integrations.md)
- **Pictures, video, music, speech and 3D.** FLUX.2, Qwen-Image, Krea, MOSS-Image, LTX-Video, MiniMax-H3, ACE-Step, YuE2, Qwen3-TTS with voice cloning, and Hunyuan3D, in the app or over HTTP. [The app](docs/app.md)
- **Transcripts and meeting notes.** Speech to text in 30 languages, with word timings and who said what. The menu bar records a call and writes the notes. [API](docs/api.md#other-endpoints)
- **Models bigger than your Mac.** GLM-5.3-Flash, MiMo-V2.6-Flash and DeepSeek-V4.1-Flash run on a 128 GB Mac by reading their experts from the SSD. Two Macs on a Thunderbolt 5 cable can run one model together, and Macs on your network can use each other's models. [CLI](docs/cli.md) · [FAQ](docs/faq.md#can-my-macs-share-models-over-the-network)
- **An assistant that lives on your Mac.** Chat with your documents, talk to it hands-free, give agents a sandboxed shell and MCP tools, or schedule tasks in plain English. [The app](docs/app.md)
- **The APIs your tools already use.** OpenAI chat, completions and Responses, Anthropic Messages and Ollama, all on one port, with streaming, tool calls, JSON schemas, vision and reasoning. [API](docs/api.md)

## What can my Mac run?

The app suggests a first model for your Mac's memory:

| Memory | Suggested model |
|---|---|
| 16 GB | Gemma 4 E4B |
| 18–24 GB | Gemma 4 12B |
| 32 GB | Qwen 3.8 27B |
| 36 GB | Qwen 3.8 27B 6-bit |
| 48–64 GB | Qwen 3.8 27B 8-bit |
| 96 GB and up | Qwen3.8-Flash-Next (125B, about 6B active per token) |
| 128 GB | Also GLM-5.3-Flash, MiMo-V2.6-Flash and DeepSeek-V4.1-Flash, streamed from the SSD |
| 256 GB | Also GLM-5.3 (744B), streamed from the SSD |

Thousands more run too: any MLX model with a supported architecture, and any GGUF. The [tier list](https://mlxserve.com/llm-tier-list/) ranks the popular ones, and [docs/models.md](docs/models.md) lists every architecture.

## Speed

Decode speed in v26.10.3 on an M5 Ultra with 256 GB, on a code prompt, with speculative decoding on:

| Model | Tokens per second |
|---|---:|
| Qwen3.6 35B-A3B, 4-bit | 424 |
| Qwen3.8 27B, 4-bit | 242 |
| Gemma 4 E4B, 4-bit | 199 |
| Gemma 4 26B-A4B, 4-bit | 186 |
| Qwen3.8-Flash-Next, 4.7 bits | 175 |

On an M4 Max, the same models decode 10% to 51% faster than they did in v26.8. Every release is measured the same way in [benchmarks.md](benchmarks.md); speculative decoding and tuning are in [docs/performance.md](docs/performance.md).

## Docs

- [Integrations](docs/integrations.md): Claude Code, Codex, OpenCode, pi, Zed and more
- [HTTP API](docs/api.md): OpenAI, Anthropic, Ollama and the media endpoints
- [CLI and server flags](docs/cli.md)
- [The Mac app](docs/app.md), with the media tour
- [Supported models](docs/models.md)
- Write a plugin without rebuilding the server: [model](docs/plugin-model.md), [media](docs/plugin-media.md), [quant](docs/plugin-quant.md)
- [Performance and tuning](docs/performance.md)
- [Building from source](docs/building.md) and [NVIDIA GPUs](docs/CUDA.md)
- [FAQ](docs/faq.md) and the [changelog](CHANGELOG.md)

## Build from source

You need Xcode 26.2 or later and [Homebrew](https://brew.sh):

```bash
git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
./app/build.sh
```

The script fetches the pinned Zig, builds MLX and llama.cpp, and signs the app ad hoc. There's no Python anywhere in the build. Server-only builds are in [docs/building.md](docs/building.md).

## Community

Questions, ideas and things you made: the [community chat on X](https://x.com/i/chat/group_join/g2108335256541741397/K38XS6aGRh) and [GitHub issues](https://github.com/ddalcu/mlx-serve/issues). Builds and deep dives go out on [YouTube](https://www.youtube.com/@DavidDalcu) and [X](https://x.com/ddalcu).

## Thank you

Thank you to the sponsors who back mlx-serve: @jcprichard, @skudinov, @davidfekke, @lojza3d, @cpko, @d-b, @alinselea, Johnny Dang and @R0xr1te. If it's useful to you, you can support it on [GitHub Sponsors](https://github.com/sponsors/ddalcu) or [Buy Me a Coffee](https://buymeacoffee.com/ddalcu). Nothing is paywalled: it's MIT now and stays MIT.

Thank you also to everyone who sends code, reports and ideas. The [contributors page](https://github.com/ddalcu/mlx-serve/graphs/contributors) shows the code side, and the [changelog](CHANGELOG.md) thanks people release by release.

## Acknowledgements

mlx-serve builds on [MLX](https://github.com/ml-explore/mlx), [mlx-c](https://github.com/ml-explore/mlx-c), [mlx-lm](https://github.com/ml-explore/mlx-lm), [llama.cpp](https://github.com/ggerganov/llama.cpp), [antirez/ds4](https://github.com/antirez/ds4), [mlx-stream](https://github.com/davidtai/mlx-stream), [Sushi](https://github.com/beamivalice/sushi), [jinja.cpp](https://github.com/wangzhaode/jinja.cpp), [nlohmann/json](https://github.com/nlohmann/json), [stb_image](https://github.com/nothings/stb), [libwebp](https://chromium.googlesource.com/webm/libwebp), [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) and [Zig](https://ziglang.org). It runs the open models their authors published, and the app uses the [Anthropic](https://github.com/anthropics/swift-sdk) and [MCP](https://github.com/modelcontextprotocol/swift-sdk) Swift SDKs.

Licenses and required notices are in [NOTICE](NOTICE). If we missed you, please open a PR.

## Star history

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/star-history-dark.svg">
  <img alt="GitHub star history for ddalcu/mlx-serve" src="docs/star-history.svg">
</picture>

<!-- regenerate with: python3 scripts/star-chart.py -->

[![ddalcu%2Fmlx-serve | Trendshift](https://trendshift.io/api/badge/repositories/43025)](https://trendshift.io/repositories/43025)

If mlx-serve is useful to you, a star helps other people find it.

## License

MIT, see [LICENSE](LICENSE). Bundled third-party code keeps its own license: [NOTICE](NOTICE) lists it with the required attributions, and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) has the Apache license text.
