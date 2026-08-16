# Supported models

| Architecture | `model_type` | Examples | Chat Format | Vision |
|---|---|---|---|---|
| **Gemma 4** | `gemma4`, `gemma4_unified` | `gemma-4-e2b-it-4bit`, `gemma-4-e4b-it-8bit`, `gemma-4-26b-a4b-it-4bit`, `gemma-4-12b-unified` | Gemma turns | SigLIP (unified adds audio) |
| **Gemma 3** | `gemma3` | `gemma-3-12b-it-qat-4bit` | Gemma turns | -- |
| **DiffusionGemma** | `diffusion_gemma` | `diffusiongemma-26B-A4B-it-4bit` | Gemma turns (block diffusion) | -- |
| **Qwen 2 / 3 / 3.5 / 3.6 / 3.8** | `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5`, `qwen3_5_moe`, `qwen3_next` | `Qwen3-4B`, `Qwen3.5-4B`, `Qwen3.6-27B`, `Qwen3.6-35B-A3B`, [`Qwen3.8-27B`](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) (18.2 GB, draft head baked in, effort levels `xhigh`/`medium`/`low`) | ChatML | Qwen3-VL |
| **Muse-Glimmer** | `muse_glimmer` | Meta's Muse-Glimmer-30B ([4-bit](https://huggingface.co/ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit) / [8-bit](https://huggingface.co/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit), DFlash draft companion built in, up to 75 tok/s on M4 Max) | Harmony channels + ATEM tools | Muse ViT (images) |
| **Ling 3.0** | `bailing_hybrid` | inclusionAI Ling 3.0, e.g. `rapid-mlx/Ling-3.0-tiny-MLX-4bit` (4.2 GB, KDA + MLA hybrid MoE) | GLM tags, thinking default on | -- |
| **DeepSeek V4 Flash** | `deepseek_v4` | DeepSeek-V4-Flash-0731 (284B-A13B, 1M ctx) — **native MLX** for safetensors builds, embedded [ds4](https://github.com/antirez/ds4) for `.gguf` | DSV4 + DSML tools | -- |
| **Inkling Small** | `inkling_mm_model` | Thinking Machines Inkling Small (276B-A12B MoE, 2-bit) | role-less channel messages | -- |
| **Hunyuan 3** | `hy_v3` | `Hy3-295B-Instruct` (295B-A21B MoE, 2-bit) | Hunyuan tags | -- |
| **Laguna** | `laguna` | poolside Laguna S 2.1 / XS (117.6B-A8.5B MoE coder, nvfp4) | GLM tags, pre-opened think | -- |
| **Nemotron-H** | `nemotron_h` | Nemotron-3-Nano-4B | ChatML | -- |
| **LFM2 / LFM2.5** | `lfm2`, `lfm2_vl` | LFM2.5-2.6B (8-bit, bf16, nvfp4, mxfp4), LFM2.5-VL 3B / 1.6B | ChatML, Pythonic tool calls | SigLIP2, big images tiled |
| **Llama** | `llama` | Llama 3, Llama 3.1, Llama 3.2 | Llama-3 | -- |
| **Mistral** | `mistral` | Mistral 7B Instruct v0.3 | Mistral turns | -- |
| **Embeddings** | `bert`, `gemma3_text`, `qwen3` | bge, mxbai, EmbeddingGemma, Qwen3-Embedding (pooling read from the checkpoint) | n/a | -- |
| **Anything else as GGUF** | via embedded llama.cpp | any `.gguf` on HuggingFace | per-template | -- |

Media models live in the same registry and are classified the same way: FLUX.2, Krea-2 and Mage-Flow (image), Qwen3-TTS, Kokoro, ACE-Step and MiniMax Music 3 (speech + music), LTX-Video 2.3 / 2.5 and MiniMax-H3 (video), Hunyuan3D-2.1 (3D). A chat request naming one of them gets a 400 that names the endpoint to use instead.

Any quantized MLX model using one of the above architectures works natively. Anything else can be served as GGUF through the embedded llama.cpp engine — just pick the `.gguf` file in the Model Browser and the server auto-routes by format. Models with unsupported architectures are flagged in the Model Browser but can still be downloaded.
