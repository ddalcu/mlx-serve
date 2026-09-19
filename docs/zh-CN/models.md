[English](../models.md) · [简体中文](models.md)

# 支持的模型

| 架构 | `model_type` | 示例 | 聊天格式 | 视觉 |
|---|---|---|---|---|
| **Gemma 4** | `gemma4`, `gemma4_unified` | `gemma-4-e2b-it-4bit`、`gemma-4-e4b-it-8bit`、`gemma-4-26b-a4b-it-4bit`、`gemma-4-12b-unified` | Gemma 轮次 | SigLIP（unified 增加音频） |
| **Gemma 3** | `gemma3` | `gemma-3-12b-it-qat-4bit` | Gemma 轮次 | -- |
| **DiffusionGemma** | `diffusion_gemma` | `diffusiongemma-26B-A4B-it-4bit` | Gemma 轮次（块扩散） | -- |
| **Qwen 2 / 3 / 3.5 / 3.6 / 3.8** | `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5`, `qwen3_5_moe`, `qwen3_next`, `qwen4_exp`, `prism_hadamard_qwen35` | `Qwen3-4B`、`Qwen3.5-4B`、`Qwen3.6-27B`、`Qwen3.6-35B-A3B`、[`Qwen3.8-27B`](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit)（18.2 GB，内置草稿头，effort 档位 `xhigh`/`medium`/`low`）、[`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)（2-bit 三值 Hadamard 打包，<16 GB 可用） | ChatML | Qwen3-VL（通过 `vision_config`） |
| **Muse-Glimmer** | `muse_glimmer` | Meta 的 Muse-Glimmer-30B（[4-bit](https://huggingface.co/ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit) / [8-bit](https://huggingface.co/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit)，内置 DFlash 配套草稿模型，M4 Max 上最高 75 tok/s） | Harmony 频道 + ATEM 工具 | Muse ViT（图像） |
| **Ling 3.0** | `bailing_hybrid` | inclusionAI Ling 3.0，例如 `rapid-mlx/Ling-3.0-tiny-MLX-4bit`（4.2 GB，KDA + MLA 混合 MoE） | GLM 标签，思考默认开启 | -- |
| **DeepSeek V4 Flash** | `deepseek_v4` | DeepSeek-V4-Flash-0731（284B-A13B，1M ctx）—— safetensors 版本用**原生 MLX**，`.gguf` 用内置 [ds4](https://github.com/antirez/ds4) | DSV4 + DSML 工具 | -- |
| **Inkling Small** | `inkling_mm_model` | Thinking Machines Inkling Small（276B-A12B MoE，2-bit） | 无 role 的频道消息 | -- |
| **Hunyuan 3** | `hy_v3` | `Hy3-295B-Instruct`（295B-A21B MoE，2-bit） | Hunyuan 标签 | -- |
| **Laguna** | `laguna` | poolside Laguna S 2.1 / XS（117.6B-A8.5B MoE 编码模型，nvfp4） | GLM 标签，预先打开的 think | -- |
| **gpt-oss** | `gpt_oss` | OpenAI gpt-oss 20B-A3.6B / 120B-A5.1B MoE（[20B MXFP4-Q8](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8) / [120B MXFP4-Q8](https://huggingface.co/mlx-community/gpt-oss-120b-MXFP4-Q8)） | Harmony 频道，思考始终开启 | -- |
| **Spark-X2.5** | `spark2_5` | XHToken 星火 X2.5（1.7B / 4B，[4-bit](https://huggingface.co/abenzerps/Spark-X2.5-4B-MLX-4bit) / [8-bit](https://huggingface.co/abenzerps/Spark-X2.5-4B-MLX-8bit)） | Spark 轮次，思考默认开启 | -- |
| **K2-Horizon** | `k2_horizon` | IFM K2-Horizon dense（0.9B / 3.7B / 7B / 32B，[7B oQ6e](https://huggingface.co/mlx-community/K2-Horizon-7B-oQ6e)） | IFM 标签，思考默认开启 | -- |
| **Nemotron-H** | `nemotron_h` | Nemotron-3-Nano-4B | ChatML | -- |
| **LFM2 / LFM2.5** | `lfm2`, `lfm2_vl` | LFM2.5-2.6B（8-bit、bf16、nvfp4、mxfp4），LFM2.5-VL 3B / 1.6B | ChatML，Python 风格工具调用 | SigLIP2，大图切片 |
| **Llama** | `llama` | Llama 3、Llama 3.1、Llama 3.2 | Llama-3 | -- |
| **Mistral** | `mistral` | Mistral 7B Instruct v0.3 | Mistral 轮次 | -- |
| **嵌入** | `bert`, `gemma3_text`, `qwen3` | bge、mxbai、EmbeddingGemma、Qwen3-Embedding（池化方式从检查点读取） | n/a | -- |
| **其它一切以 GGUF 形式** | 通过内置 llama.cpp | HuggingFace 上任何 `.gguf` | 按模板 | -- |

媒体模型位于同一个注册表里，分类方式也相同：FLUX.2、Krea-2 和 Mage-Flow（图像），Qwen3-TTS、Kokoro、ACE-Step 和 MiniMax Music 3（语音 + 音乐），LTX-Video 2.3 / 2.5 和 MiniMax-H3（视频），Hunyuan3D-2.1（3D）。聊天请求里点名其中之一，会收到一个 400，并指出应改用哪个端点。

任何使用上述架构之一的量化 MLX 模型都能原生运行。其它一切都可以通过内置 llama.cpp 引擎以 GGUF 形式提供 —— 只要在模型浏览器里挑中那个 `.gguf` 文件，服务器就会按格式自动路由。架构不受支持的模型会在模型浏览器里标注出来，但仍然可以下载。
