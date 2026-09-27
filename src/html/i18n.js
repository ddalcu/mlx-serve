/* Localization for the built-in console.
 *
 * The English source string IS the key, mirroring the app's
 * zh-Hans.lproj/Localizable.strings: a key with no entry falls back to itself,
 * so the page is complete in English by construction and a new string is never
 * blank while it is untranslated. Injected as a runtime `{s}` arg in the
 * <head>, like app.css/app.js — index.html is a std.fmt format string, so an
 * inline script would have to double every brace it contains. */

(function () {
  var STORE_KEY = 'mlx-serve-lang'

  // Empty on purpose: identity IS the English sentence. The table exists so
  // `en` is a real language the resolver can pick, not a special case.
  var EN = {}

  // zh-Hans. Product/protocol/unit names stay Latin (mlx-serve, FLUX.2, Kokoro,
  // HTTP, JSON, tok/s, ms, MB, GPU); the rest follows the app's own wording.
  var ZH = {
    // ── Sidebar and chrome ──────────────────────────────────────────────────
    "Language": "语言",
    "Hide sidebar": "隐藏侧边栏",
    "Show sidebar": "显示侧边栏",
    "Switch light/dark theme": "切换浅色/深色主题",
    "API": "API",
    "New chat": "新建聊天",
    "Monitor": "监控",
    "Recents": "最近",
    "Models": "模型",

    // ── Chat ────────────────────────────────────────────────────────────────
    "What's on your mind?": "在想什么？",
    "Attach an image": "附加图像",
    "Ask anything, or ask for an image, a song, an edit…": "问点什么，或者让它画张图、写首歌、改改图…",
    "Model": "模型",
    "model": "模型",
    "no model": "未选择模型",
    "no chat model on this server": "此服务器上没有聊天模型",
    "Voice mode: speak, and hear the reply": "语音模式 —— 说话，并听到回复",
    "Voice mode — speak, and hear the reply": "语音模式 —— 说话，并听到回复",
    "Voice mode": "语音模式",
    "Voice": "语音",
    "Listening": "聆听中",
    "Media tools": "媒体工具",
    "Thinking": "思考",
    "Speaking": "朗读中",
    "Send": "发送",
    "Stop": "停止",
    "Runs entirely on this Mac. Models can make mistakes.": "完全在这台 Mac 上运行。模型可能会出错。",
    "Copy": "复制",
    "Copied": "已复制",
    "Copy failed": "复制失败",
    "Regenerate": "重新生成",
    "thinking": "思考",
    "thinking…": "正在思考…",
    "loading %@…": "正在加载 %@…",
    "stopped": "已停止",
    "editing image…": "正在编辑图像…",
    "microphone blocked — allow it in the browser": "麦克风被阻止 —— 请在浏览器中允许",
    "Your chats show up here.": "你的对话会显示在这里。",
    "Delete": "删除",
    "remove %@": "移除 %@",
    "image attached (not stored)": "图像已附加（未保存）",
    "download png": "下载 PNG",
    "download wav": "下载 WAV",
    "could not read %@": "无法读取 %@",
    "No chat models found. Pull one with `mlx-serve pull`, or point the server at a models directory.": "没有找到聊天模型。用 `mlx-serve pull` 拉取一个，或把服务器指向一个模型目录。",
    "No models discovered. Point the server at a models directory with --model-dir, or pull one with `mlx-serve pull`.": "没有发现任何模型。用 --model-dir 把服务器指向一个模型目录，或用 `mlx-serve pull` 拉取一个。",
    "Extended thinking": "扩展思考",
    "shows the model's reasoning": "显示模型的推理过程",
    "%@ model": "%@ 个模型",
    "%@ models": "%@ 个模型",
    "%@ loaded": "已加载 %@",
    "%@ resident": "%@ 常驻",
    "%@s prefill": "%@s 预填充",
    "request failed (HTTP %@)": "请求失败（HTTP %@）",
    "Live metrics are off. Restart the server with --metrics for decode/prefill rates, TTFT and cache hit rate here.": "实时指标已关闭。用 --metrics 重启服务器，即可在这里看到解码/预填充速度、TTFT 与缓存命中率。",
    "(code block)": "（代码块）",
    "a link": "一个链接",

    // ── Server enums the UI renders: model state (server.zig), capabilities
    // (readyCapsJson) and generation stages (every `Progress.emit` site). An
    // unrecognized value falls back to itself, so a new one is never blank.
    "ready": "就绪",
    "unloaded": "未加载",
    "error": "错误",
    "chat": "对话",
    "tool_use": "工具调用",
    "streaming": "流式",
    "vision": "视觉",
    "reasoning": "推理",
    "json_schema": "JSON 模式",
    "embeddings": "嵌入",
    "image": "图像",
    "audio": "音频",
    "music": "音乐",
    "video": "视频",
    "3d": "3D",
    "working": "处理中",
    "Generating": "正在生成",
    "Generating audio": "正在生成音频",
    "Encoding prompt": "正在编码提示词",
    "Encoding image": "正在编码图像",
    "Encoding reference voice": "正在编码参考音色",
    "Decoding image": "正在解码图像",
    "Decoding audio": "正在解码音频",
    "Decoding video": "正在解码视频",
    "Upscaling": "正在放大",
    "prefill": "预填充",
    "decode": "解码",
    "encode": "编码",
    "denoise": "去噪",
    "diffuse": "扩散",
    "frames": "帧",
    "mesh": "网格",
    "volume": "体素",
    "status": "状态",
    "paint-encode": "贴图编码",
    "paint-decode": "贴图解码",
    "paint-denoise": "贴图去噪",
    "paint-inpaint": "贴图修补",
    "paint-render": "贴图渲染",
    "paint-unwrap": "贴图展开",

    // ── Live metrics panel ──────────────────────────────────────────────────
    "Live metrics": "实时指标",
    "connecting…": "正在连接…",
    "Decode": "解码",
    "Prefill": "预填充",
    "Requests": "请求",
    "running": "运行中",
    "Avg TTFT": "平均 TTFT",
    "Cache hit rate": "缓存命中率",
    "Memory": "内存",
    "Generated": "已生成",
    "physical footprint": "物理内存占用",
    "Decode tok/s · last 60s": "解码 tok/s · 最近 60 秒",
    "Prefill tok/s · last 60s": "预填充 tok/s · 最近 60 秒",
    "— ms avg": "— ms 平均",
    "%@ ms avg": "%@ ms 平均",
    "— ms e2e": "— ms 端到端",
    "%@ ms e2e": "%@ ms 端到端",
    "0 waiting · — req/s": "0 等待中 · — req/s",
    "%@ waiting · %@ req/s": "%@ 等待中 · %@ req/s",
    "— / — queries": "— / — 次查询",
    "%@ / %@ queries": "%@ / %@ 次查询",
    "%@% tokens reused": "%@% Token 复用",
    "0 requests": "0 次请求",
    "%@ requests": "%@ 次请求",
    "prefilling": "正在预填充",
    "%@ tok/s avg · %@ ms": "%@ tok/s 平均 · %@ ms",
    "metrics disabled": "指标未启用",
    "error: %@": "错误：%@",
    "● live": "● 实时",

    // ── API reference ───────────────────────────────────────────────────────
    "Streaming and non-streaming · tool calling · JSON mode · vision (when supported)": "流式与非流式 · 工具调用 · JSON 模式 · 视觉（模型支持时）",
    "Legacy text completions": "旧的文本补全接口",
    "Stateful responses with tool calling · stream/non-stream · vision": "带工具调用的有状态响应 · 流式/非流式 · 视觉",
    "Compact a conversation into a round-trippable opaque blob": "把一段对话压缩成可往返的不透明数据块",
    "Retrieve a stored response envelope": "取回已保存的响应信封",
    "Delete a stored response": "删除已保存的响应",
    "WebSocket transport · per-connection store-false cache · sequential turns": "WebSocket 传输 · 每连接 store=false 缓存 · 顺序轮次",
    "Claude SDK / Claude Code compatible · stream & non-stream · tool use · thinking blocks": "兼容 Claude SDK / Claude Code · 流式与非流式 · 工具调用 · 思考块",
    "Ollama chat · NDJSON stream (default on) · tool calls · images · think": "Ollama 聊天 · NDJSON 流（默认开启）· 工具调用 · 图像 · <code>think</code>",
    "Ollama completion · templated or raw": "Ollama 补全 · 模板或 <code>raw</code>",
    "List local models (ollama list)": "列出本地模型（<code>ollama list</code>）",
    "Model details, template and parameters": "模型详情、模板与参数",
    "Models currently resident in memory": "当前驻留在内存中的模型",
    "Download a model from Hugging Face · NDJSON progress": "从 Hugging Face 下载模型 · NDJSON 进度",
    "Version string (clients probe this to detect an Ollama server)": "版本字符串（客户端用它探测 Ollama 服务器）",
    "Embeddings, current shape (input string or array)": "嵌入，当前形状（<code>input</code> 为字符串或数组）",
    "Embeddings, legacy shape (prompt)": "嵌入，旧版形状（<code>prompt</code>）",
    "Embeddings & utilities": "嵌入与实用工具",
    "Vector embeddings (encoder-only models)": "向量嵌入（仅编码器模型）",
    "Laya typed decisions · choice / score / noul questions over a JSON state · calibrated probabilities":
      "Laya 结构化决策 · 基于 JSON 状态的 choice / score / noul 问题 · 校准概率",
    "Tokenize a string": "对字符串分词",
    "Detokenize an id sequence": "把 id 序列还原为文本",
    "Media generation": "媒体生成",
    "FLUX.2, Krea & Mage-Flow text-to-image · img2img + instruction edit · runtime LoRA · base64 PNG": "FLUX.2、Krea 与 Mage-Flow 文生图 · 图生图 + 指令编辑 · 运行时 LoRA · base64 PNG",
    "OpenAI-compatible image editing · multipart form · one or more reference images + an instruction": "兼容 OpenAI 的图像编辑 · multipart 表单 · 一张或多张参考图 + 一条指令",
    "Qwen3-TTS (zero-shot voice cloning) or Kokoro (54 blendable voices) · WAV": "Qwen3-TTS（零样本语音克隆）或 Kokoro（54 种可混合音色）· WAV",
    "ACE-Step text-to-music · 48 kHz stereo WAV": "ACE-Step 文生音乐 · 48 kHz 立体声 WAV",
    "LTX-Video or MiniMax-H3 · text / image / audio → video with its own soundtrack · frames + PCM": "LTX-Video 或 MiniMax-H3 · 文本 / 图像 / 音频 → 自带音轨的视频 · 帧 + PCM",
    "Hunyuan3D-2.1 · one photo → GLB mesh · optional PBR texturing": "Hunyuan3D-2.1 · 一张照片 → GLB 网格 · 可选 PBR 贴图",
    "Model management": "模型管理",
    "Load a discovered model, or register + load one by absolute path": "加载已发现的模型，或用绝对路径注册并加载一个",
    "Free a model's memory now": "立即释放某个模型占用的内存",
    "Pick up models added to the model folders since startup": "拾取启动之后加入模型目录的模型",
    "Configured upstream chat providers (~/.mlx-serve/providers.json) and whether each answered its last probe": "已配置的上游聊天提供商（~/.mlx-serve/providers.json），以及每个提供商上次探测是否有响应",
    "Re-read providers.json and re-probe now": "重新读取 providers.json 并立即重新探测",
    "Discovery": "发现",
    "OpenAI models list (id, capabilities, context length) — this console's model picker": "OpenAI 模型列表（id、能力、上下文长度）—— 本控制台的模型选择器",
    "llama.cpp-style server props (chat template, memory)": "llama.cpp 风格的服务器 props（聊天模板、内存）",
    "Liveness probe": "存活探测",
    "Prometheus metrics, text exposition format (enable with --metrics)": "Prometheus 指标，文本展示格式（用 <code>--metrics</code> 启用）",
    "Metrics as JSON — drives the Monitor panel (enable with --metrics)": "以 JSON 提供指标 —— 驱动监控面板（用 <code>--metrics</code> 启用）",
    "Quick start": "快速开始",
    "your-model-id": "你的模型 id"
  }

  var TABLES = { en: EN, 'zh-Hans': ZH }
  // Whole-element keys, not fragments of one. `data-i18n` carries the English
  // source text (what collectApi() feeds the model), these carry it per slot.
  var TEXT = 'data-i18n'
  var ATTRS = {
    'data-i18n-title': 'title',
    'data-i18n-aria-label': 'aria-label',
    'data-i18n-placeholder': 'placeholder'
  }
  var SEL = '[data-i18n],[data-i18n-title],[data-i18n-aria-label],[data-i18n-placeholder]'

  function fromNavigator() {
    var langs = navigator.languages && navigator.languages.length
      ? navigator.languages
      : [navigator.language || '']
    for (var i = 0; i < langs.length; i++) {
      if (/^zh\b|^zh-/i.test(String(langs[i]))) return 'zh-Hans'
    }
    return 'en'
  }

  function stored() {
    try {
      var v = localStorage.getItem(STORE_KEY)
      return v && TABLES[v] ? v : null
    } catch (e) {
      return null
    }
  }

  var lang = stored() || fromNavigator()
  var listeners = []

  /// The key's translation, or the key itself when the table has no entry, with
  /// `%@` substituted positionally (the app's L10n.format uses the same marker).
  function t(key, params) {
    var table = TABLES[lang] || EN
    var s = table[key]
    if (s === undefined) s = key
    if (!params || !params.length) return s
    var i = 0
    return String(s).replace(/%@/g, function () { return String(params[i++]) })
  }

  function applyNode(el) {
    var key = el.getAttribute(TEXT)
    if (key) {
      var out = t(key)
      // A translation may carry the inline <code> the English row has; the rest
      // are plain sentences and must never be parsed as HTML.
      if (out.indexOf('<') >= 0) el.innerHTML = out
      else el.textContent = out
    }
    for (var a in ATTRS) {
      var src = el.getAttribute(a)
      if (src) el.setAttribute(ATTRS[a], t(src))
    }
  }

  /// Translate every marked node under `root` (default: the whole document).
  function applyMarkup(root) {
    var scope = root || document
    var nodes = []
    if (scope.querySelectorAll) nodes = scope.querySelectorAll(SEL)
    if (scope.matches && scope.matches(SEL)) nodes = [scope].concat(Array.prototype.slice.call(nodes))
    for (var i = 0; i < nodes.length; i++) applyNode(nodes[i])
  }

  function setLang(next) {
    if (!TABLES[next] || next === lang) return lang
    lang = next
    api.lang = lang
    try { localStorage.setItem(STORE_KEY, lang) } catch (e) {}
    document.documentElement.setAttribute('lang', lang)
    applyMarkup(document)
    for (var i = 0; i < listeners.length; i++) listeners[i](lang)
    return lang
  }

  var api = {
    lang: lang,
    t: t,
    setLang: setLang,
    applyMarkup: applyMarkup,
    onChange: function (cb) { if (typeof cb === 'function') listeners.push(cb) }
  }

  document.documentElement.setAttribute('lang', lang)
  // The head runs before the body exists, so the markup pass waits for it. Any
  // script that built nodes already carries its own data-i18n.
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { applyMarkup(document) })
  else applyMarkup(document)

  window.mlxI18n = api
})()
