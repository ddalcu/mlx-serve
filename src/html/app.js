// Built-in console for the index page (`GET /`).
//
// Injected into index.html as a RUNTIME `{s}` arg — index.html is a std.fmt
// format string, so anything living inside it has to double every brace. Same
// reason metrics.js is a separate file: std.fmt does not re-parse runtime args,
// so this file can be ordinary JavaScript.
//
// A sidebar with three destinations — New chat, Monitor, API — plus Recents,
// which is chat history in localStorage. Chat opens simple (a greeting and a
// centred composer) and becomes a transcript once you send something; it is
// ONE composer element in two layouts, not two composers.
//
// There is no Images or Audio destination: media is something you ASK for. The
// chat carries a tool per modality this server can serve, so "generate an image
// of a fox", "make it winter" on an attached photo and "write me a lo-fi track"
// are chat turns that end in a picture or a player.
//
// Everything the page decides — which models can serve what, how an SSE byte
// stream is cut into events, which endpoint a tool call becomes, what gets
// persisted, how markdown is rendered — is a pure function exported on
// `globalThis.__mlxConsole` and unit-tested by tests/html_console_test.mjs. The
// DOM wiring below is guarded on `typeof document`, so node sees only helpers.

(function () {
  'use strict';

  // ══ Pure helpers (tested) ═════════════════════════════════════════════════

  function capsOf(m) {
    // Unloaded stubs whose config.json couldn't be read ship no `capabilities`
    // key at all, so this can never assume an array.
    return m && Array.isArray(m.capabilities) ? m.capabilities : [];
  }

  /// Models that can serve a given kind of work. `speech` excludes music
  /// backends: they advertise BOTH "audio" and "music" (the additive rule in
  /// readyCapsJson), and posting one to /v1/audio/speech is an honest 400.
  function pickModels(models, kind) {
    if (!Array.isArray(models)) return [];
    return models.filter(function (m) {
      var c = capsOf(m);
      if (kind === 'chat') return c.indexOf('chat') >= 0;
      if (kind === 'image') return c.indexOf('image') >= 0;
      if (kind === 'speech') return c.indexOf('audio') >= 0 && c.indexOf('music') < 0;
      if (kind === 'music') return c.indexOf('music') >= 0;
      return false;
    });
  }

  function modelLabel(m) {
    var state = m && m.state ? m.state : (m && m.loaded ? 'ready' : 'unloaded');
    var size = formatBytes(m && (m.bytes_resident || m.bytes_on_disk));
    return m.id + (size === '—' ? '' : '  ·  ' + size) + '  ·  ' + state;
  }

  /// The composer pill has room for a name, not an org path.
  function shortModelName(id) {
    if (!id) return 'no model';
    var s = String(id);
    var slash = s.lastIndexOf('/');
    return slash >= 0 ? s.slice(slash + 1) : s;
  }

  /// One true line under a model's name in the picker. No invented tiers —
  /// size, vision, and whether it is resident are things we actually know.
  function modelSubtitle(m) {
    var bits = [];
    var size = formatBytes(m && (m.bytes_resident || m.bytes_on_disk));
    if (size !== '—') bits.push(size);
    if (capsOf(m).indexOf('vision') >= 0) bits.push('vision');
    bits.push((m && m.state) || 'unloaded');
    return bits.join(' · ');
  }

  // ── Media as tools ────────────────────────────────────────────────────────

  /// Candidate ids for a modality, most-likely-to-work first: already resident
  /// (free, and provably loadable), then untried, then one whose last load
  /// FAILED. Live, an incomplete Qwen3-TTS download sorted first and every
  /// "say this out loud" burned a load attempt on it before retrying its
  /// sibling. Only the tool path reorders — `pickModels` keeps discovery order
  /// for the picker, which refreshes every 15s and would otherwise shuffle
  /// under the cursor.
  function rankedIds(models, kind) {
    var rank = function (m) {
      if (m.state === 'ready') return 0;
      if (m.state === 'error') return 3;
      // No weights on disk. Discovery sums the checkpoint's *.safetensors, so
      // a null size is an incomplete download — config and tokenizer present,
      // shards missing — which is exactly what the broken Qwen3-TTS bf16 dir
      // reported while its healthy sibling reported 2.3 GB. It cannot load, so
      // it must never be the first pick. Soft: if nothing is sized, nothing
      // changes.
      if (!m.bytes_resident && !m.bytes_on_disk) return 2;
      return 1;
    };
    return pickModels(models, kind)
      .map(function (m, i) { return { m: m, i: i }; })
      .sort(function (a, b) { return rank(a.m) - rank(b.m) || a.i - b.i; })
      .map(function (e) { return e.m.id; });
  }

  /// Image models that can plausibly EDIT, best first.
  ///
  /// Edit capability is not API-visible: Mage-Flow-Turbo and Mage-Flow-Edit-Turbo
  /// both report capabilities:["image"] and ship byte-identical configs, so the
  /// SERVER itself gates on the directory name (`mage_flow.dirIsEdit`). We mirror
  /// that rule. FLUX.2 has trained edit capability, so it's the fallback. When
  /// nothing looks edit-capable we return every image model rather than an empty
  /// list, so the user gets the server's honest 400 instead of "no image model".
  ///
  /// ONE list feeding both the tool's `model` enum and the resolution: offering
  /// a txt2img checkpoint as a choice for editing is offering a guaranteed 400,
  /// and live the model duly picked one out of the enum.
  function editableIds(models) {
    var ids = rankedIds(models, 'image');
    var isMageEdit = function (id) { return /mage-?flow-edit/i.test(id); };
    var isFlux = function (id) { return /flux/i.test(id); };
    var mf = ids.filter(isMageEdit);
    if (mf.length) return mf.concat(ids.filter(function (id) { return !isMageEdit(id) && isFlux(id); }));
    var fx = ids.filter(isFlux);
    return fx.length ? fx : ids;
  }

  function toolDef(name, description, properties, required) {
    return {
      type: 'function',
      function: {
        name: name,
        description: description,
        parameters: { type: 'object', properties: properties, required: required },
      },
    };
  }

  function modelArg(ids, what) {
    return { type: 'string', enum: ids, description: 'Which ' + what + ' model to use. Omit unless the user named one.' };
  }

  /// One tool per modality this server can ACTUALLY serve. Offering
  /// generate_music on a box with no music checkpoint just teaches the model
  /// to promise a song and hand back a 400.
  function mediaTools(models) {
    var tools = [];
    var img = rankedIds(models, 'image');
    var speech = rankedIds(models, 'speech');
    var music = rankedIds(models, 'music');

    if (img.length) {
      tools.push(toolDef('generate_image',
        'Generate a NEW image from a text description and show it to the user. ' +
        'ONLY when the user wants a picture made. A question that merely mentions ' +
        'images — "which endpoint edits an image?", "what fields does it take?", ' +
        '"how does image generation work?" — is answered in text, with no tool call.',
        {
          prompt: { type: 'string', description: 'What to draw. Be visually specific; expand a terse request into a full scene description.' },
          size: { type: 'string', description: 'WIDTHxHEIGHT, e.g. 1024x1024 or 1280x720. Omit for square.' },
          model: modelArg(img, 'image'),
        }, ['prompt']));
      tools.push(toolDef('edit_image',
        'Edit an EXISTING image and show the result. Use this for "make it winter", ' +
        '"give him a hat", "remove the background". The picture the user attached — or ' +
        'the last one generated in this conversation — is passed in for you ' +
        'automatically; you do not pass the image itself, only the instruction. You CAN ' +
        'do this: never tell the user you are unable to edit an existing image. ' +
        'ONLY when the user wants a picture changed — a question ABOUT editing or about ' +
        'the edit endpoint is answered in text, with no tool call.',
        {
          prompt: { type: 'string', description: 'The change to make, as an instruction. Do not describe the whole picture, only the edit.' },
          model: modelArg(editableIds(models), 'image-editing'),
        }, ['prompt']));
    }
    if (speech.length) {
      tools.push(toolDef('generate_speech',
        'Speak text aloud and give the user an audio player.',
        {
          text: { type: 'string', description: 'Exactly what should be said.' },
          model: modelArg(speech, 'speech'),
        }, ['text']));
    }
    if (music.length) {
      tools.push(toolDef('generate_music',
        'Compose a piece of music and give the user an audio player.',
        {
          prompt: { type: 'string', description: 'Style, genre, mood and instrumentation.' },
          lyrics: { type: 'string', description: 'Lyrics to sing. Omit and set instrumental for a wordless track.' },
          instrumental: { type: 'boolean', description: 'True for a wordless track. Never send lyrics with it.' },
          duration_seconds: { type: 'integer', description: 'Length in seconds, 10 to 600. Omit for 60.' },
          model: modelArg(music, 'music'),
        }, ['prompt']));
    }
    return tools;
  }

  /// Merge streamed `delta.tool_calls` fragments. Our server sends the full
  /// arguments in ONE delta, but the wire format allows any split.
  function accumulateToolCalls(acc, deltas) {
    var out = (acc || []).map(function (c) {
      return { id: c.id, name: c.name, arguments: c.arguments };
    });
    for (var i = 0; i < (deltas || []).length; i++) {
      var d = deltas[i] || {};
      var idx = typeof d.index === 'number' ? d.index : out.length;
      while (out.length <= idx) out.push({ id: '', name: '', arguments: '' });
      if (d.id) out[idx].id = d.id;
      var f = d.function || {};
      if (f.name) out[idx].name = f.name;
      if (typeof f.arguments === 'string') out[idx].arguments += f.arguments;
    }
    return out;
  }

  /// Turn a tool call into the request we'd actually send, or an error string
  /// the user (and the model) can act on. Pure: `ctx.refs` is however many
  /// reference images are on hand, opaque to this function.
  function toolInvocation(call, ctx) {
    var args = (call && call.args) || {};
    var models = (ctx && ctx.models) || [];
    var refs = (ctx && ctx.refs) || [];

    // ONE media generation per user turn. Live, a 2B model answered "generate
    // an image of a fox" by generating the fox and then inventing three more
    // edits nobody asked for — four GPU generations, tens of seconds each, off
    // one sentence. A round cap doesn't bound that (every round is another
    // picture); a budget does. Refuse in words, because a model that gets
    // silence just calls again.
    if ((ctx && ctx.mediaUsed) > 0) {
      return { error: 'Refused: already produced one result for this request. Tell the user what you made in one short sentence and let them ask for the next change.' };
    }

    function idsFor(kind) { return rankedIds(models, kind); }

    function resolveFrom(ids, label) {
      if (!ids.length) return { error: 'this server has no ' + label + ' model installed' };
      var want = args.model;
      if (want && ids.indexOf(want) >= 0) return { id: want };
      // A model that invents an id, or names the chat model it is running as,
      // must not have that id posted to a media endpoint. Fall back, but SAY
      // so — a silent substitution is the no-op class we avoid everywhere.
      if (want) {
        return { id: ids[0], note: '"' + want + '" is not a ' + label + ' model on this server — used ' + ids[0] + ' instead' };
      }
      return { id: ids[0] };
    }

    function plan(pick, extra) {
      if (pick.error) return { error: pick.error };
      if (pick.note) extra.note = pick.note;
      return extra;
    }

    switch (call && call.name) {
      case 'generate_image': {
        var pi = resolveFrom(idsFor('image'), 'image');
        if (pi.error) return { error: pi.error };
        return plan(pi, {
          kind: 'image',
          path: '/v1/images/generations',
          body: imageBody({ model: pi.id, prompt: args.prompt, size: args.size, steps: null, seed: null }),
        });
      }
      case 'edit_image': {
        // Same list the tool's `model` enum was built from, so the model cannot
        // choose a checkpoint that is guaranteed to 400.
        var pe = resolveFrom(editableIds(models), 'image-editing');
        if (pe.error) return { error: pe.error };
        if (!refs.length) {
          return { error: 'there is no image to edit — attach one with +, or generate one first' };
        }
        return plan(pe, {
          kind: 'image',
          path: '/v1/images/edits',
          fields: editFields({ model: pe.id, prompt: args.prompt, size: args.size, files: [] }),
          refs: refs,
        });
      }
      case 'generate_speech': {
        var ps = resolveFrom(idsFor('speech'), 'speech');
        if (ps.error) return { error: ps.error };
        return plan(ps, {
          kind: 'audio',
          path: '/v1/audio/speech',
          body: speechBody({ model: ps.id, text: args.text }),
        });
      }
      case 'generate_music': {
        var pm = resolveFrom(idsFor('music'), 'music');
        if (pm.error) return { error: pm.error };
        return plan(pm, {
          kind: 'audio',
          path: '/v1/audio/music-generations',
          body: musicBody({
            model: pm.id, prompt: args.prompt, lyrics: args.lyrics,
            instrumental: args.instrumental === true, duration: args.duration_seconds,
          }),
        });
      }
      default:
        return { error: 'no such tool "' + (call && call.name) + '" — answer in text instead' };
    }
  }

  // ── The console's own system prompt ───────────────────────────────────────

  function apiReferenceText(entries) {
    return (entries || []).map(function (e) {
      return e.method + ' ' + e.path + ' — ' + e.desc;
    }).join('\n');
  }

  function modelInventoryText(models) {
    return (models || []).map(function (m) {
      var caps = capsOf(m).join(', ') || 'unknown';
      var size = formatBytes(m.bytes_resident || m.bytes_on_disk);
      return '- ' + m.id + ' [' + caps + '] ' + (m.state || 'unloaded') +
        (size === '—' ? '' : ', ' + size);
    }).join('\n');
  }

  /// The request shapes people actually ask about. The endpoint list alone is
  /// a set of paths, and a model asked "how do I edit an image?" will fill the
  /// gap itself — live it produced `-F "ref1=<base64>"` against
  /// `https://your-ollama-ip-address/api/v1/images/edits`, every part of which
  /// was invented. Keep this short, and true.
  var REQUEST_FIELDS = [
    'Request fields (JSON bodies unless noted):',
    'POST /v1/chat/completions — model, messages[{role,content}], stream, max_tokens, temperature, top_p, top_k, tools, tool_choice, response_format, reasoning_effort or enable_thinking, stream_options.include_usage.',
    'POST /v1/images/generations — model, prompt, size ("1024x1024"), steps, seed, stream; mode:"edit"|"variation" with image (base64) and ref_images[] for editing; returns {data:[{b64_json}]}.',
    'POST /v1/images/edits — multipart/form-data, NOT JSON. Accepted fields: model, prompt, image[] (repeat the field once per reference file), size. REJECTED with a 400, never list these as options: mask (the editors are maskless), n greater than 1, response_format "url", any output_format other than png, stream.',
    'POST /v1/audio/speech — model, input, optional ref_audio (base64 WAV) to clone a voice, stream; returns audio/wav bytes.',
    'POST /v1/audio/music-generations — model, prompt (style/genre/mood, required), lyrics, duration_seconds (10-600), vocal_language, bpm, seed, stream; returns audio/wav bytes.',
    'POST /v1/video/generations — model, prompt, width, height, num_frames, steps, seed, stream; optional preview, preview_frames, preview_max_side (opt-in JPEG on each SSE progress event); LTX: pipeline, first_frame_image, last_frame_image, audio, cfg_scale, stg_scale; H3: turbo, fast, chain_windows, first_frame_image / last_frame_image or ref_images / ref_videos / ref_audios.',
    'POST /v1/embeddings — model, input (string or array), optional dimensions.',
    'POST /v1/load-model and /v1/unload-model — model (a discovered id, or an absolute path to register one).',
    'Media endpoints with stream:true emit SSE {"type":"progress"|"complete"|"error"} instead of a JSON body. Video progress may include preview (JPEG b64) when preview:true.',
  ].join('\n');

  /// The one system message, built from what this server actually has. Not a
  /// user-editable box: it is what makes the assistant able to run the media
  /// tools and answer questions about the API it is running inside.
  function systemPrompt(o) {
    var models = (o && o.models) || [];
    var api = (o && o.api) || [];
    var tools = mediaTools(models);
    var parts = [];
    parts.push(
      'You are the assistant built into the mlx-serve web console. mlx-serve is a native ' +
      'MLX inference server for Apple Silicon, running on this machine and serving ' +
      'OpenAI-, Anthropic- and Ollama-compatible HTTP APIs. Be concise and concrete. ' +
      'Format answers in markdown.');
    if (tools.length) {
      var names = tools.map(function (t) { return t.function.name; }).join(', ');
      parts.push(
        'You can produce media yourself by calling these tools: ' + names + '. ' +
        'When the user asks for a picture, an edit, speech or music, CALL the tool — do ' +
        'not describe what they should do instead, and never claim to have made something ' +
        'you did not call a tool for. Expand a terse request into a good prompt.\n' +
        'Call ONE tool per user message. Each call runs a real generation on this Mac and ' +
        'takes real time, so never invent extra variations, edits or follow-ups the user ' +
        'did not ask for. When the tool returns, the result is ALREADY displayed — reply ' +
        'with one short sentence and stop. Do not write markdown image links or made-up ' +
        'file names; there is no file.\n' +
        'Only call a tool when the user wants a picture, an edit, speech or music PRODUCED. ' +
        'Questions — about the API, an endpoint, a curl command, which models are ' +
        'installed, how something works — are answered in TEXT with no tool call at all. ' +
        '"Show me a curl for the edit endpoint" wants words, not a picture.');
    }
    if (models.length) {
      parts.push('Models installed on this server:\n' + modelInventoryText(models));
    }
    if (api.length) {
      var base = (o && o.origin) || '';
      parts.push(
        'HTTP API this server exposes' + (base ? ', base URL ' + base : '') + ':\n' +
        apiReferenceText(api) + '\n\n' + REQUEST_FIELDS + '\n\n' +
        'Answer API questions from the two lists above and use ' + (base || 'this server') +
        ' as the host in examples. Never invent an endpoint, a host or a parameter you ' +
        'cannot see there; if a detail is not listed, say so instead of guessing.');
    }
    return parts.join('\n\n');
  }

  // ── Chat history ──────────────────────────────────────────────────────────

  /// A conversation's name in Recents: the first thing the user said.
  function chatTitle(content) {
    var text = '';
    if (typeof content === 'string') text = content;
    else if (Array.isArray(content)) {
      for (var i = 0; i < content.length; i++) {
        if (content[i] && content[i].type === 'text') { text = content[i].text || ''; break; }
      }
    }
    text = String(text || '').replace(/\s+/g, ' ').trim();
    if (!text) return 'New chat';
    return text.length > 48 ? text.slice(0, 48).trim() + '…' : text;
  }

  /// History without the image payloads. One 1024² PNG is ~1.5 MB of base64
  /// and localStorage gives us ~5 MB for everything, so persisting a single
  /// image conversation would evict every other one. The bytes live in the DOM;
  /// the stored turn keeps a marker in their place. The agent STRUCTURE
  /// (tool_calls, tool results) survives, or a reloaded chat can't be continued.
  function storableTurns(turns) {
    return (turns || []).map(function (t) {
      var out = { role: t.role };
      if (Array.isArray(t.content)) {
        out.content = t.content.map(function (p) {
          return p && p.type === 'image_url' ? { type: 'image_omitted' } : p;
        });
      } else {
        out.content = t.content;
      }
      if (t.tool_calls) out.tool_calls = t.tool_calls;
      if (t.tool_call_id) out.tool_call_id = t.tool_call_id;
      if (t.name) out.name = t.name;
      return out;
    });
  }

  function historyUpsert(list, convo, opts) {
    var o = opts || {};
    var maxItems = o.maxItems || 40;
    var maxBytes = o.maxBytes || 1500000;
    var out = (list || []).filter(function (c) { return c.id !== convo.id; });
    out.unshift(convo);
    if (out.length > maxItems) out = out.slice(0, maxItems);
    // Quota safety: drop the oldest until the whole store fits.
    while (out.length > 1 && JSON.stringify(out).length > maxBytes) out.pop();
    return out;
  }

  function historyRemove(list, id) {
    return (list || []).filter(function (c) { return c.id !== id; });
  }

  // ── Markdown ──────────────────────────────────────────────────────────────
  // Model replies arrive as markdown and read as noise unrendered. EVERYTHING
  // here is built from escaped input — the text is model output and may be
  // quoting a user, so it is never trusted, not even inside a fence.

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /// Inline spans over ALREADY-ESCAPED text.
  function renderInline(s) {
    var held = [];
    // Code spans first, so their contents don't get further formatting.
    var out = s.replace(/`([^`]+)`/g, function (_, code) {
      // Placeholder that CANNOT occur naturally: the text is already escaped,
      // so a raw '<' can only be one we put there ourselves.
      return '<c' + (held.push('<code>' + code + '</code>') - 1) + '>';
    });
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, text, url) {
      // http(s) only. The url is escaped already, so a quote can't break out of
      // the attribute — but `javascript:` would still be a live href, so the
      // scheme is checked, and anything else degrades to plain text.
      if (!/^https?:\/\//i.test(url)) return text;
      return '<a href="' + url + '" target="_blank" rel="noreferrer">' + text + '</a>';
    });
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/(^|[^*\w])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    out = out.replace(/(^|[^_\w])_([^_\n]+)_/g, '$1<em>$2</em>');
    return out.replace(/<c(\d+)>/g, function (_, i) { return held[+i]; });
  }

  var BLOCK_START = /^\s*(```|#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s)/;

  function renderMarkdown(text) {
    var lines = escapeHtml(text).split('\n');
    var html = [];
    var i = 0;
    while (i < lines.length) {
      var line = lines[i];

      if (/^\s*```/.test(line)) {
        var buf = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
        i++;
        html.push('<pre><code>' + buf.join('\n') + '</code></pre>');
        continue;
      }
      var h = /^\s*(#{1,6})\s+(.*)$/.exec(line);
      if (h) {
        var lvl = Math.min(h[1].length, 3);
        html.push('<h' + lvl + '>' + renderInline(h[2].trim()) + '</h' + lvl + '>');
        i++;
        continue;
      }
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { html.push('<hr>'); i++; continue; }

      var items, m;
      if (/^\s*[-*+]\s+/.test(line)) {
        items = [];
        while (i < lines.length && (m = /^\s*[-*+]\s+(.*)$/.exec(lines[i]))) { items.push('<li>' + renderInline(m[1]) + '</li>'); i++; }
        html.push('<ul>' + items.join('') + '</ul>');
        continue;
      }
      if (/^\s*\d+[.)]\s+/.test(line)) {
        items = [];
        while (i < lines.length && (m = /^\s*\d+[.)]\s+(.*)$/.exec(lines[i]))) { items.push('<li>' + renderInline(m[1]) + '</li>'); i++; }
        html.push('<ol>' + items.join('') + '</ol>');
        continue;
      }
      if (!line.trim()) { i++; continue; }

      var para = [];
      while (i < lines.length && lines[i].trim() && !BLOCK_START.test(lines[i])) para.push(lines[i++]);
      html.push('<p>' + renderInline(para.join('\n')).replace(/\n/g, '<br>') + '</p>');
    }
    return html.join('\n');
  }

  // ── SSE ───────────────────────────────────────────────────────────────────

  /// One SSE frame's `data` payload, or null for a comment/keepalive frame.
  function frameData(frame) {
    var lines = frame.split(/\r?\n/);
    var parts = [];
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      // `: keepalive` — the server emits one on any stream that has gone quiet
      // for 5s. It is not data; JSON.parse on it would kill the stream.
      if (!line || line.charAt(0) === ':') continue;
      var c = line.indexOf(':');
      var field = c < 0 ? line : line.slice(0, c);
      if (field !== 'data') continue;
      var v = c < 0 ? '' : line.slice(c + 1);
      if (v.charAt(0) === ' ') v = v.slice(1);
      parts.push(v);
    }
    return parts.length ? parts.join('\n') : null;
  }

  /// Feed a stream chunk in, get complete events out plus whatever is still
  /// half-arrived. A ReadableStream cuts at arbitrary byte boundaries: frames
  /// routinely split across chunks and arrive several to a chunk.
  function sseFeed(buffer, chunk) {
    var buf = (buffer || '') + (chunk || '');
    var events = [];
    for (;;) {
      var i1 = buf.indexOf('\n\n');
      var i2 = buf.indexOf('\r\n\r\n');
      if (i1 < 0 && i2 < 0) break;
      var at, len;
      if (i2 >= 0 && (i1 < 0 || i2 < i1)) { at = i2; len = 4; } else { at = i1; len = 2; }
      var data = frameData(buf.slice(0, at));
      buf = buf.slice(at + len);
      if (data !== null) events.push(data);
    }
    return { events: events, rest: buf };
  }

  /// What one chat SSE frame means. Absent keys stay absent so a caller can
  /// tell "no content this frame" from "empty content".
  function chatDelta(obj) {
    var out = {};
    if (!obj || typeof obj !== 'object') return out;
    var ch = Array.isArray(obj.choices) ? obj.choices[0] : null;
    var d = ch && ch.delta ? ch.delta : null;
    if (d) {
      if (typeof d.content === 'string' && d.content.length) out.content = d.content;
      if (typeof d.reasoning_content === 'string' && d.reasoning_content.length) {
        out.reasoning = d.reasoning_content;
      }
      if (Array.isArray(d.tool_calls) && d.tool_calls.length) out.toolCalls = d.tool_calls;
    }
    if (ch && ch.finish_reason) out.finish = ch.finish_reason;
    if (obj.usage) out.usage = obj.usage;
    if (obj.timings) out.timings = obj.timings;
    return out;
  }

  // ── Request construction ──────────────────────────────────────────────────

  function isNum(v) {
    return v !== null && v !== undefined && v !== '' && isFinite(Number(v));
  }

  function chatBody(o) {
    var messages = [];
    var sys = (o.system || '').trim();
    if (sys) messages.push({ role: 'system', content: sys });
    var turns = o.turns || [];
    for (var i = 0; i < turns.length; i++) {
      var t = turns[i];
      var m = { role: t.role, content: t.content };
      // An agent turn is only replayable if the assistant's tool_calls and the
      // tool results that answer them survive the round trip.
      if (t.tool_calls) m.tool_calls = t.tool_calls;
      if (t.tool_call_id) m.tool_call_id = t.tool_call_id;
      if (t.name) m.name = t.name;
      messages.push(m);
    }
    var body = {
      model: o.model,
      messages: messages,
      stream: true,
      // Load-bearing: the server gates the whole final chunk on this, and that
      // chunk is what carries `timings` — the only trustworthy decode rate a
      // client can get, since buffering makes wall-clock meaningless.
      stream_options: { include_usage: true },
    };
    if (isNum(o.temperature)) body.temperature = Number(o.temperature);
    if (isNum(o.maxTokens)) body.max_tokens = Number(o.maxTokens);
    if (o.thinking) body.enable_thinking = true;
    if (o.tools && o.tools.length) body.tools = o.tools;
    return body;
  }

  function imageBody(o) {
    var body = { model: o.model, prompt: o.prompt };
    if (o.size) body.size = o.size;
    if (isNum(o.steps)) body.steps = Number(o.steps);
    if (isNum(o.seed)) body.seed = Number(o.seed);
    return body;
  }

  /// The OpenAI images-edit multipart field set, as [name, value] pairs.
  /// `model` matters: the server's multipart scan reads that field to dispatch,
  /// so dropping it silently runs the edit against the DEFAULT model.
  function editFields(o) {
    var fields = [];
    if (o.model) fields.push(['model', o.model]);
    fields.push(['prompt', o.prompt]);
    // A sizeless edit means "keep the reference's geometry"; "auto" is
    // OpenAI's spelling of the same and the server drops it, so send neither.
    if (o.size && o.size !== 'auto') fields.push(['size', o.size]);
    var files = o.files || [];
    for (var i = 0; i < files.length; i++) fields.push(['image[]', files[i]]);
    return fields;
  }

  function musicBody(o) {
    var body = { model: o.model, prompt: o.prompt };
    // `instrumental` and lyrics are a named 400 on both backends, so the flag
    // WINS here rather than letting the pair reach the server: an omitted
    // lyrics field is the only spelling of "no words" Music 3 accepts.
    if (o.instrumental) body.instrumental = true;
    else if (o.lyrics) body.lyrics = o.lyrics;
    if (isNum(o.duration)) body.duration_seconds = Number(o.duration);
    return body;
  }

  function speechBody(o) {
    var body = { model: o.model, input: o.text };
    // ref_audio and voice belong to DIFFERENT backends and each is a named 400
    // on the other, so never send both.
    if (o.voice) body.voice = o.voice;
    else if (o.refAudio) body.ref_audio = o.refAudio;
    if (o.speed) body.speed = o.speed;
    return body;
  }

  /// --api-key exempts loopback, so the common case needs nothing. But a page
  /// fetched over the LAN was authorized by its own query string, and its
  /// fetches have to carry the same key or the console 401s against the very
  /// server that served it.
  function apiKeyFrom(search) {
    if (!search) return null;
    var q = search.charAt(0) === '?' ? search.slice(1) : search;
    var parts = q.split('&');
    for (var i = 0; i < parts.length; i++) {
      var kv = parts[i].split('=');
      if (kv[0] === 'api_key' && kv.length > 1 && kv[1]) return decodeURIComponent(kv[1]);
    }
    return null;
  }

  function authHeaders(key) {
    return key ? { Authorization: 'Bearer ' + key } : {};
  }

  /// Sum the server's own timing block across the rounds of one turn.
  ///
  /// Client wall-clock cannot measure this: with `tools` present the server
  /// buffers tokens for tool-call detection and flushes at the end, so every
  /// SSE delta arrives at once and the measured decode span is ~0 — the console
  /// reported 937 tok/s on a 2B. `timings` is measured on the server around the
  /// actual forward passes, so buffering (and a minutes-long image generation
  /// between two text rounds) cannot distort it.
  function addTimings(acc, t) {
    var a = acc || { prefillMs: 0, tokens: 0, decodeMs: 0 };
    if (!t) return a;
    return {
      prefillMs: a.prefillMs + (t.prompt_ms || 0),
      tokens: a.tokens + (t.predicted_n || 0),
      decodeMs: a.decodeMs + (t.predicted_ms || 0),
    };
  }

  function formatTurnStats(acc) {
    if (!acc) return '';
    var bits = [];
    if (acc.prefillMs > 0) bits.push((acc.prefillMs / 1000).toFixed(2) + 's prefill');
    var tps = tokensPerSecond(acc.tokens, acc.decodeMs);
    if (tps) bits.push(tps + ' tok/s');
    return bits.join('  ·  ');
  }

  function tokensPerSecond(tokens, elapsedMs) {
    if (!tokens || !elapsedMs || elapsedMs <= 0) return null;
    return Math.round((tokens / (elapsedMs / 1000)) * 10) / 10;
  }

  function formatBytes(n) {
    if (!n || n <= 0) return '—';
    var gb = n / (1024 * 1024 * 1024);
    if (gb >= 1) return gb.toFixed(1) + ' GB';
    var mb = Math.round(n / (1024 * 1024));
    // MLX holds a few KB at idle, which rounded to a flat "0 MB" in the
    // header — indistinguishable from "we failed to read /props".
    return mb < 1 ? '<1 MB' : mb + ' MB';
  }

  /// The server says exactly why it refused ("instruction editing requires a
  /// FLUX.2 or Mage-Flow-Edit model"). Swallowing that for a generic "request
  /// failed" would leave the user with no idea which checkpoint to pick.
  function errorText(payload, status) {
    var e = payload && payload.error;
    if (e) {
      if (typeof e === 'string') return e;
      if (typeof e.message === 'string' && e.message) return e.message;
    }
    if (payload && typeof payload.message === 'string' && payload.message) return payload.message;
    return 'request failed (HTTP ' + status + ')';
  }

  /// Prose a TTS model should actually SAY, cut from one markdown reply.
  ///
  /// Reading raw markdown aloud is unusable: a fenced code block becomes
  /// minutes of punctuation, a URL becomes an alphabet soup, and `**bold**`
  /// becomes "star star bold star star". So this strips the markup rather than
  /// escaping it, drops whole code blocks (announcing them instead, so the
  /// listener knows something was skipped), and splits on sentence ends so
  /// playback can start on sentence 1 while the rest is still synthesizing.
  function speakableChunks(md) {
    if (!md) return [];
    var text = String(md);

    // Fenced code: replace the WHOLE block with a spoken marker.
    text = text.replace(/```[\s\S]*?```/g, ' (code block) ');
    text = text.replace(/```[\s\S]*$/, ' (code block) ');   // unterminated fence
    // Inline code keeps its content — it is usually a short identifier.
    text = text.replace(/`([^`]*)`/g, '$1');
    // Links: say the label, never the URL.
    text = text.replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1');
    text = text.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');
    // A bare URL is unspeakable.
    text = text.replace(/https?:\/\/\S+/g, ' a link ');
    // Emphasis, headings, quotes, list bullets, table pipes, rules.
    text = text.replace(/^\s{0,3}#{1,6}\s*/gm, '');
    text = text.replace(/(\*\*|__)(.*?)\1/g, '$2');
    text = text.replace(/(\*|_)(.*?)\1/g, '$2');
    text = text.replace(/^\s{0,3}>\s?/gm, '');
    text = text.replace(/^\s{0,3}[-*+]\s+/gm, '');
    text = text.replace(/^\s{0,3}\d+[.)]\s+/gm, '');
    text = text.replace(/^\s{0,3}([-*_]\s*){3,}$/gm, ' ');
    text = text.replace(/\|/g, ' ');
    text = text.replace(/\s+/g, ' ').trim();
    if (!text) return [];

    // Split after . ! ? … when followed by space/end. Keeps the terminator so
    // the model still hears the sentence type (question prosody matters).
    var parts = text.match(/[^.!?…]+[.!?…]+|[^.!?…]+$/g) || [text];
    var out = [];
    for (var i = 0; i < parts.length; i++) {
      var t = parts[i].trim();
      if (!t) continue;
      // Merge a RUNT onto the previous chunk — "OK." is a whole round trip for
      // nothing. The bar is deliberately low (8): "Yes indeed!" is a real
      // sentence and deserves its own prosody, so a generous threshold would
      // quietly flatten short replies together.
      if (out.length && t.length < 8) out[out.length - 1] += ' ' + t;
      else out.push(t);
    }
    // Cap each chunk: Kokoro's context is 510 phoneme tokens, and a very long
    // clause would 400 rather than truncate.
    var capped = [];
    for (var j = 0; j < out.length; j++) {
      var c = out[j];
      while (c.length > 300) {
        var cut = c.lastIndexOf(' ', 300);
        if (cut <= 0) cut = 300;
        capped.push(c.slice(0, cut).trim());
        c = c.slice(cut).trim();
      }
      if (c) capped.push(c);
    }
    return capped;
  }

  /// Is browser speech recognition available? STT is the BROWSER's job here
  /// (no server-side ASR endpoint exists yet), so the mic must hide rather than
  /// offer a button that cannot work — Safari and Chrome differ on the prefix.
  function sttSupported(win) {
    if (!win) return false;
    return !!(win.SpeechRecognition || win.webkitSpeechRecognition);
  }

  /// Voice-mode state machine. Explicit because the audible failure is a
  /// half-state: the mic still listening while the reply is being spoken makes
  /// the assistant transcribe ITSELF and answer its own sentence.
  var VOICE_STATES = ['off', 'listening', 'thinking', 'speaking'];
  function voiceNext(state, event) {
    switch (state + ':' + event) {
      case 'off:enable': return 'listening';
      case 'listening:transcript': return 'thinking';
      case 'listening:disable': return 'off';
      case 'thinking:reply': return 'speaking';
      case 'thinking:disable': return 'off';
      case 'thinking:error': return 'listening';
      // Back to listening only AFTER speech finishes — never during.
      case 'speaking:spoken': return 'listening';
      case 'speaking:disable': return 'off';
      default: return state;
    }
  }
  /// The mic may only be live while listening.
  function micShouldRun(state) { return state === 'listening'; }

  // Everything above is IIFE-scoped on purpose: the DOM wiring below calls
  // these too, and a helper declared inside this block would be visible to the
  // export and to the tests while being a ReferenceError on the page.
  if (typeof globalThis !== 'undefined') {
    globalThis.__mlxConsole = {
      pickModels: pickModels,
      modelLabel: modelLabel,
      shortModelName: shortModelName,
      modelSubtitle: modelSubtitle,
      mediaTools: mediaTools,
      accumulateToolCalls: accumulateToolCalls,
      toolInvocation: toolInvocation,
      apiReferenceText: apiReferenceText,
      systemPrompt: systemPrompt,
      chatTitle: chatTitle,
      storableTurns: storableTurns,
      historyUpsert: historyUpsert,
      historyRemove: historyRemove,
      renderMarkdown: renderMarkdown,
      sseFeed: sseFeed,
      chatDelta: chatDelta,
      chatBody: chatBody,
      imageBody: imageBody,
      editFields: editFields,
      musicBody: musicBody,
      speechBody: speechBody,
      speakableChunks: speakableChunks,
      sttSupported: sttSupported,
      voiceNext: voiceNext,
      micShouldRun: micShouldRun,
      VOICE_STATES: VOICE_STATES,
      apiKeyFrom: apiKeyFrom,
      authHeaders: authHeaders,
      addTimings: addTimings,
      formatTurnStats: formatTurnStats,
      tokensPerSecond: tokensPerSecond,
      formatBytes: formatBytes,
      errorText: errorText,
    };
  }

  // ══ DOM wiring ════════════════════════════════════════════════════════════
  if (typeof document === 'undefined') return;

  var $ = function (id) { return document.getElementById(id); };
  var API_KEY = apiKeyFrom(location.search);
  var MODELS = [];
  var API_ENTRIES = [];

  var STORE = { chats: 'mlx-serve.chats', model: 'mlx-serve.model', think: 'mlx-serve.thinking' };
  var lsGet = function (k, d) { try { var v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } };
  var lsSet = function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode / quota */ } };

  var CURRENT_MODEL = lsGet(STORE.model, '') || '';
  var THINKING = lsGet(STORE.think, '0') === '1';
  var HISTORY = [];
  var CHAT_ID = null;

  function jsonHeaders() {
    var h = authHeaders(API_KEY);
    h['Content-Type'] = 'application/json';
    return h;
  }

  async function failureText(res) {
    var payload = null;
    try { payload = await res.json(); } catch (e) { /* non-JSON body */ }
    return errorText(payload, res.status);
  }

  /// The one place a status is set. It drives BOTH the line under the composer
  /// and the in-transcript waiting indicator, so a cold load, a prefill and an
  /// image's per-step progress all show up where the answer is going to appear.
  function setStatus(text, kind) {
    var el = $('chat-tools');
    el.textContent = text || 'Runs entirely on this Mac. Models can make mistakes.';
    el.className = 'disclaimer' + (kind ? ' ' + kind : '');
    if (kind === 'busy' && CURRENT_OUT) {
      if (PENDING) PENDING.label.textContent = text || '';
      else showPending(CURRENT_OUT, text);
    } else if (kind !== 'busy') {
      hidePending();
    }
  }

  var PENDING = null;
  var CURRENT_OUT = null;

  function showPending(parent, text) {
    hidePending();
    var w = document.createElement('div');
    w.className = 'pending';
    var dots = document.createElement('span');
    dots.className = 'dots';
    for (var i = 0; i < 3; i++) dots.appendChild(document.createElement('i'));
    var label = document.createElement('span');
    label.className = 'plabel';
    label.textContent = text || '';
    w.appendChild(dots);
    w.appendChild(label);
    parent.appendChild(w);
    PENDING = { el: w, label: label };
    scrollLog();
    return PENDING;
  }

  function hidePending() {
    if (PENDING && PENDING.el.parentNode) PENDING.el.parentNode.removeChild(PENDING.el);
    PENDING = null;
  }

  // ── Sidebar + panels ──────────────────────────────────────────────────────
  var tabs = document.querySelectorAll('[data-tab]');

  function showTab(name) {
    var found = false;
    for (var i = 0; i < tabs.length; i++) {
      var on = tabs[i].getAttribute('data-tab') === name;
      tabs[i].classList.toggle('active', on);
      found = found || on;
    }
    if (!found) return false;
    var panels = document.querySelectorAll('.panel');
    for (var j = 0; j < panels.length; j++) {
      panels[j].classList.toggle('active', panels[j].id === 'tab-' + name);
    }
    if (history.replaceState) history.replaceState(null, '', '#' + name);
    return true;
  }

  for (var t = 0; t < tabs.length; t++) {
    tabs[t].addEventListener('click', function (e) {
      var name = e.currentTarget.getAttribute('data-tab');
      if (e.currentTarget.id === 'nav-new') newChat();
      showTab(name);
    });
  }
  window.addEventListener('hashchange', function () { showTab(location.hash.slice(1) || 'chat'); });

  $('side-toggle').addEventListener('click', function () { $('app').classList.add('collapsed'); });
  $('side-open').addEventListener('click', function () { $('app').classList.remove('collapsed'); });

  // ── The API reference, read out of the page it is already rendered in ─────
  // One list, not two: the API panel's markup IS the reference the assistant
  // answers from, so the two can't drift and the Zig drift guard (every
  // ROUTE_PATHS entry appears in index.html) covers both.
  (function collectApi() {
    var eps = document.querySelectorAll('#tab-api .ep');
    for (var i = 0; i < eps.length; i++) {
      var method = eps[i].querySelector('.m');
      var path = eps[i].querySelector('.p');
      var desc = eps[i].querySelector('.d');
      if (!method || !path) continue;
      API_ENTRIES.push({
        method: method.textContent.trim(),
        path: path.textContent.trim(),
        desc: desc ? desc.textContent.trim() : '',
      });
    }
  })();

  // ── Model picker ──────────────────────────────────────────────────────────
  function chatModels() { return pickModels(MODELS, 'chat'); }

  function currentModel() {
    var list = chatModels();
    if (CURRENT_MODEL && list.some(function (m) { return m.id === CURRENT_MODEL; })) return CURRENT_MODEL;
    // Prefer one that's already resident, so the first send doesn't cold-load.
    var ranked = rankedIds(MODELS, 'chat');
    return ranked.length ? ranked[0] : '';
  }

  function syncModelPill() {
    var id = currentModel();
    $('chat-model-name').textContent = shortModelName(id);
    $('chat-model').title = id || 'no chat model on this server';
  }

  function buildMenu() {
    var menu = $('model-menu');
    menu.innerHTML = '';
    var list = chatModels();
    var active = currentModel();
    if (!list.length) {
      var e = document.createElement('div');
      e.className = 'empty';
      e.textContent = 'No chat models found. Pull one with `mlx-serve pull`, or point the server at a models directory.';
      menu.appendChild(e);
    }
    list.forEach(function (m) {
      var b = document.createElement('button');
      b.className = 'menu-item';
      var tick = document.createElement('span');
      tick.className = 'tick';
      tick.textContent = m.id === active ? '✓' : '';
      var txt = document.createElement('span');
      var name = document.createElement('div');
      name.className = 'mt';
      name.textContent = shortModelName(m.id);
      var sub = document.createElement('div');
      sub.className = 'ms';
      sub.textContent = modelSubtitle(m);
      txt.appendChild(name); txt.appendChild(sub);
      b.appendChild(tick); b.appendChild(txt);
      b.addEventListener('click', function () {
        CURRENT_MODEL = m.id;
        lsSet(STORE.model, m.id);
        syncModelPill();
        closeMenu();
      });
      menu.appendChild(b);
    });

    var sep = document.createElement('div');
    sep.className = 'menu-sep';
    menu.appendChild(sep);

    var think = document.createElement('button');
    think.className = 'menu-item';
    var tick2 = document.createElement('span');
    tick2.className = 'tick';
    tick2.textContent = THINKING ? '✓' : '';
    var txt2 = document.createElement('span');
    var n2 = document.createElement('div');
    n2.className = 'mt';
    n2.textContent = 'Extended thinking';
    var s2 = document.createElement('div');
    s2.className = 'ms';
    s2.textContent = 'shows the model\'s reasoning';
    txt2.appendChild(n2); txt2.appendChild(s2);
    think.appendChild(tick2); think.appendChild(txt2);
    think.addEventListener('click', function () {
      THINKING = !THINKING;
      lsSet(STORE.think, THINKING ? '1' : '0');
      buildMenu();
    });
    menu.appendChild(think);
  }

  function openMenu() {
    buildMenu();
    var menu = $('model-menu');
    menu.hidden = false;
    var pill = $('chat-model').getBoundingClientRect();
    var main = document.querySelector('.main').getBoundingClientRect();
    // The menu opens UPWARD from the pill, so its height is bounded by the
    // space above it — a box of 28 models otherwise runs off the top of the
    // window and the first entries are unreachable.
    menu.style.maxHeight = Math.max(160, pill.top - main.top - 20) + 'px';
    menu.style.left = Math.max(8, Math.min(pill.right - main.left - menu.offsetWidth, main.width - menu.offsetWidth - 8)) + 'px';
    menu.style.bottom = (main.bottom - pill.top + 10) + 'px';
  }
  function closeMenu() { $('model-menu').hidden = true; }

  $('chat-model').addEventListener('click', function (e) {
    e.stopPropagation();
    if ($('model-menu').hidden) openMenu(); else closeMenu();
  });
  document.addEventListener('click', function (e) {
    if (!$('model-menu').hidden && !$('model-menu').contains(e.target)) closeMenu();
  });

  // ── Model list + monitor ──────────────────────────────────────────────────
  function renderMonitor() {
    var body = $('mon-models').querySelector('tbody');
    body.innerHTML = '';
    if (!MODELS.length) {
      var empty = document.createElement('tr');
      var td = document.createElement('td');
      td.colSpan = 4;
      td.className = 'muted';
      td.textContent = 'No models discovered. Point the server at a models directory with --model-dir, or pull one with `mlx-serve pull`.';
      empty.appendChild(td);
      body.appendChild(empty);
      return;
    }
    // Loaded first — a monitor is mostly read for "what is resident right now".
    var rows = MODELS.slice().sort(function (a, b) {
      var ar = a.state === 'ready' ? 0 : 1, br = b.state === 'ready' ? 0 : 1;
      return ar - br || a.id.localeCompare(b.id);
    });
    for (var i = 0; i < rows.length; i++) {
      var m = rows[i];
      var tr = document.createElement('tr');
      if (m.state === 'ready') tr.className = 'ready';

      var id = document.createElement('td');
      id.className = 'mono';
      id.textContent = m.id;

      var caps = document.createElement('td');
      var list = capsOf(m);
      for (var c = 0; c < list.length; c++) {
        var pill = document.createElement('span');
        pill.className = 'cap';
        pill.textContent = list[c];
        caps.appendChild(pill);
      }

      var size = document.createElement('td');
      size.className = 'num';
      size.textContent = formatBytes(m.bytes_resident || m.bytes_on_disk);

      var state = document.createElement('td');
      state.className = 'state ' + (m.state || 'unloaded');
      state.textContent = m.state || 'unloaded';

      tr.appendChild(id); tr.appendChild(caps); tr.appendChild(size); tr.appendChild(state);
      body.appendChild(tr);
    }
  }

  async function refreshModels() {
    try {
      var res = await fetch('/v1/models', { headers: authHeaders(API_KEY) });
      var payload = await res.json();
      MODELS = Array.isArray(payload.data) ? payload.data : [];
    } catch (e) {
      MODELS = [];
    }
    renderMonitor();
    syncModelPill();
    if (!$('model-menu').hidden) buildMenu();

    var loaded = MODELS.filter(function (m) { return m.state === 'ready'; }).length;
    $('hdr-models').textContent = MODELS.length + ' model' + (MODELS.length === 1 ? '' : 's') +
      (loaded ? ' · ' + loaded + ' loaded' : '');
    var first = pickModels(MODELS, 'chat')[0];
    if (first && $('curl-model')) $('curl-model').textContent = first.id;
  }

  async function refreshMemory() {
    try {
      var res = await fetch('/props', { headers: authHeaders(API_KEY) });
      var p = await res.json();
      var mem = p && p.memory ? p.memory.active_bytes : 0;
      $('hdr-mem').textContent = formatBytes(mem) + ' resident';
    } catch (e) { /* transient */ }
  }

  // ── History ───────────────────────────────────────────────────────────────
  function loadHistory() {
    try {
      var raw = JSON.parse(lsGet(STORE.chats, '[]'));
      return Array.isArray(raw) ? raw : [];
    } catch (e) { return []; }
  }

  /// Save a conversation. Takes the list and id EXPLICITLY: a turn that is
  /// still streaming when the user starts a new chat has to write back to the
  /// conversation it belongs to, not to whatever is on screen by then.
  function persist(list, id) {
    list = list || turns;
    id = id || CHAT_ID;
    if (!list.length) return;
    var first = list.filter(function (t) { return t.role === 'user'; })[0];
    HISTORY = historyUpsert(HISTORY, {
      id: id,
      title: chatTitle(first && first.content),
      updated: Date.now(),
      turns: storableTurns(list),
    });
    lsSet(STORE.chats, JSON.stringify(HISTORY));
    renderRecents();
  }

  function renderRecents() {
    var box = $('recent-list');
    box.innerHTML = '';
    if (!HISTORY.length) {
      var e = document.createElement('div');
      e.className = 'empty';
      e.textContent = 'Your chats show up here.';
      box.appendChild(e);
      return;
    }
    HISTORY.forEach(function (c) {
      var row = document.createElement('button');
      row.className = 'recent' + (c.id === CHAT_ID ? ' active' : '');
      var t = document.createElement('span');
      t.className = 't';
      t.textContent = c.title;
      var x = document.createElement('span');
      x.className = 'x';
      x.textContent = '✕';
      x.title = 'Delete';
      x.addEventListener('click', function (ev) {
        ev.stopPropagation();
        HISTORY = historyRemove(HISTORY, c.id);
        lsSet(STORE.chats, JSON.stringify(HISTORY));
        if (c.id === CHAT_ID) newChat();
        renderRecents();
      });
      row.appendChild(t); row.appendChild(x);
      row.addEventListener('click', function () { openChat(c.id); });
      box.appendChild(row);
    });
  }

  // Every conversation owns its transcript element and its turns. `#chat-log`
  // is only the scroll container, holding one `.convo` per conversation with
  // the open one shown. That is what lets a generation you navigated away from
  // finish INTO ITS OWN transcript: there is nothing to clear and nothing to
  // replay, so a picture or a player it produced is still there when you come
  // back. (Replaying from storage could not bring those back — media is
  // deliberately not persisted, it would blow the localStorage quota.)
  var LOGS = {};
  var TURNS = {};
  var LAST_IMAGE = {};

  function logFor(id) {
    if (!LOGS[id]) {
      var d = document.createElement('div');
      d.className = 'convo';
      $('chat-log').appendChild(d);
      LOGS[id] = d;
    }
    return LOGS[id];
  }

  function switchTo(id) {
    CHAT_ID = id;
    TURNS[id] = TURNS[id] || [];
    turns = TURNS[id];
    logFor(id);
    Object.keys(LOGS).forEach(function (k) { LOGS[k].classList.toggle('active', k === id); });
    $('tab-chat').classList.toggle('empty', !turns.length);
    scrollLog();
  }

  function newId() {
    return 'c' + Date.now().toString(36) + Math.floor(Math.random() * 1e6).toString(36);
  }

  function newChat() {
    // "New chat" is also the sidebar's only chat destination, so it is what you
    // click to come BACK from Monitor. On an untouched chat that must be pure
    // navigation.
    // Already sitting in an untouched chat: pure navigation. (Guarded on
    // CHAT_ID so the boot call actually creates the first conversation.)
    if (CHAT_ID && turns && !turns.length) {
      showTab('chat');
      $('chat-input').focus();
      return;
    }
    // A run in flight is NOT cancelled: it owns its own transcript and its own
    // turns, so it finishes into the conversation it was started in.
    switchTo(newId());
    ATTACHMENTS = [];
    renderAttachments();
    setStatus('');
    renderRecents();
    showTab('chat');
    $('chat-input').focus();
  }

  function openChat(id) {
    // Already the open conversation: just show it. Never tear down and replay —
    // a generation may still be streaming into these very nodes, and replaying
    // from storage would clear the log out from under it (and lose the reply,
    // since a live turn isn't persisted until it finishes).
    if (id === CHAT_ID) { showTab('chat'); return; }
    var convo = HISTORY.filter(function (c) { return c.id === id; })[0];
    if (!convo && !TURNS[id]) return;
    // Rebuild from storage only the first time it is opened this session. After
    // that its live element is the truth — and it may still be being written to.
    if (!TURNS[id]) {
      TURNS[id] = (convo.turns || []).slice();
      replay(TURNS[id], logFor(id));
    }
    switchTo(id);
    ATTACHMENTS = [];
    renderAttachments();
    showTab('chat');
    renderRecents();
    setStatus('');
  }

  /// Rebuild the transcript from stored turns. Media is gone by design
  /// (`storableTurns`), so a note stands where the picture was.
  function replay(list, logEl) {
    for (var i = 0; i < list.length; i++) {
      var t = list[i];
      if (t.role === 'user') {
        var b = bubble('user', logEl);
        if (Array.isArray(t.content)) {
          var text = t.content.filter(function (p) { return p.type === 'text'; }).map(function (p) { return p.text; }).join(' ');
          b.textContent = text;
          if (t.content.some(function (p) { return p.type === 'image_omitted'; })) {
            note(b, 'image attached (not stored)');
          }
        } else {
          b.textContent = t.content || '';
        }
      } else if (t.role === 'assistant') {
        var out = bubble('assistant', logEl);
        (t.tool_calls || []).forEach(function (tc) {
          var a = {};
          try { a = JSON.parse(tc.function.arguments || '{}'); } catch (e) { /* keep {} */ }
          note(out, tc.function.name + '(' + (a.prompt || a.text || '') + ')');
        });
        if (t.content) {
          var md = document.createElement('div');
          md.className = 'md';
          md.innerHTML = renderMarkdown(t.content);
          out.appendChild(md);
        }
      }
    }
    scrollLog();
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  function bubble(role, logEl) {
    var wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    var body = document.createElement('div');
    body.className = 'body';
    wrap.appendChild(body);
    (logEl || logFor(CHAT_ID)).appendChild(wrap);
    scrollLog();
    return body;
  }

  function scrollLog() { $('chat-log').scrollTop = $('chat-log').scrollHeight; }

  function thinkingBlock(parent) {
    var d = document.createElement('details');
    d.className = 'think';
    var s = document.createElement('summary');
    s.textContent = 'thinking';
    var pre = document.createElement('div');
    pre.className = 'think-body';
    d.appendChild(s);
    d.appendChild(pre);
    parent.appendChild(d);
    return pre;
  }

  function note(parent, text, kind) {
    var n = document.createElement('div');
    n.className = 'tool' + (kind ? ' ' + kind : '');
    n.textContent = text;
    parent.appendChild(n);
    scrollLog();
    return n;
  }

  function showImage(target, b64, id) {
    var out = document.createElement('div');
    out.className = 'out';
    var img = document.createElement('img');
    img.src = 'data:image/png;base64,' + b64;
    var a = document.createElement('a');
    a.href = img.src;
    a.download = 'mlx-serve.png';
    a.className = 'dl';
    a.textContent = 'download png';
    out.appendChild(img);
    out.appendChild(a);
    target.appendChild(out);
    LAST_IMAGE[id || CHAT_ID] = b64;
    scrollLog();
  }

  function showAudio(target, blobUrl, name) {
    var out = document.createElement('div');
    out.className = 'out';
    var au = document.createElement('audio');
    au.controls = true;
    au.src = blobUrl;
    var a = document.createElement('a');
    a.href = blobUrl;
    a.download = name;
    a.className = 'dl';
    a.textContent = 'download wav';
    out.appendChild(au);
    out.appendChild(a);
    target.appendChild(out);
    scrollLog();
  }

  function b64ToBlob(b64, type) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: type });
  }

  function readAsDataUrl(file) {
    return new Promise(function (resolve, reject) {
      var fr = new FileReader();
      fr.onerror = function () { reject(new Error('could not read ' + file.name)); };
      fr.onload = function () { resolve(String(fr.result)); };
      fr.readAsDataURL(file);
    });
  }

  // ── Attachments ───────────────────────────────────────────────────────────
  // One place images come from: the composer's +. They feed BOTH the edit tool
  // and (on a vision model) the message itself, so "make it winter" and "what is
  // in this photo?" are the same gesture.
  var ATTACHMENTS = [];

  function renderAttachments() {
    var box = $('chat-attached');
    box.innerHTML = '';
    box.style.display = ATTACHMENTS.length ? 'flex' : 'none';
    ATTACHMENTS.forEach(function (a, i) {
      var chip = document.createElement('span');
      chip.className = 'chip';
      var img = document.createElement('img');
      img.src = a.dataUrl;
      var x = document.createElement('button');
      x.className = 'x';
      x.textContent = '✕';
      x.title = 'remove ' + a.name;
      x.addEventListener('click', function () {
        ATTACHMENTS.splice(i, 1);
        renderAttachments();
      });
      chip.appendChild(img);
      chip.appendChild(x);
      box.appendChild(chip);
    });
  }

  $('chat-files').addEventListener('change', async function (e) {
    var files = e.target.files || [];
    for (var i = 0; i < files.length; i++) {
      try {
        ATTACHMENTS.push({ name: files[i].name, file: files[i], dataUrl: await readAsDataUrl(files[i]) });
      } catch (err) { setStatus(String(err.message || err), 'err'); }
    }
    e.target.value = '';
    renderAttachments();
  });

  /// Reference images for an edit: whatever is attached, else the last picture
  /// this conversation produced (so "now make it winter" just works).
  function editRefs(id) {
    if (ATTACHMENTS.length) return ATTACHMENTS.map(function (a) { return a.file; });
    var b64 = LAST_IMAGE[id || CHAT_ID];
    if (b64) return [b64ToBlob(b64, 'image/png')];
    return [];
  }

  // ── Tool execution ────────────────────────────────────────────────────────
  // The result the MODEL sees is a sentence, never bytes: the picture is
  // already on screen, and base64 in the history would blow the context. The
  // sentence has to end the turn, too — "generated the image" reads as an
  // invitation to keep going, and a small model takes it.
  var STOP_HERE = 'Reply with one short sentence now. Do not call another tool.';
  var DONE_IMAGE = 'Done. The image is already displayed to the user. ' + STOP_HERE;

  async function executeTool(call, out, mediaUsed, sId) {
    var args = {};
    try { args = JSON.parse(call.arguments || '{}'); } catch (e) { /* keep {} */ }
    var plan = toolInvocation(
      { name: call.name, args: args },
      { models: MODELS, refs: editRefs(sId), mediaUsed: mediaUsed },
    );
    if (plan.error) {
      // A budget refusal isn't a failure worth shouting about in the log; a
      // real one is.
      var refused = plan.error.indexOf('Refused:') === 0;
      if (!refused) {
        // Don't say the tool's name twice when the message already carries it.
        note(out, plan.error.indexOf(call.name) >= 0 ? plan.error : call.name + ': ' + plan.error, 'err');
      }
      return plan.error;
    }
    note(out, call.name + '(' + (args.prompt || args.text || '') + ')');

    try {
      if (plan.path === '/v1/images/edits') {
        setStatus('editing image…', 'busy');
        var form = new FormData();
        plan.fields.forEach(function (f) { form.append(f[0], f[1]); });
        plan.refs.forEach(function (f, i) { form.append('image[]', f, f.name || ('ref' + i + '.png')); });
        var res = await fetch('/v1/images/edits', {
          method: 'POST', headers: authHeaders(API_KEY), body: form, signal: chatAbort && chatAbort.signal,
        });
        if (!res.ok) throw new Error(await failureText(res));
        var payload = await res.json();
        showImage(out, payload.data[0].b64_json, sId);
        setStatus('');
        return DONE_IMAGE + (plan.note ? ' (' + plan.note + ')' : '');
      }

      var summary = '';
      await runGenStream(plan.path, plan.body, function (ev) {
        if (plan.kind === 'image') {
          showImage(out, ev.data[0].b64_json, sId);
          summary = DONE_IMAGE;
        } else {
          var name = plan.path.indexOf('music') >= 0 ? 'music.wav' : 'speech.wav';
          showAudio(out, URL.createObjectURL(b64ToBlob(ev.data, 'audio/wav')), name);
          summary = 'Done. The audio is playing in the user\'s browser. ' + STOP_HERE;
        }
      });
      setStatus('');
      return summary + (plan.note ? ' (' + plan.note + ')' : '');
    } catch (err) {
      if (err && err.name === 'AbortError') throw err;
      var msg = String(err && err.message ? err.message : err);
      note(out, msg, 'err');
      setStatus('');
      // A failed load marks that entry `error` in the registry. Pick the fresh
      // list up now, so a retry inside this same turn ranks past it instead of
      // choosing the broken checkpoint again.
      await refreshModels();
      // Hand the server's own words back: the model can then explain the
      // constraint (e.g. "that checkpoint can't do instruction edits").
      return 'Failed: ' + msg;
    }
  }

  /// /v1/images/generations and the audio endpoints all speak the same
  /// `progress` / `complete` / `error` event shape (gen_sse.zig).
  async function runGenStream(path, body, onComplete) {
    body.stream = true;
    var res = await fetch(path, {
      method: 'POST', headers: jsonHeaders(), body: JSON.stringify(body),
      signal: chatAbort && chatAbort.signal,
    });
    if (!res.ok) throw new Error(await failureText(res));
    var reader = res.body.getReader();
    var dec = new TextDecoder();
    var rest = '';
    var done = false;
    for (;;) {
      var r = await reader.read();
      if (r.done) break;
      var fed = sseFeed(rest, dec.decode(r.value, { stream: true }));
      rest = fed.rest;
      for (var i = 0; i < fed.events.length; i++) {
        var ev;
        try { ev = JSON.parse(fed.events[i]); } catch (e) { continue; }
        if (ev.type === 'progress') {
          setStatus((ev.stage || 'working') + (ev.total ? ' ' + ev.step + '/' + ev.total : '') + '…', 'busy');
        } else if (ev.type === 'complete') {
          done = true;
          onComplete(ev);
        } else if (ev.type === 'error') {
          throw new Error(ev.message || 'generation failed');
        }
      }
    }
    if (!done) throw new Error('stream ended before a result arrived');
  }

  // ── Chat ──────────────────────────────────────────────────────────────────
  var turns = [];
  var chatAbort = null;
  var MAX_TOOL_ROUNDS = 4;

  function chatBusy(busy) {
    $('chat-send').hidden = busy;
    $('chat-stop').hidden = !busy;
  }

  /// One streamed assistant turn. Renders as it goes; returns what it saw.
  /// Text streams as PLAIN text and is re-rendered as markdown once the turn
  /// closes — parsing markdown per token would be wasted work and would fight
  /// half-written syntax.
  async function streamTurn(model, out, sTurns, partial) {
    var startedAt = Date.now();
    var res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: jsonHeaders(),
      signal: chatAbort.signal,
      body: JSON.stringify(chatBody({
        model: model,
        system: systemPrompt({ models: MODELS, api: API_ENTRIES, origin: location.origin }),
        turns: sTurns,
        thinking: THINKING,
        tools: mediaTools(MODELS),
      })),
    });
    if (!res.ok) throw new Error(await failureText(res));

    var reader = res.body.getReader();
    var dec = new TextDecoder();
    var rest = '', acc = '', calls = [], timings = null, firstByte = 0;
    var thinkPre = null;
    var answer = null;
    // Created on the first content byte, not up front: the waiting indicator is
    // sitting at the end of this bubble and the answer has to land AFTER it.
    function ensureAnswer() {
      if (!answer) {
        hidePending();
        answer = document.createElement('div');
        answer.className = 'md';
        answer.style.whiteSpace = 'pre-wrap';
        out.appendChild(answer);
      }
      return answer;
    }

    for (;;) {
      var r = await reader.read();
      if (r.done) break;
      var fed = sseFeed(rest, dec.decode(r.value, { stream: true }));
      rest = fed.rest;
      for (var i = 0; i < fed.events.length; i++) {
        if (fed.events[i] === '[DONE]') continue;
        var obj;
        try { obj = JSON.parse(fed.events[i]); } catch (e) { continue; }
        var d = chatDelta(obj);
        // Time-to-first-token means the model's first EMITTED token, whatever
        // kind. A round that only produces a tool call used to leave this unset,
        // so the reported TTFT was measured from the next round — i.e. it
        // silently included the whole image generation.
        if (!firstByte && (d.reasoning || d.content || d.toolCalls)) firstByte = Date.now();
        if (d.reasoning) {
          if (!thinkPre) { hidePending(); thinkPre = thinkingBlock(out); }
          thinkPre.textContent += d.reasoning;
        }
        if (d.content) {
          acc += d.content;
          // Published as it goes: an abort throws out of this loop, so the
          // caller can only keep the partial answer if it can see it.
          if (partial) partial.text = acc;
          ensureAnswer().textContent = acc;
        }
        // A tool round emits no text, so nothing else would take the indicator
        // down before executeTool starts appending its own lines.
        if (d.toolCalls) { hidePending(); calls = accumulateToolCalls(calls, d.toolCalls); }
        if (d.timings) timings = d.timings;
        scrollLog();
      }
    }
    if (acc) {
      answer.style.whiteSpace = '';
      answer.innerHTML = renderMarkdown(acc);
    }
    return { content: acc, calls: calls, timings: timings };
  }

  function messageActions(out, text) {
    var row = document.createElement('div');
    row.className = 'actions';
    var copy = document.createElement('button');
    copy.className = 'act';
    copy.textContent = 'Copy';
    copy.addEventListener('click', function () {
      navigator.clipboard.writeText(text).then(function () {
        copy.textContent = 'Copied';
        setTimeout(function () { copy.textContent = 'Copy'; }, 1200);
      }, function () { copy.textContent = 'Copy failed'; });
    });
    var again = document.createElement('button');
    again.className = 'act';
    again.textContent = 'Regenerate';
    again.addEventListener('click', regenerate);
    row.appendChild(copy);
    row.appendChild(again);
    out.parentNode.appendChild(row);
  }

  /// Drop everything back to (and including) the last assistant answer, then
  /// re-run the same user turn.
  function regenerate() {
    if (chatAbort) return;
    var lastUser = -1;
    for (var i = turns.length - 1; i >= 0; i--) if (turns[i].role === 'user') { lastUser = i; break; }
    if (lastUser < 0) return;
    var again = turns[lastUser];
    TURNS[CHAT_ID] = turns = turns.slice(0, lastUser);
    var logEl = logFor(CHAT_ID);
    logEl.innerHTML = '';
    replay(turns, logEl);
    runTurn(again.content, []);
  }

  async function sendChat() {
    var text = $('chat-input').value.trim();
    if (!text && !ATTACHMENTS.length) return;
    $('chat-input').value = '';
    autoGrow();
    var attached = ATTACHMENTS.slice();
    var model = currentModel();

    // A vision chat model gets the attachment as an image part; every model
    // gets it as an edit reference. Sending image parts to a text-only model
    // would just corrupt its prompt.
    var vision = capsOf(MODELS.filter(function (m) { return m.id === model; })[0]).indexOf('vision') >= 0;
    var content = text;
    if (attached.length && vision) {
      content = [{ type: 'text', text: text }];
      attached.forEach(function (a) { content.push({ type: 'image_url', image_url: { url: a.dataUrl } }); });
    }
    runTurn(content, attached);
  }

  async function runTurn(content, attached) {
    var model = currentModel();
    if (!model) { setStatus('no chat model on this server', 'err'); return; }

    turns.push({ role: 'user', content: content });
    $('tab-chat').classList.remove('empty');
    // Save on SEND, not just when the turn finishes: the conversation then
    // exists in Recents (highlighted) from the first message, so it is always
    // one click away, and a navigation, refresh or crash mid-generation can't
    // take it with it.
    persist();

    var sId = CHAT_ID, sTurns = turns, sLog = logFor(sId);
    var ub = bubble('user', sLog);
    ub.textContent = typeof content === 'string'
      ? content
      : content.filter(function (p) { return p.type === 'text'; }).map(function (p) { return p.text; }).join(' ');
    (attached || []).forEach(function (a) {
      var img = document.createElement('img');
      img.src = a.dataUrl;
      img.className = 'thumb';
      ub.appendChild(img);
    });

    chatBusy(true);
    chatAbort = new AbortController();
    var out = bubble('assistant');
    // The waiting indicator lives in THIS bubble, and setStatus keeps it in
    // step from here on. A cold model can take 30s+ before one byte moves —
    // and with tools in the request the server buffers until the flush, so
    // even a warm turn shows nothing for a while. Say which it is.
    CURRENT_OUT = out;
    var picked = MODELS.filter(function (m) { return m.id === model; })[0];
    setStatus((!picked || picked.state !== 'ready')
      ? 'loading ' + shortModelName(model) + '…'
      : 'thinking…', 'busy');

    var partial = { text: '' };
    var stats = null, mediaUsed = 0, finalText = '';
    try {
      for (var round = 0; round < MAX_TOOL_ROUNDS; round++) {
        partial.text = '';
        var turn = await streamTurn(model, out, sTurns, partial);
        stats = addTimings(stats, turn.timings);
        if (turn.content) finalText = turn.content;

        if (!turn.calls.length) {
          if (turn.content) sTurns.push({ role: 'assistant', content: turn.content });
          break;
        }
        sTurns.push({
          role: 'assistant',
          content: turn.content,
          tool_calls: turn.calls.map(function (c, i) {
            return { id: c.id || ('call_' + i), type: 'function', function: { name: c.name, arguments: c.arguments } };
          }),
        });
        for (var c = 0; c < turn.calls.length; c++) {
          var call = turn.calls[c];
          var result = await executeTool(call, out, mediaUsed, sId);
          if (result.indexOf('Done.') === 0) mediaUsed++;
          sTurns.push({ role: 'tool', tool_call_id: call.id || ('call_' + c), name: call.name, content: result });
        }
        // The references were consumed by this turn; keeping them would make
        // the next unrelated question re-edit the same photo.
        ATTACHMENTS = [];
        renderAttachments();
        setStatus('thinking…', 'busy');
      }
      setStatus(formatTurnStats(stats));
      if (finalText) messageActions(out, finalText);
      // Voice mode: say the reply, then resume listening. Guarded on the state
      // so a typed turn never starts talking.
      if (VOICE.state === 'thinking' && finalText) speakReply(finalText);
      else if (VOICE.state === 'thinking') voiceSet('error');
    } catch (err) {
      // Stopped, or navigated away mid-stream: keep whatever the model had
      // already written rather than throwing the turn away.
      if (partial.text) {
        var last = sTurns[sTurns.length - 1];
        if (!last || last.role !== 'assistant' || last.content !== partial.text) {
          sTurns.push({ role: 'assistant', content: partial.text });
        }
      }
      if (err && err.name === 'AbortError') setStatus('stopped');
      else setStatus(String(err && err.message ? err.message : err), 'err');
    } finally {
      hidePending();
      CURRENT_OUT = null;
      chatBusy(false);
      chatAbort = null;
      ATTACHMENTS = [];
      renderAttachments();
      persist(sTurns, sId);
      // A turn may have cold-loaded a model — don't make the sidebar and the
      // monitor wait out the poll interval to say so.
      refreshModels();
      refreshMemory();
    }
  }

  function autoGrow() {
    var ta = $('chat-input');
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }

  // ══ Voice mode ════════════════════════════════════════════════════════════
  // STT is the BROWSER's (Web Speech); TTS is Kokoro on this server. The mic
  // only runs while `listening` — leave it live during playback and the page
  // transcribes the assistant's own voice and answers itself.
  var VOICE = { state: 'off', rec: null, audio: null, queue: [], speaking: false };

  function voiceModelId() {
    var ids = rankedIds(MODELS, 'speech');
    return ids.length ? ids[0] : null;
  }

  function voiceSet(event) {
    var next = voiceNext(VOICE.state, event);
    if (next === VOICE.state) return;
    VOICE.state = next;
    var b = $('chat-voice');
    b.className = 'pill' + (next === 'off' ? '' : ' on ' + next);
    $('chat-voice-label').textContent =
      next === 'off' ? 'Voice'
      : next === 'listening' ? 'Listening'
      : next === 'thinking' ? 'Thinking'
      : 'Speaking';
    if (micShouldRun(next)) startMic(); else stopMic();
    if (next === 'off') stopSpeaking();
  }

  function startMic() {
    if (VOICE.rec) return;
    var Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Ctor) return;
    var rec = new Ctor();
    rec.lang = navigator.language || 'en-US';
    rec.interimResults = true;
    rec.continuous = false;
    rec.onresult = function (e) {
      var finalText = '';
      for (var i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) finalText += e.results[i][0].transcript;
      }
      if (!finalText.trim()) return;
      $('chat-input').value = finalText.trim();
      voiceSet('transcript');
      sendChat();
    };
    // A recognizer that ends on its own (silence) must be restarted, or voice
    // mode looks on but stops hearing anything.
    rec.onend = function () {
      VOICE.rec = null;
      if (micShouldRun(VOICE.state)) startMic();
    };
    rec.onerror = function (e) {
      VOICE.rec = null;
      if (e && (e.error === 'not-allowed' || e.error === 'service-not-allowed')) {
        setStatus('microphone blocked — allow it in the browser', 'err');
        voiceSet('disable');
      }
    };
    VOICE.rec = rec;
    try { rec.start(); } catch (_) { VOICE.rec = null; }
  }

  function stopMic() {
    if (!VOICE.rec) return;
    var r = VOICE.rec;
    VOICE.rec = null;
    r.onend = null;
    try { r.stop(); } catch (_) {}
  }

  function stopSpeaking() {
    VOICE.queue = [];
    VOICE.speaking = false;
    if (VOICE.audio) { try { VOICE.audio.pause(); } catch (_) {} VOICE.audio = null; }
  }

  /// Speak a whole reply. Chunks are synthesized one ahead of playback so the
  /// first sentence starts while the rest is still generating.
  async function speakReply(text) {
    var model = voiceModelId();
    var chunks = speakableChunks(text);
    if (!model || !chunks.length) { voiceSet('spoken'); return; }
    VOICE.queue = chunks;
    VOICE.speaking = true;
    voiceSet('reply');

    var gen = ++speakGen;
    var pending = fetchSpeech(model, VOICE.queue[0]);
    for (var i = 0; i < chunks.length; i++) {
      if (gen !== speakGen) return;
      var blobUrl = await pending;
      if (gen !== speakGen) return;
      pending = (i + 1 < chunks.length) ? fetchSpeech(model, chunks[i + 1]) : null;
      if (blobUrl) await playClip(blobUrl, gen);
    }
    if (gen === speakGen) { VOICE.speaking = false; voiceSet('spoken'); }
  }
  var speakGen = 0;

  async function fetchSpeech(model, text) {
    try {
      var r = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: authHeaders({ 'content-type': 'application/json' }),
        body: JSON.stringify(speechBody({ model: model, text: text, voice: VOICE_NAME })),
      });
      if (!r.ok) return null;
      return URL.createObjectURL(await r.blob());
    } catch (_) { return null; }
  }

  function playClip(url, gen) {
    return new Promise(function (resolve) {
      var a = new Audio(url);
      VOICE.audio = a;
      a.onended = a.onerror = function () {
        URL.revokeObjectURL(url);
        if (VOICE.audio === a) VOICE.audio = null;
        resolve();
      };
      if (gen !== speakGen) { resolve(); return; }
      a.play().catch(function () { resolve(); });
    });
  }

  /// Which Kokoro voice the console speaks with. Not a picker yet — the Voice
  /// pill is a toggle — so this is the model's own default.
  var VOICE_NAME = 'af_heart';

  if (sttSupported(window)) {
    $('chat-voice').hidden = false;
    $('chat-voice').addEventListener('click', function () {
      voiceSet(VOICE.state === 'off' ? 'enable' : 'disable');
    });
  }

  $('chat-send').addEventListener('click', sendChat);
  $('chat-stop').addEventListener('click', function () {
    if (chatAbort) chatAbort.abort();
    speakGen++; stopSpeaking();
    if (VOICE.state !== 'off') voiceSet('spoken');
  });
  $('chat-input').addEventListener('input', autoGrow);
  $('chat-input').addEventListener('keydown', function (e) {
    // Enter sends, Shift+Enter is a newline.
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });

  // ── Boot ──────────────────────────────────────────────────────────────────
  HISTORY = loadHistory();
  renderRecents();
  renderAttachments();
  newChat();
  if (!$('mlx-metrics')) {
    var hint = document.createElement('div');
    hint.className = 'hint';
    hint.textContent = 'Live metrics are off. Restart the server with --metrics for decode/prefill rates, TTFT and cache hit rate here.';
    var scroll = $('tab-monitor').querySelector('.scroll');
    scroll.insertBefore(hint, scroll.firstChild);
  }
  showTab(location.hash.slice(1) || 'chat');
  refreshModels();
  refreshMemory();
  setInterval(refreshMemory, 5000);
  setInterval(function () { if (!chatAbort) refreshModels(); }, 15000);
})();
