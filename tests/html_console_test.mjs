// Unit tests for the built-in console's decision logic (`src/html/app.js`).
//
// Same shape as tests/metrics_panel_test.mjs: the page itself is an untestable
// surface (DOM + fetch + streams), so everything that can be gotten wrong
// WITHOUT a browser — which models each picker offers, how an SSE byte stream
// is cut into events, what body/form we actually post — is factored into pure
// functions and pinned here.
//
// Run: node --test tests/html_console_test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, '..', 'src', 'html', 'app.js'), 'utf8');

// app.js guards its DOM wiring on `typeof document`, so in node only the pure
// helpers evaluate. It hands them back through `globalThis.__mlxConsole`.
new Function(src)();
const C = globalThis.__mlxConsole;
assert.ok(C, 'app.js must expose __mlxConsole for tests');

// ── app.js ↔ index.html wiring ──────────────────────────────────────────────
// The one thing no pure test can see: a control whose id doesn't match the
// markup. `$('chat-sned')` returns null, the listener is never attached, and
// the button is silently dead — the page still renders, still serves, still
// passes every HTTP assertion. Both files are static, so the cross-reference
// is checkable without a DOM.

test('every element app.js reaches for exists in index.html', () => {
  const html = readFileSync(join(here, '..', 'src', 'html', 'index.html'), 'utf8');
  const ids = new Set();
  for (const m of html.matchAll(/\bid=(?:"([^"]+)"|([\w-]+))/g)) ids.add(m[1] || m[2]);
  // metrics.js injects its own markup into this mount at runtime.
  ids.add('mlx-metrics');

  const referenced = new Set();
  for (const m of src.matchAll(/\$\('([\w-]+)'\)/g)) referenced.add(m[1]);
  for (const m of src.matchAll(/getElementById\('([\w-]+)'\)/g)) referenced.add(m[1]);
  assert.ok(referenced.size > 15, 'the id scan found suspiciously little');

  const missing = [...referenced].filter(id => !ids.has(id));
  assert.deepEqual(missing, [], 'app.js references ids absent from index.html');
});

// The scope class no pure test can see either: the pure helpers and the DOM
// wiring live in ONE IIFE, and the helpers are handed to tests through an
// `if (typeof globalThis !== 'undefined') { … }` block at the bottom of the
// pure layer. A helper written INSIDE that block is block-scoped under
// 'use strict', so it reaches the export object (built in the same block) and
// every test here, while being invisible to the wiring below it. Live
// 2026-07-26: four voice helpers landed inside the guard, and the first call
// from the wiring (`if (sttSupported(window))`) threw
// "ReferenceError: Can't find variable: sttSupported" — which aborts the rest
// of the IIFE, so the send button, the keyboard handler and the whole boot
// block (newChat / showTab / refreshModels) never ran. The page still renders
// and still serves; node still loads app.js (it returns before the wiring) and
// every test below still passed.
test('the __mlxConsole export guard wraps ONLY the export', () => {
  const lines = src.split('\n');
  const start = lines.findIndex(l => l.includes("if (typeof globalThis !== 'undefined') {"));
  assert.ok(start >= 0, 'the export guard must exist');
  const end = lines.indexOf('  }', start + 1);
  assert.ok(end > start, 'the export guard must close at IIFE indentation');

  const declared = lines
    .slice(start + 1, end)
    .filter(l => /^\s*(?:function|var|let|const)\s/.test(l))
    .map(l => l.trim());
  assert.deepEqual(declared, [],
    'declared inside the export guard, so invisible to the DOM wiring below — ' +
    'move these above the guard, which must contain nothing but __mlxConsole');
});

// And the same class caught dynamically: BOOT the wiring. Every DOM access
// goes through a Proxy that swallows anything, so this asserts one thing only
// — the IIFE runs to the end. It is the sole test that exercises the half of
// app.js below `typeof document`, where a ReferenceError takes out every
// listener and the whole boot block after it.
test('the DOM wiring boots without reaching for something out of scope', () => {
  const saved = ['document', 'window', 'location', 'localStorage', 'fetch',
                 'setInterval', 'setTimeout', 'AbortController']
    .map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]);
  const stub = () => new Proxy(function () {}, {
    get: (t, k) => (k === Symbol.toPrimitive || k === 'toString' ? () => '' : stub()),
    set: () => true, apply: () => stub(), construct: () => stub(), has: () => true,
  });
  try {
    globalThis.document = stub();
    globalThis.window = stub();
    globalThis.location = { search: '', hash: '' };
    globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
    globalThis.fetch = () => new Promise(() => {});
    globalThis.setInterval = () => 0;
    globalThis.setTimeout = () => 0;
    globalThis.AbortController = function () { this.abort = () => {}; this.signal = {}; };
    new Function(src)();
  } finally {
    for (const [k, d] of saved) {
      if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k];
    }
    new Function(src)();   // restore the pure-layer export for the tests below
  }
});

test('the sidebar destinations are New chat, Monitor, API in that order', () => {
  const html = readFileSync(join(here, '..', 'src', 'html', 'index.html'), 'utf8');
  const order = [...html.matchAll(/data-tab="(\w+)"/g)].map(m => m[1]);
  assert.deepEqual(order, ['chat', 'monitor', 'api']);
  // The panel that ships with `active` is what a visitor sees before any JS.
  const active = /<section class="panel active" id=tab-(\w+)>/.exec(html);
  assert.equal(active && active[1], 'chat');
  // Chat opens in its empty state: greeting + a centred composer, no transcript.
  assert.match(html, /id=chat-empty/);
});

test('every interactive control in index.html is wired up in app.js', () => {
  const html = readFileSync(join(here, '..', 'src', 'html', 'index.html'), 'utf8');
  const controls = [];
  for (const m of html.matchAll(/<(?:button|select|input|textarea)\b[^>]*>/g)) {
    const id = /\bid=(?:"([^"]+)"|([\w-]+))/.exec(m[0]);
    // Tab buttons are wired by their data-tab attribute, not by id.
    if (id && !/data-tab/.test(m[0])) controls.push(id[1] || id[2]);
  }
  // Guard the scan itself, not the design: naming the controls that must exist
  // means the test can't pass vacuously if the regex stops matching, and it
  // can't fail just because the page got simpler.
  for (const must of ['chat-input', 'chat-send', 'chat-stop', 'chat-files', 'chat-model']) {
    assert.ok(controls.includes(must), `expected a control #${must}`);
  }
  const dead = controls.filter(id => !src.includes(`'${id}'`));
  assert.deepEqual(dead, [], 'controls rendered but never read by app.js');
});

const M = (id, capabilities, over = {}) => ({
  id,
  capabilities,
  loaded: false,
  state: 'unloaded',
  bytes_resident: 0,
  bytes_on_disk: 1,
  ...over,
});

// A slice of a real `/v1/models` payload: chat models, an encoder, an image
// backend, a TTS voice, a music model, and a LAN-mirrored peer entry.
const FLEET = [
  M('gemma-4-e4b-it-4bit', ['chat', 'tool_use', 'streaming', 'json_schema', 'vision']),
  M('qwen3.6-27b', ['chat', 'tool_use', 'streaming', 'json_schema'], { loaded: true, state: 'ready' }),
  M('bge-small-en-v1.5-8bit', ['embeddings']),
  M('ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit', ['image']),
  M('Qwen3-TTS-Flash-Base-MLX-8bit', ['audio']),
  M('ACE-Step-v1-3.5B-MLX-8bit', ['audio', 'music']),
  M('no-caps-gguf-shelf', undefined),
];

// ── Model selection ─────────────────────────────────────────────────────────
// The chat picker offers what it can drive; the media models are resolved by
// capability behind the scenes, because the user asks for a song in words and
// never picks a backend.

test('chat picker lists chat models only', () => {
  assert.deepEqual(C.pickModels(FLEET, 'chat').map(m => m.id), [
    'gemma-4-e4b-it-4bit',
    'qwen3.6-27b',
  ]);
});

test('image selection lists image backends only', () => {
  assert.deepEqual(C.pickModels(FLEET, 'image').map(m => m.id), [
    'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit',
  ]);
});

test('speech selection excludes the music backend', () => {
  // Music models advertise BOTH "audio" and "music" (additive rule in
  // readyCapsJson), so a naive `has("audio")` filter routes a TTS request at
  // ACE-Step, which 400s "loaded audio model is a music generator".
  assert.deepEqual(C.pickModels(FLEET, 'speech').map(m => m.id), [
    'Qwen3-TTS-Flash-Base-MLX-8bit',
  ]);
});

test('music selection lists music backends only', () => {
  assert.deepEqual(C.pickModels(FLEET, 'music').map(m => m.id), [
    'ACE-Step-v1-3.5B-MLX-8bit',
  ]);
});

test('a model with no capabilities array is never selected', () => {
  // Unloaded stubs whose config.json couldn't be read ship no `capabilities`
  // key at all — `undefined.includes` would throw and blank every list.
  for (const kind of ['chat', 'image', 'speech', 'music']) {
    assert.equal(C.pickModels(FLEET, kind).some(m => m.id === 'no-caps-gguf-shelf'), false);
  }
  assert.deepEqual(C.pickModels([], 'chat'), []);
  assert.deepEqual(C.pickModels(undefined, 'chat'), []);
});

// ── Media as tools ──────────────────────────────────────────────────────────
// "Generate an image", "make it winter", "write me a song" are chat turns, not
// separate forms. The routing is the model's job: we hand it a tool per
// modality we can actually serve and execute what it calls.

test('mediaTools offers one tool per modality that exists on this server', () => {
  const names = C.mediaTools(FLEET).map(t => t.function.name);
  assert.deepEqual(names.sort(), ['edit_image', 'generate_image', 'generate_music', 'generate_speech']);
});

test('mediaTools offers nothing a server cannot run', () => {
  // Offering generate_music on a box with no music checkpoint teaches the
  // model to promise a song and then hand back a 400.
  const chatOnly = FLEET.filter(m => (m.capabilities || []).includes('chat'));
  assert.deepEqual(C.mediaTools(chatOnly), []);
  const imageOnly = FLEET.filter(m => (m.capabilities || []).includes('image'));
  assert.deepEqual(
    C.mediaTools(imageOnly).map(t => t.function.name).sort(),
    ['edit_image', 'generate_image'],
  );
});

test('each tool is a well-formed OpenAI function definition', () => {
  for (const t of C.mediaTools(FLEET)) {
    assert.equal(t.type, 'function');
    assert.ok(t.function.name && t.function.description);
    assert.equal(t.function.parameters.type, 'object');
    assert.ok(Array.isArray(t.function.parameters.required));
    // The `model` argument enumerates the real ids for that modality, so the
    // model can honour "use FLUX" without being able to invent a checkpoint.
    const modelArg = t.function.parameters.properties.model;
    if (modelArg) assert.ok(modelArg.enum.length > 0);
  }
});

test('accumulateToolCalls merges deltas whether args arrive whole or split', () => {
  // Our server sends the full arguments in ONE delta, but the OpenAI wire
  // format allows any split and other clients on this endpoint do split.
  const whole = C.accumulateToolCalls([], [
    { index: 0, id: 'c1', function: { name: 'generate_image', arguments: '{"prompt":"a fox"}' } },
  ]);
  assert.deepEqual(whole, [{ id: 'c1', name: 'generate_image', arguments: '{"prompt":"a fox"}' }]);

  let acc = C.accumulateToolCalls([], [{ index: 0, id: 'c1', function: { name: 'generate_image', arguments: '{"pro' } }]);
  acc = C.accumulateToolCalls(acc, [{ index: 0, function: { arguments: 'mpt":"a fox"}' } }]);
  assert.deepEqual(acc, [{ id: 'c1', name: 'generate_image', arguments: '{"prompt":"a fox"}' }]);
});

test('chatDelta surfaces tool calls alongside content', () => {
  const d = C.chatDelta({
    choices: [{ delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'generate_music', arguments: '{}' } }] } }],
  });
  assert.equal(d.toolCalls.length, 1);
  assert.equal(d.toolCalls[0].function.name, 'generate_music');
});

test('toolInvocation resolves generate_image against a real backend', () => {
  const plan = C.toolInvocation(
    { name: 'generate_image', args: { prompt: 'a red fox', size: '1024x768' } },
    { models: FLEET, refs: [] },
  );
  assert.equal(plan.kind, 'image');
  assert.equal(plan.path, '/v1/images/generations');
  assert.deepEqual(plan.body, {
    model: 'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit',
    prompt: 'a red fox',
    size: '1024x768',
  });
});

test('toolInvocation honours a named model but never a hallucinated one', () => {
  const named = C.toolInvocation(
    { name: 'generate_image', args: { prompt: 'x', model: 'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit' } },
    { models: FLEET, refs: [] },
  );
  assert.equal(named.body.model, 'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit');

  // A model that invents an id, or names the CHAT model it is currently
  // running as, must not send that id to an image endpoint — it falls back to
  // a real backend, and `plan.note` says so rather than substituting silently.
  const bogus = C.toolInvocation(
    { name: 'generate_image', args: { prompt: 'x', model: 'dall-e-3' } },
    { models: FLEET, refs: [] },
  );
  assert.equal(bogus.body.model, 'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit');
  assert.match(bogus.note, /dall-e-3/);
});

test('toolInvocation refuses a modality this server has no model for', () => {
  const chatOnly = FLEET.filter(m => (m.capabilities || []).includes('chat'));
  const plan = C.toolInvocation({ name: 'generate_music', args: { prompt: 'lofi' } }, { models: chatOnly, refs: [] });
  assert.ok(plan.error);
  assert.match(plan.error, /music/i);
});

test('edit_image prefers a checkpoint that can actually edit', () => {
  // Edit capability is not API-visible: Mage-Flow-Turbo and Mage-Flow-Edit-Turbo
  // both report capabilities:["image"] and ship byte-identical configs, so the
  // SERVER itself gates on the directory name (`mage_flow.dirIsEdit`). Picking
  // the first image model instead routed the edit at the txt2img checkpoint and
  // 400'd — live, the model then burned a second generation retrying.
  const fleet = [
    M('ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit', ['image']),
    M('Runpod/FLUX.2-klein-4B-mflux-4bit', ['image']),
    M('ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit', ['image']),
  ];
  const plan = C.toolInvocation({ name: 'edit_image', args: { prompt: 'winter' } }, { models: fleet, refs: ['<b>'] });
  assert.equal(Object.fromEntries(plan.fields).model, 'ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit');

  // FLUX.2 has trained edit capability, so it wins over a plain txt2img model.
  const noMageEdit = fleet.filter(m => !/Edit/.test(m.id));
  const flux = C.toolInvocation({ name: 'edit_image', args: { prompt: 'winter' } }, { models: noMageEdit, refs: ['<b>'] });
  assert.equal(Object.fromEntries(flux.fields).model, 'Runpod/FLUX.2-klein-4B-mflux-4bit');

  // Generation is unaffected — any image model can do txt2img.
  const gen = C.toolInvocation({ name: 'generate_image', args: { prompt: 'a fox' } }, { models: fleet, refs: [] });
  assert.equal(gen.body.model, 'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit');
});

test("edit_image's model enum offers only what it would actually resolve to", () => {
  // The enum and the resolution have to be the same list. They weren't: the
  // enum listed every image model, and live the model dutifully picked
  // Mage-Flow-Turbo out of it — an explicit choice beats a preference, so the
  // edit went to the txt2img checkpoint and 400'd anyway.
  const fleet = [
    M('ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit', ['image']),
    M('mlx-community/FLUX.2-Klein-4B-3bit', ['image']),
    M('ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit', ['image']),
  ];
  const tools = C.mediaTools(fleet);
  const editEnum = tools.find(t => t.function.name === 'edit_image').function.parameters.properties.model.enum;
  assert.equal(editEnum.includes('ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit'), false);
  assert.deepEqual(editEnum, [
    'ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit',
    'mlx-community/FLUX.2-Klein-4B-3bit',
  ]);
  // Anything the enum offers must resolve to itself, not be second-guessed.
  for (const id of editEnum) {
    const plan = C.toolInvocation({ name: 'edit_image', args: { prompt: 'p', model: id } }, { models: fleet, refs: ['<b>'] });
    assert.equal(Object.fromEntries(plan.fields).model, id);
    assert.equal(plan.note, undefined);
  }
  // Generation still offers everything — any image model can do txt2img.
  const genEnum = tools.find(t => t.function.name === 'generate_image').function.parameters.properties.model.enum;
  assert.equal(genEnum.length, 3);
});

test('edit_image needs a reference image and says so when there is none', () => {
  const none = C.toolInvocation({ name: 'edit_image', args: { prompt: 'make it winter' } }, { models: FLEET, refs: [] });
  assert.ok(none.error);
  assert.match(none.error, /attach|image/i);

  const withRef = C.toolInvocation(
    { name: 'edit_image', args: { prompt: 'make it winter' } },
    { models: FLEET, refs: ['<blob>'] },
  );
  assert.equal(withRef.path, '/v1/images/edits');
  assert.equal(withRef.refs.length, 1);
  const scalars = Object.fromEntries(withRef.fields);
  assert.equal(scalars.prompt, 'make it winter');
  // `model` is the field the server's multipart scan reads to dispatch;
  // without it the edit silently runs against the default model.
  assert.equal(scalars.model, 'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit');
});

test('tool resolution prefers a loaded model and avoids one that failed to load', () => {
  // Live: two Qwen3-TTS checkpoints on disk, the bf16 one an incomplete
  // download (config + tokenizer, no safetensors). It sorted first, so "say
  // this out loud" spent a load attempt on it — `NoWeightFiles`, "Model load
  // failed" — before the retry landed on the 8-bit sibling. Already-resident
  // is also free, which matters when a wrong pick costs a multi-GB load.
  const fleet = [
    M('tts-broken-bf16', ['audio'], { state: 'error', error: 'NoWeightFiles' }),
    M('tts-cold-8bit', ['audio']),
    M('tts-resident-4bit', ['audio'], { state: 'ready', loaded: true }),
  ];
  const plan = C.toolInvocation({ name: 'generate_speech', args: { text: 'hi' } }, { models: fleet, refs: [] });
  assert.equal(plan.body.model, 'tts-resident-4bit');

  // With nothing resident, a cold model still beats a known-broken one.
  const noResident = fleet.filter(m => m.state !== 'ready');
  const cold = C.toolInvocation({ name: 'generate_speech', args: { text: 'hi' } }, { models: noResident, refs: [] });
  assert.equal(cold.body.model, 'tts-cold-8bit');

  // The picker itself keeps discovery order — it refreshes every 15s and a
  // list that reorders as models load and unload moves under the cursor.
  assert.deepEqual(C.pickModels(fleet, 'speech').map(m => m.id),
    ['tts-broken-bf16', 'tts-cold-8bit', 'tts-resident-4bit']);

  // Before anything is tried, `state` is `unloaded` for both and the ranking
  // has to read the only pre-load signal there is: discovery sums the
  // checkpoint's *.safetensors, so a null size means the shards are missing.
  // That is exactly what the broken bf16 dir reported (config + tokenizer, no
  // weights) while its sibling reported 2.3 GB.
  const untried = [
    M('tts-incomplete-download', ['audio'], { bytes_on_disk: null }),
    M('tts-complete', ['audio'], { bytes_on_disk: 2417320525 }),
  ];
  const first = C.toolInvocation({ name: 'generate_speech', args: { text: 'hi' } }, { models: untried, refs: [] });
  assert.equal(first.body.model, 'tts-complete');

  // Soft preference: if nothing reports a size, order is untouched.
  const noSizes = untried.map(m => ({ ...m, bytes_on_disk: null }));
  const same = C.toolInvocation({ name: 'generate_speech', args: { text: 'hi' } }, { models: noSizes, refs: [] });
  assert.equal(same.body.model, 'tts-incomplete-download');
});

test('toolInvocation maps speech and music onto their endpoints', () => {
  const speech = C.toolInvocation({ name: 'generate_speech', args: { text: 'hello' } }, { models: FLEET, refs: [] });
  assert.equal(speech.path, '/v1/audio/speech');
  assert.deepEqual(speech.body, { model: 'Qwen3-TTS-Flash-Base-MLX-8bit', input: 'hello' });

  const music = C.toolInvocation(
    { name: 'generate_music', args: { prompt: 'lofi', lyrics: 'la', duration_seconds: 30 } },
    { models: FLEET, refs: [] },
  );
  assert.equal(music.path, '/v1/audio/music-generations');
  assert.deepEqual(music.body, {
    model: 'ACE-Step-v1-3.5B-MLX-8bit', prompt: 'lofi', lyrics: 'la', duration_seconds: 30,
  });
});

test('an unknown tool name is an error, not a thrown exception', () => {
  const plan = C.toolInvocation({ name: 'delete_everything', args: {} }, { models: FLEET, refs: [] });
  assert.ok(plan.error);
});

test('one media generation per user turn, and the model is told why', () => {
  // Live, a 2B model answered "generate an image of a fox" by generating the
  // fox and then inventing three more edits nobody asked for — four GPU
  // generations for one request. Rounds alone don't bound this; the budget
  // does. It must come back as a sentence the model can act on, not silence,
  // or it just calls again.
  const first = C.toolInvocation({ name: 'generate_image', args: { prompt: 'a fox' } },
    { models: FLEET, refs: [], mediaUsed: 0 });
  assert.ok(!first.error);

  const second = C.toolInvocation({ name: 'generate_image', args: { prompt: 'a fox' } },
    { models: FLEET, refs: [], mediaUsed: 1 });
  assert.ok(second.error);
  assert.match(second.error, /already|one/i);
  assert.match(second.error, /ask/i); // tells it what to do instead
});

// ── The console's own system prompt ─────────────────────────────────────────

test('apiReferenceText renders one line per endpoint', () => {
  const text = C.apiReferenceText([
    { method: 'POST', path: '/v1/chat/completions', desc: 'Streaming and non-streaming' },
    { method: 'GET', path: '/api/tags', desc: 'List local models' },
  ]);
  assert.match(text, /POST \/v1\/chat\/completions — Streaming and non-streaming/);
  assert.match(text, /GET \/api\/tags — List local models/);
});

test('systemPrompt teaches the model this server, its models and its tools', () => {
  const p = C.systemPrompt({
    models: FLEET,
    api: [{ method: 'POST', path: '/v1/images/edits', desc: 'OpenAI-compatible image editing' }],
    origin: 'http://127.0.0.1:11434',
  });
  assert.match(p, /mlx-serve/);
  // It answers questions about the API, so the reference has to be in it.
  assert.match(p, /\/v1\/images\/edits/);
  // …and about what is installed.
  assert.match(p, /ACE-Step-v1-3\.5B-MLX-8bit/);
  // …and it must know the media work goes through tools, not prose.
  assert.match(p, /generate_image/);
  assert.match(p, /generate_music/);
});

test('systemPrompt carries the real base URL and real request fields', () => {
  // A path list alone is not enough to answer "how do I edit an image?": live,
  // the model filled the gap itself and produced
  // `curl -X POST https://your-ollama-ip-address/api/v1/images/edits -F "ref1=…"`
  // — wrong host, wrong path prefix, invented field names. Whatever the prompt
  // doesn't say, the model makes up.
  const p = C.systemPrompt({
    models: [], api: [{ method: 'POST', path: '/v1/images/edits', desc: 'edit' }],
    origin: 'http://192.168.1.50:11434',
  });
  assert.match(p, /http:\/\/192\.168\.1\.50:11434/);
  // The fields people actually ask about, for the endpoints they ask about.
  assert.match(p, /image\[\]/);
  assert.match(p, /max_tokens/);
  assert.match(p, /duration_seconds/);
});

test('systemPrompt promises no tool on a server that has none', () => {
  const chatOnly = FLEET.filter(m => (m.capabilities || []).includes('chat'));
  const p = C.systemPrompt({ models: chatOnly, api: [] });
  assert.equal(/generate_image/.test(p), false);
  assert.match(p, /mlx-serve/);
});

test('model labels carry the loaded/unloaded state', () => {
  assert.match(C.modelLabel(FLEET[1]), /ready|loaded/i);
  assert.match(C.modelLabel(FLEET[0]), /unloaded|not loaded/i);
});

// ── SSE parsing ─────────────────────────────────────────────────────────────
// A ReadableStream hands over arbitrary byte boundaries: a frame routinely
// arrives split across two chunks, and two frames routinely arrive in one.

test('sseFeed cuts complete frames and keeps the remainder buffered', () => {
  const a = C.sseFeed('', 'data: {"a":1}\n\ndata: {"a":2}\n\ndata: {"a":3');
  assert.deepEqual(a.events, ['{"a":1}', '{"a":2}']);
  assert.equal(a.rest, 'data: {"a":3');

  const b = C.sseFeed(a.rest, '}\n\n');
  assert.deepEqual(b.events, ['{"a":3}']);
  assert.equal(b.rest, '');
});

test('sseFeed splits a frame arriving one byte at a time', () => {
  const whole = 'data: {"hello":"world"}\n\n';
  let rest = '';
  let out = [];
  for (const ch of whole) {
    const r = C.sseFeed(rest, ch);
    rest = r.rest;
    out = out.concat(r.events);
  }
  assert.deepEqual(out, ['{"hello":"world"}']);
  assert.equal(rest, '');
});

test('sseFeed handles [DONE], comments and bare newlines', () => {
  const r = C.sseFeed('', ': keepalive\n\ndata: {"x":1}\n\ndata: [DONE]\n\n');
  // Keepalive comments are not events — the server emits them on any stream
  // that goes quiet for 5s, and treating one as data breaks JSON.parse.
  assert.deepEqual(r.events, ['{"x":1}', '[DONE]']);
});

test('sseFeed tolerates \\r\\n frame separators', () => {
  const r = C.sseFeed('', 'data: {"x":1}\r\n\r\ndata: [DONE]\r\n\r\n');
  assert.deepEqual(r.events, ['{"x":1}', '[DONE]']);
});

test('chatDelta routes content, reasoning and finish', () => {
  const d = j => C.chatDelta(JSON.parse(j));
  assert.deepEqual(d('{"choices":[{"delta":{"content":"hi"}}]}'), { content: 'hi' });
  assert.deepEqual(d('{"choices":[{"delta":{"reasoning_content":"hmm"}}]}'), { reasoning: 'hmm' });
  // The final usage frame carries no delta at all.
  assert.deepEqual(
    d('{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":7}}'),
    { finish: 'stop', usage: { completion_tokens: 7 } },
  );
  assert.deepEqual(d('{}'), {});
});

// ── Request construction ────────────────────────────────────────────────────

test('chatBody sends the conversation, streams, and asks for usage', () => {
  const body = C.chatBody({
    model: 'qwen3.6-27b',
    system: 'You are terse.',
    turns: [
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi' },
      { role: 'user', content: 'again' },
    ],
    temperature: 0.7,
    maxTokens: 512,
  });
  assert.equal(body.model, 'qwen3.6-27b');
  assert.equal(body.stream, true);
  assert.equal(body.temperature, 0.7);
  assert.equal(body.max_tokens, 512);
  assert.deepEqual(body.messages[0], { role: 'system', content: 'You are terse.' });
  assert.equal(body.messages.length, 4);
  // Without it the final frame carries no token counts and the tok/s readout
  // has no numerator.
  assert.deepEqual(body.stream_options, { include_usage: true });
});

test('chatBody omits an empty system prompt and unset sampling fields', () => {
  const body = C.chatBody({
    model: 'm',
    system: '   ',
    turns: [{ role: 'user', content: 'x' }],
    temperature: null,
    maxTokens: null,
  });
  assert.equal(body.messages.length, 1);
  assert.equal('temperature' in body, false);
  assert.equal('max_tokens' in body, false);
});

test('imageBody carries size/steps/seed only when given', () => {
  const full = C.imageBody({ model: 'm', prompt: 'a cat', size: '1024x1024', steps: 4, seed: 7 });
  assert.deepEqual(full, { model: 'm', prompt: 'a cat', size: '1024x1024', steps: 4, seed: 7 });
  const bare = C.imageBody({ model: 'm', prompt: 'a cat', size: '', steps: null, seed: null });
  assert.deepEqual(bare, { model: 'm', prompt: 'a cat' });
});

test('editFields is the OpenAI multipart field set with repeated image[]', () => {
  const fields = C.editFields({
    model: 'ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit',
    prompt: 'make it winter',
    files: ['<a>', '<b>'],
    size: '',
  });
  // Order-independent comparison; `image[]` repeats once per file.
  assert.deepEqual(fields.filter(([k]) => k === 'image[]').map(([, v]) => v), ['<a>', '<b>']);
  const scalars = Object.fromEntries(fields.filter(([k]) => k !== 'image[]'));
  assert.deepEqual(scalars, {
    model: 'ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit',
    prompt: 'make it winter',
  });
  // `model` is the field the server's multipart scan reads to dispatch —
  // dropping it runs the edit against the DEFAULT model (the Open WebUI class).
  assert.ok(fields.some(([k, v]) => k === 'model' && v));
});

test('editFields forwards an explicit size but never "auto"', () => {
  // A sizeless edit means "keep the reference's geometry"; "auto" is OpenAI's
  // spelling of the same thing and the server drops it, so don't send either.
  const sized = C.editFields({ model: 'm', prompt: 'p', files: ['<a>'], size: '1024x1024' });
  assert.equal(Object.fromEntries(sized).size, '1024x1024');
  const auto = C.editFields({ model: 'm', prompt: 'p', files: ['<a>'], size: 'auto' });
  assert.equal(auto.some(([k]) => k === 'size'), false);
});

test('musicBody requires a prompt and passes lyrics + duration through', () => {
  const b = C.musicBody({ model: 'm', prompt: 'lofi', lyrics: 'la la', duration: 30 });
  assert.deepEqual(b, { model: 'm', prompt: 'lofi', lyrics: 'la la', duration_seconds: 30 });
  const bare = C.musicBody({ model: 'm', prompt: 'lofi', lyrics: '', duration: null });
  assert.deepEqual(bare, { model: 'm', prompt: 'lofi' });
});

test('speechBody uses the OpenAI `input` field', () => {
  assert.deepEqual(C.speechBody({ model: 'm', text: 'hello' }), { model: 'm', input: 'hello' });
  assert.deepEqual(
    C.speechBody({ model: 'm', text: 'hello', refAudio: 'BASE64' }),
    { model: 'm', input: 'hello', ref_audio: 'BASE64' },
  );
});

// ── Auth passthrough ────────────────────────────────────────────────────────

test('the page carries its own ?api_key= into every fetch', () => {
  // --api-key exempts loopback, so the common case needs nothing — but a page
  // opened over the LAN was authorized by its query string and its fetches
  // must be too, or the console 401s against the server that just served it.
  assert.equal(C.apiKeyFrom('?api_key=s3cret'), 's3cret');
  assert.equal(C.apiKeyFrom('?foo=1&api_key=s3cret&bar=2'), 's3cret');
  assert.equal(C.apiKeyFrom(''), null);
  assert.equal(C.apiKeyFrom('?other=1'), null);

  assert.deepEqual(C.authHeaders(null), {});
  assert.deepEqual(C.authHeaders('s3cret'), { Authorization: 'Bearer s3cret' });
});

// ── Chat history (localStorage) ─────────────────────────────────────────────
// Conversations survive a refresh and show up under Recents. The store is a
// plain array so all the deciding is testable without a DOM or a quota.

test('chatTitle summarises the first thing the user said', () => {
  assert.equal(C.chatTitle('what is mlx-serve?'), 'what is mlx-serve?');
  assert.equal(C.chatTitle('  spaces\n and\tnewlines  '), 'spaces and newlines');
  const long = 'generate an image of a red fox standing in deep snow at golden hour with long shadows';
  assert.ok(C.chatTitle(long).length <= 52);
  assert.match(C.chatTitle(long), /^generate an image of a red fox/);
  assert.equal(C.chatTitle(''), 'New chat');
  assert.equal(C.chatTitle(null), 'New chat');
  // A vision turn's content is an array of parts, not a string.
  assert.equal(
    C.chatTitle([{ type: 'text', text: 'make it winter' }, { type: 'image_url', image_url: { url: 'data:...' } }]),
    'make it winter',
  );
});

test('storableTurns drops image payloads but keeps the conversation', () => {
  // A single 1024x1024 PNG is ~1.5 MB of base64 and localStorage gives us ~5 MB
  // for everything. Persisting one image-generating chat would evict every
  // other conversation, so the bytes stay in the DOM and the history keeps a
  // marker in their place.
  const turns = [
    { role: 'user', content: [
      { type: 'text', text: 'make it winter' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAAAAAA' } },
    ] },
    { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'edit_image', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'c1', name: 'edit_image', content: 'Done. The image is already displayed.' },
    { role: 'assistant', content: 'Made it wintry.' },
  ];
  const stored = C.storableTurns(turns);
  assert.equal(JSON.stringify(stored).includes('base64'), false);
  assert.equal(stored[0].content[0].text, 'make it winter');
  assert.equal(stored[0].content[1].type, 'image_omitted');
  // The agent structure has to survive or a reloaded chat can't be continued.
  assert.equal(stored[1].tool_calls.length, 1);
  assert.equal(stored[2].role, 'tool');
  assert.equal(stored[3].content, 'Made it wintry.');
});

test('historyUpsert keeps newest first and replaces by id', () => {
  let list = [];
  list = C.historyUpsert(list, { id: 'a', title: 'A', updated: 1, turns: [] });
  list = C.historyUpsert(list, { id: 'b', title: 'B', updated: 2, turns: [] });
  assert.deepEqual(list.map(c => c.id), ['b', 'a']);
  // Continuing an older conversation moves it back to the top, not duplicates it.
  list = C.historyUpsert(list, { id: 'a', title: 'A2', updated: 3, turns: [] });
  assert.deepEqual(list.map(c => c.id), ['a', 'b']);
  assert.equal(list[0].title, 'A2');
});

test('historyUpsert caps by count and by size', () => {
  let list = [];
  for (let i = 0; i < 10; i++) {
    list = C.historyUpsert(list, { id: 'c' + i, title: 't', updated: i, turns: [] }, { maxItems: 4 });
  }
  assert.equal(list.length, 4);
  assert.deepEqual(list.map(c => c.id), ['c9', 'c8', 'c7', 'c6']);

  // Quota safety: a few very long conversations must not wedge the store.
  const fat = n => ({ id: 'f' + n, title: 't', updated: n, turns: [{ role: 'user', content: 'x'.repeat(5000) }] });
  let big = [];
  for (let i = 0; i < 10; i++) big = C.historyUpsert(big, fat(i), { maxItems: 50, maxBytes: 12000 });
  assert.ok(big.length < 10 && big.length >= 1);
  assert.ok(JSON.stringify(big).length <= 12000);
  // Whatever survives, the one just written is in it.
  assert.equal(big[0].id, 'f9');
});

test('historyRemove drops one conversation and leaves the rest', () => {
  const list = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
  assert.deepEqual(C.historyRemove(list, 'b').map(c => c.id), ['a', 'c']);
  assert.deepEqual(C.historyRemove(list, 'zz').map(c => c.id), ['a', 'b', 'c']);
});

// ── Model pill ──────────────────────────────────────────────────────────────

test('shortModelName drops the org prefix for the composer pill', () => {
  assert.equal(C.shortModelName('mlx-community/Qwen3.5-2B-bf16'), 'Qwen3.5-2B-bf16');
  assert.equal(C.shortModelName('qwen3.6-27b'), 'qwen3.6-27b');
  assert.equal(C.shortModelName(''), 'no model');
  assert.equal(C.shortModelName(null), 'no model');
});

test('modelSubtitle says something true about the model', () => {
  assert.match(C.modelSubtitle(FLEET[1]), /ready/);           // qwen3.6-27b, loaded
  assert.match(C.modelSubtitle(FLEET[0]), /vision/);          // gemma-4-e4b, vision
  assert.match(C.modelSubtitle(FLEET[0]), /GB|MB/);
});

// ── Markdown ────────────────────────────────────────────────────────────────
// Model replies arrive as markdown and read as noise unrendered. Everything
// here is built from ESCAPED input — the text is model output and may quote a
// user, so it is never trusted.

test('renderMarkdown handles the shapes models actually emit', () => {
  assert.match(C.renderMarkdown('## Features'), /<h2>Features<\/h2>/);
  assert.match(C.renderMarkdown('**bold** and *italic*'), /<strong>bold<\/strong> and <em>italic<\/em>/);
  assert.match(C.renderMarkdown('call `POST /v1/models`'), /<code>POST \/v1\/models<\/code>/);
  const list = C.renderMarkdown('- one\n- two');
  assert.match(list, /<ul><li>one<\/li><li>two<\/li><\/ul>/);
  const ol = C.renderMarkdown('1. first\n2. second');
  assert.match(ol, /<ol><li>first<\/li><li>second<\/li><\/ol>/);
  const fence = C.renderMarkdown('```bash\ncurl -s http://x/v1/models\n```');
  assert.match(fence, /<pre><code>curl -s http:\/\/x\/v1\/models\n?<\/code><\/pre>/);
  assert.match(C.renderMarkdown('para one\n\npara two'), /<p>para one<\/p>\s*<p>para two<\/p>/);
});

test('renderMarkdown escapes everything and never emits a dangerous href', () => {
  const out = C.renderMarkdown('<script>alert(1)</script> & <img onerror=x>');
  assert.equal(/<script|<img/.test(out), false);
  assert.match(out, /&lt;script&gt;/);
  assert.match(out, /&amp;/);

  // Inside a fence too — a fenced block is still model output.
  assert.equal(/<script/.test(C.renderMarkdown('```\n<script>x</script>\n```')), false);

  // http(s) links become links; anything else stays inert text.
  assert.match(C.renderMarkdown('[docs](https://example.com/a)'), /<a href="https:\/\/example\.com\/a"[^>]*>docs<\/a>/);
  const js = C.renderMarkdown('[x](javascript:alert(1))');
  assert.equal(/href/.test(js), false);
  assert.match(js, /x/);
});

// ── Formatting ──────────────────────────────────────────────────────────────

test('turn stats come from the server, because a buffered stream has no clock', () => {
  // With `tools` present the server buffers tokens for tool-call detection and
  // flushes at the end, so every SSE delta lands at once: client wall-clock
  // decode time is ~0 and the console reported 937 tok/s on a 2B. The final
  // chunk already carries the server's own measurement, which buffering cannot
  // distort — use that.
  const t1 = { prompt_n: 900, prompt_ms: 1420.5, predicted_n: 40, predicted_ms: 700, predicted_per_second: 57.1 };
  const t2 = { prompt_n: 60, prompt_ms: 120, predicted_n: 20, predicted_ms: 300 };

  let acc = C.addTimings(null, t1);
  assert.deepEqual(acc, { prefillMs: 1420.5, tokens: 40, decodeMs: 700 });
  acc = C.addTimings(acc, t2);
  assert.deepEqual(acc, { prefillMs: 1540.5, tokens: 60, decodeMs: 1000 });

  // 60 tokens over 1.0s of DECODE — the seconds spent generating an image
  // between two text rounds are not in the denominator.
  assert.equal(C.formatTurnStats(acc), '1.54s prefill  ·  60 tok/s');

  assert.equal(C.addTimings(null, null).tokens, 0);
  assert.equal(C.formatTurnStats(null), '');
  assert.equal(C.formatTurnStats({ prefillMs: 0, tokens: 0, decodeMs: 0 }), '');
});

test('chatDelta surfaces the timings block', () => {
  const d = C.chatDelta({ choices: [{ delta: {}, finish_reason: 'stop' }], timings: { predicted_n: 5, predicted_ms: 100 } });
  assert.deepEqual(d.timings, { predicted_n: 5, predicted_ms: 100 });
});

test('tokensPerSecond needs both a count and elapsed time', () => {
  assert.equal(C.tokensPerSecond(120, 2000), 60);
  assert.equal(C.tokensPerSecond(120, 0), null);
  assert.equal(C.tokensPerSecond(0, 1000), null);
  assert.equal(C.tokensPerSecond(null, 1000), null);
});

test('formatBytes reads in GB/MB', () => {
  assert.equal(C.formatBytes(0), '—');
  assert.equal(C.formatBytes(null), '—');
  assert.equal(C.formatBytes(512 * 1024 * 1024), '512 MB');
  assert.equal(C.formatBytes(8 * 1024 * 1024 * 1024), '8.0 GB');
  // MLX holds a few KB at idle; a flat "0 MB" in the header reads as a failed
  // /props fetch rather than as an idle server.
  assert.equal(C.formatBytes(4096), '<1 MB');
});

test('errorText prefers the server\'s own message', () => {
  // Media-gen 400s say exactly why ("instruction editing requires a FLUX.2 or
  // Mage-Flow-Edit model"). Swallowing that for a generic "request failed" is
  // the whole reason the Images tab offers Edit on every image model.
  assert.equal(
    C.errorText({ error: { message: 'mask is not supported' } }, 400),
    'mask is not supported',
  );
  assert.equal(C.errorText({ error: 'flat string' }, 400), 'flat string');
  assert.match(C.errorText(null, 503), /503/);
});

// ── Voice mode ──────────────────────────────────────────────────────────────
// STT is the browser's (Web Speech); TTS is Kokoro on this server. Everything
// that can be wrong without a microphone lives here.

test('speakableChunks strips markup instead of reading it aloud', () => {
  // Raw markdown read by a TTS model is unusable: a fence becomes minutes of
  // punctuation and a URL becomes alphabet soup.
  assert.deepEqual(C.speakableChunks('Here is **bold** and _italic_ text.'),
                   ['Here is bold and italic text.']);
  assert.deepEqual(C.speakableChunks('# Heading\nBody text here.'),
                   ['Heading Body text here.']);
  assert.deepEqual(C.speakableChunks('See [the docs](https://example.com/x) now.'),
                   ['See the docs now.']);
  assert.deepEqual(C.speakableChunks('Go to https://example.com/very/long/path please.'),
                   ['Go to a link please.']);
  assert.deepEqual(C.speakableChunks('Use `git status` first.'), ['Use git status first.']);
  assert.deepEqual(C.speakableChunks('- one\n- two\n- three'), ['one two three']);
  assert.deepEqual(C.speakableChunks('> quoted line here.'), ['quoted line here.']);
});

test('speakableChunks announces a code block rather than reciting it', () => {
  const out = C.speakableChunks('First line.\n```js\nlet x = 1;\nfoo(bar);\n```\nAfter.');
  const joined = out.join(' ');
  assert.ok(joined.includes('(code block)'), 'the listener must know something was skipped');
  assert.ok(!joined.includes('let x'), 'code must not be spoken');
  assert.ok(!joined.includes('```'));
});

test('speakableChunks handles an UNTERMINATED fence (a stream cut mid-block)', () => {
  const out = C.speakableChunks('Here you go:\n```python\nimport os');
  const joined = out.join(' ');
  assert.ok(!joined.includes('import os'), 'a half-streamed fence must not be recited');
  assert.ok(joined.includes('(code block)'));
});

test('speakableChunks splits on sentence ends and keeps the terminator', () => {
  // Question prosody depends on the model still seeing the "?".
  const out = C.speakableChunks('One thing happened. Then another thing? Yes indeed!');
  assert.equal(out.length, 3);
  assert.ok(out[0].endsWith('.'));
  assert.ok(out[1].endsWith('?'));
  assert.ok(out[2].endsWith('!'));
});

test('speakableChunks merges runts so a word is not its own round trip', () => {
  const out = C.speakableChunks('This is a full sentence. OK.');
  assert.equal(out.length, 1, 'a 3-character sentence should ride along');
  assert.ok(out[0].includes('OK'));
});

test('speakableChunks caps long clauses under Kokoro context', () => {
  // Kokoro's context is 510 phoneme tokens: an uncapped clause 400s rather
  // than truncating, which would drop the sentence silently.
  const long = 'word '.repeat(400).trim() + '.';
  const out = C.speakableChunks(long);
  assert.ok(out.length > 1, 'must be split');
  for (const c of out) assert.ok(c.length <= 300, `chunk too long: ${c.length}`);
});

test('speakableChunks on empty or markup-only input yields nothing to say', () => {
  assert.deepEqual(C.speakableChunks(''), []);
  assert.deepEqual(C.speakableChunks(null), []);
  assert.deepEqual(C.speakableChunks('```\ncode only\n```'), ['(code block)']);
  assert.deepEqual(C.speakableChunks('---'), []);
});

test('sttSupported detects both the standard and webkit prefixes', () => {
  assert.equal(C.sttSupported({ SpeechRecognition: function () {} }), true);
  assert.equal(C.sttSupported({ webkitSpeechRecognition: function () {} }), true);
  assert.equal(C.sttSupported({}), false);
  assert.equal(C.sttSupported(null), false);
});

test('the mic runs ONLY while listening', () => {
  // Leaving it live during playback makes the page transcribe the assistant's
  // own voice and answer its own sentence.
  assert.equal(C.micShouldRun('listening'), true);
  assert.equal(C.micShouldRun('speaking'), false);
  assert.equal(C.micShouldRun('thinking'), false);
  assert.equal(C.micShouldRun('off'), false);
});

test('voice state machine only returns to listening AFTER speech finishes', () => {
  let s = 'off';
  s = C.voiceNext(s, 'enable');      assert.equal(s, 'listening');
  s = C.voiceNext(s, 'transcript');  assert.equal(s, 'thinking');
  s = C.voiceNext(s, 'reply');       assert.equal(s, 'speaking');
  // The event that would reopen the mic mid-playback must be inert.
  assert.equal(C.voiceNext('speaking', 'transcript'), 'speaking');
  s = C.voiceNext(s, 'spoken');      assert.equal(s, 'listening');
});

test('voice state machine: disable always wins, from any state', () => {
  for (const st of C.VOICE_STATES) {
    if (st === 'off') continue;
    assert.equal(C.voiceNext(st, 'disable'), 'off', `${st} must be cancellable`);
  }
});

test('voice state machine: a failed turn returns to listening, not a dead end', () => {
  // An empty reply or an error must not strand voice mode in "thinking" with
  // the mic off — that reads as the feature being broken.
  assert.equal(C.voiceNext('thinking', 'error'), 'listening');
});

test('voice state machine ignores nonsense transitions', () => {
  assert.equal(C.voiceNext('off', 'spoken'), 'off');
  assert.equal(C.voiceNext('listening', 'reply'), 'listening');
  assert.equal(C.voiceNext('off', 'bogus'), 'off');
});

test('speechBody never sends both voice and ref_audio', () => {
  // They belong to DIFFERENT backends and each is a named 400 on the other.
  const withVoice = C.speechBody({ model: 'm', text: 'hi', voice: 'af_heart', refAudio: 'AAAA' });
  assert.equal(withVoice.voice, 'af_heart');
  assert.equal(withVoice.ref_audio, undefined);

  const withClip = C.speechBody({ model: 'm', text: 'hi', refAudio: 'AAAA' });
  assert.equal(withClip.ref_audio, 'AAAA');
  assert.equal(withClip.voice, undefined);

  const plain = C.speechBody({ model: 'm', text: 'hi' });
  assert.deepEqual(plain, { model: 'm', input: 'hi' });
});
