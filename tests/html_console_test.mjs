// Hermetic checks of the shipped console; no DOM, network or npm packages.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { test } from 'node:test';
export const read = (name) => readFileSync(new URL('../src/html/' + name, import.meta.url), 'utf8');
export function loadConsole() {
  const context = vm.createContext({
    URL,
    Blob,
    FormData,
    TextEncoder,
    TextDecoder,
    ReadableStream,
    AbortController,
    EventTarget,
    performance,
    structuredClone,
    setTimeout,
    clearTimeout,
    atob,
    btoa,
    crypto,
    console
  });
  vm.runInContext(read('api.js') + '\n' + read('app.js'), context);
  assert.ok(context.__mlxConsole, 'the DOM-free console hook is available');
  return context.__mlxConsole;
}
export const C = loadConsole();
const plain = (v) => JSON.parse(JSON.stringify(v));
const model = (id, capabilities, architecture) => ({
  id,
  capabilities,
  architecture,
  meta: { architecture }
});
const models = [
  model('flux', ['image'], 'flux2'),
  model('mage-base', ['image'], 'mage_flow'),
  model('kokoro', ['audio'], 'kokoro'),
  model('ace', ['audio', 'music'], 'acestep'),
  model('qwen-tts', ['audio'], 'qwen3_tts')
];

test('SSE frames survive every byte split, CRLF, comments, UTF-8 and DONE', async () => {
  const bytes = new TextEncoder().encode(
    ': keepalive\r\nevent: progress\r\nid: 7\r\ndata: hé\r\ndata: two\r\n\r\ndata: [DONE]\n\n'
  );
  const expected = [
    { event: 'progress', data: 'hé\ntwo', id: '7' },
    { event: 'message', data: '[DONE]', id: '7' }
  ];
  for (let split = 1; split < bytes.length; split++) {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(bytes.slice(0, split));
        c.enqueue(bytes.slice(split));
        c.close();
      }
    });
    const out = [];
    for await (const event of C.readSSE(stream)) out.push(event);
    assert.deepEqual(plain(out), expected);
  }
  const stream = new ReadableStream({
    start(c) {
      for (const byte of bytes) c.enqueue(Uint8Array.of(byte));
      c.close();
    }
  });
  const out = [];
  for await (const event of C.readSSE(stream)) out.push(event);
  assert.deepEqual(plain(out), expected);
});

test('chat and edit builders preserve parameters and repeated image order', async () => {
  assert.deepEqual(
    plain(C.buildChatRequest({ model: 'chat', messages: [], temperature: 0, stream: false })),
    {
      model: 'chat',
      messages: [],
      temperature: 0,
      stream: true,
      stream_options: { include_usage: true }
    }
  );
  const form = C.buildEditRequest({ model: 'flux', prompt: 'blue', seed: 0, unused: undefined }, [
    new Blob(['first'], { type: 'image/png' }),
    new Blob(['second'], { type: 'image/jpeg' })
  ]);
  assert.equal(form.get('stream'), 'false');
  assert.equal(form.get('seed'), '0');
  assert.equal(form.get('response_format'), 'b64_json');
  assert.equal(form.has('unused'), false);
  assert.deepEqual(await Promise.all(form.getAll('image[]').map((b) => b.text())), [
    'first',
    'second'
  ]);
  assert.throws(() => C.buildEditRequest({ prompt: 'x' }, []), /reference image/);
});

test('capabilities separate speech, music and edit-capable image models', () => {
  const parsed = C.parseModels({ data: models });
  assert.deepEqual(
    plain(parsed.filter((m) => m.capabilities.includes('speech')).map((m) => m.id)),
    ['kokoro', 'qwen-tts']
  );
  const { tools } = C.browserTools(
    { baseUrl: 'https://host' },
    models,
    {},
    { server: 'https://host', messages: [] }
  );
  const ids = (name) =>
    plain(tools.find((t) => t.function.name === name).function.parameters.properties.model.enum);
  assert.deepEqual(ids('generate_speech'), ['kokoro', 'qwen-tts']);
  assert.deepEqual(ids('generate_music'), ['ace']);
  assert.deepEqual(
    ids('edit_image'),
    models
      .filter((m) => C.imageProfile(m)?.edit || m.capabilities.includes('image-edit'))
      .map((m) => m.id)
  );
  assert.ok(!ids('edit_image').includes('mage-base'));
});

test('edit execution accepts exactly the advertised edit-model enum', async () => {
  const opened = [];
  const client = {
    baseUrl: 'https://host',
    open: async (path, init) => {
      opened.push([path, init]);
      throw Error('fixture request');
    }
  };
  const session = {
    server: client.baseUrl,
    messages: [{ role: 'user', images: ['data:image/png;base64,YQ=='] }]
  };
  const { tools, execute } = C.browserTools(client, models, {}, session);
  const allowed = tools.find((t) => t.function.name === 'edit_image').function.parameters.properties
    .model.enum;
  for (const m of models) {
    const count = opened.length;
    await assert.rejects(
      execute(
        { name: 'edit_image', args: { model: m.id, prompt: 'blue' } },
        new AbortController().signal
      ),
      allowed.includes(m.id) ? /fixture request/ : /advertised model/
    );
    assert.equal(opened.length - count, allowed.includes(m.id) ? 1 : 0);
  }
  assert.equal(opened[0][0], '/v1/images/edits');
  assert.equal(opened[0][1].body.get('model'), 'flux');
});

test('model Markdown keeps HTML text and permits only absolute HTTP(S) links', () => {
  const html = C.renderMarkdown('<script>alert(1)</script>\n\n```html\n<script>x</script>\n```');
  assert.ok(html.includes('&lt;script&gt;'));
  assert.doesNotMatch(html, /<script[ >]/i);
  for (const url of [
    'javascript:alert(1)',
    'data:text/html,hi',
    'mailto:a@b.test',
    '/relative',
    '#anchor',
    '//host/path',
    'ftp://host/file'
  ])
    assert.doesNotMatch(C.renderMarkdown('[click](' + url + ')'), /href=/, url);
  for (const url of ['https://example.test/a', 'http://example.test/b'])
    assert.match(C.renderMarkdown('[click](' + url + ')'), /href="https?:/);
  assert.doesNotMatch(C.renderMarkdown('![tracking](https://example.test/pixel)'), /<img/);
});

test('one generation attempt per turn, including later rounds and failures; library still works', async () => {
  for (const fail of [false, true]) {
    let round = 0;
    const executed = [],
      results = [];
    const tools = ['generate_image', 'generate_speech', 'search_library'].map((name) => ({
      type: 'function',
      function: { name }
    }));
    const stream = async function* () {
      const names =
        round++ === 0
          ? ['generate_image', 'generate_speech', 'search_library']
          : round === 2
            ? ['generate_image']
            : [];
      if (names.length)
        yield {
          type: 'tools',
          delta: names.map((name, index) => ({
            index,
            id: `r${round}-${index}`,
            function: { name, arguments: '{}' }
          }))
        };
      yield { type: 'finish', reason: names.length ? 'tool_calls' : 'stop' };
    };
    for await (const e of C.toolLoop(
      { redact: (s) => s },
      { messages: [] },
      {
        tools,
        stream,
        execute: async (call) => {
          executed.push(call.name);
          if (fail && call.name === 'generate_image') throw Error('failed');
          return { text: 'ok' };
        }
      }
    ))
      if (e.type === 'tool-result') results.push(plain(e.round));
    assert.deepEqual(executed, ['generate_image', 'search_library']);
    assert.ok(
      results.some((r) =>
        r.calls.some((c) => c.status === 'error' && /one media generation/i.test(c.result))
      )
    );
  }
});

test('speakable prose omits fences and URLs; every chunk is at most 300 characters', () => {
  for (const text of [
    'Yes indeed! Hello **world**. ```js\nsecret()',
    'word '.repeat(250),
    'x'.repeat(901)
  ]) {
    const chunks = C.speechChunks(text);
    assert.ok(chunks.length);
    assert.ok(chunks.every((s) => s.length <= 300));
    assert.doesNotMatch(chunks.join(' '), /secret|```|\*\*/);
  }
  assert.deepEqual(
    plain(C.speechChunks('Read [this](https://example.test). https://example.test')),
    ['Read this.']
  );
});

test('voice waits for recognition end, never listens while speaking, and Stop cancels playback', async () => {
  let mic = false,
    recognition,
    finishPlayback;
  const phases = [];
  const loop = new C.VoiceLoop({
    recognition: () =>
      (recognition = {
        start() {
          mic = true;
        },
        abort() {}
      }),
    send: async () => {
      assert.equal(mic, false);
      return 'One. Two.';
    },
    synthesize: async (text) => {
      assert.equal(mic, false);
      return new Blob([text]);
    },
    play: async (_blob, signal) => {
      assert.equal(mic, false);
      await new Promise((resolve) => {
        finishPlayback = resolve;
        signal.addEventListener('abort', resolve, { once: true });
      });
    },
    stopChat() {},
    changed() {
      phases.push(loop.phase);
    }
  });
  const done = loop.start();
  recognition.onresult({ results: [{ isFinal: true, 0: { transcript: 'hello' } }] });
  assert.equal(loop.phase, 'listening'); // abort requested; hardware has not ended yet
  mic = false;
  recognition.onend();
  for (let i = 0; i < 20 && !finishPlayback; i++)
    await new Promise((resolve) => setImmediate(resolve));
  assert.equal(loop.phase, 'speaking');
  assert.equal(mic, false);
  loop.stop();
  await done;
  assert.equal(loop.phase, 'off');
  assert.deepEqual(phases, ['listening', 'thinking', 'speaking', 'off']);
});

test('theme and English boot honor stored/OS choice before paint', () => {
  for (const stored of [null, 'light', 'dark', 'system', 'bogus', 'broken', 'throws'])
    for (const dark of [false, true]) {
      const root = { dataset: {}, lang: '', hasAttribute: () => true };
      const env = vm.createContext({
        document: { documentElement: root, addEventListener() {} },
        window: { addEventListener() {} },
        EventTarget,
        URL,
        localStorage: {
          getItem() {
            if (stored === 'throws') throw Error('blocked');
            return stored === 'broken' ? '[' : JSON.stringify({ theme: stored });
          }
        },
        matchMedia: () => ({ matches: dark })
      });
      vm.runInContext(read('api.js') + '\n' + read('theme.js') + '\n' + read('i18n.js'), env);
      assert.equal(
        root.dataset.theme,
        ['light', 'dark'].includes(stored) ? stored : dark ? 'dark' : 'light'
      );
      assert.equal(root.lang, 'en');
      assert.equal(env.__mlxConsole, undefined);
    }
  const html = read('index.html');
  assert.ok(html.indexOf('<script>') < html.indexOf('<style>'));
});

test('type scales with browser preferences: no pixel font sizes or fixed root', () => {
  const css = read('app.css').replace(/\/\*[\s\S]*?\*\//g, '');
  assert.doesNotMatch(css, /(?:font-size|--[\w-]*font[\w-]*size)\s*:[^;}]*\dpx/i);
  assert.doesNotMatch(css, /(?:^|[;{])\s*font\s*:[^;}]*\dpx/m);
  assert.doesNotMatch(css, /(?:^|})\s*(?:html|:root)\s*\{[^}]*(?:^|[;{])\s*font-size\s*:/m);
  assert.match(css, /font-size:\s*[.\d]+rem/);
});

test('static console wiring: referenced ids exist and id-bearing controls have consumers', () => {
  const bundle = read('app.js');
  const ui = bundle.split(/(?=^\/\/ src\/)/m).filter((s) => s.startsWith('// src/ui/'));
  const source = ui.join('\n');
  const markup = source + '\n' + read('index.html');
  const ids = new Set([...markup.matchAll(/\bid="([\w-]+)"/g)].map((m) => m[1]));
  // These factories render id="${id}"; resolve their literal call sites.
  for (const m of source.matchAll(/(?:imageMenu|mediaChooser|menu)\(\s*"([\w-]+)"/g)) {
    ids.add(m[1]);
    ids.add(m[1] + '-menu');
  }
  for (const m of source.matchAll(/\.id\s*=\s*"([\w-]+)"/g)) ids.add(m[1]);
  const video = ui.find((s) => s.startsWith('// src/ui/video-view.js'));
  assert.ok(video.includes('id="video-${id}"'));
  for (const m of video.matchAll(/this\.(?:slider|toggle)\(\s*"([\w-]+)"/g))
    ids.add('video-' + m[1]);
  assert.ok(source.includes('id="bar-${id}"'));
  for (const id of ['gpu', 'prefill']) ids.add('bar-' + id); // cards' bar-${id} template
  const refs = new Set(
    [...source.matchAll(/["'`]#([\w-]+)(?:["'`](?!\s*\+)|\s)/g)].map((m) => m[1])
  );
  for (const id of ['chat', 'monitor', 'api']) refs.delete(id); // URL fragments, not DOM selectors
  assert.deepEqual([...refs].filter((id) => !ids.has(id)).sort(), [], 'selectors without markup');
  for (const part of ui) {
    const controls = [
      ...part.matchAll(/<(?:input|select|textarea|button)\b[^>]*\bid="([\w-]+)"[^>]*>/g)
    ];
    const missing = [];
    for (const [tag, id] of controls) {
      if (refs.has(id)) continue;
      // Name/data-based form delegation and type selectors also consume controls.
      const name = tag.match(/\bname="([\w-]+)"/)?.[1];
      const byName = name && part.includes('form.oninput =') && part.includes('target.name');
      const byType = id === 'model-query' && part.includes('querySelector("input")');
      const rewrite = id === 'video-rewrite-text' && part.includes('querySelector("textarea")');
      const byData = [...tag.matchAll(/\b(data-[\w-]+)=/g)].some((m) =>
        part.includes('[' + m[1] + ']')
      );
      if (!byName && !byType && !rewrite && !byData) missing.push(id);
    }
    assert.deepEqual(
      [...new Set(missing)].sort(),
      [],
      'controls without a selector or form consumer: ' + part.split('\n')[0]
    );
  }
});

test('tool resolution honors a named model and rejects a hallucinated model before HTTP', async () => {
  const sent = [];
  const client = { baseUrl: 'https://host', open: async (_path, init) => {
    sent.push(JSON.parse(init.body)); throw Error('fixture request');
  } };
  const { execute } = C.browserTools(client, models, {});
  const run = (model) => execute({ name: 'generate_image', args: { model, prompt: 'fox' } }, new AbortController().signal);
  await assert.rejects(run('flux'), /fixture request/);
  assert.equal(sent[0].model, 'flux');
  await assert.rejects(run('dall-e-3'), /advertised model/);
  assert.equal(sent.length, 1);
});

test('tool resolution refuses a modality absent from the server before HTTP', async () => {
  let requests = 0;
  const { tools, execute } = C.browserTools({ baseUrl: 'https://host', open: () => { requests++; } }, [], {});
  assert.deepEqual(plain(tools.map(t => t.function.name)), ['search_library']);
  await assert.rejects(execute({ name: 'generate_music', args: { prompt: 'lofi' } }, new AbortController().signal), /music/);
  assert.equal(requests, 0);
});

test('tool resolution prefers resident, then healthy complete models without reordering discovery', async () => {
  const fleet = [
    { ...models[2], id: 'broken', state: 'error', bytes_on_disk: 100 },
    { ...models[2], id: 'incomplete', bytes_on_disk: null },
    { ...models[2], id: 'cold', bytes_on_disk: 100 },
    { ...models[2], id: 'resident', loaded: true, bytes_on_disk: 100 }
  ];
  for (const [available, expected] of [[fleet, 'resident'], [fleet.slice(0, 3), 'cold'], [fleet.slice(1, 2), 'incomplete']]) {
    let body;
    const { tools, execute } = C.browserTools({ baseUrl: 'https://host', bytes: async (_path, init) => {
      body = JSON.parse(init.body); throw Error('fixture request');
    } }, C.parseModels({ data: available }), {});
    assert.ok(!tools.find(t => t.function.name === 'generate_speech').function.parameters.required.includes('model'));
    await assert.rejects(execute({ name: 'generate_speech', args: { prompt: 'hello' } }, new AbortController().signal), /fixture request/);
    assert.equal(body.model, expected);
  }
  assert.deepEqual(fleet.map(m => m.id), ['broken', 'incomplete', 'cold', 'resident']);
});

test('tool system prompt carries the real mounted base, inventory, API fields and question rules onto the wire', async () => {
  const baseUrl = C.pageServer({ origin: 'https://inference.example:8443', pathname: '/proxy/mlx/index.html' });
  const options = C.browserTools({ baseUrl }, models, {});
  const request = { messages: [{ role: 'system', content: 'User custom instruction.' }, { role: 'user', content: 'Show me curl for edits.' }] };
  let sent;
  for await (const _ of C.toolLoop({}, request, { ...options, stream: async function* (_client, req) {
    sent = req.messages; yield { type: 'finish', reason: 'stop' };
  } })) {}
  const prompt = sent[0].content;
  assert.match(prompt, /https:\/\/inference.example:8443\/proxy\/mlx/);
  for (const m of models) assert.ok(prompt.includes(m.id));
  assert.match(prompt, /image.*flux2|flux2.*image/);
  assert.match(prompt, /POST \/v1\/images\/edits/);
  assert.match(prompt, /multipart\/form-data/);
  assert.match(prompt, /image\[\]/);
  assert.match(prompt, /mask/);
  assert.match(prompt, /stream:true/);
  assert.match(prompt, /questions.*text.*no tool/is);
  assert.match(prompt, /one media generation/i);
  assert.equal(sent[1].content, 'User custom instruction.');
  assert.equal(request.messages.length, 2);
});

test('real-size LTX video decodes without per-byte intermediate arrays', () => {
  const length = 768 * 512 * 97 * 3;
  const data = 'AQID'.repeat(length / 3);
  const input = { format: 'rgb8', width: 768, height: 512, frames: 97, fps: 24, data };
  const raw = C.decodeVideo(input);
  assert.equal(raw.rgb.length, length);
  for (const i of [0, 1, 2, 49151, 49152, length - 3, length - 2, length - 1])
    assert.equal(raw.rgb[i], i % 3 + 1);
  assert.throws(() => C.decodeVideo({ ...input, data: data.slice(0, -1) + '!' }), /Invalid/);
  assert.throws(() => C.decodeVideo({ ...input, frames: 96 }), /large/);
});
