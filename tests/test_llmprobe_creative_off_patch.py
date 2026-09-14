"""Keep the disposable llmprobe benchmark/eval thinking-off patch honest."""

import shutil
import subprocess
import unittest

from scripts.patch_llmprobe_creative_off import patch_bundle


class ProbePatchTest(unittest.TestCase):
    def test_timed_and_eval_chat_requests_explicitly_disable_thinking(self):
        source = '''async function timedRun(ctx, surface, text, maxTokens, extra, system) {
  const body = {
    stream: true
  };
  let timed;
  timed = await ctx.client.streamTimed(
      adapter.path,
      body,
      adapter.headers(ctx.config),
      null
    );
}
async function runReasoning(ctx, opts) {
  const surface = ctx.evalSurface;
  const request = {
          allowReasoning: false
        };
        const sendOpts = {};
}
'''
        patched = patch_bundle(source)
        self.assertIn('...(surface === "chat" ? { enable_thinking: false } : {})', patched)
        self.assertIn('...(surface === "chat" ? { extra: { enable_thinking: false } } : {})', patched)
        self.assertIn('"X-MTPLX-Cache-Mode": "bypass"', patched)
        self.assertEqual(source.count("enable_thinking"), 0)
        self.assertEqual(patched.count("enable_thinking"), 2)

    @unittest.skipUnless(shutil.which("node"), "llmprobe requires Node")
    def test_timed_seed_is_stable_across_paired_requests(self):
        source = '''async function timedRun(ctx, surface, text, maxTokens, extra, system) {
  const adapter = ctx.adapters.get(surface);
  const body = {
    ...adapter.buildBody({ temperature: 1, topP: 0.95, maxTokens }),
    stream: true
  };
  let timed;
  timed = await ctx.client.streamTimed(
      adapter.path,
      body,
      adapter.headers(ctx.config),
      null
    );
}
async function runReasoning(ctx, opts) {
  const surface = ctx.evalSurface;
  const request = {
          allowReasoning: false
        };
        const sendOpts = {};
}
'''
        patched = patch_bundle(source)
        check = '''
const assert = require('node:assert/strict');
const sent = [];
const adapter = {
  path: '/v1/chat/completions',
  headers: () => ({}),
  buildBody: q => ({ temperature: q.temperature, top_p: q.topP, max_tokens: q.maxTokens })
};
const ctx = {
  adapters: new Map([['chat', adapter]]), config: {},
  client: { streamTimed: async (_path, body, headers) => { sent.push({ body, headers }); } }
};
(async () => {
  await timedRun(ctx, 'chat', 'same prompt', 192);
  await timedRun(ctx, 'chat', 'same prompt', 192);
  await timedRun(ctx, 'chat', 'other prompt', 192);
  assert.deepEqual(sent[0], sent[1]);
  assert.notEqual(sent[0].body.seed, sent[2].body.seed);
  assert.equal(sent[0].body.temperature, 1);
  assert.equal(sent[0].body.top_p, 0.95);
  assert.equal(sent[0].body.max_tokens, 192);
  assert.equal(sent[0].body.enable_thinking, false);
  assert.equal(sent[0].headers['X-MTPLX-Cache-Mode'], 'bypass');
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
        subprocess.run(["node", "-e", patched + check], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
