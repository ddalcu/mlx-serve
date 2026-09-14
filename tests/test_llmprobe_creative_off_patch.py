"""Keep the disposable llmprobe benchmark/eval thinking-off patch honest."""

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


if __name__ == "__main__":
    unittest.main()
