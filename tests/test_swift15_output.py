#!/usr/bin/env python3
"""Swift-1.5 text regression against a running server: python3 tests/test_swift15_output.py --port 11235."""

import argparse
import json
import unittest
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=11235)
parser.add_argument("--model", default="ukisai/Swift-1.5-4bit-MLX")
args, unittest_args = parser.parse_known_args()
BASE = f"http://127.0.0.1:{args.port}"


def post(path, body):
    return urllib.request.urlopen(
        urllib.request.Request(
            BASE + path,
            json.dumps(body, ensure_ascii=False).encode(),
            {"Content-Type": "application/json"},
        ),
        timeout=300,
    )


def served_models():
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as response:
            return [row["id"] for row in json.load(response)["data"]]
    except Exception:
        return []


def setUpModule():
    models = served_models()
    if not models:
        raise unittest.SkipTest(f"SKIP: no server on {BASE}")
    if args.model not in models:
        raise unittest.SkipTest(f"SKIP: {args.model} not served on {BASE} (have {models})")


def request_body(stream):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": "请用中文回答：一加一等于几？然后写一句问候。"}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": stream,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }


class SwiftOutputTests(unittest.TestCase):
    def assert_answer(self, text):
        self.assertTrue(text.strip(), "empty assistant answer")
        self.assertNotIn("\ufffd", text, "replacement characters in generated text")
        self.assertRegex(text, r"[\u4e00-\u9fff]", "no Chinese in the answer")
        self.assertRegex(text, r"二|两|2", "missing the answer to one plus one")
        print(f"\nAssistant: {text}")

    def test_nonstream_chinese(self):
        with post("/v1/chat/completions", request_body(False)) as response:
            result = json.load(response)
        self.assert_answer(result["choices"][0]["message"].get("content", ""))
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertGreater(result["usage"]["completion_tokens"], 0)
        print("Timings:", json.dumps(result.get("timings"), ensure_ascii=False))

    def test_stream_chinese(self):
        content = []
        finished = False
        done = False
        with post("/v1/chat/completions", request_body(True)) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                for choice in chunk.get("choices", []):
                    content.append(choice.get("delta", {}).get("content") or "")
                    if choice.get("finish_reason"):
                        self.assertEqual(choice["finish_reason"], "stop")
                        finished = True
        self.assert_answer("".join(content))
        self.assertTrue(finished, "no finish event")
        self.assertTrue(done, "no SSE done marker")


if __name__ == "__main__":
    unittest.main(argv=[__file__, *unittest_args])
