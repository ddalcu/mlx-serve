"""Hermetic fault injection for the live media suite's pass/fail decisions."""
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_media_surfaces as surfaces


MESSAGE = ("Stored media cannot be reconstructed. Resend the complete image "
           "history without previous_response_id.")


class MediaSurfaceContractTests(unittest.TestCase):
    def run_suite(self, mutation=None):
        statuses = [400] * 4 + [200, 200, 400] + [200] * 3 + [400, 400] + [200] * 4 + [400, 400]
        http = [{"id": "resp_test", "text": "red blue"} for _ in statuses]
        for index in (16, 17):
            http[index] = {"error": {"type": "incomplete_media_history", "code": 400, "message": MESSAGE}}
        rejection = {"type": "error", "status": 400, "error": {
            "code": "incomplete_media_history", "message": MESSAGE}}
        ws = [[{"type": "error", "status": 400, "error": {
            "code": "invalid_request_error", "message": "Cannot prepare the complete media history: InvalidImage"}}],
              [{"type": "response.completed", "response": {"status": "completed", "text": "red"}}],
              copy.deepcopy(rejection), copy.deepcopy(rejection)]
        if mutation:
            mutation(http, ws)
        replies = iter(zip(statuses, http))
        socket_replies = iter(ws)

        def urlopen(request, **kwargs):
            status, value = next(replies)
            response = io.BytesIO(json.dumps(value).encode())
            if status >= 400:
                raise surfaces.urllib.error.HTTPError(request.full_url, status, "Bad Request", {}, response)
            response.status = status
            return response

        def run(*args, **kwargs):
            return subprocess.CompletedProcess([], 0, json.dumps(next(socket_replies)), "")

        with tempfile.TemporaryDirectory() as directory, \
                patch("sys.argv", ["test_media_surfaces", "--url", "http://unused", "--output", directory]), \
                patch.object(surfaces.urllib.request, "urlopen", side_effect=urlopen), \
                patch.object(surfaces.subprocess, "run", side_effect=run), \
                patch("sys.stdout", new_callable=io.StringIO):
            surfaces.main()
            return [json.loads(line) for line in (Path(directory) / "surfaces.jsonl").read_text().splitlines()]

    def test_valid_protocol_contract_passes_all_cases(self):
        results = self.run_suite()
        self.assertEqual(len(results), 22)
        self.assertTrue(all(result["ok"] for result in results))

    def test_websocket_rejects_wrong_status_code_or_message(self):
        for index in (0, 2, 3):
            for field, value in (("status", 500), ("code", "internal_error"), ("message", "unrelated failure")):
                def mutate(http, ws, index=index, field=field, value=value):
                    event = ws[index][0] if index == 0 else ws[index]
                    target = event if field == "status" else event["error"]
                    target[field] = value
                with self.subTest(index=index, field=field), self.assertRaises(AssertionError):
                    self.run_suite(mutate)

    def test_http_continuation_requires_structured_error(self):
        for index in (16, 17):
            for field, value in (("type", "invalid_request_error"), ("code", 500), ("message", "unrelated failure")):
                def mutate(http, ws, index=index, field=field, value=value):
                    http[index]["error"][field] = value
                with self.subTest(index=index, field=field), self.assertRaises(AssertionError):
                    self.run_suite(mutate)

    def test_missing_stored_response_id_cannot_skip_continuations(self):
        for value in (None, "", 42):
            def mutate(http, ws, value=value):
                http[14]["id"] = value
            with self.subTest(value=value), self.assertRaises(AssertionError):
                self.run_suite(mutate)


if __name__ == "__main__":
    unittest.main()
