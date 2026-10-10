#!/usr/bin/env python3
"""pi_rpc's event handling: python3 tests/agent_eval/test_pi_rpc.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import pi_rpc  # noqa: E402


def drive(events, threshold=100):
    st = pi_rpc.State(threshold)
    sent = []
    for ev in events:
        sent += pi_rpc.on_event(st, ev)
    return st, [c["type"] if c["type"] != "prompt" else c["message"] for c in sent]


OVER = {"type": "turn_end", "toolResults": [{}], "message": {"usage": {"totalTokens": 150}}}


class OnEvent(unittest.TestCase):
    def test_our_compaction_continues_the_task(self):
        _, sent = drive([OVER, {"type": "compaction_start", "reason": "manual"},
                         {"type": "compaction_end", "reason": "manual", "aborted": False}])
        self.assertEqual(sent, ["compact", pi_rpc.CONTINUE])

    def test_ours_aborting_pis_own_compaction_continues_once_after_ours(self):
        # The run that ended early: pi's threshold compaction starts, ours aborts it.
        st, sent = drive([OVER, {"type": "compaction_start", "reason": "threshold"},
                          {"type": "compaction_end", "reason": "threshold", "aborted": True},
                          {"type": "compaction_start", "reason": "manual"},
                          {"type": "compaction_end", "reason": "manual", "aborted": False}])
        self.assertEqual(sent, ["compact", pi_rpc.CONTINUE])
        self.assertIsNone(st.idle_since)

    def test_pis_own_compaction_continues_the_task(self):
        _, sent = drive([{"type": "compaction_start", "reason": "threshold"},
                         {"type": "compaction_end", "reason": "threshold", "aborted": False}])
        self.assertEqual(sent, [pi_rpc.CONTINUE])

    def test_no_compact_while_one_runs_and_no_continue_when_pi_retries(self):
        _, sent = drive([{"type": "compaction_start", "reason": "overflow"}, OVER,
                         {"type": "compaction_end", "reason": "overflow", "aborted": False, "willRetry": True}])
        self.assertEqual(sent, [])

    def test_a_failed_compaction_ends_the_run(self):
        st, sent = drive([OVER, {"type": "compaction_start", "reason": "manual"},
                          {"type": "compaction_end", "reason": "manual", "aborted": False,
                           "errorMessage": "Compaction failed: boom"}])
        self.assertEqual(sent, ["compact"])
        self.assertIsNotNone(st.idle_since)


if __name__ == "__main__":
    unittest.main()
