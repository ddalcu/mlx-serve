#!/usr/bin/env python3
"""Live EOS/cancellation parity against a running server.

Start it with native MTP, MLX_SERVE_MTP_GROUP_PLANNER=1, and
MLX_SERVE_MTP_FORCE_DEPTH=4 so both arms use the same verify width.
The caller owns model loading and GPU/thermal controls.
"""
import argparse
import concurrent.futures
import http.client
import json
import re
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('port', type=int)
parser.add_argument('--server-log', type=Path, required=True)
parser.add_argument('--out', type=Path)
args = parser.parse_args()
port = args.port
out = args.out or Path(tempfile.mkdtemp(prefix='mtp-batch-stops-'))
out.mkdir(parents=True, exist_ok=True)
logfile = args.server_log
base = f'http://127.0.0.1:{port}'
shorts = [
    'Count from one to twenty in English, separated by commas, and stop after twenty.',
    'List the twelve months from January through December, one per line, with no introduction.',
    'List the seven days of the week in order, then stop. No introduction or explanation.',
    'Write the numbers 1 through 30 separated by spaces. Do not add any other text.',
    'Recite the first two sentences of A Tale of Two Cities, then stop.',
]
survivor = 'Write a Python implementation of a linked list with append, prepend, remove, find, iteration, and unit tests. Return only Python code.'
victim = 'Write a Python implementation of an AVL tree with insert, delete, rotations, lookup, iteration, and extensive unit tests. Return only Python code.'
def request(prompt, limit=160, barrier=None, cancel_after=None, grouped=False):
    if barrier is not None:
        barrier.wait()
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    body = dict(model='default', messages=[dict(role='user', content=prompt)], temperature=0,
                max_tokens=limit, enable_mtp=True, enable_batch_mtp=grouped, enable_thinking=False, stream=True,
                stream_options=dict(include_usage=True))
    conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f'HTTP {response.status}: {response.read().decode()}')
    events = []
    finish = None
    usage = None
    try:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            obj = json.loads(data)
            if 'error' in obj:
                raise RuntimeError(str(obj['error']))
            for choice in obj.get('choices', []):
                delta = choice.get('delta', {})
                for key in ['reasoning_content', 'content']:
                    if delta.get(key):
                        events.append([key, delta[key]])
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
            if obj.get('usage'):
                usage = obj['usage']
            if cancel_after and len(events) >= cancel_after:
                sock = conn.sock or response.fp.raw._sock
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                response.close()
                conn.close()
                return dict(events=events, cancelled=True)
    finally:
        response.close()
        conn.close()
    if finish is None:
        raise RuntimeError('Stream ended without a finish reason')
    return dict(events=events, finish=finish, usage=usage)

def equal(label, expected, actual):
    if expected['events'] != actual['events']:
        (out / (label + '.expected.json')).write_text(json.dumps(expected, indent=2))
        (out / (label + '.actual.json')).write_text(json.dumps(actual, indent=2))
        raise AssertionError(label + ': published event sequence differs from solo')
    if expected['finish'] != actual['finish']:
        raise AssertionError(label + ': finish reason differs from solo')
    if expected['usage']['completion_tokens'] != actual['usage']['completion_tokens']:
        raise AssertionError(label + ': completion count differs from solo')

references = [request(prompt) for prompt in shorts]
long_reference = request(survivor, 200)
victim_reference = request(victim, 512)
(out / 'references.json').write_text(json.dumps(dict(short=references, survivor=long_reference, victim=victim_reference), indent=2))
reached_eos = False
for i, prompt in enumerate(shorts):
    start = logfile.stat().st_size
    barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        short_run = pool.submit(request, prompt, 160, barrier, grouped=True)
        long_run = pool.submit(request, survivor, 200, barrier, grouped=True)
        short_result, long_result = short_run.result(), long_run.result()
    equal(f'eos-{i}-short', references[i], short_result)
    equal(f'eos-{i}-survivor', long_reference, long_result)
    log = logfile.read_bytes()[start:].decode(errors='replace')
    if '[mtp-planner-stats]' not in log:
        raise AssertionError('Two native-MTP streams never entered the group planner')
    if '[mtp-stop] accepted EOS' in log:
        survivor_stats = re.findall(r'\[mtp-planner-stats\] tokens=200 plain=(\d+)', log)
        deadline = time.monotonic() + 5
        while not survivor_stats and time.monotonic() < deadline:
            time.sleep(.01)
            log = logfile.read_bytes()[start:].decode(errors='replace')
            survivor_stats = re.findall(r'\[mtp-planner-stats\] tokens=200 plain=(\d+)', log)
        if not survivor_stats or int(survivor_stats[-1]) > 2:
            raise AssertionError('The surviving planner-owned stream stopped speculating')
        reached_eos = True
        print(f'PASS: EOS inside accepted draft block; both streams equal solo (case {i})', flush=True)
        break
if not reached_eos:
    raise AssertionError('No accepted-draft EOS was observed; gate did not run')
reached_cancel = False
for attempt in range(8):
    start = logfile.stat().st_size
    barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        gone = pool.submit(request, victim, 512, barrier, 17 + attempt, grouped=True)
        live = pool.submit(request, survivor, 200, barrier, grouped=True)
        cancelled_result, survivor_result = gone.result(), live.result()
    equal(f'cancel-{attempt}-survivor', long_reference, survivor_result)
    prefix = cancelled_result['events']
    if prefix != victim_reference['events'][:len(prefix)]:
        raise AssertionError('Cancelled stream differs from its solo prefix')
    time.sleep(.2)
    log = logfile.read_bytes()[start:].decode(errors='replace')
    if '[mtp-publish] cancelled after round' in log:
        reached_cancel = True
        print(f'PASS: mid-round cancellation suppressed publication; survivor equals solo (attempt {attempt})', flush=True)
        break
if not reached_cancel:
    raise AssertionError('No mid-round cancellation publication was observed; gate did not run')
if '[batched] row-axis mtp verify engaged (slots=2' not in logfile.read_text(errors='replace'):
    raise AssertionError('Grouped verify did not engage')
print('RESULT: 2 passed, 0 failed; artifacts=' + str(out), flush=True)
