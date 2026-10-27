#!/usr/bin/env python3
"""Verify durable emitted prefixes when overlapping MTP requests finish B before A.

Use a dedicated server with a fresh MLX_SERVE_LOOKUP_STORE, scratch HOME, native
MTP, --no-drafter --no-pld --max-concurrent 2 --ctx-size 4096. Tested with
Qwen3.8-27B. The test neither starts nor stops the caller's server.

  python3 tests/test_token_history.py PORT --store STORE \
      --tokenizer MODEL/tokenizer.json --server-log LOG [--out DIR]

Requires actual overlapping length-limited streams and history acquisition;
unsupported scheduling/model behavior fails explicitly, never silently passes.
This checks publication, not serial numerical equivalence or inference speed.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import threading
import time
import urllib.request

from lib.server import wait_health

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('port', type=int)
parser.add_argument('--store', type=Path, required=True)
parser.add_argument('--tokenizer', type=Path, required=True)
parser.add_argument('--server-log', type=Path, required=True)
parser.add_argument('--out', type=Path)
args = parser.parse_args()
base = f'http://127.0.0.1:{args.port}'
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
if args.out is None:
    scratch = Path.home() / 'claude-tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    args.out = Path(tempfile.mkdtemp(prefix='token-history-', dir=scratch))
else:
    args.out.mkdir(parents=True, exist_ok=False)


def save(name, value):
    (args.out / name).write_text(json.dumps(value, indent=2) + '\n')


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
    with http.open(req, timeout=120) as response:
        return json.load(response)


prompts = {
    'A': 'Continue this Python module, defining similar vector functions.\n\ndef add(a, b):\n    return [x + y for x, y in zip(a, b)]\n\ndef subtract(a, b):\n    return [x - y for x, y in zip(a, b)]\n\ndef multiply(a, b):\n',
    'B': 'List the integers from 1 through 80, separated by commas.\n1, 2, 3,',
}
started = threading.Event()


def stream(label, limit, signal=False):
    body = dict(model=model_id, prompt=prompts[label], max_tokens=limit,
                temperature=0, top_p=1.0, top_k=0, min_p=0.0, repeat_penalty=1.0,
                stream=True, stream_options={'include_usage': True},
                enable_mtp=True, enable_drafter=False, enable_pld=False)
    req = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    result = dict(label=label, started=time.monotonic(), events=[], text='', done=False)
    chunks = 0
    try:
        with http.open(req, timeout=120) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    result['done'] = True
                    break
                event = json.loads(payload)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                result['events'].append(event)
                for choice in event.get('choices', []):
                    text = choice.get('text') or ''
                    if text:
                        result['text'] += text
                        chunks += 1
                        if signal and chunks >= 8:
                            started.set()
                    if choice.get('finish_reason') is not None:
                        result['finish_reason'] = choice['finish_reason']
                if event.get('usage'):
                    result['usage'] = event['usage']
        assert result['done'] and result.get('finish_reason') == 'length', result
        assert result['usage']['completion_tokens'] == limit, result
        return result
    finally:
        result['finished'] = time.monotonic()
        save(f'{label}-{limit}.json', result)


wait_health(base, 30)
log_start = args.server_log.stat().st_size
logfile = args.store / 'tokens.bin'
assert not logfile.exists(), 'Use a fresh store; this test never clears existing history'
model_id = api('/v1/models')['data'][0]['id']
prompt_ids = {key: api('/tokenize', {'content': value})['tokens'] for key, value in prompts.items()}
seed = stream('A', 96)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    fa = pool.submit(stream, 'A', 192, True)
    assert started.wait(120), 'A did not emit enough output to start the overlap'
    fb = pool.submit(stream, 'B', 24)
    results = [future.result() for future in concurrent.futures.as_completed((fa, fb))]
assert results[0]['label'] == 'B', 'Reverse finish order was not exercised'
overlap = min(row['finished'] for row in results) - max(row['started'] for row in results)
assert overlap > 0, 'Requests did not overlap'

raw = logfile.read_bytes()
assert raw[:8] == b'MLXLOOK1' and len(raw) >= 40 and (len(raw) - 40) % 4 == 0
assert raw[8:40] == hashlib.sha256(args.tokenizer.read_bytes()).digest()
words = struct.unpack('<' + 'I' * ((len(raw) - 40) // 4), raw[40:])
documents, document = [], []
for word in words:
    if word == 1 << 30:
        documents.append(document)
        document = []
    else:
        document.append(word)
assert not document and len(documents) == 3, 'Incomplete, lost or duplicated history document'
checked = []
for document, (label, count), response in zip(documents, [('A', 96), ('B', 24), ('A', 192)], [seed, *results]):
    prefix = prompt_ids[label]
    assert document[:len(prefix)] == prefix, f'{label}: wrong prompt or publication order'
    emitted = document[len(prefix):]
    assert len(emitted) == count, f'{label}: persisted an unseen tail or lost emitted tokens'
    assert api('/detokenize', {'tokens': emitted})['content'] == response['text'], f'{label}: output differs from stream'
    checked.append(dict(label=label, prompt_tokens=len(prefix), output_tokens=count))
log = args.server_log.read_bytes()[log_start:].decode(errors='replace')
assert re.search(r'\[spec-stats\] mode=mtp attempts=[1-9]', log), 'Native MTP did not execute'
assert len(re.findall(r'\[history\] external_rounds=', log)) == 3, 'Requests did not acquire history snapshots'
assert not any(message in log for message in ('StoreBusy', '[history] publication failed:', '[history] publication refused:'))
summary = dict(documents=checked, finish_order=[row['label'] for row in results],
               overlap_seconds=overlap, log_bytes=len(raw), text_matches_streams=True)
save('summary.json', summary)
print('PASS: overlapping emitted prefixes persist in finish order; artifacts=' + str(args.out))
