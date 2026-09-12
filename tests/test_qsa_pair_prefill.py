#!/usr/bin/env python3
"""Run against a disposable server, MLX_SERVE_QSA_PAIR=1, cache off; check two shapes."""
import argparse
import json
import re
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:18765')
p.add_argument('--model', default='ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit')
p.add_argument('--log', type=Path, required=True)
p.add_argument('--pair', choices=('on', 'off'), required=True)
p.add_argument('--hc', choices=('on', 'off'))
p.add_argument('--gdn', choices=('on', 'off'))
p.add_argument('--first-chunk', action='store_true')
args = p.parse_args()
log_offset = args.log.stat().st_size if args.log.exists() else 0
source = (Path(__file__).resolve().parents[1] / 'src/transformer.zig').read_text()
corpus = '\n'.join(source.splitlines()[100:900])
for nonce, size in [('81492017', 40000), ('52839106', 47000)]:
    prompt = (f'Nonce {nonce}. Remember the passphrase MAGNOLIA-7731.\n'
              + (corpus * 4)[:size]
              + '\nWhat was the passphrase? Answer with the passphrase only.')
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'seed': 1234, 'max_tokens': 32,
            'enable_mtp': False, 'enable_pld': False, 'stream': False}
    req = urllib.request.Request(args.url + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    usage = result['usage']
    assert usage['prompt_tokens'] > 8192, usage
    assert usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0, usage
    answer = result['choices'][0]['message']['content']
    assert 'MAGNOLIA-7731' in answer, result
    print(json.dumps({'arm': args.pair, 'usage': usage, 'answer': answer}), flush=True)
log = args.log.read_bytes()[log_offset:].decode('utf-8', errors='replace')
expected = ('[qsa-pair] engaged:' if args.pair == 'on'
            else '[qsa-gather] engaged: msv_qsa_nax_precise')
assert expected in log, f'Missing engagement: {expected}'
if args.first_chunk:
    assert args.pair == 'on'
    assert re.search(r'\[qsa-pair\] engaged: S=8192 kv=8192 ', log), 'First chunk used the old mask arm'
if args.pair == 'off':
    assert '[qsa-pair] engaged:' not in log
if args.hc == 'on':
    assert '[hc-prefill] engaged:' in log, 'Missing HC prefill engagement'
if args.hc == 'off':
    assert '[hc-prefill] engaged:' not in log
if args.gdn == 'on':
    assert re.search(r'\[gdn-prefill\] engaged: S=8192 B=1 cold=true', log), 'Missing cold GDN prefill engagement'
if args.gdn == 'off':
    assert '[gdn-prefill] engaged:' not in log
print('PASS: long prefill answers and expected QSA arm')
