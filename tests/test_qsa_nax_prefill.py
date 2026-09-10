#!/usr/bin/env python3
"""HTTP correctness and engagement guard; point at a disposable server with cache off.

MLX_SERVE_QSA_NAX=1 MLX_SERVE_PREFILL_CHUNK=4096 zig-out/bin/mlx-serve serve \
  --port 18765 --ctx-size 131072 --prefix-cache-entries 0 > /tmp/qsa-server.log 2>&1
python3 tests/test_qsa_nax_prefill.py --log /tmp/qsa-server.log --nax on
Repeat with MLX_SERVE_QSA_NAX=0 and --nax off for the stock arm.
"""
import argparse
import json
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:18765')
p.add_argument('--model', default='ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit')
p.add_argument('--log', type=Path, required=True)
p.add_argument('--nax', choices=('on', 'off'), required=True)
args = p.parse_args()
log_offset = args.log.stat().st_size if args.log.exists() else 0
source = (Path(__file__).resolve().parents[1] / 'src/transformer.zig').read_text()
corpus = '\n'.join(source.splitlines()[100:900])
for nonce in ('81492017', '52839106', '90371648'):
    prompt = (f'Nonce {nonce}. Remember the passphrase MAGNOLIA-7731.\n'
              + (corpus * 4)[:40000]
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
    print(json.dumps({'arm': args.nax, 'usage': usage, 'answer': answer}), flush=True)
log = args.log.read_bytes()[log_offset:].decode('utf-8', errors='replace')
expected = ('[qsa-gather] engaged: msv_qsa_nax_precise' if args.nax == 'on'
            else '[qsa-gather] engaged: msv_attn_qsa256')
assert expected in log, f'Missing engagement: {expected}'
if args.nax == 'off':
    assert '[qsa-gather] engaged: msv_qsa_nax_precise' not in log
print('PASS: long prefill answers and expected QSA arm')
