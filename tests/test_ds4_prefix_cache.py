#!/usr/bin/env python3
"""Live ds4 SSD checkpoints: run seed, restart the server, then run resume.

The seed phase also replaces the conversation and verifies a disk restore.
BENCH_API_KEY authenticates requests; the output file contains no credentials.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def request(base, model, prompt):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'enable_thinking': False, 'max_tokens': 32}
    req = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ['BENCH_API_KEY']})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
        result['http_status'] = response.status
    result['elapsed_seconds'] = time.monotonic() - start
    result['request'] = body
    return result


def cached(result):
    return result.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', required=True)
    p.add_argument('--phase', choices=('seed', 'resume'), required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    lines = [f'Record {i}: the archive stores a neutral observation about the river, weather and forest.' for i in range(240)]
    lines.insert(100, 'The verification passphrase is cedar otter 7391.')
    prompt = f'Archive {args.tag}.\n' + '\n'.join(lines) + '\nReturn only the verification passphrase recorded in this archive.'
    results = []
    def run(label, text):
        result = request(args.base_url, args.model, text)
        result['case'] = label
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        content = result['choices'][0]['message'].get('content', '') or ''
        print(label, f"{result['elapsed_seconds']:.3f}s", 'cached=', cached(result), 'prompt=', result.get('usage', {}).get('prompt_tokens'), repr(content), flush=True)
        return result, content
    if args.phase == 'seed':
        result, text = run('cold', prompt)
        assert 'cedar otter 7391' in text.lower(), text
        run('replace', f'Unrelated conversation {args.tag}. Return only BLACKBIRD.')
        result, text = run('disk-after-replacement', prompt)
    else:
        result, text = run('disk-after-restart', prompt)
    assert 'cedar otter 7391' in text.lower(), text
    assert cached(result) >= 512, f'No usable prefix restored: {result.get("usage")}'
    print('PASS', flush=True)


if __name__ == '__main__':
    main()
