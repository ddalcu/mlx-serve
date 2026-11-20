#!/usr/bin/env python3
"""Verify two discovered ds4 GGUFs share one native model owner.

Use an isolated server and BENCH_API_KEY. The two model files may be hard links.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--first-model', required=True)
    parser.add_argument('--second-model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    headers = {'Authorization': 'Bearer ' + os.environ.get('BENCH_API_KEY', ''),
               'Content-Type': 'application/json'}

    def request(path, payload=None):
        return urllib.request.Request(args.base_url.rstrip('/') + path,
                data=None if payload is None else json.dumps(payload).encode(), headers=headers)

    def call(path, payload=None):
        with urllib.request.urlopen(request(path, payload), timeout=300) as response:
            return json.load(response)

    for model, other in [(args.first_model, args.second_model),
                         (args.second_model, args.first_model)]:
        call('/v1/load-model', {'model': model})
        rows = {row['id']: row for row in call('/v1/models')['data']}
        assert rows[model]['loaded'], rows[model]
        assert rows[model]['meta']['engine'] == 'ds4', rows[model]
        assert not rows[other]['loaded'], rows[other]
        owners = [row['id'] for row in rows.values()
                  if row.get('loaded') and row.get('meta', {}).get('engine') == 'ds4']
        assert owners == [model], owners
        records.append({'loaded': model, 'evicted': other})
        print('PASS sole ds4 owner:', model, flush=True)

    body = {'model': args.second_model, 'stream': True, 'ignore_eos': True,
            'max_tokens': 1024, 'temperature': 0, 'enable_thinking': False,
            'messages': [{'role': 'user', 'content': 'Repeat the word amber many times.'}]}
    with urllib.request.urlopen(request('/v1/chat/completions', body), timeout=300) as response:
        for line in response:
            if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                continue
            event = json.loads(line[5:])
            if any(choice.get('delta', {}).get('content') for choice in event.get('choices', [])):
                break
        else:
            raise AssertionError('No active decode observed')
        try:
            call('/v1/load-model', {'model': args.first_model})
            raise AssertionError('A second native model loaded during decode')
        except urllib.error.HTTPError as error:
            result = json.load(error)
            assert error.code == 503, result
            assert result.get('error', {}).get('type') == 'model_busy', result
            records.append({'busy_status': error.code, 'response': result})
    assert call('/health')['status'] == 'ok'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    print('PASS busy owner returns 503 model_busy; server remains healthy', flush=True)


if __name__ == '__main__':
    main()
