#!/usr/bin/env python3
"""Compare legacy and group MTP on one running server with the group planner enabled.

Optional hooks receive phase (warmup/timed), context, arm, and row count.
The caller owns model loading, GPU exclusivity, cooling, and clock sampling.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


PREFIX_CHARS = {}

def hook(path, phase, context, arm, count):
    if path:
        subprocess.run([str(path.resolve()), phase, context, arm, str(count)], check=True)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port', type=int)
    parser.add_argument('--before-cell', type=Path)
    parser.add_argument('--after-cell', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--corpus', choices=('functions', 'code'), default='functions')
    parser.add_argument('--n', default='2,4')
    parser.add_argument('--ctx', default='4k')
    parser.add_argument('--arms', default='off,on')
    parser.add_argument('--reqs', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=300)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--requests-jsonl', type=Path)
    parser.add_argument('--label', default='')
    parser.add_argument('--nonce', type=int, default=0)
    parser.add_argument('--calibration', type=Path, help='Saved token-calibrated prefix lengths for this source')
    return parser.parse_args()


def context_chars(context):
    if context in PREFIX_CHARS:
        return PREFIX_CHARS[context]
    value = float(context.lower().rstrip('k'))
    return int(value * (1000 if context.lower().endswith('k') else 1) * 3)


def prompt(source, corpus, context, client, request):
    prefix = f'Read this file.\n{source[:context_chars(context)]}\n'
    if corpus == 'functions':
        return prefix + f'User {client} question {request}: name three functions above and what each does.\nAssistant:'
    return prefix + (
        'write a complete Python implementation of a streaming JSONL reader that accepts '
        'an iterable of text chunks, handles records split across chunks, reports malformed '
        'records with line numbers, and includes unit tests. Return only Python code.\n'
        f'User {client} question {request}:\nAssistant:'
    )


def request(base, source, args, context, client, index, mtp, max_tokens):
    body = {
        'messages': [{'role': 'user', 'content': prompt(source, args.corpus, context, client, index + 2 * args.nonce)}],
        'max_tokens': max_tokens, 'temperature': 0, 'enable_thinking': False,
        'stream': True, 'stream_options': {'include_usage': True},
    }
    body['enable_mtp'] = True
    body['enable_batch_mtp'] = mtp
    body['enable_pld'] = False
    body['enable_drafter'] = False
    start = time.perf_counter()
    arrivals = []
    timings = {}
    usage = {}
    ended = False
    finish_reason = None
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                ended = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(event['error'])
            choices = event.get('choices') or []
            if choices:
                if choices[0].get('delta', {}).get('content'):
                    arrivals.append(time.perf_counter())
                finish_reason = choices[0].get('finish_reason') or finish_reason
            timings = event.get('timings') or timings
            if event.get('usage') and not choices:
                usage = event['usage']
    end = time.perf_counter()
    if not ended or not arrivals or finish_reason not in ('stop', 'length'):
        raise RuntimeError(f'Incomplete stream: done={ended}, chunks={len(arrivals)}, finish={finish_reason}')
    tokens = timings.get('predicted_n') or usage.get('completion_tokens')
    if not tokens:
        raise RuntimeError('Server did not report a completion-token count')
    return dict(client=client, request=index, start=start, first=arrivals[0], end=end, tokens=tokens,
                server_tps=timings.get('predicted_per_second'), prompt_tokens=usage.get('prompt_tokens'),
                cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                finish_reason=finish_reason,
                gaps_ms=[1000 * (b - a) for a, b in zip(arrivals, arrivals[1:])])


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))] if values else 0


def main():
    args = options()
    if args.reqs < 1 or args.max_tokens < 1:
        raise ValueError('Request count and token budget must be positive')
    counts = [int(value) for value in args.n.split(',')]
    if any(value < 1 or value > 8 for value in counts):
        raise ValueError('Row counts must be between 1 and 8')
    arms = args.arms.split(',')
    if any(arm not in ('off', 'on') for arm in arms):
        raise ValueError('Arms must be off or on')
    source_bytes = args.source.read_bytes()
    source = source_bytes.decode() * 8
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if args.calibration:
        saved = json.loads(args.calibration.read_text())
        if saved['source_sha256'] != source_hash:
            raise ValueError('Calibration source changed')
        PREFIX_CHARS.update(saved['prefix_chars'])
    print(json.dumps(dict(label=args.label, corpus=args.corpus, source=str(args.source), sha256=source_hash)), flush=True)
    base = f'http://127.0.0.1:{args.port}'
    for context in args.ctx.split(','):
        for arm in arms:
            hook(args.before_cell, 'warmup', context, arm, 1)
            try:
                warm = request(base, source, args, context, 99, 0, arm == 'on', 8)
            finally:
                hook(args.after_cell, 'warmup', context, arm, 1)
            for count in counts:
                hook(args.before_cell, 'timed', context, arm, count)

                def client(client_id):
                    return [request(base, source, args, context, client_id, i, arm == 'on', args.max_tokens)
                            for i in range(args.reqs)]

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                        results = [result for group in pool.map(client, range(count)) for result in group]
                finally:
                    hook(args.after_cell, 'timed', context, arm, count)
                if len(results) != count * args.reqs:
                    raise RuntimeError('Incomplete cell')
                tokens = sum(result['tokens'] for result in results)
                window = max(result['end'] for result in results) - min(result['first'] for result in results)
                wall = max(result['end'] for result in results) - min(result['start'] for result in results)
                ttfts = [result['first'] - result['start'] for result in results]
                gaps = [gap for result in results for gap in result['gaps_ms']]
                server = [result['server_tps'] for result in results if result['server_tps']]
                row = dict(label=args.label, corpus=args.corpus, ctx=context, arm=arm, n=count,
                           prompt_tokens=warm['prompt_tokens'], reqs=len(results), tokens=tokens,
                           agg_decode_tps=round(tokens / window, 1), wall_tps=round(tokens / wall, 1),
                           per_stream_tps=round(statistics.mean(result['tokens'] / (result['end'] - result['first']) for result in results), 1),
                           server_tps=round(statistics.mean(server), 1) if server else None,
                           ttft_p50=round(percentile(ttfts, .5), 2), ttft_p95=round(percentile(ttfts, .95), 2),
                           output_gap_p50_ms=round(percentile(gaps, .5), 2),
                           output_gap_p95_ms=round(percentile(gaps, .95), 2),
                           output_gap_max_ms=round(max(gaps, default=0), 2),
                           window=round(window, 3), wall=round(wall, 3), source_sha256=source_hash)
                print(json.dumps(row), flush=True)
                if args.requests_jsonl:
                    with args.requests_jsonl.open('a') as file:
                        for result in results:
                            file.write(json.dumps(dict(label=args.label, corpus=args.corpus, ctx=context, arm=arm, n=count, **result)) + '\n')
                if args.csv:
                    write_header = not args.csv.exists() or args.csv.stat().st_size == 0
                    with args.csv.open('a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=list(row))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)


if __name__ == '__main__':
    main()
