#!/usr/bin/env python3
"""Live image and image-history probes for an embedded-engine deployment.

BENCH_API_KEY supplies authentication. Images are deterministic RGB fixtures;
the answers must change when pixels change while the textual prompt stays fixed.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib


def image(left, right):
    w, h = 420, 280
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            color = (255, 255, 255)
            if 25 <= x < 175 and 65 <= y < 215:
                color = left
            elif (x - 315) ** 2 + (y - 140) ** 2 < 75 ** 2:
                color = right
            rows.extend(color)
    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value))
    return base64.b64encode(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')).decode()


def part(protocol, data):
    if protocol == 'anthropic':
        return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}}
    return {'type': 'input_image' if protocol == 'responses' else 'image_url', 'image_url': ('data:image/png;base64,' + data) if protocol == 'responses' else {'url': 'data:image/png;base64,' + data}}


def body(protocol, model, data, stream):
    q = 'Name the color of the square on the left and the circle on the right. Reply only left: COLOR; right: COLOR.'
    if protocol == 'responses':
        return {'model': model, 'stream': stream, 'max_output_tokens': 384, 'input': [{'role': 'user', 'content': [part(protocol, data), {'type': 'input_text', 'text': q}]}], 'enable_thinking': False}
    result = {'model': model, 'stream': stream, 'max_tokens': 384, 'messages': [{'role': 'user', 'content': [part(protocol, data), {'type': 'text', 'text': q}]}]}
    if protocol == 'anthropic':
        result['thinking'] = {'type': 'disabled'}
    else:
        result['enable_thinking'] = False
    return result


def call(base, protocol, payload, key):
    endpoint = {'openai': '/v1/chat/completions', 'anthropic': '/v1/messages', 'responses': '/v1/responses'}[protocol]
    headers = {'Authorization': 'Bearer ' + key, 'x-api-key': key, 'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'}
    req = urllib.request.Request(base.rstrip('/') + endpoint, data=json.dumps(payload).encode(), headers=headers)
    start = time.monotonic()
    text, events, ttft = '', [], None
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            status = response.status
            if not payload.get('stream'):
                raw = json.load(response)
                events = [raw]
                if protocol == 'anthropic':
                    text = ''.join(b.get('text', '') for b in raw.get('content', []) if b.get('type') == 'text')
                elif protocol == 'openai':
                    text = raw['choices'][0]['message'].get('content', '') or ''
                else:
                    text = ''.join(c.get('text', '') for o in raw.get('output', []) for c in o.get('content', []) if c.get('type') == 'output_text')
            else:
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if not data or data == b'[DONE]':
                        continue
                    ev = json.loads(data)
                    events.append(ev)
                    if ev.get('type') == 'error' or 'error' in ev:
                        raise RuntimeError(json.dumps(ev))
                    delta = ''
                    if protocol == 'openai':
                        delta = ''.join(c.get('delta', {}).get('content', '') or '' for c in ev.get('choices', []))
                    elif protocol == 'anthropic' and ev.get('delta', {}).get('type') == 'text_delta':
                        delta = ev['delta']['text']
                    elif protocol == 'responses' and ev.get('type') == 'response.output_text.delta':
                        delta = ev['delta']
                    if delta and ttft is None:
                        ttft = time.monotonic() - start
                    text += delta
            request_id = response.headers.get('x-request-id')
    except urllib.error.HTTPError as exc:
        return {'status': exc.code, 'error': exc.read().decode(errors='replace'), 'seconds': time.monotonic() - start}
    return {'status': status, 'text': text, 'seconds': time.monotonic() - start, 'ttft': ttft, 'request_id': request_id, 'events': events}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', required=True)
    p.add_argument('--protocols', default='openai,anthropic,responses')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    key = os.environ['BENCH_API_KEY']
    fixtures = [image((230, 20, 20), (20, 40, 230)), image((20, 190, 35), (245, 215, 10))]
    results = []
    def record(label, protocol, payload, expected):
        result = call(args.base_url, protocol, payload, key)
        result['case'] = label
        result['pass'] = result['status'] == 200 and all(word in result.get('text', '').lower() for word in expected)
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(label, 'PASS' if result['pass'] else 'FAIL', result['status'], f"{result['seconds']:.2f}s", repr(result.get('text', '')[:250]), flush=True)
        return result
    for protocol in args.protocols.split(','):
        for stream in (False, True):
            for i, colors in enumerate((('red', 'blue'), ('green', 'yellow'))):
                payload = body(protocol, args.model, fixtures[i], stream)
                record(f'{protocol}-stream={stream}-pixels={i}', protocol, payload, colors)
        if protocol in ('openai', 'anthropic'):
            payload = body(protocol, args.model, fixtures[0], False)
            first = record(protocol + '-history-start', protocol, payload, ('red', 'blue'))
            if first['pass']:
                payload['messages'].extend([{'role': 'assistant', 'content': first['text']}, {'role': 'user', 'content': 'Which shape was blue in the attached picture? Reply one word.'}])
                record(protocol + '-history-followup', protocol, payload, ('circle',))
            payload = body(protocol, args.model, 'not-a-png', False)
            invalid = call(args.base_url, protocol, payload, key)
            invalid.update(case=protocol + '-invalid-image', **{'pass': invalid['status'] == 400})
            results.append(invalid)
            args.output.write_text(json.dumps(results, indent=2))
        if protocol == 'anthropic':
            payload = body(protocol, args.model, fixtures[0], True)
            payload['tools'] = [{'name': 'inspect_image', 'description': 'Read a local image.', 'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}]
            payload['messages'] = [
                {'role': 'user', 'content': 'Inspect sample.png. Name the left square color and right circle color; use the image from the tool.'},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_image_fixture', 'name': 'inspect_image', 'input': {'path': 'sample.png'}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_image_fixture', 'content': [part(protocol, fixtures[0]), {'type': 'text', 'text': 'Image loaded successfully. Reply only left: COLOR; right: COLOR. No further tool call is needed.'}]}]},
            ]
            record('anthropic-tool-result-image', protocol, payload, ('red', 'blue'))
    failures = [r['case'] for r in results if not r['pass']]
    print(f'{len(results) - len(failures)}/{len(results)} passed', flush=True)
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
