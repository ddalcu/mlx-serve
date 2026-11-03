#!/usr/bin/env python3
"""Nemotron 3.5 Bolt/Super live checks: RADIO image and 128/256-frame video, literal HTML, LiLiCorr drafting,
prefix restore, on/off decode timing (stdlib only).

  python3 tests/test_nemotron_bolt.py --model <native-config MLX pack> --output <dir>
      --checks vision|defaults|drafter|restore|bench [--drafter <dir>] [--reps 3]

Starts and stops its own loopback server (scratch HOME; prefix cache, PLD and MTP off for timing; `restore` turns on
only the RAM prefix cache). The pack's drafter/ is found on its own; drafter checks assert default engagement and
acceptance plus explicit opt-out on Chat, raw Completions, Messages and Responses, streamed and not (`defaults` runs
only that matrix). Super's vision check needs no drafter. A failure names the surface and the check.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

from lib.server import wait_health


REPO = Path(__file__).resolve().parents[1]


def image_uri(path):
    mime = 'image/png' if path.suffix == '.png' else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


def color_uri(rgb):
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))
    pixels = (b'\0' + bytes(rgb) * 32) * 32
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 32, 32, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


class Server:
    def __init__(self, args):
        self.args = args
        self.log = args.output / 'server.log'
        self.console = (args.output / 'console.log').open('w')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.base = f'http://127.0.0.1:{self.port}'
        drafter_flags = ([] if args.drafter_auto else ['--drafter', str(args.drafter)]) if args.drafter else ['--no-drafter']
        self.command = [str(args.binary), '--model', str(args.model), *drafter_flags,
                        '--serve', '--host', '127.0.0.1', '--port', str(self.port),
                        '--ctx-size', '32768', '--max-tokens', '256', '--reasoning-budget', '0',
                        '--no-mtp', '--no-pld', '--prefix-cache-entries', '8' if args.checks == 'restore' else '0', '--prefix-cache-disk', 'off',
                        '--log-file', str(self.log), '--log-level', 'info']
        test_home = args.output / 'home'
        test_home.mkdir(exist_ok=True)
        self.process = subprocess.Popen(self.command, cwd=REPO, stdout=self.console, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'HOME': str(test_home)})
        try:
            try:
                wait_health(self.base, 240, self.process)
            except RuntimeError as error:
                raise RuntimeError(f'{error}; see {self.log}') from error
            loaded = [m for m in self.request('/v1/models')['data'] if m.get('loaded')]
            assert len(loaded) == 1, loaded
            self.model = loaded[0]
            if args.drafter:
                config = json.loads((args.drafter / 'config.json').read_text())
                if config.get('has_own_lm_head'):
                    assert '[dflash] trained own lm_head loaded:' in self.log.read_text(), 'trained draft output head was not loaded'
        except BaseException:
            self.close()
            raise

    def request(self, route, body=None, timeout=600):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + route, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if body is not None and body.get('stream'):
                    events, done = [], False
                    for line in response:
                        if not line.startswith(b'data:'):
                            continue
                        payload = line[5:].strip()
                        if payload == b'[DONE]':
                            done = True
                        elif payload:
                            events.append(json.loads(payload))
                    return {'events': events, 'done': done}
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise AssertionError(f'HTTP {error.code}: {error.read().decode()}') from error

    def chat(self, content, drafter=False, tokens=192):
        start = time.monotonic()
        body = {
            'model': self.model['id'], 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'top_k': 1, 'seed': 42, 'max_tokens': tokens, 'stream': False,
            'enable_pld': False, 'enable_mtp': False,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        if drafter is not None:
            body['enable_drafter'] = drafter
        reply = self.request('/v1/chat/completions', body)
        return {'wall_seconds': time.monotonic() - start, 'response': reply}

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.console.close()


def text_of(result):
    return result['response']['choices'][0]['message'].get('content', '')


def spec_stats(log):
    rows = re.findall(r'\[spec-stats\] mode=dflash ([^\n]+)', log)
    return [{k: float(v) for k, v in re.findall(r'\b(attempts|accepts|avg_per_round)=([0-9.]+)', row)} for row in rows]


def vision_checks(server, drafter=False):
    assert {'image', 'video'} <= set(server.model['input_modalities']), server.model
    results = {'text': server.chat('What is 17 + 25? Reply with just the integer.', tokens=16, drafter=drafter)}
    assert re.search(r'\b42\b', text_of(results['text'])), results['text']
    results['literal_html'] = server.chat('Explain how an HTML <video> element works in one sentence.', drafter=drafter)
    assert text_of(results['literal_html']).strip()
    image_offset = server.log.stat().st_size
    results['image'] = server.chat([
        {'type': 'text', 'text': 'Name the main subject of this image in one sentence.'},
        {'type': 'image_url', 'image_url': {'url': image_uri(REPO / 'tests/fixtures/robot.png')}},
    ], drafter=drafter)
    assert 'robot' in text_of(results['image']).lower(), results['image']
    if drafter:
        stats = spec_stats(server.log.read_bytes()[image_offset:].decode('utf-8'))
        assert sum(s['attempts'] for s in stats) > 0, 'image drafting did not engage'
        results['image']['spec_stats'] = stats
    red, blue = color_uri((255, 0, 0)), color_uri((0, 0, 255))
    for count in (128, 256):
        offset = server.log.stat().st_size
        prompt = 'The HTML <video> tag is mentioned here as text. Name the first and final colors in the attached clip, in order.'
        prompt += ' Then explain the color change across the clip in three sentences.' if drafter else ' Answer with just the two colors.'
        result = server.chat([
            {'type': 'text', 'text': prompt},
            {'type': 'video_url', 'video_url': {'frames': [red] * (count - 32) + [blue] * 32}},
        ], drafter=drafter)
        logs = server.log.read_bytes()[offset:].decode('utf-8')
        assert f'Decoded {count} frames → RADIO video' in logs, logs
        assert f'RADIO video EVS: {count} frames' in logs, logs
        answer = text_of(result).lower()
        assert 'red' in answer and 'blue' in answer and answer.index('red') < answer.index('blue'), result
        assert result['response']['usage']['prompt_tokens'] > 1000
        if drafter:
            stats = spec_stats(logs)
            assert sum(s['attempts'] for s in stats) > 0, f'{count}-frame video drafting did not engage'
            result['spec_stats'] = stats
        result['video_log'] = [line for line in logs.splitlines() if 'RADIO' in line or 'Multimodal:' in line]
        results[f'video_{count}'] = result
        print(f'PASS: {count} supplied frames reach temporal video; late color recognized', flush=True)
    print('PASS: image pixels and literal HTML video text', flush=True)
    return results


PROMPTS = {
    'explanation': 'Explain binary search, its loop invariant, and its time complexity. Give a worked example.',
    'code': 'Write a Python function that merges two sorted lists without sorting them again. Include type hints, a docstring, and three examples.',
    'narrative': 'Write a detailed short story about a lighthouse keeper who discovers a message hidden in the beam. Begin immediately with the story.',
}


def drafter_defaults(server):
    results = {}
    prompt = 'Explain how binary search finds an item in a sorted list.'
    surfaces = {
        'chat': ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': prompt}]}),
        'completions': ('/v1/completions', {'prompt': prompt}),
        'messages': ('/v1/messages', {'messages': [{'role': 'user', 'content': prompt}],
                                    'thinking': {'type': 'disabled'}}),
        'responses': ('/v1/responses', {'input': prompt, 'max_output_tokens': 96,
                                      'reasoning': {'effort': 'none'}}),
    }
    for name, (route, fields) in surfaces.items():
        for stream in (False, True):
            for enabled in (None, False):
                body = dict(fields, model=server.model['id'], max_tokens=96, temperature=0,
                            top_k=1, seed=42, stream=stream, enable_pld=False, enable_mtp=False,
                            enable_thinking=False, chat_template_kwargs={'enable_thinking': False})
                if enabled is not None:
                    body['enable_drafter'] = enabled
                offset = server.log.stat().st_size
                start = time.monotonic()
                reply = server.request(route, body)
                result = {'wall_seconds': time.monotonic() - start, 'response': reply}
                if stream:
                    events = reply['events']
                    assert events and all(e.get('error') is None and e.get('type') not in
                                          ('error', 'response.failed') for e in events), result
                    if name == 'responses':
                        assert any(e.get('type') in ('response.completed', 'response.incomplete') and
                                   e['response'].get('error') is None for e in events), result
                        text = ''.join(e['delta'] for e in events if e.get('type') == 'response.output_text.delta')
                    elif name == 'messages':
                        assert any(e.get('type') == 'message_stop' for e in events), result
                        text = ''.join(e['delta'].get('text', '') for e in events if e.get('type') == 'content_block_delta')
                    else:
                        assert reply['done'], result
                        assert any(c.get('finish_reason') in ('stop', 'length') for e in events
                                   for c in e.get('choices', [])), result
                        text = ''.join(c.get('text', '') if name == 'completions' else
                                       c.get('delta', {}).get('content', '') or ''
                                       for e in events for c in e.get('choices', []))
                else:
                    assert reply.get('error') is None, result
                    if name == 'responses':
                        assert reply['status'] in ('completed', 'incomplete'), result
                        text = ''.join(c['text'] for item in reply['output'] if item['type'] == 'message'
                                       for c in item['content'] if c['type'] == 'output_text')
                    elif name == 'messages':
                        text = ''.join(c['text'] for c in reply['content'] if c['type'] == 'text')
                    else:
                        choice = reply['choices'][0]
                        text = choice['text'] if name == 'completions' else choice['message'].get('content', '')
                assert text.strip(), result
                logs = server.log.read_bytes()[offset:].decode('utf-8')
                stats = spec_stats(logs)
                label = f'{name} stream={stream} enable_drafter={enabled}'
                if enabled is None:
                    assert sum(s['attempts'] for s in stats) > 0, f'{label}: default drafting did not engage'
                    assert sum(s['accepts'] for s in stats) > 0, f'{label}: default drafting accepted no tokens'
                else:
                    assert not stats, f'{label}: explicit opt-out drafted: {stats}'
                result['spec_stats'] = stats
                key = name + ('_stream' if stream else '') + ('_off' if enabled is False else '')
                results[key] = result
                print(f'PASS: {label}', flush=True)
    return results


def drafter_measurements(server, reps):
    for enabled in (False, True):
        server.chat('List five practical uses of a dictionary in Python.', drafter=enabled, tokens=64)
    rows = []
    for name, prompt in PROMPTS.items():
        for repetition in range(reps):
            pair = {}
            for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
                offset = server.log.stat().st_size
                result = server.chat(prompt, drafter=enabled)
                logs = server.log.read_bytes()[offset:].decode('utf-8')
                stats = spec_stats(logs)
                assert bool(stats) == enabled, f'drafter engagement mismatch: {logs}'
                if enabled:
                    assert sum(s['attempts'] for s in stats) > 0, stats
                row = {'case': name, 'repetition': repetition + 1, 'drafter': enabled,
                       'decode_tok_s': result['response']['timings']['predicted_per_second'],
                       'spec_stats': stats,
                       'spec_log': [line for line in logs.splitlines() if '[spec-stats]' in line], **result}
                pair[enabled] = row
                rows.append(row)
                print(f'{name} rep={repetition + 1} drafter={enabled} decode={row["decode_tok_s"]:.2f} tok/s', flush=True)
            assert text_of(pair[False]) == text_of(pair[True]), f'greedy on/off mismatch: {pair}'
    assert sum(s['accepts'] for row in rows for s in row['spec_stats']) > 0, 'no speculative draft was accepted'
    summary = []
    for name in PROMPTS:
        off = [r for r in rows if r['case'] == name and not r['drafter']]
        on = [r for r in rows if r['case'] == name and r['drafter']]
        attempts = sum(s['attempts'] for r in on for s in r['spec_stats'])
        accepts = sum(s['accepts'] for r in on for s in r['spec_stats'])
        summary.append({'case': name, 'reps_per_arm': reps,
                        'off_median_tok_s': statistics.median(r['decode_tok_s'] for r in off),
                        'on_median_tok_s': statistics.median(r['decode_tok_s'] for r in on),
                        'drafts_accepted_per_round': accepts / attempts,
                        'attempts': attempts, 'accepts': accepts})
    # Protocol checks follow the measurements so added cases cannot warm the timed arms.
    return {'defaults': drafter_defaults(server), 'rows': rows, 'summary': summary}


def restore_check(server):
    prompt = ('Reference notes:\n' + 'Binary search halves a sorted interval and preserves the target if present.\n' * 80
              + 'Explain the loop invariant concisely, using these notes.')
    first = server.chat(prompt, drafter=True, tokens=96)
    second = server.chat(prompt, drafter=True, tokens=96)
    assert text_of(first).strip() and text_of(second).strip()
    cached = second['response']['usage']['prompt_tokens_details']['cached_tokens']
    assert cached > 0, f'prefix restore did not engage: {second}'
    print(f'PASS: repeated LiLiCorr request restored {cached} prompt tokens and completed', flush=True)
    return {'first': first, 'restored': second}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='MLX pack with native vision_config.model_type=radio')
    parser.add_argument('--drafter', type=Path)
    parser.add_argument('--binary', type=Path, default=REPO / 'zig-out/bin/mlx-serve')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checks', choices=('vision', 'defaults', 'drafter', 'bench', 'restore', 'all'), default='all')
    parser.add_argument('--reps', type=int, default=3)
    args = parser.parse_args()
    args.model, args.binary, args.output = args.model.resolve(), args.binary.resolve(), args.output.resolve()
    args.drafter_auto = args.drafter is None
    if args.drafter is not None:
        args.drafter = args.drafter.resolve()
    elif (args.model / 'drafter' / 'config.json').is_file():
        args.drafter = (args.model / 'drafter').resolve()
    if args.checks != 'vision' and args.drafter is None:
        parser.error('drafter checks require --drafter or a model/drafter sidecar')
    assert args.reps > 0
    args.output.mkdir(parents=True, exist_ok=True)
    results = {'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'model': str(args.model),
               'drafter': str(args.drafter) if args.drafter else None, 'checks': args.checks, 'repetitions': args.reps}
    server = Server(args)
    try:
        results['command'] = server.command
        results['models'] = server.model
        results['props'] = server.request('/props')
        if args.checks in ('vision', 'all'):
            results['vision'] = vision_checks(server, drafter=args.checks == 'all' and args.drafter is not None)
        if args.checks in ('drafter', 'bench', 'all'):
            results['drafter'] = drafter_measurements(server, 1 if args.checks == 'drafter' else args.reps)
        if args.checks == 'defaults':
            results['defaults'] = drafter_defaults(server)
        if args.checks == 'restore':
            results['restore'] = restore_check(server)
        results['passed'] = True
    except BaseException as error:
        results['error'] = str(error)
        raise
    finally:
        server.close()
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
