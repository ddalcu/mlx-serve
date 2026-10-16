#!/usr/bin/env python3
"""Nemotron RADIO image/video integration and LiLiCorr measurements (stdlib only)."""
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
        drafter_flags = ['--drafter', str(args.drafter)] if args.drafter else ['--no-drafter']
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
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'server exited {self.process.returncode}; see {self.log}')
                try:
                    self.request('/health', timeout=2)
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise RuntimeError('server startup timed out')
            loaded = [m for m in self.request('/v1/models')['data'] if m.get('loaded')]
            assert len(loaded) == 1, loaded
            self.model = loaded[0]
        except BaseException:
            self.close()
            raise

    def request(self, route, body=None, timeout=600):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + route, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise AssertionError(f'HTTP {error.code}: {error.read().decode()}') from error

    def chat(self, content, drafter=False, tokens=192):
        start = time.monotonic()
        reply = self.request('/v1/chat/completions', {
            'model': self.model['id'], 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'top_k': 1, 'seed': 42, 'max_tokens': tokens, 'stream': False,
            'enable_drafter': drafter, 'enable_pld': False, 'enable_mtp': False,
            'chat_template_kwargs': {'enable_thinking': False},
        })
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


def vision_checks(server):
    assert {'image', 'video'} <= set(server.model['input_modalities']), server.model
    results = {'text': server.chat('What is 17 + 25? Reply with just the integer.', tokens=16)}
    assert re.search(r'\b42\b', text_of(results['text'])), results['text']
    results['literal_html'] = server.chat('Explain how an HTML <video> element works in one sentence.')
    assert text_of(results['literal_html']).strip()
    results['image'] = server.chat([
        {'type': 'text', 'text': 'Name the main subject of this image in one sentence.'},
        {'type': 'image_url', 'image_url': {'url': image_uri(REPO / 'tests/fixtures/robot.png')}},
    ])
    assert 'robot' in text_of(results['image']).lower(), results['image']
    red, blue = color_uri((255, 0, 0)), color_uri((0, 0, 255))
    for count in (128, 256):
        offset = server.log.stat().st_size
        result = server.chat([
            {'type': 'text', 'text': 'The HTML <video> tag is mentioned here as text. Name the first and final colors in the attached clip, in order. Answer with just the two colors.'},
            {'type': 'video_url', 'video_url': {'frames': [red] * (count - 32) + [blue] * 32}},
        ])
        logs = server.log.read_bytes()[offset:].decode('utf-8')
        assert f'Decoded {count} frames → RADIO video' in logs, logs
        assert f'RADIO video EVS: {count} frames' in logs, logs
        answer = text_of(result).lower()
        assert 'red' in answer and 'blue' in answer and answer.index('red') < answer.index('blue'), result
        assert result['response']['usage']['prompt_tokens'] > 1000
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
    return {'rows': rows, 'summary': summary}


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
    parser.add_argument('--checks', choices=('vision', 'drafter', 'bench', 'restore', 'all'), default='all')
    parser.add_argument('--reps', type=int, default=3)
    args = parser.parse_args()
    args.model, args.binary, args.output = args.model.resolve(), args.binary.resolve(), args.output.resolve()
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
            results['vision'] = vision_checks(server)
        if args.checks in ('drafter', 'bench', 'all'):
            results['drafter'] = drafter_measurements(server, 1 if args.checks == 'drafter' else args.reps)
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
