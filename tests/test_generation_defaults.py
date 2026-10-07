#!/usr/bin/env python3
"""Generation policies apply across APIs, reload without model reload, and enforce client locks."""
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

binary = pathlib.Path(os.environ.get('MLX_SERVE_BINARY', './zig-out/bin/mlx-serve')).resolve()
model = pathlib.Path(sys.argv[1]).resolve()
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
base = f'http://127.0.0.1:{port}'
passed = 0


def post(route, body):
    request = urllib.request.Request(base + route, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        if not body.get('stream'):
            return json.load(response)
        events = []
        for raw in response:
            if raw.startswith(b'data: ') and raw.strip() != b'data: [DONE]':
                events.append(json.loads(raw[6:]))
        if route == '/v1/messages':
            usage = {}
            for event in events:
                usage.update(event.get('message', {}).get('usage', {}))
                usage.update(event.get('usage', {}))
            return {'usage': usage, 'events': events}
        if route == '/v1/responses':
            return next(event['response'] for event in reversed(events) if event['type'] == 'response.completed')
        return next(event for event in reversed(events) if event.get('usage'))


def check(condition, message):
    global passed
    assert condition, message
    passed += 1
    print('PASS:', message, flush=True)


with tempfile.TemporaryDirectory(prefix='mlx-generation-defaults-') as home:
    settings_dir = pathlib.Path(home, '.mlx-serve')
    settings_dir.mkdir()
    global_file = settings_dir / 'generation-settings.json'
    model_file = settings_dir / 'model-settings.json'

    def write(path, value):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value))
        temporary.replace(path)
        time.sleep(1.1)

    write(global_file, {
        'temperature': {'value': 0},
        'max_tokens': {'value': 2, 'ignore_client': True},
        'enable_thinking': {'value': False, 'ignore_client': True},
    })
    log_path = pathlib.Path(home, 'server.log')
    with log_path.open('a') as log:
        process = subprocess.Popen([
            str(binary), '--serve', '--host', '127.0.0.1', '--port', str(port),
            '--model', str(model), '--ctx-size', '8192', '--kv-quant', '8',
            '--prefix-cache-entries', '0', '--no-mtp', '--no-drafter', '--no-pld',
            '--no-prevent-sleep', '--log-file', str(log_path),
        ], env={**os.environ, 'HOME': home}, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text()[-2000:])
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError('server did not start')
            message = [{'role': 'user', 'content': 'Count from 1 to 100, one number per line.'}]
            cases = [
                ('/v1/chat/completions', {'model': 'mlx-serve', 'messages': message, 'max_completion_tokens': 64, 'enable_thinking': True, 'reasoning_effort': 'xhigh', 'stream_options': {'include_usage': True}}, 'completion_tokens'),
                ('/v1/messages', {'model': 'mlx-serve', 'messages': message, 'max_tokens': 64, 'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'xhigh'}}, 'output_tokens'),
                ('/v1/responses', {'model': 'mlx-serve', 'input': message, 'max_output_tokens': 64, 'reasoning': {'effort': 'xhigh'}}, 'output_tokens'),
            ]
            for route, body, output_key in cases:
                for stream in (False, True):
                    result = post(route, {**body, 'stream': stream})
                    check(result['usage'][output_key] <= 2, f'{route} stream={stream}: locked output cap wins')
            result = post('/v1/completions', {'model': 'mlx-serve', 'prompt': 'Count from 1 to 100.', 'max_tokens': 64})
            check(result['usage']['completion_tokens'] <= 2, 'raw completions ignores the thinking lock and keeps the output lock')
            write(model_file, {str(model): {'generation_defaults': {'max_tokens': {'value': 3, 'ignore_client': False}}}})
            result = post('/v1/chat/completions', {'model': 'mlx-serve', 'messages': message, 'max_tokens': 5})
            check(result['usage']['completion_tokens'] == 5, 'model unlocked rule replaces inherited global lock')
            result = post('/v1/chat/completions', {'model': 'mlx-serve', 'messages': message})
            check(result['usage']['completion_tokens'] == 3, 'model default fills client omission without reload')
            write(model_file, {})
            result = post('/api/chat', {'model': 'mlx-serve', 'messages': message, 'stream': False, 'options': {'num_predict': 64, 'temperature': 1}})
            check(result['eval_count'] <= 2, 'Ollama native option cannot bypass output lock')
            write(global_file, {
                'temperature': {'value': 0},
                'max_tokens': {'value': 0, 'ignore_client': True},
                'enable_thinking': {'value': False, 'ignore_client': True},
            })
            for stream in (False, True):
                start = len(log_path.read_text())
                post('/v1/messages', {'model': 'mlx-serve', 'messages': message,
                                      'max_tokens': 8, 'stream': stream})
                caps = [line for line in log_path.read_text()[start:].splitlines()
                        if 'prompt=' in line and 'max_gen=' in line and 'ctx=' in line]
                check(bool(caps) and all(int(line.split('max_gen=')[1].split(',')[0]) ==
                      8192 - int(line.split('prompt=')[1].split()[0]) for line in caps),
                      f'messages stream={stream}: forced Auto uses remaining context instead of client cap')
            write(global_file, {
                'temperature': {'value': 0},
                'enable_thinking': {'value': True},
                'reasoning_effort': {'value': 'none', 'ignore_client': True},
            })
            for route, body, _ in cases:
                for stream in (False, True):
                    start = len(log_path.read_text())
                    post(route, {**body, 'stream': stream})
                    lines = [line for line in log_path.read_text()[start:].splitlines()
                             if f'POST {route} (' in line]
                    check(bool(lines) and all('thinking=false' in line for line in lines),
                          f'{route} stream={stream}: forced effort none beats unforced thinking on')
            write(global_file, {
                'temperature': {'value': 0.25},
                'max_tokens': {'value': 2, 'ignore_client': True},
            })
            for route, body, _ in cases:
                post(route, {**body, 'temperature': None})
            post('/v1/completions', {'model': 'mlx-serve', 'prompt': 'Count from 1 to 100.', 'temperature': None})
            lines = [line for line in log_path.read_text().splitlines() if 'POST /v1/' in line and 'temp=' in line]
            check(len(lines) >= 4 and all('temp=0.25' in line for line in lines[-4:]),
                  'null client fields take the global default on every text API')
            write(model_file, {str(model): {'generation_defaults': {'top_k': {'value': -1, 'ignore_client': True}}}})
            result = post('/v1/chat/completions', {'model': 'mlx-serve', 'messages': message})
            check(result['usage']['completion_tokens'] == 2 and 'generation_defaults ignored' in log_path.read_text(),
                  'malformed model policy is logged and ignored, the request is served')
            write(model_file, {})
            write(global_file, {
                'temperature': {'value': 0},
                'reasoning_budget': {'value': 16, 'ignore_client': True},
            })
            tools = [{'name': 'lookup', 'input_schema': {'type': 'object', 'properties': {}}}]
            for stream in (False, True):
                post('/v1/messages', {'model': 'mlx-serve', 'messages': [{'role': 'user', 'content': 'Think carefully about 17 times 23, then answer.'}],
                                     'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'xhigh'},
                                     'tools': tools, 'max_tokens': 96, 'stream': stream})
            check('[think-bound] reasoning budget 16 reached' in log_path.read_text(), 'adaptive xhigh request closes thought at forced budget')
            write(global_file, {'temperature': {'value': 0}, 'max_tokens': {'value': 2, 'ignore_client': True}})
            for stream in (False, True):
                result = post('/v1/completions', {'model': 'mlx-serve', 'prompt': 'Count from 1 to 100.',
                                                 'max_tokens': 64, 'stream': stream,
                                                 'stream_options': {'include_usage': True}})
                check(result['usage']['completion_tokens'] <= 2,
                      f'raw completions stream={stream}: sampling/output policy applies')
            global_file.write_text('{broken')
            time.sleep(1.1)
            result = post('/v1/messages', {'model': 'mlx-serve', 'messages': message, 'max_tokens': 8})
            check(result['usage']['output_tokens'] > 2 and 'generation-settings.json: malformed' in log_path.read_text(),
                  'malformed global file is logged and ignored, the request is served')
            print(f'{passed} checks passed', flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
