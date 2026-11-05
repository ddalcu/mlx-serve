#!/usr/bin/env python3
"""Check MLX persistence does not stat other cache files after cold initialization.

Runs an isolated server and observes libc calls without changing the runtime.
Requires macOS, a compatible MLX chat pack and its staged runtime libraries.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--files', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        print('SKIP: Darwin syscall interposition probe')
        return
    assert args.files > 0
    with tempfile.TemporaryDirectory(prefix='mlxserve-cache-io-') as scratch:
        home = Path(scratch)
        cache = home / '.mlx-serve/kv-cache'
        other = cache / 'unmanaged-pool'
        other.mkdir(parents=True)
        for i in range(args.files):
            (other / f'cache-io-witness-{i}').write_bytes(b'x')
        probe = home / 'probe.dylib'
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-dynamiclib',
                        str(Path(__file__).with_name('cache_io_probe.c')), '-o', str(probe)], check=True)
        read_fd, write_fd = os.pipe()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        key = secrets.token_urlsafe(24)
        env = os.environ.copy()
        env.update(HOME=str(home), CACHE_IO_PROBE_FD=str(write_fd),
                   DYLD_INSERT_LIBRARIES=str(probe), CACHE_IO_TEST_KEY=key)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.output.with_suffix('.server.log')
        with log_path.open('w') as log:
            proc = subprocess.Popen([str(args.binary.resolve()), '--model', str(args.model.resolve()),
                    '--serve', '--host', '127.0.0.1', '--port', str(port), '--ctx-size', '8192',
                    '--prefix-cache-disk', '2GB', '--no-prefix-cache-ram', '--shutdown-persist-secs', '1',
                    '--api-key-env', 'CACHE_IO_TEST_KEY', '--api-key-strict', '--log-file', 'off'],
                    env=env, pass_fds=(write_fd,), stdout=log, stderr=subprocess.STDOUT)
        os.close(write_fd)

        def call(path, body=None):
            req = urllib.request.Request(base + path,
                    data=None if body is None else json.dumps(body).encode(),
                    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as res:
                return json.load(res)

        def snapshot():
            proc.send_signal(signal.SIGUSR2)
            if not select.select([read_fd], [], [], 5)[0]:
                raise AssertionError('Syscall observer did not report')
            data = os.read(read_fd, 8)
            assert len(data) == 8, data
            return struct.unpack('=Q', data)[0]

        try:
            for _ in range(600):
                if proc.poll() is not None:
                    raise AssertionError(f'Server exited; see {log_path}')
                try:
                    call('/health')
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise AssertionError('Server readiness timeout')
            models = call('/v1/models')['data']
            model = next(row['id'] for row in models if row.get('loaded'))
            def request(prompt):
                result = call('/v1/chat/completions', {'model': model, 'temperature': 0,
                        'enable_thinking': False, 'max_tokens': 32,
                        'messages': [{'role': 'user', 'content': prompt}]})
                assert result['choices'][0]['message'].get('content'), result
                return result
            request('Reply only READY.')
            before = snapshot()
            assert before >= args.files, 'Cold inventory did not reach the syscall observer'
            prompt = '\n'.join(f'Record {i}: the archive stores a neutral river observation.'
                               for i in range(180))
            results = []
            for tag in ('north', 'south', 'north'):
                result = request(f'Archive {tag}.\n{prompt}\nReply only ARCHIVED.')
                assert result['usage']['prompt_tokens'] >= 512, result
                results.append(result)
            deadline = time.monotonic() + 15
            while not any(cache.glob('*/e*/meta.json')) and time.monotonic() < deadline:
                time.sleep(.05)
            assert any(cache.glob('*/e*/meta.json')), 'No persisted checkpoint was published'
            after = snapshot()
            output = {'model': model, 'witness_files': args.files, 'cold_file_stats': before,
                      'hot_file_stats': after - before, 'responses': results,
                      'health': call('/health')}
            args.output.write_text(json.dumps(output, indent=2))
            print(f'cold_file_stats={before} hot_file_stats={after - before}', flush=True)
            assert after == before, 'KV commits traversed another cache namespace'
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            os.close(read_fd)


if __name__ == '__main__':
    main()
