"""Boot-and-poll helpers for the Python live tests, the twin of tests/lib/server.sh.

    from lib.server import wait_health          # tests/ is sys.path[0] when a script runs from there
    process = subprocess.Popen([BIN, '--serve', '--port', str(port), ...], stdout=log, stderr=log)
    wait_health(f'http://127.0.0.1:{port}', 240, process)

Readiness is the STATUS: /health answers 503 "loading" while a model loads, so a bare socket check lies.
"""
import time
import urllib.request


def wait_health(base, timeout_s, process=None, interval=1.0):
    """Return once base/health answers 200; RuntimeError when process exits, TimeoutError past timeout_s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'server exited {process.returncode}')
        try:
            with urllib.request.urlopen(base + '/health', timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(interval)
    raise TimeoutError(f'{base}/health did not answer 200 within {timeout_s}s')


def stop(process, timeout_s=10):
    """Terminate the server this script started and reap it, killing it past timeout_s."""
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except Exception:
        process.kill()
        process.wait()
