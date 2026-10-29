#!/usr/bin/env python3
"""Black-box companion to test_module_prefix_cache_disk.sh. Standard library only.

Results, requests, server logs and manifests remain in the shell driver's scratch HOME.
Exit 77 means an optional fixture or a scheduling window could not be exercised.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def cache_root(home):
    return Path(home) / ".mlx-serve" / "kv-cache"


def boundary(path):
    return read(path)["usage"]["prompt_tokens"] - 1


def cached(result):
    value = result["usage"]["prompt_tokens_details"]["cached_tokens"]
    require(isinstance(value, int), f"cached_tokens is not an integer: {value!r}")
    return value


def initial_messages(conversation_id=None):
    lines = ([f"Conversation marker: {conversation_id}. Read this project log before answering."]
             if conversation_id else ["Read this project log before answering."])
    lines += [f"Entry {i}: subsystem {i % 17}, latency {i * 7 % 131} ms, checksum {i * i % 9973}."
              for i in range(1, 161)]
    lines.append("What integer is the sum of 17 and 25? Reply with just the integer.")
    return [{"role": "user", "content": "\n".join(lines)}]


def continuation(seed):
    return seed["messages"] + [{"role": "assistant", "content": seed["message"]["content"]},
                               {"role": "user", "content": "What integer is the sum of 18 and 25? Reply with just the integer."}]


def auth_headers(api_key, extra_headers=()):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for raw in extra_headers:
        require(isinstance(raw, str) and "\r" not in raw and "\n" not in raw,
                "request header must be a single line")
        name, separator, value = raw.partition(":")
        require(separator and re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name.strip()),
                "request header must use a valid name: value form")
        name = name.strip()
        require(name.lower() not in ("authorization", "content-type"),
                f"--header cannot override {name}")
        headers[name] = value.strip()
    return headers


def build_chat_request(base, messages, max_tokens=24, api_key=None, extra_headers=()):
    body = {"model": "mlx-serve", "messages": messages, "temperature": 0.0,
            "enable_thinking": False, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=auth_headers(api_key, extra_headers))
    return req, body


def chat(base, messages, max_tokens=24, first_token=None, cancel=False, api_key=None,
         extra_headers=()):
    req, body = build_chat_request(base, messages, max_tokens, api_key, extra_headers)
    started = time.monotonic()
    first, usage, finish = None, None, None
    pieces = {"content": [], "reasoning_content": []}
    with urllib.request.urlopen(req, timeout=1200) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            require("error" not in event, f"stream error: {event}")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                has_token = bool(delta.get("content") or delta.get("reasoning_content"))
                for key in pieces:
                    if delta.get(key):
                        pieces[key].append(delta[key])
                if not has_token or first is not None:
                    continue
                first = time.monotonic()
                if first_token:
                    first_token()
                if cancel:
                    return {"cancelled_after_token": True}
    require(first is not None and usage is not None and finish is not None, "incomplete SSE response")
    message = {key: "".join(value) for key, value in pieces.items()}
    return {"request": body, "messages": messages, "message": message, "usage": usage, "finish_reason": finish,
            "ttft_s": first - started, "total_s": time.monotonic() - started}


def request(a):
    msgs = continuation(read(a.seed)) if a.seed else initial_messages(a.conversation_id)
    api_key = os.environ.get(a.api_key_env) if a.request_url else None
    if a.request_url and a.api_key_env and not api_key:
        raise AssertionError(f"{a.api_key_env} is required when --request-url is set")
    result = chat(a.request_url or a.base, msgs, max_tokens=a.max_tokens, api_key=api_key,
                  extra_headers=a.header)
    write(a.output, result)
    print(f"{a.output}: cached={cached(result)}/{result['usage']['prompt_tokens']}, ttft={result['ttft_s']:.3f}s")


def result(a):
    got = read(a.result)
    n = boundary(a.boundary) if a.boundary else a.cached
    if n is not None:
        require(cached(got) == n, f"cached={cached(got)}, expected exactly {n}")
        require(0 <= n < got["usage"]["prompt_tokens"], "restore must leave a token to forward")
    else:
        require(0 <= cached(got) < got["usage"]["prompt_tokens"],
                "cached count must leave a token to forward")
    if a.min_cached is not None:
        require(cached(got) >= a.min_cached,
                f"cached={cached(got)}, expected at least {a.min_cached}")
    if a.legacy_log:
        log = Path(a.legacy_log).read_text()
        matches = re.findall(r"\[dsv41\] resumed (\d+) of (\d+) prompt tokens", log)
        require(matches, "missing EXL3 legacy idle-resume log")
        resumed, prompt = map(int, matches[-1])
        require(resumed == cached(got) and prompt == got["usage"]["prompt_tokens"],
                "legacy resume log does not match this request's cached-token counts")
    if a.cold:
        cold = read(a.cold)
        require(got["request"] == cold["request"], "cold control used a different request")
        require(got["message"] == cold["message"], f"greedy output differs: {got['message']} vs {cold['message']}")
        require(got["finish_reason"] == cold["finish_reason"], "greedy stop reason differs")
    if a.source:
        log = Path(a.log).read_text()
        pattern = rf"module restored source={a.source} id=\d+ position={n}(?:\s|$)"
        require(re.search(pattern, log), f"missing restore engagement: {pattern}")
        print(re.search(pattern, log).group())


def module_records(home):
    out = []
    for path in sorted(cache_root(home).glob("*/e*/meta.json")):
        meta = read(path)
        if meta.get("v") == 10 and meta.get("kind") == "module":
            out.append((path, meta))
    return out


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record_files(path, meta):
    names = ["meta.json", "tokens.bin", "module.header"]
    names += [f"m{part:03d}.safetensors" for part in range(len(meta["part_bytes"]))]
    return {name: digest(path.parent / name) for name in names}


def validate_record(path, meta):
    directory = path.parent
    require(re.fullmatch(r".+-m[0-9a-f]+", directory.parent.name), "module root lacks pack stat hash")
    require(meta["identity"] == directory.parent.name, "root/envelope identity disagreement")
    require(len(path.read_bytes()) <= 65536, "module envelope exceeds its bound")
    require((directory / "tokens.bin").stat().st_size == meta["n"] * 4 == meta["token_bytes"], "token size mismatch")
    header_size = (directory / "module.header").stat().st_size
    require(0 < header_size == meta["header_bytes"] <= 65536, "codec header size mismatch")
    total = path.stat().st_size + header_size + meta["token_bytes"]
    for part, size in enumerate(meta["part_bytes"]):
        tensor_file = directory / f"m{part:03d}.safetensors"
        require(tensor_file.stat().st_size == size, f"wrong part size: {tensor_file}")
        with tensor_file.open("rb") as stream:
            raw = stream.read(8)
            require(len(raw) == 8, f"missing safetensors header: {tensor_file}")
            count = struct.unpack("<Q", raw)[0]
            require(count <= 65536, "safetensors header exceeds its bound")
            descriptors = json.loads(stream.read(count))
        expected, offset = {}, 0
        for tensor in meta["tensors"]:
            if tensor["part"] != part:
                continue
            expected[tensor["name"]] = {"dtype": tensor["dtype"], "shape": tensor["shape"],
                                        "data_offsets": [offset, offset + tensor["bytes"]]}
            offset += tensor["bytes"]
        descriptors.pop("__metadata__", None)
        require(descriptors == expected, "safetensors inventory differs from the envelope")
        require(8 + count + offset == size, "part payload size mismatch")
        total += size
    require(total == meta["bytes"], f"encoded footprint {total} differs from bill {meta['bytes']}")


def layout(a):
    until = time.monotonic() + a.wait
    n = boundary(a.boundary)
    while True:
        records = [(path, meta) for path, meta in module_records(a.root) if meta["n"] == n]
        if records:
            for path, meta in records:
                validate_record(path, meta)
            break
        require(time.monotonic() < until, f"no complete module record at exact position {n}")
        time.sleep(0.1)
    for path, meta in records:
        print(f"v10 module {path}: n={meta['n']}, parts={len(meta['part_bytes'])}, bytes={meta['bytes']}")
    if a.only:
        require(all(meta["n"] == n for _, meta in module_records(a.root)), "cancelled continuation committed a replacement record")
    if a.snapshot:
        write(a.snapshot, {str(path.relative_to(cache_root(a.root))): {"meta": meta, "files": record_files(path, meta)}
                           for path, meta in records})


def preserved(a):
    for relative, expected in read(a.snapshot).items():
        path = cache_root(a.root) / relative
        meta = expected["meta"]
        require(path.is_file() and read(path) == meta, f"older record overwritten or removed: {path}")
        validate_record(path, meta)
        require(record_files(path, meta) == expected["files"], f"older record payload changed: {path}")


def copy_cache(a):
    target = cache_root(a.dest)
    require(not target.exists(), f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_root(a.source), target)


def fixture(a):
    source, dest = Path(a.model).resolve(), Path(a.dest)
    require(read(source / "config.json").get("model_type") == "deepseek_v41", "expected a DeepSeek-V4.1 pack")
    bank = (source / "experts.bin").is_file()
    if bank:
        require(any((source / name).is_file() for name in ("expert-manifest.json", "expert-manifest-v2.json")),
                "EXL3 bank fixture lacks an expert manifest")
    dest.mkdir()
    for path in source.iterdir():
        (dest / path.name).symlink_to(path)
    # Mutation candidates live in this disposable wrapper. Big inputs are cloned copy-on-write;
    # a failed clone stays a symlink and makes that mutation arm SKIP.
    requested = a.identity_file
    if requested:
        mutable = source / requested
        require(mutable.is_file(), f"identity input is absent: {requested}")
        require(Path(requested).name == requested, f"identity input must be a pack-root file: {requested}")
        require(requested in {"experts.bin", "expert-manifest.json", "expert-manifest-v2.json",
                              "engram-token-map.u32", "engram_token_map.json"} or
                requested.endswith(".safetensors"), f"unsupported identity input: {requested}")
    else:
        mutable = next((source / name for name in ("engram-token-map.u32", "engram_token_map.json")
                        if (source / name).is_file()), None)
        if mutable is None and bank:
            mutable = next((source / name for name in ("expert-manifest-v2.json", "expert-manifest.json")
                            if (source / name).is_file()), None)
        if mutable is None:
            index = source / "model.safetensors.index.json"
            selected = sorted(set(read(index).get("weight_map", {}).values())) if index.is_file() else []
            mutable = next((source / name for name in selected if (source / name).is_file()), None)
            mutable = mutable or next(source.glob("*.safetensors"), None)
    cloned = mutable
    if mutable and mutable.name in {"expert-manifest.json", "expert-manifest-v2.json",
                                    "engram-token-map.u32", "engram_token_map.json"}:
        (dest / mutable.name).unlink()
        shutil.copy2(mutable, dest / mutable.name)
    elif mutable:
        target = dest / mutable.name
        target.unlink()
        command = ["cp", "-c"] if sys.platform == "darwin" else ["cp", "--reflink=always"]
        size = mutable.stat().st_size
        free = shutil.disk_usage(dest).free
        if free < size + (2 << 30):
            print(f"SKIP: insufficient free space for a guarded identity clone ({size >> 30} GiB input)")
            target.symlink_to(mutable)
            cloned = None
        else:
            clone = subprocess.run(command + [str(mutable), str(target)], capture_output=True)
            if clone.returncode:
                target.unlink(missing_ok=True)
                target.symlink_to(mutable)
                cloned = None
    stamp = None
    if cloned:
        require(not (dest / cloned.name).is_symlink(), "fixture mutation input still points at the source pack")
        stat = (dest / cloned.name).stat()
        stamp = {"name": mutable.name, "atime_ns": stat.st_atime_ns, "mtime_ns": stat.st_mtime_ns}
    require(not (dest / "module-test-fixture.json").exists(), "fixture bookkeeping name already exists")
    write(dest / "module-test-fixture.json", stamp)


def touch_pack(a):
    stamp = read(Path(a.pack) / "module-test-fixture.json")
    if stamp is None:
        print("SKIP: no disposable copy-on-write identity input is available")
        return 77
    path = Path(a.pack) / stamp["name"]
    require(not path.is_symlink(), "refusing to touch the original pack")
    os.utime(path, ns=(stamp["atime_ns"], stamp["mtime_ns"] + 2_000_000_000))
    require(path.stat().st_mtime_ns != stamp["mtime_ns"], "pack mtime did not change")


def touch_ready(a):
    stamp = read(Path(a.pack) / "module-test-fixture.json")
    if stamp is None:
        print("SKIP: no disposable copy-on-write identity input is available")
        return 77
    path = Path(a.pack) / stamp["name"]
    require(not path.is_symlink(), "refusing to mutate an identity input linked to the source pack")
    print(f"disposable identity input ready: {path.name}")


def restore_stat(a):
    stamp = read(Path(a.pack) / "module-test-fixture.json")
    path = Path(a.pack) / stamp["name"]
    require(not path.is_symlink(), "refusing to restore stat through a source-pack symlink")
    os.utime(path, ns=(stamp["atime_ns"], stamp["mtime_ns"]))


def changed_root(a):
    old = {Path(relative).parts[0] for relative in read(a.snapshot)}
    roots = {path.parent.parent.name for path, _ in module_records(a.root)}
    require(old < roots, f"pack stat change did not publish under a distinct root: {roots}")
    preserved(a)


def truncate(a):
    candidates = [(path, meta) for path, meta in module_records(a.root) if meta["n"] == boundary(a.boundary)]
    require(len(candidates) == 1, f"expected one record to truncate, got {len(candidates)}")
    path, meta = candidates[0]
    part = path.parent / "m000.safetensors"
    require(part.stat().st_size == meta["part_bytes"][0], "part was already corrupt")
    with part.open("r+b") as stream:
        stream.truncate(part.stat().st_size - 1)


def dropped(a):
    log = Path(a.log).read_text()
    for relative in read(a.snapshot):
        entry = Path(relative).parent.name
        require(f"dropping incomplete entry {entry}" in log, f"scan did not reject {entry}")


def disabled(a):
    log = Path(a.log).read_text()
    if a.legacy_resume:
        require(not module_records(a.root), "disabled EXL3 control wrote shared module records to disk")
        require(re.search(r"\[dsv41\] resumed [1-9]\d* of \d+ prompt tokens", log),
                "EXL3 control did not preserve a legacy idle resume")
        require(not re.search(r"module restored source=(?:ram|disk)", log),
                "disabled EXL3 control restored shared module history")
    else:
        require(not list(cache_root(a.root).glob("**/*")), "disabled cache wrote to disk")
        require("[disk-cache]" not in log, "disabled cache performed SSD work")
    if a.reason:
        require("this model retains no conversation" in log, "missing disabled module history diagnostic")


def port_free(a):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", a.port))


def available_bytes():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative|purgeable):\s+(\d+)", out)}
    return page * sum(pages.values())


def mem_wait(a):
    # A stopped server's resident weights can take seconds to return to the OS. Prove the old
    # process is gone and the port is free, then wait for resident shards plus fixed headroom.
    if a.previous_pid is not None:
        if alive(a.previous_pid):
            current_start = ps(a.previous_pid, "lstart")
            require(a.previous_start is not None and current_start != a.previous_start,
                    f"previous server PID {a.previous_pid} is still present")
    if a.port is not None:
        port_free(a)
    need = required_model_bytes(a.model)
    until = time.monotonic() + a.timeout
    while available_bytes() < need and time.monotonic() < until:
        time.sleep(1)
    require(available_bytes() >= need, f"memory not released: {available_bytes() >> 30} GiB free, {need >> 30} GiB needed")


def required_model_bytes(model):
    # experts.bin is an mmap-backed streamed bank, not a resident allocation. Its file size
    # must never become a RAM threshold; resident trunk shards retain the old safety bound.
    root = Path(model)
    need = sum(path.stat().st_size for path in root.glob("*.safetensors"))
    return need + (8 << 30)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop(a):
    started = time.monotonic()
    os.kill(a.pid, signal.SIGTERM)
    while alive(a.pid) and time.monotonic() - started < a.timeout:
        time.sleep(0.1)
    if alive(a.pid):
        os.kill(a.pid, signal.SIGKILL)
        raise AssertionError(f"server did not exit within {a.timeout}s of SIGTERM")
    print(f"SIGTERM exit: {time.monotonic() - started:.3f}s")


def crash_window(a):
    until = time.monotonic() + 1200
    while time.monotonic() < until and alive(a.pid):
        entries = list(cache_root(a.root).glob("*/e*"))
        for entry in entries:
            if not (entry / "module.header").is_file() or (entry / "meta.json").exists():
                continue
            os.kill(a.pid, signal.SIGSTOP)
            if (entry / "meta.json").exists():
                os.kill(a.pid, signal.SIGCONT)
                continue
            write(a.output, {"entry": str(entry.relative_to(cache_root(a.root))),
                             "files": sorted(path.name for path in entry.iterdir())})
            os.kill(a.pid, signal.SIGKILL)
            print(f"SIGKILL before metadata: {entry}")
            return
        if any((entry / "meta.json").exists() for entry in entries):
            print("SKIP: export completed before the crash window was observed")
            return 77
        time.sleep(0.002)
    raise AssertionError("no module export observed before crash-window timeout/server exit")


def crash_scan(a):
    entry = read(a.window)["entry"]
    require(not (cache_root(a.root) / entry).exists(), "crash leftover survived scan")
    require(not module_records(a.root), "restart accepted a crash leftover")
    require(f"dropping incomplete entry {Path(entry).name}" in Path(a.log).read_text(), "missing crash scan diagnostic")


def summary(a):
    log = Path(a.log).read_text()
    lines = [line for line in log.splitlines() if "[disk-cache] shutdown persist:" in line]
    if a.deadline == 0:
        require(not lines, "disabled shutdown persistence ran a flush")
        print("disabled deadline produces no shutdown persist invocation")
        return
    require(len(lines) == 1, f"expected one shutdown summary, got {len(lines)}")
    pattern = (r"persist: (\d+) written \([\d.]+ MB\), (\d+) left \(deadline\), "
               r"(\d+) left \(store declined\), (\d+) skipped checked-out, (\d+) failures, "
               r"[\d.]+ MB dropped at the deadline, ([\d.]+) s")
    match = re.search(pattern, lines[0])
    require(match, f"unrecognized shutdown tally: {lines[0]}")
    written, left, declined, checked_out, failures = map(int, match.groups()[:5])
    require(failures == 0 and declined == 0 and checked_out == 0, f"unexpected shutdown failure: {lines[0]}")
    print(lines[0])
    if a.root:
        complete = module_records(a.root)
        for path, meta in complete:
            validate_record(path, meta)
        require(written <= len(complete), "shutdown reported more completed records than exist")
        if left:
            require("module export cancelled" in log, "deadline leftovers lacked a cancellation diagnostic")
        else:
            print("SKIP: mid-slice expiry: export completed within the deadline; unit guard owns the boundary")


def cancellation_count(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        text = response.read().decode()
    match = re.search(r"^vllm:request_cancelled_total (\d+)$", text, re.MULTILINE)
    require(match, "missing cancellation metric")
    return int(match[1])


def cancel(a):
    before = cancellation_count(a.base)
    state = {}
    offset = len(Path(a.log).read_text())

    def follow():
        try:
            msgs = initial_messages() + [{"role": "assistant", "content": "Noted."},
                                         {"role": "user", "content": "Write 4000 numbered lines, one integer per line, starting at one."}]
            api_key = os.environ.get(a.api_key_env) if a.request_url else None
            if a.request_url and a.api_key_env and not api_key:
                raise AssertionError(f"{a.api_key_env} is required when --request-url is set")
            state["reply"] = chat(a.request_url or a.base, msgs, max_tokens=8192, cancel=True,
                                  api_key=api_key,
                                  extra_headers=a.header,
                                  first_token=lambda: state.update(dirty=not bool(module_records(a.root))))
        except Exception as error:
            state["error"] = str(error)

    thread = threading.Thread(target=follow, daemon=True)
    # Start the continuation only once the seed's stream has ended: started on the seed's first
    # token it was admitted and prefilled beside the seed, before the seed committed anything to
    # restore. Starting at once keeps the idle window (and its export) short; a missed window SKIPs.
    seed = chat(a.base, initial_messages(), max_tokens=1)
    thread.start()
    write(a.output, seed)
    thread.join(timeout=1200)
    require(not thread.is_alive(), "continuation did not start")
    require("error" not in state, f"continuation failed: {state}")
    require(state.get("reply", {}).get("cancelled_after_token"), "continuation was not disconnected mid-turn")
    until = time.monotonic() + 30
    while cancellation_count(a.base) == before and time.monotonic() < until:
        time.sleep(0.1)
    require(cancellation_count(a.base) == before + 1, "disconnect did not cancel exactly one request")
    log = Path(a.log).read_text()[offset:]
    require(re.search(rf"module restored source=ram id=\d+ position={seed['usage']['prompt_tokens'] - 1}(?:\s|$)", log),
            "cancelled continuation did not consume the previous RAM snapshot")
    if not state.get("dirty"):
        print("SKIP: dirty consumption: old record was already published when continuation streamed")
        return 77
    print("consumed an unpublished RAM checkpoint, disconnected and observed cancellation")


def inject_module(a):
    kv = [(path, read(path)) for path in cache_root(a.root).glob("*/e*/meta.json")]
    kv = [(path, meta) for path, meta in kv if 2 <= meta.get("v", 0) <= 9 and meta.get("kind") != "module"]
    require(kv, "no v2-v9 KV record persisted")
    parent = kv[0][0].parent.parent
    identity = max(int(path.name[1:]) for path in parent.glob("e*") if path.is_dir()) + 1
    target = parent / f"e{identity}"
    source = module_records(a.source)[0][0].parent
    shutil.copytree(source, target)
    write(a.output, {"entry": target.name, "kv_versions": sorted({meta["v"] for _, meta in kv})})
    print(f"injected v10 module at {target}; KV versions={read(a.output)['kv_versions']}")


def kv_result(a):
    got, cold = read(a.result), read(a.cold)
    require(got["messages"] == cold["messages"] and got["message"] == cold["message"], "KV restart output differs")
    require(0 < cached(got) < got["usage"]["prompt_tokens"], "KV reader failed to restore")
    log = Path(a.log).read_text()
    entry = read(a.injected)["entry"]
    require(f"dropping incomplete entry {entry}" in log, "reader accepted the injected v10 record")
    require(re.search(r"\[disk-cache\] restored \d+/\d+ tokens", log), "KV restore did not engage")


def unit_owners(_):
    print('Unit-owned branches (also cover optional process SKIPs):')
    for filename, name in [
        ("cache/ssd/disk_tier.zig", "scanning a module crash leftover removes the whole directory"),
        ("cache/ssd/disk_tier.zig", "missing or truncated required module files reject the entire record"),
        ("load.zig", "module pack identity follows selected weights and token map precedence"),
        ("cache/prefix_cache.zig", "expired shutdown starts no module files"),
        ("cache/prefix_cache.zig", "shutdown stops after one overdue module file"),
        ("cache/ssd/disk_tier.zig", "v10 module records are isolated from every KV entry point"),
        ("cache/ssd/disk_tier.zig", "v2 through v9 KV scans retain contiguous chunk salvage"),
        ("cache/prefix_cache.zig", "failed and cancelled requests leave the old module export owed"),
    ]:
        print(f'  src/engines/mlx/{filename}: test "{name}"')


def receipt(a):
    log = Path(a.log).read_text() if a.log and Path(a.log).is_file() else ""
    passed = [line.strip() for line in log.splitlines() if line.strip().startswith("PASS:")]
    skipped = [line.strip() for line in log.splitlines() if line.strip().startswith("SKIP:")]
    binary = Path(a.binary)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    write(output, {
        "schema": 1,
        "test": "test_module_prefix_cache_disk.sh",
        "status": "passed" if a.exit_code == 0 else "failed",
        "exit_code": a.exit_code,
        "started_utc": a.started_utc,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": Path(a.model).name,
        "binary": binary.name,
        "binary_sha256": digest(binary) if binary.is_file() else None,
        "prefix_cache_ram_budget": a.ram_budget,
        "prefix_cache_disk_budget": a.disk_budget,
        "passed_cases": passed,
        "skipped_cases": skipped,
    })
    os.chmod(output, 0o600)
    print(f"receipt: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    specs = {
        "request": (request, "base output", "seed"),
        "result": (result, "", "boundary cold log source"),
        "layout": (layout, "root boundary", "snapshot"),
        "preserved": (preserved, "root snapshot", ""),
        "copy-cache": (copy_cache, "source dest", ""),
        "fixture": (fixture, "model dest", ""),
        "touch-pack": (touch_pack, "pack", ""),
        "touch-ready": (touch_ready, "pack", ""),
        "restore-stat": (restore_stat, "pack", ""),
        "changed-root": (changed_root, "root snapshot", ""),
        "truncate": (truncate, "root boundary", ""),
        "dropped": (dropped, "log snapshot", ""),
        "disabled": (disabled, "root log", ""),
        "port-free": (port_free, "", ""),
        "mem-wait": (mem_wait, "model", ""),
        "stop": (stop, "", ""),
        "crash-window": (crash_window, "root output", ""),
        "crash-scan": (crash_scan, "root log window", ""),
        "summary": (summary, "log", "root"),
        "cancel": (cancel, "base root log output", ""),
        "inject-module": (inject_module, "root source output", ""),
        "kv-result": (kv_result, "cold log injected", ""),
        "unit-owners": (unit_owners, "", ""),
        "receipt": (receipt, "output", "log"),
    }
    extra = {
        "result": [("result", {}), ("--cached", {"type": int}),
                   ("--min-cached", {"type": int}), ("--legacy-log", {})],
        "kv-result": [("result", {})],
        "disabled": [("--reason", {"action": "store_true"}),
                     ("--legacy-resume", {"action": "store_true"})],
        "layout": [("--wait", {"type": float, "default": 0}), ("--only", {"action": "store_true"})],
        "request": [("--max-tokens", {"type": int, "default": 24}),
                    ("--request-url", {}), ("--api-key-env", {"default": "MLX_SERVE_API_KEY"}),
                    ("--header", {"action": "append", "default": []}),
                    ("--conversation-id", {})],
        "fixture": [("--identity-file", {})],
        "cancel": [("--request-url", {}), ("--api-key-env", {"default": "MLX_SERVE_API_KEY"}),
                   ("--header", {"action": "append", "default": []})],
        "port-free": [("--port", {"required": True, "type": int})],
        "mem-wait": [("--timeout", {"type": float, "default": 180}),
                     ("--previous-pid", {"type": int}), ("--previous-start", {}),
                     ("--port", {"type": int})],
        "receipt": [("--exit-code", {"required": True, "type": int}),
                    ("--started-utc", {"required": True}),
                    ("--model", {"required": True}), ("--binary", {"required": True}),
                    ("--ram-budget", {"required": True}), ("--disk-budget", {"required": True})],
        "stop": [("--pid", {"required": True, "type": int}), ("--timeout", {"required": True, "type": float})],
        "crash-window": [("--pid", {"required": True, "type": int})],
        "summary": [("--deadline", {"required": True, "type": int})],
    }
    for name, (function, required, optional) in specs.items():
        sub = commands.add_parser(name)
        sub.set_defaults(run=function)
        for flag in required.split():
            sub.add_argument("--" + flag, required=True)
        for flag in optional.split():
            sub.add_argument("--" + flag)
        for flag, options in extra.get(name, []):
            sub.add_argument(flag, **options)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
