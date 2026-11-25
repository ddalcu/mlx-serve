"""dsv41_long_kld_prep.py: the long-context KLD test's documents as text (tests/dsv41_long_kld.py), built once on CPU.

GLM-5.3-Flash's kld_prep.py (the approved 32-document design) without a tokenizer: the boxes' Python has no
`tokenizers`, so this writes text documents sized generously and the driver cuts them through the server's /tokenize
(`ids` phase), where the held-out check also runs. Standard library only; run isolated, every input path an argument:

  python3 -I dsv41_long_kld_prep.py <data dir: dl/ from dsv41_long_kld_fetch.sh, no docs/ or refs/ yet>
      <code spec json> <CPython 3.14.4 Lib dir> <prompts30 manifest.json> <reference prompt json>...

- Code (12, d01-d12): each spec entry is one file, or one directory's files in path order (recursive when the entry
  says so), each file after a "---- <path> ----" comment line, cut at a line end past CODE_CHARS characters (3.2
  characters a token, twice 16,384 tokens). A source with fewer characters is kept whole down to CODE_MIN (1.25
  times 16,384 tokens); under that it stops the prep (the driver stops too on one under 16,384 tokens).
- Prose (up to 14 candidates, from d13: 10 used, the rest spares): PG-19 test books (dl/pg19, Gutenberg id order)
  after their "Produced by" credit, cut at a line end past PROSE_CHARS (4 characters a token, 1.6 times 16,384 tokens);
  a book shorter than that is skipped and named. Fewer than 10 stops the prep.
- Chat (12 candidates, after the prose: 10 used, 2 spares): see below.
- Chat: UltraChat-200k test_sft dialogues (dl/ultrachat, row order) joined
  into one conversation, user turn first, each turn's text stripped, until its message text passes CHAT_CHARS (4
  characters a token, 1.3 times 16,384 tokens). The driver renders it with DeepSeek-V4.1's chat template and drops
  trailing turns to 16,384 tokens.
- No document may hold DeepSeek-V4.1's special-token spellings ("｜", "<think>", "</think>"): the tokenizer would read
  them as control tokens. A code source is cut before the first one (and stops when that leaves it short); a prose or
  chat candidate that holds one is skipped and named.
- The held-out reference (refs/): each reference prompt JSON's message text (coding.json, prompts30 p01, p10, p14,
  p21, whatever is given), and for every prompt of the prompts30 manifest the CPython source lines it quotes (the
  manifest's file and line ranges, read from the Lib dir; each file's character count is checked against the
  manifest's). The driver tokenizes these and drops or stops on any document that shares a 64-token run with them.
- Writes docs/dNN.json, refs/*.txt and manifest.json (each document's and reference's sha256, the sources' paths or
  URLs and sha256s).
"""
import hashlib
import json
import os
import sys

N = 16384
CODE_CHARS = int(N * 3.2 * 2.0)
CODE_MIN = int(N * 3.2 * 1.25)  # a whole source shorter than CODE_CHARS is kept from this size
PROSE_CHARS = int(N * 4.0 * 1.6)
CHAT_CHARS = int(N * 4.0 * 1.3)
N_CODE, N_PROSE, N_CHAT = 12, 14, 12
FORBIDDEN = ("｜", "<think>", "</think>")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_text(p):
    with open(p, "rb") as f:
        return f.read().decode("utf-8", errors="strict")


def first_forbidden(text):
    hits = [text.find(s) for s in FORBIDDEN]
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else -1


def cut_at_line(text, chars):
    """The text up to the first line end at or past `chars` characters (all of it when shorter)."""
    if len(text) <= chars:
        return text
    e = text.find("\n", chars)
    return text if e < 0 else text[:e + 1]


def code_files(s):
    root = s["root"]
    if os.path.isfile(root):
        return os.path.dirname(root), [root]
    files = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            if any(f.endswith(e) for e in s["ext"]) and not any(f.endswith(x) for x in s.get("skip", [])):
                files.append(os.path.join(dp, f))
        if not s.get("recursive"):
            break
    files.sort(key=lambda p: os.path.relpath(p, root))
    return root, files


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    kdir, spec_path, lib_dir, p30_manifest = sys.argv[1:5]
    ref_jsons = sys.argv[5:]
    out, refs_dir = os.path.join(kdir, "docs"), os.path.join(kdir, "refs")
    for d in (out, refs_dir):
        if os.path.exists(d):
            sys.exit(f"exists: {d}")
    os.makedirs(out)
    os.makedirs(refs_dir)
    docs, skipped, refs = [], [], []

    def write(doc):
        p = os.path.join(out, doc["id"] + ".json")
        body = json.dumps(doc, ensure_ascii=False).encode("utf-8")
        with open(p, "wb") as f:
            f.write(body)
        chars = len(doc["text"]) if "text" in doc else sum(len(m["content"]) for m in doc["messages"])
        entry = {k: doc[k] for k in ("id", "kind", "label", "lang") if k in doc}
        entry.update(file=os.path.relpath(p, kdir), sha256=sha256_bytes(body), chars=chars)
        docs.append(entry)
        print(f"{doc['id']} {doc['kind']:5} {chars:7} chars  {doc['label']}", flush=True)

    # the held-out reference: the given prompt JSONs' message text, then prompts30's quoted CPython lines
    def ref(name, text, sources):
        p = os.path.join(refs_dir, name + ".txt")
        b = text.encode("utf-8")
        with open(p, "wb") as f:
            f.write(b)
        refs.append({"name": name, "file": os.path.relpath(p, kdir), "sha256": sha256_bytes(b), "chars": len(text), "sources": sources})

    for rf in ref_jsons:
        body = json.loads(read_text(rf))
        texts = [m["content"] for m in body.get("messages", []) if isinstance(m.get("content"), str)]
        if isinstance(body.get("prompt"), str):
            texts.append(body["prompt"])
        if not texts:
            sys.exit(f"{rf}: no message text")
        ref("prompt-" + os.path.splitext(os.path.basename(rf))[0], "\n".join(texts), [{"path": rf, "sha256": sha256_file(rf)}])
    man30 = json.loads(read_text(p30_manifest))
    src_notes = []
    for p in man30["prompts"]:
        parts, srcs = [], []
        for sf in p["source_files"]:
            f = os.path.join(lib_dir, sf["path"][4:] if sf["path"].startswith("Lib/") else sf["path"])
            lines = read_text(f).splitlines(keepends=True)
            if len(lines) != sf["total_lines"]:
                src_notes.append(f"{p['id']} {sf['path']}: {len(lines)} lines, the manifest's {sf['total_lines']}")
            lo, hi = (1, len(lines)) if sf["lines"] == "complete" else sf["lines"]
            seg = "".join(lines[lo - 1:hi])
            if len(seg) != sf["chars"]:
                src_notes.append(f"{p['id']} {sf['path']} lines {lo}-{hi}: {len(seg)} characters, the manifest's {sf['chars']}")
            parts.append(seg)
            srcs.append({"path": f, "sha256": sha256_file(f), "lines": [lo, hi]})
        ref(f"p30src-{p['id']}", "\n".join(parts), srcs)
    for s in src_notes:
        print("reference note:", s, flush=True)
    print(f"held-out reference: {len(refs)} texts, {sum(r['chars'] for r in refs)} characters", flush=True)

    # code
    spec = json.loads(read_text(spec_path))
    if len(spec) != N_CODE:
        sys.exit(f"the code spec names {len(spec)} sources, not {N_CODE}")
    for i, s in enumerate(spec):
        base, files = code_files(s)
        mark = "# " if s["lang"] == "python" else "// "
        text, srcs = "", []
        for f in files:
            body = read_text(f)
            text += f"{mark}---- {os.path.relpath(f, base)} ----\n" + body
            if not text.endswith("\n"):
                text += "\n"
            srcs.append({"path": f, "sha256": sha256_file(f)})
            if len(text) >= CODE_CHARS:
                break
        bad = first_forbidden(text)
        if 0 <= bad:
            cut = text.rfind("\n", 0, bad) + 1
            print(f"code {s['label']}: cut before a special-token spelling at character {bad}", flush=True)
            text = text[:cut]
        if len(text) < CODE_MIN:
            sys.exit(f"code {s['label']}: {len(text)} characters, under {CODE_MIN}: pick another source")
        write({"id": f"d{i + 1:02d}", "kind": "code", "lang": s["lang"], "label": s["label"], "sources": srcs,
               "text": cut_at_line(text, CODE_CHARS)})

    # prose
    pg = os.path.join(kdir, "dl", "pg19")
    rows = [l.rstrip("\n").split("\t") for l in read_text(os.path.join(pg, "SOURCES.tsv")).splitlines(keepends=True)][1:]
    books = sorted((r for r in rows if r[0].endswith(".txt") and r[0][:-4].isdigit()), key=lambda r: int(r[0][:-4]))
    n0 = len(docs)
    for r in books:
        if len(docs) == n0 + N_PROSE:
            break
        f = os.path.join(pg, r[0])
        if sha256_file(f) != r[2]:
            sys.exit(f"{f}: sha256 is not SOURCES.tsv's")
        body = read_text(f).lstrip()
        if body.lower().startswith("produced by"):
            cut = body.find("\n\n")
            body = body[cut:].lstrip() if cut >= 0 else body
        if len(body) < PROSE_CHARS:
            skipped.append(f"pg19 {r[0]}: {len(body)} characters, under {PROSE_CHARS}")
            continue
        body = cut_at_line(body, PROSE_CHARS)
        if first_forbidden(body) >= 0:
            skipped.append(f"pg19 {r[0]}: holds a special-token spelling")
            continue
        title = next((l.strip() for l in body.splitlines() if l.strip()), "")[:80]
        write({"id": f"d{len(docs) + 1:02d}", "kind": "prose", "label": f"PG-19 test {r[0][:-4]}: {title}",
               "sources": [{"url": r[1], "sha256": r[2], "file": os.path.relpath(f, kdir)}], "text": body})
    if len(docs) < n0 + 10:
        sys.exit(f"only {len(docs) - n0} prose candidates, under the 10 used")

    # chat
    uc = os.path.join(kdir, "dl", "ultrachat")
    urows = [l.rstrip("\n").split("\t") for l in read_text(os.path.join(uc, "SOURCES.tsv")).splitlines(keepends=True)][1:]
    dialogues = []
    for r in sorted(urows, key=lambda r: int(r[0][5:-5])):
        f = os.path.join(uc, r[0])
        if sha256_file(f) != r[2]:
            sys.exit(f"{f}: sha256 is not SOURCES.tsv's")
        page = json.loads(read_text(f))
        for row in page["rows"]:
            msgs = row["row"]["messages"]
            ok = msgs and all(isinstance(m.get("content"), str) and m["content"].strip() and m.get("role") == ("user" if j % 2 == 0 else "assistant")
                              for j, m in enumerate(msgs))
            if not ok or row.get("truncated_cells"):
                skipped.append(f"ultrachat row {row['row_idx']}: not alternating user/assistant text, or truncated")
                continue
            if any(first_forbidden(m["content"]) >= 0 for m in msgs):
                skipped.append(f"ultrachat row {row['row_idx']}: holds a special-token spelling")
                continue
            # stripped, so a server that strips a turn renders the same bytes as one that does not
            dialogues.append((row["row_idx"], r, [{"role": m["role"], "content": m["content"].strip()} for m in msgs]))
    n0 = len(docs)
    di = 0
    while len(docs) < n0 + N_CHAT:
        msgs, used, src, chars = [], [], set(), 0
        while di < len(dialogues) and chars < CHAT_CHARS:
            idx, r, dm = dialogues[di]
            msgs += dm
            chars += sum(len(m["content"]) for m in dm)
            used.append(idx)
            src.add((r[0], r[1], r[2]))
            di += 1
        if chars < CHAT_CHARS:
            sys.exit(f"the UltraChat rows ran out after {len(docs) - n0} chat candidates")
        write({"id": f"d{len(docs) + 1:02d}", "kind": "chat",
               "label": f"UltraChat-200k test_sft rows {used[0]}-{used[-1]} ({len(msgs)} messages before the driver's cut)",
               "sources": [{"url": u, "sha256": h, "file": os.path.relpath(os.path.join(uc, fn), kdir)} for fn, u, h in sorted(src)],
               "rows": used, "messages": msgs})

    for s in skipped:
        print("skipped:", s, flush=True)
    manifest = {"n": N, "chars": {"code": CODE_CHARS, "prose": PROSE_CHARS, "chat": CHAT_CHARS},
                "want": {"code": N_CODE, "prose": 10, "chat": 10}, "forbidden": list(FORBIDDEN),
                "code_spec": {"path": spec_path, "sha256": sha256_file(spec_path)},
                "prompts30_manifest": {"path": p30_manifest, "sha256": sha256_file(p30_manifest), "lib_dir": lib_dir},
                "reference_notes": src_notes, "refs": refs, "skipped": skipped, "docs": docs}
    with open(os.path.join(kdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    kinds = {k: [d for d in docs if d["kind"] == k] for k in ("code", "prose", "chat")}
    print("manifest: " + ", ".join(f"{len(v)} {k} ({min(d['chars'] for d in v)}-{max(d['chars'] for d in v)} characters)" for k, v in kinds.items())
          + f"; {len(refs)} reference texts", flush=True)


if __name__ == "__main__":
    main()
