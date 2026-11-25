#!/bin/bash
# dsv41_long_kld_fetch.sh <data dir>: the long-context KLD test's public downloads (tests/dsv41_long_kld.py), each source
# in its own new directory with SOURCES.tsv (file, url, sha256, bytes, fetched_utc) beside its files. Network and disk
# only: GLM-5.3-Flash's fetch_kld.sh with a fourth UltraChat page. Every downloaded file is read later only as text or
# JSON, by python3 -I with its path as an argument.
#   <data dir>/dl/pg19:      the PG-19 test split's listing (the bucket's JSON API, as the HF pg19 loader lists it) and
#                            the first 14 books by Gutenberg id (10 used, 4 spares), each checked against the listing's
#                            md5 and size.
#   <data dir>/dl/ultrachat: UltraChat-200k test_sft rows 0-399 from the HF datasets server (four pages of 100).
set -eu
K=${1:?data dir}
for d in "$K/dl/pg19" "$K/dl/ultrachat"; do [ ! -e "$d" ] || { echo "exists: $d"; exit 1; }; done
mkdir -p "$K/dl/pg19" "$K/dl/ultrachat"
P=/usr/bin/python3
get() { # get <dir> <file> <url>
  curl -sS -f -m 300 --retry 3 --retry-delay 5 -o "$1/$2" "$3"
  printf '%s\t%s\t%s\t%s\t%s\n' "$2" "$3" "$(shasum -a 256 "$1/$2" | cut -d' ' -f1)" "$(stat -f %z "$1/$2")" "$(date -u +%FT%TZ)" >> "$1/SOURCES.tsv"
}
printf 'file\turl\tsha256\tbytes\tfetched_utc\n' > "$K/dl/pg19/SOURCES.tsv"
printf 'file\turl\tsha256\tbytes\tfetched_utc\n' > "$K/dl/ultrachat/SOURCES.tsv"
get "$K/dl/pg19" listing.json "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?prefix=test/&maxResults=1000"
# the first 14 test books by id: name, md5 and size from the listing (read as JSON)
$P -I -c '
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
assert "nextPageToken" not in body, "the listing has more pages"
items = [i for i in body.get("items", []) if i["name"].startswith("test/") and i["name"].endswith(".txt") and i["name"][5:-4].isdigit()]
items.sort(key=lambda i: int(i["name"][5:-4]))
print(len(items), "test books listed", file=sys.stderr)
for i in items[:14]: print(i["name"][5:-4], i["md5Hash"], i["size"])
' "$K/dl/pg19/listing.json" > "$K/dl/pg19/picked.txt"
while read -r id md5 size; do
  case "$id" in *[!0-9]*|"") echo "pg19: bad id '$id' in the listing"; exit 1;; esac
  get "$K/dl/pg19" "$id.txt" "https://storage.googleapis.com/deepmind-gutenberg/test/$id.txt"
  got=$($P -I -c 'import base64, hashlib, sys; print(base64.b64encode(hashlib.md5(open(sys.argv[1], "rb").read()).digest()).decode())' "$K/dl/pg19/$id.txt")
  [ "$got" = "$md5" ] && [ "$(stat -f %z "$K/dl/pg19/$id.txt")" = "$size" ] || { echo "pg19 $id: md5 $got / size $(stat -f %z "$K/dl/pg19/$id.txt") are not the listing's $md5 / $size"; exit 1; }
done < "$K/dl/pg19/picked.txt"
for off in 0 100 200 300; do
  get "$K/dl/ultrachat" "rows-$off.json" "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2Fultrachat_200k&config=default&split=test_sft&offset=$off&length=100"
done
echo "fetched: pg19 $(($(wc -l < "$K/dl/pg19/SOURCES.tsv") - 1)) files, ultrachat $(($(wc -l < "$K/dl/ultrachat/SOURCES.tsv") - 1)) files; $(du -sh "$K/dl" | cut -f1)"
