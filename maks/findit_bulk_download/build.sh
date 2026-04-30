#!/usr/bin/env sh
set -e
ROOT="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/out"
javac -d "$ROOT/out" "$ROOT/src/BulkDownload.java"
echo "Built: $ROOT/out/BulkDownload.class"
echo "Run (Java): java -cp \"$ROOT/out\" BulkDownload ../findit_urls/dtu_theses_all_urls_deduped.txt -o ./downloads -c ../cookies.txt"
echo "Run (Python, no JDK): uv run python \"$ROOT/bulk_download.py\" ../findit_urls/dtu_theses_all_urls_deduped.txt -o ./downloads -c ../cookies.txt"
