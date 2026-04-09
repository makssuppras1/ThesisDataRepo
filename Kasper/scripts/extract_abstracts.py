#!/usr/bin/env python3
import os
import re
from pathlib import Path

DOWNLOADS_DIR = Path('downloads')
OUTPUT_DIR = Path('abstracts')

def extract_by_heading(text):
    # Accept 'Abstract' even if preceded by control chars or page breaks
    control_re = re.compile(r'(?mi)^[\x00-\x1F\s\d\-\u00A0]*abstract\s*:??\s*$', re.MULTILINE)
    m = control_re.search(text)
    if m:
        # no hash-level available, stop at next ATX or setext heading
        start_pos = m.end()
        remainder = text[start_pos:]
        # find next ATX or setext heading position
        next_heading = re.search(r'(?m)^(?:#{1,6}\s)|(^.+\n[-=]{3,}\s*$)', remainder)
        end_pos = next_heading.start() if next_heading else len(remainder)
        snippet = remainder[:end_pos]
        out_lines = snippet.splitlines()
        while out_lines and out_lines[0].strip() == '':
            out_lines.pop(0)
        while out_lines and out_lines[-1].strip() == '':
            out_lines.pop()
        return '\n'.join(out_lines).strip() or None

    # Prefer ATX headings like "## Abstract" (case-insensitive)
    atx_re = re.compile(r'(?mi)^(#{1,6})\\s*(abstract)\\s*:?\\s*$', re.MULTILINE)
    m = atx_re.search(text)
    if m:
        level = len(m.group(1))
        start_pos = m.end()
        lines = text[start_pos:].splitlines()
        out_lines = []
        stop_re = re.compile(r'^\s#{1,%d}\s' % level)
        for line in lines:
            if stop_re.match(line):
                break
            out_lines.append(line)
        while out_lines and out_lines[0].strip() == '':
            out_lines.pop(0)
        while out_lines and out_lines[-1].strip() == '':
            out_lines.pop()
        return '\n'.join(out_lines).strip() or None

    # Support setext-style headings:
    # Abstract\n------
    setext_re = re.compile(r'(?mi)^(abstract)\s*\r?\n([=-]{3,})', re.MULTILINE)
    m = setext_re.search(text)
    if m:
        start_pos = m.end()
        lines = text[start_pos:].splitlines()
        out_lines = []
        stop_re = re.compile(r'^\s#{1,6}\s')
        for line in lines:
            if stop_re.match(line):
                break
            out_lines.append(line)
        while out_lines and out_lines[0].strip() == '':
            out_lines.pop(0)
        while out_lines and out_lines[-1].strip() == '':
            out_lines.pop()
        return '\n'.join(out_lines).strip() or None

    return None

def extract_abstract(text):
    # Now only return content from an explicit "Abstract" section.
    return extract_by_heading(text)

def process_file(path: Path, out_dir: Path):
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return False, f'read error: {e}'
    abstract = extract_abstract(text)
    if not abstract:
        return False, 'no abstract found'
    out_name = out_dir / (path.stem + '_abstract.txt')
    out_name.write_text(abstract, encoding='utf-8')
    return True, str(out_name)

def main():
    if not DOWNLOADS_DIR.exists():
        print('downloads/ directory not found')
        return 2
    OUTPUT_DIR.mkdir(exist_ok=True)
    md_files = []
    for root, dirs, files in os.walk(DOWNLOADS_DIR):
        for f in files:
            if f.lower().endswith('.md'):
                md_files.append(Path(root) / f)
    md_files.sort()
    total = len(md_files)
    print(f'Found {total} markdown files')
    succeeded = 0
    failed = 0
    failed_list = []
    created = []
    for p in md_files:
        ok, info = process_file(p, OUTPUT_DIR)
        if ok:
            succeeded += 1
            created.append(info)
        else:
            failed += 1
            failed_list.append((str(p), info))
    print(f'Abstracts written: {succeeded}')
    if failed:
        print(f'Files without abstracts: {failed}')
    if created:
        print('Sample outputs:')
        for c in created[:10]:
            print(' -', c)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
