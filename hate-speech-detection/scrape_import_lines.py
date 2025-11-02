#!/usr/bin/env python3
"""
scrape_import_lines.py

Recursively scan a directory for .py files and extract the *exact source lines*
that represent imports (handles single-line and multi-line imports). Outputs a
deduplicated list (preserves first-seen order) with filename and line numbers.

Usage:
    python scrape_import_lines.py                # scan current dir -> import_lines.txt
    python scrape_import_lines.py -p src -o imports.txt
    python scrape_import_lines.py --show-files    # print results to stdout too

Output format (default file):
    path/to/file.py:12-13: from package import (
        a,
        b,
    )

Notes:
 - The script uses the AST nodes' lineno and end_lineno when available (Python 3.8+).
 - For files that can't be parsed (syntax error / encoding), it will skip and print a warning.
"""
from __future__ import annotations
import argparse
import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple, Set

def find_python_files(base: Path) -> List[Path]:
    skip_dirs = {'venv', '.venv', 'env', '.env', 'envs', '__pycache__', '.git', 'node_modules'}
    py_files: List[Path] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.pytest_cache')]
        for f in files:
            if f.endswith('.py'):
                py_files.append(Path(root) / f)
    return py_files

def extract_import_lines(path: Path) -> List[Tuple[str,int,int]]:
    """
    Return list of tuples (import_text, start_lineno, end_lineno) found in the file.
    import_text contains the source block exactly as in file (with original newlines preserved).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"warning: can't read {path}: {e}", file=sys.stderr)
        return []
    try:
        node = ast.parse(text, filename=str(path))
    except Exception as e:
        print(f"warning: can't parse {path}: {e}", file=sys.stderr)
        return []

    lines = text.splitlines(keepends=True)
    results: List[Tuple[str,int,int]] = []

    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            # lineno is 1-based. end_lineno exists on Python 3.8+; fall back to lineno if not present.
            start = getattr(n, "lineno", None)
            end = getattr(n, "end_lineno", None) or start
            if start is None:
                continue
            # clip to file bounds
            start_idx = max(0, start - 1)
            end_idx = min(len(lines), end)
            block = "".join(lines[start_idx:end_idx])
            # normalize trailing newline to single '\n' for file output consistency
            if not block.endswith("\n"):
                block = block + "\n"
            results.append((block, start, end))
    return results

def main():
    p = argparse.ArgumentParser(description="Scrape import lines from Python files")
    p.add_argument("-p", "--path", default=".", help="Path to scan (default: current dir)")
    p.add_argument("-o", "--output", default="import_lines.txt", help="Output file")
    p.add_argument("--show-files", action="store_true", help="Also print results to stdout")
    args = p.parse_args()

    base = Path(args.path).resolve()
    if not base.exists():
        print(f"error: path {base} does not exist", file=sys.stderr)
        sys.exit(2)

    py_files = find_python_files(base)
    print(f"scanning {len(py_files)} python files under {base} ...")

    seen: Set[Tuple[str, str]] = set()  # (file_path, import_block) to dedupe exact blocks per file
    ordered_entries: List[Tuple[Path, int, int, str]] = []  # (path, start, end, block)

    for f in py_files:
        extracted = extract_import_lines(f)
        for block, start, end in extracted:
            key = (str(f), block)
            if key in seen:
                continue
            seen.add(key)
            ordered_entries.append((f, start, end, block))

    # write to output
    out_lines: List[str] = []
    for path, start, end, block in ordered_entries:
        header = f"{path}:{start}-{end}:\n"
        out_lines.append(header)
        out_lines.append(block)
        out_lines.append("\n")  # spacer between entries

    out_path = Path(args.output)
    try:
        out_path.write_text("".join(out_lines), encoding="utf-8")
        print(f"wrote {len(ordered_entries)} import blocks to {out_path.resolve()}")
    except Exception as e:
        print(f"error: couldn't write output file {out_path}: {e}", file=sys.stderr)
        sys.exit(3)

    if args.show_files:
        # also print in a concise form to stdout
        for path, start, end, block in ordered_entries:
            print(f"{path}:{start}-{end} -> {block.strip().splitlines()[0]}")

if __name__ == "__main__":
    main()
