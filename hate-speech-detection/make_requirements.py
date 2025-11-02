#!/usr/bin/env python3
"""
make_requirements.py

Scan a directory tree for Python files, extract imported top-level modules,
filter out likely stdlib modules, and produce a requirements.txt.

Usage:
    python make_requirements.py            # scans current directory, writes requirements.txt
    python make_requirements.py -p src     # scan folder 'src'
    python make_requirements.py -o reqs.txt --with-versions

Notes:
 - "with-versions" will attempt to map top-level module names to installed distributions
   (using importlib.metadata) and pin versions like `requests==2.31.0`. If a mapping
   or version can't be found, the package will be written without a version.
 - The script tries to detect standard-library modules using builtin module names
   and the stdlib path for the current Python interpreter. Detection is heuristic.
"""
from __future__ import annotations
import argparse
import ast
import os
import sys
import sysconfig
import importlib.util
from pathlib import Path
from typing import Set, Dict

# optional importlib.metadata compatibility for older Python
try:
    import importlib.metadata as importlib_metadata
except Exception:
    try:
        import importlib_metadata  # type: ignore
    except Exception:
        importlib_metadata = None  # type: ignore

BUILTINS = set(sys.builtin_module_names)

def extract_imports_from_file(path: Path) -> Set[str]:
    """Return top-level module names imported in a .py file."""
    mods: Set[str] = set()
    try:
        src = path.read_text(encoding="utf-8")
        node = ast.parse(src, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        # skip files with syntax errors or unreadable encoding
        return mods

    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                name = alias.name.split('.')[0]
                if name:
                    mods.add(name)
        elif isinstance(n, ast.ImportFrom):
            if n.level and n.level > 0:
                # relative import like from . import x -> skip
                continue
            if n.module:
                name = n.module.split('.')[0]
                if name:
                    mods.add(name)
    return mods

def find_python_files(base: Path) -> Set[Path]:
    """Recursively find .py files (skip virtualenv folders)."""
    py_files = set()
    skip_dirs = {'venv', '.venv', 'env', '.env', 'envs',  '__pycache__', '.git', 'node_modules'}
    for root, dirs, files in os.walk(base):
        # mutate dirs in-place to skip heavy directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.pytest_cache')]
        for f in files:
            if f.endswith('.py'):
                py_files.add(Path(root) / f)
    return py_files

def likely_stdlib(module_name: str, stdlib_dir: str) -> bool:
    """
    Heuristic to determine if module_name is part of the python stdlib.
    Uses builtin module names, and find_spec origin path compared to stdlib_dir.
    """
    if module_name in BUILTINS:
        return True
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        spec = None
    if not spec or not getattr(spec, "origin", None):
        # could be a namespace or missing package; assume not stdlib
        return False
    origin = spec.origin  # may be a path like '/usr/lib/python3.10/...'
    # normalize strings and compare
    try:
        origin_path = str(Path(origin).resolve())
        stdlib_path = str(Path(stdlib_dir).resolve())
        if origin_path.startswith(stdlib_path):
            return True
    except Exception:
        pass
    # also treat .so/.pyd living in lib-dynload inside stdlib as stdlib
    if 'site-packages' in origin or 'dist-packages' in origin:
        return False
    return False

def map_module_to_distribution(modules: Set[str]) -> Dict[str, str]:
    """
    Try to map top-level module names to distribution names using importlib.metadata.
    Returns dict module -> distribution (first match). Requires importlib.metadata support.
    """
    mapping: Dict[str, str] = {}
    if importlib_metadata is None:
        return mapping

    # packages_distributions maps top-level package name -> list of distributions that provide it
    try:
        pkg_map = importlib_metadata.packages_distributions()
    except Exception:
        pkg_map = {}  # not available on older importlib.metadata
    for m in modules:
        dists = pkg_map.get(m)
        if dists:
            mapping[m] = dists[0]  # choose the first distribution name
    return mapping

def get_distribution_version(dist_name: str) -> str | None:
    """Return installed version for a distribution name or None if not found."""
    if importlib_metadata is None:
        return None
    try:
        return importlib_metadata.version(dist_name)
    except Exception:
        return None

def generate_requirements(modules: Set[str], with_versions: bool = False) -> list[str]:
    """Return lines for requirements.txt"""
    lines: list[str] = []
    stdlib_dir = sysconfig.get_paths().get("stdlib", "")
    non_std = sorted([m for m in modules if not likely_stdlib(m, stdlib_dir)])
    if with_versions and importlib_metadata is not None:
        mapping = map_module_to_distribution(set(non_std))
        for m in non_std:
            dist = mapping.get(m)
            if dist:
                ver = get_distribution_version(dist)
                if ver:
                    lines.append(f"{dist}=={ver}")
                else:
                    lines.append(f"{dist}")  # distribution found but version unknown
            else:
                # fallback: write module name without version
                lines.append(m)
    else:
        lines = list(non_std)
    # dedupe and keep stable order
    seen = set()
    out = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out

def main():
    p = argparse.ArgumentParser(description="Scan Python files and generate requirements.txt")
    p.add_argument("-p", "--path", default=".", help="Path to project root to scan")
    p.add_argument("-o", "--output", default="requirements.txt", help="Output requirements file")
    p.add_argument("--with-versions", action="store_true",
                   help="Attempt to map imports to installed distributions and pin versions")
    args = p.parse_args()

    base = Path(args.path).resolve()
    if not base.exists():
        print(f"error: path {base} does not exist", file=sys.stderr)
        sys.exit(2)

    print(f"scanning python files under {base} ...")
    files = find_python_files(base)
    print(f"found {len(files)} .py files")

    all_mods: Set[str] = set()
    for f in files:
        mods = extract_imports_from_file(f)
        all_mods.update(mods)
    print(f"extracted {len(all_mods)} unique top-level imports")

    stdlib_dir = sysconfig.get_paths().get("stdlib", "")
    stdlib_matches = [m for m in sorted(all_mods) if likely_stdlib(m, stdlib_dir)]
    third_party = [m for m in sorted(all_mods) if not likely_stdlib(m, stdlib_dir)]

    print(f"likely stdlib modules (ignored): {', '.join(stdlib_matches)[:200] or '(none)'}")
    print(f"likely third-party modules (candidates): {', '.join(third_party)[:400] or '(none)'}")

    req_lines = generate_requirements(all_mods, with_versions=args.with_versions)

    out_path = Path(args.output)
    out_path.write_text("\n".join(req_lines) + ("\n" if req_lines else ""), encoding="utf-8")
    print(f"wrote {len(req_lines)} lines to {out_path.resolve()}")

if __name__ == "__main__":
    main()
