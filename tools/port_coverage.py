#!/usr/bin/env python3
"""Report rough upstream tinygrad symbol coverage in the RSS port.

This is intentionally conservative: it is an inventory/direction tool, not proof
that a symbol is semantically complete.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


RSS_FN_RE = re.compile(r"^\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def py_symbols(path: Path) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text())
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(("function", node.name))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((f"{node.name}.method", item.name))
    return out


def rss_symbols(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*.rss"):
        names.update(RSS_FN_RE.findall(path.read_text(errors="ignore")))
    return names


def candidates(kind: str, name: str) -> list[str]:
    base = name.strip("_")
    if not base:
        return [name]
    out = [name, base]
    if "Tensor.method" in kind or kind.endswith(".method"):
        out += [
            f"tensor_{base}",
            f"tensor_{base}_axis",
            f"tensor_{base}_all",
            f"tensor_{base}_axes",
            f"tensor_{base}_keepdim",
        ]
    if "UOp.method" in kind:
        out += [f"uop_{base}"]
    return out


def covered(kind: str, name: str, rss: set[str]) -> bool:
    for cand in candidates(kind, name):
        if cand in rss:
            return True
    return False


def report_module(upstream_file: Path, rss: set[str], limit: int) -> tuple[int, int]:
    syms = py_symbols(upstream_file)
    hits = [(kind, name) for kind, name in syms if covered(kind, name, rss)]
    misses = [(kind, name) for kind, name in syms if not covered(kind, name, rss)]
    pct = 100.0 * len(hits) / len(syms) if syms else 100.0
    rel = upstream_file
    print(f"{rel}: {len(hits)}/{len(syms)} rough symbols covered ({pct:.1f}%)")
    if misses:
        shown = ", ".join(name for _, name in misses[:limit])
        suffix = "" if len(misses) <= limit else f", ... +{len(misses)-limit}"
        print(f"  missing: {shown}{suffix}")
    return len(hits), len(syms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, default=Path("../tinygrad/tinygrad"))
    ap.add_argument("--rss", type=Path, default=Path("tinygrad-rss/src"))
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("modules", nargs="*", default=["tensor.py", "mixin/__init__.py", "uop/ops.py"])
    args = ap.parse_args()

    rss = rss_symbols(args.rss)
    total_hits = 0
    total_syms = 0
    for module in args.modules:
        path = args.upstream / module
        if not path.exists():
            print(f"{path}: missing upstream file")
            continue
        hits, syms = report_module(path, rss, args.limit)
        total_hits += hits
        total_syms += syms
    pct = 100.0 * total_hits / total_syms if total_syms else 100.0
    print(f"total: {total_hits}/{total_syms} rough symbols covered ({pct:.1f}%)")
    print("note: rough coverage only checks symbol presence, not exact tinygrad semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
