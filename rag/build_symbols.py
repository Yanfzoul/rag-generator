#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a lightweight symbol index from headers (e.g., inc/*.h).
Outputs: store/symbols.parquet with columns:
  symbol, kind, path, start_line, end_line, snippet
Configure `symbols.base_dirs` and `symbols.header_globs` in config.yaml.
"""

import os, re, yaml
from pathlib import Path
import pandas as pd

CFG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")
CFG = yaml.safe_load(open(CFG_PATH, "r"))
INDEX_ROOT = Path(CFG["paths"]["index_root"])
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

# Configurable scan roots and header patterns
SYM_CFG = CFG.get("symbols", {}) or {}
BASE_DIRS = [Path(p) for p in (SYM_CFG.get("base_dirs") or [])]
HDR_GLOBS = SYM_CFG.get("header_globs") or ["**/inc/*.h", "**/include/*.h"]

# crude regexes (good enough for C headers)
RE_DEF      = re.compile(r'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\b')
RE_FUNC     = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_*\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*;')
RE_ENUM_BEG = re.compile(r'^\s*typedef\s+enum\b|^\s*enum\b')
RE_ENUM_VAL = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=|,|}|$)')

def scan_file(p: Path):
    rows = []
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return rows
    n = len(lines)

    # simple pass: macros + function prototypes
    for i, line in enumerate(lines, start=1):
        m = RE_DEF.match(line)
        if m:
            sym = m.group(1)
            rows.append(dict(symbol=sym, kind="macro", path=str(p), start_line=i, end_line=i, snippet=line.strip()))
            continue
        m = RE_FUNC.match(line)
        if m:
            sym = m.group(1)
            # capture a small window around the prototype
            j0 = max(1, i-3); j1 = min(n, i+3)
            snippet = "\n".join(lines[j0-1:j1])
            rows.append(dict(symbol=sym, kind="function", path=str(p), start_line=i, end_line=i, snippet=snippet.strip()))

    # enum blocks: collect values
    in_enum = False; enum_start = 0
    for i, line in enumerate(lines, start=1):
        if not in_enum and RE_ENUM_BEG.search(line):
            in_enum = True; enum_start = i
            continue
        if in_enum:
            if "}" in line:
                in_enum = False
                continue
            m = RE_ENUM_VAL.match(line)
            if m:
                sym = m.group(1)
                rows.append(dict(symbol=sym, kind="enum", path=str(p), start_line=i, end_line=i, snippet=line.strip()))

    return rows

def main():
    files = []
    for base in (BASE_DIRS or []):
        for g in HDR_GLOBS:
            files += list(base.glob(g))
    rows = []
    for p in files:
        rows.extend(scan_file(p))
    if not rows:
        print("No symbols found; check inc/ path in config.")
        return
    df = pd.DataFrame(rows).drop_duplicates(["symbol","path","start_line"])
    out = INDEX_ROOT / "symbols.parquet"
    df.to_parquet(out, index=False)
    print("OK:", out, f"({len(df)} symbols)")

if __name__ == "__main__":
    main()
