#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Doctor: quick preflight for environment and config.

Checks:
- Python version
- Required packages: pandas, numpy, faiss, sentence_transformers, pyarrow|fastparquet
- Config path (RAG_CONFIG or config.yaml), parseability
- paths.index_root exists/creatable (no write performed)
- sources: list exists; each path exists
- Embedding model id presence

Usage:
  python tools/doctor.py   # optionally set RAG_CONFIG
"""

from __future__ import annotations
import os, sys, importlib
from pathlib import Path
import traceback

def have(mod: str) -> tuple[bool, str]:
    try:
        importlib.import_module(mod)
        return True, "ok"
    except Exception as e:
        return False, str(e)

def main():
    ok = True
    print("== RAG Doctor ==")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    # Packages
    for m in ["pandas", "numpy", "faiss", "sentence_transformers"]:
        h, msg = have(m)
        print(f"pkg {m:24}: {'OK' if h else 'MISSING'} {'' if h else msg}")
        ok &= h
    # Parquet engine
    h1, _ = have("pyarrow")
    h2, _ = have("fastparquet")
    print(f"pkg parquet (pyarrow|fastparquet): {'OK' if (h1 or h2) else 'MISSING'}")
    ok &= (h1 or h2)

    # Torch / CUDA / Tensor Core checks
    ht, _ = have("torch")
    if ht:
        import torch  # type: ignore
        print(f"torch: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print("CUDA available:", cuda_ok)
        if cuda_ok:
            try:
                dev_name = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
            except Exception:
                dev_name = "<unknown>"
                cap = (0, 0)
            print("CUDA device:", dev_name)
            print("Compute capability:", cap)
            # Tensor Core FP16 generally available on Volta/Turing/Ampere+ (cc >= 7.0)
            has_tensor_cores = (cap[0] >= 7)
            print("Tensor Cores (FP16):", bool(has_tensor_cores))
            # TF32 acceleration is Ampere+ (cc >= 8.0)
            has_tf32 = (cap[0] >= 8)
            print("TF32 matmul acceleration:", bool(has_tf32))
        else:
            print("NOTE: CUDA not detected. GPU runs require a CUDA-enabled PyTorch wheel and NVIDIA drivers.")

    # Config
    cfg_path = os.environ.get("RAG_CONFIG", "config.yaml")
    print(f"Config: {cfg_path}")
    try:
        import yaml
        cfg = yaml.safe_load(open(cfg_path, "r"))
    except Exception as e:
        print("ERROR: cannot read config:", e)
        ok = False
        cfg = {}

    paths = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    idx_root = Path(paths.get("index_root" or "")) if paths else None
    if idx_root:
        print(f"index_root: {idx_root}")
        try:
            idx_root.mkdir(parents=True, exist_ok=True)
            print("index_root exists/creatable: OK")
        except Exception as e:
            print("index_root exists/creatable: FAIL", e)
            ok = False

    # Sources
    srcs = cfg.get("sources") if isinstance(cfg, dict) else None
    if not isinstance(srcs, list) or not srcs:
        print("WARNING: config has no sources defined; indexer will do nothing.")
    else:
        missing = []
        for s in srcs:
            p = Path(str(s.get("path", "")))
            if not p.exists():
                missing.append(str(p))
        if missing:
            print("WARNING: some source paths do not exist:")
            for m in missing:
                print(" -", m)

    # Embedding model id presence
    emb = (cfg.get("embedding") or {}).get("model") if isinstance(cfg, dict) else None
    if not emb:
        print("ERROR: embedding.model not set in config")
        ok = False
    else:
        print("embedding.model:", emb)

    print("Result:", "OK" if ok else "Issues found")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
