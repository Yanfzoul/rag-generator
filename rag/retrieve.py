#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid retriever (generic):
- Semantic: FAISS over BGE (or configured) embeddings
- Lexical: optional BM25 over meta['text']
- Identifier boost: exact symbol hits from symbols.parquet
- Optional reranker: CrossEncoder (configurable)
"""

import argparse, re, yaml, numpy as np, pandas as pd, torch
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# optional: pip install rank_bm25
try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

import os
CFG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")
CFG = yaml.safe_load(open(CFG_PATH, "r"))
INDEX_ROOT   = Path(CFG["paths"]["index_root"])
INDEX_NAME   = (CFG.get("index", {}) or {}).get("name", "index")
FAISS_PATH   = INDEX_ROOT / f"{INDEX_NAME}.faiss"
META_PATH    = INDEX_ROOT / f"{INDEX_NAME}.meta.parquet"
SYMS_PATH    = INDEX_ROOT / "symbols.parquet"
EMB_MODEL_ID = CFG["embedding"]["model"]
EMB_NORM     = bool(CFG["embedding"].get("normalize_embeddings", True))
if not EMB_NORM:
    print("[warn] normalize_embeddings=False with IP index; semantic scores may be skewed.")

ap = argparse.ArgumentParser()
ap.add_argument("--q", required=True)
ap.add_argument("--gpu", action="store_true", help="Use CUDA for embedding and reranker (CPU default)")
ap.add_argument("--mps", action="store_true", help="Use Apple MPS (Metal) for embedding if available")
ap.add_argument("--initial_k", type=int, default=24)
ap.add_argument("--final_k", type=int, default=5)
ap.add_argument("--use_reranker", action="store_true")
ap.add_argument("--reranker_id", default="BAAI/bge-reranker-base")
ap.add_argument("--show", action="store_true", help="print results")
args = ap.parse_args()

# --- load
print(f"[retrieve] loading meta: {META_PATH}")
meta = pd.read_parquet(META_PATH)
print(f"[retrieve] meta rows: {len(meta)}")
print(f"[retrieve] loading FAISS: {FAISS_PATH}")
index = faiss.read_index(str(FAISS_PATH))
embedder = SentenceTransformer(EMB_MODEL_ID)
HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
USE_CUDA = args.gpu and torch.cuda.is_available()
USE_MPS = args.mps and HAS_MPS
if args.gpu and args.mps:
    raise SystemExit("[error] --gpu and --mps cannot be used together.")
if args.gpu and not torch.cuda.is_available():
    raise SystemExit("[error] --gpu set but CUDA not available. Install CUDA-enabled PyTorch and proper drivers.")
if args.mps and not HAS_MPS:
    raise SystemExit("[error] --mps set but PyTorch MPS backend is unavailable.")
if USE_CUDA:
    try:
        embedder = embedder.to("cuda")
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        # Try to half the underlying transformer
        first = getattr(embedder, "_first_module", None)
        if callable(first):
            mod = first()
            auto_model = getattr(mod, "auto_model", None)
            if auto_model is not None:
                auto_model.half()
        print(f"[retrieve] embedder on CUDA (FP16): {torch.cuda.get_device_name(0)}")
    except Exception:
        USE_CUDA = False
elif USE_MPS:
    try:
        embedder = embedder.to("mps")
        print("[retrieve] embedder on MPS (Apple Metal)")
    except Exception:
        USE_MPS = False

# --- BM25 (optional, configurable)
bm25_cfg = (CFG.get("retrieval", {}) or {}).get("bm25", {}) or {}
bm25_enabled = bool(bm25_cfg.get("enabled", True))
if bm25_enabled and HAVE_BM25:
    bm25 = BM25Okapi([t.split() for t in meta["text"].astype(str)])
    print("[retrieve] BM25: enabled")
else:
    bm25 = None
    print("[retrieve] BM25: disabled")

# --- symbols (optional)
symbols = None
if SYMS_PATH.exists():
    symbols = pd.read_parquet(SYMS_PATH)
    # quick hash for exact lookup
    syms_set = set(symbols["symbol"].unique())
    print(f"[retrieve] symbols: {len(syms_set)} unique")
else:
    syms_set = set()

RE_SYM = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b')

def extract_symbols(q: str):
    toks = [t for t in RE_SYM.findall(q)]
    prefixes = (CFG.get("retrieval", {}) or {}).get("symbol_prefixes", [])
    if prefixes:
        keep = [t for t in toks if any(t.startswith(pref) for pref in prefixes)]
        return keep or toks[:6]
    # default: return a small set of identifiers if present
    return toks[:6]

def semantic_search(q: str, k: int):
    if USE_CUDA:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            qvec = embedder.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
        qv = qvec.detach().float().cpu().numpy()
    elif USE_MPS:
        qvec = embedder.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
        qv = qvec.detach().cpu().numpy()
    else:
        qv = embedder.encode([q], normalize_embeddings=EMB_NORM)
    scores, ids = index.search(np.asarray(qv, dtype="float32"), k)
    rows = meta.iloc[ids[0]].copy()
    rows["sem_score"] = scores[0]
    return rows

def lexical_search(q: str, k: int):
    if not bm25:
        return pd.DataFrame()
    scores = bm25.get_scores(q.split())
    rows = meta.copy()
    rows["lex_score"] = scores
    return rows.nlargest(k, "lex_score")

def identifier_hits(q: str):
    """Return meta rows that overlap symbol locations (strong boost)."""
    if symbols is None:
        return pd.DataFrame()
    qsyms = [s for s in extract_symbols(q) if s in syms_set]
    if not qsyms:
        return pd.DataFrame()
    hit = symbols[symbols["symbol"].isin(qsyms)].copy()
    # join with meta on path and overlapping line range when available
    m = meta.copy()
    # ensure numeric
    for col in ("start_line","end_line"):
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")
    # exact path match then overlap
    out_rows = []
    for _, r in hit.iterrows():
        cand = m[m["path"] == r["path"]]
        if cand.empty:
            continue
        sl, el = int(r["start_line"]), int(r["end_line"])
        # overlap: (chunk.start <= sym.end) and (chunk.end >= sym.start)
        c = cand[(pd.notna(cand["start_line"])) & (pd.notna(cand["end_line"]))]
        c = c[(c["start_line"].astype(int) <= el) & (c["end_line"].astype(int) >= sl)]
        if c.empty:
            # fallback: any chunk in same file
            c = cand.head(3)
        if not c.empty:
            c = c.copy()
            c["id_score"] = 1.0
            out_rows.append(c)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True).drop_duplicates("id")

def fuse(q: str, k_init: int, k_final: int):
    sem = semantic_search(q, k_init)
    lex = lexical_search(q, k_init)
    ide = identifier_hits(q)

    # merge
    all_ = pd.concat([sem, lex, ide], ignore_index=True)
    all_ = all_.drop_duplicates("id")
    print(f"[retrieve] pool sizes: sem={len(sem)}, lex={len(lex)}, id={len(ide)} -> merged={len(all_)}")

    # normalize score components
    for col in ("sem_score","lex_score","id_score"):
        if col not in all_.columns:
            all_[col] = 0.0
        else:
            s = all_[col].astype(float)
            if s.max() > 0:
                all_[col] = (s - s.min()) / (s.max() - s.min() + 1e-9)
            else:
                all_[col] = 0.0

    # weighted fusion (configurable)
    fw = (CFG.get("retrieval", {}) or {}).get("fusion_weights", {}) or {}
    w_sem = float(fw.get("semantic", 0.55))
    w_lex = float(fw.get("lexical", 0.35))
    w_id  = float(fw.get("identifier", 0.10))
    all_["fused"] = w_sem*all_["sem_score"] + w_lex*all_["lex_score"] + w_id*all_["id_score"]
    # Optional per-source weights from config
    cfg_weights = (CFG.get("retrieval", {}) or {}).get("source_weights", {})
    sw = all_.get("source_type").fillna("").map(cfg_weights).fillna(1.00)
    all_["fused"] = all_["fused"] * sw
    top = all_.nlargest(max(k_init, k_final), "fused")
    return top

def rerank(q: str, df: pd.DataFrame, k_final: int, model_id: str):
    if df.empty or len(df) <= k_final:
        return df.head(k_final)

    # load once (prefer accelerator)
    device = "cuda" if USE_CUDA else ("mps" if USE_MPS else "cpu")
    rr = CrossEncoder(model_id, device=device)
    # truncate texts to ~480 tokens equivalent (approx via chars)
    pairs = [(q, str(t)[:3000]) for t in df["text"].astype(str)]
    scores = rr.predict(pairs)
    df = df.copy()
    df["rr_score"] = scores
    return df.nlargest(k_final, "rr_score")

def main():
    q = args.q.strip()
    pool = fuse(q, args.initial_k, args.final_k)
    if args.use_reranker:
        top = rerank(q, pool, args.final_k, args.reranker_id)
    else:
        top = pool.nlargest(args.final_k, "fused")
    print(f"[retrieve] returning top {args.final_k} rows")

    if args.show:
        for _, r in top.iterrows():
            src = r.get("source_type","")
            path = r.get("path","") or r.get("url","")
            sl, el = r.get("start_line"), r.get("end_line")
            loc = ""
            if pd.notna(sl) and pd.notna(el):
                loc = f" L{int(sl)}-{int(el)}"
            print(f"[{src:8}] {path}{loc}  | fused={r.get('fused',0):.3f} rr={r.get('rr_score',0):.3f}")
    else:
        # print ids for piping
        print("\n".join(top["id"].tolist()))

if __name__ == "__main__":
    main()
