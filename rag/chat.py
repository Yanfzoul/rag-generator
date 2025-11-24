#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic RAG Chat (hybrid retrieval + sessions + continuation)
- Hybrid retrieval: FAISS (embedding) + optional BM25 + symbol boost (+ optional CrossEncoder reranker)
- Structured context with citations and headings
- Token-safe prompt assembly
- Session memory: save/load turns in sessions/<id>.jsonl
- Reuse last context if desired; include last K Q/A summaries
- Continuation-aware generation to avoid truncated answers
"""

import os, re, json, argparse, textwrap, yaml
from pathlib import Path
import numpy as np
import pandas as pd

import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup

# ------------------ CLI ------------------
ap = argparse.ArgumentParser()
ap.add_argument("--q", required=True, help="User question")
ap.add_argument("--initial_k", type=int, default=24)
ap.add_argument("--final_k", type=int, default=3)
ap.add_argument("--use_reranker", action="store_true")
ap.add_argument("--reranker_id", default="BAAI/bge-reranker-base")

ap.add_argument("--model_id", default=None)
ap.add_argument("--max_new_tokens", type=int, default=600)
ap.add_argument("--temp", type=float, default=0.2)
ap.add_argument("--gpu", action="store_true", help="Use CUDA for embedding and generation (CPU default)")
ap.add_argument("--mps", action="store_true", help="Use Apple MPS (Metal) when available")
ap.add_argument("--verbose", action="store_true", help="Print extra debug info and step-by-step status")

# Sessions
ap.add_argument("--session", default="", help="Session id (stores turns in sessions/<id>.jsonl)")
ap.add_argument("--history_k", type=int, default=0, help="Include last K Q/A summaries in prompt")
ap.add_argument("--reuse_context", action="store_true", help="Reuse previous turn's retrieved chunks (no fresh search)")
ap.add_argument("--save_context", action="store_true", help="Save prompt/context to saved_contexts/")

# Attach documents (ephemeral, per-turn)
ap.add_argument("--attach", nargs='*', default=[], help="Paths to files or directories to include as additional context (chunked & embedded on the fly)")
ap.add_argument("--attach_only", action="store_true", help="Answer using only attached documents (skip index retrieval)")
ap.add_argument("--attach_limit_files", type=int, default=200, help="Max number of files to read from attachments")
ap.add_argument("--attach_limit_chunks", type=int, default=2048, help="Max number of chunks to generate from attachments")

# Debug
ap.add_argument("--show_sources", action="store_true")
ap.add_argument("--show_prompt", action="store_true")
args = ap.parse_args()

# ------------------ Env ------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ------------------ Config ------------------
CFG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")
CFG = yaml.safe_load(open(CFG_PATH, "r"))

INDEX_ROOT   = Path(CFG["paths"]["index_root"])
INDEX_NAME   = (CFG.get("index", {}) or {}).get("name", "index")
FAISS_PATH   = INDEX_ROOT / f"{INDEX_NAME}.faiss"
META_PATH    = INDEX_ROOT / f"{INDEX_NAME}.meta.parquet"
SYMS_PATH    = INDEX_ROOT / "symbols.parquet"

EMB_MODEL_ID = CFG["embedding"]["model"]
EMB_NORM     = bool(CFG["embedding"].get("normalize_embeddings", True))

GEN_MODEL_ID = args.model_id or CFG.get("generation", {}).get(
    "model_id", "stabilityai/stable-code-instruct-3b"
)

PROMPT_CFG       = CFG.get("prompt", {})
MAX_INPUT_TOKENS = int(PROMPT_CFG.get("max_input_tokens", 3800))
CONTEXT_SEP      = PROMPT_CFG.get("context_separator", "\n---\n")
BASE_SYSTEM_MSG  = PROMPT_CFG.get(
    "system_message",
    "You are a concise domain expert. Prefer minimal, executable examples when code is requested. "
    "Answer grounded strictly in the provided CONTEXT. If uncertain, say so and suggest what to check."
)
TEMPLATE         = PROMPT_CFG.get("template",
    "SYSTEM:\n{system_message}\n\nQUESTION:\n{question}\n\nCONTEXT (CITED EXCERPTS):\n{context}\n\nANSWER:\n"
)

INDEXING_CFG = CFG.get("indexing", {}) or {}
DEF_CODE_MAX_LINES   = int(INDEXING_CFG.get("code_max_lines", 100))
DEF_PROSE_MAX_CHARS  = int(INDEXING_CFG.get("prose_max_chars", 1600))
DEF_MIN_CHUNK_CHARS  = int(INDEXING_CFG.get("min_chunk_chars", 300))
DEF_OVERLAP_LINES    = int(INDEXING_CFG.get("overlap_lines", 10))
MAX_TOKENS_PER_CHUNK = int((CFG.get("max_tokens_per_chunk") or 0) or 0)
TOKENIZER_MODEL_ID   = CFG.get("tokenizer_model_id") or GEN_MODEL_ID

SOURCE_CFGS = []
for _src in (CFG.get("sources") or []):
    root = Path(_src.get("path", "")).resolve()
    if not root.exists():
        continue
    SOURCE_CFGS.append({
        "root": root,
        "type": _src.get("type", "source"),
        "name": _src.get("name") or _src.get("type", "source"),
        "treat_as": (_src.get("treat_as") or "auto").lower(),
        "chunking": _src.get("chunking") if isinstance(_src.get("chunking"), dict) else {},
    })

CHUNK_TOKENIZER = None

# ------------------ Load models & data ------------------
print(f"[chat] loading embedder: {EMB_MODEL_ID}")
embedder = SentenceTransformer(EMB_MODEL_ID)
HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
if args.gpu and args.mps:
    raise SystemExit("[error] --gpu and --mps cannot be used together.")
if args.gpu and not torch.cuda.is_available():
    print("[error] --gpu set but CUDA not available. Install CUDA-enabled PyTorch and proper drivers.")
    raise SystemExit(1)
if args.mps and not HAS_MPS:
    raise SystemExit("[error] --mps set but PyTorch MPS backend is unavailable.")

USE_CUDA = bool(args.gpu and torch.cuda.is_available())
USE_MPS = bool(args.mps and HAS_MPS)

if USE_CUDA:
    try:
        embedder = embedder.to("cuda")
        first = getattr(embedder, "_first_module", None)
        if callable(first):
            mod = first()
            auto_model = getattr(mod, "auto_model", None)
            if auto_model is not None:
                auto_model.half()
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        print(f"[chat] embedder on CUDA (FP16): {torch.cuda.get_device_name(0)}")
    except Exception:
        USE_CUDA = False
elif USE_MPS:
    try:
        embedder = embedder.to("mps")
        print("[chat] embedder on MPS (Apple Metal)")
    except Exception:
        USE_MPS = False

print(f"[chat] loading FAISS index: {FAISS_PATH}")
index = None
meta  = pd.DataFrame()
try:
    index = faiss.read_index(str(FAISS_PATH))
    meta  = pd.read_parquet(META_PATH)
    print(f"[chat] meta rows: {len(meta)}")
except Exception:
    if not args.attach_only and not args.attach:
        raise

# Optional BM25 (configurable)
bm25 = None
HAVE_BM25 = False
try:
    from rank_bm25 import BM25Okapi
    bm25_cfg = (CFG.get("retrieval", {}) or {}).get("bm25", {}) or {}
    bm25_enabled = bool(bm25_cfg.get("enabled", True))
    if bm25_enabled and not meta.empty:
        bm25 = BM25Okapi([t.split() for t in meta["text"].astype(str)])
        HAVE_BM25 = bm25 is not None
except Exception:
    bm25 = None
    HAVE_BM25 = False

# Optional symbols
symbols = None
syms_set = set()
if SYMS_PATH.exists():
    symbols = pd.read_parquet(SYMS_PATH)
    syms_set = set(symbols["symbol"].unique())

# Generator
print(f"[chat] loading generator: {GEN_MODEL_ID}")
tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID, trust_remote_code=True)
gen = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_ID,
    dtype=(torch.float16 if (USE_CUDA or USE_MPS) else torch.float32),
    device_map=("auto" if USE_CUDA else ("mps" if USE_MPS else "cpu")),
    trust_remote_code=True
)

def model_max_ctx():
    # Prefer generator config; fall back to tokenizer’s guess
    m = getattr(gen.config, "max_position_embeddings", None)
    if not m or m <= 0 or m > 32768:
        m = getattr(tok, "model_max_length", 4096)
    # cap to something sane if tokenizer says "infinite"
    if m is None or m > 32768:
        m = 4096
    return int(m)

MODEL_MAX = model_max_ctx()
# Align tokenizer reported model_max_length to effective context to avoid warnings
try:
    tok.model_max_length = int(MODEL_MAX)
    tok.init_kwargs = dict(getattr(tok, 'init_kwargs', {}) or {}, model_max_length=int(MODEL_MAX))
except Exception:
    pass

def get_chunk_tokenizer():
    """Lazily load tokenizer for token-cap enforcement on attachment chunks."""
    global CHUNK_TOKENIZER
    if MAX_TOKENS_PER_CHUNK <= 0:
        return None
    if CHUNK_TOKENIZER is not None:
        return CHUNK_TOKENIZER
    try:
        CHUNK_TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, trust_remote_code=True)
    except Exception:
        # fallback: reuse generator tokenizer if available
        CHUNK_TOKENIZER = tok
    return CHUNK_TOKENIZER

def trim_to_token_cap(text: str, cap: int) -> tuple[str, bool]:
    if cap <= 0:
        return text, False
    tok_local = get_chunk_tokenizer()
    if tok_local is None:
        return text, False
    ids = tok_local.encode(text, add_special_tokens=False)
    if len(ids) <= cap:
        return text, False
    trimmed_text = tok_local.decode(ids[:cap], skip_special_tokens=True)
    return trimmed_text, True

def sentence_split(text: str):
    s = text.strip()
    if not s:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        out = [t.strip() for t in sent_tokenize(s) if t.strip()]
        if out:
            return out
    except Exception:
        pass
    parts = re.split(r"(?<=[.!?])\\s+(?=[A-Z0-9])", s)
    out = [p.strip() for p in parts if p.strip()]
    return out or [s]

def effective_chunk_params(src_cfg: dict | None):
    chunk_cfg = (src_cfg.get("chunking") if src_cfg else {}) or {}
    return dict(
        code_max_lines=int(chunk_cfg.get("code_max_lines") or DEF_CODE_MAX_LINES),
        overlap_lines=int(chunk_cfg.get("overlap_lines") or DEF_OVERLAP_LINES),
        prose_max_chars=int(chunk_cfg.get("prose_max_chars") or DEF_PROSE_MAX_CHARS),
        min_chunk_chars=int(chunk_cfg.get("min_chunk_chars") or DEF_MIN_CHUNK_CHARS),
        token_cap=int(chunk_cfg.get("tokenizer_cap") or MAX_TOKENS_PER_CHUNK),
    )

def dedup_key(text: str) -> str:
    return " ".join(text.split())

def find_source_for_path(p: Path) -> dict | None:
    pr = p.resolve()
    best = None
    for src in SOURCE_CFGS:
        root = src["root"]
        try:
            pr.relative_to(root)
            if best is None or len(str(root)) > len(str(best["root"])):
                best = src
        except Exception:
            continue
    return best

# ------------------ Attachments (ephemeral) ------------------
SKIP_BIN = {".png",".bmp",".gif",".jpg",".jpeg",".ogg",".wav",".mp3",".bin",".dat",".zip",".7z",".rar",".exe"}
DOC_SUFFIXES = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp", ".rtf"}

def _is_code_file(p: Path) -> bool:
    return p.suffix.lower() in {".h",".hpp",".c",".cpp",".cc",".py",".js",".ts",".go",".rs",".java",".cs",".swift",".kt",".m",".mm",".s",".asm"}

def _read_text(p: Path) -> str:
    try:
        suf = p.suffix.lower()
        if suf in {".html",".htm"}:
            html = p.read_text(encoding='utf-8', errors='ignore')
            try:
                soup = BeautifulSoup(html, 'lxml')
            except Exception:
                soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text("\n")
        if suf in DOC_SUFFIXES:
            try:
                from unstructured.partition.auto import partition
                elements = partition(filename=str(p))
                return "\n\n".join(getattr(el, "text", "") for el in elements if getattr(el, "text", ""))
            except Exception:
                return ""
        return p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""

def _resolve_treat_as(p: Path, src_cfg: dict | None) -> str:
    ta = (src_cfg.get("treat_as") if src_cfg else "auto") or "auto"
    if ta == "code":
        return "code"
    if ta == "prose":
        return "prose"
    suf = p.suffix.lower()
    if suf in DOC_SUFFIXES:
        return "prose"
    if _is_code_file(p) or ("inc" in p.parts or "include" in p.parts or "src" in p.parts or "sample" in p.parts):
        return "code"
    return "prose"

def _line_chunks(text: str, max_lines: int = 100, overlap: int = 10):
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        j = min(i + max_lines, n)
        chunk = "\n".join(lines[i:j])
        if chunk.strip():
            yield (i+1, j, chunk)
        if j == n: break
        i = max(0, j - overlap)

def _char_chunks(text: str, max_chars: int = 1600, min_chars: int = 300):
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if len(chunk) >= min_chars or j == n:
            yield chunk
        i = j

def _prose_chunks(text: str, max_chars: int, min_chars: int):
    sentences = sentence_split(text)
    if len(sentences) <= 1:
        yield from _char_chunks(text, max_chars, min_chars)
        return
    buf = []
    buf_len = 0
    for idx, sent in enumerate(sentences):
        if not sent:
            continue
        add_len = len(sent) + (1 if buf else 0)
        if buf and (buf_len + add_len) > max_chars:
            chunk = " ".join(buf).strip()
            if chunk and (len(chunk) >= min_chars or idx == len(sentences) - 1):
                yield chunk
            buf = [sent]
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += add_len
    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            yield chunk

def iter_attach_files(paths: list[str]) -> list[Path]:
    out = []
    for s in paths:
        p = Path(s)
        if p.is_dir():
            out.extend([q for q in p.rglob('*') if q.is_file()])
        elif p.is_file():
            out.append(p)
    res = []
    for p in out:
        if p.suffix.lower() in SKIP_BIN:
            continue
        res.append(p)
    # deterministic order then limit
    res = sorted(res, key=lambda x: str(x))
    if args.attach_limit_files and len(res) > args.attach_limit_files:
        res = res[: args.attach_limit_files]
    return res

def build_attachment_rows() -> list[dict]:
    rows = []
    for p in iter_attach_files(args.attach):
        txt = _read_text(p)
        if not txt.strip():
            continue
        src_cfg = find_source_for_path(p)
        params = effective_chunk_params(src_cfg)
        token_cap = params["token_cap"]
        treat_as = _resolve_treat_as(p, src_cfg)
        seen = set()
        if treat_as == "code":
            for (start, end, chunk) in _line_chunks(txt, params["code_max_lines"], params["overlap_lines"]):
                if len(chunk.strip()) < 5:
                    continue
                chunk_text, _trimmed = trim_to_token_cap(chunk, token_cap)
                key = dedup_key(chunk_text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(
                    id=f"att:{p}:{start}-{end}",
                    source_type="attached",
                    repo=None,
                    path=str(p),
                    url=None,
                    start_line=start,
                    end_line=end,
                    text=chunk_text,
                ))
                if args.attach_limit_chunks and len(rows) >= args.attach_limit_chunks:
                    return rows
        else:
            for chunk in _prose_chunks(txt, params["prose_max_chars"], params["min_chunk_chars"]):
                if not chunk:
                    continue
                chunk_text, _trimmed = trim_to_token_cap(chunk, token_cap)
                key = dedup_key(chunk_text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(
                    id=f"att:{p}:{hash(chunk_text)}",
                    source_type="attached",
                    repo=None,
                    path=str(p),
                    url=None,
                    start_line=None,
                    end_line=None,
                    text=chunk_text,
                ))
                if args.attach_limit_chunks and len(rows) >= args.attach_limit_chunks:
                    return rows
    return rows

def build_attachment_pool(q: str) -> pd.DataFrame:
    rows = build_attachment_rows()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # semantic score via dot(q, emb)
    if USE_CUDA:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            qvec = embedder.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
            tvec = embedder.encode(df['text'].astype(str).tolist(), batch_size=32, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
        qv = qvec.detach().float().cpu().numpy()
        tv = tvec.detach().float().cpu().numpy()
    elif USE_MPS:
        qvec = embedder.encode([q], batch_size=1, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
        tvec = embedder.encode(df['text'].astype(str).tolist(), batch_size=32, convert_to_tensor=True, normalize_embeddings=EMB_NORM, show_progress_bar=False)
        qv = qvec.detach().cpu().numpy()
        tv = tvec.detach().cpu().numpy()
    else:
        qv = embedder.encode([q], normalize_embeddings=EMB_NORM)
        tv = embedder.encode(df['text'].astype(str).tolist(), normalize_embeddings=EMB_NORM)
    import numpy as _np
    scores = (_np.asarray(qv) @ _np.asarray(tv).T).ravel()
    df['sem_score'] = scores
    for col in ('lex_score','id_score'):
        df[col] = 0.0
    # normalize like main pool
    s = df['sem_score'].astype(float)
    if s.max() > 0:
        df['sem_score'] = (s - s.min()) / (s.max() - s.min() + 1e-9)
    df['fused'] = 0.55*df['sem_score']
    return df

# ------------------ Retrieval utils ------------------
RE_SYM = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b')

def extract_symbols(q: str):
    toks = [t for t in RE_SYM.findall(q)]
    prefixes = (CFG.get("retrieval", {}) or {}).get("symbol_prefixes", [])
    if prefixes:
        keep = [t for t in toks if any(t.startswith(pref) for pref in prefixes)]
        return keep or toks[:6]
    return toks[:6]

def semantic_search(q: str, k: int) -> pd.DataFrame:
    # Embed query; use AMP FP16 on CUDA for Tensor Core acceleration, return float32 numpy for FAISS
    if USE_CUDA:
        # Use new torch.amp.autocast API (device_type='cuda')
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            qvec = embedder.encode(
                [q],
                batch_size=1,
                convert_to_tensor=True,
                normalize_embeddings=EMB_NORM,
                show_progress_bar=False,
            )
        qv = qvec.detach().float().cpu().numpy()
    elif USE_MPS:
        qvec = embedder.encode(
            [q],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=EMB_NORM,
            show_progress_bar=False,
        )
        qv = qvec.detach().cpu().numpy()
    else:
        qv = embedder.encode([q], normalize_embeddings=EMB_NORM)
    scores, ids = index.search(np.asarray(qv, dtype="float32"), k)
    rows = meta.iloc[ids[0]].copy()
    rows["sem_score"] = scores[0]
    return rows

def lexical_search(q: str, k: int) -> pd.DataFrame:
    if not bm25:
        return pd.DataFrame()
    scores = bm25.get_scores(q.split())
    import numpy as _np
    k = max(0, min(int(k), len(scores)))
    if k == 0:
        return pd.DataFrame()
    top_idx = _np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[_np.argsort(_np.asarray(scores)[top_idx])[::-1]]
    out = meta.iloc[top_idx].copy()
    out["lex_score"] = _np.asarray(scores)[top_idx]
    return out

def identifier_hits(q: str) -> pd.DataFrame:
    if symbols is None:
        return pd.DataFrame()
    qsyms = [s for s in extract_symbols(q) if s in syms_set]
    if not qsyms:
        return pd.DataFrame()
    hit = symbols[symbols["symbol"].isin(qsyms)]
    m = meta  # avoid full copy on each query

    out_rows = []
    for _, r in hit.iterrows():
        cand = m[m["path"] == r["path"]]
        if cand.empty:
            continue
        sl, el = int(r["start_line"]), int(r["end_line"])
        c = cand
        if "start_line" in c.columns and "end_line" in c.columns:
            c = c[(pd.notna(c["start_line"])) & (pd.notna(c["end_line"]))]
            try:
                c = c[(c["start_line"].astype(int) <= el) & (c["end_line"].astype(int) >= sl)]
            except Exception:
                pass
        if c.empty:
            c = cand.head(3)
        if not c.empty:
            c = c.copy()
            c["id_score"] = 1.0
            out_rows.append(c)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True).drop_duplicates("id")

def fuse_pool(q: str, k_init: int) -> pd.DataFrame:
    sem = semantic_search(q, k_init)
    lex = lexical_search(q, k_init)
    ide = identifier_hits(q)
    pool = pd.concat([sem, lex, ide], ignore_index=True).drop_duplicates("id")

    for col in ("sem_score","lex_score","id_score"):
        if col not in pool.columns:
            pool[col] = 0.0
        else:
            s = pool[col].astype(float)
            if s.max() > 0:
                pool[col] = (s - s.min()) / (s.max() - s.min() + 1e-9)
            else:
                pool[col] = 0.0
    fw = (CFG.get("retrieval", {}) or {}).get("fusion_weights", {}) or {}
    w_sem = float(fw.get("semantic", 0.55))
    w_lex = float(fw.get("lexical", 0.35))
    w_id  = float(fw.get("identifier", 0.10))
    pool["fused"] = w_sem*pool["sem_score"] + w_lex*pool["lex_score"] + w_id*pool["id_score"]
    return pool

def rerank_df(q: str, df: pd.DataFrame, k_final: int, model_id: str) -> pd.DataFrame:
    if df.empty or len(df) <= k_final:
        return df.head(k_final)
    rr = CrossEncoder(model_id)
    pairs = [(q, str(t)[:3000]) for t in df["text"].astype(str)]
    scores = rr.predict(pairs)
    df = df.copy()
    df["rr_score"] = scores
    return df.nlargest(k_final, "rr_score")

def hybrid_retrieve(q: str, k_init: int, k_final: int, use_reranker: bool, reranker_id: str) -> pd.DataFrame:
    pool = fuse_pool(q, k_init) if index is not None and not meta.empty and not args.attach_only else pd.DataFrame()
    # merge in any attached documents
    if args.attach:
        att = build_attachment_pool(q)
        pool = pd.concat([pool, att], ignore_index=True) if not pool.empty else att
    # Guard: if caller requested zero results or pool is empty, return empty
    if k_final <= 0 or pool.empty:
        return pool.head(0).reset_index(drop=True)
    if use_reranker:
        top = rerank_df(q, pool, k_final, reranker_id)
    else:
        top = pool.nlargest(k_final, "fused")
    return top.reset_index(drop=True)

# ------------------ Sessions ------------------
def session_path(sess_id: str) -> Path:
    d = Path("sessions"); d.mkdir(exist_ok=True)
    return d / f"{sess_id}.jsonl"

def load_session(sess_id: str) -> list:
    if not sess_id:
        return []
    p = session_path(sess_id)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def append_session(sess_id: str, record: dict):
    if not sess_id:
        return
    p = session_path(sess_id)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def last_context_from_session(sess: list) -> list[str]:
    # returns the context blocks used in the last turn, if any
    for rec in reversed(sess):
        if "context_blocks" in rec and rec["context_blocks"]:
            return rec["context_blocks"]
    return []

def summarize_text(s: str, max_tokens: int = 200) -> str:
    # crude budgeted summary: take first/last chunks around 100 tokens each
    # ensure tokenizer never encodes beyond a safe limit to avoid warnings
    cap = max(64, min(int(MODEL_MAX) - 32, max_tokens * 2))
    ids = tok.encode(s, add_special_tokens=False, truncation=True, max_length=cap)
    if len(ids) <= max_tokens:
        return s.strip()
    head = tok.decode(ids[: max_tokens // 2], skip_special_tokens=False)
    tail = tok.decode(ids[-max_tokens // 2 :], skip_special_tokens=False)
    return (head + "\n...\n" + tail).strip()

def history_prefix(sess: list, k: int) -> str:
    if k <= 0 or not sess:
        return ""
    recent = sess[-k:]
    parts = []
    for rec in recent:
        q = rec.get("question","").strip()
        a = rec.get("answer","").strip()
        if not q or not a:
            continue
        a_sum = summarize_text(a, max_tokens=240)
        parts.append(f"Prev Q: {q}\nPrev A (summary):\n{a_sum}")
    if not parts:
        return ""
    return "Conversation context:\n" + "\n---\n".join(parts) + "\n---\n"

# ------------------ Formatting & prompt ------------------
def fmt_citation(row) -> str:
    st = str(row.get("source_type", "source") or "source")
    path = str(row.get("path", "") or "")
    url  = str(row.get("url", "") or "")
    start = row.get("start_line"); end = row.get("end_line")
    label = Path(path or url).name if (path or url) else st
    if pd.notna(start) and pd.notna(end) and (path or url):
        return f"[{st}:{label}:{int(start)}-{int(end)}]"
    return f"[{st}:{label}]" if (path or url) else f"[{st}]"

def section_header(row) -> str:
    st = str(row.get("source_type", "source") or "source")
    return st.capitalize() if st else "Source"

def render_context_sections(rows: pd.DataFrame) -> list[str]:
    blocks = []
    for _, r in rows.iterrows():
        hdr = section_header(r)
        cite = fmt_citation(r)
        path = r.get("path", "") or r.get("url","")
        start = r.get("start_line"); end = r.get("end_line")
        loc = f" L{int(start)}-{int(end)}" if pd.notna(start) and pd.notna(end) else ""
        header_line = f"[{hdr}] {path}{loc} {cite}"
        text = str(r["text"]).strip()
        blocks.append(f"{header_line}\n{text}")
    return blocks

def ntoks(s: str, max_len: int | None = None) -> int:
    ml = int(max_len) if max_len else int(MODEL_MAX)
    return len(tok.encode(s, add_special_tokens=False, truncation=True, max_length=ml))

def build_prompt(question: str, ctx_blocks: list[str], session_hist: str) -> str:
    # Reserve space for generation (max_new_tokens) + a small safety buffer
    reserve = int(args.max_new_tokens) + 32
    budget  = max(256, min(MAX_INPUT_TOKENS, MODEL_MAX - reserve))

    system_message = (session_hist + BASE_SYSTEM_MSG).strip()
    def render(ctx_text: str) -> str:
        return TEMPLATE.format(system_message=system_message, question=question, context=ctx_text)

    blocks = ctx_blocks[:]
    ctx_text = CONTEXT_SEP.join(blocks)
    prompt = render(ctx_text)

    # Drop blocks until it fits (use truncation-aware length)
    while ntoks(prompt, budget) > budget and blocks:
        blocks.pop()
        ctx_text = CONTEXT_SEP.join(blocks)
        prompt = render(ctx_text)

    # If still too long, trim question then context
    if ntoks(prompt, budget) > budget:
        q_ids = tok.encode(question, add_special_tokens=False, truncation=True, max_length=256)
        question_trim = tok.decode(q_ids[-256:], skip_special_tokens=False)
        prompt = TEMPLATE.format(system_message=system_message, question=question_trim, context=ctx_text)

    if ntoks(prompt, budget) > budget and ctx_text:
        # final clamp on context tokens
        head = TEMPLATE.format(system_message=system_message, question=question, context="")
        head_ids = tok.encode(head, add_special_tokens=False, truncation=True, max_length=budget)
        head_budget = len(head_ids)
        keep_ctx = max(0, budget - head_budget)
        ctx_ids = tok.encode(ctx_text, add_special_tokens=False, truncation=True, max_length=keep_ctx)
        ctx_trim = tok.decode(ctx_ids[:keep_ctx], skip_special_tokens=False)
        prompt = TEMPLATE.format(system_message=system_message, question=question, context=ctx_trim)

    return prompt

# ------------------ Continuation-safe generation ------------------
def tail_tokens(s: str, keep: int) -> str:
    ids = tok.encode(s, add_special_tokens=False, truncation=True, max_length=max(0, keep))
    if len(ids) <= keep:
        return s
    return tok.decode(ids[-keep:], skip_special_tokens=False)

def generate_complete(prompt_base, max_tokens=600, temperature=0.2, rounds=5):
    """
    Continue generation in rounds, while ensuring the *input* stays under the model limit.
    We keep the original grounded prompt and add only the LAST ~256 tokens of the running answer
    as continuation context each round.
    """
    full_answer = ""
    prev_len = 0

    # Pre-encode constant prompt head to measure budget each round
    reserve = int(max_tokens) + 32
    total_budget = max(256, MODEL_MAX - reserve)
    # We'll allow up to 256 tokens of previous answer as continuation context
    cont_tail_tokens = 256

    for _ in range(rounds):
        # Build safe input: prompt_base + tail of current answer
        cont_tail = tail_tokens(full_answer, cont_tail_tokens) if full_answer else ""
        if cont_tail:
            inp = prompt_base + cont_tail
        else:
            inp = prompt_base

        # If still too long, trim the *cont_tail* further
        enc_len = len(tok.encode(inp, add_special_tokens=False, truncation=True, max_length=total_budget))
        if enc_len > total_budget:
            # Reduce how many answer tokens we carry over
            overflow = enc_len - total_budget
            # heuristic: drop ~overflow tokens from cont_tail
            cont_keep = max(0, cont_tail_tokens - overflow - 32)
            cont_tail = tail_tokens(full_answer, cont_keep) if full_answer else ""
            inp = prompt_base + cont_tail

        # Final guard: hard trim to budget
        inp_ids = tok.encode(inp, add_special_tokens=False, truncation=True, max_length=total_budget)
        if len(inp_ids) > total_budget:
            inp_ids = inp_ids[-total_budget:]
            inp = tok.decode(inp_ids, skip_special_tokens=False)

        # Strict tokenization guard: enforce truncation to stay under model context budget
        eff_max = max(256, total_budget)
        inputs = tok(
            inp,
            return_tensors="pt",
            truncation=True,
            max_length=eff_max,
        ).to(gen.device)
        with torch.no_grad():
            out = gen.generate(
                **inputs,
                do_sample=(temperature > 0),
                temperature=max(0.01, temperature),
                max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)

        # Strip prompt echo if template is present
        if "ANSWER:" in text:
            text = text.split("ANSWER:", 1)[-1]

        # De-dup overlap
        if full_answer and text.startswith(full_answer):
            text = text[len(full_answer):]

        full_answer += text.strip()

        # Stop if we reached a clean boundary or the model produced little
        if full_answer.endswith(('.', '!', '?', '```')) or len(tok.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)) < max_tokens * 0.25:
            break

        # Safety break if no progress
        if len(full_answer) == prev_len:
            break
        prev_len = len(full_answer)

    return full_answer.strip()

# ------------------ Run one turn ------------------
question = args.q.strip()
session_id = args.session.strip()
sess = load_session(session_id) if session_id else []

# Build context blocks
if args.reuse_context and sess:
    # reuse last context blocks
    last_ctx = last_context_from_session(sess)
    ctx_blocks = last_ctx if last_ctx else []
    hits = None
else:
    # fresh retrieval
    hits = hybrid_retrieve(
        question,
        k_init=args.initial_k,
        k_final=args.final_k,
        use_reranker=args.use_reranker,
        reranker_id=args.reranker_id
    )
    ctx_blocks = render_context_sections(hits)
    if args.verbose:
        print(f"[chat] retrieved {len(hits)} hits -> {len(ctx_blocks)} context blocks")

# System prefix with last K Q/A summaries (token-bounded)
hist_prefix = history_prefix(sess, args.history_k)

# Build prompt
prompt = build_prompt(question, ctx_blocks, hist_prefix)
if args.verbose:
    try:
        used = ntoks(prompt, MAX_INPUT_TOKENS)
        print(f"[chat] prompt tokens (approx): {used}  | max_new_tokens={args.max_new_tokens}")
    except Exception:
        pass

# Optionally save context/prompt
if args.save_context:
    out_dir = Path("saved_contexts"); out_dir.mkdir(exist_ok=True)
    safe_q = re.sub(r"[^a-zA-Z0-9]+", "_", question)[:60]
    fname = out_dir / f"context_{safe_q}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("QUESTION:\n" + question + "\n\n")
        f.write("CONTEXT USED:\n" + CONTEXT_SEP.join(ctx_blocks) + "\n\n")
        f.write("FULL PROMPT:\n" + prompt)
    print(f"[chat] Context saved to {fname}")

# Generate (continuation-safe)
answer = generate_complete(prompt, max_tokens=args.max_new_tokens, temperature=args.temp)

# Clean template echo if present
if "ANSWER:" in answer:
    answer = answer.split("ANSWER:", 1)[-1].strip()

print("\n" + answer + "\n")

# Show sources
if args.show_sources and hits is not None:
    print("Sources:")
    for _, r in hits.iterrows():
        cite = fmt_citation(r)
        path = r.get("path", "") or r.get("url", "")
        start = r.get("start_line"); end = r.get("end_line")
        loc = f" L{int(start)}-{int(end)}" if pd.notna(start) and pd.notna(end) else ""
        fused = r.get("fused", 0.0)
        rr    = r.get("rr_score", 0.0)
        print(f" - {cite}  {path}{loc}  (fused={fused:.3f}{', rr='+format(rr,'.3f') if rr else ''})")

# Persist this turn in session
append_session(session_id, {
    "question": question,
    "context_blocks": ctx_blocks,
    "answer": answer,
    "used_reranker": bool(args.use_reranker),
    "final_k": int(args.final_k)
} if session_id else {})
