#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid-Fast (CPU+GPU) index builder — with INCREMENTAL mode
- CPU: concurrent file scanning + chunking (producer)
- GPU: embeddings with OOM backoff, automatic CPU fallback (consumer)
- Large producer bundles + deeper queue for throughput
- Warm-up encode, fewer PyTorch CPU threads (less contention on Windows)
- Skips .s/.asm and large files by default
- Writes FAISS index + Parquet metadata + embeddings.npy + manifest.json
- INCREMENTAL: reuse unchanged vectors; re-embed only new/changed files; drop deleted ones
- Detailed timing & throughput
"""

import os, re, yaml, argparse, hashlib, threading, queue, time, json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Quieter HF / tokenizers; reduce PyTorch thread contention on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
try:
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
except Exception:
    pass

# -------------------------
# CLI
# -------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true", help="Use CUDA if available (CPU is default)")
ap.add_argument("--mps", action="store_true", help="Use Apple MPS (Metal) if available")
ap.add_argument("--batch", type=int, default=0, help="Override embedding batch size from config")
ap.add_argument("--limit", type=int, default=0, help="Only embed first N NEW/CHANGED chunks (quick test)")
ap.add_argument("--max_size_mb", type=int, default=2, help="Skip files larger than this size (MB)")
ap.add_argument("--threads", type=int, default=6, help="File IO/Chunking worker threads (CPU producers)")
ap.add_argument("--incremental", action="store_true", help="Reuse previous vectors; re-embed only new/changed files")
args = ap.parse_args()

# Determine accelerator availability
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

if args.gpu and args.mps:
    print("[error] --gpu and --mps cannot be used together.")
    raise SystemExit(1)

if args.gpu and not HAS_CUDA:
    print("[error] --gpu was set but CUDA is not available in this Python environment.")
    print("        Install a CUDA-enabled PyTorch build and ensure an NVIDIA GPU/driver is present.")
    print("        Example (CUDA 12.1): pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio")
    raise SystemExit(1)

if args.mps and not HAS_MPS:
    print("[error] --mps was set but PyTorch was not built with Metal (MPS) support.")
    print("        Use a recent PyTorch build on Apple Silicon or omit --mps.")
    raise SystemExit(1)

# -------------------------
# Config
# -------------------------
CFG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")
CFG = yaml.safe_load(open(CFG_PATH, "r"))
INDEX_ROOT = Path(CFG["paths"]["index_root"])
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

# Generic index name (used in artifact filenames)
INDEX_NAME = (CFG.get("index", {}) or {}).get("name", "index")

EMB_MODEL   = CFG["embedding"]["model"]
EMB_NORM    = bool(CFG["embedding"].get("normalize_embeddings", True))
BATCH_SIZE  = int(CFG["embedding"]["batch_size"])
if args.batch > 0:
    BATCH_SIZE = args.batch

CODE_MAX_LINES   = int(CFG["indexing"]["code_max_lines"])
PROSE_MAX_CHARS  = int(CFG["indexing"]["prose_max_chars"])
MIN_CHUNK_CHARS  = int(CFG["indexing"]["min_chunk_chars"])
OVERLAP_LINES    = int(CFG["indexing"]["overlap_lines"])
MAX_TOKENS_PER_CHUNK = int((CFG.get("max_tokens_per_chunk") or 0) or 0)
TOKENIZER_MODEL_ID = CFG.get("tokenizer_model_id") or CFG.get("model_id") or EMB_MODEL
TOKENIZER = None
TOKENIZER_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()

CHUNK_STATS: Counter = Counter()

# Sources are provided via config.yaml under the `sources:` list.

FAISS_PATH   = INDEX_ROOT / f"{INDEX_NAME}.faiss"
META_PATH    = INDEX_ROOT / f"{INDEX_NAME}.meta.parquet"
EMBS_PATH    = INDEX_ROOT / f"{INDEX_NAME}.embeddings.npy"
MANIFEST_PATH= INDEX_ROOT / f"{INDEX_NAME}.manifest.json"

if not EMB_NORM:
    print("[warn] normalize_embeddings=False with IndexFlatIP. Consider enabling normalization for cosine/IP scoring.")

print(f"[index] config: {CFG_PATH}")
print(f"[index] index_name: {INDEX_NAME}")
print(f"[index] embedder: {EMB_MODEL}  | batch={BATCH_SIZE}")
print(f"[index] index_root: {INDEX_ROOT}")

# -------------------------
# Helpers
# -------------------------
def get_tokenizer() -> Optional[AutoTokenizer]:
    """Lazily load tokenizer for token-cap enforcement."""
    global TOKENIZER
    if MAX_TOKENS_PER_CHUNK <= 0:
        return None
    if TOKENIZER is None:
        with TOKENIZER_LOCK:
            if TOKENIZER is None:
                try:
                    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, trust_remote_code=True)
                except Exception:
                    TOKENIZER = None
    return TOKENIZER

def trim_to_token_cap(text: str, cap: int) -> Tuple[str, bool]:
    if cap <= 0:
        return text, False
    tok = get_tokenizer()
    if tok is None:
        return text, False
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= cap:
        return text, False
    trimmed_text = tok.decode(ids[:cap], skip_special_tokens=True)
    return trimmed_text, True

def sentence_split(text: str) -> List[str]:
    """Best-effort sentence splitter with lightweight fallback."""
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
    try:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", s)
        out = [p.strip() for p in parts if p.strip()]
        if out:
            return out
    except Exception:
        pass
    return [s]

def effective_chunk_params(entry: Dict) -> Dict[str, int]:
    chunk_cfg = entry.get("chunking") if isinstance(entry.get("chunking"), dict) else {}
    return {
        "code_max_lines": int(chunk_cfg.get("code_max_lines") or CODE_MAX_LINES),
        "overlap_lines": int(chunk_cfg.get("overlap_lines") or OVERLAP_LINES),
        "prose_max_chars": int(chunk_cfg.get("prose_max_chars") or PROSE_MAX_CHARS),
        "min_chunk_chars": int(chunk_cfg.get("min_chunk_chars") or MIN_CHUNK_CHARS),
        "token_cap": int(chunk_cfg.get("tokenizer_cap") or MAX_TOKENS_PER_CHUNK),
    }

def dedup_key(text: str) -> str:
    return " ".join(text.split())

SKIP_BIN = {".png",".bmp",".gif",".jpg",".jpeg",".ogg",".wav",".mp3",".bin",".dat",".zip",".7z",".rar",".exe"}
SKIP_ASM = {".s",".asm"}   # skip heavy assembly

def is_code_file(p: Path) -> bool:
    return p.suffix.lower() in {
        ".h",".hpp",".c",".cpp",".cc",".s",".asm",".ld",".mk",
        ".py",".js",".ts",".java",".go",".rs",".cs",".swift",".kt",".m",".mm"
    }

def is_text_file(p: Path) -> bool:
    return p.suffix.lower() in {".txt",".md",".rst",".adoc",".html",".htm"}

# Office/PDF document types handled via unstructured
DOC_SUFFIXES = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp", ".rtf"}

def read_doc_text(p: Path) -> str:
    try:
        # Local import to avoid import cost if unused
        from unstructured.partition.auto import partition
        elements = partition(filename=str(p))
        return "\n\n".join(getattr(el, "text", "") for el in elements if getattr(el, "text", ""))
    except Exception:
        return ""

def line_chunks(text: str, max_lines: int, overlap: int):
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        j = min(i + max_lines, n)
        chunk = "\n".join(lines[i:j])
        if chunk.strip():
            yield (i+1, j, chunk)  # 1-based
        if j == n: break
        i = j - overlap
        if i < 0: i = 0

def char_chunks(text: str, max_chars: int, min_chars: int):
    i, n = 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if len(chunk) >= min_chars or j == n:
            yield chunk
        i = j

def prose_chunks(text: str, max_chars: int, min_chars: int) -> List[str]:
    """Sentence-aware prose chunking with fallback to char windows."""
    sentences = sentence_split(text)
    if len(sentences) <= 1:
        return list(char_chunks(text, max_chars, min_chars))

    out: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for idx, sent in enumerate(sentences):
        if not sent:
            continue
        add_len = len(sent) + (1 if buf else 0)
        if buf and (buf_len + add_len) > max_chars:
            chunk = " ".join(buf).strip()
            if chunk and (len(chunk) >= min_chars or idx == len(sentences) - 1):
                out.append(chunk)
            buf = [sent]
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += add_len
    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            out.append(chunk)
    return out

def _match_any(p: Path, globs: List[str]) -> bool:
    if not globs:
        return True
    s = str(p).replace("\\", "/")
    for g in globs:
        try:
            if Path(s).match(g):
                return True
        except Exception:
            # best-effort on odd patterns
            pass
    return False

def scan_sources() -> List[Dict]:
    """Scan inputs from config (generic). Falls back to legacy paths if needed."""
    out: List[Dict] = []
    src_cfg = CFG.get("sources")
    if not isinstance(src_cfg, list) or not src_cfg:
        return out

    for src in src_cfg:
        root = Path(src.get("path", "")).resolve()
        if not root.exists():
            continue
        source_type = src.get("type", "source")
        name = src.get("name") or source_type
        include = src.get("include") or ["**/*"]
        exclude = src.get("exclude") or []
        treat_as = (src.get("treat_as") or "auto").lower()
        chunking = src.get("chunking") if isinstance(src.get("chunking"), dict) else {}
        size_cap_mb = int(src.get("max_size_mb") or args.max_size_mb or 0)

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf in SKIP_BIN or suf in SKIP_ASM:
                continue
            if not _match_any(p, include):
                continue
            if exclude and _match_any(p, exclude):
                continue
            try:
                if size_cap_mb and p.stat().st_size > size_cap_mb * 1024 * 1024:
                    continue
            except Exception:
                pass
            out.append(dict(
                source_type=source_type,
                repo=name if source_type == "repo" else None,
                path=str(p),
                url=str(p) if source_type != "repo" else None,
                treat_as=treat_as,
                chunking=chunking,
            ))
    return out

def file_sig(p: Path) -> Dict:
    st = p.stat()
    return {"size": int(st.st_size), "mtime": float(st.st_mtime)}

def chunk_file(entry: Dict) -> List[Dict]:
    """Return chunk rows for a single file (no embeddings)."""
    p = Path(entry["path"])
    txt = ""
    try:
        if p.suffix.lower() in DOC_SUFFIXES:
            txt = read_doc_text(p)
        else:
            txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    rows: List[Dict] = []
    params = effective_chunk_params(entry)
    token_cap = params["token_cap"]
    trimmed_local = 0
    dedup_local = 0
    seen: Set[str] = set()
    ta = (entry.get("treat_as") or "auto").lower()
    if ta == "code":
        is_code = True
    elif ta == "prose":
        is_code = False
    else:
        # Treat office/PDF docs as prose; otherwise heuristic
        if p.suffix.lower() in DOC_SUFFIXES:
            is_code = False
        else:
            is_code = (is_code_file(p) or ("inc" in p.parts or "include" in p.parts or "src" in p.parts or "sample" in p.parts))
    if is_code:
        for (start, end, chunk) in line_chunks(txt, params["code_max_lines"], params["overlap_lines"]):
            if len(chunk.strip()) < 5: 
                continue
            chunk_text, trimmed = trim_to_token_cap(chunk, token_cap)
            if trimmed:
                trimmed_local += 1
            key = dedup_key(chunk_text)
            if key in seen:
                dedup_local += 1
                continue
            seen.add(key)
            rows.append({
                "id": hashlib.md5(f"{p}:{start}-{end}".encode()).hexdigest(),
                "source_type": entry["source_type"],
                "repo": entry.get("repo"),
                "path": str(p),
                "url": entry.get("url"),
                "start_line": start,
                "end_line": end,
                "text": chunk_text
            })
    else:
        for chunk in prose_chunks(txt, params["prose_max_chars"], params["min_chunk_chars"]):
            if not chunk:
                continue
            chunk_text, trimmed = trim_to_token_cap(chunk, token_cap)
            if trimmed:
                trimmed_local += 1
            key = dedup_key(chunk_text)
            if key in seen:
                dedup_local += 1
                continue
            seen.add(key)
            rows.append({
                "id": hashlib.md5(f"{p}:{hash(chunk_text)}".encode()).hexdigest(),
                "source_type": entry["source_type"],
                "repo": entry.get("repo"),
                "path": str(p),
                "url": entry.get("url"),
                "start_line": None,
                "end_line": None,
                "text": chunk_text
            })
    if trimmed_local or dedup_local:
        with STATS_LOCK:
            CHUNK_STATS["trimmed"] += trimmed_local
            CHUNK_STATS["deduped"] += dedup_local
    return rows

# -------------------------
# Producer / Consumer
# -------------------------
Row = Dict[str, object]

def producer(files: List[Dict], out_q: "queue.Queue[List[Row]]", workers: int, limit_chunks: int, timing: dict):
    """CPU: chunk files; push large bundles to queue for throughput."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.perf_counter()
    produced = 0
    bundle: List[Row] = []
    # try to send ~8 GPU batches per push
    bundle_target = max(BATCH_SIZE * 8, 256)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(chunk_file, f) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Chunking (CPU)"):
            rows = fut.result()
            if not rows:
                continue
            if limit_chunks > 0:
                remaining = limit_chunks - produced
                if remaining <= 0:
                    break
                if len(rows) > remaining:
                    rows = rows[:remaining]
            bundle.extend(rows)
            produced += len(rows)
            if len(bundle) >= bundle_target:
                out_q.put(bundle)
                bundle = []
            if limit_chunks > 0 and produced >= limit_chunks:
                break
    if bundle:
        out_q.put(bundle)
    out_q.put(None)  # sentinel
    timing["producer_secs"] = time.perf_counter() - t0
    timing["produced_chunks"] = produced

def prepare_embedder(model_id: str, want_gpu: bool, want_mps: bool) -> SentenceTransformer:
    emb = SentenceTransformer(model_id)
    if want_gpu and torch.cuda.is_available():
        emb = emb.to("cuda")
        # Try to set underlying transformer to FP16 for Tensor Core acceleration
        try:
            first = getattr(emb, "_first_module", None)
            if callable(first):
                mod = first()
                auto_model = getattr(mod, "auto_model", None)
                if auto_model is not None:
                    auto_model.half()
        except Exception:
            pass
        torch.cuda.empty_cache()
        print(f"[build_index] Using GPU for embeddings: {torch.cuda.get_device_name(0)} (AMP FP16)")
    elif want_mps and bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        emb = emb.to("mps")
        print("[build_index] Using Apple MPS for embeddings")
    else:
        print("[build_index] Using CPU for embeddings")
    # warm-up to reduce first-batch latency
    try:
        _ = emb.encode(["warmup"] * max(1, min(BATCH_SIZE, 8)),
                       batch_size=max(1, min(BATCH_SIZE, 8)),
                       convert_to_tensor=True,
                       normalize_embeddings=EMB_NORM,
                       show_progress_bar=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return emb

def consumer_embed(emb: SentenceTransformer, in_q: "queue.Queue[List[Row]]",
                   batch_size: int, norm: bool, timing: dict) -> Tuple[np.ndarray, List[Row]]:
    """GPU/CPU embeddings with OOM backoff + CPU fallback."""
    t0 = time.perf_counter()
    all_rows: List[Row] = []
    all_vecs: List[torch.Tensor] = []

    bs = max(1, batch_size)
    device = "cpu"
    try:
        if hasattr(emb, "parameters"):
            device = "cuda" if next(emb.parameters()).is_cuda else "cpu"
    except Exception:
        device = "cuda" if torch.cuda.is_available() and "cuda" in str(getattr(emb, "device", "")) else "cpu"

    with torch.inference_mode():
        pbar = tqdm(desc="Embedding (GPU/CPU)", unit="chunk")
        while True:
            bundle = in_q.get()
            if bundle is None:
                break
            texts = [str(r["text"]) for r in bundle]
            i, n = 0, len(texts)
            while i < n:
                j = min(i + bs, n)
                batch = texts[i:j]
                try:
                    if device == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            vecs = emb.encode(
                                batch,
                                batch_size=bs,
                                convert_to_tensor=True,
                                normalize_embeddings=norm,
                                show_progress_bar=False,
                            )
                    else:
                        vecs = emb.encode(
                            batch,
                            batch_size=bs,
                            convert_to_tensor=True,
                            normalize_embeddings=norm,
                            show_progress_bar=False,
                        )
                except RuntimeError as e:
                    msg = str(e)
                    if "CUDA out of memory" in msg and bs > 1 and device == "cuda":
                        bs = max(1, bs // 2)
                        print(f"[build_index] OOM → reducing batch_size to {bs} and retrying…")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    if "CUDA" in msg and device == "cuda":
                        print("[build_index] CUDA error → switching to CPU for remaining embeddings.")
                        emb = SentenceTransformer(EMB_MODEL)  # fresh CPU instance
                        device = "cpu"
                        bs = max(1, min(bs, 8))
                        vecs = emb.encode(
                            batch,
                            batch_size=bs,
                            convert_to_tensor=True,
                            normalize_embeddings=norm,
                            show_progress_bar=False,
                        )
                    else:
                        raise
                all_vecs.append(vecs.detach().cpu())
                all_rows.extend(bundle[i:j])
                pbar.update(j - i)
                i = j
        pbar.close()

    timing["consumer_secs"] = time.perf_counter() - t0
    timing["embedded_chunks"] = len(all_rows)

    if not all_rows:
        return np.zeros((0, 768), dtype="float32"), []

    embs = torch.cat(all_vecs, dim=0).numpy().astype("float32")
    return embs, all_rows

# -------------------------
# Incremental helpers
# -------------------------
def load_previous_state() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Dict[str, Dict]]:
    meta_prev = None
    embs_prev = None
    manifest_prev = {}
    if META_PATH.exists() and EMBS_PATH.exists() and MANIFEST_PATH.exists():
        try:
            meta_prev = pd.read_parquet(META_PATH)
            embs_prev = np.load(EMBS_PATH)
            manifest_prev = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            # sanity check shape
            if len(meta_prev) != len(embs_prev):
                print("[incremental] meta/embeddings size mismatch → full rebuild")
                return None, None, {}
        except Exception as e:
            print(f"[incremental] Failed to load previous state ({e}) → full rebuild")
            return None, None, {}
    return meta_prev, embs_prev, manifest_prev

def current_manifest(files: List[Dict]) -> Dict[str, Dict]:
    man = {}
    for f in files:
        p = Path(f["path"])
        try:
            man[str(p)] = file_sig(p)
        except Exception:
            pass
    return man

def diff_manifest(prev: Dict[str, Dict], now: Dict[str, Dict]) -> Tuple[Set[str], Set[str], Set[str]]:
    prev_paths = set(prev.keys())
    now_paths  = set(now.keys())
    deleted = prev_paths - now_paths
    new = now_paths - prev_paths
    changed = set()
    common = prev_paths & now_paths
    for p in common:
        a, b = prev[p], now[p]
        if a.get("size") != b.get("size") or abs(a.get("mtime",0) - b.get("mtime",0)) > 1e-6:
            changed.add(p)
    return new, changed, deleted

# -------------------------
# Main
# -------------------------
def fmt_secs(s: float) -> str:
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:0.2f}s"

def main():
    timing: Dict[str, float] = {}
    t_total0 = time.perf_counter()

    print("[build_index] Scanning file list…")
    files_all = scan_sources()
    if not files_all:
        print("No rows to index (no sources found).")
        return
    print(f"[build_index] Found {len(files_all)} source files")

    # Incremental plan
    meta_prev, embs_prev, manifest_prev = (None, None, {})
    new_paths: Set[str] = set()
    changed_paths: Set[str] = set()
    deleted_paths: Set[str] = set()

    manifest_now = current_manifest(files_all)

    if args.incremental:
        meta_prev, embs_prev, manifest_prev = load_previous_state()
        if meta_prev is not None and embs_prev is not None and manifest_prev:
            new_paths, changed_paths, deleted_paths = diff_manifest(manifest_prev, manifest_now)
            print(f"[incremental] new={len(new_paths)} changed={len(changed_paths)} deleted={len(deleted_paths)}")
        else:
            print("[incremental] No valid previous state → full rebuild")

    # Select files to (re)process
    if args.incremental and meta_prev is not None and embs_prev is not None and manifest_prev:
        reproc_paths = new_paths | changed_paths
        files_reproc = [f for f in files_all if f["path"] in reproc_paths]
        # Keep rows/vectors for unchanged paths
        keep_mask = ~meta_prev["path"].isin(list(reproc_paths | deleted_paths))
        kept_meta = meta_prev[keep_mask].reset_index(drop=True)
        kept_embs = embs_prev[keep_mask.values]
        print(f"[incremental] Keeping {len(kept_meta)} chunks; re-embedding from {len(reproc_paths)} files")
    else:
        files_reproc = files_all
        kept_meta = pd.DataFrame()
        kept_embs = np.zeros((0, 768), dtype="float32")

    # Prepare embedder
    want_gpu = args.gpu and HAS_CUDA
    want_mps = args.mps and HAS_MPS
    emb = prepare_embedder(EMB_MODEL, want_gpu, want_mps)

    # If nothing to reprocess (pure delete/rename), we still need to write new index
    if not files_reproc:
        combined_meta = kept_meta
        combined_embs = kept_embs
    else:
        # Pipeline: producer (CPU) -> consumer (GPU/CPU)
        qsize = max(32, (args.batch or BATCH_SIZE) * 4)
        work_q: "queue.Queue[List[Row]]" = queue.Queue(maxsize=qsize)

        # Start producer thread
        prod = threading.Thread(
            target=producer,
            args=(files_reproc, work_q, max(1, args.threads), int(args.limit or 0), timing),
            daemon=True
        )
        prod.start()

        # Consumer on main thread
        embs_new, rows_new = consumer_embed(emb, work_q, BATCH_SIZE, EMB_NORM, timing)
        prod.join()

        if len(rows_new) == 0 and len(kept_meta) == 0:
            print("No rows to index.")
            return

        # Merge kept + new
        meta_new = pd.DataFrame(rows_new)
        combined_meta = pd.concat([kept_meta, meta_new], ignore_index=True)
        combined_embs = np.vstack([kept_embs, embs_new]) if kept_embs.size else embs_new

    # Write outputs
    t_write0 = time.perf_counter()
    # Ensure alignment
    if len(combined_meta) != len(combined_embs):
        print("[build_index] FATAL: meta/embedding length mismatch")
        return

    # Write FAISS
    dim = combined_embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(combined_embs.astype("float32"))
    faiss.write_index(idx, str(FAISS_PATH))

    # Write parquet + embeddings + manifest
    combined_meta.to_parquet(META_PATH, index=False)
    np.save(EMBS_PATH, combined_embs.astype("float32"))
    Path(MANIFEST_PATH).write_text(json.dumps(manifest_now, ensure_ascii=False, indent=2), encoding="utf-8")
    t_write1 = time.perf_counter()

    t_total1 = time.perf_counter()

    # -------- TIMING REPORT --------
    print("\n===== Timing =====")
    total = t_total1 - t_total0
    prod_secs = timing.get("producer_secs", 0.0)
    cons_secs = timing.get("consumer_secs", 0.0)
    write_secs = t_write1 - t_write0
    produced = timing.get("produced_chunks", 0)
    embedded = timing.get("embedded_chunks", 0)

    print(f"Total time:      {fmt_secs(total)}")
    if args.incremental and meta_prev is not None:
        print(f"Kept (unchanged): {len(kept_meta)} chunks")
    print(f"Producer (CPU):  {fmt_secs(prod_secs)}  • chunks produced: {produced}")
    print(f"Consumer (Emb):  {fmt_secs(cons_secs)}  • chunks embedded: {embedded}  • batch={BATCH_SIZE}")
    print(f"Write (All):     {fmt_secs(write_secs)}")
    if total > 0 and (len(combined_meta) > 0):
        rate = (embedded if embedded else len(combined_meta)) / total
        print(f"Throughput:      {rate:.2f} chunks/sec (~{rate*60:.0f} chunks/min)")
    trimmed = CHUNK_STATS.get("trimmed", 0)
    deduped = CHUNK_STATS.get("deduped", 0)
    if trimmed or deduped:
        print(f"Chunks trimmed by token cap: {trimmed}  • deduped (skipped): {deduped}")
    print("=================\n")

    print("Index built:", FAISS_PATH, "and", META_PATH)

if __name__ == "__main__":
    main()
