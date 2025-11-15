#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare model answers with and without RAG context.

Runs rag/chat.py twice for a given question:
 - Baseline: no retrieval, no attachments (final_k=0)
 - With RAG: retrieval enabled (final_k>0, optional reranker)

Shows both answers and basic stats to help you judge impact.

Usage examples:
  # CPU
  python tools/compare_rag.py --q "How to init SGDK and draw a sprite?" --rag_final_k 3 --show_sources

  # GPU
  python tools/compare_rag.py --q "How to init SGDK and draw a sprite?" --rag_final_k 3 --gpu --show_sources

Environment:
  Respects RAG_CONFIG if set (e.g., config.sgdk.yaml)
"""

from __future__ import annotations
import argparse, subprocess, sys, os, shutil, textwrap, time


def run_chat(args_list: list[str]) -> str:
    try:
        out = subprocess.check_output([sys.executable, "rag/chat.py", *args_list], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output or str(e)
    return out


def clean_output(s: str) -> str:
    # Remove loader lines like [chat] ... to focus on the answer and sources
    lines = []
    for line in s.splitlines():
        if line.strip().startswith("[chat]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def compare(q: str, use_gpu: bool, use_mps: bool, rag_final_k: int, use_reranker: bool, show_sources: bool) -> None:
    common = ["--q", q]
    if show_sources:
        common.append("--show_sources")
    if use_gpu:
        common.append("--gpu")
    elif use_mps:
        common.append("--mps")

    # Baseline: no retrieval, no attachments
    base_args = common + ["--attach_only", "--final_k", "0"]
    t0 = time.perf_counter()
    base_out = run_chat(base_args)
    t1 = time.perf_counter()
    base_out = clean_output(base_out)

    # RAG: retrieval enabled
    rag_args = common + ["--final_k", str(max(1, int(rag_final_k)))]
    if use_reranker:
        rag_args.append("--use_reranker")
    t2 = time.perf_counter()
    rag_out = run_chat(rag_args)
    t3 = time.perf_counter()
    rag_out = clean_output(rag_out)

    # Print side-by-side-ish comparison
    bar = "=" * 72
    print(bar)
    print("QUESTION:")
    print(q)
    print(bar)
    print("BASELINE (no retrieval, no attachments)\n")
    print(base_out or "<no output>")
    print("\n" + bar)
    print(f"WITH RAG (final_k={rag_final_k}{', reranker' if use_reranker else ''})\n")
    print(rag_out or "<no output>")
    print("\n" + bar)

    # Simple stats
    def stats(s: str) -> tuple[int, int]:
        words = s.split()
        return (len(s), len(words))

    blen, bwin = stats(base_out)
    rlen, rwin = stats(rag_out)
    print(f"Lengths (chars/words): baseline={blen}/{bwin} | rag={rlen}/{rwin}")
    print(f"Durations (s): baseline={t1-t0:0.2f} | rag={t3-t2:0.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Question to compare")
    ap.add_argument("--gpu", action="store_true", help="Use CUDA for both runs if available")
    ap.add_argument("--mps", action="store_true", help="Use Apple MPS for both runs if available")
    ap.add_argument("--rag_final_k", type=int, default=3, help="final_k for RAG run")
    ap.add_argument("--use_reranker", action="store_true", help="Use CrossEncoder reranker in RAG run")
    ap.add_argument("--show_sources", action="store_true", help="Show sources inline in outputs")
    args = ap.parse_args()
    if args.gpu and args.mps:
        raise SystemExit("--gpu and --mps cannot be used together.")

    compare(args.q, args.gpu, args.mps, args.rag_final_k, args.use_reranker, args.show_sources)


if __name__ == "__main__":
    main()
