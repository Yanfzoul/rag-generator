#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic source fetcher driven by config.yaml.

Supports entries under `fetch:`:
  - type: github
    repo: https://github.com/org/repo
    dest: ./data/repo
    depth: 1           # optional
    branch: main       # optional

  - type: http_list
    urls: ["https://example/page1.html", "https://example/page2.html"]
    dest_dir: ./data/pages

  - type: http_file
    url: https://example/page.html
    dest: ./data/pages/page.html

This is optional to use; ingestion only requires local directories specified
in the `sources:` list.
"""

import subprocess, sys, os
from pathlib import Path
import os, yaml, requests

CFG_PATH = os.environ.get("RAG_CONFIG", "config.yaml")
CFG = yaml.safe_load(open(CFG_PATH, "r"))

def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def fetch_github(entry: dict):
    repo = entry.get("repo")
    dest = Path(entry.get("dest", "")).resolve()
    depth = str(entry.get("depth", 1))
    branch = entry.get("branch")
    if not repo or not dest:
        print("[skip] invalid github entry")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        print(f"[git] updating {dest}")
        run(["git", "pull", "--ff-only"], cwd=str(dest))
    else:
        print(f"[git] cloning {repo} -> {dest}")
        cmd = ["git", "clone", "--depth", depth]
        if branch:
            cmd += ["--branch", branch]
        cmd += [repo, str(dest)]
        run(cmd)

def fetch_http_list(entry: dict):
    urls = entry.get("urls") or []
    dest_dir = Path(entry.get("dest_dir", "")).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        safe = url.replace("http://", "").replace("https://", "")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe)
        out = dest_dir / f"{safe}.html"
        try:
            print(f"[get] {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            out.write_text(r.text, encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[warn] failed {url}: {e}")

def fetch_http_file(entry: dict):
    url = entry.get("url")
    dest = Path(entry.get("dest", "")).resolve()
    if not url or not dest:
        print("[skip] invalid http_file entry")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[get] {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_text(r.text, encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[warn] failed {url}: {e}")

def main():
    tasks = CFG.get("fetch") or []
    for t in tasks:
        typ = (t.get("type") or "").lower()
        if typ == "github":
            fetch_github(t)
        elif typ == "http_list":
            fetch_http_list(t)
        elif typ == "http_file":
            fetch_http_file(t)
        else:
            print(f"[skip] unknown fetch type: {typ}")

if __name__ == "__main__":
    main()
