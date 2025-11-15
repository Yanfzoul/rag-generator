#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic website crawler (config/CLI-driven).

Features:
- Respects robots.txt (per domain)
- Domain allow/deny filters
- Depth- and per-domain-page limits
- Saves raw HTML to a filesystem-safe path
- Basic throttling per host

Usage (CLI):
  py ingest/crawl_web.py \
    --seeds https://example.com/docs https://another.site/page \
    --allow_domains example.com another.site \
    --depth 2 --per_domain_limit 200 \
    --out_dir ./data/crawl

Then optionally normalize HTML to text:
  py ingest/normalize_text.py --src ./data/crawl --dst ./data/crawl_txt

You can also mirror this via config by adding a commented example in config.yaml
and invoking this script from launch.sh.
"""

from __future__ import annotations
import argparse, time, json, re
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, Dict, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import urllib.robotparser as robotparser


def safe_name_from_url(url: str) -> Path:
    u = urlparse(url)
    base = (u.netloc + u.path)
    if base.endswith('/'):
        base += 'index'
    # keep a hint of query in the filename but avoid long names
    qhint = ''
    if u.query:
        qhint = '_' + re.sub(r'[^A-Za-z0-9._-]', '_', u.query)[:60]
    safe = re.sub(r'[^A-Za-z0-9._\-/]', '_', base)
    return Path(safe + qhint + '.html')


@dataclass
class CrawlConfig:
    seeds: Tuple[str, ...]
    allow_domains: Set[str]
    deny_domains: Set[str]
    depth: int
    per_domain_limit: int
    out_dir: Path
    user_agent: str
    delay_sec: float
    respect_robots: bool


def same_domain_or_allowed(url: str, allow_domains: Set[str], deny_domains: Set[str]) -> bool:
    host = urlparse(url).netloc.lower()
    if any(host.endswith(d.lower()) for d in deny_domains if d):
        return False
    if not allow_domains:
        return True
    return any(host.endswith(d.lower()) for d in allow_domains if d)


def load_robots_for(host: str, ua: str, cache: Dict[str, robotparser.RobotFileParser]):
    if host in cache:
        return cache[host]
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(f"https://{host}/robots.txt")
        rp.read()
    except Exception:
        pass
    cache[host] = rp
    return rp


def extract_links(url: str, html: str) -> Iterable[str]:
    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception:
        soup = BeautifulSoup(html, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a.get('href')
        if not href:
            continue
        abs_url = urljoin(url, href)
        # strip fragments for dedupe
        abs_url, _ = urldefrag(abs_url)
        if abs_url.startswith('http://') or abs_url.startswith('https://'):
            yield abs_url


def crawl(cfg: CrawlConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / 'seeds.json').write_text(json.dumps(list(cfg.seeds), indent=2), encoding='utf-8')

    q = deque([(s, 0) for s in cfg.seeds])
    seen: Set[str] = set()
    saved: Set[str] = set()
    per_domain_count: Dict[str, int] = defaultdict(int)
    last_fetch: Dict[str, float] = defaultdict(lambda: 0.0)
    robots_cache: Dict[str, robotparser.RobotFileParser] = {}

    sess = requests.Session()
    sess.headers.update({'User-Agent': cfg.user_agent})

    while q:
        url, d = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        if not same_domain_or_allowed(url, cfg.allow_domains, cfg.deny_domains):
            continue

        host = urlparse(url).netloc
        if cfg.per_domain_limit and per_domain_count[host] >= cfg.per_domain_limit:
            continue

        if cfg.respect_robots:
            rp = load_robots_for(host, cfg.user_agent, robots_cache)
            try:
                if hasattr(rp, 'can_fetch') and not rp.can_fetch(cfg.user_agent, url):
                    continue
            except Exception:
                pass

        # throttle per host
        now = time.time()
        dt = now - last_fetch[host]
        if dt < cfg.delay_sec:
            time.sleep(cfg.delay_sec - dt)

        try:
            r = sess.get(url, timeout=20)
            ct = (r.headers.get('content-type') or '').lower()
            if r.status_code >= 400:
                continue
            if ('text/html' not in ct) and not url.endswith('.html'):
                continue
            html = r.text
        except Exception:
            continue
        finally:
            last_fetch[host] = time.time()

        # save
        rel = safe_name_from_url(url)
        out_path = cfg.out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(html, encoding='utf-8', errors='ignore')
            saved.add(url)
            per_domain_count[host] += 1
            print(f"[saved] {url} -> {out_path}")
        except Exception:
            pass

        # frontier
        if d < cfg.depth:
            try:
                for nxt in extract_links(url, html):
                    if nxt not in seen:
                        q.append((nxt, d + 1))
            except Exception:
                pass

    # manifest
    (cfg.out_dir / 'manifest.json').write_text(
        json.dumps({
            'saved': len(saved),
            'seen': len(seen),
            'per_domain_count': dict(per_domain_count),
        }, indent=2),
        encoding='utf-8'
    )
    print(f"Done. Saved {len(saved)} pages to {cfg.out_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-config', action='store_true', help='Read crawl tasks from RAG_CONFIG (config.yaml) and execute them')
    ap.add_argument('--seeds', nargs='+', help='One or more seed URLs')
    ap.add_argument('--allow_domains', nargs='*', default=[], help='Limit crawl to these domains (suffix match)')
    ap.add_argument('--deny_domains', nargs='*', default=[], help='Denylist domains (suffix match)')
    ap.add_argument('--depth', type=int, default=1, help='Crawl depth from seeds')
    ap.add_argument('--per_domain_limit', type=int, default=200, help='Max pages per domain')
    ap.add_argument('--out_dir', default='./data/crawl', help='Output directory for HTML')
    ap.add_argument('--user_agent', default='GenericRAGCrawler/1.0', help='User-Agent string')
    ap.add_argument('--delay_sec', type=float, default=0.5, help='Delay between requests to same host')
    ap.add_argument('--no_robots', action='store_true', help='Do not check robots.txt')
    args = ap.parse_args()
    # From-config mode will be handled in __main__
    seeds = tuple(args.seeds or [])
    return CrawlConfig(
        seeds=seeds,
        allow_domains=set(args.allow_domains or []),
        deny_domains=set(args.deny_domains or []),
        depth=int(args.depth),
        per_domain_limit=int(args.per_domain_limit),
        out_dir=Path(args.out_dir),
        user_agent=str(args.user_agent),
        delay_sec=float(args.delay_sec),
        respect_robots=(not args.no_robots),
    ), args.from_config


if __name__ == '__main__':
    parsed, from_cfg = parse_args()
    if from_cfg:
        # Read tasks from config
        import yaml, os as _os
        cfg_path = _os.environ.get('RAG_CONFIG', 'config.yaml')
        cfg = yaml.safe_load(open(cfg_path, 'r'))
        tasks = (cfg.get('crawl') or []) if isinstance(cfg, dict) else []
        if not tasks:
            print('[crawl] No crawl tasks found in config; nothing to do.')
        for i, t in enumerate(tasks, 1):
            seeds = tuple(t.get('seeds') or [])
            if not seeds:
                print(f"[crawl] Task {i}: missing seeds; skipping")
                continue
            cc = CrawlConfig(
                seeds=seeds,
                allow_domains=set(t.get('allow_domains') or []),
                deny_domains=set(t.get('deny_domains') or []),
                depth=int(t.get('depth', 1)),
                per_domain_limit=int(t.get('per_domain_limit', 200)),
                out_dir=Path(t.get('out_dir', './data/crawl')),
                user_agent=str(t.get('user_agent', 'GenericRAGCrawler/1.0')),
                delay_sec=float(t.get('delay_sec', 0.5)),
                respect_robots=(not bool(t.get('no_robots', False))),
            )
            print(f"[crawl] Task {i}/{len(tasks)}: seeds={len(seeds)} out={cc.out_dir}")
            crawl(cc)
    else:
        if not parsed.seeds:
            print('ERROR: --seeds is required (or use --from-config).')
            raise SystemExit(2)
        crawl(parsed)
