import os, re, pathlib, argparse, yaml
from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify as md

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Convert <pre><code>..</code></pre> and <pre>..</pre> into fenced code blocks
    for pre in soup.find_all("pre"):
        code_text = pre.get_text("\n")
        fenced = f"\n```\n{code_text}\n```\n"
        pre.replace_with(NavigableString(fenced))
    # Inline <code>..</code> as backticked chunks
    for code in soup.find_all("code"):
        ct = code.get_text()
        code.replace_with(NavigableString(f"`{ct}`"))
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def process_dir(src_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    total = 0; converted = 0
    print(f"[normalize] src={src_dir} -> dst={out_dir}")
    for root, _, files in os.walk(src_dir):
        for f in files:
            p = pathlib.Path(root) / f
            rel = pathlib.Path(src_dir).resolve().joinpath("").as_posix()
            try:
                if p.suffix.lower() in [".html", ".htm"]:
                    out = html_to_text(p.read_text(encoding="utf-8", errors="ignore"))
                    converted += 1
                else:
                    # pass-through for code / text
                    out = p.read_text(encoding="utf-8", errors="ignore")
                dst = pathlib.Path(out_dir) / pathlib.Path(os.path.relpath(p, src_dir))
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(out, encoding="utf-8")
                total += 1
            except Exception as e:
                print(f"Skip {p}: {e}")
    print(f"[normalize] done: files={total} html_converted={converted}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="Source directory (HTML or text)")
    ap.add_argument("--dst", help="Destination directory for text")
    ap.add_argument("--from-config", action="store_true", help="Read normalize: tasks from RAG_CONFIG and run them")
    args = ap.parse_args()
    if args.from_config:
        cfg_path = os.environ.get("RAG_CONFIG", "config.yaml")
        cfg = yaml.safe_load(open(cfg_path, "r"))
        tasks = (cfg.get("normalize") or []) if isinstance(cfg, dict) else []
        if not tasks:
            print("[normalize] No normalize tasks found in config.")
        for i, t in enumerate(tasks, 1):
            src = t.get("src"); dst = t.get("dst")
            if not src or not dst:
                print(f"[normalize] Task {i}: missing src/dst; skipping")
                continue
            process_dir(src, dst)
    else:
        if not args.src or not args.dst:
            print("ERROR: --src and --dst required (or use --from-config)")
            raise SystemExit(2)
        process_dir(args.src, args.dst)
