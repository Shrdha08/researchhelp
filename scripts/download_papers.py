"""Download the evaluation corpus listed in evaluation/datasets/papers.yaml into data/papers/.

Usage (from repo root):  py -m uv run --project backend python scripts/download_papers.py
"""

from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "evaluation" / "datasets" / "papers.yaml"
OUT = ROOT / "data" / "papers"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    papers = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))["papers"]
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for p in papers:
            dest = OUT / f"{p['short_name']}.pdf"
            if dest.exists():
                print(f"skip  {dest.name}")
                continue
            resp = client.get(f"https://arxiv.org/pdf/{p['arxiv_id']}")
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"saved {dest.name} ({len(resp.content) // 1024} KB)")


if __name__ == "__main__":
    main()
