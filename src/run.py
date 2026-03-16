from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run multiturn web app")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--index-limit", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cache_root = project_root / "cache"
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    os.environ["MFR_DEVICE"] = args.device
    if args.index_limit is None:
        os.environ.pop("MFR_INDEX_LIMIT", None)
        cache_dir = cache_root / "cache_all"
    else:
        os.environ["MFR_INDEX_LIMIT"] = str(args.index_limit)
        cache_dir = cache_root / f"cache_bench_{args.index_limit}"

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MFR_CACHE_DIR"] = str(cache_dir)

    uvicorn.run(
        "src.app.main:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
