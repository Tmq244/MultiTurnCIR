from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    app_root: Path
    cache_dir: Path
    model_checkpoint: Path
    results_json: Path
    attr_dir: Path
    data_dir: Path
    images_dir: Path
    device: str
    gpu_id: int
    index_limit: int | None


def _parse_index_limit() -> int | None:
    raw = os.getenv("MFR_INDEX_LIMIT", "").strip()
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def _parse_gpu_id() -> int:
    raw = os.getenv("MFR_GPU_ID", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if value >= 0 else 3


def _resolve_cache_dir(project_root: Path, index_limit: int | None) -> Path:
    raw = os.getenv("MFR_CACHE_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path
    if index_limit is None:
        return project_root / "cache" / "cache_all"
    return project_root / "cache" / f"cache_bench_{index_limit}"


def get_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[2]
    app_root = Path(__file__).resolve().parents[1]

    index_limit = _parse_index_limit()
    cache_dir = _resolve_cache_dir(project_root, index_limit)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = os.getenv("MFR_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        device = "cuda"
    gpu_id = _parse_gpu_id()

    return AppConfig(
        project_root=project_root,
        app_root=app_root,
        cache_dir=cache_dir,
        model_checkpoint=project_root / "best_model.pth",
        results_json=project_root
        / "results/20260314_223831_combine_all_resnet152_encode_s3_b16_e150_lr6e-05_gpu1_combine_all_r152_sa3_lr6e5_e150.json",
        attr_dir=project_root / "attr",
        data_dir=project_root / "data",
        images_dir=project_root / "images",
        device=device,
        gpu_id=gpu_id,
        index_limit=index_limit,
    )
