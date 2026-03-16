from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from .model_service import ModelService


class RetrievalService:
    def __init__(self, model_service: ModelService, cache_dir: Path, index_limit: int | None = None) -> None:
        self.model_service = model_service
        self.cache_dir = cache_dir
        self.index_limit = index_limit

        self.ids_path = cache_dir / "index_ids.json"
        self.embeddings_path = cache_dir / "index_embeddings.npy"
        self.meta_path = cache_dir / "metadata.json"

        self.image_ids: list[str] = []
        self.index_embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.image_ids)

    def ensure_index(self) -> None:
        if self._load_cache():
            return
        self._build_cache()

    def _load_cache(self) -> bool:
        if not (self.ids_path.exists() and self.embeddings_path.exists() and self.meta_path.exists()):
            return False
        with self.meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with self.ids_path.open("r", encoding="utf-8") as handle:
            self.image_ids = json.load(handle)
        self.index_embeddings = np.load(self.embeddings_path)

        if self.index_embeddings.shape[0] != len(self.image_ids):
            return False

        cached_limit = metadata.get("index_limit") if isinstance(metadata, dict) else None
        if cached_limit != self.index_limit:
            # Rebuild cache when runtime index limit changes.
            return False

        if self.index_limit is None and cached_limit is None:
            # Handle legacy caches that were built with a small subset but had no recorded limit.
            total_images = len(self.model_service.all_image_ids())
            if len(self.image_ids) < total_images:
                return False

        return True

    def _build_cache(self) -> None:
        all_ids = self.model_service.all_image_ids()
        if self.index_limit is not None:
            all_ids = all_ids[: self.index_limit]

        embeddings = []
        valid_ids = []
        first_error = None
        for image_id in all_ids:
            try:
                vec = self.model_service.embed_index_image(image_id)
            except Exception as exc:
                if first_error is None:
                    first_error = (image_id, exc)
                continue
            embeddings.append(vec)
            valid_ids.append(image_id)

        if not embeddings:
            if first_error is None:
                raise RuntimeError("No index embeddings were generated.")
            image_id, exc = first_error
            raise RuntimeError(
                f"No index embeddings were generated. First failure on image_id={image_id}: {exc}"
            ) from exc

        matrix = np.stack(embeddings, axis=0).astype(np.float32)
        self.image_ids = valid_ids
        self.index_embeddings = matrix

        with self.ids_path.open("w", encoding="utf-8") as handle:
            json.dump(self.image_ids, handle)
        np.save(self.embeddings_path, self.index_embeddings)
        with self.meta_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "built_at": datetime.now().isoformat(),
                    "count": len(self.image_ids),
                    "index_limit": self.index_limit,
                    "checkpoint": str(self.model_service.cfg.model_checkpoint),
                },
                handle,
                indent=2,
            )

    def gallery(self, limit: int = 200, offset: int = 0, random_sample: bool = False) -> list[str]:
        self.ensure_index()
        if random_sample:
            count = max(0, min(limit, len(self.image_ids)))
            if count == 0:
                return []
            return random.sample(self.image_ids, count)
        return self.image_ids[offset : offset + limit]

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        self.ensure_index()
        assert self.index_embeddings is not None

        scores = np.matmul(self.index_embeddings, query_vec.reshape(-1, 1)).reshape(-1)
        top_k = max(1, min(top_k, scores.shape[0]))
        indices = np.argpartition(-scores, top_k - 1)[:top_k]
        indices = indices[np.argsort(-scores[indices])]

        return [(self.image_ids[idx], float(scores[idx])) for idx in indices]
