from __future__ import annotations

import json
import random
from pathlib import Path
from threading import Lock


class ReferenceTextService:
    """Loads reference texts from dataset JSON files and provides two suggestions per image."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._lock = Lock()
        self._loaded = False
        self._asin_to_texts: dict[str, list[str]] = {}
        self._fallback_pool: list[str] = []

    def _iter_dataset_files(self) -> list[Path]:
        preferred = self._data_dir / "all.train.json"
        files: list[Path] = []
        if preferred.exists():
            files.append(preferred)

        for path in sorted(self._data_dir.glob("*.json")):
            if path == preferred:
                continue
            files.append(path)
        return files

    @staticmethod
    def _clean_texts(values: object) -> list[str]:
        if not isinstance(values, list):
            return []

        cleaned: list[str] = []
        for item in values:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            if text not in cleaned:
                cleaned.append(text)
        return cleaned

    def _load_once(self) -> None:
        with self._lock:
            if self._loaded:
                return

            asin_to_texts: dict[str, list[str]] = {}
            fallback_seen: set[str] = set()
            fallback_pool: list[str] = []

            for path in self._iter_dataset_files():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    continue

                if not isinstance(payload, list):
                    continue

                for sample in payload:
                    if not isinstance(sample, dict):
                        continue
                    references = sample.get("reference")
                    if not isinstance(references, list):
                        continue

                    for ref in references:
                        if not isinstance(ref, list) or len(ref) < 3:
                            continue

                        texts = self._clean_texts(ref[1])
                        asin = ref[2]
                        if not isinstance(asin, str):
                            continue
                        asin = asin.strip().upper()
                        if not asin or not texts:
                            continue

                        existing = asin_to_texts.setdefault(asin, [])
                        for text in texts:
                            if text not in existing:
                                existing.append(text)
                            if text not in fallback_seen:
                                fallback_seen.add(text)
                                fallback_pool.append(text)

            self._asin_to_texts = asin_to_texts
            self._fallback_pool = fallback_pool
            self._loaded = True

    def get_suggestions(self, image_id: str, count: int = 2) -> list[str]:
        self._load_once()

        normalized_id = image_id.strip().upper()
        suggestions = list(self._asin_to_texts.get(normalized_id, []))

        if len(suggestions) >= count:
            return suggestions[:count]

        if self._fallback_pool:
            picks = random.sample(self._fallback_pool, k=min(count, len(self._fallback_pool)))
            return picks

        return [
            "is similar in style",
            "has a different color tone",
        ][:count]
