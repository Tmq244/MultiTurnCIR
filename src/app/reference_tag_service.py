from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class ReferenceTagService:
    """Loads ASIN tag mappings from attr json files and provides flattened tags."""

    def __init__(self, attr_dir: Path) -> None:
        self._attr_dir = attr_dir
        self._lock = Lock()
        self._loaded = False
        self._asin_to_tags: dict[str, list[str]] = {}

    def _iter_attr_files(self) -> list[Path]:
        preferred = [
            self._attr_dir / "asin2attr.all.train.new.json",
            self._attr_dir / "asin2attr.all.val.new.json",
        ]

        files: list[Path] = [path for path in preferred if path.exists()]
        for path in sorted(self._attr_dir.glob("asin2attr.*.json")):
            if path in files:
                continue
            files.append(path)
        return files

    @staticmethod
    def _flatten_tags(raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []

        result: list[str] = []
        for group in raw:
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, str):
                    continue
                tag = item.strip()
                if not tag:
                    continue
                if tag not in result:
                    result.append(tag)
        return result

    def _load_once(self) -> None:
        with self._lock:
            if self._loaded:
                return

            asin_to_tags: dict[str, list[str]] = {}
            for path in self._iter_attr_files():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    continue

                if not isinstance(payload, dict):
                    continue

                for asin, raw_tags in payload.items():
                    if not isinstance(asin, str):
                        continue
                    key = asin.strip().upper()
                    if not key:
                        continue

                    tags = self._flatten_tags(raw_tags)
                    if not tags:
                        continue

                    existing = asin_to_tags.setdefault(key, [])
                    for tag in tags:
                        if tag not in existing:
                            existing.append(tag)

            self._asin_to_tags = asin_to_tags
            self._loaded = True

    def get_tags(self, image_id: str, limit: int = 12) -> list[str]:
        self._load_once()
        key = image_id.strip().upper()
        tags = self._asin_to_tags.get(key, [])
        return tags[:limit]
