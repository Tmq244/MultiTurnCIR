from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .config import AppConfig


def _add_project_to_path(project_root: Path) -> None:
    import sys

    root_str = str(project_root)
    src_str = str(project_root / "src")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _normalize_tag_groups(tags: Any, size: int) -> list[list[str]]:
    groups = list(tags) if tags is not None else []
    normalized: list[list[str]] = []
    for idx in range(size):
        group = groups[idx] if idx < len(groups) and groups[idx] is not None else []
        if isinstance(group, str):
            group = [group] if group else []
        normalized.append([str(token).strip() for token in group if str(token).strip()])
    return normalized


def _format_tag_strings(tag_groups: list[list[str]], size: int = 6) -> list[str]:
    groups = _normalize_tag_groups(tag_groups, size)
    return ["[CLS]" + " [SEP] ".join(group) for group in groups]


def _build_text_corpus_from_training_pipeline(project_root: Path, data_dir: Path) -> list[str]:
    # Mirror preprocess/dataset_tag.py text collection:
    # - all caption turns
    # - reference tag tokens (first 5 groups after dropping type group)
    texts: list[str] = []

    for path in sorted(data_dir.glob("*.train.json")):
        target = path.name.split(".")[0]
        attr_path = project_root / "attr" / f"asin2attr.{target}.train.new.json"

        try:
            with path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        except Exception:
            continue

        tags_dict: dict[str, Any] = {}
        if attr_path.exists():
            try:
                with attr_path.open("r", encoding="utf-8") as handle:
                    raw_tags = json.load(handle)
                if isinstance(raw_tags, dict):
                    tags_dict = raw_tags
            except Exception:
                tags_dict = {}

        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            references = record.get("reference", [])
            if not isinstance(references, list):
                continue
            for ref in references:
                # Expected schema: [image_id, [caption_turns...], target_id]
                if isinstance(ref, list) and len(ref) > 1 and isinstance(ref[1], list):
                    for token in ref[1]:
                        s = str(token).strip()
                        if s:
                            texts.append(s)

                # dataset_tag uses c_id = references[2], then tags_dict[c_id][1:] and first 5 groups
                if isinstance(ref, list) and len(ref) > 2:
                    c_id = str(ref[2])
                    groups = tags_dict.get(c_id, [])
                    if isinstance(groups, list):
                        groups = groups[1:]
                    else:
                        groups = []
                    for group in _normalize_tag_groups(groups, 5):
                        for token in group:
                            s = str(token).strip()
                            if s:
                                texts.append(s)

    # Keep deterministic order while de-duplicating.
    return list(dict.fromkeys(texts))


def _resolve_model_args(results_json: Path) -> dict[str, Any]:
    # These defaults match the shipped checkpoint name:
    # combine_all_r152_sa3_lr6e5_e150/best_model.pth
    defaults: dict[str, Any] = {
        "fdims": 2048,
        "normalize_scale": 5.0,
        "lr": 6e-05,
        "lrp": 0.48,
        "stack_num": 3,
        "max_turn_len": 4,
        "backbone": "resnet152",
        "text_method": "encode",
        "image_size": 224,
    }

    if not results_json.exists():
        return defaults

    with results_json.open("r", encoding="utf-8") as handle:
        run_meta = json.load(handle)
    raw_args = run_meta.get("args", {}) if isinstance(run_meta, dict) else {}
    if not isinstance(raw_args, dict):
        return defaults

    merged = {**defaults, **raw_args}
    return merged


class ModelService:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{cfg.gpu_id}" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.set_device(self.device)
        self.model = None
        self.max_turn_len = 4
        self._transform = None
        self._id_to_tags: dict[str, list[list[str]]] = {}
        self._load_metadata()
        self._load_model()

    def _load_metadata(self) -> None:
        targets = ["all", "dress", "shirt", "toptee"]
        for target in targets:
            path = self.cfg.attr_dir / f"asin2attr.{target}.val.new.json"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            for image_id, groups in raw.items():
                self._id_to_tags[image_id] = _normalize_tag_groups(groups, 6)

    def _load_model(self) -> None:
        _add_project_to_path(self.cfg.project_root)
        os.environ.setdefault("MFR_BACKBONE_PRETRAINED", "0")
        from Model.cross_attention import Combine
        from preprocess.transform import PaddedResize

        args = _resolve_model_args(self.cfg.results_json)
        self.max_turn_len = int(args["max_turn_len"])

        model_args = SimpleNamespace(
            fdims=int(args["fdims"]),
            normalize_scale=float(args["normalize_scale"]),
            lr=float(args["lr"]),
            lrp=float(args["lrp"]),
            stack_num=int(args["stack_num"]),
            max_turn_len=int(args["max_turn_len"]),
        )

        texts = _build_text_corpus_from_training_pipeline(self.cfg.project_root, self.cfg.data_dir)
        self.model = Combine(
            args=model_args,
            backbone=args["backbone"],
            texts=texts,
            stack_num=model_args.stack_num,
            max_turn_len=model_args.max_turn_len,
            normalize_scale=model_args.normalize_scale,
            text_method=args["text_method"],
            fdims=model_args.fdims,
            fc_arch="A",
            init_with_glove=False,
        )

        checkpoint = torch.load(self.cfg.model_checkpoint, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        self._align_text_vocab_with_checkpoint(state_dict)

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            self._load_state_dict_compat(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self._transform = T.Compose(
            [
                PaddedResize(int(args["image_size"])),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _align_text_vocab_with_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> None:
        key = "model.text_encoder.embedding_layer.embedding.weight"
        if key not in state_dict:
            return

        try:
            text_encoder = self.model.model["text_encoder"]
            vocab = text_encoder.vocab
            embedding = text_encoder.embedding_layer.embedding
        except Exception:
            return

        target_rows = int(state_dict[key].shape[0])
        current_rows = int(embedding.weight.shape[0])

        if current_rows <= target_rows:
            return

        # Keep token ids [0, target_rows) so ids line up with checkpoint rows.
        tokens_to_drop = [token for token, idx in vocab.word2id.items() if int(idx) >= target_rows]
        for token in tokens_to_drop:
            vocab.word2id.pop(token, None)
            vocab.wordcount.pop(token, None)

        trimmed = torch.nn.Embedding(target_rows, int(embedding.weight.shape[1]))
        with torch.no_grad():
            trimmed.weight.copy_(embedding.weight[:target_rows])
        text_encoder.embedding_layer.embedding = trimmed

    def _load_state_dict_compat(self, state_dict: dict[str, torch.Tensor]) -> None:

        # Some checkpoints are trained with a different text vocabulary size.
        # Load exact-shape tensors directly and partially copy row/column-overlap
        # for mismatched tensors (notably embedding matrices) to preserve semantics.
        model_state = self.model.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key not in model_state:
                continue

            target = model_state[key]
            if target.shape == value.shape:
                filtered_state[key] = value

        missing_keys = [k for k in model_state.keys() if k not in filtered_state]
        for key in missing_keys:
            if key not in state_dict:
                continue

            src = state_dict[key]
            dst = model_state[key].clone()

            if src.ndim != dst.ndim:
                continue

            # Copy overlapped block on every dimension.
            slices = tuple(slice(0, min(int(src.shape[i]), int(dst.shape[i]))) for i in range(dst.ndim))
            if not slices:
                continue
            dst[slices] = src[slices]
            filtered_state[key] = dst

        self.model.load_state_dict(filtered_state, strict=False)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def all_image_ids(self) -> list[str]:
        ids = [path.stem for path in self.cfg.images_dir.glob("*.jpg")]
        ids.sort()
        return ids

    def image_exists(self, image_id: str) -> bool:
        return (self.cfg.images_dir / f"{image_id}.jpg").exists()

    def image_path(self, image_id: str) -> Path:
        return self.cfg.images_dir / f"{image_id}.jpg"

    def tag_strings(self, image_id: str) -> list[str]:
        tag_groups = self._id_to_tags.get(image_id, _normalize_tag_groups([], 6))
        return _format_tag_strings(tag_groups, 6)

    def _load_image_tensor(self, image_id: str) -> torch.Tensor:
        image_path = self.image_path(image_id)
        with image_path.open("rb") as handle:
            image = Image.open(handle).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        return tensor

    def embed_index_image(self, image_id: str) -> np.ndarray:
        image_tensor = self._load_image_tensor(image_id)
        tags = [[text] for text in self.tag_strings(image_id)]
        with torch.no_grad():
            feat = self.model.get_original_combined_feature(tags, image_tensor)
        return feat[0].detach().cpu().numpy().astype(np.float32)

    def embed_query(self, reference_image_id: str, turns: list[str]) -> np.ndarray:
        turns = turns[-self.max_turn_len :]
        turns_padded = turns + [""] * (self.max_turn_len - len(turns))

        ref_image_tensor = self._load_image_tensor(reference_image_id)
        ref_tags = [[text] for text in self.tag_strings(reference_image_id)]

        dummy_img = torch.zeros_like(ref_image_tensor)
        dummy_cls = torch.zeros((1,), dtype=torch.long, device=self.device)
        dummy_tag = [""] * 5

        query_input = []
        for turn_text in turns_padded:
            query_input.append((dummy_img, dummy_cls, [turn_text], dummy_tag))

        query_input.append((ref_image_tensor, dummy_cls, [reference_image_id], ref_tags))
        query_input.append(("session_key", torch.tensor([len(turns)], device=self.device), reference_image_id))

        with torch.no_grad():
            feat = self.model(query_input)[0]
        return feat[0].detach().cpu().numpy().astype(np.float32)
