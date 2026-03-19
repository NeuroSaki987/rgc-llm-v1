from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from torch.utils.data import Dataset

from rgc_llm.config import DataConfig, Datum


class TextDataset(Dataset[Tuple[str, str]]):
    def __init__(self, samples: List[Datum]) -> None:
        self.samples = [(s.input, s.target) for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.samples[idx]


def _normalize_record(record: Dict[str, Any], text_field: str, target_field: str) -> Datum:
    if text_field not in record or target_field not in record:
        raise ValueError(f"Record missing required fields: {text_field!r}, {target_field!r}")
    meta = {k: v for k, v in record.items() if k not in {text_field, target_field}}
    return Datum(input=str(record[text_field]), target=str(record[target_field]), metadata=meta)


def load_records_from_path(path: str | Path, text_field: str = "input", target_field: str = "target", file_format: str = "auto") -> List[Datum]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    fmt = file_format.lower()
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        else:
            raise ValueError(f"Unsupported file extension for auto format: {suffix}")

    records: Iterable[Dict[str, Any]]
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    elif fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            if "samples" in raw and isinstance(raw["samples"], list):
                records = raw["samples"]
            elif "data" in raw and isinstance(raw["data"], list):
                records = raw["data"]
            else:
                raise ValueError("JSON object input must contain a 'samples' or 'data' list")
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError("Unsupported JSON structure")
    else:
        raise ValueError(f"Unsupported file format: {fmt}")

    return [_normalize_record(r, text_field=text_field, target_field=target_field) for r in records]


def build_dataset(cfg: DataConfig, split: str = "train") -> TextDataset:
    file_path = cfg.train_file if split == "train" else cfg.valid_file
    if file_path:
        samples = load_records_from_path(file_path, text_field=cfg.text_field, target_field=cfg.target_field, file_format=cfg.file_format)
    else:
        samples = cfg.toy_samples
    return TextDataset(samples)
