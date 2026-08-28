from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


EPS = 1e-9


def resolve_repo_path(path_value: Any, project_root: str) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (Path(project_root) / raw).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def quarter_key(value: str) -> Tuple[int, int]:
    text = str(value or "")
    if not text.startswith("FY") or "_Q" not in text:
        return (0, 0)
    try:
        return (int(text[2:6]), int(text.split("_Q", 1)[1]))
    except Exception:
        return (0, 0)


def in_window(quarter: str, start_q: str, end_q: str) -> bool:
    key = quarter_key(quarter)
    return quarter_key(start_q) <= key <= quarter_key(end_q)


def parse_mapping(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): safe_float(v, 0.0) for k, v in value.items()}
    text = str(value or "").strip()
    if not text or text in {"{}", "nan", "None"}:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            payload = loader(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            return {str(k): safe_float(v, 0.0) for k, v in payload.items()}
    return {}


def sort_mapping_by_abs_value(mapping: Mapping[str, float]) -> List[Tuple[str, float]]:
    return sorted(((str(k), float(v)) for k, v in mapping.items()), key=lambda item: abs(item[1]), reverse=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return Path("")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
