from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

EPS = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[1]

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_QUARTER_RE = re.compile(r"FY(\d{4})_Q([1-4])")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "been",
    "because",
    "before",
    "between",
    "continue",
    "continued",
    "driven",
    "during",
    "expect",
    "from",
    "growth",
    "have",
    "higher",
    "including",
    "into",
    "long",
    "lower",
    "more",
    "near",
    "quarter",
    "reported",
    "revenue",
    "signals",
    "similar",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "which",
    "with",
}


def is_missing(value: Any) -> bool:
    try:
        return value is None or bool(np.isnan(value))
    except Exception:
        return value is None or str(value).strip().lower() in {"", "nan", "none", "null"}


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if is_missing(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def quarter_key(quarter: str) -> int:
    match = _QUARTER_RE.fullmatch(str(quarter or "").strip())
    if not match:
        return -1
    return int(match.group(1)) * 10 + int(match.group(2))


def quarter_number(quarter: str) -> int:
    match = _QUARTER_RE.fullmatch(str(quarter or "").strip())
    if not match:
        return 0
    return int(match.group(2))


def fiscal_year(quarter: str) -> int:
    match = _QUARTER_RE.fullmatch(str(quarter or "").strip())
    if not match:
        return 0
    return int(match.group(1))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_repo_path(raw_path: Any) -> Optional[Path]:
    if is_missing(raw_path):
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "smape": float("nan"),
        }
    err = yp - yt
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(abs_err / np.maximum(np.abs(yt), EPS)))
    smape = float(np.mean(2.0 * abs_err / np.maximum(np.abs(yt) + np.abs(yp), EPS)))
    return {
        "n": int(yt.size),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
    }


def tokenize_preview(text: Any) -> List[str]:
    raw = str(text or "").lower()
    tokens = [tok for tok in _TOKEN_RE.findall(raw) if tok not in _STOPWORDS]
    return sorted(set(tokens))


def log_signed_transform(value: float) -> float:
    value = safe_float(value, 0.0)
    if not np.isfinite(value):
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def cosine_from_dicts(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    left_vec = np.array([log_signed_transform(left.get(key, 0.0)) for key in keys], dtype=float)
    right_vec = np.array([log_signed_transform(right.get(key, 0.0)) for key in keys], dtype=float)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denom <= EPS:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denom)


def weighted_jaccard(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numer = 0.0
    denom = 0.0
    for key in keys:
        left_val = max(0.0, safe_float(left.get(key, 0.0), 0.0))
        right_val = max(0.0, safe_float(right.get(key, 0.0), 0.0))
        numer += min(left_val, right_val)
        denom += max(left_val, right_val)
    if denom <= EPS:
        return 0.0
    return float(numer / denom)


def token_jaccard(left_tokens: Iterable[str], right_tokens: Iterable[str]) -> float:
    left = set(left_tokens)
    right = set(right_tokens)
    if not left and not right:
        return 0.0
    return float(len(left & right) / max(len(left | right), 1))


def softmax_weights(scores: Sequence[float], temperature: float) -> List[float]:
    if not scores:
        return []
    temp = max(float(temperature), EPS)
    score_arr = np.asarray(scores, dtype=float) / temp
    score_arr = score_arr - np.max(score_arr)
    weights = np.exp(score_arr)
    total = float(np.sum(weights))
    if total <= EPS:
        return [1.0 / len(scores)] * len(scores)
    return [float(val / total) for val in weights]


def clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))
