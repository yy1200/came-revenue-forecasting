#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from native_evidence_forecaster.common import EPS, load_jsonl, quarter_key, resolve_repo_path, safe_float, sanitize_for_json, write_json
from native_evidence_forecaster.run_backbone_v2 import _load_best_stat, _load_json, _metrics, _safe_log
from native_evidence_forecaster.run_native_cards_v1 import (
    CANONICAL_FACTORS,
    METADATA_PROXY_FEATURES,
    SEMANTIC_FEATURE_NAMES,
    _aggregate_group,
    _apply_safety_guard,
    _card_influence,
    _clip,
    _fit_zero_intercept_ridge,
    _predict_zero_intercept_ridge,
    _sign_num,
    _top_feature_contribs,
)


DEFAULT_EXPERIMENT_CONFIG = "replay_inputs/retained_336/experiment.json"
DEFAULT_CARD_TABLE_JSONL = "replay_inputs/retained_336/normalized_native_forward_cards.jsonl"
DEFAULT_BACKBONE_CSV = "replay_inputs/retained_336/native_backbone.csv"

RAW_CARD_FEATURE_NAMES = [
    "fwd_count_log",
    "fwd_signed_mass",
    "fwd_abs_mass",
    "fwd_conflict_ratio",
    "fwd_persistent_share",
    "fwd_event_share",
    "fwd_claim_share",
    "fwd_top1_shareweighted_signed",
    "fwd_other_shareweighted_signed",
    "realized_count_log",
    "realized_signed_mass",
    "realized_abs_mass",
    "realized_conflict_ratio",
    "delta_fwd_count_log",
    "delta_fwd_signed_mass",
    "delta_fwd_abs_mass",
    "delta_realized_count_log",
    "delta_realized_signed_mass",
] + [f"factor_signed__{factor}" for factor in CANONICAL_FACTORS] + [f"factor_abs__{factor}" for factor in CANONICAL_FACTORS]

FACTOR_DELTA_FEATURE_NAMES = [
    "fwd_signed_mass",
    "fwd_abs_mass",
    "fwd_conflict_ratio",
    "fwd_top1_shareweighted_signed",
    "fwd_other_shareweighted_signed",
] + [f"factor_signed__{factor}" for factor in CANONICAL_FACTORS] + [f"factor_abs__{factor}" for factor in CANONICAL_FACTORS]

GATE_FEATURE_NAMES = [
    "gate_raw_delta",
    "gate_guard_delta",
    "gate_sem_delta",
    "gate_delta_gap_raw_sem",
    "gate_fwd_conflict_ratio",
    "gate_fwd_abs_mass",
    "gate_fwd_count_log",
    "gate_top1_share",
    "gate_guidance_numeric_available",
    "gate_guidance_score_norm",
    "gate_guid_band_ratio",
    "gate_guidance_lock",
    "gate_same_quarter_support",
    "gate_segment_count",
    "gate_backbone_uncertainty",
    "gate_backbone_error_recent",
    "gate_backbone_error_same_quarter",
    "gate_memory_pred_delta",
    "gate_memory_support",
    "gate_memory_consistency",
    "gate_memory_sign_match_raw",
    "gate_metadata_proxy_flag",
    "gate_semantic_clarity",
    "gate_raw_x_conflict",
    "gate_raw_x_uncertainty",
    "gate_raw_x_weak_guidance",
    "gate_sem_x_clarity",
    "gate_mem_x_support",
]


def _quarter_number(value: str) -> int:
    key = quarter_key(value)
    return int(key[1]) if key != (0, 0) else 0


def _guidance_lock(row: Mapping[str, Any]) -> float:
    lock = (
        float(safe_float(row.get("guidance_numeric_available"), 0.0))
        * float(safe_float(row.get("guidance_score_norm"), 0.0))
        * max(0.0, 1.0 - float(safe_float(row.get("guid_band_ratio"), 0.0)) / 0.15)
    )
    return float(_clip(lock if math.isfinite(lock) else 0.0, 0.0, 1.0))


def _mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _build_backbone_uncertainty_state(
    history_rows: Sequence[Mapping[str, Any]],
    current_q: int,
    current_guidance: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    errors = [safe_float(item.get("backbone_abs_log_error"), float("nan")) for item in history_rows]
    overall_mean = _mean_or_nan(errors)
    recent_window = max(int(args.backbone_uncertainty_recent_window), 1)
    recent_mean = _mean_or_nan(errors[-recent_window:]) if errors else float("nan")
    same_q = [
        safe_float(item.get("backbone_abs_log_error"), float("nan"))
        for item in history_rows
        if _quarter_number(str(item.get("target_fiscal_quarter") or "")) == int(current_q)
    ]
    same_guidance = [
        safe_float(item.get("backbone_abs_log_error"), float("nan"))
        for item in history_rows
        if str(item.get("guidance_availability") or "none") == str(current_guidance or "none")
    ]
    same_q_mean = _mean_or_nan(same_q)
    same_guidance_mean = _mean_or_nan(same_guidance)

    proxy_parts = [overall_mean, recent_mean]
    if len([v for v in same_q if math.isfinite(v)]) >= int(args.backbone_same_quarter_min):
        proxy_parts.append(same_q_mean)
    if len([v for v in same_guidance if math.isfinite(v)]) >= int(args.backbone_same_guidance_min):
        proxy_parts.append(same_guidance_mean)
    proxy_mean = _mean_or_nan(proxy_parts)
    uncertainty = 0.0
    if math.isfinite(proxy_mean):
        uncertainty = _clip(proxy_mean / max(float(args.backbone_uncertainty_tau), EPS), 0.0, 1.0)
    return {
        "backbone_error_overall_abs_log": float(0.0 if not math.isfinite(overall_mean) else overall_mean),
        "backbone_error_recent_abs_log": float(0.0 if not math.isfinite(recent_mean) else recent_mean),
        "backbone_error_same_quarter_abs_log": float(0.0 if not math.isfinite(same_q_mean) else same_q_mean),
        "backbone_error_same_guidance_abs_log": float(0.0 if not math.isfinite(same_guidance_mean) else same_guidance_mean),
        "backbone_uncertainty": float(uncertainty),
    }


def _predict_zero_intercept_history(
    train_rows: Sequence[Mapping[str, Any]],
    x_map: Mapping[str, float],
    feature_names: Sequence[str],
    target_col: str,
    alpha: float,
    delta_cap_quantile: float,
    shrink_k: float,
) -> Dict[str, Any]:
    train_count = int(len(train_rows))
    if train_count == 0:
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": 0,
            "support": 0.0,
            "top_contribs": [],
            "coefs_by_name": {},
        }
    x_train = np.asarray([[float(safe_float(row.get(name), 0.0)) for name in feature_names] for row in train_rows], dtype=float)
    y_train = np.asarray([float(safe_float(row.get(target_col), float("nan"))) for row in train_rows], dtype=float)
    mask = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    x_train = x_train[mask]
    y_train = y_train[mask]
    if len(y_train) == 0:
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": 0,
            "support": 0.0,
            "top_contribs": [],
            "coefs_by_name": {},
        }
    model = _fit_zero_intercept_ridge(x_train, y_train, float(alpha))
    x = np.asarray([float(safe_float(x_map.get(name), 0.0)) for name in feature_names], dtype=float)
    pred_raw = _predict_zero_intercept_ridge(model, x)
    finite_abs = np.abs(y_train[np.isfinite(y_train)])
    if finite_abs.size:
        delta_cap = float(np.quantile(finite_abs, float(delta_cap_quantile)))
        if math.isfinite(delta_cap) and delta_cap > 0.0:
            pred_raw = _clip(pred_raw, -delta_cap, delta_cap)
    support = float(len(y_train) / max(len(y_train) + float(shrink_k), EPS))
    pred = float(pred_raw * support)
    coefs_by_name = {
        name: float(val)
        for name, val in zip(feature_names, np.asarray(model["coefs"], dtype=float) / np.asarray(model["stds"], dtype=float))
    }
    return {
        "pred": pred,
        "pred_raw": float(pred_raw),
        "train_count": int(len(y_train)),
        "support": support,
        "top_contribs": _top_feature_contribs(model, x, feature_names, top_k=6),
        "coefs_by_name": coefs_by_name,
    }


def _factor_signed_vector(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(safe_float(row.get(f"factor_signed__{factor}"), 0.0)) for factor in CANONICAL_FACTORS], dtype=float)


def _factor_abs_vector(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(safe_float(row.get(f"factor_abs__{factor}"), 0.0)) for factor in CANONICAL_FACTORS], dtype=float)


def _memory_pair_score(current_row: Mapping[str, Any], past_row: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    current_signed = _factor_signed_vector(current_row)
    past_signed = _factor_signed_vector(past_row)
    current_abs = _factor_abs_vector(current_row)
    past_abs = _factor_abs_vector(past_row)
    signed_norm = float(np.linalg.norm(current_signed) * np.linalg.norm(past_signed))
    signed_align = float((current_signed @ past_signed) / max(signed_norm, EPS)) if signed_norm > EPS else 0.0
    abs_overlap = float(1.0 - np.mean(np.abs(current_abs - past_abs) / np.maximum(current_abs + past_abs, 1.0))) if current_abs.size else 0.0
    same_quarter = 1.0 if int(safe_float(current_row.get("target_fiscal_q"), 0.0)) == int(safe_float(past_row.get("target_fiscal_q"), 0.0)) else 0.0
    guidance_match = 1.0 if str(current_row.get("guidance_availability") or "none") == str(past_row.get("guidance_availability") or "none") else 0.0
    top1_gap = abs(safe_float(current_row.get("segment_share_top1"), 0.0) - safe_float(past_row.get("segment_share_top1"), 0.0))
    conflict_gap = abs(safe_float(current_row.get("fwd_conflict_ratio"), 0.0) - safe_float(past_row.get("fwd_conflict_ratio"), 0.0))
    score = 1.20 * signed_align + 0.55 * abs_overlap + 0.20 * same_quarter + 0.10 * guidance_match - 0.45 * top1_gap - 0.15 * conflict_gap
    diag = {
        "signed_align": float(signed_align),
        "abs_overlap": float(abs_overlap),
        "same_quarter": float(same_quarter),
        "guidance_match": float(guidance_match),
        "top1_gap": float(top1_gap),
        "conflict_gap": float(conflict_gap),
    }
    return float(score), diag


def _softmax_weights(values: Sequence[float], temperature: float) -> List[float]:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return []
    temp = max(float(temperature), EPS)
    shifted = vals - float(np.max(vals))
    weights = np.exp(shifted / temp)
    total = float(np.sum(weights))
    if total <= EPS:
        return [1.0 / len(vals)] * len(vals)
    return [float(v / total) for v in weights]


def _memory_preview(cards: Sequence[Mapping[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for card in list(cards)[:top_k]:
        preview.append(
            {
                "instance_id": str(card.get("instance_id") or ""),
                "segment": str(card.get("segment") or ""),
                "canonical_factor": str(card.get("canonical_factor") or ""),
                "polarity": str(card.get("polarity") or ""),
                "source_text_sha256": str(card.get("source_text_sha256") or card.get("evidence_sha256") or ""),
                "release_status": str(card.get("release_status") or "quote_withheld_third_party"),
            }
        )
    return preview


def _memory_diag(current_row: Mapping[str, Any], history_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    if len(history_rows) < int(args.memory_min_train):
        return {
            "available": False,
            "pred_delta": 0.0,
            "support": 0.0,
            "consistency": 0.0,
            "top_matches": [],
        }
    scored: List[Tuple[float, Mapping[str, Any], Dict[str, float]]] = []
    for past in history_rows:
        target = safe_float(past.get("target_delta_log"), float("nan"))
        if not math.isfinite(target):
            continue
        score, diag = _memory_pair_score(current_row, past)
        scored.append((score, past, diag))
    if len(scored) < int(args.memory_min_train):
        return {
            "available": False,
            "pred_delta": 0.0,
            "support": 0.0,
            "consistency": 0.0,
            "top_matches": [],
        }
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: int(args.memory_top_k)]
    weights = _softmax_weights([item[0] for item in top], float(args.memory_temperature))
    deltas = [safe_float(item[1].get("target_delta_log"), 0.0) for item in top]
    pred_mean = float(sum(weight * delta for weight, delta in zip(weights, deltas)))
    abs_mass = float(sum(weight * abs(delta) for weight, delta in zip(weights, deltas)))
    consistency = float(abs(pred_mean) / max(abs_mass, EPS)) if abs_mass > EPS else 0.0
    weight_sq_sum = float(sum(weight ** 2 for weight in weights))
    neff = 0.0 if weight_sq_sum <= EPS else 1.0 / weight_sq_sum
    overlap_mean = float(sum(weight * item[2].get("abs_overlap", 0.0) for weight, item in zip(weights, top)))
    support = float(_clip(min(1.0, neff / max(float(args.memory_top_k), 1.0)) * consistency * (0.5 + 0.5 * overlap_mean), 0.0, 1.0))
    top_matches: List[Dict[str, Any]] = []
    for idx, item in enumerate(top):
        top_matches.append(
            {
                "target_quarter": str(item[1].get("target_fiscal_quarter") or ""),
                "weight": round(float(weights[idx]), 6),
                "score": round(float(item[0]), 6),
                "target_delta_log": round(float(safe_float(item[1].get("target_delta_log"), 0.0)), 6),
                "signed_align": round(float(item[2].get("signed_align", 0.0)), 6),
                "abs_overlap": round(float(item[2].get("abs_overlap", 0.0)), 6),
                "same_quarter": round(float(item[2].get("same_quarter", 0.0)), 6),
                "forward_cards_preview": _memory_preview(item[1].get("forward_rows") or [], top_k=2),
            }
        )
    return {
        "available": True,
        "pred_delta": pred_mean,
        "support": support,
        "consistency": consistency,
        "top_matches": top_matches,
    }


def _build_gate_feature_map(
    row: Mapping[str, Any],
    *,
    raw_delta: float,
    guard_delta: float,
    sem_delta: float,
    safety: Mapping[str, Any],
    backbone_uncertainty_state: Mapping[str, float],
    guidance_lock: float,
    memory_diag: Mapping[str, Any],
) -> Dict[str, float]:
    raw_sign = _sign_num(raw_delta)
    mem_sign = _sign_num(safe_float(memory_diag.get("pred_delta"), 0.0))
    memory_sign_match = 0.0
    if raw_sign != 0 and mem_sign != 0:
        memory_sign_match = 1.0 if raw_sign == mem_sign else -1.0
    metadata_proxy_flag = 1.0 if bool(safety.get("metadata_proxy_dominant")) else 0.0
    semantic_clarity = float(_clip(safe_float(safety.get("semantic_clarity"), 0.0), 0.0, 1.0))
    conflict = float(_clip(safe_float(row.get("fwd_conflict_ratio"), 0.0), 0.0, 1.0))
    return {
        "gate_raw_delta": float(raw_delta),
        "gate_guard_delta": float(guard_delta),
        "gate_sem_delta": float(sem_delta if math.isfinite(sem_delta) else raw_delta),
        "gate_delta_gap_raw_sem": float(raw_delta - (sem_delta if math.isfinite(sem_delta) else raw_delta)),
        "gate_fwd_conflict_ratio": conflict,
        "gate_fwd_abs_mass": float(max(safe_float(row.get("fwd_abs_mass"), 0.0), 0.0)),
        "gate_fwd_count_log": float(max(safe_float(row.get("fwd_count_log"), 0.0), 0.0)),
        "gate_top1_share": float(_clip(safe_float(row.get("segment_share_top1"), 0.0), 0.0, 1.0)),
        "gate_guidance_numeric_available": float(safe_float(row.get("guidance_numeric_available"), 0.0)),
        "gate_guidance_score_norm": float(_clip(safe_float(row.get("guidance_score_norm"), 0.0), 0.0, 1.0)),
        "gate_guid_band_ratio": float(max(safe_float(row.get("guid_band_ratio"), 0.0), 0.0)),
        "gate_guidance_lock": float(guidance_lock),
        "gate_same_quarter_support": float(_clip(safe_float(row.get("same_quarter_support"), 0.0), 0.0, 1.0)),
        "gate_segment_count": float(max(safe_float(row.get("segment_count"), 0.0), 0.0)),
        "gate_backbone_uncertainty": float(_clip(safe_float(backbone_uncertainty_state.get("backbone_uncertainty"), 0.0), 0.0, 1.0)),
        "gate_backbone_error_recent": float(max(safe_float(backbone_uncertainty_state.get("backbone_error_recent_abs_log"), 0.0), 0.0)),
        "gate_backbone_error_same_quarter": float(max(safe_float(backbone_uncertainty_state.get("backbone_error_same_quarter_abs_log"), 0.0), 0.0)),
        "gate_memory_pred_delta": float(safe_float(memory_diag.get("pred_delta"), 0.0)),
        "gate_memory_support": float(_clip(safe_float(memory_diag.get("support"), 0.0), 0.0, 1.0)),
        "gate_memory_consistency": float(_clip(safe_float(memory_diag.get("consistency"), 0.0), 0.0, 1.0)),
        "gate_memory_sign_match_raw": float(memory_sign_match),
        "gate_metadata_proxy_flag": float(metadata_proxy_flag),
        "gate_semantic_clarity": float(semantic_clarity),
        "gate_raw_x_conflict": float(raw_delta * conflict),
        "gate_raw_x_uncertainty": float(raw_delta * safe_float(backbone_uncertainty_state.get("backbone_uncertainty"), 0.0)),
        "gate_raw_x_weak_guidance": float(raw_delta * (1.0 - guidance_lock)),
        "gate_sem_x_clarity": float((sem_delta if math.isfinite(sem_delta) else raw_delta) * semantic_clarity),
        "gate_mem_x_support": float(safe_float(memory_diag.get("pred_delta"), 0.0) * safe_float(memory_diag.get("support"), 0.0)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Native CSAIS v1: native backbone plus native card deltas with a small context gate.")
    ap.add_argument("--experiment_config", default=DEFAULT_EXPERIMENT_CONFIG)
    ap.add_argument("--card_table_jsonl", default=DEFAULT_CARD_TABLE_JSONL)
    ap.add_argument("--backbone_csv", default=DEFAULT_BACKBONE_CSV)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--raw_alpha", type=float, default=10.0)
    ap.add_argument("--factor_alpha", type=float, default=10.0)
    ap.add_argument("--min_delta_train", type=int, default=12)
    ap.add_argument("--delta_shrink_k", type=float, default=8.0)
    ap.add_argument("--delta_cap_quantile", type=float, default=0.9)
    ap.add_argument("--delta_training_scope", choices=["company_local", "shared_pooled"], default="company_local")
    ap.add_argument("--gate_alpha", type=float, default=8.0)
    ap.add_argument("--min_gate_train", type=int, default=6)
    ap.add_argument("--gate_shrink_k", type=float, default=12.0)
    ap.add_argument("--gate_training_scope", choices=["company_local", "shared_pooled", "shared_prior_local_blend"], default="company_local")
    ap.add_argument("--gate_local_prior_k", type=float, default=8.0)
    ap.add_argument("--backbone_uncertainty_recent_window", type=int, default=6)
    ap.add_argument("--backbone_same_quarter_min", type=int, default=2)
    ap.add_argument("--backbone_same_guidance_min", type=int, default=3)
    ap.add_argument("--backbone_uncertainty_tau", type=float, default=0.10)
    ap.add_argument("--memory_top_k", type=int, default=3)
    ap.add_argument("--memory_temperature", type=float, default=0.35)
    ap.add_argument("--memory_min_train", type=int, default=6)
    ap.add_argument("--output_dir", default="output/native_csais_v1_all12")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = os.path.abspath(args.project_root)
    out_dir = Path(resolve_repo_path(args.output_dir, project_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = Path(resolve_repo_path(args.experiment_config, project_root))
    exp = _load_json(experiment_path)
    requested = {ticker.upper() for ticker in args.tickers}

    backbone_path = Path(resolve_repo_path(args.backbone_csv, project_root))
    backbone_df = pd.read_csv(backbone_path).copy()
    if requested:
        backbone_df = backbone_df[backbone_df["ticker"].astype(str).str.upper().isin(requested)].copy()
    backbone_df["ticker"] = backbone_df["ticker"].astype(str).str.upper()
    backbone_df["observed_filing_date"] = pd.to_datetime(backbone_df["observed_filing_date"], errors="coerce")
    backbone_df["target_filing_date"] = pd.to_datetime(backbone_df["target_filing_date"], errors="coerce")

    card_rows = load_jsonl(Path(resolve_repo_path(args.card_table_jsonl, project_root)))
    group_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in card_rows:
        ticker = str(row.get("ticker") or "").upper()
        if requested and ticker not in requested:
            continue
        key = (ticker, str(row.get("observed_quarter") or ""), str(row.get("target_quarter") or ""))
        group_map.setdefault(key, {"rows": []})["rows"].append(row)

    feature_records: List[Dict[str, Any]] = []
    for _, row in backbone_df.iterrows():
        key = (str(row.get("ticker") or "").upper(), str(row.get("observed_fiscal_quarter") or ""), str(row.get("target_fiscal_quarter") or ""))
        agg = _aggregate_group(group_map.get(key, {"rows": []})["rows"], "forward_only")
        record = {str(col): row[col] for col in backbone_df.columns}
        for name in RAW_CARD_FEATURE_NAMES:
            record[name] = float(agg["features"].get(name, 0.0))
        record["card_group_present"] = bool(key in group_map)
        record["target_log"] = _safe_log(safe_float(row.get("target_revenue")))
        record["backbone_log"] = _safe_log(safe_float(row.get("pred_backbone_v2")))
        record["target_delta_log"] = float(record["target_log"] - record["backbone_log"]) if math.isfinite(record["target_log"]) and math.isfinite(record["backbone_log"]) else float("nan")
        record["forward_rows"] = agg["forward_rows"]
        record["realized_rows"] = agg["realized_rows"]
        record["delta_forward_rows"] = agg["delta_forward_rows"]
        feature_records.append(record)

    feature_df = pd.DataFrame(feature_records)
    feature_df = feature_df.sort_values(
        ["observed_filing_date", "target_fiscal_quarter", "ticker"],
        key=lambda s: pd.to_datetime(s, errors="coerce") if s.name == "observed_filing_date" else (s.map(quarter_key) if s.name == "target_fiscal_quarter" else s),
    ).reset_index(drop=True)

    pred_rows: List[Dict[str, Any]] = []
    gate_history_records: List[Dict[str, Any]] = []
    company_lookup = {str(company.get("ticker") or "").upper(): company for company in exp.get("companies", [])}
    backbone_error_history: Dict[str, List[Dict[str, Any]]] = {}
    memory_history: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in feature_df.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        current_q = int(safe_float(row.get("target_fiscal_q"), 0.0))
        current_guidance = str(row.get("guidance_availability") or "none")
        observed_date = pd.to_datetime(row.get("observed_filing_date"), errors="coerce")

        train_mask = pd.to_datetime(feature_df["target_filing_date"], errors="coerce") < observed_date
        if str(args.delta_training_scope) == "company_local":
            train_mask = train_mask & (feature_df["ticker"].astype(str).str.upper() == ticker)
        train_df = feature_df[train_mask].copy()
        train_rows = [r.to_dict() for _, r in train_df.iterrows()]

        backbone_uncertainty_state = _build_backbone_uncertainty_state(
            backbone_error_history.get(ticker, []),
            current_q,
            current_guidance,
            args,
        )
        guidance_lock = _guidance_lock(row)

        raw_result = {"pred": 0.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": [], "coefs_by_name": {}}
        factor_result = {"pred": 0.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": [], "coefs_by_name": {}}
        if len(train_rows) >= int(args.min_delta_train):
            raw_result = _predict_zero_intercept_history(
                train_rows=train_rows,
                x_map=row.to_dict(),
                feature_names=RAW_CARD_FEATURE_NAMES,
                target_col="target_delta_log",
                alpha=float(args.raw_alpha),
                delta_cap_quantile=float(args.delta_cap_quantile),
                shrink_k=float(args.delta_shrink_k),
            )
            factor_result = _predict_zero_intercept_history(
                train_rows=train_rows,
                x_map=row.to_dict(),
                feature_names=FACTOR_DELTA_FEATURE_NAMES,
                target_col="target_delta_log",
                alpha=float(args.factor_alpha),
                delta_cap_quantile=float(args.delta_cap_quantile),
                shrink_k=float(args.delta_shrink_k),
            )

        raw_delta = float(raw_result["pred"])
        sem_delta = float(factor_result["pred"])
        safety = _apply_safety_guard(
            delta_pred=raw_delta,
            features={name: float(safe_float(row.get(name), 0.0)) for name in RAW_CARD_FEATURE_NAMES},
            top_contribs=list(raw_result["top_contribs"]),
            forward_row_count=len(list(row.get("forward_rows") or [])),
            safety_mode="semantic_blend",
            semantic_delta_pred=sem_delta,
        )
        signguard = _apply_safety_guard(
            delta_pred=raw_delta,
            features={name: float(safe_float(row.get(name), 0.0)) for name in RAW_CARD_FEATURE_NAMES},
            top_contribs=list(raw_result["top_contribs"]),
            forward_row_count=len(list(row.get("forward_rows") or [])),
            safety_mode="signguard_only",
            semantic_delta_pred=sem_delta,
        )
        guard_delta = float(signguard["delta_pred"])
        base_delta = float(safety["delta_pred"])

        memory_diag = _memory_diag(row, memory_history.get(ticker, []), args)
        gate_feature_map = _build_gate_feature_map(
            row,
            raw_delta=raw_delta,
            guard_delta=guard_delta,
            sem_delta=sem_delta,
            safety=safety,
            backbone_uncertainty_state=backbone_uncertainty_state,
            guidance_lock=guidance_lock,
            memory_diag=memory_diag,
        )

        gate_used = False
        gate_scope_applied = "off"
        gate_train_count = 0
        gate_shared_train_count = 0
        gate_local_train_count = 0
        gate_local_weight = 0.0
        gate_shared_pred = float("nan")
        gate_local_pred = float("nan")
        gate_support_used = 0.0
        gate_top_contribs: List[Tuple[str, float]] = []
        final_delta = base_delta

        gate_scope = str(args.gate_training_scope)
        local_gate_rows = [history for history in gate_history_records if str(history.get("ticker") or "").upper() == ticker]
        shared_gate_rows = list(gate_history_records)
        local_gate_result = {"pred": 0.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": [], "coefs_by_name": {}}
        shared_gate_result = {"pred": 0.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": [], "coefs_by_name": {}}
        if gate_scope == "company_local":
            local_gate_result = _predict_zero_intercept_history(
                train_rows=local_gate_rows,
                x_map=gate_feature_map,
                feature_names=GATE_FEATURE_NAMES,
                target_col="gate_target_delta",
                alpha=float(args.gate_alpha),
                delta_cap_quantile=float(args.delta_cap_quantile),
                shrink_k=float(args.gate_shrink_k),
            )
            gate_local_train_count = int(local_gate_result["train_count"])
            gate_train_count = gate_local_train_count
            if gate_local_train_count >= int(args.min_gate_train):
                gate_support_used = float(local_gate_result["support"])
                final_delta = float((1.0 - gate_support_used) * base_delta + gate_support_used * float(local_gate_result["pred"]))
                gate_top_contribs = list(local_gate_result["top_contribs"])
                gate_used = True
                gate_scope_applied = "company_local"
        elif gate_scope == "shared_pooled":
            shared_gate_result = _predict_zero_intercept_history(
                train_rows=shared_gate_rows,
                x_map=gate_feature_map,
                feature_names=GATE_FEATURE_NAMES,
                target_col="gate_target_delta",
                alpha=float(args.gate_alpha),
                delta_cap_quantile=float(args.delta_cap_quantile),
                shrink_k=float(args.gate_shrink_k),
            )
            gate_shared_train_count = int(shared_gate_result["train_count"])
            gate_train_count = gate_shared_train_count
            if gate_shared_train_count >= int(args.min_gate_train):
                gate_support_used = float(shared_gate_result["support"])
                final_delta = float((1.0 - gate_support_used) * base_delta + gate_support_used * float(shared_gate_result["pred"]))
                gate_top_contribs = list(shared_gate_result["top_contribs"])
                gate_used = True
                gate_scope_applied = "shared_pooled"
        else:
            shared_gate_result = _predict_zero_intercept_history(
                train_rows=shared_gate_rows,
                x_map=gate_feature_map,
                feature_names=GATE_FEATURE_NAMES,
                target_col="gate_target_delta",
                alpha=float(args.gate_alpha),
                delta_cap_quantile=float(args.delta_cap_quantile),
                shrink_k=float(args.gate_shrink_k),
            )
            local_gate_result = _predict_zero_intercept_history(
                train_rows=local_gate_rows,
                x_map=gate_feature_map,
                feature_names=GATE_FEATURE_NAMES,
                target_col="gate_target_delta",
                alpha=float(args.gate_alpha),
                delta_cap_quantile=float(args.delta_cap_quantile),
                shrink_k=float(args.gate_shrink_k),
            )
            gate_shared_train_count = int(shared_gate_result["train_count"])
            gate_local_train_count = int(local_gate_result["train_count"])
            gate_train_count = max(gate_shared_train_count, gate_local_train_count)
            shared_ok = gate_shared_train_count >= int(args.min_gate_train)
            local_ok = gate_local_train_count >= int(args.min_gate_train)
            if shared_ok and local_ok:
                gate_local_weight = float(gate_local_train_count / max(gate_local_train_count + float(args.gate_local_prior_k), EPS))
                gate_pred = float((1.0 - gate_local_weight) * float(shared_gate_result["pred"]) + gate_local_weight * float(local_gate_result["pred"]))
                gate_support_used = float((1.0 - gate_local_weight) * float(shared_gate_result["support"]) + gate_local_weight * float(local_gate_result["support"]))
                final_delta = float((1.0 - gate_support_used) * base_delta + gate_support_used * gate_pred)
                gate_top_contribs = list(local_gate_result["top_contribs"] if gate_local_weight >= 0.5 else shared_gate_result["top_contribs"])
                gate_used = True
                gate_scope_applied = "shared_prior_local_blend"
            elif local_ok:
                gate_support_used = float(local_gate_result["support"])
                final_delta = float((1.0 - gate_support_used) * base_delta + gate_support_used * float(local_gate_result["pred"]))
                gate_top_contribs = list(local_gate_result["top_contribs"])
                gate_used = True
                gate_scope_applied = "company_local_only"
            elif shared_ok:
                gate_support_used = float(shared_gate_result["support"])
                final_delta = float((1.0 - gate_support_used) * base_delta + gate_support_used * float(shared_gate_result["pred"]))
                gate_top_contribs = list(shared_gate_result["top_contribs"])
                gate_used = True
                gate_scope_applied = "shared_pooled_only"
        gate_shared_pred = float(shared_gate_result["pred"]) if math.isfinite(safe_float(shared_gate_result.get("pred_raw"), float("nan"))) else float("nan")
        gate_local_pred = float(local_gate_result["pred"]) if math.isfinite(safe_float(local_gate_result.get("pred_raw"), float("nan"))) else float("nan")

        backbone_log = _safe_log(safe_float(row.get("pred_backbone_v2")))
        pred_log = float(backbone_log + final_delta) if math.isfinite(backbone_log) else float("nan")
        pred = float(math.exp(pred_log)) if math.isfinite(pred_log) else float("nan")

        support_cards: List[Dict[str, Any]] = []
        conflict_cards: List[Dict[str, Any]] = []
        if raw_result["coefs_by_name"]:
            scored_cards: List[Dict[str, Any]] = []
            for card in list(row.get("forward_rows") or []):
                influence = _card_influence(card, raw_result["coefs_by_name"])
                card_copy = dict(card)
                card_copy["approx_influence"] = float(influence)
                scored_cards.append(card_copy)
            scored_cards.sort(key=lambda item: item.get("approx_influence", 0.0), reverse=True)
            support_cards = scored_cards[:5]
            conflict_cards = list(reversed(sorted(scored_cards, key=lambda item: item.get("approx_influence", 0.0))))[:5]

        out_row = {str(col): row[col] for col in feature_df.columns if col not in {"forward_rows", "realized_rows", "delta_forward_rows"}}
        out_row.update(backbone_uncertainty_state)
        out_row["native_csais_mode"] = "native_csais_context_gate" if gate_used else "native_csais_raw_delta"
        out_row["pred_native_csais_v1"] = pred
        out_row["pred_native_csais_v1_log"] = pred_log
        out_row["pred_native_csais_backbone"] = float(safe_float(row.get("pred_backbone_v2"), float("nan")))
        out_row["pred_native_csais_backbone_log"] = backbone_log
        out_row["native_csais_target_delta_log"] = float(safe_float(row.get("target_delta_log"), float("nan")))
        out_row["native_csais_raw_delta_log"] = raw_delta
        out_row["native_csais_base_delta_log"] = base_delta
        out_row["native_csais_guard_delta_log"] = guard_delta
        out_row["native_csais_factor_delta_log"] = sem_delta
        out_row["native_csais_final_delta_log"] = float(final_delta)
        out_row["native_csais_raw_train_count"] = int(raw_result["train_count"])
        out_row["native_csais_factor_train_count"] = int(factor_result["train_count"])
        out_row["native_csais_gate_used"] = bool(gate_used)
        out_row["native_csais_gate_scope"] = gate_scope_applied
        out_row["native_csais_gate_train_count"] = int(gate_train_count)
        out_row["native_csais_gate_local_train_count"] = int(gate_local_train_count)
        out_row["native_csais_gate_shared_train_count"] = int(gate_shared_train_count)
        out_row["native_csais_gate_local_weight"] = float(gate_local_weight)
        out_row["native_csais_gate_support_used"] = float(gate_support_used)
        out_row["native_csais_gate_shared_delta_log"] = gate_shared_pred
        out_row["native_csais_gate_local_delta_log"] = gate_local_pred
        out_row["native_csais_guidance_lock"] = float(guidance_lock)
        out_row["native_csais_memory_available"] = int(bool(memory_diag.get("available", False)))
        out_row["native_csais_memory_pred_delta_log"] = float(safe_float(memory_diag.get("pred_delta"), 0.0))
        out_row["native_csais_memory_support"] = float(safe_float(memory_diag.get("support"), 0.0))
        out_row["native_csais_memory_consistency"] = float(safe_float(memory_diag.get("consistency"), 0.0))
        out_row["native_csais_memory_top_matches_json"] = json.dumps(sanitize_for_json(memory_diag.get("top_matches", [])), ensure_ascii=False, allow_nan=False)
        out_row["native_csais_raw_top_contribs_json"] = json.dumps(sanitize_for_json(raw_result["top_contribs"]), ensure_ascii=False, allow_nan=False)
        out_row["native_csais_factor_top_contribs_json"] = json.dumps(sanitize_for_json(factor_result["top_contribs"]), ensure_ascii=False, allow_nan=False)
        out_row["native_csais_gate_top_contribs_json"] = json.dumps(sanitize_for_json(gate_top_contribs), ensure_ascii=False, allow_nan=False)
        out_row["native_csais_support_cards_json"] = json.dumps(sanitize_for_json(support_cards), ensure_ascii=False, allow_nan=False)
        out_row["native_csais_conflict_cards_json"] = json.dumps(sanitize_for_json(conflict_cards), ensure_ascii=False, allow_nan=False)
        out_row["native_csais_forward_card_rows_json"] = json.dumps(sanitize_for_json(list(row.get("forward_rows") or [])[:12]), ensure_ascii=False, allow_nan=False)
        pred_rows.append(out_row)

        target_log = _safe_log(safe_float(row.get("target_revenue")))
        if math.isfinite(target_log) and math.isfinite(backbone_log):
            backbone_error_history.setdefault(ticker, []).append(
                {
                    "target_fiscal_quarter": str(row.get("target_fiscal_quarter") or ""),
                    "guidance_availability": current_guidance,
                    "backbone_abs_log_error": float(abs(target_log - backbone_log)),
                }
            )
        memory_history.setdefault(ticker, []).append(
            {
                **row.to_dict(),
                "forward_rows": list(row.get("forward_rows") or []),
            }
        )
        gate_history_records.append(
            {
                "ticker": ticker,
                "gate_target_delta": float(safe_float(row.get("target_delta_log"), float("nan"))),
                **{name: float(safe_float(gate_feature_map.get(name), 0.0)) for name in GATE_FEATURE_NAMES},
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    quarterly_csv = out_dir / "native_csais_v1_quarterly.csv"
    feature_csv = out_dir / "native_csais_v1_feature_panel.csv"
    company_csv = out_dir / "native_csais_v1_company_summary.csv"
    summary_json = out_dir / "native_csais_v1_summary.json"
    feature_df_copy = feature_df.copy()
    feature_df_copy["forward_rows_json"] = feature_df_copy["forward_rows"].map(lambda rows: json.dumps(sanitize_for_json(list(rows)[:12]), ensure_ascii=False, allow_nan=False))
    feature_df_copy = feature_df_copy.drop(columns=["forward_rows", "realized_rows", "delta_forward_rows"], errors="ignore")
    feature_df_copy.to_csv(feature_csv, index=False)
    pred_df.to_csv(quarterly_csv, index=False)

    company_rows: List[Dict[str, Any]] = []
    for ticker, company_group in pred_df.groupby("ticker"):
        metrics = _metrics(company_group["target_revenue"], company_group["pred_native_csais_v1"])
        backbone_metrics = _metrics(company_group["target_revenue"], company_group["pred_native_csais_backbone"])
        company_cfg = company_lookup.get(str(ticker), {})
        best_model, best_mae = _load_best_stat(Path(resolve_repo_path(company_cfg.get("stat_baseline_metrics_csv"), project_root)))
        company_rows.append(
            {
                "ticker": str(ticker),
                "n": int(metrics["n"]),
                "native_csais_v1_mae": float(metrics["mae"]),
                "native_csais_v1_rmse": float(metrics["rmse"]),
                "native_csais_v1_mape": float(metrics["mape"]),
                "native_csais_v1_smape": float(metrics["smape"]),
                "backbone_v2_mae": float(backbone_metrics["mae"]),
                "delta_vs_backbone": float(metrics["mae"] - backbone_metrics["mae"]),
                "best_stat_model": best_model,
                "best_stat_mae": float(best_mae),
                "beats_best_stat": bool(metrics["mae"] < best_mae),
            }
        )
    company_df = pd.DataFrame(company_rows).sort_values("ticker")
    company_df.to_csv(company_csv, index=False)

    pooled = _metrics(pred_df["target_revenue"], pred_df["pred_native_csais_v1"])
    backbone_pooled = _metrics(pred_df["target_revenue"], pred_df["pred_native_csais_backbone"])
    summary = {
        "experiment_config": str(experiment_path),
        "card_table_jsonl": str(resolve_repo_path(args.card_table_jsonl, project_root)),
        "backbone_csv": str(backbone_path),
        "output_dir": str(out_dir),
        "company_count": int(len(company_df)),
        "eval_row_count": int(len(pred_df)),
        "native_csais_v1_pooled": pooled,
        "backbone_v2_pooled": backbone_pooled,
        "macro_mae": float(company_df["native_csais_v1_mae"].mean()) if not company_df.empty else float("nan"),
        "backbone_macro_mae": float(company_df["backbone_v2_mae"].mean()) if not company_df.empty else float("nan"),
        "beats_best_stat_companies": int(company_df["beats_best_stat"].sum()) if not company_df.empty else 0,
        "raw_card_feature_names": RAW_CARD_FEATURE_NAMES,
        "factor_delta_feature_names": FACTOR_DELTA_FEATURE_NAMES,
        "gate_feature_names": GATE_FEATURE_NAMES,
        "raw_alpha": float(args.raw_alpha),
        "factor_alpha": float(args.factor_alpha),
        "min_delta_train": int(args.min_delta_train),
        "delta_shrink_k": float(args.delta_shrink_k),
        "delta_cap_quantile": float(args.delta_cap_quantile),
        "delta_training_scope": str(args.delta_training_scope),
        "gate_alpha": float(args.gate_alpha),
        "min_gate_train": int(args.min_gate_train),
        "gate_shrink_k": float(args.gate_shrink_k),
        "gate_training_scope": str(args.gate_training_scope),
        "gate_local_prior_k": float(args.gate_local_prior_k),
        "backbone_uncertainty_recent_window": int(args.backbone_uncertainty_recent_window),
        "backbone_uncertainty_tau": float(args.backbone_uncertainty_tau),
        "memory_top_k": int(args.memory_top_k),
        "memory_temperature": float(args.memory_temperature),
        "memory_min_train": int(args.memory_min_train),
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit("Implementation dependency only; run scripts/run_reference_replay.sh.")
