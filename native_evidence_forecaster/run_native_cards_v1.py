#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from native_evidence_forecaster.common import EPS, load_jsonl, resolve_repo_path, safe_float, sanitize_for_json, write_json


DEFAULT_EXPERIMENT_CONFIG = "replay_inputs/retained_336/experiment.json"
DEFAULT_CARD_TABLE_JSONL = "replay_inputs/retained_336/normalized_native_forward_cards.jsonl"
DEFAULT_BACKBONE_CSV = "replay_inputs/retained_336/native_backbone.csv"
CANONICAL_FACTORS = [
    "demand",
    "supply_capacity",
    "pricing_margin",
    "product_transition",
    "inventory_channel",
    "regulation_macro",
    "revenue_boost_limit",
    "other",
]
STRENGTH_TO_NUM = {
    "low": 0.6,
    "medium": 1.0,
    "high": 1.4,
    "unknown": 1.0,
}
METADATA_PROXY_FEATURES = {
    "fwd_count_log",
    "fwd_abs_mass",
    "fwd_conflict_ratio",
    "fwd_persistent_share",
    "fwd_event_share",
    "fwd_claim_share",
    "fwd_top1_shareweighted_signed",
    "fwd_other_shareweighted_signed",
}
SEMANTIC_FEATURE_NAMES = [
    "fwd_signed_mass",
    "fwd_top1_shareweighted_signed",
    "fwd_other_shareweighted_signed",
] + [f"factor_signed__{factor}" for factor in CANONICAL_FACTORS]
CONTEXT_GATE_FEATURE_NAMES = [
    "gate_full_delta",
    "gate_guard_delta",
    "gate_sem_delta",
    "gate_full_x_conflict",
    "gate_full_x_proxy",
    "gate_full_x_sign_mismatch",
    "gate_full_x_top1",
    "gate_sem_x_clarity",
    "gate_guard_x_low_count",
]


def _quarter_key(value: str) -> Tuple[int, int]:
    text = str(value or "")
    if not text.startswith("FY") or "_Q" not in text:
        return (0, 0)
    try:
        return (int(text[2:6]), int(text.split("_Q", 1)[1]))
    except Exception:
        return (0, 0)


def _sign(polarity: str) -> float:
    text = str(polarity or "").strip().lower()
    if text == "positive":
        return 1.0
    if text == "negative":
        return -1.0
    return 0.0


def _strength_num(strength: str) -> float:
    return float(STRENGTH_TO_NUM.get(str(strength or "unknown").strip().lower(), 1.0))


def _safe_log(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(math.log(max(value, EPS)))


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _sign_num(value: float, tol: float = 1e-6) -> int:
    if not math.isfinite(value) or abs(value) <= tol:
        return 0
    return 1 if value > 0 else -1


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = pd.to_numeric(pd.Series(list(y_true)), errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(pd.Series(list(y_pred)), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "smape": float("nan")}
    err = yt - yp
    return {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), EPS))),
        "smape": float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(yt) + np.abs(yp), EPS))),
    }


def _fit_zero_intercept_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, Any]:
    mask = np.isfinite(y)
    if x.size:
        mask = mask & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        width = x.shape[1] if x.ndim == 2 else 0
        return {"stds": np.ones(width), "coefs": np.zeros(width)}
    stds = np.where(x.std(axis=0) > EPS, x.std(axis=0), 1.0)
    xz = np.nan_to_num(x / stds, nan=0.0, posinf=0.0, neginf=0.0)
    reg = float(alpha) * np.eye(x.shape[1], dtype=float)
    try:
        coefs = np.linalg.pinv(xz.T @ xz + reg) @ (xz.T @ y)
    except np.linalg.LinAlgError:
        coefs = np.zeros(x.shape[1], dtype=float)
    return {"stds": stds, "coefs": coefs}


def _predict_zero_intercept_ridge(model: Mapping[str, Any], x: np.ndarray) -> float:
    stds = np.asarray(model["stds"], dtype=float)
    coefs = np.asarray(model["coefs"], dtype=float)
    xz = np.nan_to_num(x / stds, nan=0.0, posinf=0.0, neginf=0.0)
    return float(xz @ coefs)


def _top_feature_contribs(model: Mapping[str, Any], x: np.ndarray, feature_names: Sequence[str], top_k: int = 6) -> List[Tuple[str, float]]:
    stds = np.asarray(model["stds"], dtype=float)
    coefs = np.asarray(model["coefs"], dtype=float)
    xz = np.nan_to_num(x / stds, nan=0.0, posinf=0.0, neginf=0.0)
    contribs = [(name, float(val)) for name, val in zip(feature_names, xz * coefs)]
    contribs.sort(key=lambda item: abs(item[1]), reverse=True)
    return contribs[:top_k]


def _apply_safety_guard(
    delta_pred: float,
    features: Mapping[str, float],
    top_contribs: Sequence[Tuple[str, float]],
    forward_row_count: int,
    safety_mode: str,
    semantic_delta_pred: float = float("nan"),
) -> Dict[str, Any]:
    fwd_abs_mass = max(safe_float(features.get("fwd_abs_mass"), 0.0), 0.0)
    fwd_signed_mass = safe_float(features.get("fwd_signed_mass"), 0.0)
    conflict_ratio = _clip(safe_float(features.get("fwd_conflict_ratio"), 0.0), 0.0, 1.0)
    top_feature_name = str(top_contribs[0][0]) if top_contribs else ""
    metadata_proxy_dominant = top_feature_name in METADATA_PROXY_FEATURES
    semantic_clarity = float(min(abs(fwd_signed_mass) / max(fwd_abs_mass, EPS), 1.0)) if fwd_abs_mass > EPS else 0.0
    count_scale = 1.0
    conflict_scale = float(1.0 - 0.5 * conflict_ratio)
    proxy_scale = 0.85 if metadata_proxy_dominant else 1.0
    sign_guard_triggered = False
    guard_reason = "none"
    guarded_delta = float(delta_pred)
    safety_scale = 1.0

    if str(safety_mode) in {"signguard_only", "signsafe_easyguard"}:
        if semantic_clarity >= 0.2 and _sign_num(fwd_signed_mass) != 0 and _sign_num(delta_pred) != 0 and _sign_num(fwd_signed_mass) != _sign_num(delta_pred):
            guarded_delta = 0.0
            safety_scale = 0.0
            sign_guard_triggered = True
            guard_reason = "raw_sign_mismatch"
        elif str(safety_mode) == "signsafe_easyguard":
            reasons: List[str] = []
            if conflict_ratio >= 0.25:
                safety_scale *= float(semantic_clarity * conflict_scale)
                reasons.append("conflict_shrink")
            if metadata_proxy_dominant:
                safety_scale *= float(proxy_scale)
                reasons.append("metadata_proxy_shrink")
            if forward_row_count <= 1:
                count_scale = 0.5
                safety_scale *= count_scale
                reasons.append("single_card_shrink")
            elif forward_row_count == 2:
                count_scale = 0.75
                safety_scale *= count_scale
                reasons.append("two_card_shrink")
            elif forward_row_count == 3:
                count_scale = 0.9
                safety_scale *= count_scale
                reasons.append("three_card_shrink")
            guarded_delta = float(delta_pred * safety_scale)
            guard_reason = "+".join(reasons) if reasons else "pass_through"
        else:
            guard_reason = "pass_through"
    elif str(safety_mode) == "semantic_blend":
        semantic_sign = _sign_num(semantic_delta_pred)
        full_sign = _sign_num(delta_pred)
        high_conflict = conflict_ratio >= 0.25
        if metadata_proxy_dominant or high_conflict:
            if semantic_sign != 0 and full_sign != 0 and semantic_sign != full_sign:
                guarded_delta = float(semantic_delta_pred)
                guard_reason = "semantic_sign_override"
            elif math.isfinite(semantic_delta_pred):
                guarded_delta = float(0.5 * delta_pred + 0.5 * semantic_delta_pred)
                guard_reason = "semantic_blend"
            safety_scale = float(abs(guarded_delta) / max(abs(delta_pred), EPS)) if math.isfinite(delta_pred) else 1.0
        else:
            guard_reason = "pass_through"
    else:
        safety_scale = 1.0

    if str(safety_mode) not in {"signguard_only", "signsafe_easyguard", "semantic_blend"}:
        safety_scale = 1.0

    return {
        "delta_pred": float(guarded_delta),
        "safety_scale": float(safety_scale),
        "semantic_clarity": float(semantic_clarity),
        "count_scale": float(count_scale),
        "conflict_scale": float(conflict_scale),
        "proxy_scale": float(proxy_scale),
        "top_feature_name": top_feature_name,
        "metadata_proxy_dominant": bool(metadata_proxy_dominant),
        "sign_guard_triggered": bool(sign_guard_triggered),
        "guard_reason": guard_reason,
    }


def _build_context_gate_features(
    row: Mapping[str, Any],
    features: Mapping[str, float],
    safety: Mapping[str, Any],
    delta_full: float,
    delta_guard: float,
    delta_semantic: float,
    forward_row_count: int,
) -> Dict[str, float]:
    conflict_ratio = _clip(safe_float(features.get("fwd_conflict_ratio"), 0.0), 0.0, 1.0)
    semantic_clarity = _clip(safe_float(safety.get("semantic_clarity"), 0.0), 0.0, 1.0)
    top1_share = _clip(safe_float(row.get("segment_share_top1"), 0.0), 0.0, 1.0)
    metadata_proxy_flag = 1.0 if bool(safety.get("metadata_proxy_dominant")) else 0.0
    sign_mismatch_flag = 1.0 if _sign_num(delta_full) != 0 and _sign_num(safe_float(features.get("fwd_signed_mass"), 0.0)) != 0 and _sign_num(delta_full) != _sign_num(safe_float(features.get("fwd_signed_mass"), 0.0)) else 0.0
    low_count_flag = 1.0 if int(forward_row_count) <= 2 else 0.0
    return {
        "gate_full_delta": float(delta_full),
        "gate_guard_delta": float(delta_guard),
        "gate_sem_delta": float(delta_semantic if math.isfinite(delta_semantic) else delta_full),
        "gate_full_x_conflict": float(delta_full * conflict_ratio),
        "gate_full_x_proxy": float(delta_full * metadata_proxy_flag),
        "gate_full_x_sign_mismatch": float(delta_full * sign_mismatch_flag),
        "gate_full_x_top1": float(delta_full * top1_share),
        "gate_sem_x_clarity": float((delta_semantic if math.isfinite(delta_semantic) else delta_full) * semantic_clarity),
        "gate_guard_x_low_count": float(delta_guard * low_count_flag),
    }


def _predict_history_gate(
    train_rows: Sequence[Mapping[str, Any]],
    x_map: Mapping[str, float],
    feature_names: Sequence[str],
    alpha: float,
    delta_cap_quantile: float,
    shrink_k: float,
) -> Dict[str, Any]:
    train_count = int(len(train_rows))
    if train_count == 0:
        return {
            "pred": float("nan"),
            "train_count": 0,
            "support": 0.0,
            "top_contribs": [],
        }
    x_train = np.asarray([[float(safe_float(row.get(name), 0.0)) for name in feature_names] for row in train_rows], dtype=float)
    y_train = np.asarray([float(safe_float(row.get("gate_target_delta"), float("nan"))) for row in train_rows], dtype=float)
    model = _fit_zero_intercept_ridge(x_train, y_train, float(alpha))
    x = np.asarray([float(safe_float(x_map.get(name), 0.0)) for name in feature_names], dtype=float)
    pred = _predict_zero_intercept_ridge(model, x)
    finite_abs = np.abs(y_train[np.isfinite(y_train)])
    if finite_abs.size:
        delta_cap = float(np.quantile(finite_abs, float(delta_cap_quantile)))
        if math.isfinite(delta_cap) and delta_cap > 0.0:
            pred = _clip(pred, -delta_cap, delta_cap)
    support = train_count / max(train_count + float(shrink_k), EPS)
    pred = float(pred * support)
    top_contribs = _top_feature_contribs(model, x, feature_names, top_k=6)
    return {
        "pred": float(pred),
        "train_count": train_count,
        "support": float(support),
        "top_contribs": top_contribs,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_best_stat(path: Path) -> Tuple[str, float]:
    frame = pd.read_csv(path)
    frame = frame[~frame["model"].isin(["guid_uniform", "guid_triangular"])].copy()
    best = frame.sort_values("MAE").iloc[0]
    return str(best["model"]), float(best["MAE"])


def _card_mass(row: Mapping[str, Any]) -> float:
    return float(_sign(str(row.get("polarity") or "unknown")) * _strength_num(str(row.get("strength") or "unknown")) * max(safe_float(row.get("confidence"), 0.0), 0.0))


def _aggregate_group(rows: Sequence[Mapping[str, Any]], context_mode: str) -> Dict[str, Any]:
    rows_list = list(rows)
    forward_rows = [row for row in rows_list if bool(row.get("is_forward_target_quarter")) and str(row.get("native_source") or "") != "delta_op"]
    realized_rows = [row for row in rows_list if bool(row.get("is_observed_realized")) and str(row.get("native_source") or "") != "delta_op"]
    delta_forward_rows = [row for row in rows_list if bool(row.get("is_forward_target_quarter")) and str(row.get("native_source") or "") == "delta_op"]
    delta_realized_rows = [row for row in rows_list if bool(row.get("is_observed_realized")) and str(row.get("native_source") or "") == "delta_op"]
    if str(context_mode) == "forward_only":
        realized_rows = []
        delta_forward_rows = []
        delta_realized_rows = []

    def _signed_abs(cards: Sequence[Mapping[str, Any]]) -> Tuple[float, float, float, float]:
        signed = 0.0
        pos_abs = 0.0
        neg_abs = 0.0
        for card in cards:
            mass = _card_mass(card)
            signed += mass
            if mass > 0:
                pos_abs += abs(mass)
            elif mass < 0:
                neg_abs += abs(mass)
        total_abs = pos_abs + neg_abs
        conflict_ratio = min(pos_abs, neg_abs) / total_abs if total_abs > EPS else 0.0
        return float(signed), float(total_abs), float(conflict_ratio), float(pos_abs - neg_abs)

    def _share_weighted_signed(cards: Sequence[Mapping[str, Any]], top1_only: bool) -> float:
        total = 0.0
        for card in cards:
            rank = int(safe_float(card.get("segment_rank_at_observed"), 0.0))
            share = safe_float(card.get("segment_share_at_observed"), 0.0)
            if top1_only and rank != 1:
                continue
            if not top1_only and rank == 1:
                continue
            total += _card_mass(card) * max(share, 0.0)
        return float(total)

    def _source_share(cards: Sequence[Mapping[str, Any]], source_name: str) -> float:
        if not cards:
            return 0.0
        return float(sum(1 for card in cards if str(card.get("driver_source") or "") == source_name) / len(cards))

    def _persistent_share(cards: Sequence[Mapping[str, Any]]) -> float:
        if not cards:
            return 0.0
        return float(sum(1 for card in cards if bool(card.get("persistence_hint"))) / len(cards))

    features: Dict[str, float] = {}
    fwd_signed, fwd_abs, fwd_conflict, _ = _signed_abs(forward_rows)
    realized_signed, realized_abs, realized_conflict, _ = _signed_abs(realized_rows)
    delta_fwd_signed, delta_fwd_abs, _, _ = _signed_abs(delta_forward_rows)
    delta_realized_signed, _, _, _ = _signed_abs(delta_realized_rows)

    features.update(
        {
            "fwd_count_log": float(math.log1p(len(forward_rows))),
            "fwd_signed_mass": fwd_signed,
            "fwd_abs_mass": fwd_abs,
            "fwd_conflict_ratio": fwd_conflict,
            "fwd_persistent_share": _persistent_share(forward_rows),
            "fwd_event_share": _source_share(forward_rows, "event"),
            "fwd_claim_share": _source_share(forward_rows, "claim"),
            "fwd_top1_shareweighted_signed": _share_weighted_signed(forward_rows, top1_only=True),
            "fwd_other_shareweighted_signed": _share_weighted_signed(forward_rows, top1_only=False),
            "realized_count_log": float(math.log1p(len(realized_rows))),
            "realized_signed_mass": realized_signed,
            "realized_abs_mass": realized_abs,
            "realized_conflict_ratio": realized_conflict,
            "delta_fwd_count_log": float(math.log1p(len(delta_forward_rows))),
            "delta_fwd_signed_mass": delta_fwd_signed,
            "delta_fwd_abs_mass": delta_fwd_abs,
            "delta_realized_count_log": float(math.log1p(len(delta_realized_rows))),
            "delta_realized_signed_mass": delta_realized_signed,
        }
    )
    for factor in CANONICAL_FACTORS:
        signed = 0.0
        total_abs = 0.0
        for card in forward_rows:
            if str(card.get("canonical_factor") or "") != factor:
                continue
            mass = _card_mass(card)
            signed += mass
            total_abs += abs(mass)
        features[f"factor_signed__{factor}"] = float(signed)
        features[f"factor_abs__{factor}"] = float(total_abs)

    return {
        "features": features,
        "forward_rows": forward_rows,
        "realized_rows": realized_rows,
        "delta_forward_rows": delta_forward_rows,
        "delta_realized_rows": delta_realized_rows,
    }


def _card_influence(card: Mapping[str, Any], coefs_by_name: Mapping[str, float]) -> float:
    factor = str(card.get("canonical_factor") or "other")
    mass = _card_mass(card)
    influence = float(coefs_by_name.get(f"factor_signed__{factor}", 0.0) * mass)
    influence += float(coefs_by_name.get(f"factor_abs__{factor}", 0.0) * abs(mass))
    share = max(safe_float(card.get("segment_share_at_observed"), 0.0), 0.0)
    rank = int(safe_float(card.get("segment_rank_at_observed"), 0.0))
    if rank == 1:
        influence += float(coefs_by_name.get("fwd_top1_shareweighted_signed", 0.0) * mass * share)
    else:
        influence += float(coefs_by_name.get("fwd_other_shareweighted_signed", 0.0) * mass * share)
    return influence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run native-cards V1 evidence delta on top of Backbone V2.")
    ap.add_argument("--experiment_config", default=DEFAULT_EXPERIMENT_CONFIG)
    ap.add_argument("--card_table_jsonl", default=DEFAULT_CARD_TABLE_JSONL)
    ap.add_argument("--backbone_csv", default=DEFAULT_BACKBONE_CSV)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--min_train", type=int, default=12)
    ap.add_argument("--delta_shrink_k", type=float, default=8.0)
    ap.add_argument("--delta_cap_quantile", type=float, default=0.9)
    ap.add_argument("--training_scope", choices=["company_local", "shared_pooled"], default="company_local")
    ap.add_argument("--context_mode", choices=["full", "forward_only"], default="full")
    ap.add_argument("--safety_mode", choices=["none", "signguard_only", "signsafe_easyguard", "semantic_blend", "context_gate"], default="none")
    ap.add_argument("--gate_alpha", type=float, default=8.0)
    ap.add_argument("--min_gate_train", type=int, default=10)
    ap.add_argument("--gate_shrink_k", type=float, default=12.0)
    ap.add_argument("--gate_training_scope", choices=["company_local", "shared_pooled", "shared_prior_local_blend"], default="company_local")
    ap.add_argument("--gate_local_prior_k", type=float, default=8.0)
    ap.add_argument("--output_dir", default="output/native_cards_v1_all12")
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
    feature_names = [
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

    for _, row in backbone_df.iterrows():
        key = (str(row.get("ticker") or "").upper(), str(row.get("observed_fiscal_quarter") or ""), str(row.get("target_fiscal_quarter") or ""))
        agg = _aggregate_group(group_map.get(key, {"rows": []})["rows"], str(args.context_mode))
        record = {str(col): row[col] for col in backbone_df.columns}
        for name in feature_names:
            record[name] = float(agg["features"].get(name, 0.0))
        record["card_group_present"] = bool(key in group_map)
        record["target_log"] = _safe_log(safe_float(row.get("target_revenue")))
        record["backbone_log"] = _safe_log(safe_float(row.get("pred_backbone_v2")))
        record["native_card_delta_target_log"] = float(record["target_log"] - record["backbone_log"]) if math.isfinite(record["target_log"]) and math.isfinite(record["backbone_log"]) else float("nan")
        record["forward_card_rows_json"] = json.dumps(sanitize_for_json(agg["forward_rows"][:12]), ensure_ascii=False, allow_nan=False)
        feature_records.append(record)

    feature_df = pd.DataFrame(feature_records)
    feature_df = feature_df.sort_values(["observed_filing_date", "target_fiscal_quarter", "ticker"], key=lambda s: pd.to_datetime(s, errors="coerce") if s.name == "observed_filing_date" else (s.map(_quarter_key) if s.name == "target_fiscal_quarter" else s)).reset_index(drop=True)

    pred_rows: List[Dict[str, Any]] = []
    gate_history_records: List[Dict[str, Any]] = []
    company_lookup = {str(company.get("ticker") or "").upper(): company for company in exp.get("companies", [])}

    for _, row in feature_df.iterrows():
        observed_date = pd.to_datetime(row.get("observed_filing_date"), errors="coerce")
        train_mask = pd.to_datetime(feature_df["target_filing_date"], errors="coerce") < observed_date
        if str(args.training_scope) == "company_local":
            train_mask = train_mask & (feature_df["ticker"].astype(str).str.upper() == str(row.get("ticker") or "").upper())
        train_df = feature_df[train_mask].copy()

        x = np.asarray([float(safe_float(row.get(name), 0.0)) for name in feature_names], dtype=float)
        delta_pred = 0.0
        train_count = int(len(train_df))
        top_contribs: List[Tuple[str, float]] = []
        support_cards: List[Dict[str, Any]] = []
        conflict_cards: List[Dict[str, Any]] = []
        mode = "backbone_only"
        semantic_delta_pred = float("nan")
        full_delta_pred = 0.0
        signguard_delta_pred = 0.0
        gate_train_count = 0
        gate_shared_train_count = 0
        gate_local_train_count = 0
        gate_local_weight = 0.0
        gate_shared_delta_pred = float("nan")
        gate_local_delta_pred = float("nan")
        gate_used = False
        gate_top_contribs: List[Tuple[str, float]] = []
        gate_feature_map = {name: 0.0 for name in CONTEXT_GATE_FEATURE_NAMES}
        if train_count >= int(args.min_train):
            x_train = np.asarray([[float(safe_float(r.get(name), 0.0)) for name in feature_names] for _, r in train_df.iterrows()], dtype=float)
            y_train = pd.to_numeric(train_df["native_card_delta_target_log"], errors="coerce").to_numpy(dtype=float)
            model = _fit_zero_intercept_ridge(x_train, y_train, float(args.alpha))
            delta_pred = _predict_zero_intercept_ridge(model, x)
            finite_abs = np.abs(y_train[np.isfinite(y_train)])
            if finite_abs.size:
                delta_cap = float(np.quantile(finite_abs, float(args.delta_cap_quantile)))
                if math.isfinite(delta_cap) and delta_cap > 0.0:
                    delta_pred = _clip(delta_pred, -delta_cap, delta_cap)
            support = train_count / max(train_count + float(args.delta_shrink_k), EPS)
            delta_pred = float(delta_pred * support)
            full_delta_pred = float(delta_pred)
            top_contribs = _top_feature_contribs(model, x, feature_names, top_k=6)
            coefs_by_name = {name: float(val) for name, val in zip(feature_names, np.asarray(model["coefs"], dtype=float) / np.asarray(model["stds"], dtype=float))}
            current_key = (str(row.get("ticker") or "").upper(), str(row.get("observed_fiscal_quarter") or ""), str(row.get("target_fiscal_quarter") or ""))
            current_forward_rows = _aggregate_group(group_map.get(current_key, {"rows": []})["rows"], str(args.context_mode))["forward_rows"]
            if str(args.safety_mode) in {"semantic_blend", "context_gate"}:
                x_sem = np.asarray([float(safe_float(row.get(name), 0.0)) for name in SEMANTIC_FEATURE_NAMES], dtype=float)
                x_sem_train = np.asarray([[float(safe_float(r.get(name), 0.0)) for name in SEMANTIC_FEATURE_NAMES] for _, r in train_df.iterrows()], dtype=float)
                semantic_model = _fit_zero_intercept_ridge(x_sem_train, y_train, float(args.alpha))
                semantic_delta_pred = _predict_zero_intercept_ridge(semantic_model, x_sem)
                if finite_abs.size:
                    delta_cap = float(np.quantile(finite_abs, float(args.delta_cap_quantile)))
                    if math.isfinite(delta_cap) and delta_cap > 0.0:
                        semantic_delta_pred = _clip(semantic_delta_pred, -delta_cap, delta_cap)
                semantic_delta_pred = float(semantic_delta_pred * support)
            base_safety_mode = str(args.safety_mode) if str(args.safety_mode) in {"signguard_only", "signsafe_easyguard", "semantic_blend"} else "none"
            safety = _apply_safety_guard(
                delta_pred=full_delta_pred,
                features={name: float(safe_float(row.get(name), 0.0)) for name in feature_names},
                top_contribs=top_contribs,
                forward_row_count=len(current_forward_rows),
                safety_mode=base_safety_mode,
                semantic_delta_pred=semantic_delta_pred,
            )
            signguard = _apply_safety_guard(
                delta_pred=full_delta_pred,
                features={name: float(safe_float(row.get(name), 0.0)) for name in feature_names},
                top_contribs=top_contribs,
                forward_row_count=len(current_forward_rows),
                safety_mode="signguard_only",
                semantic_delta_pred=semantic_delta_pred,
            )
            signguard_delta_pred = float(signguard["delta_pred"])
            gate_feature_map = _build_context_gate_features(
                row=row,
                features={name: float(safe_float(row.get(name), 0.0)) for name in feature_names},
                safety=safety,
                delta_full=full_delta_pred,
                delta_guard=signguard_delta_pred,
                delta_semantic=semantic_delta_pred,
                forward_row_count=len(current_forward_rows),
            )
            if str(args.safety_mode) == "context_gate":
                gate_scope = str(args.gate_training_scope)
                local_gate_rows = [history for history in gate_history_records if str(history.get("ticker") or "").upper() == str(row.get("ticker") or "").upper()]
                shared_gate_rows = list(gate_history_records)
                local_gate_result = {"pred": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": []}
                shared_gate_result = {"pred": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": []}
                if gate_scope == "company_local":
                    local_gate_result = _predict_history_gate(
                        train_rows=local_gate_rows,
                        x_map=gate_feature_map,
                        feature_names=CONTEXT_GATE_FEATURE_NAMES,
                        alpha=float(args.gate_alpha),
                        delta_cap_quantile=float(args.delta_cap_quantile),
                        shrink_k=float(args.gate_shrink_k),
                    )
                    gate_local_train_count = int(local_gate_result["train_count"])
                    gate_train_count = gate_local_train_count
                    if gate_local_train_count >= int(args.min_gate_train):
                        delta_pred = float(local_gate_result["pred"])
                        gate_top_contribs = list(local_gate_result["top_contribs"])
                        gate_used = True
                elif gate_scope == "shared_pooled":
                    shared_gate_result = _predict_history_gate(
                        train_rows=shared_gate_rows,
                        x_map=gate_feature_map,
                        feature_names=CONTEXT_GATE_FEATURE_NAMES,
                        alpha=float(args.gate_alpha),
                        delta_cap_quantile=float(args.delta_cap_quantile),
                        shrink_k=float(args.gate_shrink_k),
                    )
                    gate_shared_train_count = int(shared_gate_result["train_count"])
                    gate_train_count = gate_shared_train_count
                    if gate_shared_train_count >= int(args.min_gate_train):
                        delta_pred = float(shared_gate_result["pred"])
                        gate_top_contribs = list(shared_gate_result["top_contribs"])
                        gate_used = True
                else:
                    shared_gate_result = _predict_history_gate(
                        train_rows=shared_gate_rows,
                        x_map=gate_feature_map,
                        feature_names=CONTEXT_GATE_FEATURE_NAMES,
                        alpha=float(args.gate_alpha),
                        delta_cap_quantile=float(args.delta_cap_quantile),
                        shrink_k=float(args.gate_shrink_k),
                    )
                    local_gate_result = _predict_history_gate(
                        train_rows=local_gate_rows,
                        x_map=gate_feature_map,
                        feature_names=CONTEXT_GATE_FEATURE_NAMES,
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
                        gate_local_weight = gate_local_train_count / max(gate_local_train_count + float(args.gate_local_prior_k), EPS)
                        delta_pred = float((1.0 - gate_local_weight) * float(shared_gate_result["pred"]) + gate_local_weight * float(local_gate_result["pred"]))
                        gate_top_contribs = list(local_gate_result["top_contribs"] if gate_local_weight >= 0.5 else shared_gate_result["top_contribs"])
                        gate_used = True
                    elif shared_ok:
                        delta_pred = float(shared_gate_result["pred"])
                        gate_top_contribs = list(shared_gate_result["top_contribs"])
                        gate_used = True
                    elif local_ok:
                        delta_pred = float(local_gate_result["pred"])
                        gate_top_contribs = list(local_gate_result["top_contribs"])
                        gate_used = True
                gate_shared_delta_pred = float(shared_gate_result["pred"]) if math.isfinite(safe_float(shared_gate_result.get("pred"), float("nan"))) else float("nan")
                gate_local_delta_pred = float(local_gate_result["pred"]) if math.isfinite(safe_float(local_gate_result.get("pred"), float("nan"))) else float("nan")
                if gate_used:
                    mode = "native_cards_context_gate"
                    safety["guard_reason"] = f"context_gate::{gate_scope}"
                    safety["safety_scale"] = float(abs(delta_pred) / max(abs(full_delta_pred), EPS)) if abs(full_delta_pred) > EPS else 1.0
                else:
                    delta_pred = full_delta_pred
                    mode = "native_cards_ridge"
            else:
                delta_pred = float(safety["delta_pred"])
            scored_cards = []
            for card in current_forward_rows:
                influence = _card_influence(card, coefs_by_name)
                card_copy = dict(card)
                card_copy["approx_influence"] = float(influence)
                scored_cards.append(card_copy)
            scored_cards.sort(key=lambda c: c["approx_influence"], reverse=True)
            support_cards = scored_cards[:5]
            conflict_cards = list(reversed(sorted(scored_cards, key=lambda c: c["approx_influence"])))[:5]
            if not gate_used:
                mode = "native_cards_ridge"
        else:
            safety = _apply_safety_guard(
                delta_pred=delta_pred,
                features={name: float(safe_float(row.get(name), 0.0)) for name in feature_names},
                top_contribs=top_contribs,
                forward_row_count=0,
                safety_mode=str(args.safety_mode),
                semantic_delta_pred=float("nan"),
            )

        backbone_log = _safe_log(safe_float(row.get("pred_backbone_v2")))
        pred_log = float(backbone_log + delta_pred) if math.isfinite(backbone_log) else float("nan")
        pred = float(math.exp(pred_log)) if math.isfinite(pred_log) else float("nan")
        out_row = {str(col): row[col] for col in feature_df.columns}
        out_row["native_cards_mode"] = mode
        out_row["train_row_count"] = train_count
        out_row["pred_native_cards_v1"] = pred
        out_row["pred_native_cards_v1_log"] = pred_log
        out_row["pred_native_cards_delta_log"] = float(delta_pred)
        out_row["native_cards_full_delta_log"] = float(full_delta_pred)
        out_row["native_cards_signguard_delta_log"] = float(signguard_delta_pred)
        out_row["native_cards_semantic_delta_log"] = float(semantic_delta_pred) if math.isfinite(semantic_delta_pred) else float("nan")
        out_row["native_cards_safety_mode"] = str(args.safety_mode)
        out_row["native_cards_safety_scale"] = float(safety["safety_scale"])
        out_row["native_cards_semantic_clarity"] = float(safety["semantic_clarity"])
        out_row["native_cards_count_scale"] = float(safety["count_scale"])
        out_row["native_cards_conflict_scale"] = float(safety["conflict_scale"])
        out_row["native_cards_proxy_scale"] = float(safety["proxy_scale"])
        out_row["native_cards_top_feature_name"] = str(safety["top_feature_name"])
        out_row["native_cards_metadata_proxy_dominant"] = bool(safety["metadata_proxy_dominant"])
        out_row["native_cards_sign_guard_triggered"] = bool(safety["sign_guard_triggered"])
        out_row["native_cards_guard_reason"] = str(safety["guard_reason"])
        out_row["native_cards_context_gate_used"] = bool(gate_used)
        out_row["native_cards_context_gate_train_count"] = int(gate_train_count)
        out_row["native_cards_context_gate_shared_train_count"] = int(gate_shared_train_count)
        out_row["native_cards_context_gate_local_train_count"] = int(gate_local_train_count)
        out_row["native_cards_context_gate_local_weight"] = float(gate_local_weight)
        out_row["native_cards_context_gate_shared_delta_log"] = float(gate_shared_delta_pred) if math.isfinite(gate_shared_delta_pred) else float("nan")
        out_row["native_cards_context_gate_local_delta_log"] = float(gate_local_delta_pred) if math.isfinite(gate_local_delta_pred) else float("nan")
        out_row["native_cards_context_gate_top_contribs_json"] = json.dumps(sanitize_for_json(gate_top_contribs), ensure_ascii=False, allow_nan=False)
        out_row["top_feature_contribs_json"] = json.dumps(sanitize_for_json(top_contribs), ensure_ascii=False, allow_nan=False)
        out_row["top_support_cards_json"] = json.dumps(sanitize_for_json(support_cards), ensure_ascii=False, allow_nan=False)
        out_row["top_conflict_cards_json"] = json.dumps(sanitize_for_json(conflict_cards), ensure_ascii=False, allow_nan=False)
        pred_rows.append(out_row)
        if train_count >= int(args.min_train):
            gate_history_records.append(
                {
                    "ticker": str(row.get("ticker") or "").upper(),
                    "gate_target_delta": float(safe_float(row.get("native_card_delta_target_log"), float("nan"))),
                    **{name: float(safe_float(gate_feature_map.get(name), 0.0)) for name in CONTEXT_GATE_FEATURE_NAMES},
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    quarterly_csv = out_dir / "native_cards_v1_quarterly.csv"
    feature_csv = out_dir / "native_cards_v1_feature_panel.csv"
    company_csv = out_dir / "native_cards_v1_company_summary.csv"
    summary_json = out_dir / "native_cards_v1_summary.json"
    feature_df.to_csv(feature_csv, index=False)
    pred_df.to_csv(quarterly_csv, index=False)

    company_rows: List[Dict[str, Any]] = []
    for ticker, company_group in pred_df.groupby("ticker"):
        metrics = _metrics(company_group["target_revenue"], company_group["pred_native_cards_v1"])
        backbone_metrics = _metrics(company_group["target_revenue"], company_group["pred_backbone_v2"])
        company_cfg = company_lookup.get(str(ticker), {})
        best_model, best_mae = _load_best_stat(Path(resolve_repo_path(company_cfg.get("stat_baseline_metrics_csv"), project_root)))
        company_rows.append(
            {
                "ticker": str(ticker),
                "n": int(metrics["n"]),
                "native_cards_v1_mae": float(metrics["mae"]),
                "native_cards_v1_rmse": float(metrics["rmse"]),
                "native_cards_v1_mape": float(metrics["mape"]),
                "native_cards_v1_smape": float(metrics["smape"]),
                "backbone_v2_mae": float(backbone_metrics["mae"]),
                "delta_vs_backbone": float(metrics["mae"] - backbone_metrics["mae"]),
                "best_stat_model": best_model,
                "best_stat_mae": float(best_mae),
                "beats_best_stat": bool(metrics["mae"] < best_mae),
            }
        )
    company_df = pd.DataFrame(company_rows).sort_values("ticker")
    company_df.to_csv(company_csv, index=False)

    pooled = _metrics(pred_df["target_revenue"], pred_df["pred_native_cards_v1"])
    backbone_pooled = _metrics(pred_df["target_revenue"], pred_df["pred_backbone_v2"])
    summary = {
        "experiment_config": str(experiment_path),
        "card_table_jsonl": str(resolve_repo_path(args.card_table_jsonl, project_root)),
        "backbone_csv": str(backbone_path),
        "output_dir": str(out_dir),
        "company_count": int(len(company_df)),
        "eval_row_count": int(len(pred_df)),
        "native_cards_v1_pooled": pooled,
        "backbone_v2_pooled": backbone_pooled,
        "macro_mae": float(company_df["native_cards_v1_mae"].mean()) if not company_df.empty else float("nan"),
        "backbone_macro_mae": float(company_df["backbone_v2_mae"].mean()) if not company_df.empty else float("nan"),
        "beats_best_stat_companies": int(company_df["beats_best_stat"].sum()) if not company_df.empty else 0,
        "feature_names": feature_names,
        "alpha": float(args.alpha),
        "min_train": int(args.min_train),
        "delta_shrink_k": float(args.delta_shrink_k),
        "delta_cap_quantile": float(args.delta_cap_quantile),
        "training_scope": str(args.training_scope),
        "context_mode": str(args.context_mode),
        "safety_mode": str(args.safety_mode),
        "gate_alpha": float(args.gate_alpha),
        "min_gate_train": int(args.min_gate_train),
        "gate_shrink_k": float(args.gate_shrink_k),
        "gate_training_scope": str(args.gate_training_scope),
        "gate_local_prior_k": float(args.gate_local_prior_k),
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit("Implementation dependency only; run scripts/run_reference_replay.sh.")
