#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evidence_memory_residual.common import softmax_weights
from temporal_kg_memory_attention.pair_features import internal_item_alignment, top_item_matches


EPS = 1e-9
DEFAULT_EXPERIMENT_CONFIG = "replay_inputs/retained_336/experiment.json"
STAT_EXCLUDE = {"pred__guid_uniform", "pred__guid_triangular"}
DIRECT_GUIDANCE_PRED_COLS = ["pred__guid_mid", "pred__guid_affine", "pred__guid_blend"]
GUIDANCE_SANITY_MIN_PRIOR_RATIO = 0.55
GUIDANCE_SANITY_MAX_PRIOR_RATIO = 1.55
FORECAST_COLS = [
    "target_fiscal_quarter",
    "guid_low",
    "guid_high",
    "guid_mid",
    "guidance_score",
    "baseline_eligible",
    "guidance_availability",
    "demand_mentions",
    "supply_constraint_mentions",
    "margin_cost_mentions",
    "segment_guidance_mentions",
    "tone_score",
]
FORECAST_ZERO_FILL_COLS = {
    "demand_mentions",
    "supply_constraint_mentions",
    "margin_cost_mentions",
    "segment_guidance_mentions",
    "tone_score",
}
ATTR_COLS = [
    "quarter",
    "actual",
    "baseline_pred",
    "segment_share_count",
    "segment_share_top1",
    "segment_share_top2",
    "l2_mode",
    "l2_gate_label",
    "l2_rule_shock_signed",
    "l2_gate_internal_forward_total_used",
    "retrieve_path",
    "fg_peer_qoq_sum",
    "fg_peer_qoq_abs_sum",
    "fg_peer_yoy_sum",
    "fg_peer_yoy_abs_sum",
    "fg_drv_mass_pos_total",
    "fg_drv_mass_neg_total",
    "fg_shock_abs_mean_nz",
    "fg_shock_fwd_abs_sum",
    "fg_churn_conf_mass",
    "fg_kg_num_edges",
]
ATTR_ZERO_FILL_COLS = {
    col
    for col in ATTR_COLS
    if col
    not in {
        "quarter",
        "actual",
        "baseline_pred",
        "l2_mode",
        "l2_gate_label",
        "retrieve_path",
    }
}
REGIME_FEATURES = [
    "reg_recent_qoq",
    "reg_last_yoy",
    "reg_trend_slope4",
    "reg_vol_qoq4",
    "reg_vol_yoy4",
    "reg_same_quarter_support",
    "reg_recent_level_log",
    "guidance_numeric_available",
    "guidance_score_norm",
    "guid_band_ratio",
    "segment_share_top1",
    "segment_share_count",
    "tone_score",
    "demand_mentions",
    "supply_constraint_mentions",
]
RAW_SHOCK_FEATURES = [
    "guidance_numeric_available",
    "guidance_score_norm",
    "guid_band_ratio",
    "segment_share_top1",
    "segment_share_count",
    "tone_score",
    "demand_mentions",
    "supply_constraint_mentions",
    "margin_cost_mentions",
    "segment_guidance_mentions",
    "internal_balance",
    "internal_strength",
    "fg_peer_qoq_sum",
    "fg_peer_qoq_abs_sum",
    "fg_peer_yoy_sum",
    "fg_peer_yoy_abs_sum",
    "fg_drv_mass_pos_total",
    "fg_drv_mass_neg_total",
    "fg_shock_abs_mean_nz",
    "fg_shock_fwd_abs_sum",
    "fg_churn_conf_mass",
    "fg_kg_num_edges",
    "anchor_uncertainty",
    "anchor_blend_weight",
    "anchor_top1_gap_ratio",
    "reg_recent_qoq",
    "reg_last_yoy",
    "reg_trend_slope4",
    "reg_vol_qoq4",
    "reg_vol_yoy4",
    "reg_same_quarter_support",
]
FACTORIZED_SHOCK_FEATURES = [
    "factor_demand",
    "factor_supply",
    "factor_margin",
    "factor_tone",
    "factor_segment_focus",
    "factor_internal_balance",
    "factor_internal_confidence",
    "factor_transition",
    "anchor_uncertainty",
    "anchor_blend_weight",
    "guidance_numeric_available",
    "guidance_score_norm",
    "guid_band_ratio",
    "reg_same_quarter_support",
    "reg_vol_qoq4",
    "reg_last_yoy",
]
CONTEXT_GATE_FEATURES = [
    "gate_base_shock_log",
    "gate_raw_shock_log",
    "gate_base_weight",
    "gate_guidance_lock",
    "gate_guidance_numeric_available",
    "gate_guidance_score_norm",
    "gate_guid_band_ratio",
    "gate_internal_strength",
    "gate_internal_balance_abs",
    "gate_anchor_uncertainty",
    "gate_anchor_error_recent",
    "gate_anchor_error_same_quarter",
    "gate_anchor_error_same_guidance",
    "gate_regime_vol_qoq4",
    "gate_regime_same_quarter_support",
    "gate_regime_recent_qoq",
    "gate_factor_demand",
    "gate_factor_supply",
    "gate_factor_margin",
    "gate_factor_transition",
    "gate_shock_x_weak_guidance",
    "gate_shock_x_anchor_easy",
    "gate_shock_x_balance",
    "gate_shock_x_factor_disagreement",
]
TEMPORAL_KG_LEARNED_FEATURES = [
    "tkg_kg_abs_log",
    "tkg_kg_pred_var",
    "tkg_kg_effective_memory_count",
    "tkg_kg_attention_focus",
    "tkg_kg_segment_relation_overlap",
    "tkg_kg_directional_consistency",
    "tkg_kg_sign_agreement",
    "tkg_kg_support",
    "tkg_shock_base_abs_log",
    "tkg_shock_raw_abs_log",
    "tkg_shock_memory_sign_match",
    "tkg_anchor_uncertainty",
    "tkg_guidance_numeric_available",
    "tkg_guidance_score_norm",
    "tkg_guid_band_ratio",
    "tkg_internal_strength",
    "tkg_internal_balance_abs",
]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _resolve(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path_value).resolve()


def _quarter_key(quarter: str) -> Tuple[int, int]:
    if not isinstance(quarter, str) or not quarter.startswith("FY") or "_Q" not in quarter:
        return (0, 0)
    return (int(quarter[2:6]), int(quarter.split("_Q", 1)[1]))


def _quarter_number(quarter: str) -> int:
    return int(_quarter_key(quarter)[1])


def _sign(value: float, eps: float = 1e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = pd.to_numeric(pd.Series(list(y_true)), errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(pd.Series(list(y_pred)), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "smape": float("nan")}
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(yt), EPS)))
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(yt) + np.abs(yp), EPS)))
    return {"n": int(yt.size), "mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, np.ndarray | float]:
    if x.ndim != 2:
        raise ValueError("Expected 2D array")
    mask = np.isfinite(y)
    if x.size:
        mask = mask & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return {
            "means": np.zeros(x.shape[1] if x.ndim == 2 else 0, dtype=float),
            "stds": np.ones(x.shape[1] if x.ndim == 2 else 0, dtype=float),
            "intercept": 0.0,
            "coefs": np.zeros(x.shape[1] if x.ndim == 2 else 0, dtype=float),
        }
    means = x.mean(axis=0) if len(x) else np.zeros(x.shape[1], dtype=float)
    stds = x.std(axis=0)
    stds = np.where(stds > EPS, stds, 1.0)
    xz = (x - means) / stds
    xz = np.nan_to_num(xz, nan=0.0, posinf=0.0, neginf=0.0)
    intercept = float(y.mean()) if len(y) else 0.0
    y_center = y - intercept
    reg = float(alpha) * np.eye(x.shape[1], dtype=float)
    try:
        coefs = np.linalg.pinv(xz.T @ xz + reg) @ (xz.T @ y_center)
    except np.linalg.LinAlgError:
        coefs = np.zeros(x.shape[1], dtype=float)
    return {"means": means, "stds": stds, "intercept": intercept, "coefs": coefs}


def _predict_ridge(model: Mapping[str, np.ndarray | float], x: np.ndarray) -> float:
    means = np.asarray(model["means"], dtype=float)
    stds = np.asarray(model["stds"], dtype=float)
    coefs = np.asarray(model["coefs"], dtype=float)
    intercept = float(model["intercept"])
    xz = (x - means) / stds
    xz = np.nan_to_num(xz, nan=0.0, posinf=0.0, neginf=0.0)
    return float(intercept + xz @ coefs)


def _mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _safe_log(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(math.log(max(value, EPS)))


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(list(values), dtype=float)
    x = x - float(x.mean())
    denom = float((x ** 2).sum())
    if denom <= EPS:
        return 0.0
    return float((x @ (y - float(y.mean()))) / denom)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=FORECAST_COLS).copy()
    df = df.rename(columns={"target_fiscal_quarter": "quarter"})
    df["quarter"] = df["quarter"].astype(str)
    for col in [c for c in FORECAST_COLS if c not in {"target_fiscal_quarter", "guidance_availability"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FORECAST_ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    df["guidance_availability"] = df["guidance_availability"].fillna("none").astype(str)
    return df.drop_duplicates(subset=["quarter"], keep="last")


def _load_attr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [col for col in ATTR_COLS if col in df.columns]
    df = df[keep].copy()
    df["quarter"] = df["quarter"].astype(str)
    non_numeric = {"quarter", "l2_mode", "l2_gate_label", "retrieve_path"}
    for col in keep:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ATTR_ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df.drop_duplicates(subset=["quarter"], keep="last")


def _load_stat_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["quarter"] = df["fiscal_quarter"].astype(str)
    keep = ["quarter", "y_true", "guid_available"] + [col for col in df.columns if col.startswith("pred__") and col not in STAT_EXCLUDE]
    df = df[keep].copy()
    df = df.rename(columns={"y_true": "actual_stat", "guid_available": "stat_guid_available"})
    for col in df.columns:
        if col != "quarter":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["quarter"], keep="last")


def _load_best_stat_metrics(path: Path) -> Tuple[str, float]:
    df = pd.read_csv(path)
    df = df[~df["model"].isin(["guid_uniform", "guid_triangular"])].copy()
    best = df.sort_values("MAE").iloc[0]
    return str(best["model"]), float(best["MAE"])


def _repair_annual_total_like_q4_panel_rows(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "actual" not in panel.columns or "quarter" not in panel.columns:
        return panel

    out = panel.copy()
    out["quarter"] = out["quarter"].astype(str)
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    actual_by_quarter = {
        str(row["quarter"]): float(row["actual"])
        for _, row in out.iterrows()
        if pd.notna(row.get("actual"))
    }
    repaired_quarters = set()

    for idx, row in out.iterrows():
        quarter = str(row.get("quarter") or "")
        if not quarter.endswith("_Q4"):
            continue
        actual = _safe_float(row.get("actual"), float("nan"))
        if not math.isfinite(actual):
            continue
        fy = quarter[2:6]
        q1 = actual_by_quarter.get(f"FY{fy}_Q1")
        q2 = actual_by_quarter.get(f"FY{fy}_Q2")
        q3 = actual_by_quarter.get(f"FY{fy}_Q3")
        if q1 is None or q2 is None or q3 is None:
            continue
        q1q3 = np.asarray([q1, q2, q3], dtype=float)
        median_q1q3 = float(np.median(q1q3))
        max_q1q3 = float(np.max(q1q3))
        repaired_actual = float(actual - q1 - q2 - q3)
        if not (
            repaired_actual > 0.0
            and median_q1q3 > 0.0
            and actual >= 2.25 * median_q1q3
            and repaired_actual <= 1.75 * max_q1q3
        ):
            continue
        out.at[idx, "actual"] = repaired_actual
        actual_by_quarter[quarter] = repaired_actual
        repaired_quarters.add(quarter)

    if not repaired_quarters:
        return out

    out = out.sort_values("quarter", key=lambda s: s.map(_quarter_key)).reset_index(drop=True)
    actual_values = pd.to_numeric(out["actual"], errors="coerce")
    if "baseline_pred" in out.columns:
        out["baseline_pred"] = actual_values.shift(1)
    if "pred__naive" in out.columns:
        out["pred__naive"] = actual_values.shift(1)
    if "pred__seasonal_naive_q4" in out.columns:
        out["pred__seasonal_naive_q4"] = actual_values.shift(4)

    has_guid_mid = pd.to_numeric(out.get("guid_mid", pd.Series(np.nan, index=out.index)), errors="coerce").notna()
    if "pred__naive" in out.columns:
        for col in ["pred__guid_mid", "pred__guid_affine", "pred__guid_blend"]:
            if col in out.columns:
                out.loc[~has_guid_mid, col] = out.loc[~has_guid_mid, "pred__naive"]
    return out


def _apply_explicit_guidance_sanity_guard(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "guid_mid" not in panel.columns or "guidance_availability" not in panel.columns:
        return panel

    out = panel.copy()
    out["guidance_sanity_guardrail_applied"] = 0
    out["guidance_sanity_reason"] = ""
    out["guidance_sanity_original_availability"] = out["guidance_availability"].astype(str)
    for col in ["guid_low", "guid_high", "guid_mid"]:
        if col in out.columns:
            out[f"guidance_sanity_original_{col}"] = pd.to_numeric(out[col], errors="coerce")

    for idx, row in out.iterrows():
        if str(row.get("guidance_availability") or "") != "explicit_numeric":
            continue
        guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
        prior = _safe_float(row.get("pred__naive"), float("nan"))
        if not math.isfinite(prior):
            prior = _safe_float(row.get("baseline_pred"), float("nan"))
        if not (math.isfinite(guid_mid) and math.isfinite(prior) and guid_mid > 0.0 and prior > 0.0):
            continue
        ratio = float(guid_mid / prior)
        if GUIDANCE_SANITY_MIN_PRIOR_RATIO <= ratio <= GUIDANCE_SANITY_MAX_PRIOR_RATIO:
            continue

        out.at[idx, "guidance_sanity_guardrail_applied"] = 1
        out.at[idx, "guidance_sanity_reason"] = (
            f"explicit_guid_mid_prior_ratio_{ratio:.3f}_outside_"
            f"[{GUIDANCE_SANITY_MIN_PRIOR_RATIO:.2f},{GUIDANCE_SANITY_MAX_PRIOR_RATIO:.2f}]"
        )
        out.at[idx, "guidance_availability"] = "none"
        if "baseline_eligible" in out.columns:
            out.at[idx, "baseline_eligible"] = 0
        if "stat_guid_available" in out.columns:
            out.at[idx, "stat_guid_available"] = 0
        if "guidance_score" in out.columns:
            out.at[idx, "guidance_score"] = 0
        for col in ["guid_low", "guid_high", "guid_mid"]:
            if col in out.columns:
                out.at[idx, col] = np.nan
        naive = _safe_float(row.get("pred__naive"), float("nan"))
        if math.isfinite(naive):
            for col in DIRECT_GUIDANCE_PRED_COLS:
                if col in out.columns:
                    out.at[idx, col] = naive
    return out


def _history_regime(actual_hist: Sequence[float], quarter_hist: Sequence[str], current_q: int) -> Dict[str, float]:
    logs = [_safe_log(v) for v in actual_hist]
    logs = [v for v in logs if math.isfinite(v)]
    qoq = [logs[i] - logs[i - 1] for i in range(1, len(logs))]
    yoy = [logs[i] - logs[i - 4] for i in range(4, len(logs))]
    same_q_support = 0.0
    if current_q > 0 and len(quarter_hist) > 0:
        q_numbers = [_quarter_number(str(q)) for q in quarter_hist]
        same_q_support = min(1.0, q_numbers.count(current_q) / 3.0)
    return {
        "reg_recent_qoq": float(qoq[-1]) if qoq else 0.0,
        "reg_last_yoy": float(yoy[-1]) if yoy else 0.0,
        "reg_trend_slope4": float(_slope(logs[-4:])) if logs else 0.0,
        "reg_vol_qoq4": float(np.std(qoq[-4:], ddof=0)) if len(qoq) >= 2 else 0.0,
        "reg_vol_yoy4": float(np.std(yoy[-4:], ddof=0)) if len(yoy) >= 2 else 0.0,
        "reg_same_quarter_support": float(_clip(same_q_support, 0.0, 1.0)),
        "reg_recent_level_log": float(np.mean(logs[-4:])) if logs else 0.0,
    }


def _guidance_features(row: Mapping[str, Any]) -> Dict[str, float]:
    mid = _safe_float(row.get("guid_mid"), float("nan"))
    low = _safe_float(row.get("guid_low"), float("nan"))
    high = _safe_float(row.get("guid_high"), float("nan"))
    band_ratio = 0.0
    if math.isfinite(mid) and abs(mid) > EPS and math.isfinite(low) and math.isfinite(high):
        band_ratio = abs(high - low) / max(abs(mid), EPS)
    availability = str(row.get("guidance_availability") or "none")
    return {
        "guidance_numeric_available": 1.0 if availability == "explicit_numeric" and int(_safe_float(row.get("baseline_eligible"), 0.0)) == 1 else 0.0,
        "guidance_score_norm": float(_clip(_safe_float(row.get("guidance_score"), 0.0) / 20.0, 0.0, 1.0)),
        "guid_band_ratio": float(_clip(band_ratio, 0.0, 2.0)),
    }


def _guidance_lock(guidance_dict: Mapping[str, float]) -> float:
    lock = (
        float(_safe_float(guidance_dict.get("guidance_numeric_available"), 0.0))
        * float(_safe_float(guidance_dict.get("guidance_score_norm"), 0.0))
        * max(0.0, 1.0 - float(_safe_float(guidance_dict.get("guid_band_ratio"), 0.0)) / 0.15)
    )
    return float(_clip(0.0 if not math.isfinite(lock) else lock, 0.0, 1.0))


def _internal_features(row: Mapping[str, Any]) -> Dict[str, float]:
    pos = max(0.0, _safe_float(row.get("fg_drv_mass_pos_total"), 0.0))
    neg = max(0.0, _safe_float(row.get("fg_drv_mass_neg_total"), 0.0))
    total = pos + neg
    balance = 0.0 if total <= EPS else (pos - neg) / total
    strength = math.log1p(total + abs(_safe_float(row.get("fg_peer_qoq_abs_sum"), 0.0)) + abs(_safe_float(row.get("fg_peer_yoy_abs_sum"), 0.0)) + abs(_safe_float(row.get("fg_shock_fwd_abs_sum"), 0.0)))
    return {
        "internal_balance": float(_clip(balance, -1.0, 1.0)),
        "internal_strength": float(_clip(strength, 0.0, 8.0)),
    }


def _company_abstain_scale(ticker: str) -> float:
    if ticker in {"AVGO", "WMT", "ASML", "MU", "INTC"}:
        return 0.65
    if ticker in {"NVDA", "TSLA"}:
        return 1.00
    if ticker in {"AAPL", "MSFT", "GOOGL", "META", "AMZN"}:
        return 0.85
    return 0.80


def _build_anchor_features(row: Mapping[str, Any], model_cols: Sequence[str], regime: Mapping[str, float]) -> np.ndarray:
    vec: List[float] = []
    for col in model_cols:
        vec.append(_safe_log(_safe_float(row.get(col))))
    vec.extend(float(_safe_float(regime.get(name), 0.0)) for name in REGIME_FEATURES)
    return np.asarray([0.0 if not math.isfinite(v) else float(v) for v in vec], dtype=float)


def _build_shock_features(row: Mapping[str, Any], regime: Mapping[str, float], anchor_diag: Mapping[str, float]) -> np.ndarray:
    values = {
        **_guidance_features(row),
        **_internal_features(row),
        **regime,
        "segment_share_top1": float(_clip(_safe_float(row.get("segment_share_top1"), 0.0), 0.0, 1.0)),
        "segment_share_count": float(max(_safe_float(row.get("segment_share_count"), 0.0), 0.0)),
        "tone_score": float(_clip(_safe_float(row.get("tone_score"), 0.0), -10.0, 10.0)),
        "demand_mentions": float(_clip(_safe_float(row.get("demand_mentions"), 0.0), 0.0, 30.0)),
        "supply_constraint_mentions": float(_clip(_safe_float(row.get("supply_constraint_mentions"), 0.0), 0.0, 30.0)),
        "margin_cost_mentions": float(_clip(_safe_float(row.get("margin_cost_mentions"), 0.0), 0.0, 30.0)),
        "segment_guidance_mentions": float(_clip(_safe_float(row.get("segment_guidance_mentions"), 0.0), 0.0, 30.0)),
        "fg_peer_qoq_sum": float(_safe_float(row.get("fg_peer_qoq_sum"), 0.0)),
        "fg_peer_qoq_abs_sum": float(_safe_float(row.get("fg_peer_qoq_abs_sum"), 0.0)),
        "fg_peer_yoy_sum": float(_safe_float(row.get("fg_peer_yoy_sum"), 0.0)),
        "fg_peer_yoy_abs_sum": float(_safe_float(row.get("fg_peer_yoy_abs_sum"), 0.0)),
        "fg_drv_mass_pos_total": float(_safe_float(row.get("fg_drv_mass_pos_total"), 0.0)),
        "fg_drv_mass_neg_total": float(_safe_float(row.get("fg_drv_mass_neg_total"), 0.0)),
        "fg_shock_abs_mean_nz": float(_safe_float(row.get("fg_shock_abs_mean_nz"), 0.0)),
        "fg_shock_fwd_abs_sum": float(_safe_float(row.get("fg_shock_fwd_abs_sum"), 0.0)),
        "fg_churn_conf_mass": float(_safe_float(row.get("fg_churn_conf_mass"), 0.0)),
        "fg_kg_num_edges": float(_safe_float(row.get("fg_kg_num_edges"), 0.0)),
        "anchor_uncertainty": float(_safe_float(anchor_diag.get("anchor_uncertainty"), 0.0)),
        "anchor_blend_weight": float(_safe_float(anchor_diag.get("anchor_blend_weight"), 0.0)),
        "anchor_top1_gap_ratio": float(_safe_float(anchor_diag.get("anchor_top1_gap_ratio"), 0.0)),
    }
    return np.asarray([float(values[name]) for name in RAW_SHOCK_FEATURES], dtype=float)


def _factorized_internal_features(row: Mapping[str, Any], regime: Mapping[str, float], anchor_diag: Mapping[str, float]) -> Dict[str, float]:
    pos = max(0.0, _safe_float(row.get("fg_drv_mass_pos_total"), 0.0))
    neg = max(0.0, _safe_float(row.get("fg_drv_mass_neg_total"), 0.0))
    total = pos + neg
    balance = 0.0 if total <= EPS else (pos - neg) / total
    strength = math.log1p(total + abs(_safe_float(row.get("fg_peer_qoq_abs_sum"), 0.0)) + abs(_safe_float(row.get("fg_peer_yoy_abs_sum"), 0.0)) + abs(_safe_float(row.get("fg_shock_fwd_abs_sum"), 0.0)))
    return {
        "factor_demand": float(0.08 * _safe_float(row.get("demand_mentions"), 0.0) + 0.15 * _safe_float(row.get("fg_peer_qoq_sum"), 0.0) + 0.10 * _safe_float(row.get("fg_peer_yoy_sum"), 0.0) + 0.12 * pos),
        "factor_supply": float(-(0.10 * _safe_float(row.get("supply_constraint_mentions"), 0.0) + 0.15 * neg + 0.04 * _safe_float(row.get("fg_shock_fwd_abs_sum"), 0.0) + 0.04 * _safe_float(row.get("fg_churn_conf_mass"), 0.0))),
        "factor_margin": float(-0.08 * _safe_float(row.get("margin_cost_mentions"), 0.0)),
        "factor_tone": float(0.20 * _safe_float(row.get("tone_score"), 0.0)),
        "factor_segment_focus": float(0.20 * _clip(_safe_float(row.get("segment_share_top1"), 0.0), 0.0, 1.0)),
        "factor_internal_balance": float(0.40 * balance),
        "factor_internal_confidence": float(0.25 * _clip(strength, 0.0, 6.0)),
        "factor_transition": float(0.06 * _safe_float(row.get("segment_guidance_mentions"), 0.0) + 0.02 * _safe_float(row.get("fg_kg_num_edges"), 0.0)),
        "anchor_uncertainty": float(_safe_float(anchor_diag.get("anchor_uncertainty"), 0.0)),
        "anchor_blend_weight": float(_safe_float(anchor_diag.get("anchor_blend_weight"), 0.0)),
        "guidance_numeric_available": float(_guidance_features(row)["guidance_numeric_available"]),
        "guidance_score_norm": float(_guidance_features(row)["guidance_score_norm"]),
        "guid_band_ratio": float(_guidance_features(row)["guid_band_ratio"]),
        "reg_same_quarter_support": float(_safe_float(regime.get("reg_same_quarter_support"), 0.0)),
        "reg_vol_qoq4": float(_safe_float(regime.get("reg_vol_qoq4"), 0.0)),
        "reg_last_yoy": float(_safe_float(regime.get("reg_last_yoy"), 0.0)),
    }


def _build_anchor_error_seed_records(frame: pd.DataFrame, pred_col: str) -> List[Dict[str, Any]]:
    if pred_col not in frame.columns:
        return []
    records: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        actual = _safe_float(row.get("actual"))
        pred = _safe_float(row.get(pred_col))
        actual_log = _safe_log(actual)
        pred_log = _safe_log(pred)
        if not (math.isfinite(actual_log) and math.isfinite(pred_log)):
            continue
        records.append(
            {
                "quarter": str(row.get("quarter") or ""),
                "guidance_availability": str(row.get("guidance_availability") or "none"),
                "anchor_abs_log_error": float(abs(actual_log - pred_log)),
            }
        )
    return records


def _anchor_error_state(
    history: Sequence[Mapping[str, Any]],
    current_q: int,
    current_guidance: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    records = list(history)
    errors = [_safe_float(item.get("anchor_abs_log_error"), float("nan")) for item in records]
    overall_mean = _mean_or_nan(errors)
    recent_window = max(int(args.anchor_uncertainty_recent_window), 1)
    recent_mean = _mean_or_nan(errors[-recent_window:]) if errors else float("nan")
    same_q = [
        _safe_float(item.get("anchor_abs_log_error"), float("nan"))
        for item in records
        if _quarter_number(str(item.get("quarter") or "")) == int(current_q)
    ]
    same_guidance = [
        _safe_float(item.get("anchor_abs_log_error"), float("nan"))
        for item in records
        if str(item.get("guidance_availability") or "none") == str(current_guidance or "none")
    ]
    same_q_mean = _mean_or_nan(same_q)
    same_guidance_mean = _mean_or_nan(same_guidance)

    parts = [overall_mean, recent_mean]
    if len([v for v in same_q if math.isfinite(v)]) >= int(args.anchor_uncertainty_same_quarter_min):
        parts.append(same_q_mean)
    if len([v for v in same_guidance if math.isfinite(v)]) >= int(args.anchor_uncertainty_same_guidance_min):
        parts.append(same_guidance_mean)
    proxy_mean = _mean_or_nan(parts)
    uncertainty = 0.0
    if math.isfinite(proxy_mean):
        uncertainty = _clip(proxy_mean / max(float(args.anchor_uncertainty_tau), EPS), 0.0, 1.0)
    return {
        "anchor_error_history_n": float(len([v for v in errors if math.isfinite(v)])),
        "anchor_error_overall_abs_log": float(0.0 if not math.isfinite(overall_mean) else overall_mean),
        "anchor_error_recent_abs_log": float(0.0 if not math.isfinite(recent_mean) else recent_mean),
        "anchor_error_same_quarter_abs_log": float(0.0 if not math.isfinite(same_q_mean) else same_q_mean),
        "anchor_error_same_guidance_abs_log": float(0.0 if not math.isfinite(same_guidance_mean) else same_guidance_mean),
        "anchor_uncertainty_proxy": float(uncertainty),
    }


def _build_context_gate_feature_map(
    *,
    shock_raw: float,
    shock_weight_base: float,
    guidance_lock: float,
    guidance_dict: Mapping[str, float],
    internal_state: Mapping[str, float],
    factor_map: Mapping[str, float],
    regime: Mapping[str, float],
    anchor_diag: Mapping[str, float],
    anchor_error_state: Mapping[str, float],
) -> Dict[str, float]:
    shock_base = float(shock_raw) * float(shock_weight_base)
    factor_net = (
        float(_safe_float(factor_map.get("factor_demand"), 0.0))
        + float(_safe_float(factor_map.get("factor_supply"), 0.0))
        + float(_safe_float(factor_map.get("factor_margin"), 0.0))
        + 0.25 * float(_safe_float(factor_map.get("factor_tone"), 0.0))
        + 0.50 * float(_safe_float(factor_map.get("factor_transition"), 0.0))
    )
    factor_disagreement = 0.0
    if _sign(float(shock_raw)) != 0 and _sign(factor_net) != 0 and _sign(float(shock_raw)) != _sign(factor_net):
        factor_disagreement = 1.0
    return {
        "gate_base_shock_log": float(shock_base),
        "gate_raw_shock_log": float(shock_raw),
        "gate_base_weight": float(shock_weight_base),
        "gate_guidance_lock": float(guidance_lock),
        "gate_guidance_numeric_available": float(_safe_float(guidance_dict.get("guidance_numeric_available"), 0.0)),
        "gate_guidance_score_norm": float(_safe_float(guidance_dict.get("guidance_score_norm"), 0.0)),
        "gate_guid_band_ratio": float(_safe_float(guidance_dict.get("guid_band_ratio"), 0.0)),
        "gate_internal_strength": float(_safe_float(internal_state.get("internal_strength"), 0.0)),
        "gate_internal_balance_abs": float(abs(_safe_float(internal_state.get("internal_balance"), 0.0))),
        "gate_anchor_uncertainty": float(_safe_float(anchor_diag.get("anchor_uncertainty"), 0.0)),
        "gate_anchor_error_recent": float(_safe_float(anchor_error_state.get("anchor_error_recent_abs_log"), 0.0)),
        "gate_anchor_error_same_quarter": float(_safe_float(anchor_error_state.get("anchor_error_same_quarter_abs_log"), 0.0)),
        "gate_anchor_error_same_guidance": float(_safe_float(anchor_error_state.get("anchor_error_same_guidance_abs_log"), 0.0)),
        "gate_regime_vol_qoq4": float(_safe_float(regime.get("reg_vol_qoq4"), 0.0)),
        "gate_regime_same_quarter_support": float(_safe_float(regime.get("reg_same_quarter_support"), 0.0)),
        "gate_regime_recent_qoq": float(_safe_float(regime.get("reg_recent_qoq"), 0.0)),
        "gate_factor_demand": float(_safe_float(factor_map.get("factor_demand"), 0.0)),
        "gate_factor_supply": float(_safe_float(factor_map.get("factor_supply"), 0.0)),
        "gate_factor_margin": float(_safe_float(factor_map.get("factor_margin"), 0.0)),
        "gate_factor_transition": float(_safe_float(factor_map.get("factor_transition"), 0.0)),
        "gate_shock_x_weak_guidance": float(shock_base * (1.0 - float(guidance_lock))),
        "gate_shock_x_anchor_easy": float(shock_base * (1.0 - float(_safe_float(anchor_diag.get("anchor_uncertainty"), 0.0)))),
        "gate_shock_x_balance": float(shock_base * abs(_safe_float(internal_state.get("internal_balance"), 0.0))),
        "gate_shock_x_factor_disagreement": float(shock_base * factor_disagreement),
    }


def _optimal_gate_multiplier_target(shock_raw: float, shock_weight_base: float, shock_target_log: float) -> float:
    base_effect = float(shock_raw) * float(shock_weight_base)
    if not math.isfinite(base_effect) or abs(base_effect) <= EPS or not math.isfinite(float(shock_target_log)):
        return float("nan")
    if base_effect * float(shock_target_log) <= 0.0:
        return 0.0
    return float(_clip(abs(float(shock_target_log)) / max(abs(base_effect), EPS), 0.0, 1.0))


def _predict_context_gate(
    train_rows: Sequence[Mapping[str, Any]],
    current_features: Mapping[str, float],
    alpha: float,
    shrink_k: float,
) -> Dict[str, Any]:
    if not train_rows:
        return {
            "multiplier": 1.0,
            "pred_raw": float("nan"),
            "train_count": 0,
            "support": 0.0,
            "top_contribs": [],
        }
    x_rows = np.asarray(
        [[float(_safe_float(row.get(name), 0.0)) for name in CONTEXT_GATE_FEATURES] for row in train_rows],
        dtype=float,
    )
    y_rows = np.asarray([float(_safe_float(row.get("shock_gate_target"), float("nan"))) for row in train_rows], dtype=float)
    mask = np.isfinite(y_rows) & np.all(np.isfinite(x_rows), axis=1)
    x_train = x_rows[mask]
    y_train = y_rows[mask]
    train_count = int(len(y_train))
    if train_count == 0:
        return {
            "multiplier": 1.0,
            "pred_raw": float("nan"),
            "train_count": 0,
            "support": 0.0,
            "top_contribs": [],
        }
    model = _fit_ridge(x_train, y_train, float(alpha))
    x_cur = np.asarray([float(_safe_float(current_features.get(name), 0.0)) for name in CONTEXT_GATE_FEATURES], dtype=float)
    pred_raw = float(_clip(_predict_ridge(model, x_cur), 0.0, 1.0))
    support = float(train_count / max(train_count + float(shrink_k), EPS))
    multiplier = float(_clip((1.0 - support) * 1.0 + support * pred_raw, 0.0, 1.0))
    return {
        "multiplier": multiplier,
        "pred_raw": pred_raw,
        "train_count": train_count,
        "support": support,
        "top_contribs": _top_contributions(model, x_cur, CONTEXT_GATE_FEATURES, top_k=6),
    }


def _load_retrieve_payload(path_value: Any, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    path = str(path_value or "").strip()
    if not path:
        return {}
    if path in cache:
        return cache[path]
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    cache[path] = payload
    return payload


def _extract_internal_cards(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    cards_by_segment = payload.get("cards_by_segment", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(cards_by_segment, Mapping):
        return items
    for segment_name, cards in cards_by_segment.items():
        if not isinstance(cards, list):
            continue
        for raw_card in cards:
            if not isinstance(raw_card, Mapping):
                continue
            items.append(
                {
                    "segment": str(raw_card.get("segment") or segment_name or "unknown"),
                    "relation_family": str(raw_card.get("relation_family") or raw_card.get("category") or "unknown"),
                    "category": str(raw_card.get("category") or raw_card.get("relation_family") or "unknown"),
                    "polarity": str(raw_card.get("polarity") or "unknown"),
                    "strength": str(raw_card.get("strength") or "unknown"),
                    "confidence": _safe_float(raw_card.get("confidence"), 0.0),
                    "weight": _safe_float(raw_card.get("weight"), 0.0),
                    "persistence_hint": bool(raw_card.get("persistence_hint")),
                    "verbatim": str(raw_card.get("verbatim") or ""),
                }
            )
    return items


def _build_temporal_kg_stub(row: Mapping[str, Any], payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    items = _extract_internal_cards(payload)
    if not items:
        return None
    return {
        "quarter": str(row.get("quarter") or ""),
        "context": {
            "guidance_availability": str(row.get("guidance_availability") or "none"),
            "segment_share_top1": float(_clip(_safe_float(row.get("segment_share_top1"), 0.0), 0.0, 1.0)),
            "segment_share_top2": float(_clip(_safe_float(row.get("segment_share_top2"), 0.0), 0.0, 1.0)),
            "segment_share_count": float(max(_safe_float(row.get("segment_share_count"), 0.0), 0.0)),
            "tone_score": float(_clip(_safe_float(row.get("tone_score"), 0.0), -10.0, 10.0)),
            "demand_mentions": float(_clip(_safe_float(row.get("demand_mentions"), 0.0), 0.0, 30.0)),
            "supply_constraint_mentions": float(_clip(_safe_float(row.get("supply_constraint_mentions"), 0.0), 0.0, 30.0)),
        },
        "internal": {"items": items},
    }


def _temporal_kg_pair_score(current_stub: Mapping[str, Any], past_stub: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    current_ctx = current_stub.get("context", {}) if isinstance(current_stub, Mapping) else {}
    past_ctx = past_stub.get("context", {}) if isinstance(past_stub, Mapping) else {}
    same_quarter = 1.0 if _quarter_number(str(current_stub.get("quarter") or "")) == _quarter_number(str(past_stub.get("quarter") or "")) else 0.0
    guidance_match = 1.0 if str(current_ctx.get("guidance_availability") or "none") == str(past_ctx.get("guidance_availability") or "none") else 0.0
    segment_gap = (
        abs(_safe_float(current_ctx.get("segment_share_top1"), 0.0) - _safe_float(past_ctx.get("segment_share_top1"), 0.0))
        + 0.5 * abs(_safe_float(current_ctx.get("segment_share_top2"), 0.0) - _safe_float(past_ctx.get("segment_share_top2"), 0.0))
        + 0.1 * abs(_safe_float(current_ctx.get("segment_share_count"), 0.0) - _safe_float(past_ctx.get("segment_share_count"), 0.0))
    )
    tone_gap = abs(_safe_float(current_ctx.get("tone_score"), 0.0) - _safe_float(past_ctx.get("tone_score"), 0.0))
    demand_gap = abs(_safe_float(current_ctx.get("demand_mentions"), 0.0) - _safe_float(past_ctx.get("demand_mentions"), 0.0))
    supply_gap = abs(_safe_float(current_ctx.get("supply_constraint_mentions"), 0.0) - _safe_float(past_ctx.get("supply_constraint_mentions"), 0.0))
    align = internal_item_alignment(current_stub, past_stub)
    score = (
        1.20 * float(align.get("item_best_alignment", 0.0))
        + 0.90 * float(align.get("item_segment_relation_overlap", 0.0))
        + 0.55 * float(align.get("item_direction_alignment", 0.0))
        + 0.20 * float(align.get("item_polarity_alignment", 0.0))
        + 0.15 * same_quarter
        + 0.10 * guidance_match
        - 0.75 * segment_gap
        - 0.02 * tone_gap
        - 0.01 * demand_gap
        - 0.01 * supply_gap
    )
    diag = {
        "same_quarter": float(same_quarter),
        "guidance_match": float(guidance_match),
        "segment_gap": float(segment_gap),
        "tone_gap": float(tone_gap),
        "demand_gap": float(demand_gap),
        "supply_gap": float(supply_gap),
        **{str(k): float(v) for k, v in align.items()},
    }
    return float(score), diag


def _temporal_kg_gate_diag(
    current_stub: Optional[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": bool(args.temporal_kg_gate),
        "available": False,
        "reason": "off" if not bool(args.temporal_kg_gate) else "missing_current_retrieve",
        "predicted_correction_log": 0.0,
        "predicted_var": float("nan"),
        "effective_memory_count": 0.0,
        "attention_focus": 0.0,
        "segment_relation_overlap": 0.0,
        "directional_consistency": 0.0,
        "sign_agreement": 0.0,
        "support": 0.0,
        "retrieved_quarters": "",
        "top_matches_json": "",
    }
    if not bool(args.temporal_kg_gate):
        return out
    if current_stub is None:
        return out

    candidates: List[Tuple[float, Mapping[str, Any], Dict[str, float]]] = []
    for past in history_rows:
        past_stub = past.get("temporal_kg_stub")
        target = _safe_float(past.get("shock_target_log"), float("nan"))
        if past_stub is None or not math.isfinite(target):
            continue
        score, diag = _temporal_kg_pair_score(current_stub, past_stub)
        candidates.append((float(score), past, diag))

    if len(candidates) < int(args.temporal_kg_min_matches):
        out["reason"] = "insufficient_history_matches"
        return out

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_pairs = candidates[: int(args.temporal_kg_top_k)]
    weights = softmax_weights([item[0] for item in top_pairs], float(args.temporal_kg_temperature))
    target_logs = [_safe_float(item[1].get("shock_target_log"), 0.0) for item in top_pairs]
    pred_mean = sum(weight * target for weight, target in zip(weights, target_logs))
    pred_var = sum(weight * (target - pred_mean) ** 2 for weight, target in zip(weights, target_logs))
    weight_sq_sum = sum(weight ** 2 for weight in weights)
    effective_memory_count = 0.0 if weight_sq_sum <= EPS else 1.0 / weight_sq_sum
    abs_mass = sum(weight * abs(target) for weight, target in zip(weights, target_logs))
    directional_consistency = 0.0 if abs_mass <= EPS else abs(pred_mean) / abs_mass
    sign_agreement = abs(sum(weight * float(_sign(target)) for weight, target in zip(weights, target_logs)))
    attention_focus = sum(weight * float(item[2].get("item_best_alignment", 0.0)) for weight, item in zip(weights, top_pairs))
    segment_relation_overlap = sum(weight * float(item[2].get("item_segment_relation_overlap", 0.0)) for weight, item in zip(weights, top_pairs))
    var_conf = math.exp(-pred_var / max(float(args.temporal_kg_var_tau), EPS))
    neff_factor = min(1.0, effective_memory_count / max(float(args.temporal_kg_neff_scale), 1.0))
    support = var_conf * neff_factor * (0.5 + 0.5 * attention_focus) * (0.5 + 0.5 * segment_relation_overlap) * max(directional_consistency, 0.0)
    support = float(_clip(support, 0.0, 1.0))

    retrieved_quarters = "; ".join(f"{item[1]['quarter']}:{weights[idx]:.3f}" for idx, item in enumerate(top_pairs))
    top_matches_payload = []
    for idx, item in enumerate(top_pairs):
        top_matches_payload.append(
            {
                "quarter": str(item[1].get("quarter") or ""),
                "weight": round(float(weights[idx]), 6),
                "score": round(float(item[0]), 6),
                "target_correction_log": round(float(_safe_float(item[1].get("shock_target_log"), 0.0)), 6),
                "item_best_alignment": round(float(item[2].get("item_best_alignment", 0.0)), 6),
                "item_segment_relation_overlap": round(float(item[2].get("item_segment_relation_overlap", 0.0)), 6),
                "item_direction_alignment": round(float(item[2].get("item_direction_alignment", 0.0)), 6),
                "top_item_matches": top_item_matches(current_stub, item[1].get("temporal_kg_stub") or {}, top_k=2),
            }
        )

    out.update(
        {
            "available": True,
            "reason": "ok",
            "predicted_correction_log": float(pred_mean),
            "predicted_var": float(pred_var),
            "effective_memory_count": float(effective_memory_count),
            "attention_focus": float(attention_focus),
            "segment_relation_overlap": float(segment_relation_overlap),
            "directional_consistency": float(directional_consistency),
            "sign_agreement": float(sign_agreement),
            "support": float(support),
            "retrieved_quarters": retrieved_quarters,
            "top_matches_json": json.dumps(top_matches_payload, ensure_ascii=False),
        }
    )
    return out


def _temporal_kg_gate_multiplier(kg_diag: Mapping[str, Any], shock_raw: float, args: argparse.Namespace) -> Dict[str, Any]:
    out = {
        "multiplier": 1.0,
        "relation": "off",
        "shock_sign": int(_sign(shock_raw)),
        "memory_sign": int(_sign(_safe_float(kg_diag.get("predicted_correction_log"), 0.0))),
    }
    if not bool(args.temporal_kg_gate) or not bool(kg_diag.get("available", False)):
        return out

    support = float(_clip(_safe_float(kg_diag.get("support"), 0.0), 0.0, 1.0))
    shock_sign = int(out["shock_sign"])
    memory_sign = int(out["memory_sign"])
    memory_abs = abs(_safe_float(kg_diag.get("predicted_correction_log"), 0.0))

    relation = "neutral"
    multiplier = 1.0
    if shock_sign != 0 and memory_sign != 0:
        if shock_sign == memory_sign:
            relation = "confirm"
        elif memory_abs >= float(args.temporal_kg_conflict_min_abs):
            relation = "conflict"
            multiplier = 1.0 - support * (1.0 - float(args.temporal_kg_conflict_multiplier))
        else:
            relation = "weak_conflict"
            multiplier = 1.0 - 0.5 * support * (1.0 - float(args.temporal_kg_min_multiplier))
    out.update({"multiplier": float(_clip(multiplier, 0.0, 1.0)), "relation": relation})
    return out


def _optimal_temporal_kg_gate_target(shock_raw: float, shock_weight_base: float, shock_target_log: float) -> float:
    return _optimal_gate_multiplier_target(shock_raw, shock_weight_base, shock_target_log)


def _build_temporal_kg_learned_feature_map(
    *,
    kg_diag: Mapping[str, Any],
    shock_raw: float,
    shock_weight_base: float,
    guidance_dict: Mapping[str, float],
    internal_state: Mapping[str, float],
    anchor_diag: Mapping[str, float],
) -> Dict[str, float]:
    kg_pred = float(_safe_float(kg_diag.get("predicted_correction_log"), 0.0))
    shock_base = float(shock_raw) * float(shock_weight_base)
    shock_sign = _sign(shock_base)
    kg_sign = _sign(kg_pred)
    sign_match = 0.0
    if shock_sign != 0 and kg_sign != 0:
        sign_match = 1.0 if shock_sign == kg_sign else -1.0
    return {
        "tkg_kg_abs_log": float(abs(kg_pred)),
        "tkg_kg_pred_var": float(max(_safe_float(kg_diag.get("predicted_var"), 0.0), 0.0)),
        "tkg_kg_effective_memory_count": float(max(_safe_float(kg_diag.get("effective_memory_count"), 0.0), 0.0)),
        "tkg_kg_attention_focus": float(_clip(_safe_float(kg_diag.get("attention_focus"), 0.0), 0.0, 1.0)),
        "tkg_kg_segment_relation_overlap": float(_clip(_safe_float(kg_diag.get("segment_relation_overlap"), 0.0), 0.0, 1.0)),
        "tkg_kg_directional_consistency": float(_clip(_safe_float(kg_diag.get("directional_consistency"), 0.0), 0.0, 1.0)),
        "tkg_kg_sign_agreement": float(_clip(_safe_float(kg_diag.get("sign_agreement"), 0.0), 0.0, 1.0)),
        "tkg_kg_support": float(_clip(_safe_float(kg_diag.get("support"), 0.0), 0.0, 1.0)),
        "tkg_shock_base_abs_log": float(abs(shock_base)),
        "tkg_shock_raw_abs_log": float(abs(float(shock_raw))),
        "tkg_shock_memory_sign_match": float(sign_match),
        "tkg_anchor_uncertainty": float(_clip(_safe_float(anchor_diag.get("anchor_uncertainty"), 0.0), 0.0, 1.0)),
        "tkg_guidance_numeric_available": float(_clip(_safe_float(guidance_dict.get("guidance_numeric_available"), 0.0), 0.0, 1.0)),
        "tkg_guidance_score_norm": float(_clip(_safe_float(guidance_dict.get("guidance_score_norm"), 0.0), 0.0, 1.0)),
        "tkg_guid_band_ratio": float(max(_safe_float(guidance_dict.get("guid_band_ratio"), 0.0), 0.0)),
        "tkg_internal_strength": float(max(_safe_float(internal_state.get("internal_strength"), 0.0), 0.0)),
        "tkg_internal_balance_abs": float(abs(_safe_float(internal_state.get("internal_balance"), 0.0))),
    }


def _temporal_kg_learned_gate_state(
    *,
    gate_history: pd.DataFrame,
    kg_diag: Mapping[str, Any],
    heuristic_state: Mapping[str, Any],
    shock_raw: float,
    shock_weight_base: float,
    guidance_dict: Mapping[str, float],
    internal_state: Mapping[str, float],
    anchor_diag: Mapping[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = dict(heuristic_state)
    out.update(
        {
            "mode": "heuristic_fallback",
            "learned_train_n": 0,
            "learned_pred_raw": float("nan"),
            "learned_blend_weight": 0.0,
            "learned_top_contribs": "",
        }
    )
    if not bool(args.temporal_kg_gate) or str(args.temporal_kg_gate_mode) != "learned":
        out["mode"] = "heuristic"
        return out
    if not bool(kg_diag.get("available", False)):
        return out
    if str(heuristic_state.get("relation") or "neutral") not in {"conflict", "weak_conflict"}:
        out["mode"] = "heuristic_confirm"
        return out

    feature_map = _build_temporal_kg_learned_feature_map(
        kg_diag=kg_diag,
        shock_raw=shock_raw,
        shock_weight_base=shock_weight_base,
        guidance_dict=guidance_dict,
        internal_state=internal_state,
        anchor_diag=anchor_diag,
    )
    current_x = np.asarray([float(feature_map[name]) for name in TEMPORAL_KG_LEARNED_FEATURES], dtype=float)

    train_rows = gate_history.copy()
    if train_rows.empty or "temporal_kg_gate_target" not in train_rows.columns:
        return out
    if "temporal_kg_relation" in train_rows.columns:
        train_rows = train_rows[train_rows["temporal_kg_relation"].astype(str).isin(["conflict", "weak_conflict"])].copy()
    train_rows["temporal_kg_gate_target"] = pd.to_numeric(train_rows["temporal_kg_gate_target"], errors="coerce")
    train_rows = train_rows[np.isfinite(train_rows["temporal_kg_gate_target"].to_numpy(dtype=float))].copy()
    if train_rows.empty:
        return out

    usable_cols = [name for name in TEMPORAL_KG_LEARNED_FEATURES if name in train_rows.columns]
    if len(usable_cols) != len(TEMPORAL_KG_LEARNED_FEATURES):
        return out
    x_train = train_rows[usable_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_train = train_rows["temporal_kg_gate_target"].to_numpy(dtype=float)
    mask = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    x_train = x_train[mask]
    y_train = y_train[mask]
    if len(y_train) < int(args.temporal_kg_learned_min_train):
        return out

    model = _fit_ridge(x_train, y_train, float(args.temporal_kg_learned_alpha))
    pred_raw = float(_clip(_predict_ridge(model, current_x), 0.0, 1.0))
    blend_weight = float(len(y_train) / max(len(y_train) + float(args.temporal_kg_learned_shrink_k), EPS))
    multiplier = float(_clip((1.0 - blend_weight) * float(heuristic_state.get("multiplier", 1.0)) + blend_weight * pred_raw, 0.0, 1.0))
    contribs = _top_contributions(model, current_x, TEMPORAL_KG_LEARNED_FEATURES, top_k=5)
    out.update(
        {
            "multiplier": multiplier,
            "mode": "learned_blend",
            "learned_train_n": int(len(y_train)),
            "learned_pred_raw": pred_raw,
            "learned_blend_weight": blend_weight,
            "learned_top_contribs": json.dumps(contribs, ensure_ascii=False),
        }
    )
    return out


def _top_contributions(model: Mapping[str, Any], x: np.ndarray, feature_names: Sequence[str], top_k: int = 3) -> List[Tuple[str, float]]:
    means = np.asarray(model["means"], dtype=float)
    stds = np.asarray(model["stds"], dtype=float)
    coefs = np.asarray(model["coefs"], dtype=float)
    xz = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    pairs = [(str(name), float(value)) for name, value in zip(feature_names, xz * coefs)]
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs[:top_k]


def _prepare_company_panel(company: Mapping[str, Any], project_root: Path) -> Tuple[pd.DataFrame, str, float, str, float, List[Dict[str, Any]]]:
    company_cfg = _load_json(_resolve(str(company.get("company_config")), project_root))
    forecast_csv = _resolve(str(company_cfg["data_paths"]["forecast_dataset_csv"]), project_root)
    forecast_df = _load_forecast(forecast_csv)
    attr_df = _load_attr(_resolve(str(company.get("internal_attribution_csv")), project_root))
    stat_df = _load_stat_predictions(_resolve(str(company.get("stat_baseline_predictions_csv")), project_root))
    metrics_csv = _resolve(str(company.get("stat_baseline_metrics_csv")), project_root)
    best_model, best_mae = _load_best_stat_metrics(metrics_csv)

    panel_full = attr_df.merge(forecast_df, on="quarter", how="left")
    panel_full = panel_full.merge(stat_df, on="quarter", how="inner")
    if "actual_stat" in panel_full.columns:
        panel_full = panel_full.drop(columns=["actual_stat"])
    panel_full = _repair_annual_total_like_q4_panel_rows(panel_full)
    panel_full = _apply_explicit_guidance_sanity_guard(panel_full)
    start_q = str(company.get("evaluation_start_fq"))
    prehist = panel_full[panel_full["quarter"].map(lambda q: _quarter_key(q) < _quarter_key(start_q))].copy()
    prehist_best_model = best_model
    prehist_best_mae = float("nan")
    if not prehist.empty:
        candidates = [col for col in panel_full.columns if col.startswith("pred__") and col not in STAT_EXCLUDE]
        scores = []
        for col in candidates:
            met = _metrics(prehist["actual"], prehist[col])
            if int(met["n"]) > 0 and math.isfinite(float(met["mae"])):
                scores.append((col, float(met["mae"])))
        if scores:
            scores.sort(key=lambda item: item[1])
            prehist_best_model = scores[0][0][len("pred__") :]
            prehist_best_mae = float(scores[0][1])
    prehist_anchor_error_history = _build_anchor_error_seed_records(prehist, f"pred__{prehist_best_model}")
    end_q = str(company.get("evaluation_end_fq"))
    panel = panel_full[panel_full["quarter"].map(lambda q: _quarter_key(start_q) <= _quarter_key(q) <= _quarter_key(end_q))].copy()
    panel["quarter"] = panel["quarter"].astype(str)
    panel = panel.sort_values("quarter", key=lambda s: s.map(_quarter_key)).reset_index(drop=True)
    return panel, best_model, best_mae, prehist_best_model, prehist_best_mae, prehist_anchor_error_history


def _historical_model_scores(hist: pd.DataFrame, current_q: int, current_guidance: str, model_cols: Sequence[str], args: argparse.Namespace) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for col in model_cols:
        cur = hist[["actual", "quarter", "guidance_availability", col]].copy()
        cur["actual"] = pd.to_numeric(cur["actual"], errors="coerce")
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
        cur = cur.dropna(subset=["actual", col])
        if len(cur) < int(args.anchor_min_train):
            continue
        overall = _metrics(cur["actual"], cur[col])["mae"]
        score = float(overall)
        same_q = cur[cur["quarter"].astype(str).map(_quarter_number) == current_q]
        if len(same_q) >= int(args.anchor_same_quarter_min):
            same_q_mae = _metrics(same_q["actual"], same_q[col])["mae"]
            score = (1.0 - float(args.anchor_same_quarter_weight)) * score + float(args.anchor_same_quarter_weight) * float(same_q_mae)
        same_guid = cur[cur["guidance_availability"].astype(str) == current_guidance]
        if len(same_guid) >= int(args.anchor_same_guidance_min):
            same_guid_mae = _metrics(same_guid["actual"], same_guid[col])["mae"]
            score = (1.0 - float(args.anchor_same_guidance_weight)) * score + float(args.anchor_same_guidance_weight) * float(same_guid_mae)
        scores[col] = float(score)
    return scores


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Company-specific statistical-anchor plus internal-shock forecasting prototype.")
    ap.add_argument("--experiment_config", default=DEFAULT_EXPERIMENT_CONFIG)
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--top_model_pool", type=int, default=5)
    ap.add_argument("--anchor_mode", default="online_top_pool", choices=["online_top_pool", "frozen_prehistory_best"])
    ap.add_argument("--anchor_uncertainty_mode", default="legacy_zero", choices=["legacy_zero", "historical_proxy"])
    ap.add_argument("--anchor_uncertainty_recent_window", type=int, default=6)
    ap.add_argument("--anchor_uncertainty_same_quarter_min", type=int, default=2)
    ap.add_argument("--anchor_uncertainty_same_guidance_min", type=int, default=3)
    ap.add_argument("--anchor_uncertainty_tau", type=float, default=0.10)
    ap.add_argument("--shock_feature_mode", default="raw", choices=["raw", "factorized"])
    ap.add_argument("--shock_gate_mode", default="legacy_company_scale", choices=["legacy_company_scale", "context_ridge"])
    ap.add_argument("--shock_gate_training_scope", default="company_local", choices=["company_local", "shared_pooled", "shared_blend"])
    ap.add_argument("--shock_gate_alpha", type=float, default=8.0)
    ap.add_argument("--shock_gate_min_train", type=int, default=6)
    ap.add_argument("--shock_gate_shrink_k", type=float, default=12.0)
    ap.add_argument("--shock_gate_local_prior_k", type=float, default=8.0)
    ap.add_argument("--shock_global_scale", type=float, default=1.0)
    ap.add_argument("--anchor_min_train", type=int, default=8)
    ap.add_argument("--anchor_same_quarter_min", type=int, default=2)
    ap.add_argument("--anchor_same_guidance_min", type=int, default=3)
    ap.add_argument("--anchor_same_quarter_weight", type=float, default=0.35)
    ap.add_argument("--anchor_same_guidance_weight", type=float, default=0.20)
    ap.add_argument("--anchor_alpha", type=float, default=8.0)
    ap.add_argument("--anchor_shrink_k", type=float, default=16.0)
    ap.add_argument("--anchor_dispersion_tau", type=float, default=0.12)
    ap.add_argument("--shock_min_train", type=int, default=10)
    ap.add_argument("--shock_alpha", type=float, default=8.0)
    ap.add_argument("--shock_shrink_k", type=float, default=16.0)
    ap.add_argument("--shock_max_abs_log_delta", type=float, default=0.14)
    ap.add_argument("--temporal_kg_gate", action="store_true", default=False)
    ap.add_argument("--temporal_kg_gate_mode", default="heuristic", choices=["heuristic", "learned"])
    ap.add_argument("--temporal_kg_learned_scope", default="shared", choices=["same_company", "shared"])
    ap.add_argument("--temporal_kg_top_k", type=int, default=3)
    ap.add_argument("--temporal_kg_min_matches", type=int, default=3)
    ap.add_argument("--temporal_kg_temperature", type=float, default=0.35)
    ap.add_argument("--temporal_kg_neff_scale", type=float, default=3.0)
    ap.add_argument("--temporal_kg_var_tau", type=float, default=0.04)
    ap.add_argument("--temporal_kg_min_multiplier", type=float, default=0.80)
    ap.add_argument("--temporal_kg_conflict_multiplier", type=float, default=0.45)
    ap.add_argument("--temporal_kg_conflict_min_abs", type=float, default=0.03)
    ap.add_argument("--temporal_kg_learned_min_train", type=int, default=8)
    ap.add_argument("--temporal_kg_learned_alpha", type=float, default=8.0)
    ap.add_argument("--temporal_kg_learned_shrink_k", type=float, default=8.0)
    ap.add_argument("--output_dir", default="output/csais_v1_all12")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    exp = _load_json(_resolve(args.experiment_config, project_root))
    requested = {t.upper() for t in args.tickers}
    out_dir = _resolve(args.output_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_quarterly: List[pd.DataFrame] = []
    per_company_summary: List[Dict[str, Any]] = []
    temporal_kg_global_history: List[Dict[str, Any]] = []
    shock_gate_global_history: List[Dict[str, Any]] = []
    for company in exp.get("companies", []):
        ticker = str(company.get("ticker", "")).upper()
        if requested and ticker not in requested:
            continue
        panel, best_stat_model, best_stat_mae, prehist_best_model, prehist_best_mae, prehist_anchor_error_history = _prepare_company_panel(company, project_root)
        model_cols = [col for col in panel.columns if col.startswith("pred__") and col not in STAT_EXCLUDE]
        quarterly_rows: List[Dict[str, Any]] = []
        actual_hist: List[float] = []
        quarter_hist: List[str] = []
        retrieve_cache: Dict[str, Dict[str, Any]] = {}
        temporal_kg_history: List[Dict[str, Any]] = []
        anchor_error_history: List[Dict[str, Any]] = list(prehist_anchor_error_history)
        for idx, row in panel.iterrows():
            hist_processed = pd.DataFrame(quarterly_rows)
            current_q = _quarter_number(str(row.get("quarter") or ""))
            current_guidance = str(row.get("guidance_availability") or "none")
            regime = _history_regime(actual_hist, quarter_hist, current_q)
            guidance_dict = _guidance_features(row)
            regime.update(guidance_dict)
            regime.update(
                {
                    "segment_share_top1": float(_clip(_safe_float(row.get("segment_share_top1"), 0.0), 0.0, 1.0)),
                    "segment_share_count": float(max(_safe_float(row.get("segment_share_count"), 0.0), 0.0)),
                    "tone_score": float(_clip(_safe_float(row.get("tone_score"), 0.0), -10.0, 10.0)),
                    "demand_mentions": float(_clip(_safe_float(row.get("demand_mentions"), 0.0), 0.0, 30.0)),
                    "supply_constraint_mentions": float(_clip(_safe_float(row.get("supply_constraint_mentions"), 0.0), 0.0, 30.0)),
                }
            )
            anchor_error_state = _anchor_error_state(anchor_error_history, current_q, current_guidance, args)

            if str(args.anchor_mode) == "frozen_prehistory_best":
                frozen_col = f"pred__{prehist_best_model}"
                top_models = [frozen_col] if frozen_col in model_cols and math.isfinite(_safe_float(row.get(frozen_col))) else []
                scores = {frozen_col: float(prehist_best_mae)} if top_models else {}
            else:
                scores = _historical_model_scores(hist_processed, current_q, current_guidance, model_cols, args) if len(hist_processed) else {}
                if scores:
                    top_models = [name for name, _ in sorted(scores.items(), key=lambda item: item[1])[: int(args.top_model_pool)]]
                else:
                    top_models = [col for col in model_cols if math.isfinite(_safe_float(row.get(col)))][: int(args.top_model_pool)]
            top_models = [col for col in top_models if math.isfinite(_safe_float(row.get(col)))]
            if not top_models:
                continue
            best_model = top_models[0]
            best_model_pred = _safe_float(row.get(best_model))
            best_model_log = _safe_log(best_model_pred)

            anchor_blend_weight = 0.0
            anchor_uncertainty = 0.0
            anchor_top1_gap_ratio = 0.0
            anchor_raw_log = best_model_log
            anchor_log = best_model_log
            if len(hist_processed) >= int(args.anchor_min_train):
                x_rows: List[np.ndarray] = []
                y_rows: List[float] = []
                for _, past in hist_processed.iterrows():
                    if any(not math.isfinite(_safe_float(past.get(col))) for col in top_models):
                        continue
                    x_rows.append(_build_anchor_features(past, top_models, {name: _safe_float(past.get(name), 0.0) for name in REGIME_FEATURES}))
                    y_rows.append(_safe_log(_safe_float(past.get("actual"))))
                if len(x_rows) >= int(args.anchor_min_train):
                    x_train = np.vstack(x_rows)
                    y_train = np.asarray(y_rows, dtype=float)
                    model = _fit_ridge(x_train, y_train, float(args.anchor_alpha))
                    x_cur = _build_anchor_features(row, top_models, regime)
                    if str(args.anchor_mode) == "frozen_prehistory_best":
                        anchor_raw_log = best_model_log
                        anchor_uncertainty = 0.0
                        anchor_blend_weight = 0.0
                        anchor_log = best_model_log
                        anchor_top1_gap_ratio = 0.0
                    else:
                        anchor_raw_log = _predict_ridge(model, x_cur)
                        pred_logs = [_safe_log(_safe_float(row.get(col))) for col in top_models]
                        pred_logs = [v for v in pred_logs if math.isfinite(v)]
                        dispersion = float(np.std(pred_logs, ddof=0)) if len(pred_logs) >= 2 else 0.0
                        anchor_uncertainty = _clip(dispersion / max(float(args.anchor_dispersion_tau), EPS), 0.0, 1.0)
                        support_weight = len(x_rows) / max(len(x_rows) + float(args.anchor_shrink_k), EPS)
                        anchor_blend_weight = _clip(support_weight * (0.4 + 0.6 * anchor_uncertainty), 0.0, 1.0)
                        anchor_log = (1.0 - anchor_blend_weight) * best_model_log + anchor_blend_weight * anchor_raw_log
                        anchor_top1_gap_ratio = (math.exp(anchor_raw_log) - best_model_pred) / max(abs(best_model_pred), 1.0)
            if str(args.anchor_uncertainty_mode) == "historical_proxy":
                anchor_uncertainty = max(anchor_uncertainty, float(_safe_float(anchor_error_state.get("anchor_uncertainty_proxy"), 0.0)))
            anchor_pred = math.exp(anchor_log) if math.isfinite(anchor_log) else float(best_model_pred)

            anchor_diag_state = {
                "anchor_uncertainty": anchor_uncertainty,
                "anchor_blend_weight": anchor_blend_weight,
                "anchor_top1_gap_ratio": anchor_top1_gap_ratio,
            }
            retrieve_payload = _load_retrieve_payload(row.get("retrieve_path"), retrieve_cache)
            temporal_kg_stub = _build_temporal_kg_stub(row, retrieve_payload)
            temporal_kg_diag = _temporal_kg_gate_diag(temporal_kg_stub, temporal_kg_history, args)
            factor_map = _factorized_internal_features(row, regime, anchor_diag_state)
            if str(args.shock_feature_mode) == "factorized":
                shock_feature_names = FACTORIZED_SHOCK_FEATURES
                shock_signal = np.asarray([float(factor_map[name]) for name in shock_feature_names], dtype=float)
            else:
                shock_feature_names = RAW_SHOCK_FEATURES
                shock_signal = _build_shock_features(row, regime, anchor_diag_state)
            internal_state = _internal_features(row)
            guidance_lock = _guidance_lock(guidance_dict)
            shock_raw = 0.0
            shock_weight = 0.0
            shock_weight_base = 0.0
            shock_weight_pre_tkg = 0.0
            shock_gate_state: Dict[str, Any] = {
                "multiplier": 1.0,
                "used": False,
                "scope": "off",
                "train_count": 0,
                "local_train_count": 0,
                "shared_train_count": 0,
                "local_weight": 0.0,
                "pred_raw": float("nan"),
                "pred_local": float("nan"),
                "pred_shared": float("nan"),
                "support": 0.0,
                "top_contribs": [],
            }
            temporal_kg_gate_state = _temporal_kg_gate_multiplier(temporal_kg_diag, shock_raw, args)
            temporal_kg_feature_map = _build_temporal_kg_learned_feature_map(
                kg_diag=temporal_kg_diag,
                shock_raw=shock_raw,
                shock_weight_base=shock_weight_base,
                guidance_dict=guidance_dict,
                internal_state=internal_state,
                anchor_diag=anchor_diag_state,
            )
            if len(hist_processed) >= int(args.shock_min_train):
                x_rows = []
                y_rows = []
                for _, past in hist_processed.iterrows():
                    target = _safe_float(past.get("shock_target_log"), float("nan"))
                    if not math.isfinite(target):
                        continue
                    x_rows.append(np.asarray([_safe_float(past.get(name), 0.0) for name in shock_feature_names], dtype=float))
                    y_rows.append(target)
                if len(x_rows) >= int(args.shock_min_train):
                    x_train = np.vstack(x_rows)
                    y_train = np.asarray(y_rows, dtype=float)
                    shock_model = _fit_ridge(x_train, y_train, float(args.shock_alpha))
                    shock_raw = _predict_ridge(shock_model, shock_signal)
                    shock_raw = _clip(shock_raw, -float(args.shock_max_abs_log_delta), float(args.shock_max_abs_log_delta))
                    evidence_gate = (1.0 - math.exp(-internal_state["internal_strength"] / 1.5)) * (0.4 + 0.6 * abs(internal_state["internal_balance"]))
                    evidence_gate = float(0.0 if not math.isfinite(evidence_gate) else evidence_gate)
                    support = len(x_rows) / max(len(x_rows) + float(args.shock_shrink_k), EPS)
                    shock_weight_base_raw = support * evidence_gate * (0.35 + 0.65 * anchor_uncertainty) * (1.0 - 0.6 * guidance_lock)
                    if str(args.shock_gate_mode) == "legacy_company_scale":
                        shock_weight_base_raw *= _company_abstain_scale(ticker)
                    else:
                        shock_weight_base_raw *= float(args.shock_global_scale)
                    shock_weight_base = _clip(shock_weight_base_raw, 0.0, 1.0)
                    shock_weight_base = float(0.0 if not math.isfinite(shock_weight_base) else shock_weight_base)
                    if str(args.shock_gate_mode) == "context_ridge":
                        gate_feature_map = _build_context_gate_feature_map(
                            shock_raw=shock_raw,
                            shock_weight_base=shock_weight_base,
                            guidance_lock=guidance_lock,
                            guidance_dict=guidance_dict,
                            internal_state=internal_state,
                            factor_map=factor_map,
                            regime=regime,
                            anchor_diag=anchor_diag_state,
                            anchor_error_state=anchor_error_state,
                        )
                        gate_scope = str(args.shock_gate_training_scope)
                        local_gate_rows = list(quarterly_rows)
                        shared_gate_rows = [
                            item
                            for item in shock_gate_global_history
                            if _quarter_key(str(item.get("quarter") or "")) < _quarter_key(str(row.get("quarter") or ""))
                        ]
                        local_result = {"multiplier": 1.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": []}
                        shared_result = {"multiplier": 1.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": []}
                        if gate_scope == "company_local":
                            local_result = _predict_context_gate(local_gate_rows, gate_feature_map, float(args.shock_gate_alpha), float(args.shock_gate_shrink_k))
                            shock_gate_state["train_count"] = int(local_result["train_count"])
                            shock_gate_state["local_train_count"] = int(local_result["train_count"])
                            if int(local_result["train_count"]) >= int(args.shock_gate_min_train):
                                shock_gate_state.update(
                                    {
                                        "multiplier": float(local_result["multiplier"]),
                                        "used": True,
                                        "scope": "company_local",
                                        "pred_raw": float(local_result["pred_raw"]),
                                        "pred_local": float(local_result["pred_raw"]),
                                        "support": float(local_result["support"]),
                                        "top_contribs": list(local_result["top_contribs"]),
                                    }
                                )
                        elif gate_scope == "shared_pooled":
                            shared_result = _predict_context_gate(shared_gate_rows, gate_feature_map, float(args.shock_gate_alpha), float(args.shock_gate_shrink_k))
                            shock_gate_state["train_count"] = int(shared_result["train_count"])
                            shock_gate_state["shared_train_count"] = int(shared_result["train_count"])
                            if int(shared_result["train_count"]) >= int(args.shock_gate_min_train):
                                shock_gate_state.update(
                                    {
                                        "multiplier": float(shared_result["multiplier"]),
                                        "used": True,
                                        "scope": "shared_pooled",
                                        "pred_raw": float(shared_result["pred_raw"]),
                                        "pred_shared": float(shared_result["pred_raw"]),
                                        "support": float(shared_result["support"]),
                                        "top_contribs": list(shared_result["top_contribs"]),
                                    }
                                )
                        else:
                            shared_result = _predict_context_gate(shared_gate_rows, gate_feature_map, float(args.shock_gate_alpha), float(args.shock_gate_shrink_k))
                            local_result = _predict_context_gate(local_gate_rows, gate_feature_map, float(args.shock_gate_alpha), float(args.shock_gate_shrink_k))
                            shared_ok = int(shared_result["train_count"]) >= int(args.shock_gate_min_train)
                            local_ok = int(local_result["train_count"]) >= int(args.shock_gate_min_train)
                            shock_gate_state["shared_train_count"] = int(shared_result["train_count"])
                            shock_gate_state["local_train_count"] = int(local_result["train_count"])
                            shock_gate_state["train_count"] = max(int(shared_result["train_count"]), int(local_result["train_count"]))
                            if shared_ok and local_ok:
                                local_weight = float(local_result["train_count"]) / max(float(local_result["train_count"]) + float(args.shock_gate_local_prior_k), EPS)
                                use_local = local_weight >= 0.5
                                shock_gate_state.update(
                                    {
                                        "multiplier": float((1.0 - local_weight) * float(shared_result["multiplier"]) + local_weight * float(local_result["multiplier"])),
                                        "used": True,
                                        "scope": "shared_blend",
                                        "local_weight": float(local_weight),
                                        "pred_local": float(local_result["pred_raw"]),
                                        "pred_shared": float(shared_result["pred_raw"]),
                                        "pred_raw": float((1.0 - local_weight) * float(shared_result["pred_raw"]) + local_weight * float(local_result["pred_raw"])),
                                        "support": float((1.0 - local_weight) * float(shared_result["support"]) + local_weight * float(local_result["support"])),
                                        "top_contribs": list(local_result["top_contribs"] if use_local else shared_result["top_contribs"]),
                                    }
                                )
                            elif local_ok:
                                shock_gate_state.update(
                                    {
                                        "multiplier": float(local_result["multiplier"]),
                                        "used": True,
                                        "scope": "company_local_only",
                                        "pred_raw": float(local_result["pred_raw"]),
                                        "pred_local": float(local_result["pred_raw"]),
                                        "support": float(local_result["support"]),
                                        "top_contribs": list(local_result["top_contribs"]),
                                    }
                                )
                            elif shared_ok:
                                shock_gate_state.update(
                                    {
                                        "multiplier": float(shared_result["multiplier"]),
                                        "used": True,
                                        "scope": "shared_pooled_only",
                                        "pred_raw": float(shared_result["pred_raw"]),
                                        "pred_shared": float(shared_result["pred_raw"]),
                                        "support": float(shared_result["support"]),
                                        "top_contribs": list(shared_result["top_contribs"]),
                                    }
                                )
                    shock_weight_pre_tkg = float(_clip(shock_weight_base * float(shock_gate_state.get("multiplier", 1.0)), 0.0, 1.0))
                    temporal_kg_gate_state = _temporal_kg_gate_multiplier(temporal_kg_diag, shock_raw, args)
                    temporal_kg_feature_map = _build_temporal_kg_learned_feature_map(
                        kg_diag=temporal_kg_diag,
                        shock_raw=shock_raw,
                        shock_weight_base=shock_weight_pre_tkg,
                        guidance_dict=guidance_dict,
                        internal_state=internal_state,
                        anchor_diag=anchor_diag_state,
                    )
                    if str(args.temporal_kg_learned_scope) == "shared" and temporal_kg_global_history:
                        gate_history = pd.DataFrame(temporal_kg_global_history)
                        if "quarter" in gate_history.columns:
                            gate_history = gate_history[
                                gate_history["quarter"].astype(str).map(_quarter_key) < _quarter_key(str(row.get("quarter") or ""))
                            ].copy()
                    else:
                        gate_history = hist_processed.copy()
                    temporal_kg_gate_state = _temporal_kg_learned_gate_state(
                        gate_history=gate_history,
                        kg_diag=temporal_kg_diag,
                        heuristic_state=temporal_kg_gate_state,
                        shock_raw=shock_raw,
                        shock_weight_base=shock_weight_pre_tkg,
                        guidance_dict=guidance_dict,
                        internal_state=internal_state,
                        anchor_diag=anchor_diag_state,
                        args=args,
                    )
                    shock_weight = float(_clip(shock_weight_pre_tkg * float(temporal_kg_gate_state.get("multiplier", 1.0)), 0.0, 1.0))
                    top_contribs = _top_contributions(shock_model, shock_signal, shock_feature_names, top_k=3)
                else:
                    top_contribs = []
            else:
                top_contribs = []
            shock_final = shock_raw * shock_weight
            final_pred = anchor_pred * math.exp(shock_final) if math.isfinite(anchor_pred) and anchor_pred > 0.0 else float(anchor_pred)

            out_row = dict(row)
            out_row.update(regime)
            out_row.update(internal_state)
            out_row.update(anchor_error_state)
            out_row.update(
                {
                    "ticker": ticker,
                    "best_stat_model": best_stat_model,
                    "best_stat_mae_company": float(best_stat_mae),
                    "prehistory_best_model": prehist_best_model,
                    "prehistory_best_mae": float(prehist_best_mae) if math.isfinite(prehist_best_mae) else float(best_stat_mae),
                    "anchor_best_model": best_model,
                    "anchor_best_model_pred": float(best_model_pred),
                    "anchor_raw_log": float(anchor_raw_log),
                    "anchor_blend_weight": float(anchor_blend_weight),
                    "anchor_uncertainty": float(anchor_uncertainty),
                    "anchor_top1_gap_ratio": float(anchor_top1_gap_ratio),
                    "anchor_uncertainty_mode": str(args.anchor_uncertainty_mode),
                    "pred_csais_anchor": float(anchor_pred),
                    "guidance_lock": float(guidance_lock),
                    "shock_raw_log": float(shock_raw),
                    "shock_weight_base": float(shock_weight_base),
                    "shock_weight_pre_tkg": float(shock_weight_pre_tkg),
                    "shock_weight": float(shock_weight),
                    "shock_final_log": float(shock_final),
                    "shock_gate_mode": str(args.shock_gate_mode),
                    "shock_gate_training_scope": str(args.shock_gate_training_scope),
                    "shock_gate_multiplier": float(_safe_float(shock_gate_state.get("multiplier"), 1.0)),
                    "shock_gate_used": int(bool(shock_gate_state.get("used", False))),
                    "shock_gate_scope_applied": str(shock_gate_state.get("scope") or "off"),
                    "shock_gate_train_n": int(_safe_float(shock_gate_state.get("train_count"), 0.0)),
                    "shock_gate_local_train_n": int(_safe_float(shock_gate_state.get("local_train_count"), 0.0)),
                    "shock_gate_shared_train_n": int(_safe_float(shock_gate_state.get("shared_train_count"), 0.0)),
                    "shock_gate_local_weight": float(_safe_float(shock_gate_state.get("local_weight"), 0.0)),
                    "shock_gate_pred_raw": float(_safe_float(shock_gate_state.get("pred_raw"), float("nan"))),
                    "shock_gate_pred_local": float(_safe_float(shock_gate_state.get("pred_local"), float("nan"))),
                    "shock_gate_pred_shared": float(_safe_float(shock_gate_state.get("pred_shared"), float("nan"))),
                    "shock_gate_support": float(_safe_float(shock_gate_state.get("support"), 0.0)),
                    "shock_gate_top_contribs": json.dumps(shock_gate_state.get("top_contribs", []), ensure_ascii=False),
                    "pred_csais_v1": float(final_pred),
                    "temporal_kg_available": int(bool(temporal_kg_diag.get("available", False))),
                    "temporal_kg_reason": str(temporal_kg_diag.get("reason") or ""),
                    "temporal_kg_predicted_correction_log": float(_safe_float(temporal_kg_diag.get("predicted_correction_log"), 0.0)),
                    "temporal_kg_predicted_var": float(_safe_float(temporal_kg_diag.get("predicted_var"), float("nan"))),
                    "temporal_kg_effective_memory_count": float(_safe_float(temporal_kg_diag.get("effective_memory_count"), 0.0)),
                    "temporal_kg_attention_focus": float(_safe_float(temporal_kg_diag.get("attention_focus"), 0.0)),
                    "temporal_kg_segment_relation_overlap": float(_safe_float(temporal_kg_diag.get("segment_relation_overlap"), 0.0)),
                    "temporal_kg_directional_consistency": float(_safe_float(temporal_kg_diag.get("directional_consistency"), 0.0)),
                    "temporal_kg_sign_agreement": float(_safe_float(temporal_kg_diag.get("sign_agreement"), 0.0)),
                    "temporal_kg_support": float(_safe_float(temporal_kg_diag.get("support"), 0.0)),
                    "temporal_kg_multiplier": float(_safe_float(temporal_kg_gate_state.get("multiplier"), 1.0)),
                    "temporal_kg_relation": str(temporal_kg_gate_state.get("relation") or "off"),
                    "temporal_kg_gate_mode": str(temporal_kg_gate_state.get("mode") or "heuristic"),
                    "temporal_kg_gate_train_n": int(_safe_float(temporal_kg_gate_state.get("learned_train_n"), 0.0)),
                    "temporal_kg_gate_pred_raw": float(_safe_float(temporal_kg_gate_state.get("learned_pred_raw"), float("nan"))),
                    "temporal_kg_gate_blend_weight": float(_safe_float(temporal_kg_gate_state.get("learned_blend_weight"), 0.0)),
                    "temporal_kg_gate_top_contribs": str(temporal_kg_gate_state.get("learned_top_contribs") or ""),
                    "temporal_kg_retrieved_quarters": str(temporal_kg_diag.get("retrieved_quarters") or ""),
                    "temporal_kg_top_matches": str(temporal_kg_diag.get("top_matches_json") or ""),
                    "shock_feature_mode": str(args.shock_feature_mode),
                    "top_internal_factor_contribs": json.dumps(top_contribs, ensure_ascii=False),
                    "shock_target_log": float("nan"),
                    "shock_gate_target": float("nan"),
                    "temporal_kg_gate_target": float("nan"),
                }
            )
            for name, value in factor_map.items():
                out_row[name] = float(value)
            for name, value in temporal_kg_feature_map.items():
                out_row[name] = float(value)
            actual_log = _safe_log(_safe_float(row.get("actual")))
            anchor_log_for_target = _safe_log(anchor_pred)
            if math.isfinite(actual_log) and math.isfinite(anchor_log_for_target):
                out_row["shock_target_log"] = float(actual_log - anchor_log_for_target)
                anchor_error_history.append(
                    {
                        "quarter": str(row.get("quarter") or ""),
                        "guidance_availability": current_guidance,
                        "anchor_abs_log_error": float(abs(actual_log - anchor_log_for_target)),
                    }
                )
            out_row["shock_gate_target"] = _optimal_gate_multiplier_target(
                float(shock_raw),
                float(shock_weight_base),
                float(_safe_float(out_row.get("shock_target_log"), float("nan"))),
            )
            out_row["temporal_kg_gate_target"] = _optimal_temporal_kg_gate_target(
                float(shock_raw),
                float(shock_weight_pre_tkg),
                float(_safe_float(out_row.get("shock_target_log"), float("nan"))),
            )
            quarterly_rows.append(out_row)
            shock_gate_global_history.append(dict(out_row))
            temporal_kg_global_history.append(dict(out_row))
            temporal_kg_history.append(
                {
                    "quarter": str(row.get("quarter") or ""),
                    "shock_target_log": float(out_row.get("shock_target_log", float("nan"))),
                    "temporal_kg_stub": temporal_kg_stub,
                }
            )
            actual_hist.append(_safe_float(row.get("actual")))
            quarter_hist.append(str(row.get("quarter") or ""))

        quarterly_df = pd.DataFrame(quarterly_rows)
        if quarterly_df.empty:
            continue
        company_metrics = {
            "baseline": _metrics(quarterly_df["actual"], quarterly_df["baseline_pred"]),
            "best_stat": _metrics(quarterly_df["actual"], quarterly_df[f"pred__{best_stat_model}"]),
            "csais_anchor": _metrics(quarterly_df["actual"], quarterly_df["pred_csais_anchor"]),
            "csais_v1": _metrics(quarterly_df["actual"], quarterly_df["pred_csais_v1"]),
        }
        per_company_summary.append(
            {
                "ticker": ticker,
                "panel_role": str(company.get("panel_role", "")),
                "panel_type": str(company.get("panel_type", "")),
                "archetype": str(company.get("archetype", "")),
                "n": int(len(quarterly_df)),
                "best_stat_model": best_stat_model,
                "baseline_mae": float(company_metrics["baseline"]["mae"]),
                "best_stat_mae": float(company_metrics["best_stat"]["mae"]),
                "csais_anchor_mae": float(company_metrics["csais_anchor"]["mae"]),
                "csais_v1_mae": float(company_metrics["csais_v1"]["mae"]),
            }
        )
        all_quarterly.append(quarterly_df)

    quarterly = pd.concat(all_quarterly, axis=0).sort_values(["ticker", "quarter"], key=lambda s: s.map(_quarter_key) if s.name == "quarter" else s).reset_index(drop=True)
    company_summary_df = pd.DataFrame(per_company_summary)
    pooled_metrics = {
        "baseline": _metrics(quarterly["actual"], quarterly["baseline_pred"]),
        "csais_anchor": _metrics(quarterly["actual"], quarterly["pred_csais_anchor"]),
        "csais_v1": _metrics(quarterly["actual"], quarterly["pred_csais_v1"]),
    }
    macro = {
        "baseline": float(company_summary_df["baseline_mae"].mean()),
        "best_stat": float(company_summary_df["best_stat_mae"].mean()),
        "csais_anchor": float(company_summary_df["csais_anchor_mae"].mean()),
        "csais_v1": float(company_summary_df["csais_v1_mae"].mean()),
    }
    company_summary_df.to_csv(out_dir / "csais_v1_company_summary.csv", index=False)
    quarterly.to_csv(out_dir / "csais_v1_quarterly.csv", index=False)
    summary = {
        "inputs": {
            "experiment_config": str(_resolve(args.experiment_config, project_root)),
            "tickers": sorted(company_summary_df["ticker"].tolist()),
            "top_model_pool": int(args.top_model_pool),
            "anchor_mode": str(args.anchor_mode),
            "anchor_uncertainty_mode": str(args.anchor_uncertainty_mode),
            "anchor_uncertainty_recent_window": int(args.anchor_uncertainty_recent_window),
            "anchor_uncertainty_tau": float(args.anchor_uncertainty_tau),
            "shock_feature_mode": str(args.shock_feature_mode),
            "shock_gate_mode": str(args.shock_gate_mode),
            "shock_gate_training_scope": str(args.shock_gate_training_scope),
            "shock_gate_alpha": float(args.shock_gate_alpha),
            "shock_gate_min_train": int(args.shock_gate_min_train),
            "shock_gate_shrink_k": float(args.shock_gate_shrink_k),
            "shock_global_scale": float(args.shock_global_scale),
            "anchor_alpha": float(args.anchor_alpha),
            "shock_alpha": float(args.shock_alpha),
            "anchor_shrink_k": float(args.anchor_shrink_k),
            "shock_shrink_k": float(args.shock_shrink_k),
            "temporal_kg_gate": bool(args.temporal_kg_gate),
            "temporal_kg_gate_mode": str(args.temporal_kg_gate_mode),
            "temporal_kg_learned_scope": str(args.temporal_kg_learned_scope),
            "temporal_kg_top_k": int(args.temporal_kg_top_k),
            "temporal_kg_temperature": float(args.temporal_kg_temperature),
            "temporal_kg_learned_min_train": int(args.temporal_kg_learned_min_train),
            "temporal_kg_learned_alpha": float(args.temporal_kg_learned_alpha),
        },
        "metrics": {
            "pooled": pooled_metrics,
            "macro_mae": macro,
        },
        "wins": {
            "csais_v1_beats_baseline_companies": int((company_summary_df["csais_v1_mae"] < company_summary_df["baseline_mae"]).sum()),
            "csais_v1_beats_best_stat_companies": int((company_summary_df["csais_v1_mae"] < company_summary_df["best_stat_mae"]).sum()),
            "csais_anchor_beats_best_stat_companies": int((company_summary_df["csais_anchor_mae"] < company_summary_df["best_stat_mae"]).sum()),
        },
        "outputs": {
            "quarterly_csv": str(out_dir / "csais_v1_quarterly.csv"),
            "company_summary_csv": str(out_dir / "csais_v1_company_summary.csv"),
        },
    }
    (out_dir / "csais_v1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit("Implementation dependency only; run scripts/run_reference_replay.sh.")
