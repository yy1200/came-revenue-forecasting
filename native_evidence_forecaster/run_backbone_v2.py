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

from native_evidence_forecaster.common import EPS, ensure_dir, parse_mapping, quarter_key, resolve_repo_path, safe_float, write_json


DEFAULT_EXPERIMENT_CONFIG = "replay_inputs/retained_336/experiment.json"


def _safe_log(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(math.log(max(value, EPS)))


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _quarter_number(value: str) -> int:
    key = quarter_key(value)
    return int(key[1]) if key != (0, 0) else 0


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(list(values), dtype=float)
    x = np.arange(len(arr), dtype=float)
    x = x - float(x.mean())
    denom = float((x ** 2).sum())
    if denom <= EPS:
        return 0.0
    return float((x @ (arr - float(arr.mean()))) / denom)


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


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, Any]:
    mask = np.isfinite(y)
    if x.size:
        mask = mask & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        width = x.shape[1] if x.ndim == 2 else 0
        return {"means": np.zeros(width), "stds": np.ones(width), "intercept": 0.0, "coefs": np.zeros(width)}
    means = x.mean(axis=0)
    stds = np.where(x.std(axis=0) > EPS, x.std(axis=0), 1.0)
    xz = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    intercept = float(y.mean())
    centered = y - intercept
    reg = float(alpha) * np.eye(x.shape[1], dtype=float)
    try:
        coefs = np.linalg.pinv(xz.T @ xz + reg) @ (xz.T @ centered)
    except np.linalg.LinAlgError:
        coefs = np.zeros(x.shape[1], dtype=float)
    return {"means": means, "stds": stds, "intercept": intercept, "coefs": coefs}


def _predict_ridge(model: Mapping[str, Any], x: np.ndarray) -> float:
    means = np.asarray(model["means"], dtype=float)
    stds = np.asarray(model["stds"], dtype=float)
    coefs = np.asarray(model["coefs"], dtype=float)
    intercept = float(model["intercept"])
    xz = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    return float(intercept + xz @ coefs)


def _top_feature_contribs(model: Mapping[str, Any], x: np.ndarray, feature_names: Sequence[str], top_k: int = 5) -> List[Tuple[str, float]]:
    means = np.asarray(model["means"], dtype=float)
    stds = np.asarray(model["stds"], dtype=float)
    coefs = np.asarray(model["coefs"], dtype=float)
    xz = np.nan_to_num((x - means) / stds, nan=0.0, posinf=0.0, neginf=0.0)
    contribs = [(name, float(val)) for name, val in zip(feature_names, xz * coefs)]
    contribs.sort(key=lambda item: abs(item[1]), reverse=True)
    return contribs[:top_k]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_best_stat(path: Path) -> Tuple[str, float]:
    frame = pd.read_csv(path)
    frame = frame[~frame["model"].isin(["guid_uniform", "guid_triangular"])].copy()
    best = frame.sort_values("MAE").iloc[0]
    return str(best["model"]), float(best["MAE"])


def _load_forecast(path: Path) -> pd.DataFrame:
    cols = [
        "ticker",
        "observed_fiscal_quarter",
        "target_fiscal_quarter",
        "observed_revenue",
        "observed_filing_date",
        "target_revenue",
        "target_filing_date",
        "guid_low",
        "guid_high",
        "guid_mid",
        "guidance_score",
        "baseline_eligible",
        "guidance_availability",
        "guidance_source",
    ]
    try:
        frame = pd.read_csv(path, usecols=lambda c: c in cols).copy()
    except Exception:
        frame = pd.read_csv(path).copy()
        keep = [col for col in cols if col in frame.columns]
        frame = frame[keep].copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = np.nan
    for col in ["observed_revenue", "target_revenue", "guid_low", "guid_high", "guid_mid", "guidance_score", "baseline_eligible"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ["observed_filing_date", "target_filing_date"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    frame["observed_fiscal_quarter"] = frame["observed_fiscal_quarter"].astype(str)
    frame["target_fiscal_quarter"] = frame["target_fiscal_quarter"].astype(str)
    frame["guidance_availability"] = frame["guidance_availability"].fillna("none").astype(str)
    frame["guidance_source"] = frame["guidance_source"].fillna("").astype(str)
    return frame


def _load_actuals(path: Path) -> pd.DataFrame:
    cols = ["fiscal_quarter", "filing_date", "revenue", "actual_source"]
    try:
        frame = pd.read_csv(path, usecols=lambda c: c in set(cols)).copy()
    except Exception:
        frame = pd.read_csv(path).copy()
        keep = [col for col in cols if col in frame.columns]
        frame = frame[keep].copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["fiscal_quarter"] = frame["fiscal_quarter"].astype(str)
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    return frame.sort_values("fiscal_quarter", key=lambda s: s.map(quarter_key)).reset_index(drop=True)


def _load_segments(path: Path) -> pd.DataFrame:
    cols = {
        "fiscal_quarter",
        "segment_revenue_m",
        "major_segment_revenue_m",
        "segment_count",
        "segment_sum_m",
        "major_segment_count",
        "major_segment_sum_m",
        "sanity_flag",
        "extraction_source",
    }
    try:
        frame = pd.read_csv(path, usecols=lambda c: c in cols).copy()
    except Exception:
        frame = pd.read_csv(path).copy()
        keep = [col for col in cols if col in frame.columns]
        frame = frame[keep].copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["fiscal_quarter"] = frame["fiscal_quarter"].astype(str)
    return frame.sort_values("fiscal_quarter", key=lambda s: s.map(quarter_key)).reset_index(drop=True)


def _segment_map(row: Mapping[str, Any]) -> Dict[str, float]:
    direct = parse_mapping(row.get("segment_revenue_m"))
    if direct:
        return direct
    return parse_mapping(row.get("major_segment_revenue_m"))


def _share_map(seg_map: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(max(v, 0.0) for v in seg_map.values()))
    if total <= EPS:
        return {}
    return {str(k): float(max(v, 0.0) / total) for k, v in seg_map.items()}


def _segment_state(current_map: Mapping[str, float], prev_map: Mapping[str, float], yoy_map: Mapping[str, float]) -> Dict[str, float]:
    current_shares = _share_map(current_map)
    prev_shares = _share_map(prev_map)
    yoy_shares = _share_map(yoy_map)
    ordered = sorted(current_shares.items(), key=lambda item: item[1], reverse=True)
    top1 = float(ordered[0][1]) if ordered else 0.0
    top2 = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    gap12 = top1 - top2
    hhi = float(sum(v ** 2 for v in current_shares.values())) if current_shares else 0.0
    entropy = float(-sum(v * math.log(max(v, EPS)) for v in current_shares.values())) if current_shares else 0.0
    union_prev = set(current_map) | set(prev_map)
    union_yoy = set(current_map) | set(yoy_map)

    def _weighted_growth(base: Mapping[str, float], other: Mapping[str, float]) -> float:
        if not base:
            return 0.0
        shares = _share_map(base)
        values: List[float] = []
        weights: List[float] = []
        for seg, share in shares.items():
            cur = max(float(base.get(seg, 0.0)), 0.0)
            prev = max(float(other.get(seg, 0.0)), 0.0)
            values.append(_safe_log(cur + 1.0) - _safe_log(prev + 1.0))
            weights.append(share)
        if not weights or sum(weights) <= EPS:
            return 0.0
        return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))

    share_turnover_qoq = 0.5 * sum(abs(current_shares.get(seg, 0.0) - prev_shares.get(seg, 0.0)) for seg in union_prev)
    share_turnover_yoy = 0.5 * sum(abs(current_shares.get(seg, 0.0) - yoy_shares.get(seg, 0.0)) for seg in union_yoy)
    top_seg = ordered[0][0] if ordered else ""
    top1_qoq = _safe_log(max(float(current_map.get(top_seg, 0.0)), 0.0) + 1.0) - _safe_log(max(float(prev_map.get(top_seg, 0.0)), 0.0) + 1.0) if top_seg else 0.0
    top1_yoy = _safe_log(max(float(current_map.get(top_seg, 0.0)), 0.0) + 1.0) - _safe_log(max(float(yoy_map.get(top_seg, 0.0)), 0.0) + 1.0) if top_seg else 0.0
    return {
        "segment_available": 1.0 if current_map else 0.0,
        "segment_count": float(len(current_map)),
        "segment_share_top1": top1,
        "segment_share_top2": top2,
        "segment_share_gap12": gap12,
        "segment_hhi": hhi,
        "segment_entropy": entropy,
        "segment_weighted_qoq_log": _weighted_growth(current_map, prev_map),
        "segment_weighted_yoy_log": _weighted_growth(current_map, yoy_map),
        "segment_share_turnover_qoq": float(share_turnover_qoq),
        "segment_share_turnover_yoy": float(share_turnover_yoy),
        "segment_top1_qoq_log": float(top1_qoq),
        "segment_top1_yoy_log": float(top1_yoy),
        "segment_prev_available": 1.0 if prev_map else 0.0,
        "segment_yoy_available": 1.0 if yoy_map else 0.0,
    }


def _component_logs(history_actuals: pd.DataFrame, target_quarter: str, guidance_mid: float, guidance_numeric_available: float) -> Dict[str, float]:
    revenues = [safe_float(v) for v in history_actuals.get("revenue", pd.Series(index=history_actuals.index, dtype=float)).tolist()]
    revenues = [v for v in revenues if math.isfinite(v) and v > 0.0]
    logs = [_safe_log(v) for v in revenues]
    out: Dict[str, float] = {
        "comp_level_log": float(np.mean(logs[-4:])) if logs else float("nan"),
        "comp_trend_log": float(logs[-1] + _slope(logs[-4:])) if logs else float("nan"),
        "comp_seasonal_log": float("nan"),
        "comp_guidance_log": float("nan"),
    }
    target_q_num = _quarter_number(target_quarter)
    quarter_series = history_actuals.get("fiscal_quarter", pd.Series(index=history_actuals.index, dtype=str)).astype(str)
    revenue_series = history_actuals.get("revenue", pd.Series(index=history_actuals.index, dtype=float))
    same_q_vals = [
        safe_float(revenue_series.iloc[i])
        for i, q in enumerate(quarter_series.tolist())
        if _quarter_number(str(q)) == target_q_num and math.isfinite(safe_float(revenue_series.iloc[i])) and safe_float(revenue_series.iloc[i]) > 0.0
    ]
    if same_q_vals:
        out["comp_seasonal_log"] = float(np.mean([_safe_log(v) for v in same_q_vals[-3:]]))
    if guidance_numeric_available == 1.0 and math.isfinite(guidance_mid) and guidance_mid > 0.0:
        out["comp_guidance_log"] = _safe_log(guidance_mid)
    return out


def _fallback_backbone_log(component_logs: Mapping[str, float]) -> float:
    vals = [float(component_logs[name]) for name in ["comp_level_log", "comp_trend_log", "comp_seasonal_log", "comp_guidance_log"] if math.isfinite(float(component_logs[name]))]
    return float(np.mean(vals)) if vals else float("nan")


def _build_feature_row(
    company_row: Mapping[str, Any],
    actuals_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    archetype_names: Sequence[str],
) -> Dict[str, Any]:
    observed_quarter = str(company_row["observed_fiscal_quarter"])
    target_quarter = str(company_row["target_fiscal_quarter"])
    observed_date = pd.to_datetime(company_row.get("observed_filing_date"), errors="coerce")
    target_date = pd.to_datetime(company_row.get("target_filing_date"), errors="coerce")
    fq_series = actuals_df.get("fiscal_quarter", pd.Series(index=actuals_df.index, dtype=str)).astype(str)
    rev_series = actuals_df.get("revenue", pd.Series(index=actuals_df.index, dtype=float))
    history_actuals = actuals_df[fq_series.map(lambda q: quarter_key(q) <= quarter_key(observed_quarter))].copy()
    history_actuals = history_actuals[rev_series.loc[history_actuals.index].map(lambda v: math.isfinite(safe_float(v)) and safe_float(v) > 0.0)].copy()
    history_actuals = history_actuals.assign(__quarter_sort=history_actuals.get("fiscal_quarter", pd.Series(index=history_actuals.index, dtype=str)).astype(str).map(quarter_key))
    history_actuals = history_actuals.sort_values("__quarter_sort").drop(columns=["__quarter_sort"], errors="ignore").reset_index(drop=True)
    logs = [_safe_log(v) for v in history_actuals.get("revenue", pd.Series(index=history_actuals.index, dtype=float)).tolist()]
    qoq = [logs[i] - logs[i - 1] for i in range(1, len(logs))]
    yoy = [logs[i] - logs[i - 4] for i in range(4, len(logs))]

    guidance_numeric_available = 1.0 if str(company_row.get("guidance_availability") or "none") == "explicit_numeric" and int(safe_float(company_row.get("baseline_eligible"), 0.0) > 0.0) == 1 else 0.0
    guidance_mid = safe_float(company_row.get("guid_mid"))
    guidance_low = safe_float(company_row.get("guid_low"))
    guidance_high = safe_float(company_row.get("guid_high"))
    guid_band_ratio = 0.0
    if math.isfinite(guidance_mid) and guidance_mid > 0.0 and math.isfinite(guidance_low) and math.isfinite(guidance_high):
        guid_band_ratio = abs(guidance_high - guidance_low) / max(abs(guidance_mid), EPS)
    component_logs = _component_logs(history_actuals, target_quarter, guidance_mid, guidance_numeric_available)
    fallback_log = _fallback_backbone_log(component_logs)
    level_log = component_logs["comp_level_log"] if math.isfinite(component_logs["comp_level_log"]) else (logs[-1] if logs else float("nan"))
    trend_log = component_logs["comp_trend_log"] if math.isfinite(component_logs["comp_trend_log"]) else level_log
    seasonal_log = component_logs["comp_seasonal_log"] if math.isfinite(component_logs["comp_seasonal_log"]) else level_log
    guidance_log = component_logs["comp_guidance_log"] if math.isfinite(component_logs["comp_guidance_log"]) else level_log
    fallback_log = fallback_log if math.isfinite(fallback_log) else level_log

    seg_fq_series = segments_df.get("fiscal_quarter", pd.Series(index=segments_df.index, dtype=str)).astype(str)
    segment_history = segments_df[seg_fq_series.map(lambda q: quarter_key(q) <= quarter_key(observed_quarter))].copy()
    segment_history = segment_history.assign(__quarter_sort=segment_history.get("fiscal_quarter", pd.Series(index=segment_history.index, dtype=str)).astype(str).map(quarter_key))
    segment_history = segment_history.sort_values("__quarter_sort").drop(columns=["__quarter_sort"], errors="ignore").reset_index(drop=True)
    current_seg_map: Dict[str, float] = {}
    prev_seg_map: Dict[str, float] = {}
    yoy_seg_map: Dict[str, float] = {}
    if not segment_history.empty:
        seg_hist_quarters = segment_history.get("fiscal_quarter", pd.Series(index=segment_history.index, dtype=str)).astype(str)
        current_candidates = segment_history[seg_hist_quarters == observed_quarter]
        if not current_candidates.empty:
            current_seg_map = _segment_map(current_candidates.iloc[-1].to_dict())
        prev_candidates = segment_history[seg_hist_quarters.map(lambda q: quarter_key(q) < quarter_key(observed_quarter))]
        if not prev_candidates.empty:
            prev_seg_map = _segment_map(prev_candidates.iloc[-1].to_dict())
        observed_q_num = _quarter_number(observed_quarter)
        yoy_quarters = prev_candidates.get("fiscal_quarter", pd.Series(index=prev_candidates.index, dtype=str)).astype(str)
        yoy_candidates = prev_candidates[yoy_quarters.map(lambda q: _quarter_number(q) == observed_q_num)] if not prev_candidates.empty else pd.DataFrame()
        if not yoy_candidates.empty:
            yoy_seg_map = _segment_map(yoy_candidates.iloc[-1].to_dict())
    segment_state = _segment_state(current_seg_map, prev_seg_map, yoy_seg_map)

    target_q_num = _quarter_number(target_quarter)
    same_q_support = 0.0
    if not history_actuals.empty:
        same_q_support = min(1.0, sum(1 for q in history_actuals.get("fiscal_quarter", pd.Series(index=history_actuals.index, dtype=str)).astype(str).tolist() if _quarter_number(str(q)) == target_q_num) / 3.0)
    feature_map: Dict[str, float] = {
        "recent_level_log": float(np.mean(logs[-4:])) if logs else 0.0,
        "recent_last_log": float(logs[-1]) if logs else 0.0,
        "recent_qoq_log": float(qoq[-1]) if qoq else 0.0,
        "recent_yoy_log": float(yoy[-1]) if yoy else 0.0,
        "trend_slope4": float(_slope(logs[-4:])) if logs else 0.0,
        "vol_qoq4": float(np.std(qoq[-4:], ddof=0)) if len(qoq) >= 2 else 0.0,
        "vol_yoy4": float(np.std(yoy[-4:], ddof=0)) if len(yoy) >= 2 else 0.0,
        "same_quarter_support": float(_clip(same_q_support, 0.0, 1.0)),
        "history_len_log": float(math.log1p(len(history_actuals))),
        "guidance_numeric_available": float(guidance_numeric_available),
        "guidance_score_norm": float(_clip(safe_float(company_row.get("guidance_score"), 0.0) / 20.0, 0.0, 1.0)),
        "guid_band_ratio": float(_clip(guid_band_ratio, 0.0, 2.0)),
        "comp_level_log": float(level_log),
        "comp_trend_log": float(trend_log),
        "comp_seasonal_log": float(seasonal_log),
        "comp_guidance_log": float(guidance_log),
        "comp_guidance_available": float(1.0 if math.isfinite(component_logs["comp_guidance_log"]) else 0.0),
        "comp_seasonal_available": float(1.0 if math.isfinite(component_logs["comp_seasonal_log"]) else 0.0),
        "gap_trend_minus_level": float(trend_log - level_log),
        "gap_seasonal_minus_level": float(seasonal_log - level_log),
        "gap_guidance_minus_level": float(guidance_log - level_log),
        "gap_guidance_minus_seasonal": float(guidance_log - seasonal_log),
        "fallback_backbone_log": float(fallback_log),
        "target_fiscal_q": float(target_q_num),
        **segment_state,
    }
    for name in archetype_names:
        feature_map[f"arch__{name}"] = 1.0 if str(company_row.get("archetype") or "") == name else 0.0
    record = {str(key): company_row[key] for key in company_row.keys()}
    record.update(feature_map)
    record["fallback_backbone_pred"] = float(math.exp(fallback_log)) if math.isfinite(fallback_log) else float("nan")
    record["target_log"] = _safe_log(safe_float(company_row.get("target_revenue")))
    record["target_delta_log"] = float(record["target_log"] - fallback_log) if math.isfinite(record["target_log"]) and math.isfinite(fallback_log) else float("nan")
    record["is_eval_row"] = bool(company_row.get("is_eval_row"))
    return record


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Backbone V2 for the native evidence forecaster branch.")
    ap.add_argument("--experiment_config", default=DEFAULT_EXPERIMENT_CONFIG)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--min_train", type=int, default=24)
    ap.add_argument("--training_scope", choices=["company_local", "shared_pooled"], default="company_local")
    ap.add_argument("--delta_shrink_k", type=float, default=16.0)
    ap.add_argument("--delta_cap_quantile", type=float, default=0.9)
    ap.add_argument("--output_dir", default="output/backbone_v2_all12")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = os.path.abspath(args.project_root)
    experiment_path = resolve_repo_path(args.experiment_config, project_root)
    out_dir = ensure_dir(resolve_repo_path(args.output_dir, project_root))
    exp = _load_json(experiment_path)
    requested = {ticker.upper() for ticker in args.tickers}

    companies = [company for company in exp.get("companies", []) if not requested or str(company.get("ticker", "")).upper() in requested]
    archetype_names = sorted({str(company.get("archetype") or "") for company in companies if str(company.get("archetype") or "")})

    all_rows: List[Dict[str, Any]] = []
    best_stat_by_company: Dict[str, Tuple[str, float]] = {}

    for company in companies:
        ticker = str(company.get("ticker") or "").upper()
        company_cfg = _load_json(resolve_repo_path(company.get("company_config"), project_root))
        data_paths = company_cfg.get("data_paths", {}) if isinstance(company_cfg, dict) else {}
        forecast_df = _load_forecast(resolve_repo_path(data_paths.get("forecast_dataset_csv"), project_root))
        actuals_df = _load_actuals(resolve_repo_path(data_paths.get("quarter_actuals_csv"), project_root))
        segments_df = _load_segments(resolve_repo_path(data_paths.get("segment_actuals_csv"), project_root))
        best_stat_by_company[ticker] = _load_best_stat(resolve_repo_path(company.get("stat_baseline_metrics_csv"), project_root))

        start_q = str(company.get("evaluation_start_fq") or "")
        end_q = str(company.get("evaluation_end_fq") or exp.get("default_end_fq") or "")

        forecast_df = forecast_df[forecast_df["target_revenue"].map(lambda v: math.isfinite(safe_float(v)) and safe_float(v) > 0.0)].copy()
        forecast_df["ticker"] = ticker
        forecast_df["archetype"] = str(company.get("archetype") or "")
        forecast_df["is_eval_row"] = forecast_df["target_fiscal_quarter"].map(lambda q: quarter_key(start_q) <= quarter_key(str(q)) <= quarter_key(end_q))
        for _, row in forecast_df.iterrows():
            record = _build_feature_row(row.to_dict(), actuals_df, segments_df, archetype_names)
            all_rows.append(record)

    state_df = pd.DataFrame(all_rows)
    state_df = state_df.sort_values(["observed_filing_date", "target_fiscal_quarter", "ticker"], key=lambda s: pd.to_datetime(s, errors="coerce") if s.name == "observed_filing_date" else (s.map(quarter_key) if s.name == "target_fiscal_quarter" else s)).reset_index(drop=True)

    feature_names = [
        "recent_level_log",
        "recent_last_log",
        "recent_qoq_log",
        "recent_yoy_log",
        "trend_slope4",
        "vol_qoq4",
        "vol_yoy4",
        "same_quarter_support",
        "history_len_log",
        "guidance_numeric_available",
        "guidance_score_norm",
        "guid_band_ratio",
        "comp_level_log",
        "comp_trend_log",
        "comp_seasonal_log",
        "comp_guidance_log",
        "comp_guidance_available",
        "comp_seasonal_available",
        "gap_trend_minus_level",
        "gap_seasonal_minus_level",
        "gap_guidance_minus_level",
        "gap_guidance_minus_seasonal",
        "target_fiscal_q",
        "segment_available",
        "segment_count",
        "segment_share_top1",
        "segment_share_top2",
        "segment_share_gap12",
        "segment_hhi",
        "segment_entropy",
        "segment_weighted_qoq_log",
        "segment_weighted_yoy_log",
        "segment_share_turnover_qoq",
        "segment_share_turnover_yoy",
        "segment_top1_qoq_log",
        "segment_top1_yoy_log",
        "segment_prev_available",
        "segment_yoy_available",
    ] + [f"arch__{name}" for name in archetype_names]

    pred_rows: List[Dict[str, Any]] = []
    eval_df = state_df[state_df["is_eval_row"] == True].copy()
    for _, row in eval_df.iterrows():
        observed_date = pd.to_datetime(row.get("observed_filing_date"), errors="coerce")
        train_mask = pd.to_datetime(state_df["target_filing_date"], errors="coerce") < observed_date
        if str(args.training_scope) == "company_local":
            train_mask = train_mask & (state_df["ticker"].astype(str) == str(row.get("ticker") or ""))
        train_df = state_df[train_mask].copy()
        x = np.asarray([float(safe_float(row.get(name), 0.0)) for name in feature_names], dtype=float)
        fallback_log = float(safe_float(row.get("fallback_backbone_log"), float("nan")))
        pred_delta = 0.0
        train_count = int(len(train_df))
        top_contribs: List[Tuple[str, float]] = []
        mode = "fallback"
        if train_count >= int(args.min_train):
            x_train = np.asarray([[float(safe_float(r.get(name), 0.0)) for name in feature_names] for _, r in train_df.iterrows()], dtype=float)
            y_train = pd.to_numeric(train_df["target_delta_log"], errors="coerce").to_numpy(dtype=float)
            model = _fit_ridge(x_train, y_train, float(args.alpha))
            pred_delta = _predict_ridge(model, x)
            finite_abs = np.abs(y_train[np.isfinite(y_train)])
            if finite_abs.size:
                delta_cap = float(np.quantile(finite_abs, float(args.delta_cap_quantile)))
                if math.isfinite(delta_cap) and delta_cap > 0.0:
                    pred_delta = _clip(pred_delta, -delta_cap, delta_cap)
            support = train_count / max(train_count + float(args.delta_shrink_k), EPS)
            pred_delta = float(pred_delta * support)
            top_contribs = _top_feature_contribs(model, x, feature_names, top_k=5)
            mode = "ridge_delta"
        pred_log = fallback_log + pred_delta if math.isfinite(fallback_log) else float("nan")
        pred = float(math.exp(pred_log)) if math.isfinite(pred_log) else float("nan")
        out_row = {str(key): row[key] for key in eval_df.columns}
        out_row["backbone_mode"] = mode
        out_row["train_row_count"] = train_count
        out_row["pred_backbone_v2"] = pred
        out_row["pred_backbone_v2_log"] = pred_log
        out_row["pred_backbone_v2_fallback"] = float(math.exp(fallback_log)) if math.isfinite(fallback_log) else float("nan")
        out_row["pred_backbone_v2_delta_log"] = float(pred_delta)
        out_row["top_state_contribs_json"] = json.dumps(top_contribs, ensure_ascii=False)
        pred_rows.append(out_row)

    pred_df = pd.DataFrame(pred_rows)
    quarterly_csv = out_dir / "backbone_v2_quarterly.csv"
    state_csv = out_dir / "native_state_panel.csv"
    company_csv = out_dir / "backbone_v2_company_summary.csv"
    summary_json = out_dir / "backbone_v2_summary.json"
    state_df.to_csv(state_csv, index=False)
    pred_df.to_csv(quarterly_csv, index=False)

    company_rows: List[Dict[str, Any]] = []
    for ticker, company_group in pred_df.groupby("ticker"):
        metrics = _metrics(company_group["target_revenue"], company_group["pred_backbone_v2"])
        fallback_metrics = _metrics(company_group["target_revenue"], company_group["pred_backbone_v2_fallback"])
        best_model, best_mae = best_stat_by_company[str(ticker)]
        company_rows.append(
            {
                "ticker": str(ticker),
                "n": int(metrics["n"]),
                "backbone_v2_mae": float(metrics["mae"]),
                "backbone_v2_rmse": float(metrics["rmse"]),
                "backbone_v2_mape": float(metrics["mape"]),
                "backbone_v2_smape": float(metrics["smape"]),
                "fallback_mae": float(fallback_metrics["mae"]),
                "best_stat_model": best_model,
                "best_stat_mae": float(best_mae),
                "beats_best_stat": bool(metrics["mae"] < best_mae),
            }
        )
    company_df = pd.DataFrame(company_rows).sort_values("ticker")
    company_df.to_csv(company_csv, index=False)

    pooled = _metrics(pred_df["target_revenue"], pred_df["pred_backbone_v2"])
    fallback_pooled = _metrics(pred_df["target_revenue"], pred_df["pred_backbone_v2_fallback"])
    summary = {
        "experiment_config": str(experiment_path),
        "output_dir": str(out_dir),
        "company_count": int(len(company_df)),
        "eval_row_count": int(len(pred_df)),
        "backbone_v2_pooled": pooled,
        "fallback_pooled": fallback_pooled,
        "macro_mae": float(company_df["backbone_v2_mae"].mean()) if not company_df.empty else float("nan"),
        "fallback_macro_mae": float(company_df["fallback_mae"].mean()) if not company_df.empty else float("nan"),
        "beats_best_stat_companies": int(company_df["beats_best_stat"].sum()) if not company_df.empty else 0,
        "feature_names": feature_names,
        "alpha": float(args.alpha),
        "min_train": int(args.min_train),
        "training_scope": str(args.training_scope),
        "delta_shrink_k": float(args.delta_shrink_k),
        "delta_cap_quantile": float(args.delta_cap_quantile),
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit("Implementation dependency only; run scripts/run_reference_replay.sh.")
