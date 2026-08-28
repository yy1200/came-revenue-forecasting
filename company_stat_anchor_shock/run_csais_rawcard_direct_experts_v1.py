#!/usr/bin/env python3

"""Frozen CAME implementation dependency.

Only ``scripts/run_reference_replay.sh`` defines the supported public execution
contract. This module retains machine identifiers required by the frozen replay.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from company_stat_anchor_shock.run_csais_v1 import (
    DEFAULT_EXPERIMENT_CONFIG,
    FACTORIZED_SHOCK_FEATURES,
    RAW_SHOCK_FEATURES,
    STAT_EXCLUDE,
    _anchor_error_state,
    _build_shock_features,
    _clip,
    _factorized_internal_features,
    _load_retrieve_payload,
    _fit_ridge,
    _guidance_features,
    _guidance_lock,
    _history_regime,
    _internal_features,
    _load_stat_predictions,
    _metrics,
    _prepare_company_panel,
    _predict_ridge,
    _quarter_key,
    _quarter_number,
    _safe_float,
    _safe_log,
    _top_contributions,
)
from company_stat_anchor_shock.run_csais_candidate_bridge_v1 import (
    _build_native_surface,
    _card_fusion_key,
    _card_additive_influence,
    _card_segment_name,
    _load_backbone_lookup,
    _load_card_groups,
    _load_frozen_best_stat_map,
    _native_card_admissibility_weight,
    _native_rows_to_attention_record,
    _predict_compressed_candidate,
    _predict_native_card_additive_candidate,
    _sign_num,
    _split_native_support_conflict_cards,
    _weighted_native_card_mass,
    _weighted_average,
)
from evidence_memory_residual.common import EPS, softmax_weights
from native_evidence_forecaster.common import resolve_repo_path, sanitize_for_json, write_json
from native_evidence_forecaster.run_native_cards_v1 import CANONICAL_FACTORS, _apply_safety_guard, _fit_zero_intercept_ridge, _predict_zero_intercept_ridge
from native_evidence_forecaster.run_native_csais_v1 import DEFAULT_BACKBONE_CSV, DEFAULT_CARD_TABLE_JSONL, RAW_CARD_FEATURE_NAMES, _memory_diag, _predict_zero_intercept_history
from temporal_kg_memory_attention.pair_features import cross_card_attention


DIRECT_EXPERT_GATE_FEATURES = [
    "gate_base_delta",
    "gate_intrinsic_delta",
    "gate_intrinsic_support",
    "gate_temporal_delta",
    "gate_temporal_support",
    "gate_gap_base_intrinsic",
    "gate_gap_base_temporal",
    "gate_intrinsic_temporal_sign_match",
    "gate_guidance_lock",
    "gate_anchor_uncertainty",
    "gate_internal_strength",
    "gate_fwd_conflict_ratio",
    "gate_memory_support",
    "gate_memory_consistency",
]

TEMPORAL_ACTION_GATE_FEATURES = [
    "temporal_action_guidance_explicit",
    "temporal_action_guidance_non_explicit",
    "temporal_action_guidance_none",
    "temporal_action_delta",
    "temporal_action_abs_delta",
    "temporal_action_support",
    "temporal_action_effective_memory_count",
    "temporal_action_directional_consistency",
    "temporal_action_attention_focus",
    "temporal_action_mean_direction_alignment",
    "temporal_action_top_direction_alignment",
    "temporal_action_context_attention_score",
    "temporal_action_context_attention_focus",
    "temporal_action_context_quality_scale",
    "temporal_action_context_quality_weak_score",
    "temporal_action_intrinsic_delta",
    "temporal_action_intrinsic_abs_delta",
    "temporal_action_intrinsic_support",
    "temporal_action_base_delta",
    "temporal_action_base_abs_delta",
    "temporal_action_base_support",
    "temporal_action_intrinsic_temporal_sign_match",
    "temporal_action_intrinsic_temporal_duplicate_ratio",
    "temporal_action_intrinsic_conflict",
    "temporal_action_base_conflict",
    "temporal_action_guidance_expert_active",
    "temporal_action_guidance_lock",
    "temporal_action_anchor_uncertainty",
    "temporal_action_internal_strength",
    "temporal_action_forward_conflict_ratio",
    "temporal_action_memory_support",
    "temporal_action_memory_consistency",
    "temporal_action_reliability_scale",
    "temporal_action_reliability_trust_score",
]

INTRINSIC_ACTION_GATE_FEATURES = [
    "intrinsic_action_guidance_explicit",
    "intrinsic_action_guidance_non_explicit",
    "intrinsic_action_guidance_none",
    "intrinsic_action_delta",
    "intrinsic_action_abs_delta",
    "intrinsic_action_support",
    "intrinsic_action_residual_candidate_delta",
    "intrinsic_action_abs_residual_candidate_delta",
    "intrinsic_action_train_count",
    "intrinsic_action_coverage",
    "intrinsic_action_temporal_delta",
    "intrinsic_action_temporal_abs_delta",
    "intrinsic_action_temporal_support",
    "intrinsic_action_base_delta",
    "intrinsic_action_base_abs_delta",
    "intrinsic_action_base_support",
    "intrinsic_action_temporal_sign_match",
    "intrinsic_action_temporal_duplicate_ratio",
    "intrinsic_action_guidance_expert_active",
    "intrinsic_action_guidance_lock",
    "intrinsic_action_anchor_uncertainty",
    "intrinsic_action_internal_strength",
    "intrinsic_action_forward_conflict_ratio",
    "intrinsic_action_memory_support",
    "intrinsic_action_memory_consistency",
    "intrinsic_action_reliability_scale",
    "intrinsic_action_reliability_trust_score",
    "intrinsic_action_explicit_guidance_scale",
    "intrinsic_action_dedup_scale",
    "intrinsic_action_dedup_duplicate_ratio",
]

STATE_ANALOG_TEMPORAL_FEATURES = [
    "state_reg_recent_qoq",
    "state_reg_last_yoy",
    "state_reg_trend_slope4",
    "state_reg_vol_qoq4",
    "state_reg_vol_yoy4",
    "state_reg_same_quarter_support",
    "state_reg_recent_level_log",
    "state_guidance_explicit",
    "state_guidance_non_explicit",
    "state_guidance_none",
    "state_anchor_uncertainty",
    "state_internal_strength",
    "state_fq1",
    "state_fq2",
    "state_fq3",
    "state_fq4",
]

DIRECT_EXPERT_CONTRACTS = {
    "full": ("compressed_base", "intrinsic_direct", "temporal_direct"),
    "anchor_only": (),
    "base_only": ("compressed_base",),
    "intrinsic_only": ("intrinsic_direct",),
    "temporal_only": ("temporal_direct",),
    "base_plus_intrinsic": ("compressed_base", "intrinsic_direct"),
    "base_plus_temporal": ("compressed_base", "temporal_direct"),
}

INTEGRATED_EXPERT_ARBITRATION_MODE = "integrated_expert_arbitration_v0"
SHARED_RESIDUAL_BACKBONE_MODE = "shared_residual_backbone_v0"
EVIDENCE_ORTHOGONAL_ARBITRATION_MODE = "evidence_orthogonal_arbitration_v0"
CURRENT_EVIDENCE_ORTHOGONAL_ARBITRATION_MODE = "current_evidence_orthogonal_arbitration_v0"
INTEGRATED_ARBITRATION_DEV_TICKERS = ("AAPL", "NVDA", "AVGO")
INTEGRATED_ARBITRATION_DEV_END_QUARTER = "FY2023_Q4"
INTEGRATED_ARBITRATION_FLOOR_RATIO_FALLBACK = 0.9010075079067871
INTEGRATED_ARBITRATION_INTERNAL_POLICY = {
    "scope": "no_guid_mid_numeric",
    "source": "max_abs_same_sign",
    "min_total_support": 0.5,
    "min_component_support": 0.0,
    "min_agree": 2,
    "max_oppose": 1,
    "min_target_gap": 0.0,
    "max_abs_delta": 0.15,
    "min_anchor_recent_abs_log": 0.05,
    "min_internal_strength": 0.0,
}
INTEGRATED_NEGATIVE_SUPPLY_TERMS = {
    "delay",
    "delays",
    "delayed",
    "capacity expansion",
    "capacity",
    "supply",
    "constraint",
    "constraints",
    "shortage",
    "shortages",
    "limited supply",
}
INTEGRATED_FUTURE_EASING_PATTERNS = (
    r"\bshould ease\b",
    r"\bwill ease\b",
    r"\bease in\b",
    r"\beasing\b",
    r"\bunblock\b",
    r"\brecover\b",
    r"\brecovery\b",
    r"\bimprove\b",
    r"\bimprovement\b",
)


def _quarter_label(key: Tuple[int, int]) -> str:
    return f"FY{int(key[0]):04d}_Q{int(key[1])}"


def _quarter_labels_between(start_key: Tuple[int, int], end_key: Tuple[int, int]) -> List[str]:
    if start_key > end_key:
        return []
    year, quarter = int(start_key[0]), int(start_key[1])
    out: List[str] = []
    while (year, quarter) <= end_key:
        out.append(_quarter_label((year, quarter)))
        quarter += 1
        if quarter > 4:
            year += 1
            quarter = 1
    return out


def _quarter_ordinal(quarter: str) -> int:
    year, fiscal_q = _quarter_key(str(quarter or ""))
    return int(year) * 4 + int(fiscal_q)

GUIDANCE_QUALITY_GUARDRAIL_ALPHA = {
    "derived_weak_numeric": 0.0,
    "no_total_revenue_guidance_but_forward_commentary": 0.5,
    "qualitative_only": 0.5,
}

ANCHOR_FALLBACK_MODELS = ("guid_mid", "robust_momentum", "seasonal_naive_q4", "naive", "ma", "mean")
DIRECT_GUIDANCE_ANCHOR_COLS = {"pred__guid_mid", "pred__guid_affine", "pred__guid_blend"}
ROBUST_MOMENTUM_ANCHOR_COL = "pred__robust_momentum"
GUIDANCE_DEPENDENT_ANCHOR_COLS = {
    "pred__linear_lag_guid",
    "pred__ridge_lag_guid",
    "pred__elasticnet_lag_guid",
    "pred__rf_lag_guid",
    "pred__xgb_lag_guid",
    "pred__lgbm_lag_guid",
    "pred__sarimax_guid",
}
REGIME_AWARE_BASE_ANCHOR_DEFAULT_PRED_COL = "pred_regime_aware_base_anchor_proposal"
REGIME_AWARE_BASE_ANCHOR_NATIVE_DELTA_MODE = "signed_strength_delta_v0"
REGIME_AWARE_BASE_ANCHOR_SCALED_DELTA_MODE = "signed_strength_scaled_delta_v0"


def _load_anchor_override_map(path_value: str, project_root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {}
    path = Path(resolve_repo_path(path_text, str(project_root)))
    df = pd.read_csv(path)
    model_col = "best_stat_model" if "best_stat_model" in df.columns else "anchor_best_model"
    required = {"ticker", "quarter", model_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"anchor override CSV is missing required columns: {missing}")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pred_col = next(
        (col for col in ["pred_selected_anchor", "pred_online_stat_portfolio_anchor", "pred_anchor_override"] if col in df.columns),
        "",
    )
    for _, row in df.iterrows():
        ticker = str(row.get("ticker") or "").upper().strip()
        quarter = str(row.get("quarter") or "").strip()
        model = str(row.get(model_col) or "").strip()
        if not ticker or not quarter or not model:
            continue
        out[(ticker, quarter)] = {
            "best_stat_model": model,
            "anchor_override_pred": _safe_float(row.get(pred_col), float("nan")) if pred_col else float("nan"),
            "selector_score": _safe_float(row.get("selector_score"), float("nan")),
            "selector_history_n": int(_safe_float(row.get("selector_history_n"), 0.0)),
            "selector_reason": str(row.get("selector_reason") or "row_override_csv"),
        }
    return out


def _load_regime_aware_base_anchor_proposals(
    path_value: str,
    project_root: Path,
    pred_col: str,
    filter_col: str = "",
    filter_value: str = "",
) -> Dict[Tuple[str, str], float]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {}
    path = Path(resolve_repo_path(path_text, str(project_root)))
    df = pd.read_csv(path)
    pred_col = str(pred_col or "").strip() or REGIME_AWARE_BASE_ANCHOR_DEFAULT_PRED_COL
    if filter_col:
        if filter_col not in df.columns:
            raise ValueError(f"regime-aware proposal CSV is missing filter column: {filter_col}")
        df = df[df[filter_col].astype(str).eq(str(filter_value))].copy()
    required = {"ticker", "quarter", pred_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"regime-aware proposal CSV is missing required columns: {missing}")
    if df.duplicated(["ticker", "quarter"]).any():
        dupes = df.loc[df.duplicated(["ticker", "quarter"], keep=False), ["ticker", "quarter"]].head(8).to_dict("records")
        raise ValueError(f"regime-aware proposal CSV has duplicate ticker-quarter rows after filtering: {dupes}")
    out: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker") or "").upper().strip()
        quarter = str(row.get("quarter") or "").strip()
        if not ticker or not quarter:
            continue
        out[(ticker, quarter)] = _safe_float(row.get(pred_col), float("nan"))
    return out


def _row_has_numeric_guidance(row: Mapping[str, Any]) -> bool:
    if "guid_mid" in row:
        guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
        return bool(np.isfinite(guid_mid) and guid_mid > 0.0)
    for key in ["stat_guid_available", "guid_available"]:
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "1.0", "true", "yes"}
        available = _safe_float(value, float("nan"))
        if np.isfinite(available):
            return bool(available > 0.0)
    return False


def _row_guidance_bucket(row: Mapping[str, Any]) -> str:
    label = str(row.get("guidance_availability") or "none")
    if label in {"explicit_numeric", "derived_weak_numeric", "no_total_revenue_guidance_but_forward_commentary", "qualitative_only"}:
        return label
    if _row_has_numeric_guidance(row):
        return "explicit_numeric"
    return "none"


def _regime_aware_base_anchor_threshold_suffix(threshold: float) -> str:
    suffix = str(float(threshold)).replace(".", "p")
    if suffix.endswith("0") and "p" in suffix:
        suffix = suffix.rstrip("0").rstrip("p")
    if "p" not in suffix:
        suffix = f"{suffix}p0"
    return suffix


def _regime_aware_base_anchor_pred_col(threshold: float) -> str:
    suffix = _regime_aware_base_anchor_threshold_suffix(float(threshold))
    return f"pred_came_ram_v1_signed_strength_guard_t{suffix}"


def _regime_aware_base_anchor_native_delta_pred_col(threshold: float) -> str:
    suffix = _regime_aware_base_anchor_threshold_suffix(float(threshold))
    return f"pred_came_ram_native_delta_v0_signed_strength_guard_t{suffix}"


def _regime_aware_base_anchor_scaled_delta_pred_col(threshold: float) -> str:
    suffix = _regime_aware_base_anchor_threshold_suffix(float(threshold))
    return f"pred_came_ram_scaled_delta_v0_signed_strength_guard_t{suffix}"


def _regime_aware_base_anchor_scaled_delta_confidence(
    row: Mapping[str, Any],
    *,
    raw_memory_delta: float,
    signed_error: float,
    signed_strength_threshold: float,
) -> Dict[str, Any]:
    history_strength = float(_clip((float(signed_error) - float(signed_strength_threshold)) / 0.12, 0.0, 1.0)) if np.isfinite(signed_error) else 0.0
    proposal_gap_strength = float(_clip(float(raw_memory_delta) / 0.12, 0.0, 1.0)) if np.isfinite(raw_memory_delta) else 0.0
    agree_support = 0.0
    oppose_support = 0.0
    for delta_col, support_col in [
        ("csais_compressed_base_candidate_delta_log", "csais_compressed_base_candidate_support"),
        ("csais_intrinsic_direct_candidate_delta_log", "csais_intrinsic_direct_candidate_support"),
        ("csais_temporal_direct_candidate_delta_log", "csais_temporal_direct_candidate_support"),
    ]:
        delta = _safe_float(row.get(delta_col), 0.0)
        support = float(_clip(_safe_float(row.get(support_col), 0.0), 0.0, 1.0))
        if not np.isfinite(delta) or support <= 0.0:
            continue
        if delta > 0.0:
            agree_support += float(support)
        elif delta < 0.0:
            oppose_support += float(support)
    evidence_total = float(agree_support + oppose_support)
    evidence_agreement = float(agree_support / evidence_total) if evidence_total > EPS else 0.65
    scale = 1.0
    reason = "full_strength_history_gap_or_evidence_support"
    if proposal_gap_strength < 0.25 and history_strength < 0.70:
        scale = 0.55
        reason = "small_gap_and_moderate_history_shrink"
    elif oppose_support > 0.25 and evidence_agreement < 0.55:
        scale = 0.65
        reason = "current_temporal_evidence_conflict_shrink"
    return {
        "scale": float(scale),
        "reason": reason,
        "history_strength": float(history_strength),
        "proposal_gap_strength": float(proposal_gap_strength),
        "evidence_agreement": float(evidence_agreement),
        "evidence_agree_support": float(agree_support),
        "evidence_oppose_support": float(oppose_support),
    }


def _apply_regime_aware_base_anchor_panel(
    frame: pd.DataFrame,
    *,
    proposal_map: Mapping[Tuple[str, str], float],
    mode: str,
    min_history: int,
    signed_strength_threshold: float,
    upward_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mode = str(mode or "off")
    if mode == "off":
        return frame, {"mode": "off", "active_rows": 0}
    if mode not in {"signed_strength_v1", REGIME_AWARE_BASE_ANCHOR_NATIVE_DELTA_MODE, REGIME_AWARE_BASE_ANCHOR_SCALED_DELTA_MODE}:
        raise ValueError(f"Unsupported regime-aware base-anchor mode: {mode}")
    native_delta_mode = bool(mode in {REGIME_AWARE_BASE_ANCHOR_NATIVE_DELTA_MODE, REGIME_AWARE_BASE_ANCHOR_SCALED_DELTA_MODE})
    scaled_delta_mode = bool(mode == REGIME_AWARE_BASE_ANCHOR_SCALED_DELTA_MODE)

    out = frame.copy()
    pre_col = "pred_csais_rawcard_direct_experts_v1_pre_regime_aware_base_anchor"
    proposal_col = "pred_regime_aware_base_anchor_proposal"
    pred_col = (
        _regime_aware_base_anchor_scaled_delta_pred_col(float(signed_strength_threshold))
        if scaled_delta_mode
        else _regime_aware_base_anchor_native_delta_pred_col(float(signed_strength_threshold))
        if mode == REGIME_AWARE_BASE_ANCHOR_NATIVE_DELTA_MODE
        else _regime_aware_base_anchor_pred_col(float(signed_strength_threshold))
    )
    active_col = "regime_aware_base_anchor_active"
    reason_col = "regime_aware_base_anchor_reason"
    history_n_col = "regime_aware_base_anchor_history_n"
    signed_error_col = "regime_aware_base_anchor_prior_signed_log_error_actual_over_current"
    raw_native_delta_col = "regime_aware_base_anchor_raw_native_delta_log"
    native_delta_col = "regime_aware_base_anchor_native_delta_log"
    native_pre_delta_col = "regime_aware_base_anchor_pre_memory_delta_log"
    native_post_delta_col = "regime_aware_base_anchor_post_memory_delta_log"
    delta_scale_col = "regime_aware_base_anchor_delta_scale"
    delta_scale_reason_col = "regime_aware_base_anchor_delta_scale_reason"
    history_strength_col = "regime_aware_base_anchor_history_strength"
    proposal_gap_strength_col = "regime_aware_base_anchor_proposal_gap_strength"
    evidence_agreement_col = "regime_aware_base_anchor_evidence_agreement"
    evidence_oppose_support_col = "regime_aware_base_anchor_evidence_oppose_support"

    out[pre_col] = pd.to_numeric(out["pred_csais_rawcard_direct_experts_v1"], errors="coerce")
    out[proposal_col] = [
        _safe_float(proposal_map.get((str(row.get("ticker") or "").upper(), str(row.get("quarter") or ""))), float("nan"))
        for _, row in out.iterrows()
    ]
    out[pred_col] = out[pre_col]
    out[active_col] = 0
    out[reason_col] = "not_evaluated"
    out[history_n_col] = 0
    out[signed_error_col] = np.nan
    out[raw_native_delta_col] = 0.0
    out[native_delta_col] = 0.0
    out[native_pre_delta_col] = pd.to_numeric(out.get("csais_final_candidate_delta_log", np.nan), errors="coerce")
    out[native_post_delta_col] = pd.to_numeric(out.get("csais_final_candidate_delta_log", np.nan), errors="coerce")
    out[delta_scale_col] = 1.0
    out[delta_scale_reason_col] = "not_scaled"
    out[history_strength_col] = 0.0
    out[proposal_gap_strength_col] = 0.0
    out[evidence_agreement_col] = np.nan
    out[evidence_oppose_support_col] = 0.0

    history: List[Dict[str, float]] = []
    same_or_future_history_rows = 0
    same_or_future_seen_buffer_rows = 0
    max_history_delta = -1
    threshold = float(signed_strength_threshold)
    min_history = int(min_history)
    upward_ratio = float(upward_ratio)

    qnums = out["quarter"].astype(str).map(lambda value: int(_quarter_key(value)[0] * 4 + _quarter_key(value)[1]))
    sorted_index = out.assign(__ram_qnum=qnums).sort_values(["__ram_qnum", "ticker"]).index.tolist()
    for idx in sorted_index:
        row = out.loc[idx]
        qnum = int(qnums.loc[idx])
        actual = _safe_float(row.get("actual", row.get("actual_stat", row.get("y_true"))), float("nan"))
        current = _safe_float(row.get(pre_col), float("nan"))
        proposal = _safe_float(row.get(proposal_col), float("nan"))
        anchor_pred = _safe_float(row.get("pred_csais_anchor"), float("nan"))
        no_guidance = _row_guidance_bucket(row) == "none"
        chosen = current
        active = 0
        reason = "not_no_guidance_current_kept"
        history_n = 0
        signed_error = float("nan")
        raw_memory_delta = 0.0
        memory_delta = 0.0
        memory_delta_scale = 1.0
        memory_delta_scale_reason = "not_scaled"
        memory_delta_confidence: Dict[str, Any] = {
            "history_strength": 0.0,
            "proposal_gap_strength": 0.0,
            "evidence_agreement": float("nan"),
            "evidence_oppose_support": 0.0,
        }
        pre_memory_delta = _safe_float(row.get("csais_final_candidate_delta_log"), float("nan"))
        post_memory_delta = pre_memory_delta

        if no_guidance:
            reason = "invalid_current_or_proposal_current_kept"
            if np.isfinite(current) and current > 0.0 and np.isfinite(proposal) and proposal > 0.0:
                prior = [item for item in history if int(item["qnum"]) < qnum]
                same_or_future_seen_buffer_rows += sum(1 for item in history if int(item["qnum"]) >= qnum)
                same_or_future_history_rows += sum(1 for item in prior if int(item["qnum"]) >= qnum)
                if prior:
                    max_history_delta = max(max_history_delta, max(int(item["qnum"]) for item in prior) - qnum)
                history_n = int(len(prior))
                if history_n < min_history:
                    reason = "insufficient_prior_no_guidance_history_current_kept"
                else:
                    signed_error = float(np.mean([np.log(float(item["actual"]) / float(item["current_pred"])) for item in prior]))
                    if signed_error > threshold and proposal >= upward_ratio * current:
                        raw_memory_delta = float(np.log(float(proposal) / float(current)))
                        memory_delta = float(raw_memory_delta)
                        if scaled_delta_mode:
                            memory_delta_confidence = _regime_aware_base_anchor_scaled_delta_confidence(
                                row,
                                raw_memory_delta=float(raw_memory_delta),
                                signed_error=float(signed_error),
                                signed_strength_threshold=float(threshold),
                            )
                            memory_delta_scale = float(_safe_float(memory_delta_confidence.get("scale"), 1.0))
                            memory_delta_scale_reason = str(memory_delta_confidence.get("reason") or "scaled_delta")
                            memory_delta = float(raw_memory_delta * memory_delta_scale)
                            chosen = float(current * np.exp(memory_delta))
                        else:
                            chosen = proposal
                            memory_delta_scale_reason = "exact_native_delta" if native_delta_mode else "post_panel_replacement"
                        active = int(abs(chosen - current) > 1e-6)
                        reason = "prior_signed_error_strength_active"
                        if native_delta_mode and active:
                            if np.isfinite(anchor_pred) and anchor_pred > 0.0:
                                pre_memory_delta = float(np.log(float(current) / float(anchor_pred)))
                                post_memory_delta = float(pre_memory_delta + memory_delta)
                    elif signed_error <= threshold:
                        reason = "prior_signed_error_strength_below_threshold_current_kept"
                    else:
                        reason = "proposal_not_upward_current_kept"

        out.at[idx, pred_col] = chosen
        out.at[idx, "pred_csais_rawcard_direct_experts_v1"] = chosen
        if native_delta_mode:
            out.at[idx, raw_native_delta_col] = raw_memory_delta
            out.at[idx, native_delta_col] = memory_delta
            out.at[idx, native_pre_delta_col] = pre_memory_delta
            out.at[idx, native_post_delta_col] = post_memory_delta
            out.at[idx, delta_scale_col] = memory_delta_scale
            out.at[idx, delta_scale_reason_col] = memory_delta_scale_reason
            out.at[idx, history_strength_col] = float(_safe_float(memory_delta_confidence.get("history_strength"), 0.0))
            out.at[idx, proposal_gap_strength_col] = float(_safe_float(memory_delta_confidence.get("proposal_gap_strength"), 0.0))
            out.at[idx, evidence_agreement_col] = float(_safe_float(memory_delta_confidence.get("evidence_agreement"), float("nan")))
            out.at[idx, evidence_oppose_support_col] = float(_safe_float(memory_delta_confidence.get("evidence_oppose_support"), 0.0))
            if active:
                out.at[idx, "pred_csais_rawcard_direct_experts_v1_pre_guidance_guardrail"] = chosen
                out.at[idx, "csais_pre_guidance_guardrail_candidate_delta_log"] = post_memory_delta
                out.at[idx, "csais_final_candidate_delta_log"] = post_memory_delta
        out.at[idx, active_col] = active
        out.at[idx, reason_col] = reason
        out.at[idx, history_n_col] = history_n
        out.at[idx, signed_error_col] = signed_error

        if no_guidance and np.isfinite(actual) and actual > 0.0 and np.isfinite(current) and current > 0.0 and np.isfinite(proposal) and proposal > 0.0:
            history.append({"qnum": float(qnum), "actual": float(actual), "current_pred": float(current), "proposal": float(proposal)})

    active_series = out[active_col].astype(int).eq(1)
    active_scales = pd.to_numeric(out.loc[active_series, delta_scale_col], errors="coerce") if native_delta_mode else pd.Series(dtype=float)
    validation = {
        "mode": mode,
        "pred_col": pred_col,
        "pre_pred_col": pre_col,
        "proposal_col": proposal_col,
        "native_delta_mode": bool(native_delta_mode),
        "scaled_delta_mode": bool(scaled_delta_mode),
        "raw_native_delta_col": raw_native_delta_col,
        "native_delta_col": native_delta_col,
        "delta_scale_col": delta_scale_col,
        "proposal_rows": int(sum(1 for value in proposal_map.values() if np.isfinite(_safe_float(value, float("nan"))))),
        "min_history": int(min_history),
        "signed_strength_threshold": float(threshold),
        "upward_ratio": float(upward_ratio),
        "same_or_future_history_rows": int(same_or_future_history_rows),
        "same_or_future_seen_buffer_rows": int(same_or_future_seen_buffer_rows),
        "max_history_qnum_minus_target_qnum": int(max_history_delta),
        "active_rows": int(active_series.sum()),
        "guidance_bearing_active_rows": int((active_series & out.apply(lambda cur: _row_guidance_bucket(cur) != "none", axis=1)).sum()),
        "scaled_active_rows": int((active_series & (pd.to_numeric(out[delta_scale_col], errors="coerce") < 0.999999)).sum()) if native_delta_mode else 0,
        "active_delta_scale_mean": float(active_scales.mean()) if native_delta_mode and not active_scales.empty else float("nan"),
        "active_delta_scale_min": float(active_scales.min()) if native_delta_mode and not active_scales.empty else float("nan"),
    }
    return out, validation


def _shared_residual_backbone_regime(row: Mapping[str, Any]) -> str:
    bucket = _row_guidance_bucket(row)
    if bucket == "explicit_numeric":
        return "explicit"
    if bucket in {"derived_weak_numeric", "no_total_revenue_guidance_but_forward_commentary", "qualitative_only"}:
        return "non_explicit"
    return "no_guidance"


def _shared_residual_backbone_caps(regime: str) -> Dict[str, float]:
    if regime == "explicit":
        return {
            "gamma": 0.85,
            "current_card": 0.60,
            "temporal_memory": 0.65,
            "auxiliary": 0.85,
            "score_threshold": 0.035,
        }
    if regime == "non_explicit":
        return {
            "gamma": 0.60,
            "current_card": 0.65,
            "temporal_memory": 0.65,
            "auxiliary": 0.25,
            "score_threshold": 0.035,
        }
    return {
        "gamma": 0.70,
        "current_card": 0.65,
        "temporal_memory": 0.65,
        "auxiliary": 0.70,
        "score_threshold": 0.035,
    }


def _shared_residual_backbone_reference_delta(candidates: Sequence[Mapping[str, Any]]) -> float:
    usable = []
    for candidate in candidates:
        delta = _safe_float(candidate.get("delta"), float("nan"))
        support = _safe_float(candidate.get("support"), 0.0)
        if np.isfinite(delta) and support > 0.0:
            usable.append((float(delta), float(support)))
    if not usable:
        return 0.0
    return float(_weighted_average(usable))


def _shared_residual_backbone_mix(
    *,
    regime: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    caps = _shared_residual_backbone_caps(regime)
    finite = []
    for candidate in candidates:
        delta = _safe_float(candidate.get("delta"), float("nan"))
        support = float(_clip(_safe_float(candidate.get("support"), 0.0), 0.0, 1.0))
        if np.isfinite(delta) and support > 0.0:
            finite.append({**dict(candidate), "delta": float(delta), "support": float(support)})

    consensus_numer = float(sum(float(item["delta"]) * float(item["support"]) for item in finite))
    consensus_sign = _sign_num(consensus_numer)
    score_threshold = float(caps["score_threshold"])
    support_sum = 0.0
    weighted_delta = 0.0
    scored: Dict[str, Dict[str, Any]] = {}
    for item in finite:
        name = str(item.get("name") or "")
        delta = float(item["delta"])
        support = float(item["support"])
        candidate_sign = _sign_num(delta)
        if consensus_sign == 0.0 or candidate_sign == 0.0:
            agreement_scale = 0.75
        elif candidate_sign == consensus_sign:
            agreement_scale = 1.0
        else:
            agreement_scale = 0.25
        cap = float(caps.get(name, caps["auxiliary"]))
        conflict_penalty = float(_clip(_safe_float(item.get("conflict_penalty"), 1.0), 0.0, 1.0))
        prior_reliability = float(_clip(_safe_float(item.get("prior_reliability"), 1.0), 0.0, 1.0))
        raw_score = float(support * cap * agreement_scale * conflict_penalty * prior_reliability)
        score = float(raw_score if raw_score >= score_threshold else 0.0)
        scored[name] = {
            **item,
            "score": float(score),
            "raw_score": float(raw_score),
            "agreement_scale": float(agreement_scale),
            "cap": float(cap),
            "conflict_penalty": float(conflict_penalty),
            "prior_reliability": float(prior_reliability),
            "active": int(score > 0.0),
        }
        if score > 0.0:
            support_sum += float(score)
            weighted_delta += float(score * delta)

    delta_mix = float(weighted_delta / support_sum) if support_sum > EPS else 0.0
    gamma = float(caps["gamma"] * _clip(support_sum / 1.20, 0.0, 1.0))
    weights = {
        name: (float(item["score"] / support_sum) if support_sum > EPS else 0.0)
        for name, item in scored.items()
    }
    if regime == "no_guidance" and float(_safe_float(scored.get("auxiliary", {}).get("score") if isinstance(scored.get("auxiliary"), dict) else 0.0, 0.0)) > 0.0:
        gamma = float(max(gamma, _clip(0.80 + 0.20 * float(weights.get("auxiliary", 0.0)), 0.80, 1.0)))
    return {
        "delta_mix": float(delta_mix),
        "gamma": float(gamma),
        "final_delta": float(gamma * delta_mix),
        "support_sum": float(support_sum),
        "consensus_sign": float(consensus_sign),
        "scores": scored,
        "weights": weights,
        "caps": caps,
    }


def _apply_shared_residual_backbone_panel(
    frame: pd.DataFrame,
    *,
    proposal_map: Mapping[Tuple[str, str], float],
    guidance_quality_guardrail_mode: str,
    min_history: int,
    signed_strength_threshold: float,
    upward_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = frame.copy()
    pre_pred_col = "pred_csais_rawcard_direct_experts_v1_pre_shared_residual_backbone"
    pre_delta_col = "csais_final_candidate_delta_log_pre_shared_residual_backbone"
    out[pre_pred_col] = pd.to_numeric(out["pred_csais_rawcard_direct_experts_v1"], errors="coerce")
    out[pre_delta_col] = pd.to_numeric(out["csais_final_candidate_delta_log"], errors="coerce") if "csais_final_candidate_delta_log" in out.columns else np.nan

    for col, default in [
        ("csais_srb_mode", SHARED_RESIDUAL_BACKBONE_MODE),
        ("csais_srb_regime", ""),
        ("csais_srb_auxiliary_type", "none"),
        ("csais_srb_auxiliary_reason", "not_evaluated"),
    ]:
        out[col] = default
    numeric_defaults = {
        "csais_srb_gamma": 0.0,
        "csais_srb_delta_mix_log": 0.0,
        "csais_srb_support_sum": 0.0,
        "csais_srb_consensus_sign": 0.0,
        "csais_srb_current_card_delta_log": 0.0,
        "csais_srb_current_card_support": 0.0,
        "csais_srb_current_card_score": 0.0,
        "csais_srb_current_card_weight": 0.0,
        "csais_srb_temporal_memory_delta_log": 0.0,
        "csais_srb_temporal_memory_support": 0.0,
        "csais_srb_temporal_memory_score": 0.0,
        "csais_srb_temporal_memory_weight": 0.0,
        "csais_srb_auxiliary_delta_log": 0.0,
        "csais_srb_auxiliary_support": 0.0,
        "csais_srb_auxiliary_score": 0.0,
        "csais_srb_auxiliary_weight": 0.0,
        "csais_srb_auxiliary_active": 0,
        "csais_srb_no_guidance_history_n": 0,
        "csais_srb_no_guidance_signed_error": np.nan,
        "csais_srb_reference_delta_no_aux_log": 0.0,
        "csais_srb_reference_pred_no_aux": np.nan,
        "csais_srb_memory_gate_reference_pred": np.nan,
        "pred_csais_rawcard_direct_experts_v1_srb_pre_guidance_guardrail": np.nan,
    }
    for col, default in numeric_defaults.items():
        out[col] = default
    out["csais_srb_scores_json"] = "{}"
    out["csais_srb_weights_json"] = "{}"

    history: List[Dict[str, float]] = []
    same_or_future_history_rows = 0
    same_or_future_seen_buffer_rows = 0
    max_history_delta = -1
    active_aux_rows = 0
    active_no_guidance_aux_rows = 0
    active_explicit_aux_rows = 0
    active_weak_aux_rows = 0

    qnums = out["quarter"].astype(str).map(lambda value: int(_quarter_key(value)[0] * 4 + _quarter_key(value)[1]))
    sorted_index = out.assign(__srb_qnum=qnums).sort_values(["__srb_qnum", "ticker"]).index.tolist()
    for idx in sorted_index:
        row = out.loc[idx]
        ticker = str(row.get("ticker") or "").upper().strip()
        quarter = str(row.get("quarter") or "")
        qnum = int(qnums.loc[idx])
        anchor_pred = _safe_float(row.get("pred_csais_anchor"), float("nan"))
        regime = _shared_residual_backbone_regime(row)
        bucket = _row_guidance_bucket(row)
        total_support_scale = float(_clip(_safe_float(row.get("csais_intrinsic_direct_total_support_scale"), 1.0), 0.0, 2.0))
        current_delta = _safe_float(row.get("csais_intrinsic_direct_residual_candidate_delta_log"), float("nan"))
        if not np.isfinite(current_delta):
            current_delta = _safe_float(row.get("csais_intrinsic_direct_candidate_delta_log"), 0.0)
        current_delta = float(_clip(float(current_delta) * min(total_support_scale, 1.0), -0.30, 0.30)) if np.isfinite(current_delta) else 0.0
        current_support = float(_clip(_safe_float(row.get("csais_intrinsic_direct_candidate_support"), 0.0), 0.0, 1.0))
        compressed_delta = float(_clip(_safe_float(row.get("csais_compressed_base_candidate_delta_log"), 0.0), -0.30, 0.30))
        compressed_support = float(_clip(_safe_float(row.get("csais_compressed_base_candidate_support"), 0.0), 0.0, 1.0))
        current_delta = _shared_residual_backbone_reference_delta(
            [
                {"delta": compressed_delta, "support": compressed_support},
                {"delta": current_delta, "support": current_support},
            ]
        )
        current_support = float(_clip(max(current_support, compressed_support), 0.0, 1.0))
        temporal_delta = float(_clip(_safe_float(row.get("csais_temporal_direct_candidate_delta_log"), 0.0), -0.30, 0.30))
        temporal_support = float(_clip(_safe_float(row.get("csais_temporal_direct_candidate_support"), 0.0), 0.0, 1.0))
        conflict_ratio = float(_clip(_safe_float(row.get("fwd_conflict_ratio"), 0.0), 0.0, 1.0))
        current_conflict_penalty = float(_clip(1.0 - 0.50 * conflict_ratio, 0.35, 1.0))
        temporal_alignment = float(_clip(
            min(
                _safe_float(row.get("csais_temporal_direct_candidate_mean_direction_alignment"), 1.0),
                _safe_float(row.get("csais_temporal_direct_candidate_top_direction_alignment"), 1.0),
            ),
            0.35,
            1.0,
        ))
        candidates: List[Dict[str, Any]] = [
            {
                "name": "current_card",
                "delta": current_delta,
                "support": current_support,
                "conflict_penalty": current_conflict_penalty,
                "prior_reliability": 1.0,
            },
            {
                "name": "temporal_memory",
                "delta": temporal_delta,
                "support": temporal_support,
                "conflict_penalty": temporal_alignment,
                "prior_reliability": 1.0,
            },
        ]
        reference_delta = _shared_residual_backbone_reference_delta(candidates)
        reference_pred = float(anchor_pred * np.exp(reference_delta)) if np.isfinite(anchor_pred) and anchor_pred > 0.0 else float("nan")
        memory_gate_reference_pred = _safe_float(row.get(pre_pred_col), float("nan"))
        if not (np.isfinite(memory_gate_reference_pred) and memory_gate_reference_pred > 0.0):
            memory_gate_reference_pred = reference_pred

        aux_type = "none"
        aux_reason = "no_regime_auxiliary"
        aux_delta = 0.0
        aux_support = 0.0
        if regime == "explicit":
            aux_type = "strict_guidance_delta"
            aux_delta = float(_clip(_safe_float(row.get("csais_guidance_expert_delta_log"), 0.0), -0.30, 0.30))
            aux_support = float(_clip(_safe_float(row.get("csais_guidance_expert_support"), 0.0), 0.0, 1.0))
            aux_reason = str(row.get("csais_guidance_expert_reason") or "strict_guidance_expert")
        elif regime == "non_explicit":
            aux_type = "weak_signal_delta"
            guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
            if bucket == "derived_weak_numeric" and np.isfinite(anchor_pred) and anchor_pred > 0.0 and np.isfinite(guid_mid) and guid_mid > 0.0:
                guidance_lock = float(_clip(_safe_float(row.get("guidance_lock"), 0.0), 0.0, 1.0))
                aux_delta = float(_clip(0.35 * np.log(float(guid_mid) / float(anchor_pred)), -0.12, 0.12))
                aux_support = float(_clip(0.18 * (0.35 + 0.65 * guidance_lock), 0.0, 0.18))
                aux_reason = "derived_weak_numeric_guid_mid_soft_auxiliary"
            else:
                aux_reason = "no_numeric_weak_guidance_auxiliary"
        else:
            aux_type = "anchor_memory_delta"
            proposal = _safe_float(proposal_map.get((ticker, quarter)), float("nan"))
            prior = [item for item in history if int(item["qnum"]) < qnum]
            same_or_future_seen_buffer_rows += sum(1 for item in history if int(item["qnum"]) >= qnum)
            same_or_future_history_rows += sum(1 for item in prior if int(item["qnum"]) >= qnum)
            if prior:
                max_history_delta = max(max_history_delta, max(int(item["qnum"]) for item in prior) - qnum)
            history_n = int(len(prior))
            signed_error = float("nan")
            if history_n >= int(min_history):
                signed_error = float(np.mean([np.log(float(item["actual"]) / float(item["reference_pred"])) for item in prior]))
            out.at[idx, "csais_srb_no_guidance_history_n"] = history_n
            out.at[idx, "csais_srb_no_guidance_signed_error"] = signed_error
            if not (np.isfinite(anchor_pred) and anchor_pred > 0.0 and np.isfinite(memory_gate_reference_pred) and memory_gate_reference_pred > 0.0 and np.isfinite(proposal) and proposal > 0.0):
                aux_reason = "invalid_anchor_reference_or_memory_proposal"
            elif history_n < int(min_history):
                aux_reason = "insufficient_prior_no_guidance_history"
            elif not (np.isfinite(signed_error) and signed_error > float(signed_strength_threshold)):
                aux_reason = "prior_signed_error_strength_below_threshold"
            elif proposal < float(upward_ratio) * float(memory_gate_reference_pred):
                aux_reason = "memory_proposal_not_upward_vs_backbone_reference"
            else:
                strength = float(_clip((float(signed_error) - float(signed_strength_threshold)) / 0.08, 0.0, 1.0))
                count_scale = float(_clip(history_n / max(float(min_history), 1.0), 0.0, 1.0))
                aux_delta = float(_clip(np.log(float(proposal) / float(anchor_pred)), -0.30, 0.30))
                aux_support = float(_clip(0.50 * count_scale + 0.25 * strength, 0.0, 0.75))
                aux_reason = "prior_signed_error_memory_auxiliary_active"

        candidates.append(
            {
                "name": "auxiliary",
                "delta": float(aux_delta),
                "support": float(aux_support),
                "conflict_penalty": 1.0,
                "prior_reliability": 1.0,
                "auxiliary_type": aux_type,
            }
        )
        mix = _shared_residual_backbone_mix(regime=regime, candidates=candidates)
        pre_guardrail_delta = float(mix["final_delta"])
        pre_guardrail_pred = float(anchor_pred * np.exp(pre_guardrail_delta)) if np.isfinite(anchor_pred) and anchor_pred > 0.0 else float(anchor_pred)
        guardrail = _apply_guidance_quality_guardrail(
            anchor_pred=float(anchor_pred),
            final_pred=float(pre_guardrail_pred),
            guidance_label=str(row.get("guidance_availability") or bucket),
            mode=str(guidance_quality_guardrail_mode),
            anchor_history_score=_safe_float(row.get("anchor_best_model_history_mae"), float("nan")),
        )
        final_pred = float(guardrail["post_guardrail_pred"])
        final_delta = pre_guardrail_delta
        if np.isfinite(anchor_pred) and anchor_pred > 0.0 and np.isfinite(final_pred) and final_pred > 0.0:
            final_delta = float(np.log(final_pred / anchor_pred))

        scores = dict(mix.get("scores") or {})
        weights = dict(mix.get("weights") or {})
        aux_score = float(_safe_float(scores.get("auxiliary", {}).get("score") if isinstance(scores.get("auxiliary"), dict) else 0.0, 0.0))
        aux_active = int(aux_score > 0.0)
        active_aux_rows += int(aux_active)
        if aux_active and aux_type == "anchor_memory_delta":
            active_no_guidance_aux_rows += 1
        elif aux_active and aux_type == "strict_guidance_delta":
            active_explicit_aux_rows += 1
        elif aux_active and aux_type == "weak_signal_delta":
            active_weak_aux_rows += 1

        out.at[idx, "csais_srb_regime"] = regime
        out.at[idx, "csais_srb_auxiliary_type"] = aux_type
        out.at[idx, "csais_srb_auxiliary_reason"] = aux_reason
        out.at[idx, "csais_srb_gamma"] = float(mix["gamma"])
        out.at[idx, "csais_srb_delta_mix_log"] = float(mix["delta_mix"])
        out.at[idx, "csais_srb_support_sum"] = float(mix["support_sum"])
        out.at[idx, "csais_srb_consensus_sign"] = float(mix["consensus_sign"])
        out.at[idx, "csais_srb_current_card_delta_log"] = current_delta
        out.at[idx, "csais_srb_current_card_support"] = current_support
        out.at[idx, "csais_srb_current_card_score"] = float(_safe_float(scores.get("current_card", {}).get("score") if isinstance(scores.get("current_card"), dict) else 0.0, 0.0))
        out.at[idx, "csais_srb_current_card_weight"] = float(_safe_float(weights.get("current_card"), 0.0))
        out.at[idx, "csais_srb_temporal_memory_delta_log"] = temporal_delta
        out.at[idx, "csais_srb_temporal_memory_support"] = temporal_support
        out.at[idx, "csais_srb_temporal_memory_score"] = float(_safe_float(scores.get("temporal_memory", {}).get("score") if isinstance(scores.get("temporal_memory"), dict) else 0.0, 0.0))
        out.at[idx, "csais_srb_temporal_memory_weight"] = float(_safe_float(weights.get("temporal_memory"), 0.0))
        out.at[idx, "csais_srb_auxiliary_delta_log"] = float(aux_delta)
        out.at[idx, "csais_srb_auxiliary_support"] = float(aux_support)
        out.at[idx, "csais_srb_auxiliary_score"] = float(aux_score)
        out.at[idx, "csais_srb_auxiliary_weight"] = float(_safe_float(weights.get("auxiliary"), 0.0))
        out.at[idx, "csais_srb_auxiliary_active"] = int(aux_active)
        out.at[idx, "csais_srb_reference_delta_no_aux_log"] = float(reference_delta)
        out.at[idx, "csais_srb_reference_pred_no_aux"] = float(reference_pred) if np.isfinite(reference_pred) else float("nan")
        out.at[idx, "csais_srb_memory_gate_reference_pred"] = float(memory_gate_reference_pred) if np.isfinite(memory_gate_reference_pred) else float("nan")
        out.at[idx, "pred_csais_rawcard_direct_experts_v1_srb_pre_guidance_guardrail"] = float(pre_guardrail_pred)
        out.at[idx, "pred_csais_rawcard_direct_experts_v1_pre_guidance_guardrail"] = float(pre_guardrail_pred)
        out.at[idx, "pred_csais_rawcard_direct_experts_v1"] = float(final_pred)
        out.at[idx, "csais_pre_guidance_guardrail_candidate_delta_log"] = float(pre_guardrail_delta)
        out.at[idx, "csais_base_candidate_delta_log"] = float(pre_guardrail_delta)
        out.at[idx, "csais_final_candidate_delta_log"] = float(final_delta)
        out.at[idx, "csais_direct_expert_blend_intrinsic_weight"] = float(_safe_float(weights.get("current_card"), 0.0))
        out.at[idx, "csais_direct_expert_blend_temporal_weight"] = float(_safe_float(weights.get("temporal_memory"), 0.0))
        out.at[idx, "csais_direct_expert_blend_guidance_weight"] = float(_safe_float(weights.get("auxiliary"), 0.0))
        out.at[idx, "csais_direct_expert_blend_support_sum"] = float(mix["support_sum"])
        out.at[idx, "csais_srb_scores_json"] = json.dumps(sanitize_for_json(scores), ensure_ascii=False, allow_nan=False)
        out.at[idx, "csais_srb_weights_json"] = json.dumps(sanitize_for_json(weights), ensure_ascii=False, allow_nan=False)

        actual = _safe_float(row.get("actual", row.get("actual_stat", row.get("y_true"))), float("nan"))
        if regime == "no_guidance" and np.isfinite(actual) and actual > 0.0 and np.isfinite(memory_gate_reference_pred) and memory_gate_reference_pred > 0.0:
            history.append({"qnum": float(qnum), "actual": float(actual), "reference_pred": float(memory_gate_reference_pred)})

    validation = {
        "mode": SHARED_RESIDUAL_BACKBONE_MODE,
        "proposal_rows": int(sum(1 for value in proposal_map.values() if np.isfinite(_safe_float(value, float("nan"))))),
        "min_history": int(min_history),
        "signed_strength_threshold": float(signed_strength_threshold),
        "upward_ratio": float(upward_ratio),
        "same_or_future_history_rows": int(same_or_future_history_rows),
        "same_or_future_seen_buffer_rows": int(same_or_future_seen_buffer_rows),
        "max_history_qnum_minus_target_qnum": int(max_history_delta),
        "active_aux_rows": int(active_aux_rows),
        "active_no_guidance_anchor_memory_rows": int(active_no_guidance_aux_rows),
        "active_explicit_guidance_rows": int(active_explicit_aux_rows),
        "active_weak_signal_rows": int(active_weak_aux_rows),
    }
    return out, validation


def _company_summary_from_quarterly(quarterly: pd.DataFrame, method_family: str, expert_contract: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ticker, group in quarterly.groupby("ticker", sort=True):
        company_metrics = {
            "baseline": _metrics(group["actual"], group["baseline_pred"]),
            "csais_anchor": _metrics(group["actual"], group["pred_csais_anchor"]),
            "rawcard_direct_experts": _metrics(group["actual"], group["pred_csais_rawcard_direct_experts_v1"]),
        }
        best_stat_mae = float(_safe_float(group["best_stat_mae_company"].dropna().iloc[0], float("nan"))) if "best_stat_mae_company" in group and not group["best_stat_mae_company"].dropna().empty else float("nan")
        best_stat_model = str(group["best_stat_model"].dropna().iloc[0]) if "best_stat_model" in group and not group["best_stat_model"].dropna().empty else ""
        rows.append(
            {
                "ticker": str(ticker),
                "n": int(len(group)),
                "method_family": str(method_family),
                "expert_contract": str(expert_contract),
                "baseline_mae": float(company_metrics["baseline"]["mae"]),
                "csais_anchor_mae": float(company_metrics["csais_anchor"]["mae"]),
                "rawcard_direct_experts_mae": float(company_metrics["rawcard_direct_experts"]["mae"]),
                "best_stat_model": best_stat_model,
                "best_stat_mae": float(best_stat_mae),
                "anchor_selection_mode": str(group["anchor_selection_mode"].dropna().iloc[0]) if "anchor_selection_mode" in group and not group["anchor_selection_mode"].dropna().empty else "",
                "beats_best_stat": bool(float(company_metrics["rawcard_direct_experts"]["mae"]) < float(best_stat_mae)) if np.isfinite(best_stat_mae) else False,
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def _is_guidance_dependent_anchor_col(col: str) -> bool:
    text = str(col)
    return text in GUIDANCE_DEPENDENT_ANCHOR_COLS or text.endswith("_lag_guid") or text.endswith("sarimax_guid")


def _is_any_guidance_anchor_model(model: str) -> bool:
    text = str(model or "")
    col = text if text.startswith("pred__") else f"pred__{text}"
    return col in DIRECT_GUIDANCE_ANCHOR_COLS or _is_guidance_dependent_anchor_col(col)


def _add_robust_momentum_anchor_candidate(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if str(mode) == "off" or df.empty:
        return df
    out = df.copy()
    if str(mode) == "median_naive_auto_ma":
        cols = [col for col in ["pred__naive", "pred__auto_ets", "pred__auto_theta", "pred__ma"] if col in out.columns]
    else:
        cols = []
    if not cols:
        return out
    values = out[cols].apply(pd.to_numeric, errors="coerce")
    out[ROBUST_MOMENTUM_ANCHOR_COL] = values.median(axis=1, skipna=True)
    return out


def _select_missing_anchor_fallback(
    row: Mapping[str, Any],
    model_cols: Sequence[str],
    policy: str,
) -> Tuple[str, float, str]:
    if str(policy) == "skip":
        return "", float("nan"), "skip"
    if str(policy) != "repaired_quarter_prior_v1":
        return "", float("nan"), f"unsupported_{policy}"

    candidates = [
        ("baseline_pred", row.get("baseline_pred")),
        ("pred__naive", row.get("pred__naive")),
        ("guid_mid", row.get("guid_mid")),
        ("pred__seasonal_naive_q4", row.get("pred__seasonal_naive_q4")),
        ("pred__ma", row.get("pred__ma")),
        ("pred__mean", row.get("pred__mean")),
    ]
    for name, value in candidates:
        pred = _safe_float(value, float("nan"))
        if np.isfinite(pred) and pred > 0.0:
            return name, float(pred), f"repaired_quarter_prior_v1_{name}"

    for col in model_cols:
        pred = _safe_float(row.get(col), float("nan"))
        if np.isfinite(pred) and pred > 0.0:
            return str(col), float(pred), "repaired_quarter_prior_v1_first_finite_model"
    return "", float("nan"), "repaired_quarter_prior_v1_no_finite_candidate"


def _guidance_quality_guardrail_alpha(
    mode: str,
    guidance_label: str,
    anchor_history_score: float | None = None,
) -> float:
    if str(mode) == "explicit_strong_anchor_history_v1":
        if (
            str(guidance_label or "") == "explicit_numeric"
            and np.isfinite(_safe_float(anchor_history_score, float("nan")))
            and _safe_float(anchor_history_score, float("nan")) <= 0.06
        ):
            return 0.0
        return float(GUIDANCE_QUALITY_GUARDRAIL_ALPHA.get(str(guidance_label or "none"), 1.0))
    if str(mode) == "unified_v1":
        return float(GUIDANCE_QUALITY_GUARDRAIL_ALPHA.get(str(guidance_label or "none"), 1.0))
    return 1.0


def _apply_guidance_quality_guardrail(
    *,
    anchor_pred: float,
    final_pred: float,
    guidance_label: str,
    mode: str,
    anchor_history_score: float | None = None,
) -> Dict[str, float | str | int]:
    alpha = _guidance_quality_guardrail_alpha(mode, guidance_label, anchor_history_score)
    pre_guardrail_pred = float(final_pred)
    guarded_pred = float(final_pred)
    if alpha < 1.0 and np.isfinite(anchor_pred) and np.isfinite(final_pred):
        guarded_pred = float(anchor_pred + alpha * (final_pred - anchor_pred))
    return {
        "mode": str(mode),
        "guidance_label": str(guidance_label or "none"),
        "anchor_history_score": float(_safe_float(anchor_history_score, float("nan"))),
        "alpha": float(alpha),
        "applied": int(alpha < 1.0),
        "pre_guardrail_pred": pre_guardrail_pred,
        "post_guardrail_pred": guarded_pred,
    }


def _guidance_expert_candidate(
    *,
    row: Mapping[str, Any],
    anchor_pred: float,
    anchor_model: str,
    guidance_lock: float,
    anchor_error_state: Mapping[str, Any],
    mode: str,
    min_history: int,
    history_tau: float,
    support_scale: float,
) -> Dict[str, Any]:
    mode = str(mode)
    result: Dict[str, Any] = {
        "mode": mode,
        "active": 0,
        "delta": 0.0,
        "support": 0.0,
        "guid_mid": float("nan"),
        "anchor_model": str(anchor_model or ""),
        "history_abs_log": float("nan"),
        "history_count_scale": 0.0,
        "history_trust_scale": 0.0,
        "reason": "guidance_expert_off" if mode == "off" else "not_explicit_numeric_guidance",
    }
    if mode == "off":
        return result
    if mode not in {"explicit_history_trust_v1", "explicit_history_trust_non_guidance_anchor_v1"}:
        result["reason"] = f"unsupported_mode_{mode}"
        return result
    if _row_guidance_bucket(row) != "explicit_numeric":
        return result
    if mode == "explicit_history_trust_non_guidance_anchor_v1" and _is_any_guidance_anchor_model(anchor_model):
        result["reason"] = "guidance_dependent_anchor_abstain"
        return result
    guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
    result["guid_mid"] = float(guid_mid)
    if not (np.isfinite(anchor_pred) and anchor_pred > 0.0 and np.isfinite(guid_mid) and guid_mid > 0.0):
        result["reason"] = "missing_anchor_or_guid_mid"
        return result

    history_abs_log = _safe_float(anchor_error_state.get("anchor_error_same_guidance_abs_log"), float("nan"))
    if not np.isfinite(history_abs_log):
        history_abs_log = _safe_float(anchor_error_state.get("anchor_error_recent_abs_log"), float("nan"))
    if not np.isfinite(history_abs_log):
        history_abs_log = _safe_float(anchor_error_state.get("anchor_error_overall_abs_log"), float("nan"))
    history_count = int(max(_safe_float(anchor_error_state.get("anchor_error_history_n"), 0.0), 0.0))
    count_scale = 1.0 if int(min_history) <= 0 else float(_clip(history_count / max(float(min_history), 1.0), 0.0, 1.0))
    tau = max(float(history_tau), EPS)
    history_scale = float(np.exp(-max(float(history_abs_log), 0.0) / tau)) if np.isfinite(history_abs_log) else 0.5
    lock_scale = float(_clip(0.35 + 0.65 * float(guidance_lock), 0.0, 1.0))
    support = float(_clip(float(support_scale) * lock_scale * history_scale * count_scale, 0.0, 1.0))
    delta = float(np.log(float(guid_mid) / float(anchor_pred)))
    result.update(
        {
            "active": int(support > 0.0),
            "delta": float(delta),
            "support": float(support),
            "history_abs_log": float(history_abs_log) if np.isfinite(history_abs_log) else float("nan"),
            "history_count_scale": float(count_scale),
            "history_trust_scale": float(history_scale),
            "reason": "explicit_numeric_guidance_history_trust",
        }
    )
    return result


def _derive_integrated_arbitration_floor_ratio(
    *,
    exp: Mapping[str, Any],
    project_root: Path,
    dev_tickers: Sequence[str],
    dev_end_quarter: str,
) -> Dict[str, Any]:
    dev_set = {str(ticker).upper() for ticker in dev_tickers}
    dev_end_key = _quarter_key(str(dev_end_quarter))
    abs_log_errors: List[float] = []
    for company in exp.get("companies", []):
        ticker = str(company.get("ticker") or "").upper()
        if ticker not in dev_set:
            continue
        panel, _best_model, _best_mae, _prehist_model, _prehist_mae, _history = _prepare_company_panel(company, project_root)
        for _idx, row in panel.iterrows():
            quarter = str(row.get("quarter") or "")
            if _quarter_key(quarter) > dev_end_key:
                continue
            if str(row.get("guidance_availability") or _row_guidance_bucket(row)) != "none":
                continue
            guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
            if np.isfinite(guid_mid):
                continue
            actual = _actual_value_for_anchor_history(row)
            seasonal = _safe_float(row.get("pred__seasonal_naive_q4"), float("nan"))
            if not (np.isfinite(actual) and actual > 0.0 and np.isfinite(seasonal) and seasonal > 0.0):
                continue
            abs_log_errors.append(float(abs(np.log(seasonal / actual))))
    if not abs_log_errors:
        return {
            "floor_ratio": float(INTEGRATED_ARBITRATION_FLOOR_RATIO_FALLBACK),
            "floor_ratio_source": "fallback_no_dev_no_guidance_rows",
            "support_rows": 0,
            "median_abs_log_error": float("nan"),
        }
    median_abs_log_error = float(np.nanmedian(np.asarray(abs_log_errors, dtype=float)))
    return {
        "floor_ratio": float(np.exp(-median_abs_log_error)),
        "floor_ratio_source": "retained_id_pretest_no_guidance_seasonal_naive_q4_median_abs_log_error",
        "support_rows": int(len(abs_log_errors)),
        "median_abs_log_error": median_abs_log_error,
    }


def _actual_value_for_anchor_history(row: Mapping[str, Any]) -> float:
    return _safe_float(row.get("actual", row.get("actual_stat", row.get("y_true"))), float("nan"))


def _anchor_history_error_record(row: Mapping[str, Any], pred: float) -> Dict[str, Any]:
    actual = _actual_value_for_anchor_history(row)
    if not (np.isfinite(actual) and np.isfinite(pred) and pred > 0.0):
        return {}
    quarter = str(row.get("quarter") or row.get("fiscal_quarter") or "")
    denom = max(abs(actual) + abs(pred), EPS)
    record = {
        "qnum": float(_quarter_key(quarter)[0] * 4 + _quarter_key(quarter)[1]),
        "mae": float(abs(actual - pred)),
        "smape": float(2.0 * abs(actual - pred) / denom),
        "mape": float(abs(actual - pred) / max(abs(actual), EPS)),
        "guidance_bucket": _row_guidance_bucket(row),
    }
    actual_log = _safe_log(actual)
    pred_log = _safe_log(pred)
    record["logabs"] = float(abs(actual_log - pred_log)) if np.isfinite(actual_log) and np.isfinite(pred_log) else float("nan")
    guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
    if np.isfinite(guid_mid) and guid_mid > 0.0:
        record["guidance_abs_log_ratio"] = float(abs(np.log(pred / guid_mid)))
    else:
        record["guidance_abs_log_ratio"] = float("nan")
    return record


def _append_anchor_model_errors(
    row: Mapping[str, Any],
    model_cols: Sequence[str],
    model_error_history: Dict[str, List[Dict[str, Any]]],
) -> None:
    has_numeric_guidance = _row_has_numeric_guidance(row)
    for col in model_cols:
        if str(col) in DIRECT_GUIDANCE_ANCHOR_COLS and not has_numeric_guidance:
            continue
        pred = _safe_float(row.get(col), float("nan"))
        record = _anchor_history_error_record(row, pred)
        if record:
            model_error_history.setdefault(str(col), []).append(record)


def _select_online_anchor_col(
    row: Mapping[str, Any],
    model_cols: Sequence[str],
    model_error_history: Mapping[str, Sequence[Mapping[str, Any]]],
    min_history: int,
    score_metric: str = "mae",
    window: int = 0,
    half_life: float = 0.0,
    same_quarter_weight: float = 0.0,
    guidance_regime_mode: str = "off",
    guidance_same_regime_min_history: int = 4,
    guidance_mismatch_penalty: float = 0.0,
    explicit_guidance_proximity_mode: str = "off",
    explicit_guidance_proximity_weight: float = 0.0,
    explicit_guidance_kernel_min_history: int = 4,
    explicit_guidance_kernel_band: float = 0.20,
    explicit_guidance_kernel_shrink_k: float = 8.0,
) -> Tuple[str, float, int, str]:
    current_bucket = _row_guidance_bucket(row)
    finite_cols = [str(col) for col in model_cols if np.isfinite(_safe_float(row.get(col), float("nan")))]
    if not _row_has_numeric_guidance(row):
        finite_cols = [col for col in finite_cols if col not in DIRECT_GUIDANCE_ANCHOR_COLS]
    if current_bucket != "none":
        finite_cols = [col for col in finite_cols if col != ROBUST_MOMENTUM_ANCHOR_COL]
    if not finite_cols:
        return "", float("nan"), 0, "no_finite_anchor_candidate"

    quarter = str(row.get("quarter") or row.get("fiscal_quarter") or "")
    current_key = _quarter_key(quarter)
    current_qnum = int(current_key[0] * 4 + current_key[1])
    current_fq = int(current_key[1])
    guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
    explicit_proximity_mode = str(explicit_guidance_proximity_mode or "off")
    use_explicit_guidance_proximity = (
        current_bucket == "explicit_numeric"
        and np.isfinite(guid_mid)
        and guid_mid > 0.0
        and explicit_proximity_mode != "off"
    )
    scored: List[Tuple[float, str, int, float, int, float, float, int, float, float]] = []
    for col in finite_cols:
        values: List[Tuple[float, float, str, float]] = []
        for item in model_error_history.get(col, []):
            value = _safe_float(item.get(score_metric), float("nan"))
            qnum = int(_safe_float(item.get("qnum"), 0.0))
            if not (np.isfinite(value) and qnum > 0 and qnum < current_qnum):
                continue
            if int(window) > 0 and qnum < current_qnum - int(window):
                continue
            weight = 1.0
            if float(half_life) > 0.0:
                age = max(current_qnum - qnum, 1)
                weight *= 0.5 ** (float(age) / float(half_life))
            if float(same_quarter_weight) > 0.0 and ((qnum - 1) % 4 + 1) == current_fq:
                weight *= 1.0 + float(same_quarter_weight)
            values.append(
                (
                    float(value),
                    float(weight),
                    str(item.get("guidance_bucket") or "none"),
                    float(_safe_float(item.get("guidance_abs_log_ratio"), float("nan"))),
                )
            )
        if len(values) >= int(min_history):
            value_arr = np.asarray([item[0] for item in values], dtype=float)
            weight_arr = np.asarray([item[1] for item in values], dtype=float)
            all_score = float(np.average(value_arr, weights=weight_arr))
            same_values = [item for item in values if item[2] == current_bucket]
            same_n = int(len(same_values))
            raw_score = all_score
            score = all_score
            penalty = 0.0
            kernel_score = float("nan")
            kernel_n = 0
            kernel_effective_n = 0.0
            kernel_blend_lambda = 0.0
            if str(guidance_regime_mode) in {"penalized_v1", "mismatch_penalty_v1"} and current_bucket == "none":
                if str(guidance_regime_mode) == "penalized_v1" and same_n >= int(guidance_same_regime_min_history):
                    same_value_arr = np.asarray([item[0] for item in same_values], dtype=float)
                    same_weight_arr = np.asarray([item[1] for item in same_values], dtype=float)
                    raw_score = float(np.average(same_value_arr, weights=same_weight_arr))
                    score = raw_score
                if (
                    _is_guidance_dependent_anchor_col(col)
                    and same_n < int(guidance_same_regime_min_history)
                ):
                    penalty = float(guidance_mismatch_penalty)
                    score += penalty
            if use_explicit_guidance_proximity:
                pred = _safe_float(row.get(col), float("nan"))
                if np.isfinite(pred) and pred > 0.0:
                    current_ratio = abs(float(np.log(pred / guid_mid)))
                    if explicit_proximity_mode == "fixed_penalty_v1" and float(explicit_guidance_proximity_weight) > 0.0:
                        proximity_penalty = float(explicit_guidance_proximity_weight) * current_ratio
                        score += proximity_penalty
                        penalty += proximity_penalty
                    elif explicit_proximity_mode in {"kernel_history_v1", "kernel_blend_v1"}:
                        band = max(float(explicit_guidance_kernel_band), EPS)
                        kernel_values: List[Tuple[float, float]] = []
                        for hist_value, hist_weight, hist_bucket, hist_ratio in values:
                            if hist_bucket != "explicit_numeric" or not np.isfinite(hist_ratio):
                                continue
                            kernel_weight = float(hist_weight) * float(np.exp(-abs(hist_ratio - current_ratio) / band))
                            if kernel_weight > 0.0:
                                kernel_values.append((float(hist_value), kernel_weight))
                        kernel_n = int(len(kernel_values))
                        if kernel_n >= int(explicit_guidance_kernel_min_history):
                            kernel_weight_arr = np.asarray([item[1] for item in kernel_values], dtype=float)
                            kernel_weight_sum = float(np.sum(kernel_weight_arr))
                            kernel_weight_sq_sum = float(np.sum(np.square(kernel_weight_arr)))
                            if kernel_weight_sum > 0.0 and kernel_weight_sq_sum > 0.0:
                                kernel_effective_n = float((kernel_weight_sum * kernel_weight_sum) / kernel_weight_sq_sum)
                            kernel_score = float(
                                np.average(
                                    np.asarray([item[0] for item in kernel_values], dtype=float),
                                    weights=kernel_weight_arr,
                                )
                            )
                            prior_score = float(score)
                            if explicit_proximity_mode == "kernel_history_v1":
                                score = float(kernel_score)
                            else:
                                kernel_blend_lambda = float(
                                    kernel_effective_n / max(kernel_effective_n + float(explicit_guidance_kernel_shrink_k), EPS)
                                )
                                score = float((1.0 - kernel_blend_lambda) * score + kernel_blend_lambda * kernel_score)
                            penalty += float(score - prior_score)
            scored.append(
                (
                    float(score),
                    col,
                    int(len(values)),
                    float(raw_score),
                    same_n,
                    float(penalty),
                    float(kernel_score),
                    int(kernel_n),
                    float(kernel_effective_n),
                    float(kernel_blend_lambda),
                )
            )
    if scored:
        scored.sort(key=lambda item: (item[0], item[1]))
        mae, col, n_hist, raw_score, same_n, penalty, kernel_score, kernel_n, kernel_effective_n, kernel_blend_lambda = scored[0]
        reason = "historical_" + str(score_metric)
        if int(window) > 0:
            reason += f"_rolling{int(window)}"
        if float(half_life) > 0.0:
            reason += f"_decay{float(half_life):g}"
        if float(same_quarter_weight) > 0.0:
            reason += f"_sameq{float(same_quarter_weight):g}"
        if str(guidance_regime_mode) != "off":
            reason += f"_guidregime_{current_bucket}_same{int(same_n)}_raw{float(raw_score):.4f}_penalty{float(penalty):.4f}"
        if use_explicit_guidance_proximity:
            reason += f"_explicitguidprox_{explicit_proximity_mode}"
            if explicit_proximity_mode == "fixed_penalty_v1":
                reason += f"_w{float(explicit_guidance_proximity_weight):g}"
            elif explicit_proximity_mode in {"kernel_history_v1", "kernel_blend_v1"}:
                reason += f"_kmin{int(explicit_guidance_kernel_min_history)}_band{float(explicit_guidance_kernel_band):g}"
                if explicit_proximity_mode == "kernel_blend_v1":
                    reason += f"_shrink{float(explicit_guidance_kernel_shrink_k):g}"
                reason += f"_kn{int(kernel_n)}_keff{float(kernel_effective_n):.2f}_lambda{float(kernel_blend_lambda):.3f}_ks{float(kernel_score):.4f}"
        return col, float(mae), int(n_hist), reason

    for model in ANCHOR_FALLBACK_MODELS:
        col = f"pred__{model}"
        if col in finite_cols:
            return col, float("nan"), 0, f"fallback_{model}"
    return finite_cols[0], float("nan"), 0, "fallback_first_finite"


def _seed_online_anchor_history(
    stat_df: pd.DataFrame,
    model_cols: Sequence[str],
    first_panel_quarter: str,
    min_history: int,
    score_metric: str = "mae",
    window: int = 0,
    half_life: float = 0.0,
    same_quarter_weight: float = 0.0,
    guidance_regime_mode: str = "off",
    guidance_same_regime_min_history: int = 4,
    guidance_mismatch_penalty: float = 0.0,
    explicit_guidance_proximity_mode: str = "off",
    explicit_guidance_proximity_weight: float = 0.0,
    explicit_guidance_kernel_min_history: int = 4,
    explicit_guidance_kernel_band: float = 0.20,
    explicit_guidance_kernel_shrink_k: float = 8.0,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    model_error_history: Dict[str, List[Dict[str, Any]]] = {str(col): [] for col in model_cols}
    selected_anchor_error_history: List[Dict[str, Any]] = []
    if stat_df.empty:
        return model_error_history, selected_anchor_error_history

    hist = stat_df[stat_df["quarter"].astype(str).map(_quarter_key) < _quarter_key(str(first_panel_quarter))].copy()
    if hist.empty:
        return model_error_history, selected_anchor_error_history
    hist = hist.sort_values("quarter", key=lambda s: s.astype(str).map(_quarter_key)).reset_index(drop=True)
    for _, row in hist.iterrows():
        col, _mae, _n_hist, _reason = _select_online_anchor_col(
            row,
            model_cols,
            model_error_history,
            min_history,
            score_metric=score_metric,
            window=window,
            half_life=half_life,
            same_quarter_weight=same_quarter_weight,
            guidance_regime_mode=guidance_regime_mode,
            guidance_same_regime_min_history=guidance_same_regime_min_history,
            guidance_mismatch_penalty=guidance_mismatch_penalty,
            explicit_guidance_proximity_mode=explicit_guidance_proximity_mode,
            explicit_guidance_proximity_weight=explicit_guidance_proximity_weight,
            explicit_guidance_kernel_min_history=explicit_guidance_kernel_min_history,
            explicit_guidance_kernel_band=explicit_guidance_kernel_band,
            explicit_guidance_kernel_shrink_k=explicit_guidance_kernel_shrink_k,
        )
        actual = _actual_value_for_anchor_history(row)
        pred = _safe_float(row.get(col), float("nan")) if col else float("nan")
        if np.isfinite(actual) and np.isfinite(pred):
            selected_anchor_error_history.append(
                {
                    "quarter": str(row.get("quarter") or ""),
                    "guidance_availability": _row_guidance_bucket(row),
                    "anchor_abs_log_error": float(abs(_safe_log(actual) - _safe_log(pred))) if np.isfinite(_safe_log(actual)) and np.isfinite(_safe_log(pred)) else 0.0,
                }
            )
        _append_anchor_model_errors(row, model_cols, model_error_history)
    return model_error_history, selected_anchor_error_history

def _same_fiscal_quarter_bonus(current_quarter: str, past_quarter: str) -> float:
    try:
        return 1.0 if int(str(current_quarter).split("_Q", 1)[1]) == int(str(past_quarter).split("_Q", 1)[1]) else 0.0
    except Exception:
        return 0.0


def _retrieve_context_card_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cards_by_segment = payload.get("cards_by_segment", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(cards_by_segment, Mapping):
        return []
    shares = payload.get("segment_shares_base_canonical") or payload.get("segment_shares_base_raw") or {}
    shares = shares if isinstance(shares, Mapping) else {}
    ranked_segments = sorted(
        [(str(segment), float(_safe_float(value, 0.0))) for segment, value in shares.items()],
        key=lambda item: item[1],
        reverse=True,
    )
    rank_map = {segment: idx + 1 for idx, (segment, _) in enumerate(ranked_segments)}
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for segment_name, raw_cards in cards_by_segment.items():
        if not isinstance(raw_cards, list):
            continue
        for raw_card in raw_cards:
            if not isinstance(raw_card, Mapping):
                continue
            segment = str(raw_card.get("segment") or segment_name or "UNKNOWN")
            relation_family = str(raw_card.get("relation_family") or raw_card.get("category") or "unknown")
            category = str(raw_card.get("category") or raw_card.get("relation_family") or "other")
            evidence = str(raw_card.get("verbatim") or raw_card.get("evidence") or "")
            instance_id = str(raw_card.get("instance_id") or "")
            dedupe_key = instance_id or json.dumps(
                [segment, relation_family, category, raw_card.get("polarity"), evidence[:120]],
                ensure_ascii=False,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(
                {
                    "instance_id": instance_id,
                    "signature_key": dedupe_key,
                    "native_source": "retrieve_context",
                    "driver_source": str(raw_card.get("driver_source") or "retrieve_context"),
                    "segment": segment,
                    "segment_normalized": segment,
                    "relation_family": relation_family,
                    "category": category,
                    "canonical_factor": str(raw_card.get("canonical_factor") or category or relation_family or "other"),
                    "polarity": str(raw_card.get("polarity") or "unknown"),
                    "strength": str(raw_card.get("strength") or "unknown"),
                    "confidence": float(_safe_float(raw_card.get("confidence"), 0.0)),
                    "weight": float(_safe_float(raw_card.get("weight"), 0.0)),
                    "persistence_hint": bool(raw_card.get("persistence_hint")),
                    "attribution_anchor": str(raw_card.get("attribution_anchor") or relation_family or "retrieve_context"),
                    "evidence": evidence,
                    "release_token_ids": list(raw_card.get("release_token_ids") or []),
                    "source_text_sha256": str(raw_card.get("source_text_sha256") or raw_card.get("verbatim_sha256") or ""),
                    "release_status": str(raw_card.get("release_status") or ""),
                    "segment_share_at_observed": float(_safe_float(shares.get(segment), 0.0)),
                    "segment_rank_at_observed": int(rank_map.get(segment, 0)),
                }
            )
    return rows


def _segment_share_map_from_context(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        segment = str(row.get("segment") or row.get("segment_normalized") or "").strip()
        if not segment:
            continue
        share = float(_clip(_safe_float(row.get("segment_share_at_observed"), 0.0), 0.0, 1.0))
        weight = abs(float(_safe_float(row.get("weight"), 0.0))) * max(float(_safe_float(row.get("confidence"), 0.0)), 0.25)
        out[segment] = max(float(out.get(segment, 0.0)), float(share * max(weight, 0.25)))
    return out


def _weighted_jaccard_dict(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left.keys()) | set(right.keys())
    if not keys:
        return 0.0
    numerator = sum(min(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    denominator = sum(max(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    return float(0.0 if denominator <= EPS else numerator / denominator)


def _segment_scale_compatibility(
    current_state: Mapping[str, Any],
    past_state: Mapping[str, Any],
    current_context_rows: Sequence[Mapping[str, Any]],
    past_context_rows: Sequence[Mapping[str, Any]],
) -> float:
    top_gap = abs(float(_safe_float(current_state.get("segment_share_top1"), 0.0)) - float(_safe_float(past_state.get("segment_share_top1"), 0.0)))
    top2_gap = abs(float(_safe_float(current_state.get("segment_share_top2"), 0.0)) - float(_safe_float(past_state.get("segment_share_top2"), 0.0)))
    count_gap = abs(float(_safe_float(current_state.get("segment_share_count"), 0.0)) - float(_safe_float(past_state.get("segment_share_count"), 0.0)))
    profile_gap = float(top_gap + 0.5 * top2_gap + 0.08 * count_gap)
    profile_compat = float(np.exp(-profile_gap / 0.75))
    context_compat = _weighted_jaccard_dict(
        _segment_share_map_from_context(current_context_rows),
        _segment_share_map_from_context(past_context_rows),
    )
    if context_compat <= 0.0:
        return float(_clip(profile_compat, 0.0, 1.0))
    return float(_clip(0.65 * profile_compat + 0.35 * context_compat, 0.0, 1.0))


def _temporal_novelty_value(state: Mapping[str, Any]) -> float:
    churn = max(0.0, float(_safe_float(state.get("fg_churn_conf_mass"), 0.0)))
    shock = max(0.0, float(_safe_float(state.get("fg_shock_fwd_abs_sum"), state.get("fg_shock_abs_mean_nz", 0.0))))
    edge_count = max(0.0, float(_safe_float(state.get("fg_kg_num_edges"), 0.0)))
    raw = float(np.log1p(churn + 8.0 * shock) / 4.0)
    if edge_count > 0.0:
        raw *= float(_clip(1.0 + 0.05 * np.log1p(edge_count), 1.0, 1.25))
    return float(_clip(raw, 0.0, 1.0))


def _guidance_quality_value(state: Mapping[str, Any]) -> float:
    if "guidance_lock" in state:
        return float(_clip(_safe_float(state.get("guidance_lock"), 0.0), 0.0, 1.0))
    numeric = float(_clip(_safe_float(state.get("guidance_numeric_available"), 0.0), 0.0, 1.0))
    score = float(_clip(_safe_float(state.get("guidance_score_norm"), 0.0), 0.0, 1.0))
    band = max(0.0, float(_safe_float(state.get("guid_band_ratio"), 0.0)))
    return float(_clip(numeric * score * max(0.0, 1.0 - band / 0.15), 0.0, 1.0))


_IMMEDIATE_REVENUE_TERMS = (
    "revenue",
    "sales",
    "demand",
    "order",
    "backlog",
    "delivery",
    "deliveries",
    "ship",
    "shipment",
    "production",
    "inventory",
    "guide",
    "guidance",
    "outlook",
    "headwind",
    "tailwind",
    "foreign exchange",
    "fx",
    "currency",
    "pricing",
    "price",
    "asp",
    "q1",
    "q2",
    "q3",
    "q4",
    "quarter",
    "first quarter",
    "second quarter",
    "third quarter",
    "fourth quarter",
)

_LONG_TERM_TERMS = (
    "long-term",
    "long term",
    "over time",
    "in the long run",
    "eventually",
    "years from now",
    "multi-year",
    "multiyear",
    "next generation",
    "future opportunity",
    "longer term",
)

_COST_MARGIN_TERMS = (
    "gross margin",
    "operating margin",
    "margin was",
    "margin is",
    "cost reduction",
    "cost reductions",
    "lowering the production costs",
    "opex",
    "operating expenses",
    "r&d spend",
    "capital spending",
)


def _temporal_evidence_filter_reason(row: Mapping[str, Any], mode: str) -> str:
    mode = str(mode or "off")
    if mode == "off":
        return "keep_off"
    text = str(row.get("evidence") or row.get("verbatim") or "").lower()
    relation_family = str(row.get("relation_family") or row.get("category") or "").lower()
    canonical_factor = str(row.get("canonical_factor") or "").lower()
    segment = str(row.get("segment") or row.get("segment_normalized") or "").strip().lower()
    has_immediate = any(term in text for term in _IMMEDIATE_REVENUE_TERMS)
    has_long_term = any(term in text for term in _LONG_TERM_TERMS)
    has_cost_margin = any(term in text for term in _COST_MARGIN_TERMS)
    is_target_forward = bool(row.get("is_forward_target_quarter"))
    temporal_type = str(row.get("temporal_type") or "").lower()
    is_forecast_like = (not temporal_type) or "forecast" in temporal_type or "guidance" in temporal_type
    if has_long_term and not any(term in text for term in ("q1", "q2", "q3", "q4", "quarter", "guidance", "guide")):
        return "drop_long_term"
    if has_cost_margin and not any(term in text for term in ("revenue", "sales", "price", "pricing", "asp", "demand", "delivery", "deliveries", "ship")):
        return "drop_cost_margin_only"
    if mode == "immediate_revenue_strict_v1":
        if segment in {"", "unknown", "theme"} and not any(term in text for term in ("revenue", "headwind", "tailwind", "foreign exchange", "fx", "guidance", "guide")):
            return "drop_weak_unknown_segment"
        if relation_family in {"product_transition", "other"} and canonical_factor in {"product_transition", "other", "revenue_boost_limit"} and not has_immediate:
            return "drop_weak_product_transition"
        if is_target_forward and not is_forecast_like and not has_immediate:
            return "drop_nonforecast_weak_forward"
    if not has_immediate and relation_family in {"product_transition", "other", "revenue_limit"} and mode in {"immediate_revenue_v1", "immediate_revenue_strict_v1"}:
        return "drop_nonimmediate_weak_relation"
    return "keep"


def _filter_temporal_evidence_rows(rows: Sequence[Mapping[str, Any]], mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = str(mode or "off")
    if mode == "off":
        return [dict(row) for row in rows if isinstance(row, Mapping)], {
            "mode": "off",
            "before": int(len(rows)),
            "after": int(len(rows)),
            "dropped": 0,
            "reason_counts": {},
        }
    kept: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reason = _temporal_evidence_filter_reason(row, mode)
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        if not reason.startswith("drop_"):
            kept.append(dict(row))
    return kept, {
        "mode": mode,
        "before": int(len(rows)),
        "after": int(len(kept)),
        "dropped": int(max(0, len(rows) - len(kept))),
        "reason_counts": reason_counts,
    }


def _temporal_evidence_quality_diag(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    _, diag = _filter_temporal_evidence_rows(rows, mode)
    before = max(int(diag.get("before", 0)), 1)
    reason_counts = dict(diag.get("reason_counts") or {})
    weak_score = 0.0
    weak_score += 0.40 * float(reason_counts.get("drop_nonimmediate_weak_relation", 0)) / before
    weak_score += 0.55 * float(reason_counts.get("drop_cost_margin_only", 0)) / before
    weak_score += 0.70 * float(reason_counts.get("drop_long_term", 0)) / before
    weak_score += 0.75 * float(reason_counts.get("drop_weak_product_transition", 0)) / before
    weak_score += 0.85 * float(reason_counts.get("drop_weak_unknown_segment", 0)) / before
    weak_score += 0.50 * float(reason_counts.get("drop_nonforecast_weak_forward", 0)) / before
    diag["weak_score"] = float(_clip(weak_score, 0.0, 1.0))
    diag["weak_ratio"] = float(_clip(float(diag.get("dropped", 0)) / before, 0.0, 1.0))
    return diag


def _temporal_evidence_quality_scale(diag: Mapping[str, Any], weight: float) -> float:
    return float(_clip(1.0 - float(_clip(weight, 0.0, 1.0)) * float(_safe_float(diag.get("weak_score"), 0.0)), 0.0, 1.0))


def _temporal_guidance_bucket_scale(
    *,
    guidance_bucket: str,
    mode: str,
    explicit_scale: float,
    non_explicit_scale: float,
    no_guidance_scale: float,
) -> float:
    """Default-off diagnostic shrink for Temporal Direct by guidance bucket.

    Under partitioned Intrinsic residuals, changing Temporal also changes the
    no-Intrinsic residual target that later rows can learn from. This is
    intentional for same-contract diagnostics, but it is not a pure row-local
    Temporal ablation.
    """
    if str(mode) == "off":
        return 1.0
    if str(mode) != "fixed_v1":
        raise ValueError(f"Unsupported temporal_guidance_bucket_scale_mode: {mode}")
    if str(guidance_bucket) == "explicit_numeric":
        scale = float(explicit_scale)
    elif str(guidance_bucket) == "none":
        scale = float(no_guidance_scale)
    else:
        scale = float(non_explicit_scale)
    return float(_clip(scale, 0.0, 1.0))


def _temporal_interaction_guard_scale(
    *,
    guidance_bucket: str,
    temporal_delta: float,
    temporal_support: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    base_delta: float,
    base_support: float,
    guidance_expert_active: bool,
    mode: str,
    duplicate_threshold: float,
    duplicate_scale: float,
    explicit_scale: float,
    non_explicit_scale: float,
    guidance_active_scale: float,
    conflict_scale: float,
    min_support: float,
) -> Dict[str, Any]:
    mode = str(mode or "off")
    sign_match = _expert_sign_match_score(float(intrinsic_delta), float(temporal_delta))
    duplicate_ratio = _expert_duplicate_ratio(float(intrinsic_delta), float(temporal_delta))
    temporal_intrinsic_conflict = bool(
        sign_match <= 0.0
        and float(temporal_support) > 0.10
        and float(intrinsic_support) > 0.10
    )
    temporal_base_conflict = bool(
        _sign_num(float(temporal_delta)) != 0.0
        and _sign_num(float(base_delta)) != 0.0
        and _sign_num(float(temporal_delta)) != _sign_num(float(base_delta))
        and float(temporal_support) > 0.10
        and float(base_support) > 0.10
    )
    out = {
        "mode": mode,
        "scale": 1.0,
        "reason": "off",
        "guidance_scale": 1.0,
        "duplicate_scale": 1.0,
        "guidance_active_scale": 1.0,
        "conflict_scale": 1.0,
        "min_support_scale": 1.0,
        "sign_match": float(sign_match),
        "duplicate_ratio": float(duplicate_ratio),
        "temporal_intrinsic_conflict": int(temporal_intrinsic_conflict),
        "temporal_base_conflict": int(temporal_base_conflict),
        "guidance_expert_active": int(bool(guidance_expert_active)),
    }
    if mode == "off":
        return out
    valid_modes = {"duplicate_only_v0", "guidance_bucket_only_v0", "duplicate_plus_guidance_v0", "full_v0"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported temporal_interaction_guard_mode: {mode}")

    scale = 1.0
    reasons: List[str] = []
    if mode in {"guidance_bucket_only_v0", "duplicate_plus_guidance_v0", "full_v0"}:
        if str(guidance_bucket) == "explicit_numeric":
            guidance_scale = float(_clip(explicit_scale, 0.0, 1.0))
            reasons.append("explicit_guidance_shrink")
        elif str(guidance_bucket) != "none":
            guidance_scale = float(_clip(non_explicit_scale, 0.0, 1.0))
            reasons.append("non_explicit_guidance_shrink")
        else:
            guidance_scale = 1.0
        scale *= guidance_scale
        out["guidance_scale"] = float(guidance_scale)

    if mode in {"duplicate_only_v0", "duplicate_plus_guidance_v0", "full_v0"}:
        if float(duplicate_ratio) >= float(duplicate_threshold):
            dup_scale = float(_clip(duplicate_scale, 0.0, 1.0))
            scale *= dup_scale
            out["duplicate_scale"] = float(dup_scale)
            reasons.append("duplicate_temporal_intrinsic_shrink")

    if mode == "full_v0":
        if bool(guidance_expert_active):
            active_scale = float(_clip(guidance_active_scale, 0.0, 1.0))
            scale *= active_scale
            out["guidance_active_scale"] = float(active_scale)
            reasons.append("guidance_expert_active_shrink")
        if temporal_intrinsic_conflict:
            conflict_support_scale = float(_clip(conflict_scale, 0.0, 1.0))
            scale *= conflict_support_scale
            out["conflict_scale"] = float(conflict_support_scale)
            reasons.append("temporal_intrinsic_conflict_shrink")
        if float(temporal_support) < float(min_support):
            scale = 0.0
            out["min_support_scale"] = 0.0
            reasons.append("temporal_support_below_min_abstain")

    out["scale"] = float(_clip(scale, 0.0, 1.0))
    out["reason"] = "+".join(reasons) if reasons else "no_interaction_shrink"
    return out


def _predict_temporal_graph_attention_direct_expert(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    current_rows: Sequence[Mapping[str, Any]],
    current_quarter: str,
    target_col: str,
    min_train: int,
    history_cap: int,
    quarter_top_k: int,
    quarter_temperature: float,
    item_top_k: int,
    item_temperature: float,
    same_quarter_bonus: float,
    time_decay_quarters: float,
    var_tau: float,
    neff_scale: float,
    directional_consistency_power: float,
    attention_focus_power: float,
    max_abs_log_delta: float,
    score_mode: str,
    support_mode: str,
    direction_mode: str,
    current_context_rows: Sequence[Mapping[str, Any]] = (),
    current_state: Mapping[str, Any] = {},
    history_rows_field: str = "forward_rows",
    context_rows_field: str = "context_rows",
    context_score_weight: float = 0.0,
    reliability_mode: str = "off",
    segment_score_weight: float = 0.0,
    context_support_weight: float = 0.0,
    novelty_shrink_weight: float = 0.0,
    guidance_trust_weight: float = 0.0,
    segment_support_weight: float = 0.0,
    soft_agreement_weight: float = 0.0,
) -> Dict[str, Any]:
    out = {
        "pred": 0.0,
        "pred_raw": float("nan"),
        "train_count": 0,
        "support": 0.0,
        "support_pre_direction": 0.0,
        "direction_scale": 1.0,
        "effective_memory_count": 0.0,
        "directional_consistency": 0.0,
        "attention_focus": 0.0,
        "mean_direction_alignment": 0.0,
        "top_direction_alignment": 0.0,
        "pred_post_support": 0.0,
        "context_score_weight": 0.0,
        "context_attention_score": 0.0,
        "context_attention_focus": 0.0,
        "reliability_mode": "off",
        "segment_compatibility": 1.0,
        "segment_support_scale": 1.0,
        "context_support_scale": 1.0,
        "novelty_value": 0.0,
        "novelty_compatibility": 1.0,
        "novelty_shrink_scale": 1.0,
        "guidance_quality": 0.0,
        "guidance_trust_scale": 1.0,
        "typed_reliability_scale": 1.0,
        "context_pred_raw": float("nan"),
        "context_support_proxy": 0.0,
        "context_directional_consistency": 0.0,
        "context_magnitude_alignment": 0.0,
        "soft_state_compatibility": 1.0,
        "soft_agreement_value": 0.0,
        "soft_agreement_scale": 1.0,
        "retrieved_quarters": [],
        "top_matches": [],
        "context_top_matches": [],
    }
    if not current_rows:
        return out

    valid_history: List[Dict[str, Any]] = []
    for row in train_rows:
        target_value = float(_safe_float(row.get(target_col), float("nan")))
        if not np.isfinite(target_value):
            continue
        history_rows = list(row.get(history_rows_field) or [])
        if not history_rows:
            continue
        valid_history.append(dict(row))
    out["train_count"] = int(len(valid_history))
    if len(valid_history) < int(min_train):
        return out

    query_record = _native_rows_to_attention_record(current_rows)
    context_weight = float(_clip(_safe_float(context_score_weight, 0.0), 0.0, 1.0))
    context_query_record = _native_rows_to_attention_record(current_context_rows) if context_weight > 0.0 and current_context_rows else {}
    reliability_active = str(reliability_mode) in {"typed_reliability", "typed_soft_agreement"}
    soft_agreement_active = str(reliability_mode) == "typed_soft_agreement"
    current_state_dict = dict(current_state or {})
    current_novelty = _temporal_novelty_value(current_state_dict) if reliability_active else 0.0
    current_guidance_quality = _guidance_quality_value(current_state_dict) if reliability_active else 0.0
    history_slice = valid_history[-int(history_cap) :] if int(history_cap) > 0 else list(valid_history)
    current_q_num = _quarter_number(str(current_quarter or ""))
    raw_pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any], float, float, Dict[str, Any], float, float, float]] = []
    for past_row in history_slice:
        past_quarter = str(past_row.get("quarter") or "")
        past_rows = list(past_row.get(history_rows_field) or [])
        attention = cross_card_attention(
            query_record,
            _native_rows_to_attention_record(past_rows),
            top_k_per_query=int(item_top_k),
            temperature=float(item_temperature),
            top_k_overall=6,
        )
        past_q_num = _quarter_number(past_quarter)
        quarter_distance = max(current_q_num - past_q_num, 0)
        recency = float(np.exp(-float(quarter_distance) / max(float(time_decay_quarters), EPS)))
        same_q = float(same_quarter_bonus) * _same_fiscal_quarter_bonus(current_quarter, past_quarter)
        attention_score = float(attention.get("attention_score", 0.0))
        if str(score_mode) == "attention_only":
            score = attention_score
            context_component_scale = 1.0
        elif str(score_mode) == "attention_plus_recency":
            score = attention_score * recency
            context_component_scale = recency
        else:
            score = attention_score * recency + same_q
            context_component_scale = recency
        context_attention = {
            "attention_score": 0.0,
            "attention_focus": 0.0,
            "direction_alignment": 0.0,
            "query_count": 0,
            "matched_query_count": 0,
            "top_matches": [],
        }
        context_component = 0.0
        past_context_rows = list(past_row.get(context_rows_field) or [])
        if context_query_record:
            if past_context_rows:
                context_attention = cross_card_attention(
                    context_query_record,
                    _native_rows_to_attention_record(past_context_rows),
                    top_k_per_query=int(item_top_k),
                    temperature=float(item_temperature),
                    top_k_overall=6,
                )
                context_component = float(context_attention.get("attention_score", 0.0)) * float(context_component_scale)
                score += float(context_weight) * float(context_component)
        segment_compat = 1.0
        novelty_compat = 1.0
        if reliability_active:
            segment_compat = _segment_scale_compatibility(current_state_dict, past_row, current_context_rows, past_context_rows)
            novelty_compat = float(np.exp(-2.0 * abs(current_novelty - _temporal_novelty_value(past_row))))
            score += float(_clip(segment_score_weight, 0.0, 1.0)) * float(segment_compat) * float(recency)
        raw_pairs.append((score, dict(past_row), attention, recency, same_q, context_attention, context_component, segment_compat, novelty_compat))
    raw_pairs.sort(key=lambda item: item[0], reverse=True)
    top_pairs = raw_pairs[: int(quarter_top_k)]
    if not top_pairs:
        return out

    weights = softmax_weights([item[0] for item in top_pairs], float(quarter_temperature))
    target_logs = [float(_safe_float(item[1].get(target_col), 0.0)) for item in top_pairs]
    pred_mean = float(sum(weight * target for weight, target in zip(weights, target_logs)))
    pred_var = float(sum(weight * (target - pred_mean) ** 2 for weight, target in zip(weights, target_logs)))
    weight_sq_sum = float(sum(weight ** 2 for weight in weights))
    effective_memory_count = 0.0 if weight_sq_sum <= EPS else 1.0 / weight_sq_sum
    abs_mass = float(sum(weight * abs(target) for weight, target in zip(weights, target_logs)))
    directional_consistency = 0.0 if abs_mass <= EPS else abs(pred_mean) / abs_mass
    attention_focus = float(sum(weight * float(item[2].get("attention_focus", 0.0)) for weight, item in zip(weights, top_pairs)))
    mean_direction_alignment = float(sum(weight * float(item[2].get("direction_alignment", 0.0)) for weight, item in zip(weights, top_pairs)))
    top_direction_alignment = float(_safe_float(top_pairs[0][2].get("direction_alignment"), 0.0))
    context_attention_score = float(sum(weight * float(item[5].get("attention_score", 0.0)) for weight, item in zip(weights, top_pairs)))
    context_attention_focus = float(sum(weight * float(item[5].get("attention_focus", 0.0)) for weight, item in zip(weights, top_pairs)))
    segment_compatibility = float(sum(weight * float(item[7]) for weight, item in zip(weights, top_pairs)))
    novelty_compatibility = float(sum(weight * float(item[8]) for weight, item in zip(weights, top_pairs)))
    context_pred_mean = float("nan")
    context_support_proxy = 0.0
    context_directional_consistency = 0.0
    context_magnitude_alignment = 0.0
    soft_state_compatibility = 1.0
    soft_agreement_value = 0.0
    soft_agreement_scale = 1.0
    if soft_agreement_active and context_query_record:
        context_pairs = [item for item in raw_pairs if float(item[6]) > EPS]
        context_pairs.sort(key=lambda item: float(item[6]) + float(item[4]), reverse=True)
        context_top_pairs = context_pairs[: int(quarter_top_k)]
        if context_top_pairs:
            context_scores = [float(item[6]) + float(item[4]) for item in context_top_pairs]
            context_weights = softmax_weights(context_scores, float(quarter_temperature))
            context_targets = [float(_safe_float(item[1].get(target_col), 0.0)) for item in context_top_pairs]
            context_pred_mean = float(sum(weight * target for weight, target in zip(context_weights, context_targets)))
            context_var = float(sum(weight * (target - context_pred_mean) ** 2 for weight, target in zip(context_weights, context_targets)))
            context_weight_sq_sum = float(sum(weight ** 2 for weight in context_weights))
            context_neff = 0.0 if context_weight_sq_sum <= EPS else 1.0 / context_weight_sq_sum
            context_abs_mass = float(sum(weight * abs(target) for weight, target in zip(context_weights, context_targets)))
            context_directional_consistency = 0.0 if context_abs_mass <= EPS else abs(context_pred_mean) / context_abs_mass
            context_focus = float(sum(weight * float(item[5].get("attention_focus", 0.0)) for weight, item in zip(context_weights, context_top_pairs)))
            context_support_proxy = float(np.exp(-context_var / max(float(var_tau), EPS)))
            context_support_proxy *= min(1.0, context_neff / max(float(neff_scale), 1.0))
            if str(support_mode) in {"var_neff_focus", "current"} and float(attention_focus_power) > 0.0:
                context_support_proxy *= max(context_focus, 0.0) ** float(attention_focus_power)
            if str(support_mode) == "current" and float(directional_consistency_power) > 0.0:
                context_support_proxy *= max(context_directional_consistency, 0.0) ** float(directional_consistency_power)
            context_support_proxy = float(_clip(context_support_proxy, 0.0, 1.0))
            if abs(pred_mean) > EPS and abs(context_pred_mean) > EPS:
                context_magnitude_alignment = float(min(abs(pred_mean), abs(context_pred_mean)) / max(abs(pred_mean), abs(context_pred_mean)))
                agreement_sign = float(_sign_num(pred_mean) * _sign_num(context_pred_mean))
                soft_state_compatibility = float(
                    _clip(
                        np.mean(
                            [
                                context_support_proxy,
                                segment_compatibility,
                                novelty_compatibility,
                                1.0 - current_guidance_quality,
                            ]
                        ),
                        0.0,
                        1.0,
                    )
                )
                soft_agreement_value = float(agreement_sign * context_support_proxy * context_magnitude_alignment * soft_state_compatibility)
                weight = float(_clip(soft_agreement_weight, 0.0, 0.5))
                soft_agreement_scale = float(_clip(1.0 + weight * soft_agreement_value, 1.0 - weight, 1.0 + weight))
    support = 1.0
    if str(support_mode) != "none":
        support = float(np.exp(-pred_var / max(float(var_tau), EPS)))
        support *= min(1.0, effective_memory_count / max(float(neff_scale), 1.0))
        if str(support_mode) in {"var_neff_focus", "current"} and float(attention_focus_power) > 0.0:
            support *= max(attention_focus, 0.0) ** float(attention_focus_power)
        if str(support_mode) == "current" and float(directional_consistency_power) > 0.0:
            support *= max(directional_consistency, 0.0) ** float(directional_consistency_power)
    support = float(_clip(support, 0.0, 1.0))
    context_support_scale = 1.0
    segment_support_scale = 1.0
    novelty_shrink_scale = 1.0
    guidance_trust_scale = 1.0
    typed_reliability_scale = 1.0
    if str(reliability_mode) == "typed_reliability":
        if context_query_record:
            context_support_scale = float(_clip(1.0 - float(_clip(context_support_weight, 0.0, 1.0)) * (1.0 - context_attention_score), 0.0, 1.0))
        segment_support_scale = float(_clip(1.0 - float(_clip(segment_support_weight, 0.0, 1.0)) * (1.0 - segment_compatibility), 0.0, 1.0))
        novelty_shrink_scale = float(_clip(1.0 - float(_clip(novelty_shrink_weight, 0.0, 1.0)) * current_novelty * (1.0 - novelty_compatibility), 0.0, 1.0))
        guidance_trust_scale = float(_clip(1.0 - float(_clip(guidance_trust_weight, 0.0, 1.0)) * current_guidance_quality, 0.0, 1.0))
        typed_reliability_scale = float(_clip(context_support_scale * segment_support_scale * novelty_shrink_scale * guidance_trust_scale, 0.0, 1.0))
        support *= typed_reliability_scale
        support = float(_clip(support, 0.0, 1.0))
    elif soft_agreement_active:
        typed_reliability_scale = float(soft_agreement_scale)
        support *= typed_reliability_scale
        support = float(_clip(support, 0.0, 1.0))
    direction_scale = 1.0
    if str(direction_mode) == "support_scale":
        direction_scale = float(_clip(mean_direction_alignment, 0.0, 1.0))
    elif str(direction_mode) == "min_align":
        direction_scale = float(_clip(min(mean_direction_alignment, top_direction_alignment), 0.0, 1.0))
    pred_post_support = float(_clip(support * pred_mean, -float(max_abs_log_delta), float(max_abs_log_delta)))
    support_final = float(_clip(support * direction_scale, 0.0, 1.0))
    pred = float(_clip(support_final * pred_mean, -float(max_abs_log_delta), float(max_abs_log_delta)))
    out.update(
        {
            "pred": pred,
            "pred_raw": float(pred_mean),
            "support": float(support_final),
            "support_pre_direction": float(support),
            "direction_scale": float(direction_scale),
            "effective_memory_count": float(effective_memory_count),
            "directional_consistency": float(directional_consistency),
            "attention_focus": float(attention_focus),
            "mean_direction_alignment": float(mean_direction_alignment),
            "top_direction_alignment": float(top_direction_alignment),
            "pred_post_support": float(pred_post_support),
            "context_score_weight": float(context_weight),
            "context_attention_score": float(context_attention_score),
            "context_attention_focus": float(context_attention_focus),
            "reliability_mode": str(reliability_mode),
            "segment_compatibility": float(segment_compatibility),
            "segment_support_scale": float(segment_support_scale),
            "context_support_scale": float(context_support_scale),
            "novelty_value": float(current_novelty),
            "novelty_compatibility": float(novelty_compatibility),
            "novelty_shrink_scale": float(novelty_shrink_scale),
            "guidance_quality": float(current_guidance_quality),
            "guidance_trust_scale": float(guidance_trust_scale),
            "typed_reliability_scale": float(typed_reliability_scale),
            "context_pred_raw": float(context_pred_mean),
            "context_support_proxy": float(context_support_proxy),
            "context_directional_consistency": float(context_directional_consistency),
            "context_magnitude_alignment": float(context_magnitude_alignment),
            "soft_state_compatibility": float(soft_state_compatibility),
            "soft_agreement_value": float(soft_agreement_value),
            "soft_agreement_scale": float(soft_agreement_scale),
            "retrieved_quarters": [
                {
                    "quarter": str(item[1].get("quarter") or ""),
                    "weight": round(float(weights[idx]), 6),
                    "score": round(float(item[0]), 6),
                    "target_value": round(float(_safe_float(item[1].get(target_col), 0.0)), 6),
                    "attention_score": round(float(item[2].get("attention_score", 0.0)), 6),
                    "attention_focus": round(float(item[2].get("attention_focus", 0.0)), 6),
                    "direction_alignment": round(float(item[2].get("direction_alignment", 0.0)), 6),
                    "context_attention_score": round(float(item[5].get("attention_score", 0.0)), 6),
                    "context_attention_focus": round(float(item[5].get("attention_focus", 0.0)), 6),
                    "context_score_component": round(float(item[6]), 6),
                    "segment_compatibility": round(float(item[7]), 6),
                    "novelty_compatibility": round(float(item[8]), 6),
                    "recency": round(float(item[3]), 6),
                    "same_quarter_bonus": round(float(item[4]), 6),
                    "target_sign": int(_sign_num(_safe_float(item[1].get(target_col), 0.0))),
                }
                for idx, item in enumerate(top_pairs)
            ],
            "top_matches": sanitize_for_json(top_pairs[0][2].get("top_matches", [])) if top_pairs else [],
            "context_top_matches": sanitize_for_json(top_pairs[0][5].get("top_matches", [])) if top_pairs else [],
        }
    )
    return out


def _expert_sign_match_score(intrinsic_delta: float, temporal_delta: float) -> float:
    intrinsic_sign = _sign_num(float(intrinsic_delta))
    temporal_sign = _sign_num(float(temporal_delta))
    if intrinsic_sign == 0.0 or temporal_sign == 0.0:
        return 0.5
    return 1.0 if intrinsic_sign == temporal_sign else 0.0


def _expert_duplicate_ratio(intrinsic_delta: float, temporal_delta: float) -> float:
    intrinsic_sign = _sign_num(float(intrinsic_delta))
    temporal_sign = _sign_num(float(temporal_delta))
    if intrinsic_sign == 0.0 or temporal_sign == 0.0 or intrinsic_sign != temporal_sign:
        return 0.0
    intrinsic_abs = abs(float(intrinsic_delta))
    temporal_abs = abs(float(temporal_delta))
    return float(min(intrinsic_abs, temporal_abs) / max(max(intrinsic_abs, temporal_abs), EPS))


def _local_smape(actual: float, pred: float) -> float:
    actual = float(actual)
    pred = float(pred)
    denom = abs(actual) + abs(pred)
    if denom <= EPS:
        return 0.0
    return float(2.0 * abs(pred - actual) / denom)


def _intrinsic_reliability_guidance_bucket(row: Mapping[str, Any]) -> str:
    bucket = _row_guidance_bucket(row)
    if bucket == "explicit_numeric":
        return "explicit_numeric"
    if bucket == "none":
        return "none"
    return "non_explicit_guidance"


def _intrinsic_reliability_sign_bucket(intrinsic_delta: float, temporal_delta: float) -> str:
    sign_match = _expert_sign_match_score(float(intrinsic_delta), float(temporal_delta))
    if sign_match >= 1.0:
        return "agree_temporal"
    if sign_match <= 0.0:
        return "oppose_temporal"
    return "temporal_abstain"


def _intrinsic_reliability_bucket(
    *,
    row: Mapping[str, Any],
    intrinsic_delta: float,
    temporal_delta: float,
    conflict_ratio: float,
    guidance_expert_active: bool,
) -> str:
    guidance_bucket = _intrinsic_reliability_guidance_bucket(row)
    sign_bucket = _intrinsic_reliability_sign_bucket(float(intrinsic_delta), float(temporal_delta))
    conflict_bucket = "high_conflict" if float(conflict_ratio) >= 0.25 else "low_conflict"
    guidance_expert_bucket = "guidance_expert_active" if guidance_expert_active else "guidance_expert_inactive"
    return "||".join([guidance_bucket, sign_bucket, conflict_bucket, guidance_expert_bucket])


def _intrinsic_history_reliability_scale(
    history: Sequence[Mapping[str, Any]],
    *,
    row: Mapping[str, Any],
    intrinsic_delta: float,
    temporal_delta: float,
    conflict_ratio: float,
    guidance_expert_active: bool,
    mode: str,
    min_history: int,
    tau: float,
    min_scale: float,
    max_scale: float,
) -> Dict[str, Any]:
    bucket = _intrinsic_reliability_bucket(
        row=row,
        intrinsic_delta=float(intrinsic_delta),
        temporal_delta=float(temporal_delta),
        conflict_ratio=float(conflict_ratio),
        guidance_expert_active=bool(guidance_expert_active),
    )
    guidance_bucket = _intrinsic_reliability_guidance_bucket(row)
    if str(mode) == "off":
        return {
            "scale": 1.0,
            "mode": "off",
            "bucket": bucket,
            "source": "off",
            "history_n": 0,
            "effect_mean": float("nan"),
            "win_rate": float("nan"),
        }
    if str(mode) != "history_bucket_v1":
        raise ValueError(f"Unsupported intrinsic_reliability_mode: {mode}")

    usable: List[Mapping[str, Any]] = []
    for item in history:
        effect = _safe_float(item.get("intrinsic_raw_marginal_smape_effect"), float("nan"))
        if not np.isfinite(effect):
            continue
        usable.append(item)

    exact = [item for item in usable if str(item.get("intrinsic_reliability_bucket") or "") == bucket]
    source = "exact_bucket"
    selected = exact
    if len(selected) < int(min_history):
        selected = [
            item for item in usable
            if str(item.get("intrinsic_reliability_guidance_bucket") or "") == guidance_bucket
        ]
        source = "guidance_bucket_fallback"
    if len(selected) < int(min_history):
        selected = usable
        source = "all_history_fallback"
    if len(selected) < int(min_history):
        return {
            "scale": 1.0,
            "mode": str(mode),
            "bucket": bucket,
            "source": "insufficient_history",
            "history_n": int(len(selected)),
            "effect_mean": float("nan"),
            "win_rate": float("nan"),
        }

    effects = np.asarray([_safe_float(item.get("intrinsic_raw_marginal_smape_effect"), float("nan")) for item in selected], dtype=float)
    effects = effects[np.isfinite(effects)]
    if effects.size < int(min_history):
        return {
            "scale": 1.0,
            "mode": str(mode),
            "bucket": bucket,
            "source": "insufficient_finite_history",
            "history_n": int(effects.size),
            "effect_mean": float("nan"),
            "win_rate": float("nan"),
        }
    effect_mean = float(np.mean(effects))
    win_rate = float(np.mean(effects > 0.0))
    effect_score = float(np.tanh(effect_mean / max(float(tau), EPS)))
    win_score = float(2.0 * win_rate - 1.0)
    trust_score = float(_clip(0.65 * effect_score + 0.35 * win_score, -1.0, 1.0))
    if trust_score >= 0.0:
        scale = 1.0 + trust_score * (float(max_scale) - 1.0)
    else:
        scale = 1.0 + trust_score * (1.0 - float(min_scale))
    return {
        "scale": float(_clip(scale, float(min_scale), float(max_scale))),
        "mode": str(mode),
        "bucket": bucket,
        "source": source,
        "history_n": int(effects.size),
        "effect_mean": effect_mean,
        "win_rate": win_rate,
        "trust_score": trust_score,
    }


def _temporal_reliability_guidance_bucket(row: Mapping[str, Any]) -> str:
    bucket = _row_guidance_bucket(row)
    if bucket == "explicit_numeric":
        return "explicit_numeric"
    if bucket == "none":
        return "none"
    return "non_explicit_guidance"


def _temporal_reliability_support_bucket(temporal_support: float, *, medium_threshold: float, high_threshold: float) -> str:
    support = float(_safe_float(temporal_support, 0.0))
    if support >= float(high_threshold):
        return "high_support"
    if support >= float(medium_threshold):
        return "medium_support"
    return "low_support"


def _temporal_reliability_conflict_bucket(
    *,
    temporal_delta: float,
    temporal_support: float,
    intrinsic_delta: float,
    intrinsic_support: float,
) -> str:
    sign_match = _expert_sign_match_score(float(intrinsic_delta), float(temporal_delta))
    if (
        sign_match <= 0.0
        and float(temporal_support) > 0.10
        and float(intrinsic_support) > 0.10
    ):
        return "intrinsic_conflict"
    if sign_match >= 1.0:
        return "intrinsic_agree"
    return "intrinsic_weak_or_abstain"


def _temporal_reliability_bucket(
    *,
    row: Mapping[str, Any],
    temporal_delta: float,
    temporal_support: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    duplicate_threshold: float,
    support_medium_threshold: float,
    support_high_threshold: float,
) -> Dict[str, Any]:
    guidance_bucket = _temporal_reliability_guidance_bucket(row)
    duplicate_ratio = _expert_duplicate_ratio(float(intrinsic_delta), float(temporal_delta))
    duplicate_bucket = "duplicate_high" if duplicate_ratio >= float(duplicate_threshold) else "duplicate_low"
    conflict_bucket = _temporal_reliability_conflict_bucket(
        temporal_delta=float(temporal_delta),
        temporal_support=float(temporal_support),
        intrinsic_delta=float(intrinsic_delta),
        intrinsic_support=float(intrinsic_support),
    )
    support_bucket = _temporal_reliability_support_bucket(
        float(temporal_support),
        medium_threshold=float(support_medium_threshold),
        high_threshold=float(support_high_threshold),
    )
    bucket = "||".join([guidance_bucket, duplicate_bucket, conflict_bucket, support_bucket])
    return {
        "bucket": bucket,
        "guidance_bucket": guidance_bucket,
        "duplicate_bucket": duplicate_bucket,
        "conflict_bucket": conflict_bucket,
        "support_bucket": support_bucket,
        "duplicate_ratio": float(duplicate_ratio),
    }


def _temporal_history_reliability_scale(
    history: Sequence[Mapping[str, Any]],
    *,
    row: Mapping[str, Any],
    temporal_delta: float,
    temporal_support: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    mode: str,
    min_history: int,
    tau: float,
    min_scale: float,
    duplicate_threshold: float,
    support_medium_threshold: float,
    support_high_threshold: float,
    abstain_trust_threshold: float,
) -> Dict[str, Any]:
    bucket_diag = _temporal_reliability_bucket(
        row=row,
        temporal_delta=float(temporal_delta),
        temporal_support=float(temporal_support),
        intrinsic_delta=float(intrinsic_delta),
        intrinsic_support=float(intrinsic_support),
        duplicate_threshold=float(duplicate_threshold),
        support_medium_threshold=float(support_medium_threshold),
        support_high_threshold=float(support_high_threshold),
    )
    base_out = {
        "scale": 1.0,
        "mode": str(mode or "off"),
        "bucket": str(bucket_diag["bucket"]),
        "guidance_bucket": str(bucket_diag["guidance_bucket"]),
        "duplicate_bucket": str(bucket_diag["duplicate_bucket"]),
        "conflict_bucket": str(bucket_diag["conflict_bucket"]),
        "support_bucket": str(bucket_diag["support_bucket"]),
        "duplicate_ratio": float(bucket_diag["duplicate_ratio"]),
        "source": "off",
        "history_n": 0,
        "effect_mean": float("nan"),
        "win_rate": float("nan"),
        "trust_score": float("nan"),
    }
    if str(mode) == "off":
        return base_out
    valid_modes = {"history_bucket_shrink_v0", "history_bucket_abstain_v0"}
    if str(mode) not in valid_modes:
        raise ValueError(f"Unsupported temporal_reliability_memory_mode: {mode}")

    usable: List[Mapping[str, Any]] = []
    for item in history:
        effect = _safe_float(item.get("temporal_marginal_smape_effect"), float("nan"))
        if not np.isfinite(effect):
            continue
        usable.append(item)

    selected = [item for item in usable if str(item.get("temporal_reliability_bucket") or "") == str(bucket_diag["bucket"])]
    source = "exact_bucket"
    if len(selected) < int(min_history):
        selected = [
            item for item in usable
            if str(item.get("temporal_reliability_guidance_bucket") or "") == str(bucket_diag["guidance_bucket"])
            and str(item.get("temporal_reliability_conflict_bucket") or "") == str(bucket_diag["conflict_bucket"])
        ]
        source = "guidance_conflict_fallback"
    if len(selected) < int(min_history):
        selected = [
            item for item in usable
            if str(item.get("temporal_reliability_guidance_bucket") or "") == str(bucket_diag["guidance_bucket"])
        ]
        source = "guidance_bucket_fallback"
    if len(selected) < int(min_history):
        selected = usable
        source = "all_history_fallback"
    if len(selected) < int(min_history):
        return {
            **base_out,
            "mode": str(mode),
            "source": "insufficient_history",
            "history_n": int(len(selected)),
        }

    effects = np.asarray([_safe_float(item.get("temporal_marginal_smape_effect"), float("nan")) for item in selected], dtype=float)
    effects = effects[np.isfinite(effects)]
    if effects.size < int(min_history):
        return {
            **base_out,
            "mode": str(mode),
            "source": "insufficient_finite_history",
            "history_n": int(effects.size),
        }

    effect_mean = float(np.mean(effects))
    win_rate = float(np.mean(effects > 0.0))
    effect_score = float(np.tanh(effect_mean / max(float(tau), EPS)))
    win_score = float(2.0 * win_rate - 1.0)
    trust_score = float(_clip(0.65 * effect_score + 0.35 * win_score, -1.0, 1.0))
    scale = 1.0
    reason = "prior_temporal_nonnegative_noop"
    if trust_score < 0.0:
        scale = float(1.0 + trust_score * (1.0 - float(min_scale)))
        reason = "prior_temporal_negative_shrink"
    if str(mode) == "history_bucket_abstain_v0" and trust_score <= float(abstain_trust_threshold) and effect_mean < 0.0:
        scale = 0.0
        reason = "prior_temporal_negative_abstain"
    return {
        **base_out,
        "mode": str(mode),
        "scale": float(_clip(scale, 0.0, 1.0)),
        "source": source,
        "history_n": int(effects.size),
        "effect_mean": effect_mean,
        "win_rate": win_rate,
        "trust_score": trust_score,
        "reason": reason,
    }


def _temporal_action_gate_feature_map(
    *,
    guidance_bucket: str,
    temporal_delta: float,
    temporal_support: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    base_delta: float,
    base_support: float,
    guidance_expert_active: bool,
    guidance_lock: float,
    anchor_uncertainty: float,
    internal_strength: float,
    forward_conflict_ratio: float,
    memory_support: float,
    memory_consistency: float,
    temporal_result: Mapping[str, Any],
    temporal_interaction_diag: Mapping[str, Any],
    temporal_context_quality_scale: float,
    temporal_context_quality_weak_score: float,
    temporal_reliability_scale: float,
    temporal_reliability_trust_score: float,
) -> Dict[str, float]:
    guidance_bucket = str(guidance_bucket or "")
    sign_match = float(_safe_float(temporal_interaction_diag.get("sign_match"), _expert_sign_match_score(intrinsic_delta, temporal_delta)))
    duplicate_ratio = float(_safe_float(temporal_interaction_diag.get("duplicate_ratio"), _expert_duplicate_ratio(intrinsic_delta, temporal_delta)))
    return {
        "temporal_action_guidance_explicit": float(1.0 if guidance_bucket == "explicit_numeric" else 0.0),
        "temporal_action_guidance_non_explicit": float(1.0 if guidance_bucket not in {"", "none", "explicit_numeric"} else 0.0),
        "temporal_action_guidance_none": float(1.0 if guidance_bucket == "none" else 0.0),
        "temporal_action_delta": float(_safe_float(temporal_delta, 0.0)),
        "temporal_action_abs_delta": float(abs(_safe_float(temporal_delta, 0.0))),
        "temporal_action_support": float(_safe_float(temporal_support, 0.0)),
        "temporal_action_effective_memory_count": float(_safe_float(temporal_result.get("effective_memory_count"), 0.0)),
        "temporal_action_directional_consistency": float(_safe_float(temporal_result.get("directional_consistency"), 0.0)),
        "temporal_action_attention_focus": float(_safe_float(temporal_result.get("attention_focus"), 0.0)),
        "temporal_action_mean_direction_alignment": float(_safe_float(temporal_result.get("mean_direction_alignment"), 0.0)),
        "temporal_action_top_direction_alignment": float(_safe_float(temporal_result.get("top_direction_alignment"), 0.0)),
        "temporal_action_context_attention_score": float(_safe_float(temporal_result.get("context_attention_score"), 0.0)),
        "temporal_action_context_attention_focus": float(_safe_float(temporal_result.get("context_attention_focus"), 0.0)),
        "temporal_action_context_quality_scale": float(_safe_float(temporal_context_quality_scale, 1.0)),
        "temporal_action_context_quality_weak_score": float(_safe_float(temporal_context_quality_weak_score, 0.0)),
        "temporal_action_intrinsic_delta": float(_safe_float(intrinsic_delta, 0.0)),
        "temporal_action_intrinsic_abs_delta": float(abs(_safe_float(intrinsic_delta, 0.0))),
        "temporal_action_intrinsic_support": float(_safe_float(intrinsic_support, 0.0)),
        "temporal_action_base_delta": float(_safe_float(base_delta, 0.0)),
        "temporal_action_base_abs_delta": float(abs(_safe_float(base_delta, 0.0))),
        "temporal_action_base_support": float(_safe_float(base_support, 0.0)),
        "temporal_action_intrinsic_temporal_sign_match": float(sign_match),
        "temporal_action_intrinsic_temporal_duplicate_ratio": float(duplicate_ratio),
        "temporal_action_intrinsic_conflict": float(_safe_float(temporal_interaction_diag.get("temporal_intrinsic_conflict"), 0.0)),
        "temporal_action_base_conflict": float(_safe_float(temporal_interaction_diag.get("temporal_base_conflict"), 0.0)),
        "temporal_action_guidance_expert_active": float(1.0 if bool(guidance_expert_active) else 0.0),
        "temporal_action_guidance_lock": float(_safe_float(guidance_lock, 0.0)),
        "temporal_action_anchor_uncertainty": float(_safe_float(anchor_uncertainty, 0.0)),
        "temporal_action_internal_strength": float(_safe_float(internal_strength, 0.0)),
        "temporal_action_forward_conflict_ratio": float(_safe_float(forward_conflict_ratio, 0.0)),
        "temporal_action_memory_support": float(_safe_float(memory_support, 0.0)),
        "temporal_action_memory_consistency": float(_safe_float(memory_consistency, 0.0)),
        "temporal_action_reliability_scale": float(_safe_float(temporal_reliability_scale, 1.0)),
        "temporal_action_reliability_trust_score": float(_safe_float(temporal_reliability_trust_score, 0.0)),
    }


def _feature_vector(feature_map: Mapping[str, Any], feature_names: Sequence[str]) -> np.ndarray:
    values: List[float] = []
    for name in feature_names:
        value = _safe_float(feature_map.get(name), 0.0)
        values.append(float(value if np.isfinite(value) else 0.0))
    return np.asarray(values, dtype=float)


def _json_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _load_action_gate_panel_history(path: str, project_root: Path) -> List[Dict[str, Any]]:
    raw_path = str(path or "").strip()
    if not raw_path:
        return []
    csv_path = Path(resolve_repo_path(raw_path, str(project_root)))
    if not csv_path.exists():
        raise FileNotFoundError(f"action_gate_panel_history_csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    history: List[Dict[str, Any]] = []
    for rec in df.to_dict("records"):
        quarter = str(rec.get("quarter") or "")
        if not quarter:
            continue
        item: Dict[str, Any] = {
            "ticker": str(rec.get("ticker") or ""),
            "quarter": quarter,
            "quarter_qnum": int(_quarter_ordinal(quarter)),
        }
        temporal_features = _json_mapping(rec.get("csais_temporal_action_gate_features_json"))
        temporal_effect = _safe_float(rec.get("csais_temporal_marginal_smape_effect"), float("nan"))
        if not np.isfinite(temporal_effect):
            temporal_effect = _safe_float(rec.get("csais_temporal_action_marginal_smape_effect"), float("nan"))
        if temporal_features and np.isfinite(temporal_effect):
            item["temporal_action_gate_features"] = temporal_features
            item["temporal_marginal_smape_effect"] = float(temporal_effect)
        intrinsic_features = _json_mapping(rec.get("csais_intrinsic_action_gate_features_json"))
        intrinsic_effect = _safe_float(rec.get("csais_intrinsic_action_marginal_smape_effect"), float("nan"))
        if intrinsic_features and np.isfinite(intrinsic_effect):
            item["intrinsic_action_gate_features"] = intrinsic_features
            item["intrinsic_action_marginal_smape_effect"] = float(intrinsic_effect)
        if "temporal_action_gate_features" in item or "intrinsic_action_gate_features" in item:
            history.append(item)
    return history


def _prior_history_by_quarter(history: Sequence[Mapping[str, Any]], current_quarter: str) -> List[Mapping[str, Any]]:
    current_qnum = int(_quarter_ordinal(str(current_quarter or "")))
    out: List[Mapping[str, Any]] = []
    for item in history:
        item_qnum = _safe_float(item.get("quarter_qnum"), float("nan"))
        if not np.isfinite(item_qnum):
            item_qnum = _quarter_ordinal(str(item.get("quarter") or ""))
        if np.isfinite(item_qnum) and int(item_qnum) < current_qnum:
            out.append(item)
    return out


def _state_analog_temporal_feature_map(
    *,
    regime: Mapping[str, Any],
    row: Mapping[str, Any],
    current_quarter: str,
    anchor_uncertainty: float,
    internal_strength: float,
) -> Dict[str, float]:
    guidance_bucket = _row_guidance_bucket(row)
    _, fiscal_q = _quarter_key(str(current_quarter or ""))
    return {
        "state_reg_recent_qoq": float(_safe_float(regime.get("reg_recent_qoq"), 0.0)),
        "state_reg_last_yoy": float(_safe_float(regime.get("reg_last_yoy"), 0.0)),
        "state_reg_trend_slope4": float(_safe_float(regime.get("reg_trend_slope4"), 0.0)),
        "state_reg_vol_qoq4": float(_safe_float(regime.get("reg_vol_qoq4"), 0.0)),
        "state_reg_vol_yoy4": float(_safe_float(regime.get("reg_vol_yoy4"), 0.0)),
        "state_reg_same_quarter_support": float(_safe_float(regime.get("reg_same_quarter_support"), 0.0)),
        "state_reg_recent_level_log": float(_safe_float(regime.get("reg_recent_level_log"), 0.0)),
        "state_guidance_explicit": float(1.0 if guidance_bucket == "explicit_numeric" else 0.0),
        "state_guidance_non_explicit": float(1.0 if guidance_bucket not in {"", "none", "explicit_numeric"} else 0.0),
        "state_guidance_none": float(1.0 if guidance_bucket == "none" else 0.0),
        "state_anchor_uncertainty": float(_safe_float(anchor_uncertainty, 0.0)),
        "state_internal_strength": float(_safe_float(internal_strength, 0.0)),
        "state_fq1": float(1.0 if fiscal_q == 1 else 0.0),
        "state_fq2": float(1.0 if fiscal_q == 2 else 0.0),
        "state_fq3": float(1.0 if fiscal_q == 3 else 0.0),
        "state_fq4": float(1.0 if fiscal_q == 4 else 0.0),
    }


def _predict_state_analog_temporal_memory(
    history: Sequence[Mapping[str, Any]],
    *,
    feature_map: Mapping[str, Any],
    current_quarter: str,
    target_col: str,
    mode: str,
    min_history: int,
    neighbor_k: int,
    history_cap: int,
    distance_tau: float,
    var_tau: float,
    neff_scale: float,
    support_scale: float,
    max_abs_log_delta: float,
) -> Dict[str, Any]:
    base_out = {
        "mode": str(mode or "off"),
        "pred": 0.0,
        "pred_raw": float("nan"),
        "support": 0.0,
        "train_count": 0,
        "neighbor_n": 0,
        "effective_memory_count": 0.0,
        "mean_distance": float("nan"),
        "pred_post_support": 0.0,
        "reason": "off",
        "top_matches": [],
    }
    mode = str(mode or "off")
    if mode == "off":
        return base_out
    if mode not in {"state_analog_blend_v0", "state_analog_replace_v0"}:
        raise ValueError(f"Unsupported temporal_state_analog_mode: {mode}")

    usable: List[Tuple[int, np.ndarray, float, Mapping[str, Any]]] = []
    current_qnum = int(_quarter_ordinal(str(current_quarter or "")))
    for item in history:
        item_qnum = _safe_float(item.get("quarter_qnum"), float("nan"))
        if not np.isfinite(item_qnum):
            item_qnum = _quarter_ordinal(str(item.get("quarter") or ""))
        if not np.isfinite(item_qnum) or int(item_qnum) >= current_qnum:
            continue
        target_value = _safe_float(item.get(target_col), float("nan"))
        prior_features = item.get("state_analog_temporal_features")
        if not np.isfinite(target_value) or not isinstance(prior_features, Mapping):
            continue
        usable.append((int(item_qnum), _feature_vector(prior_features, STATE_ANALOG_TEMPORAL_FEATURES), float(target_value), item))
    usable.sort(key=lambda item: item[0])
    if int(history_cap) > 0:
        usable = usable[-int(history_cap) :]
    if len(usable) < int(min_history):
        return {
            **base_out,
            "mode": mode,
            "reason": "insufficient_history_noop",
            "train_count": int(len(usable)),
        }

    x_cur = _feature_vector(feature_map, STATE_ANALOG_TEMPORAL_FEATURES)
    x_hist = np.vstack([item[1] for item in usable])
    targets = np.asarray([item[2] for item in usable], dtype=float)
    scale = np.nanstd(x_hist, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    distances = np.sqrt(np.mean(((x_hist - x_cur) / scale) ** 2, axis=1))
    finite_mask = np.isfinite(distances) & np.isfinite(targets)
    if int(np.sum(finite_mask)) < int(min_history):
        return {
            **base_out,
            "mode": mode,
            "reason": "insufficient_finite_history_noop",
            "train_count": int(np.sum(finite_mask)),
        }
    distances = distances[finite_mask]
    targets = targets[finite_mask]
    usable_finite = [item for item, keep in zip(usable, finite_mask.tolist()) if keep]
    order = np.argsort(distances)
    k = int(max(1, min(int(neighbor_k), len(order))))
    neighbor_idx = order[:k]
    neighbor_dist = distances[neighbor_idx]
    neighbor_targets = targets[neighbor_idx]
    weights = np.exp(-neighbor_dist / max(float(distance_tau), EPS))
    if not np.isfinite(float(np.sum(weights))) or float(np.sum(weights)) <= EPS:
        weights = 1.0 / np.maximum(neighbor_dist, 1e-3)
    weights = weights / max(float(np.sum(weights)), EPS)
    pred_raw = float(np.sum(weights * neighbor_targets))
    pred_var = float(np.sum(weights * (neighbor_targets - pred_raw) ** 2))
    weight_sq_sum = float(np.sum(weights ** 2))
    neff = 0.0 if weight_sq_sum <= EPS else 1.0 / weight_sq_sum
    mean_distance = float(np.mean(neighbor_dist))
    support = float(np.exp(-pred_var / max(float(var_tau), EPS)))
    support *= min(1.0, neff / max(float(neff_scale), 1.0))
    support *= float(np.exp(-mean_distance / max(float(distance_tau), EPS)))
    support = float(_clip(float(support_scale) * support, 0.0, 1.0))
    pred = float(_clip(support * pred_raw, -float(max_abs_log_delta), float(max_abs_log_delta)))
    top_matches = []
    for weight, idx in zip(weights, neighbor_idx):
        source = usable_finite[int(idx)][3]
        top_matches.append(
            {
                "ticker": str(source.get("ticker") or ""),
                "quarter": str(source.get("quarter") or ""),
                "weight": round(float(weight), 6),
                "distance": round(float(distances[int(idx)]), 6),
                "target_value": round(float(targets[int(idx)]), 6),
            }
        )
    return {
        **base_out,
        "mode": mode,
        "pred": pred,
        "pred_raw": pred_raw,
        "support": support,
        "train_count": int(len(usable)),
        "neighbor_n": int(k),
        "effective_memory_count": float(neff),
        "mean_distance": mean_distance,
        "pred_post_support": float(pred),
        "reason": "state_analog_prior_residual",
        "top_matches": sanitize_for_json(top_matches),
    }


def _temporal_action_gate_scale(
    history: Sequence[Mapping[str, Any]],
    *,
    feature_map: Mapping[str, Any],
    mode: str,
    min_history: int,
    neighbor_k: int,
    tau: float,
    min_scale: float,
    abstain_effect_threshold: float,
    abstain_win_rate_threshold: float,
) -> Dict[str, Any]:
    base_out = {
        "mode": str(mode or "off"),
        "scale": 1.0,
        "source": "off",
        "reason": "off",
        "history_n": 0,
        "neighbor_n": 0,
        "pred_effect": float("nan"),
        "neighbor_effect_mean": float("nan"),
        "neighbor_effect_std": float("nan"),
        "win_rate": float("nan"),
        "trust_score": float("nan"),
        "mean_distance": float("nan"),
    }
    mode = str(mode or "off")
    if mode == "off":
        return base_out
    valid_modes = {"knn_shrink_v0", "knn_abstain_v0"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported temporal_action_gate_mode: {mode}")

    usable: List[Tuple[np.ndarray, float]] = []
    for item in history:
        effect = _safe_float(item.get("temporal_marginal_smape_effect"), float("nan"))
        prior_features = item.get("temporal_action_gate_features")
        if not np.isfinite(effect) or not isinstance(prior_features, Mapping):
            continue
        usable.append((_feature_vector(prior_features, TEMPORAL_ACTION_GATE_FEATURES), float(effect)))
    if len(usable) < int(min_history):
        return {
            **base_out,
            "mode": mode,
            "source": "insufficient_history",
            "reason": "insufficient_history_noop",
            "history_n": int(len(usable)),
        }

    x_cur = _feature_vector(feature_map, TEMPORAL_ACTION_GATE_FEATURES)
    x_hist = np.vstack([vec for vec, _ in usable])
    effects = np.asarray([effect for _, effect in usable], dtype=float)
    scale = np.nanstd(x_hist, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    distances = np.sqrt(np.mean(((x_hist - x_cur) / scale) ** 2, axis=1))
    finite_mask = np.isfinite(distances) & np.isfinite(effects)
    if int(np.sum(finite_mask)) < int(min_history):
        return {
            **base_out,
            "mode": mode,
            "source": "insufficient_finite_history",
            "reason": "insufficient_finite_history_noop",
            "history_n": int(np.sum(finite_mask)),
        }
    distances = distances[finite_mask]
    effects = effects[finite_mask]
    order = np.argsort(distances)
    k = int(max(1, min(int(neighbor_k), len(order))))
    neighbor_idx = order[:k]
    neighbor_dist = distances[neighbor_idx]
    neighbor_effects = effects[neighbor_idx]
    weights = 1.0 / np.maximum(neighbor_dist, 1e-3)
    weights = weights / max(float(np.sum(weights)), EPS)
    pred_effect = float(np.sum(weights * neighbor_effects))
    win_rate = float(np.sum(weights * (neighbor_effects > 0.0)))
    effect_score = float(np.tanh(pred_effect / max(float(tau), EPS)))
    win_score = float(2.0 * win_rate - 1.0)
    trust_score = float(_clip(0.70 * effect_score + 0.30 * win_score, -1.0, 1.0))
    scale_value = 1.0
    reason = "prior_action_nonnegative_noop"
    if trust_score < 0.0:
        scale_value = float(1.0 + trust_score * (1.0 - float(min_scale)))
        reason = "prior_action_negative_shrink"
    if (
        mode == "knn_abstain_v0"
        and pred_effect <= float(abstain_effect_threshold)
        and win_rate <= float(abstain_win_rate_threshold)
    ):
        scale_value = 0.0
        reason = "prior_action_negative_abstain"
    return {
        **base_out,
        "mode": mode,
        "scale": float(_clip(scale_value, 0.0, 1.0)),
        "source": "prior_knn_margin",
        "reason": reason,
        "history_n": int(len(usable)),
        "neighbor_n": int(k),
        "pred_effect": pred_effect,
        "neighbor_effect_mean": float(np.mean(neighbor_effects)),
        "neighbor_effect_std": float(np.std(neighbor_effects)),
        "win_rate": win_rate,
        "trust_score": trust_score,
        "mean_distance": float(np.mean(neighbor_dist)),
    }


def _intrinsic_action_gate_feature_map(
    *,
    guidance_bucket: str,
    intrinsic_delta: float,
    intrinsic_support: float,
    intrinsic_residual_candidate_delta: float,
    temporal_delta: float,
    temporal_support: float,
    base_delta: float,
    base_support: float,
    guidance_expert_active: bool,
    guidance_lock: float,
    anchor_uncertainty: float,
    internal_strength: float,
    forward_conflict_ratio: float,
    memory_support: float,
    memory_consistency: float,
    intrinsic_result: Mapping[str, Any],
    intrinsic_reliability_diag: Mapping[str, Any],
    intrinsic_explicit_guidance_scale: float,
    intrinsic_temporal_dedup_diag: Mapping[str, Any],
    intrinsic_temporal_dedup_scale: float,
) -> Dict[str, float]:
    guidance_bucket = str(guidance_bucket or "")
    sign_match = _expert_sign_match_score(float(intrinsic_residual_candidate_delta), float(temporal_delta))
    duplicate_ratio = _expert_duplicate_ratio(float(intrinsic_residual_candidate_delta), float(temporal_delta))
    return {
        "intrinsic_action_guidance_explicit": float(1.0 if guidance_bucket == "explicit_numeric" else 0.0),
        "intrinsic_action_guidance_non_explicit": float(1.0 if guidance_bucket not in {"", "none", "explicit_numeric"} else 0.0),
        "intrinsic_action_guidance_none": float(1.0 if guidance_bucket == "none" else 0.0),
        "intrinsic_action_delta": float(_safe_float(intrinsic_delta, 0.0)),
        "intrinsic_action_abs_delta": float(abs(_safe_float(intrinsic_delta, 0.0))),
        "intrinsic_action_support": float(_safe_float(intrinsic_support, 0.0)),
        "intrinsic_action_residual_candidate_delta": float(_safe_float(intrinsic_residual_candidate_delta, 0.0)),
        "intrinsic_action_abs_residual_candidate_delta": float(abs(_safe_float(intrinsic_residual_candidate_delta, 0.0))),
        "intrinsic_action_train_count": float(_safe_float(intrinsic_result.get("train_count"), 0.0)),
        "intrinsic_action_coverage": float(_safe_float(intrinsic_result.get("coverage"), 0.0)),
        "intrinsic_action_temporal_delta": float(_safe_float(temporal_delta, 0.0)),
        "intrinsic_action_temporal_abs_delta": float(abs(_safe_float(temporal_delta, 0.0))),
        "intrinsic_action_temporal_support": float(_safe_float(temporal_support, 0.0)),
        "intrinsic_action_base_delta": float(_safe_float(base_delta, 0.0)),
        "intrinsic_action_base_abs_delta": float(abs(_safe_float(base_delta, 0.0))),
        "intrinsic_action_base_support": float(_safe_float(base_support, 0.0)),
        "intrinsic_action_temporal_sign_match": float(sign_match),
        "intrinsic_action_temporal_duplicate_ratio": float(duplicate_ratio),
        "intrinsic_action_guidance_expert_active": float(1.0 if bool(guidance_expert_active) else 0.0),
        "intrinsic_action_guidance_lock": float(_safe_float(guidance_lock, 0.0)),
        "intrinsic_action_anchor_uncertainty": float(_safe_float(anchor_uncertainty, 0.0)),
        "intrinsic_action_internal_strength": float(_safe_float(internal_strength, 0.0)),
        "intrinsic_action_forward_conflict_ratio": float(_safe_float(forward_conflict_ratio, 0.0)),
        "intrinsic_action_memory_support": float(_safe_float(memory_support, 0.0)),
        "intrinsic_action_memory_consistency": float(_safe_float(memory_consistency, 0.0)),
        "intrinsic_action_reliability_scale": float(_safe_float(intrinsic_reliability_diag.get("scale"), 1.0)),
        "intrinsic_action_reliability_trust_score": float(_safe_float(intrinsic_reliability_diag.get("trust_score"), 0.0)),
        "intrinsic_action_explicit_guidance_scale": float(_safe_float(intrinsic_explicit_guidance_scale, 1.0)),
        "intrinsic_action_dedup_scale": float(_safe_float(intrinsic_temporal_dedup_scale, 1.0)),
        "intrinsic_action_dedup_duplicate_ratio": float(_safe_float(intrinsic_temporal_dedup_diag.get("duplicate_ratio"), 0.0)),
    }


def _intrinsic_action_gate_scale(
    history: Sequence[Mapping[str, Any]],
    *,
    feature_map: Mapping[str, Any],
    mode: str,
    min_history: int,
    neighbor_k: int,
    tau: float,
    min_scale: float,
    abstain_effect_threshold: float,
    abstain_win_rate_threshold: float,
) -> Dict[str, Any]:
    base_out = {
        "mode": str(mode or "off"),
        "scale": 1.0,
        "source": "off",
        "reason": "off",
        "history_n": 0,
        "neighbor_n": 0,
        "pred_effect": float("nan"),
        "neighbor_effect_mean": float("nan"),
        "neighbor_effect_std": float("nan"),
        "win_rate": float("nan"),
        "trust_score": float("nan"),
        "mean_distance": float("nan"),
    }
    mode = str(mode or "off")
    if mode == "off":
        return base_out
    valid_modes = {"knn_shrink_v0", "knn_abstain_v0"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported intrinsic_action_gate_mode: {mode}")

    usable: List[Tuple[np.ndarray, float]] = []
    for item in history:
        effect = _safe_float(item.get("intrinsic_action_marginal_smape_effect"), float("nan"))
        prior_features = item.get("intrinsic_action_gate_features")
        if not np.isfinite(effect) or not isinstance(prior_features, Mapping):
            continue
        usable.append((_feature_vector(prior_features, INTRINSIC_ACTION_GATE_FEATURES), float(effect)))
    if len(usable) < int(min_history):
        return {
            **base_out,
            "mode": mode,
            "source": "insufficient_history",
            "reason": "insufficient_history_noop",
            "history_n": int(len(usable)),
        }

    x_cur = _feature_vector(feature_map, INTRINSIC_ACTION_GATE_FEATURES)
    x_hist = np.vstack([vec for vec, _ in usable])
    effects = np.asarray([effect for _, effect in usable], dtype=float)
    scale = np.nanstd(x_hist, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    distances = np.sqrt(np.mean(((x_hist - x_cur) / scale) ** 2, axis=1))
    finite_mask = np.isfinite(distances) & np.isfinite(effects)
    if int(np.sum(finite_mask)) < int(min_history):
        return {
            **base_out,
            "mode": mode,
            "source": "insufficient_finite_history",
            "reason": "insufficient_finite_history_noop",
            "history_n": int(np.sum(finite_mask)),
        }
    distances = distances[finite_mask]
    effects = effects[finite_mask]
    order = np.argsort(distances)
    k = int(max(1, min(int(neighbor_k), len(order))))
    neighbor_idx = order[:k]
    neighbor_dist = distances[neighbor_idx]
    neighbor_effects = effects[neighbor_idx]
    weights = 1.0 / np.maximum(neighbor_dist, 1e-3)
    weights = weights / max(float(np.sum(weights)), EPS)
    pred_effect = float(np.sum(weights * neighbor_effects))
    win_rate = float(np.sum(weights * (neighbor_effects > 0.0)))
    effect_score = float(np.tanh(pred_effect / max(float(tau), EPS)))
    win_score = float(2.0 * win_rate - 1.0)
    trust_score = float(_clip(0.70 * effect_score + 0.30 * win_score, -1.0, 1.0))
    scale_value = 1.0
    reason = "prior_action_nonnegative_noop"
    if trust_score < 0.0:
        scale_value = float(1.0 + trust_score * (1.0 - float(min_scale)))
        reason = "prior_action_negative_shrink"
    if (
        mode == "knn_abstain_v0"
        and pred_effect <= float(abstain_effect_threshold)
        and win_rate <= float(abstain_win_rate_threshold)
    ):
        scale_value = 0.0
        reason = "prior_action_negative_abstain"
    return {
        **base_out,
        "mode": mode,
        "scale": float(_clip(scale_value, 0.0, 1.0)),
        "source": "prior_knn_margin",
        "reason": reason,
        "history_n": int(len(usable)),
        "neighbor_n": int(k),
        "pred_effect": pred_effect,
        "neighbor_effect_mean": float(np.mean(neighbor_effects)),
        "neighbor_effect_std": float(np.std(neighbor_effects)),
        "win_rate": win_rate,
        "trust_score": trust_score,
        "mean_distance": float(np.mean(neighbor_dist)),
    }


def _intrinsic_temporal_dedup_support_scale(
    *,
    intrinsic_module_delta: float,
    temporal_delta: float,
    intrinsic_support: float,
    temporal_support: float,
    mode: str,
    duplicate_threshold: float,
    min_scale: float,
    strength: float,
) -> Dict[str, Any]:
    sign_match = _expert_sign_match_score(float(intrinsic_module_delta), float(temporal_delta))
    duplicate_ratio = _expert_duplicate_ratio(float(intrinsic_module_delta), float(temporal_delta))
    if str(mode) == "off":
        return {
            "mode": "off",
            "active": 0,
            "scale": 1.0,
            "sign_match": float(sign_match),
            "duplicate_ratio": float(duplicate_ratio),
            "reason": "off",
        }
    if str(mode) != "support_shrink_v1":
        raise ValueError(f"Unsupported intrinsic_temporal_dedup_mode: {mode}")
    if not (
        float(intrinsic_support) > 0.0
        and float(temporal_support) > 0.0
        and np.isfinite(float(intrinsic_module_delta))
        and np.isfinite(float(temporal_delta))
    ):
        return {
            "mode": str(mode),
            "active": 0,
            "scale": 1.0,
            "sign_match": float(sign_match),
            "duplicate_ratio": float(duplicate_ratio),
            "reason": "inactive_missing_supported_pair",
        }
    if sign_match < 1.0:
        return {
            "mode": str(mode),
            "active": 0,
            "scale": 1.0,
            "sign_match": float(sign_match),
            "duplicate_ratio": float(duplicate_ratio),
            "reason": "inactive_not_same_direction",
        }
    threshold = float(_clip(duplicate_threshold, 0.0, 1.0))
    if duplicate_ratio < threshold:
        return {
            "mode": str(mode),
            "active": 0,
            "scale": 1.0,
            "sign_match": float(sign_match),
            "duplicate_ratio": float(duplicate_ratio),
            "reason": "inactive_below_duplicate_threshold",
        }
    excess = float((duplicate_ratio - threshold) / max(1.0 - threshold, EPS))
    scale = float(_clip(1.0 - float(strength) * excess, float(min_scale), 1.0))
    return {
        "mode": str(mode),
        "active": int(scale < 1.0),
        "scale": float(scale),
        "sign_match": float(sign_match),
        "duplicate_ratio": float(duplicate_ratio),
        "reason": "same_direction_duplicate_support_shrunk" if scale < 1.0 else "inactive_scale_one",
    }


def _guarded_delta_from_base_delta(
    *,
    row: Mapping[str, Any],
    anchor_pred: float,
    base_delta: float,
    guidance_quality_guardrail_mode: str,
    anchor_history_score: float,
) -> Tuple[float, float]:
    if not (np.isfinite(anchor_pred) and float(anchor_pred) > 0.0 and np.isfinite(base_delta)):
        return float("nan"), float("nan")
    pred = float(anchor_pred * np.exp(float(base_delta)))
    guardrail = _apply_guidance_quality_guardrail(
        anchor_pred=float(anchor_pred),
        final_pred=float(pred),
        guidance_label=str(row.get("guidance_availability") or "none"),
        mode=str(guidance_quality_guardrail_mode),
        anchor_history_score=float(anchor_history_score),
    )
    pred = float(guardrail["post_guardrail_pred"])
    delta = float(base_delta)
    if np.isfinite(pred) and pred > 0.0:
        delta = float(np.log(pred / float(anchor_pred)))
    return pred, delta


def _specialist_gap_guard(
    *,
    base_delta: float,
    expert_delta: float,
    gap_floor: float = 0.02,
    sign_conflict_penalty: float = 0.55,
) -> Dict[str, float]:
    base_sign = _sign_num(float(base_delta))
    expert_sign = _sign_num(float(expert_delta))
    denom = max(abs(float(base_delta)), float(gap_floor), EPS)
    gap_ratio = float(abs(float(expert_delta) - float(base_delta)) / denom)
    gap_scale = float(1.0 / (1.0 + gap_ratio))
    sign_scale = 1.0
    if base_sign != 0.0 and expert_sign != 0.0 and base_sign != expert_sign:
        sign_scale = float(sign_conflict_penalty)
    specialist_scale = float(max(0.15, sign_scale * gap_scale))
    base_boost = float(1.0 + 0.75 * (1.0 - gap_scale))
    if sign_scale < 1.0:
        base_boost *= 1.15
    return {
        "gap_ratio": float(gap_ratio),
        "gap_scale": float(gap_scale),
        "sign_scale": float(sign_scale),
        "specialist_scale": float(specialist_scale),
        "base_boost": float(base_boost),
    }


def _outlier_specialist_scale(
    *,
    base_delta: float,
    candidate_delta: float,
    peer_delta: float,
    min_gap: float = 0.01,
    ratio_threshold: float = 1.5,
) -> float:
    candidate_gap = float(abs(float(candidate_delta) - float(base_delta)))
    peer_gap = float(abs(float(peer_delta) - float(base_delta)))
    if candidate_gap <= float(min_gap):
        return 1.0
    if candidate_gap <= float(ratio_threshold) * max(peer_gap, float(min_gap)):
        return 1.0
    return float(max(0.25, max(peer_gap, float(min_gap)) / max(candidate_gap, EPS)))


def _evidence_overlap_scale(
    *,
    target_delta: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    temporal_delta: float,
    temporal_support: float,
    duplicate_threshold: float,
    min_scale: float,
    strength: float,
) -> Dict[str, Any]:
    overlaps: List[Tuple[str, float]] = []
    if float(intrinsic_support) > 0.0 and np.isfinite(float(intrinsic_delta)):
        overlaps.append(("intrinsic_direct", _expert_duplicate_ratio(float(target_delta), float(intrinsic_delta))))
    if float(temporal_support) > 0.0 and np.isfinite(float(temporal_delta)):
        overlaps.append(("temporal_direct", _expert_duplicate_ratio(float(target_delta), float(temporal_delta))))
    if not overlaps or not np.isfinite(float(target_delta)):
        return {
            "scale": 1.0,
            "duplicate_ratio": 0.0,
            "source": "none",
            "reason": "no_supported_evidence_overlap",
        }
    source, duplicate_ratio = max(overlaps, key=lambda item: item[1])
    threshold = float(_clip(duplicate_threshold, 0.0, 1.0))
    if duplicate_ratio < threshold:
        return {
            "scale": 1.0,
            "duplicate_ratio": float(duplicate_ratio),
            "source": source,
            "reason": "below_duplicate_threshold",
        }
    excess = float((duplicate_ratio - threshold) / max(1.0 - threshold, EPS))
    scale = float(_clip(1.0 - float(strength) * excess, float(min_scale), 1.0))
    return {
        "scale": float(scale),
        "duplicate_ratio": float(duplicate_ratio),
        "source": source,
        "reason": "evidence_duplicate_shrink" if scale < 1.0 else "duplicate_scale_one",
    }


def _integrated_evidence_text(item: Mapping[str, Any] | Any) -> str:
    if not isinstance(item, Mapping):
        return str(item or "")
    parts = [
        str(item.get("segment") or item.get("memory_segment") or ""),
        str(item.get("relation_family") or item.get("memory_relation_family") or ""),
        str(item.get("polarity") or item.get("memory_polarity") or ""),
        str(item.get("evidence") or item.get("quote") or item.get("text") or ""),
    ]
    return " ".join(part for part in parts if part)


def _integrated_timing_risk(
    *,
    selected_delta: float,
    target_source: str,
    native_forward_rows: Sequence[Mapping[str, Any]],
    temporal_top_matches: Sequence[Mapping[str, Any]],
) -> bool:
    if not np.isfinite(float(selected_delta)) or float(selected_delta) >= 0.0:
        return False
    if str(target_source) == "temporal_direct" and temporal_top_matches:
        items = list(temporal_top_matches)
    else:
        items = list(native_forward_rows)
    combined = " ".join(_integrated_evidence_text(item) for item in items).lower()
    if not any(term in combined for term in INTEGRATED_NEGATIVE_SUPPLY_TERMS):
        return False
    return any(re.search(pattern, combined) for pattern in INTEGRATED_FUTURE_EASING_PATTERNS)


def _integrated_expert_arbitration_adjustment(
    *,
    row: Mapping[str, Any],
    anchor_pred: float,
    retained_pre_guardrail_delta: float,
    retained_post_guardrail_delta: float,
    compressed_base_delta: float,
    compressed_base_support: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    temporal_delta: float,
    temporal_support: float,
    anchor_error_recent_abs_log: float,
    internal_strength: float,
    native_forward_rows: Sequence[Mapping[str, Any]],
    temporal_top_matches: Sequence[Mapping[str, Any]],
    floor_ratio: float,
) -> Dict[str, Any]:
    policy = INTEGRATED_ARBITRATION_INTERNAL_POLICY
    selected_delta = float(retained_pre_guardrail_delta)
    action = "retained_came"
    reason = "retained_came_expert_blend_kept"
    target_source = "retained_came"
    target_delta = float("nan")
    internal_active = 0
    floor_active = 0
    timing_blocked = 0
    floor_pred = float("nan")

    deltas = np.asarray([compressed_base_delta, intrinsic_delta, temporal_delta], dtype=float)
    supports = np.asarray([compressed_base_support, intrinsic_support, temporal_support], dtype=float)
    source_names = np.asarray(["compressed_base", "intrinsic_direct", "temporal_direct"], dtype=object)
    retained_sign = _sign_num(float(retained_post_guardrail_delta))
    guid_mid = _safe_float(row.get("guid_mid"), float("nan"))
    no_guid_mid_numeric = not np.isfinite(guid_mid)
    supported = (supports >= float(policy["min_component_support"])) & np.isfinite(deltas)
    agree = (np.sign(deltas) == retained_sign) & supported & (retained_sign != 0.0)
    oppose = (np.sign(deltas) == -retained_sign) & supported & (retained_sign != 0.0)
    total_support = float(np.nansum(np.where(np.isfinite(supports), supports, 0.0)))

    active = bool(
        np.isfinite(anchor_pred)
        and anchor_pred > 0.0
        and np.isfinite(retained_post_guardrail_delta)
        and no_guid_mid_numeric
        and total_support >= float(policy["min_total_support"])
        and int(np.sum(agree)) >= int(policy["min_agree"])
        and int(np.sum(oppose)) <= int(policy["max_oppose"])
        and float(anchor_error_recent_abs_log) >= float(policy["min_anchor_recent_abs_log"])
        and float(internal_strength) >= float(policy["min_internal_strength"])
    )
    if active:
        signed = np.where(agree, deltas, np.nan)
        abs_signed = np.where(np.isnan(signed), -1.0, np.abs(signed))
        best_idx = int(np.argmax(abs_signed))
        if float(abs_signed[best_idx]) >= 0.0:
            target_delta = float(signed[best_idx])
            target_source = str(source_names[best_idx])
            active = bool(
                np.isfinite(target_delta)
                and _sign_num(target_delta) == retained_sign
                and abs(target_delta) > abs(float(retained_post_guardrail_delta)) + float(policy["min_target_gap"])
            )
        else:
            active = False
    if active:
        proposed_delta = float(_clip(target_delta, -float(policy["max_abs_delta"]), float(policy["max_abs_delta"])))
        if _integrated_timing_risk(
            selected_delta=proposed_delta,
            target_source=target_source,
            native_forward_rows=native_forward_rows,
            temporal_top_matches=temporal_top_matches,
        ):
            timing_blocked = 1
            action = "timing_guard_kept_retained_came"
            reason = "negative_supply_or_delay_evidence_points_to_future_easing"
            target_source = str(target_source)
        else:
            selected_delta = proposed_delta
            internal_active = 1
            action = "internal_consensus_delta"
            reason = "same_direction_internal_expert_has_stronger_anchor_correction"

    pre_guardrail_pred = float(anchor_pred * np.exp(selected_delta)) if np.isfinite(anchor_pred) and anchor_pred > 0.0 and np.isfinite(selected_delta) else float("nan")
    seasonal = _safe_float(row.get("pred__seasonal_naive_q4"), float("nan"))
    if np.isfinite(seasonal) and seasonal > 0.0:
        floor_pred = float(floor_ratio) * float(seasonal)
    no_guidance_floor_scope = str(row.get("guidance_availability") or _row_guidance_bucket(row)) == "none" and no_guid_mid_numeric
    if no_guidance_floor_scope and np.isfinite(pre_guardrail_pred) and np.isfinite(floor_pred) and pre_guardrail_pred < floor_pred:
        pre_guardrail_pred = floor_pred
        if np.isfinite(anchor_pred) and anchor_pred > 0.0:
            selected_delta = float(np.log(pre_guardrail_pred / anchor_pred))
        floor_active = 1
        if action == "retained_came":
            action = "compressed_base_seasonal_prior_floor"
            reason = "no_guidance_prediction_below_pretest_derived_seasonal_prior"
        else:
            action = f"{action}+compressed_base_seasonal_prior_floor"
            reason = f"{reason}; no_guidance_prediction_below_pretest_derived_seasonal_prior"

    return {
        "delta": float(selected_delta),
        "action": action,
        "reason": reason,
        "target_source": target_source,
        "target_delta_log": float(target_delta) if np.isfinite(target_delta) else float("nan"),
        "retained_post_guardrail_delta_log": float(retained_post_guardrail_delta),
        "internal_active": int(internal_active),
        "base_prior_floor_active": int(floor_active),
        "timing_blocked": int(timing_blocked),
        "floor_ratio": float(floor_ratio),
        "floor_pred": float(floor_pred) if np.isfinite(floor_pred) else float("nan"),
    }


def _arbitrate_direct_rawcard_experts(
    *,
    compressed_base_delta: float,
    compressed_base_support: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    temporal_delta: float,
    temporal_support: float,
    active_experts: Sequence[str],
    arbitration_mode: str,
    guidance_delta: float = 0.0,
    guidance_support: float = 0.0,
    guidance_expert_mode: str = "off",
    guidance_bucket: str = "",
    evidence_orthogonal_duplicate_threshold: float = 0.35,
    evidence_orthogonal_base_min_scale: float = 0.50,
    evidence_orthogonal_guidance_min_scale: float = 0.70,
    evidence_orthogonal_base_strength: float = 0.60,
    evidence_orthogonal_guidance_strength: float = 0.40,
) -> Dict[str, Any]:
    active = set(active_experts)
    sign_match = 0.5
    duplicate_ratio = 0.0
    base_weight = float(_clip(compressed_base_support, 0.0, 1.0)) if "compressed_base" in active else 0.0
    intrinsic_weight = float(_clip(intrinsic_support, 0.0, 1.0)) if "intrinsic_direct" in active else 0.0
    temporal_weight = float(_clip(temporal_support, 0.0, 1.0)) if "temporal_direct" in active else 0.0

    # Always expose intrinsic/temporal agreement diagnostics; only the current mode shapes weights with them.
    if "intrinsic_direct" in active and "temporal_direct" in active:
        sign_match = _expert_sign_match_score(intrinsic_delta, temporal_delta)
        duplicate_ratio = _expert_duplicate_ratio(intrinsic_delta, temporal_delta)
        if str(arbitration_mode) in {"current", "current_gap_guard", CURRENT_EVIDENCE_ORTHOGONAL_ARBITRATION_MODE}:
            if sign_match >= 1.0:
                duplicate_scale = float(_clip(1.0 - 0.35 * duplicate_ratio, 0.45, 1.0))
                intrinsic_weight *= duplicate_scale
                temporal_weight *= duplicate_scale
                base_weight *= float(1.0 + 0.25 * duplicate_ratio)
            elif sign_match <= 0.0:
                intrinsic_weight *= 0.5
                temporal_weight *= 0.5
                base_weight *= 1.75
            else:
                intrinsic_weight *= 0.75
                temporal_weight *= 0.75
                base_weight *= 1.35

    if str(arbitration_mode) == "current_gap_guard" and "compressed_base" in active and base_weight > 0.0:
        if "intrinsic_direct" in active and intrinsic_weight > 0.0:
            intrinsic_guard = _specialist_gap_guard(base_delta=compressed_base_delta, expert_delta=intrinsic_delta)
            intrinsic_weight *= float(intrinsic_guard["specialist_scale"])
            base_weight *= float(intrinsic_guard["base_boost"])
        if "temporal_direct" in active and temporal_weight > 0.0:
            temporal_guard = _specialist_gap_guard(base_delta=compressed_base_delta, expert_delta=temporal_delta)
            temporal_weight *= float(temporal_guard["specialist_scale"])
            base_weight *= float(temporal_guard["base_boost"])

    if (
        str(arbitration_mode) == "current_consensus_guard"
        and "compressed_base" in active
        and "intrinsic_direct" in active
        and "temporal_direct" in active
        and intrinsic_weight > 0.0
        and temporal_weight > 0.0
        and _sign_num(float(intrinsic_delta)) != 0.0
        and _sign_num(float(intrinsic_delta)) == _sign_num(float(temporal_delta))
    ):
        intrinsic_scale = _outlier_specialist_scale(
            base_delta=compressed_base_delta,
            candidate_delta=intrinsic_delta,
            peer_delta=temporal_delta,
        )
        temporal_scale = _outlier_specialist_scale(
            base_delta=compressed_base_delta,
            candidate_delta=temporal_delta,
            peer_delta=intrinsic_delta,
        )
        if intrinsic_scale < 1.0:
            intrinsic_weight *= float(intrinsic_scale)
            base_weight *= 1.10
            temporal_weight *= 1.05
        if temporal_scale < 1.0:
            temporal_weight *= float(temporal_scale)
            base_weight *= 1.10
            intrinsic_weight *= 1.05

    support_map = {
        "compressed_base": float(base_weight),
        "intrinsic_direct": float(intrinsic_weight),
        "temporal_direct": float(temporal_weight),
    }
    delta_map = {
        "compressed_base": float(compressed_base_delta),
        "intrinsic_direct": float(intrinsic_delta),
        "temporal_direct": float(temporal_delta),
    }
    if str(guidance_expert_mode) != "off" and guidance_support > 0.0 and np.isfinite(guidance_delta):
        support_map["guidance_expert"] = float(_clip(guidance_support, 0.0, 1.0))
        delta_map["guidance_expert"] = float(guidance_delta)
    orthogonal_diag = {
        "mode": str(arbitration_mode),
        "active": 0,
        "base_scale": 1.0,
        "guidance_scale": 1.0,
        "base_duplicate_ratio": 0.0,
        "guidance_duplicate_ratio": 0.0,
        "base_overlap_source": "",
        "guidance_overlap_source": "",
        "reason": "not_enabled",
    }
    if str(arbitration_mode) in {EVIDENCE_ORTHOGONAL_ARBITRATION_MODE, CURRENT_EVIDENCE_ORTHOGONAL_ARBITRATION_MODE}:
        guidance_bucket_norm = str(guidance_bucket or "")
        base_diag = {
            "scale": 1.0,
            "duplicate_ratio": 0.0,
            "source": "",
            "reason": "base_protected_no_guidance_or_inactive",
        }
        guidance_diag = {
            "scale": 1.0,
            "duplicate_ratio": 0.0,
            "source": "",
            "reason": "guidance_inactive",
        }
        if guidance_bucket_norm != "none" and support_map.get("compressed_base", 0.0) > 0.0:
            base_diag = _evidence_overlap_scale(
                target_delta=float(delta_map.get("compressed_base", 0.0)),
                intrinsic_delta=float(intrinsic_delta),
                intrinsic_support=float(support_map.get("intrinsic_direct", 0.0)),
                temporal_delta=float(temporal_delta),
                temporal_support=float(support_map.get("temporal_direct", 0.0)),
                duplicate_threshold=float(evidence_orthogonal_duplicate_threshold),
                min_scale=float(evidence_orthogonal_base_min_scale),
                strength=float(evidence_orthogonal_base_strength),
            )
            support_map["compressed_base"] = float(support_map["compressed_base"] * float(base_diag["scale"]))
        if support_map.get("guidance_expert", 0.0) > 0.0:
            guidance_diag = _evidence_overlap_scale(
                target_delta=float(delta_map.get("guidance_expert", 0.0)),
                intrinsic_delta=float(intrinsic_delta),
                intrinsic_support=float(support_map.get("intrinsic_direct", 0.0)),
                temporal_delta=float(temporal_delta),
                temporal_support=float(support_map.get("temporal_direct", 0.0)),
                duplicate_threshold=float(evidence_orthogonal_duplicate_threshold),
                min_scale=float(evidence_orthogonal_guidance_min_scale),
                strength=float(evidence_orthogonal_guidance_strength),
            )
            support_map["guidance_expert"] = float(support_map["guidance_expert"] * float(guidance_diag["scale"]))
        active_orthogonal = int(float(base_diag.get("scale", 1.0)) < 0.999999 or float(guidance_diag.get("scale", 1.0)) < 0.999999)
        orthogonal_diag = {
            "mode": str(arbitration_mode),
            "active": active_orthogonal,
            "base_scale": float(base_diag.get("scale", 1.0)),
            "guidance_scale": float(guidance_diag.get("scale", 1.0)),
            "base_duplicate_ratio": float(base_diag.get("duplicate_ratio", 0.0)),
            "guidance_duplicate_ratio": float(guidance_diag.get("duplicate_ratio", 0.0)),
            "base_overlap_source": str(base_diag.get("source") or ""),
            "guidance_overlap_source": str(guidance_diag.get("source") or ""),
            "reason": ";".join([str(base_diag.get("reason") or ""), str(guidance_diag.get("reason") or "")]),
        }
    usable = [(delta_map[name], support_map[name]) for name in support_map if support_map[name] > 0.0 and np.isfinite(delta_map[name])]
    pred = 0.0
    denom = float(sum(weight for _, weight in usable))
    if denom > EPS:
        pred = float(sum(value * weight for value, weight in usable) / denom)
    normalized_weights = {
        name: (float(weight / denom) if denom > EPS else 0.0)
        for name, weight in support_map.items()
    }
    return {
        "pred": float(pred),
        "sign_match": float(sign_match),
        "duplicate_ratio": float(duplicate_ratio),
        "support_sum": float(denom),
        "active_experts": list(active_experts),
        "candidate_supports": support_map,
        "candidate_weights": normalized_weights,
        "orthogonal_diag": orthogonal_diag,
    }


def _build_direct_gate_features(
    *,
    base_delta: float,
    intrinsic_delta: float,
    intrinsic_support: float,
    temporal_delta: float,
    temporal_support: float,
    sign_match: float,
    guidance_lock: float,
    anchor_uncertainty: float,
    internal_strength: float,
    fwd_conflict_ratio: float,
    memory_support: float,
    memory_consistency: float,
) -> Dict[str, float]:
    return {
        "gate_base_delta": float(base_delta),
        "gate_intrinsic_delta": float(intrinsic_delta),
        "gate_intrinsic_support": float(intrinsic_support),
        "gate_temporal_delta": float(temporal_delta),
        "gate_temporal_support": float(temporal_support),
        "gate_gap_base_intrinsic": float(base_delta - intrinsic_delta),
        "gate_gap_base_temporal": float(base_delta - temporal_delta),
        "gate_intrinsic_temporal_sign_match": float(sign_match),
        "gate_guidance_lock": float(guidance_lock),
        "gate_anchor_uncertainty": float(anchor_uncertainty),
        "gate_internal_strength": float(internal_strength),
        "gate_fwd_conflict_ratio": float(fwd_conflict_ratio),
        "gate_memory_support": float(memory_support),
        "gate_memory_consistency": float(memory_consistency),
    }


def _direct_gate_presence_scale(
    *,
    base_delta: float,
    base_weight: float,
    sign_match: float,
    max_abs_log_delta: float,
) -> float:
    if not np.isfinite(base_delta):
        return 0.0
    presence = float(_clip(np.sqrt(abs(float(base_delta)) / max(float(max_abs_log_delta), EPS)), 0.0, 1.0))
    reliability = float(_clip(0.25 + 0.75 * max(float(sign_match), 0.25), 0.0, 1.0))
    base_term = float(_clip(0.35 + 0.65 * float(base_weight), 0.0, 1.0))
    return float(_clip(presence * reliability * base_term, 0.0, 1.0))


def _intrinsic_effective_support(
    *,
    raw_support: float,
    coverage: float,
    semantic_clarity: float,
    conflict_ratio: float,
    safety_scale: float,
    delta_pred: float,
) -> float:
    return float(
        _clip(
            float(raw_support)
            * (0.5 + 0.5 * float(coverage))
            * (0.5 + 0.5 * float(semantic_clarity))
            * (1.0 - 0.5 * float(conflict_ratio))
            * float(safety_scale if float(delta_pred) == 0.0 else 1.0),
            0.0,
            1.0,
        )
    )


def _intrinsic_strict_group_feature_name(card: Mapping[str, Any]) -> str:
    return "strict::" + "||".join(str(piece) for piece in _card_fusion_key(card))


def _intrinsic_loose_group_feature_name(card: Mapping[str, Any]) -> str:
    return "loose::" + "||".join(
        [
            str(_card_segment_name(card) or "UNKNOWN").strip().lower() or "unknown",
            str(card.get("canonical_factor") or "other").strip().lower() or "other",
            str(card.get("polarity") or "unknown").strip().lower() or "unknown",
        ]
    )


def _intrinsic_grouped_feature_candidates(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    admissibility_mode: str,
    max_strict_features: int,
    max_loose_features: int,
    min_feature_occurrence: int,
) -> List[str]:
    stats: Dict[str, Dict[str, float]] = {}
    for row in train_rows:
        cards = list(row.get("forward_rows") or [])
        if not cards:
            continue
        seen_in_quarter = set()
        for card in cards:
            mass = abs(float(_weighted_native_card_mass(card, admissibility_mode)))
            if mass <= EPS:
                continue
            for name in (_intrinsic_strict_group_feature_name(card), _intrinsic_loose_group_feature_name(card)):
                entry = stats.setdefault(name, {"occurrence": 0.0, "abs_mass": 0.0})
                entry["abs_mass"] += mass
                if name not in seen_in_quarter:
                    entry["occurrence"] += 1.0
                    seen_in_quarter.add(name)
    strict_names = [
        name for name, entry in stats.items()
        if name.startswith("strict::") and int(entry["occurrence"]) >= int(min_feature_occurrence)
    ]
    loose_names = [
        name for name, entry in stats.items()
        if name.startswith("loose::") and int(entry["occurrence"]) >= int(min_feature_occurrence)
    ]
    strict_names.sort(key=lambda name: (float(stats[name]["abs_mass"]), float(stats[name]["occurrence"])), reverse=True)
    loose_names.sort(key=lambda name: (float(stats[name]["abs_mass"]), float(stats[name]["occurrence"])), reverse=True)
    return list(strict_names[: int(max_strict_features)]) + list(loose_names[: int(max_loose_features)])


def _intrinsic_grouped_feature_map(
    cards: Sequence[Mapping[str, Any]],
    *,
    admissibility_mode: str,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, float], float, float]:
    feature_set = set(feature_names)
    out = {name: 0.0 for name in feature_names}
    covered_abs_mass = 0.0
    total_abs_mass = 0.0
    for card in cards:
        mass = float(_weighted_native_card_mass(card, admissibility_mode))
        abs_mass = abs(mass)
        total_abs_mass += abs_mass
        matched = False
        strict_name = _intrinsic_strict_group_feature_name(card)
        loose_name = _intrinsic_loose_group_feature_name(card)
        if strict_name in feature_set:
            out[strict_name] += mass
            matched = True
        if loose_name in feature_set:
            out[loose_name] += mass
            matched = True
        if matched:
            covered_abs_mass += abs_mass
    coverage = float(covered_abs_mass / max(total_abs_mass, EPS)) if total_abs_mass > EPS else 0.0
    return out, float(coverage), float(total_abs_mass)


def _intrinsic_grouped_card_influence(
    card: Mapping[str, Any],
    *,
    admissibility_mode: str,
    coefs_by_name: Mapping[str, float],
) -> float:
    mass = float(_weighted_native_card_mass(card, admissibility_mode))
    total = 0.0
    strict_name = _intrinsic_strict_group_feature_name(card)
    loose_name = _intrinsic_loose_group_feature_name(card)
    if strict_name in coefs_by_name:
        total += mass * float(coefs_by_name[strict_name])
    if loose_name in coefs_by_name:
        total += mass * float(coefs_by_name[loose_name])
    return float(total)


SEGMENT_BRIDGE_BUCKETS = ("top1", "top2", "rest")
SEGMENT_BRIDGE_FEATURE_NAMES = [
    *(f"segbridge_bucket_signed::{bucket}" for bucket in SEGMENT_BRIDGE_BUCKETS),
    *(f"segbridge_factor_signed::{bucket}||{factor}" for bucket in SEGMENT_BRIDGE_BUCKETS for factor in CANONICAL_FACTORS),
]


def _segment_bridge_bucket(card: Mapping[str, Any]) -> str:
    share = float(_clip(_safe_float(card.get("segment_share_at_observed"), 0.0), 0.0, 1.0))
    rank = int(_safe_float(card.get("segment_rank_at_observed"), 0.0))
    if share <= 0.0 or rank <= 0:
        return ""
    if rank == 1:
        return "top1"
    if rank == 2:
        return "top2"
    return "rest"


def _segment_bridge_feature_map(
    cards: Sequence[Mapping[str, Any]],
    *,
    admissibility_mode: str,
) -> Tuple[Dict[str, float], float, float]:
    out = {name: 0.0 for name in SEGMENT_BRIDGE_FEATURE_NAMES}
    covered_abs_mass = 0.0
    total_abs_mass = 0.0
    for card in cards:
        mass = float(_weighted_native_card_mass(card, admissibility_mode))
        abs_mass = abs(mass)
        total_abs_mass += abs_mass
        bucket = _segment_bridge_bucket(card)
        if not bucket:
            continue
        share = float(_clip(_safe_float(card.get("segment_share_at_observed"), 0.0), 0.0, 1.0))
        factor = str(card.get("canonical_factor") or "other")
        factor = factor if factor in CANONICAL_FACTORS else "other"
        bridged_mass = float(mass * share)
        out[f"segbridge_bucket_signed::{bucket}"] += bridged_mass
        out[f"segbridge_factor_signed::{bucket}||{factor}"] += bridged_mass
        covered_abs_mass += abs_mass
    coverage = float(covered_abs_mass / max(total_abs_mass, EPS)) if total_abs_mass > EPS else 0.0
    return out, float(coverage), float(total_abs_mass)


def _predict_segment_bridge_candidate(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    current_cards: Sequence[Mapping[str, Any]],
    target_col: str,
    admissibility_mode: str,
    alpha: float,
    min_train: int,
    shrink_k: float,
    delta_cap_quantile: float = 0.9,
) -> Dict[str, Any]:
    valid_train = []
    for row in train_rows:
        cards = list(row.get("forward_rows") or [])
        target_value = float(_safe_float(row.get(target_col), float("nan")))
        if not cards or not np.isfinite(target_value):
            continue
        valid_train.append(dict(row))
    if len(valid_train) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(valid_train)),
            "support": 0.0,
            "coverage": 0.0,
            "feature_names": list(SEGMENT_BRIDGE_FEATURE_NAMES),
            "top_contribs": [],
            "coefs_by_name": {},
        }

    x_rows: List[List[float]] = []
    y_rows: List[float] = []
    for row in valid_train:
        fmap, _, _ = _segment_bridge_feature_map(
            list(row.get("forward_rows") or []),
            admissibility_mode=admissibility_mode,
        )
        x_rows.append([float(fmap.get(name, 0.0)) for name in SEGMENT_BRIDGE_FEATURE_NAMES])
        y_rows.append(float(_safe_float(row.get(target_col), float("nan"))))
    x_train = np.asarray(x_rows, dtype=float)
    y_train = np.asarray(y_rows, dtype=float)
    mask = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    x_train = x_train[mask]
    y_train = y_train[mask]
    if len(y_train) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(y_train)),
            "support": 0.0,
            "coverage": 0.0,
            "feature_names": list(SEGMENT_BRIDGE_FEATURE_NAMES),
            "top_contribs": [],
            "coefs_by_name": {},
        }

    model = _fit_zero_intercept_ridge(x_train, y_train, float(alpha))
    current_feature_map, coverage, _ = _segment_bridge_feature_map(
        current_cards,
        admissibility_mode=admissibility_mode,
    )
    x_cur = np.asarray([float(current_feature_map.get(name, 0.0)) for name in SEGMENT_BRIDGE_FEATURE_NAMES], dtype=float)
    pred_raw = float(_predict_zero_intercept_ridge(model, x_cur))
    finite_abs = np.abs(y_train[np.isfinite(y_train)])
    if finite_abs.size:
        delta_cap = float(np.quantile(finite_abs, float(delta_cap_quantile)))
        if np.isfinite(delta_cap) and delta_cap > 0.0:
            pred_raw = float(_clip(pred_raw, -delta_cap, delta_cap))
    train_support = float(len(y_train) / max(len(y_train) + float(shrink_k), EPS))
    support = float(_clip(train_support * (0.5 + 0.5 * coverage), 0.0, 1.0))
    coefs_by_name = {
        name: float(val)
        for name, val in zip(SEGMENT_BRIDGE_FEATURE_NAMES, np.asarray(model["coefs"], dtype=float) / np.asarray(model["stds"], dtype=float))
    }
    return {
        "pred": float(pred_raw * support),
        "pred_raw": float(pred_raw),
        "train_count": int(len(y_train)),
        "support": float(support),
        "coverage": float(coverage),
        "feature_names": list(SEGMENT_BRIDGE_FEATURE_NAMES),
        "top_contribs": _top_linear_feature_contribs(feature_names=SEGMENT_BRIDGE_FEATURE_NAMES, x_cur=x_cur, model=model, top_k=6),
        "coefs_by_name": coefs_by_name,
    }


def _top_linear_feature_contribs(
    *,
    feature_names: Sequence[str],
    x_cur: np.ndarray,
    model: Mapping[str, Any],
    top_k: int = 6,
) -> List[Tuple[str, float]]:
    stds = np.asarray(model["stds"], dtype=float)
    coefs = np.asarray(model["coefs"], dtype=float)
    xz = np.nan_to_num(x_cur / stds, nan=0.0, posinf=0.0, neginf=0.0)
    contribs = [(name, float(val)) for name, val in zip(feature_names, xz * coefs)]
    contribs.sort(key=lambda item: abs(item[1]), reverse=True)
    return contribs[:top_k]


def _predict_intrinsic_grouped_direct_candidate(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    current_cards: Sequence[Mapping[str, Any]],
    target_col: str,
    admissibility_mode: str,
    alpha: float,
    min_train: int,
    shrink_k: float,
    max_strict_features: int,
    max_loose_features: int,
    min_feature_occurrence: int,
    delta_cap_quantile: float = 0.9,
) -> Dict[str, Any]:
    valid_train = []
    for row in train_rows:
        cards = list(row.get("forward_rows") or [])
        target_value = float(_safe_float(row.get(target_col), float("nan")))
        if not cards or not np.isfinite(target_value):
            continue
        valid_train.append(dict(row))
    if len(valid_train) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(valid_train)),
            "support": 0.0,
            "coverage": 0.0,
            "feature_names": [],
            "top_contribs": [],
            "coefs_by_name": {},
        }
    feature_names = _intrinsic_grouped_feature_candidates(
        valid_train,
        admissibility_mode=admissibility_mode,
        max_strict_features=int(max_strict_features),
        max_loose_features=int(max_loose_features),
        min_feature_occurrence=int(min_feature_occurrence),
    )
    if not feature_names:
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(valid_train)),
            "support": 0.0,
            "coverage": 0.0,
            "feature_names": [],
            "top_contribs": [],
            "coefs_by_name": {},
        }

    x_rows: List[List[float]] = []
    y_rows: List[float] = []
    for row in valid_train:
        fmap, _, _ = _intrinsic_grouped_feature_map(
            list(row.get("forward_rows") or []),
            admissibility_mode=admissibility_mode,
            feature_names=feature_names,
        )
        x_rows.append([float(fmap.get(name, 0.0)) for name in feature_names])
        y_rows.append(float(_safe_float(row.get(target_col), float("nan"))))
    x_train = np.asarray(x_rows, dtype=float)
    y_train = np.asarray(y_rows, dtype=float)
    mask = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    x_train = x_train[mask]
    y_train = y_train[mask]
    if len(y_train) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(y_train)),
            "support": 0.0,
            "coverage": 0.0,
            "feature_names": feature_names,
            "top_contribs": [],
            "coefs_by_name": {},
        }

    model = _fit_zero_intercept_ridge(x_train, y_train, float(alpha))
    current_feature_map, coverage, _ = _intrinsic_grouped_feature_map(
        current_cards,
        admissibility_mode=admissibility_mode,
        feature_names=feature_names,
    )
    x_cur = np.asarray([float(current_feature_map.get(name, 0.0)) for name in feature_names], dtype=float)
    pred_raw = float(_predict_zero_intercept_ridge(model, x_cur))
    finite_abs = np.abs(y_train[np.isfinite(y_train)])
    if finite_abs.size:
        delta_cap = float(np.quantile(finite_abs, float(delta_cap_quantile)))
        if np.isfinite(delta_cap) and delta_cap > 0.0:
            pred_raw = float(_clip(pred_raw, -delta_cap, delta_cap))
    train_support = float(len(y_train) / max(len(y_train) + float(shrink_k), EPS))
    support = float(_clip(train_support * (0.5 + 0.5 * coverage), 0.0, 1.0))
    coefs_by_name = {
        name: float(val)
        for name, val in zip(feature_names, np.asarray(model["coefs"], dtype=float) / np.asarray(model["stds"], dtype=float))
    }
    return {
        "pred": float(pred_raw * support),
        "pred_raw": float(pred_raw),
        "train_count": int(len(y_train)),
        "support": float(support),
        "coverage": float(coverage),
        "feature_names": list(feature_names),
        "top_contribs": _top_linear_feature_contribs(feature_names=feature_names, x_cur=x_cur, model=model, top_k=6),
        "coefs_by_name": coefs_by_name,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run CSAIS raw-card direct experts v1 with direct intrinsic and temporal anchor-correction experts.")
    ap.add_argument("--experiment_config", default=DEFAULT_EXPERIMENT_CONFIG)
    ap.add_argument("--native_backbone_csv", default=DEFAULT_BACKBONE_CSV)
    ap.add_argument("--native_card_table_jsonl", default=DEFAULT_CARD_TABLE_JSONL)
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--raw_alpha", type=float, default=8.0)
    ap.add_argument("--factor_alpha", type=float, default=8.0)
    ap.add_argument("--intrinsic_alpha", type=float, default=10.0)
    ap.add_argument("--intrinsic_mode", choices=["additive_v1", "grouped_ridge_v3", "additive_grouped_hybrid_v3"], default="additive_v1")
    ap.add_argument(
        "--intrinsic_target_mode",
        choices=["anchor_residual", "post_base_temporal_guidance_residual_v1", "partitioned_residual_v1"],
        default="anchor_residual",
        help="Default preserves retained behavior. residual_v1 trains current-card evidence on the prior-row residual left after base/temporal/guidance without intrinsic; partitioned_residual_v1 uses that residual only on guidance-bearing rows.",
    )
    ap.add_argument(
        "--intrinsic_reliability_mode",
        choices=["off", "history_bucket_v1"],
        default="off",
        help="Default-off online support calibration using prior realized intrinsic marginal value in shared evidence buckets.",
    )
    ap.add_argument("--intrinsic_reliability_min_history", type=int, default=4)
    ap.add_argument("--intrinsic_reliability_tau", type=float, default=0.01)
    ap.add_argument("--intrinsic_reliability_min_scale", type=float, default=0.25)
    ap.add_argument("--intrinsic_reliability_max_scale", type=float, default=1.35)
    ap.add_argument(
        "--intrinsic_explicit_guidance_guard_mode",
        choices=["off", "support_shrink_v1"],
        default="off",
        help="Default-off anti-double-counting shrink for explicit numeric guidance rows.",
    )
    ap.add_argument("--intrinsic_explicit_guidance_support_scale", type=float, default=0.65)
    ap.add_argument(
        "--intrinsic_temporal_dedup_mode",
        choices=["off", "support_shrink_v1"],
        default="off",
        help="Default-off Intrinsic support shrink when current-card and temporal-memory signals duplicate the same correction.",
    )
    ap.add_argument("--intrinsic_temporal_dedup_duplicate_threshold", type=float, default=0.35)
    ap.add_argument("--intrinsic_temporal_dedup_min_scale", type=float, default=0.55)
    ap.add_argument("--intrinsic_temporal_dedup_strength", type=float, default=0.60)
    ap.add_argument(
        "--intrinsic_action_gate_mode",
        choices=["off", "knn_shrink_v0", "knn_abstain_v0"],
        default="off",
        help="Default-off prior-only kNN harm-risk gate for the Intrinsic/current-evidence action; diagnostic only unless promoted.",
    )
    ap.add_argument("--intrinsic_action_gate_min_history", type=int, default=8)
    ap.add_argument("--intrinsic_action_gate_neighbor_k", type=int, default=8)
    ap.add_argument("--intrinsic_action_gate_tau", type=float, default=0.003)
    ap.add_argument("--intrinsic_action_gate_min_scale", type=float, default=0.65)
    ap.add_argument("--intrinsic_action_gate_abstain_effect_threshold", type=float, default=-0.00025)
    ap.add_argument("--intrinsic_action_gate_abstain_win_rate_threshold", type=float, default=0.45)
    ap.add_argument("--candidate_min_train", type=int, default=10)
    ap.add_argument("--intrinsic_min_train", type=int, default=12)
    ap.add_argument("--candidate_shrink_k", type=float, default=16.0)
    ap.add_argument("--intrinsic_shrink_k", type=float, default=8.0)
    ap.add_argument("--intrinsic_grouped_max_strict_features", type=int, default=48)
    ap.add_argument("--intrinsic_grouped_max_loose_features", type=int, default=24)
    ap.add_argument("--intrinsic_grouped_min_feature_occurrence", type=int, default=2)
    ap.add_argument("--segment_bridge_mode", choices=["off", "intrinsic_hybrid_v1"], default="off")
    ap.add_argument("--segment_bridge_alpha", type=float, default=10.0)
    ap.add_argument("--segment_bridge_min_train", type=int, default=12)
    ap.add_argument("--segment_bridge_shrink_k", type=float, default=8.0)
    ap.add_argument("--segment_bridge_blend_weight", type=float, default=0.35)
    ap.add_argument("--candidate_base_scale", type=float, default=0.75)
    ap.add_argument("--shock_max_abs_log_delta", type=float, default=0.14)
    ap.add_argument("--anchor_uncertainty_recent_window", type=int, default=6)
    ap.add_argument("--anchor_uncertainty_same_quarter_min", type=int, default=2)
    ap.add_argument("--anchor_uncertainty_same_guidance_min", type=int, default=3)
    ap.add_argument("--anchor_uncertainty_tau", type=float, default=0.10)
    ap.add_argument("--gate_alpha", type=float, default=8.0)
    ap.add_argument("--gate_min_train", type=int, default=6)
    ap.add_argument("--gate_shrink_k", type=float, default=12.0)
    ap.add_argument("--gate_mode", choices=["off", "company_local", "company_local_scaled"], default="company_local")
    ap.add_argument("--method_family_label", default="")
    ap.add_argument(
        "--expert_contract",
        choices=tuple(DIRECT_EXPERT_CONTRACTS.keys()),
        default="full",
        help="Final blend contract used for frozen same-contract ablations.",
    )
    ap.add_argument("--report_start_quarter", default="FY2019_Q1")
    ap.add_argument("--report_end_quarter", default="FY2025_Q4")
    ap.add_argument(
        "--coverage_drift_mode",
        choices=["warn", "fail", "off"],
        default="warn",
        help="Surface report-quarter coverage drift without changing predictions; fail returns a non-zero status when rows are missing or skipped.",
    )
    ap.add_argument(
        "--missing_anchor_policy",
        choices=["skip", "repaired_quarter_prior_v1"],
        default="skip",
        help="Opt-in policy for non-finite selected anchors. Default skip preserves existing artifacts; repaired_quarter_prior_v1 is for current repaired-quarter ORCL reruns.",
    )
    ap.add_argument("--memory_top_k", type=int, default=3)
    ap.add_argument("--memory_temperature", type=float, default=0.35)
    ap.add_argument("--memory_min_train", type=int, default=6)
    ap.add_argument("--temporal_min_train", type=int, default=6)
    ap.add_argument("--temporal_history_cap", type=int, default=12)
    ap.add_argument("--temporal_top_k", type=int, default=3)
    ap.add_argument("--temporal_temperature", type=float, default=0.35)
    ap.add_argument("--temporal_item_top_k", type=int, default=2)
    ap.add_argument("--temporal_item_temperature", type=float, default=0.20)
    ap.add_argument("--temporal_same_quarter_bonus", type=float, default=0.10)
    ap.add_argument("--temporal_time_decay_quarters", type=float, default=8.0)
    ap.add_argument("--temporal_var_tau", type=float, default=0.03)
    ap.add_argument("--temporal_neff_scale", type=float, default=4.0)
    ap.add_argument("--temporal_directional_consistency_power", type=float, default=0.5)
    ap.add_argument("--temporal_attention_focus_power", type=float, default=1.0)
    ap.add_argument("--temporal_max_abs_log_correction", type=float, default=0.50)
    ap.add_argument("--temporal_score_mode", choices=["attention_only", "attention_plus_recency", "attention_plus_same_quarter", "current"], default="current")
    ap.add_argument("--temporal_support_mode", choices=["none", "var_neff", "var_neff_focus", "current"], default="current")
    ap.add_argument("--temporal_direction_mode", choices=["off", "support_scale", "min_align"], default="off")
    ap.add_argument(
        "--temporal_context_guard_mode",
        choices=["off", "guidance_available_min_align", "weak_guidance_min_align", "derived_weak_min_align"],
        default="off",
        help="Post-temporal trust calibration. The retained public wrapper uses derived_weak_min_align.",
    )
    ap.add_argument(
        "--temporal_context_memory_mode",
        choices=["off", "retrieve_cards", "typed_retrieval", "typed_reliability", "typed_soft_agreement"],
        default="off",
        help="Context-memory behavior inside temporal_direct. The retained public wrapper uses typed_retrieval.",
    )
    ap.add_argument(
        "--temporal_evidence_filter_mode",
        choices=["off", "immediate_revenue_v1", "immediate_revenue_strict_v1"],
        default="off",
        help="Default-off diagnostic filter for temporal expert cards; the release contract uses off.",
    )
    ap.add_argument(
        "--temporal_evidence_filter_scope",
        choices=["all", "forward_only", "context_only"],
        default="all",
        help="Scope for non-off temporal evidence filters; the release contract uses filter mode off.",
    )
    ap.add_argument(
        "--temporal_context_quality_mode",
        choices=["off", "immediate_revenue_v1", "immediate_revenue_strict_v1"],
        default="off",
        help="Default-off soft context reliability diagnostic; reduces typed-context retrieval weight without deleting cards.",
    )
    ap.add_argument(
        "--temporal_context_quality_weight",
        type=float,
        default=0.0,
        help="Soft downweight strength for noisy context retrieval when temporal_context_quality_mode is enabled.",
    )
    ap.add_argument(
        "--temporal_context_retrieval_weight",
        type=float,
        default=0.35,
        help="Weight for typed context cards when they are used only to retrieve comparable memory quarters.",
    )
    ap.add_argument("--temporal_segment_compat_weight", type=float, default=0.12)
    ap.add_argument("--temporal_context_support_weight", type=float, default=0.20)
    ap.add_argument("--temporal_novelty_shrink_weight", type=float, default=0.35)
    ap.add_argument("--temporal_guidance_trust_weight", type=float, default=0.30)
    ap.add_argument("--temporal_segment_support_weight", type=float, default=0.20)
    ap.add_argument("--temporal_soft_agreement_weight", type=float, default=0.20)
    ap.add_argument(
        "--temporal_context_memory_sparsity_power",
        type=float,
        default=1.0,
        help="Continuous shrink for retrieve-card context memory as target-forward card availability increases.",
    )
    ap.add_argument(
        "--temporal_guidance_bucket_scale_mode",
        choices=["off", "fixed_v1"],
        default="off",
        help="Default-off shared shrink for temporal_direct by guidance bucket; diagnostic only unless promoted. With partitioned residuals, enabled runs can also affect later Intrinsic residual-history targets.",
    )
    ap.add_argument("--temporal_explicit_guidance_scale", type=float, default=1.0)
    ap.add_argument("--temporal_non_explicit_guidance_scale", type=float, default=1.0)
    ap.add_argument("--temporal_no_guidance_scale", type=float, default=1.0)
    ap.add_argument(
        "--temporal_anchor_confidence_guard_mode",
        choices=["off", "strong_anchor_no_guidance_v1"],
        default="off",
        help="Default-off Temporal support guard: when the online anchor has strong prior history, suppress large no-guidance Temporal corrections; diagnostic only unless promoted.",
    )
    ap.add_argument("--temporal_anchor_confidence_guard_history_mae_threshold", type=float, default=0.02)
    ap.add_argument("--temporal_anchor_confidence_guard_min_abs_delta", type=float, default=0.02)
    ap.add_argument("--temporal_anchor_confidence_guard_min_support", type=float, default=0.0)
    ap.add_argument("--temporal_anchor_confidence_guard_scale", type=float, default=0.0)
    ap.add_argument(
        "--temporal_interaction_guard_mode",
        choices=["off", "duplicate_only_v0", "guidance_bucket_only_v0", "duplicate_plus_guidance_v0", "full_v0"],
        default="off",
        help="Default-off Temporal support calibration for duplicate/guidance interactions; diagnostic only unless promoted.",
    )
    ap.add_argument("--temporal_interaction_duplicate_threshold", type=float, default=0.50)
    ap.add_argument("--temporal_interaction_duplicate_scale", type=float, default=0.50)
    ap.add_argument("--temporal_interaction_explicit_scale", type=float, default=0.85)
    ap.add_argument("--temporal_interaction_non_explicit_scale", type=float, default=0.60)
    ap.add_argument("--temporal_interaction_guidance_active_scale", type=float, default=0.50)
    ap.add_argument("--temporal_interaction_conflict_scale", type=float, default=0.75)
    ap.add_argument("--temporal_interaction_min_support", type=float, default=0.0)
    ap.add_argument(
        "--temporal_reliability_memory_mode",
        choices=["off", "history_bucket_shrink_v0", "history_bucket_abstain_v0"],
        default="off",
        help="Default-off prior-only Temporal support calibration from historical row-level Temporal marginal effects; diagnostic only unless promoted.",
    )
    ap.add_argument("--temporal_reliability_min_history", type=int, default=4)
    ap.add_argument("--temporal_reliability_tau", type=float, default=0.01)
    ap.add_argument("--temporal_reliability_min_scale", type=float, default=0.35)
    ap.add_argument("--temporal_reliability_duplicate_threshold", type=float, default=0.50)
    ap.add_argument("--temporal_reliability_support_medium_threshold", type=float, default=0.10)
    ap.add_argument("--temporal_reliability_support_high_threshold", type=float, default=0.35)
    ap.add_argument("--temporal_reliability_abstain_trust_threshold", type=float, default=-0.75)
    ap.add_argument(
        "--temporal_state_analog_mode",
        choices=["off", "state_analog_blend_v0", "state_analog_replace_v0"],
        default="off",
        help="Default-off state/regime analog Temporal memory candidate using prior realized anchor residuals; diagnostic only unless promoted.",
    )
    ap.add_argument("--temporal_state_analog_min_history", type=int, default=6)
    ap.add_argument("--temporal_state_analog_neighbor_k", type=int, default=8)
    ap.add_argument("--temporal_state_analog_history_cap", type=int, default=48)
    ap.add_argument("--temporal_state_analog_distance_tau", type=float, default=1.0)
    ap.add_argument("--temporal_state_analog_var_tau", type=float, default=0.03)
    ap.add_argument("--temporal_state_analog_neff_scale", type=float, default=4.0)
    ap.add_argument("--temporal_state_analog_support_scale", type=float, default=1.0)
    ap.add_argument("--temporal_state_analog_blend_weight", type=float, default=0.50)
    ap.add_argument(
        "--temporal_action_gate_mode",
        choices=["off", "knn_shrink_v0", "knn_abstain_v0"],
        default="off",
        help="Default-off prior-only kNN harm-risk gate for the Temporal action; diagnostic only unless promoted.",
    )
    ap.add_argument("--temporal_action_gate_min_history", type=int, default=8)
    ap.add_argument("--temporal_action_gate_neighbor_k", type=int, default=12)
    ap.add_argument("--temporal_action_gate_tau", type=float, default=0.003)
    ap.add_argument("--temporal_action_gate_min_scale", type=float, default=0.35)
    ap.add_argument("--temporal_action_gate_abstain_effect_threshold", type=float, default=-0.00025)
    ap.add_argument("--temporal_action_gate_abstain_win_rate_threshold", type=float, default=0.45)
    ap.add_argument(
        "--action_gate_history_scope",
        choices=["company_local", "panel_prior"],
        default="company_local",
        help="History source for default-off action gates. panel_prior filters all supplied/completed rows to quarters strictly before the target quarter.",
    )
    ap.add_argument(
        "--action_gate_panel_history_csv",
        default="",
        help="Optional retained-off quarterly CSV used as order-independent panel-prior action-gate history.",
    )
    ap.add_argument(
        "--arbitration_mode",
        choices=["simple_weighted", "current", "current_gap_guard", "current_consensus_guard", EVIDENCE_ORTHOGONAL_ARBITRATION_MODE, CURRENT_EVIDENCE_ORTHOGONAL_ARBITRATION_MODE, INTEGRATED_EXPERT_ARBITRATION_MODE, SHARED_RESIDUAL_BACKBONE_MODE],
        default="current",
    )
    ap.add_argument(
        "--evidence_orthogonal_duplicate_threshold",
        type=float,
        default=0.35,
        help="Duplicate-ratio threshold for default-off evidence_orthogonal_arbitration_v0.",
    )
    ap.add_argument(
        "--evidence_orthogonal_base_min_scale",
        type=float,
        default=0.50,
        help="Minimum compressed-base support scale when base duplicates Intrinsic/Temporal evidence on guidance-bearing rows.",
    )
    ap.add_argument(
        "--evidence_orthogonal_guidance_min_scale",
        type=float,
        default=0.70,
        help="Minimum strict-guidance support scale when guidance duplicates Intrinsic/Temporal evidence.",
    )
    ap.add_argument("--evidence_orthogonal_base_strength", type=float, default=0.60)
    ap.add_argument("--evidence_orthogonal_guidance_strength", type=float, default=0.40)
    ap.add_argument(
        "--integrated_arbitration_floor_ratio",
        type=float,
        default=float("nan"),
        help="Optional override for integrated_expert_arbitration_v0 seasonal floor ratio; default derives it from retained-ID pretest no-guidance rows.",
    )
    ap.add_argument(
        "--integrated_arbitration_dev_tickers",
        nargs="*",
        default=list(INTEGRATED_ARBITRATION_DEV_TICKERS),
        help="Tickers used to derive the integrated v0 no-guidance seasonal floor ratio.",
    )
    ap.add_argument(
        "--integrated_arbitration_dev_end_quarter",
        default=INTEGRATED_ARBITRATION_DEV_END_QUARTER,
        help="Last quarter used when deriving the integrated v0 no-guidance seasonal floor ratio.",
    )
    ap.add_argument(
        "--frozen_best_stat_csv",
        default="",
        help="Optional frozen comparator map CSV with columns ticker,best_stat_model,best_stat_mae.",
    )
    ap.add_argument(
        "--anchor_selection_mode",
        choices=["frozen_prehistory_best", "online_historical_mae", "row_override_csv"],
        default="frozen_prehistory_best",
        help="Default preserves existing artifacts; online_historical_mae selects from past realized errors only; row_override_csv uses a precomputed row-level no-leakage selector output.",
    )
    ap.add_argument(
        "--anchor_override_csv",
        default="",
        help="CSV with ticker,quarter,best_stat_model for anchor_selection_mode=row_override_csv.",
    )
    ap.add_argument(
        "--anchor_online_min_history",
        type=int,
        default=4,
        help="Minimum past realized errors required before a model can win online anchor selection.",
    )
    ap.add_argument(
        "--anchor_online_score_metric",
        choices=["mae", "smape", "mape", "logabs"],
        default="mae",
        help="Past-error metric used by online_historical_mae anchor selection.",
    )
    ap.add_argument(
        "--anchor_online_window",
        type=int,
        default=0,
        help="If positive, use only this many previous quarters when scoring online anchor candidates.",
    )
    ap.add_argument(
        "--anchor_online_half_life",
        type=float,
        default=0.0,
        help="If positive, exponentially decay online anchor candidate errors by this half-life in quarters.",
    )
    ap.add_argument(
        "--anchor_online_same_quarter_weight",
        type=float,
        default=0.0,
        help="Extra multiplicative weight for historical rows with the same fiscal quarter number.",
    )
    ap.add_argument(
        "--anchor_guidance_regime_mode",
        choices=["off", "penalized_v1", "mismatch_penalty_v1"],
        default="off",
        help="Default-off diagnostic anchor scoring that conditions on guidance regime and penalizes guidance-dependent anchors without same-regime support.",
    )
    ap.add_argument(
        "--anchor_guidance_same_regime_min_history",
        type=int,
        default=4,
        help="Same-guidance-regime realized rows required before regime-specific anchor history is trusted.",
    )
    ap.add_argument(
        "--anchor_guidance_mismatch_penalty",
        type=float,
        default=0.08,
        help="Additive score penalty for guidance-dependent anchors on non-explicit rows without enough same-regime support.",
    )
    ap.add_argument(
        "--anchor_explicit_guidance_proximity_mode",
        choices=["off", "fixed_penalty_v1", "kernel_history_v1", "kernel_blend_v1"],
        default="off",
        help="Default-off root-level explicit-guidance anchor scoring using current candidate/guid_mid divergence.",
    )
    ap.add_argument(
        "--anchor_explicit_guidance_proximity_weight",
        type=float,
        default=0.0,
        help="Penalty weight used only by anchor_explicit_guidance_proximity_mode=fixed_penalty_v1.",
    )
    ap.add_argument(
        "--anchor_explicit_guidance_kernel_min_history",
        type=int,
        default=4,
        help="Minimum explicit numeric history rows for kernel_history_v1/kernel_blend_v1 candidate scoring.",
    )
    ap.add_argument(
        "--anchor_explicit_guidance_kernel_band",
        type=float,
        default=0.20,
        help="Abs-log candidate/guid_mid bandwidth for kernel_history_v1/kernel_blend_v1 anchor scoring.",
    )
    ap.add_argument(
        "--anchor_explicit_guidance_kernel_shrink_k",
        type=float,
        default=8.0,
        help="Effective-history shrinkage for kernel_blend_v1; larger values keep more weight on rolling history.",
    )
    ap.add_argument(
        "--anchor_robust_momentum_mode",
        choices=["off", "median_naive_auto_ma"],
        default="off",
        help="Default-off robust momentum anchor candidate for diagnostic regime-aware anchor selection.",
    )
    ap.add_argument(
        "--guidance_quality_guardrail_mode",
        choices=["off", "unified_v1", "explicit_strong_anchor_history_v1"],
        default="off",
        help="Post-blend guidance-quality guardrail for non-explicit guidance rows; default off preserves older artifacts.",
    )
    ap.add_argument(
        "--guidance_expert_mode",
        choices=["off", "explicit_history_trust_v1", "explicit_history_trust_non_guidance_anchor_v1"],
        default="off",
        help="Default-off candidate guidance expert. explicit_history_trust_v1 abstains unless current-row explicit numeric guidance exists; non_guidance_anchor_v1 also abstains when the selected anchor already uses guidance.",
    )
    ap.add_argument(
        "--guidance_expert_min_history",
        type=int,
        default=4,
        help="Minimum prior anchor-error rows used to activate the candidate guidance expert at full strength.",
    )
    ap.add_argument(
        "--guidance_expert_history_tau",
        type=float,
        default=0.08,
        help="Abs-log-error decay scale for the candidate guidance expert history-trust support.",
    )
    ap.add_argument(
        "--guidance_expert_support_scale",
        type=float,
        default=1.0,
        help="Global support multiplier for the candidate guidance expert.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_mode",
        choices=["off", "signed_strength_v1", REGIME_AWARE_BASE_ANCHOR_NATIVE_DELTA_MODE, REGIME_AWARE_BASE_ANCHOR_SCALED_DELTA_MODE],
        default="off",
        help="Opt-in CAME-RAM base-anchor memory update. signed_strength_v1 preserves retained post-panel behavior; signed_strength_delta_v0 records the same no-guidance memory correction as a pre-final diagnostic delta; signed_strength_scaled_delta_v0 scales that delta using shared history/gap/evidence confidence.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_proposal_csv",
        default="",
        help="CSV keyed by ticker,quarter containing the no-guidance RAM proposal used when regime_aware_base_anchor_mode is enabled.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_proposal_pred_col",
        default=REGIME_AWARE_BASE_ANCHOR_DEFAULT_PRED_COL,
        help="Prediction column in --regime_aware_base_anchor_proposal_csv.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_proposal_filter_col",
        default="",
        help="Optional column used to filter proposal CSV before ticker-quarter alignment, e.g. portfolio_policy.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_proposal_filter_value",
        default="",
        help="Value paired with --regime_aware_base_anchor_proposal_filter_col.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_min_history",
        type=int,
        default=6,
        help="Minimum prior no-guidance rows for signed-strength CAME-RAM admission.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_signed_strength_threshold",
        type=float,
        default=0.02,
        help="Prior signed log-error threshold for signed_strength_v1 CAME-RAM admission.",
    )
    ap.add_argument(
        "--regime_aware_base_anchor_upward_ratio",
        type=float,
        default=1.0,
        help="Proposal must be at least this multiple of the current CAME prediction to activate.",
    )
    ap.add_argument("--output_dir", default="output/csais_rawcard_direct_experts_v1_all12")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    expert_contract = str(args.expert_contract)
    active_experts = tuple(DIRECT_EXPERT_CONTRACTS[expert_contract])
    default_method_family = "rawcard_direct_experts_v1" if str(args.gate_mode) == "company_local" else f"rawcard_direct_experts_v2_{str(args.gate_mode)}"
    if expert_contract != "full":
        default_method_family = f"{default_method_family}_{expert_contract}"
    if str(args.arbitration_mode) in {EVIDENCE_ORTHOGONAL_ARBITRATION_MODE, CURRENT_EVIDENCE_ORTHOGONAL_ARBITRATION_MODE, INTEGRATED_EXPERT_ARBITRATION_MODE, SHARED_RESIDUAL_BACKBONE_MODE}:
        default_method_family = f"{default_method_family}_{str(args.arbitration_mode)}"
    if str(args.guidance_expert_mode) != "off":
        default_method_family = f"{default_method_family}_{str(args.guidance_expert_mode)}"
    if str(args.regime_aware_base_anchor_mode) != "off":
        default_method_family = f"{default_method_family}_regime_aware_base_anchor_{str(args.regime_aware_base_anchor_mode)}"
    method_family = str(args.method_family_label or default_method_family)
    report_start_key = _quarter_key(str(args.report_start_quarter))
    report_end_key = _quarter_key(str(args.report_end_quarter))
    requested = {ticker.upper() for ticker in args.tickers}
    out_dir = Path(resolve_repo_path(args.output_dir, str(project_root)))
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_best_stat_map = _load_frozen_best_stat_map(args.frozen_best_stat_csv, project_root)
    anchor_override_map = _load_anchor_override_map(args.anchor_override_csv, project_root)
    if str(args.anchor_selection_mode) == "row_override_csv" and not anchor_override_map:
        raise ValueError("anchor_selection_mode=row_override_csv requires a non-empty --anchor_override_csv")
    regime_aware_base_anchor_proposals = _load_regime_aware_base_anchor_proposals(
        args.regime_aware_base_anchor_proposal_csv,
        project_root,
        str(args.regime_aware_base_anchor_proposal_pred_col),
        str(args.regime_aware_base_anchor_proposal_filter_col),
        str(args.regime_aware_base_anchor_proposal_filter_value),
    )
    if str(args.regime_aware_base_anchor_mode) != "off" and not regime_aware_base_anchor_proposals:
        raise ValueError("regime_aware_base_anchor_mode requires a non-empty --regime_aware_base_anchor_proposal_csv")
    action_gate_panel_history_seed = _load_action_gate_panel_history(str(args.action_gate_panel_history_csv), project_root)

    backbone_lookup = _load_backbone_lookup(Path(resolve_repo_path(args.native_backbone_csv, str(project_root))), requested)
    card_groups = _load_card_groups(Path(resolve_repo_path(args.native_card_table_jsonl, str(project_root))), requested)
    exp = json.loads(Path(resolve_repo_path(args.experiment_config, str(project_root))).read_text(encoding="utf-8"))
    integrated_arbitration_floor_derivation: Dict[str, Any] = {
        "floor_ratio": float("nan"),
        "floor_ratio_source": "not_used",
        "support_rows": 0,
        "median_abs_log_error": float("nan"),
    }
    if str(args.arbitration_mode) == INTEGRATED_EXPERT_ARBITRATION_MODE:
        if np.isfinite(float(args.integrated_arbitration_floor_ratio)) and float(args.integrated_arbitration_floor_ratio) > 0.0:
            integrated_arbitration_floor_derivation = {
                "floor_ratio": float(args.integrated_arbitration_floor_ratio),
                "floor_ratio_source": "cli_override",
                "support_rows": 0,
                "median_abs_log_error": float("nan"),
            }
        else:
            integrated_arbitration_floor_derivation = _derive_integrated_arbitration_floor_ratio(
                exp=exp,
                project_root=project_root,
                dev_tickers=list(args.integrated_arbitration_dev_tickers),
                dev_end_quarter=str(args.integrated_arbitration_dev_end_quarter),
            )
    integrated_arbitration_floor_ratio = float(_safe_float(integrated_arbitration_floor_derivation.get("floor_ratio"), float("nan")))

    all_quarterly: List[pd.DataFrame] = []
    company_summaries: List[Dict[str, Any]] = []
    coverage_records: List[Dict[str, Any]] = []
    coverage_skip_records: List[Dict[str, Any]] = []
    panel_action_gate_history: List[Dict[str, Any]] = []

    for company in exp.get("companies", []):
        ticker = str(company.get("ticker") or "").upper()
        if requested and ticker not in requested:
            continue
        panel, best_stat_model, best_stat_mae, prehist_best_model, prehist_best_mae, prehist_anchor_error_history = _prepare_company_panel(company, project_root)
        if ticker in frozen_best_stat_map:
            best_stat_model = str(frozen_best_stat_map[ticker]["best_stat_model"])
            best_stat_mae = float(frozen_best_stat_map[ticker]["best_stat_mae"])
        panel = _add_robust_momentum_anchor_candidate(panel, str(args.anchor_robust_momentum_mode))
        model_cols = [col for col in panel.columns if col.startswith("pred__") and col not in STAT_EXCLUDE]
        company_report_start_key = max(_quarter_key(str(company.get("evaluation_start_fq") or args.report_start_quarter)), report_start_key)
        company_report_end_key = min(_quarter_key(str(company.get("evaluation_end_fq") or args.report_end_quarter)), report_end_key)
        expected_report_quarters = _quarter_labels_between(company_report_start_key, company_report_end_key)
        panel_report_quarters = [
            str(q)
            for q in panel["quarter"].astype(str).tolist()
            if company_report_start_key <= _quarter_key(str(q)) <= company_report_end_key
        ]
        company_skip_records: List[Dict[str, Any]] = []
        anchor_model_error_history: Dict[str, List[Dict[str, Any]]] = {str(col): [] for col in model_cols}
        if str(args.anchor_selection_mode) == "online_historical_mae" and not panel.empty:
            stat_df_for_anchor = _load_stat_predictions(Path(resolve_repo_path(str(company.get("stat_baseline_predictions_csv")), str(project_root))))
            stat_df_for_anchor = _add_robust_momentum_anchor_candidate(stat_df_for_anchor, str(args.anchor_robust_momentum_mode))
            anchor_model_error_history, online_seed_anchor_errors = _seed_online_anchor_history(
                stat_df_for_anchor,
                model_cols,
                str(panel.iloc[0].get("quarter") or ""),
                int(args.anchor_online_min_history),
                score_metric=str(args.anchor_online_score_metric),
                window=int(args.anchor_online_window),
                half_life=float(args.anchor_online_half_life),
                same_quarter_weight=float(args.anchor_online_same_quarter_weight),
                guidance_regime_mode=str(args.anchor_guidance_regime_mode),
                guidance_same_regime_min_history=int(args.anchor_guidance_same_regime_min_history),
                guidance_mismatch_penalty=float(args.anchor_guidance_mismatch_penalty),
                explicit_guidance_proximity_mode=str(args.anchor_explicit_guidance_proximity_mode),
                explicit_guidance_proximity_weight=float(args.anchor_explicit_guidance_proximity_weight),
                explicit_guidance_kernel_min_history=int(args.anchor_explicit_guidance_kernel_min_history),
                explicit_guidance_kernel_band=float(args.anchor_explicit_guidance_kernel_band),
                explicit_guidance_kernel_shrink_k=float(args.anchor_explicit_guidance_kernel_shrink_k),
            )
            prehist_anchor_error_history = list(online_seed_anchor_errors)
        quarterly_rows: List[Dict[str, Any]] = []
        actual_hist: List[float] = []
        quarter_hist: List[str] = []
        anchor_error_history: List[Dict[str, Any]] = list(prehist_anchor_error_history)
        native_memory_history: List[Dict[str, Any]] = []
        retrieve_cache: Dict[str, Dict[str, Any]] = {}

        for _, row in panel.iterrows():
            current_q = _quarter_number(str(row.get("quarter") or ""))
            current_guidance = str(row.get("guidance_availability") or "none")
            current_guidance_bucket = _row_guidance_bucket(row)
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
            anchor_override_pred = float("nan")

            if str(args.anchor_selection_mode) == "online_historical_mae":
                anchor_col, anchor_history_mae, anchor_history_n, anchor_selection_reason = _select_online_anchor_col(
                    row,
                    model_cols,
                    anchor_model_error_history,
                    int(args.anchor_online_min_history),
                    score_metric=str(args.anchor_online_score_metric),
                    window=int(args.anchor_online_window),
                    half_life=float(args.anchor_online_half_life),
                    same_quarter_weight=float(args.anchor_online_same_quarter_weight),
                    guidance_regime_mode=str(args.anchor_guidance_regime_mode),
                    guidance_same_regime_min_history=int(args.anchor_guidance_same_regime_min_history),
                    guidance_mismatch_penalty=float(args.anchor_guidance_mismatch_penalty),
                    explicit_guidance_proximity_mode=str(args.anchor_explicit_guidance_proximity_mode),
                    explicit_guidance_proximity_weight=float(args.anchor_explicit_guidance_proximity_weight),
                    explicit_guidance_kernel_min_history=int(args.anchor_explicit_guidance_kernel_min_history),
                    explicit_guidance_kernel_band=float(args.anchor_explicit_guidance_kernel_band),
                    explicit_guidance_kernel_shrink_k=float(args.anchor_explicit_guidance_kernel_shrink_k),
                )
            elif str(args.anchor_selection_mode) == "row_override_csv":
                override = anchor_override_map.get((ticker, str(row.get("quarter") or "")), {})
                override_model = str(override.get("best_stat_model") or "")
                anchor_col = f"pred__{override_model}" if override_model else ""
                anchor_override_pred = float(_safe_float(override.get("anchor_override_pred"), float("nan")))
                anchor_history_mae = float(_safe_float(override.get("selector_score"), float("nan")))
                anchor_history_n = int(_safe_float(override.get("selector_history_n"), 0.0))
                anchor_selection_reason = str(override.get("selector_reason") or "row_override_csv_missing_override")
            else:
                anchor_col = f"pred__{prehist_best_model}"
                anchor_history_mae = float(prehist_best_mae) if np.isfinite(prehist_best_mae) else float(best_stat_mae)
                anchor_history_n = 0
                anchor_selection_reason = "frozen_prehistory_best"
            if str(args.anchor_selection_mode) == "row_override_csv" and np.isfinite(anchor_override_pred):
                anchor_value = float(anchor_override_pred)
            else:
                anchor_value = _safe_float(row.get(anchor_col), float("nan")) if anchor_col in model_cols else float("nan")
            if not np.isfinite(anchor_value) and str(args.missing_anchor_policy) != "skip":
                fallback_col, fallback_value, fallback_reason = _select_missing_anchor_fallback(
                    row,
                    model_cols,
                    str(args.missing_anchor_policy),
                )
                if np.isfinite(fallback_value):
                    anchor_col = str(fallback_col)
                    anchor_value = float(fallback_value)
                    anchor_history_mae = float("nan")
                    anchor_history_n = 0
                    anchor_selection_reason = f"{anchor_selection_reason}__missing_anchor_{fallback_reason}"
            if not np.isfinite(anchor_value):
                company_skip_records.append(
                    {
                        "ticker": ticker,
                        "quarter": str(row.get("quarter") or ""),
                        "reason": "missing_or_nonfinite_anchor",
                        "anchor_col": str(anchor_col),
                        "anchor_selection_mode": str(args.anchor_selection_mode),
                        "missing_anchor_policy": str(args.missing_anchor_policy),
                        "prehistory_best_model": str(prehist_best_model),
                        "anchor_value": float(anchor_value),
                    }
                )
                continue
            anchor_pred = float(anchor_value)
            anchor_model = str(anchor_col)[len("pred__") :] if str(anchor_col).startswith("pred__") else str(anchor_col)
            anchor_uncertainty = float(_safe_float(anchor_error_state.get("anchor_uncertainty_proxy"), 0.0))
            anchor_diag_state = {
                "anchor_uncertainty": anchor_uncertainty,
                "anchor_blend_weight": 0.0,
                "anchor_top1_gap_ratio": 0.0,
            }

            factor_map = _factorized_internal_features(row, regime, anchor_diag_state)
            internal_state = _internal_features(row)
            guidance_lock = _guidance_lock(guidance_dict)
            state_analog_temporal_features = _state_analog_temporal_feature_map(
                regime=regime,
                row=row,
                current_quarter=str(row.get("quarter") or ""),
                anchor_uncertainty=float(anchor_uncertainty),
                internal_strength=float(_safe_float(internal_state.get("internal_strength"), 0.0)),
            )

            raw_features = {name: float(_safe_float(value, 0.0)) for name, value in zip(RAW_SHOCK_FEATURES, _build_shock_features(row, regime, anchor_diag_state))}
            factor_features = {name: float(_safe_float(factor_map.get(name), 0.0)) for name in FACTORIZED_SHOCK_FEATURES}
            guidance_expert = _guidance_expert_candidate(
                row=row,
                anchor_pred=float(anchor_pred),
                anchor_model=str(anchor_model),
                guidance_lock=float(guidance_lock),
                anchor_error_state=anchor_error_state,
                mode=str(args.guidance_expert_mode),
                min_history=int(args.guidance_expert_min_history),
                history_tau=float(args.guidance_expert_history_tau),
                support_scale=float(args.guidance_expert_support_scale),
            )

            hist_processed = list(quarterly_rows)
            raw_result = _predict_compressed_candidate(
                hist_processed,
                feature_names=RAW_SHOCK_FEATURES,
                current_features=raw_features,
                target_col="shock_target_log",
                alpha=float(args.raw_alpha),
                min_train=int(args.candidate_min_train),
                shrink_k=float(args.candidate_shrink_k),
                max_abs_log_delta=float(args.shock_max_abs_log_delta),
            )
            factor_result = _predict_compressed_candidate(
                hist_processed,
                feature_names=FACTORIZED_SHOCK_FEATURES,
                current_features=factor_features,
                target_col="shock_target_log",
                alpha=float(args.factor_alpha),
                min_train=int(args.candidate_min_train),
                shrink_k=float(args.candidate_shrink_k),
                max_abs_log_delta=float(args.shock_max_abs_log_delta),
            )
            evidence_gate = (1.0 - np.exp(-float(internal_state["internal_strength"]) / 1.5)) * (0.4 + 0.6 * abs(float(internal_state["internal_balance"])))
            evidence_gate = float(0.0 if not np.isfinite(evidence_gate) else evidence_gate)
            base_support = float(_clip(raw_result["support"] * evidence_gate * (0.35 + 0.65 * anchor_uncertainty) * (1.0 - 0.6 * guidance_lock) * float(args.candidate_base_scale), 0.0, 1.0))
            raw_delta = float(raw_result["pred_raw"] if np.isfinite(raw_result["pred_raw"]) else 0.0) * base_support
            factor_delta = float(factor_result["pred_raw"] if np.isfinite(factor_result["pred_raw"]) else 0.0) * base_support
            compressed_base_delta = float(
                sum(value * weight for value, weight in [(raw_delta, raw_result["support"]), (factor_delta, factor_result["support"])] if np.isfinite(value) and weight > 0.0)
                / max(sum(weight for value, weight in [(raw_delta, raw_result["support"]), (factor_delta, factor_result["support"])] if np.isfinite(value) and weight > 0.0), EPS)
            ) if ((raw_result["support"] > 0.0 and np.isfinite(raw_delta)) or (factor_result["support"] > 0.0 and np.isfinite(factor_delta))) else 0.0

            native_backbone_row = backbone_lookup.get((ticker, str(row.get("quarter") or "")), {})
            observed_quarter = str(native_backbone_row.get("observed_fiscal_quarter") or "")
            native_surface = _build_native_surface(
                ticker=ticker,
                observed_quarter=observed_quarter,
                target_quarter=str(row.get("quarter") or ""),
                group_map=card_groups,
            )
            native_current_record = {
                "ticker": ticker,
                **row.to_dict(),
                **guidance_dict,
                **regime,
                **native_surface,
                "target_fiscal_q": float(current_q),
                "target_delta_log": float("nan"),
                "guidance_lock": float(guidance_lock),
                "anchor_uncertainty": float(anchor_uncertainty),
                "forward_rows": list(native_surface.get("forward_rows") or []),
                "state_analog_temporal_features": dict(state_analog_temporal_features),
            }
            native_forward_rows = list(native_surface.get("forward_rows") or [])
            native_context_rows: List[Dict[str, Any]] = []
            temporal_context_memory_mode = str(args.temporal_context_memory_mode)
            if temporal_context_memory_mode in {"retrieve_cards", "typed_retrieval", "typed_reliability", "typed_soft_agreement"}:
                retrieve_payload = _load_retrieve_payload(row.get("retrieve_path"), retrieve_cache)
                native_context_rows = _retrieve_context_card_rows(retrieve_payload)
            native_current_record["context_rows"] = list(native_context_rows)
            temporal_evidence_filter_mode = str(args.temporal_evidence_filter_mode)
            temporal_evidence_filter_scope = str(args.temporal_evidence_filter_scope)
            forward_filter_mode = temporal_evidence_filter_mode if temporal_evidence_filter_scope in {"all", "forward_only"} else "off"
            context_filter_mode = temporal_evidence_filter_mode if temporal_evidence_filter_scope in {"all", "context_only"} else "off"
            temporal_forward_rows, temporal_forward_filter_diag = _filter_temporal_evidence_rows(
                native_forward_rows,
                forward_filter_mode,
            )
            temporal_context_rows, temporal_context_filter_diag = _filter_temporal_evidence_rows(
                native_context_rows,
                context_filter_mode,
            )
            temporal_forward_filter_diag["scope"] = temporal_evidence_filter_scope
            temporal_context_filter_diag["scope"] = temporal_evidence_filter_scope
            temporal_context_quality_mode = str(args.temporal_context_quality_mode)
            temporal_context_quality_diag = _temporal_evidence_quality_diag(
                temporal_context_rows,
                temporal_context_quality_mode,
            )
            temporal_context_quality_scale = _temporal_evidence_quality_scale(
                temporal_context_quality_diag,
                float(args.temporal_context_quality_weight),
            )
            native_current_record["temporal_forward_rows"] = list(temporal_forward_rows)
            native_current_record["temporal_context_rows"] = list(temporal_context_rows)
            native_current_record["temporal_context_quality_scale"] = float(temporal_context_quality_scale)
            native_current_record["temporal_context_quality_weak_score"] = float(_safe_float(temporal_context_quality_diag.get("weak_score"), 0.0))

            intrinsic_mode = str(args.intrinsic_mode)
            intrinsic_target_mode = str(args.intrinsic_target_mode)
            intrinsic_target_col = "target_delta_log"
            if intrinsic_target_mode == "post_base_temporal_guidance_residual_v1":
                intrinsic_target_col = "intrinsic_residual_target_log"
            elif intrinsic_target_mode == "partitioned_residual_v1":
                intrinsic_target_col = "intrinsic_partitioned_residual_target_log"
            intrinsic_additive_result = _predict_native_card_additive_candidate(
                train_rows=native_memory_history,
                current_cards=native_forward_rows,
                target_col=intrinsic_target_col,
                admissibility_mode="conservative_admissibility_v2",
                alpha=float(args.intrinsic_alpha),
                min_train=int(args.intrinsic_min_train),
                shrink_k=float(args.intrinsic_shrink_k),
            )
            intrinsic_grouped_result = _predict_intrinsic_grouped_direct_candidate(
                train_rows=native_memory_history,
                current_cards=native_forward_rows,
                target_col=intrinsic_target_col,
                admissibility_mode="conservative_admissibility_v2",
                alpha=float(args.intrinsic_alpha),
                min_train=int(args.intrinsic_min_train),
                shrink_k=float(args.intrinsic_shrink_k),
                max_strict_features=int(args.intrinsic_grouped_max_strict_features),
                max_loose_features=int(args.intrinsic_grouped_max_loose_features),
                min_feature_occurrence=int(args.intrinsic_grouped_min_feature_occurrence),
            )
            intrinsic_additive_safety = _apply_safety_guard(
                delta_pred=float(intrinsic_additive_result["pred"]),
                features={name: float(_safe_float(native_surface.get(name), 0.0)) for name in RAW_CARD_FEATURE_NAMES},
                top_contribs=list(intrinsic_additive_result.get("top_contribs", [])),
                forward_row_count=len(native_forward_rows),
                safety_mode="signguard_only",
                semantic_delta_pred=float("nan"),
            )
            intrinsic_grouped_safety = _apply_safety_guard(
                delta_pred=float(intrinsic_grouped_result["pred"]),
                features={name: float(_safe_float(native_surface.get(name), 0.0)) for name in RAW_CARD_FEATURE_NAMES},
                top_contribs=list(intrinsic_grouped_result.get("top_contribs", [])),
                forward_row_count=len(native_forward_rows),
                safety_mode="signguard_only",
                semantic_delta_pred=float("nan"),
            )
            memory_diag = _memory_diag(native_current_record, native_memory_history, args)
            conflict_ratio = float(_safe_float(native_surface.get("fwd_conflict_ratio"), 0.0))
            semantic_clarity = float(_safe_float(intrinsic_additive_safety.get("semantic_clarity"), 0.0))

            additive_delta = float(_safe_float(intrinsic_additive_safety.get("delta_pred"), 0.0))
            additive_support = _intrinsic_effective_support(
                raw_support=float(_safe_float(intrinsic_additive_result.get("support"), 0.0)),
                coverage=float(_safe_float(intrinsic_additive_result.get("coverage"), 1.0)),
                semantic_clarity=semantic_clarity,
                conflict_ratio=conflict_ratio,
                safety_scale=float(_safe_float(intrinsic_additive_safety.get("safety_scale"), 0.0)),
                delta_pred=additive_delta,
            )
            grouped_delta = float(_safe_float(intrinsic_grouped_safety.get("delta_pred"), 0.0))
            grouped_coverage = float(_safe_float(intrinsic_grouped_result.get("coverage"), 0.0))
            grouped_support = _intrinsic_effective_support(
                raw_support=float(_safe_float(intrinsic_grouped_result.get("support"), 0.0)),
                coverage=grouped_coverage,
                semantic_clarity=semantic_clarity,
                conflict_ratio=conflict_ratio,
                safety_scale=float(_safe_float(intrinsic_grouped_safety.get("safety_scale"), 0.0)),
                delta_pred=grouped_delta,
            )
            segment_bridge_result = _predict_segment_bridge_candidate(
                train_rows=native_memory_history,
                current_cards=native_forward_rows,
                target_col=intrinsic_target_col,
                admissibility_mode="conservative_admissibility_v2",
                alpha=float(args.segment_bridge_alpha),
                min_train=int(args.segment_bridge_min_train),
                shrink_k=float(args.segment_bridge_shrink_k),
            )
            segment_bridge_delta = float(
                _safe_float(segment_bridge_result.get("pred_raw"), 0.0)
                if np.isfinite(_safe_float(segment_bridge_result.get("pred_raw"), float("nan")))
                else 0.0
            )
            segment_bridge_coverage = float(_safe_float(segment_bridge_result.get("coverage"), 0.0))
            segment_bridge_support = _intrinsic_effective_support(
                raw_support=float(_safe_float(segment_bridge_result.get("support"), 0.0)),
                coverage=segment_bridge_coverage,
                semantic_clarity=semantic_clarity,
                conflict_ratio=conflict_ratio,
                safety_scale=1.0,
                delta_pred=segment_bridge_delta,
            )

            intrinsic_result = intrinsic_additive_result
            intrinsic_safety = intrinsic_additive_safety
            intrinsic_delta = float(additive_delta)
            intrinsic_support = float(additive_support)
            intrinsic_hybrid_lambda = 0.0
            segment_bridge_lambda = 0.0
            segment_bridge_sign_match = 0.5
            if intrinsic_mode == "grouped_ridge_v3":
                intrinsic_result = intrinsic_grouped_result
                intrinsic_safety = intrinsic_grouped_safety
                intrinsic_delta = float(grouped_delta)
                intrinsic_support = float(grouped_support)
            elif intrinsic_mode == "additive_grouped_hybrid_v3":
                add_group_sign_match = _expert_sign_match_score(additive_delta, grouped_delta)
                grouped_advantage = float(grouped_support - additive_support)
                if add_group_sign_match >= 1.0:
                    intrinsic_hybrid_lambda = float(_clip(0.15 + 0.35 * grouped_support, 0.0, 0.45))
                elif grouped_advantage > 0.15 and grouped_coverage >= 0.65:
                    intrinsic_hybrid_lambda = float(_clip(0.10 + 0.50 * grouped_advantage, 0.0, 0.35))
                intrinsic_delta = float(additive_delta + intrinsic_hybrid_lambda * (grouped_delta - additive_delta))
                intrinsic_support = float(
                    _clip(
                        max(additive_support, grouped_support * (0.7 if add_group_sign_match >= 1.0 else 0.45)),
                        0.0,
                        1.0,
                    )
                )
                intrinsic_result = dict(intrinsic_additive_result)
                intrinsic_result["coverage"] = float(
                    _weighted_average(
                        [
                            (float(_safe_float(intrinsic_additive_result.get("coverage"), 1.0)), 1.0 - intrinsic_hybrid_lambda),
                            (float(grouped_coverage), intrinsic_hybrid_lambda),
                        ]
                    )
                )
                intrinsic_result["top_contribs"] = list(intrinsic_additive_result.get("top_contribs", []))

            segment_bridge_mode = str(args.segment_bridge_mode)
            if segment_bridge_mode == "intrinsic_hybrid_v1" and segment_bridge_support > 0.0:
                intrinsic_was_zero = abs(float(intrinsic_delta)) <= EPS
                segment_bridge_sign_match = _expert_sign_match_score(intrinsic_delta, segment_bridge_delta)
                bridge_blend_weight = float(_clip(args.segment_bridge_blend_weight, 0.0, 0.6))
                if intrinsic_was_zero:
                    segment_bridge_lambda = float(_clip(bridge_blend_weight * (0.5 + 0.5 * segment_bridge_support), 0.0, bridge_blend_weight))
                elif segment_bridge_sign_match >= 1.0:
                    segment_bridge_lambda = float(_clip(bridge_blend_weight * segment_bridge_support, 0.0, bridge_blend_weight))
                else:
                    support_advantage = max(float(segment_bridge_support - intrinsic_support), 0.0)
                    if support_advantage > 0.15 and segment_bridge_coverage >= 0.5:
                        segment_bridge_lambda = float(_clip(0.5 * bridge_blend_weight * support_advantage, 0.0, 0.15))
                intrinsic_delta = float(intrinsic_delta + segment_bridge_lambda * (segment_bridge_delta - intrinsic_delta))
                intrinsic_support = float(
                    _clip(
                        max(
                            intrinsic_support,
                            segment_bridge_support
                            * (0.75 if (segment_bridge_sign_match >= 1.0 or intrinsic_was_zero) else 0.35),
                        ),
                        0.0,
                        1.0,
                    )
                )

            temporal_result = _predict_temporal_graph_attention_direct_expert(
                train_rows=native_memory_history,
                current_rows=temporal_forward_rows,
                current_quarter=str(row.get("quarter") or ""),
                target_col="target_delta_log",
                min_train=int(args.temporal_min_train),
                history_cap=int(args.temporal_history_cap),
                quarter_top_k=int(args.temporal_top_k),
                quarter_temperature=float(args.temporal_temperature),
                item_top_k=int(args.temporal_item_top_k),
                item_temperature=float(args.temporal_item_temperature),
                same_quarter_bonus=float(args.temporal_same_quarter_bonus),
                time_decay_quarters=float(args.temporal_time_decay_quarters),
                var_tau=float(args.temporal_var_tau),
                neff_scale=float(args.temporal_neff_scale),
                directional_consistency_power=float(args.temporal_directional_consistency_power),
                attention_focus_power=float(args.temporal_attention_focus_power),
                max_abs_log_delta=float(args.temporal_max_abs_log_correction),
                score_mode=("attention_plus_same_quarter" if str(args.temporal_score_mode) == "current" else str(args.temporal_score_mode)),
                support_mode=str(args.temporal_support_mode),
                direction_mode=str(args.temporal_direction_mode),
                current_context_rows=(temporal_context_rows if temporal_context_memory_mode in {"typed_retrieval", "typed_reliability", "typed_soft_agreement"} else []),
                current_state=native_current_record,
                history_rows_field="temporal_forward_rows",
                context_rows_field="temporal_context_rows",
                context_score_weight=((float(args.temporal_context_retrieval_weight) * float(temporal_context_quality_scale)) if temporal_context_memory_mode in {"typed_retrieval", "typed_reliability", "typed_soft_agreement"} else 0.0),
                reliability_mode=(temporal_context_memory_mode if temporal_context_memory_mode in {"typed_reliability", "typed_soft_agreement"} else "off"),
                segment_score_weight=(float(args.temporal_segment_compat_weight) if temporal_context_memory_mode == "typed_reliability" else 0.0),
                context_support_weight=(float(args.temporal_context_support_weight) if temporal_context_memory_mode == "typed_reliability" else 0.0),
                novelty_shrink_weight=(float(args.temporal_novelty_shrink_weight) if temporal_context_memory_mode == "typed_reliability" else 0.0),
                guidance_trust_weight=(float(args.temporal_guidance_trust_weight) if temporal_context_memory_mode == "typed_reliability" else 0.0),
                segment_support_weight=(float(args.temporal_segment_support_weight) if temporal_context_memory_mode == "typed_reliability" else 0.0),
                soft_agreement_weight=(float(args.temporal_soft_agreement_weight) if temporal_context_memory_mode == "typed_soft_agreement" else 0.0),
            )
            temporal_delta = float(_safe_float(temporal_result.get("pred"), 0.0))
            temporal_support = float(_safe_float(temporal_result.get("support"), 0.0))
            temporal_forward_delta = float(temporal_delta)
            temporal_forward_support = float(temporal_support)
            temporal_context_result: Dict[str, Any] = {
                "pred": 0.0,
                "pred_raw": float("nan"),
                "train_count": 0,
                "support": 0.0,
                "support_pre_direction": 0.0,
                "direction_scale": 1.0,
                "effective_memory_count": 0.0,
                "directional_consistency": 0.0,
                "attention_focus": 0.0,
                "mean_direction_alignment": 0.0,
                "top_direction_alignment": 0.0,
                "pred_post_support": 0.0,
                "retrieved_quarters": [],
                "top_matches": [],
            }
            temporal_context_forward_sparsity = 0.0
            temporal_context_delta = 0.0
            temporal_context_support = 0.0
            temporal_context_guard_scale = 1.0
            temporal_context_guard_mode = str(args.temporal_context_guard_mode)
            weak_guidance_contexts = {"derived_weak_numeric", "no_total_revenue_guidance_but_forward_commentary"}
            temporal_context_guard_active = (
                temporal_context_guard_mode == "guidance_available_min_align" and str(current_guidance) not in {"", "none"}
            ) or (
                temporal_context_guard_mode == "weak_guidance_min_align" and str(current_guidance) in weak_guidance_contexts
            ) or (
                temporal_context_guard_mode == "derived_weak_min_align" and str(current_guidance) == "derived_weak_numeric"
            )
            if temporal_context_guard_active and temporal_support > 0.0:
                temporal_context_guard_scale = float(
                    _clip(
                        min(
                            float(_safe_float(temporal_result.get("mean_direction_alignment"), 1.0)),
                            float(_safe_float(temporal_result.get("top_direction_alignment"), 1.0)),
                        ),
                        0.0,
                        1.0,
                    )
                )
                temporal_delta = float(temporal_delta * temporal_context_guard_scale)
                temporal_support = float(temporal_support * temporal_context_guard_scale)
            temporal_forward_delta = float(temporal_delta)
            temporal_forward_support = float(temporal_support)
            if temporal_context_memory_mode == "retrieve_cards":
                temporal_context_result = _predict_temporal_graph_attention_direct_expert(
                    train_rows=native_memory_history,
                    current_rows=temporal_context_rows,
                    current_quarter=str(row.get("quarter") or ""),
                    target_col="target_delta_log",
                    min_train=int(args.temporal_min_train),
                    history_cap=int(args.temporal_history_cap),
                    quarter_top_k=int(args.temporal_top_k),
                    quarter_temperature=float(args.temporal_temperature),
                    item_top_k=int(args.temporal_item_top_k),
                    item_temperature=float(args.temporal_item_temperature),
                    same_quarter_bonus=float(args.temporal_same_quarter_bonus),
                    time_decay_quarters=float(args.temporal_time_decay_quarters),
                    var_tau=float(args.temporal_var_tau),
                    neff_scale=float(args.temporal_neff_scale),
                    directional_consistency_power=float(args.temporal_directional_consistency_power),
                    attention_focus_power=float(args.temporal_attention_focus_power),
                    max_abs_log_delta=float(args.temporal_max_abs_log_correction),
                    score_mode=("attention_plus_same_quarter" if str(args.temporal_score_mode) == "current" else str(args.temporal_score_mode)),
                    support_mode=str(args.temporal_support_mode),
                    direction_mode=str(args.temporal_direction_mode),
                    history_rows_field="temporal_context_rows",
                )
                temporal_context_forward_sparsity = float(
                    _clip(
                        (1.0 / max(1.0 + float(len(native_forward_rows)), EPS)) ** float(args.temporal_context_memory_sparsity_power),
                        0.0,
                        1.0,
                    )
                )
                temporal_context_delta = float(_safe_float(temporal_context_result.get("pred"), 0.0) * temporal_context_forward_sparsity)
                temporal_context_support = float(_safe_float(temporal_context_result.get("support"), 0.0) * temporal_context_forward_sparsity)
                temporal_pairs = [
                    (float(temporal_forward_delta), float(temporal_forward_support)),
                    (float(temporal_context_delta), float(temporal_context_support)),
                ]
                temporal_delta = float(_weighted_average(temporal_pairs))
                temporal_support = float(_clip(max(float(temporal_forward_support), float(temporal_context_support)), 0.0, 1.0))

            temporal_guidance_bucket_scale = _temporal_guidance_bucket_scale(
                guidance_bucket=str(current_guidance_bucket),
                mode=str(args.temporal_guidance_bucket_scale_mode),
                explicit_scale=float(args.temporal_explicit_guidance_scale),
                non_explicit_scale=float(args.temporal_non_explicit_guidance_scale),
                no_guidance_scale=float(args.temporal_no_guidance_scale),
            )
            temporal_delta = float(temporal_delta * temporal_guidance_bucket_scale)
            temporal_support = float(temporal_support * temporal_guidance_bucket_scale)
            temporal_interaction_guard_diag = _temporal_interaction_guard_scale(
                guidance_bucket=str(current_guidance_bucket),
                temporal_delta=float(temporal_delta),
                temporal_support=float(temporal_support),
                intrinsic_delta=float(intrinsic_delta),
                intrinsic_support=float(intrinsic_support),
                base_delta=float(compressed_base_delta),
                base_support=float(base_support),
                guidance_expert_active=bool(_safe_float(guidance_expert.get("active"), 0.0) > 0.0),
                mode=str(args.temporal_interaction_guard_mode),
                duplicate_threshold=float(args.temporal_interaction_duplicate_threshold),
                duplicate_scale=float(args.temporal_interaction_duplicate_scale),
                explicit_scale=float(args.temporal_interaction_explicit_scale),
                non_explicit_scale=float(args.temporal_interaction_non_explicit_scale),
                guidance_active_scale=float(args.temporal_interaction_guidance_active_scale),
                conflict_scale=float(args.temporal_interaction_conflict_scale),
                min_support=float(args.temporal_interaction_min_support),
            )
            temporal_interaction_guard_scale = float(_safe_float(temporal_interaction_guard_diag.get("scale"), 1.0))
            temporal_support = float(_clip(float(temporal_support) * temporal_interaction_guard_scale, 0.0, 1.0))
            temporal_reliability_diag = _temporal_history_reliability_scale(
                native_memory_history,
                row=row,
                temporal_delta=float(temporal_delta),
                temporal_support=float(temporal_support),
                intrinsic_delta=float(intrinsic_delta),
                intrinsic_support=float(intrinsic_support),
                mode=str(args.temporal_reliability_memory_mode),
                min_history=int(args.temporal_reliability_min_history),
                tau=float(args.temporal_reliability_tau),
                min_scale=float(args.temporal_reliability_min_scale),
                duplicate_threshold=float(args.temporal_reliability_duplicate_threshold),
                support_medium_threshold=float(args.temporal_reliability_support_medium_threshold),
                support_high_threshold=float(args.temporal_reliability_support_high_threshold),
                abstain_trust_threshold=float(args.temporal_reliability_abstain_trust_threshold),
            )
            temporal_reliability_scale = float(_safe_float(temporal_reliability_diag.get("scale"), 1.0))
            temporal_support = float(_clip(float(temporal_support) * temporal_reliability_scale, 0.0, 1.0))
            temporal_pre_state_analog_delta = float(temporal_delta)
            temporal_pre_state_analog_support = float(temporal_support)
            temporal_state_analog_result = _predict_state_analog_temporal_memory(
                native_memory_history,
                feature_map=state_analog_temporal_features,
                current_quarter=str(row.get("quarter") or ""),
                target_col="target_delta_log",
                mode=str(args.temporal_state_analog_mode),
                min_history=int(args.temporal_state_analog_min_history),
                neighbor_k=int(args.temporal_state_analog_neighbor_k),
                history_cap=int(args.temporal_state_analog_history_cap),
                distance_tau=float(args.temporal_state_analog_distance_tau),
                var_tau=float(args.temporal_state_analog_var_tau),
                neff_scale=float(args.temporal_state_analog_neff_scale),
                support_scale=float(args.temporal_state_analog_support_scale),
                max_abs_log_delta=float(args.temporal_max_abs_log_correction),
            )
            temporal_state_analog_delta = float(_safe_float(temporal_state_analog_result.get("pred"), 0.0))
            temporal_state_analog_support = float(_safe_float(temporal_state_analog_result.get("support"), 0.0))
            temporal_state_analog_blend_support = 0.0
            if str(args.temporal_state_analog_mode) == "state_analog_replace_v0" and temporal_state_analog_support > 0.0:
                temporal_delta = float(temporal_state_analog_delta)
                temporal_support = float(temporal_state_analog_support)
            elif str(args.temporal_state_analog_mode) == "state_analog_blend_v0" and temporal_state_analog_support > 0.0:
                temporal_state_analog_blend_support = float(
                    _clip(float(temporal_state_analog_support) * float(args.temporal_state_analog_blend_weight), 0.0, 1.0)
                )
                temporal_delta = float(
                    _weighted_average(
                        [
                            (float(temporal_delta), float(temporal_support)),
                            (float(temporal_state_analog_delta), float(temporal_state_analog_blend_support)),
                        ]
                    )
                )
                temporal_support = float(_clip(max(float(temporal_support), temporal_state_analog_blend_support), 0.0, 1.0))
            temporal_action_gate_features = _temporal_action_gate_feature_map(
                guidance_bucket=str(current_guidance_bucket),
                temporal_delta=float(temporal_delta),
                temporal_support=float(temporal_support),
                intrinsic_delta=float(intrinsic_delta),
                intrinsic_support=float(intrinsic_support),
                base_delta=float(compressed_base_delta),
                base_support=float(base_support),
                guidance_expert_active=bool(_safe_float(guidance_expert.get("active"), 0.0) > 0.0),
                guidance_lock=float(guidance_lock),
                anchor_uncertainty=float(anchor_uncertainty),
                internal_strength=float(_safe_float(internal_state.get("internal_strength"), 0.0)),
                forward_conflict_ratio=float(conflict_ratio),
                memory_support=float(_safe_float(memory_diag.get("support"), 0.0)),
                memory_consistency=float(_safe_float(memory_diag.get("consistency"), 0.0)),
                temporal_result=temporal_result,
                temporal_interaction_diag=temporal_interaction_guard_diag,
                temporal_context_quality_scale=float(temporal_context_quality_scale),
                temporal_context_quality_weak_score=float(_safe_float(temporal_context_quality_diag.get("weak_score"), 0.0)),
                temporal_reliability_scale=float(temporal_reliability_scale),
                temporal_reliability_trust_score=float(_safe_float(temporal_reliability_diag.get("trust_score"), 0.0)),
            )
            action_gate_history_scope = str(args.action_gate_history_scope)
            action_gate_history_source = "company_local"
            action_gate_history: Sequence[Mapping[str, Any]] = native_memory_history
            if action_gate_history_scope == "panel_prior":
                if action_gate_panel_history_seed:
                    action_gate_history = _prior_history_by_quarter(action_gate_panel_history_seed, str(row.get("quarter") or ""))
                    action_gate_history_source = "panel_prior_csv"
                else:
                    action_gate_history = _prior_history_by_quarter(panel_action_gate_history, str(row.get("quarter") or ""))
                    action_gate_history_source = "panel_prior_traversal"
            temporal_action_gate_diag = _temporal_action_gate_scale(
                action_gate_history,
                feature_map=temporal_action_gate_features,
                mode=str(args.temporal_action_gate_mode),
                min_history=int(args.temporal_action_gate_min_history),
                neighbor_k=int(args.temporal_action_gate_neighbor_k),
                tau=float(args.temporal_action_gate_tau),
                min_scale=float(args.temporal_action_gate_min_scale),
                abstain_effect_threshold=float(args.temporal_action_gate_abstain_effect_threshold),
                abstain_win_rate_threshold=float(args.temporal_action_gate_abstain_win_rate_threshold),
            )
            temporal_action_gate_diag["history_scope"] = action_gate_history_scope
            temporal_action_gate_diag["history_source"] = action_gate_history_source
            temporal_action_gate_scale = float(_safe_float(temporal_action_gate_diag.get("scale"), 1.0))
            temporal_support = float(_clip(float(temporal_support) * temporal_action_gate_scale, 0.0, 1.0))
            temporal_anchor_confidence_guard_scale = 1.0
            temporal_anchor_confidence_guard_active = 0
            temporal_anchor_confidence_guard_reason = "off"
            temporal_anchor_confidence_guard_anchor_history_mae = float(_safe_float(anchor_history_mae, float("nan")))
            if str(args.temporal_anchor_confidence_guard_mode) == "strong_anchor_no_guidance_v1":
                temporal_anchor_confidence_guard_reason = "conditions_not_met"
                no_guidance_temporal_context = str(current_guidance_bucket) == "none"
                strong_anchor_history = (
                    np.isfinite(temporal_anchor_confidence_guard_anchor_history_mae)
                    and temporal_anchor_confidence_guard_anchor_history_mae <= float(args.temporal_anchor_confidence_guard_history_mae_threshold)
                )
                large_temporal_delta = abs(float(_safe_float(temporal_delta, 0.0))) >= float(args.temporal_anchor_confidence_guard_min_abs_delta)
                enough_temporal_support = float(_safe_float(temporal_support, 0.0)) >= float(args.temporal_anchor_confidence_guard_min_support)
                if no_guidance_temporal_context and strong_anchor_history and large_temporal_delta and enough_temporal_support:
                    temporal_anchor_confidence_guard_scale = float(_clip(float(args.temporal_anchor_confidence_guard_scale), 0.0, 1.0))
                    temporal_anchor_confidence_guard_active = int(temporal_anchor_confidence_guard_scale < 1.0)
                    temporal_anchor_confidence_guard_reason = "strong_anchor_no_guidance_large_temporal_delta"
                temporal_support = float(_clip(float(temporal_support) * temporal_anchor_confidence_guard_scale, 0.0, 1.0))

            intrinsic_residual_candidate_delta = float(intrinsic_delta)
            no_intrinsic_active_experts = tuple(name for name in active_experts if name != "intrinsic_direct")
            no_intrinsic_reference_arbitration = _arbitrate_direct_rawcard_experts(
                compressed_base_delta=float(compressed_base_delta if "compressed_base" in active_experts else 0.0),
                compressed_base_support=float(base_support if "compressed_base" in active_experts else 0.0),
                intrinsic_delta=0.0,
                intrinsic_support=0.0,
                temporal_delta=float(temporal_delta if "temporal_direct" in active_experts else 0.0),
                temporal_support=float(temporal_support if "temporal_direct" in active_experts else 0.0),
                active_experts=no_intrinsic_active_experts,
                arbitration_mode=str("current" if str(args.arbitration_mode) in {INTEGRATED_EXPERT_ARBITRATION_MODE, SHARED_RESIDUAL_BACKBONE_MODE} else args.arbitration_mode),
                guidance_delta=float(_safe_float(guidance_expert.get("delta"), 0.0)),
                guidance_support=float(_safe_float(guidance_expert.get("support"), 0.0)),
                guidance_expert_mode=str(args.guidance_expert_mode),
                guidance_bucket=str(current_guidance_bucket),
                evidence_orthogonal_duplicate_threshold=float(args.evidence_orthogonal_duplicate_threshold),
                evidence_orthogonal_base_min_scale=float(args.evidence_orthogonal_base_min_scale),
                evidence_orthogonal_guidance_min_scale=float(args.evidence_orthogonal_guidance_min_scale),
                evidence_orthogonal_base_strength=float(args.evidence_orthogonal_base_strength),
                evidence_orthogonal_guidance_strength=float(args.evidence_orthogonal_guidance_strength),
            )
            no_intrinsic_reference_delta = float(no_intrinsic_reference_arbitration["pred"])
            current_intrinsic_target_is_residual = bool(
                intrinsic_target_mode == "post_base_temporal_guidance_residual_v1"
                or (intrinsic_target_mode == "partitioned_residual_v1" and current_guidance_bucket != "none")
            )
            if current_intrinsic_target_is_residual and np.isfinite(no_intrinsic_reference_delta):
                intrinsic_delta = float(no_intrinsic_reference_delta + intrinsic_residual_candidate_delta)

            intrinsic_pre_calibration_delta = float(intrinsic_delta)
            intrinsic_pre_calibration_support = float(intrinsic_support)
            intrinsic_raw_reference_arbitration = _arbitrate_direct_rawcard_experts(
                compressed_base_delta=float(compressed_base_delta if "compressed_base" in active_experts else 0.0),
                compressed_base_support=float(base_support if "compressed_base" in active_experts else 0.0),
                intrinsic_delta=float(intrinsic_pre_calibration_delta if "intrinsic_direct" in active_experts else 0.0),
                intrinsic_support=float(intrinsic_pre_calibration_support if "intrinsic_direct" in active_experts else 0.0),
                temporal_delta=float(temporal_delta if "temporal_direct" in active_experts else 0.0),
                temporal_support=float(temporal_support if "temporal_direct" in active_experts else 0.0),
                active_experts=active_experts,
                arbitration_mode=str("current" if str(args.arbitration_mode) in {INTEGRATED_EXPERT_ARBITRATION_MODE, SHARED_RESIDUAL_BACKBONE_MODE} else args.arbitration_mode),
                guidance_delta=float(_safe_float(guidance_expert.get("delta"), 0.0)),
                guidance_support=float(_safe_float(guidance_expert.get("support"), 0.0)),
                guidance_expert_mode=str(args.guidance_expert_mode),
                guidance_bucket=str(current_guidance_bucket),
                evidence_orthogonal_duplicate_threshold=float(args.evidence_orthogonal_duplicate_threshold),
                evidence_orthogonal_base_min_scale=float(args.evidence_orthogonal_base_min_scale),
                evidence_orthogonal_guidance_min_scale=float(args.evidence_orthogonal_guidance_min_scale),
                evidence_orthogonal_base_strength=float(args.evidence_orthogonal_base_strength),
                evidence_orthogonal_guidance_strength=float(args.evidence_orthogonal_guidance_strength),
            )
            intrinsic_raw_reference_delta = float(intrinsic_raw_reference_arbitration["pred"])
            intrinsic_reliability_diag = _intrinsic_history_reliability_scale(
                native_memory_history,
                row=row,
                intrinsic_delta=float(intrinsic_pre_calibration_delta),
                temporal_delta=float(temporal_delta),
                conflict_ratio=float(conflict_ratio),
                guidance_expert_active=bool(_safe_float(guidance_expert.get("active"), 0.0) > 0.0),
                mode=str(args.intrinsic_reliability_mode),
                min_history=int(args.intrinsic_reliability_min_history),
                tau=float(args.intrinsic_reliability_tau),
                min_scale=float(args.intrinsic_reliability_min_scale),
                max_scale=float(args.intrinsic_reliability_max_scale),
            )
            intrinsic_explicit_guidance_scale = 1.0
            if str(args.intrinsic_explicit_guidance_guard_mode) == "support_shrink_v1" and _row_guidance_bucket(row) == "explicit_numeric":
                intrinsic_explicit_guidance_scale = float(_clip(args.intrinsic_explicit_guidance_support_scale, 0.0, 1.0))
            intrinsic_total_support_scale = float(_clip(
                float(_safe_float(intrinsic_reliability_diag.get("scale"), 1.0)) * intrinsic_explicit_guidance_scale,
                0.0,
                max(float(args.intrinsic_reliability_max_scale), 1.0),
            ))
            if intrinsic_total_support_scale < 1.0:
                intrinsic_delta = float(intrinsic_delta * intrinsic_total_support_scale)
            intrinsic_support = float(_clip(float(intrinsic_support) * intrinsic_total_support_scale, 0.0, 1.0))
            intrinsic_temporal_dedup_diag = _intrinsic_temporal_dedup_support_scale(
                intrinsic_module_delta=float(intrinsic_residual_candidate_delta),
                temporal_delta=float(temporal_delta),
                intrinsic_support=float(intrinsic_support),
                temporal_support=float(temporal_support),
                mode=str(args.intrinsic_temporal_dedup_mode),
                duplicate_threshold=float(args.intrinsic_temporal_dedup_duplicate_threshold),
                min_scale=float(args.intrinsic_temporal_dedup_min_scale),
                strength=float(args.intrinsic_temporal_dedup_strength),
            )
            intrinsic_temporal_dedup_scale = float(_safe_float(intrinsic_temporal_dedup_diag.get("scale"), 1.0))
            intrinsic_support = float(_clip(float(intrinsic_support) * intrinsic_temporal_dedup_scale, 0.0, 1.0))
            intrinsic_action_gate_features = _intrinsic_action_gate_feature_map(
                guidance_bucket=str(current_guidance_bucket),
                intrinsic_delta=float(intrinsic_delta),
                intrinsic_support=float(intrinsic_support),
                intrinsic_residual_candidate_delta=float(intrinsic_residual_candidate_delta),
                temporal_delta=float(temporal_delta),
                temporal_support=float(temporal_support),
                base_delta=float(compressed_base_delta),
                base_support=float(base_support),
                guidance_expert_active=bool(_safe_float(guidance_expert.get("active"), 0.0) > 0.0),
                guidance_lock=float(guidance_lock),
                anchor_uncertainty=float(anchor_uncertainty),
                internal_strength=float(_safe_float(internal_state.get("internal_strength"), 0.0)),
                forward_conflict_ratio=float(conflict_ratio),
                memory_support=float(_safe_float(memory_diag.get("support"), 0.0)),
                memory_consistency=float(_safe_float(memory_diag.get("consistency"), 0.0)),
                intrinsic_result=intrinsic_result,
                intrinsic_reliability_diag=intrinsic_reliability_diag,
                intrinsic_explicit_guidance_scale=float(intrinsic_explicit_guidance_scale),
                intrinsic_temporal_dedup_diag=intrinsic_temporal_dedup_diag,
                intrinsic_temporal_dedup_scale=float(intrinsic_temporal_dedup_scale),
            )
            intrinsic_action_gate_diag = _intrinsic_action_gate_scale(
                action_gate_history,
                feature_map=intrinsic_action_gate_features,
                mode=str(args.intrinsic_action_gate_mode),
                min_history=int(args.intrinsic_action_gate_min_history),
                neighbor_k=int(args.intrinsic_action_gate_neighbor_k),
                tau=float(args.intrinsic_action_gate_tau),
                min_scale=float(args.intrinsic_action_gate_min_scale),
                abstain_effect_threshold=float(args.intrinsic_action_gate_abstain_effect_threshold),
                abstain_win_rate_threshold=float(args.intrinsic_action_gate_abstain_win_rate_threshold),
            )
            intrinsic_action_gate_diag["history_scope"] = action_gate_history_scope
            intrinsic_action_gate_diag["history_source"] = action_gate_history_source
            intrinsic_action_gate_scale = float(_safe_float(intrinsic_action_gate_diag.get("scale"), 1.0))
            intrinsic_support = float(_clip(float(intrinsic_support) * intrinsic_action_gate_scale, 0.0, 1.0))
            contract_base_delta = float(compressed_base_delta if "compressed_base" in active_experts else 0.0)
            contract_base_support = float(base_support if "compressed_base" in active_experts else 0.0)
            contract_intrinsic_delta = float(intrinsic_delta if "intrinsic_direct" in active_experts else 0.0)
            contract_intrinsic_support = float(intrinsic_support if "intrinsic_direct" in active_experts else 0.0)
            contract_temporal_delta = float(temporal_delta if "temporal_direct" in active_experts else 0.0)
            contract_temporal_support = float(temporal_support if "temporal_direct" in active_experts else 0.0)
            no_temporal_active_experts = tuple(name for name in active_experts if name != "temporal_direct")
            no_temporal_reference_arbitration = _arbitrate_direct_rawcard_experts(
                compressed_base_delta=float(contract_base_delta),
                compressed_base_support=float(contract_base_support),
                intrinsic_delta=float(contract_intrinsic_delta),
                intrinsic_support=float(contract_intrinsic_support),
                temporal_delta=0.0,
                temporal_support=0.0,
                active_experts=no_temporal_active_experts,
                arbitration_mode=str("current" if str(args.arbitration_mode) in {INTEGRATED_EXPERT_ARBITRATION_MODE, SHARED_RESIDUAL_BACKBONE_MODE} else args.arbitration_mode),
                guidance_delta=float(_safe_float(guidance_expert.get("delta"), 0.0)),
                guidance_support=float(_safe_float(guidance_expert.get("support"), 0.0)),
                guidance_expert_mode=str(args.guidance_expert_mode),
                guidance_bucket=str(current_guidance_bucket),
                evidence_orthogonal_duplicate_threshold=float(args.evidence_orthogonal_duplicate_threshold),
                evidence_orthogonal_base_min_scale=float(args.evidence_orthogonal_base_min_scale),
                evidence_orthogonal_guidance_min_scale=float(args.evidence_orthogonal_guidance_min_scale),
                evidence_orthogonal_base_strength=float(args.evidence_orthogonal_base_strength),
                evidence_orthogonal_guidance_strength=float(args.evidence_orthogonal_guidance_strength),
            )
            no_temporal_reference_delta = float(no_temporal_reference_arbitration["pred"])

            # The frozen ablation matrix keeps the candidate generators fixed and only restricts which experts may enter the final blend.
            arbitration_blend_mode = "current" if str(args.arbitration_mode) in {INTEGRATED_EXPERT_ARBITRATION_MODE, SHARED_RESIDUAL_BACKBONE_MODE} else str(args.arbitration_mode)
            arbitration = _arbitrate_direct_rawcard_experts(
                compressed_base_delta=float(contract_base_delta),
                compressed_base_support=float(contract_base_support),
                intrinsic_delta=float(contract_intrinsic_delta),
                intrinsic_support=float(contract_intrinsic_support),
                temporal_delta=float(contract_temporal_delta),
                temporal_support=float(contract_temporal_support),
                active_experts=active_experts,
                arbitration_mode=str(arbitration_blend_mode),
                guidance_delta=float(_safe_float(guidance_expert.get("delta"), 0.0)),
                guidance_support=float(_safe_float(guidance_expert.get("support"), 0.0)),
                guidance_expert_mode=str(args.guidance_expert_mode),
                guidance_bucket=str(current_guidance_bucket),
                evidence_orthogonal_duplicate_threshold=float(args.evidence_orthogonal_duplicate_threshold),
                evidence_orthogonal_base_min_scale=float(args.evidence_orthogonal_base_min_scale),
                evidence_orthogonal_guidance_min_scale=float(args.evidence_orthogonal_guidance_min_scale),
                evidence_orthogonal_base_strength=float(args.evidence_orthogonal_base_strength),
                evidence_orthogonal_guidance_strength=float(args.evidence_orthogonal_guidance_strength),
            )
            evidence_orthogonal_diag = dict(arbitration.get("orthogonal_diag") or {})
            base_delta = float(arbitration["pred"])
            integrated_arbitration = {
                "delta": float(base_delta),
                "action": "not_used",
                "reason": "integrated_expert_arbitration_v0_not_enabled",
                "target_source": "",
                "target_delta_log": float("nan"),
                "retained_post_guardrail_delta_log": float("nan"),
                "internal_active": 0,
                "base_prior_floor_active": 0,
                "timing_blocked": 0,
                "floor_ratio": integrated_arbitration_floor_ratio,
                "floor_pred": float("nan"),
            }
            if str(args.arbitration_mode) == INTEGRATED_EXPERT_ARBITRATION_MODE:
                retained_pre_guardrail_pred = float(anchor_pred * np.exp(base_delta)) if np.isfinite(anchor_pred) and anchor_pred > 0.0 else float(anchor_pred)
                retained_guardrail = _apply_guidance_quality_guardrail(
                    anchor_pred=float(anchor_pred),
                    final_pred=float(retained_pre_guardrail_pred),
                    guidance_label=current_guidance,
                    mode=str(args.guidance_quality_guardrail_mode),
                    anchor_history_score=float(anchor_history_mae),
                )
                retained_post_guardrail_pred = float(retained_guardrail["post_guardrail_pred"])
                retained_post_guardrail_delta = float(base_delta)
                if np.isfinite(anchor_pred) and anchor_pred > 0.0 and np.isfinite(retained_post_guardrail_pred) and retained_post_guardrail_pred > 0.0:
                    retained_post_guardrail_delta = float(np.log(retained_post_guardrail_pred / anchor_pred))
                integrated_arbitration = _integrated_expert_arbitration_adjustment(
                    row=row,
                    anchor_pred=float(anchor_pred),
                    retained_pre_guardrail_delta=float(base_delta),
                    retained_post_guardrail_delta=float(retained_post_guardrail_delta),
                    compressed_base_delta=float(contract_base_delta),
                    compressed_base_support=float(contract_base_support),
                    intrinsic_delta=float(contract_intrinsic_delta),
                    intrinsic_support=float(contract_intrinsic_support),
                    temporal_delta=float(contract_temporal_delta),
                    temporal_support=float(contract_temporal_support),
                    anchor_error_recent_abs_log=float(_safe_float(anchor_error_state.get("anchor_error_recent_abs_log"), 0.0)),
                    internal_strength=float(_safe_float(internal_state.get("internal_strength"), 0.0)),
                    native_forward_rows=list(native_forward_rows),
                    temporal_top_matches=list(temporal_result.get("top_matches", [])),
                    floor_ratio=float(integrated_arbitration_floor_ratio),
                )
                base_delta = float(integrated_arbitration["delta"])

            anchor_log = _safe_log(anchor_pred)
            actual_log = _safe_log(_safe_float(row.get("actual")))
            shock_target_log = float(actual_log - anchor_log) if np.isfinite(actual_log) and np.isfinite(anchor_log) else float("nan")
            gate_residual_target_log = float(shock_target_log - base_delta) if np.isfinite(shock_target_log) else float("nan")
            intrinsic_raw_reference_pred, intrinsic_raw_reference_guarded_delta = _guarded_delta_from_base_delta(
                row=row,
                anchor_pred=float(anchor_pred),
                base_delta=float(intrinsic_raw_reference_delta),
                guidance_quality_guardrail_mode=str(args.guidance_quality_guardrail_mode),
                anchor_history_score=float(anchor_history_mae),
            )
            no_intrinsic_reference_pred, no_intrinsic_reference_guarded_delta = _guarded_delta_from_base_delta(
                row=row,
                anchor_pred=float(anchor_pred),
                base_delta=float(no_intrinsic_reference_delta),
                guidance_quality_guardrail_mode=str(args.guidance_quality_guardrail_mode),
                anchor_history_score=float(anchor_history_mae),
            )
            no_temporal_reference_pred, no_temporal_reference_guarded_delta = _guarded_delta_from_base_delta(
                row=row,
                anchor_pred=float(anchor_pred),
                base_delta=float(no_temporal_reference_delta),
                guidance_quality_guardrail_mode=str(args.guidance_quality_guardrail_mode),
                anchor_history_score=float(anchor_history_mae),
            )
            intrinsic_raw_marginal_smape_effect = float("nan")
            intrinsic_residual_target_log = float("nan")
            intrinsic_partitioned_residual_target_log = float("nan")
            actual_value = _safe_float(row.get("actual"), float("nan"))
            if (
                np.isfinite(actual_value)
                and actual_value > 0.0
                and np.isfinite(intrinsic_raw_reference_pred)
                and intrinsic_raw_reference_pred > 0.0
                and np.isfinite(no_intrinsic_reference_pred)
                and no_intrinsic_reference_pred > 0.0
            ):
                intrinsic_raw_marginal_smape_effect = float(
                    _local_smape(float(actual_value), float(no_intrinsic_reference_pred))
                    - _local_smape(float(actual_value), float(intrinsic_raw_reference_pred))
                )
                intrinsic_residual_target_log = float(np.log(float(actual_value) / float(no_intrinsic_reference_pred)))
            if current_guidance_bucket == "none":
                intrinsic_partitioned_residual_target_log = float(shock_target_log)
            elif np.isfinite(intrinsic_residual_target_log):
                intrinsic_partitioned_residual_target_log = float(intrinsic_residual_target_log)

            gate_feature_map = _build_direct_gate_features(
                base_delta=base_delta,
                intrinsic_delta=contract_intrinsic_delta,
                intrinsic_support=contract_intrinsic_support,
                temporal_delta=contract_temporal_delta,
                temporal_support=contract_temporal_support,
                sign_match=float(arbitration["sign_match"]),
                guidance_lock=guidance_lock,
                anchor_uncertainty=anchor_uncertainty,
                internal_strength=float(_safe_float(internal_state.get("internal_strength"), 0.0)),
                fwd_conflict_ratio=conflict_ratio,
                memory_support=float(_safe_float(memory_diag.get("support"), 0.0)),
                memory_consistency=float(_safe_float(memory_diag.get("consistency"), 0.0)),
            )

            local_gate_result = _predict_zero_intercept_history(
                train_rows=list(quarterly_rows),
                x_map=gate_feature_map,
                feature_names=DIRECT_EXPERT_GATE_FEATURES,
                target_col="gate_residual_target_log",
                alpha=float(args.gate_alpha),
                delta_cap_quantile=0.9,
                shrink_k=float(args.gate_shrink_k),
            )
            gate_used = False
            gate_support_used = 0.0
            gate_scale_applied = 0.0
            gate_scope_applied = "off"
            final_delta = float(base_delta)
            gate_mode = str(args.gate_mode)
            if expert_contract != "anchor_only" and gate_mode != "off" and int(local_gate_result["train_count"]) >= int(args.gate_min_train):
                gate_support_used = float(local_gate_result["support"])
                gate_scale_applied = 1.0
                if gate_mode == "company_local_scaled":
                    gate_scale_applied = _direct_gate_presence_scale(
                        base_delta=float(base_delta),
                        base_weight=float(_safe_float((arbitration.get("candidate_weights") or {}).get("compressed_base"), 0.0)),
                        sign_match=float(arbitration["sign_match"]),
                        max_abs_log_delta=float(args.temporal_max_abs_log_correction),
                    )
                final_delta = float(base_delta + gate_support_used * gate_scale_applied * float(local_gate_result["pred"]))
                gate_used = bool(gate_scale_applied > 0.0)
                gate_scope_applied = gate_mode

            final_pred = float(anchor_pred * np.exp(final_delta)) if np.isfinite(anchor_pred) and anchor_pred > 0.0 else float(anchor_pred)
            pre_guidance_guardrail_delta = float(final_delta)
            pre_guidance_guardrail_pred = float(final_pred)
            guidance_guardrail = _apply_guidance_quality_guardrail(
                anchor_pred=float(anchor_pred),
                final_pred=float(final_pred),
                guidance_label=current_guidance,
                mode=str(args.guidance_quality_guardrail_mode),
                anchor_history_score=float(anchor_history_mae),
            )
            final_pred = float(guidance_guardrail["post_guardrail_pred"])
            if np.isfinite(anchor_pred) and anchor_pred > 0.0 and np.isfinite(final_pred) and final_pred > 0.0:
                final_delta = float(np.log(final_pred / anchor_pred))
            temporal_marginal_smape_effect = float("nan")
            intrinsic_action_marginal_smape_effect = float("nan")
            if (
                np.isfinite(actual_value)
                and actual_value > 0.0
                and np.isfinite(final_pred)
                and final_pred > 0.0
                and np.isfinite(no_temporal_reference_pred)
                and no_temporal_reference_pred > 0.0
            ):
                temporal_marginal_smape_effect = float(
                    _local_smape(float(actual_value), float(no_temporal_reference_pred))
                    - _local_smape(float(actual_value), float(final_pred))
                )
            if (
                np.isfinite(actual_value)
                and actual_value > 0.0
                and np.isfinite(final_pred)
                and final_pred > 0.0
                and np.isfinite(no_intrinsic_reference_pred)
                and no_intrinsic_reference_pred > 0.0
            ):
                intrinsic_action_marginal_smape_effect = float(
                    _local_smape(float(actual_value), float(no_intrinsic_reference_pred))
                    - _local_smape(float(actual_value), float(final_pred))
                )

            intrinsic_support_cards: List[Dict[str, Any]] = []
            intrinsic_conflict_cards: List[Dict[str, Any]] = []
            if intrinsic_result.get("coefs_by_name"):
                scored_cards: List[Dict[str, Any]] = []
                for card in native_forward_rows:
                    if intrinsic_mode == "grouped_ridge_v3":
                        influence = _intrinsic_grouped_card_influence(
                            card,
                            admissibility_mode="conservative_admissibility_v2",
                            coefs_by_name=intrinsic_result["coefs_by_name"],
                        )
                    elif intrinsic_mode == "additive_grouped_hybrid_v3":
                        additive_influence = _card_additive_influence(card, intrinsic_additive_result.get("coefs_by_name", {}), "conservative_admissibility_v2")
                        grouped_influence = _intrinsic_grouped_card_influence(
                            card,
                            admissibility_mode="conservative_admissibility_v2",
                            coefs_by_name=intrinsic_grouped_result.get("coefs_by_name", {}),
                        )
                        influence = float(additive_influence + intrinsic_hybrid_lambda * (grouped_influence - additive_influence))
                    else:
                        influence = _card_additive_influence(card, intrinsic_result["coefs_by_name"], "conservative_admissibility_v2")
                    card_copy = dict(card)
                    card_copy["admissibility_weight"] = float(_native_card_admissibility_weight(card, "conservative_admissibility_v2"))
                    card_copy["approx_influence"] = float(influence)
                    scored_cards.append(card_copy)
                intrinsic_support_cards, intrinsic_conflict_cards = _split_native_support_conflict_cards(
                    scored_cards,
                    target_delta=intrinsic_delta,
                    top_k=5,
                )

            out_row = dict(row)
            out_row.update(regime)
            out_row.update(internal_state)
            out_row.update(anchor_error_state)
            out_row.update({name: float(value) for name, value in factor_map.items()})
            out_row.update({name: float(value) for name, value in native_surface.items() if name in RAW_CARD_FEATURE_NAMES})
            out_row.update({name: float(value) for name, value in gate_feature_map.items()})
            out_row.update(
                {
                    "ticker": ticker,
                    "best_stat_model": best_stat_model,
                    "best_stat_mae_company": float(best_stat_mae),
                    "prehistory_best_model": prehist_best_model,
                    "prehistory_best_mae": float(prehist_best_mae) if np.isfinite(prehist_best_mae) else float(best_stat_mae),
                    "anchor_selection_mode": str(args.anchor_selection_mode),
                    "anchor_guidance_regime_mode": str(args.anchor_guidance_regime_mode),
                    "anchor_guidance_same_regime_min_history": int(args.anchor_guidance_same_regime_min_history),
                    "anchor_guidance_mismatch_penalty": float(args.anchor_guidance_mismatch_penalty),
                    "anchor_explicit_guidance_proximity_mode": str(args.anchor_explicit_guidance_proximity_mode),
                    "anchor_explicit_guidance_proximity_weight": float(args.anchor_explicit_guidance_proximity_weight),
                    "anchor_explicit_guidance_kernel_min_history": int(args.anchor_explicit_guidance_kernel_min_history),
                    "anchor_explicit_guidance_kernel_band": float(args.anchor_explicit_guidance_kernel_band),
                    "anchor_explicit_guidance_kernel_shrink_k": float(args.anchor_explicit_guidance_kernel_shrink_k),
                    "anchor_robust_momentum_mode": str(args.anchor_robust_momentum_mode),
                    "anchor_current_guidance_bucket": _row_guidance_bucket(row),
                    "anchor_best_model": str(anchor_model),
                    "anchor_best_model_history_mae": float(anchor_history_mae),
                    "anchor_best_model_history_n": int(anchor_history_n),
                    "anchor_selection_reason": str(anchor_selection_reason),
                    "pred_csais_anchor": float(anchor_pred),
                    "anchor_uncertainty": float(anchor_uncertainty),
                    "guidance_lock": float(guidance_lock),
                    "method_family": str(method_family),
                    "expert_contract": str(expert_contract),
                    "intrinsic_mode": str(intrinsic_mode),
                    "intrinsic_target_mode": str(intrinsic_target_mode),
                    "intrinsic_target_col": str(intrinsic_target_col),
                    "csais_intrinsic_direct_target_is_residual": int(current_intrinsic_target_is_residual),
                    "csais_intrinsic_direct_residual_candidate_delta_log": float(intrinsic_residual_candidate_delta),
                    "csais_intrinsic_direct_pre_calibration_delta_log": float(intrinsic_pre_calibration_delta),
                    "csais_intrinsic_direct_pre_calibration_support": float(intrinsic_pre_calibration_support),
                    "csais_intrinsic_direct_reliability_mode": str(args.intrinsic_reliability_mode),
                    "csais_intrinsic_direct_reliability_bucket": str(intrinsic_reliability_diag.get("bucket") or ""),
                    "csais_intrinsic_direct_reliability_source": str(intrinsic_reliability_diag.get("source") or ""),
                    "csais_intrinsic_direct_reliability_history_n": int(_safe_float(intrinsic_reliability_diag.get("history_n"), 0.0)),
                    "csais_intrinsic_direct_reliability_effect_mean": float(_safe_float(intrinsic_reliability_diag.get("effect_mean"), float("nan"))),
                    "csais_intrinsic_direct_reliability_win_rate": float(_safe_float(intrinsic_reliability_diag.get("win_rate"), float("nan"))),
                    "csais_intrinsic_direct_reliability_scale": float(_safe_float(intrinsic_reliability_diag.get("scale"), 1.0)),
                    "csais_intrinsic_direct_explicit_guidance_guard_mode": str(args.intrinsic_explicit_guidance_guard_mode),
                    "csais_intrinsic_direct_explicit_guidance_scale": float(intrinsic_explicit_guidance_scale),
                    "csais_intrinsic_direct_total_support_scale": float(intrinsic_total_support_scale),
                    "csais_intrinsic_temporal_dedup_mode": str(args.intrinsic_temporal_dedup_mode),
                    "csais_intrinsic_temporal_dedup_active": int(_safe_float(intrinsic_temporal_dedup_diag.get("active"), 0.0)),
                    "csais_intrinsic_temporal_dedup_scale": float(intrinsic_temporal_dedup_scale),
                    "csais_intrinsic_temporal_dedup_duplicate_threshold": float(args.intrinsic_temporal_dedup_duplicate_threshold),
                    "csais_intrinsic_temporal_dedup_duplicate_ratio": float(_safe_float(intrinsic_temporal_dedup_diag.get("duplicate_ratio"), 0.0)),
                    "csais_intrinsic_temporal_dedup_sign_match": float(_safe_float(intrinsic_temporal_dedup_diag.get("sign_match"), 0.5)),
                    "csais_intrinsic_temporal_dedup_reason": str(intrinsic_temporal_dedup_diag.get("reason") or ""),
                    "csais_intrinsic_action_gate_mode": str(args.intrinsic_action_gate_mode),
                    "csais_action_gate_history_scope": str(action_gate_history_scope),
                    "csais_action_gate_history_source": str(action_gate_history_source),
                    "csais_action_gate_history_available_n": int(len(action_gate_history)),
                    "csais_intrinsic_action_gate_scale": float(intrinsic_action_gate_scale),
                    "csais_intrinsic_action_gate_source": str(intrinsic_action_gate_diag.get("source") or ""),
                    "csais_intrinsic_action_gate_history_scope": str(intrinsic_action_gate_diag.get("history_scope") or ""),
                    "csais_intrinsic_action_gate_history_source": str(intrinsic_action_gate_diag.get("history_source") or ""),
                    "csais_intrinsic_action_gate_reason": str(intrinsic_action_gate_diag.get("reason") or ""),
                    "csais_intrinsic_action_gate_history_n": int(_safe_float(intrinsic_action_gate_diag.get("history_n"), 0.0)),
                    "csais_intrinsic_action_gate_neighbor_n": int(_safe_float(intrinsic_action_gate_diag.get("neighbor_n"), 0.0)),
                    "csais_intrinsic_action_gate_pred_effect": float(_safe_float(intrinsic_action_gate_diag.get("pred_effect"), float("nan"))),
                    "csais_intrinsic_action_gate_neighbor_effect_mean": float(_safe_float(intrinsic_action_gate_diag.get("neighbor_effect_mean"), float("nan"))),
                    "csais_intrinsic_action_gate_neighbor_effect_std": float(_safe_float(intrinsic_action_gate_diag.get("neighbor_effect_std"), float("nan"))),
                    "csais_intrinsic_action_gate_win_rate": float(_safe_float(intrinsic_action_gate_diag.get("win_rate"), float("nan"))),
                    "csais_intrinsic_action_gate_trust_score": float(_safe_float(intrinsic_action_gate_diag.get("trust_score"), float("nan"))),
                    "csais_intrinsic_action_gate_mean_distance": float(_safe_float(intrinsic_action_gate_diag.get("mean_distance"), float("nan"))),
                    "csais_intrinsic_action_gate_features_json": json.dumps(sanitize_for_json(intrinsic_action_gate_features), ensure_ascii=False, allow_nan=False),
                    "csais_intrinsic_direct_raw_reference_delta_log": float(intrinsic_raw_reference_delta),
                    "csais_intrinsic_direct_raw_reference_guarded_delta_log": float(intrinsic_raw_reference_guarded_delta),
                    "csais_without_intrinsic_reference_delta_log": float(no_intrinsic_reference_delta),
                    "csais_without_intrinsic_reference_guarded_delta_log": float(no_intrinsic_reference_guarded_delta),
                    "csais_intrinsic_direct_raw_marginal_smape_effect": float(intrinsic_raw_marginal_smape_effect),
                    "csais_intrinsic_action_marginal_smape_effect": float(intrinsic_action_marginal_smape_effect),
                    "csais_intrinsic_direct_residual_target_log": float(intrinsic_residual_target_log),
                    "csais_intrinsic_direct_partitioned_residual_target_log": float(intrinsic_partitioned_residual_target_log),
                    "csais_raw_candidate_delta_log": float(raw_delta),
                    "csais_factor_candidate_delta_log": float(factor_delta),
                    "csais_compressed_base_candidate_delta_log": float(compressed_base_delta),
                    "csais_compressed_base_candidate_support": float(base_support),
                    "csais_intrinsic_direct_candidate_delta_raw_log": float(_safe_float(intrinsic_result.get("pred_raw"), 0.0) if np.isfinite(_safe_float(intrinsic_result.get("pred_raw"), float("nan"))) else 0.0),
                    "csais_intrinsic_direct_candidate_delta_log": float(intrinsic_delta),
                    "csais_intrinsic_direct_candidate_support": float(intrinsic_support),
                    "csais_intrinsic_direct_candidate_train_count": int(_safe_float(intrinsic_result.get("train_count"), 0.0)),
                    "csais_intrinsic_direct_candidate_coverage": float(_safe_float(intrinsic_result.get("coverage"), 1.0)),
                    "csais_intrinsic_grouped_aux_delta_log": float(grouped_delta),
                    "csais_intrinsic_grouped_aux_support": float(grouped_support),
                    "csais_intrinsic_grouped_aux_coverage": float(grouped_coverage),
                    "csais_intrinsic_hybrid_lambda": float(intrinsic_hybrid_lambda),
                    "csais_segment_bridge_mode": str(args.segment_bridge_mode),
                    "csais_segment_bridge_candidate_delta_raw_log": float(
                        _safe_float(segment_bridge_result.get("pred_raw"), 0.0)
                        if np.isfinite(_safe_float(segment_bridge_result.get("pred_raw"), float("nan")))
                        else 0.0
                    ),
                    "csais_segment_bridge_candidate_delta_log": float(segment_bridge_delta),
                    "csais_segment_bridge_candidate_support": float(segment_bridge_support),
                    "csais_segment_bridge_candidate_train_count": int(_safe_float(segment_bridge_result.get("train_count"), 0.0)),
                    "csais_segment_bridge_candidate_coverage": float(segment_bridge_coverage),
                    "csais_segment_bridge_hybrid_lambda": float(segment_bridge_lambda),
                    "csais_segment_bridge_sign_match": float(segment_bridge_sign_match),
                    "csais_temporal_direct_candidate_delta_raw_log": float(_safe_float(temporal_result.get("pred_raw"), 0.0) if np.isfinite(_safe_float(temporal_result.get("pred_raw"), float("nan"))) else 0.0),
                    "csais_temporal_direct_candidate_score_mean_target_log": float(_safe_float(temporal_result.get("pred_raw"), 0.0) if np.isfinite(_safe_float(temporal_result.get("pred_raw"), float("nan"))) else 0.0),
                    "csais_temporal_direct_candidate_post_support_delta_log": float(_safe_float(temporal_result.get("pred_post_support"), 0.0)),
                    "csais_temporal_direct_candidate_delta_log": float(temporal_delta),
                    "csais_temporal_direct_candidate_support": float(temporal_support),
                    "csais_temporal_forward_candidate_delta_log": float(temporal_forward_delta),
                    "csais_temporal_forward_candidate_support": float(temporal_forward_support),
                    "csais_temporal_pre_state_analog_delta_log": float(temporal_pre_state_analog_delta),
                    "csais_temporal_pre_state_analog_support": float(temporal_pre_state_analog_support),
                    "csais_temporal_state_analog_mode": str(args.temporal_state_analog_mode),
                    "csais_temporal_state_analog_blend_weight": float(args.temporal_state_analog_blend_weight),
                    "csais_temporal_state_analog_blend_support": float(temporal_state_analog_blend_support),
                    "csais_temporal_state_analog_delta_raw_log": float(
                        _safe_float(temporal_state_analog_result.get("pred_raw"), 0.0)
                        if np.isfinite(_safe_float(temporal_state_analog_result.get("pred_raw"), float("nan")))
                        else 0.0
                    ),
                    "csais_temporal_state_analog_delta_log": float(temporal_state_analog_delta),
                    "csais_temporal_state_analog_support": float(temporal_state_analog_support),
                    "csais_temporal_state_analog_train_count": int(_safe_float(temporal_state_analog_result.get("train_count"), 0.0)),
                    "csais_temporal_state_analog_neighbor_n": int(_safe_float(temporal_state_analog_result.get("neighbor_n"), 0.0)),
                    "csais_temporal_state_analog_effective_memory_count": float(_safe_float(temporal_state_analog_result.get("effective_memory_count"), 0.0)),
                    "csais_temporal_state_analog_mean_distance": float(_safe_float(temporal_state_analog_result.get("mean_distance"), float("nan"))),
                    "csais_temporal_state_analog_reason": str(temporal_state_analog_result.get("reason") or ""),
                    "csais_temporal_direct_candidate_support_pre_direction": float(_safe_float(temporal_result.get("support_pre_direction"), 0.0)),
                    "csais_temporal_direct_candidate_direction_scale": float(_safe_float(temporal_result.get("direction_scale"), 1.0)),
                    "csais_temporal_direct_candidate_train_count": int(_safe_float(temporal_result.get("train_count"), 0.0)),
                    "csais_temporal_direct_candidate_effective_memory_count": float(_safe_float(temporal_result.get("effective_memory_count"), 0.0)),
                    "csais_temporal_direct_candidate_directional_consistency": float(_safe_float(temporal_result.get("directional_consistency"), 0.0)),
                    "csais_temporal_direct_candidate_attention_focus": float(_safe_float(temporal_result.get("attention_focus"), 0.0)),
                    "csais_temporal_direct_candidate_mean_direction_alignment": float(_safe_float(temporal_result.get("mean_direction_alignment"), 0.0)),
                    "csais_temporal_direct_candidate_top_direction_alignment": float(_safe_float(temporal_result.get("top_direction_alignment"), 0.0)),
                    "csais_temporal_evidence_filter_mode": str(args.temporal_evidence_filter_mode),
                    "csais_temporal_evidence_filter_scope": str(args.temporal_evidence_filter_scope),
                    "csais_temporal_forward_rows_before_filter": int(temporal_forward_filter_diag.get("before", 0)),
                    "csais_temporal_forward_rows_after_filter": int(temporal_forward_filter_diag.get("after", 0)),
                    "csais_temporal_forward_rows_dropped_filter": int(temporal_forward_filter_diag.get("dropped", 0)),
                    "csais_temporal_context_rows_before_filter": int(temporal_context_filter_diag.get("before", 0)),
                    "csais_temporal_context_rows_after_filter": int(temporal_context_filter_diag.get("after", 0)),
                    "csais_temporal_context_rows_dropped_filter": int(temporal_context_filter_diag.get("dropped", 0)),
                    "csais_temporal_context_quality_mode": str(args.temporal_context_quality_mode),
                    "csais_temporal_context_quality_weight": float(args.temporal_context_quality_weight),
                    "csais_temporal_context_quality_scale": float(temporal_context_quality_scale),
                    "csais_temporal_context_quality_weak_score": float(_safe_float(temporal_context_quality_diag.get("weak_score"), 0.0)),
                    "csais_temporal_context_quality_weak_ratio": float(_safe_float(temporal_context_quality_diag.get("weak_ratio"), 0.0)),
                    "csais_temporal_context_guard_mode": str(args.temporal_context_guard_mode),
                    "csais_temporal_context_guard_scale": float(temporal_context_guard_scale),
                    "csais_temporal_context_memory_mode": str(args.temporal_context_memory_mode),
                    "csais_temporal_context_memory_card_count": int(len(temporal_context_rows)),
                    "csais_temporal_context_retrieval_weight": float(_safe_float(temporal_result.get("context_score_weight"), 0.0)),
                    "csais_temporal_context_retrieval_attention_score": float(_safe_float(temporal_result.get("context_attention_score"), 0.0)),
                    "csais_temporal_context_retrieval_attention_focus": float(_safe_float(temporal_result.get("context_attention_focus"), 0.0)),
                    "csais_temporal_reliability_mode": str(temporal_result.get("reliability_mode") or "off"),
                    "csais_temporal_reliability_scale": float(_safe_float(temporal_result.get("typed_reliability_scale"), 1.0)),
                    "csais_temporal_context_support_scale": float(_safe_float(temporal_result.get("context_support_scale"), 1.0)),
                    "csais_temporal_segment_compatibility": float(_safe_float(temporal_result.get("segment_compatibility"), 1.0)),
                    "csais_temporal_segment_support_scale": float(_safe_float(temporal_result.get("segment_support_scale"), 1.0)),
                    "csais_temporal_novelty_value": float(_safe_float(temporal_result.get("novelty_value"), 0.0)),
                    "csais_temporal_novelty_compatibility": float(_safe_float(temporal_result.get("novelty_compatibility"), 1.0)),
                    "csais_temporal_novelty_shrink_scale": float(_safe_float(temporal_result.get("novelty_shrink_scale"), 1.0)),
                    "csais_temporal_guidance_quality": float(_safe_float(temporal_result.get("guidance_quality"), 0.0)),
                    "csais_temporal_guidance_trust_scale": float(_safe_float(temporal_result.get("guidance_trust_scale"), 1.0)),
                    "csais_temporal_context_pred_raw_log": float(_safe_float(temporal_result.get("context_pred_raw"), 0.0) if np.isfinite(_safe_float(temporal_result.get("context_pred_raw"), float("nan"))) else 0.0),
                    "csais_temporal_context_support_proxy": float(_safe_float(temporal_result.get("context_support_proxy"), 0.0)),
                    "csais_temporal_context_directional_consistency": float(_safe_float(temporal_result.get("context_directional_consistency"), 0.0)),
                    "csais_temporal_context_magnitude_alignment": float(_safe_float(temporal_result.get("context_magnitude_alignment"), 0.0)),
                    "csais_temporal_soft_state_compatibility": float(_safe_float(temporal_result.get("soft_state_compatibility"), 1.0)),
                    "csais_temporal_soft_agreement_value": float(_safe_float(temporal_result.get("soft_agreement_value"), 0.0)),
                    "csais_temporal_soft_agreement_scale": float(_safe_float(temporal_result.get("soft_agreement_scale"), 1.0)),
                    "csais_temporal_context_memory_forward_sparsity": float(temporal_context_forward_sparsity),
                    "csais_temporal_context_memory_delta_raw_log": float(_safe_float(temporal_context_result.get("pred_raw"), 0.0) if np.isfinite(_safe_float(temporal_context_result.get("pred_raw"), float("nan"))) else 0.0),
                    "csais_temporal_context_memory_delta_log": float(temporal_context_delta),
                    "csais_temporal_context_memory_support": float(temporal_context_support),
                    "csais_temporal_context_memory_support_pre_sparsity": float(_safe_float(temporal_context_result.get("support"), 0.0)),
                    "csais_temporal_context_memory_train_count": int(_safe_float(temporal_context_result.get("train_count"), 0.0)),
                    "csais_temporal_context_memory_effective_memory_count": float(_safe_float(temporal_context_result.get("effective_memory_count"), 0.0)),
                    "csais_temporal_context_memory_directional_consistency": float(_safe_float(temporal_context_result.get("directional_consistency"), 0.0)),
                    "csais_temporal_context_memory_attention_focus": float(_safe_float(temporal_context_result.get("attention_focus"), 0.0)),
                    "csais_temporal_context_memory_mean_direction_alignment": float(_safe_float(temporal_context_result.get("mean_direction_alignment"), 0.0)),
                    "csais_temporal_context_memory_top_direction_alignment": float(_safe_float(temporal_context_result.get("top_direction_alignment"), 0.0)),
                    "csais_temporal_guidance_bucket_scale_mode": str(args.temporal_guidance_bucket_scale_mode),
                    "csais_temporal_guidance_bucket_scale": float(temporal_guidance_bucket_scale),
                    "csais_temporal_anchor_confidence_guard_mode": str(args.temporal_anchor_confidence_guard_mode),
                    "csais_temporal_anchor_confidence_guard_active": int(temporal_anchor_confidence_guard_active),
                    "csais_temporal_anchor_confidence_guard_scale": float(temporal_anchor_confidence_guard_scale),
                    "csais_temporal_anchor_confidence_guard_reason": str(temporal_anchor_confidence_guard_reason),
                    "csais_temporal_anchor_confidence_guard_anchor_history_mae": float(temporal_anchor_confidence_guard_anchor_history_mae),
                    "csais_temporal_interaction_guard_mode": str(args.temporal_interaction_guard_mode),
                    "csais_temporal_interaction_guard_scale": float(temporal_interaction_guard_scale),
                    "csais_temporal_interaction_guard_reason": str(temporal_interaction_guard_diag.get("reason") or ""),
                    "csais_temporal_interaction_guard_guidance_scale": float(_safe_float(temporal_interaction_guard_diag.get("guidance_scale"), 1.0)),
                    "csais_temporal_interaction_guard_duplicate_scale": float(_safe_float(temporal_interaction_guard_diag.get("duplicate_scale"), 1.0)),
                    "csais_temporal_interaction_guard_guidance_active_scale": float(_safe_float(temporal_interaction_guard_diag.get("guidance_active_scale"), 1.0)),
                    "csais_temporal_interaction_guard_conflict_scale": float(_safe_float(temporal_interaction_guard_diag.get("conflict_scale"), 1.0)),
                    "csais_temporal_interaction_guard_duplicate_ratio": float(_safe_float(temporal_interaction_guard_diag.get("duplicate_ratio"), 0.0)),
                    "csais_temporal_interaction_guard_sign_match": float(_safe_float(temporal_interaction_guard_diag.get("sign_match"), 0.5)),
                    "csais_temporal_interaction_guard_intrinsic_conflict": int(_safe_float(temporal_interaction_guard_diag.get("temporal_intrinsic_conflict"), 0.0)),
                    "csais_temporal_interaction_guard_base_conflict": int(_safe_float(temporal_interaction_guard_diag.get("temporal_base_conflict"), 0.0)),
                    "csais_temporal_reliability_memory_mode": str(args.temporal_reliability_memory_mode),
                    "csais_temporal_reliability_memory_scale": float(temporal_reliability_scale),
                    "csais_temporal_reliability_memory_bucket": str(temporal_reliability_diag.get("bucket") or ""),
                    "csais_temporal_reliability_memory_guidance_bucket": str(temporal_reliability_diag.get("guidance_bucket") or ""),
                    "csais_temporal_reliability_memory_duplicate_bucket": str(temporal_reliability_diag.get("duplicate_bucket") or ""),
                    "csais_temporal_reliability_memory_conflict_bucket": str(temporal_reliability_diag.get("conflict_bucket") or ""),
                    "csais_temporal_reliability_memory_support_bucket": str(temporal_reliability_diag.get("support_bucket") or ""),
                    "csais_temporal_reliability_memory_source": str(temporal_reliability_diag.get("source") or ""),
                    "csais_temporal_reliability_memory_reason": str(temporal_reliability_diag.get("reason") or ""),
                    "csais_temporal_reliability_memory_history_n": int(_safe_float(temporal_reliability_diag.get("history_n"), 0.0)),
                    "csais_temporal_reliability_memory_effect_mean": float(_safe_float(temporal_reliability_diag.get("effect_mean"), float("nan"))),
                    "csais_temporal_reliability_memory_win_rate": float(_safe_float(temporal_reliability_diag.get("win_rate"), float("nan"))),
                    "csais_temporal_reliability_memory_trust_score": float(_safe_float(temporal_reliability_diag.get("trust_score"), float("nan"))),
                    "csais_temporal_reliability_memory_duplicate_ratio": float(_safe_float(temporal_reliability_diag.get("duplicate_ratio"), 0.0)),
                    "csais_temporal_action_gate_mode": str(args.temporal_action_gate_mode),
                    "csais_temporal_action_gate_scale": float(temporal_action_gate_scale),
                    "csais_temporal_action_gate_source": str(temporal_action_gate_diag.get("source") or ""),
                    "csais_temporal_action_gate_history_scope": str(temporal_action_gate_diag.get("history_scope") or ""),
                    "csais_temporal_action_gate_history_source": str(temporal_action_gate_diag.get("history_source") or ""),
                    "csais_temporal_action_gate_reason": str(temporal_action_gate_diag.get("reason") or ""),
                    "csais_temporal_action_gate_history_n": int(_safe_float(temporal_action_gate_diag.get("history_n"), 0.0)),
                    "csais_temporal_action_gate_neighbor_n": int(_safe_float(temporal_action_gate_diag.get("neighbor_n"), 0.0)),
                    "csais_temporal_action_gate_pred_effect": float(_safe_float(temporal_action_gate_diag.get("pred_effect"), float("nan"))),
                    "csais_temporal_action_gate_neighbor_effect_mean": float(_safe_float(temporal_action_gate_diag.get("neighbor_effect_mean"), float("nan"))),
                    "csais_temporal_action_gate_neighbor_effect_std": float(_safe_float(temporal_action_gate_diag.get("neighbor_effect_std"), float("nan"))),
                    "csais_temporal_action_gate_win_rate": float(_safe_float(temporal_action_gate_diag.get("win_rate"), float("nan"))),
                    "csais_temporal_action_gate_trust_score": float(_safe_float(temporal_action_gate_diag.get("trust_score"), float("nan"))),
                    "csais_temporal_action_gate_mean_distance": float(_safe_float(temporal_action_gate_diag.get("mean_distance"), float("nan"))),
                    "csais_temporal_action_gate_features_json": json.dumps(sanitize_for_json(temporal_action_gate_features), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_marginal_smape_effect": float(temporal_marginal_smape_effect),
                    "csais_without_temporal_reference_delta_log": float(no_temporal_reference_delta),
                    "csais_without_temporal_reference_guarded_delta_log": float(no_temporal_reference_guarded_delta),
                    "csais_temporal_direct_candidate_score_mode": str(args.temporal_score_mode),
                    "csais_temporal_direct_candidate_support_mode": str(args.temporal_support_mode),
                    "csais_temporal_direct_candidate_direction_mode": str(args.temporal_direction_mode),
                    "csais_direct_expert_sign_match": float(arbitration["sign_match"]),
                    "csais_direct_expert_duplicate_ratio": float(arbitration["duplicate_ratio"]),
                    "csais_direct_expert_blend_support_sum": float(arbitration["support_sum"]),
                    "csais_direct_expert_blend_base_weight": float(_safe_float((arbitration.get("candidate_weights") or {}).get("compressed_base"), 0.0)),
                    "csais_direct_expert_blend_intrinsic_weight": float(_safe_float((arbitration.get("candidate_weights") or {}).get("intrinsic_direct"), 0.0)),
                    "csais_direct_expert_blend_temporal_weight": float(_safe_float((arbitration.get("candidate_weights") or {}).get("temporal_direct"), 0.0)),
                    "csais_direct_expert_blend_guidance_weight": float(_safe_float((arbitration.get("candidate_weights") or {}).get("guidance_expert"), 0.0)),
                    "csais_evidence_orthogonal_arbitration_active": int(_safe_float(evidence_orthogonal_diag.get("active"), 0.0)),
                    "csais_evidence_orthogonal_base_scale": float(_safe_float(evidence_orthogonal_diag.get("base_scale"), 1.0)),
                    "csais_evidence_orthogonal_guidance_scale": float(_safe_float(evidence_orthogonal_diag.get("guidance_scale"), 1.0)),
                    "csais_evidence_orthogonal_base_duplicate_ratio": float(_safe_float(evidence_orthogonal_diag.get("base_duplicate_ratio"), 0.0)),
                    "csais_evidence_orthogonal_guidance_duplicate_ratio": float(_safe_float(evidence_orthogonal_diag.get("guidance_duplicate_ratio"), 0.0)),
                    "csais_evidence_orthogonal_base_overlap_source": str(evidence_orthogonal_diag.get("base_overlap_source") or ""),
                    "csais_evidence_orthogonal_guidance_overlap_source": str(evidence_orthogonal_diag.get("guidance_overlap_source") or ""),
                    "csais_evidence_orthogonal_reason": str(evidence_orthogonal_diag.get("reason") or ""),
                    "csais_guidance_expert_mode": str(args.guidance_expert_mode),
                    "csais_guidance_expert_active": int(_safe_float(guidance_expert.get("active"), 0.0)),
                    "csais_guidance_expert_anchor_model": str(guidance_expert.get("anchor_model") or ""),
                    "csais_guidance_expert_delta_log": float(_safe_float(guidance_expert.get("delta"), 0.0)),
                    "csais_guidance_expert_support": float(_safe_float(guidance_expert.get("support"), 0.0)),
                    "csais_guidance_expert_guid_mid": float(_safe_float(guidance_expert.get("guid_mid"), float("nan"))),
                    "csais_guidance_expert_history_abs_log": float(_safe_float(guidance_expert.get("history_abs_log"), float("nan"))),
                    "csais_guidance_expert_history_count_scale": float(_safe_float(guidance_expert.get("history_count_scale"), 0.0)),
                    "csais_guidance_expert_history_trust_scale": float(_safe_float(guidance_expert.get("history_trust_scale"), 0.0)),
                    "csais_guidance_expert_reason": str(guidance_expert.get("reason") or ""),
                    "csais_direct_expert_arbitration_mode": str(args.arbitration_mode),
                    "csais_direct_expert_arbitration_blend_mode": str(arbitration_blend_mode),
                    "csais_integrated_expert_arbitration_action": str(integrated_arbitration.get("action", "")),
                    "csais_integrated_expert_arbitration_reason": str(integrated_arbitration.get("reason", "")),
                    "csais_integrated_expert_arbitration_target_source": str(integrated_arbitration.get("target_source", "")),
                    "csais_integrated_expert_arbitration_target_delta_log": float(_safe_float(integrated_arbitration.get("target_delta_log"), float("nan"))),
                    "csais_integrated_expert_arbitration_retained_post_guardrail_delta_log": float(_safe_float(integrated_arbitration.get("retained_post_guardrail_delta_log"), float("nan"))),
                    "csais_integrated_expert_arbitration_internal_active": int(_safe_float(integrated_arbitration.get("internal_active"), 0.0)),
                    "csais_integrated_expert_arbitration_base_prior_floor_active": int(_safe_float(integrated_arbitration.get("base_prior_floor_active"), 0.0)),
                    "csais_integrated_expert_arbitration_timing_blocked": int(_safe_float(integrated_arbitration.get("timing_blocked"), 0.0)),
                    "csais_integrated_expert_arbitration_floor_ratio": float(_safe_float(integrated_arbitration.get("floor_ratio"), float("nan"))),
                    "csais_integrated_expert_arbitration_floor_pred": float(_safe_float(integrated_arbitration.get("floor_pred"), float("nan"))),
                    "csais_base_candidate_delta_log": float(base_delta),
                    "csais_final_candidate_delta_log": float(final_delta),
                    "csais_pre_guidance_guardrail_candidate_delta_log": float(pre_guidance_guardrail_delta),
                    "csais_guidance_quality_guardrail_mode": str(guidance_guardrail["mode"]),
                    "csais_guidance_quality_guardrail_alpha": float(guidance_guardrail["alpha"]),
                    "csais_guidance_quality_guardrail_applied": int(guidance_guardrail["applied"]),
                    "csais_direct_expert_gate_used": int(bool(gate_used)),
                    "csais_direct_expert_gate_scope": str(gate_scope_applied),
                    "csais_direct_expert_gate_support_used": float(gate_support_used),
                    "csais_direct_expert_gate_scale_applied": float(gate_scale_applied),
                    "csais_direct_expert_gate_train_count": int(local_gate_result["train_count"]),
                    "pred_csais_rawcard_direct_experts_v1_pre_guidance_guardrail": float(pre_guidance_guardrail_pred),
                    "pred_csais_rawcard_direct_experts_v1": float(final_pred),
                    "shock_target_log": float(shock_target_log),
                    "gate_residual_target_log": float(gate_residual_target_log),
                    "csais_raw_candidate_top_contribs": json.dumps(sanitize_for_json(raw_result["top_contribs"]), ensure_ascii=False, allow_nan=False),
                    "csais_factor_candidate_top_contribs": json.dumps(sanitize_for_json(factor_result["top_contribs"]), ensure_ascii=False, allow_nan=False),
                    "csais_intrinsic_direct_candidate_top_contribs": json.dumps(sanitize_for_json(intrinsic_result.get("top_contribs", [])), ensure_ascii=False, allow_nan=False),
                    "csais_segment_bridge_candidate_top_contribs": json.dumps(sanitize_for_json(segment_bridge_result.get("top_contribs", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_direct_candidate_retrieved_quarters": json.dumps(sanitize_for_json(temporal_result.get("retrieved_quarters", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_direct_candidate_top_matches": json.dumps(sanitize_for_json(temporal_result.get("top_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_state_analog_features_json": json.dumps(sanitize_for_json(state_analog_temporal_features), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_state_analog_top_matches": json.dumps(sanitize_for_json(temporal_state_analog_result.get("top_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_context_retrieval_top_matches": json.dumps(sanitize_for_json(temporal_result.get("context_top_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_context_memory_retrieved_quarters": json.dumps(sanitize_for_json(temporal_context_result.get("retrieved_quarters", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_context_memory_top_matches": json.dumps(sanitize_for_json(temporal_context_result.get("top_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_forward_filter_diag_json": json.dumps(sanitize_for_json(temporal_forward_filter_diag), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_context_filter_diag_json": json.dumps(sanitize_for_json(temporal_context_filter_diag), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_context_quality_diag_json": json.dumps(sanitize_for_json(temporal_context_quality_diag), ensure_ascii=False, allow_nan=False),
                    "csais_direct_expert_blend_supports_json": json.dumps(sanitize_for_json(arbitration.get("candidate_supports", {})), ensure_ascii=False, allow_nan=False),
                    "csais_direct_expert_blend_weights_json": json.dumps(sanitize_for_json(arbitration.get("candidate_weights", {})), ensure_ascii=False, allow_nan=False),
                    "csais_evidence_orthogonal_diag_json": json.dumps(sanitize_for_json(evidence_orthogonal_diag), ensure_ascii=False, allow_nan=False),
                    "csais_direct_expert_gate_top_contribs": json.dumps(sanitize_for_json(local_gate_result.get("top_contribs", [])), ensure_ascii=False, allow_nan=False),
                    "csais_candidate_memory_top_matches": json.dumps(sanitize_for_json(memory_diag.get("top_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_intrinsic_direct_support_cards_json": json.dumps(sanitize_for_json(intrinsic_support_cards), ensure_ascii=False, allow_nan=False),
                    "csais_intrinsic_direct_conflict_cards_json": json.dumps(sanitize_for_json(intrinsic_conflict_cards), ensure_ascii=False, allow_nan=False),
                    "csais_native_forward_rows_json": json.dumps(sanitize_for_json(list(native_forward_rows)[:12]), ensure_ascii=False, allow_nan=False),
                }
            )
            quarterly_rows.append(out_row)
            memory_record = {
                **native_current_record,
                "quarter_qnum": int(_quarter_ordinal(str(row.get("quarter") or ""))),
                "target_delta_log": float(shock_target_log),
                "intrinsic_residual_target_log": float(intrinsic_residual_target_log),
                "intrinsic_partitioned_residual_target_log": float(intrinsic_partitioned_residual_target_log),
                "intrinsic_raw_marginal_smape_effect": float(intrinsic_raw_marginal_smape_effect),
                "intrinsic_action_marginal_smape_effect": float(intrinsic_action_marginal_smape_effect),
                "intrinsic_reliability_bucket": str(intrinsic_reliability_diag.get("bucket") or ""),
                "intrinsic_reliability_guidance_bucket": _intrinsic_reliability_guidance_bucket(row),
                "intrinsic_action_gate_features": dict(intrinsic_action_gate_features),
                "intrinsic_action_gate_scale": float(intrinsic_action_gate_scale),
                "intrinsic_action_gate_mode": str(args.intrinsic_action_gate_mode),
                "intrinsic_action_gate_reason": str(intrinsic_action_gate_diag.get("reason") or ""),
                "temporal_marginal_smape_effect": float(temporal_marginal_smape_effect),
                "temporal_reliability_bucket": str(temporal_reliability_diag.get("bucket") or ""),
                "temporal_reliability_guidance_bucket": str(temporal_reliability_diag.get("guidance_bucket") or ""),
                "temporal_reliability_conflict_bucket": str(temporal_reliability_diag.get("conflict_bucket") or ""),
                "temporal_action_gate_features": dict(temporal_action_gate_features),
                "temporal_action_gate_scale": float(temporal_action_gate_scale),
                "temporal_action_gate_mode": str(args.temporal_action_gate_mode),
                "temporal_action_gate_reason": str(temporal_action_gate_diag.get("reason") or ""),
                "state_analog_temporal_features": dict(state_analog_temporal_features),
                "forward_rows": list(native_forward_rows),
                "context_rows": list(native_context_rows),
                "temporal_forward_rows": list(temporal_forward_rows),
                "temporal_context_rows": list(temporal_context_rows),
            }
            native_memory_history.append(memory_record)
            panel_action_gate_history.append(memory_record)
            if np.isfinite(actual_log) and np.isfinite(anchor_log):
                anchor_error_history.append(
                    {
                        "quarter": str(row.get("quarter") or ""),
                        "guidance_availability": current_guidance,
                        "anchor_abs_log_error": float(abs(actual_log - anchor_log)),
                    }
                )
            if str(args.anchor_selection_mode) == "online_historical_mae":
                _append_anchor_model_errors(row, model_cols, anchor_model_error_history)
            actual_hist.append(_safe_float(row.get("actual")))
            quarter_hist.append(str(row.get("quarter") or ""))

        quarterly_df = pd.DataFrame(quarterly_rows)
        if not quarterly_df.empty:
            quarterly_df = quarterly_df[
                quarterly_df["quarter"].map(_quarter_key).map(lambda key: report_start_key <= key <= report_end_key)
            ].reset_index(drop=True)
        emitted_report_quarters = quarterly_df["quarter"].astype(str).tolist() if not quarterly_df.empty else []
        expected_set = set(expected_report_quarters)
        panel_report_set = set(panel_report_quarters)
        emitted_set = set(emitted_report_quarters)
        missing_from_panel = [q for q in expected_report_quarters if q not in panel_report_set]
        skipped_after_panel = [q for q in panel_report_quarters if q not in emitted_set]
        report_skip_records = [rec for rec in company_skip_records if str(rec.get("quarter") or "") in expected_set]
        status_parts: List[str] = []
        if missing_from_panel:
            status_parts.append("missing_from_panel")
        if skipped_after_panel:
            status_parts.append("skipped_after_panel")
        coverage_status = "+".join(status_parts) if status_parts else "ok"
        coverage_record = {
            "ticker": ticker,
            "status": coverage_status,
            "evaluation_start_fq": str(company.get("evaluation_start_fq") or ""),
            "evaluation_end_fq": str(company.get("evaluation_end_fq") or ""),
            "report_start_quarter": str(args.report_start_quarter),
            "report_end_quarter": str(args.report_end_quarter),
            "expected_start_quarter": _quarter_label(company_report_start_key) if expected_report_quarters else "",
            "expected_end_quarter": _quarter_label(company_report_end_key) if expected_report_quarters else "",
            "expected_report_n": int(len(expected_report_quarters)),
            "panel_report_n": int(len(panel_report_quarters)),
            "emitted_report_n": int(len(emitted_report_quarters)),
            "missing_from_panel_n": int(len(missing_from_panel)),
            "skipped_after_panel_n": int(len(skipped_after_panel)),
            "anchor_invalid_skip_n": int(sum(1 for rec in report_skip_records if str(rec.get("reason") or "") == "missing_or_nonfinite_anchor")),
            "expected_report_quarters_json": json.dumps(expected_report_quarters, ensure_ascii=False),
            "panel_report_quarters_json": json.dumps(panel_report_quarters, ensure_ascii=False),
            "emitted_report_quarters_json": json.dumps(emitted_report_quarters, ensure_ascii=False),
            "missing_from_panel_json": json.dumps(missing_from_panel, ensure_ascii=False),
            "skipped_after_panel_json": json.dumps(skipped_after_panel, ensure_ascii=False),
        }
        coverage_records.append(coverage_record)
        coverage_skip_records.extend(report_skip_records)
        if quarterly_df.empty:
            continue
        company_metrics = {
            "baseline": _metrics(quarterly_df["actual"], quarterly_df["baseline_pred"]),
            "csais_anchor": _metrics(quarterly_df["actual"], quarterly_df["pred_csais_anchor"]),
            "rawcard_direct_experts": _metrics(quarterly_df["actual"], quarterly_df["pred_csais_rawcard_direct_experts_v1"]),
        }
        company_summaries.append(
            {
                "ticker": ticker,
                "n": int(len(quarterly_df)),
                "method_family": str(method_family),
                "expert_contract": str(expert_contract),
                "baseline_mae": float(company_metrics["baseline"]["mae"]),
                "csais_anchor_mae": float(company_metrics["csais_anchor"]["mae"]),
                "rawcard_direct_experts_mae": float(company_metrics["rawcard_direct_experts"]["mae"]),
                "best_stat_model": best_stat_model,
                "best_stat_mae": float(best_stat_mae),
                "anchor_selection_mode": str(args.anchor_selection_mode),
                "beats_best_stat": bool(float(company_metrics["rawcard_direct_experts"]["mae"]) < float(best_stat_mae)),
            }
        )
        all_quarterly.append(quarterly_df)

    quarterly = pd.concat(all_quarterly, axis=0).sort_values(["ticker", "quarter"], key=lambda s: s.map(_quarter_key) if s.name == "quarter" else s).reset_index(drop=True)
    regime_aware_base_anchor_validation: Dict[str, Any] = {"mode": str(args.regime_aware_base_anchor_mode), "active_rows": 0}
    shared_residual_backbone_validation: Dict[str, Any] = {"mode": str(args.arbitration_mode), "active_aux_rows": 0}
    if str(args.arbitration_mode) == SHARED_RESIDUAL_BACKBONE_MODE:
        quarterly, shared_residual_backbone_validation = _apply_shared_residual_backbone_panel(
            quarterly,
            proposal_map=regime_aware_base_anchor_proposals,
            guidance_quality_guardrail_mode=str(args.guidance_quality_guardrail_mode),
            min_history=int(args.regime_aware_base_anchor_min_history),
            signed_strength_threshold=float(args.regime_aware_base_anchor_signed_strength_threshold),
            upward_ratio=float(args.regime_aware_base_anchor_upward_ratio),
        )
        if str(args.regime_aware_base_anchor_mode) != "off":
            regime_aware_base_anchor_validation = {
                "mode": str(args.regime_aware_base_anchor_mode),
                "active_rows": 0,
                "reason": f"not_applied_when_{SHARED_RESIDUAL_BACKBONE_MODE}_uses_memory_as_auxiliary_delta",
            }
        company_df = _company_summary_from_quarterly(quarterly, method_family, expert_contract)
    elif str(args.regime_aware_base_anchor_mode) != "off":
        quarterly, regime_aware_base_anchor_validation = _apply_regime_aware_base_anchor_panel(
            quarterly,
            proposal_map=regime_aware_base_anchor_proposals,
            mode=str(args.regime_aware_base_anchor_mode),
            min_history=int(args.regime_aware_base_anchor_min_history),
            signed_strength_threshold=float(args.regime_aware_base_anchor_signed_strength_threshold),
            upward_ratio=float(args.regime_aware_base_anchor_upward_ratio),
        )
        company_df = _company_summary_from_quarterly(quarterly, method_family, expert_contract)
    else:
        company_df = pd.DataFrame(company_summaries).sort_values("ticker")
    quarterly_csv = out_dir / f"{method_family}_quarterly.csv"
    company_csv = out_dir / f"{method_family}_company_summary.csv"
    coverage_csv = out_dir / f"{method_family}_coverage_diagnostics.csv"
    coverage_json = out_dir / f"{method_family}_coverage_diagnostics.json"
    quarterly.to_csv(quarterly_csv, index=False)
    company_df.to_csv(company_csv, index=False)
    pd.DataFrame(coverage_records).to_csv(coverage_csv, index=False)
    write_json(
        coverage_json,
        {
            "coverage_drift_mode": str(args.coverage_drift_mode),
            "missing_anchor_policy": str(args.missing_anchor_policy),
            "records": coverage_records,
            "skipped_rows": coverage_skip_records,
        },
    )

    contract_payload = {
        "method_family": str(method_family),
        "expert_contract": str(expert_contract),
        "active_experts": list(active_experts),
        "intrinsic_expert_mode": ("grouped_raw_card_direct_anchor_correction_v3" if str(args.intrinsic_mode) == "grouped_ridge_v3" else "card_level_additive_direct_anchor_correction_v1"),
        "temporal_expert_mode": (
            "raw_card_graph_attention_direct_anchor_correction_v5_typed_soft_agreement"
            if str(args.temporal_context_memory_mode) == "typed_soft_agreement"
            else (
                "raw_card_graph_attention_direct_anchor_correction_v4_typed_reliability"
                if str(args.temporal_context_memory_mode) == "typed_reliability"
                else (
                    "raw_card_graph_attention_direct_anchor_correction_v3_typed_context_retrieval"
                    if str(args.temporal_context_memory_mode) == "typed_retrieval"
                    else (
                        "raw_card_graph_attention_direct_anchor_correction_v2_context_memory"
                        if str(args.temporal_context_memory_mode) != "off"
                        else "raw_card_graph_attention_direct_anchor_correction_v1"
                    )
                )
            )
        ),
        "temporal_score_mode": str(args.temporal_score_mode),
        "temporal_support_mode": str(args.temporal_support_mode),
        "temporal_direction_mode": str(args.temporal_direction_mode),
        "temporal_context_guard_mode": str(args.temporal_context_guard_mode),
        "temporal_context_memory_mode": str(args.temporal_context_memory_mode),
        "temporal_evidence_filter_mode": str(args.temporal_evidence_filter_mode),
        "temporal_evidence_filter_scope": str(args.temporal_evidence_filter_scope),
        "temporal_context_quality_mode": str(args.temporal_context_quality_mode),
        "temporal_context_quality_weight": float(args.temporal_context_quality_weight),
        "temporal_context_retrieval_weight": float(args.temporal_context_retrieval_weight),
        "temporal_segment_compat_weight": float(args.temporal_segment_compat_weight),
        "temporal_context_support_weight": float(args.temporal_context_support_weight),
        "temporal_novelty_shrink_weight": float(args.temporal_novelty_shrink_weight),
        "temporal_guidance_trust_weight": float(args.temporal_guidance_trust_weight),
        "temporal_segment_support_weight": float(args.temporal_segment_support_weight),
        "temporal_soft_agreement_weight": float(args.temporal_soft_agreement_weight),
        "temporal_context_memory_sparsity_power": float(args.temporal_context_memory_sparsity_power),
        "temporal_guidance_bucket_scale_mode": str(args.temporal_guidance_bucket_scale_mode),
        "temporal_explicit_guidance_scale": float(args.temporal_explicit_guidance_scale),
        "temporal_non_explicit_guidance_scale": float(args.temporal_non_explicit_guidance_scale),
        "temporal_no_guidance_scale": float(args.temporal_no_guidance_scale),
        "temporal_anchor_confidence_guard_mode": str(args.temporal_anchor_confidence_guard_mode),
        "temporal_anchor_confidence_guard_history_mae_threshold": float(args.temporal_anchor_confidence_guard_history_mae_threshold),
        "temporal_anchor_confidence_guard_min_abs_delta": float(args.temporal_anchor_confidence_guard_min_abs_delta),
        "temporal_anchor_confidence_guard_min_support": float(args.temporal_anchor_confidence_guard_min_support),
        "temporal_anchor_confidence_guard_scale": float(args.temporal_anchor_confidence_guard_scale),
        "temporal_interaction_guard_mode": str(args.temporal_interaction_guard_mode),
        "temporal_interaction_duplicate_threshold": float(args.temporal_interaction_duplicate_threshold),
        "temporal_interaction_duplicate_scale": float(args.temporal_interaction_duplicate_scale),
        "temporal_interaction_explicit_scale": float(args.temporal_interaction_explicit_scale),
        "temporal_interaction_non_explicit_scale": float(args.temporal_interaction_non_explicit_scale),
        "temporal_interaction_guidance_active_scale": float(args.temporal_interaction_guidance_active_scale),
        "temporal_interaction_conflict_scale": float(args.temporal_interaction_conflict_scale),
        "temporal_interaction_min_support": float(args.temporal_interaction_min_support),
        "temporal_reliability_memory_mode": str(args.temporal_reliability_memory_mode),
        "temporal_reliability_min_history": int(args.temporal_reliability_min_history),
        "temporal_reliability_tau": float(args.temporal_reliability_tau),
        "temporal_reliability_min_scale": float(args.temporal_reliability_min_scale),
        "temporal_reliability_duplicate_threshold": float(args.temporal_reliability_duplicate_threshold),
        "temporal_reliability_support_medium_threshold": float(args.temporal_reliability_support_medium_threshold),
        "temporal_reliability_support_high_threshold": float(args.temporal_reliability_support_high_threshold),
        "temporal_reliability_abstain_trust_threshold": float(args.temporal_reliability_abstain_trust_threshold),
        "temporal_state_analog_mode": str(args.temporal_state_analog_mode),
        "temporal_state_analog_min_history": int(args.temporal_state_analog_min_history),
        "temporal_state_analog_neighbor_k": int(args.temporal_state_analog_neighbor_k),
        "temporal_state_analog_history_cap": int(args.temporal_state_analog_history_cap),
        "temporal_state_analog_distance_tau": float(args.temporal_state_analog_distance_tau),
        "temporal_state_analog_var_tau": float(args.temporal_state_analog_var_tau),
        "temporal_state_analog_neff_scale": float(args.temporal_state_analog_neff_scale),
        "temporal_state_analog_support_scale": float(args.temporal_state_analog_support_scale),
        "temporal_state_analog_blend_weight": float(args.temporal_state_analog_blend_weight),
        "temporal_state_analog_feature_names": list(STATE_ANALOG_TEMPORAL_FEATURES),
        "temporal_action_gate_mode": str(args.temporal_action_gate_mode),
        "temporal_action_gate_min_history": int(args.temporal_action_gate_min_history),
        "temporal_action_gate_neighbor_k": int(args.temporal_action_gate_neighbor_k),
        "temporal_action_gate_tau": float(args.temporal_action_gate_tau),
        "temporal_action_gate_min_scale": float(args.temporal_action_gate_min_scale),
        "temporal_action_gate_abstain_effect_threshold": float(args.temporal_action_gate_abstain_effect_threshold),
        "temporal_action_gate_abstain_win_rate_threshold": float(args.temporal_action_gate_abstain_win_rate_threshold),
        "temporal_action_gate_feature_names": list(TEMPORAL_ACTION_GATE_FEATURES),
        "action_gate_history_scope": str(args.action_gate_history_scope),
        "action_gate_panel_history_csv": str(resolve_repo_path(args.action_gate_panel_history_csv, str(project_root))) if str(args.action_gate_panel_history_csv or "").strip() else "",
        "action_gate_panel_history_seed_rows": int(len(action_gate_panel_history_seed)),
        "guidance_quality_guardrail_mode": str(args.guidance_quality_guardrail_mode),
        "guidance_quality_guardrail_alpha": dict(GUIDANCE_QUALITY_GUARDRAIL_ALPHA),
        "guidance_expert_mode": str(args.guidance_expert_mode),
        "guidance_expert_min_history": int(args.guidance_expert_min_history),
        "guidance_expert_history_tau": float(args.guidance_expert_history_tau),
        "guidance_expert_support_scale": float(args.guidance_expert_support_scale),
        "regime_aware_base_anchor_mode": str(args.regime_aware_base_anchor_mode),
        "regime_aware_base_anchor_proposal_csv": str(resolve_repo_path(args.regime_aware_base_anchor_proposal_csv, str(project_root))) if str(args.regime_aware_base_anchor_proposal_csv or "").strip() else "",
        "regime_aware_base_anchor_proposal_pred_col": str(args.regime_aware_base_anchor_proposal_pred_col),
        "regime_aware_base_anchor_proposal_filter_col": str(args.regime_aware_base_anchor_proposal_filter_col),
        "regime_aware_base_anchor_proposal_filter_value": str(args.regime_aware_base_anchor_proposal_filter_value),
        "regime_aware_base_anchor_min_history": int(args.regime_aware_base_anchor_min_history),
        "regime_aware_base_anchor_signed_strength_threshold": float(args.regime_aware_base_anchor_signed_strength_threshold),
        "regime_aware_base_anchor_upward_ratio": float(args.regime_aware_base_anchor_upward_ratio),
        "regime_aware_base_anchor_validation": sanitize_for_json(regime_aware_base_anchor_validation),
        "shared_residual_backbone_validation": sanitize_for_json(shared_residual_backbone_validation),
        "anchor_selection_mode": str(args.anchor_selection_mode),
        "anchor_override_csv": str(resolve_repo_path(args.anchor_override_csv, str(project_root))) if str(args.anchor_override_csv or "").strip() else "",
        "anchor_online_min_history": int(args.anchor_online_min_history),
        "anchor_online_score_metric": str(args.anchor_online_score_metric),
        "anchor_online_window": int(args.anchor_online_window),
        "anchor_online_half_life": float(args.anchor_online_half_life),
        "anchor_online_same_quarter_weight": float(args.anchor_online_same_quarter_weight),
        "anchor_guidance_regime_mode": str(args.anchor_guidance_regime_mode),
        "anchor_guidance_same_regime_min_history": int(args.anchor_guidance_same_regime_min_history),
        "anchor_guidance_mismatch_penalty": float(args.anchor_guidance_mismatch_penalty),
        "anchor_explicit_guidance_proximity_mode": str(args.anchor_explicit_guidance_proximity_mode),
        "anchor_explicit_guidance_proximity_weight": float(args.anchor_explicit_guidance_proximity_weight),
        "anchor_explicit_guidance_kernel_min_history": int(args.anchor_explicit_guidance_kernel_min_history),
        "anchor_explicit_guidance_kernel_band": float(args.anchor_explicit_guidance_kernel_band),
        "anchor_explicit_guidance_kernel_shrink_k": float(args.anchor_explicit_guidance_kernel_shrink_k),
        "anchor_robust_momentum_mode": str(args.anchor_robust_momentum_mode),
        "coverage_drift_mode": str(args.coverage_drift_mode),
        "missing_anchor_policy": str(args.missing_anchor_policy),
        "arbitration_mode": str(args.arbitration_mode),
        "evidence_orthogonal_duplicate_threshold": float(args.evidence_orthogonal_duplicate_threshold),
        "evidence_orthogonal_base_min_scale": float(args.evidence_orthogonal_base_min_scale),
        "evidence_orthogonal_guidance_min_scale": float(args.evidence_orthogonal_guidance_min_scale),
        "evidence_orthogonal_base_strength": float(args.evidence_orthogonal_base_strength),
        "evidence_orthogonal_guidance_strength": float(args.evidence_orthogonal_guidance_strength),
        "integrated_arbitration_floor_derivation": sanitize_for_json(integrated_arbitration_floor_derivation),
        "integrated_arbitration_internal_policy": sanitize_for_json(INTEGRATED_ARBITRATION_INTERNAL_POLICY),
        "gate_mode": str(args.gate_mode),
        "intrinsic_mode": str(args.intrinsic_mode),
        "intrinsic_target_mode": str(args.intrinsic_target_mode),
        "intrinsic_reliability_mode": str(args.intrinsic_reliability_mode),
        "intrinsic_reliability_min_history": int(args.intrinsic_reliability_min_history),
        "intrinsic_reliability_tau": float(args.intrinsic_reliability_tau),
        "intrinsic_reliability_min_scale": float(args.intrinsic_reliability_min_scale),
        "intrinsic_reliability_max_scale": float(args.intrinsic_reliability_max_scale),
        "intrinsic_explicit_guidance_guard_mode": str(args.intrinsic_explicit_guidance_guard_mode),
        "intrinsic_explicit_guidance_support_scale": float(args.intrinsic_explicit_guidance_support_scale),
        "intrinsic_temporal_dedup_mode": str(args.intrinsic_temporal_dedup_mode),
        "intrinsic_temporal_dedup_duplicate_threshold": float(args.intrinsic_temporal_dedup_duplicate_threshold),
        "intrinsic_temporal_dedup_min_scale": float(args.intrinsic_temporal_dedup_min_scale),
        "intrinsic_temporal_dedup_strength": float(args.intrinsic_temporal_dedup_strength),
        "intrinsic_action_gate_mode": str(args.intrinsic_action_gate_mode),
        "intrinsic_action_gate_min_history": int(args.intrinsic_action_gate_min_history),
        "intrinsic_action_gate_neighbor_k": int(args.intrinsic_action_gate_neighbor_k),
        "intrinsic_action_gate_tau": float(args.intrinsic_action_gate_tau),
        "intrinsic_action_gate_min_scale": float(args.intrinsic_action_gate_min_scale),
        "intrinsic_action_gate_abstain_effect_threshold": float(args.intrinsic_action_gate_abstain_effect_threshold),
        "intrinsic_action_gate_abstain_win_rate_threshold": float(args.intrinsic_action_gate_abstain_win_rate_threshold),
        "intrinsic_action_gate_feature_names": list(INTRINSIC_ACTION_GATE_FEATURES),
        "intrinsic_grouped_max_strict_features": int(args.intrinsic_grouped_max_strict_features),
        "intrinsic_grouped_max_loose_features": int(args.intrinsic_grouped_max_loose_features),
        "intrinsic_grouped_min_feature_occurrence": int(args.intrinsic_grouped_min_feature_occurrence),
        "segment_bridge_mode": str(args.segment_bridge_mode),
        "segment_bridge_alpha": float(args.segment_bridge_alpha),
        "segment_bridge_min_train": int(args.segment_bridge_min_train),
        "segment_bridge_shrink_k": float(args.segment_bridge_shrink_k),
        "segment_bridge_blend_weight": float(args.segment_bridge_blend_weight),
        "gate_feature_names": list(DIRECT_EXPERT_GATE_FEATURES),
        "gate_feature_count": int(len(DIRECT_EXPERT_GATE_FEATURES)),
    }
    contract_json = out_dir / f"{method_family}_contract.json"
    write_json(contract_json, contract_payload)
    coverage_issue_records = [rec for rec in coverage_records if str(rec.get("status") or "ok") != "ok"]

    summary = {
        "inputs": {
            "experiment_config": str(resolve_repo_path(args.experiment_config, str(project_root))),
            "native_backbone_csv": str(resolve_repo_path(args.native_backbone_csv, str(project_root))),
            "native_card_table_jsonl": str(resolve_repo_path(args.native_card_table_jsonl, str(project_root))),
            "tickers": sorted(company_df["ticker"].tolist()),
            "frozen_best_stat_csv": str(resolve_repo_path(args.frozen_best_stat_csv, str(project_root))) if str(args.frozen_best_stat_csv or "").strip() else "",
            "anchor_override_csv": str(resolve_repo_path(args.anchor_override_csv, str(project_root))) if str(args.anchor_override_csv or "").strip() else "",
            "method_family": str(method_family),
            "expert_contract": str(expert_contract),
            "active_experts": list(active_experts),
            "gate_mode": str(args.gate_mode),
            "intrinsic_mode": str(args.intrinsic_mode),
            "intrinsic_target_mode": str(args.intrinsic_target_mode),
            "intrinsic_reliability_mode": str(args.intrinsic_reliability_mode),
            "intrinsic_reliability_min_history": int(args.intrinsic_reliability_min_history),
            "intrinsic_reliability_tau": float(args.intrinsic_reliability_tau),
            "intrinsic_reliability_min_scale": float(args.intrinsic_reliability_min_scale),
            "intrinsic_reliability_max_scale": float(args.intrinsic_reliability_max_scale),
            "intrinsic_explicit_guidance_guard_mode": str(args.intrinsic_explicit_guidance_guard_mode),
            "intrinsic_explicit_guidance_support_scale": float(args.intrinsic_explicit_guidance_support_scale),
            "intrinsic_temporal_dedup_mode": str(args.intrinsic_temporal_dedup_mode),
            "intrinsic_temporal_dedup_duplicate_threshold": float(args.intrinsic_temporal_dedup_duplicate_threshold),
            "intrinsic_temporal_dedup_min_scale": float(args.intrinsic_temporal_dedup_min_scale),
            "intrinsic_temporal_dedup_strength": float(args.intrinsic_temporal_dedup_strength),
            "intrinsic_action_gate_mode": str(args.intrinsic_action_gate_mode),
            "intrinsic_action_gate_min_history": int(args.intrinsic_action_gate_min_history),
            "intrinsic_action_gate_neighbor_k": int(args.intrinsic_action_gate_neighbor_k),
            "intrinsic_action_gate_tau": float(args.intrinsic_action_gate_tau),
            "intrinsic_action_gate_min_scale": float(args.intrinsic_action_gate_min_scale),
            "intrinsic_action_gate_abstain_effect_threshold": float(args.intrinsic_action_gate_abstain_effect_threshold),
            "intrinsic_action_gate_abstain_win_rate_threshold": float(args.intrinsic_action_gate_abstain_win_rate_threshold),
            "intrinsic_action_gate_feature_names": list(INTRINSIC_ACTION_GATE_FEATURES),
            "report_start_quarter": str(args.report_start_quarter),
            "report_end_quarter": str(args.report_end_quarter),
            "raw_alpha": float(args.raw_alpha),
            "factor_alpha": float(args.factor_alpha),
            "intrinsic_alpha": float(args.intrinsic_alpha),
            "candidate_min_train": int(args.candidate_min_train),
            "intrinsic_min_train": int(args.intrinsic_min_train),
            "candidate_shrink_k": float(args.candidate_shrink_k),
            "intrinsic_shrink_k": float(args.intrinsic_shrink_k),
            "intrinsic_grouped_max_strict_features": int(args.intrinsic_grouped_max_strict_features),
            "intrinsic_grouped_max_loose_features": int(args.intrinsic_grouped_max_loose_features),
            "intrinsic_grouped_min_feature_occurrence": int(args.intrinsic_grouped_min_feature_occurrence),
            "segment_bridge_mode": str(args.segment_bridge_mode),
            "segment_bridge_alpha": float(args.segment_bridge_alpha),
            "segment_bridge_min_train": int(args.segment_bridge_min_train),
            "segment_bridge_shrink_k": float(args.segment_bridge_shrink_k),
            "segment_bridge_blend_weight": float(args.segment_bridge_blend_weight),
            "candidate_base_scale": float(args.candidate_base_scale),
            "gate_alpha": float(args.gate_alpha),
            "gate_min_train": int(args.gate_min_train),
            "gate_shrink_k": float(args.gate_shrink_k),
            "memory_top_k": int(args.memory_top_k),
            "memory_temperature": float(args.memory_temperature),
            "memory_min_train": int(args.memory_min_train),
            "temporal_min_train": int(args.temporal_min_train),
            "temporal_history_cap": int(args.temporal_history_cap),
            "temporal_top_k": int(args.temporal_top_k),
            "temporal_temperature": float(args.temporal_temperature),
            "temporal_item_top_k": int(args.temporal_item_top_k),
            "temporal_item_temperature": float(args.temporal_item_temperature),
            "temporal_same_quarter_bonus": float(args.temporal_same_quarter_bonus),
            "temporal_time_decay_quarters": float(args.temporal_time_decay_quarters),
            "temporal_var_tau": float(args.temporal_var_tau),
            "temporal_neff_scale": float(args.temporal_neff_scale),
            "temporal_directional_consistency_power": float(args.temporal_directional_consistency_power),
            "temporal_attention_focus_power": float(args.temporal_attention_focus_power),
            "temporal_max_abs_log_correction": float(args.temporal_max_abs_log_correction),
            "temporal_score_mode": str(args.temporal_score_mode),
            "temporal_support_mode": str(args.temporal_support_mode),
            "temporal_direction_mode": str(args.temporal_direction_mode),
            "temporal_context_guard_mode": str(args.temporal_context_guard_mode),
            "temporal_context_memory_mode": str(args.temporal_context_memory_mode),
            "temporal_evidence_filter_mode": str(args.temporal_evidence_filter_mode),
            "temporal_evidence_filter_scope": str(args.temporal_evidence_filter_scope),
            "temporal_context_quality_mode": str(args.temporal_context_quality_mode),
            "temporal_context_quality_weight": float(args.temporal_context_quality_weight),
            "temporal_context_retrieval_weight": float(args.temporal_context_retrieval_weight),
            "temporal_segment_compat_weight": float(args.temporal_segment_compat_weight),
            "temporal_context_support_weight": float(args.temporal_context_support_weight),
            "temporal_novelty_shrink_weight": float(args.temporal_novelty_shrink_weight),
            "temporal_guidance_trust_weight": float(args.temporal_guidance_trust_weight),
            "temporal_segment_support_weight": float(args.temporal_segment_support_weight),
            "temporal_soft_agreement_weight": float(args.temporal_soft_agreement_weight),
            "temporal_context_memory_sparsity_power": float(args.temporal_context_memory_sparsity_power),
            "temporal_guidance_bucket_scale_mode": str(args.temporal_guidance_bucket_scale_mode),
            "temporal_explicit_guidance_scale": float(args.temporal_explicit_guidance_scale),
            "temporal_non_explicit_guidance_scale": float(args.temporal_non_explicit_guidance_scale),
            "temporal_no_guidance_scale": float(args.temporal_no_guidance_scale),
            "temporal_interaction_guard_mode": str(args.temporal_interaction_guard_mode),
            "temporal_interaction_duplicate_threshold": float(args.temporal_interaction_duplicate_threshold),
            "temporal_interaction_duplicate_scale": float(args.temporal_interaction_duplicate_scale),
            "temporal_interaction_explicit_scale": float(args.temporal_interaction_explicit_scale),
            "temporal_interaction_non_explicit_scale": float(args.temporal_interaction_non_explicit_scale),
            "temporal_interaction_guidance_active_scale": float(args.temporal_interaction_guidance_active_scale),
            "temporal_interaction_conflict_scale": float(args.temporal_interaction_conflict_scale),
            "temporal_interaction_min_support": float(args.temporal_interaction_min_support),
            "temporal_reliability_memory_mode": str(args.temporal_reliability_memory_mode),
            "temporal_reliability_min_history": int(args.temporal_reliability_min_history),
            "temporal_reliability_tau": float(args.temporal_reliability_tau),
            "temporal_reliability_min_scale": float(args.temporal_reliability_min_scale),
            "temporal_reliability_duplicate_threshold": float(args.temporal_reliability_duplicate_threshold),
            "temporal_reliability_support_medium_threshold": float(args.temporal_reliability_support_medium_threshold),
            "temporal_reliability_support_high_threshold": float(args.temporal_reliability_support_high_threshold),
            "temporal_reliability_abstain_trust_threshold": float(args.temporal_reliability_abstain_trust_threshold),
            "temporal_state_analog_mode": str(args.temporal_state_analog_mode),
            "temporal_state_analog_min_history": int(args.temporal_state_analog_min_history),
            "temporal_state_analog_neighbor_k": int(args.temporal_state_analog_neighbor_k),
            "temporal_state_analog_history_cap": int(args.temporal_state_analog_history_cap),
            "temporal_state_analog_distance_tau": float(args.temporal_state_analog_distance_tau),
            "temporal_state_analog_var_tau": float(args.temporal_state_analog_var_tau),
            "temporal_state_analog_neff_scale": float(args.temporal_state_analog_neff_scale),
            "temporal_state_analog_support_scale": float(args.temporal_state_analog_support_scale),
            "temporal_state_analog_blend_weight": float(args.temporal_state_analog_blend_weight),
            "temporal_state_analog_feature_names": list(STATE_ANALOG_TEMPORAL_FEATURES),
            "temporal_action_gate_mode": str(args.temporal_action_gate_mode),
            "temporal_action_gate_min_history": int(args.temporal_action_gate_min_history),
            "temporal_action_gate_neighbor_k": int(args.temporal_action_gate_neighbor_k),
            "temporal_action_gate_tau": float(args.temporal_action_gate_tau),
            "temporal_action_gate_min_scale": float(args.temporal_action_gate_min_scale),
            "temporal_action_gate_abstain_effect_threshold": float(args.temporal_action_gate_abstain_effect_threshold),
            "temporal_action_gate_abstain_win_rate_threshold": float(args.temporal_action_gate_abstain_win_rate_threshold),
            "temporal_action_gate_feature_names": list(TEMPORAL_ACTION_GATE_FEATURES),
            "action_gate_history_scope": str(args.action_gate_history_scope),
            "action_gate_panel_history_csv": str(resolve_repo_path(args.action_gate_panel_history_csv, str(project_root))) if str(args.action_gate_panel_history_csv or "").strip() else "",
            "action_gate_panel_history_seed_rows": int(len(action_gate_panel_history_seed)),
            "guidance_quality_guardrail_mode": str(args.guidance_quality_guardrail_mode),
            "guidance_quality_guardrail_alpha": dict(GUIDANCE_QUALITY_GUARDRAIL_ALPHA),
            "guidance_expert_mode": str(args.guidance_expert_mode),
            "guidance_expert_min_history": int(args.guidance_expert_min_history),
            "guidance_expert_history_tau": float(args.guidance_expert_history_tau),
            "guidance_expert_support_scale": float(args.guidance_expert_support_scale),
            "regime_aware_base_anchor_mode": str(args.regime_aware_base_anchor_mode),
            "regime_aware_base_anchor_proposal_csv": str(resolve_repo_path(args.regime_aware_base_anchor_proposal_csv, str(project_root))) if str(args.regime_aware_base_anchor_proposal_csv or "").strip() else "",
            "regime_aware_base_anchor_proposal_pred_col": str(args.regime_aware_base_anchor_proposal_pred_col),
            "regime_aware_base_anchor_proposal_filter_col": str(args.regime_aware_base_anchor_proposal_filter_col),
            "regime_aware_base_anchor_proposal_filter_value": str(args.regime_aware_base_anchor_proposal_filter_value),
            "regime_aware_base_anchor_min_history": int(args.regime_aware_base_anchor_min_history),
            "regime_aware_base_anchor_signed_strength_threshold": float(args.regime_aware_base_anchor_signed_strength_threshold),
            "regime_aware_base_anchor_upward_ratio": float(args.regime_aware_base_anchor_upward_ratio),
            "regime_aware_base_anchor_validation": sanitize_for_json(regime_aware_base_anchor_validation),
            "shared_residual_backbone_validation": sanitize_for_json(shared_residual_backbone_validation),
            "anchor_selection_mode": str(args.anchor_selection_mode),
            "anchor_override_csv": str(resolve_repo_path(args.anchor_override_csv, str(project_root))) if str(args.anchor_override_csv or "").strip() else "",
            "anchor_online_min_history": int(args.anchor_online_min_history),
            "anchor_online_score_metric": str(args.anchor_online_score_metric),
            "anchor_online_window": int(args.anchor_online_window),
            "anchor_online_half_life": float(args.anchor_online_half_life),
            "anchor_online_same_quarter_weight": float(args.anchor_online_same_quarter_weight),
            "anchor_guidance_regime_mode": str(args.anchor_guidance_regime_mode),
            "anchor_guidance_same_regime_min_history": int(args.anchor_guidance_same_regime_min_history),
            "anchor_guidance_mismatch_penalty": float(args.anchor_guidance_mismatch_penalty),
            "anchor_explicit_guidance_proximity_mode": str(args.anchor_explicit_guidance_proximity_mode),
            "anchor_explicit_guidance_proximity_weight": float(args.anchor_explicit_guidance_proximity_weight),
            "anchor_explicit_guidance_kernel_min_history": int(args.anchor_explicit_guidance_kernel_min_history),
            "anchor_explicit_guidance_kernel_band": float(args.anchor_explicit_guidance_kernel_band),
            "anchor_explicit_guidance_kernel_shrink_k": float(args.anchor_explicit_guidance_kernel_shrink_k),
            "anchor_robust_momentum_mode": str(args.anchor_robust_momentum_mode),
            "coverage_drift_mode": str(args.coverage_drift_mode),
            "missing_anchor_policy": str(args.missing_anchor_policy),
            "arbitration_mode": str(args.arbitration_mode),
            "evidence_orthogonal_duplicate_threshold": float(args.evidence_orthogonal_duplicate_threshold),
            "evidence_orthogonal_base_min_scale": float(args.evidence_orthogonal_base_min_scale),
            "evidence_orthogonal_guidance_min_scale": float(args.evidence_orthogonal_guidance_min_scale),
            "evidence_orthogonal_base_strength": float(args.evidence_orthogonal_base_strength),
            "evidence_orthogonal_guidance_strength": float(args.evidence_orthogonal_guidance_strength),
            "integrated_arbitration_floor_derivation": sanitize_for_json(integrated_arbitration_floor_derivation),
            "integrated_arbitration_internal_policy": sanitize_for_json(INTEGRATED_ARBITRATION_INTERNAL_POLICY),
            "gate_feature_names": list(DIRECT_EXPERT_GATE_FEATURES),
            "gate_feature_count": int(len(DIRECT_EXPERT_GATE_FEATURES)),
        },
        "metrics": {
            "pooled": {
                "baseline": _metrics(quarterly["actual"], quarterly["baseline_pred"]),
                "csais_anchor": _metrics(quarterly["actual"], quarterly["pred_csais_anchor"]),
                "rawcard_direct_experts": _metrics(quarterly["actual"], quarterly["pred_csais_rawcard_direct_experts_v1"]),
            },
            "macro_mae": {
                "baseline": float(company_df["baseline_mae"].mean()),
                "csais_anchor": float(company_df["csais_anchor_mae"].mean()),
                "rawcard_direct_experts": float(company_df["rawcard_direct_experts_mae"].mean()),
            },
        },
        "wins": {
            "rawcard_direct_experts_beats_best_stat_companies": int(company_df["beats_best_stat"].sum()),
        },
        "coverage": {
            "coverage_drift_mode": str(args.coverage_drift_mode),
            "missing_anchor_policy": str(args.missing_anchor_policy),
            "company_count": int(len(coverage_records)),
            "issue_count": int(len(coverage_issue_records)),
            "issues": sanitize_for_json(coverage_issue_records),
        },
        "outputs": {
            "quarterly_csv": str(quarterly_csv),
            "company_summary_csv": str(company_csv),
            "coverage_diagnostics_csv": str(coverage_csv),
            "coverage_diagnostics_json": str(coverage_json),
            "contract_json": str(contract_json),
        },
    }
    write_json(out_dir / f"{method_family}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if str(args.coverage_drift_mode) != "off" and coverage_issue_records:
        issue_text = "; ".join(
            f"{rec.get('ticker')}: status={rec.get('status')}, missing_from_panel={rec.get('missing_from_panel_json')}, skipped_after_panel={rec.get('skipped_after_panel_json')}"
            for rec in coverage_issue_records
        )
        print(f"Coverage drift detected: {issue_text}", file=sys.stderr)
        if str(args.coverage_drift_mode) == "fail":
            return 2
    return 0


if __name__ == "__main__":
    if os.environ.get("CAME_REFERENCE_REPLAY") != "1":
        raise SystemExit("Implementation dependency only; run scripts/run_reference_replay.sh.")
    raise SystemExit(main())
