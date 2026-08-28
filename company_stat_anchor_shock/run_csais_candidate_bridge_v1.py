#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from evidence_memory_residual.common import EPS, softmax_weights
from company_stat_anchor_shock.run_csais_v1 import (
    DEFAULT_EXPERIMENT_CONFIG,
    FACTORIZED_SHOCK_FEATURES,
    RAW_SHOCK_FEATURES,
    STAT_EXCLUDE,
    _anchor_error_state,
    _build_shock_features,
    _clip,
    _factorized_internal_features,
    _fit_ridge,
    _guidance_features,
    _guidance_lock,
    _history_regime,
    _internal_features,
    _metrics,
    _prepare_company_panel,
    _predict_ridge,
    _quarter_key,
    _quarter_number,
    _safe_float,
    _safe_log,
    _top_contributions,
)
from temporal_kg_memory_attention.pair_features import _item_match, cross_card_attention
from native_evidence_forecaster.common import load_jsonl, resolve_repo_path, sanitize_for_json, write_json
from native_evidence_forecaster.run_native_cards_v1 import (
    CANONICAL_FACTORS,
    _aggregate_group,
    _apply_safety_guard,
    _card_mass,
    _fit_zero_intercept_ridge,
    _predict_zero_intercept_ridge,
    _top_feature_contribs,
)
from native_evidence_forecaster.run_native_csais_v1 import (
    DEFAULT_BACKBONE_CSV,
    DEFAULT_CARD_TABLE_JSONL,
    RAW_CARD_FEATURE_NAMES,
    _memory_diag,
    _predict_zero_intercept_history,
)


MINIMAL_CANDIDATE_GATE_FEATURES = [
    "gate_base_delta",
    "gate_native_delta",
    "gate_native_reliability",
    "gate_guidance_lock",
    "gate_anchor_uncertainty",
    "gate_internal_strength",
    "gate_native_fwd_conflict_ratio",
    "gate_memory_support",
    "gate_memory_consistency",
]

MINIMAL_CANDIDATE_GATE_FEATURES_V2_GAP_WEAKGUID = MINIMAL_CANDIDATE_GATE_FEATURES + [
    "gate_gap_base_native",
    "gate_base_x_weak_guidance",
]

MINIMAL_CANDIDATE_GATE_FEATURES_V3_TEMPORAL = MINIMAL_CANDIDATE_GATE_FEATURES + [
    "gate_temporal_delta",
    "gate_temporal_support",
    "gate_temporal_attention_focus",
    "gate_gap_base_temporal",
    "gate_gap_native_temporal",
]

MINIMAL_CANDIDATE_GATE_FEATURES_V4_DUALSTREAM = MINIMAL_CANDIDATE_GATE_FEATURES + [
    "gate_dualstream_delta",
    "gate_dualstream_support",
    "gate_gap_base_dualstream",
    "gate_gap_native_dualstream",
]

STANDARD_CANDIDATE_GATE_FEATURES = [
    "gate_base_delta",
    "gate_raw_delta",
    "gate_factor_delta",
    "gate_native_delta",
    "gate_native_available",
    "gate_native_reliability",
    "gate_gap_raw_factor",
    "gate_gap_base_native",
    "gate_sign_raw_factor",
    "gate_sign_base_native",
    "gate_guidance_lock",
    "gate_guidance_numeric_available",
    "gate_guidance_score_norm",
    "gate_guid_band_ratio",
    "gate_anchor_uncertainty",
    "gate_anchor_error_recent",
    "gate_anchor_error_same_quarter",
    "gate_anchor_error_same_guidance",
    "gate_internal_strength",
    "gate_internal_balance_abs",
    "gate_regime_vol_qoq4",
    "gate_regime_same_quarter_support",
    "gate_segment_share_top1",
    "gate_segment_share_count",
    "gate_native_fwd_abs_mass",
    "gate_native_fwd_conflict_ratio",
    "gate_native_fwd_count_log",
    "gate_native_metadata_proxy_flag",
    "gate_native_semantic_clarity",
    "gate_memory_pred_delta",
    "gate_memory_support",
    "gate_memory_consistency",
    "gate_base_x_weak_guidance",
    "gate_raw_x_uncertainty",
    "gate_native_x_support",
]

CANDIDATE_GATE_FEATURE_SETS = {
    "standard_candidate_gate_v1": STANDARD_CANDIDATE_GATE_FEATURES,
    "minimal_base_native_context_v1": MINIMAL_CANDIDATE_GATE_FEATURES,
    "minimal_base_native_context_v2_gap_weakguid": MINIMAL_CANDIDATE_GATE_FEATURES_V2_GAP_WEAKGUID,
    "minimal_base_native_temporal_context_v1": MINIMAL_CANDIDATE_GATE_FEATURES_V3_TEMPORAL,
    "minimal_base_native_dualstream_context_v1": MINIMAL_CANDIDATE_GATE_FEATURES_V4_DUALSTREAM,
}

NATIVE_CANDIDATE_FEATURE_MODE = "card_level_additive_shared_semantics"
TEMPORAL_CANDIDATE_FEATURE_MODE = "raw_card_graph_attention_v1"
DUALSTREAM_CANDIDATE_FEATURE_MODE = "dual_stream_raw_temporal_fusion_candidate_v1"
NATIVE_RELIABILITY_MODE = "memory_support_x_memory_consistency_x_one_minus_fwd_conflict_ratio"
CANDIDATE_GATE_TARGET_MODE = "residual_over_base_candidate"
NATIVE_TARGET_MODES = {"full_target_delta", "residual_over_base_candidate"}
NATIVE_CARD_ADMISSIBILITY_MODES = {"none", "conservative_admissibility_v1", "conservative_admissibility_v2"}

GENERIC_FACTOR_ADMISSIBILITY_WEIGHTS = {
    "demand": 1.0,
    "supply_capacity": 1.0,
    "inventory_channel": 0.9,
    "pricing_margin": 0.8,
    "product_transition": 0.8,
    "regulation_macro": 0.65,
    "revenue_boost_limit": 0.7,
    "other": 0.75,
}

GENERIC_FACTOR_ADMISSIBILITY_WEIGHTS_V2 = {
    "demand": 1.0,
    "supply_capacity": 1.0,
    "inventory_channel": 0.9,
    "pricing_margin": 0.8,
    "product_transition": 0.8,
    "regulation_macro": 0.5,
    "revenue_boost_limit": 0.55,
    "other": 0.65,
}

MAINLINE_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "raw_factor_control_v1": {
        "candidate_profile": "raw_factor",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "standard_candidate_gate_v1",
        "profile_role": "candidate-bridge non-native control",
        "profile_notes": "Raw/factor compressed candidate bridge with the broader standard gate. Useful as the E2 control row.",
    },
    "frontier_card_native_v1": {
        "candidate_profile": "raw_factor_native",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "standard_candidate_gate_v1",
        "profile_role": "performance frontier",
        "profile_notes": "Card-level native residual bridge with the broader standard gate. Useful as the E3 performance frontier.",
    },
    "conservative_card_native_v1": {
        "candidate_profile": "raw_factor_native",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "minimal_base_native_context_v1",
        "native_target_mode": "full_target_delta",
        "native_card_admissibility_mode": "none",
        "profile_role": "conservative retained target",
        "profile_notes": "Card-level native residual bridge with the minimal shared gate. Useful as the E4 conservative retained target.",
    },
    "conservative_card_native_v2_gap_weakguid": {
        "candidate_profile": "raw_factor_native",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "minimal_base_native_context_v2_gap_weakguid",
        "native_target_mode": "full_target_delta",
        "native_card_admissibility_mode": "none",
        "profile_role": "conservative retained target candidate",
        "profile_notes": "Conservative card-native bridge that extends the minimal gate with base-native gap and weak-guidance interaction cues for diagnostic recovery on weak-guidance watchlist slices.",
    },
    "conservative_card_native_residual_target_v1": {
        "candidate_profile": "raw_factor_native",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "minimal_base_native_context_v1",
        "native_target_mode": "residual_over_base_candidate",
        "native_card_admissibility_mode": "none",
        "profile_role": "conservative retained target candidate",
        "profile_notes": "Conservative card-native bridge that trains the native scorer on residual-over-base targets so native evidence stays a smaller supplement rather than a competing full-delta head.",
    },
    "conservative_card_native_residual_target_v2_admissibility": {
        "candidate_profile": "raw_factor_native",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "minimal_base_native_context_v1",
        "native_target_mode": "residual_over_base_candidate",
        "native_card_admissibility_mode": "conservative_admissibility_v1",
        "profile_role": "conservative retained target candidate",
        "profile_notes": "Conservative residual-target native scorer with explicit admissibility weighting for unknown-segment and generic-factor cards.",
    },
    "conservative_card_native_residual_target_v3_admissibility_cleanup": {
        "candidate_profile": "raw_factor_native",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "minimal_base_native_context_v1",
        "native_target_mode": "residual_over_base_candidate",
        "native_card_admissibility_mode": "conservative_admissibility_v2",
        "profile_role": "conservative retained target candidate",
        "profile_notes": "Conservative residual-target native scorer with a narrower second-round admissibility cleanup focused on unresolved macro and comparison-style cards.",
    },
    "conservative_card_native_temporal_residual_target_v1": {
        "candidate_profile": "raw_factor_native_temporal",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "minimal_base_native_temporal_context_v1",
        "native_target_mode": "residual_over_base_candidate",
        "native_card_admissibility_mode": "conservative_admissibility_v1",
        "profile_role": "exploratory temporal candidate bridge",
        "profile_notes": "Conservative residual-target bridge that keeps the native additive candidate and adds a non-flattened raw-card temporal graph-attention candidate as an explicit bridge input.",
    },
    "conservative_card_native_dualstream_residual_target_v1": {
        "candidate_profile": "raw_factor_native_dualstream",
        "base_candidate_mode": "compressed_only",
        "candidate_gate_feature_mode": "minimal_base_native_dualstream_context_v1",
        "native_target_mode": "residual_over_base_candidate",
        "native_card_admissibility_mode": "conservative_admissibility_v2",
        "profile_role": "exploratory dual-stream candidate bridge",
        "profile_notes": "Conservative residual-target bridge that keeps the compressed base and native additive candidates, then adds a dual-stream fused raw-card candidate built from intrinsic card scoring plus historical raw-card temporal aggregation.",
    },
    "baseblend_diagnostic_v1": {
        "candidate_profile": "raw_factor_native",
        "base_candidate_mode": "native_guided_blend",
        "candidate_gate_feature_mode": "minimal_base_native_context_v1",
        "native_target_mode": "full_target_delta",
        "native_card_admissibility_mode": "none",
        "profile_role": "negative-transfer diagnostic",
        "profile_notes": "Diagnostic-only base candidate weakening path. Useful as the E5 negative-transfer row, not as a retained default.",
    },
}

NATIVE_CARD_ADDITIVE_FEATURE_NAMES = [
    "card_signed_mass",
    "card_abs_mass",
    "card_top1_shareweighted_signed",
    "card_other_shareweighted_signed",
    "card_persistent_signed",
    "card_event_signed",
    "card_claim_signed",
] + [f"factor_signed__{factor}" for factor in CANONICAL_FACTORS] + [f"factor_abs__{factor}" for factor in CANONICAL_FACTORS]


def _sign_num(value: float, tol: float = 1e-6) -> float:
    if not np.isfinite(value) or abs(float(value)) <= tol:
        return 0.0
    return 1.0 if float(value) > 0.0 else -1.0


def _normalize_key_piece(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _card_segment_name(card: Mapping[str, Any]) -> str:
    return str(card.get("segment_normalized") or card.get("segment") or "UNKNOWN").strip() or "UNKNOWN"


def _card_strict_fusion_key(card: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        _normalize_key_piece(card.get("segment_normalized") or card.get("segment")),
        _normalize_key_piece(card.get("canonical_factor") or "other"),
        _normalize_key_piece(card.get("polarity")),
        _normalize_key_piece(card.get("driver_source")),
        _normalize_key_piece(card.get("attribution_anchor")),
    )


def _card_fallback_fusion_key(card: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        _normalize_key_piece(card.get("segment_normalized") or card.get("segment")),
        _normalize_key_piece(card.get("canonical_factor") or "other"),
        _normalize_key_piece(card.get("polarity")),
        _normalize_key_piece(card.get("driver_source")),
        _normalize_key_piece(card.get("relation_family") or card.get("category") or card.get("attribution_anchor")),
    )


def _card_fusion_key(card: Mapping[str, Any]) -> Tuple[str, str, str, str, str]:
    strict_key = _card_strict_fusion_key(card)
    if strict_key[-1] != "unknown":
        return strict_key
    return _card_fallback_fusion_key(card)


def _weighted_native_card_mass(card: Mapping[str, Any], admissibility_mode: str) -> float:
    return float(_card_mass(card) * _native_card_admissibility_weight(card, admissibility_mode))


def _dualstream_card_stub(card: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "instance_id": str(card.get("instance_id") or ""),
        "segment": _card_segment_name(card),
        "canonical_factor": str(card.get("canonical_factor") or "other"),
        "polarity": str(card.get("polarity") or "unknown"),
        "driver_source": str(card.get("driver_source") or "unknown"),
        "attribution_anchor": str(card.get("attribution_anchor") or card.get("relation_family") or "UNKNOWN"),
        "relation_family": str(card.get("relation_family") or card.get("category") or "unknown"),
        "segment_share_at_observed": float(_safe_float(card.get("segment_share_at_observed"), 0.0)),
        "segment_rank_at_observed": int(_safe_float(card.get("segment_rank_at_observed"), 0.0)),
        "novelty_state": str(card.get("novelty_state") or ""),
        "source_text_sha256": str(card.get("source_text_sha256") or card.get("evidence_sha256") or ""),
        "release_status": str(card.get("release_status") or "quote_withheld_third_party"),
    }


def _weighted_average(pairs: Sequence[Tuple[float, float]]) -> float:
    usable = [(float(value), float(weight)) for value, weight in pairs if np.isfinite(value) and np.isfinite(weight) and weight > 0.0]
    if not usable:
        return 0.0
    denom = float(sum(weight for _, weight in usable))
    if denom <= EPS:
        return 0.0
    return float(sum(value * weight for value, weight in usable) / denom)


def _split_scored_rows(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    target_delta: float,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    positive = [dict(row) for row in scored_rows if float(_safe_float(row.get(value_field), 0.0)) > 0.0]
    negative = [dict(row) for row in scored_rows if float(_safe_float(row.get(value_field), 0.0)) < 0.0]
    positive.sort(key=lambda item: float(_safe_float(item.get(value_field), 0.0)), reverse=True)
    negative.sort(key=lambda item: float(_safe_float(item.get(value_field), 0.0)))
    if float(target_delta) > 1e-9:
        return positive[:top_k], negative[:top_k]
    if float(target_delta) < -1e-9:
        return negative[:top_k], positive[:top_k]
    strongest = [dict(row) for row in scored_rows]
    strongest.sort(key=lambda item: abs(float(_safe_float(item.get(value_field), 0.0))), reverse=True)
    return strongest[:top_k], []


def _load_backbone_lookup(path: Path, requested: set[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    df = pd.read_csv(path).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if requested:
        df = df[df["ticker"].isin(requested)].copy()
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[(str(row.get("ticker") or "").upper(), str(row.get("target_fiscal_quarter") or ""))] = row.to_dict()
    return out


def _load_card_groups(path: Path, requested: set[str]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    group_map: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in load_jsonl(path):
        ticker = str(row.get("ticker") or "").upper()
        if requested and ticker not in requested:
            continue
        key = (ticker, str(row.get("observed_quarter") or ""), str(row.get("target_quarter") or ""))
        group_map.setdefault(key, []).append(row)
    return group_map


def _native_card_admissibility_weight(card: Mapping[str, Any], mode: str) -> float:
    mode = str(mode or "none")
    if mode == "none":
        return 1.0
    if mode not in {"conservative_admissibility_v1", "conservative_admissibility_v2"}:
        raise ValueError(f"Unsupported native_card_admissibility_mode: {mode}")
    factor = str(card.get("canonical_factor") or "other")
    segment = str(card.get("segment_normalized") or card.get("segment") or "").strip()
    source = str(card.get("driver_source") or "")
    share = float(_clip(_safe_float(card.get("segment_share_at_observed"), 0.0), 0.0, 1.0))
    rank = int(_safe_float(card.get("segment_rank_at_observed"), 0.0))
    factor_weights = GENERIC_FACTOR_ADMISSIBILITY_WEIGHTS if mode == "conservative_admissibility_v1" else GENERIC_FACTOR_ADMISSIBILITY_WEIGHTS_V2
    weight = float(factor_weights.get(factor, factor_weights["other"]))
    if not segment or segment.upper() == "UNKNOWN":
        weight *= 0.6 if mode == "conservative_admissibility_v1" else 0.5
    if rank <= 0 or share <= 0.0:
        weight *= 0.85 if mode == "conservative_admissibility_v1" else 0.8
    if source not in {"event", "claim"}:
        weight *= 0.9
    return float(_clip(weight, 0.1, 1.0))


def _native_card_additive_features(card: Mapping[str, Any], admissibility_mode: str = "none") -> Dict[str, float]:
    admissibility_weight = _native_card_admissibility_weight(card, admissibility_mode)
    mass = float(_card_mass(card) * admissibility_weight)
    share = float(_clip(_safe_float(card.get("segment_share_at_observed"), 0.0), 0.0, 1.0))
    rank = int(_safe_float(card.get("segment_rank_at_observed"), 0.0))
    driver_source = str(card.get("driver_source") or "")
    factor = str(card.get("canonical_factor") or "other")
    out = {
        "card_signed_mass": mass,
        "card_abs_mass": abs(mass),
        "card_top1_shareweighted_signed": float(mass * share) if rank == 1 else 0.0,
        "card_other_shareweighted_signed": float(mass * share) if rank != 1 else 0.0,
        "card_persistent_signed": float(mass) if bool(card.get("persistence_hint")) else 0.0,
        "card_event_signed": float(mass) if driver_source == "event" else 0.0,
        "card_claim_signed": float(mass) if driver_source == "claim" else 0.0,
    }
    for name in CANONICAL_FACTORS:
        out[f"factor_signed__{name}"] = float(mass) if factor == name else 0.0
        out[f"factor_abs__{name}"] = float(abs(mass)) if factor == name else 0.0
    return out


def _sum_native_card_additive_features(cards: Sequence[Mapping[str, Any]], admissibility_mode: str = "none") -> Dict[str, float]:
    totals = {name: 0.0 for name in NATIVE_CARD_ADDITIVE_FEATURE_NAMES}
    for card in cards:
        feature_map = _native_card_additive_features(card, admissibility_mode)
        for name in NATIVE_CARD_ADDITIVE_FEATURE_NAMES:
            totals[name] += float(feature_map.get(name, 0.0))
    return totals


def _card_additive_influence(card: Mapping[str, Any], coefs_by_name: Mapping[str, float], admissibility_mode: str = "none") -> float:
    feature_map = _native_card_additive_features(card, admissibility_mode)
    return float(sum(float(feature_map.get(name, 0.0)) * float(coefs_by_name.get(name, 0.0)) for name in NATIVE_CARD_ADDITIVE_FEATURE_NAMES))


def _split_native_support_conflict_cards(
    scored_cards: Sequence[Mapping[str, Any]],
    *,
    target_delta: float,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    positive = [dict(card) for card in scored_cards if float(_safe_float(card.get("approx_influence"), 0.0)) > 0.0]
    negative = [dict(card) for card in scored_cards if float(_safe_float(card.get("approx_influence"), 0.0)) < 0.0]
    positive.sort(key=lambda item: float(_safe_float(item.get("approx_influence"), 0.0)), reverse=True)
    negative.sort(key=lambda item: float(_safe_float(item.get("approx_influence"), 0.0)))
    if float(target_delta) > 1e-9:
        return positive[:top_k], negative[:top_k]
    if float(target_delta) < -1e-9:
        return negative[:top_k], positive[:top_k]
    strongest = [dict(card) for card in scored_cards]
    strongest.sort(key=lambda item: abs(float(_safe_float(item.get("approx_influence"), 0.0))), reverse=True)
    return strongest[:top_k], []


def _predict_native_card_additive_candidate(
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
    x_rows: List[List[float]] = []
    y_rows: List[float] = []
    for row in train_rows:
        cards = list(row.get("forward_rows") or [])
        if not cards:
            continue
        feature_map = _sum_native_card_additive_features(cards, admissibility_mode)
        x_rows.append([float(feature_map.get(name, 0.0)) for name in NATIVE_CARD_ADDITIVE_FEATURE_NAMES])
        y_rows.append(float(_safe_float(row.get(target_col), float("nan"))))
    if len(y_rows) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(y_rows)),
            "support": 0.0,
            "top_contribs": [],
            "coefs_by_name": {},
        }
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
            "top_contribs": [],
            "coefs_by_name": {},
        }
    model = _fit_zero_intercept_ridge(x_train, y_train, float(alpha))
    current_feature_map = _sum_native_card_additive_features(current_cards, admissibility_mode)
    x_cur = np.asarray([float(current_feature_map.get(name, 0.0)) for name in NATIVE_CARD_ADDITIVE_FEATURE_NAMES], dtype=float)
    pred_raw = float(_predict_zero_intercept_ridge(model, x_cur))
    finite_abs = np.abs(y_train[np.isfinite(y_train)])
    if finite_abs.size:
        delta_cap = float(np.quantile(finite_abs, float(delta_cap_quantile)))
        if np.isfinite(delta_cap) and delta_cap > 0.0:
            pred_raw = float(_clip(pred_raw, -delta_cap, delta_cap))
    support = float(len(y_train) / max(len(y_train) + float(shrink_k), 1e-9))
    coefs_by_name = {
        name: float(val)
        for name, val in zip(
            NATIVE_CARD_ADDITIVE_FEATURE_NAMES,
            np.asarray(model["coefs"], dtype=float) / np.asarray(model["stds"], dtype=float),
        )
    }
    return {
        "pred": float(pred_raw * support),
        "pred_raw": float(pred_raw),
        "train_count": int(len(y_train)),
        "support": support,
        "top_contribs": _top_feature_contribs(model, x_cur, NATIVE_CARD_ADDITIVE_FEATURE_NAMES, top_k=6) if current_cards else [],
        "coefs_by_name": coefs_by_name,
    }


def _native_card_to_attention_item(card: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "segment": str(card.get("segment") or "unknown"),
        "segment_normalized": _card_segment_name(card),
        "relation_family": str(card.get("relation_family") or card.get("category") or "unknown"),
        "category": str(card.get("category") or card.get("relation_family") or "unknown"),
        "canonical_factor": str(card.get("canonical_factor") or "other"),
        "polarity": str(card.get("polarity") or "unknown"),
        "strength": str(card.get("strength") or "unknown"),
        "confidence": float(_safe_float(card.get("confidence"), 0.0)),
        "weight": float(_card_mass(card)),
        "persistence_hint": bool(card.get("persistence_hint")),
        "driver_source": str(card.get("driver_source") or "unknown"),
        "attribution_anchor": str(card.get("attribution_anchor") or card.get("relation_family") or "unknown"),
        "instance_id": str(card.get("instance_id") or ""),
        "release_token_ids": list(card.get("release_token_ids") or []),
        "verbatim": str(card.get("evidence") or ""),
    }


def _native_rows_to_attention_record(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "internal": {
            "items": [_native_card_to_attention_item(card) for card in rows if isinstance(card, Mapping)]
        }
    }


def _temporal_match_structure_bonus(current_card: Mapping[str, Any], past_card: Mapping[str, Any]) -> Tuple[float, float, float]:
    strict_match = 1.0 if _card_strict_fusion_key(current_card) == _card_strict_fusion_key(past_card) else 0.0
    fallback_match = 1.0 if strict_match <= 0.0 and _card_fallback_fusion_key(current_card) == _card_fallback_fusion_key(past_card) else 0.0
    bonus = 0.15 * strict_match + 0.07 * fallback_match
    return strict_match, fallback_match, bonus


def _dualstream_card_summary(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "segment": str(entry.get("segment") or "UNKNOWN"),
        "canonical_factor": str(entry.get("canonical_factor") or entry.get("factor") or "other"),
        "polarity": str(entry.get("polarity") or "unknown"),
        "driver_source": str(entry.get("driver_source") or "unknown"),
        "attribution_anchor": str(entry.get("attribution_anchor") or entry.get("anchor") or "UNKNOWN"),
        "relation_family": str(entry.get("relation_family") or "unknown"),
        "card_count": int(_safe_float(entry.get("card_count"), 0.0)),
        "intrinsic_card_score": float(_safe_float(entry.get("intrinsic_card_score"), 0.0)),
        "temporal_card_score": float(_safe_float(entry.get("temporal_card_score"), 0.0)),
        "fused_card_score": float(_safe_float(entry.get("fused_card_score"), 0.0)),
        "temporal_card_support": float(_safe_float(entry.get("temporal_card_support"), 0.0)),
        "temporal_card_attention_focus": float(_safe_float(entry.get("temporal_card_attention_focus"), 0.0)),
        "temporal_directional_consistency": float(_safe_float(entry.get("temporal_directional_consistency"), 0.0)),
        "analog_disagreement_penalty": float(_safe_float(entry.get("analog_disagreement_penalty"), 0.0)),
        "intrinsic_temporal_agreement": str(entry.get("intrinsic_temporal_agreement") or "neutral"),
        "fused_alpha_t": float(_safe_float(entry.get("fused_alpha_t"), 0.0)),
        "top_evidence": str(entry.get("top_evidence") or ""),
        "temporal_matched_quarters": list(entry.get("temporal_matched_quarters") or [])[:5],
        "temporal_top_matches": list(entry.get("temporal_top_matches") or [])[:3],
    }


def _predict_dualstream_raw_temporal_fusion_candidate(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    current_cards: Sequence[Mapping[str, Any]],
    current_quarter: str,
    target_col: str,
    native_train_support: float,
    native_reliability: float,
    native_coefs_by_name: Mapping[str, float],
    admissibility_mode: str,
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
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pred": 0.0,
        "pred_raw": 0.0,
        "support": float(_clip(native_reliability, 0.0, 1.0)),
        "train_count": 0,
        "attention_focus": 0.0,
        "directional_consistency": 0.0,
        "sign_agreement": 0.0,
        "intrinsic_component_delta_log": 0.0,
        "temporal_component_delta_raw_log": 0.0,
        "temporal_component_delta_log": 0.0,
        "fused_cards": [],
        "top_support_cards": [],
        "top_conflict_cards": [],
        "top_temporal_matches": [],
    }
    if not current_cards:
        out["support"] = 0.0
        return out

    valid_history: List[Dict[str, Any]] = []
    for row in train_rows:
        target_value = float(_safe_float(row.get(target_col), float("nan")))
        if not np.isfinite(target_value):
            continue
        forward_rows = list(row.get("forward_rows") or [])
        if not forward_rows:
            continue
        valid_history.append(dict(row))
    out["train_count"] = int(len(valid_history))

    weighted_current_rows: List[Dict[str, Any]] = []
    total_abs_current_mass = 0.0
    for raw_card in current_cards:
        card = dict(raw_card)
        weighted_mass = _weighted_native_card_mass(card, admissibility_mode)
        total_abs_current_mass += abs(weighted_mass)
        intrinsic_raw = 0.0
        intrinsic_final = 0.0
        if native_coefs_by_name:
            intrinsic_base = _card_additive_influence(card, native_coefs_by_name, admissibility_mode)
            intrinsic_raw = float(intrinsic_base * native_train_support)
            intrinsic_final = float(intrinsic_raw * native_reliability)
        weighted_current_rows.append(
            {
                "card": card,
                "weighted_mass": float(weighted_mass),
                "intrinsic_card_score_raw": float(intrinsic_raw),
                "intrinsic_card_score": float(intrinsic_final),
            }
        )

    query_record = _native_rows_to_attention_record(current_cards)
    history_slice = valid_history[-int(history_cap) :] if int(history_cap) > 0 else list(valid_history)
    raw_pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any], float, float]] = []
    current_q_num = _quarter_number(str(current_quarter or ""))
    for past_row in history_slice:
        past_quarter = str(past_row.get("quarter") or "")
        past_rows = list(past_row.get("forward_rows") or [])
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
        score = float(attention.get("attention_score", 0.0)) * recency + same_q
        raw_pairs.append((score, dict(past_row), attention, recency, same_q))
    raw_pairs.sort(key=lambda item: item[0], reverse=True)
    top_pairs = raw_pairs[: int(quarter_top_k)] if len(valid_history) >= int(min_train) else []
    quarter_weights = softmax_weights([item[0] for item in top_pairs], float(quarter_temperature)) if top_pairs else []
    history_support = float(len(valid_history) / max(len(valid_history) + max(float(min_train), 1.0), EPS))

    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for weighted_row in weighted_current_rows:
        card = dict(weighted_row["card"])
        current_share = float(abs(weighted_row["weighted_mass"]) / max(total_abs_current_mass, EPS)) if total_abs_current_mass > EPS else 0.0
        query_item = _native_card_to_attention_item(card)
        card_pair_rows: List[Dict[str, Any]] = []
        for quarter_weight, pair in zip(quarter_weights, top_pairs):
            _, past_row, _, recency, same_q = pair
            target_value = float(_safe_float(past_row.get(target_col), 0.0))
            past_quarter = str(past_row.get("quarter") or "")
            scored_matches: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, float], float, float]] = []
            for raw_past_card in list(past_row.get("forward_rows") or []):
                past_card = dict(raw_past_card)
                match = _item_match(query_item, _native_card_to_attention_item(past_card))
                strict_match, fallback_match, structure_bonus = _temporal_match_structure_bonus(card, past_card)
                adjusted_score = float(match["score"] + structure_bonus)
                scored_matches.append((adjusted_score, past_card, match, match, strict_match, fallback_match))
            if not scored_matches:
                continue
            scored_matches.sort(key=lambda item: item[0], reverse=True)
            top_scored = scored_matches[: max(int(item_top_k), 1)]
            item_weights = softmax_weights([item[0] for item in top_scored], float(item_temperature)) if len(top_scored) > 1 else [1.0]
            pooled_match_score = float(sum(weight * item[0] for weight, item in zip(item_weights, top_scored)))
            attention_focus = float(item_weights[0]) if item_weights else 0.0
            structure_match_score = float(sum(weight * (item[4] + 0.5 * item[5]) for weight, item in zip(item_weights, top_scored)))
            pair_weight_raw = float(max(quarter_weight, 0.0) * max(min(pooled_match_score, 1.0), 0.0))
            match_preview: List[Dict[str, Any]] = []
            for match_weight, item in zip(item_weights, top_scored):
                adjusted_score, past_card, match_diag, _, strict_match, fallback_match = item
                match_preview.append(
                    {
                        "matched_quarter": past_quarter,
                        "quarter_weight": round(float(quarter_weight), 6),
                        "match_weight": round(float(match_weight), 6),
                        "adjusted_match_score": round(float(adjusted_score), 6),
                        "match_score": round(float(match_diag.get("score", 0.0)), 6),
                        "token_match": round(float(match_diag.get("token_match", 0.0)), 6),
                        "strict_key_match": int(strict_match > 0.0),
                        "fallback_key_match": int(fallback_match > 0.0),
                        "matched_segment": _card_segment_name(past_card),
                        "matched_canonical_factor": str(past_card.get("canonical_factor") or "other"),
                        "matched_polarity": str(past_card.get("polarity") or "unknown"),
                        "matched_driver_source": str(past_card.get("driver_source") or "unknown"),
                        "matched_attribution_anchor": str(past_card.get("attribution_anchor") or past_card.get("relation_family") or "UNKNOWN"),
                        "matched_evidence": str(past_card.get("evidence") or "")[:180],
                    }
                )
            card_pair_rows.append(
                {
                    "quarter": past_quarter,
                    "target_value": float(target_value),
                    "pair_weight_raw": float(pair_weight_raw),
                    "pooled_match_score": float(min(max(pooled_match_score, 0.0), 1.0)),
                    "attention_focus": float(attention_focus),
                    "structure_match_score": float(structure_match_score),
                    "recency": float(recency),
                    "same_quarter_bonus": float(same_q),
                    "top_matches": match_preview,
                }
            )

        card_pair_rows = [entry for entry in card_pair_rows if float(entry["pair_weight_raw"]) > 0.0]
        pair_weights = [float(entry["pair_weight_raw"]) for entry in card_pair_rows]
        card_weights = [float(weight / max(sum(pair_weights), EPS)) for weight in pair_weights] if pair_weights else []
        target_values = [float(entry["target_value"]) for entry in card_pair_rows]
        pred_mean = float(sum(weight * value for weight, value in zip(card_weights, target_values))) if card_weights else 0.0
        pred_var = float(sum(weight * (value - pred_mean) ** 2 for weight, value in zip(card_weights, target_values))) if card_weights else 0.0
        abs_mass = float(sum(weight * abs(value) for weight, value in zip(card_weights, target_values))) if card_weights else 0.0
        directional_consistency = 0.0 if abs_mass <= EPS else float(abs(pred_mean) / abs_mass)
        attention_focus = float(sum(weight * float(entry["attention_focus"]) for weight, entry in zip(card_weights, card_pair_rows))) if card_weights else 0.0
        mean_match_score = float(sum(weight * float(entry["pooled_match_score"]) for weight, entry in zip(card_weights, card_pair_rows))) if card_weights else 0.0
        structure_match_score = float(sum(weight * float(entry["structure_match_score"]) for weight, entry in zip(card_weights, card_pair_rows))) if card_weights else 0.0
        weight_sq_sum = float(sum(weight ** 2 for weight in card_weights)) if card_weights else 0.0
        effective_memory_count = 0.0 if weight_sq_sum <= EPS else 1.0 / weight_sq_sum
        card_support = float(np.exp(-pred_var / max(float(var_tau), EPS))) if card_weights else 0.0
        card_support *= min(1.0, effective_memory_count / max(float(neff_scale), 1.0)) if card_weights else 0.0
        card_support *= float(_clip(mean_match_score, 0.0, 1.0))
        card_support *= float(_clip(0.5 + 0.5 * structure_match_score, 0.0, 1.0))
        if float(directional_consistency_power) > 0.0:
            card_support *= max(directional_consistency, 0.0) ** float(directional_consistency_power)
        if float(attention_focus_power) > 0.0:
            card_support *= max(attention_focus, 0.0) ** float(attention_focus_power)
        card_support = float(_clip(history_support * card_support, 0.0, 1.0))
        temporal_raw = float(current_share * pred_mean)
        temporal_final = float(temporal_raw * card_support)
        strict_key = _card_fusion_key(card)
        entry = grouped.setdefault(
            strict_key,
            {
                **_dualstream_card_stub(card),
                "card_count": 0,
                "intrinsic_card_score_raw": 0.0,
                "intrinsic_card_score": 0.0,
                "intrinsic_card_support": float(_clip(native_reliability, 0.0, 1.0)),
                "temporal_card_score_raw": 0.0,
                "temporal_card_score": 0.0,
                "temporal_support_weight": 0.0,
                "temporal_card_support": 0.0,
                "temporal_card_attention_focus": 0.0,
                "temporal_directional_consistency": 0.0,
                "temporal_structure_match": 0.0,
                "temporal_matched_quarters": [],
                "temporal_top_matches": [],
                "top_evidence": str(card.get("evidence") or "")[:240],
            },
        )
        temporal_weight = max(abs(temporal_raw), current_share)
        entry["card_count"] += 1
        entry["intrinsic_card_score_raw"] += float(weighted_row["intrinsic_card_score_raw"])
        entry["intrinsic_card_score"] += float(weighted_row["intrinsic_card_score"])
        entry["temporal_card_score_raw"] += float(temporal_raw)
        entry["temporal_card_score"] += float(temporal_final)
        entry["temporal_support_weight"] += float(temporal_weight)
        entry["temporal_card_support"] += float(card_support * temporal_weight)
        entry["temporal_card_attention_focus"] += float(attention_focus * temporal_weight)
        entry["temporal_directional_consistency"] += float(directional_consistency * temporal_weight)
        entry["temporal_structure_match"] += float(structure_match_score * temporal_weight)
        seen_quarters = set(entry["temporal_matched_quarters"])
        for item in sorted(card_pair_rows, key=lambda row_item: float(row_item["pair_weight_raw"]), reverse=True):
            quarter = str(item["quarter"] or "")
            if quarter and quarter not in seen_quarters:
                entry["temporal_matched_quarters"].append(quarter)
                seen_quarters.add(quarter)
        entry["temporal_top_matches"].extend(
            [
                {
                    **match,
                    "query_segment": entry["segment"],
                    "query_canonical_factor": entry["canonical_factor"],
                    "query_polarity": entry["polarity"],
                    "query_driver_source": entry["driver_source"],
                    "query_attribution_anchor": entry["attribution_anchor"],
                    "temporal_card_support": round(float(card_support), 6),
                }
                for item in card_pair_rows[:3]
                for match in list(item.get("top_matches") or [])[:2]
            ]
        )

    fused_rows: List[Dict[str, Any]] = []
    for entry in grouped.values():
        temporal_weight = float(entry.pop("temporal_support_weight", 0.0))
        if temporal_weight > EPS:
            entry["temporal_card_support"] = float(entry["temporal_card_support"] / temporal_weight)
            entry["temporal_card_attention_focus"] = float(entry["temporal_card_attention_focus"] / temporal_weight)
            entry["temporal_directional_consistency"] = float(entry["temporal_directional_consistency"] / temporal_weight)
            entry["temporal_structure_match"] = float(entry["temporal_structure_match"] / temporal_weight)
        else:
            entry["temporal_card_support"] = 0.0
            entry["temporal_card_attention_focus"] = 0.0
            entry["temporal_directional_consistency"] = 0.0
            entry["temporal_structure_match"] = 0.0
        entry["temporal_matched_quarters"] = entry["temporal_matched_quarters"][:5]
        entry["temporal_top_matches"] = sorted(
            list(entry["temporal_top_matches"]),
            key=lambda item: (
                float(_safe_float(item.get("quarter_weight"), 0.0))
                * float(_safe_float(item.get("adjusted_match_score"), 0.0))
                * (0.5 + 0.5 * float(_safe_float(item.get("temporal_card_support"), 0.0)))
            ),
            reverse=True,
        )[:6]
        agreement = "neutral"
        intrinsic_sign = _sign_num(float(entry["intrinsic_card_score"]))
        temporal_sign = _sign_num(float(entry["temporal_card_score"]))
        sign_agreement = 0.5
        if intrinsic_sign != 0.0 and temporal_sign != 0.0:
            if intrinsic_sign == temporal_sign:
                agreement = "same_sign"
                sign_agreement = 1.0
            else:
                agreement = "opposing_sign"
                sign_agreement = 0.0
        elif intrinsic_sign != 0.0 or temporal_sign != 0.0:
            agreement = "one_sided"
            sign_agreement = 0.5
        analog_disagreement_penalty = float(1.0 - _clip(entry["temporal_directional_consistency"], 0.0, 1.0))
        alpha_t = float(
            _clip(
                float(entry["temporal_card_support"])
                * (0.5 + 0.5 * float(entry["temporal_card_attention_focus"]))
                * (0.5 + 0.5 * float(entry["temporal_directional_consistency"]))
                * (0.35 + 0.65 * sign_agreement)
                * (1.0 - 0.5 * analog_disagreement_penalty),
                0.0,
                1.0,
            )
        )
        entry["intrinsic_temporal_agreement"] = agreement
        entry["sign_agreement_score"] = float(sign_agreement)
        entry["analog_disagreement_penalty"] = float(analog_disagreement_penalty)
        entry["fused_alpha_t"] = float(alpha_t)
        entry["fused_card_score_raw"] = float(entry["intrinsic_card_score_raw"] + alpha_t * entry["temporal_card_score_raw"])
        entry["fused_card_score"] = float(entry["intrinsic_card_score"] + alpha_t * entry["temporal_card_score"])
        fused_rows.append(dict(entry))

    if not fused_rows:
        return out

    fused_rows.sort(key=lambda item: abs(float(_safe_float(item.get("fused_card_score"), 0.0))), reverse=True)
    out["pred_raw"] = float(sum(float(_safe_float(item.get("fused_card_score_raw"), 0.0)) for item in fused_rows))
    out["pred"] = float(sum(float(_safe_float(item.get("fused_card_score"), 0.0)) for item in fused_rows))
    out["intrinsic_component_delta_log"] = float(sum(float(_safe_float(item.get("intrinsic_card_score"), 0.0)) for item in fused_rows))
    out["temporal_component_delta_raw_log"] = float(sum(float(_safe_float(item.get("fused_alpha_t"), 0.0)) * float(_safe_float(item.get("temporal_card_score_raw"), 0.0)) for item in fused_rows))
    out["temporal_component_delta_log"] = float(sum(float(_safe_float(item.get("fused_alpha_t"), 0.0)) * float(_safe_float(item.get("temporal_card_score"), 0.0)) for item in fused_rows))
    temporal_mass = [abs(float(_safe_float(item.get("temporal_card_score"), 0.0))) for item in fused_rows]
    out["attention_focus"] = _weighted_average(
        [(float(_safe_float(item.get("temporal_card_attention_focus"), 0.0)), weight) for item, weight in zip(fused_rows, temporal_mass)]
    )
    out["directional_consistency"] = _weighted_average(
        [(float(_safe_float(item.get("temporal_directional_consistency"), 0.0)), weight) for item, weight in zip(fused_rows, temporal_mass)]
    )
    out["sign_agreement"] = _weighted_average(
        [(float(_safe_float(item.get("sign_agreement_score"), 0.5)), weight) for item, weight in zip(fused_rows, temporal_mass)]
    )
    temporal_support = _weighted_average(
        [(float(_safe_float(item.get("temporal_card_support"), 0.0)), weight) for item, weight in zip(fused_rows, temporal_mass)]
    )
    out["support"] = float(
        _clip(
            float(native_reliability)
            + (1.0 - float(native_reliability)) * temporal_support * (0.5 + 0.5 * float(out["sign_agreement"])),
            0.0,
            1.0,
        )
    )
    support_rows, conflict_rows = _split_scored_rows(
        fused_rows,
        value_field="fused_card_score",
        target_delta=float(out["pred"]),
        top_k=5,
    )
    out["fused_cards"] = [_dualstream_card_summary(item) for item in fused_rows]
    out["top_support_cards"] = [_dualstream_card_summary(item) for item in support_rows]
    out["top_conflict_cards"] = [_dualstream_card_summary(item) for item in conflict_rows]
    flat_matches = []
    for item in fused_rows[:5]:
        for match in list(item.get("temporal_top_matches") or [])[:3]:
            match_copy = dict(match)
            match_copy["fused_card_score"] = round(float(_safe_float(item.get("fused_card_score"), 0.0)), 6)
            flat_matches.append(match_copy)
    flat_matches.sort(
        key=lambda item: (
            abs(float(_safe_float(item.get("fused_card_score"), 0.0)))
            * float(_safe_float(item.get("quarter_weight"), 0.0))
            * float(_safe_float(item.get("adjusted_match_score"), 0.0))
        ),
        reverse=True,
    )
    out["top_temporal_matches"] = flat_matches[:10]
    return out


def _arbitrate_candidate_bridge(
    *,
    compressed_base_delta: float,
    compressed_base_support: float,
    native_delta: float,
    native_support: float,
    dualstream_delta: float,
    dualstream_support: float,
) -> Dict[str, Any]:
    support_map = {
        "compressed_base": float(_clip(compressed_base_support, 0.0, 1.0)),
        "native_additive": float(_clip(native_support, 0.0, 1.0)),
        "dualstream_fused": float(_clip(dualstream_support, 0.0, 1.0)),
    }
    delta_map = {
        "compressed_base": float(compressed_base_delta),
        "native_additive": float(native_delta),
        "dualstream_fused": float(dualstream_delta),
    }
    pred = _blend_candidates([(delta_map[name], support_map[name]) for name in support_map])
    support_sum = float(sum(support_map.values()))
    normalized_weights = {
        name: (float(weight / support_sum) if support_sum > EPS else 0.0)
        for name, weight in support_map.items()
    }
    return {
        "pred": float(pred),
        "support_sum": float(support_sum),
        "candidate_supports": support_map,
        "candidate_weights": normalized_weights,
    }


def _same_fiscal_quarter_bonus(current_quarter: str, past_quarter: str) -> float:
    try:
        return 1.0 if int(str(current_quarter).split("_Q", 1)[1]) == int(str(past_quarter).split("_Q", 1)[1]) else 0.0
    except Exception:
        return 0.0


def _predict_temporal_graph_attention_candidate(
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
    shrink_k: float,
) -> Dict[str, Any]:
    if not current_rows:
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": 0,
            "support": 0.0,
            "effective_memory_count": 0.0,
            "directional_consistency": 0.0,
            "attention_focus": 0.0,
            "retrieved_quarters": [],
            "top_matches": [],
        }
    valid_history = []
    for row in train_rows:
        target_value = float(_safe_float(row.get(target_col), float("nan")))
        if not np.isfinite(target_value):
            continue
        past_rows = list(row.get("forward_rows") or [])
        if not past_rows:
            continue
        valid_history.append(dict(row))
    if len(valid_history) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(valid_history)),
            "support": 0.0,
            "effective_memory_count": 0.0,
            "directional_consistency": 0.0,
            "attention_focus": 0.0,
            "retrieved_quarters": [],
            "top_matches": [],
        }

    query_record = _native_rows_to_attention_record(current_rows)
    history_slice = valid_history[-int(history_cap) :] if int(history_cap) > 0 else valid_history
    raw_pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any], float, float]] = []
    current_q_num = _quarter_number(str(current_quarter or ""))
    for past_row in history_slice:
        past_quarter = str(past_row.get("quarter") or "")
        past_rows = list(past_row.get("forward_rows") or [])
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
        score = float(attention.get("attention_score", 0.0)) * recency + same_q
        raw_pairs.append((score, past_row, attention, recency, same_q))
    raw_pairs.sort(key=lambda item: item[0], reverse=True)
    top_pairs = raw_pairs[: int(quarter_top_k)]
    if not top_pairs:
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(valid_history)),
            "support": 0.0,
            "effective_memory_count": 0.0,
            "directional_consistency": 0.0,
            "attention_focus": 0.0,
            "retrieved_quarters": [],
            "top_matches": [],
        }

    weights = softmax_weights([item[0] for item in top_pairs], float(quarter_temperature))
    target_logs = [float(_safe_float(item[1].get(target_col), 0.0)) for item in top_pairs]
    pred_mean = float(sum(weight * target for weight, target in zip(weights, target_logs)))
    pred_var = float(sum(weight * (target - pred_mean) ** 2 for weight, target in zip(weights, target_logs)))
    weight_sq_sum = float(sum(weight ** 2 for weight in weights))
    effective_memory_count = 0.0 if weight_sq_sum <= EPS else 1.0 / weight_sq_sum
    abs_mass = float(sum(weight * abs(target) for weight, target in zip(weights, target_logs)))
    directional_consistency = 0.0 if abs_mass <= EPS else abs(pred_mean) / abs_mass
    attention_focus = float(sum(weight * float(item[2].get("attention_focus", 0.0)) for weight, item in zip(weights, top_pairs)))
    history_support = float(len(valid_history) / max(len(valid_history) + float(shrink_k), EPS))
    support = float(np.exp(-pred_var / max(float(var_tau), EPS)))
    support *= min(1.0, effective_memory_count / max(float(neff_scale), 1.0))
    if float(directional_consistency_power) > 0.0:
        support *= max(directional_consistency, 0.0) ** float(directional_consistency_power)
    if float(attention_focus_power) > 0.0:
        support *= max(attention_focus, 0.0) ** float(attention_focus_power)
    support = float(_clip(history_support * support, 0.0, 1.0))
    pred_raw = float(_clip(pred_mean, -float(max_abs_log_delta), float(max_abs_log_delta)))
    pred = float(pred_raw * support)
    return {
        "pred": pred,
        "pred_raw": pred_raw,
        "train_count": int(len(valid_history)),
        "support": support,
        "effective_memory_count": float(effective_memory_count),
        "directional_consistency": float(directional_consistency),
        "attention_focus": float(attention_focus),
        "retrieved_quarters": [
            {
                "quarter": str(item[1].get("quarter") or ""),
                "weight": round(float(weights[idx]), 6),
                "score": round(float(item[0]), 6),
                "target_value": round(float(_safe_float(item[1].get(target_col), 0.0)), 6),
                "attention_score": round(float(item[2].get("attention_score", 0.0)), 6),
                "attention_focus": round(float(item[2].get("attention_focus", 0.0)), 6),
                "direction_alignment": round(float(item[2].get("direction_alignment", 0.0)), 6),
                "recency": round(float(item[3]), 6),
                "same_quarter_bonus": round(float(item[4]), 6),
            }
            for idx, item in enumerate(top_pairs)
        ],
        "top_matches": sanitize_for_json(top_pairs[0][2].get("top_matches", [])) if top_pairs else [],
    }


def _build_native_surface(
    *,
    ticker: str,
    observed_quarter: str,
    target_quarter: str,
    group_map: Mapping[Tuple[str, str, str], Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    rows = list(group_map.get((ticker, observed_quarter, target_quarter), []))
    agg = _aggregate_group(rows, "forward_only")
    out = {name: float(agg["features"].get(name, 0.0)) for name in RAW_CARD_FEATURE_NAMES}
    out["forward_rows"] = list(agg.get("forward_rows") or [])
    out["card_group_present"] = bool(rows)
    return out


def _predict_compressed_candidate(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    current_features: Mapping[str, float],
    target_col: str,
    alpha: float,
    min_train: int,
    shrink_k: float,
    max_abs_log_delta: float,
) -> Dict[str, Any]:
    if len(train_rows) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": 0,
            "support": 0.0,
            "top_contribs": [],
        }
    x_rows = np.asarray([[float(_safe_float(row.get(name), 0.0)) for name in feature_names] for row in train_rows], dtype=float)
    y_rows = np.asarray([float(_safe_float(row.get(target_col), float("nan"))) for row in train_rows], dtype=float)
    mask = np.isfinite(y_rows) & np.all(np.isfinite(x_rows), axis=1)
    x_train = x_rows[mask]
    y_train = y_rows[mask]
    if len(y_train) < int(min_train):
        return {
            "pred": 0.0,
            "pred_raw": float("nan"),
            "train_count": int(len(y_train)),
            "support": 0.0,
            "top_contribs": [],
        }
    model = _fit_ridge(x_train, y_train, float(alpha))
    x_cur = np.asarray([float(_safe_float(current_features.get(name), 0.0)) for name in feature_names], dtype=float)
    pred_raw = float(_predict_ridge(model, x_cur))
    pred_raw = float(_clip(pred_raw, -float(max_abs_log_delta), float(max_abs_log_delta)))
    support = float(len(y_train) / max(len(y_train) + float(shrink_k), 1e-9))
    return {
        "pred": float(pred_raw * support),
        "pred_raw": pred_raw,
        "train_count": int(len(y_train)),
        "support": support,
        "top_contribs": _top_contributions(model, x_cur, feature_names, top_k=6),
    }


def _blend_candidates(candidates: Sequence[Tuple[float, float]]) -> float:
    usable = [(float(value), float(weight)) for value, weight in candidates if np.isfinite(value) and np.isfinite(weight) and weight > 0.0]
    if not usable:
        return 0.0
    weight_sum = float(sum(weight for _, weight in usable))
    if weight_sum <= 1e-9:
        return 0.0
    return float(sum(value * weight for value, weight in usable) / weight_sum)


def _base_presence_scale(base_delta: float, max_abs_log_delta: float) -> float:
    if not np.isfinite(base_delta):
        return 0.0
    ratio = abs(float(base_delta)) / max(float(max_abs_log_delta), 1e-9)
    return float(_clip(np.sqrt(ratio), 0.0, 1.0))


def _native_reliability(memory_support: float, memory_consistency: float, fwd_conflict_ratio: float) -> float:
    support = float(_clip(memory_support, 0.0, 1.0))
    consistency = float(_clip(memory_consistency, 0.0, 1.0))
    conflict = float(_clip(fwd_conflict_ratio, 0.0, 1.0))
    return float(_clip(support * consistency * (1.0 - conflict), 0.0, 1.0))


def _native_guided_base_mix(native_reliability: float, guidance_lock: float) -> float:
    return float(_clip(native_reliability * (1.0 - guidance_lock), 0.0, 1.0))


def _build_candidate_gate_features(
    *,
    row: Mapping[str, Any],
    base_delta: float,
    raw_delta: float,
    factor_delta: float,
    native_delta: float,
    native_reliability: float,
    native_available: float,
    guidance_lock: float,
    anchor_diag: Mapping[str, float],
    anchor_error_state: Mapping[str, float],
    internal_state: Mapping[str, float],
    regime: Mapping[str, float],
    native_surface: Mapping[str, Any],
    native_metadata_proxy_flag: float,
    native_semantic_clarity: float,
    memory_diag: Mapping[str, Any],
    temporal_delta: float,
    temporal_support: float,
    temporal_attention_focus: float,
    dualstream_delta: float,
    dualstream_support: float,
) -> Dict[str, float]:
    return {
        "gate_base_delta": float(base_delta),
        "gate_raw_delta": float(raw_delta),
        "gate_factor_delta": float(factor_delta),
        "gate_native_delta": float(native_delta),
        "gate_native_available": float(native_available),
        "gate_native_reliability": float(native_reliability),
        "gate_gap_raw_factor": float(raw_delta - factor_delta),
        "gate_gap_base_native": float(base_delta - native_delta),
        "gate_sign_raw_factor": float(_sign_num(raw_delta) * _sign_num(factor_delta)),
        "gate_sign_base_native": float(_sign_num(base_delta) * _sign_num(native_delta)),
        "gate_guidance_lock": float(guidance_lock),
        "gate_guidance_numeric_available": float(_safe_float(row.get("guidance_numeric_available"), 0.0)),
        "gate_guidance_score_norm": float(_safe_float(row.get("guidance_score_norm"), 0.0)),
        "gate_guid_band_ratio": float(_safe_float(row.get("guid_band_ratio"), 0.0)),
        "gate_anchor_uncertainty": float(_safe_float(anchor_diag.get("anchor_uncertainty"), 0.0)),
        "gate_anchor_error_recent": float(_safe_float(anchor_error_state.get("anchor_error_recent_abs_log"), 0.0)),
        "gate_anchor_error_same_quarter": float(_safe_float(anchor_error_state.get("anchor_error_same_quarter_abs_log"), 0.0)),
        "gate_anchor_error_same_guidance": float(_safe_float(anchor_error_state.get("anchor_error_same_guidance_abs_log"), 0.0)),
        "gate_internal_strength": float(_safe_float(internal_state.get("internal_strength"), 0.0)),
        "gate_internal_balance_abs": float(abs(_safe_float(internal_state.get("internal_balance"), 0.0))),
        "gate_regime_vol_qoq4": float(_safe_float(regime.get("reg_vol_qoq4"), 0.0)),
        "gate_regime_same_quarter_support": float(_safe_float(regime.get("reg_same_quarter_support"), 0.0)),
        "gate_segment_share_top1": float(_safe_float(row.get("segment_share_top1"), 0.0)),
        "gate_segment_share_count": float(_safe_float(row.get("segment_share_count"), 0.0)),
        "gate_native_fwd_abs_mass": float(_safe_float(native_surface.get("fwd_abs_mass"), 0.0)),
        "gate_native_fwd_conflict_ratio": float(_safe_float(native_surface.get("fwd_conflict_ratio"), 0.0)),
        "gate_native_fwd_count_log": float(_safe_float(native_surface.get("fwd_count_log"), 0.0)),
        "gate_native_metadata_proxy_flag": float(native_metadata_proxy_flag),
        "gate_native_semantic_clarity": float(native_semantic_clarity),
        "gate_memory_pred_delta": float(_safe_float(memory_diag.get("pred_delta"), 0.0)),
        "gate_memory_support": float(_safe_float(memory_diag.get("support"), 0.0)),
        "gate_memory_consistency": float(_safe_float(memory_diag.get("consistency"), 0.0)),
        "gate_temporal_delta": float(temporal_delta),
        "gate_temporal_support": float(temporal_support),
        "gate_temporal_attention_focus": float(temporal_attention_focus),
        "gate_gap_base_temporal": float(base_delta - temporal_delta),
        "gate_gap_native_temporal": float(native_delta - temporal_delta),
        "gate_dualstream_delta": float(dualstream_delta),
        "gate_dualstream_support": float(dualstream_support),
        "gate_gap_base_dualstream": float(base_delta - dualstream_delta),
        "gate_gap_native_dualstream": float(native_delta - dualstream_delta),
        "gate_base_x_weak_guidance": float(base_delta * (1.0 - guidance_lock)),
        "gate_raw_x_uncertainty": float(raw_delta * _safe_float(anchor_diag.get("anchor_uncertainty"), 0.0)),
        "gate_native_x_support": float(native_delta * _safe_float(memory_diag.get("support"), 0.0)),
    }


def _resolve_mainline_contract(args: argparse.Namespace) -> Dict[str, Any]:
    requested_profile = str(getattr(args, "mainline_profile", "manual") or "manual")
    resolved = {
        "mainline_profile": requested_profile,
        "candidate_profile": str(args.candidate_profile),
        "base_candidate_mode": str(args.base_candidate_mode),
        "candidate_gate_feature_mode": str(args.candidate_gate_feature_mode),
        "native_target_mode": str(getattr(args, "native_target_mode", "full_target_delta") or "full_target_delta"),
        "native_card_admissibility_mode": str(getattr(args, "native_card_admissibility_mode", "none") or "none"),
        "native_candidate_feature_mode": NATIVE_CANDIDATE_FEATURE_MODE,
        "native_reliability_mode": NATIVE_RELIABILITY_MODE,
        "candidate_gate_target_mode": CANDIDATE_GATE_TARGET_MODE,
        "profile_role": "manual",
        "profile_notes": "Manual argument-driven contract.",
    }
    if requested_profile != "manual":
        preset = MAINLINE_PROFILE_PRESETS[requested_profile]
        resolved.update(
            {
                "candidate_profile": str(preset["candidate_profile"]),
                "base_candidate_mode": str(preset["base_candidate_mode"]),
                "candidate_gate_feature_mode": str(preset["candidate_gate_feature_mode"]),
                "native_target_mode": str(preset.get("native_target_mode", "full_target_delta")),
                "native_card_admissibility_mode": str(preset.get("native_card_admissibility_mode", "none")),
                "profile_role": str(preset["profile_role"]),
                "profile_notes": str(preset["profile_notes"]),
            }
        )
    native_target_mode = str(resolved["native_target_mode"])
    if native_target_mode not in NATIVE_TARGET_MODES:
        raise ValueError(f"Unsupported native_target_mode: {native_target_mode}")
    native_card_admissibility_mode = str(resolved["native_card_admissibility_mode"])
    if native_card_admissibility_mode not in NATIVE_CARD_ADMISSIBILITY_MODES:
        raise ValueError(f"Unsupported native_card_admissibility_mode: {native_card_admissibility_mode}")
    feature_mode = str(resolved["candidate_gate_feature_mode"])
    resolved["candidate_gate_feature_names"] = list(CANDIDATE_GATE_FEATURE_SETS[feature_mode])
    resolved["candidate_gate_feature_count"] = int(len(resolved["candidate_gate_feature_names"]))
    if str(resolved["candidate_profile"]) == "raw_factor_native_temporal" and "gate_temporal_delta" not in resolved["candidate_gate_feature_names"]:
        raise ValueError("Temporal candidate profile requires a temporal-aware candidate gate feature set.")
    if str(resolved["candidate_profile"]) == "raw_factor_native_dualstream" and "gate_dualstream_delta" not in resolved["candidate_gate_feature_names"]:
        raise ValueError("Dual-stream candidate profile requires a dualstream-aware candidate gate feature set.")
    return resolved


def _load_frozen_best_stat_map(path: str, project_root: Path) -> Dict[str, Dict[str, Any]]:
    if not str(path or "").strip():
        return {}
    csv_path = Path(resolve_repo_path(path, str(project_root)))
    df = pd.read_csv(csv_path).copy()
    need = {"ticker", "best_stat_model", "best_stat_mae"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"Frozen best-stat CSV missing columns: {missing}")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[str(row["ticker"])] = {
            "best_stat_model": str(row["best_stat_model"]),
            "best_stat_mae": float(_safe_float(row["best_stat_mae"], float("nan"))),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run CSAIS candidate bridge with strong anchor and multiple correction candidates.")
    ap.add_argument("--experiment_config", default=DEFAULT_EXPERIMENT_CONFIG)
    ap.add_argument("--native_backbone_csv", default=DEFAULT_BACKBONE_CSV)
    ap.add_argument("--native_card_table_jsonl", default=DEFAULT_CARD_TABLE_JSONL)
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--candidate_profile", choices=["raw_factor", "raw_factor_native", "raw_factor_native_temporal", "raw_factor_native_dualstream"], default="raw_factor_native")
    ap.add_argument("--raw_alpha", type=float, default=8.0)
    ap.add_argument("--factor_alpha", type=float, default=8.0)
    ap.add_argument("--native_alpha", type=float, default=10.0)
    ap.add_argument("--candidate_min_train", type=int, default=10)
    ap.add_argument("--native_min_train", type=int, default=12)
    ap.add_argument("--candidate_shrink_k", type=float, default=16.0)
    ap.add_argument("--native_shrink_k", type=float, default=8.0)
    ap.add_argument("--candidate_base_scale", type=float, default=0.75)
    ap.add_argument("--base_candidate_mode", choices=["compressed_only", "native_guided_blend"], default="compressed_only")
    ap.add_argument("--native_target_mode", choices=sorted(NATIVE_TARGET_MODES), default="full_target_delta")
    ap.add_argument("--native_card_admissibility_mode", choices=sorted(NATIVE_CARD_ADMISSIBILITY_MODES), default="none")
    ap.add_argument(
        "--candidate_gate_feature_mode",
        choices=sorted(CANDIDATE_GATE_FEATURE_SETS.keys()),
        default="minimal_base_native_context_v1",
        help="Explicit candidate-gate feature contract used when --mainline_profile=manual.",
    )
    ap.add_argument(
        "--mainline_profile",
        choices=["manual"] + sorted(MAINLINE_PROFILE_PRESETS.keys()),
        default="manual",
        help="Named mainline contract preset. When not manual, this freezes candidate profile, base mode, and gate feature mode to an auditable preset.",
    )
    ap.add_argument("--shock_max_abs_log_delta", type=float, default=0.14)
    ap.add_argument("--anchor_uncertainty_recent_window", type=int, default=6)
    ap.add_argument("--anchor_uncertainty_same_quarter_min", type=int, default=2)
    ap.add_argument("--anchor_uncertainty_same_guidance_min", type=int, default=3)
    ap.add_argument("--anchor_uncertainty_tau", type=float, default=0.10)
    ap.add_argument("--gate_alpha", type=float, default=8.0)
    ap.add_argument("--gate_min_train", type=int, default=6)
    ap.add_argument("--gate_shrink_k", type=float, default=12.0)
    ap.add_argument("--gate_training_scope", choices=["company_local", "shared_pooled", "shared_blend"], default="company_local")
    ap.add_argument("--gate_local_prior_k", type=float, default=8.0)
    ap.add_argument("--memory_top_k", type=int, default=3)
    ap.add_argument("--memory_temperature", type=float, default=0.35)
    ap.add_argument("--memory_min_train", type=int, default=6)
    ap.add_argument("--temporal_candidate_min_train", type=int, default=10)
    ap.add_argument("--temporal_candidate_history_cap", type=int, default=12)
    ap.add_argument("--temporal_candidate_top_k", type=int, default=3)
    ap.add_argument("--temporal_candidate_temperature", type=float, default=0.35)
    ap.add_argument("--temporal_item_top_k", type=int, default=2)
    ap.add_argument("--temporal_item_temperature", type=float, default=0.20)
    ap.add_argument("--temporal_same_quarter_bonus", type=float, default=0.10)
    ap.add_argument("--temporal_time_decay_quarters", type=float, default=8.0)
    ap.add_argument("--temporal_var_tau", type=float, default=0.03)
    ap.add_argument("--temporal_neff_scale", type=float, default=4.0)
    ap.add_argument("--temporal_directional_consistency_power", type=float, default=0.5)
    ap.add_argument("--temporal_attention_focus_power", type=float, default=1.0)
    ap.add_argument("--temporal_candidate_shrink_k", type=float, default=12.0)
    ap.add_argument(
        "--frozen_best_stat_csv",
        default="",
        help="Optional frozen comparator map CSV with columns ticker,best_stat_model,best_stat_mae. When provided, company summaries and wins use this comparator instead of the dynamically loaded branch-local comparator.",
    )
    ap.add_argument("--output_dir", default="output/csais_candidate_bridge_v1_all12")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    requested = {ticker.upper() for ticker in args.tickers}
    out_dir = Path(resolve_repo_path(args.output_dir, str(project_root)))
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = _resolve_mainline_contract(args)
    frozen_best_stat_map = _load_frozen_best_stat_map(args.frozen_best_stat_csv, project_root)

    backbone_lookup = _load_backbone_lookup(Path(resolve_repo_path(args.native_backbone_csv, str(project_root))), requested)
    card_groups = _load_card_groups(Path(resolve_repo_path(args.native_card_table_jsonl, str(project_root))), requested)
    exp = json.loads(Path(resolve_repo_path(args.experiment_config, str(project_root))).read_text(encoding="utf-8"))

    all_quarterly: List[pd.DataFrame] = []
    company_summaries: List[Dict[str, Any]] = []
    gate_global_history: List[Dict[str, Any]] = []
    native_memory_global_history: List[Dict[str, Any]] = []

    for company in exp.get("companies", []):
        ticker = str(company.get("ticker") or "").upper()
        if requested and ticker not in requested:
            continue
        panel, best_stat_model, best_stat_mae, prehist_best_model, prehist_best_mae, prehist_anchor_error_history = _prepare_company_panel(company, project_root)
        if ticker in frozen_best_stat_map:
            best_stat_model = str(frozen_best_stat_map[ticker]["best_stat_model"])
            best_stat_mae = float(frozen_best_stat_map[ticker]["best_stat_mae"])
        model_cols = [col for col in panel.columns if col.startswith("pred__") and col not in STAT_EXCLUDE]
        quarterly_rows: List[Dict[str, Any]] = []
        actual_hist: List[float] = []
        quarter_hist: List[str] = []
        anchor_error_history: List[Dict[str, Any]] = list(prehist_anchor_error_history)
        native_memory_history: List[Dict[str, Any]] = []

        for _, row in panel.iterrows():
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

            frozen_col = f"pred__{prehist_best_model}"
            if frozen_col not in model_cols or not np.isfinite(_safe_float(row.get(frozen_col))):
                continue
            anchor_pred = float(_safe_float(row.get(frozen_col)))
            anchor_uncertainty = float(_safe_float(anchor_error_state.get("anchor_uncertainty_proxy"), 0.0))
            anchor_diag_state = {
                "anchor_uncertainty": anchor_uncertainty,
                "anchor_blend_weight": 0.0,
                "anchor_top1_gap_ratio": 0.0,
            }

            factor_map = _factorized_internal_features(row, regime, anchor_diag_state)
            internal_state = _internal_features(row)
            guidance_lock = _guidance_lock(guidance_dict)

            raw_features = {name: float(_safe_float(value, 0.0)) for name, value in zip(RAW_SHOCK_FEATURES, _build_shock_features(row, regime, anchor_diag_state))}
            factor_features = {name: float(_safe_float(factor_map.get(name), 0.0)) for name in FACTORIZED_SHOCK_FEATURES}

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
            base_weight = float(_clip(raw_result["support"] * evidence_gate * (0.35 + 0.65 * anchor_uncertainty) * (1.0 - 0.6 * guidance_lock) * float(args.candidate_base_scale), 0.0, 1.0))
            raw_delta = float(raw_result["pred_raw"] if np.isfinite(raw_result["pred_raw"]) else 0.0) * base_weight
            factor_delta = float(factor_result["pred_raw"] if np.isfinite(factor_result["pred_raw"]) else 0.0) * base_weight
            compressed_base_delta = _blend_candidates([(raw_delta, raw_result["support"]), (factor_delta, factor_result["support"])])

            native_backbone_row = backbone_lookup.get((ticker, str(row.get("quarter") or "")), {})
            observed_quarter = str(native_backbone_row.get("observed_fiscal_quarter") or "")
            native_surface = _build_native_surface(
                ticker=ticker,
                observed_quarter=observed_quarter,
                target_quarter=str(row.get("quarter") or ""),
                group_map=card_groups,
            )
            native_current_record = {
                **row.to_dict(),
                **native_surface,
                "target_fiscal_q": float(current_q),
                "target_delta_log": float("nan"),
                "native_residual_target_log": float("nan"),
                "base_candidate_delta_log": float("nan"),
                "forward_rows": list(native_surface.get("forward_rows") or []),
            }
            native_forward_rows = list(native_surface.get("forward_rows") or [])
            native_target_col = "target_delta_log" if str(contract["native_target_mode"]) == "full_target_delta" else "native_residual_target_log"
            native_result = _predict_native_card_additive_candidate(
                train_rows=native_memory_history,
                current_cards=native_forward_rows,
                target_col=native_target_col,
                admissibility_mode=str(contract["native_card_admissibility_mode"]),
                alpha=float(args.native_alpha),
                min_train=int(args.native_min_train),
                shrink_k=float(args.native_shrink_k),
            )
            native_safety = _apply_safety_guard(
                delta_pred=float(native_result["pred"]),
                features={name: float(_safe_float(native_surface.get(name), 0.0)) for name in RAW_CARD_FEATURE_NAMES},
                top_contribs=list(native_result["top_contribs"]),
                forward_row_count=len(list(native_surface.get("forward_rows") or [])),
                safety_mode="signguard_only",
                semantic_delta_pred=float("nan"),
            )
            native_metadata_proxy_flag = 1.0 if bool(native_safety.get("metadata_proxy_dominant")) else 0.0
            native_semantic_clarity = float(_safe_float(native_safety.get("semantic_clarity"), 0.0))
            memory_diag = _memory_diag(native_current_record, native_memory_history, args)
            candidate_profile = str(contract["candidate_profile"])
            native_enabled = candidate_profile in {"raw_factor_native", "raw_factor_native_temporal", "raw_factor_native_dualstream"}
            temporal_enabled = candidate_profile in {"raw_factor_native_temporal", "raw_factor_native_dualstream"}
            dualstream_enabled = candidate_profile == "raw_factor_native_dualstream"
            native_reliability = _native_reliability(
                float(_safe_float(memory_diag.get("support"), 0.0)),
                float(_safe_float(memory_diag.get("consistency"), 0.0)),
                float(_safe_float(native_surface.get("fwd_conflict_ratio"), 0.0)),
            ) if native_enabled else 0.0
            native_delta_raw = float(native_result["pred"]) if native_enabled else 0.0
            native_delta = float(native_delta_raw * native_reliability)
            base_native_mix_weight = 0.0
            if str(contract["base_candidate_mode"]) == "native_guided_blend" and native_enabled:
                base_native_mix_weight = _native_guided_base_mix(native_reliability, guidance_lock)
            seed_base_delta = float((1.0 - base_native_mix_weight) * compressed_base_delta + base_native_mix_weight * native_delta)
            seed_base_support = float(_clip(max(base_weight, base_native_mix_weight * native_reliability), 0.0, 1.0))

            temporal_result = {
                "pred": 0.0,
                "pred_raw": float("nan"),
                "train_count": 0,
                "support": 0.0,
                "effective_memory_count": 0.0,
                "directional_consistency": 0.0,
                "attention_focus": 0.0,
                "retrieved_quarters": [],
                "top_matches": [],
            }
            if temporal_enabled:
                temporal_result = _predict_temporal_graph_attention_candidate(
                    train_rows=native_memory_history,
                    current_rows=native_forward_rows,
                    current_quarter=str(row.get("quarter") or ""),
                    target_col=native_target_col,
                    min_train=int(args.temporal_candidate_min_train),
                    history_cap=int(args.temporal_candidate_history_cap),
                    quarter_top_k=int(args.temporal_candidate_top_k),
                    quarter_temperature=float(args.temporal_candidate_temperature),
                    item_top_k=int(args.temporal_item_top_k),
                    item_temperature=float(args.temporal_item_temperature),
                    same_quarter_bonus=float(args.temporal_same_quarter_bonus),
                    time_decay_quarters=float(args.temporal_time_decay_quarters),
                    var_tau=float(args.temporal_var_tau),
                    neff_scale=float(args.temporal_neff_scale),
                    directional_consistency_power=float(args.temporal_directional_consistency_power),
                    attention_focus_power=float(args.temporal_attention_focus_power),
                    max_abs_log_delta=float(args.shock_max_abs_log_delta),
                    shrink_k=float(args.temporal_candidate_shrink_k),
                )
            temporal_delta_raw = float(temporal_result["pred_raw"]) if np.isfinite(_safe_float(temporal_result.get("pred_raw"), float("nan"))) else 0.0
            temporal_delta = float(temporal_result["pred"]) if temporal_enabled else 0.0

            dualstream_result = {
                "pred": 0.0,
                "pred_raw": 0.0,
                "support": 0.0,
                "train_count": 0,
                "attention_focus": 0.0,
                "directional_consistency": 0.0,
                "sign_agreement": 0.0,
                "intrinsic_component_delta_log": 0.0,
                "temporal_component_delta_raw_log": 0.0,
                "temporal_component_delta_log": 0.0,
                "fused_cards": [],
                "top_support_cards": [],
                "top_conflict_cards": [],
                "top_temporal_matches": [],
            }
            if dualstream_enabled:
                dualstream_result = _predict_dualstream_raw_temporal_fusion_candidate(
                    train_rows=native_memory_history,
                    current_cards=native_forward_rows,
                    current_quarter=str(row.get("quarter") or ""),
                    target_col=native_target_col,
                    native_train_support=float(_safe_float(native_result.get("support"), 0.0)),
                    native_reliability=float(native_reliability),
                    native_coefs_by_name=dict(native_result.get("coefs_by_name") or {}),
                    admissibility_mode=str(contract["native_card_admissibility_mode"]),
                    min_train=int(args.temporal_candidate_min_train),
                    history_cap=int(args.temporal_candidate_history_cap),
                    quarter_top_k=int(args.temporal_candidate_top_k),
                    quarter_temperature=float(args.temporal_candidate_temperature),
                    item_top_k=int(args.temporal_item_top_k),
                    item_temperature=float(args.temporal_item_temperature),
                    same_quarter_bonus=float(args.temporal_same_quarter_bonus),
                    time_decay_quarters=float(args.temporal_time_decay_quarters),
                    var_tau=float(args.temporal_var_tau),
                    neff_scale=float(args.temporal_neff_scale),
                    directional_consistency_power=float(args.temporal_directional_consistency_power),
                    attention_focus_power=float(args.temporal_attention_focus_power),
                )
            dualstream_delta_raw = float(_safe_float(dualstream_result.get("pred_raw"), 0.0)) if dualstream_enabled else 0.0
            dualstream_delta = float(_safe_float(dualstream_result.get("pred"), 0.0)) if dualstream_enabled else 0.0
            candidate_arbitration = {
                "pred": float(seed_base_delta),
                "support_sum": float(seed_base_support),
                "candidate_supports": {
                    "compressed_base": float(seed_base_support),
                    "native_additive": float(native_reliability if native_enabled else 0.0),
                    "dualstream_fused": float(_safe_float(dualstream_result.get("support"), 0.0) if dualstream_enabled else 0.0),
                },
                "candidate_weights": {
                    "compressed_base": 1.0,
                    "native_additive": 0.0,
                    "dualstream_fused": 0.0,
                },
            }
            if dualstream_enabled:
                candidate_arbitration = _arbitrate_candidate_bridge(
                    compressed_base_delta=seed_base_delta,
                    compressed_base_support=seed_base_support,
                    native_delta=native_delta,
                    native_support=native_reliability,
                    dualstream_delta=dualstream_delta,
                    dualstream_support=float(_safe_float(dualstream_result.get("support"), 0.0)),
                )
            base_delta = float(candidate_arbitration["pred"])

            anchor_log = _safe_log(anchor_pred)
            actual_log = _safe_log(_safe_float(row.get("actual")))
            shock_target_log = float(actual_log - anchor_log) if np.isfinite(actual_log) and np.isfinite(anchor_log) else float("nan")
            gate_residual_target_log = float(shock_target_log - base_delta) if np.isfinite(shock_target_log) else float("nan")

            gate_feature_map = _build_candidate_gate_features(
                row=row,
                base_delta=base_delta,
                raw_delta=raw_delta,
                factor_delta=factor_delta,
                native_delta=native_delta,
                native_reliability=native_reliability,
                native_available=1.0 if native_enabled else 0.0,
                guidance_lock=guidance_lock,
                anchor_diag=anchor_diag_state,
                anchor_error_state=anchor_error_state,
                internal_state=internal_state,
                regime=regime,
                native_surface=native_surface,
                native_metadata_proxy_flag=native_metadata_proxy_flag,
                native_semantic_clarity=native_semantic_clarity,
                memory_diag=memory_diag,
                temporal_delta=temporal_delta,
                temporal_support=float(_safe_float(temporal_result.get("support"), 0.0)),
                temporal_attention_focus=float(_safe_float(temporal_result.get("attention_focus"), 0.0)),
                dualstream_delta=dualstream_delta,
                dualstream_support=float(_safe_float(dualstream_result.get("support"), 0.0)),
            )

            gate_used = False
            gate_scope_applied = "off"
            gate_support_used = 0.0
            gate_support_base_scaled = 0.0
            gate_train_count = 0
            gate_local_train_count = 0
            gate_shared_train_count = 0
            gate_local_weight = 0.0
            gate_shared_pred = float("nan")
            gate_local_pred = float("nan")
            gate_top_contribs: List[Tuple[str, float]] = []
            final_delta = base_delta

            local_gate_rows = list(quarterly_rows)
            shared_gate_rows = [item for item in gate_global_history if _quarter_key(str(item.get("quarter") or "")) < _quarter_key(str(row.get("quarter") or ""))]
            local_gate_result = {"pred": 0.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": []}
            shared_gate_result = {"pred": 0.0, "pred_raw": float("nan"), "train_count": 0, "support": 0.0, "top_contribs": []}

            scope = str(args.gate_training_scope)
            if scope == "company_local":
                local_gate_result = _predict_zero_intercept_history(
                    train_rows=local_gate_rows,
                    x_map=gate_feature_map,
                    feature_names=contract["candidate_gate_feature_names"],
                    target_col="gate_residual_target_log",
                    alpha=float(args.gate_alpha),
                    delta_cap_quantile=0.9,
                    shrink_k=float(args.gate_shrink_k),
                )
                gate_local_train_count = int(local_gate_result["train_count"])
                gate_train_count = gate_local_train_count
                if gate_local_train_count >= int(args.gate_min_train):
                    gate_support_used = float(local_gate_result["support"])
                    gate_support_base_scaled = float(gate_support_used * _base_presence_scale(base_delta, float(args.shock_max_abs_log_delta)))
                    final_delta = float(base_delta + gate_support_base_scaled * float(local_gate_result["pred"]))
                    gate_top_contribs = list(local_gate_result["top_contribs"])
                    gate_used = True
                    gate_scope_applied = "company_local"
            elif scope == "shared_pooled":
                shared_gate_result = _predict_zero_intercept_history(
                    train_rows=shared_gate_rows,
                    x_map=gate_feature_map,
                    feature_names=contract["candidate_gate_feature_names"],
                    target_col="gate_residual_target_log",
                    alpha=float(args.gate_alpha),
                    delta_cap_quantile=0.9,
                    shrink_k=float(args.gate_shrink_k),
                )
                gate_shared_train_count = int(shared_gate_result["train_count"])
                gate_train_count = gate_shared_train_count
                if gate_shared_train_count >= int(args.gate_min_train):
                    gate_support_used = float(shared_gate_result["support"])
                    gate_support_base_scaled = float(gate_support_used * _base_presence_scale(base_delta, float(args.shock_max_abs_log_delta)))
                    final_delta = float(base_delta + gate_support_base_scaled * float(shared_gate_result["pred"]))
                    gate_top_contribs = list(shared_gate_result["top_contribs"])
                    gate_used = True
                    gate_scope_applied = "shared_pooled"
            else:
                shared_gate_result = _predict_zero_intercept_history(
                    train_rows=shared_gate_rows,
                    x_map=gate_feature_map,
                    feature_names=contract["candidate_gate_feature_names"],
                    target_col="gate_residual_target_log",
                    alpha=float(args.gate_alpha),
                    delta_cap_quantile=0.9,
                    shrink_k=float(args.gate_shrink_k),
                )
                local_gate_result = _predict_zero_intercept_history(
                    train_rows=local_gate_rows,
                    x_map=gate_feature_map,
                    feature_names=contract["candidate_gate_feature_names"],
                    target_col="gate_residual_target_log",
                    alpha=float(args.gate_alpha),
                    delta_cap_quantile=0.9,
                    shrink_k=float(args.gate_shrink_k),
                )
                gate_shared_train_count = int(shared_gate_result["train_count"])
                gate_local_train_count = int(local_gate_result["train_count"])
                gate_train_count = max(gate_shared_train_count, gate_local_train_count)
                shared_ok = gate_shared_train_count >= int(args.gate_min_train)
                local_ok = gate_local_train_count >= int(args.gate_min_train)
                if shared_ok and local_ok:
                    gate_local_weight = float(gate_local_train_count / max(gate_local_train_count + float(args.gate_local_prior_k), 1e-9))
                    gate_pred = float((1.0 - gate_local_weight) * float(shared_gate_result["pred"]) + gate_local_weight * float(local_gate_result["pred"]))
                    gate_support_used = float((1.0 - gate_local_weight) * float(shared_gate_result["support"]) + gate_local_weight * float(local_gate_result["support"]))
                    gate_support_base_scaled = float(gate_support_used * _base_presence_scale(base_delta, float(args.shock_max_abs_log_delta)))
                    final_delta = float(base_delta + gate_support_base_scaled * gate_pred)
                    gate_top_contribs = list(local_gate_result["top_contribs"] if gate_local_weight >= 0.5 else shared_gate_result["top_contribs"])
                    gate_used = True
                    gate_scope_applied = "shared_blend"
                elif local_ok:
                    gate_support_used = float(local_gate_result["support"])
                    gate_support_base_scaled = float(gate_support_used * _base_presence_scale(base_delta, float(args.shock_max_abs_log_delta)))
                    final_delta = float(base_delta + gate_support_base_scaled * float(local_gate_result["pred"]))
                    gate_top_contribs = list(local_gate_result["top_contribs"])
                    gate_used = True
                    gate_scope_applied = "company_local_only"
                elif shared_ok:
                    gate_support_used = float(shared_gate_result["support"])
                    gate_support_base_scaled = float(gate_support_used * _base_presence_scale(base_delta, float(args.shock_max_abs_log_delta)))
                    final_delta = float(base_delta + gate_support_base_scaled * float(shared_gate_result["pred"]))
                    gate_top_contribs = list(shared_gate_result["top_contribs"])
                    gate_used = True
                    gate_scope_applied = "shared_pooled_only"
            gate_shared_pred = float(shared_gate_result["pred"]) if np.isfinite(_safe_float(shared_gate_result.get("pred_raw"), float("nan"))) else float("nan")
            gate_local_pred = float(local_gate_result["pred"]) if np.isfinite(_safe_float(local_gate_result.get("pred_raw"), float("nan"))) else float("nan")

            final_pred = float(anchor_pred * np.exp(final_delta)) if np.isfinite(anchor_pred) and anchor_pred > 0.0 else float(anchor_pred)

            native_support_cards: List[Dict[str, Any]] = []
            native_conflict_cards: List[Dict[str, Any]] = []
            if native_result.get("coefs_by_name"):
                scored_cards: List[Dict[str, Any]] = []
                for card in native_forward_rows:
                    influence = _card_additive_influence(card, native_result["coefs_by_name"], str(contract["native_card_admissibility_mode"]))
                    card_copy = dict(card)
                    card_copy["admissibility_weight"] = float(_native_card_admissibility_weight(card, str(contract["native_card_admissibility_mode"])))
                    card_copy["approx_influence"] = float(influence)
                    scored_cards.append(card_copy)
                native_support_cards, native_conflict_cards = _split_native_support_conflict_cards(
                    scored_cards,
                    target_delta=native_delta,
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
                    "pred_csais_anchor": float(anchor_pred),
                    "anchor_uncertainty": float(anchor_uncertainty),
                    "guidance_lock": float(guidance_lock),
                    "candidate_profile": str(contract["candidate_profile"]),
                    "mainline_profile": str(contract["mainline_profile"]),
                    "candidate_gate_feature_mode": str(contract["candidate_gate_feature_mode"]),
                    "candidate_gate_feature_count": int(contract["candidate_gate_feature_count"]),
                    "candidate_gate_feature_names_json": json.dumps(contract["candidate_gate_feature_names"], ensure_ascii=False, allow_nan=False),
                    "profile_role": str(contract["profile_role"]),
                    "csais_raw_candidate_delta_log": float(raw_delta),
                    "csais_factor_candidate_delta_log": float(factor_delta),
                    "csais_native_candidate_delta_raw_log": float(native_delta_raw),
                    "csais_native_candidate_delta_log": float(native_delta),
                    "csais_native_candidate_reliability": float(native_reliability),
                    "csais_native_candidate_forward_card_count": int(len(native_forward_rows)),
                    "csais_temporal_candidate_delta_raw_log": float(temporal_delta_raw),
                    "csais_temporal_candidate_delta_log": float(temporal_delta),
                    "csais_temporal_candidate_support": float(_safe_float(temporal_result.get("support"), 0.0)),
                    "csais_temporal_candidate_effective_memory_count": float(_safe_float(temporal_result.get("effective_memory_count"), 0.0)),
                    "csais_temporal_candidate_directional_consistency": float(_safe_float(temporal_result.get("directional_consistency"), 0.0)),
                    "csais_temporal_candidate_attention_focus": float(_safe_float(temporal_result.get("attention_focus"), 0.0)),
                    "csais_compressed_base_candidate_delta_log": float(compressed_base_delta),
                    "csais_seed_base_candidate_delta_log": float(seed_base_delta),
                    "csais_seed_base_candidate_support": float(seed_base_support),
                    "csais_base_native_mix_weight": float(base_native_mix_weight),
                    "csais_dualstream_candidate_delta_raw_log": float(dualstream_delta_raw),
                    "csais_dualstream_candidate_delta_log": float(dualstream_delta),
                    "csais_dualstream_candidate_support": float(_safe_float(dualstream_result.get("support"), 0.0)),
                    "csais_dualstream_candidate_directional_consistency": float(_safe_float(dualstream_result.get("directional_consistency"), 0.0)),
                    "csais_dualstream_candidate_attention_focus": float(_safe_float(dualstream_result.get("attention_focus"), 0.0)),
                    "csais_dualstream_candidate_sign_agreement": float(_safe_float(dualstream_result.get("sign_agreement"), 0.0)),
                    "csais_dualstream_intrinsic_component_delta_log": float(_safe_float(dualstream_result.get("intrinsic_component_delta_log"), 0.0)),
                    "csais_dualstream_temporal_component_delta_raw_log": float(_safe_float(dualstream_result.get("temporal_component_delta_raw_log"), 0.0)),
                    "csais_dualstream_temporal_component_delta_log": float(_safe_float(dualstream_result.get("temporal_component_delta_log"), 0.0)),
                    "csais_candidate_blend_support_sum": float(_safe_float(candidate_arbitration.get("support_sum"), 0.0)),
                    "csais_candidate_blend_base_weight": float(_safe_float((candidate_arbitration.get("candidate_weights") or {}).get("compressed_base"), 0.0)),
                    "csais_candidate_blend_native_weight": float(_safe_float((candidate_arbitration.get("candidate_weights") or {}).get("native_additive"), 0.0)),
                    "csais_candidate_blend_dualstream_weight": float(_safe_float((candidate_arbitration.get("candidate_weights") or {}).get("dualstream_fused"), 0.0)),
                    "csais_base_candidate_delta_log": float(base_delta),
                    "csais_final_candidate_delta_log": float(final_delta),
                    "csais_raw_candidate_train_count": int(raw_result["train_count"]),
                    "csais_factor_candidate_train_count": int(factor_result["train_count"]),
                    "csais_native_candidate_train_count": int(native_result["train_count"]),
                    "csais_temporal_candidate_train_count": int(_safe_float(temporal_result.get("train_count"), 0.0)),
                    "csais_dualstream_candidate_train_count": int(_safe_float(dualstream_result.get("train_count"), 0.0)),
                    "csais_candidate_gate_used": int(bool(gate_used)),
                    "csais_candidate_gate_scope": str(gate_scope_applied),
                    "csais_candidate_gate_train_count": int(gate_train_count),
                    "csais_candidate_gate_local_train_count": int(gate_local_train_count),
                    "csais_candidate_gate_shared_train_count": int(gate_shared_train_count),
                    "csais_candidate_gate_local_weight": float(gate_local_weight),
                    "csais_candidate_gate_support_used": float(gate_support_used),
                    "csais_candidate_gate_support_base_scaled": float(gate_support_base_scaled),
                    "csais_candidate_gate_shared_pred_delta_log": float(gate_shared_pred),
                    "csais_candidate_gate_local_pred_delta_log": float(gate_local_pred),
                    "csais_candidate_gate_shared_pred_residual_log": float(gate_shared_pred),
                    "csais_candidate_gate_local_pred_residual_log": float(gate_local_pred),
                    "csais_candidate_memory_available": int(bool(memory_diag.get("available", False))),
                    "csais_candidate_memory_pred_delta_log": float(_safe_float(memory_diag.get("pred_delta"), 0.0)),
                    "csais_candidate_memory_support": float(_safe_float(memory_diag.get("support"), 0.0)),
                    "csais_candidate_memory_consistency": float(_safe_float(memory_diag.get("consistency"), 0.0)),
                    "pred_csais_candidate_bridge_v1": float(final_pred),
                    "shock_target_log": float(shock_target_log),
                    "gate_residual_target_log": float(gate_residual_target_log),
                    "csais_raw_candidate_top_contribs": json.dumps(sanitize_for_json(raw_result["top_contribs"]), ensure_ascii=False, allow_nan=False),
                    "csais_factor_candidate_top_contribs": json.dumps(sanitize_for_json(factor_result["top_contribs"]), ensure_ascii=False, allow_nan=False),
                    "csais_native_candidate_top_contribs": json.dumps(sanitize_for_json(native_result["top_contribs"]), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_candidate_retrieved_quarters": json.dumps(sanitize_for_json(temporal_result.get("retrieved_quarters", [])), ensure_ascii=False, allow_nan=False),
                    "csais_temporal_candidate_top_matches": json.dumps(sanitize_for_json(temporal_result.get("top_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_dualstream_candidate_fused_cards_json": json.dumps(sanitize_for_json(dualstream_result.get("fused_cards", [])), ensure_ascii=False, allow_nan=False),
                    "csais_dualstream_candidate_top_support_cards_json": json.dumps(sanitize_for_json(dualstream_result.get("top_support_cards", [])), ensure_ascii=False, allow_nan=False),
                    "csais_dualstream_candidate_top_conflict_cards_json": json.dumps(sanitize_for_json(dualstream_result.get("top_conflict_cards", [])), ensure_ascii=False, allow_nan=False),
                    "csais_dualstream_candidate_top_temporal_matches_json": json.dumps(sanitize_for_json(dualstream_result.get("top_temporal_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_candidate_blend_supports_json": json.dumps(sanitize_for_json(candidate_arbitration.get("candidate_supports", {})), ensure_ascii=False, allow_nan=False),
                    "csais_candidate_blend_weights_json": json.dumps(sanitize_for_json(candidate_arbitration.get("candidate_weights", {})), ensure_ascii=False, allow_nan=False),
                    "csais_candidate_gate_top_contribs": json.dumps(sanitize_for_json(gate_top_contribs), ensure_ascii=False, allow_nan=False),
                    "csais_candidate_memory_top_matches": json.dumps(sanitize_for_json(memory_diag.get("top_matches", [])), ensure_ascii=False, allow_nan=False),
                    "csais_native_support_cards_json": json.dumps(sanitize_for_json(native_support_cards), ensure_ascii=False, allow_nan=False),
                    "csais_native_conflict_cards_json": json.dumps(sanitize_for_json(native_conflict_cards), ensure_ascii=False, allow_nan=False),
                    "csais_native_forward_rows_json": json.dumps(sanitize_for_json(list(native_surface.get("forward_rows") or [])[:12]), ensure_ascii=False, allow_nan=False),
                }
            )
            quarterly_rows.append(out_row)
            gate_global_history.append(dict(out_row))
            native_memory_history.append(
                {
                    **native_current_record,
                    "target_delta_log": float(shock_target_log),
                    "native_residual_target_log": float(gate_residual_target_log),
                    "base_candidate_delta_log": float(base_delta),
                    "forward_rows": list(native_surface.get("forward_rows") or []),
                }
            )
            native_memory_global_history.append(dict(native_memory_history[-1]))
            if np.isfinite(actual_log) and np.isfinite(anchor_log):
                anchor_error_history.append(
                    {
                        "quarter": str(row.get("quarter") or ""),
                        "guidance_availability": current_guidance,
                        "anchor_abs_log_error": float(abs(actual_log - anchor_log)),
                    }
                )
            actual_hist.append(_safe_float(row.get("actual")))
            quarter_hist.append(str(row.get("quarter") or ""))

        quarterly_df = pd.DataFrame(quarterly_rows)
        if quarterly_df.empty:
            continue
        company_metrics = {
            "baseline": _metrics(quarterly_df["actual"], quarterly_df["baseline_pred"]),
            "csais_anchor": _metrics(quarterly_df["actual"], quarterly_df["pred_csais_anchor"]),
            "candidate_bridge": _metrics(quarterly_df["actual"], quarterly_df["pred_csais_candidate_bridge_v1"]),
        }
        company_summaries.append(
            {
                "ticker": ticker,
                "n": int(len(quarterly_df)),
                "candidate_profile": str(contract["candidate_profile"]),
                "mainline_profile": str(contract["mainline_profile"]),
                "candidate_gate_feature_mode": str(contract["candidate_gate_feature_mode"]),
                "native_target_mode": str(contract["native_target_mode"]),
                "native_card_admissibility_mode": str(contract["native_card_admissibility_mode"]),
                "baseline_mae": float(company_metrics["baseline"]["mae"]),
                "csais_anchor_mae": float(company_metrics["csais_anchor"]["mae"]),
                "candidate_bridge_mae": float(company_metrics["candidate_bridge"]["mae"]),
                "best_stat_model": best_stat_model,
                "best_stat_mae": float(best_stat_mae),
                "beats_best_stat": bool(float(company_metrics["candidate_bridge"]["mae"]) < float(best_stat_mae)),
            }
        )
        all_quarterly.append(quarterly_df)

    quarterly = pd.concat(all_quarterly, axis=0).sort_values(["ticker", "quarter"], key=lambda s: s.map(_quarter_key) if s.name == "quarter" else s).reset_index(drop=True)
    company_df = pd.DataFrame(company_summaries).sort_values("ticker")
    quarterly_csv = out_dir / "csais_candidate_bridge_v1_quarterly.csv"
    company_csv = out_dir / "csais_candidate_bridge_v1_company_summary.csv"
    quarterly.to_csv(quarterly_csv, index=False)
    company_df.to_csv(company_csv, index=False)

    contract_payload = {
        "mainline_profile": str(contract["mainline_profile"]),
        "profile_role": str(contract["profile_role"]),
        "profile_notes": str(contract["profile_notes"]),
        "candidate_profile": str(contract["candidate_profile"]),
        "base_candidate_mode": str(contract["base_candidate_mode"]),
        "candidate_gate_feature_mode": str(contract["candidate_gate_feature_mode"]),
        "candidate_gate_feature_names": list(contract["candidate_gate_feature_names"]),
        "candidate_gate_feature_count": int(contract["candidate_gate_feature_count"]),
        "native_target_mode": str(contract["native_target_mode"]),
        "native_card_admissibility_mode": str(contract["native_card_admissibility_mode"]),
        "native_candidate_feature_mode": str(contract["native_candidate_feature_mode"]),
        "temporal_candidate_feature_mode": str(TEMPORAL_CANDIDATE_FEATURE_MODE),
        "dualstream_candidate_feature_mode": str(DUALSTREAM_CANDIDATE_FEATURE_MODE),
        "native_reliability_mode": str(contract["native_reliability_mode"]),
        "candidate_gate_target_mode": str(contract["candidate_gate_target_mode"]),
    }
    contract_json = out_dir / "csais_candidate_bridge_v1_contract.json"
    write_json(contract_json, contract_payload)

    summary = {
        "inputs": {
            "experiment_config": str(resolve_repo_path(args.experiment_config, str(project_root))),
            "native_backbone_csv": str(resolve_repo_path(args.native_backbone_csv, str(project_root))),
            "native_card_table_jsonl": str(resolve_repo_path(args.native_card_table_jsonl, str(project_root))),
            "tickers": sorted(company_df["ticker"].tolist()),
            "frozen_best_stat_csv": str(resolve_repo_path(args.frozen_best_stat_csv, str(project_root))) if str(args.frozen_best_stat_csv or "").strip() else "",
            "candidate_profile": str(contract["candidate_profile"]),
            "mainline_profile": str(contract["mainline_profile"]),
            "profile_role": str(contract["profile_role"]),
            "profile_notes": str(contract["profile_notes"]),
            "native_target_mode": str(contract["native_target_mode"]),
            "native_card_admissibility_mode": str(contract["native_card_admissibility_mode"]),
            "temporal_candidate_feature_mode": str(TEMPORAL_CANDIDATE_FEATURE_MODE),
            "dualstream_candidate_feature_mode": str(DUALSTREAM_CANDIDATE_FEATURE_MODE),
            "raw_alpha": float(args.raw_alpha),
            "factor_alpha": float(args.factor_alpha),
            "native_alpha": float(args.native_alpha),
            "candidate_min_train": int(args.candidate_min_train),
            "native_min_train": int(args.native_min_train),
            "candidate_shrink_k": float(args.candidate_shrink_k),
            "native_shrink_k": float(args.native_shrink_k),
            "candidate_base_scale": float(args.candidate_base_scale),
            "base_candidate_mode": str(contract["base_candidate_mode"]),
            "gate_alpha": float(args.gate_alpha),
            "gate_min_train": int(args.gate_min_train),
            "gate_shrink_k": float(args.gate_shrink_k),
            "gate_training_scope": str(args.gate_training_scope),
            "gate_local_prior_k": float(args.gate_local_prior_k),
            "memory_top_k": int(args.memory_top_k),
            "memory_temperature": float(args.memory_temperature),
            "memory_min_train": int(args.memory_min_train),
            "temporal_candidate_min_train": int(args.temporal_candidate_min_train),
            "temporal_candidate_history_cap": int(args.temporal_candidate_history_cap),
            "temporal_candidate_top_k": int(args.temporal_candidate_top_k),
            "temporal_candidate_temperature": float(args.temporal_candidate_temperature),
            "temporal_item_top_k": int(args.temporal_item_top_k),
            "temporal_item_temperature": float(args.temporal_item_temperature),
            "temporal_same_quarter_bonus": float(args.temporal_same_quarter_bonus),
            "temporal_time_decay_quarters": float(args.temporal_time_decay_quarters),
            "temporal_var_tau": float(args.temporal_var_tau),
            "temporal_neff_scale": float(args.temporal_neff_scale),
            "temporal_directional_consistency_power": float(args.temporal_directional_consistency_power),
            "temporal_attention_focus_power": float(args.temporal_attention_focus_power),
            "temporal_candidate_shrink_k": float(args.temporal_candidate_shrink_k),
            "native_candidate_feature_mode": str(contract["native_candidate_feature_mode"]),
            "candidate_gate_feature_mode": str(contract["candidate_gate_feature_mode"]),
            "candidate_gate_feature_names": list(contract["candidate_gate_feature_names"]),
            "candidate_gate_feature_count": int(contract["candidate_gate_feature_count"]),
            "native_target_mode": str(contract["native_target_mode"]),
            "native_card_admissibility_mode": str(contract["native_card_admissibility_mode"]),
            "native_reliability_mode": str(contract["native_reliability_mode"]),
            "candidate_gate_target_mode": str(contract["candidate_gate_target_mode"]),
        },
        "metrics": {
            "pooled": {
                "baseline": _metrics(quarterly["actual"], quarterly["baseline_pred"]),
                "csais_anchor": _metrics(quarterly["actual"], quarterly["pred_csais_anchor"]),
                "candidate_bridge": _metrics(quarterly["actual"], quarterly["pred_csais_candidate_bridge_v1"]),
            },
            "macro_mae": {
                "baseline": float(company_df["baseline_mae"].mean()),
                "csais_anchor": float(company_df["csais_anchor_mae"].mean()),
                "candidate_bridge": float(company_df["candidate_bridge_mae"].mean()),
            },
        },
        "wins": {
            "candidate_bridge_beats_best_stat_companies": int(company_df["beats_best_stat"].sum()),
        },
        "outputs": {
            "quarterly_csv": str(quarterly_csv),
            "company_summary_csv": str(company_csv),
            "contract_json": str(contract_json),
        },
    }
    write_json(out_dir / "csais_candidate_bridge_v1_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit("Implementation dependency only; run scripts/run_reference_replay.sh.")
