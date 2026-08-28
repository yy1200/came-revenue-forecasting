from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from evidence_memory_residual.common import EPS, cosine_from_dicts, fiscal_year, quarter_number, safe_float, token_jaccard, weighted_jaccard


FEATURE_NAMES = [
    "same_quarter",
    "guidance_match",
    "baseline_gap",
    "segment_gap",
    "guidance_gap",
    "tone_gap",
    "demand_gap",
    "supply_gap",
    "internal_cosine",
    "external_cosine",
    "internal_tag_overlap",
    "external_tag_overlap",
    "preview_overlap",
    "internal_item_count_gap",
    "external_item_count_gap",
]


WEAK_GUIDANCE_VALUES = {
    "none",
    "qualitative_only",
    "derived_weak_numeric",
    "no_total_revenue_guidance_but_forward_commentary",
}


def is_weak_guidance(guidance_availability: Any) -> bool:
    return str(guidance_availability or "none").strip().lower() in WEAK_GUIDANCE_VALUES


def product_cycle_state(row: Mapping[str, Any]) -> Dict[str, float]:
    summary_features = row.get("summary_features") or {}
    numeric_sources = [summary_features.get("internal_numeric") or {}, summary_features.get("external_numeric") or {}]
    pos = 0.0
    neg = 0.0
    neutral = 0.0
    for numeric_map in numeric_sources:
        for key, value in numeric_map.items():
            name = str(key)
            if "product_transition" not in name and "launch_cycle" not in name:
                continue
            numeric = max(0.0, safe_float(value, 0.0))
            if name.endswith("_pos"):
                pos += numeric
            elif name.endswith("_neg"):
                neg += numeric
            else:
                neutral += numeric
    total = pos + neg + neutral
    signed = pos - neg
    sign = 0.0
    if total > EPS:
        if signed > EPS:
            sign = 1.0
        elif signed < -EPS:
            sign = -1.0
    return {
        "pos": pos,
        "neg": neg,
        "neutral": neutral,
        "total": total,
        "signed": signed,
        "sign": sign,
    }


def product_transition_alignment(current_state: Mapping[str, float], past_state: Mapping[str, float]) -> Dict[str, float]:
    current_total = max(0.0, safe_float(current_state.get("total"), 0.0))
    past_total = max(0.0, safe_float(past_state.get("total"), 0.0))
    current_signed = safe_float(current_state.get("signed"), 0.0)
    past_signed = safe_float(past_state.get("signed"), 0.0)
    gap = abs(current_total - past_total) + 0.5 * abs(current_signed - past_signed)
    alignment = 1.0 / (1.0 + gap)
    sign_match = 1.0
    if current_total > EPS and past_total > EPS:
        sign_match = 1.0 if safe_float(current_state.get("sign"), 0.0) == safe_float(past_state.get("sign"), 0.0) else 0.0
    return {
        "gap": gap,
        "alignment": alignment,
        "sign_match": sign_match,
    }


def evidence_direction_state(row: Mapping[str, Any]) -> Dict[str, float]:
    internal_tags = (row.get("internal") or {}).get("tag_counts") or {}
    external_tags = (row.get("external") or {}).get("tag_counts") or {}
    internal_pos = max(0.0, safe_float(internal_tags.get("polarity=positive"), 0.0))
    internal_neg = max(0.0, safe_float(internal_tags.get("polarity=negative"), 0.0))
    internal_mixed = max(0.0, safe_float(internal_tags.get("polarity=mixed"), 0.0))
    external_pos = max(0.0, safe_float(external_tags.get("polarity=positive"), 0.0))
    external_neg = max(0.0, safe_float(external_tags.get("polarity=negative"), 0.0))
    external_mixed = max(0.0, safe_float(external_tags.get("polarity=mixed"), 0.0))

    def _balance(pos: float, neg: float, mixed: float) -> float:
        denom = pos + neg + 0.5 * mixed
        if denom <= EPS:
            return 0.0
        return (pos - neg) / denom

    product_state = product_cycle_state(row)
    product_balance = 0.0
    if safe_float(product_state.get("total"), 0.0) > EPS:
        product_balance = safe_float(product_state.get("signed"), 0.0) / max(safe_float(product_state.get("total"), 0.0), EPS)

    directional_count = internal_pos + internal_neg + external_pos + external_neg
    mixed_count = internal_mixed + external_mixed
    support = directional_count / max(directional_count + mixed_count + 1.0, 1.0)

    return {
        "internal_balance": float(_balance(internal_pos, internal_neg, internal_mixed)),
        "external_balance": float(_balance(external_pos, external_neg, external_mixed)),
        "product_balance": float(product_balance),
        "support": float(min(max(support, 0.0), 1.0)),
        "internal_pos_count": float(internal_pos),
        "internal_neg_count": float(internal_neg),
        "internal_mixed_count": float(internal_mixed),
        "external_pos_count": float(external_pos),
        "external_neg_count": float(external_neg),
        "external_mixed_count": float(external_mixed),
        "directional_count": float(directional_count),
        "mixed_count": float(mixed_count),
    }


def _guidance_gap(current: Mapping[str, Any], past: Mapping[str, Any], missing_penalty: float) -> float:
    cur = safe_float(current.get("guidance_mid_ratio"))
    hist = safe_float(past.get("guidance_mid_ratio"))
    if math.isfinite(cur) and math.isfinite(hist):
        return abs(cur - hist)
    if math.isfinite(cur) or math.isfinite(hist):
        return float(missing_penalty)
    return 0.0


def _segment_gap(current: Mapping[str, Any], past: Mapping[str, Any]) -> float:
    return (
        abs(safe_float(current.get("segment_share_top1"), 0.0) - safe_float(past.get("segment_share_top1"), 0.0))
        + 0.5 * abs(safe_float(current.get("segment_share_top2"), 0.0) - safe_float(past.get("segment_share_top2"), 0.0))
        + 0.1 * abs(safe_float(current.get("segment_share_count"), 0.0) - safe_float(past.get("segment_share_count"), 0.0))
    )


def pair_components(current: Mapping[str, Any], past: Mapping[str, Any], guidance_missing_penalty: float = 0.35) -> Dict[str, float]:
    cur_context = current.get("context") or {}
    past_context = past.get("context") or {}
    cur_internal = (current.get("summary_features") or {}).get("internal_numeric") or {}
    past_internal = (past.get("summary_features") or {}).get("internal_numeric") or {}
    cur_external = (current.get("summary_features") or {}).get("external_numeric") or {}
    past_external = (past.get("summary_features") or {}).get("external_numeric") or {}
    cur_internal_tags = (current.get("internal") or {}).get("tag_counts") or {}
    past_internal_tags = (past.get("internal") or {}).get("tag_counts") or {}
    cur_external_tags = (current.get("external") or {}).get("tag_counts") or {}
    past_external_tags = (past.get("external") or {}).get("tag_counts") or {}
    cur_previews = current.get("previews") or {}
    past_previews = past.get("previews") or {}
    cur_internal_count = safe_float((current.get("internal") or {}).get("item_count"), 0.0)
    past_internal_count = safe_float((past.get("internal") or {}).get("item_count"), 0.0)
    cur_external_count = safe_float((current.get("external") or {}).get("item_count"), 0.0)
    past_external_count = safe_float((past.get("external") or {}).get("item_count"), 0.0)
    baseline_gap = abs(
        math.log(max(safe_float(cur_context.get("baseline_pred"), 1.0), EPS))
        - math.log(max(safe_float(past_context.get("baseline_pred"), 1.0), EPS))
    )
    internal_preview_overlap = token_jaccard(cur_previews.get("internal_tokens") or [], past_previews.get("internal_tokens") or [])
    external_preview_overlap = token_jaccard(cur_previews.get("external_tokens") or [], past_previews.get("external_tokens") or [])
    current_quarter_num = quarter_number(str(current.get("quarter") or ""))
    past_quarter_num = quarter_number(str(past.get("quarter") or ""))
    current_year = fiscal_year(str(current.get("quarter") or ""))
    past_year = fiscal_year(str(past.get("quarter") or ""))
    same_quarter = 1.0 if current_quarter_num == past_quarter_num and current_quarter_num > 0 else 0.0
    current_cycle = product_cycle_state(current)
    past_cycle = product_cycle_state(past)
    cycle_alignment = product_transition_alignment(current_cycle, past_cycle)
    guidance_match = 1.0 if str(cur_context.get("guidance_availability") or "none") == str(past_context.get("guidance_availability") or "none") else 0.0
    return {
        "same_quarter": same_quarter,
        "guidance_match": guidance_match,
        "baseline_gap": baseline_gap,
        "segment_gap": _segment_gap(cur_context, past_context),
        "guidance_gap": _guidance_gap(cur_context, past_context, float(guidance_missing_penalty)),
        "tone_gap": abs(safe_float(cur_context.get("tone_score"), 0.0) - safe_float(past_context.get("tone_score"), 0.0)),
        "demand_gap": abs(safe_float(cur_context.get("demand_mentions"), 0.0) - safe_float(past_context.get("demand_mentions"), 0.0)),
        "supply_gap": abs(safe_float(cur_context.get("supply_constraint_mentions"), 0.0) - safe_float(past_context.get("supply_constraint_mentions"), 0.0)),
        "internal_cosine": cosine_from_dicts(cur_internal, past_internal),
        "external_cosine": cosine_from_dicts(cur_external, past_external),
        "internal_tag_overlap": weighted_jaccard(cur_internal_tags, past_internal_tags),
        "external_tag_overlap": weighted_jaccard(cur_external_tags, past_external_tags),
        "product_transition_alignment": float(cycle_alignment["alignment"]),
        "preview_overlap": 0.5 * (internal_preview_overlap + external_preview_overlap),
        "internal_item_count_gap": abs(cur_internal_count - past_internal_count),
        "external_item_count_gap": abs(cur_external_count - past_external_count),
    }


def feature_vector(components: Mapping[str, float], feature_names: Sequence[str] = FEATURE_NAMES) -> Sequence[float]:
    return [float(components[name]) for name in feature_names]
