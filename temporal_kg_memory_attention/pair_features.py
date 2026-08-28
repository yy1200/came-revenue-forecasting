from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from evidence_memory_residual.common import EPS, cosine_from_dicts, safe_float, softmax_weights, token_jaccard, tokenize_preview, weighted_jaccard
from evidence_memory_residual.pair_features import pair_components as base_pair_components


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
    "internal_tag_overlap",
    "preview_overlap",
    "product_transition_alignment",
    "item_best_alignment",
    "item_segment_overlap",
    "item_relation_overlap",
    "item_segment_relation_overlap",
    "item_polarity_alignment",
    "item_persistence_alignment",
    "item_strength_alignment",
    "item_token_alignment",
    "item_mass_alignment",
    "item_direction_alignment",
]


_NORM_RE = re.compile(r"[^a-z0-9]+")
_STRENGTH_MAP = {"low": 0.33, "medium": 0.66, "high": 1.0}


def _norm_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = _NORM_RE.sub("_", text).strip("_")
    return text or "unknown"


def _segment_key(value: Any) -> str:
    text = str(value or "unknown").strip()
    if "::" in text:
        text = text.split("::", 1)[1]
    return _norm_label(text)


def _strength_value(value: Any) -> float:
    return float(_STRENGTH_MAP.get(_norm_label(value), 0.5))


def _item_importance(item: Mapping[str, Any]) -> float:
    confidence = max(0.0, safe_float(item.get("confidence"), 0.0))
    weight = abs(safe_float(item.get("weight"), 0.0))
    if weight <= EPS:
        weight = max(_strength_value(item.get("strength")), 0.1)
    return float(weight * max(confidence, 0.25))


def _polarity_value(value: Any) -> int:
    key = _norm_label(value)
    if key in {"positive", "pos", "up"}:
        return 1
    if key in {"negative", "neg", "down"}:
        return -1
    return 0


def _item_tokens(item: Mapping[str, Any]) -> List[str]:
    release_tokens = item.get("release_token_ids")
    if isinstance(release_tokens, list):
        return sorted({str(token) for token in release_tokens if str(token)})
    return tokenize_preview(item.get("verbatim"))


def _source_ref(item: Mapping[str, Any]) -> str:
    return str(item.get("instance_id") or item.get("source_record_id") or "")


def _typed_item_maps(items: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    segment_map: Dict[str, float] = defaultdict(float)
    relation_map: Dict[str, float] = defaultdict(float)
    segment_relation_map: Dict[str, float] = defaultdict(float)
    pos_mass = 0.0
    neg_mass = 0.0
    total_mass = 0.0
    for item in items:
        importance = _item_importance(item)
        if importance <= EPS:
            continue
        seg = _segment_key(item.get("segment"))
        rel = _norm_label(item.get("relation_family") or item.get("category"))
        segment_map[seg] += importance
        relation_map[rel] += importance
        segment_relation_map[f"{seg}|{rel}"] += importance
        sign = _polarity_value(item.get("polarity"))
        if sign > 0:
            pos_mass += importance
        elif sign < 0:
            neg_mass += importance
        total_mass += importance
    return {
        "segment": dict(segment_map),
        "relation": dict(relation_map),
        "segment_relation": dict(segment_relation_map),
        "direction": {
            "positive": float(pos_mass),
            "negative": float(neg_mass),
            "total": float(total_mass),
        },
    }


def _item_match(query_item: Mapping[str, Any], memory_item: Mapping[str, Any]) -> Dict[str, float]:
    segment_match = 1.0 if _segment_key(query_item.get("segment")) == _segment_key(memory_item.get("segment")) else 0.0
    relation_match = 1.0 if _norm_label(query_item.get("relation_family") or query_item.get("category")) == _norm_label(memory_item.get("relation_family") or memory_item.get("category")) else 0.0
    query_polarity = _polarity_value(query_item.get("polarity"))
    memory_polarity = _polarity_value(memory_item.get("polarity"))
    if query_polarity == 0 or memory_polarity == 0:
        polarity_match = 0.5
    else:
        polarity_match = 1.0 if query_polarity == memory_polarity else 0.0
    persistence_match = 1.0 if bool(query_item.get("persistence_hint")) == bool(memory_item.get("persistence_hint")) else 0.0
    strength_match = max(0.0, 1.0 - abs(_strength_value(query_item.get("strength")) - _strength_value(memory_item.get("strength"))))
    token_match = token_jaccard(_item_tokens(query_item), _item_tokens(memory_item))
    score = (
        0.30 * segment_match
        + 0.25 * relation_match
        + 0.15 * polarity_match
        + 0.10 * persistence_match
        + 0.10 * strength_match
        + 0.10 * token_match
    )
    return {
        "score": float(score),
        "segment_match": float(segment_match),
        "relation_match": float(relation_match),
        "polarity_match": float(polarity_match),
        "persistence_match": float(persistence_match),
        "strength_match": float(strength_match),
        "token_match": float(token_match),
    }


def _internal_items(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    items = ((row.get("internal") or {}).get("items") or []) if isinstance(row, Mapping) else []
    return [dict(item) for item in items if isinstance(item, Mapping)]


def internal_item_alignment(current: Mapping[str, Any], past: Mapping[str, Any]) -> Dict[str, float]:
    current_items = _internal_items(current)
    past_items = _internal_items(past)
    if not current_items or not past_items:
        return {
            "item_best_alignment": 0.0,
            "item_segment_overlap": 0.0,
            "item_relation_overlap": 0.0,
            "item_segment_relation_overlap": 0.0,
            "item_polarity_alignment": 0.0,
            "item_persistence_alignment": 0.0,
            "item_strength_alignment": 0.0,
            "item_token_alignment": 0.0,
            "item_mass_alignment": 0.0,
            "item_direction_alignment": 0.0,
        }

    total_weight = 0.0
    best_score_sum = 0.0
    polarity_sum = 0.0
    persistence_sum = 0.0
    strength_sum = 0.0
    token_sum = 0.0

    for query_item in current_items:
        importance = _item_importance(query_item)
        if importance <= EPS:
            continue
        best = None
        for memory_item in past_items:
            cur = _item_match(query_item, memory_item)
            if best is None or float(cur["score"]) > float(best["score"]):
                best = cur
        if best is None:
            continue
        total_weight += importance
        best_score_sum += importance * float(best["score"])
        polarity_sum += importance * float(best["polarity_match"])
        persistence_sum += importance * float(best["persistence_match"])
        strength_sum += importance * float(best["strength_match"])
        token_sum += importance * float(best["token_match"])

    current_maps = _typed_item_maps(current_items)
    past_maps = _typed_item_maps(past_items)
    direction_current = current_maps["direction"]
    direction_past = past_maps["direction"]
    total_current = max(float(direction_current["total"]), 0.0)
    total_past = max(float(direction_past["total"]), 0.0)
    mass_alignment = 0.0
    if total_current > EPS and total_past > EPS:
        mass_alignment = min(total_current, total_past) / max(total_current, total_past)
    cur_balance = 0.0 if total_current <= EPS else (float(direction_current["positive"]) - float(direction_current["negative"])) / total_current
    past_balance = 0.0 if total_past <= EPS else (float(direction_past["positive"]) - float(direction_past["negative"])) / total_past
    direction_alignment = max(0.0, 1.0 - abs(cur_balance - past_balance) / 2.0)

    denom = max(total_weight, EPS)
    return {
        "item_best_alignment": float(best_score_sum / denom),
        "item_segment_overlap": float(weighted_jaccard(current_maps["segment"], past_maps["segment"])),
        "item_relation_overlap": float(weighted_jaccard(current_maps["relation"], past_maps["relation"])),
        "item_segment_relation_overlap": float(weighted_jaccard(current_maps["segment_relation"], past_maps["segment_relation"])),
        "item_polarity_alignment": float(polarity_sum / denom),
        "item_persistence_alignment": float(persistence_sum / denom),
        "item_strength_alignment": float(strength_sum / denom),
        "item_token_alignment": float(token_sum / denom),
        "item_mass_alignment": float(mass_alignment),
        "item_direction_alignment": float(direction_alignment),
    }


def pair_components(current: Mapping[str, Any], past: Mapping[str, Any], guidance_missing_penalty: float = 0.35) -> Dict[str, float]:
    base = base_pair_components(current, past, guidance_missing_penalty=guidance_missing_penalty)
    base.update(internal_item_alignment(current, past))
    return base


def feature_vector(components: Mapping[str, float], feature_names: Sequence[str] = FEATURE_NAMES) -> List[float]:
    return [float(components.get(name, 0.0)) for name in feature_names]


def top_item_matches(current: Mapping[str, Any], past: Mapping[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    current_items = sorted(_internal_items(current), key=_item_importance, reverse=True)
    past_items = _internal_items(past)
    if not current_items or not past_items:
        return []
    matches: List[Dict[str, Any]] = []
    for query_item in current_items:
        importance = _item_importance(query_item)
        best_match = None
        best_item = None
        for memory_item in past_items:
            cur = _item_match(query_item, memory_item)
            if best_match is None or float(cur["score"]) > float(best_match["score"]):
                best_match = cur
                best_item = memory_item
        if best_match is None or best_item is None:
            continue
        matches.append(
            {
                "query_segment": str(query_item.get("segment") or ""),
                "query_relation_family": str(query_item.get("relation_family") or query_item.get("category") or ""),
                "query_polarity": str(query_item.get("polarity") or ""),
                "query_weight": float(safe_float(query_item.get("weight"), 0.0)),
                "memory_segment": str(best_item.get("segment") or ""),
                "memory_relation_family": str(best_item.get("relation_family") or best_item.get("category") or ""),
                "memory_polarity": str(best_item.get("polarity") or ""),
                "memory_weight": float(safe_float(best_item.get("weight"), 0.0)),
                "match_score": float(best_match["score"]),
                "token_match": float(best_match["token_match"]),
                "importance": float(importance),
                "query_source_ref": _source_ref(query_item),
                "memory_source_ref": _source_ref(best_item),
                "source_text_release_status": "quote_withheld_third_party",
            }
        )
    matches.sort(key=lambda item: float(item["importance"]) * float(item["match_score"]), reverse=True)
    return matches[: int(top_k)]


def cross_card_attention(
    current: Mapping[str, Any],
    past: Mapping[str, Any],
    *,
    top_k_per_query: int = 2,
    temperature: float = 0.20,
    top_k_overall: int = 6,
) -> Dict[str, Any]:
    current_items = sorted(_internal_items(current), key=_item_importance, reverse=True)
    past_items = _internal_items(past)
    if not current_items or not past_items:
        return {
            "attention_score": 0.0,
            "attention_focus": 0.0,
            "direction_alignment": 0.0,
            "query_count": 0,
            "matched_query_count": 0,
            "top_matches": [],
        }

    total_importance = 0.0
    score_sum = 0.0
    focus_sum = 0.0
    direction_sum = 0.0
    matched_query_count = 0
    top_matches: List[Dict[str, Any]] = []
    for query_item in current_items:
        importance = _item_importance(query_item)
        if importance <= EPS:
            continue
        scored = []
        for memory_item in past_items:
            match = _item_match(query_item, memory_item)
            scored.append((float(match["score"]), dict(memory_item), match))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        top_scored = scored[: max(int(top_k_per_query), 1)]
        weights = softmax_weights([item[0] for item in top_scored], float(temperature)) if len(top_scored) > 1 else [1.0]
        pooled_score = sum(weight * item[0] for weight, item in zip(weights, top_scored))
        pooled_direction = sum(weight * float(item[2].get("polarity_match", 0.0)) for weight, item in zip(weights, top_scored))
        total_importance += importance
        score_sum += importance * pooled_score
        focus_sum += importance * float(weights[0])
        direction_sum += importance * pooled_direction
        matched_query_count += 1
        best_score, best_item, best_match = top_scored[0]
        top_matches.append(
            {
                "query_segment": str(query_item.get("segment") or ""),
                "query_relation_family": str(query_item.get("relation_family") or query_item.get("category") or ""),
                "query_polarity": str(query_item.get("polarity") or ""),
                "query_weight": float(safe_float(query_item.get("weight"), 0.0)),
                "memory_segment": str(best_item.get("segment") or ""),
                "memory_relation_family": str(best_item.get("relation_family") or best_item.get("category") or ""),
                "memory_polarity": str(best_item.get("polarity") or ""),
                "memory_weight": float(safe_float(best_item.get("weight"), 0.0)),
                "match_score": float(best_score),
                "token_match": float(best_match.get("token_match", 0.0)),
                "importance": float(importance),
                "query_source_ref": _source_ref(query_item),
                "memory_source_ref": _source_ref(best_item),
                "source_text_release_status": "quote_withheld_third_party",
            }
        )

    denom = max(total_importance, EPS)
    top_matches.sort(key=lambda item: float(item["importance"]) * float(item["match_score"]), reverse=True)
    return {
        "attention_score": float(score_sum / denom),
        "attention_focus": float(focus_sum / denom),
        "direction_alignment": float(direction_sum / denom),
        "query_count": int(len(current_items)),
        "matched_query_count": int(matched_query_count),
        "top_matches": top_matches[: int(top_k_overall)],
    }


def internal_numeric_cosine(current: Mapping[str, Any], past: Mapping[str, Any]) -> float:
    current_numeric = ((current.get("summary_features") or {}).get("internal_numeric") or {}) if isinstance(current, Mapping) else {}
    past_numeric = ((past.get("summary_features") or {}).get("internal_numeric") or {}) if isinstance(past, Mapping) else {}
    return float(cosine_from_dicts(current_numeric, past_numeric))


def internal_direction_balance(row: Mapping[str, Any]) -> float:
    items = _internal_items(row)
    if not items:
        return 0.0
    signed_mass = 0.0
    total_mass = 0.0
    for item in items:
        importance = _item_importance(item)
        if importance <= EPS:
            continue
        sign = _polarity_value(item.get("polarity"))
        signed_mass += importance * float(sign)
        total_mass += importance
    if total_mass <= EPS:
        return 0.0
    return float(signed_mass / total_mass)
