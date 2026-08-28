"""Strict target-quarter CAME evidence-card extraction prompt contract."""

from __future__ import annotations

import re


STRICT_TARGET_QUARTER_EVENT_POLICY = "strict_target_quarter_causal_driver"
LEGACY_EVENT_POLICY_ALIAS = "u0_strict_driver"
RETAINED_EVENT_POLICY = STRICT_TARGET_QUARTER_EVENT_POLICY
REQUESTED_MODEL_ALIAS = "gpt-4.1-mini"
PRIMARY_DOCUMENT_CHAR_LIMIT = 100000
EVENT_FALLBACK_DOCUMENT_CHAR_LIMIT = 22000
EVENT_TEMPERATURE = 0.0
EVENT_FALLBACK_SYSTEM_SUFFIX = """

FALLBACK OUTPUT LIMIT:
- The transcript below is prefiltered to target-quarter/guidance context after an overlong structured-output attempt.
- Return at most 12 highest-confidence, non-duplicate claims.
- Prefer explicit management guidance over Q&A or generic commentary.
"""


def _bullet_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _json_safe_text(value: str) -> str:
    if not isinstance(value, str):
        value = str(value or "")
    return value.encode("utf-8", "ignore").decode("utf-8", "ignore")


def _normalize_text(value: str) -> str:
    text = (value or "").replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(value: str) -> list[str]:
    text = _normalize_text(value).replace("\u2022", "\n\u2022")
    chunks: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("\u2022"):
            chunks.append(line)
            continue
        parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z(\u2022])", line)
        chunks.extend(part.strip() for part in parts if part.strip())
    return chunks


def build_relation_user_prompt(document_text: str, observed_quarter: str) -> str:
    return f"Transcript ({observed_quarter}):\n{_json_safe_text(document_text)[:PRIMARY_DOCUMENT_CHAR_LIMIT]}"


def build_event_user_prompt(document_text: str, observed_quarter: str) -> str:
    return f"Transcript ({observed_quarter}):\n\n{_json_safe_text(document_text)[:PRIMARY_DOCUMENT_CHAR_LIMIT]}"


def build_relation_system_prompt(allowed_segments: list[str], allowed_relations: list[str]) -> str:
    segments = _bullet_list(allowed_segments)
    relations = ", ".join(allowed_relations)
    return f"""
You are a Knowledge Graph Engineer extracting ATOMIC relation claims from an earnings-call transcript.

Your goal is to produce a CLEAN forecasting KG update. Follow all rules strictly.

Return JSON that matches the provided schema exactly.

====================
HARD CONSTRAINTS
====================

GROUNDING (STRICT):
- evidence MUST be copied verbatim from the transcript (one sentence-like chunk).
- head_mention and tail_mention MUST be exact substrings that appear in evidence (case-insensitive OK). If not available, set them to "".
- Do NOT use external knowledge.

SEGMENTS (STRICT ENUM):
- The ONLY valid Segment nodes are the allowed segment strings listed below. You MUST NOT create any other Segment names.

ALLOWED SEGMENTS:
{segments}

RELATIONS (STRICT ENUM):
- Choose relation ONLY from this list (exact string match):
[{relations}]

NOISE BAN (DO NOT OUTPUT as head or tail):
- Do NOT output finance/metric pseudo-entities such as: revenue, gross margin, operating income, YoY, QoQ, forecast, guidance, Q1/Q2/Q3/Q4, FY2018, "revenue growth".
- If a sentence talks about these metrics but does NOT give a causal driver, output NOTHING.

====================
CLAIM TYPES
====================

(A) STRUCTURAL CLAIMS (slow-changing):
- Allowed relations: IS_VARIANT_OF, SUPPLIES_COMPONENTS_FOR, PARTNERS_WITH, COMPETES_WITH.
- edge_class MUST be "structural".
- Segment nodes (if used) MUST be one of the allowed segments listed above.

(B) DRIVER CLAIMS (used for forecasting):
- edge_class MUST be "driver".
- head_type MUST NOT be "Segment" (head is a driver factor like Theme/Product/Company/CustomerType).
- head MUST NOT equal tail (no self-loops).

IMPORTANT (DUAL-TRACK OUTPUT):
You may output TWO kinds of driver claims, but keep them in the same "claims" list:

(1) Segment-anchored DriverClaim (STRICT; used for shock/SR)
- tail_type MUST be "Segment".
- tail MUST be EXACTLY one of the allowed segments.
- segment MUST equal tail (same exact string).
- Prefer this form whenever the evidence explicitly supports a segment.

(2) RawFactor DriverClaim (LOOSE; NOT used for shock/SR)
- Only output if the evidence clearly states a driver but does NOT clearly mention a segment.
- In this case tail_type can be Theme/ProductFamily/ProductSKU/Company/CustomerType and tail must be grounded by evidence.
- segment MUST be "UNKNOWN" (do NOT infer).
- Keep these rare: output at most 5 RawFactor driver claims total for the whole transcript.

TARGET-LEVEL DISCIPLINE:
- First identify the most faithful grounded target level supported by the evidence.
- If the evidence explicitly names a reporting segment or an unambiguous segment phrase, use a Segment-anchored DriverClaim.
- If the evidence supports only a company-level, product-level, service-level, geography-level, or theme-level driver, keep the claim at that grounded level and set segment="UNKNOWN".
- Do NOT force company-wide commentary such as FX, macro pressure, pricing environment, channel conditions, or installed-base commentary into a single segment unless the evidence explicitly does so.

QUESTION / Q&A BAN (CRITICAL):
- If evidence contains a question mark "?" OR starts like an analyst question (e.g., "Do you", "Can you", "Could you", "Would you", "What", "How", "Any color"), DO NOT output that claim/event.

EVIDENCE LENGTH (CRITICAL):
- evidence MUST be a single sentence-like chunk and MUST be <= 240 characters.
- If the relevant text is longer, select the single most relevant clause/sentence within the limit.

ADDITIONAL REQUIRED FIELDS FOR DRIVER CLAIMS:
- theme MUST be set (use the theme enum) regardless of whether segment is known.

TEMPORAL TYPE (from evidence cues):
- Forecast/Guidance: expect, outlook, will, next quarter, guidance, anticipate
- Condition/External: due to, headwind, macro, supply, inventory, channel, FX, pricing pressure, export restriction
- else Realized/Reporting

HORIZON:
- current_q: impacts the observed quarter
- next_q: explicitly impacts next quarter / the coming quarter
- multi_q: explicitly impacts multiple future quarters (e.g., "next couple quarters")

POLARITY:
- positive: uplift / tailwind
- negative: headwind / downside / constraint
- mixed: both directions present

STRENGTH:
- Use only: low / medium / high (or unknown if unclear).

PERSISTENCE HINT:
- Set persistence_hint=true only if evidence explicitly suggests continued impact ("next couple quarters", "continue", "remain constrained").

OUTPUT REQUIREMENTS:
- head and tail MUST each be a SINGLE entity (split lists).
- Do NOT add extra keys. Do NOT invent new segments. Do NOT output banned metric entities.

THEME / CATEGORY (REQUIRED FOR DRIVER CLAIMS):
- theme MUST be one of: demand, supply, inventory, pricing, macro, competition, product_transition, geopolitics, regulation, other
- Choose the MOST appropriate theme based on evidence. Avoid "other" unless no category fits.

SEGMENT FIELD (REQUIRED FOR DRIVER CLAIMS):
- For Segment-anchored DriverClaim: segment MUST be the exact canonical Segment string.
- For RawFactor DriverClaim: segment MUST be "UNKNOWN".
"""


def build_event_system_prompt(
    allowed_segments: list[str],
    observed_quarter: str,
    affected_quarter: str,
) -> str:
    segments = _bullet_list(allowed_segments)
    return f"""
    You are a Financial Analyst. Extract quarter-specific CAUSAL DRIVERS (events/factors) mentioned in an earnings call that are explicitly stated to affect revenue in a target affected quarter.

    1) Allowed Segments
    {segments}

    2) Target-quarter constraint
    - Observed quarter: {observed_quarter}
    - Target affected quarter: {affected_quarter}
    - Only output drivers explicitly stated to affect {affected_quarter}.
    - If the sentence does not clearly reference the target quarter, do NOT output.

    3) Grounding (STRICT)
    - evidence MUST be copied verbatim from the transcript.
    - Use ONE sentence-like chunk as evidence.
    - head_mention MUST be an exact substring inside evidence (driver phrase).
    - tail_mention MUST be an exact substring inside evidence (segment/theme phrase), or "" if not present.

    3b) Q&A BAN + LENGTH
    - If evidence contains a question mark "?" or is an analyst question, output NOTHING.
    - evidence MUST be <= 240 characters; if longer, select the most relevant single clause within the limit.

    4) DRIVER vs OUTCOME (CRITICAL)
    Extract the DRIVER phrase, NOT the outcome statement.
    - Good head_mention (drivers): "export restrictions", "inventory correction", "supply constraints", "channel normalization", "FX headwind".
    - Bad head_mention (outcomes, DO NOT USE): "Gaming revenue to decline", "strong growth in Data Center".

    If the sentence only states an outcome without a causal driver, output NOTHING.

    5) Multi-driver splitting
    If a single evidence sentence contains multiple independent drivers, create MULTIPLE claims (one per driver).

    6) Canonical naming for Event node
    - head must be: EVENT::{observed_quarter}::{{short_slug}}
    - short_slug must be derived from head_mention (lower-case, hyphenated)

    7) Relation direction (required)
    - LIMITS_REVENUE_OF: driver reduces revenue (headwind/constraint)
    - BOOSTS_REVENUE_OF: driver increases revenue (tailwind/uplift)
    - SUPPORTS_REVENUE_OF: direction ambiguous but impact exists (use lower confidence)

    8) Event category and strength (required)
    - event_category MUST be one of: geopolitics, macro, regulation, supply, demand, inventory, pricing, competition, product_transition, other
    - impact_strength MUST be one of: low, medium, high (use unknown only if truly unclear)
    - attr_polarity MUST be: positive/negative/mixed/unknown (should be consistent with relation)

    9) Target-level handling (NO hallucination)
    - First determine the most faithful grounded target in the evidence.
    - If the evidence explicitly states a reporting segment or an unambiguous segment phrase, you MAY set tail_type="Segment" and tail to one of the allowed segments.
    - Otherwise, set tail_type to a schema-groundable entity type such as Theme, ProductFamily, ProductSKU, Company, or CustomerType, and set segment="UNKNOWN".
    - Do NOT force broad company-level or cross-segment drivers into a single segment unless the evidence explicitly does so.
    - Avoid creating brand new entities; prefer nodes already grounded by the transcript wording.

    10) Examples of correct restraint
    - "iPhone 7 Plus supply constrained" can target a segment if the evidence clearly supports iPhone revenue impact.
    - "FX headwind" should remain unresolved or non-segment if the evidence does not explicitly tie the impact to one reporting segment.
    - "installed base growth" or "Apple Pay adoption" can stay product/service-level if the evidence is about that surface rather than a formal segment label.

    Return JSON that matches the schema exactly. No extra keys.
    """


def build_event_fallback_system_prompt(
    allowed_segments: list[str],
    observed_quarter: str,
    affected_quarter: str,
) -> str:
    return build_event_system_prompt(allowed_segments, observed_quarter, affected_quarter) + EVENT_FALLBACK_SYSTEM_SUFFIX


def select_event_fallback_document(document_text: str, affected_quarter: str) -> str:
    sentences = _split_sentences(document_text)
    quarter_terms: list[str] = []
    match = re.search(r"_Q([1-4])", affected_quarter or "")
    if match:
        quarter_number = int(match.group(1))
        quarter_terms.extend(
            {
                1: ["q1", "q 1", "first quarter"],
                2: ["q2", "q 2", "second quarter"],
                3: ["q3", "q 3", "third quarter"],
                4: ["q4", "q 4", "fourth quarter"],
            }.get(quarter_number, [])
        )
    quarter_terms.extend(["next quarter", "coming quarter", (affected_quarter or "").lower().replace("_", " ")])
    quarter_terms = [term for term in quarter_terms if term]

    forward_terms = [
        "guidance",
        "guide",
        "outlook",
        "expect",
        "expects",
        "expected",
        "anticipate",
        "forecast",
        "next quarter",
        "coming quarter",
    ]
    impact_terms = [
        "revenue",
        "sales",
        "demand",
        "growth",
        "decline",
        "headwind",
        "tailwind",
        "shipment",
        "order",
        "booking",
        "pricing",
        "inventory",
        "supply",
        "constraint",
        "capacity",
        "tariff",
        "currency",
        "fx",
    ]

    selected_ids: set[int] = set()
    for index, sentence in enumerate(sentences):
        text = (sentence or "").strip()
        if not text or "?" in text:
            continue
        lower = text.lower()
        target_hit = any(term in lower for term in quarter_terms)
        forward_hit = any(term in lower for term in forward_terms)
        impact_hit = any(term in lower for term in impact_terms)
        if (target_hit and impact_hit) or ("guidance" in lower and impact_hit) or ("revenue" in lower and forward_hit):
            for selected_index in (index - 1, index, index + 1):
                if 0 <= selected_index < len(sentences) and "?" not in (sentences[selected_index] or ""):
                    selected_ids.add(selected_index)

    selected: list[str] = []
    total_chars = 0
    for index in sorted(selected_ids):
        text = (sentences[index] or "").strip()
        if not text:
            continue
        if total_chars + len(text) > EVENT_FALLBACK_DOCUMENT_CHAR_LIMIT:
            break
        selected.append(text)
        total_chars += len(text) + 1
    if selected:
        return "\n".join(selected)
    return document_text[:EVENT_FALLBACK_DOCUMENT_CHAR_LIMIT]


def build_relation_request_spec(
    document_text: str,
    observed_quarter: str,
    allowed_segments: list[str],
    allowed_relations: list[str],
    model_alias: str = REQUESTED_MODEL_ALIAS,
) -> dict[str, object]:
    # The retained relation call omitted temperature and therefore used the provider default.
    return {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": build_relation_system_prompt(allowed_segments, allowed_relations)},
            {"role": "user", "content": build_relation_user_prompt(document_text, observed_quarter)},
        ],
        "response_format_model": "ClaimOutput",
    }


def build_event_request_spec(
    document_text: str,
    observed_quarter: str,
    affected_quarter: str,
    allowed_segments: list[str],
    model_alias: str = REQUESTED_MODEL_ALIAS,
    *,
    fallback: bool = False,
) -> dict[str, object]:
    selected_document = select_event_fallback_document(document_text, affected_quarter) if fallback else document_text
    system_prompt = (
        build_event_fallback_system_prompt(allowed_segments, observed_quarter, affected_quarter)
        if fallback
        else build_event_system_prompt(allowed_segments, observed_quarter, affected_quarter)
    )
    return {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_event_user_prompt(selected_document, observed_quarter)},
        ],
        "response_format_model": "EventClaimOutput",
        "temperature": EVENT_TEMPERATURE,
    }
