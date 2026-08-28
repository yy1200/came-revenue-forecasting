"""Exact identity-masked direct same-task prompt and parser contract."""

from __future__ import annotations

import json
import math
import re
from typing import Any


def unit_description(unit: str, reporting_currency: str) -> tuple[str, float]:
    currency = str(reporting_currency or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError(f"Invalid reporting currency: {reporting_currency}")
    normalized_unit = unit.lower().strip()
    if normalized_unit == "raw":
        return currency, 1.0
    if normalized_unit == "billion":
        return f"billion {currency}", 1e9
    if normalized_unit == "million":
        return f"million {currency}", 1e6
    raise ValueError(f"Unknown unit: {unit}")


def build_system_msg(company_label: str) -> str:
    return (
        "You are a financial forecasting assistant.\n"
        f"You will be given {company_label}'s historical quarterly data and optionally management guidance.\n"
        "You must forecast the revenue for ONE target quarter.\n\n"
        "Rules:\n"
        "- Use only the provided tables; do not assume you know any future realized values.\n"
        "- Output MUST follow the required JSON schema (no extra keys, no commentary).\n"
        "- Return a single object with fields fiscal_quarter and pred_revenue.\n"
    )


def json_schema_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "revenue_forecast",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "fiscal_quarter": {"type": "string", "minLength": 1},
                    "pred_revenue": {"type": "number"},
                },
                "required": ["fiscal_quarter", "pred_revenue"],
                "additionalProperties": False,
            },
        },
    }


def build_user_prompt_hist(
    history_csv: str,
    target_fq: str,
    unit_desc: str,
    company_label: str,
) -> str:
    return (
        f"Below is {company_label}'s historical quarterly revenue (realized).\n"
        f"The unit of all values is {unit_desc}.\n\n"
        f"Historical data (past quarters):\n"
        f"```text\n{history_csv}\n```\n\n"
        f"Task: Forecast the realized revenue for the target quarter: {target_fq}.\n"
        f"Return only the JSON object."
    )


def build_user_prompt_hist_guid(
    history_csv: str,
    target_guidance_csv: str,
    target_fq: str,
    unit_desc: str,
    company_label: str,
) -> str:
    return (
        f"Below is {company_label}'s historical quarterly data.\n"
        f"The unit of all revenue and guidance values is {unit_desc}.\n\n"
        f"Historical data (past quarters; realized revenue + guidance):\n"
        f"```text\n{history_csv}\n```\n\n"
        f"For the target quarter {target_fq}, you ONLY see management guidance (revenue is unknown):\n"
        f"```text\n{target_guidance_csv}\n```\n\n"
        f"Task: Using the historical relationship between guidance and realized revenue (bias, under/over-shoot, etc.), "
        f"forecast the realized revenue for {target_fq}.\n"
        f"Return only the JSON object."
    )


def parse_response(content: str, expected_fiscal_quarter: str) -> dict[str, Any]:
    """Apply the post-response checks used by the identity-masked runner."""
    parsed = json.loads(content)
    returned_quarter = str(parsed.get("fiscal_quarter", "")).strip()
    if returned_quarter != str(expected_fiscal_quarter):
        raise ValueError(
            f"Returned fiscal_quarter {returned_quarter!r} does not match target "
            f"{expected_fiscal_quarter!r}"
        )
    pred_revenue = float(parsed["pred_revenue"])
    if not math.isfinite(pred_revenue) or pred_revenue <= 0.0:
        raise ValueError(f"Invalid pred_revenue: {parsed.get('pred_revenue')!r}")
    return parsed
