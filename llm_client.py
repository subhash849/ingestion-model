"""
llm_client.py — LLM interaction layer.

Key design choices:
  - Single call per chunk returns BOTH domain and glossary (no 2-call split)
  - Domain confidence is explicit in the response ("high"/"medium"/"low")
  - Retry with exponential backoff for transient errors (429, 503)
  - Markdown fence stripping as a safety net for non-compliant responses
  - All responses validated through Pydantic before returning to caller
"""

from __future__ import annotations
import json
import re
import time
import logging
from typing import Any
from groq import Groq, APIError as GroqAPIError

from config import LLM_PROVIDER, GROQ_MODEL, OLLAMA_MODEL, MAX_TOKENS, DOMAIN_CATEGORIES, SUPPORTED_DOMAINS, FALLBACK_DOMAIN
from models import ChunkLLMResponse, GlossaryEntry, PYDANTIC_AVAILABLE

logger = logging.getLogger(__name__)

_MAX_RETRIES   = 3
_RETRY_BACKOFF = 2.0   # seconds (doubles each retry)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(domain_hint: str | None = None) -> str:
    """
    Build a domain-aware system prompt.
    If domain_hint is provided (from dataset-level cache), inject it.
    Otherwise ask the model to detect domain itself.
    """
    domain_list   = ", ".join(SUPPORTED_DOMAINS)

    if domain_hint:
        categories    = DOMAIN_CATEGORIES.get(domain_hint, DOMAIN_CATEGORIES[FALLBACK_DOMAIN])
        category_list = " | ".join(categories)
        domain_instruction = (
            f'The domain of this text is "{domain_hint}".'
        )
        category_instruction = (
            f'You MUST assign each term one of these exact categories for the "{domain_hint}" domain:\n'
            f'{category_list}\n'
            f'Do not invent new category names. Use only the values listed above, exactly as written.'
        )
        confidence_instruction = (
            'Set confidence to "high" if the text clearly matches the domain, '
            '"medium" if partially, "low" if unclear.'
        )
    else:
        # Build a full domain→categories reference so the model can self-select
        domain_category_map = "\n".join(
            f'  "{d}": {" | ".join(cats)}'
            for d, cats in DOMAIN_CATEGORIES.items()
            if d != FALLBACK_DOMAIN
        )
        fallback_cats = " | ".join(DOMAIN_CATEGORIES[FALLBACK_DOMAIN])
        domain_instruction = (
            f"Detect the domain from this list: {domain_list}. "
            f'Use "general" only if none fit.'
        )
        category_instruction = (
            f'Once you detect the domain, assign each term a category from that domain\'s list below.\n'
            f'Use ONLY the exact category values listed — do not invent new ones.\n\n'
            f'{domain_category_map}\n'
            f'  "general": {fallback_cats}'
        )
        confidence_instruction = (
            'Set confidence to "high" (>80% sure), "medium" (50–80%), or "low" (<50%).'
        )

    return f"""You are an expert multi-domain terminology analyst.
Your task: given a text chunk, detect its domain and extract a glossary of domain-specific CONCEPTS.

{domain_instruction}

Rules for glossary extraction:
1. Extract reusable CONCEPTS only (e.g., informed_consent, liability_clause, gear_ratio).
   Do NOT extract raw values: IDs, codes, names, dates, numbers.
2. Use snake_case for all term names.
3. Write concise definitions (1–2 sentences) grounded in the provided text.
4. {category_instruction}
5. If no valid domain terms exist, return an empty glossary list.
6. {confidence_instruction}

Respond ONLY with a valid JSON object — no markdown fences, no prose:
{{
  "domain": "detected_domain",
  "confidence": "high|medium|low",
  "glossary": [
    {{"term": "string", "definition": "string", "category": "string"}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def call_llm_for_chunk(
    chunk: str,
    client: Any,
    domain_hint: str | None = None,
) -> ChunkLLMResponse | None:
    """
    Send one text chunk to the LLM. Returns a validated ChunkLLMResponse
    or None on unrecoverable failure.

    Retries up to _MAX_RETRIES times on transient API errors.
    """
    system_prompt = _build_system_prompt(domain_hint)
    user_message  = f"Extract glossary from the following text:\n\n{chunk}"

    delay = _RETRY_BACKOFF
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if LLM_PROVIDER == "ollama":
                response = client.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    format="json",
                    options={"num_predict": MAX_TOKENS}
                )
                raw = response["message"]["content"].strip()
                return _parse_and_validate(raw)
            else:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                return _parse_and_validate(raw)

        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (429, 503) and attempt < _MAX_RETRIES:
                logger.warning(
                    "API %s error (attempt %d/%d), retrying in %.1fs",
                    status, attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            logger.error("API error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            return None

    return None


# ---------------------------------------------------------------------------
# Parse + validate
# ---------------------------------------------------------------------------

def _parse_and_validate(raw: str) -> ChunkLLMResponse | None:
    """Strip markdown fences, parse JSON, validate with Pydantic."""
    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON: %s | raw: %.200s", exc, raw)
        return None

    if PYDANTIC_AVAILABLE:
        try:
            return ChunkLLMResponse(**data)
        except Exception as exc:
            logger.warning("Schema validation failed: %s", exc)
            return None
    else:
        # Fallback: manual construction without Pydantic
        entries = [
            GlossaryEntry(**e)  # type: ignore[arg-type]
            for e in data.get("glossary", [])
            if isinstance(e, dict) and e.get("term") and e.get("definition")
        ]
        return ChunkLLMResponse(          # type: ignore[call-arg]
            domain     = data.get("domain", FALLBACK_DOMAIN),
            confidence = data.get("confidence", "low"),
            glossary   = entries,
        )