"""
Infer a one-line domain description from dataset columns and corpus paper titles.
Called once before hypothesis screening begins; result threads into all LLM prompts.
"""

from __future__ import annotations

import re

from src.reporting.narrative import _llm_call_with_retry
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_HAS_CJK = re.compile(r"[一-鿿぀-ヿ]")
_DOMAIN_LABEL_RE = re.compile(
    r"^(output|answer|domain|context|result)\s*[:\-]\s*",
    re.IGNORECASE,
)
_BAD_DOMAIN_TERMS = (
    "column",
    "dataset",
    "uploaded",
    "paper",
    "title",
    "infer",
    "unknown",
)


def _fallback_domain_context(col_names: list[str]) -> str:
    return ", ".join(str(c) for c in col_names[:5] if c)


def _clean_domain_context(text: str) -> str:
    line = text.strip().splitlines()[-1].strip() if text.strip() else ""
    line = _DOMAIN_LABEL_RE.sub("", line).strip().strip("\"'").rstrip(".")
    return re.sub(r"\s+", " ", line)


def _is_clean_domain_context(text: str) -> bool:
    """Validate a short scientific domain label."""
    text = _clean_domain_context(text)
    if not text or _HAS_CJK.search(text):
        return False
    words = text.split()
    if not (2 <= len(words) <= 15):
        return False
    lower = text.lower()
    if any(term in lower for term in _BAD_DOMAIN_TERMS):
        return False
    if "?" in text or "\n" in text:
        return False
    # Keep this stricter than prose: a domain label should be mostly letters,
    # numbers, spaces, slashes, and hyphens.
    if re.search(r"[*|\\@#{}_%<>]", text):
        return False
    alpha_space = sum(1 for c in text if c.isalpha() or c.isdigit() or c in " /-&")
    if alpha_space / max(1, len(text)) < 0.80:
        return False
    return True


def infer_domain_context(col_names: list[str], paper_titles: list[str]) -> str:
    fallback = _fallback_domain_context(col_names)
    cols = ", ".join(col_names[:20])
    titles = "\n".join(f"- {t}" for t in paper_titles[:5] if t and t != "unknown")
    messages = [
        {
            "role": "system",
            "content": (
                "Output ONLY a short scientific domain label. "
                "No preamble, no reasoning, no punctuation-heavy text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given dataset column names and uploaded paper titles, infer the scientific "
                "domain in 2-15 words. Be specific: name the process, substance, or system "
                "being studied.\n\n"
                f"Dataset columns: {cols}\n\n"
                f"Corpus papers:\n{titles if titles else '(none uploaded)'}\n\n"
                "Domain label:"
            ),
        },
    ]
    try:
        settings = get_settings()
        result = _llm_call_with_retry(
            messages,
            max_tokens=60,
            temperature=0.1,
            timeout=min(60, settings.llm.request_timeout),
            use_chat_model=True,
            validator=_is_clean_domain_context,
        )
        if result:
            context = _clean_domain_context(result)
            logger.info("Inferred domain context: {}", context)
            return context
    except Exception as e:
        logger.warning("Domain context inference failed: {}", e)
    return fallback
