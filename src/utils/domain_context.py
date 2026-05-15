"""
Infer a one-line domain description from dataset columns and corpus paper titles.
Called once before hypothesis screening begins; result threads into all LLM prompts.
"""

from __future__ import annotations

import re

from src.utils.logging import get_logger

logger = get_logger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HAS_CJK = re.compile(r"[一-鿿぀-ヿ]")


def _extract_domain_line(text: str) -> str:
    """Return the final clean English line from a potentially CoT-heavy response."""
    text = _THINK_BLOCK.sub("", text).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Walk from the end; take the first line that is English, ≤20 words, no meta-markers
    for line in reversed(lines):
        if _HAS_CJK.search(line):
            continue
        words = line.split()
        if 2 <= len(words) <= 20:
            # Strip leading labels like "Output:" / "Answer:" / "Domain:"
            line = re.sub(
                r"^(output|answer|domain|context|result)\s*[:\-]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            )
            return line.rstrip(".")
    return lines[-1].rstrip(".") if lines else ""


def infer_domain_context(col_names: list[str], paper_titles: list[str]) -> str:
    cols = ", ".join(col_names[:20])
    titles = "\n".join(f"- {t}" for t in paper_titles[:5] if t and t != "unknown")
    prompt = (
        "You are given column names from an experimental dataset and titles of "
        "uploaded scientific papers. Infer the scientific domain in ONE line "
        "(max 15 words). Be specific — name the process, substance, or system "
        "being studied. Output only that line, nothing else.\n\n"
        f"Dataset columns: {cols}\n\n"
        f"Corpus papers:\n{titles if titles else '(none uploaded)'}"
    )
    try:
        from src.utils.llm_client import chat_completion

        result = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
            use_chat_model=True,
        )
        if result and len(result.strip()) > 4:
            context = _extract_domain_line(result)
            if context:
                logger.info("Inferred domain context: {}", context)
                return context
    except Exception as e:
        logger.warning("Domain context inference failed: {}", e)
    return ", ".join(col_names[:5])
