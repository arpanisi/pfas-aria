"""
Narrative Generator.
LLM writes three things — everything else is templated:
  1. Executive summary (150-200 words, plain English)
  2. Finding narratives (one paragraph per significant finding)
  3. Contradiction notes (if findings contradict cited papers)

The LLM never generates numbers. It only interprets them.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable

from src.reporting.sections import (
    CrossOutputConsistency,
    ParameterEffectRow,
    ReportSections,
)
from src.utils.config import get_settings
from src.utils.llm_client import chat_completion
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NarrativeGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(self, sections: ReportSections) -> ReportSections:
        logger.info("Generating report narratives...")
        sections.executive_summary = self._executive_summary(sections)
        sections.finding_narratives = self._finding_narratives(sections)
        sections.contradiction_notes = self._contradiction_notes(sections)
        logger.info("Narratives complete")
        return sections

    def _executive_summary(self, sections: ReportSections) -> str:
        c = sections.classification
        cv = sections.convergence
        n_sig = sum(1 for e in sections.parameter_effects if e.is_significant)
        top_vars = [
            x.variable
            for x in sections.cross_output_consistency[:3]
            if x.consistency_score > 0.5
        ]
        context = {
            "experimental_group": c.experimental_group,
            "n_experiments": c.n_experiments,
            "n_rows": c.n_rows,
            "outcome_variable": c.outcome_variable,
            "active_outputs": c.active_outputs,
            "converged": cv.converged,
            "total_rounds": cv.total_rounds,
            "final_score": round(cv.final_score, 3),
            "stop_reason": cv.stop_reason,
            "n_significant_findings": n_sig,
            "top_variables": top_vars,
            "time_explains_pct": sections.time_structure.time_explains_pct,
            "n_citations": len(sections.citations),
            "pathway_metrics": sections.pathway_metrics.interpretation,
        }
        prompt = (
            "You are writing the executive summary of a scientific convergence report.\n\n"
            f"Dataset context:\n{json.dumps(context, indent=2)}\n\n"
            "Write a 150-200 word executive summary for a domain scientist.\n"
            "Requirements:\n"
            "- Plain English, no statistical jargon\n"
            "- State which variables matter and in which direction\n"
            "- State convergence status\n"
            "- State literature agreement\n"
            "- Do NOT invent numbers not provided\n"
            "- Do NOT mention OLS, fixed effects, LASSO, or model families\n"
            "- Third person past tense\n"
            "Write only the paragraph, no heading."
        )
        return self._call_llm(
            prompt, max_tokens=300, fallback=self._fallback_summary(sections)
        )

    def _finding_narratives(self, sections: ReportSections) -> list[str]:
        narratives = []
        seen: set[str] = set()
        sig = [e for e in sections.parameter_effects if e.is_significant]
        for consistency in sections.cross_output_consistency[:5]:
            if consistency.consistency_score < 0.3 or consistency.variable in seen:
                continue
            seen.add(consistency.variable)
            var_effects = [e for e in sig if e.variable == consistency.variable]
            if not var_effects:
                continue
            supporting = next(
                (
                    c
                    for c in sections.citations
                    if c.supporting_variable == consistency.variable
                ),
                None,
            )
            context = {
                "variable": consistency.variable,
                "direction": consistency.dominant_direction,
                "outputs_significant": consistency.outputs_significant,
                "consistency_score": consistency.consistency_score,
                "effects": [
                    {
                        "output": e.output,
                        "coefficient": round(e.coefficient, 4),
                        "p_value": round(e.p_value, 4),
                        "effect_size": e.effect_size,
                        "ci": [round(e.ci_lower, 4), round(e.ci_upper, 4)],
                    }
                    for e in var_effects[:3]
                ],
                "literature_support": supporting.title if supporting else None,
                "similarity": supporting.similarity_score if supporting else None,
            }
            prompt = (
                "Write a 2-3 sentence scientific finding narrative.\n\n"
                f"Context:\n{json.dumps(context, indent=2)}\n\n"
                "Requirements:\n"
                "- 2-3 sentences only\n"
                "- Written for a domain scientist\n"
                "- State what the variable does, how strongly, whether literature agrees\n"
                "- Do NOT mention model families\n"
                "- Start with the variable name\n"
                "Write only the narrative."
            )
            narratives.append(
                self._call_llm(
                    prompt,
                    max_tokens=150,
                    fallback=self._fallback_finding(consistency, var_effects),
                )
            )
        return narratives

    def _contradiction_notes(self, sections: ReportSections) -> list[str]:
        notes = []
        for finding in sections.unresolved.contradicted_findings:
            prompt = (
                "Write one sentence noting a contradiction between a model finding "
                "and published literature.\n\n"
                f"Finding: {finding}\n\n"
                "Be factual. One sentence only."
            )
            notes.append(
                self._call_llm(prompt, max_tokens=80, fallback=f"Note: {finding}")
            )
        return notes

    def _call_llm(self, prompt: str, max_tokens: int = 200, fallback: str = "") -> str:
        try:
            from src.utils.resilience import get_llm_circuit

            def _do() -> str:
                return chat_completion(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3,
                    timeout=min(120, self.settings.llm.request_timeout),
                )

            return get_llm_circuit().call(_do) or fallback
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return fallback

    def _fallback_summary(self, sections: ReportSections) -> str:
        c = sections.classification
        cv = sections.convergence
        top = [
            x.variable
            for x in sections.cross_output_consistency[:3]
            if x.consistency_score > 0.5
        ]
        n_sig = sum(1 for e in sections.parameter_effects if e.is_significant)
        status = "converged" if cv.converged else "reached maximum rounds"
        return (
            f"The PFAS-ARIA system analyzed {c.n_experiments} experiments "
            f"({c.n_rows} observations) from a {c.experimental_group} dataset "
            f"across {cv.total_rounds} rounds of autonomous hypothesis testing. "
            f"The pipeline {status} with a match score of {cv.final_score:.2f}. "
            f"{n_sig} statistically significant parameter effects were identified "
            f"across {len(c.active_outputs)} response variables. "
            + (f"Key variables: {', '.join(top)}. " if top else "")
            + f"Results were grounded against {len(sections.citations)} literature sources."
        )

    def _fallback_finding(
        self, consistency: CrossOutputConsistency, effects: list[ParameterEffectRow]
    ) -> str:
        return (
            f"{consistency.variable} showed a {consistency.dominant_direction} and statistically significant "
            f"effect on {', '.join(consistency.outputs_significant) if consistency.outputs_significant else 'the outcome'}, "
            f"consistent across {consistency.consistency_score:.0%} of tested outputs."
        )


# ── Per-hypothesis mechanistic rationale (dashboard / Postgres) ───────────────


def _hypothesis_rationale_llm(
    prompt: str,
    fallback: str,
    *,
    max_tokens: int = 220,
    request_timeout_seconds: int | None = None,
    use_chat_model: bool = False,
) -> str:
    try:
        from src.utils.resilience import get_llm_circuit

        settings = get_settings()
        cap = request_timeout_seconds if request_timeout_seconds is not None else 120
        effective = min(int(cap), int(settings.llm.request_timeout))

        def _do() -> str:
            return chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.35,
                timeout=effective,
                use_chat_model=use_chat_model,
            )

        return get_llm_circuit().call(_do) or fallback
    except Exception as e:  # noqa: BLE001
        logger.warning("Hypothesis rationale LLM failed: {}", e)
        return fallback


def _compact_coefs(coefs: dict[str, float], *, limit: int = 14) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, (k, v) in enumerate(coefs.items()):
        if i >= limit:
            break
        try:
            out[str(k)] = round(float(v), 4)
        except (TypeError, ValueError):
            continue
    return out


def _fallback_hypothesis_rationale(
    *,
    significant_variables: list[str],
    coefficients: dict[str, float],
    r_squared: float,
    adj_r_squared: float,
    validation_passed: bool,
    citation_titles: list[str],
) -> str:
    parts: list[str] = []
    if significant_variables:
        dirs = []
        for v in significant_variables[:5]:
            c = coefficients.get(v)
            if c is None:
                continue
            try:
                dirs.append(f"{v} ({'+' if float(c) >= 0 else ''}{float(c):.3f})")
            except (TypeError, ValueError):
                dirs.append(v)
        if dirs:
            parts.append(
                "The strongest statistical signal involves "
                + ", ".join(dirs)
                + ", suggesting these predictors align most closely with the fitted response."
            )
    parts.append(
        f"In-sample fit is moderate (R²={r_squared:.3f}, adj. R²={adj_r_squared:.3f}) "
        f"and validation {'passed' if validation_passed else 'did not fully pass'}."
    )
    if citation_titles:
        parts.append(
            "Corpus matches include: "
            + "; ".join(t[:120] for t in citation_titles[:2])
            + "."
        )
    return (
        " ".join(parts)
        if parts
        else (
            f"Model fit summary: R²={r_squared:.3f}, adj. R²={adj_r_squared:.3f}. "
            "Interpretation pending richer variable structure."
        )
    )


_BAD_STARTS = (
    "we need",
    "let me",
    "i need",
    "the user",
    "let's",
    "sure,",
    "here is",
    "here's",
    "rewrite",
    "input:",
    "certainly",
)

_GARBAGE_PATTERNS = (
    "<|",
    "#include",
    ".map(",
    "Array.",
    "NSAttributed",
    "dp-thinking",
    "dp-answer",
    "---------",
    "####",
    # URL / encoding fragments
    ".com/",
    "http://",
    "https://",
    "%20",
    # LaTeX / math markup
    "<math>",
    "^{",
    ".rseqtable",
    "{\\",
)

# Consecutive non-word/non-space punctuation (e.g. ))))))), .>}.., ---)
_REPEATED_PUNCT = re.compile(r"[^\w\s]{3,}")

# Chars that should never appear inside a prose word
_INVALID_WORD_CHARS = re.compile(r"[*|\\@#{}_%]")

# Common English stopwords excluded from the repetition check
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "is", "are", "was", "were",
    "to", "for", "with", "that", "this", "it", "as", "at", "be", "by", "on",
    "not", "from", "but", "its", "their", "which", "have", "has", "been",
    "also", "both", "more", "than", "into", "over", "such", "these",
})

# ── Thinking-strip helpers ─────────────────────────────────────────────────────

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Phrases that mark the end of chain-of-thought and the start of the real answer
_COT_END = re.compile(
    r"(?:let\s+(?:final|me\s+output|the\s+final)|final\s+(?:output|answer|sentence)[:\s]"
    r"|ok(?:ay)?\s*[,:.]\s*(?:sent|here|so|final|output|give|the)"
    r"|output\s+is\s+body"
    r"|i\s+must\s+output"
    r"|\bsent\.\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

# Residual CoT label patterns that shouldn't survive into clean prose
_COT_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:think|say|body|tax|ok|end|also|but|note|for\s+force)\s*[:.]\s*",
    re.IGNORECASE,
)


def _strip_thinking(text: str) -> str:
    """Remove DeepSeek <think> blocks and chain-of-thought preamble; keep the real answer."""
    text = _THINK_BLOCK.sub("", text).strip()
    # Take everything after the last CoT-end marker if one exists
    matches = list(_COT_END.finditer(text))
    if matches:
        candidate = text[matches[-1].end() :].strip()
        if len(candidate) > 20:
            text = candidate
    # Second pass: split on sentence boundaries (". Capital" or ".Capital"),
    # then keep only chunks that look like real prose output.
    parts = re.split(r"\.(?:\s+(?=[A-Z])|(?=[A-Z]))", text)
    real = [
        p.strip()
        for p in parts
        if len(p.strip().split()) >= 8  # substantial length
        and not _COT_LABEL_RE.search(p)  # no CoT labels
        and p.strip()[:1].isupper()  # starts with capital (not ":" or '"')
    ]
    if real:
        # Deduplicate before joining — some LLMs repeat a sentence verbatim
        seen_s: set[str] = set()
        deduped: list[str] = []
        for s in real:
            key = s.lower().strip()
            if key not in seen_s:
                seen_s.add(key)
                deduped.append(s)
        text = ". ".join(deduped[-2:]).strip()
        if not text.endswith("."):
            text += "."
    return text.strip()


def _is_clean_sentence(text: str) -> bool:
    """Return False if the text looks like garbage, chain-of-thought, or multilingual noise."""
    if not text or len(text) < 15:
        return False
    # Non-ASCII ratio — catches German/Polish/Chinese/emoji leakage (tightened from 0.03)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.015:
        return False
    # Alpha+space density — catches punctuation-heavy garbage (tightened from 0.60)
    alpha_space = sum(1 for c in text if c.isalpha() or c == " ")
    if alpha_space / len(text) < 0.65:
        return False
    if text.lower().startswith(_BAD_STARTS):
        return False
    if any(p in text for p in _GARBAGE_PATTERNS):
        return False
    # Consecutive punctuation block — catches ))))))), .>>}.., etc.
    if _REPEATED_PUNCT.search(text):
        return False
    words = text.split()
    # Minimum word count — single-word or token-fragment output is not prose
    if len(words) < 6:
        return False
    # Dirty-word check — words containing chars that should never appear in prose
    strip_chars = ".,!?;:()[]\"'"
    dirty = sum(
        1 for w in words
        if _INVALID_WORD_CHARS.search(w.strip(strip_chars))
    )
    if dirty / len(words) > 0.15:
        return False
    # Repetition check — non-stopword (len≥4) appearing in >20% of all words
    # catches "narudi narudi narudi …" style looping output
    significant = [
        w.lower().strip(strip_chars)
        for w in words
        if len(w.strip(strip_chars)) >= 4
        and w.lower().strip(strip_chars) not in _STOPWORDS
    ]
    if significant:
        top_count = Counter(significant).most_common(1)[0][1]
        if top_count / len(words) > 0.20:
            return False
    return True


def _is_clean_paragraph(text: str) -> bool:
    """Validate a short prose paragraph (next steps / summary): no residual CoT markers."""
    if not _is_clean_sentence(text):
        return False
    # Colon density: more than 2 colons suggests label:value thinking leakage
    if text.count(":") > 2:
        return False
    # Residual CoT labels anywhere in the text
    if _COT_LABEL_RE.search(text):
        return False
    # Single-word or two-word "sentences" (e.g. "Tax." "Also." "But.") indicate thinking fragments
    for fragment in re.split(r"(?<=[.!])\s+", text):
        words_in_frag = fragment.strip().split()
        if 1 <= len(words_in_frag) <= 2 and fragment.strip().endswith("."):
            return False
    return True


def _llm_call_with_retry(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    use_chat_model: bool = True,
    max_attempts: int = 3,
    validator: Callable[[str], bool] | None = None,
) -> str | None:
    """Call the LLM and retry up to max_attempts times if output fails validation.

    validator defaults to _is_clean_sentence. Pass _is_clean_title for stricter
    title-specific checks.
    """
    from src.utils.resilience import get_llm_circuit

    validate = validator if validator is not None else _is_clean_sentence
    circuit = get_llm_circuit()
    for attempt in range(1, max_attempts + 1):
        try:

            def _do() -> str:
                return chat_completion(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    use_chat_model=use_chat_model,
                )

            result = _strip_thinking(circuit.call(_do) or "").strip()
            # Strip common label-leak prefixes
            for prefix in ("output:", "title:", "answer:"):
                if result.lower().startswith(prefix):
                    result = result[len(prefix) :].strip()
                    break
            result = result.lstrip(":").strip()
            if validate(result):
                return result
            logger.warning(
                "LLM attempt {}/{} failed validation, retrying", attempt, max_attempts
            )
        except Exception as e:
            logger.warning("LLM attempt {}/{} failed: {}", attempt, max_attempts, e)
    return None


def aggregate_system_summary(rationales: list[str]) -> str | None:
    """Summarise all per-hypothesis rationales into one short paragraph for the UI ribbon."""
    clean = [r for r in rationales if _is_clean_sentence(r)]
    if not clean:
        return None

    bullet_block = "\n".join(f"- {r[:200]}" for r in clean)
    messages = [
        {
            "role": "system",
            "content": "Output ONLY 3 sentences. No preamble, no thinking, no markers.",
        },
        {
            "role": "user",
            "content": (
                "Based on these findings, write 3 sentences summarising "
                "the key drivers and literature alignment. Be specific. No method jargon.\n\n"
                + bullet_block
            ),
        },
    ]
    try:
        settings = get_settings()
        return _llm_call_with_retry(
            messages,
            max_tokens=120,
            temperature=0.2,
            timeout=min(90, settings.llm.request_timeout),
            validator=_is_clean_paragraph,
        )
    except Exception as e:
        logger.warning("System summary aggregation failed: {}", e)
        return None


_KNOWN_ACRONYMS = {
    "ph",
    "uv",
    "ntp",
    "cap",
}
_UNIT_SUFFIXES = re.compile(
    r"\s*(mg[_\s]l|ug[_\s]l|ng[_\s]l|mg[_\s]kg|ppm|ppb|pct|percent|min|sec|hrs?|degc|celsius)\s*$",
    re.IGNORECASE,
)


def _plain_name(col: str) -> str:
    """Convert a snake_case column name to plain English, preserving common acronyms."""
    label = _UNIT_SUFFIXES.sub("", col.replace("_", " ")).strip()
    words = label.split()
    return " ".join(w.upper() if w.lower() in _KNOWN_ACRONYMS else w for w in words)


def _deterministic_title(bundles: list) -> str:
    """Build a coherent decision title purely from bundle data — no LLM, never fails."""
    if not bundles:
        return "Key Predictors Identified for Degradation Outcome"

    best = max(
        bundles,
        key=lambda b: getattr(getattr(b, "model_result", None), "r_squared", 0) or 0,
    )
    sig: list[str] = list(
        getattr(best.model_result, "significant_variables", None) or []
    )[:2]
    desc: str = getattr(best.hypothesis, "description", "") or ""

    # Parse outcome from "predict X in regime" pattern in the description
    outcome = ""
    m = re.search(r"predict\s+(.+?)\s+in\s+regime", desc, re.IGNORECASE)
    if m:
        outcome = _plain_name(m.group(1).strip())

    if sig and outcome:
        preds = " and ".join(_plain_name(v).title() for v in sig)
        return f"{preds} Drive {outcome.title()}"
    if sig:
        preds = " and ".join(_plain_name(v).title() for v in sig)
        return f"{preds} Are Key Predictors"
    if outcome:
        return f"Multiple Factors Drive {outcome.title()}"
    return "Key Predictors Identified for Degradation Outcome"


def _is_clean_title(text: str) -> bool:
    """Validate a decision-making title: right length, no fragments, no questions, no loops."""
    if not text or not text[0].isupper():
        return False
    words = text.split()
    if not (5 <= len(words) <= 12):
        return False
    if "?" in text:
        return False
    # Only a trailing period is acceptable — mid-title full stops signal fragment leakage
    if "." in text.rstrip("."):
        return False
    # Repeated trigram means the model looped (bigrams allowed to repeat once — predictor
    # and outcome often share a term, e.g. "C4 Concentration" in initial and final)
    if len(words) >= 3:
        trigrams = [
            f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}"
            for i in range(len(words) - 2)
        ]
        if len(trigrams) != len(set(trigrams)):
            return False
    # Must contain at least one "action" word so it reads as a finding, not a topic
    action_words = {
        "drive",
        "drives",
        "determine",
        "determines",
        "predict",
        "predicts",
        "control",
        "controls",
        "govern",
        "governs",
        "increase",
        "increases",
        "reduce",
        "reduces",
        "dominate",
        "dominates",
        "shape",
        "shapes",
        "define",
        "defines",
        "explain",
        "explains",
        "amplify",
        "amplifies",
        "are",
        "is",
    }
    if not any(
        w.lower().rstrip("s") in action_words or w.lower() in action_words
        for w in words
    ):
        return False
    return True


def _strip_llm_label_prefix(text: str) -> str:
    for prefix in ("output:", "title:", "answer:"):
        if text.lower().startswith(prefix):
            return text[len(prefix) :].strip()
    return text.lstrip(":").strip()


def clean_llm_sentence(text: str | None) -> str | None:
    """Return stripped LLM prose only if it passes the sentence guardrail."""
    cleaned = _strip_llm_label_prefix(_strip_thinking(text or "").strip())
    return cleaned if _is_clean_sentence(cleaned) else None


def clean_llm_paragraph(text: str | None) -> str | None:
    """Return stripped LLM prose only if it passes the paragraph guardrail."""
    cleaned = _strip_llm_label_prefix(_strip_thinking(text or "").strip())
    return cleaned if _is_clean_paragraph(cleaned) else None


def clean_llm_title(text: str | None) -> str | None:
    """Return stripped LLM title text only if it passes the title guardrail."""
    cleaned = _strip_llm_label_prefix(_strip_thinking(text or "").strip()).rstrip(".")
    return cleaned if _is_clean_title(cleaned) else None


def generate_display_title(
    bundles: list,
    system_summary: str | None = None,
) -> str:
    """Generate a decision-making insight title from structured bundle data.

    Builds a deterministic fallback from the top bundle first, then asks the LLM
    to humanise it into plain English. Returns the deterministic title if the LLM
    output fails validation — so this function always returns a coherent string.
    """
    fallback = _deterministic_title(bundles)

    # Extract structured facts from the best bundle for the prompt
    best = (
        max(
            bundles,
            key=lambda b: getattr(getattr(b, "model_result", None), "r_squared", 0)
            or 0,
        )
        if bundles
        else None
    )

    if best is None:
        return fallback

    sig: list[str] = list(
        getattr(best.model_result, "significant_variables", None) or []
    )[:3]
    r2: float = float(getattr(best.model_result, "r_squared", 0) or 0)
    desc: str = getattr(best.hypothesis, "description", "") or ""
    top_cit: str = ""
    for cit in getattr(best, "citations", []):
        t = getattr(cit, "title", "") or ""
        if t:
            top_cit = t[:80]
            break

    outcome = ""
    m = re.search(r"predict\s+(.+?)\s+in\s+regime", desc, re.IGNORECASE)
    if m:
        outcome = _plain_name(m.group(1).strip())

    sig_plain = [_plain_name(v) for v in sig]

    messages = [
        {
            "role": "system",
            "content": (
                "Output ONLY the title. 6-9 words. "
                "No question marks. No periods. No quotes. No explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a 6-9 word decision-making slide title that states the single most "
                "important finding. Use plain English — no method words (OLS, regression, model, plasma).\n\n"
                "Format: [Key predictor(s)] [action verb] [outcome]. Examples:\n"
                "- Gas Type and Additives Jointly Control Final Analyte Concentration\n"
                "- Initial Concentration Determines Degradation Outcome Across Conditions\n"
                "- Surfactant Choice Amplifies Fluoride Ion Yield in Plasma Treatment\n\n"
                f"Key predictors: {', '.join(sig_plain) or 'see description'}\n"
                f"Outcome: {outcome or 'see description'}\n"
                f"Model fit (R²): {r2:.2f}\n"
                + (f"Literature match: {top_cit}\n" if top_cit else "")
                + (f"Raw description: {desc[:120]}\n" if desc else "")
                + "\nTitle:"
            ),
        },
    ]

    try:
        settings = get_settings()
        result = _llm_call_with_retry(
            messages,
            max_tokens=25,
            temperature=0.2,
            timeout=min(60, settings.llm.request_timeout),
            validator=_is_clean_title,
        )
        if result:
            return result.rstrip(".")
    except Exception as e:
        logger.warning("Display title LLM failed: {}", e)

    return fallback


def generate_next_steps(
    *,
    significant_vars: list[str],
    non_significant_vars: list[str],
    system_summary: str | None,
    citation_titles: list[str],
) -> str | None:
    """Suggest what analyses to run next based on what was and wasn't significant."""
    if not system_summary or not _is_clean_sentence(system_summary):
        return None

    lit = "; ".join(t for t in citation_titles[:2] if t) or "none"
    messages = [
        {
            "role": "system",
            "content": "Output ONLY 1-2 sentences. No preamble, no bullets.",
        },
        {
            "role": "user",
            "content": (
                "Based on what was found, suggest in 1-2 sentences "
                "what analyses should be investigated next. Be specific and concise — what gaps remain, "
                "what interactions were untested, or what the literature points to but data didn't cover.\n\n"
                f"What drove the outcome: {', '.join(significant_vars) or 'none identified'}\n"
                f"What had no effect: {', '.join(non_significant_vars) or 'none'}\n"
                f"Key finding: {system_summary}\n"
                f"Literature covered: {lit}\n\n"
                "Next steps suggestion (1-2 sentences):"
            ),
        },
    ]
    try:
        settings = get_settings()
        return _llm_call_with_retry(
            messages,
            max_tokens=80,
            temperature=0.4,
            timeout=min(60, settings.llm.request_timeout),
            validator=_is_clean_paragraph,
        )
    except Exception as e:
        logger.warning("Next steps generation failed: {}", e)
        return None


def _build_label_map(variables: list[str]) -> str:
    """Auto-generate col_name = human label entries from column names."""
    seen: set[str] = set()
    lines: list[str] = []
    for v in variables:
        if v in seen:
            continue
        seen.add(v)
        label = v.replace("_", " ").strip()
        for unit in (" mg l", " mg kg", " ug l", " ng l", " pct", " percent"):
            label = label.replace(unit, "").strip()
        lines.append(f"{v} = {label}")
    return "\n".join(lines) if lines else "none"


def generate_hypothesis_description(
    *,
    raw_description: str,
    primary_variables: list[str],
    significant_variables: list[str],
    request_timeout_seconds: int | None = None,
) -> str:
    """
    Rewrite a raw column-name description into one plain-English sentence
    that a domain expert immediately understands.
    Falls back to the original raw description if the LLM is unavailable.
    """
    all_vars = list(dict.fromkeys(primary_variables + significant_variables))
    label_map = _build_label_map(all_vars)
    fallback = raw_description

    messages = [
        {
            "role": "system",
            "content": "Output ONLY the rewritten sentence. No preamble, no explanation, no repetition of the input.",
        },
        {
            "role": "user",
            "content": (
                "Rewrite as one concise scientific finding a domain expert would write in a paper.\n\n"
                "Example\n"
                "Input:  temperature_c · reaction_time_min · catalyst_conc jointly associate with yield_pct in regime 2 (n=44 observations).\n"
                "Output: Temperature, reaction time, and catalyst concentration jointly predict product yield in regime 2 (n=44).\n\n"
                f"Variable labels:\n{label_map}\n\n"
                f"Input:  {raw_description}\n"
                "Output:"
            ),
        },
    ]

    try:
        settings = get_settings()
        cap = request_timeout_seconds if request_timeout_seconds is not None else 120
        effective = min(int(cap), int(settings.llm.request_timeout))
        result = _llm_call_with_retry(
            messages, max_tokens=80, temperature=0.2, timeout=effective
        )
        return result or fallback
    except Exception as e:
        logger.warning("Hypothesis description LLM failed: {}", e)
        return fallback


def generate_hypothesis_rationale_from_model_evidence(
    *,
    hypothesis_description: str,
    primary_variables: list[str],
    model_family: str,
    r_squared: float,
    adj_r_squared: float,
    significant_variables: list[str],
    coefficients: dict[str, float],
    validation_passed: bool,
    citation_titles: list[str],
    request_timeout_seconds: int | None = None,
) -> str:
    """
    Write 2–3 sentences interpreting a fitted model for dashboard ``rationale``.

    Uses the configured LLM when available; otherwise a deterministic fallback.

    ``request_timeout_seconds`` caps the HTTP timeout for this call only (e.g. batch
    screening uses a lower cap so several bundles stay within one API request).
    """
    coef_compact = _compact_coefs(coefficients)
    titles = [t for t in citation_titles if t][:3]
    fallback = _fallback_hypothesis_rationale(
        significant_variables=significant_variables,
        coefficients=coefficients,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        validation_passed=validation_passed,
        citation_titles=titles,
    )
    lit_block = "\n".join(f"- {t}" for t in titles) if titles else "- none"
    messages = [
        {
            "role": "system",
            "content": "Output ONLY 2 sentences. No preamble, no bullet points, no chain-of-thought.",
        },
        {
            "role": "user",
            "content": (
                "Write 2 sentences connecting model results to literature.\n\n"
                "Sentence 1: Which variable drives the outcome, in which direction, and how strongly.\n"
                "Sentence 2: How the matched literature supports or contextualizes this.\n\n"
                f"Hypothesis: {hypothesis_description}\n"
                f"Significant predictors (p<0.05): {json.dumps(significant_variables)}\n"
                f"Coefficients: {json.dumps(coef_compact)}\n"
                f"R²={r_squared:.3f}, validation passed={validation_passed}\n"
                f"Matched literature:\n{lit_block}\n\n"
                "Write only the 2 sentences."
            ),
        },
    ]

    try:
        settings = get_settings()
        cap = request_timeout_seconds if request_timeout_seconds is not None else 120
        effective = min(int(cap), int(settings.llm.request_timeout))
        result = _llm_call_with_retry(
            messages, max_tokens=120, temperature=0.3, timeout=effective
        )
        return result or fallback
    except Exception as e:
        logger.warning("Hypothesis rationale LLM failed: {}", e)
        return fallback
