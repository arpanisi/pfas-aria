# LLM-Based Grounding and Literature Evidence

This document describes how PFAS-ARIA uses language models around literature grounding. It is separate from the statistical model design because the LLM is not part of coefficient estimation, model fitting, or statistical scoring.

The intended role of the LLM is:

```text
computed model evidence + retrieved literature evidence
  -> guarded scientific explanation
  -> concise rationale for expert review
```

The LLM does not decide whether a model is statistically valid. It does not create R2, p-values, coefficients, diagnostics, or literature similarity scores. Those are computed before the LLM is asked to summarize anything.

---

## 1. Why Use LLMs Here

The statistical pipeline can identify a candidate relationship, but raw outputs are hard to read:

- column names may be cryptic
- coefficients need outcome-specific interpretation
- literature matches may be semantically related but not obviously connected
- multiple hypotheses need a short expert-facing summary

The LLM layer converts structured evidence into readable scientific language. It is used for communication and synthesis, not inference.

Typical LLM tasks include:

- rewriting a raw column-name hypothesis into one clear sentence
- explaining which predictors drove the fitted outcome
- connecting model evidence to matched citation titles/snippets
- summarizing several grounded hypotheses into a short system-level finding
- suggesting follow-up analyses based on significant and non-significant variables

---

## 2. Preference for Free and Open-Source Models

PFAS-ARIA is designed to work primarily with free or open-source model options. This keeps the system cheaper to run, easier to reproduce, and less dependent on a proprietary model provider.

The current configuration supports three deployment styles:

| Provider mode | Use case | Notes |
|---------------|----------|-------|
| OpenRouter free models | Hosted access to free/open-weight or free-tier chat models. | Useful when local hardware is unavailable. Availability and rate limits can change. |
| Ollama | Local open-source inference. | Best for privacy and reproducibility when the user can run models locally. |
| RunPod / OpenAI-compatible endpoint | Self-hosted or rented GPU inference. | Useful for larger open-weight models without depending on a closed API. |

The configured model list favors free models such as DeepSeek, Llama, Qwen, Gemma, GLM, Hermes, Nemotron, and other open/free model routes where available. The exact hosted availability can vary, so the system includes fallback models rather than assuming one model will always respond.

### Why This Matters

Free/open models are attractive for scientific tooling, but they are less predictable than a single managed frontier model:

- some models time out
- some return reasoning text instead of final prose
- some produce malformed or multilingual fragments
- some hosted free routes have rate limits or intermittent outages
- model behavior can change across provider updates
- long context or citation-heavy prompts may degrade output quality

The architecture assumes this variability and treats LLM output as optional. If generation fails or validation rejects the text, the system falls back to deterministic wording.

---

## 3. Literature-Based Scoring

Literature scoring is computed before the LLM writes anything.

For each candidate statistical relationship, the system builds retrieval queries from the outcome and predictor names. These queries are used to search uploaded papers and external literature metadata.

There are two related query styles:

1. **Semantic retrieval query**
   A compact natural-language query such as:

   ```text
   final concentration as a function of voltage, treatment time, pH
   ```

   This is embedded and compared with uploaded corpus chunks.

2. **External literature search query**
   A keyword-focused query containing scientific terms such as PFAS, PFOA, plasma, UV photolysis, electrochemical oxidation, defluorination, or water treatment.

   This is used for external literature APIs and then scored by embedding similarity.

### Corpus Scoring

Uploaded papers are split into chunks and embedded with a pretrained sentence-transformer. Candidate queries are embedded with the same model. Similarity is computed using normalized vector dot product, equivalent to cosine similarity.

The literature score for a candidate is the best matching corpus similarity among retrieved chunks:

```text
lit_score = max(similarity(query, retrieved_chunk))
```

The current screening logic uses:

- a minimum retrieval similarity threshold for candidate chunks
- a stronger literature threshold for marking a candidate as well-grounded
- the top retrieved chunk as the primary corpus citation for a candidate

### External Source Scoring

External literature candidates are retrieved from sources such as:

- OpenAlex
- Europe PMC
- Crossref
- arXiv
- Semantic Scholar

For external results, the system embeds available title/abstract/snippet text and compares it with the candidate's external search query embedding. The best result per source can be attached as supporting context with a similarity score.

### Ranking Formula

In the grounded screening path, candidates are ranked with a blended evidence score:

```text
rank_score = literature_score * statistical_fit * diagnostic_score
```

This means a candidate should not rise to the top only because it has a high R2, and it should not rise only because it matches a paper topically. Strong candidates should have statistical signal, tolerable diagnostics, and relevant literature resemblance.

Important limitation: embedding similarity measures topical or semantic resemblance. It does not prove that the cited paper supports the direction or causality of the fitted relationship.

---

## 4. What the LLM Receives

The LLM receives structured evidence after scoring:

- hypothesis description or raw predictor/outcome relationship
- primary variables
- significant variables
- coefficients
- R2 and adjusted R2
- validation or diagnostic status
- citation titles or short snippets

The prompt asks for short, constrained scientific prose. It should connect model evidence to literature context without inventing new measurements.

The LLM should not receive unrestricted raw documents when a shorter evidence packet is sufficient. Smaller prompts are more reliable for free/open models and reduce the chance of irrelevant generation.

---

## 5. Main Challenges With Free/Open Models

### Unstable Availability

Free hosted routes may be unavailable, rate-limited, or slow. A model that works one day may timeout the next day. The system therefore tries fallback models and caps per-model timeout.

### Reasoning Leakage

Some reasoning-tuned models return hidden reasoning, `<think>` blocks, or planning text. This is unacceptable in the UI. The system strips and validates outputs before display.

### Prompt Echoing

Models sometimes repeat labels such as `Input:`, `Output:`, `Answer:`, or parts of the prompt. These are rejected or stripped.

### Hallucinated Specificity

A model may invent mechanistic claims, numbers, or citation implications that were not present in the evidence packet. Prompts explicitly prohibit invented numbers, and validation/fallback behavior limits damage.

### Inconsistent Scientific Tone

Free/open models can vary in style. Some are too verbose, too vague, or too confident. The prompts request one or two sentences and deterministic fallback text is used when outputs fail validation.

### Multilingual or Garbage Output

Some models occasionally emit non-English fragments, code-like text, markup, repeated tokens, or punctuation-heavy output. Guardrails reject these cases.

### Context Dilution

If too many citations or variables are provided, the model may ignore the important ones. The system passes compact evidence: top variables, key coefficients, and a small number of citation titles.

---

## 6. Guardrails Imposed

The LLM guardrail layer has three goals:

1. prevent malformed text from reaching the user
2. prevent chain-of-thought or prompt residue from appearing
3. prevent the prose from exceeding the evidence actually computed

### Input Guardrails

Prompts are constrained to structured evidence:

- coefficients and fit metrics already computed by the model
- selected variables
- validation status
- top citation titles or snippets
- explicit instruction not to invent numbers
- explicit instruction to output only the requested sentence or paragraph

### Output Guardrails

Generated text is validated before use. The system rejects or strips:

- `<think>...</think>` blocks
- chain-of-thought preambles
- `output:`, `answer:`, `title:`, and similar label leaks
- prompt echoes
- code fragments
- URLs and markup artifacts
- repeated punctuation
- repeated-token loops
- very short non-prose fragments
- excessive non-ASCII or multilingual leakage
- text with abnormal punctuation/word density

### Retry and Fallback

If an output fails validation, the LLM call can be retried. If all attempts fail, the system uses deterministic fallback text built from the same structured evidence.

This is important for free/open models. The system expects occasional bad outputs and treats them as recoverable.

### Readback and Frontend Defense

Stored results may have been generated before the latest guardrails existed. Therefore, persisted LLM-authored fields are cleaned again when read back, and the frontend applies a final defensive filter before rendering.

---

## 7. Separation of Scoring and Explanation

The core safety principle is separation:

```text
statistics and embeddings compute evidence
LLMs explain evidence
guardrails constrain explanation
```

The LLM does not:

- select the best statistical model
- compute the literature score
- decide whether diagnostics passed
- create citation similarity
- invent support from a paper that was not retrieved

The LLM may:

- restate computed results in clearer language
- connect a coefficient direction to a citation title cautiously
- summarize where the evidence is strong or weak
- suggest follow-up analyses based on gaps

---

## 8. How to Interpret Grounded Output

A grounded hypothesis should be read as:

> This statistical relationship was observed in the uploaded data, and the literature retrieval system found semantically related scientific text.

It should not be read as:

> The literature proves this fitted relationship is causal.

When reviewing an LLM-generated rationale, check:

1. Does the explanation match the reported coefficient direction?
2. Does it avoid inventing numbers?
3. Does it correctly distinguish association from causation?
4. Are the cited papers mechanistically relevant or just topically similar?
5. Does the statistical model have acceptable diagnostics?
6. Is the result regime-specific or global?

The LLM text is a convenience layer. The actual evidence remains the model result, diagnostics, and literature similarity metadata.

---

## 9. Design Philosophy

PFAS-ARIA uses free/open LLMs because the scientific value should come from the evidence pipeline, not from dependence on a proprietary text generator.

The tradeoff is that open/free models require stricter engineering:

- fallback models for availability
- short prompts for reliability
- validation for malformed output
- deterministic fallback text
- clear separation between scoring and explanation

This keeps the system useful even when the LLM is imperfect. A failed or rejected LLM output should degrade the user experience, not corrupt the scientific evidence.
