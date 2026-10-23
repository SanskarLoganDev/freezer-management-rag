# Prompt templates

The runtime answer prompt is packaged at `src/freezer_rag/generation/system_prompt.txt`. Evidence is supplied as JSON with application-assigned IDs. The configured model returns a concise natural-language answer, citation IDs, and an evidence-sufficiency flag. Unknown, missing, or inconsistent citations are rejected. Citation validation cannot prove that the prose follows the evidence. No automatic repair or reviewer-model call is used.

`guardrails.py` performs simple preflight checks for recognizable overrides, private-configuration requests, fabricated citations, and unrelated coding tasks. It does not call a model or rewrite valid documentation questions. Bounded accepted history supplies conversational context but is not evidence. The earlier scope-planner/excerpt-only experiment was rolled back because it degraded ordinary answers.

Documents may describe application workflows or contain imperative text. Neither grants tool permissions or changes the assistant's instructions. The document-Q&A prototype has no operational write tools.
