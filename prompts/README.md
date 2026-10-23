# Prompt templates

The runtime evidence-selection prompt is packaged at `src/freezer_rag/generation/system_prompt.txt`. Evidence is supplied as JSON with application-assigned IDs. The model returns only an evidence-sufficiency flag and selected IDs; application code renders the original excerpts. Free-form model prose, invented facts, and generated code are not accepted answer fields. Unknown IDs and incomplete requested-source coverage are rejected, with one bounded repair attempt using the same model.

`guardrails.py` supplies the separate scope/query-planning prompt to that same configured Ollama model. It distinguishes documentation, mixed requests, unrelated questions, live data, actions, and overrides. False premises should be rewritten as neutral document lookups, not refused. Only accepted previous questions help resolve follow-ups; prior model answers never enter the planning or selection prompts. Explicit source requirements are derived by application code from the user request. No additional reviewer model is used.

Documents may describe application workflows or contain imperative text. Neither grants tool permissions or changes the assistant's instructions. The document-Q&A prototype has no operational write tools.
