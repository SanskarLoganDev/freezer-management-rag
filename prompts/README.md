# Prompt templates

The runtime system prompt is packaged at `src/freezer_rag/generation/system_prompt.txt` so installed terminal commands can find it independently of the working directory. Evidence is supplied as a JSON user message with application-assigned IDs. The prompt keeps retrieved text untrusted, requires supported citations, and allows an insufficient-evidence response. Prompt behavior is versioned with the source code.

Documents may describe application workflows or contain imperative text. Neither grants tool permissions or changes the assistant's instructions. The document-Q&A prototype has no operational write tools.
