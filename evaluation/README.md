# Retrieval and answer evaluation

`questions.jsonl` contains a small manually reviewed starting set grounded in the supplied corpus. It is not a benchmark result or a replacement for a larger held-out set.

Expand to at least 30 representative questions before selecting an embedding model. Each record needs an ID, category, question, relevant source/locator, expected behavior, and answerability. Use paragraph- and requirement-level relevance, allowing multiple valid chunks after chunking changes.

Run all candidate models over the same source snapshot, chunking strategy, candidate counts, and hardware conditions. Build a separate collection per model/dimension. Report retrieval-only results separately from generation. Save generated reports in `data/evaluation/` with configuration and model digests; never replace this seed set with generated answers.

Measure Recall@5, MRR@5, citation correctness, unsupported-answer rate, cold/warm latency, and peak RAM/VRAM. Evaluation targets and experiment design are in `plan.md`.
