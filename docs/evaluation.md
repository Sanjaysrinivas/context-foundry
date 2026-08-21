# RAG evaluation

Unit tests prove pipeline behavior; the included live evaluator measures whether retrieval and answers remain useful on representative documents. It does not create training data during ingestion and it does not treat model output as truth.

## Build the dataset

Copy `evaluation/cases.example.jsonl` to `evaluation/private/` and create at least 30 human-verified cases spanning direct facts, facts distributed across passages, ambiguous wording, and questions whose answer is absent. Private datasets, result files, and source documents are ignored by Git.

Each JSON Lines record has this shape:

```json
{
  "id": "policy-cancellation-001",
  "question": "What is the cancellation window?",
  "answerable": true,
  "expected_evidence": [
    {"source": "policy.pdf", "page": 4, "contains": ["30 days"]}
  ],
  "required_facts": ["30 days"],
  "reference_answer": "The cancellation window is 30 days.",
  "document_ids": []
}
```

- `id`, `question`, and `answerable` identify the case.
- `expected_evidence` identifies a relevant citation by filename, one-based page, and stable text fragments. It is required for answerable cases.
- `required_facts` lists normalized phrases that must appear in both the answer and retrieved evidence. It is required for answerable cases.
- `reference_answer` is documentation for reviewers; deterministic scoring uses `required_facts`.
- `document_ids` optionally scopes the case to specific indexed documents. Leave it empty to search the whole collection.

For an unanswerable case, use an empty evidence/facts list and `null` reference answer:

```json
{"id":"policy-absent-001","question":"Who approved this policy?","answerable":false,"expected_evidence":[],"required_facts":[],"reference_answer":null}
```

Chunk IDs are deliberately excluded because changing the parser or chunk size would invalidate them. Review expected passages when a source document itself changes.

## Run it

Start the application, index the matching corpus, then run:

```powershell
uv run local-rag-eval evaluation/private/cases.jsonl
```

Use `--base-url` for another local port. The command prints JSON and exits with status 1 if a quality threshold is missed. Run `uv run local-rag-eval --help` to see every configurable threshold.

## Metrics and initial gates

| Layer | Metric | Default gate | Meaning |
|---|---|---:|---|
| Retrieval | Hit@k | ≥ 0.85 | at least one expected passage appears in returned citations |
| Retrieval | Mean reciprocal rank | ≥ 0.70 | expected evidence ranks near the top |
| Retrieval | Citation precision | ≥ 0.90 | returned citations match annotated evidence |
| Generation | Fact coverage | ≥ 0.90 | required facts appear in the answer |
| Grounding | Evidence support | ≥ 0.90 | required answer facts also occur in retrieved evidence |
| Abstention | Accuracy | ≥ 0.90 | absent-answer cases return no citations and an explicit insufficient-evidence response |
| Runtime | p95 latency | recorded | end-to-end API latency for the local machine/model |

Fact matching is a deterministic, case-insensitive phrase check, not an LLM judge. This makes regressions reproducible but means paraphrases must be represented by suitable required phrases or reviewed manually. The gates are initial portfolio targets, not universal production guarantees.

## Change protocol

1. Freeze the corpus, evaluation cases, model tags, and retrieval configuration.
2. Record the date, machine, corpus size, and p95 latency with the baseline output.
3. Run the same cases against the proposed parser, chunking, retrieval, prompt, or model change.
4. Compare retrieval metrics before judging generated prose and inspect every abstention regression.
5. Adopt a neural reranker or more complex framework only when the measured gain justifies its latency and maintenance cost.
