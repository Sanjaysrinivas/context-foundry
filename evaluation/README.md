# Golden evaluation datasets

Keep private source documents and their datasets outside Git. Copy `cases.example.jsonl`, add at least 30 human-verified cases, and identify evidence by source hash, page, stable text, and optional page coordinates rather than generated chunk IDs.

Each case supports:

- `case_id`, dataset/corpus versions, split, `question`, and `answerable`;
- `expected_evidence`: source-hash/page passages, evidence groups, and required text;
- `required_facts`: facts that must appear in both the answer and retrieved evidence;
- provenance, `review_status`, tags, and optional `reference_answer`/`document_ids`.

Only `approved_gold` records enter release metrics. Automatically generated records remain synthetic silver until source-first review.

With the app running and the matching corpus indexed:

```powershell
uv run local-rag-eval path\to\private-cases.jsonl
```

The command reports Hit@k, MRR, evidence recall and group coverage, nDCG, citation precision, fact coverage, evidence support, abstention errors, and p50/p95 latency. It exits non-zero when a documented quality gate is missed.
