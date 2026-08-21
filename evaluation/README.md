# Golden evaluation datasets

Keep private source documents and their datasets outside Git. Copy `cases.example.jsonl`, add at least 30 human-verified cases, and identify evidence by source, page, and stable text rather than generated chunk IDs.

Each case supports:

- `id`, `question`, and `answerable`;
- `expected_evidence`: source/page passages with required text;
- `required_facts`: facts that must appear in both the answer and retrieved evidence;
- optional `reference_answer` and `document_ids`.

With the app running and the matching corpus indexed:

```powershell
uv run local-rag-eval path\to\private-cases.jsonl
```

The command reports Hit@k, mean reciprocal rank, citation precision, fact coverage, evidence support, abstention accuracy, and p95 latency. It exits non-zero when the documented baseline targets are missed.
