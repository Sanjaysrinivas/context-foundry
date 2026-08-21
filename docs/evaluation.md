# RAG evaluation

Unit tests prove pipeline behavior; the live evaluator measures whether retrieval and answers
remain useful on representative documents. It does not create training data during ingestion and
it never treats model output as truth.

## Evaluation lifecycle

Ingestion records the raw file SHA-256 and carries it through pages, chunks, Qdrant payloads,
document responses, and citations. Evaluation stays separate:

~~~text
source ingestion -> raw source hash -> extraction -> chunks -> index
                                             |
                                             v
candidate generation -> review -> approved gold JSONL -> regression evaluation
~~~

This separation lets one frozen dataset compare parser, chunking, retrieval, prompt, and model
changes. Automatically generated cases are synthetic silver until a reviewer checks them directly
against the source.

After upgrading an existing installation, clear and re-index documents once so older Qdrant records
receive the source_sha256 field.

## Build the dataset

Copy evaluation/cases.example.jsonl to evaluation/private/ and create at least 30 human-verified
cases spanning direct facts, facts distributed across passages, ambiguous wording, OCR or tables,
and questions whose answer is absent. Private datasets, result files, and source documents are
ignored by Git.

A versioned answerable record has this shape:

~~~json
{
  "case_id": "policy-cancellation-001",
  "dataset_version": "1.0.0",
  "split": "holdout",
  "corpus_id": "policy-demo",
  "corpus_version": "2026-08",
  "question": "What is the cancellation window?",
  "question_type": "direct_fact",
  "answerable": true,
  "expected_evidence": [
    {
      "source": "policy.pdf",
      "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "page": 4,
      "contains": ["30 days"],
      "evidence_group": "window",
      "bbox": [72.0, 214.0, 527.0, 278.0],
      "evidence_hash": "optional-evidence-fingerprint"
    }
  ],
  "required_facts": ["30 days"],
  "reference_answer": "The cancellation window is 30 days.",
  "document_ids": [],
  "generation_method": "human",
  "review_status": "approved_gold",
  "tags": ["direct_fact", "policy"]
}
~~~

- case_id (or legacy id), question, and answerable identify the case.
- dataset_version, corpus_id, and corpus_version freeze the labels and source collection
  independently of the RAG configuration.
- split should be dev, holdout, or smoke; tune on dev and reserve holdout for release
  comparisons.
- expected_evidence identifies source evidence by raw file hash, one-based PDF page, and stable
  text. Filename-only matching remains supported for older cases.
- evidence_group marks independently required evidence. A multi-passage case should use groups
  such as A and B; alternative acceptable spans can share a group.
- bbox and evidence_hash preserve optional parser-independent provenance. The current live runner
  matches citations by source hash, page, and stable text.
- required_facts lists normalized phrases that must appear in both the answer and retrieved
  evidence.
- reference_answer documents a human-approved exemplar; deterministic scoring uses required_facts.
- document_ids optionally scopes a case to indexed documents. New document IDs equal the raw
  source SHA-256.
- generation_method preserves whether a human, local model, or observed failure proposed the case.
- review_status must be approved_gold before the release evaluator will accept the case.

For an unanswerable case, use empty evidence/facts and a null reference answer:

~~~json
{"case_id":"policy-absent-001","dataset_version":"1.0.0","split":"holdout","corpus_id":"policy-demo","corpus_version":"2026-08","question":"Who approved this policy?","question_type":"unanswerable","answerable":false,"expected_evidence":[],"required_facts":[],"reference_answer":null,"generation_method":"human","review_status":"approved_gold","tags":["unanswerable"]}
~~~

Chunk IDs are deliberately excluded because parser and chunk-size changes would invalidate them.
Review or migrate evidence anchors when the source document itself changes.

## Run it

Start the application, index the matching frozen corpus, then run:

~~~powershell
uv run local-rag-eval evaluation/private/cases.jsonl
~~~

Use --base-url for another local port. The command prints JSON and exits with status 1 if a quality
gate is missed. Run uv run local-rag-eval --help to see configurable thresholds.

## Metrics and initial gates

| Layer | Metric | Default gate | Meaning |
|---|---|---:|---|
| Retrieval | Hit@k | >= 0.85 | at least one expected passage appears |
| Retrieval | Mean reciprocal rank | >= 0.70 | first expected passage ranks near the top |
| Retrieval | Evidence recall@k | >= 0.90 | expected evidence anchors are recovered |
| Retrieval | Required evidence coverage@k | >= 0.85 | all required evidence groups are represented |
| Retrieval | nDCG@k | >= 0.70 | distinct required evidence is ranked early |
| Citations | Citation precision | >= 0.90 | returned citations match annotated evidence |
| Generation | Fact coverage | >= 0.90 | required facts occur in the answer |
| Grounding | Evidence support | >= 0.90 | required facts also occur in retrieved evidence |
| Abstention | Accuracy | >= 0.90 | absent-answer cases return no citations and abstain |
| Abstention | False-abstention rate | <= 0.10 | answerable cases are not incorrectly refused |
| Runtime | p50/p95 latency | recorded | end-to-end API latency on fixed hardware |

Fact matching is a deterministic, case-insensitive phrase check, not an LLM judge. This makes
regressions reproducible but means acceptable paraphrases must be represented by suitable required
phrases or reviewed manually. The gates are initial portfolio targets, not universal production
guarantees.

## Change protocol

1. Freeze the corpus, source hashes, evaluation cases, model tags, and retrieval configuration.
2. Record dataset, corpus, extraction, index, and model versions with the baseline output.
3. Run the same approved cases against the proposed parser, chunking, retrieval, prompt, or model
   change.
4. Compare retrieval metrics before judging generated prose and inspect every abstention regression.
5. Keep a development split for tuning and a locked holdout for release comparisons.
6. Add local judges, neural rerankers, or larger frameworks only when measured gains justify their
   latency and maintenance cost.
