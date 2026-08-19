# RAG evaluation plan

The automated suite proves pipeline behavior; it does not prove answer quality. Quality should be measured against representative documents before changing chunking, retrieval, prompts, or models.

## Evaluation set

Create a private JSON Lines file outside Git with at least 30 cases spanning direct facts, facts distributed across passages, ambiguous questions, and questions whose answer is absent.

```json
{"question":"What is the cancellation window?","expected_source":"policy.pdf","expected_page":4,"reference_answer":"30 days"}
{"question":"Does the policy guarantee refunds after 30 days?","expected_source":null,"expected_page":null,"reference_answer":null}
```

Source documents and evaluation cases may contain private text and therefore must not be committed unless they are intentionally public fixtures.

## Baseline metrics

| Layer | Metric | Baseline target | Meaning |
|---|---|---:|---|
| Retrieval | Hit@4 | ≥ 0.85 | expected evidence appears in the top four chunks |
| Retrieval | Mean reciprocal rank | ≥ 0.70 | relevant evidence ranks near the top |
| Generation | Citation correctness | ≥ 0.90 | cited passage supports the associated claim |
| Generation | Faithfulness | ≥ 0.90 | answer claims are supported by retrieved context |
| Abstention | Accuracy | ≥ 0.90 | unanswered questions produce an explicit insufficient-evidence response |
| Runtime | p95 latency | record locally | separates retrieval cost from model-generation cost |

Targets are initial portfolio thresholds, not universal production guarantees. Record the machine, model tag, configuration, corpus size, and date with every result.

## Change protocol

1. Freeze the evaluation set and baseline configuration.
2. Run the same questions against the current and proposed configurations.
3. Compare retrieval metrics before judging generated prose.
4. Inspect every regression, especially false answers on absent evidence.
5. Adopt a tokenizer-aware splitter, reranker, or larger model only when it improves the measured weakness enough to justify its cost.

The first likely experiment is chunk size and overlap. A reranker is not warranted until correct evidence frequently appears below the top retrieved positions.

