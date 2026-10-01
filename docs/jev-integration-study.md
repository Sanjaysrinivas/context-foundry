# Jev in Context Foundry: an observable citation auditor

Reviewed 2026-10-01 against repository commit `cb706b4` and public TypeSafe documentation.
The observer proposed here and subsequent opt-in citation repair are implemented on
`feature/jev-observer`, branched from `dev`. Repair verifies individual alternative retrieved
passages before changing a citation; it preserves claim text and records the original assessment.
See [measured live tests and reproduction instructions](jev-testing.md). The accompanying
[interactive simulator](jev-experiment.html) still uses invented probabilities. Live testing uses
public synthetic documents; no private-corpus evaluation has been performed.
The [API research](jev-api-research.md) records the verified contract and its limits.

## Recommendation

Start with **an observer-only auditor of each generated claim and its actual citations**.
Keep Ollama as the writer, Qdrant/BM25 as retrieval, and Python as the decision maker.
This tests a specific weakness already reproduced in this repository, makes the result visible,
and needs no new database, orchestration graph, or model framework.

Jev's three primitives are Choice, Score, and Noul. For this experiment, use Choice to distinguish
support, contradiction, and insufficient evidence. Jev supplies the distribution; an explicit
application policy supplies the action. Its citation cookbook already uses this pattern.
[Primitives](https://docs.typesafe.ai/primitives),
[citation recipe](https://docs.typesafe.ai/cookbooks/citation_check)

The decision to start with observation is mine, following the user's preference to choose the
experiment. It is not an architecture decision to make cloud inference mandatory.

## What the repository actually does

The source is a small, inspectable RAG system with useful seams:

1. [documents.py](../src/local_rag/documents.py) hashes source bytes, extracts/OCRs PDFs, cleans
   Markdown, and creates deterministic overlapping chunks.
2. [store.py](../src/local_rag/store.py) filters selected document IDs, combines cosine and
   BM25Plus rankings with normalized reciprocal-rank fusion, and returns a shortlist.
3. [service.py](../src/local_rag/service.py) splits clear compound questions into at most three
   queries. Each retrieves four passages by default, or eight for exhaustive requests.
4. `_select_generation_context()` keeps the first match when it covers at least 60% of query
   terms; otherwise it selects passages adding two new terms. Exhaustive requests select one
   direct definition or the passage containing the most structured items.
5. [providers.py](../src/local_rag/providers.py) either extracts explicit exhaustive lists or
   requests a structured answer. It validates the schema/citation bounds, applies a lexical
   grounding check, retries invalid drafts once, and can fall back to extraction with one source.
6. `RAGService.ask()` maps local citation numbers to global ledger numbers and renders claims.
   It retains other retrieved passages in the ledger even when they were excluded from generation.
7. [api.py](../src/local_rag/api.py) renders safe Markdown and returns answer text plus the ledger.
   The [browser](../src/local_rag/web/index.html) shows each passage and its retrieval match score.

Existing [evaluation](../src/local_rag/evaluation.py), [silver review](../src/local_rag/evaluation_data.py),
and [Ragas diagnostics](../src/local_rag/ragas_evaluation.py) already cover most of the experiment's
measurement and review needs. Reuse them.

## Reproduced gaps, rather than speculative use cases

These checks used synthetic text and the real existing functions; they did not call a model:

| Probe | Existing behavior | What a Jev audit would test |
| --- | --- | --- |
| Claim removes “not” from otherwise identical source text | Lexical validator accepts it | Does the cited source contradict the assertion? |
| Claim cites passage 1, but only passage 2 contains the answer | Lexical validator accepts it when both are in context | Does the claim's actual cited set establish it? |
| “Alice approved it” when no approver is recorded | Lexical validator accepts it | Is the named person established by evidence? |
| Top passage discusses finding the cancellation window; lower-ranked passage says “thirty days” | Context selector sends only the top passage | Which passage actually contains answer evidence? |

The first three arise because `_validate_claim_grounding()` compares terms against the entire
context and skips the overlap requirement for claims with fewer than six unique terms.
Valid JSON and citation bounds alone do not resolve these cases. The fourth makes semantic
evidence selection a worthwhile second experiment.

These are demonstrations of what the current checks admit, not measured rates of bad live answers.
The existing quality checks passed: Ruff, mypy, and 52 tests, with 84.67% coverage.

## Ranked integration opportunities

| Priority | Integration | Exact location | Outcome and what the user sees |
| --- | --- | --- | --- |
| First | Claim/citation audit | `ask()`, immediately after `chat_provider.answer()`, before rendering | Per-claim support/contradiction/missing-evidence judgments beside the numbered citations |
| Second | Semantic evidence selection plus failure diagnosis | After `_search_groups()`, around `_select_generation_context()` | Why a passage was used or omitted; whether omitted retrieved evidence could answer a refused part |
| Later | Review-queue prioritization | `evaluation_data.py`, before source-first human review | Which silver cases appear ambiguous or unsupported, without promoting them to gold |

A universal answer-quality score would obscure the failure we need to inspect. A narrow judgment
with identifiable evidence is more useful for this project.

## The first experiment, end to end

~~~mermaid
flowchart LR
    R[Local hybrid retrieval] --> S[Select generation evidence]
    S --> O[Ollama structured draft]
    O --> V[Existing schema and lexical checks]
    V --> J[Jev: claim plus its cited excerpts]
    J --> P[Python policy]
    P --> T[Per-claim audit trace]
    V --> A[Baseline answer]
    A --> U[Answer and citation ledger]
    T --> U
~~~

The initial trace is observational; it does not alter the baseline answer.

At the proposed seam, `draft.claims`, `scoped`, and the citation mapping already exist.
Construct an audit record from each structured claim. Do not recover claims by parsing the final
Markdown. Resolve each claim's local citation IDs against `scoped` in code, and preserve their
global IDs for display. Pass the same transformed excerpt text used for generation
(`_context_text()`) to the auditor and record its digest; the ledger can still show the original.

Assess all excerpts cited by a claim **collectively**. Two complementary citations may support a
claim together even when neither proves it alone. A joint-support judgment does not establish that
every attached citation is individually necessary; checking redundant citations is a later scope.

Use one compact state per distinct cited excerpt set. Batch claims sharing that exact set if useful.
Every question in a Jev call sees the entire state; instruction-only paths do not hide uncited
passages. The [API research](jev-api-research.md) includes a proposed REST request and an explicitly
labelled shared-state alternative. This grouping is our scoping policy.

The auditor sees drafts that survived existing checks, including extractive drafts. It does not
repair paraphrases rejected inside `_grounded_answer()`, or detect omissions where no claim was
produced. Those are different experiments.

## How code would decide the outcome

Define one Choice question: does the provided cited set establish the complete claim, establish
its contradiction, or leave it unestablished? Use explicit criteria for negation, named entities,
quantities, and conditions. Include ambiguous or conflicting evidence in the insufficient category;
do not force it into support.

This example policy matches the simulator:

~~~python
def proposed_action(probabilities, threshold=0.90):
    if probabilities["supported"] >= threshold:
        return "keep"
    if max(probabilities["contradicted"], probabilities["insufficient"]) >= threshold:
        return "withhold"
    return "review"
~~~

The function assumes a validated three-option distribution. A future adapter must check finite
numbers in [0, 1], the exact option keys, a sum approximately equal to 1, matching answer IDs/types,
and the returned model. API failures produce `unavailable`, never a fabricated zero or a pass.

**0.90 is an illustrative threshold, not a validated operating point.** Tune on reviewed development
claims, then freeze it before holdout evaluation. Separate thresholds for accepting and withholding
are justified only if the measured error costs require them.

Choice confidence summarizes the distribution and is not an independent correctness estimate.
The policy above reads option probabilities directly. Noul is useful for a later atomic coverage
question, but returns only a yes probability, without a separate confidence field.
[Confidence](https://docs.typesafe.ai/confidence),
[Noul](https://docs.typesafe.ai/primitives/noul)

| Action | Observer experiment | Possible later enforcement |
| --- | --- | --- |
| Keep | Show the baseline claim and its support estimate | Publish the supported claim |
| Withhold | Show a clear concern beside the unchanged baseline | Omit the unverified claim and explicitly identify the unsupported requested part |
| Review | Show uncertainty | Recheck with relevant evidence once, then extract a supported answer or abstain |
| Unavailable | Show “Audit unavailable”; keep baseline behavior | Withhold unaudited claims or explicitly disclose baseline fallback, according to the chosen mode |

Any enforcement must report partial outcomes truthfully. Removing one leaf from an “every item”
request makes the answer incomplete; it cannot retain a claim of completeness. Do not average
probabilities into an answer-wide “verified” badge or multiply them as if errors were independent.
Jev provides no generated explanation: the application derives short reason labels from the
distribution and rule that fired.

## How we could see it in the evidence desk

Add a small audit panel under the answer, using the existing numbered ledger:

~~~text
Citation audit · Observer mode
Claim: “The cancellation window is thirty days after submission.” [1]
Support 2% · Contradiction 3% · Insufficient evidence 95%
Suggested action: Withhold
Reason: The cited passage does not establish this claim.
Baseline answer unchanged.
~~~

These numbers are fixtures. The [interactive page](jev-experiment.html) lets you change the example,
threshold, and filtering preview. It includes joint evidence and provider unavailability.

In a live trace, expose claim ID/text, actual global citation IDs, probabilities, raw model
confidence, selected action, audit status, returned model, policy/prompt versions, latency, and
token usage. Reuse source hashes/page numbers and record the exact excerpt digest locally.
Keep retrieval “Match” and semantic support separately labelled: the current normalized RRF
score is a ranking signal, not an answer probability.

Default to one audit result per claim, with details expandable. Keep cloud-call details inside
the experimental inspector. The ordinary user needs the evidence and any uncertainty.

## The second experiment: locate where evidence was lost

Compare three distinct sets: **retrieved**, **selected for generation**, and **actually cited**.
They already exist in `ask()`; expose their membership rather than merging their meaning.

For a refused question part, ask an atomic Noul about whether the selected set contains explicit
answer evidence, and another about the full retrieved shortlist. For a flagged claim, audit the
cited set first, and only then test the broader set as a separate diagnostic request.

| Diagnostic pattern | Likely location | Bounded next action |
| --- | --- | --- |
| Cited set lacks support; another retrieved source supports the draft claim | Citation assignment | Regenerate with the actual supporting source; verify again |
| Selected set lacks answer evidence; omitted retrieved passage has it | Context selection | Add that passage and regenerate once |
| Selected set contains answer evidence; generated answer refuses | Generation | Retry once with that evidence; compare to extraction |
| Retrieved shortlist lacks explicit evidence | Retrieval or genuinely absent answer | Observe the limitation; test a larger shortlist on reviewed cases before changing search |
| Sources explicitly disagree | Source conflict | Present the disagreement with citations |

These labels are hypotheses for inspection, not proof of root cause. A larger shortlist cannot
guarantee that a document contains no answer. A grounded claim can also be irrelevant to the
question, so question coverage needs its own test.

For passage selection, start with the existing shortlist and ask whether each passage contains
usable answer evidence. Retain complementary and contradictory passages; preserve the ledger.
Do not automatically remove uncertain passages merely for being uncertain.
TypeSafe's official RAG recipe demonstrates separate passage judgments, including conflict.
[RAG passage recipe](https://docs.typesafe.ai/cookbooks/classifying_rag_passages)

This second step could help compound or exhaustive questions that need multiple passages.
Keep deterministic extraction when an explicit relevant list is already sufficient. Count items,
validate source IDs, and compare quantities in code; Jev's current limitations include counting
and numeric precision. [Model limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

## How we would know it helps

Use the repository's approved dataset and source-first review. Start with a smoke set spanning
direct facts, negation, wrong citations, absent entities, paraphrases, combined evidence, exhaustive
lists, source conflict, OCR, and attempted instruction injection. Add enough reviewed examples
and repeated failures to estimate accuracy; a few smoke examples cannot validate “90%”.

The first experiment needs **claim-level reviewed labels** alongside existing question-level gold.
Label whether each claim's cited set supports, contradicts, or leaves it unestablished. Compare:

- Existing checks alone, and the same surviving drafts plus Jev.
- Fraction of actually unsupported claims accepted, and fraction of supported claims withheld.
- Accuracy versus fraction of claims handled automatically as the threshold changes.
- Probability calibration on held-out claims, latency p50/p95, cost, and unavailable-audit rate.

Keep Jev's judgments separate from gold labels. They may prioritize the silver review queue but
cannot approve their own labels. Freeze corpus/source hashes, generator/model tags, prompts,
policy version, and Jev version; use identical captured drafts for the paired auditor comparison.
For later enforcement, rerun the same question cases and inspect answer coverage and false refusals.
Pin the model rather than silently comparing different `jev-latest` versions.

Two existing metric details matter before enforcement:

- `citation_precision` measures relevant **returned ledger passages**, and `evidence_support`
  checks required phrases anywhere in that ledger. Neither checks the claim-to-citation relation.
  A generation selector change can leave `/api/retrieve` and retrieval metrics unchanged; add
  selected-context evidence-group coverage for that experiment.
- An unanswerable case counts as a correct abstention only when it has no citations. A real
  abstention with preserved retrieved passages scored 0% abstention accuracy in a synthetic probe.
  Also, any partial “Insufficient evidence” phrase makes an answerable case count as a false
  abstention. Add explicit per-part outcomes for the experiment and report them beside the
  existing gates, rather than quietly changing what the old metrics mean.

Promote enforcement only if paired held-out review shows fewer unsupported accepted claims while
preserving an acceptable share of correct answers and meeting the agreed latency budget.
Keeping Jev diagnostic-only is a successful experiment if it helps identify failures.

## Minimal implementation scope and boundaries

The first live slice would use one concrete HTTPX adapter in `providers.py` and one optional audit
call at the existing service seam. Add small audit domain/API records and an expandable UI panel.
Use `off` as the default and `observe` as the initial opt-in mode. Build the adapter only when
enabled; no Jev key should be required to run the existing local application.

Use separate server-side Jev credentials and a short bounded timeout. Account for total audit
latency across several distinct evidence sets, not just one call; start sequential on the smoke
set, measure, and bound concurrency only if needed. Keep provider errors distinct from verdicts.
Do not reuse a transport that exposes remote error bodies to the browser without scrubbing them.

HTTPX is already installed. No SDK, new provider framework, cloud database, agents, or streaming
system is needed. Keep the existing chat/embedding interfaces independent; Jev cannot satisfy
the current text-producing `ChatProvider` contract.

Enabling Jev sends claim text and cited excerpts to a hosted service. Start with public or synthetic
documents. Ordinary privacy disclosures must reflect that change; the local default remains free
and private. The API study records US hosting and the retention terms.

At the reviewed price, 10,000 input tokens cost approximately $0.00042 per audit; retries and
separate states add cost. Measure all input tokens actually billed rather than estimating from
document characters. Published speed figures do not establish end-to-end improvement here.
[Model pricing](https://docs.typesafe.ai/models)

For any implementation, leave one deterministic mocked check covering the failure boundary:
uncited evidence cannot authorize a claim; provider unavailability cannot become “verified”;
observer mode preserves the answer. Run the repository gates and matching approved corpus.

For now, the complete deliverable is this study, cited API research, and a working fixture
simulator. The simulator runs ten embedded policy checks. The remaining unknown is empirical:
how well Jev judges this corpus. A live, reviewed replay is the smallest next experiment that
can settle that question.
