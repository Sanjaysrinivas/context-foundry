# Public document for the Jev experiment

Checked 2026-10-01 against original publisher downloads. These are development candidates, not an approved evaluation corpus.

Use **NIST FIPS 197, updated May 9, 2023**, as the first document. It is small, contains selectable text, and gives reproducible distinctions between keys, blocks, rounds, requirements, and optional behavior. NIST says the 2023 update made no technical algorithm changes. [Publication record](https://csrc.nist.gov/pubs/fips/197/final), [original PDF](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf).

| Original document | Measured bytes | PDF pages | Extracted text characters | Assessment |
|---|---:|---:|---:|---|
| [FIPS 197-upd1](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf) | 1,184,436 | 46 | 70,685 | Best first choice: precise and stable. |
| [Cassini: End of Mission, September 2017](https://solarsystem.nasa.gov/system/downloadable_items/934_cassiniEndofMission.pdf) | 8,146,778 | 24 | 61,589 | Good second choice: readable narrative, numerical facts, explanations, FAQ. |
| [Artemis I press kit](https://www.nasa.gov/wp-content/uploads/2026/01/artemis-i-press-kit.pdf) | 11,908,670 | 41 | 65,927 | Exceeds the application's 10 MiB upload limit. |

Sizes and extraction counts above were measured by downloading the linked originals with HTTPX and reading their text with PyMuPDF; they are observations of these particular downloads. No PDF was compressed or rewritten. The Artemis archive URL returned the same byte count and extraction counts. [Archived PDF](https://www3.nasa.gov/specials/artemis-i-press-kit/img/Artemis%20I_Press%20Kit.pdf).

## First checks on FIPS 197

Use only this document when asking the questions. The checks below are agent-drafted development cases, not human-approved gold labels. Page references use one-based PDF page numbers; the numbered main text has an offset of eight pages from its printed page numbers.

| Question or deliberately submitted claim | Expected reading | Source location |
|---|---|---|
| What block size does AES-256 use? | 128 bits; do not confuse key size with block size. | PDF page 9, printed page 1. |
| How many rounds do AES-128, AES-192, and AES-256 use? | 10, 12, and 14 respectively. | PDF page 19, printed page 11, Table 3. |
| Must an implementation support all three key lengths? | At least one is required; two or three are optional. | PDF page 34, printed page 26, section 6.1. |
| “Implementations must support all three key lengths.” | Contradicted by section 6.1. | Same section. |
| “A future revision could extend the allowed parameters.” | Supported; this is a possibility, not an adopted change. | Section 6.3, same page. |
| What throughput in MB/s does the document guarantee? | No guaranteed rate is supplied; avoid inventing a benchmark. | Whole-document absence check. |

These expectations come from the [original specification](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf). The specification points elsewhere for modes of operation; a reference to another publication does not supply that other publication's contents. NIST permits copying unmarked public information with appropriate attribution. [Copyright policy](https://www.nist.gov/copyrights-disclaimers).

## What this experiment can reveal

The round table extracts as column-oriented text. If a generated answer swaps the rows, record whether the problem arose during PDF extraction, retrieval, generation, or the Jev assessment. Visually check Table 3 before treating extracted strings as gold evidence.

A fact can be correct in the full PDF and still lack support in its actual cited excerpt. Jev receives the cited excerpts, so assess its judgment against those excerpts. Use a correct claim paired with an unrelated passage as a separate wrong-citation test.

For absent answers, establish absence by checking the document; do not infer absence from failed retrieval. Successful abstention produces no factual claim to audit. For deliberate errors, replay source-derived claims through the evaluation CLI rather than expecting the local model to make the particular error on demand.

## Other candidates and temporal pitfalls

Cassini is a useful follow-up because it describes mission discoveries and the rationale for ending the mission. It is written before the final plunge: its September 15, 2017 encounter is future tense. Keep planned events distinct from measured outcomes, and treat its contact and broadcast details as historical. [Cassini original PDF](https://solarsystem.nasa.gov/system/downloadable_items/934_cassiniEndofMission.pdf).

Artemis I has an updated HTML version containing actual mission duration and launch details. Do not use that page as answer truth for an older, prospective PDF. [NASA HTML press kit](https://www.nasa.gov/specials/artemis-i-press-kit/).

The [Mars 2020 launch kit](https://www.jpl.nasa.gov/news/press_kits/mars_2020/download/mars_2020_launch_press_kit.pdf) was also inspected: 16,127,366 bytes and 63 pages, above the upload limit. The [2017 NIST SP 800-63B PDF](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf) is small and searchable but expressly withdrawn and superseded. It would require historical-policy framing, making FIPS 197 a simpler first experiment.

FIPS 197 download SHA-256: `62c86eb567f13edb8f71826e985da870b04ef6381634f303cdb16e84d47becd1`.

Cassini download SHA-256: `c78ad478ba6bedb7fb31a7c18796e2a063a514537b4d57d5077c40a754708469`.

## Downloaded and tested in the application

The original FIPS PDF is saved locally at
`evaluation/results/jev-documents/NIST.FIPS.197-upd1.pdf`. It was uploaded to the isolated app at
`http://127.0.0.1:8767`, which indexed all 46 pages into 154 chunks in 33.31 seconds. Select only
`NIST.FIPS.197-upd1.pdf` when repeating the questions; the earlier synthetic policy is also indexed.

Five real application queries returned HTTP 200, with live Jev observation enabled. The local
chat model was `context-foundry-jev-smoke`, using the `llama3.2:3b` weights and a 4096-token
context; embeddings used `embeddinggemma`. These observations are an exploratory run, not
approved-gold release metrics. The original PDF and application source were not rewritten.

| Question | Expected reading | Observed answer and Jev judgment |
|---|---|---|
| What is the block size of AES-256? | 128 bits, equivalent to 16 bytes. | Answered 16 bytes, but cited a passage preceding the table rather than its values. Jev assigned 0.87 insufficient evidence and suggested review. |
| How many rounds do AES-128, AES-192, and AES-256 use? | 10, 12, and 14 respectively, Table 3. | Answered only AES-128: 10 rounds. Jev correctly supported that cited claim, but did not flag the missing answers. |
| Must an implementation support all three AES key lengths? | Section 6.1 requires at least one; two or three are optional. | Returned six statements from the document outline. Jev supported the statements, but the requirement question was unanswered. |
| Which transformation is omitted in the final CIPHER round? | MIXCOLUMNS(), section 5.1. | Answered correctly with a supporting page-20 citation; Jev assigned support 1.00 and suggested keep. |
| What guaranteed encryption throughput in MB/s does FIPS 197 specify? | No guaranteed rate is stated. | Returned only the fragment “FIPS 197” rather than an abstention. Jev assigned support 0.91 to the cited fragment and suggested keep. |

All ten claim audits completed: nine keep suggestions and one review. End-to-end query times
were 0.98–5.70 seconds. The full responses, actual cited excerpts, probabilities, and runtime
settings are saved in `evaluation/results/jev-documents/nist-aes-pipeline.json`.

This is a useful test document because it exposes separate issues. A true answer can cite the
wrong chunk; an accurate cited claim can omit other requested facts; a source-supported outline
can fail to answer a requirement question. The current Jev prompt checks claim support and does
not assess answer completeness, question relevance, or whether a fragment is a useful answer.
Keep those judgments separate when testing. No prompt or application behavior was changed to
make this exploratory run pass.

## Subsequent citation repair experiment

After this observer run, opt-in `RAG_JEV_MODE=repair` was added. A live query of the same PDF
returned the same 16-byte block-size claim with citation `[2]` instead of `[1]`. Jev gave the
replacement passage support 0.97; the details retain the original insufficient-evidence
assessment and the replacement check. The retrieval ledger was unchanged. The MIXCOLUMNS
control remained on its existing supporting citation with no replacement attempts.

This repairs the demonstrated citation gap. It does not resolve the incomplete round answer,
irrelevant outline, or fragmentary absent-throughput response reported above. See
[the repair policy and measured checks](jev-testing.md) for the bounded search, failure behavior,
and additional synthetic negative test.
