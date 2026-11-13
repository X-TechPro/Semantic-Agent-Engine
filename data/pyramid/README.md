# Pyramid synthetic meetings (EN / FR)

Two hand-authored, held-out meeting transcripts used to compare Rexu v3 with
the pyramid engine and Semantic-Agent-Engine on identical gold.

- `meeting_en.json` — 206 turns, 50 queries (20 direct 1-hop, 10 owner 2-hop,
  10 supplier 3-hop, 10 phone 4-hop).
- `meeting_fr.json` — the same structure and query mix in French.
- No query or evidence turn was used to train the Rexu late head.

## Format

```json
{
  "domain": "aurora_launch_meeting",
  "turns": [
    {"idx": 0, "ts": "0:00", "speaker": "Bruno", "text": "...", "ante": null}
  ],
  "queries": [
    {"text": "...", "qtype": "owner2hop", "hops": 2,
     "evidence": [8, 13], "main": 13}
  ]
}
```

`evidence` is the set of turn indices that support the answer; `main` is the
single turn that states the answer.

## Metrics used in the comparison

- `R@1` — the `main` turn is ranked first.
- `R@k` — at least one `evidence` turn is in the top k.
- `MRR` — reciprocal rank of the `main` turn (0 if outside the ranked list).
- `ev@5` — mean fraction of all `evidence` turns in the top 5.

## Results

`results/` holds the raw runs behind the table in the main README:

- `baselines_pyramid_eval.json` — BM25, BGE-M3 dense, pyramid v3.1.
- `rexu_pyramid_eval.json` — Rexu v3 (pplx-embed-context + late head).
- `sae_pyramid_eval.json` — Semantic-Agent-Engine v4 with its Qwen3 reranker.
- `sae_pyramid_norerank.json` — the same SAE stack without the reranker.

Caveats: n = 50 per language and one meeting per language. SAE is truncated to
its top 30, so its MRR and full-recall are floors. Rexu's doc vectors are built
once per meeting and cached; doc-build time is reported separately from query
latency.

The comparison scripts live in the pyramid repository (`eval_baselines.py`,
`eval_rexu.py`, `eval_sae.py`), because they import its metric and data code.
