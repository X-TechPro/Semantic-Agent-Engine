# Experiments (frozen)

One-off measurements that produced the numbers quoted in `docs/`. They are kept
for provenance, not shipped: nothing in `src/` imports them, and none of them is
a Make target.

Run one with the product on the path and the folder on `sys.path` (Python does
that automatically when you call the script directly):

```bash
PYTHONPATH=src .venv/bin/python experiments/late_chunk.py --help
```

The shipped measurement entry points stay in `src/engine/eval/`:
`meetings_eval`, `steps_eval`, `step7_debug`, `rerank_qwen`.

| file | what it measured / why it is here |
|---|---|
| `jina_ab`, `late_chunk`, `mlm_link`, `pretrained_link`, `ssm_link`, `ssm_model`, `jina_tags`, `laya_real` | encoder and linker ablations behind `docs/EMBEDDING_AB.md` |
| `continuity_stages*`, `relation_graph_real`, `cross_encoder_*`, `candidate_generation*`, `ce_finetune_relation`, `reranker_*` | cross-encoder and candidate-generation attempts on real pairs |
| `flaubert_finetune_*`, `change_head`, `change_ranker` | the FlauBERT relation head that lost to the deterministic linker |
| `event_*`, `n2_model_real`, `pretrained_event_real`, `summre_real_temporal`, `v4_real`, `v4_fixed` | SUMM-RE real-data probes, kept as the record of why that corpus was dropped |
| `decision_nodes`, `node_view_gate`, `state_nodes_gate`, `linker_local`, `linker_probe`, `currency_guard`, `token_attention*` | v3 node/state/attention gates |
