# MODELCARDS.md — verified model candidates (M0 gate)

Generated 2026-10-20T12:17:41Z by `python -m engine.tools.verify_models` against the live HF API (25 candidates).

Verification method: live `huggingface.co/api/models/<id>` plus `config.json` / `tokenizer_config.json` / `README.md` per repo. BUILD_GUIDE 0.5 required this pass because the guide's model names were unverified research-pass output.

| role | model | exists | licence | task | params | max len | prefixes | notes |
|---|---|---|---|---|---|---|---|---|
| embedder | `BAAI/bge-m3` | yes | mit | sentence-similarity | — | 8192 | retrieval:  | dense + sparse (ColBERT-style multi-vector) in one model; no query prefix needed |
| embedder | `intfloat/multilingual-e5-base` | yes | mit | sentence-similarity | 514/278M | 512 | query:; passage: | requires 'query: '/'passage: ' prefixes; wrong prefixes silently cost ~5-10 pts nDCG |
| embedder | `intfloat/multilingual-e5-large` | yes | mit | feature-extraction | 514/560M | 512 | query:; passage: | — |
| embedder | `Qwen/Qwen3-Embedding-0.6B` | yes | apache_2.0 | feature-extraction | 596M | 131072 | query:; Instruct: | — |
| embedder | `ibm-granite/granite-embedding-311m-multilingual-r2` | yes | apache_2.0 | feature-extraction | 312M | 32768 | retrieval:  | — |
| embedder | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | yes | apache_2.0 | sentence-similarity | 512/118M | 512 | — | — |
| embedder | `microsoft/harrier-oss-v1-0.6b` | yes | mit | feature-extraction | 596M | 131072 | query:; Instruct: | — |
| embedder | `microsoft/harrier-oss-v1-270m` | yes | mit | feature-extraction | 268M | 1000000000000000019884624838656 | query:; Instruct: | — |
| reader_encoder | `convaiinnovations/laya` | yes | apache_2.0 | text-classification | 3/421M | — | — | NOT a text encoder: typed-decision model (5 tensor inputs incl. marker_pos/qtype, no feature-extraction run). Cannot be fine-tuned as the 6.2.3 reader encoder; AutoModel.from_pretrained fails (no root config.json). Baseline card scores 0.362 vs 0.461 for always-pick-modal — worse than a constant, zero-shot.; Requires a 512-length-baked ONNX export unless torch.onnx.export(dynamo=True, opset_version=18). |
| reader_encoder | `camembert/camembert-base` | yes | ? | fill-mask | 111M | — | — | — |
| reader_encoder | `almanach/moderncamembert-base` | yes | mit | fill-mask | 136M | 8192 | — | — |
| reader_encoder | `almanach/camembertav2-base-cls` | yes | mit | text-classification | 111M | 1000000000000000019884624838656 | — | — |
| reader_encoder | `FacebookAI/xlm-roberta-base` | yes | mit | fill-mask | 279M | 512 | — | — |
| reader_encoder | `distilbert/distilbert-base-multilingual-cased` | yes | apache_2.0 | fill-mask | 135M | 512 | — | — |
| reader_encoder | `google-bert/bert-base-multilingual-cased` | yes | apache_2.0 | fill-mask | 179M | 512 | — | — |
| linker_nli | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | yes | mit | zero-shot-classification | 512/279M | 512 | — | — |
| linker_nli | `joeddav/xlm-roberta-large-xnli` | yes | mit | zero-shot-classification | 514/561M | 512 | — | — |
| linker_nli | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | yes | ? | text-classification | — | — | — | — |
| reranker | `BAAI/bge-reranker-v2-m3` | yes | apache_2.0 | text-classification | 568M | 8192 | — | — |
| reranker | `Qwen/Qwen3-Reranker-0.6B` | yes | apache_2.0 | text-ranking | 596M | 131072 | — | — |
| reranker | `jinaai/jina-reranker-v3` | yes | cc_by_nc_4.0 | text-ranking | 597M | 131072 | query:; document: | licence is NOT Apache-2.0 (v2 family was CC-BY-NC-4.0); 6.9 flags this — check before use |
| reranker | `jinaai/jina-reranker-v2-base-multilingual` | yes | cc_by_nc_4.0 | text-ranking | 278M | 1024 | query: | — |
| slots_ner | `urchade/gliner_multi-v2.1` | yes | apache_2.0 | token-classification | 289M | — | — | — |
| slots_ner | `fastino/gliner2.5-base-v1` | yes | apache_2.0 | — | 194M | 1000000000000000019884624838656 | — | — |
| slots_ner | `onnx-community/gliner_multi-v2.1` | yes | ? | — | — | 1000000000000000019884624838656 | — | — |

## Consequences for the build

- **Reader encoder:** `convaiinnovations/laya` is verified to exist but is a typed-decision model, not an encoder — it cannot initialise the 6.2.3 multi-head classifier. Primary encoder becomes a real masked-LM (CamemBERT family / XLM-R), and BUILD_GUIDE's own fallback list applies.
- **Embedder:** the guide's `harrier-oss-*` names must be confirmed below; until confirmed, `BAAI/bge-m3` and `intfloat/multilingual-e5-base` are the working candidates (both verified, both permissive).
- **Reranker:** `jinaai/jina-reranker-v3` is not usable as an Apache-2.0 substitute — `BAAI/bge-reranker-v2-m3` is the default.
- Prefix conventions are enforced in `engine/embed/hf_embedder.py` (`KNOWN_PREFIXES`), keyed by these verified ids.

## v4 addendum — FlauBERT XNLI/STS

| role | model | status |
|---|---|---|
| contextual linker / reranker | `inokufu/flaubert-base-uncased-xnli-sts` | loaded locally through `AutoModelForSequenceClassification`; 2 output logits in this checkpoint; `sacremoses` required by `FlaubertTokenizer` |

This is a **sentence-pair reranker**, not a bi-encoder and not a relation
classifier. The 2-logit head is treated as an STS-style pair score; it must be
fine-tuned with a relation/status head before it can be a shipped linker.
Licence/model-card verification is still required in the live registry before
promotion.
