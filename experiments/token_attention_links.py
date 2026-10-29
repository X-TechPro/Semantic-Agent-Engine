"""Token-level link probe for the owner's anaphora example (D-048).

The corpus test (`engine.eval.token_attention`) scores aggregate signals; this
tool records the qualitative link that motivated it, so the report can cite a
number that regenerates: for "C'est bon je les ai acheté" against "Quelqu'un veut
manger des pizzas ?", which earlier word does "les" attend to, at which layer,
and by how much over the runner-up.

No model is trained and nothing is wired. The sealed holdout is untouched.

Run: PYTHONPATH=src python -m engine.eval.token_attention_links [--layer 19]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

import numpy as np

from engine.config import repo_root
from engine.eval.token_attention import BASE, PIZZA, _words

QUERY_WORD = "les"
TARGET_WORD = "pizzas"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE, attn_implementation="eager").eval()
    if args.device != "cpu":
        model.to(args.device)
    old, new = PIZZA
    o = tok(old, add_special_tokens=False)["input_ids"]
    n = tok(new, add_special_tokens=False)["input_ids"]
    ids = [tok.cls_token_id] + o + [tok.sep_token_id] + n + [tok.sep_token_id]
    oi = list(range(1, 1 + len(o)))
    ni = list(range(2 + len(o), 2 + len(o) + len(n)))
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids)),
                    output_attentions=True)
    A = out.attentions[args.layer][0].mean(0).double().numpy()
    o_spans = _words(tok, [ids[i] for i in oi])
    n_spans = _words(tok, [ids[i] for i in ni])
    o_full = [list(range(oi[0] + s, oi[0] + e)) for s, e, _ in o_spans]
    n_full = [list(range(ni[0] + s, ni[0] + e)) for s, e, _ in n_spans]
    M = np.zeros((len(n_spans), len(o_spans)))
    for a in range(len(n_spans)):
        for b in range(len(o_spans)):
            M[a, b] = A[np.ix_(n_full[a], o_full[b])].sum() / max(1, len(n_full[a]))
    words_new = [w for _s, _e, w in n_spans]
    words_old = [w for _s, _e, w in o_spans]
    if QUERY_WORD not in words_new or TARGET_WORD not in words_old:
        raise SystemExit(f"expected {QUERY_WORD!r} -> {TARGET_WORD!r} in the tokens; "
                         f"got {words_new} -> {words_old}")
    i = words_new.index(QUERY_WORD)
    j = words_old.index(TARGET_WORD)
    order = np.argsort(-M[i])
    top = [{"word": words_old[k], "mass": round(float(M[i, k]), 4)}
           for k in order[:4]]
    a, b = np.unravel_index(M.argmax(), M.shape)
    out_doc = {
        "date": date.today().isoformat(), "layer": args.layer,
        "old": old, "new": new,
        "new_words": words_new, "old_words": words_old,
        "query_word": QUERY_WORD, "target_word": TARGET_WORD,
        "link_mass": round(float(M[i, j]), 4),
        "link_is_top": words_old[int(order[0])] == TARGET_WORD,
        "runner_up": top[1] if len(top) > 1 else None,
        "top_targets_for_query": top,
        "global_argmax": {"new_word": words_new[a], "old_word": words_old[b],
                          "mass": round(float(M[a, b]), 4)},
        "note": ("the aggregate max at this layer is a different pair; the "
                 "anaphora is read from the query word's own row"),
    }
    dest = repo_root() / "runs" / "token_attention"
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"links-{out_doc['date']}.json"
    p.write_text(json.dumps(out_doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out_doc, indent=1, ensure_ascii=False))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
