"""Linker training inputs (6.8 baseline (b)) and code chunking (6.1).

`link/train.py` was 116 statements at 0% coverage. Two of its behaviours are worth
pinning, and both were measured rather than assumed:

* `harvest()` iterated every unit against every other unit *and* emitted both
  orders at each step, so each ordered pair was produced twice. `precision_recall`
  survived that (identical rows scale tp/fp/fn uniformly) but the reported counts
  did not: `runs/linker_lr/result.json` said 5096 pairs / 240 positives where the
  distinct numbers are 2548 / 120.
* the module's own docstring records writing the dev-chosen threshold into a key
  inference never read. Because sigmoid is strictly monotonic,
  `decision_function >= b` is exactly `score >= sigmoid(b)`, so the operating point
  has to live in the one field `PairClassifier` consults.

`train()` itself is NOT called here: it writes `runs/linker_lr/result.json` and
`models/linker_lr.json`, so running it from the suite would overwrite the shipped
artefact with a toy-corpus result. Its invariance under the dedupe was checked by
re-running the real 20-project training instead (see DECISIONS).

`segment/code_chunker.py`: `_symbol_of` used to look for the definition name
*after* the regex match, but the match includes the name — so every code chunk
silently reported `symbol: None`, on both chunklet and regex paths. chunklet is
also not usable in this environment (it needs a `littletree` version whose API it
pins; installing 0.9.1 did not satisfy it), so the fallback is the real path and
the per-chunk `source` field is what records that.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from engine.link.model import FEATURES, PairClassifier, build_features
from engine.link.train import _matrix, harvest, precision_recall
from engine.schemas import Slots, Unit, utcnow
from engine.segment.code_chunker import _symbol_of, chunk_code

pytestmark = pytest.mark.integration


def _unit(text, uid, *, on="2026-10-12", subject=None, project="atlas",
          status="confirmed", type="fact") -> Unit:
    return Unit(unit_id=uid, source_id="meeting-01", source_type="meeting",
                source_date=date.fromisoformat(on), ingested_at=utcnow(),
                text=text, sentences=[], project=project, type=type, status=status,
                slots=Slots(subject=subject))


# ------------------------------------------------------------- pair duplication

@pytest.fixture(scope="module")
def harvested():
    """A real ingest, because `harvest` reads stored units — a hand-made pair list
    would not prove the loop's iteration behaviour."""
    from engine.config import load_config
    from engine.engine import Engine
    from gen.ledger import build_ledger
    from gen.render import render_meeting

    import tempfile
    import pathlib

    root = pathlib.Path(tempfile.mkdtemp())
    ledger = build_ledger(n_projects=3, seed=43)
    projects = {p["project"] for p in ledger.projects}
    cfg = load_config(overrides={"store": {"path": str(root / "h.db")},
                                 "embed": {"backend": "hash"}})
    eng = Engine(cfg, path=root / "h.db", linker_enabled=False)
    for m in ledger.meetings:
        eng.ingest(render_meeting(ledger, m.meeting_id, seed=43))
    try:
        return harvest(eng, ledger, projects)
    finally:
        eng.close()


def test_each_ordered_pair_is_harvested_exactly_once(harvested):
    """The double-count, pinned: rows == distinct pairs, nothing repeated."""
    pairs = [tuple(r["pair"]) for r in harvested]
    assert pairs, "a 3-project ledger must yield candidate pairs"
    assert len(pairs) == len(set(pairs)), (
        f"{len(pairs) - len(set(pairs))} duplicated ordered pairs — harvest() is "
        f"emitting each pair twice again")


def test_both_directions_are_present_for_a_pair(harvested):
    """(later, earlier) can be a supersede and (earlier, later) must not be, so the
    reversed row is the cheapest evidence the model learned date order."""
    pairs = {tuple(r["pair"]) for r in harvested}
    a, b = next(iter(pairs))
    assert (b, a) in pairs, "reversed direction missing: date order is unlearnable"


def test_harvest_gold_labels_are_real_relations(harvested):
    assert {r["gold"] for r in harvested} <= {"supersedes", "supports", "contradicts",
                                              "creates_action", "answers", "unrelated"}


def test_harvest_rows_are_wide_as_features_are_long(harvested):
    X, y = _matrix(harvested)
    assert X.shape == (len(harvested), len(FEATURES))
    assert y.dtype == bool and y.sum() >= 5, "need positives to train at all"


# ------------------------------------------------------------ precision_recall

def test_precision_recall_covers_the_degenerate_cases_without_dividing_by_zero():
    pos = np.array([True, True, False, False])
    assert precision_recall(pos, np.array([True, True, False, False])) == (1.0, 1.0, 1.0)
    assert precision_recall(pos, np.array([True] * 4))[:2] == (0.5, 1.0)
    assert precision_recall(pos, np.array([False] * 4)) == (0.0, 0.0, 0.0)
    assert precision_recall(np.array([False] * 4), np.array([False] * 4)) == (0.0, 0.0, 0.0)


def test_the_metric_is_invariant_under_duplication_even_though_the_fit_is_not():
    """Scoped deliberately narrowly.

    Duplicating rows leaves tp/fp/fn ratios alone, so a metric computed over
    already-fixed predictions is invariant — that much is true, and it is all this
    test claims. It is NOT evidence that training on duplicated rows is harmless:
    fitted side by side on identical data, duplication moved the coefficients by up
    to 1.541 and held-out F1 from 0.3487 to 0.3299, because the rows count twice
    against the fixed L2 penalty. `test_each_ordered_pair_is_harvested_exactly_once`
    is what guards the real defect.
    """
    pos = np.array([True, True, False, False, False])
    pred = np.array([True, False, True, False, False])
    dup = np.concatenate([pos, pos]), np.concatenate([pred, pred])
    assert precision_recall(pos, pred) == precision_recall(*dup)


# ------------------------------------------------- operating point round-trip

def test_saved_threshold_is_the_one_the_classifier_uses(tmp_path):
    """`load` must hand back the same operating point `save` wrote, and it must be
    expressed as sigmoid(dev_bias) — the value inference actually consults."""
    bias = 1.25
    clf = PairClassifier([0.5 if f == "same_subject" else 0.0 for f in FEATURES],
                         bias=-0.2, threshold=1.0 / (1.0 + math.exp(-bias)),
                         meta={"dev_bias": bias})
    path = clf.save(tmp_path / "linker_lr.json")
    again = PairClassifier.load(path)
    assert again is not None
    assert again.threshold == clf.threshold == pytest.approx(0.7773, abs=1e-4)
    assert again.meta["dev_bias"] == bias
    old = _unit("Le budget est de 1,8 M€.", "u-old", on="2026-10-01",
                subject="atlas.budget", status="confirmed")
    new = _unit("Le budget est validé à 2,4 M€.", "u-new", subject="atlas.budget")
    assert again.score(new, old) == pytest.approx(clf.score(new, old))
    assert again.proposes_supersede(new, old) == clf.proposes_supersede(new, old)


def test_features_used_by_training_match_the_model_module():
    """`_matrix` reads `features[f]`; a key training and inference disagree on
    becomes a silent 0.0 column."""
    row = build_features(_unit("a", "n", subject="atlas.budget"),
                         _unit("b", "o", subject="atlas.budget", on="2026-10-01"))
    assert set(row) == set(FEATURES)
    assert all(isinstance(v, float) for v in row.values())


# ---------------------------------------------------------------- code chunker

PY_SRC = '''"""Module docstring."""


def alpha(a, b):
    return a + b


class Beta:
    """Doc."""

    def gamma(self):
        return 3
'''


@pytest.mark.parametrize("text,want", [
    ("def compute_total(x):\n    ...", "compute_total"),
    ("class Loader:\n    pass", "Loader"),
    ("async def fetch():\n    ...", "fetch"),
    ("func delta() {}", "delta"),
    ("x = 1\nreturn x", None),
])
def test_symbol_of_reads_the_declared_name(text, want):
    """The name is INSIDE `_DEF_RE`'s match, so anything looking after the match
    finds the parameter list. This returned None for every real input before."""
    assert _symbol_of(text) == want


def test_chunk_code_keeps_symbols_on_the_path_that_actually_runs():
    chunks = chunk_code(PY_SRC, language="python", max_tokens=64)
    assert chunks, "a real python file must chunk"
    assert all(text.strip() for text, _m in chunks)
    symbols = {m.get("symbol") for _t, m in chunks if m.get("symbol")}
    assert symbols & {"alpha", "Beta", "gamma"}, f"no symbol survived: {symbols}"
    # provenance of the engine: chunklet is unavailable in this environment, and
    # `source` is what records which path produced each chunk
    assert {m.get("source") for _t, m in chunks} & {
        "chunklet", "regex", "regex-window"}


def test_chunk_code_on_empty_input_is_empty_not_error():
    assert chunk_code("", "python") == []
    assert chunk_code("   \n  ", None) == []


def test_fallback_windows_stay_contiguous_and_invented_free():
    """Oversize blocks are windowed by line. If a window dropped or invented a
    line, a citation into a later window would point at the wrong bytes."""
    code = "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(40))
    chunks = chunk_code(code, "python", max_tokens=24)
    rebuilt = "\n".join(t for t, _m in chunks)
    for i in (0, 20, 39):
        assert f"def f{i}(x)" in rebuilt, f"function {i} dropped"
    src_lines = set(code.split("\n"))
    out_lines = {ln for t, _m in chunks for ln in t.split("\n")}
    assert out_lines <= src_lines, "chunker emitted a line the source lacks"
