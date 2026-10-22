"""The NLI head end to end — load the trained checkpoint, forward real French.

This is the project's first trained component that beat the shipped baseline on
external data (DECISIONS D-022: test accuracy 0.798 vs the pair classifier's
chance-level 0.45 AUC), so the load-and-forward path itself gets a test, not just
the pure metric helpers in `test_frbench_core.py`.

Skips cleanly when the checkpoint or the pinned corpus is absent — this tier must
not require `make frdata`, but when both are present it runs the REAL model on 24
validation pairs on CPU. The assertion is deliberately weak on score and strong on
mechanics: accuracy lands in [0,1], confusion rows sum to 24, one-shot test-read
protection is asserted on the *test* directory only (reading it here would burn the
one honest test number the frbench harness refuses to re-read).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.config import repo_root

pytestmark = pytest.mark.integration

CKPT = repo_root() / "models/linker_nli"
CORPUS = repo_root() / "data/external/facebook__xnli"


@pytest.fixture(scope="module")
def head():
    if not (CKPT / "weights.pt").exists():
        pytest.skip("NLI head not trained (make nli)")
    from engine.link.nli_finetune import load_head

    got = load_head(device="cpu")
    assert got is not None
    return got


@pytest.mark.skipif(not CORPUS.exists(), reason="XNLI-fr not fetched (make frdata)")
def test_head_forwards_real_pairs_and_reports_a_full_confusion(head):
    from engine.link.nli_finetune import evaluate, load_fr

    rows = load_fr("validation")[:24]
    rep = evaluate(head[0], head[1], rows, "cpu")
    assert 0.0 <= rep["accuracy"] <= 1.0
    assert 0.0 <= rep["macro_f1"] <= 1.0
    total = sum(sum(row.values()) for row in rep["confusion"].values())
    assert total == 24, "confusion must account for every row scored"
    # sanity floor, not tuning: a head that collapsed to one class would still have
    # accuracy 1/3; require it at least beats constant-prediction by a point.
    assert rep["accuracy"] > 0.4, f"head collapsed: {rep['accuracy']}"


def test_test_split_read_is_one_event(head, tmp_path, monkeypatch):
    """The 9.5 discipline is code, not prose: once `xnli-nli ... test` exists, the
    harness refuses a second read rather than letting a tuned number replace it."""
    from eval import frbench

    fake = tmp_path / "xnli-nli_camembert-head_test_full"
    fake.mkdir()
    (fake / "result.json").write_text("{}")
    monkeypatch.setattr(frbench, "run_dir_for", lambda c, t, lim, split: fake)
    with pytest.raises(RuntimeError, match="already read once"):
        frbench.eval_xnli_with_head("test")
