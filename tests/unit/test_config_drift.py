"""The shipped YAML must not drift from what the code actually builds (0.3, 10).

Found by measurement, not by reading. Every engine here is built through
`load_config()` **with no path** — the bench, the kill test, the report, `Engine`
itself — so `configs/system_B.yaml`, the file §10 *names* System B, is never read.
That is only dangerous because the two had drifted: the YAML declared
`link.value_topical_min: 0.5` and pointed at a `docs/REPORT.md` tuning section that
does not exist, while the code default every recorded number ran on was 0.0.

The tuning itself is real and is recorded where it was actually applied —
`engine.config.DEFAULTS` ("0.5 gave P 0.4545 / R 0.2632, 0.0 gave P 0.52 / R 0.6842
-- 0.0 dominates on BOTH axes"), a dev sweep per 9.5. I re-ran it on dev and it
reproduces: P 0.5172 / R 0.7143 at 0.0 against P 0.4375 / R 0.3333 at 0.5. So the
value 0.0 is right and the file's *pointer* was wrong. What this test defends is
the thing that let a stale value sit in a shipped config unnoticed: nothing
compared the file to the code.
"""

from __future__ import annotations

import json

import pytest
import yaml

from engine.config import CONFIGS_DIR, DEFAULTS, load_config, repo_root

pytestmark = pytest.mark.unit

#: keys that label a run instead of changing its behaviour. `exp_id` is the one
#: legitimate difference between a shipped YAML and the bare code defaults.
IDENTITY_KEYS = {"exp_id"}

SHIPPED = "system_B"
OTHER_ROWS = [p.stem for p in CONFIGS_DIR.glob("*.yaml") if p.stem != SHIPPED]

#: 10 fairness rule: "same candidate budget, same context budget, same prompts".
#: An ablation row may differ from System B in the layer under test, NEVER here.
BUDGET_KEYS = (
    "embed.backend", "embed.dim", "embed.batch_size",
    "lexical.backend", "lexical.strip_accents", "lexical.stem", "lexical.over",
    "retrieve.top_k_dense", "retrieve.top_k_lexical", "retrieve.rrf_k",
    "retrieve.weights", "retrieve.final_k", "retrieve.rerank.backend",
    "retrieve.rerank.candidates",
)


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _drift(other: dict, base: dict) -> dict:
    return {k: (base.get(k), other.get(k))
            for k in set(base) | set(other)
            if k not in IDENTITY_KEYS and base.get(k) != other.get(k)}


def test_shipped_config_matches_the_code_defaults_it_annotates():
    """`configs/system_B.yaml` reads as the definition of the running system.

    It is not loaded, so without this test it could contradict the engine line by
    line and every artefact would still be produced from the code defaults — which
    is exactly what happened to the G9 threshold.
    """
    from_file = _flat(dict(load_config(f"{SHIPPED}.yaml")))
    from_code = _flat(DEFAULTS)
    drift = _drift(from_file, from_code)
    assert not drift, (
        f"engine.config.DEFAULTS and configs/{SHIPPED}.yaml disagree on {drift}; "
        f"the engine is built from DEFAULTS because load_config() takes no path, so "
        f"the YAML is documentation and must not be allowed to lie")


@pytest.mark.parametrize("row", OTHER_ROWS)
def test_ablation_rows_keep_the_fairness_budget(row: str):
    """"Same candidate budget" enforced, not promised: rows may differ in the layer
    under study (linker, validity, slot boost), never in retrieval budget."""
    a = _flat(dict(load_config(f"{row}.yaml")))
    b = _flat(dict(load_config(f"{SHIPPED}.yaml")))
    differ = {k: (a.get(k), b.get(k)) for k in BUDGET_KEYS if a.get(k) != b.get(k)}
    assert not differ, f"configs/{row}.yaml changes the comparison budget: {differ}"


def test_g9_threshold_is_the_value_every_recorded_number_ran_on():
    """0.0 is what `runs/` and `baselines/last_good.json` were produced with, and it
    dominates 0.5 on both dev axes (D-014). Moving it changes the linker, so it has
    to move with a fresh dev sweep, not with a comment."""
    assert DEFAULTS["link"]["value_topical_min"] == 0.0

    from engine.link.supervised import build_linker

    linker, backend = build_linker(load_config())
    assert linker.topical_min == 0.0, (
        f"{backend} applied G9's floor as {linker.topical_min}, not the configured "
        f"0.0, so the recorded linker numbers describe a different guard")


@pytest.mark.parametrize("backend", ["rules", "logreg", "nli"])
def test_both_linker_backends_resolve_g9_from_the_same_key(backend: str):
    """The rules and model linkers used to fall back to *different* thresholds
    (0.5 and 0.0) for the same missing key, so a rules-vs-model ablation compared
    two different guard settings and called it a comparison of classifiers.

    All three backends are reached through the one builder the engine uses, and
    the model backends are asked for checkpoints that cannot exist so they report
    the fallback rather than loading a trained file into a unit test:
    `build_linker` is the only place linker kwargs are assembled.
    """
    from engine.link.supervised import build_linker

    stripped = {"link": {k: v for k, v in DEFAULTS["link"].items()
                         if k != "value_topical_min"}}
    stripped["link"]["backend"] = backend
    # paths that cannot exist: logreg and nli must degrade to rules, never to a
    # different G9 threshold on the way
    stripped["link"]["model_path"] = str(repo_root() / "models" / "no-such-linker.json")
    stripped["link"]["nli_model_dir"] = str(repo_root() / "models" / "no-such-linker-nli")

    linker, resolved = build_linker(stripped)
    assert linker.topical_min == 0.0, (
        f"{backend} -> {resolved}: G9 fell back to {linker.topical_min} with the key "
        f"absent; the shipped value is 0.0 and both backends must agree on that")


def test_live_config_hash_is_the_one_the_regression_gate_records():
    """`config_hash` is the only thing tying a recorded number to a configuration,
    so a silent change here invalidates `baselines/last_good.json`'s provenance."""
    gate = repo_root() / "baselines" / "fast_gate.json"
    recorded = json.loads(gate.read_text(encoding="utf-8")).get("config_hash")
    live = load_config().hash()
    if recorded is None:
        pytest.skip("fast gate records no config_hash yet")
    assert live == recorded, (
        f"the engine's configuration changed ({recorded} -> {live}) while "
        f"baselines/fast_gate.json still describes {recorded}: re-run "
        f"`make eval-fast --update-gate` so the gate matches the system it guards")
