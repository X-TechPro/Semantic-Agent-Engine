"""Splits by project, never by sentence (BUILD_GUIDE 7.4 — leakage rule).

The manifest is written to `data/splits/ledger.json` (the path every caller passes;
an earlier version of this docstring named a `split_manifest.json` that no caller
writes, which is the dangling-pointer class of bug this project already fixed once
in the config comments).

What is guaranteed, and what is not:

* **Project ids are disjoint across splits by construction** — `assign()` places each
  project in exactly one bucket. `tests/unit/test_split_leakage.py` asserts it, and
  asserts the manifest's own `leakage.projects` list is empty.
* **Person names are NOT disjoint.** The ledger draws speakers from a shared pool, so
  at 20 projects / seed 42 **16 names appear in more than one split** (measured). The
  manifest records that in `leakage.persons` rather than hiding it, and the test
  asserts the field is *populated*, because a leakage list that is empty by accident
  would read as a clean split. BUILD_GUIDE 7.4 asks for unseen people only "si
  possible", and this generator does not deliver it — so no claim here may say the
  test set holds out persons.

The manifest is also a shared mutable file: it is rewritten by whichever bench or
stage run touched it last, so its split sizes depend on that run's project count
(measured: a 6-project run leaves `4/1/1` where a 20-project run writes `12/4/4`).
Read it with the same `--projects` you intend to evaluate.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from engine.seeding import subseed

SPLIT_RATIOS = {"train": 0.6, "dev": 0.2, "test": 0.2}


def assign(projects: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Deterministic round-robin over a seeded permutation of the project list."""
    import random

    order = list(projects)
    random.Random(subseed(seed, "split", len(order))).shuffle(order)
    out: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    # Atlas is the demo project (13); it must live in *test* so no tuning
    # decision can accidentally be made on it
    if "Atlas" in order:
        order.remove("Atlas")
        order.append("Atlas")
    n = len(order)
    n_dev = max(1, round(n * SPLIT_RATIOS["dev"]))
    n_test = max(1, round(n * SPLIT_RATIOS["test"]))
    out["test"] = order[-n_test:]
    out["dev"] = order[-n_test - n_dev:-n_test]
    out["train"] = order[:-n_test - n_dev] or order[:1]
    if "Atlas" not in out["test"]:
        out["test"].append("Atlas")
    return out


def person_sets(ledger_projects: list[dict[str, Any]],
                split: dict[str, list[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, members in split.items():
        people: set[str] = set()
        for p in ledger_projects:
            if p["project"] in members:
                people |= set(p["people"])
        out[name] = people
    return out


def write_manifest(path: Path, ledger_projects: list[dict[str, Any]],
                   seed: int = 42) -> dict[str, Any]:
    projects = [p["project"] for p in ledger_projects]
    split = assign(projects, seed=seed)
    people = person_sets(ledger_projects, split)
    overlap = {
        "projects": [p for p, s in _pairs(split) if _in_two(p, split)],
        "persons": sorted({n for n in _all(people)
                           if sum(1 for s in people.values() if n in s) > 1}),
    }
    manifest = {"seed": seed, "ratios": SPLIT_RATIOS, "split": split,
                "n_projects": {k: len(v) for k, v in split.items()},
                "persons_per_split": {k: sorted(v) for k, v in people.items()},
                "leakage": overlap}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pairs(d):
    for k, v in d.items():
        for x in v:
            yield x, k


def _all(d):
    s: set[str] = set()
    for v in d.values():
        s |= v
    return s


def _in_two(item, split) -> bool:
    return sum(1 for v in split.values() if item in v) > 1
