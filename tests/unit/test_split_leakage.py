"""The split's leakage contract (7.4) — what the manifest guarantees and what it only admits.

`gen/splice.py`'s docstring used to assert that "a test asserts that no project id
and no person-name tuple appears in two splits". No such test existed, and the
person half of the claim was false: the ledger draws speakers from a shared pool, so
names recur across splits. This file is the test that claim should have been, and it
pins the two facts separately rather than asserting one clean-sounding property that
is only true for half of it.

* **Projects are disjoint.** `assign()` puts each project in exactly one bucket, and
  that is what makes the headline comparable across seeds -- it is asserted, and the
  manifest's own `leakage.projects` list must agree (an empty list is the pass
  condition, so a bug that stopped computing leakage would fail here rather than
  silently report a clean split).
* **Persons are not disjoint, and the manifest says so.** `leakage.persons` must be
  *populated* on a corpus large enough to force reuse. Asserting non-emptiness is
  deliberate: if someone later "fixes" the generator to disjoint people, this test
  fails loudly and the change becomes a decision with a re-measurement attached,
  instead of a quiet edit that invalidates every recorded number.

Both directions are checked because a leakage report that is empty by accident reads
identically to a leakage report that is empty by design.
"""

from __future__ import annotations

import json

import pytest

from gen.ledger import build_ledger
from gen.splice import SPLIT_RATIOS, assign, person_sets, write_manifest

pytestmark = pytest.mark.unit

NAMES = ("train", "dev", "test")


@pytest.fixture(scope="module")
def corpus():
    """The canonical 20-project ledger at seed 42 -- the corpus the headline runs on."""
    return build_ledger(n_projects=20, seed=42)


@pytest.fixture(scope="module")
def manifest(corpus, tmp_path_factory):
    """Module-scoped, so the temp dir must come from `tmp_path_factory`: depending
    on the function-scoped `tmp_path` here is a ScopeMismatch at collection."""
    return write_manifest(tmp_path_factory.mktemp("split") / "ledger.json",
                          corpus.projects, seed=42)


# --------------------------------------------------------------- project disjointness

def test_assign_places_every_project_in_exactly_one_split(corpus):
    # `Ledger.projects` is a list of dicts, not objects: `write_manifest` itself
    # reads `p["project"]`, and this file's first draft invented a `.project`
    # attribute that does not exist.
    projects = [p["project"] for p in corpus.projects]
    split = assign(projects, seed=42)
    assert set(split) == set(NAMES), "the split vocabulary drifted"
    for name in NAMES:
        assert len(split[name]) == len(set(split[name])), f"{name} has duplicates"
    seen = [p for name in NAMES for p in split[name]]
    assert len(seen) == len(set(seen)), f"a project is in two splits: {seen}"
    assert set(seen) == set(projects), "assign() dropped or invented a project"


def test_the_demo_project_lives_in_test_only(corpus):
    """Atlas is scenario 1 of 13, so tuning on it would contaminate the demo claim."""
    split = assign([p["project"] for p in corpus.projects], seed=42)
    assert "Atlas" in split["test"]
    assert "Atlas" not in split["train"] and "Atlas" not in split["dev"]


def test_assign_is_deterministic_per_seed_and_ratios_are_roughly_honoured(corpus):
    projects = [p["project"] for p in corpus.projects]
    assert assign(projects, 42) == assign(projects, 42)
    split = assign(projects, 42)
    n = len(projects)
    assert len(split["train"]) >= round(n * SPLIT_RATIOS["train"]) - 1
    assert len(split["test"]) >= 1 and len(split["dev"]) >= 1


def test_the_manifest_reports_no_project_leakage(manifest):
    """The pass condition is an *empty* list, so this also catches leakage-detection
    itself breaking: no list at all, or a key renamed, must not read as clean."""
    assert "leakage" in manifest, "the manifest stopped computing leakage"
    assert manifest["leakage"]["projects"] == [], manifest["leakage"]
    assert manifest["n_projects"] == {k: len(v) for k, v in manifest["split"].items()}


# ------------------------------------------------------------------ person overlap

def test_persons_are_shared_across_splits_and_the_manifest_admits_it(corpus, manifest):
    """The honest half: 7.4 asks for unseen people only "si possible", and this
    generator does not deliver it. Recorded, not hidden."""
    people = person_sets(corpus.projects, manifest["split"])
    overlap = {n for n in set().union(*people.values())
               if sum(1 for s in people.values() if n in s) > 1}
    assert overlap, ("person names are now disjoint across splits -- the generator "
                     "changed, so every recorded number needs re-measuring before "
                     "this assertion is deleted (see docs/LIMITS.md)")
    assert sorted(overlap) == manifest["leakage"]["persons"], \
        "the manifest's leakage report disagrees with the generator"


def test_some_test_people_are_still_unseen_in_train(corpus, manifest):
    """Not a guarantee, a measurement: the overlap is partial, and the report should
    not overstate it either way."""
    people = person_sets(corpus.projects, manifest["split"])
    fresh = people["test"] - people["train"]
    assert fresh, "no test-project speaker is unseen in train any more"
    assert len(fresh) < len(people["test"]), (
        "every test speaker is unseen in train -- if the generator now holds people "
        "out, docs/LIMITS.md and the claim above must be revised, not this test kept")


def test_manifest_written_to_disk_is_valid_json_and_carries_the_split(manifest, tmp_path):
    """`write_manifest` is called by the bench, the stage runner and the concept
    corpus with different project counts, so the file is shared mutable state.
    Its shape is what callers rely on; its *sizes* are not."""
    path = tmp_path / "ledger.json"
    write_manifest(path, build_ledger(n_projects=6, seed=42).projects, seed=42)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk["split"]) == set(NAMES)
    assert on_disk["n_projects"] == {k: len(v) for k, v in on_disk["split"].items()}
    assert on_disk["n_projects"] != manifest["n_projects"], \
        "a 6-project run must not look identical to the 20-project canonical split"
