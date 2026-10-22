"""The CLI surface (BUILD_GUIDE 2.3, M2 gate "end-to-end CLI works").

`engine/cli.py` was 221 statements at 0% coverage: the product's primary entry
point was never executed by a test. Everything the guide insists on lives here —
the backend flags, the JSON contract the API and UI parse, and the rule that a run
reports which backends *actually* ran rather than which ones were asked for.

Tests drive `cli.main(argv)` in-process (no subprocess): the thing under test is
argument plumbing and output shape, not process spawning.
"""

from __future__ import annotations

import io
import json
import pathlib
from contextlib import redirect_stderr, redirect_stdout

import pytest

from engine import cli
from engine.schemas import Query

pytestmark = pytest.mark.unit

FIXTURE = "tests/golden/demo_scenarios.json"


def _source_files(tmp_path):
    """The §13 fixture as one Source per file, which is what `ingest` takes."""
    from engine.cli_demo import _sources

    out = []
    for i, s in enumerate(_sources(pathlib.Path(FIXTURE))):
        p = tmp_path / f"src-{i}.json"
        p.write_text(s.model_dump_json(), encoding="utf-8")
        out.append(p)
    return out


@pytest.fixture
def store(tmp_path):
    return tmp_path / "cli.db"


@pytest.fixture
def loaded(tmp_path, store):
    """A store with the demo corpus already ingested, plus the global flags."""
    glob = ["--store", str(store), "--embed", "hash"]
    files = _source_files(tmp_path)
    code, out = _run(glob + ["ingest"] + [str(p) for p in files])
    assert code == 0
    return glob


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


# --------------------------------------------------------------- flag plumbing

def test_subcommand_backend_flags_beat_the_global_ones():
    """`sae --rerank overlap ask ... --rerank none` must resolve to `none`: the
    narrowest scope wins, and one resolver per knob means no second rule to
    disagree with."""
    args = cli.build_parser().parse_args(
        ["--rerank", "overlap", "--reader", "rules", "ask", "x",
         "--rerank", "none", "--reader", "model"])
    assert cli._rerank_of(args) == "none"
    assert cli._reader_of(args) == "model"
    glob = cli.build_parser().parse_args(["ask", "x"])
    assert cli._rerank_of(glob) is None and cli._reader_of(glob) is None


def test_every_retrieval_subcommand_accepts_rerank():
    """D-011: a knob reachable only on `bench` is an inert knob everywhere else."""
    for cmd in (["ingest", "f.json"], ["query", "q"], ["ask", "q"], ["bench"],
                ["latency"]):
        args = cli.build_parser().parse_args(cmd + ["--rerank", "overlap"])
        assert cli._rerank_of(args) == "overlap", cmd


def test_open_engine_reports_the_backend_that_actually_ran(store):
    eng = cli._open_engine(cli.build_parser().parse_args(
        ["--store", str(store), "--embed", "hash", "--rerank", "overlap",
         "query", "x"]))
    try:
        assert eng.rerank_backend == "overlap"          # read back, not assumed
        assert eng.reader_backend == "rules"
        assert eng.stats()["reader_backend"] == "rules"
    finally:
        eng.close()


def test_unknown_rerank_backend_fails_loudly(store):
    """A backend the factory doesn't know must raise, not quietly fall back to
    `none` — a silent fallback would let an ablation row lie about itself.

    Built by hand rather than through the parser: argparse restricts `--rerank` to
    its choices, so an unknown value can only arrive from a config file.

    And asserted on the *pipeline*, not on `Engine(...)`: the pipeline is built
    lazily on first retrieval, so construction alone is not where this surfaces —
    asserting a raise at construction time would have been testing a behaviour the
    product does not have.
    """
    args = cli.build_parser().parse_args(
        ["--store", str(store), "--embed", "hash", "query", "x"])
    args.rerank_sc = "nope"
    eng = cli._open_engine(args)
    try:
        with pytest.raises(ValueError, match="unknown rerank backend"):
            _ = eng.rerank_backend
        with pytest.raises(ValueError, match="unknown rerank backend"):
            eng.retrieve(Query(text="budget", project="Atlas"))
    finally:
        eng.close()


# ------------------------------------------------------------------- commands

def test_ledger_then_render_then_ingest_then_query(tmp_path, store):
    led = tmp_path / "ledger"
    code, out = _run(["ledger", "--projects", "2", "--seed", "42", "--out", str(led)])
    assert code == 0 and (led / "ledger-42.json").exists() and (led / "audit.md").exists()
    assert json.loads((led / "ledger-42.json").read_text())

    rendered = tmp_path / "rendered"
    code, out = _run(["render", "--ledger", str(led / "ledger-42.json"),
                      "--out", str(rendered)])
    assert code == 0 and any(rendered.glob("*.json")) and "turns" in out

    glob = ["--store", str(store), "--embed", "hash"]
    code, out = _run(glob + ["ingest", str(rendered)])        # a directory of sources
    assert code == 0 and "units=" in out and "store:" in out

    code, out = _run(glob + ["--json", "query", "budget"])
    assert code == 0
    rows = json.loads(out)
    assert rows and ({"rank", "unit_id", "score", "status", "current", "path",
                      "evidence", "score_parts"} <= set(rows[0]))


def test_query_pretty_marks_exactly_the_non_current_rows(loaded):
    """The `[SUPERSEDED]` flag is the only place a human reads validity off the
    CLI; it must agree row-for-row with the JSON contract the API parses.

    Queried with `--no-linker` so the stale units are actually still in the list:
    with the linker on, System B demotes them and the assertion below would be
    vacuous rather than verified.
    """
    argv = loaded + ["query", "budget", "--project", "Atlas", "--no-linker", "--k", "20"]
    code, pretty = _run(argv)
    assert code == 0
    rows = json.loads(_run(loaded + ["--json"] + argv[2:])[1])
    assert rows, "the fixture has plenty of budget rows"
    assert pretty.count("[SUPERSEDED]") == len([r for r in rows if not r["current"]])


def test_ask_emits_the_contract_the_api_parses(loaded):
    code, out = _run(loaded + ["--json", "ask",
                               "Rédige un mail à la personne responsable du budget "
                               "d'Atlas."])
    assert code == 0
    payload = json.loads(out)
    assert {"intent", "operation", "used_llm", "notes", "hits", "payload"} <= set(payload)
    assert payload["used_llm"] is False                  # template-only by default
    assert '"executed": true' not in out                # nothing ever executes


def test_history_lists_every_assertion_with_validity(loaded):
    code, out = _run(loaded + ["--json", "history", "atlas.budget", "--project",
                               "Atlas"])
    assert code == 0
    rows = json.loads(out)
    assert rows, "the fixture makes several claims about atlas.budget"
    assert {"unit_id", "date", "status", "valid_to", "superseded_by", "text"} \
        <= set(rows[0])
    assert any(r["valid_to"] for r in rows), "a retired assertion must show valid_to"


def test_export_schemas_writes_valid_json(tmp_path):
    out_dir = tmp_path / "schemas"
    code, out = _run(["export-schemas", "--out", str(out_dir)])
    assert code == 0 and "JSON schemas" in out
    written = list(out_dir.glob("*.json"))
    assert written
    for p in written:
        assert json.loads(p.read_text(encoding="utf-8"))


def test_exported_index_carries_every_contract_and_a_non_empty_label_set(tmp_path):
    """2.3: these files are what an external agent validates against.

    `label_sets` shipped as `{"source_type": []}` because the expression was
    `metadata and []`, which is `[]` whatever the metadata holds — an empty list
    that *looks* like a finished field, so nothing downstream noticed. A string
    field like `Unit.type` has no allowed-value list in JSON Schema at all, so if
    the index does not carry it the consumer can only guess.
    """
    from engine.schemas import SOURCE_TYPES
    from engine.tools.export_schemas import CONTRACTS

    out_dir = tmp_path / "schemas"
    code, _out = _run(["export-schemas", "--out", str(out_dir)])
    assert code == 0
    idx = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    assert set(idx["contracts"]) == set(CONTRACTS)
    unit = json.loads((out_dir / "unit.schema.json").read_text(encoding="utf-8"))
    assert set(idx["contracts"]["unit"]["required"]) == set(unit["required"])
    ls = idx["label_sets"]
    assert ls["source_type"] == list(SOURCE_TYPES)
    assert all(v for v in ls.values()), f"an empty label set shipped again: {ls}"


def test_the_schemas_module_entry_point_writes_where_its_own_root_says(monkeypatch,
                                                                       tmp_path):
    """`python -m engine.tools.export_schemas` is a second entry point besides
    `sae export-schemas`, and it had never run: it resolves its output through
    `repo_root()`, which the CLI path bypasses with an explicit `--out`.

    The root is redirected so the test cannot rewrite the shipped `schemas/`
    directory — a unit test that mutates committed artefacts would make the very
    file it is checking the test's own output.
    """
    from engine.tools import export_schemas

    monkeypatch.setattr("engine.config.repo_root", lambda: tmp_path)
    assert export_schemas.main([]) == 0
    written = sorted(p.name for p in (tmp_path / "schemas").glob("*.json"))
    assert "unit.schema.json" in written and "index.json" in written
    assert len(written) == len(export_schemas.CONTRACTS) + 1


# ------------------------------------------------------- output and error paths

def test_emit_pretty_renders_dicts_and_scalars(capsys):
    cli._emit({"a": 1, "b": "x"}, False)
    assert capsys.readouterr().out == "a: 1\nb: x\n"
    cli._emit("plain", False)
    assert capsys.readouterr().out == "plain\n"
    cli._emit({"a": 1}, True)
    assert json.loads(capsys.readouterr().out) == {"a": 1}


def test_main_turns_an_exception_into_exit_1_with_a_cause(store):
    """No traceback at the user, and no silent failure either: the exception type
    and message are the diagnosis. `/proc` cannot hold a new sqlite file."""
    err = io.StringIO()
    with redirect_stderr(err):
        code = cli.main(["--store", "/proc/nope/x.db", "--embed", "hash",
                         "query", "budget"])
    assert code == 1
    assert err.getvalue().startswith("error:")
    assert ":" in err.getvalue()[7:]                    # type and message


def test_broken_pipe_is_not_a_failure(loaded, monkeypatch):
    """`sae ask ... | head` closes the pipe mid-print. That is success, not exit 1.

    Patched on `_emit`, which is how `ask`/`history` print — `query`'s own print
    path bypasses it, so choosing the wrong command here would have tested nothing.
    """
    def boom(*a, **k):
        raise BrokenPipeError()

    monkeypatch.setattr(cli, "_emit", boom)
    err = io.StringIO()
    with redirect_stderr(err):
        assert cli.main(loaded + ["--json", "ask", "Quel est le budget ?"]) == 0


# ------------------------------------------- heavy commands: wiring, not runtime

def test_cmd_bench_passes_every_backend_choice_to_the_harness(monkeypatch):
    import eval.temporal as T

    seen = {}

    def fake_seeds(seeds, **kw):
        seen.update(seeds=seeds, **kw)
        return {"temporal@1_mean": {"baseline_A": 0.25, "system_B": 0.55,
                                    "delta": 0.3},
                "temporal@1_spread": 0.17, "seeds_with_significant_win": 3,
                "per_seed": {"42": {"temporal@1": {
                    "baseline_A": 0.23, "system_B": 0.51, "delta": 0.29,
                    "delta_ci95": [0.1, 0.5], "significant_win": True}}},
                # real provenance: the CLI reports which space was searched, and a
                # refined row must never be mistaken for the headline
                "provenance": {"vector_column": kw.get("vector_column") or "base"},
                "canonical": (kw.get("vector_column") or "base") == "base",
                "_out": "runs/x"}

    monkeypatch.setattr(T, "run_bench_seeds", fake_seeds)
    code, out = _run(["bench", "--seeds", "42,43,44", "--split", "test",
                      "--rerank", "cross_encoder", "--reader", "model"])
    assert code == 0
    assert seen["rerank_backend"] == "cross_encoder" and seen["reader_backend"] == "model"
    assert seen["seeds"] == [42, 43, 44]
    assert "3/3" in out and "seed 42" in out
    # unset means "whatever the config says", not "base": the harness resolves it
    assert seen["vector_column"] is None

    code, out = _run(["bench", "--vectors", "refined", "--split", "dev"])
    assert code == 0
    assert seen["vector_column"] == "refined", "6.13's row 9 is unreachable from the CLI"
    assert "vector column: refined" in out
    assert "canonical=False" in out, "a refined run must announce it cannot be the headline"


def test_cmd_stage_forwards_split_and_project_count(monkeypatch):
    from eval import run_stage as RS

    seen = {}

    def fake(stage, args):
        seen.update(stage=stage, split=args.split, projects=args.projects)
        return 0

    monkeypatch.setattr(RS, "run_stage", fake)
    code, _ = _run(["stage", "reader", "--split", "dev", "--projects", "3"])
    assert code == 0
    assert seen == {"stage": "reader", "split": "dev", "projects": 3}


def test_cmd_demo_returns_the_suites_own_verdict(monkeypatch):
    """The §13 gate's exit code must reach `make demo`; swallowing a 1 would make a
    broken demo look green."""
    import engine.cli_demo as CD

    monkeypatch.setattr(CD, "run_demos", lambda args: 1)
    assert cli.main(["demo"]) == 1


def test_cmd_latency_fails_when_nothing_was_timed(monkeypatch):
    """An empty holdout must not print `p50=None` as if it were a measurement."""
    from eval import latency as L

    monkeypatch.setattr(L, "measure", lambda **kw: {
        "end_to_end_ms": None, "cold_start_ms": None, "stages_ms": {},
        "peak": {"ram_mb": 0.0, "vram_alloc_mib": 0.0},
        "ingest": {"units_per_s": 0.0}, "not_measured": [], "n_queries": 0,
        "split": "dev"})
    err = io.StringIO()
    with redirect_stderr(err):
        code = cli.main(["latency", "--projects", "20", "--split", "dev"])
    assert code == 1 and "no queries to time" in err.getvalue()


def test_cmd_latency_prints_stages_cold_start_and_gaps(monkeypatch):
    from eval import latency as L

    monkeypatch.setattr(L, "measure", lambda **kw: {
        "end_to_end_ms": {"p50": 14.4, "p95": 38.3, "mean": 190.0, "n": 52},
        "cold_start_ms": 9027.0,
        "stages_ms": {"dense_s": {"p50": 0.6, "p95": 0.8, "mean": 0.6, "n": 52},
                      "rerank_s": {"p50": 11.9, "p95": 40.7, "mean": 30.4, "n": 52}},
        "peak": {"ram_mb": 3891.2, "vram_alloc_mib": 2182.2},
        "ingest": {"units_per_s": 89.7}, "not_measured": ["validity_s"],
        "n_queries": 52, "split": "test", "_out_dir": "runs/latency/x"})
    code, out = _run(["latency", "--rerank", "cross_encoder"])
    assert code == 0
    assert "cold start 9027.0 ms" in out and "rerank_s" in out
    assert "not measured: validity_s" in out


def test_cmd_verify_models_returns_the_tool_exit_code(monkeypatch):
    """A licence failure is the gate; swallowing its code would let one through."""
    from engine.tools import verify_models as VM

    monkeypatch.setattr(VM, "main", lambda argv: 2)
    assert cli.main(["verify-models"]) == 2
