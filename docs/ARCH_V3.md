# Architecture v3 — immutable meetings, decision nodes, a dated view

Status: **v3 is the rollback path**; v4 is the new architecture (`docs/ARCH_V4.md`). v2's pairwise
`supersedes` writer is disabled; the meeting is immutable.

## The constraint

A meeting is a source of truth about what was said at a precise date. A later
message ("the date moved") is a separate source; it must never rewrite the
meeting. The meeting must still retrace the truth **as of its own date**.

## The pipeline

```
finished session
  → segment
  → reader (kind / type / status / intent)
  → node builder: group related messages / continuations
        DecisionNode(children = messages, latest = current child)
  → change detector (optional, review queue)
        node.current = latest detected change
  → immutable store + node view
  → answer: node.current, or node.as_of(date) / node.at_turn(turn)
```

## Components

| stage | file | role |
|---|---|---|
| segment | `engine/segment/` | split a session into message units |
| reader | `engine/read/` | kind, type, status, intent |
| topic tracker | `engine/link/thread.py` | names a topic once, carries it across turns |
| node builder | `engine/threads/session.py` | `DecisionNode`, `build_session_nodes` |
| node view | `engine/threads/view.py` | `NodeView.find`, `current`, `as_of`, `at_turn` |
| detector | `engine/tools/session_changes.py` | review rows; never writes `valid_to` |
| ask wiring | `engine/engine.py`, `engine/agent/router.py` | node current leads answer/history |
| gate | `engine/eval/node_view_gate.py` | all-six measurement |

## Invariants

- A node never writes `valid_to`, `superseded_by` or any child's text.
- `latest` is the last child; `as_of(day)` is the last child on or before that
  day; `at_turn(i)` is the last child at or before that turn.
- The detector emits **review rows only**; it cannot retire a fact.
- Cross-session children append in date order; a later user update is a new
  child, not an edit of the meeting.

## Measured on the six SUMM-RE gold meetings

Artefact: `runs/decision_nodes/v3-all6.json`.

| measure | value |
|---|---|
| coverage (messages filed into a node) | **1,518/1,525 = 99.5%** |
| gold old+new in one node | **25/40 = 0.625** |
| `node.at_turn(new)` returns the gold new | **40/40 = 1.000** |
| `node.latest` is the gold new (final state) | 2/40 = 0.050 |
| mutated children | **0** |

The last row is not a defect: the gold pairs are intermediate changes, and the
thread usually continues afterwards. The node's **current** is the latest
message, as specified; the historical truth is read with `at_turn`/`as_of`.

### Additive state-point nodes (D-063)

`retrieve.node_state: true` adds a second node set keyed by the explicit
`DecisionState` point (`food`, `budget`, `date`, `mode`, `venue`, ...). The
topic nodes remain; a pair is reached if it shares either key. `state: false`
(or the config flag) keeps the old topic-only view.

| measure | topic only | + state |
|---|---|---|
| gold old+new in one node | **25/40 = 0.625** | **33/40 = 0.825** |
| `at_turn(new)` returns the new | 40/40 = 1.000 | **40/40 = 1.000** |
| current is gold new | 2/40 = 0.050 | **6/40 = 0.150** |
| mutated children | 0 | 0 |

This is a reach and current improvement; `at_turn` is unchanged. The state
reader is deterministic and writes a point only when the value is grounded in
the turn.


## Detector (2 labelled meetings)

| meeting | rows | true | precision | gold new | recall |
|---|---|---|---|---|---|
| 024c | 17 | 17 | 1.000 | 4/12 | 0.333 |
| 047a | 22 | 18 | 0.818 | 4/7 | 0.571 |
| node currents | 20 | 18 | **0.900** | 8/19 | **0.421** |

The other four meetings are measured for node reach and historical state. A
targeted 7B scan (only pairs whose new turn is a gold change) selected 4 rows,
2 true, and reached **2/21 gold new turns = 0.095**. The detector does not
generalize to those meetings; more labels alone will not fix it, because the
miss is in candidate discovery/selection, not in the node view.

## Retrieval and ask

`Engine.ask()` runs the node view beside retrieval for `answer`/`history`.
The matching node's current child is prepended to the hits, so the answer leads
with the node's current message and cites it. `as_of=YYYY-MM-DD` selects the
child current on that date. With no matching node, retrieval is unchanged.
`make eval-fast` stays at temporal@1 A=0.4444 / B=0.4444; the demo is 19/19.
