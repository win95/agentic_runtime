# Phase 3 skill-subgraph — verification after `67e767f` + `2e9ab1a`

- **Date:** 2026-11-14
- **Commit:** `2e9ab1a` (Loop fix: booking-lookup fallback + per-node tool-call cap), on top of `67e767f` (END only terminal, `NEXT: WAIT`, branch conditions)
- **Agent / mission:** `hotel_receptionist` / `hotel_reception`, AgentVersion **v2**
- **Model:** `qwen2.5-3b`, temperature `0.5` (pinned by v2)
- **Stack:** local editable-venv Celery worker (compose worker stopped), mock hotel MCP on `0.0.0.0:8081 --reseed`, API healthy
- **Command:** `.venv/bin/python scripts/manual/hotel_reception.py --session-id sess-hotel-manual-v4b --timeout 420`
- **Raw log:** `reports/phase3-subgraph-verify-2e9ab1a.log`
- **MLflow report:** `reports/hotel-mlflow-report-v4b.md` + `reports/hotel-mlflow-traces-v4b.csv`
- **Unit tests:** 38 passed (`test_skill_subgraph.py`, `test_skill_maps.py`, `test_hotel_reception.py`)

## Verdict

**The two commits did not fix the main failures; one fix (per-node cap) works but
introduced a new regression on `policy`.**

| | Result |
|---|---|
| Suite | **FAIL** — 109 passed, 32 failed, 27 warnings (32 traces, 29 SUCCEEDED, 3 FAILED) |
| Subgraph engaged | Yes — MLflow spans show map nodes (`route`, `booking-lookup`, `booking-change`, …) |
| Meaningful guest answers | **17/32** (15/32 answers are *only* a `NEXT:` control line; 25/32 contain one) |
| Runaway tool loop | **Contained** — max 15 audit rows (was ~160 `get_booking`) |
| Correct routing | Still unreliable (quotes and service tasks misrouted) |

The 3 FAILED traces are a mid-run LLM-proxy `Connection error` (11:18), not subgraph
behaviour; they hit `multi_request` turns 1–2.

## Root cause found: the `NEXT:` regex rejects the model's real output

`runtime/skill_subgraph.py`:

```python
NEXT_RE = re.compile(r"^NEXT:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)
```

The model almost always appends a reason:

```
NEXT: WAIT — khách chưa xác nhận đặt phòng.
NEXT: END — task đã được giao và mã booking đã được tạo.
NEXT: booking-dates — phòng còn trống, cho biết giá…
```

`\s*$` rejects every one of those lines, so `parse_next()` returns `None` and
`_route()` falls back to `_next_in_order()` — the next node in map order. The commit
that added branch conditions (`NEXT: <id> (when)`) makes the trailing text *more*
likely, so the failure got worse.

Concrete sequence from `exec_e23229bfa98d` (`soldout_switch` turn 2, a quote request):

```
route            model: NEXT: booking-lookup — khách muốn đặt phòng   -> unparsed -> fallback booking-dates
booking-dates    model: NEXT: check-availability — …                  -> unparsed -> fallback check-availability
check-availability model: NEXT: check-availability — …                -> unparsed -> fallback quote-price
quote-price      model: NEXT: booking-dates — …                       -> unparsed -> fallback confirm-booking
confirm-booking  model: NEXT: WAIT — khách chưa xác nhận đặt phòng.   -> unparsed -> fallback create-booking (!)
create-booking   model: NEXT: WAIT — khách chưa xác nhận…             -> unparsed -> fallback report-booking
report-booking   model: NEXT: END — khách đã xác nhận…                -> unparsed -> fallback task-details (!)
task-details     model: NEXT: WAIT — …                                -> unparsed -> fallback find-staff
… then find-staff -> create-task -> assign-task -> report-task
```

Consequences:
- `NEXT: WAIT` at `confirm-booking` was ignored, so the router advanced into
  `create-booking` **without guest confirmation** (the write did not execute only
  because the model refused again).
- After the terminal `report-booking`, an unparsed `NEXT: END` walked the map tail
  and **created and assigned a service task** that the guest never asked for.
- The last node's raw control line becomes the answer.

Map paths observed (MLflow spans):

| Session | Path |
|---|---|
| availability | `route -> booking-dates` (stops; no PMS tool) |
| soldout | `route -> booking-dates` (stops; no PMS tool) |
| task | `route -> task-details` (stops; no task) |
| task_full | `route -> booking-dates` (wrong branch) |
| booking / journey / cancel | `route -> booking-lookup -> booking-change -> task-status` |
| policy | `route -> policy-lookup` (cap fired; read `POL-REFUND`, not `POL-CANCEL`) |
| soldout_switch t2 | full booking chain + `task-details -> … -> report-task` |
| multi_request t1 | `booking-lookup` cap fired: `find_guest`×9 + `get_booking`×6 audit rows |

## Per-claim check

### `67e767f` — "END only at terminal nodes"

**Not effective in live behaviour.** The guard exists, but because `NEXT: END — …`
is unparsed, the fallback still walks the tail from terminal nodes
(`report-booking -> task-details -> … -> report-task`). A bare `NEXT: END`
(`exec_409e9d7daf15`) is honoured correctly.

### `67e767f` — "`NEXT: WAIT`"

**Mechanically present, but unsafe as shipped.** It is honoured only when the line
is a bare `NEXT: WAIT`; with an explanation it is ignored and the flow advances.
When honoured, it becomes the *entire* guest answer:

- 15/32 answers are only a control line (`NEXT: WAIT` / `NEXT: END` / `NEXT: <node>`)
- availability, soldout, task, task_full, memory all stop on the first tool-less node

So the old "premature END" failure is not fixed — it is renamed to `WAIT`.

### `67e767f` — "branch conditions in node instruction"

**Present but not effective.** The prompt now shows `node (when)` pairs (verified in
the trace), yet `route` still sends "hỏi giá phòng Deluxe, chưa đặt phòng" to
`booking-lookup` and the `task_full` housekeeping request to `booking-dates`.

### `2e9ab1a` — per-node tool-call cap

**Works.** `exec_1051f200ea2d` (`booking-lookup`) stopped with:

```
Step 'booking-lookup' stopped: the tool-call budget for this step is exhausted.
NEXT: WAIT
```

The old 100+ `get_booking` loop is gone (max 15 audit rows, 21 s instead of 147 s).

**Regression:** the budget is too small for `policy-lookup`. `exec_94a042ad21a1`
spent 5 calls on search/read retries, hit the cap, and stopped before reading
`POL-CANCEL` (it read `POL-REFUND`). Policy now fails "read the POL-CANCEL document"
and the answer check, where the previous run passed the tool check.

### `2e9ab1a` — exact-repeat guard

**No live evidence** (0 occurrences of "You already called"). The repeated calls in
this run used different arguments, so the guard never fired; it remains unverified.

### `2e9ab1a` — `booking-lookup` fallback rule in the skill

**Not effective.** `exec_1051f200ea2d` kept calling `find_guest`/`get_booking` until
the cap; the "don't repeat the same id, use find_guest" wording did not change the
model's behaviour. The cap, not the skill rule, stopped it.

## Other observations

- **Answer composition is still missing.** The node's control line (and, on cap,
  the diagnostic) is returned as the guest answer. `outcome_from_messages` takes
  the last AI message and nothing strips `NEXT:` or the "Step … stopped" text.
- **Tool usage is narrow:** 8/14 contract tools were used; `list_staff`, `assign_task`,
  `get_task`, `cancel_booking`, `change_booking_dates`, `register_guest` were never
  called. `task`, `task_full`, `cancel` fail at that missing step.
- **`stop_completion` still fires 3/3 on tool-less nodes** (`Hook Stop continues
  turn N/3`), so each tool-less node pays up to 3 extra model turns before `WAIT`.
- **Policy output hit `finish_reason=length`** (5126 output tokens), same repeated
  instruction-text pathology as the previous run.
- The Docker skill-discovery bug was not re-tested: the compose worker is stopped and
  the run used the editable `.venv` worker (skills load, 6 skills).

## Recommended fixes (highest impact first)

1. **Accept trailing text in `NEXT`** and strip the control line from the answer:
   `^\s*NEXT:\s*([A-Za-z0-9_-]+)\b` (take the first token), then remove that line
   before returning the guest answer. This alone fixes confirmation-skip, the map
   tail-walk, and most of the answer leakage.
2. **Stop on invalid/missing `NEXT`** instead of `_next_in_order()` fallback; keep the
   fallback only for `None` at a non-terminal node when explicitly intended.
3. **Never advance from a terminal node** unless the declaration is an allowed
   resume node.
4. **Raise / specialize the per-node budget** (`policy-lookup` needs ≥8 calls;
   search + read + retries), and make the cap message *not* the answer.
5. **Separate control plane from content**: `NEXT`, diagnostics and errors go to
   `errors`/`notes`; the answer is the last non-control text from the node.
6. **Harden `route`** (deterministic keyword/structured classifier); it is the single
   point of failure for the whole spine with a 3B model.

## Reproduce

```bash
docker compose stop worker
.venv/bin/celery -A app.worker.celery_app worker --loglevel=info --concurrency=1
.venv/bin/python scripts/mock_hotel_mcp_server.py --reseed --host 0.0.0.0 \
    --allow-host 'host.docker.internal:*'
.venv/bin/python scripts/manual/hotel_reception.py \
    --session-id sess-hotel-manual-v4b --timeout 420
.venv/bin/python scripts/manual/mlflow_report.py \
    --tracking-uri http://localhost:5000 \
    --session-prefix sess-hotel-manual-v4b- \
    --out reports/hotel-mlflow-report-v4b.md \
    --csv reports/hotel-mlflow-traces-v4b.csv
```
