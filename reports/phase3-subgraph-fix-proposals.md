# Đề xuất sửa lỗi — Phase 3 skill-subgraph

- **Ngày:** 2026-11-14
- **Mốc kiểm chứng:** `2e9ab1a` (trên `67e767f`)
- **Tài liệu bằng chứng:** `reports/phase3-subgraph-verify-2e9ab1a.md`
- **Log suite:** `reports/phase3-subgraph-verify-2e9ab1a.log`
- **MLflow:** `reports/hotel-mlflow-report-v4b.md`, `reports/hotel-mlflow-traces-v4b.csv`
- **Trạng thái:** **chỉ đề xuất — không có thay đổi source code nào trong tài liệu này.**

Mỗi mục dưới đây gồm: hiện trạng → nguyên nhân → diff đề xuất → test → cách kiểm chứng.
Số dòng là theo commit `2e9ab1a`; sau khi sửa nên bám theo tên hàm/khối code.

## 0. Tóm tắt và thứ tự thực hiện

| ID | Mức | Vấn đề | Tác động | Công sức |
|---|---|---|---|---|
| P0-1 | Chặn | `NEXT_RE` từ chối dòng `NEXT: X — lý do` | Gây phần lớn 32 fail | ~20 dòng |
| P0-2 | Chặn | `NEXT` có marker nhưng parse lỗi → vẫn fallback `_next_in_order()` | Đi sai node, ghi booking/task ngoài ý khách | ~15 dòng |
| P0-3 | Chặn | Node `terminal` vẫn rơi xuống node kế trong list | `report-booking → task-details → … → report-task` | ~6 dòng |
| P1-1 | Cao | Budget 5 quá thấp cho `policy-lookup` | Đọc nhầm `POL-REFUND`, fail policy | ~10 dòng |
| P1-2 | Cao | Repeat-guard chỉ so call lỗi *ngay trước* | `find_guest`/`get_booking` xen kẽ lách guard | ~15 dòng |
| P1-3 | Cao | Answer = control line / dòng chẩn đoán | 15/32 câu trả lời rỗng nghĩa | ~25 dòng |
| P2-1 | Trung bình | `route` do model 3B quyết định | Misroute quote → `booking-lookup`, task → `booking-dates` | ~80 dòng |
| P2-2 | Trung bình | Prompt node chưa cấm text sau `NEXT` / lặp chẩn đoán | Lặp lỗi, rò rỉ nội bộ | ~10 dòng |
| P2-3 | Thấp | `stop_completion` đốt 3 lượt ở node không gọi tool | Chậm, tốn token | ~10 dòng |

Thứ tự an toàn: **P0-1 → P0-3 → P0-2 → P1-1 → P1-2 → P1-3 → P2-1 → P2-2 → P2-3**.
P0-1 làm P0-2/P0-3 dễ đo hơn (sau P0-1, `NEXT` hợp lệ gần như luôn parse được).

---

## P0-1 — Parse `NEXT` chịu được phần giải thích; không trả control line làm câu trả lời

### Hiện trạng

`runtime/skill_subgraph.py:208`:

```python
NEXT_RE = re.compile(r"^NEXT:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)
```

`\s*$` bắt buộc token phải kết thúc dòng. Model thực tế luôn viết thêm lý do:

```
NEXT: WAIT — khách chưa xác nhận đặt phòng.
NEXT: END — task đã được giao và mã booking đã được tạo.
NEXT: booking-dates — phòng còn trống, cho biết giá…
```

`parse_next()` trả `None` → `_route()` rơi vào `_next_in_order()`.

Bằng chứng chuỗi `exec_e23229bfa98d` (`soldout_switch` turn 2):

```
route             NEXT: booking-lookup — khách muốn đặt phòng   -> booking-dates
booking-dates     NEXT: check-availability — …                  -> check-availability
check-availability NEXT: check-availability — …                 -> quote-price
quote-price       NEXT: booking-dates — …                       -> confirm-booking
confirm-booking   NEXT: WAIT — khách chưa xác nhận…             -> create-booking (!)
create-booking    NEXT: WAIT — khách chưa xác nhận…             -> report-booking
report-booking    NEXT: END — khách đã xác nhận…                -> task-details (!)
task-details      NEXT: WAIT — …                                -> find-staff
… find-staff -> create-task -> assign-task -> report-task
```

Hệ quả kép: (a) router đi sai node, (b) dòng `NEXT:` cuối trở thành câu trả lời cho khách.

### Diff đề xuất

Thay khối regex + thêm helper (đặt ngay dưới `NEXT_RE`):

```python
# The model's real output puts an explanation after the token, often in
# bold, sometimes indented:
#   NEXT: WAIT — khách chưa xác nhận đặt phòng.
#   **NEXT:** booking-dates — phòng còn trống.
# Only the first token is the declaration; everything after it is prose.
NEXT_RE = re.compile(
    r"(?m)^[ \t]*(?:\*\*)?NEXT(?:\*\*)?\s*:\s*(?:\*\*)?([A-Za-z0-9_-]+)"
)

# A whole control line, used to keep NEXT/diagnostics out of the answer.
CONTROL_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:\*\*)?NEXT(?:\*\*)?\s*:.*(?:\n|$)"
)
DIAGNOSTIC_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:Step '[^']+' stopped:|Tool-call budget).*$"
)


def has_next_marker(text: str) -> bool:
    """True when the text contains a NEXT line, parseable or not."""
    return bool(text) and bool(CONTROL_LINE_RE.search(text))


def strip_control_lines(text: str) -> str:
    """Remove control/diagnostic lines from guest-facing text."""
    if not text:
        return ""
    cleaned = DIAGNOSTIC_LINE_RE.sub("", text)
    cleaned = CONTROL_LINE_RE.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
```

`parse_next()` giữ nguyên (dùng `NEXT_RE.search`, lấy match đầu tiên — nếu model lỡ in
nhiều dòng `NEXT:` thì dòng đầu thắng, có tính tất định).

Thêm hàm chọn answer (đặt cạnh `_last_text`, `runtime/skill_subgraph.py:288`):

```python
def _message_text(message) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return " ".join(t for t in texts if t)
    return ""


def _last_answer_text(messages: list) -> str:
    """The last assistant text that is actual content, not a NEXT line.

    Nodes that only emitted a control line (the 15/32 empty answers in
    the v4b run) fall back to the previous node's answer instead of
    returning "NEXT: WAIT" to the guest.
    """
    for message in reversed(messages or []):
        if not isinstance(message, AIMessage):
            continue
        text = strip_control_lines(_message_text(message))
        if text:
            return text
    return ""
```

Sửa `run_skill_subgraph` (`runtime/skill_subgraph.py:576`):

```python
    outcome = outcome_from_messages(messages, skill_maps=skill_maps)
    answer = strip_control_lines(outcome.get("summary") or "")
    if not answer:
        answer = _last_answer_text(messages)
    outcome["summary"] = answer
    for item in outcome.get("evidence") or []:
        if item.get("kind") == "summary":
            item["summary"] = answer
```

> `outcome_from_messages` không đổi: nó vẫn là "last non-empty text" cho cả free-form
> agent; chỉ subgraph (nơi có `NEXT`) cần sanitize.

### Test

Thêm vào `tests/test_skill_subgraph.py`:

```python
def test_parse_next_tolerates_trailing_explanation():
    assert parse_next("NEXT: WAIT — khách chưa xác nhận") == "WAIT"
    assert parse_next("NEXT: END — đã xong") == "END"
    assert parse_next("**NEXT:** booking-dates — còn phòng") == "booking-dates"
    assert parse_next("  NEXT: create-booking") == "create-booking"
    # first declaration wins, deterministic
    assert parse_next("NEXT: END\nNEXT: other") == "END"
    # a mere mention mid-sentence is not a declaration
    assert parse_next("the NEXT: thing is") is None


def test_strip_control_lines_removes_next_and_diagnostics():
    text = (
        "Dạ còn phòng Deluxe ạ.\n"
        "NEXT: END — đã báo giá\n"
        "Step 'booking-lookup' stopped: the tool-call budget for this step is exhausted."
    )
    assert strip_control_lines(text) == "Dạ còn phòng Deluxe ạ."


def test_run_keeps_control_line_out_of_answer():
    flow = _map([{"id": "route"}, {"id": "report", "terminal": True}])
    result = _run(flow, {
        "route": ([AIMessage(content="routing")], "report"),
        "report": ([AIMessage(content="Dạ đã xong.\nNEXT: END — hoàn tất")], None),
    })
    assert result["summary"] == "Dạ đã xong."


def test_run_falls_back_to_previous_answer_when_node_only_declares():
    flow = _map([{"id": "route"}, {"id": "dates"}, {"id": "report", "terminal": True}])
    result = _run(flow, {
        "route": ([AIMessage(content="routing")], "dates"),
        "dates": ([AIMessage(content="Anh cho em xin ngày nhận phòng.\nNEXT: WAIT")], None),
        "report": ([AIMessage(content="NEXT: END")], None),
    })
    assert result["summary"] == "Anh cho em xin ngày nhận phòng."
```

### Kiểm chứng live

Sau fix, chạy lại và đo trên CSV: `answer` bắt đầu bằng `NEXT:` phải là **0/32**;
`answer` rỗng phải là **0/32**.

---

## P0-2 — `NEXT` có marker nhưng không parse/không hợp lệ: dừng turn, không đoán node

### Hiện trạng

`runtime/skill_subgraph.py:477` (`_route`):

```python
        if declared is not None and declared in allowed:
            …
            return declared
        nxt = _next_in_order(skill_map, current)
        …
        return nxt if nxt is not None else "END"
```

Mọi declaration không hợp lệ (kể cả do parse lỗi) đều rơi xuống node kế tiếp.
Đây chính là cơ chế biến `confirm-booking` thành `create-booking` và biến terminal
thành đầu tàu của cả map tail.

### Diff đề xuất

1) Thêm key state (`SubgraphState`, `runtime/skill_subgraph.py:213`):

```python
    declared_next: str | None
    unparsed_next: bool
```

2) Trong `_run_node` (`runtime/skill_subgraph.py:468`):

```python
            raw = _last_text(new_messages)
            declared = parse_next(raw)
            unparsed = declared is None and has_next_marker(raw)
            if unparsed:
                errors.append(
                    f"Node '{node.id}' wrote an unparseable NEXT line: "
                    f"{raw.strip().splitlines()[-1][:200]!r}"
                )
            return {
                "messages": new_messages,
                "current": node.id,
                "visited": list(state.get("visited") or []) + [node.id],
                "declared_next": declared,
                "unparsed_next": unparsed,
                "steps": steps,
                "errors": errors,
            }
```

3) Trong `_route`, ngay sau `declared = state.get("declared_next")`:

```python
        if declared is None and state.get("unparsed_next"):
            # The model wrote a NEXT line we could not parse. Advancing
            # would guess a target (2026-11-14: confirm-booking declared
            # "NEXT: WAIT — …" and the run booked instead). Stop the
            # turn and report the error; never guess.
            return "END"
```

Giữ fallback `_next_in_order()` **chỉ** cho trường hợp node không viết `NEXT` nào
(hợp đồng map: "default flow is the next node in list order"). Thêm log để đếm:

```python
        logger.warning(
            "Subgraph: node %r declared no NEXT line; falling back to %r",
            current, nxt,
        )
```

### Test

```python
def test_router_stops_on_unparseable_next():
    flow = _map([
        {"id": "route", "branches": [{"when": "book", "goto": "check"}]},
        {"id": "check", "tools": ["t_check"]},
        {"id": "report", "terminal": True},
    ])
    # A marker with no parseable token is a model error, not a fallback.
    result = _run(flow, {
        "route": ([AIMessage(content="routing\nNEXT:")], None),
        "check": ([AIMessage(content="checked")], "report"),
        "report": ([AIMessage(content="done")], "END"),
    })
    assert result["visited_nodes"] == ["route"]
    assert any("unparseable NEXT" in e for e in result["errors"])


def test_router_missing_next_still_falls_back():
    flow = _map([{"id": "route"}, {"id": "check", "tools": ["t_check"]}])
    result = _run(flow, {
        "route": ([AIMessage(content="routing, no control line")], None),
        "check": ([AIMessage(content="checked")], "END"),
    })
    assert result["visited_nodes"] == ["route", "check"]
```

Test cũ `test_router_falls_back_to_next_in_order_on_bad_next` phải đổi: nó đang
dùng `NEXT: create` (token hợp lệ nhưng không nằm trong `allowed`) — đó là
trường hợp **không hợp lệ**, theo đề xuất phải dừng. Tách thành:

```python
def test_router_stops_on_disallowed_next():
    result = _run(FLOW_MAP, {
        "route": ([AIMessage(content="routing")], "create"),  # not allowed, no history
    })
    assert result["visited_nodes"] == ["route"]
```

> Lưu ý: `test_router_never_books_before_confirmation` và
> `test_router_resume_rule_needs_history` vẫn đúng — chúng dựa trên resume rule,
> không dựa trên fallback.

---

## P0-3 — Node `terminal` không bao giờ rơi sang node kế trong list

### Hiện trạng

`_route` xử lý `declared == "END"` đúng, nhưng nếu node **không khai báo gì** hoặc
khai báo không hợp lệ thì vẫn đi tiếp bằng `_next_in_order()`. Vì map hotel đánh
`terminal: true` cho các node nằm *giữa* danh sách (`policy-lookup` → `booking-lookup`
→ `booking-change` → `task-status`; `report-booking` → `task-details` → …), fallback
nối các flow rời rạc lại với nhau.

Bằng chứng: `exec_e23229bfa98d` đi `report-booking → task-details → find-staff →
create-task → assign-task → report-task` và tạo task khách không yêu cầu.

### Diff đề xuất

Trong `_route`, sau khi tính `allowed`, **trước** `nxt = _next_in_order(...)`:

```python
        node = skill_map.node_by_id(current)
        if node is not None and node.terminal:
            # A terminal node ends its flow. Falling through to the next
            # node in list order is how report-booking walked into
            # task-details and created a task the guest never asked for
            # (2026-11-14). Only an explicit, allowed declaration may
            # continue (e.g. a map that later adds a terminal branch).
            return declared if declared in allowed else "END"
```

### Test

```python
def test_terminal_node_never_falls_through():
    flow = _map([
        {"id": "route", "branches": [{"when": "task", "goto": "details"}]},
        {"id": "details"},
        {"id": "report-task", "terminal": True},
        {"id": "create-task", "tools": ["t_create"]},
    ])
    # report-task declares nothing: the run must stop, not create a task.
    result = _run(flow, {
        "route": ([AIMessage(content="routing")], "details"),
        "details": ([AIMessage(content="got details")], "report-task"),
        "report-task": ([AIMessage(content="NEXT: END — done")], None),
        "create-task": ([AIMessage(content="created")], "END"),
    })
    assert result["visited_nodes"] == ["route", "details", "report-task"]
```

Thêm một test hồi quy cho đúng chuỗi đã xảy ra:

```python
def test_report_booking_does_not_walk_into_task_flow():
    # Map con rút gọn theo đúng thứ tự node của skills/reception/SKILL.md
    ...
```

---

## P1-1 — Budget tool-call theo từng node (và không dùng message cap làm answer)

### Hiện trạng

`runtime/skill_subgraph.py:93`: `MAX_TOOL_CALLS_PER_NODE = 5`, áp cho mọi node.
`policy-lookup` cần tối thiểu: 1 search + 1 read đúng tài liệu, cộng retry khi
search trả nhiều kết quả. Trong v4b:

```
exec_94a042ad21a1: search_policy_documents ×2, read_policy_document ×3 (5 call thật)
answer: "Step 'policy-lookup' stopped: the tool-call budget for this step is exhausted.\nNEXT: WAIT"
FAIL: read the POL-CANCEL document — {"doc_id": "POL-REFUND"}
```

> Ghi chú số liệu: 10 audit row = 5 call thật. `runtime/middleware/callbacks.py`
> ghi 2 row/tool do instrumentation lồng nhau (đã nêu ở mục 12 của
> `scripts/manual/mlflow_report.py`). Không phải gọi MCP hai lần.

### Diff đề xuất (khuyến nghị: khai báo trong map)

1) `runtime/skills/map_plan.py` — `MapNode` (`:46`):

```python
    stop_when: str = ""
    terminal: bool = False
    # Per-node tool-call budget. 0 = use the runner default. Search +
    # read nodes need more than the 1–2 calls a write node needs.
    max_tool_calls: int = 0
```

2) `parse_skill_map` (`:132`, `:192`):

```python
            if key not in {"id", "goal", "tools", "branches", "stop_when",
                           "terminal", "max_tool_calls"}:
                ...
        raw_budget = entry.get("max_tool_calls")
        max_tool_calls = 0
        if raw_budget is not None:
            if isinstance(raw_budget, int) and raw_budget > 0:
                max_tool_calls = raw_budget
            else:
                logger.warning(
                    "Skill %s: node %r max_tool_calls must be a positive int; ignoring",
                    skill_name, node_id,
                )
        nodes.append(
            MapNode(id=node_id, goal=goal, tools=tools, branches=branches,
                    stop_when=stop_when, terminal=terminal,
                    max_tool_calls=max_tool_calls)
        )
```

3) `_run_node`/`node_agent` (`runtime/skill_subgraph.py:368`):

```python
            cap = _NodeToolCallCap(
                max_calls=node.max_tool_calls or max_tool_calls_per_node
            )
```

4) `skills/reception/SKILL.md` — thêm budget cho các node nặng:

```yaml
    - id: policy-lookup
      ...
      tools: [search_policy_documents, read_policy_document]
      terminal: true
      max_tool_calls: 10

    - id: booking-lookup
      ...
      tools: [find_guest, register_guest, get_booking]
      terminal: true
      max_tool_calls: 6
```

`booking-change` (2 tool ghi) và `assign-task` giữ default 5; `find-staff` 2.

### Bản vá nhanh (nếu chưa muốn đổi schema map)

```python
TOOL_CALL_BUDGET_OVERRIDES = {
    "booking-lookup": 6,
    "policy-lookup": 10,
}
...
            cap = _NodeToolCallCap(
                max_calls=TOOL_CALL_BUDGET_OVERRIDES.get(
                    node.id, max_tool_calls_per_node
                )
            )
```

### Test

```python
def test_policy_lookup_gets_a_larger_budget(monkeypatch):
    flow = _map([
        {"id": "route", "branches": [{"when": "policy", "goto": "policy"}]},
        {"id": "policy", "tools": ["t_search", "t_read"], "terminal": True,
         "max_tool_calls": 10},
    ])
    # 6 calls succeed, no budget error.
    ...
    assert result["errors"] == []


def test_node_budget_override_falls_back_to_default():
    ...
```

> Không chỉ nâng budget: cần thêm chỉ dẫn trong `SKILL.md` cho `policy-lookup`:
> "đọc đúng tài liệu có tiêu đề khớp câu hỏi (hủy → `POL-CANCEL`, hoàn tiền →
> `POL-REFUND`)". Nếu không, model vẫn đọc sai tài liệu dù budget 10.

---

## P1-2 — Repeat-guard phải nhớ mọi call đã lỗi trong node, không chỉ call ngay trước

### Hiện trạng

`runtime/skill_subgraph.py:158`:

```python
    def __init__(self, max_calls: int = MAX_TOOL_CALLS_PER_NODE):
        ...
        self._last_failed: tuple[str, str] | None = None
```

Chỉ chặn khi call trùng **liền trước** cũng lỗi. Model xen kẽ hai call lỗi là lách được:

```
find_guest (not found) -> get_booking (not found) -> find_guest (not found) -> …
```

Đúng như `exec_1051f200ea2d`: `find_guest` + `get_booking` xen kẽ, guard 0 lần kích hoạt,
chỉ budget dừng được.

### Diff đề xuất

```python
    def __init__(self, max_calls: int = MAX_TOOL_CALLS_PER_NODE):
        self.max_calls = max_calls
        self.calls = 0
        self.exceeded = False
        self._failed_keys: set[tuple[str, str]] = set()
        self._failed_errors: dict[tuple[str, str], str] = {}

    def reset(self) -> None:
        self.calls = 0
        self.exceeded = False
        self._failed_keys.clear()
        self._failed_errors.clear()

    def wrap_tool_call(self, request, handler):
        ...
        key = _call_key(name, args)
        if key in self._failed_keys:
            return ToolMessage(
                content=(
                    f"You already called '{name}' with these exact arguments "
                    f"earlier in this step and it failed "
                    f"({self._failed_errors[key]}). Do not repeat it — call a "
                    f"different tool, use different arguments, or end your "
                    f"reply with NEXT: WAIT if you need information from the "
                    f"guest first."
                ),
                tool_call_id=tool_call_id,
                status="error",
            )
        result = handler(request)
        error = _error_in_result(result)
        if error is not None:
            self._failed_keys.add(key)
            self._failed_errors[key] = error
        return result
```

### Test

```python
def test_node_tool_call_cap_denies_repeat_after_interleaved_failure():
    cap = _NodeToolCallCap(max_calls=10)
    failing = {"a": "not found"} 
    # call t_x(bad) -> fail; call t_y(bad) -> fail; t_x(bad) again -> denied
    ...
    assert denied.status == "error"
    assert "earlier in this step" in denied.content
```

---

## P1-3 — Answer composition: control plane tách khỏi nội dung trả khách

P0-1 đã xử lý `NEXT:` và dòng "Step … stopped". Bổ sung hai điểm:

1. **Fallback answer** khi toàn bộ run chỉ có control line (ví dụ node đầu đã `WAIT`):
   thêm tham số `fallback_answer: str = ""` cho `run_skill_subgraph`, và
   `runtime/mission_graph.py` truyền câu chờ cấu hình theo mission (không hard-code
   tiếng Việt trong runtime):

```python
# runtime/skill_subgraph.py, sau khi strip:
    if not answer:
        answer = fallback_answer
```

```python
# runtime/mission_graph.py, chỗ gọi run_skill_subgraph:
            outcome = run_skill_subgraph(
                subgraph, mission, input_data,
                ...
                fallback_answer=str(
                    (mission.get("execution") or {}).get("wait_answer") or ""
                ),
            )
```

2. **Không đưa diagnostics vào `summary`**; chúng ở `outcome["errors"]` (đã có sẵn
   từ `NodeToolCallBudgetExceededError`). Chỉ cần đảm bảo `summary` không lấy dòng
   chẩn đoán — `strip_control_lines` đã bắt `Step '…' stopped:`.

### Test

```python
def test_run_uses_fallback_answer_when_only_control_line():
    flow = _map([{"id": "route", "terminal": True}])
    compiled = build_skill_subgraph(flow, tools=[], system_prompt="base",
                                    node_executor=_executor_script(
                                        {"route": ([AIMessage(content="no text")], "WAIT")}))
    result = run_skill_subgraph(
        compiled, {"mission_id": "t", "objective": "t"}, {"message": "hi"},
        skill_maps=[flow], fallback_answer="Em cần thêm thông tin ạ.",
    )
    assert result["summary"] == "Em cần thêm thông tin ạ."
```

---

## P2-1 — `route` nên tất định (classifier bằng code), model chỉ xử lý ca mơ hồ

### Hiện trạng

`route` là node đầu, quyết định toàn bộ spine, nhưng giao cho `qwen2.5-3b`.
Hai lỗi tái diễn:

| Input | Đi đâu | Đúng phải là |
|---|---|---|
| "hỏi giá phòng Deluxe, chưa đặt phòng" | `booking-lookup` | `booking-dates` |
| "phòng 305 hỏng điều hòa, cử người sửa" (`task_full`) | `booking-dates` | `task-details` |

Branch hints trong prompt không đủ; thêm chữ cũng không giải quyết với model 3B.

### Đề xuất

Thêm `runtime/skills/reception_intent.py`:

```python
"""Deterministic first-pass intent router for the reception map.

Only unambiguous shapes are classified; anything else returns None and
the model's `route` node decides. This exists because `route` is the
single point of failure for the whole spine with a 3B model (2026-11-14:
a quote request went to booking-lookup, a room-305 repair went to
booking-dates).
"""

import re

BOOKING_CODE = re.compile(r"\bBK-\d{4}\b", re.I)
PHONE = re.compile(r"\b0\d{9,10}\b")
ROOM = re.compile(r"\b(?:phòng|p\.?)\s*(\d{2,4})\b", re.I)
CANCEL = re.compile(r"\b(hủy|huỷ|đổi ngày|dời ngày|phụ thu)\b", re.I)
POLICY = re.compile(r"\b(chính sách|quy định|phí hủy|mất phí|hoàn tiền)\b", re.I)
TASK = re.compile(
    r"\b(hỏng|sửa|thay|vệ sinh|dọn phòng|buồng phòng|kỹ thuật|"
    r"điều hòa|điều hoà|nước nóng|tivi|wifi|khăn|thêm giường)\b", re.I
)
TASK_STATUS = re.compile(r"\b(tiến độ|task|xong chưa|đến đâu|ai làm)\b", re.I)
QUOTE = re.compile(
    r"\b(giá|bao nhiêu|một đêm|1 đêm|còn phòng|trống không|đặt phòng|"
    r"check-?in|check-?out|nhận phòng|trả phòng)\b", re.I
)


def classify(message: str) -> str | None:
    text = message or ""
    has_room = bool(ROOM.search(text))
    if BOOKING_CODE.search(text) and CANCEL.search(text):
        return "booking-change"
    if BOOKING_CODE.search(text) or PHONE.search(text):
        return "booking-lookup"
    if has_room and TASK.search(text):
        return "task-details"
    if TASK_STATUS.search(text):
        return "task-status"
    if CANCEL.search(text) and POLICY.search(text):
        return "policy-lookup"
    if CANCEL.search(text):
        return "booking-change" if has_room else "policy-lookup"
    if POLICY.search(text):
        return "policy-lookup"
    if TASK.search(text) and has_room:
        return "task-details"
    if QUOTE.search(text):
        # "chưa đặt phòng" is the tell that this is a quote, not a lookup.
        return "booking-dates"
    return None
```

Tích hợp: thêm tham số `start_router=None` cho `build_skill_subgraph`; nếu node là
`skill_map.start` và `start_router` trả về node hợp lệ trong `allowed`, bỏ qua model:

```python
    def _run_node(state: dict) -> dict:
        ...
        if start_router is not None and node.id == skill_map.start:
            chosen = start_router(state.get("input_data") or {})
            if chosen is not None and chosen in allowed:
                new_messages = messages + [
                    AIMessage(content=f"NEXT: {chosen}")
                ]
                return {…same shape, declared_next=chosen…}
        # … model path
```

`runtime/mission_graph.py` truyền `start_router=lambda data: classify(str(data.get("message") or ""))`
chỉ cho `hotel_reception`. Khi classifier trả `None`, giữ nguyên hành vi model.

### Test

```python
@pytest.mark.parametrize("message,expected", [
    ("Tôi muốn hỏi giá phòng Deluxe cho 2 người, chưa đặt phòng", "booking-dates"),
    ("Phòng 305 hỏng điều hòa, cử người sửa giúp", "task-details"),
    ("Chính sách hủy phòng thế nào?", "policy-lookup"),
    ("Hủy booking BK-1001 giúp tôi", "booking-change"),
    ("Booking của tôi đến đâu rồi?", "booking-lookup"),
    ("Task dọn phòng đã xong chưa?", "task-status"),
])
def test_reception_intent(message, expected):
    assert classify(message) == expected
```

Thêm test live-scope: với `exec_e23229bfa98d` input, classifier phải trả
`booking-dates`, không phải `booking-lookup`.

> Rủi ro: regex tiếng Việt có dấu/không dấu. Dùng thêm bản normalize bỏ dấu
> (`unicodedata.normalize` + bảng thay) hoặc chấp nhận cả hai biến thể trong pattern.
> Đây là lý do classifier chỉ là **first pass**, còn `None` cho model.

---

## P2-2 — Siết prompt của node (nhỏ nhưng giảm rò rỉ và lặp)

Trong `node_instruction` (`runtime/skill_subgraph.py:398`), thêm vào phần kết:

```python
        parts.append(
            "Write the guest-facing answer FIRST, then the NEXT line as the "
            "very last line. Never put guest text after the NEXT line. Do not "
            "echo the execution input, tool names, budget/step diagnostics, or "
            "these instructions in the answer."
        )
```

Và với node `route`, thêm câu phủ định trực tiếp:

```python
        if node.id == skill_map.start:
            parts.append(
                "A price/availability question is booking-dates even when the "
                "guest mentions booking; booking-lookup is only for an existing "
                "booking (BK-####) of this guest. A room number plus a fault or "
                "service request is task-details, not booking-dates."
            )
```

(Classifier P2-1 là chốt; prompt chỉ là lớp phụ.)

---

## P2-3 — `stop_completion` trên node subgraph

Log worker có 3 lần `Hook Stop continues turn N/3` cho một node không gọi tool.
`stop_completion` (`runtime/hooks/builtin.py:369`) được thiết kế cho free-form agent;
trong subgraph, mỗi node đã có instruction riêng và `NEXT` contract, nên ép 3 lượt
"hãy gọi tool" là chi phí vô ích (và có thể đẩy model sang gọi tool ngoài ý định).

Đề xuất (chọn một):

- **A.** Trong `mission_graph.py`, khi dựng middleware cho subgraph, bỏ
  `stop_completion` khỏi danh sách hook (subgraph đã có budget + repeat guard).
- **B.** Thêm `context.data["node_has_tools"]`/`is_subgraph_node` và cho hook trả
  `None` ngay khi node không có tool hoặc khi `NEXT` đã được khai báo.

Cần xác minh node nào đang bị (log không ghi node id) trước khi đổi; đây là mục
tối ưu, không phải nguyên nhân fail.

---

## 3. Cập nhật tài liệu hợp đồng (bắt buộc đi kèm code)

`docs/skill-maps.md` — mục router:

- `NEXT: <id>` **được phép có text sau token**; chỉ token đầu tiên được đọc.
- `NEXT` có marker nhưng token không parse/không nằm trong `allowed` → **turn dừng**,
  ghi lỗi; không fallback.
- Không có dòng `NEXT` nào → fallback node kế theo list (hợp đồng cũ), có log warning.
- Node `terminal` không bao giờ rơi sang node kế; chỉ declaration hợp lệ mới đi tiếp.
- Budget tool-call có thể khai báo theo node (`max_tool_calls`), mặc định 5.

`docs/hotel-reception.md` — bảng budget:

| Node | `max_tool_calls` | Lý do |
|---|---|---|
| `policy-lookup` | 10 | search + đọc 2–3 tài liệu + retry |
| `booking-lookup` | 6 | find_guest + get_booking + register_guest |
| các node còn lại | 5 (default) | 1–2 call |

`skills/reception/SKILL.md` — thêm `max_tool_calls` và câu: "trả lời khách trước,
dòng `NEXT:` cuối cùng; không lặp lại chẩn đoán/budget trong câu trả lời".

---

## 4. Kế hoạch test

### Unit (mở rộng `tests/test_skill_subgraph.py`, `tests/test_skill_maps.py`)

| Test | Mục |
|---|---|
| `test_parse_next_tolerates_trailing_explanation` | P0-1 |
| `test_strip_control_lines_removes_next_and_diagnostics` | P0-1 |
| `test_run_keeps_control_line_out_of_answer` | P0-1 |
| `test_router_stops_on_unparseable_next` | P0-2 |
| `test_router_missing_next_still_falls_back` | P0-2 |
| `test_router_stops_on_disallowed_next` (đổi test cũ) | P0-2 |
| `test_terminal_node_never_falls_through` | P0-3 |
| `test_policy_lookup_gets_a_larger_budget` | P1-1 |
| `test_node_budget_override_falls_back_to_default` | P1-1 |
| `test_node_tool_call_cap_denies_repeat_after_interleaved_failure` | P1-2 |
| `test_run_uses_fallback_answer_when_only_control_line` | P1-3 |
| `test_reception_intent` (parametrize 6 ca) | P2-1 |
| `test_reception_map_budgets_parse` | P1-1 + map schema |

Chạy:

```bash
.venv/bin/python -m pytest tests/test_skill_subgraph.py tests/test_skill_maps.py \
    tests/test_hotel_reception.py -q
```

### Live (sau khi merge)

```bash
docker compose stop worker
.venv/bin/celery -A app.worker.celery_app worker --loglevel=info --concurrency=1
.venv/bin/python scripts/mock_hotel_mcp_server.py --reseed --host 0.0.0.0 \
    --allow-host 'host.docker.internal:*'
.venv/bin/python scripts/manual/hotel_reception.py \
    --session-id sess-hotel-manual-v5 --timeout 420
.venv/bin/python scripts/manual/mlflow_report.py \
    --tracking-uri http://localhost:5000 \
    --session-prefix sess-hotel-manual-v5- \
    --out reports/hotel-mlflow-report-v5.md \
    --csv reports/hotel-mlflow-traces-v5.csv
```

### Tiêu chí chấp nhận

| Chỉ số | v4b (baseline) | Mục tiêu v5 |
|---|---|---|
| Suite pass | 109 | ≥ 130 (hết fail cơ học) |
| Answer chỉ là `NEXT:` | 15/32 | **0** |
| Answer có `NEXT:` | 25/32 | **0** |
| Terminal rơi sang flow khác | có (`report-booking → task-details`) | **0** |
| Booking trước xác nhận | có (`confirm-booking → create-booking`) | **0** |
| `policy` đọc đúng `POL-CANCEL` | không | **có** |
| Task flow gọi `list_staff`/`create_task`/`assign_task` | 0/3 | 3/3 |
| Misroute quote → `booking-lookup` | có | **0** |
| Tool loop | cap 15 audit row | cap theo node, không loop |

---

## 5. Rủi ro và lưu ý

1. **P0-2 thay đổi hợp đồng**: một số map/flow có thể dựa vào fallback khi model
   quên `NEXT`. Giữ fallback cho "không có marker", chỉ dừng khi "có marker mà hỏng";
   log warning để theo dõi tần suất.
2. **P0-1 regex rộng hơn** có thể bắt nhầm dòng văn xuôi bắt đầu bằng "NEXT:" —
   hiếm; giới hạn ở đầu dòng + 1 token giảm rủi ro.
3. **P1-1 budget**: nâng budget làm tăng trần thời gian xấu nhất; repeat guard
   (P1-2) là thứ giữ chi phí, budget chỉ là chốt cuối. Đo lại p90 latency.
4. **P2-1 classifier**: false positive sẽ cứng nhắc hơn model; chỉ phân loại ca rõ,
   trả `None` cho phần còn lại, và log `classifier=...` để đo.
5. **`terminal: true` trên `route`**: hiện `NEXT: END` ở route kết thúc turn ngay
   (an toàn, không advance) nhưng nghĩa là route phải luôn chọn branch. Khi có
   classifier P2-1, cân nhắc bỏ `terminal: true` khỏi `route` để tránh END sớm.
6. **Không đổi** `outcome_from_messages` cho free-form agent: sanitize `NEXT` chỉ
   áp trong `run_skill_subgraph`.
7. **Số audit row gấp đôi không phải bug**: đã ghi ở mục 12 của
   `scripts/manual/mlflow_report.py`; đừng dùng audit row để kết luận số call thật.

---

## 6. Checklist thực hiện

- [ ] P0-1 regex + `strip_control_lines` + `_last_answer_text` + sanitize outcome
- [ ] P0-2 `unparsed_next` state + dừng turn + log fallback
- [ ] P0-3 terminal không fallthrough
- [ ] P1-1 `max_tool_calls` trong `MapNode`/`SKILL.md` + override
- [ ] P1-2 repeat guard theo set
- [ ] P1-3 `fallback_answer`
- [ ] P2-1 classifier + `start_router`
- [ ] P2-2 prompt hardening
- [ ] P2-3 đo và tối ưu `stop_completion`
- [ ] Cập nhật `docs/skill-maps.md`, `docs/hotel-reception.md`, `skills/reception/SKILL.md`
- [ ] Unit tests + live suite v5 + MLflow report v5
