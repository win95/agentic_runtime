"""Outcome graders for the hotel manual suite (architecture Phase 1).

The old suite graded the agent by (a) exact tool sequences and (b) keyword
lists on free-form prose. Both are brittle: a correct answer in different
words failed, and a correct-looking answer could pass without the world
changing. Phase 1 replaces them with three grader layers, in priority
order (proposal 'kien-truc-moi-cho-agentic-runtime'):

1. **State verifiers (code, authoritative).** Inspect the world after the
   run through the same MCP read tools the agent uses: no booking was
   created for a sold-out window; the task row in the store is OPEN /
   room 201 / technicians. A state check alone has the do-nothing trap
   (doing nothing leaves the store unchanged, so it "passes" a sold-out
   check) — so every scenario verdict is ``state AND semantic``, the
   tau-bench product ``r = r_action x r_output``.
2. **Semantic checks (interim).** The prose answer is graded on meaning,
   not wording: a small equivalence set of paraphrases for one fact
   ("no rooms available"), matched on normalized text (case, punctuation,
   whitespace, plus a diacritics-insensitive fallback tier). INTERIM: the
   proposal's binary LLM judge replaces these once the judge model is
   chosen (open decision — see the proposal page). The set is data, kept
   small, and every entry is a paraphrase of the same fact, never a
   growing keyword list. Known weakness, stated plainly: substring
   matching cannot see negation ("chưa hết phòng" contains "hết phòng"),
   which is exactly what the judge will fix.
3. **Trajectory diagnostics (advisory only).** Required/forbidden tool
   sets and pairwise ordering constraints, reported as warnings, never
   as pass/fail. A step is a genuine requirement only when Phase 3 makes
   it a data dependency or enforces it in the runtime gate; until then
   the trajectory is diagnostic evidence, not the verdict.

Pure functions only: no API, DB, or LLM access. The live suite passes in
the answer text, the audit calls, and a ``get_booking`` callable; pytest
drives the same functions with fakes. Negative controls live in
``tests/test_manual_verifiers.py``: a do-nothing run must fail every
verdict, and a gold run (right calls + right state + ideal prose) must
pass all of them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    """One grader's answer: pass/fail plus the evidence for the report."""

    passed: bool
    detail: str


# --- Text normalization ------------------------------------------------------


def strip_diacritics(text: str) -> str:
    """Drop combining marks; 'đ'/'Đ' have no decomposition, map by hand."""
    decomposed = unicodedata.normalize("NFD", text or "")
    stripped = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    return stripped.replace("đ", "d").replace("Đ", "D")


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace (diacritics kept)."""
    lowered = (text or "").lower()
    no_punct = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", no_punct).strip()


def _normalize_set(phrases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalize_answer(p) for p in phrases)


# --- Semantic checks (interim — see module docstring) -------------------------


# Every entry is a paraphrase of the same fact: "no rooms are available".
# Kept tight on purpose: bare "hết" is excluded (negation "chưa hết"
# would match), and bare "0 phòng" is excluded ("10 phòng" contains it).
UNAVAILABILITY_PHRASES = _normalize_set((
    "không có phòng",
    "không còn phòng",
    "hết phòng",
    "không còn trống",
    "không còn phòng trống",
    "đã kín phòng",
    "sold out",
    "fully booked",
    "no rooms available",
    "no room available",
    "no vacant rooms",
    "all booked",
))

# One fact per family; both must be present. Numbers cannot be paraphrased
# away as freely as prose, so these families stay small by design.
POLICY_WINDOW_PHRASES = _normalize_set((
    "3 ngày",
    "ba ngày",
    "3 days",
    "three days",
))
POLICY_FEE_PHRASES = _normalize_set((
    "1 đêm",
    "một đêm",
    "1 dem",
    "1 night",
    "one night",
))


def _contains_phrase(answer: str, phrases: tuple[str, ...]) -> str | None:
    """First matching phrase, or None. Tier 1 keeps diacritics; tier 2 drops them."""
    normalized = normalize_answer(answer)
    for phrase in phrases:
        if phrase and phrase in normalized:
            return phrase
    stripped = strip_diacritics(normalized)
    for phrase in phrases:
        if phrase and strip_diacritics(phrase) in stripped:
            return phrase
    return None


def check_unavailability(answer: str) -> Verdict:
    """Does the answer convey that no rooms are available?"""
    matched = _contains_phrase(answer, UNAVAILABILITY_PHRASES)
    if matched:
        return Verdict(True, f"conveys unavailability ({matched!r})")
    return Verdict(False, "no unavailability phrase found")


def check_policy_facts(answer: str) -> Verdict:
    """Does the answer keep both policy facts (3-day window + 1-night fee)?"""
    window = _contains_phrase(answer, POLICY_WINDOW_PHRASES)
    fee = _contains_phrase(answer, POLICY_FEE_PHRASES)
    missing = [
        name for name, hit in (("3-day window", window), ("1-night fee", fee))
        if not hit
    ]
    if missing:
        return Verdict(False, f"missing policy fact(s): {', '.join(missing)}")
    return Verdict(True, f"keeps 3-day window ({window!r}) + 1-night fee ({fee!r})")


# --- State verifiers -----------------------------------------------------------


def find_bookings_by_window(
    get_booking,
    room_type: str,
    check_in: str,
    check_out: str,
    start: int = 1004,
    end: int = 2000,
) -> list[dict]:
    """All bookings for one window, via the same read tool the agent uses.

    ``get_booking`` takes a booking id and returns the parsed tool result
    (``{"error": ...}`` when the id does not exist). Booking ids are
    contiguous from BK-1004 (the counter after the seed), so the scan
    stops at the first missing id — the same convention as the suite's
    ``_find_created_booking``. Seeded bookings (BK-1001..1003) are
    outside the scan range by design.
    """
    found: list[dict] = []
    for number in range(start, end):
        candidate = f"BK-{number}"
        booking = get_booking(candidate) or {}
        if booking.get("error"):
            break
        if (
            str(booking.get("room_type") or "").casefold()
            == room_type.casefold()
            and booking.get("check_in") == check_in
            and booking.get("check_out") == check_out
        ):
            found.append(booking)
    return found


def verify_no_new_booking(bookings: list[dict]) -> Verdict:
    """Pass when no booking exists for the window (nothing was created)."""
    if bookings:
        codes = ", ".join(
            str(b.get("booking_id") or b.get("code") or "?") for b in bookings
        )
        return Verdict(False, f"booking(s) created for the window: {codes}")
    return Verdict(True, "no booking created for the window")


def verify_task_row(
    task: dict | None, room: str = "201", team: str = "technicians"
) -> Verdict:
    """Pass when the store's task row is OPEN for the right room + team."""
    if not task:
        return Verdict(False, "no task row found in the store")
    if (
        task.get("status") == "OPEN"
        and str(task.get("room")) == room
        and str(task.get("team") or "").casefold() == team.casefold()
    ):
        return Verdict(True, "MCP task is OPEN / room 201 / technicians")
    return Verdict(
        False,
        f"task row mismatch: status={task.get('status')} "
        f"room={task.get('room')} team={task.get('team')}",
    )


# --- Scenario verdicts (state AND semantic) ------------------------------------


def soldout_verdict(answer: str, bookings: list[dict]) -> Verdict:
    """Sold-out gate: no booking created AND the answer says so.

    Either half alone is insufficient: the state check passes for a
    do-nothing run, and the semantic check passes for an answer that
    claims sold-out while the agent secretly booked. The product of
    the two is the verdict.
    """
    state = verify_no_new_booking(bookings)
    meaning = check_unavailability(answer)
    if state.passed and meaning.passed:
        return Verdict(True, f"{state.detail}; {meaning.detail}")
    reasons = "; ".join(
        d for v, d in ((state, state.detail), (meaning, meaning.detail))
        if not v.passed
    )
    return Verdict(False, reasons)


def policy_verdict(answer: str, read_done: bool) -> Verdict:
    """Policy gate: both facts in the answer AND the document was read."""
    facts = check_policy_facts(answer)
    if facts.passed and read_done:
        return Verdict(True, f"{facts.detail}; POL-CANCEL was read")
    reasons = []
    if not facts.passed:
        reasons.append(facts.detail)
    if not read_done:
        reasons.append("read_policy_document was not called")
    return Verdict(False, "; ".join(reasons))


# --- Trajectory diagnostics (advisory only — never the verdict) ------------------


def trajectory_findings(
    calls: list[dict],
    required: tuple[str, ...] = (),
    forbidden: tuple[str, ...] = (),
    order: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    """Diagnostic findings about a tool trace; empty means clean.

    ``calls`` are audit rows (``{"tool": ..., "status": ...}``). ``required``
    tools must appear with status ``"success"``; ``forbidden`` tools must
    not appear at all; each ``(before, after)`` pair requires the first
    successful ``before`` to precede the first successful ``after``.
    """
    findings: list[str] = []
    succeeded: dict[str, int] = {}
    attempted: set[str] = set()
    for index, call in enumerate(calls):
        attempted.add(call.get("tool", ""))
        if call.get("status") == "success":
            succeeded.setdefault(call.get("tool", ""), index)
    for tool in required:
        if tool not in succeeded:
            findings.append(f"required tool {tool!r} was not called successfully")
    for tool in forbidden:
        if tool in attempted:
            findings.append(f"forbidden tool {tool!r} was attempted")
    for before, after in order:
        if before in succeeded and after in succeeded:
            if succeeded[before] > succeeded[after]:
                findings.append(
                    f"{after!r} ran before {before!r} "
                    f"(positions {succeeded[after]} < {succeeded[before]})"
                )
    return findings
