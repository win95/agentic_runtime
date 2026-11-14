"""Phase 4 — QueryPlan: link + intent + gated expansion, one object.

Pilot mission: ``fault_analysis`` only. Fake-only in this VM: no
live LLM, Qdrant, or Neo4j; the generators are pluggable and the
live ones are built lazily (see ``runtime/retrieval_expansion.py``).

A QueryPlan is the per-question understanding that the retriever
may use. It is built by ``build_query_plan`` in four steps:

(a) linking — ``gazetteer_seeds`` (exact + Phase 4 normalisation,
    ``runtime/retrieval_graph.py``);
(b) intent — ``score_intents`` (``runtime/retrieval_intent.py``,
    fusion-style, deterministic, no LLM);
(c) decomposition — ``decompose_question``
    (``runtime/retrieval_decompose.py``), which judges whether a
    multi-part question's parts are RELATED (shared neighbourhood)
    or INDEPENDENT;
(d) gating — ``gating_decision`` on the base retriever's
    PRE-fusion candidate rows (``runtime/retrieval_expansion.py``:
    top-1 < ``LOW_TOP1_VEC`` = 0.35, or margin <
    ``SMALL_MARGIN_VEC`` = 0.15, or gazetteer miss).

GATING INTERPRETATION (the decision this module documents)
----------------------------------------------------------
Gating controls the expensive / risky parts ONLY — rewrite
generation and decomposition-driven expansion. Intent scoring and
linking are cheap and deterministic, so they run ALWAYS: a
gated-off plan still carries the scored intent and the linked
nodes as-is, plus NO rewrite variants (``gated_off=True``,
``fallback_reason="confident — no expansion"``). This satisfies
"confident lookups pass through untouched" (no rewrite, no
expansion) WITHOUT mislabelling a confident question: a
confident comparison keeps its scored intent (``comparison``),
it is not relabelled ``lookup`` just because it did not expand.

Two consequences, both deliberate and different from Phase 3:

* In Phase 3, decomposition bypassed gating: a detected
  multi-part question was decomposed even when confident
  (C02/C03/C05, M07). In Phase 4 the gate is single: a
  gated-off plan has NO sub-questions either — the
  decomposition generator (a live LLM call) is not invoked for
  a question the vector arm already answers confidently.
  Detection itself (``detect_multi_part``, cheap regex) still
  runs always and is recorded on the plan
  (``is_multi_part`` / ``multi_part_reasons``), so the audit
  trail shows what was NOT expanded and why.
* When the gate DOES open, decomposition takes precedence over
  paraphrasing (the Phase 3 pattern): if the decomposition
  generator splits the question, the plan carries the
  sub-questions and no rewrite variants; paraphrases are the
  fallback when the question is not multi-part or the generator
  cannot split it (C02/C04/C05 under the stub splitter).

When there is no retriever to gate on (``retriever=None``),
gating falls back to gazetteer-miss-only: the plan expands iff
linking found nothing. The unmeasured vector values are
``None`` in ``plan.gating`` (never invented zeros), and the
fallback is named in the gating reasons
(``["gazetteer_miss"]``).

When linking fails entirely (``linked_nodes == []``) the plan
says so explicitly: ``fallback_reason="no confident link"``,
and it carries NO rewrite variants, ever (guardrail (c) in
``check_rewrite_variants``). Decomposition sub-questions MAY
still be present in that case — they are retrieved as written
through the vector arm, and the linking-failure rule covers
rewrites only, not splits.

Rewrite guardrails (``check_rewrite_variants``)
-----------------------------------------------
Pure checks, no LLM. A variant is KEPT only if it passes all of:

(a) exact-preservation floor — it links to every seed the
    original linked to (``gazetteer_seeds`` on both, so a
    normalised re-mention counts), it keeps every fault code
    the original names (``FAULT_CODE_RE`` from
    ``retrieval_expansion``, imported not copied), and it keeps
    every number-with-unit the original names (``NUMBER_RE``
    from ``retrieval_intent``, imported not copied). Known
    limit: bare numbers with no unit ("20 open/close cycles",
    "8 faults") are NOT in the floor — ``NUMBER_RE`` only
    matches a number with a unit — so a variant dropping one
    passes (a). The floor is what the intent module can prove,
    not every number in the sentence.
(b) drift cap — it adds no entity outside the original question
    plus its 1-hop neighbourhood: ``gazetteer_seeds(variant)``
    must be a subset of the original seeds union
    ``expand_one_hop(original seeds)``. The same subset rule
    applies to the two lexical sets the preservation floor
    reads: a variant may not name a fault code the original did
    not name, nor a number-with-unit the original did not name
    (a variant inventing "92C" is drift, even though "92C" is
    not a gazetteer node — this is an extension of (b) beyond
    gazetteer entities, same reason code ``drift``).
(c) linking-failure rule — if the plan has no seeds, every
    variant is dropped and the original is used verbatim.
(d) cap — at most ``MAX_VARIANTS`` (3) variants are kept, in
    generator order; the rest are dropped as ``over_cap``.

HyDE is EXPLICITLY EXCLUDED: no hypothetical-document
generation anywhere in this module. Variants come only from
the existing stub / live paraphrase generators.

``check_rewrite_variants`` returns ``(kept, dropped, reasons)``
where ``reasons`` maps each dropped variant string to its list
of reason codes: ``missing_entities``, ``missing_fault_codes``,
``missing_numbers``, ``drift``, ``no_seeds``, ``over_cap``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from runtime.retrieval_decompose import decompose_question, detect_multi_part
from runtime.retrieval_expansion import (
    FAULT_CODE_RE,
    MAX_VARIANTS,
    gating_decision,
)
from runtime.retrieval_graph import (
    build_demo_adjacency,
    expand_one_hop,
    gazetteer_seeds,
)
from runtime.retrieval_intent import NUMBER_RE, score_intents

logger = logging.getLogger(__name__)

__all__ = [
    "QueryPlan",
    "build_query_plan",
    "check_rewrite_variants",
]

# Fallback reasons (exact strings; the eval greps for them).
FALLBACK_CONFIDENT = "confident — no expansion"
FALLBACK_NO_LINK = "no confident link"
FALLBACK_NO_VARIANTS = "no usable variants"
FALLBACK_ALL_DROPPED = "all rewrite variants dropped by guardrails"


@dataclass
class QueryPlan:
    """Per-question understanding for the retriever seam.

    ``intent`` / ``confidence``: the scored intent and its margin
    (winner minus runner-up final, from ``score_intents``) — always
    present, even on a gated-off plan. ``linked_nodes``: gazetteer
    seeds ``[{"type", "key", "confidence"}]``. ``sub_questions``:
    decomposition parts ``[{"text", "linked_nodes": [keys]}]`` —
    empty unless the gate opened AND the generator split the
    question. ``rewrite_variants``: guardrail-kept paraphrases —
    empty unless the gate opened, seeds exist, and no
    decomposition applied. ``gated_off``: True when gating said
    "no expansion". ``fallback_reason``: why there is no expansion
    (``None`` when the plan expands normally).
    Extra audit fields: ``intent_scores`` (full per-intent
    breakdown), ``gating`` (the ``gating_decision`` dict),
    ``is_multi_part`` / ``multi_part_reasons`` (cheap detection,
    always computed), ``related`` / ``shared_nodes`` (the
    decomposition verdict, ``None``/``[]`` when no decomposition
    ran).
    """

    intent: str
    linked_nodes: list[dict] = field(default_factory=list)
    sub_questions: list[dict] = field(default_factory=list)
    rewrite_variants: list[str] = field(default_factory=list)
    confidence: float = 0.0
    gated_off: bool = False
    fallback_reason: str | None = None
    intent_scores: dict = field(default_factory=dict)
    gating: dict = field(default_factory=dict)
    is_multi_part: bool = False
    multi_part_reasons: list[str] = field(default_factory=list)
    related: bool | None = None
    shared_nodes: list[str] = field(default_factory=list)


# --- Normalisation helpers (for the preservation floor) -------------------

def _norm_code(text: str) -> str:
    """Lowercase, diacritics stripped, separators removed.

    "F-204", "F 204" and "F204" all normalise to "f204", so a
    variant that re-spells a code with a different separator
    still counts as preserving it.
    """
    from runtime.retrieval_graph import _strip_diacritics

    return re.sub(r"[-_\s]", "", _strip_diacritics(text).lower())


def _norm_number(text: str) -> str:
    """Lowercase, whitespace removed (units keep their spelling).

    "6.8 mm/s" -> "6.8mm/s"; "92C" -> "92c". The degree sign is
    kept: NUMBER_RE matches it, and both sides normalise the
    same way.
    """
    return re.sub(r"\s+", "", text.lower())


# --- Rewrite guardrails ------------------------------------------------------

def check_rewrite_variants(
    question: str,
    variants: list[str],
    *,
    linked_nodes: list[dict],
    adjacency: dict | None = None,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Guardrail-check paraphrase variants against the original.

    Returns ``(kept, dropped, reasons)``; ``reasons`` maps each
    dropped variant string to its reason codes (see the module
    docstring). A variant identical to the original passes
    trivially — the caller (``build_query_plan``) filters those
    out before calling, as Phase 3 does; the check itself does
    not special-case them. Deterministic, no LLM.
    """
    adjacency = adjacency if adjacency is not None else build_demo_adjacency()
    kept: list[str] = []
    dropped: list[str] = []
    reasons: dict[str, list[str]] = {}
    if not linked_nodes and variants:
        # Rule (c): no seeds — every variant is dropped, the
        # original is used verbatim. No per-variant analysis.
        for variant in variants:
            dropped.append(variant)
            reasons[variant] = ["no_seeds"]
        return kept, dropped, reasons

    original_keys = {s["key"] for s in linked_nodes}
    allowed_keys = set(original_keys)
    if linked_nodes:
        allowed_keys |= set(
            expand_one_hop(linked_nodes, adjacency).get("nodes", {})
        )
    fault_codes = {m.group(0) for m in FAULT_CODE_RE.finditer(question or "")}
    norm_fault_codes = {_norm_code(code) for code in fault_codes}
    numbers = {_norm_number(m) for m in NUMBER_RE.findall(question or "")}

    for variant in variants:
        variant_reasons: list[str] = []
        if len(kept) >= MAX_VARIANTS:
            variant_reasons.append("over_cap")
            dropped.append(variant)
            reasons[variant] = variant_reasons
            continue
        # (a) preservation floor.
        variant_keys = {s["key"] for s in gazetteer_seeds(variant)}
        if not original_keys <= variant_keys:
            variant_reasons.append("missing_entities")
        norm_variant_code = _norm_code(variant)
        if any(code not in norm_variant_code for code in norm_fault_codes):
            variant_reasons.append("missing_fault_codes")
        norm_variant_number = _norm_number(variant)
        if any(number not in norm_variant_number for number in numbers):
            variant_reasons.append("missing_numbers")
        # (b) drift cap: no new entity, fault code, or number.
        if not variant_keys <= allowed_keys:
            variant_reasons.append("drift")
        variant_fault_codes = {
            _norm_code(m.group(0)) for m in FAULT_CODE_RE.finditer(variant)
        }
        if not variant_fault_codes <= norm_fault_codes:
            if "drift" not in variant_reasons:
                variant_reasons.append("drift")
        variant_numbers = {
            _norm_number(m) for m in NUMBER_RE.findall(variant)
        }
        if not variant_numbers <= numbers:
            if "drift" not in variant_reasons:
                variant_reasons.append("drift")
        if variant_reasons:
            dropped.append(variant)
            reasons[variant] = variant_reasons
        else:
            kept.append(variant)
    return kept, dropped, reasons


# --- QueryPlan construction ----------------------------------------------------

def build_query_plan(
    question: str,
    *,
    retriever=None,
    embeddings=None,
    adjacency: dict | None = None,
    paraphrase_generator=None,
    decomposition_generator=None,
) -> QueryPlan:
    """Build the QueryPlan for one question.

    Order: link + intent (always) -> gate -> expand only if the
    gate opens. ``retriever`` is a ``HybridRetriever`` (or None);
    its ``_candidate_rows`` supply the pre-fusion rows gating
    reads. ``embeddings`` / ``adjacency`` are passed through to
    ``score_intents`` / the guardrails (defaults: the fake
    embeddings and the demo adjacency). Generators are pluggable
    callables; when omitted the LIVE generators apply (lazy
    model build — no LLM call until a gated query needs one,
    and no call at all for a gated-off question).
    """
    adjacency = adjacency if adjacency is not None else build_demo_adjacency()
    seeds = gazetteer_seeds(question or "")
    intent_result = score_intents(
        question or "", seeds=seeds, embeddings=embeddings, adjacency=adjacency
    )
    detection = detect_multi_part(question or "")

    # Gating: on the base retriever's pre-fusion rows when one is
    # available, else gazetteer-miss-only (vector values None =
    # unmeasured, never invented).
    if retriever is not None:
        base_rows, _s, _e, _gm = retriever._candidate_rows(question or "")
        gating = gating_decision(question or "", base_rows)
    else:
        gating = {
            "triggered": not seeds,
            "reasons": ["gazetteer_miss"] if not seeds else [],
            "top1_vec": None,
            "top2_vec": None,
            "margin_vec": None,
            "has_exact_match": bool(seeds),
        }
    gated_off = not gating["triggered"]

    plan = QueryPlan(
        intent=intent_result["intent"],
        linked_nodes=seeds,
        confidence=intent_result["margin"],
        gated_off=gated_off,
        intent_scores=intent_result["scores"],
        gating=gating,
        is_multi_part=detection["is_multi_part"],
        multi_part_reasons=detection["reasons"],
    )

    if gated_off:
        # Cheap + deterministic parts are already on the plan; the
        # expensive/risky parts (generators) are not invoked.
        plan.fallback_reason = FALLBACK_CONFIDENT
        return plan

    if not seeds:
        # Linking failed entirely: no rewrite variants, ever — but
        # decomposition may still split (retrieved as written
        # through the vector arm; see module docstring).
        plan.fallback_reason = FALLBACK_NO_LINK

    # Gate open: decomposition first (Phase 3 precedence).
    sub_questions: list[dict] = []
    related: bool | None = None
    shared_nodes: list[str] = []
    if detection["is_multi_part"]:
        decomposition = decompose_question(
            question or "",
            decomposition_generator=decomposition_generator,
            adjacency=adjacency,
        )
        sub_questions = decomposition["sub_questions"]
        related = decomposition["related"]
        shared_nodes = decomposition["shared_nodes"]
    if sub_questions:
        plan.sub_questions = sub_questions
        plan.related = related
        plan.shared_nodes = shared_nodes
        # If linking failed, fallback_reason stays FALLBACK_NO_LINK:
        # it names the missing link (no rewrite variants were ever
        # built), even though decomposition still split the question.
        if plan.fallback_reason != FALLBACK_NO_LINK:
            plan.fallback_reason = None
        return plan

    # No decomposition: paraphrase fallback (seeds required).
    if not seeds:
        return plan
    if paraphrase_generator is None:
        from runtime.retrieval_expansion import build_live_variant_generator

        paraphrase_generator = build_live_variant_generator()
    raw_variants = list(paraphrase_generator(question or "") or [])
    raw_variants = [
        v.strip()
        for v in raw_variants
        if v and v.strip()
        and v.strip().lower() != (question or "").strip().lower()
    ]
    raw_variants = list(dict.fromkeys(raw_variants))[:MAX_VARIANTS]
    if not raw_variants:
        plan.fallback_reason = FALLBACK_NO_VARIANTS
        return plan
    kept, _dropped, _reasons = check_rewrite_variants(
        question or "", raw_variants, linked_nodes=seeds, adjacency=adjacency
    )
    plan.rewrite_variants = kept
    plan.fallback_reason = None if kept else FALLBACK_ALL_DROPPED
    return plan
