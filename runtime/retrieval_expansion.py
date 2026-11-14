"""Phase 3 — gated query expansion + sub-question decomposition
(report §2, §8 Phase 3; locked decisions: multi-query gated only,
HyDE deferred, decomposition gated only, no reranker model).

This module wraps ``HybridRetriever`` (Phases 1–2) and changes
nothing when expansion does not trigger: the base ranking is
returned untouched. When it triggers, each variant / sub-question
is retrieved through the SAME base retriever
(``_candidate_rows`` — dense top-40 + 1-hop graph per variant),
the candidate rows are merged, and the merged set is reranked by
the SAME fusion (``_rank_candidates``: RRF k=60 -> top-30 ->
weighted fusion -> guardrails -> top-6). Expansion never invents
candidates and never bypasses the fusion.

Gating (deterministic, named constants — no magic numbers):

* ``LOW_TOP1_VEC = 0.35`` — the original question's top-1 vector
  score (over the PRE-fusion candidate rows, i.e. the vector arm's
  top hit before reranking) is below this: the vector arm has no
  confident hit. (Measured on the fake setups: confident lookups
  sit at 0.50–0.66 with margin 0.27–0.48; the weak cases — L06
  0.21, A02 0.23, A03 0.17, M01/C01/C06 0.29–0.32 — sit below
  0.35. Limit: gating sees the vector arm only, so a case whose
  vector top-1 is confident but whose gold docs rank low AFTER
  fusion (A01/A04, top-1 0.39–0.46, nDCG@6 0.838) does not expand.)
* ``SMALL_MARGIN_VEC = 0.15`` — top-1 minus top-2 vector score is
  below this: the vector arm cannot tell the top two apart
  (M01 0.10, M04 0.08, M05 0.03, C01 0.03, A03 0.01 on the blurred
  setup).
* Gazetteer miss — ``gazetteer_seeds(question)`` is empty, i.e. the
  question names no exact asset/component/building/document/fault
  code (L06, L11, A02, A03, U04). The graph arm has no seeds, so a
  paraphrase is the only way a variant might surface one — though
  the generator is forbidden from adding codes, so this trigger
  mostly buys a second vector wording, not new seeds.

Expand iff top-1 < LOW_TOP1_VEC **or** margin < SMALL_MARGIN_VEC
**or** gazetteer miss. Thresholds are cosine-similarity values on
the configured embedding model; re-tune them live (the fake BoW
scores are not a live calibration — see docs/hybrid-retrieval.md).

Decomposition detection (deterministic heuristic first — §2 of the
task, report §2 "Sub-question decomposition") fires iff ANY of:

(a) >= 2 distinct asset codes (``O21`` + ``O22`` — C01–C06);
(b) >= 2 distinct fault codes (``F-410`` + ``F-207`` — C02);
(c) a comparison word: vs / versus / compare(d) / comparison /
    hotter / colder / higher / lower / difference(s) (C05);
(d) >= 2 distinct wh-words (which/what/how/...) AND >= 2 distinct
    codes of any kind (M01–M04: "Which document …, and what …").

Limits, documented honestly: the rule UNDER-triggers when the
second part names no code (M06: "Which building is O21 in, and
which document covers its motor overheating?" — 1 code, no
comparison word, not detected) and OVER-triggers on multi-hop
questions that name asset + component + fault (M01–M04 are
two-fact questions, so the split is defensible, but they are not
comparisons). "List all …" aggregate requests (A01) are NOT
decomposed: one list request, one retrieval — splitting it into
per-code sub-questions would need codes the question does not
name. A learned router (or the LLM itself judging multi-part)
is the live follow-up, not built here.

Generators are PLUGGABLE callables ``question -> list[str]``:

* ``build_live_variant_generator`` / ``build_live_decomposition_generator``
  — the default live implementations. They call the chat model
  (``runtime/llm_factory.build_chat_model``, LiteLLM -> vLLM 3B)
  with a constrained few-shot prompt ("paraphrase only; do not add
  facts or asset names not in the question"). The model is built
  LAZILY on first use, never at import/construction, so importing
  this module never touches LiteLLM config. Verified only with
  stub generators in tests/eval — no live LLM exists in this VM.
* ``stub_paraphrase_variants`` / ``stub_decompose_question`` —
  deterministic FAKE generators for tests and the fake eval only.
  They are not paraphrases a model would write; they stand in for
  it so the merge/rerank plumbing is measurable without a model.

Merge rule (prototyped on the golden set, both fake setups):
per candidate, ``vec_score`` and ``graph_score`` are the MEAN
across the original question + all variants, ``exact`` is the MAX,
and the §4.1 feature row (G1–G8, best path, arms) is taken from
the variant row with the highest ``graph_score`` (the strongest
graph evidence). The mean — not the max — is deliberate: a max
lets one drifting paraphrase single-handedly promote a doc, and
on the blurred setup max-merge regressed nDCG@6 0.958 -> 0.957
while mean-merge held it at 0.958 (BoW: 0.958 -> 0.965). One
variant's retrieval can still surface a doc the original missed
(it enters the union with its own scores), but no single variant
dominates. Provenance per candidate: the variants whose fused
top-6 contained it (``provenance_variants``).

No LTR training lives here (or anywhere in this repo): the
harness logs the feature rows (see ``runtime/retrieval_harness.
ltr_feature_rows``); training is PENDING DATA — see
docs/hybrid-retrieval-phase3.md §4.
"""

from __future__ import annotations

import logging
import re

from runtime.retrieval_graph import gazetteer_seeds
from runtime.retrieval_hybrid import trace_feature_rows

logger = logging.getLogger(__name__)

# --- Gating thresholds (named constants; see module docstring) --------
MAX_VARIANTS = 3
LOW_TOP1_VEC = 0.35
SMALL_MARGIN_VEC = 0.15

# --- Decomposition-detection vocabulary ---------------------------------
COMPARISON_RE = re.compile(
    r"\b(vs|versus|compare|compared|comparison|hotter|colder|higher|lower"
    r"|difference|differences)\b",
    re.IGNORECASE,
)
WH_RE = re.compile(r"\b(which|what|how|where|when|who|why)\b", re.IGNORECASE)
ASSET_CODE_RE = re.compile(r"\b(o\d{2})\b", re.IGNORECASE)
FAULT_CODE_RE = re.compile(r"\b(f-\d{3})\b", re.IGNORECASE)
ANY_CODE_RE = re.compile(
    r"\b(o\d{2}|f-\d{3}|motor-m\d|door-sensor-ds\d|controller-c\d"
    r"|brake-bk\d|building-b\d|doc-[a-z0-9-]+)\b",
    re.IGNORECASE,
)

# --- Live generator prompts (few-shot, constrained) ----------------------
EXPANSION_PROMPT = """You paraphrase a maintenance question so a search engine can find
the same answer with different words.

Rules:
- Paraphrase only. Do NOT add facts, asset names, codes, temperatures,
  or numbers that are not in the question.
- Keep each paraphrase a single question, about the same fault or document.
- Return exactly 3 paraphrases, one per line, with no numbering, no bullets, and no extra text.

Question: Elevator O21's traction motor M2 overheated (F-204). Which document covers the remedy?
Paraphrases:
What guide describes the fix for the overheated traction motor M2 on elevator O21 (F-204)?
Which manual covers the remedy when the traction motor M2 on O21 overheats (F-204)?
For elevator O21, where is the remedy documented for traction motor M2 overheating (F-204)?

Question: {question}
Paraphrases:"""

DECOMPOSITION_PROMPT = """You split a multi-part maintenance question into standalone
sub-questions, one per fact the question asks for.

Rules:
- Each sub-question must be answerable on its own. Copy asset names and codes exactly as written.
- Do NOT add facts, asset names, codes, or numbers that are not in the question.
- Return at most 3 sub-questions, one per line, with no numbering, no bullets, and no extra text.
- If the question has only one part, return it unchanged on a single line.

Question: O21's motor reached 92C (F-204) and O22's motor reached 88.5C (F-206).
Which had the hotter motor?
Sub-questions:
Which elevator had the hotter motor, O21 at 92C (F-204) or O22 at 88.5C (F-206)?
Which document covers motor overheating for O21 and O22?

Question: {question}
Sub-questions:"""


# --- Gating ---------------------------------------------------------------

def has_exact_match(question: str) -> bool:
    """True when the question names at least one gazetteer code.

    The gazetteer is the deterministic seed linker
    (``runtime/retrieval_graph.gazetteer_seeds``): asset / component /
    building / document / fault codes. A miss means the graph arm
    has no seeds for this question.
    """
    return bool(gazetteer_seeds(question))


def gating_decision(question: str, candidate_rows: list[dict]) -> dict:
    """Should this question be expanded? Deterministic gating rule.

    ``candidate_rows`` are the base retriever's pre-fusion rows for
    the ORIGINAL question (each with ``vec_score``). Expand iff any
    of: top-1 vector score < ``LOW_TOP1_VEC``; top-1/top-2 margin <
    ``SMALL_MARGIN_VEC``; gazetteer miss (no exact code match).
    Returns the decision plus the measured values, so the eval can
    report WHY each query expanded.
    """
    scores = sorted(
        (float(r.get("vec_score", 0.0)) for r in candidate_rows), reverse=True
    )
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2
    exact = has_exact_match(question)
    reasons: list[str] = []
    if top1 < LOW_TOP1_VEC:
        reasons.append("low_top1")
    if margin < SMALL_MARGIN_VEC:
        reasons.append("small_margin")
    if not exact:
        reasons.append("gazetteer_miss")
    return {
        "triggered": bool(reasons),
        "reasons": reasons,
        "top1_vec": round(top1, 4),
        "top2_vec": round(top2, 4),
        "margin_vec": round(margin, 4),
        "has_exact_match": exact,
    }


# --- Decomposition detection (deterministic heuristic) --------------------

def detect_multi_part(question: str) -> dict:
    """Is this a multi-part question? Returns decision + reasons.

    Rules (a)–(d) from the module docstring. ``reasons`` names the
    rule(s) that fired, so over/under-triggering is auditable per
    query in the eval trace.
    """
    text = question or ""
    assets = {m.group(1).upper() for m in ASSET_CODE_RE.finditer(text)}
    faults = {m.group(0).upper() for m in FAULT_CODE_RE.finditer(text)}
    codes = {m.group(1).lower() for m in ANY_CODE_RE.finditer(text)}
    wh_words = {m.group(1).lower() for m in WH_RE.finditer(text)}
    reasons: list[str] = []
    if len(assets) >= 2:
        reasons.append("two_assets")
    if len(faults) >= 2:
        reasons.append("two_fault_codes")
    if COMPARISON_RE.search(text):
        reasons.append("comparison_word")
    if len(wh_words) >= 2 and len(codes) >= 2:
        reasons.append("two_wh_words_and_codes")
    return {"is_multi_part": bool(reasons), "reasons": reasons}


# --- Stub (fake) generators — tests / fake eval only ----------------------

_SYNONYM_SWAPS = [
    ("overheated", "overheating"),
    ("overheating", "overheated"),
    ("document", "guide"),
    ("fault", "failure"),
    ("covers", "describes"),
    ("elevator", "lift"),
    ("Which document", "What guide"),
    ("which document", "what guide"),
    ("inspection", "check"),
    ("remedy", "fix"),
]


def stub_paraphrase_variants(question: str) -> list[str]:
    """Deterministic fake paraphrases (NOT model output).

    Variant 1: synonym swaps (topic words only — codes, numbers and
    asset names are copied verbatim, as the live prompt requires).
    Variant 2: the question with parenthetical codes removed (the
    vocabulary-mismatch case: same topic, no code to lean on).
    Returns at most ``MAX_VARIANTS`` distinct, non-empty variants
    that differ from the original. Used by tests and the fake eval
    so the merge/rerank plumbing is measurable without a live LLM.
    """
    swapped = question
    for old, new in _SYNONYM_SWAPS:
        # Word-boundary swaps only: a naive str.replace turns
        # "documented" into "guideed" and "overheating" into
        # "overheated" inside longer words.
        swapped = re.sub(r"\b" + re.escape(old) + r"\b", new, swapped)
    stripped = re.sub(r"\s*\([^)]*\)", "", question).strip()
    out: list[str] = []
    for variant in (swapped, stripped):
        variant = variant.strip()
        if variant and variant != question.strip() and variant not in out:
            out.append(variant)
    return out[:MAX_VARIANTS]


def stub_decompose_question(question: str) -> list[str]:
    """Deterministic fake decomposition: split on clause boundaries.

    Splits on ", and " / " and " / "; " and keeps clauses with >= 3
    words, each re-terminated with "?". Returns ``[question]`` when
    the question does not split — the caller treats that as "no
    decomposition". Fake only: a live deployment uses
    ``build_live_decomposition_generator``.
    """
    parts = re.split(r",\s+and\s+|\s+and\s+|;\s*", question)
    clauses = []
    for part in parts:
        part = part.strip().strip(",")
        if len(part.split()) >= 3:
            clauses.append(part if part.endswith("?") else part + "?")
    return clauses if len(clauses) >= 2 else [question]


# --- Live generators (LLM-backed; lazy model build) -----------------------

def _parse_variant_lines(text: str, question: str) -> list[str]:
    """Parse one-per-line model output into <= MAX_VARIANTS variants.

    Strips numbering/bullets (``1. ``, ``1) ``, ``- ``, ``* ``),
    drops empty lines, drops lines identical to the original
    question, dedups preserving order. A model that returns prose
    instead of a list yields at most its first usable line — the
    caller treats an empty result as "no expansion".
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|\-|\*)\s*", "", line).strip()
        if not line or line.lower() == question.strip().lower():
            continue
        if line not in out:
            out.append(line)
        if len(out) >= MAX_VARIANTS:
            break
    return out


def _invoke_chat_model(model, prompt: str) -> str:
    response = model.invoke(prompt)
    content = getattr(response, "content", None)
    return content if isinstance(content, str) else str(response)


def build_live_variant_generator(model=None):
    """Live multi-query generator: chat model + constrained prompt.

    ``model`` is an optional prebuilt chat model (tests inject a
    stub); otherwise ``runtime/llm_factory.build_chat_model`` is
    called LAZILY on first use — constructing the generator never
    touches LiteLLM config, and no LLM call happens until a gated
    query actually needs variants.
    """
    def generate(question: str) -> list[str]:
        chat = model
        if chat is None:
            from runtime.llm_factory import build_chat_model

            chat = build_chat_model()
        text = _invoke_chat_model(chat, EXPANSION_PROMPT.format(question=question))
        return _parse_variant_lines(text, question)

    return generate


def build_live_decomposition_generator(model=None):
    """Live decomposition generator: same lazy-model contract."""
    def generate(question: str) -> list[str]:
        chat = model
        if chat is None:
            from runtime.llm_factory import build_chat_model

            chat = build_chat_model()
        text = _invoke_chat_model(chat, DECOMPOSITION_PROMPT.format(question=question))
        return _parse_variant_lines(text, question)

    return generate


# --- Merge + rerank ---------------------------------------------------------

def merge_candidate_rows(all_rows: list[list[dict]]) -> list[dict]:
    """Merge per-variant candidate rows into one set for the fusion.

    ``all_rows[0]`` is the ORIGINAL question's rows; the rest are
    one list per variant/sub-question, in order. Per candidate:
    ``vec_score`` / ``graph_score`` = MEAN across the rows that
    contain it (every variant retrieves the same doc ids on this
    corpus, so this is a mean over variants, not over a subset);
    ``exact`` = MAX; the §4.1 feature row, ``best_path_rels`` and
    ``arms`` come from the row with the highest ``graph_score``.
    ``provenance_variants`` is filled in by the caller (it needs
    the per-variant fused rankings, not just the rows).
    """
    by_id: dict[str, list[dict]] = {}
    for rows in all_rows:
        for row in rows:
            by_id.setdefault(row["doc_id"], []).append(row)
    merged: list[dict] = []
    for doc_id, rows in by_id.items():
        best_graph = max(rows, key=lambda r: (r.get("graph_score", 0.0),
                                              r.get("vec_score", 0.0)))
        merged_row = dict(best_graph)
        merged_row["vec_score"] = round(
            sum(float(r.get("vec_score", 0.0)) for r in rows) / len(rows), 4)
        merged_row["graph_score"] = round(
            sum(float(r.get("graph_score", 0.0)) for r in rows) / len(rows), 4)
        merged_row["exact"] = max(int(r.get("exact", 0)) for r in rows)
        merged_row["provenance_variants"] = []
        merged.append(merged_row)
    return merged


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


class ExpandedRetriever:
    """Hybrid retriever + gated expansion (Phase 3 wrapper).

    ``base`` is a ``HybridRetriever`` (Phases 1–2). Generators are
    pluggable callables; defaults are the LIVE LLM generators
    (lazy model build — no LLM call until a gated query needs one).
    The fake eval and the tests pass the stub generators explicitly.
    """

    def __init__(
        self,
        base,
        *,
        paraphrase_generator=None,
        decomposition_generator=None,
        max_variants: int = MAX_VARIANTS,
    ):
        self.base = base
        self.paraphrase_generator = (
            paraphrase_generator or build_live_variant_generator()
        )
        self.decomposition_generator = (
            decomposition_generator or build_live_decomposition_generator()
        )
        self.max_variants = max_variants

    def __call__(self, question: str):
        return self.retrieve(question)

    def retrieve(self, question: str):
        base_rows, _seeds, _expansion, graph_missing = self.base._candidate_rows(question)
        base_final = self.base._rank_candidates(base_rows, graph_missing)
        base_ranked = [c["doc_id"] for c in base_final]
        base_top6 = set(base_ranked)
        gating = gating_decision(question, base_rows)
        multi = detect_multi_part(question)

        kind = "none"
        variants: list[str] = []
        if multi["is_multi_part"]:
            kind = "decomposition"
            variants = list(self.decomposition_generator(question) or [])[: self.max_variants]
            if len(variants) == 1 and variants[0].strip() == question.strip():
                # Generator says single-part (or cannot split this
                # wording — the stub splitter fails on "Both X and Y"
                # / "Compare X with Y" with no "and"): fall through
                # to gated multi-query instead of expanding nothing.
                variants = []
                kind = "none"
        if kind == "none" and gating["triggered"]:
            kind = "multi-query"
            variants = list(self.paraphrase_generator(question) or [])[: self.max_variants]
            variants = [v for v in variants if v.strip() and v.strip() != question.strip()]

        if not variants:
            kind = "none"
            trace = {
                "fake": True,
                "phase2": self.base.phase2,
                "seeds": [s["key"] for s in _seeds],
                "graph_missing": graph_missing,
                "weights": dict(self.base.weights),
                "expansion": {
                    "triggered": False,
                    "kind": "none",
                    "gating": gating,
                    "multi_part": multi,
                    "variants": [],
                    "variant_overlap": {},
                },
                "feature_rows": trace_feature_rows(base_rows, base_final),
                "candidates": {
                    c["doc_id"]: {
                        "vec": round(c["vec_score"], 4),
                        "graph": round(c["graph_score"], 4),
                        "exact": c["exact"],
                        "final": c.get("final_score"),
                        "g1": c.get("graph_g1_min_hops"),
                        "g2": c.get("graph_g2_path_score"),
                        "g4": c.get("graph_g4_type_match"),
                        "g5": c.get("graph_g5_direct"),
                        "g6_coverage": c.get("graph_g6_coverage"),
                        "g7_ppr": c.get("graph_g7_ppr"),
                        "g8_hub": c.get("graph_g8_hub"),
                        "g8_transit": c.get("graph_g8_transit_key"),
                    }
                    for c in base_final
                },
            }
            return base_ranked, trace

        # Retrieve per variant (dense + graph expansion per variant).
        variant_rows = []
        variant_rankings: list[list[str]] = []
        for variant in variants:
            rows, _s, _e, _gm = self.base._candidate_rows(variant)
            variant_rows.append(rows)
            variant_rankings.append(
                [c["doc_id"] for c in self.base._rank_candidates(rows, graph_missing)]
            )
        merged = merge_candidate_rows([base_rows, *variant_rows])
        # Provenance: variants whose fused top-6 contained the doc
        # (variant 0 is the original question).
        all_rankings = [base_ranked, *variant_rankings]
        for row in merged:
            row["provenance_variants"] = [
                i for i, ranked in enumerate(all_rankings)
                if row["doc_id"] in set(ranked)
            ]
        merged_final = self.base._rank_candidates(merged, graph_missing)
        ranked_ids = [c["doc_id"] for c in merged_final]

        overlap = {
            f"variant_{i + 1}": round(
                _jaccard(set(ranked), base_top6), 4)
            for i, ranked in enumerate(variant_rankings)
        }
        overlap["mean_top6_jaccard"] = round(
            sum(overlap.values()) / len(overlap), 4) if overlap else 0.0
        # Candidate-set overlap is 1.0 by construction on this corpus
        # (the vector arm returns all 7 docs for every variant);
        # report it as measured, not as a finding.
        cand_sets = [set(r["doc_id"] for r in rows) for rows in [base_rows, *variant_rows]]
        overlap["mean_candidate_jaccard"] = round(
            sum(_jaccard(cand_sets[0], s) for s in cand_sets[1:]) / max(len(cand_sets) - 1, 1), 4
        ) if len(cand_sets) > 1 else 1.0

        feature_rows = trace_feature_rows(merged, merged_final)
        base_rank = {doc_id: i + 1 for i, doc_id in enumerate(base_ranked)}
        for row in feature_rows:
            row["rank_before"] = base_rank.get(row["doc_id"])
            row["provenance_variants"] = next(
                (m.get("provenance_variants", []) for m in merged
                 if m["doc_id"] == row["doc_id"]),
                [],
            )
        trace = {
            "fake": True,
            "phase2": self.base.phase2,
            "seeds": [s["key"] for s in _seeds],
            "graph_missing": graph_missing,
            "weights": dict(self.base.weights),
            "expansion": {
                "triggered": True,
                "kind": kind,
                "gating": gating,
                "multi_part": multi,
                "variants": variants,
                "variant_overlap": overlap,
            },
            "feature_rows": feature_rows,
            "candidates": {
                c["doc_id"]: {
                    "vec": round(c["vec_score"], 4),
                    "graph": round(c["graph_score"], 4),
                    "exact": c["exact"],
                    "final": c.get("final_score"),
                    "g1": c.get("graph_g1_min_hops"),
                    "g2": c.get("graph_g2_path_score"),
                    "g4": c.get("graph_g4_type_match"),
                    "g5": c.get("graph_g5_direct"),
                    "g6_coverage": c.get("graph_g6_coverage"),
                    "g7_ppr": c.get("graph_g7_ppr"),
                    "g8_hub": c.get("graph_g8_hub"),
                    "g8_transit": c.get("graph_g8_transit_key"),
                }
                for c in merged_final
            },
        }
        return ranked_ids, trace


def build_expanded_retriever(*, blurred: bool = False, **kwargs):
    """Expanded retriever over the fake corpus, with STUB generators.

    Fake-eval / test convenience: same fake setups as
    ``build_hybrid_retriever`` (``blurred=True`` = code-blurred
    embeddings, the exit-criterion setup), but the generators are
    the deterministic stubs — no LLM. Live use must construct
    ``ExpandedRetriever`` directly so the live generators apply.
    """
    from runtime.retrieval_hybrid import build_hybrid_retriever

    base = build_hybrid_retriever(blurred=blurred, **kwargs)
    retriever = ExpandedRetriever(
        base,
        paraphrase_generator=stub_paraphrase_variants,
        decomposition_generator=stub_decompose_question,
    )
    retriever.__name__ = "hybrid_expanded_blurred" if blurred else "hybrid_expanded_bow"
    return retriever
