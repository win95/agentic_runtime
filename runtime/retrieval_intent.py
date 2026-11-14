"""Phase 4 — intent scoring, fusion-style (Query Understanding).

Deterministic, NO LLM call anywhere in this module. Each candidate
intent is scored with the same three-arm shape as the retrieval
fusion (``runtime/retrieval_fusion.py``)::

    final = W_VEC * z(vec) + W_GRAPH * graph + W_EXACT * exact

with priors ``W_VEC=0.50, W_GRAPH=0.30, W_EXACT=0.20`` (documented
priors, not tuned: the vector weight mirrors the fusion's 0.50; the
graph weight is 0.30 rather than the fusion's 0.40 because the graph
arm here is a 0–1 plausibility, not a z-scored feature; the exact
weight is doubled from the fusion's 0.10 because intent signals
are lexical and sparse — a single fault code or "list" word is
often the only thing separating diagnosis from lookup).

All data is INJECTED — there is no demo corpus, no demo table and
no default embedding model in this module:

* ``embeddings`` (``score_intents``): an object with
  ``embed_query``. When ``None`` the vector signal is 0 for every
  intent (``vec`` = 0.0, ``z_vec`` = 0.0) and the graph + exact
  signals carry the score — documented, never a silent fake.
* ``adjacency`` (REQUIRED everywhere): the live graph adjacency
  (node key -> ``{"type", "neighbors": [(rel, to_key, to_type)]}``).
  It resolves component ownership (``HAS_COMPONENT`` edges),
  document linkage (``DOCUMENTED_BY`` edges) and 1-hop doc
  neighbours. A missing adjacency raises ``ValueError``.
* ``asset_data``: ``{asset key: {"faults": int, "sensors": int,
  "maintenance": int}}`` — per-asset ground truth counts,
  supplied by the caller from the real store. When ``None`` all
  counts read as 0, i.e. UNKNOWN, not "known zero": rules that
  conclude from an ABSENCE of data (the unanswerable fault rule)
  do not fire without it — absence of injected data is not
  evidence of absence.
* ``doc_links``: ``{doc id: [node ids]}`` — the Qdrant payload
  links (see ``runtime/retrieval_linking.py``). They resolve a
  Document seed's owning asset and add docs per asset, alongside
  the adjacency's ``DOCUMENTED_BY`` edges.
* ``vocabulary``: an optional ``set[str]`` of content words the
  question can share with the corpus. It gates exactly one
  sub-rule (unanswerable rule (a), below); when ``None`` that
  sub-rule is skipped.

The three arms:

1. Vector arm — cosine similarity between the question embedding
   and a short English prototype text per intent, using the
   INJECTED embeddings. Raw cosines are z-scored across the six
   intents with ``runtime.retrieval_fusion.zscores`` (clip [-2, 2],
   std 0 -> all zeros), exactly like the fusion's per-query
   z-scores. There is no cosine-similarity helper anywhere in
   ``runtime/``, so this module defines its own small ``_cosine``.
   With no embeddings the arm is 0 for every intent.
2. Graph plausibility arm — deterministic, graded 0.0 / 0.5 / 1.0
   per intent (see ``graph_plausibility``). Every rule reads only
   the injected ``adjacency`` / ``asset_data`` / ``doc_links``.
3. Exact-signal arm — 0/1 per intent from regexes over the raw
   question text (``runtime/retrieval_expansion.COMPARISON_RE`` is
   imported, not copied; fault codes reuse its ``FAULT_CODE_RE``).

Prototype texts (short English topic phrases; embedded by the
injected model — no bag-of-words vocabulary is involved)::

    lookup:       "fault code guide causes"
    diagnosis:    "fault causes motor temperature vibration"
    procedure:    "guide clean check replace inspection"
    comparison:   "compare motor temperature vibration"
    aggregate:    "codes history"
    unanswerable: "warranty part number firmware"

Plausibility rules (``graph_plausibility``)
-------------------------------------------
``seeds`` are ``gazetteer_seeds`` rows (``{"type", "key",
"confidence"}``); a seed's owning asset is resolved through the
adjacency (``HAS_COMPONENT`` edges) and the injected doc links.
Per-asset ground truth comes from ``asset_data`` (fault /
sensor / maintenance counts) and from the docs linked to the
asset (``doc_links`` + ``DOCUMENTED_BY`` adjacency edges).

- lookup: 1.0 when seeds are linked and something exists to look
  up (a linked asset has fault / sensor / maintenance rows or a
  doc link, or a Document seed / doc neighbour exists); 0.5 when
  seeds are linked but the linked assets hold nothing at all
  (there is a node, but no retrievable data behind it), or when
  there are no seeds but the question has content words (the
  permissive default); 0.0 for an empty question.
- diagnosis: 1.0 when any linked asset has fault rows; 0.5 when
  no fault rows but a linked asset has sensor readings (data to
  investigate, but no fault to diagnose); 0.0 otherwise.
- procedure: 1.0 when a document is linked to a seed (the seed
  IS a document, or a document node is a 1-hop neighbour of a
  seed in the adjacency — payload links included); 0.0 otherwise.
- comparison: 1.0 when >= 2 DISTINCT linked Asset seeds, or >= 2
  distinct fault codes are named. Refinement: an asset and its
  own component (O21 + motor-M2, the expansion of a single fault
  code F-204) are ONE subject, not a comparison — counting them
  would make every diagnosis question "comparable". The compared
  things must be peer subjects: two assets, or two fault codes.
- aggregate: 0.0 unless the question asks for a list/count
  (``list`` / ``all`` / ``how many`` / ``total``, or plural
  "documents"); then 1.0 when a linked asset has fault rows or
  doc links to enumerate, else 0.5 (wording with no linked data,
  e.g. a seedless set question).
- unanswerable: 1.0 when (a) there are no seeds AND the question
  shares no vocabulary word with the corpus (only when a
  ``vocabulary`` was injected — without one this sub-rule is
  skipped), or (b) the question is fault-shaped (mentions
  fault / cause / leak / diagnos*, or a fault code) AND every
  linked asset has zero fault rows — and ``asset_data`` was
  supplied, so the zero is known, not assumed; else 0.0. Note
  the deliberate asymmetry with diagnosis: diagnosis needs
  faults to be plausible, unanswerable needs their absence.

Exact signals (``exact_signal``, 0/1 per intent)
-------------------------------------------------
- lookup: a threshold/number is present (``92C``, ``12mm``,
  ``6.8 mm/s`` — a number with a unit; bare years like ``2026``
  do not count).
- diagnosis: a fault code (``F-###``) OR a threshold/number.
- procedure: none — no lexical signal separates a how-to
  question from a lookup in this corpus ("how" would promote
  L06, a lookup, over the lookup intent); procedure wins on
  vector + graph.
- comparison: a comparison word (``COMPARISON_RE`` imported from
  ``retrieval_expansion``: vs / versus / compare / hotter /
  colder / higher / lower / difference) OR ``more`` / ``most`` /
  ``fewer`` / ``less`` / ``least`` (supplement: "more faults" in
  C06 is comparative, and ``more`` is the only comparison word
  in that question).
- aggregate: a list/count word (``list`` / ``all`` / ``how many``
  / ``total``) OR plural "documents" (A02/A04 ask for "documents"
  with no list word).
- unanswerable: none — its signal is absence, in the graph arm.

Ties break deterministically in INTENT_ORDER: lookup, diagnosis,
procedure, comparison, aggregate, unanswerable. ``score_intents``
returns the winner, the margin (winner minus runner-up final),
and the per-intent breakdown (``vec`` / ``z_vec`` / ``graph`` /
``exact`` / ``final``) so the eval can report WHY an intent won.

Signals DROPPED by the fake removal (no live equivalent)
--------------------------------------------------------
- The default vector arm: there is no default embedding model
  any more (the old default was a bag-of-words fake over a
  vocabulary built from seeded document texts). With no
  injected embeddings the vec signal is 0 for every intent.
- The per-asset demo indexes (fault / sensor / maintenance /
  doc counts read from the seeded demo tables at import time):
  replaced by the injected ``asset_data`` / ``doc_links`` /
  ``adjacency``. With nothing injected the counts are all 0
  (unknown), so diagnosis scores 0.0, lookup falls back to
  its 0.5 branches, aggregate to 0.5, and the unanswerable
  fault-absence rule does not fire.
- Unanswerable rule (a) without an injected ``vocabulary``:
  the old module compared the question's tokens against the
  seeded-doc vocabulary; no corpus vocabulary exists live
  (it would have to be built from Qdrant payloads and no
  factory supplies one), so the sub-rule is skipped unless
  the caller injects one.
"""

from __future__ import annotations

import logging
import math
import re

from runtime.retrieval_expansion import COMPARISON_RE, FAULT_CODE_RE
from runtime.retrieval_fusion import zscores
from runtime.retrieval_graph import gazetteer_seeds

logger = logging.getLogger(__name__)

# --- Fusion-style weights (priors; see module docstring) ---------------
W_VEC = 0.50
W_GRAPH = 0.30
W_EXACT = 0.20

# --- Candidate intents + tie-break order ---------------------------------
INTENT_ORDER = [
    "lookup",
    "diagnosis",
    "procedure",
    "comparison",
    "aggregate",
    "unanswerable",
]

# Prototype texts: short English topic phrases, embedded by the
# injected model (see module docstring).
PROTOTYPES: dict[str, str] = {
    "lookup": "fault code guide causes",
    "diagnosis": "fault causes motor temperature vibration",
    "procedure": "guide clean check replace inspection",
    "comparison": "compare motor temperature vibration",
    "aggregate": "codes history",
    "unanswerable": "warranty part number firmware",
}

# --- Exact-signal patterns (raw question text, not tokens) ---------------
# Number with a unit: 92C, 12mm, 6.8 mm/s, 4.5 mm/s, 3mm, 6s.
# mm/s before mm before s; a bare year (2026) has no unit and
# does not match.
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mm/s|mm|°?c|s)\b", re.IGNORECASE)
LIST_COUNT_RE = re.compile(r"\b(list|all|how many|total)\b", re.IGNORECASE)
PLURAL_DOCS_RE = re.compile(r"\bdocuments\b", re.IGNORECASE)
# Supplement to COMPARISON_RE (imported): "more faults" (C06) is
# comparative, and no COMPARISON_RE word appears in that question.
MORE_RE = re.compile(r"\b(more|most|fewer|less|least)\b", re.IGNORECASE)
# Fault-shaped question: asks about a fault, its cause, a leak, or
# a diagnosis. Used by the unanswerable absence rule.
FAULT_WORD_RE = re.compile(
    r"\b(fault|faults|cause|causes|caused|leak|leaks|diagnos\w*)\b",
    re.IGNORECASE,
)

# --- Local content tokenizer (replaces the deleted baseline tokenizer) ----
# Lowercase word tokens (hyphenated codes like f-204 stay whole),
# minus a small generic stopword list. Used only for the "does the
# question have content words" check (lookup, seedless) and the
# optional vocabulary-overlap check (unanswerable rule (a)).
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "at", "from", "it", "its",
    "this", "that", "what", "which", "how", "when", "should", "does",
    "do", "can", "if", "then", "than", "as", "after", "before", "all",
    "each", "every", "one", "two", "also", "only", "no", "not", "but",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def _tokenize(text: str) -> list[str]:
    """Lowercase content-word tokens of ``text`` (stopwords dropped)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 when either vector is all zeros.

    No cosine helper exists in ``runtime/`` (only inline dot
    products), so this module defines its own. A zero prototype
    vector yields 0.0, never a division-by-zero.
    """
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _asset_type(node_key: str, adjacency: dict) -> str | None:
    """The adjacency's recorded type for a node, or None if unknown."""
    return adjacency.get(node_key, {}).get("type")


def _component_owner(component_key: str, adjacency: dict) -> str | None:
    """Owning asset of a component, via HAS_COMPONENT adjacency edges."""
    for node, entry in adjacency.items():
        for rel, to_key, _to_type in entry.get("neighbors", []):
            if rel == "HAS_COMPONENT" and to_key == component_key:
                return node
    return None


def _docs_by_asset(doc_links: dict | None, adjacency: dict) -> dict[str, set[str]]:
    """Docs per asset, from injected payload links + DOCUMENTED_BY edges.

    ``doc_links`` maps doc id -> node ids; a doc counts for every
    Asset node in its link list (when the adjacency knows the
    node's type). Adjacency edges add the same: an Asset node's
    DOCUMENTED_BY neighbours that are Documents.
    """
    docs: dict[str, set[str]] = {}
    for doc_id, node_ids in (doc_links or {}).items():
        for node_key in node_ids or []:
            if _asset_type(node_key, adjacency) == "Asset":
                docs.setdefault(node_key, set()).add(doc_id)
    for node, entry in adjacency.items():
        if entry.get("type") != "Asset":
            continue
        for rel, to_key, to_type in entry.get("neighbors", []):
            if rel == "DOCUMENTED_BY" and to_type == "Document":
                docs.setdefault(node, set()).add(to_key)
    return docs


def _linked_assets(
    seeds: list[dict], adjacency: dict, doc_links: dict | None
) -> set[str]:
    """Owning asset per seed (Asset seeds resolve to themselves)."""
    assets: set[str] = set()
    for seed in seeds:
        key, kind = seed["key"], seed["type"]
        if kind == "Asset":
            assets.add(key)
        elif kind == "Component":
            owner = _component_owner(key, adjacency)
            if owner:
                assets.add(owner)
        elif kind == "Document":
            owner = None
            for node_key in (doc_links or {}).get(key, []) or []:
                if _asset_type(node_key, adjacency) == "Asset":
                    owner = node_key
                    break
            if owner is None:
                for _rel, to_key, to_type in adjacency.get(key, {}).get("neighbors", []):
                    if to_type == "Asset":
                        owner = to_key
                        break
            if owner:
                assets.add(owner)
    return assets


def _linked_doc_ids(seeds: list[dict], adjacency: dict) -> set[str]:
    """Docs linked to the seeds: Document seeds + 1-hop doc neighbours.

    Uses the supplied adjacency, so a doc counts as linked even
    when it has no ontology DOCUMENTED_BY relation (e.g. a
    payload-linked doc whose edge the adjacency records).
    """
    docs = {s["key"] for s in seeds if s["type"] == "Document"}
    for seed in seeds:
        for _rel, to_key, to_type in adjacency.get(seed["key"], {}).get("neighbors", []):
            if to_type == "Document":
                docs.add(to_key)
    return docs


def _asset_counts(asset_data: dict | None, asset: str) -> tuple[int, int, int]:
    """(faults, sensors, maintenance) for an asset; all 0 when unknown.

    ``asset_data`` maps asset key -> ``{"faults", "sensors",
    "maintenance"}`` (each optional, default 0). A missing entry
    means the caller supplied no ground truth for that asset.
    """
    entry = (asset_data or {}).get(asset) or {}
    return (
        int(entry.get("faults", 0) or 0),
        int(entry.get("sensors", 0) or 0),
        int(entry.get("maintenance", 0) or 0),
    )


def _has_asset_data(
    asset: str, asset_data: dict | None, docs_by_asset: dict[str, set[str]]
) -> bool:
    """Any injected row or doc link behind this asset."""
    faults, sensors, maintenance = _asset_counts(asset_data, asset)
    return bool(faults or sensors or maintenance or docs_by_asset.get(asset))


def graph_plausibility(
    intent: str,
    question: str,
    seeds: list[dict] | None = None,
    adjacency: dict | None = None,
    *,
    asset_data: dict | None = None,
    doc_links: dict | None = None,
    vocabulary: set[str] | None = None,
) -> float:
    """Graph arm for one intent: 0.0 / 0.5 / 1.0 (see module docstring).

    ``seeds`` defaults to ``gazetteer_seeds(question)``;
    ``adjacency`` is REQUIRED (the live graph adjacency).
    ``asset_data`` / ``doc_links`` / ``vocabulary`` are the
    injected data (see module docstring); with nothing injected
    the counts read as unknown (0) and the unanswerable
    absence rules do not fire.
    """
    if intent not in INTENT_ORDER:
        raise ValueError(f"unknown intent: {intent!r}")
    if adjacency is None:
        raise ValueError(
            "graph_plausibility requires 'adjacency' (the live graph "
            "adjacency); no default adjacency exists."
        )
    seeds = seeds if seeds is not None else gazetteer_seeds(question or "")
    assets = _linked_assets(seeds, adjacency, doc_links)
    docs = _linked_doc_ids(seeds, adjacency)
    docs_by_asset = _docs_by_asset(doc_links, adjacency)
    text = question or ""

    if intent == "lookup":
        if not seeds:
            return 0.5 if _tokenize(text) else 0.0
        if any(_has_asset_data(a, asset_data, docs_by_asset) for a in assets) or docs:
            return 1.0
        return 0.5

    if intent == "diagnosis":
        if any(_asset_counts(asset_data, a)[0] > 0 for a in assets):
            return 1.0
        if any(_asset_counts(asset_data, a)[1] > 0 for a in assets):
            return 0.5
        return 0.0

    if intent == "procedure":
        return 1.0 if docs else 0.0

    if intent == "comparison":
        asset_seeds = {s["key"] for s in seeds if s["type"] == "Asset"}
        fault_codes = {m.group(0).upper() for m in FAULT_CODE_RE.finditer(text)}
        if len(asset_seeds) >= 2 or len(fault_codes) >= 2:
            return 1.0
        return 0.0

    if intent == "aggregate":
        list_wording = bool(LIST_COUNT_RE.search(text) or PLURAL_DOCS_RE.search(text))
        if not list_wording:
            return 0.0
        if any(_asset_counts(asset_data, a)[0] > 0 for a in assets) or any(
            docs_by_asset.get(a) for a in assets
        ):
            return 1.0
        return 0.5

    # unanswerable: scores from the ABSENCE of links/data.
    if vocabulary is not None and not seeds and not (set(_tokenize(text)) & vocabulary):
        return 1.0
    fault_shaped = bool(FAULT_WORD_RE.search(text) or FAULT_CODE_RE.search(text))
    if asset_data is not None and assets and fault_shaped:
        if all(_asset_counts(asset_data, a)[0] == 0 for a in assets):
            return 1.0
    return 0.0


def exact_signal(intent: str, question: str) -> int:
    """Exact arm for one intent: 0 or 1 (see module docstring)."""
    if intent not in INTENT_ORDER:
        raise ValueError(f"unknown intent: {intent!r}")
    text = question or ""
    if intent == "lookup":
        return 1 if NUMBER_RE.search(text) else 0
    if intent == "diagnosis":
        return 1 if (FAULT_CODE_RE.search(text) or NUMBER_RE.search(text)) else 0
    if intent == "comparison":
        return 1 if (COMPARISON_RE.search(text) or MORE_RE.search(text)) else 0
    if intent == "aggregate":
        return 1 if (LIST_COUNT_RE.search(text) or PLURAL_DOCS_RE.search(text)) else 0
    # procedure / unanswerable: no lexical exact signal (documented).
    return 0


def score_intents(
    question: str,
    *,
    seeds: list[dict] | None = None,
    embeddings=None,
    adjacency: dict | None = None,
    asset_data: dict | None = None,
    doc_links: dict | None = None,
    vocabulary: set[str] | None = None,
) -> dict:
    """Score all six intents for a question, fusion-style.

    ``embeddings`` is an object with ``embed_query``; when ``None``
    the vector signal is 0 for every intent (``vec`` = 0.0,
    ``z_vec`` = 0.0) and the graph + exact signals carry the
    score. ``seeds`` defaults to ``gazetteer_seeds(question)``.
    ``adjacency`` is REQUIRED (the live graph adjacency).
    ``asset_data`` / ``doc_links`` / ``vocabulary`` are passed
    through to ``graph_plausibility`` (see its docstring).
    Returns ``{"intent", "margin", "scores", "weights",
    "prototypes", "seeds"}`` where ``scores[intent]`` is
    ``{"vec", "z_vec", "graph", "exact", "final"}``. Ties break in
    INTENT_ORDER (lookup first, unanswerable last). Deterministic:
    no randomness, no LLM, no live store.
    """
    if adjacency is None:
        raise ValueError(
            "score_intents requires 'adjacency' (the live graph "
            "adjacency); no default adjacency exists."
        )
    seeds = seeds if seeds is not None else gazetteer_seeds(question or "")
    if embeddings is None:
        raw_vec = {intent: 0.0 for intent in INTENT_ORDER}
    else:
        q_vec = embeddings.embed_query(question or "")
        raw_vec = {
            intent: _cosine(q_vec, embeddings.embed_query(PROTOTYPES[intent]))
            for intent in INTENT_ORDER
        }
    z_vec = dict(zip(INTENT_ORDER, zscores([raw_vec[i] for i in INTENT_ORDER])))
    scores: dict[str, dict] = {}
    for intent in INTENT_ORDER:
        graph = graph_plausibility(
            intent,
            question,
            seeds,
            adjacency,
            asset_data=asset_data,
            doc_links=doc_links,
            vocabulary=vocabulary,
        )
        exact = exact_signal(intent, question)
        final = W_VEC * z_vec[intent] + W_GRAPH * graph + W_EXACT * exact
        scores[intent] = {
            "vec": round(raw_vec[intent], 4),
            "z_vec": round(z_vec[intent], 4),
            "graph": graph,
            "exact": exact,
            "final": round(final, 4),
        }
    ranked = sorted(scores, key=lambda i: (-scores[i]["final"], INTENT_ORDER.index(i)))
    winner, runner_up = ranked[0], ranked[1]
    return {
        "intent": winner,
        "margin": round(scores[winner]["final"] - scores[runner_up]["final"], 4),
        "scores": scores,
        "weights": {"w_vec": W_VEC, "w_graph": W_GRAPH, "w_exact": W_EXACT},
        "prototypes": dict(PROTOTYPES),
        "seeds": seeds,
    }
