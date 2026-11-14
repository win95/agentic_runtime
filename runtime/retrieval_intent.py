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

The three arms:

1. Vector arm — cosine similarity between the question embedding
   and a short English prototype text per intent, using INJECTED
   embeddings (default ``DeterministicFakeEmbeddings``; the caller
   passes the setup's embeddings so BoW and code-blurred both
   work). Raw cosines are z-scored across the six intents with
   ``runtime.retrieval_fusion.zscores`` (clip [-2, 2], std 0 -> all
   zeros), exactly like the fusion's per-query z-scores. There is
   no cosine-similarity helper anywhere in ``runtime/`` (only
   inline dot products, e.g. ``FakeQdrantClient.query_points``),
   so this module defines its own small ``_cosine``.
2. Graph plausibility arm — deterministic, graded 0.0 / 0.5 / 1.0
   per intent (see ``graph_plausibility``). Every rule is grounded
   in what ``seeds/demo_data.py`` really contains; the module reads
   the FAULTS / MAINTENANCE / SENSOR_READINGS / DOCUMENTS /
   RELATIONS tables (via ``runtime/retrieval_linking.DOCUMENT_LINKS``
   and ``runtime/retrieval_graph.build_demo_adjacency``) and
   invents no data.
3. Exact-signal arm — 0/1 per intent from regexes over the raw
   question text (``runtime/retrieval_expansion.COMPARISON_RE`` is
   imported, not copied; fault codes reuse its ``FAULT_CODE_RE``).

Prototype texts and the fake-BoW vocabulary
--------------------------------------------
The fake embeddings are bag-of-words over a vocabulary built from
the seeded DOCUMENT texts only (``build_vocabulary``), and
``tokenize`` drops stopwords. A prototype word outside that
vocabulary contributes NOTHING to the fake vector — it is silently
dropped — so every prototype below uses only in-vocabulary,
digit-free words (digit-free so the code-blurred setup, which
drops every digit-bearing token, embeds the prototypes
identically). Verified against ``build_vocabulary()``::

    lookup:       "fault code guide causes"
    diagnosis:    "fault causes motor temperature vibration"
    procedure:    "guide clean check replace inspection"
    comparison:   "compare motor temperature vibration"
    aggregate:    "codes history"
    unanswerable: "warranty part number firmware"   (see below)

``unanswerable`` is the deliberate exception: all three words are
OUTSIDE the doc vocabulary, so its fake vector is the zero vector
and its cosine is 0.0 for every question. That is the intended
behaviour — "no prototype matches" is itself the signal — and
under a real embedding model the same words would embed to an
off-topic region with low (but nonzero) cosine. Its score therefore
comes from the graph arm (absence of data), not the vector arm.

Plausibility rules (``graph_plausibility``)
-------------------------------------------
``seeds`` are ``gazetteer_seeds`` rows (``{"type", "key",
"confidence"}``); a seed's owning asset is resolved through
RELATIONS (HAS_COMPONENT) and DOCUMENTS (``asset_id`` /
``component_id``). Per-asset ground truth (``demo_data``):

- O21: 8 faults, 6 maintenance rows, 62 sensor readings, 6 docs
  (3 DOCUMENTED_BY + payload links).
- O22: 2 faults, 1 maintenance row, 6 sensor readings.
- O23: 0 faults, 0 maintenance rows, 4 sensor readings, 1 doc
  (doc-o23-overview via payload link).
- O24: 0 faults, 2 maintenance rows, 4 sensor readings, no docs.
- O20: ontology seed only (archived) — 0 faults, 0 maintenance,
  0 sensor readings, no docs.

Correction to the brief: the brief stated O24 "has no
faults/maintenance/sensor rows (ontology seed only)". That is
wrong for maintenance and sensors — ``demo_data`` gives O24 two
maintenance rows and four sensor readings (days 6 and 20); only
the fault count is zero. The diagnosis rule below is therefore
GRADED, not binary: sensors-only is 0.5, faults is 1.0, neither
is 0.0 — so O24 scores 0.5 (below O21's 1.0) and O20 scores 0.0.

- lookup: 1.0 when seeds are linked and something exists to look
  up (a linked asset has fault / sensor / maintenance rows or a
  doc link, or a Document seed / doc neighbour exists); 0.5 when
  seeds are linked but the linked assets hold nothing at all
  (O20 — there is a node, but no retrievable data behind it), or
  when there are no seeds but the question has content words
  (the permissive default); 0.0 for an empty question.
- diagnosis: 1.0 when any linked asset has fault rows; 0.5 when
  no fault rows but a linked asset has sensor readings (data to
  investigate, but no fault to diagnose); 0.0 otherwise.
- procedure: 1.0 when a document is linked to a seed (the seed
  IS a document, or a document node is a 1-hop neighbour of a
  seed in the demo adjacency — payload links included, so
  doc-controller-errors counts for O21 even though it has no
  ontology DOCUMENTED_BY relation); 0.0 otherwise.
- comparison: 1.0 when >= 2 DISTINCT linked Asset seeds, or >= 2
  distinct fault codes are named. Refinement of the brief's ">= 2
  DISTINCT linked nodes": an asset and its own component (O21 +
  motor-M2, the expansion of a single fault code F-204) are ONE
  subject, not a comparison — counting them would make every
  diagnosis question "comparable". The compared things must be
  peer subjects: two assets, or two fault codes.
- aggregate: 0.0 unless the question asks for a list/count
  (``list`` / ``all`` / ``how many`` / ``total``, or plural
  "documents"); then 1.0 when a linked asset has fault rows or
  doc links to enumerate, else 0.5 (wording with no linked data,
  e.g. a seedless set question).
- unanswerable: 1.0 when (a) there are no seeds AND the question
  shares no vocabulary word with the doc corpus (nothing grounds
  it at all), or (b) the question is fault-shaped (mentions
  fault / cause / leak / diagnos*, or a fault code) AND every
  linked asset has zero fault rows (the fault it asks about has
  no row to explain it — O23's fire alarm, O24's faults, O20's
  leak); else 0.0. Note the deliberate asymmetry with diagnosis:
  diagnosis needs faults to be plausible, unanswerable needs
  their absence.

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

Known misses on the golden set (measured, not hidden)
-----------------------------------------------------
The BoW vector arm is lexical overlap over 174 doc words, so
topic words dominate it and several golden labels lose (measured
identically on BoW and code-blurred — the prototypes are
digit-free, so blurring changes nothing):
L01/L02/L04/L09/M01/M06/M07/M08/C02/C03 score diagnosis over
their golden label (fault-code / number exact signals plus
motor-temperature topic overlap); L10/M03 score procedure
("inspection"/"guide" overlap); L11 scores comparison and M05
scores comparison (both assets are named in M05, so the
comparison graph fires) over lookup; L07 (O23 normal ranges)
scores diagnosis over lookup (temperature/vibration overlap);
A02 (seedless, "motor and ropes") scores comparison over
aggregate (shared "motor" topic word, no seeds to ground
aggregate); A03 ("all the documented causes of vibration")
scores diagnosis over aggregate (the "all" exact signal at
w_exact=0.20 cannot overcome a vector z gap of ~2.6);
U01 (warranty for O21's motor) scores diagnosis — the data
exists, the asked detail does not, and this module only detects
absence, not silence; U02 (fire alarm on O23) scores lookup —
its unanswerable graph is 1.0 but the lookup vector z (1.53)
outweighs it; U03 (brake-pad part number) and U06 (firmware
procedure) tie at lookup and lose the tie-break (O21/O22 do
have data); U04 ("how many floors does B2 have") scores
aggregate — its "how many" shape fires while the answer (an
ontology property, 18 floors) lives outside the doc corpus.
Representatives that DO win (locked in by the tests): lookup
L08/M04, diagnosis L04/L09, procedure M03, comparison C01/C04/
C05/C06, aggregate A01/A04, unanswerable U05.
"""

from __future__ import annotations

import logging
import math
import re

from runtime.retrieval_baseline import (
    DeterministicFakeEmbeddings,
    build_vocabulary,
    tokenize,
)
from runtime.retrieval_expansion import COMPARISON_RE, FAULT_CODE_RE
from runtime.retrieval_fusion import zscores
from runtime.retrieval_graph import (
    build_demo_adjacency,
    gazetteer_seeds,
)
from runtime.retrieval_linking import DOCUMENT_LINKS
from seeds.demo_data import (
    DOCUMENTS,
    FAULTS,
    MAINTENANCE,
    RELATIONS,
    SENSOR_READINGS,
)

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

# Prototype texts: short English topic phrases. Every word except
# the unanswerable prototype's is in the doc-derived BoW vocabulary
# and digit-free (see module docstring for why).
PROTOTYPES: dict[str, str] = {
    "lookup": "fault code guide causes",
    "diagnosis": "fault causes motor temperature vibration",
    "procedure": "guide clean check replace inspection",
    "comparison": "compare motor temperature vibration",
    "aggregate": "codes history",
    "unanswerable": "warranty part number firmware",
}

# --- Exact-signal patterns (raw question text, not BoW tokens) -----------
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

# --- Demo-data indexes (built once at import; deterministic) -------------
_FAULTS_BY_ASSET: dict[str, int] = {}
for _row in FAULTS:
    _FAULTS_BY_ASSET[_row["asset_id"]] = _FAULTS_BY_ASSET.get(_row["asset_id"], 0) + 1
_SENSORS_BY_ASSET: dict[str, int] = {}
for _row in SENSOR_READINGS:
    _SENSORS_BY_ASSET[_row["asset_id"]] = _SENSORS_BY_ASSET.get(_row["asset_id"], 0) + 1
_MAINT_BY_ASSET: dict[str, int] = {}
for _row in MAINTENANCE:
    _MAINT_BY_ASSET[_row["asset_id"]] = _MAINT_BY_ASSET.get(_row["asset_id"], 0) + 1
_DOCS_BY_ASSET: dict[str, set[str]] = {}
for _doc in DOCUMENTS:
    if _doc.get("asset_id"):
        _DOCS_BY_ASSET.setdefault(_doc["asset_id"], set()).add(_doc["id"])
for _from_id, _from_label, _rel, _to_id, _to_label in RELATIONS:
    if _rel == "DOCUMENTED_BY":
        _DOCS_BY_ASSET.setdefault(_from_id, set()).add(_to_id)
_ASSET_OF_COMPONENT: dict[str, str] = {}
for _from_id, _from_label, _rel, _to_id, _to_label in RELATIONS:
    if _rel == "HAS_COMPONENT":
        _ASSET_OF_COMPONENT[_to_id] = _from_id
for _doc in DOCUMENTS:
    if _doc.get("component_id") and _doc.get("asset_id"):
        _ASSET_OF_COMPONENT.setdefault(_doc["component_id"], _doc["asset_id"])
_ASSET_OF_DOC: dict[str, str] = {}
for _doc_id, _links in DOCUMENT_LINKS.items():
    if _links.get("asset_id"):
        _ASSET_OF_DOC[_doc_id] = _links["asset_id"]
for _doc in DOCUMENTS:
    if _doc.get("asset_id"):
        _ASSET_OF_DOC.setdefault(_doc["id"], _doc["asset_id"])

_VOCABULARY: set[str] = set(build_vocabulary())


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 when either vector is all zeros.

    No cosine helper exists in ``runtime/`` (only inline dot
    products), so this module defines its own. A zero prototype
    vector (``unanswerable`` under the fake) yields 0.0, never a
    division-by-zero.
    """
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _linked_assets(seeds: list[dict]) -> set[str]:
    """Owning asset per seed (Asset seeds resolve to themselves)."""
    assets: set[str] = set()
    for seed in seeds:
        key, kind = seed["key"], seed["type"]
        if kind == "Asset":
            assets.add(key)
        elif kind == "Component":
            owner = _ASSET_OF_COMPONENT.get(key)
            if owner:
                assets.add(owner)
        elif kind == "Document":
            owner = _ASSET_OF_DOC.get(key)
            if owner:
                assets.add(owner)
    return assets


def _linked_doc_ids(seeds: list[dict], adjacency: dict) -> set[str]:
    """Docs linked to the seeds: Document seeds + 1-hop doc neighbours.

    Uses the demo adjacency (RELATIONS + payload links), so a doc
    counts as linked even when it has no ontology DOCUMENTED_BY
    relation (e.g. doc-controller-errors for O21).
    """
    docs = {s["key"] for s in seeds if s["type"] == "Document"}
    for seed in seeds:
        for _rel, to_key, to_type in adjacency.get(seed["key"], {}).get("neighbors", []):
            if to_type == "Document":
                docs.add(to_key)
    return docs


def _has_asset_data(asset: str) -> bool:
    """Any demo row or doc link behind this asset."""
    return bool(
        _FAULTS_BY_ASSET.get(asset, 0)
        or _SENSORS_BY_ASSET.get(asset, 0)
        or _MAINT_BY_ASSET.get(asset, 0)
        or _DOCS_BY_ASSET.get(asset)
    )


def graph_plausibility(
    intent: str,
    question: str,
    seeds: list[dict] | None = None,
    adjacency: dict | None = None,
) -> float:
    """Graph arm for one intent: 0.0 / 0.5 / 1.0 (see module docstring).

    ``seeds`` defaults to ``gazetteer_seeds(question)``;
    ``adjacency`` defaults to ``build_demo_adjacency()``.
    """
    if intent not in INTENT_ORDER:
        raise ValueError(f"unknown intent: {intent!r}")
    seeds = seeds if seeds is not None else gazetteer_seeds(question or "")
    adjacency = adjacency if adjacency is not None else build_demo_adjacency()
    assets = _linked_assets(seeds)
    docs = _linked_doc_ids(seeds, adjacency)
    text = question or ""

    if intent == "lookup":
        if not seeds:
            return 0.5 if tokenize(text) else 0.0
        if any(_has_asset_data(a) for a in assets) or docs:
            return 1.0
        return 0.5

    if intent == "diagnosis":
        if any(_FAULTS_BY_ASSET.get(a, 0) > 0 for a in assets):
            return 1.0
        if any(_SENSORS_BY_ASSET.get(a, 0) > 0 for a in assets):
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
        if any(_FAULTS_BY_ASSET.get(a, 0) > 0 for a in assets) or any(
            _DOCS_BY_ASSET.get(a) for a in assets
        ):
            return 1.0
        return 0.5

    # unanswerable: scores from the ABSENCE of links/data.
    if not seeds and not (set(tokenize(text)) & _VOCABULARY):
        return 1.0
    fault_shaped = bool(FAULT_WORD_RE.search(text) or FAULT_CODE_RE.search(text))
    if assets and fault_shaped:
        if all(_FAULTS_BY_ASSET.get(a, 0) == 0 for a in assets):
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
) -> dict:
    """Score all six intents for a question, fusion-style.

    ``embeddings`` defaults to ``DeterministicFakeEmbeddings``; the
    caller passes the setup's embeddings (BoW or code-blurred).
    ``seeds`` defaults to ``gazetteer_seeds(question)``.
    Returns ``{"intent", "margin", "scores", "weights",
    "prototypes", "seeds"}`` where ``scores[intent]`` is
    ``{"vec", "z_vec", "graph", "exact", "final"}``. Ties break in
    INTENT_ORDER (lookup first, unanswerable last). Deterministic:
    no randomness, no LLM, no live store.
    """
    embeddings = embeddings or DeterministicFakeEmbeddings()
    seeds = seeds if seeds is not None else gazetteer_seeds(question or "")
    adjacency = adjacency if adjacency is not None else build_demo_adjacency()

    q_vec = embeddings.embed_query(question or "")
    raw_vec = {
        intent: _cosine(q_vec, embeddings.embed_query(PROTOTYPES[intent]))
        for intent in INTENT_ORDER
    }
    z_vec = dict(zip(INTENT_ORDER, zscores([raw_vec[i] for i in INTENT_ORDER])))
    scores: dict[str, dict] = {}
    for intent in INTENT_ORDER:
        graph = graph_plausibility(intent, question, seeds, adjacency)
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
