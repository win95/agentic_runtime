"""Neo4j 1-hop expansion + per-candidate graph features — Phases 1–2
(report §3.2, §4.1, §4.2, §4.7; locked decisions: 1 hop, fan-out
<=10 per (seed, relation type), total <=50 nodes).

Seeds come from a deterministic gazetteer first (exact match on the
demo vocabulary: asset/component/building/document codes and fault
codes — no LLM, no embedding similarity in Phase 1). Phase 4 adds a
normalisation-only variant pass on top of the exact gazetteer
(``gazetteer_seeds``): diacritics-insensitive matching plus
hyphen/space/underscore separator variants of the same codes
(``O-21`` / ``O 21`` / ``o21`` -> ``O21``, ``F204`` -> ``F-204``,
``motor m2`` / ``motorM2`` -> ``motor-M2``), at confidence 0.8
instead of 1.0. Exact hits are byte-identical to Phase 1; a variant
hit for a node that also has an exact hit keeps the 1.0.
over an allowlisted relation set on an in-memory adjacency built from
``seeds/demo_data.py`` RELATIONS + the payload links
(``runtime/retrieval_linking.py``); the live path calls the existing
``ontology/knowledge/graph_query.py`` read functions instead
(``expand_via_graph_query``) — no new Cypher in Phase 1.

Per candidate the module returns G1 (min hop distance), G2
(path_score = max over paths of prod(w_rel) * 0.6**hops, §4.2
weights), G4 (entity-type match +1/0/-1), G5 (direct link 1 / 0.5
via a 1-hop neighbour / 0 + distinct seeds linked), plus the Phase
2 features G6 (seed coverage within 2 hops), G7 (PPR of the
document node, from ``runtime/retrieval_ppr.py``) and G8 (hub
value of the best path's transit node). G3 is the relation-type
weight itself (it enters G2). ``graph_score`` is the Phase 1
combination (G1/G2/G4/G5); ``phase2_graph_scores`` is the locked
full formula ``0.45 * minmax_log(PPR) + 0.30 * z(path_term)
+ 0.25 * coverage`` with the hub penalty inside the path term
(see docs/hybrid-retrieval-phase2.md).

Timeout: expansion takes ``timeout_seconds``; a store call that
raises (including ``TimeoutError``) degrades to an empty expansion
and the caller falls back to vector-only (guardrail 3).
"""

from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
from collections.abc import Callable

from runtime.retrieval_fusion import zscores

logger = logging.getLogger(__name__)

# --- Relation weights (§4.2, starting values = locked priors) --------
RELATION_WEIGHTS: dict[str, float] = {
    "DOCUMENTED_BY": 1.00,
    "HAS_COMPONENT": 0.90,
    "PART_OF": 0.90,
    "LOCATED_IN": 0.40,
    "RELATED_TO": 0.25,
}
# LOCATED_IN through a hub building (degree > HUB_DEGREE) is weaker.
HUB_DEGREE = 50
HUB_LOCATED_IN_WEIGHT = 0.15
HOP_DECAY = 0.6

ALLOWED_RELATIONS = frozenset(RELATION_WEIGHTS)

# Caps (§3.2, locked).
MAX_FANOUT_PER_TYPE = 10
MAX_EXPANSION_NODES = 50
DEFAULT_TIMEOUT_SECONDS = 0.15  # 150 ms Neo4j budget, report §3.2

# --- Phase 2 graph_score weights (locked full formula, §4.7) ----------
# graph_score = W_PPR * ppr_norm + W_PATH * z(path_term)
#             + W_COVERAGE * coverage
#   ppr_norm  = log1p(PPR) then min-max to [0, 1] per query
#               (report §4.7: "PPR via log then min-max")
#   path_term = Phase 1 graph_score (G1/G2/G4/G5) minus the hub
#               penalty (G8, below); z = per-query z-score clipped
#               to [-2, 2] (``runtime/retrieval_fusion.py``)
#   coverage  = G6, raw fraction in [0, 1] (NOT z-scored)
# The hub penalty lives INSIDE the path term (not subtracted from
# the final graph_score): on a live graph a hub can have degree
# 200+ (log1p ~ 5.3), and subtracting that from the final score
# would let one hub dominate the fusion; inside the z-scored path
# term the penalty saturates at z = -2. Prototyped placements:
# subtractive-on-final and inside-path both fix M08 at lambda 1.0,
# inside-path is strictly safer live (see docs/hybrid-retrieval-
# phase2.md). W_PPR = 0 is the PPR ablation flag (no renormalising
# of the surviving sub-weights — the path:coverage ratio stays
# 0.30:0.25, so the ablation measures PPR alone).
W_PPR = 0.45
W_PATH = 0.30
W_COVERAGE = 0.25
# Hub penalty (G8): HUB_LAMBDA * log(1 + degree(transit node)),
# subtracted from the path term, only when the best path transits
# through a Building or a generic Document (report §4.3/§4.1 G8:
# "penalise paths through hub Buildings / generic Documents").
# Prototyped lambda sweep on the blurred setup: 0.6–1.25 all give
# identical numbers (the z-clip saturates the effect), so 1.0 —
# the full log-degree value — is used, not a tuned knife-edge.
HUB_LAMBDA = 1.0
HUB_TRANSIT_TYPES = frozenset({"Building", "Document"})

# --- Gazetteer ---------------------------------------------------------
# Fault code -> (asset, component-or-None), grounded in
# seeds/demo_data.py FAULTS rows + the component named in each fault
# title. Codes with no component node (F-102 leveling, F-103 safety,
# F-110 door timeout, F-205 motor vibration) map to the asset only.
FAULT_CODE_LINKS: dict[str, tuple[str, str | None]] = {
    "F-101": ("O21", "door-sensor-DS1"),
    "F-204": ("O21", "motor-M2"),
    "F-301": ("O21", "brake-BK1"),
    "F-410": ("O21", "controller-C1"),
    "F-206": ("O22", "motor-M3"),
    "F-207": ("O22", "controller-C2"),
    "F-102": ("O21", None),
    "F-103": ("O21", None),
    "F-110": ("O21", None),
    "F-205": ("O21", None),
}
# Unambiguous component names (exactly one such component exists in
# the demo ontology). Bare "motor" / "controller" are ambiguous
# (M2/M3/M0, C1/C2) and are deliberately NOT mapped.
COMPONENT_NAME_LINKS: dict[str, tuple[str, str]] = {
    "door sensor": ("Component", "door-sensor-DS1"),
    "brake": ("Component", "brake-BK1"),
}

_CODE_RE = re.compile(
    r"\b(o\d{2}|building-b\d|motor-m\d|door-sensor-ds\d|controller-c\d"
    r"|brake-bk\d|doc-[a-z0-9-]+|f-\d{3})\b"
)


def _canonical_code(code: str) -> str:
    """Normalise a lowercased code match to the ontology key spelling.

    Adjacency keys are case-sensitive (``O21``, ``motor-M2``,
    ``door-sensor-DS1``); a raw lowercase match (``o21``, ``motor-m2``)
    would silently match nothing and starve the graph arm.
    """
    if re.fullmatch(r"o\d{2}", code):
        return code.upper()
    if re.fullmatch(r"f-\d{3}", code):
        return code.upper()
    if re.fullmatch(r"motor-m\d", code):
        return f"motor-M{code[-1]}"
    if re.fullmatch(r"door-sensor-ds\d", code):
        return f"door-sensor-DS{code[-1]}"
    if re.fullmatch(r"controller-c\d", code):
        return f"controller-C{code[-1]}"
    if re.fullmatch(r"brake-bk\d", code):
        return f"brake-BK{code[-1]}"
    return code  # building-b*, doc-* are already lowercase keys


# --- Phase 4: gazetteer normalisation (no LLM entity linker) -----------
# The exact gazetteer above is unchanged. The variant pass below runs
# on a diacritics-stripped copy of the lowercased question and matches
# separator variants of the SAME codes: hyphens, spaces and
# underscores between the code's parts are optional, so "O-21",
# "O 21" and "o21" all resolve to O21, "F204" to F-204, "motor m2" /
# "motorM2" to motor-M2, and likewise for door-sensor-DS1,
# controller-C1, brake-BK1 and building-B1. Every variant pattern
# still requires word boundaries on the normalised text, so a code
# embedded in a longer token ("XO21", "O21x", "motorm2x") never
# matches. A variant hit scores NORMALISED_CONFIDENCE (0.8), not
# 1.0 — downstream PPR already weighs seeds by confidence
# (``runtime/retrieval_ppr.py``), so a normalised hit counts
# slightly less than an exact one. When nothing links confidently
# the gazetteer returns [] — the QueryPlan caller says so
# explicitly; there is no scored-guess fallback here.

NORMALISED_CONFIDENCE = 0.8

_SEP = r"[-_\s]*"  # optional separator(s) between code parts


def _strip_diacritics(text: str) -> str:
    """NFKD strip of combining marks; Vietnamese đ/Đ are not
    decomposable, so they are mapped to d by hand."""
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    return stripped.replace("đ", "d").replace("Đ", "D")


def _variant_asset(match: re.Match[str]) -> list[tuple[str, str]]:
    return [("Asset", f"O{match.group(1)}")]


def _variant_fault(match: re.Match[str]) -> list[tuple[str, str]]:
    """Fault-code expansion through a normalised code: same
    FAULT_CODE_LINKS as the exact pass (unknown fault -> no seeds)."""
    asset, component = FAULT_CODE_LINKS.get(f"F-{match.group(1)}", (None, None))
    out: list[tuple[str, str]] = []
    if asset:
        out.append(("Asset", asset))
    if component:
        out.append(("Component", component))
    return out


def _variant_motor(match: re.Match[str]) -> list[tuple[str, str]]:
    return [("Component", f"motor-M{match.group(1)}")]


def _variant_door_sensor(match: re.Match[str]) -> list[tuple[str, str]]:
    return [("Component", f"door-sensor-DS{match.group(1)}")]


def _variant_controller(match: re.Match[str]) -> list[tuple[str, str]]:
    return [("Component", f"controller-C{match.group(1)}")]


def _variant_brake(match: re.Match[str]) -> list[tuple[str, str]]:
    return [("Component", f"brake-BK{match.group(1)}")]


def _variant_building(match: re.Match[str]) -> list[tuple[str, str]]:
    return [("Building", f"building-B{match.group(1)}")]


_VARIANT_CODE_PATTERNS: list[
    tuple[re.Pattern[str], Callable[[re.Match[str]], list[tuple[str, str]]]]
] = [
    (re.compile(r"\bo" + _SEP + r"(\d{2})\b"), _variant_asset),
    (re.compile(r"\bf" + _SEP + r"(\d{3})\b"), _variant_fault),
    (re.compile(r"\bmotor" + _SEP + r"m" + _SEP + r"(\d)\b"), _variant_motor),
    (
        re.compile(r"\bdoor" + _SEP + r"sensor" + _SEP + r"ds" + _SEP + r"(\d)\b"),
        _variant_door_sensor,
    ),
    (re.compile(r"\bcontroller" + _SEP + r"c" + _SEP + r"(\d)\b"), _variant_controller),
    (re.compile(r"\bbrake" + _SEP + r"bk" + _SEP + r"(\d)\b"), _variant_brake),
    (re.compile(r"\bbuilding" + _SEP + r"b" + _SEP + r"(\d)\b"), _variant_building),
]

# Document ids in the demo corpus. A doc id's hyphens are the only
# thing separating its words, so a fully concatenated mention
# ("docmotoroverheat") can only be segmented back with this
# vocabulary; a hyphen/space variant of an UNKNOWN doc id does not
# link (its exact hyphenated form still does, via _CODE_RE).
KNOWN_DOC_IDS = frozenset({
    "doc-door-sensor",
    "doc-motor-overheat",
    "doc-brake-inspection",
    "doc-controller-errors",
    "doc-leveling",
    "doc-rope-inspection",
    "doc-o23-overview",
})
_DOC_COMPACT = {doc_id.replace("-", ""): doc_id for doc_id in KNOWN_DOC_IDS}
# At most two word-segments after "doc": every known doc id has one
# or two, and capping the run keeps a following doc mention (or any
# following word) from being swallowed into this match.
_DOC_RUN_RE = re.compile(r"\bdoc" + _SEP + r"([a-z0-9]+(?:[-_\s]+[a-z0-9]+)?)\b")


def _doc_variant_matches(norm_text: str) -> list[tuple[int, str]]:
    """(start, canonical doc id) for separator variants of doc codes.

    Resolution order per match: (1) the separator-stripped mention
    equals a known doc id ("docmotoroverheat" -> doc-motor-overheat);
    (2) the longest leading segments form a known doc id
    ("doc motor overheat manual" -> doc-motor-overheat, "manual" is
    not part of the id; "doc leveling for ..." -> doc-leveling).
    Nothing else links: the run is greedy over up to two segments,
    so a following word can land inside it ("docmotoroverheated
    manual"), and reconstructing an unknown id from those segments
    would invent a node — unknown doc ids link only through their
    exact hyphenated form.
    """
    matches: list[tuple[int, str]] = []
    for match in _DOC_RUN_RE.finditer(norm_text):
        segments = re.split(r"[-_\s]+", match.group(1))
        known = _DOC_COMPACT.get("doc" + "".join(segments))
        if known is not None:
            matches.append((match.start(), known))
            continue
        for end in range(len(segments), 0, -1):
            candidate = "doc-" + "-".join(segments[:end])
            if candidate in KNOWN_DOC_IDS:
                matches.append((match.start(), candidate))
                break
    return matches


def _variant_name_pattern(name: str) -> re.Pattern[str]:
    """Component-name pattern on normalised text: the words may be
    joined by a hyphen/space/underscore or nothing ("door-sensor",
    "doorsensor"), and diacritic forms match via the stripped text."""
    parts = [re.escape(word) for word in name.split(" ")]
    return re.compile(r"\b" + _SEP.join(parts) + r"\b")


def _node_type(key: str) -> str:
    if key.startswith("building-"):
        return "Building"
    if key.startswith("doc-"):
        return "Document"
    if re.match(r"^O\d{2}$", key):
        return "Asset"
    return "Component"


def gazetteer_seeds(question: str) -> list[dict]:
    """Deterministic seed entities from a question.

    Returns ``[{"type": ..., "key": ..., "confidence": ...}]`` in
    first-mention order, deduped by key. Confidence is 1.0 for an
    exact gazetteer hit (today's behaviour, unchanged) and 0.8 for
    a normalised-variant hit (Phase 4: diacritics-insensitive,
    hyphen/space/underscore separator variants of the same codes —
    see the Phase 4 block above). An exact hit and its normalised
    twin are the same node: the 1.0 is kept.

    Ordering note (deliberate): exact seeds keep today's order
    (codes in mention order, then component names), so an
    exact-only question returns byte-identical output to Phase 1.
    Variant-only seeds slot in by first-mention position among the
    seeds already placed — for the usual all-code question that is
    exact first-mention order; a variant mentioned before a
    component-name exact seed lands before the code seeds but the
    name seed itself stays last, as it does today. [] when nothing
    links confidently: no LLM linker, no scored guesses.
    """
    text = (question or "").lower()
    seeds: list[dict] = []
    seen: set[str] = set()
    positions: list[int] = []  # mention position per seed, same order

    def _add(node_type: str, key: str, position: int) -> None:
        if key not in seen:
            seen.add(key)
            seeds.append({"type": node_type, "key": key, "confidence": 1.0})
            positions.append(position)

    # Exact pass — unchanged from Phase 1.
    for match in _CODE_RE.finditer(text):
        code = _canonical_code(match.group(1))
        if code.startswith("F-"):
            asset, component = FAULT_CODE_LINKS.get(code, (None, None))
            if asset:
                _add("Asset", asset, match.start())
            if component:
                _add("Component", component, match.start())
        else:
            _add(_node_type(code), code, match.start())
    for name, (node_type, key) in COMPONENT_NAME_LINKS.items():
        name_match = re.search(rf"\b{re.escape(name)}\b", text)
        if name_match:
            _add(node_type, key, name_match.start())

    # Phase 4 variant pass on diacritics-stripped text (confidence 0.8).
    norm_text = _strip_diacritics(text)
    offers: list[tuple[int, str, str]] = []
    for pattern, expand in _VARIANT_CODE_PATTERNS:
        for match in pattern.finditer(norm_text):
            for node_type, key in expand(match):
                offers.append((match.start(), node_type, key))
    for start, key in _doc_variant_matches(norm_text):
        offers.append((start, "Document", key))
    for name, (node_type, key) in COMPONENT_NAME_LINKS.items():
        name_match = _variant_name_pattern(name).search(norm_text)
        if name_match:
            offers.append((name_match.start(), node_type, key))
    offers.sort(key=lambda offer: offer[0])  # stable: first-mention order
    for position, node_type, key in offers:
        if key in seen:
            continue  # exact twin already placed at 1.0
        seen.add(key)
        index = next(
            (i for i, pos in enumerate(positions) if pos > position), len(seeds)
        )
        seeds.insert(
            index,
            {"type": node_type, "key": key, "confidence": NORMALISED_CONFIDENCE},
        )
        positions.insert(index, position)
    return seeds


# --- Adjacency ---------------------------------------------------------
# adjacency: node key -> {"type": str, "neighbors": [(rel, to_key, to_type)]}
# Edges are stored directed as modelled, but distance/path features
# are computed on the underlying undirected graph (report §3.2).

def build_demo_adjacency() -> dict:
    """Adjacency from RELATIONS + payload doc links (fake / eval graph).

    Payload links add the doc<->entity edges the ontology seed does
    not model for the 4 non-Document docs (e.g. doc-controller-errors
    <-> O21 / controller-C1); without them those docs would be
    unreachable from any seed.
    """
    from runtime.retrieval_linking import DOCUMENT_LINKS
    from seeds.demo_data import RELATIONS

    adjacency: dict[str, dict] = {}

    def _ensure(key: str, node_type: str) -> None:
        adjacency.setdefault(key, {"type": node_type, "neighbors": []})

    def _edge(from_key: str, from_type: str, rel: str, to_key: str, to_type: str) -> None:
        _ensure(from_key, from_type)
        _ensure(to_key, to_type)
        adjacency[from_key]["neighbors"].append((rel, to_key, to_type))
        adjacency[to_key]["neighbors"].append((rel, from_key, from_type))

    for from_id, from_label, rel, to_id, to_label in RELATIONS:
        _edge(from_id, from_label, rel, to_id, to_label)
    for doc_id, links in DOCUMENT_LINKS.items():
        for node_key in links["node_ids"]:
            if node_key == doc_id:
                continue
            # Payload link = the DOCUMENTED_BY-style doc<->entity edge.
            _edge(doc_id, "Document", "DOCUMENTED_BY", node_key, _node_type(node_key))
    return adjacency


def node_degree(adjacency: dict, key: str) -> int:
    return len(adjacency.get(key, {}).get("neighbors", []))


def _relation_weight(rel: str, via_key: str, adjacency: dict) -> float:
    if rel == "LOCATED_IN" and node_degree(adjacency, via_key) > HUB_DEGREE:
        return HUB_LOCATED_IN_WEIGHT
    return RELATION_WEIGHTS.get(rel, 0.0)


def expand_one_hop(
    seeds: list[dict],
    adjacency: dict,
    *,
    max_fanout_per_type: int = MAX_FANOUT_PER_TYPE,
    max_nodes: int = MAX_EXPANSION_NODES,
) -> dict:
    """1-hop expansion from seeds over the allowlisted relations.

    Returns ``{"nodes": {key: {"type", "rel", "from_seed", "via_hub"}},
    "truncated": bool, "nodes_returned": int}``. ``nodes`` are the
    distinct neighbours reached (seeds themselves excluded). Fan-out
    is capped per (seed, relation type); the total is capped at
    ``max_nodes``; both caps are deterministic (neighbours sorted by
    key) and truncation is reported, never silent.
    """
    nodes: dict[str, dict] = {}
    truncated = False
    for seed in seeds:
        seed_key = seed["key"]
        by_type: dict[str, list] = {}
        for rel, to_key, to_type in adjacency.get(seed_key, {}).get("neighbors", []):
            if rel not in ALLOWED_RELATIONS or to_key == seed_key:
                continue
            by_type.setdefault(rel, []).append((to_key, to_type))
        for rel, neighbours in by_type.items():
            neighbours.sort(key=lambda item: item[0])
            if len(neighbours) > max_fanout_per_type:
                truncated = True
            for to_key, to_type in neighbours[:max_fanout_per_type]:
                if to_key in nodes:
                    continue
                if len(nodes) >= max_nodes:
                    truncated = True
                    break
                nodes[to_key] = {
                    "type": to_type,
                    "rel": rel,
                    "from_seed": seed_key,
                    "via_hub": rel == "LOCATED_IN"
                    and node_degree(adjacency, to_key) > HUB_DEGREE,
                }
            if len(nodes) >= max_nodes:
                truncated = True
                break
        if len(nodes) >= max_nodes:
            truncated = True
            break
    return {"nodes": nodes, "truncated": truncated, "nodes_returned": len(nodes)}


def expand_via_graph_query(
    seeds: list[dict],
    *,
    store=None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """1-hop expansion through the live graph_query read functions.

    Calls ``ontology/knowledge/graph_query.py::neighbors`` (depth 1)
    per seed and converts the rows into the same shape as
    ``expand_one_hop``. Any store error / timeout degrades to an
    empty expansion (``graph_missing`` upstream) instead of failing
    the query.
    """
    from ontology.knowledge.graph_query import neighbors as graph_neighbors

    started = time.monotonic()
    nodes: dict[str, dict] = {}
    truncated = False
    for seed in seeds:
        if time.monotonic() - started > timeout_seconds:
            truncated = True
            break
        try:
            rows = graph_neighbors(
                seed["type"], seed["key"], depth=1, limit=MAX_FANOUT_PER_TYPE * 4,
                store=store,
            )
        except Exception as exc:  # noqa: BLE001 — graph is advisory only
            logger.warning("graph expansion failed for seed %s: %s", seed["key"], exc)
            return {"nodes": {}, "truncated": True, "nodes_returned": 0, "error": str(exc)}
        by_type: dict[str, list] = {}
        for row in rows:
            if row.get("type") not in ALLOWED_RELATIONS:
                continue
            by_type.setdefault(row["type"], []).append(row)
        for rel, group in by_type.items():
            group.sort(key=lambda r: r.get("to") or "")
            if len(group) > MAX_FANOUT_PER_TYPE:
                truncated = True
            for row in group[:MAX_FANOUT_PER_TYPE]:
                to_key = row.get("to")
                if not to_key or to_key in nodes:
                    continue
                if len(nodes) >= MAX_EXPANSION_NODES:
                    truncated = True
                    break
                nodes[to_key] = {
                    "type": row.get("to_label") or "Entity",
                    "rel": rel,
                    "from_seed": seed["key"],
                    "via_hub": False,
                }
    return {"nodes": nodes, "truncated": truncated, "nodes_returned": len(nodes)}


# --- Per-candidate graph features --------------------------------------

def doc_linked_nodes(doc_id: str, payload: dict | None = None) -> list[str]:
    """Node ids a doc is linked to: payload first, linking table fallback."""
    if payload and payload.get("node_ids"):
        return list(payload["node_ids"])
    from runtime.retrieval_linking import DOCUMENT_LINKS

    return list(DOCUMENT_LINKS.get(doc_id, {}).get("node_ids", []))


def seed_coverage(doc_id: str, seeds: list[dict], adjacency: dict, *, max_hops: int = 2) -> float:
    """G6 seed coverage (report §4.1): ``|{seeds within 2 hops of d}| / |S|``.

    BFS from the document node over the undirected adjacency, up to
    ``max_hops``. A candidate sitting between two seeds (e.g. the
    Asset *and* its Component both named in the question) scores
    higher than one near a single seed — the conjunction ``d_min``
    misses. 0.0 when there are no seeds. Note the coverage radius
    counts hops on the *undirected* graph, like G1/G2.
    """
    if not seeds:
        return 0.0
    seen = {doc_id}
    frontier = {doc_id}
    for _ in range(max_hops):
        nxt: set[str] = set()
        for node in frontier:
            for _rel, to_key, _to_type in adjacency.get(node, {}).get("neighbors", []):
                if to_key not in seen:
                    seen.add(to_key)
                    nxt.add(to_key)
        frontier = nxt
    seed_keys = {s["key"] for s in seeds}
    return round(len(seen & seed_keys) / len(seed_keys), 4)


def graph_features(
    doc_id: str,
    seeds: list[dict],
    expansion: dict,
    adjacency: dict,
    *,
    payload: dict | None = None,
    ppr_scores: dict | None = None,
) -> dict:
    """G1/G2/G4/G5 for one candidate doc, against the seed set.

    - G1: 0 if a linked entity (or the doc itself) IS a seed; 1 if a
      linked entity / the doc is a 1-hop neighbour of a seed; else
      None (unreachable within 1 hop).
    - G2 path_score: max over seeds x linked entities of
      w_rel * 0.6**hops, with a direct seed link scoring 1.0
      (w=1.0, hops=0). A doc node reached from a seed via
      DOCUMENTED_BY scores 1.0 * 0.6 = 0.6.
    - G4: +1 if the doc links to a seed component (or, with no
      component seed, to a seed asset); -1 if it links to a *different*
      component than the seed component, or a different asset than
      the seed asset; else 0.
    - G5: direct=1 if the doc links to a seed (payload/DOCUMENTED_BY
      link), 0.5 if it links to a 1-hop neighbour of a seed, else 0;
      plus ``n_seeds_linked`` = distinct seeds touched.
    - G6 (Phase 2): seed coverage within 2 hops (``seed_coverage``).
    - G7 (Phase 2): PPR of the *document node* from ``ppr_scores``
      (``runtime/retrieval_ppr.py``); 0.0 when no scores are passed.
    - G8 (Phase 2): hub value of the best path's transit node —
      ``log(1 + degree)`` when the best path transits through a
      Building / generic Document, else 0.0. A direct seed link and
      a doc that IS a 1-hop neighbour have no transit node (the doc
      is the endpoint), so they are never hub-penalised; only the
      "linked entity is a 1-hop neighbour" branch has a transit node
      (that entity). ``g8_transit_key`` names it for the trace.
    """
    seed_keys = {s["key"] for s in seeds}
    seed_assets = {s["key"] for s in seeds if s["type"] == "Asset"}
    seed_comps = {s["key"] for s in seeds if s["type"] == "Component"}
    linked = doc_linked_nodes(doc_id, payload)
    linked_set = set(linked)
    expanded = expansion.get("nodes", {})

    # Best path: (path_score, hops, rels tuple, transit node or None).
    best = (0.0, None, (), None)
    if doc_id in seed_keys:
        best = (1.0, 0, (), None)
    for entity in linked:
        if entity in seed_keys:
            if 1.0 > best[0]:
                best = (1.0, 0, ("DOCUMENTED_BY",), None)
            continue
        if entity in expanded:
            rel = expanded[entity]["rel"]
            weight = _relation_weight(rel, entity, adjacency)
            score = weight * HOP_DECAY
            if score > best[0]:
                best = (score, 1, (rel,), entity)
    if doc_id in expanded:
        rel = expanded[doc_id]["rel"]
        weight = _relation_weight(rel, doc_id, adjacency)
        score = weight * HOP_DECAY
        if score > best[0]:
            best = (score, 1, (rel,), None)

    path_score, hops, best_rels, transit = best
    if doc_id in seed_keys or linked_set & seed_keys:
        g1: int | None = 0
        direct = 1.0
    elif hops == 1:
        g1 = 1
        direct = 0.5
    else:
        g1 = None
        direct = 0.0
    n_seeds_linked = len(linked_set & seed_keys) + (1 if doc_id in seed_keys else 0)

    # G4 entity-type match / mismatch.
    doc_comps = {k for k in linked_set if _node_type(k) == "Component"}
    doc_assets = {k for k in linked_set if _node_type(k) == "Asset"}
    if seed_comps:
        if doc_comps & seed_comps:
            g4 = 1
        elif doc_comps:
            g4 = -1
        else:
            g4 = 0
    elif seed_assets:
        if doc_assets & seed_assets:
            g4 = 1
        elif doc_assets:
            g4 = -1
        else:
            g4 = 0
    else:
        g4 = 0

    # G8 hub value: only a Building / generic Document transit node
    # counts — a Component/Asset transit is structural, not a hub.
    hub = 0.0
    if transit is not None:
        transit_type = adjacency.get(transit, {}).get("type")
        if transit_type in HUB_TRANSIT_TYPES:
            hub = round(math.log1p(node_degree(adjacency, transit)), 4)

    return {
        "g1_min_hops": g1,
        "g2_path_score": round(path_score, 4),
        "g4_type_match": g4,
        "g5_direct": direct,
        "g5_n_seeds_linked": n_seeds_linked,
        "best_path_rels": list(best_rels),
        "g6_coverage": seed_coverage(doc_id, seeds, adjacency),
        "g7_ppr": round(float((ppr_scores or {}).get(doc_id, 0.0)), 6),
        "g8_hub": hub,
        "g8_transit_key": transit,
    }


def graph_score(features: dict) -> float:
    """Phase 1 graph_score from G1/G2/G4/G5 — the exact formula.

    ``graph_score = clip(path_score
        + 0.10 * G4                     # type match +1 / mismatch -1
        + 0.10 * (G5_direct - 0.5)      # direct 1 -> +0.05, 0.5 -> 0, 0 -> -0.05
        + hop_adj                       # G1 0 -> +0.05, 1 -> 0, None -> -0.05
        + 0.02 * min(max(n_seeds_linked - 1, 0), 2),  # G5 seed count
        -0.3, 1.2)``

    The floor is -0.3, not 0: a G4 mismatch (linked to a *different*
    asset/component than the seeds) must score *below* an unlinked
    doc, not tie with it — otherwise the wrong-asset doc (e.g. the
    O23 overview for an O21 question) keeps its vector rank. A doc
    with no path and no mismatch still scores 0 (its negative
    direct/hop terms are -0.05 each, clipped... in practice a
    no-path, no-mismatch doc lands at -0.10; the -0.3 floor only
    binds for a doc that is both mismatched and unlinked).

    G1 enters twice: through the 0.6**hops decay already inside
    path_score (G2), and through the small ``hop_adj`` term above.
    G3 (relation weight) is inside path_score. When no seed is
    linked at all (graph_missing) the caller sets the score to 0
    for every candidate instead of using this formula.
    """
    g1 = features.get("g1_min_hops")
    hop_adj = 0.05 if g1 == 0 else (0.0 if g1 == 1 else -0.05)
    extra_seeds = min(max(features.get("g5_n_seeds_linked", 0) - 1, 0), 2)
    score = (
        features.get("g2_path_score", 0.0)
        + 0.10 * features.get("g4_type_match", 0)
        + 0.10 * (features.get("g5_direct", 0.0) - 0.5)
        + hop_adj
        + 0.02 * extra_seeds
    )
    return round(max(-0.3, min(1.2, score)), 4)


def _minmax_log(values: list[float]) -> list[float]:
    """``log1p`` then min-max to [0, 1]; all-equal input -> all zeros.

    PPR masses are tiny and heavy-tailed (a seed-adjacent doc can
    hold 10x the mass of the next candidate); the log compresses
    the tail and the min-max puts the query's best PPR at 1.0.
    A z-score here would be wrong: on this corpus usually exactly
    one candidate has nonzero PPR, and a z-score would hand it
    z = +2.0 however small its mass — prototyped, it regressed
    blurred multi-hop nDCG@6 0.954 -> 0.908 (see
    docs/hybrid-retrieval-phase2.md). Report §4.7 sanctions this
    choice explicitly: "PPR via log then min-max".
    """
    logs = [math.log1p(max(v, 0.0)) for v in values]
    lo, hi = min(logs), max(logs)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in logs]


def phase2_graph_scores(
    feature_rows: list[dict],
    *,
    w_ppr: float = W_PPR,
    w_path: float = W_PATH,
    w_coverage: float = W_COVERAGE,
    hub_lambda: float = HUB_LAMBDA,
) -> list[float]:
    """Phase 2 graph_score — the locked full formula (§4.7).

    One score per feature row (one per candidate, same order)::

        path_term  = graph_score(row)   # Phase 1 G1/G2/G4/G5
                     - hub_lambda * row["g8_hub"]
        graph_score = w_ppr * minmax_log(PPR)      # G7
                    + w_path * z(path_term)        # G2 + G1/G4/G5 + G8
                    + w_coverage * coverage        # G6, raw [0, 1]

    with ``w_ppr=0.45, w_path=0.30, w_coverage=0.25`` by default.
    ``z`` is the per-query z-score clipped to [-2, 2]. Passing
    ``w_ppr=0`` is the PPR ablation: it zeroes the PPR term with
    NO renormalising of the surviving sub-weights, so the
    ablation measures PPR alone. Rows are ``graph_features``
    dicts; when a row has no G6/G7/G8 keys (pre-Phase-2 caller)
    they default to 0.0 and the result equals the z-scored
    Phase 1 score.
    """
    if not feature_rows:
        return []
    path_terms = [
        graph_score(row) - hub_lambda * float(row.get("g8_hub", 0.0))
        for row in feature_rows
    ]
    path_z = zscores(path_terms)
    ppr_norm = _minmax_log([float(row.get("g7_ppr", 0.0)) for row in feature_rows])
    return [
        round(
            w_ppr * pp + w_path * pz + w_coverage * float(row.get("g6_coverage", 0.0)),
            4,
        )
        for pp, pz, row in zip(ppr_norm, path_z, feature_rows)
    ]
