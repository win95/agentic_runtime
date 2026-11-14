"""Neo4j 1-hop expansion + per-candidate graph features — Phases 1–2
(report §3.2, §4.1, §4.2, §4.7; locked decisions: 1 hop, fan-out
<=10 per (seed, relation type), total <=50 nodes).

Seeds come ONLY from the dynamic gazetteer (``refresh_gazetteer_codes``):
the live ontology's entity keys (Asset, Component, Building,
Document), fetched explicitly into an in-memory cache — NO TTL, NO
background/beat job, refresh on seed/drain or by hand only, per the
user's sync principle (codes change slowly; sync fixed things once,
never duplicate the tools' fetch logic). There are NO hardcoded
demo patterns: until ``refresh_gazetteer_codes`` has loaded keys,
``gazetteer_seeds`` links NOTHING (it fails closed to ``[]``), and
a refresh that fails leaves the cache EMPTY, so the gazetteer
still links nothing rather than falling back to any built-in
vocabulary. A wrong link is worse than no link.

Fault codes are NOT ontology entities — there is no Fault entity
type — so fault codes do not link at all. The dynamic list covers
Asset/Component/Building/Document keys only.

Expansion runs over an allowlisted relation set on an adjacency
supplied by the caller (``build_live_adjacency`` /
``build_expansion_adjacency`` over real store data); the live path
calls the existing ``ontology/knowledge/graph_query.py`` read
functions instead (``expand_via_graph_query``) — no new Cypher.

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

Dynamic matching reuses the Phase 4 normalisation: verbatim
(case-insensitive whole-token) hits score 1.0, separator /
diacritics variants score 0.8, and a 1.0 twin always wins the
dedup. Patterns are compiled once at refresh, not per question.
Scores, fusion weights, guardrails and gating are untouched:
this only changes WHICH keys can become seeds.
"""

from __future__ import annotations

import logging
import math
import re
import time
import unicodedata

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

# Live-adjacency cap (Phase 5 follow-up): the PPR candidate subgraph
# is <= ~200 nodes by design (report §11.3 PPR cap). The live builder
# counts seeds + expansion + second-hop nodes against this cap.

MAX_LIVE_ADJACENCY_NODES = 200


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
# Seeds come ONLY from the dynamic gazetteer (see module docstring
# and ``refresh_gazetteer_codes`` below). No hardcoded patterns.
# The dynamic matcher reuses the Phase 4 normalisation: verbatim
# hits score 1.0, separator / diacritics variants score 0.8.

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
    first-mention order, deduped by key. ONLY keys in the dynamic
    cache (loaded by ``refresh_gazetteer_codes`` from the live
    ontology) can link — verbatim hits score 1.0, separator /
    diacritics variants score 0.8 (``NORMALISED_CONFIDENCE``), a
    1.0 twin wins the dedup. ``[]`` when the cache is empty
    (never refreshed, or the refresh failed) or nothing listed
    is mentioned: no hardcoded fallback, no LLM linker, no
    scored guesses.
    """
    return _dynamic_gazetteer_seeds(question)


# --- Phase 5: dynamic gazetteer (live ontology keys) ----------------
# See the module docstring for the contract. Summary:
# - ``refresh_gazetteer_codes(store=None)`` is the ONLY way keys
#   enter the cache; it is called explicitly by the live factory
#   (and by hand), never on a timer, never implicitly by
#   ``gazetteer_seeds``.
# - The cache is in-memory only, no TTL. ``clear_gazetteer_codes``
#   empties it (the gazetteer then links nothing until refreshed).
# - A refresh fetches ALL FOUR labels first and only commits on
#   full success: any store failure leaves the cache EMPTY and
#   never raises.
# - Fault codes are not fetched: there is no Fault entity type.
#   A fault code links nothing.
# - Dynamic matching reuses the Phase 4 normalisation: verbatim
#   (case-insensitive whole-token) hits score 1.0, separator /
#   diacritics variants score 0.8, and a 1.0 twin always wins the
#   dedup. Patterns are compiled once at refresh, not per question.
# - Scores, fusion weights, guardrails and gating are untouched:
#   this only changes WHICH keys can become seeds.

_DYNAMIC_ENTITY_TYPES: tuple[tuple[str, str], ...] = (
    ("Asset", "assets"),
    ("Component", "components"),
    ("Building", "buildings"),
    ("Document", "documents"),
)

_DYNAMIC_CODES: dict[str, list[str]] = {}
_DYNAMIC_PATTERNS: list[tuple[str, str, re.Pattern[str], re.Pattern[str]]] = []


def _dynamic_key_parts(key: str) -> list[str]:
    """Atomic parts of a key for the normalised-variant pattern.

    Split on non-alphanumerics, then split each segment into
    letter / digit runs (``M9`` -> ``m``, ``9``; ``O99`` -> ``o``,
    ``99``), so separators may appear between a code's letters
    and its digits exactly as in the Phase 4 variant pass
    (``O-99`` / ``O 99`` -> ``O99``, ``motor M 9`` -> ``motor-M9``).
    Diacritics are stripped so a key with a diacritic matches its
    plain spelling. A key with no alphanumeric part has no safe
    pattern and is skipped by the caller.
    """
    parts: list[str] = []
    for segment in re.split(r"[^A-Za-z0-9]+", _strip_diacritics(key.lower())):
        parts.extend(re.findall(r"[A-Za-z]+|[0-9]+", segment))
    return parts


def _dynamic_patterns_for(
    key: str,
) -> tuple[re.Pattern[str], re.Pattern[str]] | None:
    """(verbatim, variant) patterns for one cached key, or None.

    Verbatim: the key's exact spelling, case-insensitive, as a
    whole token (``\b`` both ends). Variant: the key's atomic
    parts joined by optional separators (``_SEP``), also
    whole-token bounded, on diacritics-stripped text — the same
    normalisation as Phase 4. Word boundaries on both patterns
    mean a key embedded in a longer token (``XO99``, ``O99x``)
    never matches.
    """
    parts = _dynamic_key_parts(key)
    if not parts:
        return None
    verbatim = re.compile(r"\b" + re.escape(key.lower()) + r"\b")
    variant = re.compile(r"\b" + _SEP.join(re.escape(part) for part in parts) + r"\b")
    return verbatim, variant


def refresh_gazetteer_codes(store=None) -> dict:
    """Load the live ontology's entity keys into the gazetteer cache.

    Fetches keys per label via ``ontology/knowledge/graph_query.py::
    list_entity_keys`` (Asset, Component, Building, Document),
    stores them in the module-level cache and
    returns ``{"assets": n, "components": n, "buildings": n,
    "documents": n, "source": "ontology"}``. Called explicitly by
    the live factory (and by hand); never on a timer.

    On ANY store failure — including a failure halfway through
    the four fetches, or no graph store being configured — it
    returns ``{"assets": 0, "components": 0, "buildings": 0,
    "documents": 0, "source": "unreachable"}``, leaves the cache
    EMPTY (clearing any earlier successful refresh: a stale cache
    could link codes the ontology no longer has, and a wrong link
    is worse than no link), so the gazetteer links nothing, and
    never raises.
    """
    global _DYNAMIC_CODES, _DYNAMIC_PATTERNS
    try:
        from ontology.knowledge.graph_query import list_entity_keys

        fetched: dict[str, list[str]] = {}
        for entity_type, _summary_key in _DYNAMIC_ENTITY_TYPES:
            fetched[entity_type] = list_entity_keys(entity_type, store=store)
    except Exception as exc:  # noqa: BLE001 — refresh must never raise
        logger.warning("dynamic gazetteer refresh failed: %s", exc)
        _DYNAMIC_CODES = {}
        _DYNAMIC_PATTERNS = []
        return {
            "assets": 0,
            "components": 0,
            "buildings": 0,
            "documents": 0,
            "source": "unreachable",
        }
    codes: dict[str, list[str]] = {}
    patterns: list[tuple[str, str, re.Pattern[str], re.Pattern[str]]] = []
    for entity_type, _summary_key in _DYNAMIC_ENTITY_TYPES:
        keys = sorted(set(fetched[entity_type]))
        codes[entity_type] = keys
        for key in keys:
            built = _dynamic_patterns_for(key)
            if built is not None:
                patterns.append((entity_type, key, built[0], built[1]))
    # Commit only on full success: a second refresh replaces the
    # cache wholesale, so no stale keys survive it.
    _DYNAMIC_CODES = codes
    _DYNAMIC_PATTERNS = patterns
    return {
        "assets": len(codes["Asset"]),
        "components": len(codes["Component"]),
        "buildings": len(codes["Building"]),
        "documents": len(codes["Document"]),
        "source": "ontology",
    }


def clear_gazetteer_codes() -> None:
    """Drop the dynamic cache: the gazetteer links nothing until refreshed.

    Test helper (also the manual way to clear the cache): after
    this call ``gazetteer_seeds`` returns [] until
    ``refresh_gazetteer_codes`` loads keys again.
    """
    global _DYNAMIC_CODES, _DYNAMIC_PATTERNS
    _DYNAMIC_CODES = {}
    _DYNAMIC_PATTERNS = []


def _dynamic_gazetteer_seeds(question: str) -> list[dict]:
    """Seeds from the dynamic cache only.

    Verbatim hits score 1.0, normalised-variant hits 0.8; a key
    with both keeps the 1.0. Seeds come out in first-mention
    order. [] when the cache is empty (never refreshed, or the
    ontology was unreachable at refresh) or nothing listed is
    mentioned — there is no fallback vocabulary.
    """
    text = (question or "").lower()
    norm_text = _strip_diacritics(text)
    best: dict[str, dict] = {}
    for node_type, key, verbatim_re, variant_re in _DYNAMIC_PATTERNS:
        match = verbatim_re.search(text)
        if match is not None:
            current = best.get(key)
            if current is None or current["confidence"] < 1.0:
                best[key] = {
                    "type": node_type,
                    "key": key,
                    "confidence": 1.0,
                    "position": match.start(),
                }
            continue
        match = variant_re.search(norm_text)
        if match is not None and key not in best:
            best[key] = {
                "type": node_type,
                "key": key,
                "confidence": NORMALISED_CONFIDENCE,
                "position": match.start(),
            }
    ordered = sorted(best.values(), key=lambda seed: seed["position"])
    return [
        {"type": seed["type"], "key": seed["key"], "confidence": seed["confidence"]}
        for seed in ordered
    ]


# --- Adjacency ---------------------------------------------------------
# adjacency: node key -> {"type": str, "neighbors": [(rel, to_key, to_type)]}
# Edges are stored directed as modelled, but distance/path features
# are computed on the underlying undirected graph (report §3.2).

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


def _live_adjacency_edge(
    adjacency: dict, from_key: str, rel: str, to_key: str, to_type: str
) -> None:
    """Add one undirected edge (both directions), deduped.

    ``adjacency[from_key]`` and ``adjacency[to_key]`` must already
    exist (``_ensure``-ed by the caller); the neighbour lists are
    re-sorted by the caller once the build finishes.
    """
    forward = (rel, to_key, to_type)
    if forward not in adjacency[from_key]["neighbors"]:
        adjacency[from_key]["neighbors"].append(forward)
    back_type = adjacency[from_key]["type"]
    backward = (rel, from_key, back_type)
    if backward not in adjacency[to_key]["neighbors"]:
        adjacency[to_key]["neighbors"].append(backward)


def build_expansion_adjacency(seeds: list[dict], expansion: dict) -> dict:
    """Seeds + expansion nodes + their seed->node edges only.

    The degradation substrate for the live path (see
    ``build_live_adjacency``): real store data (the 1-hop expansion
    rows), partial — no second-hop edges, so PPR/coverage/hub see
    only the seed fringe. Same adjacency shape (node key ->
    ``{"type": str, "neighbors": [(rel, to_key, to_type)]}``).
    Never the demo adjacency: on the live path a demo edge would be
    a fabricated edge about the live ontology.
    """
    adjacency: dict[str, dict] = {}

    def _ensure(key: str, node_type: str) -> None:
        adjacency.setdefault(key, {"type": node_type, "neighbors": []})

    seed_types = {s["key"]: s.get("type") or "Entity" for s in seeds}
    for seed in seeds:
        _ensure(seed["key"], seed_types[seed["key"]])
    for to_key, node in (expansion.get("nodes") or {}).items():
        _ensure(to_key, node.get("type") or "Entity")
        from_seed = node.get("from_seed")
        if from_seed in adjacency:
            _live_adjacency_edge(
                adjacency,
                from_seed,
                node.get("rel") or "RELATED_TO",
                to_key,
                node.get("type") or "Entity",
            )
    for entry in adjacency.values():
        entry["neighbors"] = sorted(set(entry["neighbors"]), key=lambda e: (e[1], e[0]))
    return adjacency


def build_live_adjacency(
    seeds: list[dict],
    expansion: dict,
    *,
    store=None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_nodes: int = MAX_LIVE_ADJACENCY_NODES,
) -> dict:
    """Live adjacency for PPR (G7), coverage (G6) and hub (G8).

    Same adjacency shape (node key ->
    ``{"type": str, "neighbors": [(rel, to_key, to_type)]}``,
    stored directed but used undirected), built from REAL store
    data only: the seeds + the 1-hop expansion nodes + the 1-hop
    neighbours of each expansion node, so PPR/coverage/hub see
    real edges beyond the seed fringe. Without this the live path
    scored those features over the demo adjacency — a live doc's
    PPR would have been computed from demo edges.

    Rules:
    - Only ``ALLOWED_RELATIONS`` edges; neighbours are fetched via
      ``ontology/knowledge/graph_query.py::neighbors`` (depth 1)
      per node — no new Cypher.
    - Deterministic under the caps: nodes are visited in BFS
      order (seeds in given order, then expansion nodes sorted by
      key, then second-hop nodes in discovery order) and each
      node's neighbours are processed sorted by key, so the same
      store contents always yield the same adjacency.
    - Total nodes capped at ``max_nodes`` (default 200, the §11.3
      PPR cap — counting seeds + expansion + second-hop nodes).
      Seeds + expansion nodes are always present (they are the
      query's own fringe); the cap binds on second-hop additions
      only: an edge to a not-yet-known node past the cap is
      dropped, edges between known nodes are still added, and the
      truncation is reported with a warning log.
    - One shared wall-clock budget (``timeout_seconds``, default
      the same 0.15 s as the 1-hop expansion, §3.2): the budget is
      checked before every ``neighbors`` call, NOT per call — a
      per-call budget over up to ~51 calls (seeds + expansion)
      could stretch to ~7.6 s and blow the retrieval latency
      budget. On timeout the build stops and returns the partial
      adjacency built so far (seeds + expansion nodes are always
      in it, so it is never empty for a non-empty seed set).
    - A store error RAISES (it is not swallowed here): the caller
      (``HybridRetriever._candidate_rows``) treats a failed live
      build as a degradation and falls back to
      ``build_expansion_adjacency``. A partial graph after an
      error would be indistinguishable from a complete small
      graph, so the builder never returns one.

    Returns ``{}`` only when there are no seeds (the caller never
    calls it then — ``graph_missing`` applies instead).
    """
    from ontology.knowledge.graph_query import neighbors as graph_neighbors

    adjacency: dict[str, dict] = {}

    def _ensure(key: str, node_type: str) -> None:
        adjacency.setdefault(key, {"type": node_type, "neighbors": []})

    for seed in seeds:
        _ensure(seed["key"], seed.get("type") or "Entity")
    expansion_nodes = expansion.get("nodes") or {}
    for key, node in expansion_nodes.items():
        _ensure(key, node.get("type") or "Entity")

    started = time.monotonic()
    truncated = False
    # Seeds may also appear as expansion nodes (the live expansion
    # does not exclude them); dedup so each node is fetched once.
    queue = list(dict.fromkeys([s["key"] for s in seeds] + sorted(expansion_nodes)))
    queued = set(queue)
    index = 0
    while index < len(queue):
        key = queue[index]
        index += 1
        if time.monotonic() - started > timeout_seconds:
            truncated = True
            break
        # Limit 200 (the graph_query max): degree fidelity matters
        # for PPR/hub — truncating a node's neighbour list at the
        # expansion fan-out (40) would understate real hub degrees.
        # The total-node cap above bounds the memory instead.
        rows = graph_neighbors(
            adjacency[key]["type"],
            key,
            depth=1,
            limit=200,
            store=store,
        )
        rows.sort(key=lambda r: (r.get("to") or "", r.get("type") or ""))
        for row in rows:
            rel = row.get("type")
            to_key = row.get("to")
            if rel not in ALLOWED_RELATIONS or not to_key or to_key == key:
                continue
            to_type = row.get("to_label") or "Entity"
            if to_key not in adjacency:
                if len(adjacency) >= max_nodes:
                    truncated = True
                    continue
                _ensure(to_key, to_type)
                if to_key not in queued:
                    queued.add(to_key)
                    queue.append(to_key)
            _live_adjacency_edge(adjacency, key, rel, to_key, to_type)
    for entry in adjacency.values():
        entry["neighbors"] = sorted(set(entry["neighbors"]), key=lambda e: (e[1], e[0]))
    if truncated:
        logger.warning(
            "live adjacency truncated: %d nodes (cap %d, budget %.3fs)",
            len(adjacency),
            max_nodes,
            timeout_seconds,
        )
    return adjacency


# --- Per-candidate graph features --------------------------------------

def doc_linked_nodes(doc_id: str, payload: dict | None = None) -> list[str]:
    """Node ids a doc is linked to, from its Qdrant payload only.

    ``payload["node_ids"]`` is the link (see
    ``runtime/retrieval_linking.py``: every Qdrant point carries
    ``node_ids[]`` in its payload). A doc with no payload node
    ids links to nothing — there is no table fallback; an
    unlinked doc scores ``graph_score = 0`` and the vector arm
    carries it (guardrail 3 fallback).
    """
    if payload and payload.get("node_ids"):
        return list(payload["node_ids"])
    return []


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
    A z-score here would be wrong: usually exactly one candidate
    has nonzero PPR, and a z-score would hand it z = +2.0 however
    small its mass (see docs/hybrid-retrieval-phase2.md).
    Report §4.7 sanctions this choice explicitly: "PPR via log
    then min-max".
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
