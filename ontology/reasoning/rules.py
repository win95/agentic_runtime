"""Reasoning v1 rule catalog (design §4.3, R1–R3).

Only three rules in v1 — enough to prove the mechanism; extend by adding
entries here plus the matching logic in ``materialize.py``.
"""

from __future__ import annotations

RULES: list[dict[str, str]] = [
    {
        "id": "R1",
        "name": "subclass",
        "description": (
            "An instance of class C is also an instance of every ancestor "
            "of C. Query-time only (see subclass_ancestors); class labels "
            "are never materialized on instance nodes."
        ),
    },
    {
        "id": "R2",
        "name": "inverse",
        "description": (
            "PART_OF <-> HAS_COMPONENT: writing one direction derives the "
            "other direction as a derived edge."
        ),
    },
    {
        "id": "R3",
        "name": "transitive",
        "description": (
            "PART_OF is transitive: A -PART_OF-> B and B -PART_OF-> C "
            "derives A -PART_OF-> C (both directions expanded at write "
            "time, see materialize_for_write)."
        ),
    },
]

# Inverse object property for each direction (R2). Design §4.1 stores
# this on the (:ObjectProperty) metadata node as ``inverse_of``; v1 keeps
# the same mapping here as the single source of truth for the reasoner
# until the metadata module (B2) lands.
INVERSE_RELATIONS: dict[str, str] = {
    "PART_OF": "HAS_COMPONENT",
    "HAS_COMPONENT": "PART_OF",
}
