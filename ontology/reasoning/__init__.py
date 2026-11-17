"""Ontology reasoning v1 (R1–R3) — materialize derived edges at write time."""

from __future__ import annotations

from ontology.reasoning.materialize import (
    base_ref_for,
    materialize_for_write,
    recompute_all,
    recompute_for_entity,
    remove_derived_for_base,
    subclass_ancestors,
    subclass_descendants,
)
from ontology.reasoning.rules import INVERSE_RELATIONS, RULES

__all__ = [
    "INVERSE_RELATIONS",
    "RULES",
    "base_ref_for",
    "materialize_for_write",
    "recompute_all",
    "recompute_for_entity",
    "remove_derived_for_base",
    "subclass_ancestors",
    "subclass_descendants",
]
