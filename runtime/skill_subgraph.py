"""Compatibility shim — flow subgraph replaced the legacy node compiler."""

from runtime.subgraph.flow import (
    build_flow_subgraph as build_skill_subgraph,
    run_flow_subgraph as run_skill_subgraph,
    use_flow_subgraph as use_skill_subgraph,
)

__all__ = [
    "build_skill_subgraph",
    "run_skill_subgraph",
    "use_skill_subgraph",
]