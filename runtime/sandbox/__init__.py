"""Sandbox backends for the Deep Agent (docs/sandbox.md).

The Deep Agent's file tools (ls/read/write/edit/delete/glob/grep) and
the ``execute`` tool run on a *backend*. The default is deepagents'
in-memory ``StateBackend``: files live only in the agent's LangGraph
state and there is no shell. This package adds a self-hosted
OpenSandbox backend: one OpenSandbox sandbox per session, so the agent
can write and run real analysis code in an isolated container.

Only the filesystem/`execute` surface moves into the sandbox. The
business tools (Postgres/Neo4j/Qdrant) stay host-side in the tool
registry — the sandbox never receives credentials or database access.
"""

from __future__ import annotations

from runtime.sandbox.opensandbox_backend import OpenSandboxBackend
from runtime.sandbox.provider import (
    OpenSandboxClient,
    build_connection_config,
    connect_sandbox,
    create_sandbox,
    default_client,
    destroy_sandbox,
)

__all__ = [
    "OpenSandboxBackend",
    "OpenSandboxClient",
    "build_connection_config",
    "connect_sandbox",
    "create_sandbox",
    "default_client",
    "destroy_sandbox",
]
