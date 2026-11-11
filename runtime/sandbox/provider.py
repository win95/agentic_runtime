"""OpenSandbox client factory — the only place that touches the SDK.

The ``opensandbox`` package is imported lazily inside these functions
so the runtime still imports (and runs with the in-memory backend)
when the SDK is not installed. A missing SDK raises a clear error at
the point a sandbox is actually created, never at import time.

``OpenSandboxClient`` is the injectable seam the service layer uses:
tests pass a fake client with the same three methods instead of
talking to a server.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_connection_config(settings):
    """Connection config for the OpenSandbox server from settings.

    ``domain`` is ``host:port`` of the server (default localhost:8080);
    ``api_key`` is the server's API key (secret — .env only). When the
    server runs in Docker and this client on the host, the sandbox
    container IPs are unreachable — set ``sandbox_use_server_proxy``
    so traffic goes through the server proxy instead.
    """
    from opensandbox.config.connection_sync import ConnectionConfigSync

    return ConnectionConfigSync(
        domain=settings.sandbox_domain,
        api_key=settings.sandbox_api_key or None,
        use_server_proxy=bool(settings.sandbox_use_server_proxy),
    )


def create_sandbox(image, connection_config):
    """Create a sandbox from ``image`` and return the live handle."""
    from opensandbox import SandboxSync

    return SandboxSync.create(image=image, connection_config=connection_config)


def connect_sandbox(sandbox_id, connection_config):
    """Re-attach to an existing sandbox by id and return the handle."""
    from opensandbox import SandboxSync

    return SandboxSync.connect(sandbox_id, connection_config=connection_config)


def destroy_sandbox(handle) -> None:
    """Destroy the sandbox behind ``handle`` (best-effort for callers)."""
    handle.destroy()


class OpenSandboxClient:
    """Thin wrapper over the SDK classmethods (injectable for tests)."""

    def __init__(self, connection_config):
        self._config = connection_config

    def create(self, image):
        return create_sandbox(image, self._config)

    def connect(self, sandbox_id):
        return connect_sandbox(sandbox_id, self._config)

    def destroy(self, handle) -> None:
        destroy_sandbox(handle)


def default_client(settings) -> OpenSandboxClient:
    """Client from settings (the production path)."""
    return OpenSandboxClient(build_connection_config(settings))
