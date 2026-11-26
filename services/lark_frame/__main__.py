"""Lark frame entrypoint — `python -m services.lark_frame`.

Reads the environment, picks the transport (``LARK_TRANSPORT``:
``webhook`` default, ``websocket`` option — D-LARK-15) and runs the
frame. Only the transport layer changes between modes; routing, user
sync, enqueue, session, agent, tools and skills are identical.

Environment (see services/lark_frame/README.md):
- LARK_APP_ID / LARK_APP_SECRET — the agent's own Custom App
- LARK_VERIFICATION_TOKEN / LARK_ENCRYPT_KEY — webhook mode only
- PLATFORM_BASE_URL / PLATFORM_API_KEY — the platform API
- LARK_ALLOWLIST — comma-separated sender open_ids (D-LARK-06)
- LARK_BOT_OPEN_ID — override for @bot detection (optional; the
  frame fetches it via bot/v3/info when unset)
- LARK_TRANSPORT / FRAME_HOST / FRAME_PORT / REDIS_URL

The frame also loads the repo-root ``.env`` at startup (shell env
wins over file values), so the vars can live in the same file the
platform reads instead of being passed on every command line.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from services.lark_frame.webhook import FrameConfig, FrameState, run_webhook

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_env_file(path: Path | None = None) -> Path | None:
    """Load the repo-root .env into os.environ; returns the path loaded.

    The path is resolved from this file's location, not the CWD, so it
    works from anywhere. Shell env wins over file values (``override``
    is False). A missing file is a silent no-op — shell env alone
    still works.
    """
    env_path = path or Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return None
    load_dotenv(env_path, override=False)
    return env_path


def load_config() -> FrameConfig:
    app_id = _env("LARK_APP_ID")
    app_secret = _env("LARK_APP_SECRET")
    if not app_id or not app_secret:
        raise SystemExit(
            "LARK_APP_ID / LARK_APP_SECRET are not set — create the agent's "
            "own Custom App first (Step 1 of the lark_me plan)"
        )
    allowlist = {
        part.strip()
        for part in _env("LARK_ALLOWLIST").split(",")
        if part.strip()
    }
    transport = _env("LARK_TRANSPORT", "webhook").lower()
    if transport not in ("webhook", "websocket"):
        raise SystemExit(f"unknown LARK_TRANSPORT: {transport!r}")
    return FrameConfig(
        app_id=app_id,
        app_secret=app_secret,
        verification_token=_env("LARK_VERIFICATION_TOKEN"),
        encrypt_key=_env("LARK_ENCRYPT_KEY"),
        platform_base_url=_env("PLATFORM_BASE_URL", "http://localhost:8000"),
        platform_api_key=_env("PLATFORM_API_KEY"),
        allowlist=allowlist,
        bot_open_id=_env("LARK_BOT_OPEN_ID") or None,
        host=_env("FRAME_HOST", "0.0.0.0"),
        port=int(_env("FRAME_PORT", "8000") or 8000),
        health_port=int(_env("FRAME_HEALTH_PORT", "8002") or 8002),
        redis_url=_env("REDIS_URL") or None,
        transport=transport,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    env_path = load_env_file()
    if env_path is not None:
        logger.info("Lark frame loaded env from %s", env_path)
    config = load_config()
    if not config.platform_api_key:
        logger.warning(
            "PLATFORM_API_KEY is empty — the frame cannot call the "
            "platform (every enqueue will fail)"
        )
    if not config.allowlist:
        logger.warning(
            "LARK_ALLOWLIST is empty — every sender will be refused "
            "until tran adds open_ids to it"
        )
    state = FrameState(config)
    if config.transport == "websocket":
        import asyncio

        from services.lark_frame.channel import run_channel

        asyncio.run(run_channel(state))
    else:
        run_webhook(state)


if __name__ == "__main__":
    sys.exit(main())
