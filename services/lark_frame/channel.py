"""WebSocket frame for agent_lark_me (Step 2, D-LARK-15).

Same routing + sync + enqueue as the webhook frame — only the
transport differs. The SDK channel (``lark_oapi.channel.FeishuChannel``)
owns the connection: outbound WS (no public URL needed, runs local /
behind NAT), auto-reconnect (``reconnecting`` / ``reconnected`` events),
and its own Deduper on top of the frame's seen-set. Card callbacks
arrive on the same connection as ``cardAction`` events — no
``/lark/card`` URL, no 1s HTTP challenge, no hand-rolled SHA1.

The channel SDK is legacy (supported until 2027-06-02); its successor
is the standalone ``lark-channel-sdk`` (import ``lark_channel``) — that
swap is unverified (README claim only) and NOT done here.
"""

from __future__ import annotations

import asyncio
import json
import logging

from services.lark_frame.routing import (
    NormalizedMessage,
)
from services.lark_frame.webhook import (
    FrameState,
    dispatch_card_action,
    handle_message_event,
)

logger = logging.getLogger(__name__)


def normalize_channel_message(msg) -> NormalizedMessage | None:
    """SDK ``InboundMessage`` -> NormalizedMessage."""
    message_id = getattr(msg, "message_id", "") or ""
    conversation = getattr(msg, "conversation", None)
    chat_id = getattr(conversation, "chat_id", "") or ""
    if not message_id or not chat_id:
        return None
    sender = getattr(msg, "sender", None)
    sender_id = {
        "union_id": getattr(sender, "union_id", "") or "",
        "user_id": getattr(sender, "user_id", "") or "",
        "open_id": getattr(sender, "open_id", "") or "",
    }
    mentions = []
    for m in getattr(msg, "mentions", None) or []:
        open_id = getattr(m, "open_id", "") or ""
        if open_id:
            mentions.append(open_id)
    # The SDK normalizes content into content_text; raw_content_type
    # carries the original Lark msg_type ("text", "image", ...).
    msg_type = getattr(msg, "raw_content_type", "") or "text"
    return NormalizedMessage(
        message_id=message_id,
        root_id=getattr(msg, "reply_to_message_id", None) or None,
        chat_id=chat_id,
        chat_type=getattr(conversation, "chat_type", "") or "",
        sender_type="app" if getattr(sender, "is_bot", False) else "user",
        sender_id=sender_id,
        text=getattr(msg, "content_text", "") or "",
        mentions=mentions,
        msg_type=msg_type,
    )


async def run_channel(state: FrameState) -> None:
    """Connect the WS channel forever (auto-reconnect is the SDK's)."""
    from lark_oapi.channel import FeishuChannel

    cfg = state.config
    state.ensure_bot_open_id()
    channel = FeishuChannel(
        app_id=cfg.app_id,
        app_secret=cfg.app_secret,
        domain="https://open.larksuite.com",
    )

    async def on_message(msg) -> None:
        normalized = normalize_channel_message(msg)
        if normalized is None:
            return
        # The message path is sync (urllib) — run it off the loop.
        await asyncio.to_thread(handle_message_event, state, {
            "message": {
                "message_id": normalized.message_id,
                "root_id": normalized.root_id,
                "chat_id": normalized.chat_id,
                "chat_type": normalized.chat_type,
                "message_type": normalized.msg_type,
                "content": json.dumps({"text": normalized.text}, ensure_ascii=False),
                "mentions": [
                    {"id": {"open_id": m}} for m in normalized.mentions
                ],
            },
            "sender": {
                "sender_type": normalized.sender_type,
                "sender_id": normalized.sender_id,
            },
        })

    async def on_card_action(event) -> None:
        # Rebuild the webhook-shaped payload from the SDK event so the
        # one card dispatcher serves both transports.
        action = getattr(event, "action", None)
        value = getattr(action, "value", None)
        operator = getattr(event, "operator", None)
        payload = {
            "type": "card.action.trigger",
            "action": {"value": value},
            "context": {"open_message_id": getattr(event, "message_id", "")},
            "operator": {
                "operator_id": {
                    "open_id": getattr(operator, "open_id", "") or "",
                    "user_id": getattr(operator, "user_id", "") or "",
                }
            },
        }
        await asyncio.to_thread(dispatch_card_action, state, payload)

    async def on_reconnecting() -> None:
        logger.warning("Lark WS channel reconnecting...")

    async def on_reconnected() -> None:
        logger.info("Lark WS channel reconnected")

    async def on_error(err) -> None:
        logger.warning("Lark WS channel error: %s", err)

    channel.on("message", on_message)
    channel.on("cardAction", on_card_action)
    channel.on("reconnecting", on_reconnecting)
    channel.on("reconnected", on_reconnected)
    channel.on("error", on_error)
    logger.info("Lark WS channel connecting (transport=websocket)...")
    await channel.connect()
