"""WS channel — SDK import placement + run_channel wiring (no live Lark).

Regression (seen live 2026-11-26): lark_oapi/ws/client.py captures
asyncio.get_event_loop() at import time into a module-global loop
that start() later drives with run_until_complete. Importing the SDK
from inside run_channel (under asyncio.run) bound that global to the
already-running loop, and start() died with "This event loop is
already running". The import must stay at the top of channel.py.
"""

import asyncio
import sys

from services.lark_frame.channel import run_channel
from services.lark_frame.webhook import FrameConfig, FrameState


def test_channel_imports_sdk_at_module_level():
    # Importing services.lark_frame.channel must already have imported
    # the SDK — if the import moves back inside run_channel, this fails.
    assert "lark_oapi.channel" in sys.modules
    assert "lark_oapi.ws.client" in sys.modules


def test_sdk_loop_is_not_the_running_loop():
    # The invariant the fix relies on: the SDK's module-global loop
    # (captured at import time, when no loop was running) is a
    # different loop from any asyncio.run loop.
    import lark_oapi.ws.client as ws_client

    assert ws_client.loop is not None

    async def main():
        assert asyncio.get_running_loop() is not ws_client.loop

    asyncio.run(main())


class _FakeChannel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.handlers = {}
        self.connect_called = False
        _FakeChannel.instances.append(self)

    def on(self, event, handler):
        self.handlers[event] = handler

    async def connect(self):
        self.connect_called = True


def test_run_channel_wires_handlers_and_health(monkeypatch):
    _FakeChannel.instances.clear()
    health_calls = []
    monkeypatch.setattr(
        "services.lark_frame.channel.FeishuChannel", _FakeChannel
    )
    monkeypatch.setattr(
        "services.lark_frame.channel.start_health_server",
        lambda state: health_calls.append(state),
    )
    state = FrameState(
        FrameConfig(
            app_id="cli_smoke",
            app_secret="s",
            platform_base_url="http://127.0.0.1:9",
            allowlist={"ou_1"},
            bot_open_id="ou_bot",
            transport="websocket",
        )
    )
    asyncio.run(run_channel(state))

    assert len(_FakeChannel.instances) == 1
    channel = _FakeChannel.instances[0]
    assert channel.kwargs["domain"] == "https://open.larksuite.com"
    assert channel.kwargs["app_id"] == "cli_smoke"
    assert channel.connect_called
    assert set(channel.handlers) == {
        "message",
        "cardAction",
        "reconnecting",
        "reconnected",
        "error",
    }
    assert health_calls == [state]
