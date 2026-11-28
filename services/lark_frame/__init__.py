"""Lark frame — the webhook / WebSocket front door for agent_lark_me.

The frame only verifies + routes + enqueues, then returns 200
immediately. It never calls run_mission directly and never sends the
final answer (the agent replies via its Lark tools — D-LARK-09).
"""

from services.lark_frame.webhook import FrameConfig, FrameState

__all__ = ["FrameConfig", "FrameState"]
