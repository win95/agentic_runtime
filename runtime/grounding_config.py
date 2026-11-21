"""Per-agent grounding config — which tool results ground a guest reply.

The shared grounding modules (``runtime/reply_grounding.py``,
``runtime/policy_grounding.py``) must not name any agent's tools: the
agent declares its own sources here, and the modules only apply the
branches the agent configured. An empty config means no grounding —
the reply passes through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingConfig:
    """Tool names + price field names that ground a guest reply.

    Quote sources (hotel example: availability + rates tools):

    - ``quote_tools`` — tools whose results count as quote evidence.
    - ``availability_tool`` / ``rates_tool`` — the two quote tools,
      read separately so the reply can combine availability with
      per-night and total prices.
    - ``total_price_field`` / ``per_night_field`` / ``currency_field``
      — price field names inside those tools' JSON.
    - ``room_field`` — the room-name field inside a rate row.
    - ``rates_key`` — the key holding the rate rows in a tool result.
    - ``booking_id_field`` — the booking-reference field.

    Policy sources (hotel example: search + read document tools):

    - ``policy_search_tool`` / ``policy_read_tool`` — the document
      search and read tools; when both are empty, policy grounding
      does not run.
    """

    quote_tools: frozenset[str] = frozenset()
    availability_tool: str = ""
    rates_tool: str = ""
    total_price_field: str = ""
    per_night_field: str = ""
    currency_field: str = ""
    room_field: str = ""
    rates_key: str = ""
    booking_id_field: str = ""
    policy_search_tool: str = ""
    policy_read_tool: str = ""

    @property
    def policy_tools(self) -> frozenset[str]:
        return frozenset(
            t for t in (self.policy_search_tool, self.policy_read_tool) if t
        )
