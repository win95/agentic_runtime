"""HotelReceptionistAgent — the hotel agent's own policy (plan §3.3).

Phase 2 (guest scope) + Phase 3 (write pipeline). Everything in here
used to be hard-coded hotel logic in the generic core:
``runtime/tools/registry.py`` (``_GUEST_SCOPED_TOOLS`` /
``_guest_scoped_tool`` / ``_booking_owner``),
``runtime/tool_preconditions.py`` (``_GUEST_READ_TOOLS`` /
``_booking_exists_for_session``) and ``runtime/write_pipeline_runner.py``
(the ``room_type`` / ``get_room_rates`` special-cases, the
``confirm_guest_booking`` name and the guest-facing booking strings).
The core now only calls the agent's override points
(``scope_tool`` / ``write_pipeline`` / ``confirm_tool``); an agent
without these overrides gets no scoping and the generic pipeline.
"""

from __future__ import annotations

from runtime.agents.base import BaseAgent
from runtime.agents.registry import register_agent_class
from runtime.grounding_config import GroundingConfig


class HotelReceptionistAgent(BaseAgent):
    """The hotel receptionist — one guest per session, one pipeline."""

    # Guest/booking tools whose data is scoped to the session's guest
    # (Layer 1: guest identity IS the session). Reads are never blocked
    # by name — out-of-scope data is withheld from the result instead.
    GUEST_SCOPED_TOOLS = frozenset(
        {
            "find_guest",
            "register_guest",
            "get_booking",
            "create_booking",
            "cancel_booking",
            "change_booking_dates",
        }
    )

    GUEST_READ_TOOLS = frozenset(
        {
            "find_guest",
            "get_booking",
            "list_payments",
            "get_task",
            "search_policy_documents",
            "read_policy_document",
        }
    )

    # The confirm tool speaks to a guest about a booking, so its
    # description keeps the hotel wording (the generic default on
    # BaseAgent says "write pipeline" instead).
    confirm_tool_description = (
        "Record that the guest explicitly confirmed the quoted booking "
        "in this conversation. Call once after they agree, before the "
        "create step runs."
    )

    # Grounding sources (Phase 4): the quote tools and price fields
    # the shared grounding modules read — no hotel tool name lives in
    # those modules anymore.
    grounding_config = GroundingConfig(
        quote_tools=frozenset({"check_room_availability", "get_room_rates"}),
        availability_tool="check_room_availability",
        rates_tool="get_room_rates",
        total_price_field="total_price",
        per_night_field="price_per_night",
        currency_field="currency",
        room_field="room_type",
        rates_key="rates",
        booking_id_field="booking_id",
        policy_search_tool="search_policy_documents",
        policy_read_tool="read_policy_document",
    )

    # -- Guest scope (Phase 2) -------------------------------------------

    @classmethod
    def guest_read_tool_names(cls) -> frozenset[str]:
        return cls.GUEST_READ_TOOLS

    @staticmethod
    def booking_owner(booking_id: str | None) -> str | None:
        """Guest id owning one booking, via the original get_booking tool.

        Returns None when the booking cannot be read (missing, error,
        or get_booking not registered) — the caller treats that as
        unverifiable and refuses the write (fail-closed).
        """
        if not booking_id:
            return None
        from runtime.tools.registry import registry

        entry = registry.get("get_booking")
        if entry is None or entry.tool is None:
            return None
        try:
            result = entry.tool.invoke({"booking_id": booking_id})
        except Exception:  # noqa: BLE001 — unverifiable means refuse
            return None
        if not isinstance(result, dict) or result.get("error"):
            return None
        guest_id = result.get("guest_id")
        return guest_id if isinstance(guest_id, str) and guest_id else None

    @staticmethod
    def booking_exists_for_session(booking_id: str, session_id: str) -> str | None:
        """Refusal message when a booking is not this session's, else None."""
        from runtime.scope import SCOPE_UNKNOWN, get_bound_guest_id

        if not booking_id:
            return "missing booking_id"
        owner = HotelReceptionistAgent.booking_owner(booking_id)
        if owner is None:
            return f"booking {booking_id!r} does not exist or is not readable"
        bound = get_bound_guest_id(session_id)
        if bound is SCOPE_UNKNOWN:
            return "could not verify session guest scope"
        if isinstance(bound, str) and owner != bound:
            return f"booking {booking_id!r} belongs to another guest"
        return None

    def scope_tool(self, tool, session_id: str, mission_id: str = ""):
        """Wrap a guest/booking tool with the session's guest scope.

        First sight binds the session: the first guest_id a session
        touches (a find_guest hit, a new registration, a booking read
        or a booking create) becomes the session's guest, and later
        calls naming another guest are refused (writes) or have that
        guest's data withheld (reads). The binding lives in
        ``session_guest_scopes`` (see ``runtime/scope.py``) so it
        survives across the session's executions. The refusal is a
        structured error dict, so the agent sees it as a tool result,
        not a crash. Scope helpers are resolved per call (not per
        wrap) so tests can stub the binding store without rebuilding
        the tools. Tools outside ``GUEST_SCOPED_TOOLS`` pass through
        unchanged.
        """
        if tool.name not in self.GUEST_SCOPED_TOOLS:
            return tool
        name = tool.name

        def _refusal(message: str, **extra):
            return {"error": message, "stub": False, **extra}

        def _guarded(**kwargs):
            from runtime.scope import SCOPE_UNKNOWN, bind_guest, get_bound_guest_id

            bound = get_bound_guest_id(session_id)
            # A store failure is SCOPE_UNKNOWN, not "unbound": the
            # session may already be bound to a guest we cannot see.
            # Writes are refused until the scope is verifiable again;
            # reads keep the unbound behaviour (first-sight bind on
            # success).
            scope_unknown = bound is SCOPE_UNKNOWN
            if scope_unknown:
                bound = None

            if scope_unknown:
                from runtime.tools.registry import registry

                entry = registry.get(name)
                read_ok = name in self.guest_read_tool_names() or (
                    entry is not None and entry.read_only
                )
                if not read_ok:
                    return _refusal(
                        "could not verify this session's guest scope; "
                        f"{name} refused"
                    )
            if name == "find_guest":
                result = tool.invoke(kwargs)
                if not isinstance(result, dict) or result.get("error"):
                    return result
                guests = result.get("guests") or []
                if bound:
                    kept = [
                        g
                        for g in guests
                        if isinstance(g, dict) and g.get("guest_id") == bound
                    ]
                    return {"guests": kept, "count": len(kept)}
                if len(guests) == 1 and isinstance(guests[0], dict):
                    guest_id = guests[0].get("guest_id")
                    if isinstance(guest_id, str) and guest_id:
                        bind_guest(session_id, guest_id, guests[0].get("phone"), mission_id)
                return result
            if name == "register_guest":
                if scope_unknown:
                    return _refusal(
                        "could not verify this session's guest scope; "
                        "register_guest refused"
                    )
                if bound:
                    return _refusal(
                        f"session is already bound to guest '{bound}'; "
                        "register_guest refused"
                    )
                result = tool.invoke(kwargs)
                if isinstance(result, dict) and not result.get("error"):
                    guest_id = result.get("guest_id")
                    if isinstance(guest_id, str) and guest_id:
                        bind_guest(session_id, guest_id, result.get("phone"), mission_id)
                return result
            if name == "get_booking":
                result = tool.invoke(kwargs)
                if not isinstance(result, dict) or result.get("error"):
                    return result
                guest_id = result.get("guest_id")
                if not isinstance(guest_id, str) or not guest_id:
                    return result
                if bound and guest_id != bound:
                    return _refusal(
                        f"booking {kwargs.get('booking_id')!r} belongs to "
                        "another guest; refused",
                        booking_id=kwargs.get("booking_id"),
                    )
                if not bound:
                    bind_guest(session_id, guest_id, None, mission_id)
                return result
            if name == "create_booking":
                if scope_unknown:
                    return _refusal(
                        "could not verify this session's guest scope; "
                        "create_booking refused",
                        guest_id=kwargs.get("guest_id"),
                    )
                guest_id = kwargs.get("guest_id")
                if bound and isinstance(guest_id, str) and guest_id and guest_id != bound:
                    return _refusal(
                        f"guest_id '{guest_id}' is outside this session's "
                        f"scope (guest '{bound}'); refused",
                        guest_id=guest_id,
                    )
                result = tool.invoke(kwargs)
                if (
                    isinstance(result, dict)
                    and not result.get("error")
                    and isinstance(guest_id, str)
                    and guest_id
                ):
                    bind_guest(session_id, guest_id, None, mission_id)
                return result
            if name in ("cancel_booking", "change_booking_dates"):
                if scope_unknown:
                    return _refusal(
                        "could not verify this session's guest scope; refused",
                        booking_id=kwargs.get("booking_id"),
                    )
                booking_id = kwargs.get("booking_id")
                owner = self.booking_owner(booking_id)
                if owner is None:
                    return _refusal(
                        f"cannot verify owner of booking {booking_id!r}; refused",
                        booking_id=booking_id,
                    )
                if bound and owner != bound:
                    return _refusal(
                        f"booking {booking_id} belongs to another guest; refused",
                        booking_id=booking_id,
                    )
                result = tool.invoke(kwargs)
                if isinstance(result, dict) and not result.get("error") and not bound:
                    bind_guest(session_id, owner, None, mission_id)
                return result
            return tool.invoke(kwargs)

        return tool.model_copy(update={"func": _guarded})

    # -- Write pipeline (Phase 3) ----------------------------------------
    #
    # The engine stays generic in ``runtime/write_pipeline_runner.py``;
    # this agent only supplies what is hotel about the booking_create
    # pipeline: the room-type row is picked by the room the guest
    # asked for (not always the first rate row), and the guest-facing
    # answer is a booking confirmation. Everything else (pending
    # state, idempotency, preconditions) is the engine's.

    @staticmethod
    def _requested_room_type(tools: dict) -> str | None:
        for data in (tools or {}).values():
            if not isinstance(data, dict):
                continue
            for key in ("room_type", "requested_room_type"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            guests = data.get("guests")
            if isinstance(guests, dict):
                rt = guests.get("room_type")
                if isinstance(rt, str) and rt.strip():
                    return rt.strip()
        return None

    @staticmethod
    def _pick_rate_row(rates: list, room_type: str | None) -> dict | None:
        if not isinstance(rates, list) or not rates:
            return None
        if room_type:
            for row in rates:
                if isinstance(row, dict) and row.get("room_type") == room_type:
                    return row
        first = rates[0]
        return first if isinstance(first, dict) else None

    def _resolve_room_type(self, tools: dict) -> str | None:
        """The quoted room type: the row matching the guest's request.

        Falls back to the first rate row when nothing was requested
        (the generic ``arg_paths`` would already resolve that row, so
        this only matters when the guest asked for a specific type).
        """
        rates_data = tools.get("get_room_rates") if isinstance(tools, dict) else None
        if not isinstance(rates_data, dict):
            return None
        rates = rates_data.get("rates")
        rows = rates if isinstance(rates, list) else []
        row = self._pick_rate_row(rows, self._requested_room_type(tools))
        if row and row.get("room_type"):
            return row["room_type"]
        return None

    def write_pipeline(
        self,
        pipeline,
        *,
        session_id: str,
        execution_id: str,
        tools_by_name: dict,
        session_turns: int,
    ):
        """Run the booking pipeline's automated write, hotel-flavoured."""
        from runtime.write_pipeline_runner import try_automated_write

        return try_automated_write(
            pipeline,
            session_id=session_id,
            execution_id=execution_id,
            tools_by_name=tools_by_name,
            session_turns=session_turns,
            agent=self,
            field_resolvers={"room_type": self._resolve_room_type},
            confirm_tool_name=self.confirm_tool_name(pipeline.id),
            success_template="Your booking has been created{ref}.",
        )


register_agent_class("hotel_receptionist", HotelReceptionistAgent)
