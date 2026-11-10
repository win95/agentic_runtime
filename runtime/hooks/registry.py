"""Hook registry — register handlers per event, dispatch them in order.

Matcher semantics mirror Hooks v2: ``None``/``""``/``"*"`` matches
everything; an exact tool name matches that tool; anything else is an
unanchored regex over the tool name (``"Write|Edit"`` matches either).
Only ``PreToolUse``/``PostToolUse``/``PostToolUseFailure`` use the
matcher (they are the tool events); other events ignore it.

Dispatch never raises: a handler that raises is logged and skipped, so
a broken hook cannot fail a run. Dispatch returns every decision the
matching handlers produced, in registration order — the caller applies
them per event (deny wins on PreToolUse, all feedback is concatenated
on PostToolUse, first ``continue`` wins on Stop).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from runtime.hooks.events import HookContext, HookDecision, HookEvent

logger = logging.getLogger(__name__)

# Handler: (context) -> decision or None ("no opinion").
HookHandler = Callable[[HookContext], HookDecision | None]


def _matcher_matches(matcher: str | None, tool_name: str) -> bool:
    """True when a Hooks v2 style matcher selects this tool call:
    empty/``*`` matches everything, an exact name matches that tool,
    anything else is an unanchored regex. An invalid regex never
    matches — a broken matcher must fail closed, not open."""
    if not matcher or matcher == "*":
        return True
    if matcher == tool_name:
        return True
    try:
        return re.search(matcher, tool_name) is not None
    except re.error:
        return False


class HookRegistry:
    """Per-run registry: handlers are registered once, dispatched many
    times. One instance is built per execution (see ``default_registry``)
    so nothing leaks across concurrent runs."""

    def __init__(self) -> None:
        self._handlers: dict[HookEvent, list[tuple[str | None, HookHandler, str]]] = {}

    def register(
        self,
        event: HookEvent,
        handler: HookHandler,
        matcher: str | None = None,
        name: str = "",
    ) -> None:
        self._handlers.setdefault(event, []).append((matcher, handler, name))

    def handler_names(self) -> list[str]:
        """Names of all registered handlers (registration order)."""
        names: list[str] = []
        for entries in self._handlers.values():
            for _, _, name in entries:
                if name and name not in names:
                    names.append(name)
        return names

    def with_only(self, names: list[str]) -> HookRegistry:
        """A new registry keeping only handlers whose name is listed."""
        wanted = set(names)
        narrowed = HookRegistry()
        for event, entries in self._handlers.items():
            for matcher, handler, name in entries:
                if name in wanted:
                    narrowed.register(event, handler, matcher, name)
        return narrowed

    def dispatch(self, event: HookEvent, context: HookContext) -> list[HookDecision]:
        """Run every matching handler for ``event``; never raises."""
        decisions: list[HookDecision] = []
        # Stamp the event on the context so handlers (e.g. the audit
        # hook, which serves five events) can tell which one fired.
        context.data["event_name"] = event.value
        for matcher, handler, name in self._handlers.get(event, []):
            if event in (
                HookEvent.PRE_TOOL_USE,
                HookEvent.POST_TOOL_USE,
                HookEvent.POST_TOOL_USE_FAILURE,
            ) and not _matcher_matches(matcher, context.tool_name):
                continue
            try:
                decision = handler(context)
            except Exception as exc:  # noqa: BLE001 — a hook never fails a run
                logger.warning("Hook %r for %s raised, skipping: %s", name, event, exc)
                continue
            if decision is not None:
                decisions.append(decision)
        return decisions


def default_registry() -> HookRegistry:
    """Registry with all built-in handlers (see ``runtime/hooks/builtin.py``)."""
    from runtime.hooks import builtin

    registry = HookRegistry()
    for name, events in builtin.BUILTIN_HOOKS.items():
        for event, handler in events:
            registry.register(event, handler, name=name)
    return registry


def select_hooks(mission: dict) -> list[str]:
    """Hook names active for one mission, deterministically.

    Explicit declaration wins: ``mission["hooks"]`` or
    ``mission["execution"]["hooks"]`` lists exactly the built-in hooks
    to run (unknown names are skipped with a warning). Otherwise every
    built-in hook runs — the safe default, since the dangerous-call
    blocker and the input validator are safety nets, not opt-ins.
    """
    from runtime.hooks import builtin

    declared = mission.get("hooks") or (mission.get("execution") or {}).get("hooks") or []
    if declared:
        known = set(builtin.BUILTIN_HOOKS)
        selected: list[str] = []
        for name in declared:
            if str(name) not in known:
                logger.warning(
                    "Mission %s declares unknown hook %r; skipping",
                    mission.get("mission_id"),
                    name,
                )
                continue
            selected.append(str(name))
        logger.debug(
            "Hooks for mission %s (declared): %s",
            mission.get("mission_id"),
            selected,
        )
        return selected
    selected = list(builtin.BUILTIN_HOOKS)
    logger.debug(
        "Hooks for mission %s (default, all built-ins): %s",
        mission.get("mission_id"),
        selected,
    )
    return selected
