"""Projection handlers — state reconciliation (design §5.4)."""


class UnrecoverableDriftError(RuntimeError):
    """L1 says an object is active but it is gone from Neo4j.

    L1 holds no object data, so the drain cannot recreate it. The
    drain marks the row dead at once (no retry — retrying cannot
    heal this) and logs a warning: an operator must restore the
    object from the Neo4j backup.
    """
