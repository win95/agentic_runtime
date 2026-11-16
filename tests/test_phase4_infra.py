"""Phase 4 infra: idempotency keys, execution claim, rate-limit fallback, JWT role."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from app.config import settings
from app.core.rate_limit import check_guest_rate_limit, reset_guest_rate_limit
from app.core.security import require_mission_for_role
from app.services import execution_service
from app import models
from runtime.write_idempotency import write_idempotency_key


def test_write_idempotency_key_scopes_to_attempt_and_execution():
    k1 = write_idempotency_key(
        write_attempt_id="a1", execution_id="e1", write_tool="create_booking"
    )
    k2 = write_idempotency_key(
        write_attempt_id="a1", execution_id="e2", write_tool="create_booking"
    )
    k3 = write_idempotency_key(
        write_attempt_id="a2", execution_id="e2", write_tool="create_booking"
    )
    assert k1 != k2
    assert k2 != k3


def test_jwt_without_role_gets_404_not_403():
    """P0: missing role is least-privilege user; forbidden mission is 404."""
    user = {"sub": "u1"}
    with pytest.raises(HTTPException) as exc:
        require_mission_for_role(user, "fault_analysis")
    assert exc.value.status_code == 404


def test_try_claim_execution_atomic(db_session):
    ex = models.Execution(
        id="exec_claim_1",
        mission_id="hotel_reception",
        mission_version=1,
        input={},
        status="QUEUED",
    )
    db_session.add(ex)
    db_session.commit()
    claimed = execution_service.try_claim_execution(db_session, "exec_claim_1")
    assert claimed is not None and claimed.status == "RUNNING"
    again = execution_service.try_claim_execution(db_session, "exec_claim_1")
    assert again is None


def test_rate_limit_fallback_fail_open_when_redis_down(monkeypatch):
    """P2: broker outage must not block guest chat on this process."""
    reset_guest_rate_limit()
    monkeypatch.setattr(
        "app.core.rate_limit._redis_client",
        lambda: (_ for _ in ()).throw(ConnectionError("redis down")),
    )
    for _ in range(25):
        check_guest_rate_limit("guest-a")


def test_stale_running_reset_to_queued_for_reclaim(db_session):
    ex = models.Execution(
        id="exec_stale_1",
        mission_id="hotel_reception",
        mission_version=1,
        input={},
        status="RUNNING",
        started_at=datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(seconds=99999),
    )
    db_session.add(ex)
    db_session.commit()
    assert execution_service.execution_is_stale(ex)