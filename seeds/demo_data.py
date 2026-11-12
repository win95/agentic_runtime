"""Demo seed data — real rows in real stores, not hardcoded stub returns.

Assets: O21, O22, O23 and O24, October 2026. O21 is the original
passenger elevator in building B1 (8 fault_history rows, 6
maintenance_records rows, 62 sensor readings, 6 technical documents,
5 relations) — it is UNTOUCHED: the classic eval cases and several
tests depend on its exact rows. O22 is a second elevator in the new
building B2 (2 faults, 1 maintenance row, 6 sensor readings, one
temperature spike coherent with its F-206 fault). O23 is a healthy
elevator in B2 (no faults, no maintenance — it exercises the
skill-map "no faults → maintenance → skip sensors" branch). O24 is
a maintenance-only elevator in B2 (no faults, 2 maintenance rows).
O20 is a retired (archived) elevator in B1, present only in the
ontology seed — it has no faults, maintenance or sensor rows.

- PostgreSQL: 10 fault_history rows + 9 maintenance_records rows +
  76 sensor readings in total.
- Qdrant: 7 technical documents, embedded with the configured model.
- Ontology (PG + Neo4j): 15 asset relations via canonical write +
  projection, plus Document nodes linked by DOCUMENTED_BY.

All seed functions are idempotent: re-running them changes nothing
(deterministic ids / MERGE / point-id skip).
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

ASSET_ID = "O21"

FAULTS = [
    {
        "id": "fh-o21-001",
        "asset_id": ASSET_ID,
        "code": "F-101",
        "title": "Door sensor intermittent fault",
        "description": "Door reopen sensor triggers spuriously on floors 3 and 7.",
        "severity": "medium",
        "occurred_at": datetime(2026, 10, 3, 9, 14),
        "resolved_at": datetime(2026, 10, 3, 11, 40),
        "technician": "Nguyen Van A",
        "action": "Cleaned and realigned door sensor; verified 20 open/close cycles.",
    },
    {
        "id": "fh-o21-002",
        "asset_id": ASSET_ID,
        "code": "F-204",
        "title": "Traction motor overheating",
        "description": "Motor winding temperature reached 92C during peak hours.",
        "severity": "high",
        "occurred_at": datetime(2026, 10, 7, 14, 2),
        "resolved_at": datetime(2026, 10, 7, 18, 25),
        "technician": "Tran Thi B",
        "action": "Cleaned cooling fan and vents; replaced clogged air filter.",
    },
    {
        "id": "fh-o21-003",
        "asset_id": ASSET_ID,
        "code": "F-110",
        "title": "Door closing timeout",
        "description": "Doors failed to close within 6s on ground floor.",
        "severity": "medium",
        "occurred_at": datetime(2026, 10, 9, 8, 47),
        "resolved_at": datetime(2026, 10, 9, 10, 5),
        "technician": "Nguyen Van A",
        "action": "Adjusted door operator belt tension; lubricated guide rails.",
    },
    {
        "id": "fh-o21-004",
        "asset_id": ASSET_ID,
        "code": "F-301",
        "title": "Brake wear warning",
        "description": "Brake lining thickness below 4mm threshold.",
        "severity": "high",
        "occurred_at": datetime(2026, 10, 12, 16, 30),
        "resolved_at": datetime(2026, 10, 13, 12, 0),
        "technician": "Le Van C",
        "action": "Replaced brake pads; measured stopping distance within spec.",
    },
    {
        "id": "fh-o21-005",
        "asset_id": ASSET_ID,
        "code": "F-102",
        "title": "Leveling sensor drift",
        "description": "Car stops 12mm above floor level at floor 5.",
        "severity": "low",
        "occurred_at": datetime(2026, 10, 16, 10, 21),
        "resolved_at": datetime(2026, 10, 16, 13, 15),
        "technician": "Tran Thi B",
        "action": "Recalibrated leveling sensor; re-tested all floors.",
    },
    {
        "id": "fh-o21-006",
        "asset_id": ASSET_ID,
        "code": "F-205",
        "title": "Motor vibration above limit",
        "description": "Vibration 6.8 mm/s (limit 4.5 mm/s) at full load.",
        "severity": "high",
        "occurred_at": datetime(2026, 10, 19, 11, 3),
        "resolved_at": datetime(2026, 10, 20, 15, 45),
        "technician": "Le Van C",
        "action": "Rebalanced traction sheave; checked motor mounts and coupling.",
    },
    {
        "id": "fh-o21-007",
        "asset_id": ASSET_ID,
        "code": "F-410",
        "title": "Controller communication error",
        "description": "Intermittent CAN bus errors between controller and drive.",
        "severity": "medium",
        "occurred_at": datetime(2026, 10, 23, 7, 58),
        "resolved_at": datetime(2026, 10, 23, 12, 20),
        "technician": "Nguyen Van A",
        "action": "Reseated CAN connectors; replaced one damaged cable segment.",
    },
    {
        "id": "fh-o21-008",
        "asset_id": ASSET_ID,
        "code": "F-103",
        "title": "Safety circuit trip",
        "description": "Safety circuit opened without a clear trigger; car stopped.",
        "severity": "high",
        "occurred_at": datetime(2026, 10, 27, 18, 44),
        "resolved_at": datetime(2026, 10, 28, 9, 30),
        "technician": "Tran Thi B",
        "action": "Inspected all safety switches; replaced faulty pit switch.",
    },
    # --- Asset O22 (passenger elevator, building B2) ----------------------
    {
        "id": "fh-o22-001",
        "asset_id": "O22",
        "code": "F-206",
        "title": "Traction motor overheating",
        "description": "Motor winding temperature reached 88.5C during peak hours.",
        "severity": "high",
        "occurred_at": datetime(2026, 10, 11, 15, 20),
        "resolved_at": datetime(2026, 10, 11, 19, 5),
        "technician": "Le Van C",
        "action": "Cleaned cooling fan; replaced air filter.",
    },
    {
        "id": "fh-o22-002",
        "asset_id": "O22",
        "code": "F-207",
        "title": "Controller communication error",
        "description": "Intermittent CAN bus errors between controller and drive.",
        "severity": "medium",
        "occurred_at": datetime(2026, 10, 18, 9, 40),
        "resolved_at": datetime(2026, 10, 18, 12, 10),
        "technician": "Nguyen Van A",
        "action": "Reseated CAN connectors; verified bus load.",
    },
    # O23 (healthy) and O24 (maintenance-only) have NO fault rows — that
    # absence is the point: it exercises the skill-map branches.
]

MAINTENANCE = [
    {
        "id": "mr-o21-001",
        "asset_id": ASSET_ID,
        "date": datetime(2026, 10, 2, 9, 0),
        "action": "Monthly routine inspection: doors, ropes, brakes, safety gear.",
        "technician": "Nguyen Van A",
        "notes": "All items within spec; door rollers show early wear.",
    },
    {
        "id": "mr-o21-002",
        "asset_id": ASSET_ID,
        "date": datetime(2026, 10, 5, 10, 30),
        "action": "Door sensor cleaning and alignment (follow-up to F-101).",
        "technician": "Nguyen Van A",
        "notes": "Sensor gap reset to 3mm.",
    },
    {
        "id": "mr-o21-003",
        "asset_id": ASSET_ID,
        "date": datetime(2026, 10, 8, 14, 0),
        "action": "Motor cooling system service: fan, vents, air filter.",
        "technician": "Tran Thi B",
        "notes": "Follow-up to overheating F-204.",
    },
    {
        "id": "mr-o21-004",
        "asset_id": ASSET_ID,
        "date": datetime(2026, 10, 14, 9, 0),
        "action": "Brake pads replaced; stopping distance test passed.",
        "technician": "Le Van C",
        "notes": "Follow-up to brake warning F-301.",
    },
    {
        "id": "mr-o21-005",
        "asset_id": ASSET_ID,
        "date": datetime(2026, 10, 21, 11, 0),
        "action": "Traction sheave rebalancing and coupling inspection.",
        "technician": "Le Van C",
        "notes": "Follow-up to vibration fault F-205.",
    },
    {
        "id": "mr-o21-006",
        "asset_id": ASSET_ID,
        "date": datetime(2026, 10, 29, 9, 0),
        "action": "Safety circuit full test; pit switch replaced.",
        "technician": "Tran Thi B",
        "notes": "Follow-up to safety trip F-103.",
    },
    # --- Asset O22 (building B2) ------------------------------------------
    {
        "id": "mr-o22-001",
        "asset_id": "O22",
        "date": datetime(2026, 10, 12, 10, 0),
        "action": "Motor cooling system service: fan, vents, air filter.",
        "technician": "Le Van C",
        "notes": "Follow-up to overheating F-206.",
    },
    # --- Asset O24 (maintenance-only, building B2) ------------------------
    {
        "id": "mr-o24-001",
        "asset_id": "O24",
        "date": datetime(2026, 10, 6, 9, 0),
        "action": "Monthly routine inspection: doors, ropes, brakes, safety gear.",
        "technician": "Nguyen Van A",
        "notes": "All items within spec.",
    },
    {
        "id": "mr-o24-002",
        "asset_id": "O24",
        "date": datetime(2026, 10, 20, 14, 0),
        "action": "Door operator check and lubrication.",
        "technician": "Tran Thi B",
        "notes": "Door closing time within spec.",
    },
    # O23 (healthy) has NO maintenance rows.
]

# Technical documents for the Qdrant collection. `id` is stable so the
# seed can skip documents that are already present.
DOCUMENTS = [
    {
        "id": "doc-door-sensor",
        "title": "Door sensor troubleshooting — passenger elevators",
        "content": (
            "Intermittent door sensor faults (code F-101) are usually caused by "
            "dust on the photoelectric sensor, misalignment after door "
            "operator wear, or a loose connector. Clean the sensor lens, "
            "check the 3mm gap, and run 20 open/close cycles to verify. "
            "Replace the sensor if faults recur within one week."
        ),
        "category": "door_system",
    },
    {
        "id": "doc-motor-overheat",
        "title": "Traction motor overheating — causes and remedies",
        "content": (
            "Traction motor overheating (code F-204) is commonly caused by a "
            "clogged cooling fan, blocked vents, or a dirty air filter. "
            "Winding temperature above 90C requires immediate load reduction. "
            "Clean the fan and vents, replace the filter, and log the "
            "temperature trend for one week after service."
        ),
        "category": "traction",
    },
    {
        "id": "doc-brake-inspection",
        "title": "Brake system inspection — lining wear limits",
        "content": (
            "Brake lining below 4mm (code F-301) must be replaced before the "
            "car returns to service. After replacement, measure the stopping "
            "distance at rated load and compare with the manufacturer table. "
            "Vibration above 4.5 mm/s often points to sheave imbalance or "
            "worn motor mounts rather than the brake itself."
        ),
        "category": "brake_system",
    },
    {
        "id": "doc-controller-errors",
        "title": "Controller error codes — CAN bus communication",
        "content": (
            "Controller communication errors (code F-410) appear as "
            "intermittent CAN bus faults between controller and drive. Check "
            "connector seating first, then inspect cable segments for damage. "
            "A safety circuit trip (F-103) with no clear trigger usually "
            "means a failing safety switch — test each switch individually."
        ),
        "category": "controller",
    },
    {
        "id": "doc-leveling",
        "title": "Leveling accuracy — sensor calibration",
        "content": (
            "Leveling drift (code F-102) shows as the car stopping more than "
            "10mm above or below floor level. Recalibrate the leveling sensor "
            "and re-test every floor. Persistent drift after calibration "
            "indicates a worn encoder or a loose sensor bracket."
        ),
        "category": "leveling",
    },
    {
        "id": "doc-rope-inspection",
        "title": "Rope inspection and lubrication guide",
        "content": (
            "Inspect traction ropes monthly for broken wires, diameter loss "
            "and uneven tension. Lubricate with the manufacturer-approved "
            "rope dressing only. Rope-related vibration is usually accompanied "
            "by uneven wear on the sheave grooves."
        ),
        "category": "ropes",
    },
    {
        "id": "doc-o23-overview",
        "title": "Elevator O23 — operation and routine checks",
        "content": (
            "Elevator O23 is a passenger elevator in building B2. Monthly "
            "routine checks cover doors, ropes, brakes and safety gear. "
            "Normal motor temperature is 68–73C and normal vibration is "
            "2.5–3.5 mm/s at full load. No fault history is recorded for "
            "O23 in October 2026."
        ),
        "category": "operation",
    },
]

# (from_id, from_label, relation, to_id, to_label)
RELATIONS = [
    ("O21", "Asset", "PART_OF", "building-B1", "Building"),
    ("O21", "Asset", "HAS_COMPONENT", "motor-M2", "Component"),
    ("O21", "Asset", "HAS_COMPONENT", "door-sensor-DS1", "Component"),
    ("O21", "Asset", "HAS_COMPONENT", "controller-C1", "Component"),
    ("O21", "Asset", "HAS_COMPONENT", "brake-BK1", "Component"),
    # O22 (building B2) + its components.
    ("O22", "Asset", "PART_OF", "building-B2", "Building"),
    ("O22", "Asset", "HAS_COMPONENT", "motor-M3", "Component"),
    ("O22", "Asset", "HAS_COMPONENT", "controller-C2", "Component"),
    # O20 (archived, building B1) + its one component.
    ("O20", "Asset", "PART_OF", "building-B1", "Building"),
    ("O20", "Asset", "HAS_COMPONENT", "motor-M0", "Component"),
    # O23 / O24 (building B2).
    ("O23", "Asset", "PART_OF", "building-B2", "Building"),
    ("O24", "Asset", "PART_OF", "building-B2", "Building"),
    # Document links — the Document key is the Qdrant doc id, which is
    # what ontology/knowledge/graph_query.py `linked_documents` matches.
    ("O21", "Asset", "DOCUMENTED_BY", "doc-door-sensor", "Document"),
    ("O21", "Asset", "DOCUMENTED_BY", "doc-motor-overheat", "Document"),
    ("O21", "Asset", "DOCUMENTED_BY", "doc-brake-inspection", "Document"),
]


def _sensor_readings() -> list[dict]:
    """Daily O21 readings for October 2026, deterministic (no randomness).

    Two metrics, one reading each at noon: motor temperature (C) and
    vibration (mm/s). The spikes are coherent with the fault rows: the
    motor runs hot 5–8 Oct (92C on 7 Oct = F-204) and cools after the
    8 Oct service; vibration exceeds the 4.5 mm/s limit 18–20 Oct (6.8
    on 19 Oct = F-205) and settles after the 21 Oct rebalancing.
    """
    temp_spike = {5: 78.5, 6: 85.0, 7: 92.0, 8: 80.0, 9: 74.0}
    vib_spike = {18: 5.4, 19: 6.8, 20: 5.0, 21: 3.6}
    rows = []
    for day in range(1, 32):
        at = datetime(2026, 10, day, 12, 0)
        temp = temp_spike.get(day, 69.0 + (day % 5) * 0.8)
        vib = vib_spike.get(day, 2.8 + (day % 4) * 0.4)
        rows.append({
            "id": f"sr-o21-temp-{day:02d}", "asset_id": ASSET_ID,
            "metric": "motor_temperature_c", "value": round(temp, 1),
            "unit": "C", "at": at,
        })
        rows.append({
            "id": f"sr-o21-vib-{day:02d}", "asset_id": ASSET_ID,
            "metric": "vibration_mm_s", "value": round(vib, 1),
            "unit": "mm/s", "at": at,
        })
    # A handful of readings for the newer assets — same noon cadence, no
    # full month. O22's temperature spike (88.5C on 11 Oct) is coherent
    # with its F-206 fault; O23/O24 stay inside normal ranges.
    for day, temp, vib in (
        (10, 74.0, 3.0),
        (11, 88.5, 3.2),  # F-206 day
        (12, 71.5, 2.9),
    ):
        rows.append({
            "id": f"sr-o22-temp-{day:02d}", "asset_id": "O22",
            "metric": "motor_temperature_c", "value": temp,
            "unit": "C", "at": datetime(2026, 10, day, 12, 0),
        })
        rows.append({
            "id": f"sr-o22-vib-{day:02d}", "asset_id": "O22",
            "metric": "vibration_mm_s", "value": vib,
            "unit": "mm/s", "at": datetime(2026, 10, day, 12, 0),
        })
    for day, temp, vib in ((15, 70.5, 2.9), (16, 71.0, 3.1)):
        rows.append({
            "id": f"sr-o23-temp-{day:02d}", "asset_id": "O23",
            "metric": "motor_temperature_c", "value": temp,
            "unit": "C", "at": datetime(2026, 10, day, 12, 0),
        })
        rows.append({
            "id": f"sr-o23-vib-{day:02d}", "asset_id": "O23",
            "metric": "vibration_mm_s", "value": vib,
            "unit": "mm/s", "at": datetime(2026, 10, day, 12, 0),
        })
    for day, temp, vib in ((6, 70.2, 3.0), (20, 70.8, 2.8)):
        rows.append({
            "id": f"sr-o24-temp-{day:02d}", "asset_id": "O24",
            "metric": "motor_temperature_c", "value": temp,
            "unit": "C", "at": datetime(2026, 10, day, 12, 0),
        })
        rows.append({
            "id": f"sr-o24-vib-{day:02d}", "asset_id": "O24",
            "metric": "vibration_mm_s", "value": vib,
            "unit": "mm/s", "at": datetime(2026, 10, day, 12, 0),
        })
    return rows


SENSOR_READINGS = _sensor_readings()


def seed_postgres_demo(db) -> dict:
    """Upsert demo fault + maintenance + sensor rows (idempotent by fixed ids)."""
    from app.models import FaultHistory, MaintenanceRecord, SensorReading

    for row in FAULTS:
        db.merge(FaultHistory(**row))
    for row in MAINTENANCE:
        db.merge(MaintenanceRecord(**row))
    for row in SENSOR_READINGS:
        db.merge(SensorReading(**row))
    db.commit()
    logger.info(
        "Seeded %d faults + %d maintenance rows + %d sensor readings",
        len(FAULTS), len(MAINTENANCE), len(SENSOR_READINGS),
    )
    return {
        "faults": len(FAULTS),
        "maintenance": len(MAINTENANCE),
        "sensor_readings": len(SENSOR_READINGS),
    }


def seed_qdrant_demo(embeddings=None) -> dict:
    """Embed + upsert the 6 technical documents.

    The collection is created only if missing, with the dimension measured
    from a probe embedding of the configured model. Documents already
    present are skipped.
    """
    from infra.qdrant_client import ensure_collection, probe_dimension, upsert_documents

    dim = probe_dimension(embeddings=embeddings)
    created = ensure_collection(dim)
    result = upsert_documents(DOCUMENTS, embeddings=embeddings)
    logger.info(
        "Qdrant seed: collection %s, dim=%d, upserted=%d, skipped=%d",
        created,
        dim,
        result["upserted"],
        result["skipped"],
    )
    return {"collection": created, "dim": dim, **result}


def seed_falkor_demo() -> dict:
    """Seed ontology relations (legacy entrypoint name)."""
    from seeds.seed_ontology_demo import seed_ontology_demo

    return seed_ontology_demo()


def seed_tools_demo(db) -> dict:
    """Mirror the code registry into the `tools` table (idempotent).

    The registry stays the permission source of truth — the DB rows only
    make GET /tools/db return the same 17 tools the runtime sees. The
    input schema is the langchain args JSON schema when obtainable,
    otherwise {} (the registry's ``args`` dict holds pydantic field
    objects, which are not JSON-serializable).
    """
    # Lazy import: importing the tool modules registers them.
    import runtime.tools.admin_tools  # noqa: F401
    import runtime.tools.fault_tools  # noqa: F401
    from app.models import Tool
    from runtime.tools.registry import registry

    count = 0
    for entry in registry.list():
        schema: dict = {}
        try:
            args_schema = getattr(entry.tool, "args_schema", None)
            if args_schema is not None:
                schema = args_schema.model_json_schema()
        except Exception:  # noqa: BLE001 — schema is best-effort metadata
            schema = {}
        db.merge(Tool(
            id=f"tool_{entry.name}",
            name=entry.name,
            description=entry.description,
            capability=entry.capability,
            input_schema=schema,
            output_schema={},
            permission=entry.permission,
            implementation=entry.implementation,
            timeout_seconds=entry.timeout_seconds,
            retry_count=entry.retry_count,
            # The registry is the source of truth for read_only; the DB
            # row must mirror it, otherwise GET /tools/db reports every
            # tool (including the admin write tools) as read-only.
            read_only=entry.read_only,
            is_active=True,
        ))
        count += 1
    db.commit()
    logger.info("Seeded %d tool rows from the registry", count)
    return {"tools": count}


def seed_policies_demo(db) -> dict:
    """Seed one default tenant policy (idempotent).

    Tenant "default" may use every registered capability and tool, up to
    100 executions/day. Real per-tenant policies replace this row; the
    policy table is metadata until the auth/tenant work (roadmap Step 2)
    enforces it.
    """
    import runtime.tools.admin_tools  # noqa: F401
    import runtime.tools.fault_tools  # noqa: F401
    from app.models import Policy
    from runtime.tools.registry import registry

    capabilities = sorted({e.capability for e in registry.list()})
    tool_names = sorted(e.name for e in registry.list())
    db.merge(Policy(
        id="policy_default",
        name="Default policy",
        tenant="default",
        allowed_capabilities=capabilities,
        allowed_tools=tool_names,
        max_executions_per_day=100,
        is_active=True,
    ))
    db.commit()
    logger.info("Seeded default policy (%d capabilities, %d tools)",
                len(capabilities), len(tool_names))
    return {"policies": 1}


def validate() -> dict:
    """Check the seed data's own invariants (no store access).

    Used by `seeds/seed_demo_data.py --dry-run` and the seed tests:
    ids unique per list, per-asset row counts, all fault/maintenance
    dates inside October 2026, resolved_at after occurred_at, every
    document has a non-empty title + content, every relation starts at
    a known asset. Raises ValueError on the first violation (plain
    checks, never ``assert`` — validation must survive ``python -O``).
    """
    # Per-asset counts — the O21 October 2026 dataset is fixed (8/6/62);
    # O23 must stay empty of faults + maintenance and O24 empty of faults:
    # those absences are what exercise the skill-map branches.
    expected_per_asset = {
        "O21": (8, 6, 62),
        "O22": (2, 1, 6),
        "O23": (0, 0, 4),
        "O24": (0, 2, 4),
    }
    for name, rows, key in (
        ("faults", FAULTS, "id"),
        ("maintenance", MAINTENANCE, "id"),
        ("documents", DOCUMENTS, "id"),
        ("sensor_readings", SENSOR_READINGS, "id"),
    ):
        ids = [row[key] for row in rows]
        if not (len(ids) == len(set(ids))):
            raise ValueError(f"duplicate ids in {name}: {ids}")
    for asset, (faults, maintenance, sensors) in expected_per_asset.items():
        if not (sum(1 for r in FAULTS if r["asset_id"] == asset) == faults):
            raise ValueError(f"asset {asset} fault count changed")
        if not (sum(1 for r in MAINTENANCE if r["asset_id"] == asset) == maintenance):
            raise ValueError(f"asset {asset} maintenance count changed")
        if not (sum(1 for r in SENSOR_READINGS if r["asset_id"] == asset) == sensors):
            raise ValueError(f"asset {asset} sensor count changed")
    for row in FAULTS + MAINTENANCE:
        if row["asset_id"] not in expected_per_asset:
            raise ValueError(f"wrong asset in {row['id']}")
    for row in FAULTS:
        if not (row["occurred_at"].month == 10 and row["occurred_at"].year == 2026):
            raise ValueError(f"fault {row['id']} outside October 2026")
        if not (row["resolved_at"] is None or row["resolved_at"] >= row["occurred_at"]):
            raise ValueError(f"fault {row['id']} resolves before it occurs")
        if row["severity"] not in ("low", "medium", "high"):
            raise ValueError(f"fault {row['id']} has unknown severity {row['severity']!r}")
    for row in MAINTENANCE:
        if not (row["date"].month == 10 and row["date"].year == 2026):
            raise ValueError(f"maintenance {row['id']} outside October 2026")
    for doc in DOCUMENTS:
        if not (doc["title"].strip() and doc["content"].strip()):
            raise ValueError(f"document {doc['id']} has empty title or content")
    doc_ids = {doc["id"] for doc in DOCUMENTS}
    for rel in RELATIONS:
        if not (len(rel) == 5):
            raise ValueError(f"malformed relation: {rel}")
        if rel[0] not in {*expected_per_asset, "O20"}:
            raise ValueError(f"relation does not start at a known asset: {rel}")
        if rel[2] == "DOCUMENTED_BY" and (rel[4] != "Document" or rel[3] not in doc_ids):
            raise ValueError(f"DOCUMENTED_BY target is not a seeded document: {rel}")
    for row in SENSOR_READINGS:
        if row["asset_id"] not in expected_per_asset:
            raise ValueError(f"wrong asset in {row['id']}")
        if row["metric"] not in ("motor_temperature_c", "vibration_mm_s"):
            raise ValueError(f"unknown sensor metric in {row['id']}")
        if not (row["at"].month == 10 and row["at"].year == 2026):
            raise ValueError(f"sensor reading {row['id']} outside October 2026")
    # The spikes must match the fault rows they explain.
    by_id = {r["id"]: r for r in SENSOR_READINGS}
    if not (by_id["sr-o21-temp-07"]["value"] == 92.0):
        raise ValueError("F-204 temperature spike missing")
    if not (by_id["sr-o21-vib-19"]["value"] == 6.8):
        raise ValueError("F-205 vibration spike missing")
    if not (by_id["sr-o22-temp-11"]["value"] == 88.5):
        raise ValueError("F-206 temperature spike missing")
    return {
        "faults": len(FAULTS),
        "maintenance": len(MAINTENANCE),
        "documents": len(DOCUMENTS),
        "relations": len(RELATIONS),
        "sensor_readings": len(SENSOR_READINGS),
    }


def seed_executions_demo(db) -> dict:
    """Seed 3 finished executions (idempotent by fixed ids).

    One SUCCEEDED fault_analysis (O21), one FAILED fault_analysis
    (O22, store outage), one SUCCEEDED system_admin. Each has a
    matching ExecutionResult row — the Execution table has no result
    column, so the output/evidence live there, as in real runs.

    These rows do NOT trigger Slack notifications: the notifier only
    runs inside real ``run_execution`` (app/worker/tasks.py), never on
    seeded rows.
    """
    from app.models import Execution, ExecutionResult

    executions = [
        Execution(
            id="exec-seed-fa-001",
            mission_id="fault_analysis",
            mission_version=1,
            agent_id="fault_investigator",
            agent_version=None,
            input={
                "asset_id": "O21",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            },
            status="SUCCEEDED",
            created_at=datetime(2026, 10, 28, 9, 0),
            started_at=datetime(2026, 10, 28, 9, 0, 5),
            finished_at=datetime(2026, 10, 28, 9, 2, 40),
        ),
        Execution(
            id="exec-seed-fa-002",
            mission_id="fault_analysis",
            mission_version=1,
            agent_id="fault_investigator",
            agent_version=None,
            input={
                "asset_id": "O22",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
            },
            status="FAILED",
            error="Neo4j unavailable during projection drain; graph evidence missing.",
            created_at=datetime(2026, 10, 29, 10, 0),
            started_at=datetime(2026, 10, 29, 10, 0, 4),
            finished_at=datetime(2026, 10, 29, 10, 1, 12),
        ),
        Execution(
            id="exec-seed-sa-001",
            mission_id="system_admin",
            mission_version=1,
            agent_id="admin_bot",
            agent_version=None,
            input={"request": "Kiểm tra sức khỏe hệ thống và báo cáo các mission đang chạy."},
            status="SUCCEEDED",
            created_at=datetime(2026, 10, 28, 7, 0),
            started_at=datetime(2026, 10, 28, 7, 0, 2),
            finished_at=datetime(2026, 10, 28, 7, 0, 55),
        ),
    ]
    results = [
        ExecutionResult(
            id="res-seed-fa-001",
            execution_id="exec-seed-fa-001",
            output={
                "root_causes": [
                    "Traction motor overheating (F-204): clogged cooling fan, 92C on 2026-10-07.",
                    "Motor vibration (F-205): sheave imbalance, 6.8 mm/s on 2026-10-19.",
                ],
                "recommendations": [
                    "Clean cooling fan and vents monthly during peak season.",
                    "Rebalance traction sheave after any vibration above 4.5 mm/s.",
                ],
                "answer": "Root causes for O21 (October 2026): motor overheating and vibration.",
            },
            evidence=[
                {"kind": "fault", "asset_id": "O21", "code": "F-204"},
                {"kind": "fault", "asset_id": "O21", "code": "F-205"},
                {"kind": "sensor", "asset_id": "O21", "metric": "motor_temperature_c"},
            ],
        ),
        ExecutionResult(
            id="res-seed-fa-002",
            execution_id="exec-seed-fa-002",
            output={
                "errors": ["Neo4j unavailable during projection drain; graph evidence missing."],
            },
            evidence=[],
        ),
        ExecutionResult(
            id="res-seed-sa-001",
            execution_id="exec-seed-sa-001",
            output={
                "answer": "System healthy: 2 missions, 1 schedule, no failures in the last 24h.",
                "actions_taken": ["get_executions", "get_missions", "system_health"],
            },
            evidence=[
                {"kind": "system", "detail": "system_health"},
            ],
        ),
    ]
    for row in executions:
        db.merge(row)
    for row in results:
        db.merge(row)
    db.commit()
    logger.info("Seeded %d historical executions", len(executions))
    return {"executions": len(executions)}

