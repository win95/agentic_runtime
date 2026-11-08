"""Demo seed data — real rows in real stores, not hardcoded stub returns.

Asset: O21, a passenger elevator in building B1, October 2026.
- PostgreSQL: 8 fault_history rows + 6 maintenance_records rows.
- Qdrant: 6 technical documents, embedded with the configured model.
- FalkorDB: asset relations, seeded with MERGE (idempotent).

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
]

# (from_id, from_label, relation, to_id, to_label)
RELATIONS = [
    ("O21", "Asset", "PART_OF", "building-B1", "Building"),
    ("O21", "Asset", "HAS_COMPONENT", "motor-M2", "Component"),
    ("O21", "Asset", "HAS_COMPONENT", "door-sensor-DS1", "Component"),
    ("O21", "Asset", "HAS_COMPONENT", "controller-C1", "Component"),
    ("O21", "Asset", "HAS_COMPONENT", "brake-BK1", "Component"),
]


def seed_postgres_demo(db) -> dict:
    """Upsert demo fault + maintenance rows (idempotent by fixed ids)."""
    from app.models import FaultHistory, MaintenanceRecord

    for row in FAULTS:
        db.merge(FaultHistory(**row))
    for row in MAINTENANCE:
        db.merge(MaintenanceRecord(**row))
    db.commit()
    logger.info("Seeded %d faults + %d maintenance rows", len(FAULTS), len(MAINTENANCE))
    return {"faults": len(FAULTS), "maintenance": len(MAINTENANCE)}


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
    """Seed asset relations with MERGE (idempotent)."""
    from infra.falkor_client import merge_asset_relations

    merge_asset_relations(RELATIONS)
    logger.info("Seeded %d FalkorDB relations", len(RELATIONS))
    return {"relations": len(RELATIONS)}
