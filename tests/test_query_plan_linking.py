"""Phase 4 QueryPlan linking tests — dynamic gazetteer normalisation.

Covers ``runtime/retrieval_graph.py::gazetteer_seeds`` against the
dynamic cache (the ``demo_gazetteer`` fixture in ``tests/conftest.py`` loads
the demo ontology's keys): exact hits unchanged, separator
variants (hyphen/space/case/diacritics), 0.8 confidence for
normalised hits, no false positive on embedded codes, empty
input. There is no Fault entity type and no bare-name alias table
any more, so fault codes and bare names ("door sensor", "motor")
link NOTHING — those cases are pinned as empty, not as links.
No LLM, no Qdrant/Neo4j. Main features only.
"""

from __future__ import annotations

from runtime.retrieval_graph import gazetteer_seeds


def _seed_map(question: str) -> dict[str, float]:
    return {s["key"]: s["confidence"] for s in gazetteer_seeds(question)}


def _seed_keys(question: str) -> list[str]:
    return [s["key"] for s in gazetteer_seeds(question)]


# --- Exact hits ------------------------------------------------------------

def test_exact_hits_with_confidence_1(demo_gazetteer):
    assert [(s["key"], s["confidence"]) for s in
            gazetteer_seeds("Elevator O22's motor-M3 overheated.")] == [
        ("O22", 1.0),
        ("motor-M3", 1.0),
    ]
    assert _seed_map("See doc-motor-overheat for O21.") == {
        "doc-motor-overheat": 1.0,
        "O21": 1.0,
    }


def test_bare_names_link_nothing(demo_gazetteer):
    # No name-alias table any more: only cached keys link.
    assert _seed_map("The door sensor triggers spuriously.") == {}
    assert _seed_map("The motor overheated.") == {}


def test_exact_and_normalised_twin_dedup_keeps_1(demo_gazetteer):
    assert _seed_map("O21 and O-21") == {"O21": 1.0}
    assert _seed_map("O21 and O 21") == {"O21": 1.0}


# --- Separator / case variants ------------------------------------------

def test_asset_hyphen_space_and_case_variants(demo_gazetteer):
    assert _seed_map("Elevator O-21 is stuck.") == {"O21": 0.8}
    assert _seed_map("Elevator O 21 is stuck.") == {"O21": 0.8}
    assert _seed_map("Elevator o-21 is stuck.") == {"O21": 0.8}
    # Bare "o21" is already an exact hit, so it stays 1.0.
    assert _seed_map("Elevator o21 is stuck.") == {"O21": 1.0}


def test_component_separator_variants(demo_gazetteer):
    assert _seed_map("The motor m2 overheated.") == {"motor-M2": 0.8}
    assert _seed_map("The motorM2 overheated.") == {"motor-M2": 0.8}
    # Hyphenated form is already exact.
    assert _seed_map("The motor-m2 overheated.") == {"motor-M2": 1.0}
    assert _seed_map("Check controller C1.") == {"controller-C1": 0.8}
    assert _seed_map("Check controllerC1.") == {"controller-C1": 0.8}
    assert _seed_map("Inspect brake BK1.") == {"brake-BK1": 0.8}
    assert _seed_map("Inspect brakeBK1.") == {"brake-BK1": 0.8}
    assert _seed_map("In building B1.") == {"building-B1": 0.8}
    assert _seed_map("In buildingB1.") == {"building-B1": 0.8}


def test_door_sensor_separator_variants(demo_gazetteer):
    # The bare name alone links nothing; with DS1 it is a variant hit.
    assert _seed_map("The door sensor DS1 is faulty.") == {"door-sensor-DS1": 0.8}
    assert _seed_map("The door-sensor DS1 is faulty.") == {"door-sensor-DS1": 0.8}
    assert _seed_map("The doorSensorDS1 is faulty.") == {"door-sensor-DS1": 0.8}
    assert _seed_map("The door-sensor-DS1 is faulty.") == {"door-sensor-DS1": 1.0}


def test_doc_separator_variants(demo_gazetteer):
    assert _seed_map("See doc motor overheat.") == {"doc-motor-overheat": 0.8}
    assert _seed_map("See docmotoroverheat.") == {"doc-motor-overheat": 0.8}
    assert _seed_map("See doc-motor-overheat.") == {"doc-motor-overheat": 1.0}
    assert _seed_map("See doc leveling for the procedure.") == {"doc-leveling": 0.8}
    # A following word is not swallowed into the doc id.
    assert _seed_map("See doc motor overheat manual.") == {"doc-motor-overheat": 0.8}


# --- Diacritics -----------------------------------------------------------

def test_diacritics_insensitive_matching(demo_gazetteer):
    assert _seed_map("Môtơr M2 quá nhiệt.") == {"motor-M2": 0.8}
    assert _seed_map("Thang máy O-21 bị kẹt.") == {"O21": 0.8}
    # "đoor sensor" without DS1 links nothing (no bare-name alias).
    assert _seed_map("Cảm biến cửa đoor sensor báo lỗi.") == {}


# --- Fault codes link nothing ---------------------------------------------

def test_fault_codes_link_nothing(demo_gazetteer):
    # There is no Fault entity type: fault codes link nothing,
    # exact or normalised — the old F-204 -> O21/motor-M2
    # expansion is gone with the demo tables.
    assert gazetteer_seeds("Fault F204 reported.") == []
    assert gazetteer_seeds("Fault f 204 reported.") == []
    assert gazetteer_seeds("Fault F301 reported.") == []
    assert gazetteer_seeds("Fault F999 reported.") == []
    assert gazetteer_seeds("Fault F-999 reported.") == []


# --- No false positives on embedded codes ---------------------------------

def test_no_false_positive_on_embedded_codes(demo_gazetteer):
    assert gazetteer_seeds("XO21 is not an asset.") == []
    assert gazetteer_seeds("O21x is not an asset.") == []
    assert gazetteer_seeds("The motorm2x unit.") == []
    assert gazetteer_seeds("FO204 is not a fault.") == []
    assert gazetteer_seeds("F2045 is not a fault.") == []
    assert gazetteer_seeds("See the docmotoroverheated manual.") == []
    assert gazetteer_seeds("No codes here at all.") == []


# --- Empty input / ordering -------------------------------------------------

def test_empty_and_none_input_return_empty_list(demo_gazetteer):
    assert gazetteer_seeds("") == []
    assert gazetteer_seeds(None) == []  # type: ignore[arg-type]


def test_variant_seeds_keep_first_mention_order(demo_gazetteer):
    assert _seed_keys("O22 then O21") == ["O22", "O21"]
    assert _seed_keys("O-21 then O22") == ["O21", "O22"]
    assert _seed_keys("motor-M2 then O21") == ["motor-M2", "O21"]
