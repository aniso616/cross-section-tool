"""RestorationEvent/Sequence: UUID-keyed element removal, serialization
round-trip, rename invariance, and v1 (name) → v2 (UUID) migration. (Step 1.)"""
from __future__ import annotations

import json
from types import SimpleNamespace

from section_tool.core.restoration import (
    RestorationEvent, RestorationSequence,
    resolve_removed_entities, migrate_names_to_ids, restore_remove_layer)
from section_tool.core.surfaces import HorizonPick
from section_tool.core.polygons import SectionPolygon


def _hp(name):
    return HorizonPick([0.0, 1000.0], [100.0, 200.0], name=name)


def _poly(name):
    return SectionPolygon([[0, 0], [100, 0], [100, 100]], name=name, section_name="L1")


def _project(horizons=(), faults=(), polygons=()):
    return SimpleNamespace(horizon_picks=list(horizons),
                           fault_picks=list(faults),
                           polygons=list(polygons))


def test_event_roundtrip_uuid_keyed():
    ev = RestorationEvent(1, "Remove Oligocene", age_ma=34.0,
                          remove_element_ids=["uuid-a", "uuid-b"])
    seq = RestorationSequence(events=[ev], current_step=1)
    seq2 = RestorationSequence.from_json(seq.to_json())
    assert seq2.current_step == 1
    assert seq2.events[0].remove_element_ids == ["uuid-a", "uuid-b"]
    assert seq2.events[0].name == "Remove Oligocene"
    assert json.loads(seq.to_json())["schema_version"] == 2      # version stamped


def test_resolve_returns_the_targeted_entities():
    h1, h2 = _hp("Top Chalk"), _hp("Base Chalk")
    proj = _project(horizons=[h1, h2])
    seq = RestorationSequence(events=[RestorationEvent(1, "e",
                                                       remove_element_ids=[h2.uuid])])
    assert resolve_removed_entities(seq, 0, proj) == []          # present day
    assert resolve_removed_entities(seq, 1, proj) == [h2]        # only the target
    assert seq.removed_ids_at_step(1) == {h2.uuid}


def test_rename_invariance():
    """The whole point: rename the entity, the event still resolves to it."""
    h = _hp("Top Chalk")
    proj = _project(horizons=[h])
    seq = RestorationSequence(events=[RestorationEvent(1, "e",
                                                       remove_element_ids=[h.uuid])])
    h.name = "Top Chalk (renamed)"                               # rename
    assert resolve_removed_entities(seq, 1, proj) == [h]         # still resolves


def test_resolver_is_type_agnostic():
    h, f, p = _hp("H"), _hp("F"), _poly("P")
    proj = _project(horizons=[h], faults=[f], polygons=[p])
    seq = RestorationSequence(events=[RestorationEvent(
        1, "e", remove_element_ids=[h.uuid, p.uuid])])
    got = {id(o) for o in resolve_removed_entities(seq, 1, proj)}
    assert got == {id(h), id(p)} and id(f) not in got           # horizon + polygon, not fault


def test_v1_name_schema_migrates_to_uuids():
    h = _hp("Top Chalk")
    proj = _project(horizons=[h])
    v1 = ('{"current_step": 1, "events": [{"event_id": 1, "name": "e", '
          '"remove_elements": ["Top Chalk", "Ghost Bed"]}]}')          # legacy schema
    seq = RestorationSequence.from_json(v1)
    assert seq.events[0].remove_element_ids == []                # not yet migrated
    assert seq.events[0].remove_element_names == ["Top Chalk", "Ghost Bed"]

    unresolved = migrate_names_to_ids(seq, proj)
    assert seq.events[0].remove_element_ids == [h.uuid]          # resolved by name
    assert unresolved == ["Ghost Bed"]                          # flagged, not dropped
    assert seq.events[0].remove_element_names == ["Ghost Bed"]  # kept on the event

    d = json.loads(seq.to_json())                               # next save is v2/UUID
    assert d["schema_version"] == 2
    assert d["events"][0]["remove_element_ids"] == [h.uuid]
    assert migrate_names_to_ids(seq, proj) == ["Ghost Bed"]     # idempotent


def test_restore_remove_layer_filters_by_uuid():
    h1, h2 = _hp("A"), _hp("B")
    seq = RestorationSequence(events=[RestorationEvent(1, "e",
                                                       remove_element_ids=[h2.uuid])])
    hz, fl, pl = restore_remove_layer(seq, 1, [h1, h2], [], [])
    assert hz == [h1] and fl == [] and pl == []                 # h2 removed, by uuid


def test_polygon_has_stable_uuid():
    p = _poly("P")
    assert isinstance(p.uuid, str) and len(p.uuid) >= 32
    assert _poly("P").uuid != p.uuid                            # distinct per instance
    assert SectionPolygon([[0, 0], [1, 0], [1, 1]], uuid="fixed-id").uuid == "fixed-id"
