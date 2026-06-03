"""Slice-observation round-trip: ONE fault entity carries a section trace AND a
horizontal plan-slice trace, both linked to the same entity, surviving save→reload.

This is the proof for the slice generalization (Step 2/3): the observation model
is slice-agnostic. No drawing UI, no lofting, no view — storage round-trip only.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice
from section_tool.core.surfaces import HorizonPick

NAN = float("nan")


def _open(folder):
    dst = AppState()
    dst.open_project(folder)
    return dst


# ---------------------------------------------------------------------------
# One entity, two observation kinds
# ---------------------------------------------------------------------------

class TestFaultCarriesBothSliceKinds:

    def _build(self, folder):
        src = AppState()
        src.new_project(name="SL", crs_epsg=32631, folder_path=folder)
        sec = Section([(0.0, 0.0), (2000.0, 0.0)], name="L1")
        src.add_section(sec)
        src.set_active_section(sec)
        src.add_horizontal_slice(HorizontalSlice(name="Z-1500", elevation=-1500.0,
                                                 crs_epsg=32631))
        # ONE fault entity: a section trace (2 pts on L1) + a horizontal trace
        # (2 pts on Z-1500). Horizontal pts: depth = -z0 = 1500, world e/n in map_x/y.
        fault = HorizonPick(
            distances=[0.0, 2000.0, 0.0, 0.0],
            depths=[700.0, 800.0, 1500.0, 1500.0],
            name="F1", color="#d62728",
            section_names=["L1", "L1", "Z-1500", "Z-1500"],
            slice_kinds=["section", "section", "horizontal", "horizontal"],
            map_x=[NAN, NAN, 600100.0, 600200.0],
            map_y=[NAN, NAN, 6080100.0, 6080200.0],
        )
        src.add_fault_pick(fault)
        uuid = fault.uuid
        src.save_project()
        return uuid

    def test_both_observations_on_one_entity(self, tmp_path):
        folder = str(tmp_path / "proj")
        uuid = self._build(folder)
        dst = _open(folder)

        assert len(dst.project.fault_picks) == 1
        f = dst.project.fault_picks[0]
        assert f.uuid == uuid                                   # same entity
        assert set(f.section_names()) == {"L1"}                 # section observation
        assert set(f.horizontal_slice_refs()) == {"Z-1500"}     # horizontal observation

    def test_horizontal_world_coords_and_elevation_round_trip(self, tmp_path):
        folder = str(tmp_path / "proj")
        self._build(folder)
        f = _open(folder).project.fault_picks[0]
        h = f.indices_for_slice("horizontal", "Z-1500")
        assert len(h) == 2
        assert sorted(f._map_x[h].tolist()) == [600100.0, 600200.0]
        assert sorted(f._map_y[h].tolist()) == [6080100.0, 6080200.0]
        # depth = -elevation; z0 = -1500 → depth 1500
        assert np.allclose(f._depths[h], 1500.0)

    def test_section_observation_unchanged(self, tmp_path):
        folder = str(tmp_path / "proj")
        self._build(folder)
        f = _open(folder).project.fault_picks[0]
        s = f.indices_for_slice("section", "L1")
        assert len(s) == 2
        assert sorted(f._depths[s].tolist()) == [700.0, 800.0]

    def test_horizontal_slice_registry_restored(self, tmp_path):
        folder = str(tmp_path / "proj")
        self._build(folder)
        dst = _open(folder)
        hs = {s.name: s.elevation for s in dst.project.horizontal_slices}
        assert hs == {"Z-1500": -1500.0}


# ---------------------------------------------------------------------------
# Horizontal-slice registry round-trips on its own
# ---------------------------------------------------------------------------

class TestHorizontalSliceRegistry:

    def test_multiple_slices_round_trip(self, tmp_path):
        folder = str(tmp_path / "proj")
        src = AppState()
        src.new_project(name="SL", crs_epsg=32631, folder_path=folder)
        src.add_horizontal_slice(HorizontalSlice("Z-2000", -2000.0, 32631))
        src.add_horizontal_slice(HorizontalSlice("Z-1000", -1000.0, 32631))
        src.save_project()

        dst = _open(folder)
        got = {s.name: (s.elevation, s.kind) for s in dst.project.horizontal_slices}
        assert got == {"Z-2000": (-2000.0, "horizontal"),
                       "Z-1000": (-1000.0, "horizontal")}


# ---------------------------------------------------------------------------
# Backward compatibility: section-only projects carry no horizontal contamination
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_section_only_project_all_section_kind(self, tmp_path):
        folder = str(tmp_path / "proj")
        src = AppState()
        src.new_project(name="SO", crs_epsg=32631, folder_path=folder)
        sec = Section([(0.0, 0.0), (2000.0, 0.0)], name="L1")
        src.add_section(sec)
        src.set_active_section(sec)
        src.add_fault_pick(HorizonPick([0.0, 2000.0], [700.0, 800.0], name="F1",
                                       section_names=["L1", "L1"]))
        src.save_project()

        dst = _open(folder)
        assert dst.project.horizontal_slices == []
        for e in dst.project.horizon_picks + dst.project.fault_picks:
            assert set(e._slice_kinds.tolist()) <= {"section"}
            assert e.horizontal_slice_refs() == []
