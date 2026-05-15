"""Tests for section_tool.app_state.AppState."""

import numpy as np
import pytest

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick, Surface
from section_tool.core.wells import DeviationSurvey, Well
from section_tool.io.project import Project, SeismicRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Capture:
    """Records all positional args from every signal emission."""
    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, *args):
        self.calls.append(args)

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def last(self) -> tuple | None:
        return self.calls[-1] if self.calls else None

    def first_arg(self):
        return self.calls[0][0] if self.calls else None


def capture(signal) -> Capture:
    c = Capture()
    signal.connect(c)
    return c


def _sec(name="L1") -> Section:
    return Section([(0.0, 0.0), (1000.0, 0.0)], name=name)


def _surf(name="S1") -> Surface:
    return Surface([0, 1, 2], [0, 1, 2], [0, 1, 2], name=name)


def _pick(name="P1") -> HorizonPick:
    return HorizonPick([0.0, 500.0], [100.0, 200.0], name=name)


def _well(name="W1") -> Well:
    return Well(name=name, x=500.0, y=200.0)


def _ref(path="/data/line.segy", name="L1") -> SeismicRef:
    return SeismicRef(path=path, name=name)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_project_is_empty(self):
        s = AppState()
        assert len(s.project.sections) == 0
        assert len(s.project.wells) == 0

    def test_not_modified(self):
        assert not AppState().is_modified

    def test_no_path(self):
        assert AppState().project_path is None

    def test_no_active_section(self):
        assert AppState().active_section is None

    def test_no_active_well(self):
        assert AppState().active_well is None

    def test_active_tool_default_select(self):
        assert AppState().active_tool == "select"

    def test_repr_does_not_crash(self):
        s = AppState()
        assert "AppState" in repr(s)


# ---------------------------------------------------------------------------
# new_project
# ---------------------------------------------------------------------------

class TestNewProject:
    def test_project_is_replaced(self):
        s = AppState()
        s.add_section(_sec())
        s.new_project()
        assert len(s.project.sections) == 0

    def test_crs_epsg_applied(self):
        s = AppState()
        s.new_project(crs_epsg=27700)
        assert s.project.crs_epsg == 27700

    def test_name_applied(self):
        s = AppState()
        s.new_project(name="Fresh")
        assert s.project.name == "Fresh"

    def test_path_cleared(self):
        s = AppState()
        s._project_path = "/old/path.h5"
        s.new_project()
        assert s.project_path is None

    def test_not_modified_after_new(self):
        s = AppState()
        s.add_section(_sec())
        s.new_project()
        assert not s.is_modified

    def test_active_section_cleared(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        s.set_active_section(sec)
        s.new_project()
        assert s.active_section is None

    def test_active_well_cleared(self):
        s = AppState()
        well = _well()
        s.add_well(well)
        s.set_active_well(well)
        s.new_project()
        assert s.active_well is None

    def test_project_changed_signal(self):
        s = AppState()
        c = capture(s.project_changed)
        s.new_project()
        assert c.count == 1

    def test_path_changed_emits_empty_string(self):
        s = AppState()
        c = capture(s.project_path_changed)
        s.new_project()
        assert c.count == 1
        assert c.last == ("",)

    def test_modified_changed_emits_false(self):
        s = AppState()
        c = capture(s.project_modified_changed)
        s.new_project()
        assert (False,) in c.calls


# ---------------------------------------------------------------------------
# open_project / save_project / save_project_as
# ---------------------------------------------------------------------------

class TestProjectIO:
    def test_open_project_loads_sections(self, tmp_path):
        p = Project()
        p.sections.append(_sec("Loaded"))
        p.save(tmp_path / "proj.h5")

        s = AppState()
        s.open_project(tmp_path / "proj.h5")
        assert len(s.project.sections) == 1
        assert s.project.sections[0].name == "Loaded"

    def test_open_sets_path(self, tmp_path):
        path = tmp_path / "proj.h5"
        Project().save(path)
        s = AppState()
        s.open_project(path)
        assert s.project_path == str(path)

    def test_open_clears_modified(self, tmp_path):
        path = tmp_path / "proj.h5"
        Project().save(path)
        s = AppState()
        s.add_section(_sec())   # set modified first
        s.open_project(path)
        assert not s.is_modified

    def test_open_emits_project_changed(self, tmp_path):
        path = tmp_path / "proj.h5"
        Project().save(path)
        s = AppState()
        c = capture(s.project_changed)
        s.open_project(path)
        assert c.count == 1

    def test_open_emits_path_changed(self, tmp_path):
        path = tmp_path / "proj.h5"
        Project().save(path)
        s = AppState()
        c = capture(s.project_path_changed)
        s.open_project(path)
        assert c.last == (str(path),)

    def test_save_project_as_sets_path(self, tmp_path):
        path = tmp_path / "out.h5"
        s = AppState()
        s.save_project_as(path)
        assert s.project_path == str(path)

    def test_save_project_as_clears_modified(self, tmp_path):
        path = tmp_path / "out.h5"
        s = AppState()
        s.add_section(_sec())
        s.save_project_as(path)
        assert not s.is_modified

    def test_save_project_as_emits_path_changed(self, tmp_path):
        path = tmp_path / "out.h5"
        s = AppState()
        c = capture(s.project_path_changed)
        s.save_project_as(path)
        assert c.last == (str(path),)

    def test_save_project_clears_modified(self, tmp_path):
        path = tmp_path / "out.h5"
        s = AppState()
        s.save_project_as(path)          # establishes path
        s.add_section(_sec())            # re-dirties
        s.save_project()
        assert not s.is_modified

    def test_save_project_no_path_raises(self):
        s = AppState()
        with pytest.raises(ValueError, match="save_project_as"):
            s.save_project()

    def test_save_project_roundtrip(self, tmp_path):
        path = tmp_path / "rt.h5"
        s = AppState()
        s.add_section(_sec("MyLine"))
        s.save_project_as(path)
        s2 = AppState()
        s2.open_project(path)
        assert s2.project.sections[0].name == "MyLine"

    def test_open_clears_active_section(self, tmp_path):
        path = tmp_path / "p.h5"
        Project().save(path)
        s = AppState()
        s._active_section = _sec()
        s.open_project(path)
        assert s.active_section is None


# ---------------------------------------------------------------------------
# modified flag
# ---------------------------------------------------------------------------

class TestModifiedFlag:
    def test_add_section_sets_modified(self):
        s = AppState()
        s.add_section(_sec())
        assert s.is_modified

    def test_modified_emitted_once_on_first_change(self):
        s = AppState()
        c = capture(s.project_modified_changed)
        s.add_section(_sec())
        s.add_section(_sec())   # already modified — no second emit
        assert c.count == 1
        assert c.last == (True,)

    def test_set_active_section_does_not_dirty(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        # save to clear modified
        # (simulate save without file)
        s._is_modified = False
        s.set_active_section(sec)
        assert not s.is_modified

    def test_set_active_well_does_not_dirty(self):
        s = AppState()
        well = _well()
        s.add_well(well)
        s._is_modified = False
        s.set_active_well(well)
        assert not s.is_modified

    def test_remove_section_sets_modified(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        s._is_modified = False
        s.remove_section(sec)
        assert s.is_modified


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

class TestSections:
    def test_add_section_appends(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        assert sec in s.project.sections

    def test_add_section_emits_signal(self):
        s = AppState()
        c = capture(s.section_added)
        sec = _sec()
        s.add_section(sec)
        assert c.count == 1
        assert c.first_arg() is sec

    def test_remove_section_removes(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        s.remove_section(sec)
        assert sec not in s.project.sections

    def test_remove_section_emits_signal(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        c = capture(s.section_removed)
        s.remove_section(sec)
        assert c.first_arg() is sec

    def test_remove_active_section_clears_active(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        s.set_active_section(sec)
        s.remove_section(sec)
        assert s.active_section is None

    def test_remove_active_section_falls_back_to_next(self):
        s = AppState()
        sec1, sec2 = _sec("A"), _sec("B")
        s.add_section(sec1)
        s.add_section(sec2)
        s.set_active_section(sec1)
        s.remove_section(sec1)
        assert s.active_section is sec2

    def test_update_section_replaces(self):
        s = AppState()
        sec1, sec2 = _sec("Old"), _sec("New")
        s.add_section(sec1)
        s.update_section(0, sec2)
        assert s.project.sections[0] is sec2

    def test_update_section_emits_modified_signal(self):
        s = AppState()
        s.add_section(_sec())
        c = capture(s.section_modified)
        new = _sec("New")
        s.update_section(0, new)
        assert c.count == 1
        assert c.last == (0, new)

    def test_update_active_section_updates_reference(self):
        s = AppState()
        sec1, sec2 = _sec("Old"), _sec("New")
        s.add_section(sec1)
        s.set_active_section(sec1)
        s.update_section(0, sec2)
        assert s.active_section is sec2

    def test_set_active_section_emits_once(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        c = capture(s.active_section_changed)
        s.set_active_section(sec)
        s.set_active_section(sec)   # same — no second emit
        assert c.count == 1

    def test_set_active_section_none(self):
        s = AppState()
        sec = _sec()
        s.add_section(sec)
        s.set_active_section(sec)
        s.set_active_section(None)
        assert s.active_section is None

    def test_multiple_sections(self):
        s = AppState()
        for i in range(5):
            s.add_section(_sec(f"L{i}"))
        assert len(s.project.sections) == 5


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

class TestSurfaces:
    def test_add_surface(self):
        s = AppState()
        surf = _surf()
        s.add_surface(surf)
        assert surf in s.project.surfaces

    def test_add_surface_emits(self):
        s = AppState()
        c = capture(s.surface_added)
        surf = _surf()
        s.add_surface(surf)
        assert c.first_arg() is surf

    def test_remove_surface(self):
        s = AppState()
        surf = _surf()
        s.add_surface(surf)
        s.remove_surface(surf)
        assert surf not in s.project.surfaces

    def test_remove_surface_emits(self):
        s = AppState()
        surf = _surf()
        s.add_surface(surf)
        c = capture(s.surface_removed)
        s.remove_surface(surf)
        assert c.first_arg() is surf

    def test_update_surface(self):
        s = AppState()
        s1, s2 = _surf("A"), _surf("B")
        s.add_surface(s1)
        s.update_surface(0, s2)
        assert s.project.surfaces[0] is s2

    def test_update_surface_emits(self):
        s = AppState()
        s.add_surface(_surf())
        c = capture(s.surface_modified)
        s2 = _surf("New")
        s.update_surface(0, s2)
        assert c.last == (0, s2)


# ---------------------------------------------------------------------------
# Horizon picks
# ---------------------------------------------------------------------------

class TestHorizonPicks:
    def test_add_pick(self):
        s = AppState()
        p = _pick()
        s.add_horizon_pick(p)
        assert p in s.project.horizon_picks

    def test_add_pick_emits(self):
        s = AppState()
        c = capture(s.horizon_pick_added)
        p = _pick()
        s.add_horizon_pick(p)
        assert c.first_arg() is p

    def test_remove_pick(self):
        s = AppState()
        p = _pick()
        s.add_horizon_pick(p)
        s.remove_horizon_pick(p)
        assert p not in s.project.horizon_picks

    def test_remove_pick_emits(self):
        s = AppState()
        p = _pick()
        s.add_horizon_pick(p)
        c = capture(s.horizon_pick_removed)
        s.remove_horizon_pick(p)
        assert c.first_arg() is p

    def test_update_pick(self):
        s = AppState()
        p1, p2 = _pick("A"), _pick("B")
        s.add_horizon_pick(p1)
        s.update_horizon_pick(0, p2)
        assert s.project.horizon_picks[0] is p2

    def test_update_pick_emits(self):
        s = AppState()
        s.add_horizon_pick(_pick())
        c = capture(s.horizon_pick_modified)
        p2 = _pick("New")
        s.update_horizon_pick(0, p2)
        assert c.last == (0, p2)


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------

class TestWells:
    def test_add_well(self):
        s = AppState()
        w = _well()
        s.add_well(w)
        assert w in s.project.wells

    def test_add_well_emits(self):
        s = AppState()
        c = capture(s.well_added)
        w = _well()
        s.add_well(w)
        assert c.first_arg() is w

    def test_remove_well(self):
        s = AppState()
        w = _well()
        s.add_well(w)
        s.remove_well(w)
        assert w not in s.project.wells

    def test_remove_well_emits(self):
        s = AppState()
        w = _well()
        s.add_well(w)
        c = capture(s.well_removed)
        s.remove_well(w)
        assert c.first_arg() is w

    def test_remove_active_well_clears(self):
        s = AppState()
        w = _well()
        s.add_well(w)
        s.set_active_well(w)
        s.remove_well(w)
        assert s.active_well is None

    def test_remove_active_well_emits_none(self):
        s = AppState()
        w = _well()
        s.add_well(w)
        s.set_active_well(w)
        c = capture(s.active_well_changed)
        s.remove_well(w)
        assert c.last == (None,)

    def test_update_well(self):
        s = AppState()
        w1, w2 = _well("A"), _well("B")
        s.add_well(w1)
        s.update_well(0, w2)
        assert s.project.wells[0] is w2

    def test_update_well_emits(self):
        s = AppState()
        s.add_well(_well())
        c = capture(s.well_modified)
        w2 = _well("New")
        s.update_well(0, w2)
        assert c.last == (0, w2)

    def test_update_active_well_updates_reference(self):
        s = AppState()
        w1, w2 = _well("Old"), _well("New")
        s.add_well(w1)
        s.set_active_well(w1)
        s.update_well(0, w2)
        assert s.active_well is w2

    def test_set_active_well_emits_once(self):
        s = AppState()
        w = _well()
        s.add_well(w)
        c = capture(s.active_well_changed)
        s.set_active_well(w)
        s.set_active_well(w)   # same — no second emit
        assert c.count == 1

    def test_set_active_well_none(self):
        s = AppState()
        w = _well()
        s.add_well(w)
        s.set_active_well(w)
        s.set_active_well(None)
        assert s.active_well is None


# ---------------------------------------------------------------------------
# Seismic refs
# ---------------------------------------------------------------------------

class TestSeismicRefs:
    def test_add_ref(self):
        s = AppState()
        r = _ref()
        s.add_seismic_ref(r)
        assert r in s.project.seismic_refs

    def test_add_ref_emits(self):
        s = AppState()
        c = capture(s.seismic_ref_added)
        r = _ref()
        s.add_seismic_ref(r)
        assert c.first_arg() is r

    def test_remove_ref(self):
        s = AppState()
        r = _ref()
        s.add_seismic_ref(r)
        s.remove_seismic_ref(r)
        assert r not in s.project.seismic_refs

    def test_remove_ref_emits(self):
        s = AppState()
        r = _ref()
        s.add_seismic_ref(r)
        c = capture(s.seismic_ref_removed)
        s.remove_seismic_ref(r)
        assert c.first_arg() is r

    def test_remove_missing_ref_raises(self):
        s = AppState()
        with pytest.raises(ValueError):
            s.remove_seismic_ref(_ref())


# ---------------------------------------------------------------------------
# Cross-cutting signal behaviour
# ---------------------------------------------------------------------------

class TestSignalBehaviour:
    def test_project_changed_not_emitted_on_add(self):
        s = AppState()
        c = capture(s.project_changed)
        s.add_section(_sec())
        assert c.count == 0

    def test_active_section_changed_passes_none(self):
        s = AppState()
        c = capture(s.active_section_changed)
        s.set_active_section(None)
        # None→None: no emission (already None)
        assert c.count == 0

    def test_active_section_changed_passes_section(self):
        s = AppState()
        sec = _sec()
        c = capture(s.active_section_changed)
        s.set_active_section(sec)
        assert c.first_arg() is sec

    def test_all_adds_set_modified(self):
        s = AppState()
        ops = [
            lambda: s.add_section(_sec()),
            lambda: s.add_surface(_surf()),
            lambda: s.add_horizon_pick(_pick()),
            lambda: s.add_well(_well()),
            lambda: s.add_seismic_ref(_ref()),
        ]
        for op in ops:
            s._is_modified = False
            op()
            assert s.is_modified

    def test_all_removes_set_modified(self):
        s = AppState()
        sec = _sec(); s.add_section(sec)
        surf = _surf(); s.add_surface(surf)
        pick = _pick(); s.add_horizon_pick(pick)
        well = _well(); s.add_well(well)
        ref = _ref(); s.add_seismic_ref(ref)

        for obj, remove_fn in [
            (sec, s.remove_section),
            (surf, s.remove_surface),
            (pick, s.remove_horizon_pick),
            (well, s.remove_well),
            (ref, s.remove_seismic_ref),
        ]:
            s._is_modified = False
            remove_fn(obj)
            assert s.is_modified

    def test_update_operations_set_modified(self):
        s = AppState()
        s.add_section(_sec())
        s.add_surface(_surf())
        s.add_horizon_pick(_pick())
        s.add_well(_well())

        ops = [
            (s.update_section, 0, _sec("New")),
            (s.update_surface, 0, _surf("New")),
            (s.update_horizon_pick, 0, _pick("New")),
            (s.update_well, 0, _well("New")),
        ]
        for fn, idx, obj in ops:
            s._is_modified = False
            fn(idx, obj)
            assert s.is_modified


# ---------------------------------------------------------------------------
# Active tool
# ---------------------------------------------------------------------------

class TestActiveTool:
    def test_default_is_select(self):
        assert AppState().active_tool == "select"

    def test_set_active_tool(self):
        s = AppState()
        s.set_active_tool("pan")
        assert s.active_tool == "pan"

    def test_tool_changed_signal_emitted(self):
        s = AppState()
        received = []
        s.tool_changed.connect(lambda t: received.append(t))
        s.set_active_tool("zoom")
        assert received == ["zoom"]

    def test_tool_changed_not_emitted_for_same_tool(self):
        s = AppState()
        s.set_active_tool("select")
        received = []
        s.tool_changed.connect(lambda t: received.append(t))
        s.set_active_tool("select")
        assert received == []

    def test_all_palette_tools_settable(self):
        from section_tool.views.tool_palette import _TOOL_IDS
        s = AppState()
        for tid in _TOOL_IDS:
            s.set_active_tool(tid)
            assert s.active_tool == tid


# ---------------------------------------------------------------------------
# Active pick target
# ---------------------------------------------------------------------------

class TestActivePickTarget:
    def test_defaults_none(self):
        s = AppState()
        assert s.active_pick_category is None
        assert s.active_pick_index is None

    def test_set_active_pick_target(self):
        s = AppState()
        s.set_active_pick_target("Horizons", 2)
        assert s.active_pick_category == "Horizons"
        assert s.active_pick_index == 2

    def test_set_fault_pick_target(self):
        s = AppState()
        s.set_active_pick_target("Faults", 0)
        assert s.active_pick_category == "Faults"
        assert s.active_pick_index == 0

    def test_signal_emitted_on_change(self):
        s = AppState()
        received = []
        s.active_pick_target_changed.connect(lambda c, i: received.append((c, i)))
        s.set_active_pick_target("Horizons", 1)
        assert received == [("Horizons", 1)]

    def test_signal_emitted_every_call(self):
        s = AppState()
        received = []
        s.active_pick_target_changed.connect(lambda c, i: received.append((c, i)))
        s.set_active_pick_target("Horizons", 0)
        s.set_active_pick_target("Horizons", 0)
        assert len(received) == 2

    def test_overwrite_category(self):
        s = AppState()
        s.set_active_pick_target("Horizons", 1)
        s.set_active_pick_target("Faults", 0)
        assert s.active_pick_category == "Faults"
        assert s.active_pick_index == 0


# ---------------------------------------------------------------------------
# HorizonPick add / remove / sort (integration via AppState)
# ---------------------------------------------------------------------------

class TestPickPointOperations:
    def test_add_pick_increments_count(self):
        s = AppState()
        hp = HorizonPick([0.0], [100.0], name="H1")
        s.add_horizon_pick(hp)
        copy_hp = hp.__class__(
            hp.distances.tolist() + [500.0],
            hp.depths.tolist() + [200.0],
            name=hp.name, color=hp.color,
        )
        s.update_horizon_pick(0, copy_hp)
        assert s.project.horizon_picks[0].n_picks == 2

    def test_insert_pick_sorts_by_distance(self):
        import copy as _copy
        s = AppState()
        hp = HorizonPick([0.0, 1000.0], [100.0, 150.0], name="H1")
        s.add_horizon_pick(hp)
        hp2 = _copy.deepcopy(s.project.horizon_picks[0])
        hp2.insert_pick(500.0, 125.0)
        s.update_horizon_pick(0, hp2)
        result = s.project.horizon_picks[0]
        import numpy as np
        assert np.all(np.diff(result.distances) >= 0)
        assert result.n_picks == 3

    def test_remove_pick_decrements_count(self):
        import copy as _copy
        s = AppState()
        hp = HorizonPick([0.0, 500.0, 1000.0], [100.0, 120.0, 150.0])
        s.add_horizon_pick(hp)
        hp2 = _copy.deepcopy(s.project.horizon_picks[0])
        hp2.delete_pick(1)
        s.update_horizon_pick(0, hp2)
        assert s.project.horizon_picks[0].n_picks == 2

    def test_update_pick_emits_modified(self):
        import copy as _copy
        s = AppState()
        hp = HorizonPick([0.0], [100.0])
        s.add_horizon_pick(hp)
        received = []
        s.horizon_pick_modified.connect(lambda i, p: received.append((i, p)))
        hp2 = _copy.deepcopy(hp)
        hp2.insert_pick(500.0, 200.0)
        s.update_horizon_pick(0, hp2)
        assert len(received) == 1
        assert received[0][0] == 0

    def test_empty_pick_can_receive_points(self):
        import copy as _copy
        s = AppState()
        hp = HorizonPick.empty(name="Empty")
        s.add_horizon_pick(hp)
        assert s.project.horizon_picks[0].n_picks == 0
        hp2 = _copy.deepcopy(s.project.horizon_picks[0])
        hp2.insert_pick(300.0, 500.0)
        s.update_horizon_pick(0, hp2)
        assert s.project.horizon_picks[0].n_picks == 1


# ---------------------------------------------------------------------------
# active_pick_target_changed signal fires on section-panel interaction
# ---------------------------------------------------------------------------

class TestPickTargetSignalChain:
    def test_pick_target_changed_fires_with_correct_args(self):
        s = AppState()
        s.add_horizon_pick(HorizonPick([0.0], [100.0], name="Top"))
        s.add_horizon_pick(HorizonPick([0.0], [200.0], name="Base"))
        received = []
        s.active_pick_target_changed.connect(lambda c, i: received.append((c, i)))
        s.set_active_pick_target("Horizons", 1)
        assert received == [("Horizons", 1)]

    def test_active_pick_index_matches_set_value(self):
        s = AppState()
        for i in range(3):
            s.add_horizon_pick(HorizonPick([float(i * 100)], [float(i * 50)]))
        s.set_active_pick_target("Horizons", 2)
        assert s.active_pick_index == 2

    def test_fault_pick_target_independent_of_horizon(self):
        s = AppState()
        s.add_horizon_pick(HorizonPick([0.0], [100.0]))
        from section_tool.core.surfaces import HorizonPick as HP
        s.add_fault_pick(HP([0.0], [100.0], name="F1"))
        s.set_active_pick_target("Horizons", 0)
        assert s.active_pick_category == "Horizons"
        s.set_active_pick_target("Faults", 0)
        assert s.active_pick_category == "Faults"
        assert s.active_pick_index == 0
