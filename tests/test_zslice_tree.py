"""Project tree: the Z-Slices category lists horizontal slices and activates them."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication

from section_tool.app_state import AppState
from section_tool.core.section import Section
from section_tool.core.slices import HorizontalSlice
from section_tool.views.project_panel import ProjectPanel, _CATEGORIES, _ObjectRow


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def panel(qapp):
    st = AppState()
    st.add_section(Section([(0.0, 0.0), (1000.0, 0.0)], name="L1"))
    st.project.horizontal_slices.append(HorizontalSlice("Z-1500", -1500.0))
    st.project.horizontal_slices.append(HorizontalSlice("Z-2000", -2000.0))
    p = ProjectPanel(st)
    p._rebuild()
    return p, st


def _cat_item(panel, name):
    return panel._category_items.get(name)


class TestZSliceCategory:
    def test_category_present(self):
        assert "Z-Slices" in _CATEGORIES

    def test_lists_registered_slices(self, panel):
        p, _ = panel
        cat = _cat_item(p, "Z-Slices")
        assert cat is not None
        names = [p._tree.itemWidget(cat.child(i), 0)._name
                 for i in range(cat.childCount())]
        assert names == ["Z-1500", "Z-2000"]

    def test_objects_for_category(self, panel):
        p, _ = panel
        objs = p._objects_for_category("Z-Slices")
        assert [o[0] for o in objs] == ["Z-1500", "Z-2000"]


class TestZSliceActivation:
    def test_click_sets_active_slice(self, panel):
        p, st = panel
        seen = []
        st.active_slice_changed.connect(lambda s: seen.append(s))
        cat = _cat_item(p, "Z-Slices")
        p._on_item_clicked(cat.child(1), 0)            # click Z-2000
        assert st.active_slice is st.project.horizontal_slices[1]
        assert st.active_slice.kind == "horizontal"
        assert seen and seen[-1].name == "Z-2000"
        # active_section must NOT be disturbed by activating a z-slice
        assert st.active_section is None

    def _marks(self, p, cat):
        ci = _cat_item(p, cat)
        return [p._tree.itemWidget(ci.child(i), 0)._label.font().bold()
                for i in range(ci.childCount())]

    def test_marker_moves_to_active_slice(self, panel):
        p, st = panel
        # activate a z-slice, then check exactly that row is marked active
        st.set_active_slice(st.project.horizontal_slices[0])
        assert self._marks(p, "Z-Slices") == [True, False]
        # and the section row is not marked (z-slice is the active workspace)
        assert self._marks(p, "Sections") == [False]

    def test_section_click_still_marks_section(self, panel):
        p, st = panel
        scat = _cat_item(p, "Sections")
        p._on_item_clicked(scat.child(0), 0)
        assert st.active_section is st.project.sections[0]   # delegation intact
        assert st.active_slice is st.project.sections[0]
