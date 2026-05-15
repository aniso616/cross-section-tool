"""Tests for section_tool.io.export — file output verification.

All tests are headless (no Qt).  CSV tests use a real in-memory SQLite
database; figure tests use in-memory Python objects and verify file creation
and basic content.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from section_tool.core.polygons import SectionPolygon
from section_tool.core.section import Section
from section_tool.core.surfaces import HorizonPick
from section_tool.io.database import ProjectDatabase
from section_tool.io.export import (
    export_faults_csv,
    export_horizons_csv,
    export_section_figure,
    export_wells_csv,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_path_str(tmp_path):
    return str(tmp_path)


@pytest.fixture
def db(tmp_path):
    """Fresh in-memory-style SQLite database (file in temp dir)."""
    db_path = tmp_path / "test.db"
    return ProjectDatabase(db_path)


def _ew_section(name="EW", length=10_000.0) -> Section:
    return Section([(0.0, 0.0), (length, 0.0)], name=name)


def _horizon_pick(name="TopSand", section_name="EW",
                  distances=None, depths=None) -> HorizonPick:
    d = np.array(distances or [0.0, 5000.0, 10000.0])
    z = np.array(depths or [1000.0, 1200.0, 1100.0])
    return HorizonPick(d, z, name=name, color="#0000ff",
                       section_names=[section_name] * len(d))


def _fault_pick(name="F1", section_name="EW",
                distances=None, depths=None) -> HorizonPick:
    d = np.array(distances or [3000.0, 4000.0])
    z = np.array(depths or [500.0, 2000.0])
    return HorizonPick(d, z, name=name, color="#ff0000",
                       section_names=[section_name] * len(d))


def _populate_db(db) -> None:
    """Insert one section, one horizon, one fault, two wells into db."""
    sec = _ew_section()
    db.upsert_section(sec)

    hp = _horizon_pick()
    db.upsert_horizon(hp)

    fp = _fault_pick()
    db.upsert_fault(fp)

    from section_tool.core.wells import Well
    w1 = Well(name="F02-01", x=5_000.0, y=0.0, kb=25.0)
    w1.add_formation_top("TopSand", 980.0)
    w1.add_formation_top("BaseSand", 1120.0)
    db.upsert_well(w1)

    w2 = Well(name="F03-01", x=8_000.0, y=0.0, kb=22.0)
    db.upsert_well(w2)


# ---------------------------------------------------------------------------
# 1. export_horizons_csv
# ---------------------------------------------------------------------------

class TestExportHorizonsCSV:

    def test_creates_file(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        assert out.exists()

    def test_returns_correct_row_count(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        n = export_horizons_csv(db, out)
        assert n == 3  # TopSand has 3 picks

    def test_csv_has_header(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert "horizon" in rows[0]
        assert "section" in rows[0]
        assert "distance" in rows[0]
        assert "depth" in rows[0]
        assert "x" in rows[0]
        assert "y" in rows[0]

    def test_horizon_name_column(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert all(r["horizon"] == "TopSand" for r in rows)

    def test_section_name_column(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert all(r["section"] == "EW" for r in rows)

    def test_distance_values_correct(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        distances = [float(r["distance"]) for r in rows]
        assert distances == pytest.approx([0.0, 5000.0, 10000.0], abs=0.01)

    def test_depth_values_correct(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        depths = [float(r["depth"]) for r in rows]
        assert depths == pytest.approx([1000.0, 1200.0, 1100.0], abs=0.01)

    def test_map_coordinates_computed(self, db, tmp_path):
        """x/y must reflect section geometry: EW section at y=0."""
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        xs = [float(r["x"]) for r in rows]
        ys = [float(r["y"]) for r in rows]
        # Section runs from (0,0) to (10000,0), so y should be ~0 everywhere
        assert xs == pytest.approx([0.0, 5000.0, 10000.0], abs=0.1)
        assert ys == pytest.approx([0.0, 0.0, 0.0], abs=0.1)

    def test_elevation_is_negative_depth(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "horizons.csv"
        export_horizons_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        for r in rows:
            assert float(r["elevation"]) == pytest.approx(-float(r["depth"]), abs=0.01)

    def test_section_filter(self, db, tmp_path):
        """section_name kwarg should only export picks for that section."""
        _populate_db(db)
        # Add a second section and a pick on it
        sec2 = Section([(0.0, 1000.0), (10_000.0, 1000.0)], name="NS")
        db.upsert_section(sec2)
        # Upsert a new horizon that spans both EW (3 picks) and NS (1 pick)
        all_d = np.array([0.0, 2000.0, 5000.0, 10000.0])
        all_z = np.array([1000.0, 800.0, 1200.0, 1100.0])
        all_s = ["EW", "NS", "EW", "EW"]
        merged = HorizonPick(all_d, all_z, name="Merged", color="#00ff00",
                             section_names=all_s)
        db.upsert_horizon(merged)

        out = tmp_path / "filtered.csv"
        n = export_horizons_csv(db, out, section_name="NS")
        # Only the "Merged" NS pick (1) plus none from TopSand (which is EW-only)
        assert n == 1
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert rows[0]["section"] == "NS"

    def test_empty_db_creates_header_only(self, db, tmp_path):
        out = tmp_path / "empty.csv"
        n = export_horizons_csv(db, out)
        assert n == 0
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert rows == []
        # Header row must still be present
        with open(out, encoding="utf-8") as fh:
            header = fh.readline().strip()
        assert "horizon" in header


# ---------------------------------------------------------------------------
# 2. export_faults_csv
# ---------------------------------------------------------------------------

class TestExportFaultsCSV:

    def test_creates_file(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "faults.csv"
        export_faults_csv(db, out)
        assert out.exists()

    def test_returns_correct_row_count(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "faults.csv"
        n = export_faults_csv(db, out)
        assert n == 2  # F1 has 2 picks

    def test_csv_has_fault_specific_columns(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "faults.csv"
        export_faults_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert "fault_type" in rows[0]
        assert "dip_direction" in rows[0]

    def test_fault_name_column(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "faults.csv"
        export_faults_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert all(r["fault"] == "F1" for r in rows)

    def test_depth_and_elevation_opposite_sign(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "faults.csv"
        export_faults_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        for r in rows:
            assert float(r["elevation"]) == pytest.approx(-float(r["depth"]), abs=0.01)

    def test_section_filter(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "filtered.csv"
        n = export_faults_csv(db, out, section_name="NonExistent")
        assert n == 0

    def test_map_coordinates_on_ew_section(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "faults.csv"
        export_faults_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        xs = [float(r["x"]) for r in rows]
        assert xs == pytest.approx([3000.0, 4000.0], abs=0.1)

    def test_empty_db_creates_header_only(self, db, tmp_path):
        out = tmp_path / "empty.csv"
        n = export_faults_csv(db, out)
        assert n == 0
        with open(out, encoding="utf-8") as fh:
            header = fh.readline().strip()
        assert "fault" in header


# ---------------------------------------------------------------------------
# 3. export_wells_csv
# ---------------------------------------------------------------------------

class TestExportWellsCSV:

    def test_creates_file(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "wells.csv"
        export_wells_csv(db, out)
        assert out.exists()

    def test_well_with_tops_produces_one_row_per_top(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "wells.csv"
        export_wells_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        f02_rows = [r for r in rows if r["well"] == "F02-01"]
        assert len(f02_rows) == 2  # TopSand + BaseSand

    def test_well_without_tops_produces_one_row(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "wells.csv"
        export_wells_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        f03_rows = [r for r in rows if r["well"] == "F03-01"]
        assert len(f03_rows) == 1

    def test_total_row_count(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "wells.csv"
        n = export_wells_csv(db, out)
        # F02-01: 2 tops, F03-01: 0 tops → 2 + 1 = 3 rows
        assert n == 3

    def test_csv_header_columns(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "wells.csv"
        export_wells_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert "well" in rows[0]
        assert "x" in rows[0]
        assert "y" in rows[0]
        assert "kb_elevation" in rows[0]
        assert "formation" in rows[0]
        assert "depth_md" in rows[0]

    def test_formation_top_depths_correct(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "wells.csv"
        export_wells_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        f02 = {r["formation"]: float(r["depth_md"])
               for r in rows if r["well"] == "F02-01" and r["formation"]}
        assert f02["TopSand"] == pytest.approx(980.0, abs=0.01)
        assert f02["BaseSand"] == pytest.approx(1120.0, abs=0.01)

    def test_well_coordinates_present(self, db, tmp_path):
        _populate_db(db)
        out = tmp_path / "wells.csv"
        export_wells_csv(db, out)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        f02 = next(r for r in rows if r["well"] == "F02-01")
        assert float(f02["x"]) == pytest.approx(5000.0, abs=0.1)

    def test_empty_db_creates_header_only(self, db, tmp_path):
        out = tmp_path / "empty.csv"
        n = export_wells_csv(db, out)
        assert n == 0
        with open(out, encoding="utf-8") as fh:
            header = fh.readline().strip()
        assert "well" in header


# ---------------------------------------------------------------------------
# 4. export_section_figure
# ---------------------------------------------------------------------------

class TestExportSectionFigure:

    @pytest.fixture
    def section(self):
        return _ew_section()

    @pytest.fixture
    def picks(self):
        return [_horizon_pick()]

    @pytest.fixture
    def faults(self):
        return [_fault_pick()]

    @pytest.fixture
    def polygon(self):
        verts = [(0.0, 0.0), (5000.0, 0.0), (5000.0, 2000.0), (0.0, 2000.0)]
        return SectionPolygon(verts, name="UnitA", fill_color="#aabbcc")

    def test_creates_png(self, tmp_path, section, picks, faults, polygon):
        out = tmp_path / "section.png"
        export_section_figure(section, picks, faults, [polygon], None, out)
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG

    def test_creates_pdf(self, tmp_path, section, picks, faults):
        out = tmp_path / "section.pdf"
        export_section_figure(section, picks, faults, [], None, out)
        assert out.exists()
        # PDF files start with "%PDF"
        with open(out, "rb") as fh:
            assert fh.read(4) == b"%PDF"

    def test_creates_svg(self, tmp_path, section, picks):
        out = tmp_path / "section.svg"
        export_section_figure(section, picks, [], [], None, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content

    def test_no_picks_no_crash(self, tmp_path, section):
        out = tmp_path / "empty.png"
        export_section_figure(section, [], [], [], None, out)
        assert out.exists()

    def test_respects_dpi(self, tmp_path, section, picks):
        """A 300 DPI file should be noticeably larger than a 72 DPI file."""
        hi_dpi = tmp_path / "hi.png"
        lo_dpi = tmp_path / "lo.png"
        export_section_figure(section, picks, [], [], None, hi_dpi, dpi=300)
        export_section_figure(section, picks, [], [], None, lo_dpi, dpi=72)
        assert hi_dpi.stat().st_size > lo_dpi.stat().st_size

    def test_respects_figure_size(self, tmp_path, section):
        """Larger figure → larger PNG file."""
        big = tmp_path / "big.png"
        small = tmp_path / "small.png"
        export_section_figure(section, [], [], [], None, big,
                              width_inches=20, height_inches=10, dpi=100)
        export_section_figure(section, [], [], [], None, small,
                              width_inches=4, height_inches=2, dpi=100)
        assert big.stat().st_size > small.stat().st_size

    def test_multiple_horizons(self, tmp_path, section):
        h1 = _horizon_pick("H1", depths=[1000.0, 1050.0, 1100.0])
        h2 = _horizon_pick("H2", depths=[2000.0, 2100.0, 2200.0])
        out = tmp_path / "multi.png"
        export_section_figure(section, [h1, h2], [], [], None, out)
        assert out.exists()

    def test_picks_on_other_section_not_rendered(self, tmp_path, section):
        """Picks for a different section should not cause errors."""
        other = _horizon_pick("TopSand", section_name="OTHER_SEC")
        out = tmp_path / "other.png"
        export_section_figure(section, [other], [], [], None, out)
        assert out.exists()

    def test_with_synthetic_seismic(self, tmp_path, section):
        """A SeismicDataset mock should render without errors."""
        from unittest.mock import MagicMock
        import numpy as np
        seis = MagicMock()
        n_traces = 20
        n_samples = 50
        seis.data = np.random.randn(n_traces, n_samples).astype(np.float32)
        distances = np.linspace(0, 10_000, n_traces)
        seis.traces_sorted_by_section.return_value = (
            distances, seis.data, np.zeros(n_traces)
        )
        seis.time_range = (0.0, 2000.0)
        out = tmp_path / "seismic.png"
        export_section_figure(section, [], [], [], seis, out)
        assert out.exists()

    def test_twt_section_ylabel(self, tmp_path):
        """TWT domain section should produce a valid file."""
        sec = Section([(0.0, 0.0), (10_000.0, 0.0)], name="TWT",
                      depth_domain="twt", depth_units="ms")
        out = tmp_path / "twt.png"
        export_section_figure(sec, [], [], [], None, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# 5. _distance_to_xy helper
# ---------------------------------------------------------------------------

class TestDistanceToXY:
    """White-box tests for the coordinate interpolation helper."""

    def test_midpoint_of_ew_section(self):
        from section_tool.io.export import _distance_to_xy
        nodes = {"EW": np.array([[0.0, 0.0], [10_000.0, 0.0]])}
        x, y = _distance_to_xy(nodes, "EW", 5000.0)
        assert x == pytest.approx(5000.0, abs=0.1)
        assert y == pytest.approx(0.0, abs=0.1)

    def test_start_of_section(self):
        from section_tool.io.export import _distance_to_xy
        nodes = {"EW": np.array([[100.0, 200.0], [1100.0, 200.0]])}
        x, y = _distance_to_xy(nodes, "EW", 0.0)
        assert x == pytest.approx(100.0, abs=0.1)
        assert y == pytest.approx(200.0, abs=0.1)

    def test_end_of_section(self):
        from section_tool.io.export import _distance_to_xy
        nodes = {"EW": np.array([[0.0, 0.0], [1000.0, 500.0]])}
        x, y = _distance_to_xy(nodes, "EW", 1000.0 * np.sqrt(1.25))
        assert x == pytest.approx(1000.0, abs=1.0)
        assert y == pytest.approx(500.0, abs=1.0)

    def test_unknown_section_returns_none(self):
        from section_tool.io.export import _distance_to_xy
        x, y = _distance_to_xy({}, "NoSuch", 500.0)
        assert x is None
        assert y is None

    def test_dogleg_section(self):
        """L-shaped section: midpoint of second segment."""
        from section_tool.io.export import _distance_to_xy
        nodes = {"L": np.array([[0.0, 0.0], [5000.0, 0.0], [5000.0, 5000.0]])}
        # 7500 m along: 5000 on seg1 + 2500 on seg2
        x, y = _distance_to_xy(nodes, "L", 7500.0)
        assert x == pytest.approx(5000.0, abs=0.1)
        assert y == pytest.approx(2500.0, abs=0.1)
