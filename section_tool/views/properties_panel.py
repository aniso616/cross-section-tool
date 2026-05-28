"""Phase 3 — Context-sensitive Properties panel (QDockWidget)."""
from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDockWidget,
    QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from section_tool.app_state import AppState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep_label(text: str) -> QLabel:
    lbl = QLabel(text)
    font = QFont()
    font.setBold(True)
    font.setPointSize(9)
    lbl.setFont(font)
    lbl.setStyleSheet(
        "color: #CCCCCC; margin-top: 6px; margin-bottom: 1px; "
        "border-bottom: 1px solid #444; padding-bottom: 2px;"
    )
    return lbl


def _val_label(text: str) -> QLabel:
    if not text or not str(text).strip():
        lbl = QLabel("—")
        lbl.setStyleSheet("color: #666; font-style: italic; font-size: 8pt;")
    else:
        lbl = QLabel(str(text))
        lbl.setStyleSheet("color: #CCCCCC; font-size: 8pt;")
        lbl.setWordWrap(False)
    return lbl


def _grid():
    """Return (QGridLayout, add_row_fn).

    add_row_fn(label_text, value) appends one property row. *value* may be a
    str (rendered as a read-only QLabel) or any QWidget. Labels are fixed at
    70px wide with no word-wrap so they never overlap the value column.
    """
    g = QGridLayout()
    g.setContentsMargins(4, 4, 4, 4)
    g.setHorizontalSpacing(8)
    g.setVerticalSpacing(6)
    g.setColumnStretch(0, 0)          # label column — fixed
    g.setColumnStretch(1, 1)          # value column — expands
    g.setColumnMinimumWidth(0, 70)

    _row = [0]

    def add_row(label_text: str, value) -> None:
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #888888; font-size: 8pt;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setFixedWidth(70)
        lbl.setWordWrap(False)
        g.addWidget(lbl, _row[0], 0,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if isinstance(value, str):
            val = QLabel(str(value) if value else "—")
            val.setStyleSheet("color: #CCCCCC; font-size: 8pt;")
            val.setWordWrap(False)
            g.addWidget(val, _row[0], 1,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        else:
            g.addWidget(value, _row[0], 1)
        _row[0] += 1

    return g, add_row


def _swatch(color: str) -> QLabel:
    s = QLabel()
    s.setFixedSize(18, 14)
    s.setStyleSheet(f"background:{color}; border:1px solid #888; border-radius:2px;")
    return s


# ---------------------------------------------------------------------------
# PropertiesPanel
# ---------------------------------------------------------------------------

class PropertiesPanel(QDockWidget):
    """Dockable Properties panel — content changes with selection.

    Listens to AppState signals to know what is currently selected.
    """

    # Emitted when user edits a property (so views can update)
    property_changed = Signal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__("Properties", parent)
        self._state = state
        self._selected_node: tuple[str, int, int] | None = None
        self._selected_object: tuple[str, int] | None = None

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setMinimumWidth(220)
        self.setMaximumWidth(300)

        # Scrollable inner area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._inner = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(6, 6, 6, 6)
        scroll.setWidget(self._inner)
        self.setWidget(scroll)
        self.setMinimumHeight(180)
        self.setMinimumWidth(200)

        self._rebuilding = False   # re-entry guard
        self._connect_signals()
        self._rebuild()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        s = self._state
        s.active_section_changed.connect(lambda _: self._rebuild())
        s.active_pick_target_changed.connect(lambda *_: self._rebuild())
        s.project_changed.connect(self._rebuild)
        s.horizon_pick_modified.connect(lambda *_: self._rebuild())
        s.fault_pick_modified.connect(lambda *_: self._rebuild())
        s.section_modified.connect(lambda *_: self._rebuild())
        s.polygon_modified.connect(lambda *_: self._rebuild())

    def set_selected_node(
        self, node: tuple[str, int, int] | None
    ) -> None:
        self._selected_node = node
        self._rebuild()

    def set_selected_object(self, category: str, index: int) -> None:
        self._selected_object = (category, index)
        self._rebuild()

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _rebuild(self, *_) -> None:
        """Clear and rebuild content based on current selection."""
        if self._rebuilding:
            return
        self._rebuilding = True
        try:
            self._do_rebuild()
        finally:
            self._rebuilding = False

    def _clear_layout(self, layout) -> None:
        """Recursively remove and schedule deletion of all items in *layout*."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.blockSignals(True)   # prevent editingFinished on focus loss
                w.deleteLater()
            else:
                child = item.layout()
                if child is not None:
                    self._clear_layout(child)

    def _do_rebuild(self) -> None:
        # Clear — recursively remove widgets AND child layouts so that grids
        # added via addLayout() are also cleaned up (not just top-level widgets).
        self._clear_layout(self._layout)

        # Selected node has highest priority
        if self._selected_node is not None:
            cat, oi, pi = self._selected_node
            proj = self._state.project
            picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
            if oi < len(picks) and pi < picks[oi].n_picks:
                self._build_node(cat, oi, pi, picks[oi])
                return
            self._selected_node = None

        # Selected polygon (from tree click)
        if self._selected_object is not None:
            so_cat, so_idx = self._selected_object
            if so_cat == "Polygons":
                proj = self._state.project
                if so_idx < len(proj.polygons):
                    self._build_polygon(so_idx, proj.polygons[so_idx])
                    return
            self._selected_object = None

        # Active pick target (horizon/fault selected)
        cat = self._state.active_pick_category
        idx = self._state.active_pick_index
        if cat == "Horizons" and idx is not None:
            proj = self._state.project
            if idx < len(proj.horizon_picks):
                self._build_horizon(idx, proj.horizon_picks[idx])
                return
        if cat == "Faults" and idx is not None:
            proj = self._state.project
            if idx < len(proj.fault_picks):
                self._build_fault(idx, proj.fault_picks[idx])
                return

        # Active section
        sec = self._state.active_section
        if sec is not None:
            self._build_section(sec)
            return

        # Default
        self._build_default()

    # ------------------------------------------------------------------
    # Content builders
    # ------------------------------------------------------------------

    def _build_default(self) -> None:
        proj = self._state.project
        self._layout.addWidget(_sep_label("Project"))
        grid, ar = _grid()
        ar("Name:",     proj.name or "Untitled")
        ar("CRS:",      f"EPSG:{proj.crs_epsg}")
        ar("Sections:", str(len(proj.sections)))
        ar("Horizons:", str(len(proj.horizon_picks)))
        ar("Faults:",   str(len(proj.fault_picks)))
        ar("Polygons:", str(len(proj.polygons)))
        self._layout.addLayout(grid)
        self._layout.addStretch()

    def _build_section(self, sec) -> None:
        self._layout.addWidget(_sep_label("Section"))
        grid, ar = _grid()

        name_ed = QLineEdit(sec.name)
        name_ed.returnPressed.connect(lambda: self._commit_section_name(name_ed.text()))
        name_ed.editingFinished.connect(lambda: self._commit_section_name(name_ed.text()))
        ar("Name:", name_ed)

        try:
            idx = self._state.project.sections.index(sec)
            ar("Index:", str(idx))
        except ValueError:
            pass
        ar("Nodes:",  str(sec.n_nodes))
        ar("Length:", f"{sec.total_length():.1f} {sec.depth_units}")

        try:
            azs = sec.segment_azimuths()
            az_str = f"{azs[0]:.1f}°" if len(azs) == 1 else f"{azs[0]:.1f}°…{azs[-1]:.1f}°"
        except Exception:
            az_str = "—"
        ar("Azimuth:", az_str)
        ar("Domain:",  sec.depth_domain)
        ar("Units:",   sec.depth_units)

        ve_spin = QDoubleSpinBox()
        ve_spin.setRange(0.5, 20.0); ve_spin.setSingleStep(0.5); ve_spin.setDecimals(1)
        ve_spin.blockSignals(True); ve_spin.setValue(sec.vertical_exaggeration)
        ve_spin.blockSignals(False)
        ve_spin.valueChanged.connect(lambda v: self._commit_section_ve(v))
        ar("VE:", ve_spin)

        self._layout.addLayout(grid)
        self._layout.addStretch()

    def _build_horizon(self, idx: int, hp) -> None:
        self._layout.addWidget(_sep_label("Horizon"))
        grid, ar = _grid()

        name_ed = QLineEdit(hp.name)
        name_ed.editingFinished.connect(
            lambda: self._commit_pick_name("Horizons", idx, name_ed.text()))
        ar("Name:", name_ed)

        from section_tool.views.horizon_dialog import CONTACT_TYPES
        ct_combo = QComboBox()
        for ct in CONTACT_TYPES:
            ct_combo.addItem(ct.replace("_", " ").title(), ct)
        ci = CONTACT_TYPES.index(hp.contact_type) if hp.contact_type in CONTACT_TYPES else 0
        ct_combo.blockSignals(True); ct_combo.setCurrentIndex(ci); ct_combo.blockSignals(False)
        ct_combo.currentIndexChanged.connect(
            lambda _: self._commit_pick_ct("Horizons", idx, ct_combo.currentData()))
        ar("Type:", ct_combo)

        ar("Color:", self._make_color_row(hp.color,
            lambda c: self._commit_pick_color("Horizons", idx, c)))

        lw = QDoubleSpinBox()
        lw.setRange(0.5, 6.0); lw.setSingleStep(0.5); lw.setDecimals(1)
        lw.blockSignals(True); lw.setValue(getattr(hp, "line_width", 1.5)); lw.blockSignals(False)
        lw.valueChanged.connect(lambda v: self._commit_pick_lw("Horizons", idx, v))
        ar("Width:", lw)

        fa_ed = QLineEdit(getattr(hp, "formation_above", ""))
        fa_ed.editingFinished.connect(
            lambda: self._commit_pick_formation("Horizons", idx, "above", fa_ed.text()))
        ar("Fm above:", fa_ed)

        fb_ed = QLineEdit(getattr(hp, "formation_below", ""))
        fb_ed.editingFinished.connect(
            lambda: self._commit_pick_formation("Horizons", idx, "below", fb_ed.text()))
        ar("Fm below:", fb_ed)

        n_secs = len({str(s) for s in hp._section_names if s != ""}) if hp.n_picks else 0
        ar("Picks:", f"{hp.n_picks} on {n_secs} section(s)")

        self._layout.addLayout(grid)
        self._layout.addStretch()

    def _build_fault(self, idx: int, fp) -> None:
        self._layout.addWidget(_sep_label("Fault"))
        grid, ar = _grid()

        name_ed = QLineEdit(fp.name)
        name_ed.editingFinished.connect(
            lambda: self._commit_pick_name("Faults", idx, name_ed.text()))
        ar("Name:", name_ed)

        from section_tool.views.fault_dialog import FAULT_TYPES
        ft_combo = QComboBox()
        for ft in FAULT_TYPES:
            ft_combo.addItem(ft.replace("_", " ").title(), ft)
        fi = FAULT_TYPES.index(fp.fault_type) if fp.fault_type in FAULT_TYPES else 0
        ft_combo.blockSignals(True); ft_combo.setCurrentIndex(fi); ft_combo.blockSignals(False)
        ft_combo.currentIndexChanged.connect(
            lambda _: self._commit_pick_ft("Faults", idx, ft_combo.currentData()))
        ar("Type:", ft_combo)

        dd_combo = QComboBox()
        dd_combo.addItems(["Right", "Left"])
        _di = 0 if getattr(fp, "dip_direction", "right") == "right" else 1
        dd_combo.blockSignals(True); dd_combo.setCurrentIndex(_di); dd_combo.blockSignals(False)
        dd_combo.currentIndexChanged.connect(
            lambda i: self._commit_pick_dd("Faults", idx,
                                            "right" if i == 0 else "left"))
        ar("Dip dir:", dd_combo)

        ar("Color:", self._make_color_row(fp.color,
            lambda c: self._commit_pick_color("Faults", idx, c)))

        lw = QDoubleSpinBox()
        lw.setRange(0.5, 6.0); lw.setSingleStep(0.5); lw.setDecimals(1)
        lw.blockSignals(True); lw.setValue(getattr(fp, "line_width", 1.5)); lw.blockSignals(False)
        lw.valueChanged.connect(lambda v: self._commit_pick_lw("Faults", idx, v))
        ar("Width:", lw)

        n_secs = len({str(s) for s in fp._section_names if s != ""}) if fp.n_picks else 0
        ar("Picks:", f"{fp.n_picks} on {n_secs} section(s)")

        self._layout.addLayout(grid)
        self._layout.addStretch()

    def _build_polygon(self, idx: int, poly) -> None:
        self._layout.addWidget(_sep_label(f"Polygon: {poly.name or 'Unnamed'}"))
        grid, ar = _grid()
        ar("Vertices:", str(poly.n_vertices))
        if getattr(poly, "formation", ""):
            ar("Formation:", poly.formation)
        self._layout.addLayout(grid)

        self._layout.addWidget(_sep_label("Fill"))
        fill_grid, far = _grid()

        alpha_row = QWidget()
        alpha_hb = QHBoxLayout(alpha_row)
        alpha_hb.setContentsMargins(0, 0, 0, 0)
        alpha_hb.setSpacing(6)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(round(poly.fill_alpha * 100))
        alpha_lbl = QLabel(f"{round(poly.fill_alpha * 100)}%")
        alpha_lbl.setFixedWidth(34)
        alpha_lbl.setStyleSheet("color: #CCCCCC; font-size: 8pt;")

        def _on_alpha(v: int) -> None:
            alpha_lbl.setText(f"{v}%")
            self._commit_polygon_alpha(idx, v / 100.0)

        slider.valueChanged.connect(_on_alpha)
        alpha_hb.addWidget(slider, 1)
        alpha_hb.addWidget(alpha_lbl)
        far("Alpha:", alpha_row)
        self._layout.addLayout(fill_grid)
        self._layout.addStretch()

    def _commit_polygon_alpha(self, idx: int, alpha: float) -> None:
        proj = self._state.project
        if idx >= len(proj.polygons):
            return
        poly = copy.deepcopy(proj.polygons[idx])
        poly.fill_alpha = float(np.clip(alpha, 0.0, 1.0))
        self._state.update_polygon(idx, poly)

    def _build_node(self, cat: str, oi: int, pi: int, hp) -> None:
        self._layout.addWidget(_sep_label(f"Node: {hp.name or cat[:-1]}"))
        grid, ar = _grid()

        d = float(hp._distances[pi]); z = float(hp._depths[pi])

        d_ed = QLineEdit(f"{d:.2f}")
        d_ed.editingFinished.connect(
            lambda: self._commit_node_coord(cat, oi, pi, "d", d_ed.text()))
        ar("Dist (m):", d_ed)

        z_ed = QLineEdit(f"{z:.2f}")
        z_ed.editingFinished.connect(
            lambda: self._commit_node_coord(cat, oi, pi, "z", z_ed.text()))
        ar("Depth (m):", z_ed)

        sec = self._state.active_section
        if sec is not None:
            try:
                mx, my = sec.section_to_map(d)
                ar("Map X:", f"{mx:.1f}")
                ar("Map Y:", f"{my:.1f}")
            except Exception:
                pass

        conf = float(hp._confidence[pi]) if len(hp._confidence) > pi else 1.0
        conf_spin = QDoubleSpinBox()
        conf_spin.setRange(0, 1); conf_spin.setSingleStep(0.1); conf_spin.setDecimals(2)
        conf_spin.blockSignals(True); conf_spin.setValue(conf); conf_spin.blockSignals(False)
        conf_spin.valueChanged.connect(
            lambda v: self._commit_node_meta(cat, oi, pi, "confidence", v))
        ar("Confidence:", conf_spin)

        qual_vals = ["picked", "interpolated", "projected", "inferred"]
        qual_combo = QComboBox()
        qual_combo.addItems([q.title() for q in qual_vals])
        cur_q = str(hp._quality[pi]) if len(hp._quality) > pi else "picked"
        qi = qual_vals.index(cur_q) if cur_q in qual_vals else 0
        qual_combo.blockSignals(True); qual_combo.setCurrentIndex(qi); qual_combo.blockSignals(False)
        qual_combo.currentIndexChanged.connect(
            lambda i: self._commit_node_meta(cat, oi, pi, "quality", qual_vals[i]))
        ar("Quality:", qual_combo)

        note_ed = QLineEdit(str(hp._note[pi]) if len(hp._note) > pi else "")
        note_ed.editingFinished.connect(
            lambda: self._commit_node_meta(cat, oi, pi, "note", note_ed.text()))
        ar("Note:", note_ed)

        self._layout.addLayout(grid)
        self._layout.addStretch()

    # ------------------------------------------------------------------
    # Commit helpers
    # ------------------------------------------------------------------

    def _make_color_row(self, color: str, on_change) -> QWidget:
        w = QWidget()
        hb = QHBoxLayout(w); hb.setContentsMargins(0, 0, 0, 0)
        swatch = _swatch(color)
        hb.addWidget(swatch)
        btn = QPushButton("…")
        btn.setFixedWidth(24)

        def _pick():
            from PySide6.QtWidgets import QColorDialog
            c = QColorDialog.getColor(QColor(color), self)
            if c.isValid():
                col = c.name()
                swatch.setStyleSheet(
                    f"background:{col}; border:1px solid #888; border-radius:2px;")
                on_change(col)
        btn.clicked.connect(_pick)
        hb.addWidget(btn); hb.addStretch()
        return w

    def _commit_section_name(self, name: str) -> None:
        sec = self._state.active_section
        if sec is None: return
        idx = self._state.project.sections.index(sec)
        s2 = copy.deepcopy(sec)
        s2.name = name.strip() or sec.name
        self._state.update_section(idx, s2)

    def _commit_section_ve(self, ve: float) -> None:
        sec = self._state.active_section
        if sec is None:
            return
        if abs(getattr(sec, "vertical_exaggeration", 1.0) - ve) < 0.001:
            return
        idx = self._state.project.sections.index(sec)
        s2 = copy.deepcopy(sec)
        s2.vertical_exaggeration = ve
        # Block panel rebuild so the spinbox isn't deleted while valueChanged is in flight.
        # The section_view handles the re-render via its own section_modified connection.
        self._rebuilding = True
        try:
            self._state.update_section(idx, s2)
        finally:
            self._rebuilding = False

    def _commit_pick_name(self, cat, idx, name):
        self._pick_op(cat, idx, lambda h: setattr(h, "name", name.strip()))

    def _commit_pick_ct(self, cat, idx, ct):
        self._pick_op(cat, idx, lambda h: setattr(h, "contact_type", ct))

    def _commit_pick_ft(self, cat, idx, ft):
        self._pick_op(cat, idx, lambda h: setattr(h, "fault_type", ft))

    def _commit_pick_dd(self, cat, idx, dd):
        self._pick_op(cat, idx, lambda h: setattr(h, "dip_direction", dd))

    def _commit_pick_color(self, cat, idx, color):
        self._pick_op(cat, idx, lambda h: setattr(h, "color", color))

    def _commit_pick_lw(self, cat, idx, lw):
        self._pick_op(cat, idx, lambda h: setattr(h, "line_width", lw))

    def _commit_pick_formation(self, cat, idx, side, name):
        attr = "formation_above" if side == "above" else "formation_below"
        self._pick_op(cat, idx, lambda h: setattr(h, attr, name))

    def _pick_op(self, cat, idx, mutate_fn) -> None:
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if idx >= len(picks): return
        h2 = copy.deepcopy(picks[idx])
        mutate_fn(h2)
        if cat == "Horizons":
            self._state.update_horizon_pick(idx, h2)
        else:
            self._state.update_fault_pick(idx, h2)

    def _commit_node_coord(self, cat, oi, pi, axis, text) -> None:
        try:
            val = float(text)
        except ValueError:
            return
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if oi >= len(picks): return
        h2 = copy.deepcopy(picks[oi])
        if pi >= h2.n_picks: return
        if axis == "d":
            h2._distances[pi] = val
        else:
            h2._depths[pi] = val
        # Re-sort
        order = np.argsort(h2._distances, kind="stable")
        for attr in ("_distances", "_depths", "_section_names",
                     "_confidence", "_quality", "_note"):
            arr = getattr(h2, attr, None)
            if arr is not None and len(arr) == len(order):
                setattr(h2, attr, arr[order])
        if cat == "Horizons":
            self._state.update_horizon_pick(oi, h2)
        else:
            self._state.update_fault_pick(oi, h2)

    def _commit_node_meta(self, cat, oi, pi, field, value) -> None:
        proj = self._state.project
        picks = proj.horizon_picks if cat == "Horizons" else proj.fault_picks
        if oi >= len(picks): return
        h2 = copy.deepcopy(picks[oi])
        if pi >= h2.n_picks: return
        arr = getattr(h2, f"_{field}", None)
        if arr is not None and len(arr) > pi:
            arr[pi] = value
        if cat == "Horizons":
            self._state.update_horizon_pick(oi, h2)
        else:
            self._state.update_fault_pick(oi, h2)
