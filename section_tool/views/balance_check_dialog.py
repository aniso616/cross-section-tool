"""Balance check dialog.

Accessible via Model ▸ Check Section Balance. Reports line lengths of
horizon picks and areas of section polygons for the active section.
"""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _pick_line_length(distances: np.ndarray, depths: np.ndarray) -> float:
    """Arc length of a horizon pick polyline (section-space units)."""
    if len(distances) < 2:
        return 0.0
    return float(np.hypot(np.diff(distances), np.diff(depths)).sum())


class BalanceCheckDialog(QDialog):
    """Balance check report for the active section.

    Shows horizon pick line lengths and polygon areas (both in km / km²).
    """

    def __init__(self, app_state, section, parent=None, snapshot=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Balance Check — {section.name}")
        self.setMinimumWidth(560)
        self._cmp_rows: list = []          # (result, kind) for the live flag refresh

        proj = app_state._project
        sec_length_m = section.total_length()
        sec_length_km = sec_length_m / 1000.0

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Section summary ───────────────────────────────────────────────
        hdr = QLabel(f"<b>Section:</b> {section.name}   "
                     f"<b>Length:</b> {sec_length_km:.3f} km")
        hdr.setTextFormat(Qt.RichText)
        layout.addWidget(hdr)

        # ── Horizon line lengths ──────────────────────────────────────────
        hz_box = QGroupBox("Horizon Line Lengths")
        hz_layout = QVBoxLayout(hz_box)

        picks = [hp for hp in proj.horizon_picks if hp.n_picks_for_section(section.name) >= 2]
        if picks:
            hz_table = QTableWidget(len(picks), 4)
            hz_table.setHorizontalHeaderLabels(["Name", "Picks", "Length (km)", "Extent (km)"])
            hz_table.horizontalHeader().setStretchLastSection(True)
            hz_table.verticalHeader().setVisible(False)
            hz_table.setEditTriggers(QTableWidget.NoEditTriggers)
            hz_table.setSelectionMode(QTableWidget.SingleSelection)

            for row, hp in enumerate(picks):
                dist, depth = hp.picks_for_section(section.name)
                length_km = _pick_line_length(dist, depth) / 1000.0
                d_min_km = float(dist.min()) / 1000.0
                d_max_km = float(dist.max()) / 1000.0
                hz_table.setItem(row, 0, _cell(hp.name or "(unnamed)"))
                hz_table.setItem(row, 1, _cell(str(len(dist)), align=Qt.AlignRight | Qt.AlignVCenter))
                hz_table.setItem(row, 2, _cell(f"{length_km:.3f}", align=Qt.AlignRight | Qt.AlignVCenter))
                hz_table.setItem(row, 3, _cell(f"{d_min_km:.2f} – {d_max_km:.2f}", align=Qt.AlignRight | Qt.AlignVCenter))

            hz_table.resizeColumnsToContents()
            hz_layout.addWidget(hz_table)
        else:
            hz_layout.addWidget(QLabel("No horizon picks on this section."))

        layout.addWidget(hz_box)

        # ── Polygon areas ─────────────────────────────────────────────────
        poly_box = QGroupBox("Polygon Areas")
        poly_layout = QVBoxLayout(poly_box)

        section_polys = [p for p in proj.polygons if p.section_name == section.name]
        if section_polys:
            poly_table = QTableWidget(len(section_polys), 4)
            poly_table.setHorizontalHeaderLabels(["Name", "Formation", "Area (km²)", "Vertices"])
            poly_table.horizontalHeader().setStretchLastSection(True)
            poly_table.verticalHeader().setVisible(False)
            poly_table.setEditTriggers(QTableWidget.NoEditTriggers)
            poly_table.setSelectionMode(QTableWidget.SingleSelection)

            total_area_m2 = 0.0
            for row, poly in enumerate(section_polys):
                area_km2 = poly.area / 1e6
                total_area_m2 += poly.area
                poly_table.setItem(row, 0, _cell(poly.name or "(unnamed)"))
                poly_table.setItem(row, 1, _cell(poly.formation or ""))
                poly_table.setItem(row, 2, _cell(f"{area_km2:.4f}", align=Qt.AlignRight | Qt.AlignVCenter))
                poly_table.setItem(row, 3, _cell(str(poly.n_vertices), align=Qt.AlignRight | Qt.AlignVCenter))

            poly_table.resizeColumnsToContents()
            poly_layout.addWidget(poly_table)

            total_area_km2 = total_area_m2 / 1e6
            summary_row = QHBoxLayout()
            summary_row.addWidget(QLabel(f"<b>Total area:</b> {total_area_km2:.4f} km²"))
            if sec_length_km > 0:
                depth_km = total_area_km2 / sec_length_km
                summary_row.addWidget(QLabel(
                    f"   <b>Depth to detachment:</b> {depth_km:.3f} km"
                    f"<span style='color:#777; font-size:small;'> (area ÷ length)</span>"
                ))
            summary_row.addStretch()
            poly_layout.addLayout(summary_row)
        else:
            poly_layout.addWidget(QLabel("No polygons on this section."))

        layout.addWidget(poly_box)

        # ── Deformed vs restored comparison (when a snapshot is present) ───
        # Graceful degradation: with no snapshot the single-section report above
        # is the whole dialog.
        if snapshot is not None:
            self._add_comparison(layout, app_state, section, snapshot)

        # ── Close button ─────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Deformed-vs-restored (Dahlstrom)
    # ------------------------------------------------------------------

    def _add_comparison(self, layout, app_state, section, snapshot) -> None:
        import numpy as np
        from section_tool.core import balance as B

        proj = app_state._project
        sec = section.name

        def line_pts(pick):
            d, z = pick.picks_for_section(sec)
            return np.column_stack([d, z]) if len(d) >= 2 else None

        # Compare the LIVE (deformed) section against the RESTORED state — apply the
        # current restoration step's algorithm to the snapshot. With no algorithm
        # step active, compare against the raw captured baseline (drift since
        # capture). Dahlstrom: areas/lengths conserve between deformed and restored.
        restored = snapshot
        seq = getattr(app_state, "restoration_sequence", None)
        if seq is not None and seq.events and seq.current_step >= 1:
            event = seq.events[seq.current_step - 1]
            if getattr(event, "algorithm", "none") not in ("none", None):
                from section_tool.core import kinematics as _K
                try:
                    restored = _K.restore_snapshot(
                        snapshot, event, section_name=sec,
                        reference_lines=proj.reference_lines)
                except Exception:
                    restored = snapshot

        # Beds + polygons keyed by UUID; preserved-UUID snapshots (Step 3) pair
        # restored↔deformed by equality.
        def_lines, names = {}, {}
        for pick in list(proj.horizon_picks) + list(proj.fault_picks):
            pts = line_pts(pick)
            if pts is not None:
                def_lines[pick.uuid] = pts
                names[pick.uuid] = pick.name or "(unnamed)"
        res_lines = {}
        for pick in list(restored.horizons) + list(restored.faults):
            pts = line_pts(pick)
            if pts is not None:
                res_lines[pick.uuid] = pts

        def_polys = {p.uuid: p.vertices for p in proj.polygons
                     if getattr(p, "section_name", "") in ("", sec)}
        res_polys = {p.uuid: p.vertices for p in restored.polygons}
        poly_names = {p.uuid: (p.name or "(unnamed)") for p in proj.polygons}

        line_results = B.line_length_balance(def_lines, res_lines, names)
        area_results = [B.area_balance(def_polys[k], res_polys[k],
                                       name=poly_names.get(k, k))
                        for k in def_polys if k in res_polys]

        box = QGroupBox("Deformed vs Restored — Dahlstrom balance")
        box_layout = QVBoxLayout(box)

        if not line_results and not area_results:
            box_layout.addWidget(QLabel(
                "No elements pair between the current section and the snapshot."))
            layout.addWidget(box)
            return

        # User-adjustable discrepancy threshold (surfaced, not buried).
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Discrepancy threshold:"))
        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(0.1, 100.0)
        self._tol_spin.setDecimals(1)
        self._tol_spin.setSuffix(" %")
        self._tol_spin.setValue(B.DEFAULT_AREA_TOLERANCE * 100.0)
        self._tol_spin.valueChanged.connect(self._refresh_balance_flags)
        tol_row.addWidget(self._tol_spin)
        tol_row.addStretch()
        box_layout.addLayout(tol_row)

        self._cmp_table = QTableWidget(0, 6)
        self._cmp_table.setHorizontalHeaderLabels(
            ["Element", "Type", "Deformed", "Restored", "Discrepancy", "Balanced?"])
        self._cmp_table.horizontalHeader().setStretchLastSection(True)
        self._cmp_table.verticalHeader().setVisible(False)
        self._cmp_table.setEditTriggers(QTableWidget.NoEditTriggers)

        for ab in area_results:
            self._add_cmp_row(ab, kind="area")
        for ll in line_results:
            self._add_cmp_row(ll, kind="line")

        self._cmp_table.resizeColumnsToContents()
        self._cmp_table.horizontalHeader().setStretchLastSection(True)
        box_layout.addWidget(self._cmp_table)

        # Depth to detachment — excess area ÷ shortening, showing its work.
        total_excess = sum(abs(B.polygon_area(v)) for v in def_polys.values())
        total_shortening = sum(abs(ll.shortening) for ll in line_results)
        if total_excess > 0 and total_shortening > 1e-6:
            dd = B.depth_to_detachment(total_excess, total_shortening)
            lbl = QLabel(f"<b>Depth to detachment:</b> {dd.depth:,.0f} m"
                         f"<br><span style='color:#888; font-size:small;'>"
                         f"{dd.explain()} (excess area = Σ|deformed polygon area|, "
                         f"shortening = Σ|restored − deformed length|)</span>")
            lbl.setTextFormat(Qt.RichText)
            box_layout.addWidget(lbl)

        note = QLabel("<span style='color:#888; font-size:small;'>"
                      "Discrepancy = |deformed − restored| / |restored|. Areas "
                      "(Dahlstrom plane-strain) and bed lengths should be conserved; "
                      "rows over threshold are flagged. Hover a row to see its inputs."
                      "</span>")
        note.setTextFormat(Qt.RichText)
        note.setWordWrap(True)
        box_layout.addWidget(note)

        layout.addWidget(box)
        self._refresh_balance_flags()

    def _add_cmp_row(self, result, *, kind: str) -> None:
        row = self._cmp_table.rowCount()
        self._cmp_table.insertRow(row)
        if kind == "area":
            deformed, restored = result.deformed_area, result.restored_area
            unit, type_label = "km²", "Polygon area"
            deformed_s = f"{deformed / 1e6:.4f}"
            restored_s = f"{restored / 1e6:.4f}"
            tip = (f"area_balance: |{deformed:,.0f} − {restored:,.0f}| / "
                   f"{restored:,.0f} m²")
        else:
            deformed, restored = result.deformed_length, result.restored_length
            unit, type_label = "km", "Bed length"
            deformed_s = f"{deformed / 1000.0:.3f}"
            restored_s = f"{restored / 1000.0:.3f}"
            tip = (f"line-length: deformed {deformed:,.1f} m, restored "
                   f"{restored:,.1f} m, shortening = {result.shortening:,.1f} m")
        disc = result.discrepancy
        disc_s = "∞" if disc == float("inf") else f"{disc * 100:.2f} %"
        cells = [result.name, type_label, f"{deformed_s} {unit}",
                 f"{restored_s} {unit}", disc_s, ""]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            if col in (2, 3, 4):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item.setToolTip(tip)
            self._cmp_table.setItem(row, col, item)
        self._cmp_rows.append((result, row))

    def _refresh_balance_flags(self) -> None:
        """Recolour the Balanced? column against the current threshold."""
        if not getattr(self, "_cmp_table", None):
            return
        tol = self._tol_spin.value() / 100.0
        for result, row in self._cmp_rows:
            balanced = result.is_balanced(tol)
            item = self._cmp_table.item(row, 5)
            if item is None:
                continue
            item.setText("✓ balanced" if balanced else "✗ over threshold")
            item.setForeground(QColor("#4caf50") if balanced else QColor("#ff6060"))


def _cell(text: str, align: Qt.Alignment = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    return item
