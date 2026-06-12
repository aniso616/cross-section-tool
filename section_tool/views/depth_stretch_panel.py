"""DepthStretchPanel — the recommendation-first time→depth panel.

Replaces the old Method×Setting dialog.  One panel, no steps: an inventory strip,
an accented recommendation card (its knobs behind disclosure + the single Apply),
the other rungs ordered by groundedness (unlocked = selectable, locked = a door
with a reason + Import action), and a footer carrying the construction-metadata
caption (the SAME string the section shows) plus a residual line when the method
has one.  All reasoning lives in :class:`DepthStretchController`; this is the view.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from section_tool.core.depth_stretch_controller import DepthStretchController
from section_tool.views.depth_stretch_common import (
    VelocityModelSchematic, format_model_summary_html,
    _TEXT_BASE, _TEXT_MUTED)

_ACCENT = "#4A90D9"
_PRESENT = "#5FB85F"
_ABSENT = "#9AA0A6"


def _chip(text: str, present: bool) -> QLabel:
    lbl = QLabel(text)
    fg = _PRESENT if present else _ABSENT
    border = _PRESENT if present else "#5A5A5A"
    style = ("font-weight:bold;" if present else "font-style:italic;")
    lbl.setStyleSheet(
        f"QLabel {{ color:{fg}; border:1px solid {border}; border-radius:8px; "
        f"padding:2px 8px; {style} font-size:8pt; }}")
    return lbl


class DepthStretchPanel(QDialog):
    def __init__(self, state, on_apply=None, on_import=None, parent=None) -> None:
        super().__init__(parent)
        self._c = DepthStretchController(state)
        self._state = state
        self._on_apply = on_apply
        self._on_import = on_import            # callable(action_token) → launches importer
        self._selected = self._c.recommended_rung()
        self.setWindowTitle("Depth stretch")
        self.setMinimumWidth(720)
        self._knob_widgets: dict = {}
        self._build_ui()
        self._render()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        sec = getattr(self._state, "active_section", None)
        sec_name = getattr(sec, "name", "") if sec is not None else "(no active section)"
        header = QLabel(f"<b style='font-size:11pt;'>Depth stretch</b>"
                        f"<span style='color:{_TEXT_MUTED};'>  ·  {sec_name}</span>")
        outer.addWidget(header)

        # Inventory strip
        self._inv_row = QHBoxLayout()
        self._inv_row.setSpacing(6)
        inv_holder = QWidget(); inv_holder.setLayout(self._inv_row)
        outer.addWidget(inv_holder)

        # Non-blocking upgrade banner (a more-grounded method became available).
        self._upgrade_bar = QFrame()
        self._upgrade_bar.setStyleSheet(
            "QFrame { background:#3A3320; border:1px solid #C9A24A; border-radius:4px; }")
        ub = QHBoxLayout(self._upgrade_bar); ub.setContentsMargins(8, 3, 8, 3)
        self._upgrade_lbl = QLabel()
        self._upgrade_lbl.setStyleSheet("color:#E0C474; font-size:8pt; border:none;")
        ub.addWidget(self._upgrade_lbl, 1)
        self._upgrade_keep = QPushButton("Keep current")
        self._upgrade_keep.clicked.connect(self._keep_current)
        ub.addWidget(self._upgrade_keep)
        outer.addWidget(self._upgrade_bar)
        self._upgrade_bar.setVisible(False)

        cols = QHBoxLayout()
        outer.addLayout(cols, 1)
        left = QVBoxLayout()
        cols.addLayout(left, 3)

        # Recommendation / selected card
        self._card = QFrame()
        self._card.setFrameShape(QFrame.Shape.StyledPanel)
        self._card_layout = QVBoxLayout(self._card)
        left.addWidget(self._card)

        # Other methods list
        left.addWidget(QLabel(f"<span style='color:{_TEXT_MUTED};'>Other methods</span>"))
        self._list = QVBoxLayout()
        self._list.setSpacing(3)
        list_holder = QWidget(); list_holder.setLayout(self._list)
        left.addWidget(list_holder)
        left.addStretch()

        # Footer: caption + residuals (single-source caption)
        self._footer = QLabel()
        self._footer.setWordWrap(True)
        self._footer.setStyleSheet(f"color:{_TEXT_MUTED}; font-size:8pt;")
        left.addWidget(self._footer)

        # Right: live schematic
        cols.addSpacing(12)
        self._schematic = VelocityModelSchematic()
        cols.addWidget(self._schematic, 2)

        buttons = QDialogButtonBox()
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        outer.addWidget(buttons)

    # ------------------------------------------------------------------

    def _clear_layout(self, layout) -> None:
        """Tear a layout down completely — widgets AND nested layouts.

        The card mixes addWidget() with an addLayout() knob form; a non-recursive
        clear leaves the nested form (and its knob widgets) parented to the card,
        where they survive a rung swap and render under the new rung's knobs.
        Reparent before deleteLater so the widgets leave the display atomically
        (deleteLater alone is async — they'd briefly overlap)."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            else:
                child = item.layout()
                if child is not None:
                    self._clear_layout(child)
                    child.deleteLater()

    def _knobs(self) -> dict:
        out = {}
        for key, w in self._knob_widgets.items():
            if isinstance(w, QComboBox):
                out[key] = w.currentData()
            elif isinstance(w, QDoubleSpinBox):
                out[key] = w.value()
        # Map setting-derived TWT defaults
        if out.get("setting") == "marine":
            out["seafloor_twt_s"] = 0.4
        return out

    def _render(self) -> None:
        specs = {s.key: s for s in self._c.rung_specs()}
        rec = self._c.recommended_rung()

        # Inventory strip
        self._clear_layout(self._inv_row)
        for text, present in self._c.inventory_chips():
            self._inv_row.addWidget(_chip(text, present))
        self._inv_row.addStretch()

        # Upgrade banner (non-blocking; never restretches)
        up = self._c.upgrade_rung()
        if up is not None:
            label = {s.key: s.label for s in self._c.rung_specs()}.get(up, up)
            self._upgrade_lbl.setText(
                f"★ A more grounded method is available: <b>{label}</b> — "
                f"select it to apply, or keep the current method.")
            self._upgrade_bar.setVisible(True)
        else:
            self._upgrade_bar.setVisible(False)

        # Card for the selected rung
        self._render_card(specs[self._selected], rec)

        # Other-methods list (everything except the selected)
        self._clear_layout(self._list)
        for spec in self._c.rung_specs():
            if spec.key == self._selected:
                continue
            self._list.addWidget(self._render_row(spec, rec))

        self._render_footer()

    def _render_card(self, spec, rec: str) -> None:
        self._clear_layout(self._card_layout)
        self._knob_widgets = {}
        accent = _ACCENT if spec.key == rec else "#5A5A5A"
        self._card.setStyleSheet(
            f"QFrame {{ border:2px solid {accent}; border-radius:6px; }}")

        tag = "  ★ recommended" if spec.key == rec else ""
        title = QLabel(f"<b style='font-size:10pt;color:{_TEXT_BASE};'>{spec.label}</b>"
                       f"<span style='color:{_ACCENT};'>{tag}</span>")
        self._card_layout.addWidget(title)
        ol = QLabel(spec.one_liner); ol.setWordWrap(True)
        ol.setStyleSheet(f"color:{_TEXT_MUTED}; font-size:9pt;")
        self._card_layout.addWidget(ol)
        prov = QLabel(f"provenance: {spec.provenance_label}")
        prov.setStyleSheet(f"color:{_TEXT_MUTED}; font-style:italic; font-size:8pt;")
        self._card_layout.addWidget(prov)

        # Disclosure: this rung's knobs, in a single owner widget. Clearing the
        # card deletes this container whole, so no previous-rung knob can survive
        # a swap and overlap the new ones (the leak was a bare nested layout).
        self._knob_container = QWidget()
        self._knob_container.setObjectName("knob_container")
        form = QFormLayout(self._knob_container)
        form.setContentsMargins(0, 0, 0, 0)
        self._build_knobs(spec.key, form)
        self._card_layout.addWidget(self._knob_container)

        apply_btn = QPushButton(f"Apply  ·  {spec.label}")
        apply_btn.setStyleSheet(
            f"QPushButton {{ background:{_ACCENT}; color:white; font-weight:bold; "
            f"padding:5px 12px; border-radius:4px; }}")
        apply_btn.clicked.connect(self._apply)
        self._card_layout.addWidget(apply_btn)

    def _build_knobs(self, rung: str, form: QFormLayout) -> None:
        def spin(lo, hi, val, step, suffix, dec=0):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setSuffix(suffix); s.setDecimals(dec)
            s.valueChanged.connect(self._render_footer)
            return s

        setting = QComboBox()
        setting.addItem("Land", "onshore"); setting.addItem("Marine", "marine")
        setting.currentIndexChanged.connect(self._render_footer)
        form.addRow("Setting:", setting)
        self._knob_widgets["setting"] = setting

        if rung == "bulk":
            self._knob_widgets["bulk_v"] = spin(500, 6000, 2400, 50, " m/s")
            form.addRow("Bulk velocity:", self._knob_widgets["bulk_v"])
        elif rung == "average_vz":
            self._knob_widgets["v0"] = spin(500, 6000, 1800, 50, " m/s")
            self._knob_widgets["k"] = spin(0.0, 3.0, 0.6, 0.05, " s⁻¹", dec=2)
            form.addRow("V₀:", self._knob_widgets["v0"])
            form.addRow("k:", self._knob_widgets["k"])
        elif rung in ("checkshot", "sonic_checkshot", "sonic_anchors"):
            w = self._c._checkshot_well() or self._c._sonic_well()
            note = QLabel(f"well: {getattr(w, 'name', '—')}")
            note.setStyleSheet(f"color:{_TEXT_MUTED}; font-size:8pt;")
            form.addRow("Source:", note)

        # Stable handles for assertions / debugging: knob_<key>.
        for key, widget in self._knob_widgets.items():
            widget.setObjectName(f"knob_{key}")

    def _render_row(self, spec, rec: str) -> QWidget:
        row = QFrame()
        h = QHBoxLayout(row); h.setContentsMargins(6, 3, 6, 3)
        rec_tag = "  ★" if spec.key == rec else ""
        name = QLabel(f"{spec.label}{rec_tag}")
        if spec.unlocked:
            name.setStyleSheet(f"color:{_TEXT_BASE}; font-size:9pt;")
            h.addWidget(name)
            sub = QLabel(spec.one_liner)
            sub.setStyleSheet(f"color:{_TEXT_MUTED}; font-size:8pt;")
            h.addWidget(sub, 1)
            select = QPushButton("Select")
            select.clicked.connect(lambda _=False, k=spec.key: self._select(k))
            h.addWidget(select)
        else:
            name.setStyleSheet(f"color:{_ABSENT}; font-size:9pt; font-style:italic;")
            h.addWidget(name)
            reason = QLabel(f"— {spec.reason}")
            reason.setStyleSheet(f"color:{_ABSENT}; font-size:8pt; font-style:italic;")
            h.addWidget(reason, 1)
            if spec.import_action and self._on_import is not None:
                imp = QPushButton(spec.import_label)
                imp.clicked.connect(
                    lambda _=False, a=spec.import_action: self._do_import(a))
                h.addWidget(imp)
        return row

    def _render_footer(self) -> None:
        try:
            model = self._c.build_model(self._selected, **self._knobs())
        except Exception as exc:
            self._footer.setText(f"⚠ {exc}")
            self._schematic.set_model(None, 3.0, None)
            return
        strat = getattr(self._state.project, "strat_column", None)
        from section_tool.core.velocity_model import conversion_caption
        caption = conversion_caption(model) or "unconverted"
        residual = self._c.residual_summary(model)
        text = caption + (f"\n{residual}" if residual else "")
        self._footer.setText(text)
        max_twt = max((L.top_twt_s for L in model.layers), default=1.0) + 0.5
        self._schematic.set_model(model, max_twt, strat)

    # ------------------------------------------------------------------

    def _select(self, rung: str) -> None:
        self._selected = rung
        self._render()

    def _keep_current(self) -> None:
        self._c.keep_current()                 # record the deliberate keep
        if self._on_apply is not None:
            self._on_apply()                   # persist the ack
        self._render()                         # banner clears

    def _apply(self) -> None:
        try:
            self._c.apply(self._selected, **self._knobs())
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Apply failed", str(exc))
            return
        try:
            self._state._set_modified(True)
        except Exception:
            pass
        if self._on_apply is not None:
            self._on_apply()
        self._render()

    def _do_import(self, action_token: str) -> None:
        if self._on_import is None:
            return
        self._on_import(action_token)          # launches the Prompt-05 importer
        # Refresh inventory + recommendation after the import completes.
        self._selected = self._c.recommended_rung()
        self._render()
