"""High-level construction tool objects for dip-constrained, parallel-offset,
and kink-band horizon drawing.

Each tool is a small state machine that accepts click coordinates and returns
a new :class:`~section_tool.core.surfaces.HorizonPick` when the tool
sequence is complete.  Tools are stateful — call :meth:`reset` when
cancelling or switching away.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from section_tool.core.construction import (
    DipConstrainedRule,
    KinkBandRule,
    ParallelToBedRule,
)
from section_tool.core.surfaces import HorizonPick


class DipConstrainedTool:
    """Two-click tool: anchor point → extent point.

    Depth at the extent point is constrained by the configured dip so the
    resulting segment has exactly *dip_deg* apparent dip in the section plane.
    Positive *dip_deg* → dipping to the right (increasing distance).
    """

    def __init__(self, dip_deg: float = 0.0) -> None:
        self.dip_deg: float = float(dip_deg)
        self._anchor: tuple[float, float] | None = None

    # ------------------------------------------------------------------

    @property
    def state(self) -> Literal["idle", "anchor_set"]:
        return "idle" if self._anchor is None else "anchor_set"

    @property
    def anchor(self) -> tuple[float, float] | None:
        return self._anchor

    def reset(self) -> None:
        self._anchor = None

    def hint(self) -> str:
        if self._anchor is None:
            return "Click to set anchor point (Dip-constrained)"
        return f"Click to set extent  [dip = {self.dip_deg:.1f}°]"

    def constrain_depth(self, d: float) -> float | None:
        """Return depth at *d* constrained by dip from anchor, or None if idle."""
        if self._anchor is None:
            return None
        d0, z0 = self._anchor
        return z0 + (d - d0) * math.tan(math.radians(self.dip_deg))

    def handle_click(
        self, d: float, z: float, sec_name: str, name: str = ""
    ) -> HorizonPick | None:
        """Process one click.

        First call → stores anchor, returns None.
        Second call → creates and returns the constrained HorizonPick.
        """
        if self._anchor is None:
            self._anchor = (d, z)
            return None

        d0, z0 = self._anchor
        self._anchor = None
        z_end = z0 + (d - d0) * math.tan(math.radians(self.dip_deg))

        hp = HorizonPick(
            distances=[d0, d],
            depths=[z0, z_end],
            name=name or "Dip-constrained horizon",
            section_names=[sec_name, sec_name],
        )
        hp.construction_rule = DipConstrainedRule(dip_deg=self.dip_deg)
        return hp


class ParallelOffsetTool:
    """Two-click tool: click reference horizon → click to place offset copy.

    The vertical offset is measured at the placement click position (depth
    of cursor minus depth of reference at the same distance).
    """

    def __init__(self) -> None:
        self._ref_hp: HorizonPick | None   = None
        self._ref_sec_name: str            = ""

    # ------------------------------------------------------------------

    @property
    def state(self) -> Literal["idle", "ref_selected"]:
        return "idle" if self._ref_hp is None else "ref_selected"

    @property
    def reference(self) -> HorizonPick | None:
        return self._ref_hp

    def reset(self) -> None:
        self._ref_hp      = None
        self._ref_sec_name = ""

    def hint(self) -> str:
        if self._ref_hp is None:
            return "Click on a reference horizon (Parallel Offset)"
        return "Click to place the offset copy"

    def set_reference(self, hp: HorizonPick, sec_name: str) -> None:
        self._ref_hp       = hp
        self._ref_sec_name = sec_name

    def offset_at(self, d: float, z: float, sec_name: str) -> float | None:
        """Return vertical offset (z_cursor − z_ref at d) for live preview."""
        if self._ref_hp is None:
            return None
        si = self._ref_hp.section_indices(sec_name)
        if len(si) < 2:
            return None
        d_ref = self._ref_hp._distances[si]
        z_ref = self._ref_hp._depths[si]
        z_at_d = float(np.interp(d, d_ref, z_ref,
                                  left=float(z_ref[0]), right=float(z_ref[-1])))
        return z - z_at_d

    def handle_placement(
        self, d: float, z: float, sec_name: str, name: str = ""
    ) -> HorizonPick | None:
        """Create the offset HorizonPick at the cursor position."""
        if self._ref_hp is None:
            return None
        si = self._ref_hp.section_indices(sec_name)
        if len(si) < 2:
            return None
        d_ref = self._ref_hp._distances[si]
        z_ref = self._ref_hp._depths[si]
        z_at_d = float(np.interp(d, d_ref, z_ref,
                                  left=float(z_ref[0]), right=float(z_ref[-1])))
        offset = z - z_at_d
        ref_name = self._ref_hp.name
        hp_new = HorizonPick(
            distances=d_ref.copy(),
            depths=(z_ref + offset).copy(),
            name=name or f"{ref_name} (parallel)",
            color=self._ref_hp.color,
            section_names=[sec_name] * len(d_ref),
        )
        hp_new.construction_rule = ParallelToBedRule(
            reference_name=ref_name, offset_m=offset
        )
        self.reset()
        return hp_new


class KinkBandTool:
    """Two-click tool: click backlimb horizon → click axial trace position.

    The backlimb horizon is tagged with :class:`KinkBandRule`.  A new
    forelimb horizon is created extending from the axial trace with
    *fore_dip_deg*.
    """

    def __init__(
        self,
        axial_surface_dip_deg: float = 45.0,
        fore_dip_deg:          float = 30.0,
        back_dip_deg:          float = 0.0,
    ) -> None:
        self.axial_surface_dip_deg: float = float(axial_surface_dip_deg)
        self.fore_dip_deg:          float = float(fore_dip_deg)
        self.back_dip_deg:          float = float(back_dip_deg)
        self._ref_hp: HorizonPick | None  = None

    # ------------------------------------------------------------------

    @property
    def state(self) -> Literal["idle", "ref_selected"]:
        return "idle" if self._ref_hp is None else "ref_selected"

    @property
    def reference(self) -> HorizonPick | None:
        return self._ref_hp

    def reset(self) -> None:
        self._ref_hp = None

    def hint(self) -> str:
        if self._ref_hp is None:
            return "Click on the backlimb horizon (Kink Band)"
        return f"Click to place axial trace  [fore = {self.fore_dip_deg:.0f}°]"

    def set_reference(self, hp: HorizonPick) -> None:
        self._ref_hp = hp

    def handle_axial_click(
        self,
        axial_d: float,
        sec_name: str,
        extent_d: float | None = None,
        name: str = "",
    ) -> HorizonPick | None:
        """Create a forelimb horizon from the axial trace.

        *axial_d* is the distance of the axial trace.  *extent_d* sets how
        far the forelimb extends; defaults to half the backlimb length (min
        1 000 m, max 2 000 m) from the axial trace.
        """
        if self._ref_hp is None:
            return None
        si = self._ref_hp.section_indices(sec_name)
        if len(si) < 2:
            return None
        d_ref = self._ref_hp._distances[si]
        z_ref = self._ref_hp._depths[si]

        self._ref_hp.construction_rule = KinkBandRule(
            axial_surface_dip_deg=self.axial_surface_dip_deg,
            fore_dip_deg=self.fore_dip_deg,
            back_dip_deg=self.back_dip_deg,
        )

        z_axial = float(np.interp(axial_d, d_ref, z_ref,
                                   left=float(z_ref[0]), right=float(z_ref[-1])))

        if extent_d is None:
            half_len = abs(float(d_ref[-1]) - float(d_ref[0])) / 2
            extent_d = axial_d + min(max(half_len, 1_000.0), 2_000.0)

        slope   = math.tan(math.radians(self.fore_dip_deg))
        z_extent = z_axial + (extent_d - axial_d) * slope

        hp_new = HorizonPick(
            distances=[axial_d, extent_d],
            depths=[z_axial, z_extent],
            name=name or f"{self._ref_hp.name} (forelimb)",
            color=self._ref_hp.color,
            section_names=[sec_name, sec_name],
        )
        hp_new.construction_rule = KinkBandRule(
            axial_surface_dip_deg=self.axial_surface_dip_deg,
            fore_dip_deg=self.fore_dip_deg,
            back_dip_deg=self.back_dip_deg,
        )
        self.reset()
        return hp_new
