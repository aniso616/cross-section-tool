from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import h5py
import numpy as np

from cross_section_tool.core.formation import Formation, StratigraphicColumn
from cross_section_tool.core.polygons import SectionPolygon
from cross_section_tool.core.reference_line import ReferenceLine
from cross_section_tool.core.section import Section
from cross_section_tool.core.surfaces import HorizonPick, Surface
from cross_section_tool.core.wells import DeviationSurvey, LogCurve, Well

FORMAT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# SeismicRef
# ---------------------------------------------------------------------------

@dataclass
class SeismicRef:
    """Reference to an external SEG-Y file kept inside a project.

    Stores the file path and all :func:`~cross_section_tool.io.segy.read_segy`
    keyword arguments so the file can be reloaded with the same parameters.
    """
    path: str
    name: str = ""
    x_field: int = 181      # segyio.TraceField.CDP_X
    y_field: int = 185      # segyio.TraceField.CDP_Y
    scalar_field: int = 71  # segyio.TraceField.SourceGroupScalar
    apply_scalar: bool = True
    domain: str = "twt"
    depth_units: str = "ms"
    crs_epsg: int = 32632

    def load(self):
        """Load and return the referenced :class:`~cross_section_tool.io.segy.SeismicDataset`."""
        from cross_section_tool.io.segy import read_segy  # local to avoid circular at module level
        return read_segy(
            self.path,
            x_field=self.x_field,
            y_field=self.y_field,
            scalar_field=self.scalar_field,
            apply_scalar=self.apply_scalar,
            domain=self.domain,
            depth_units=self.depth_units,
            crs_epsg=self.crs_epsg,
        )


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class Project:
    """Container for all objects in a cross-section interpretation project.

    Attributes
    ----------
    name:          Human-readable project name.
    crs_epsg:      Default projected CRS for new objects.
    sections:      Ordered list of :class:`~cross_section_tool.core.section.Section`.
    surfaces:      Ordered list of :class:`~cross_section_tool.core.surfaces.Surface`.
    horizon_picks: Ordered list of :class:`~cross_section_tool.core.surfaces.HorizonPick`.
    wells:         Ordered list of :class:`~cross_section_tool.core.wells.Well`.
    seismic_refs:  Ordered list of :class:`SeismicRef` (paths to SEG-Y files).
    """

    def __init__(self, name: str = "", crs_epsg: int = 32632) -> None:
        self.name = name
        self.crs_epsg = int(crs_epsg)
        self.sections: list[Section] = []
        self.surfaces: list[Surface] = []
        self.horizon_picks: list[HorizonPick] = []
        self.wells: list[Well] = []
        self.seismic_refs: list[SeismicRef] = []
        self.fault_picks: list[HorizonPick] = []
        self.polygons: list[SectionPolygon] = []
        self.reference_lines: list[ReferenceLine] = []
        self.strat_column: StratigraphicColumn = StratigraphicColumn()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """Serialise the project to an HDF5 file at *path* (overwrites if exists)."""
        with h5py.File(str(path), "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["project_name"] = self.name
            f.attrs["crs_epsg"] = self.crs_epsg
            f.attrs["created_at"] = datetime.now().isoformat()
            _save_sections(f, self.sections)
            _save_surfaces(f, self.surfaces)
            _save_horizon_picks(f, self.horizon_picks)
            _save_wells(f, self.wells)
            _save_seismic_refs(f, self.seismic_refs)
            _save_horizon_picks_group(f, "fault_picks", self.fault_picks)
            _save_polygons(f, self.polygons)
            _save_reference_lines(f, self.reference_lines)
            _save_strat_column(f, self.strat_column)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Project":
        """Deserialise a project from an HDF5 file."""
        with h5py.File(str(path), "r") as f:
            proj = cls(
                name=_str(f.attrs.get("project_name", "")),
                crs_epsg=int(f.attrs.get("crs_epsg", 32632)),
            )
            proj.sections = _load_sections(f)
            proj.surfaces = _load_surfaces(f)
            proj.horizon_picks = _load_horizon_picks(f)
            proj.wells = _load_wells(f)
            proj.seismic_refs = _load_seismic_refs(f)
            proj.fault_picks     = _load_horizon_picks_group(f, "fault_picks")
            proj.polygons        = _load_polygons(f)
            proj.reference_lines = _load_reference_lines(f)
            proj.strat_column    = _load_strat_column(f)
        return proj

    def __repr__(self) -> str:
        return (
            f"Project(name={self.name!r}, crs_epsg={self.crs_epsg}, "
            f"sections={len(self.sections)}, surfaces={len(self.surfaces)}, "
            f"horizon_picks={len(self.horizon_picks)}, "
            f"wells={len(self.wells)}, seismic_refs={len(self.seismic_refs)})"
        )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_sections(f: h5py.File, sections: list[Section]) -> None:
    grp = f.create_group("sections")
    for i, sec in enumerate(sections):
        sg = grp.create_group(str(i))
        sg.attrs["name"] = sec.name
        sg.attrs["depth_domain"] = sec.depth_domain
        sg.attrs["depth_units"] = sec.depth_units
        sg.attrs["vertical_exaggeration"] = sec.vertical_exaggeration
        sg.attrs["crs_epsg"] = sec.crs_epsg
        sg.create_dataset("nodes", data=sec._nodes, dtype="float64")


def _save_surfaces(f: h5py.File, surfaces: list[Surface]) -> None:
    grp = f.create_group("surfaces")
    for i, surf in enumerate(surfaces):
        sg = grp.create_group(str(i))
        sg.attrs["name"] = surf.name
        sg.attrs["kind"] = surf.kind
        sg.attrs["z_units"] = surf.z_units
        sg.attrs["crs_epsg"] = surf.crs_epsg
        sg.attrs["is_grid"] = int(surf._is_grid)
        sg.create_dataset("x", data=surf._x, dtype="float64")
        sg.create_dataset("y", data=surf._y, dtype="float64")
        sg.create_dataset("z", data=surf._z, dtype="float64")
        if surf._is_grid:
            sg.create_dataset("grid_x", data=surf._grid_x, dtype="float64")
            sg.create_dataset("grid_y", data=surf._grid_y, dtype="float64")
            sg.create_dataset("grid_z", data=surf._grid_z, dtype="float64")


def _save_horizon_picks_group(
    f: h5py.File, group_name: str, picks: list[HorizonPick]
) -> None:
    grp = f.create_group(group_name)
    for i, hp in enumerate(picks):
        sg = grp.create_group(str(i))
        sg.attrs["name"]       = hp.name
        sg.attrs["z_units"]    = hp.z_units
        sg.attrs["color"]      = hp.color
        sg.attrs["line_width"] = float(getattr(hp, "line_width", 1.5))
        sg.attrs["line_style"] = str(getattr(hp, "line_style", "solid"))
        # Phase A / B attributes
        sg.attrs["contact_type"]      = str(getattr(hp, "contact_type", "conformable"))
        sg.attrs["formation_above"]   = str(getattr(hp, "formation_above", ""))
        sg.attrs["formation_below"]   = str(getattr(hp, "formation_below", ""))
        sg.attrs["confidence"]        = float(getattr(hp, "confidence", 1.0))
        sg.attrs["fault_type"]        = str(getattr(hp, "fault_type", "normal"))
        sg.attrs["dip_direction"]     = str(getattr(hp, "dip_direction", "right"))
        sg.attrs["sense_of_slip"]     = str(getattr(hp, "sense_of_slip", "dip_slip"))
        if getattr(hp, "age_ma", None) is not None:
            sg.attrs["age_ma"] = float(hp.age_ma)
        if getattr(hp, "displacement", None) is not None:
            sg.attrs["displacement"] = float(hp.displacement)
        sg.create_dataset("distances", data=hp._distances, dtype="float64")
        sg.create_dataset("depths",    data=hp._depths,    dtype="float64")
        snames = getattr(hp, "_section_names", None)
        if snames is not None and len(snames) > 0:
            sg.create_dataset("section_names",
                              data=[s.encode() if isinstance(s, str) else s
                                    for s in snames.tolist()])


def _save_horizon_picks(f: h5py.File, picks: list[HorizonPick]) -> None:
    _save_horizon_picks_group(f, "horizon_picks", picks)


def _save_wells(f: h5py.File, wells: list[Well]) -> None:
    grp = f.create_group("wells")
    for i, well in enumerate(wells):
        wg = grp.create_group(str(i))
        wg.attrs["name"] = well.name
        wg.attrs["x"] = well.x
        wg.attrs["y"] = well.y
        wg.attrs["kb"] = well.kb
        wg.attrs["uwi"] = well.uwi
        # Deviation survey — store original inc/azi so reconstruction is exact
        dg = wg.create_group("deviation")
        dg.attrs["surface_x"] = well.deviation.surface_x
        dg.attrs["surface_y"] = well.deviation.surface_y
        dg.create_dataset("md", data=well.deviation._md, dtype="float64")
        dg.create_dataset("inc_deg", data=well.deviation._inc_deg, dtype="float64")
        dg.create_dataset("azi_deg", data=well.deviation._azi_deg, dtype="float64")
        # Log curves
        lg = wg.create_group("logs")
        for curve in well._logs.values():
            cg = lg.create_group(_safe_key(curve.name))
            cg.attrs["name"] = curve.name
            cg.attrs["units"] = curve.units
            cg.create_dataset("depths", data=curve._depths, dtype="float64")
            cg.create_dataset("values", data=curve._values, dtype="float64")
        # Formation tops — two parallel arrays
        tg = wg.create_group("formation_tops")
        tops = well._formation_tops
        if tops:
            names_arr = np.array(list(tops.keys()), dtype=object)
            mds_arr = np.array(list(tops.values()), dtype=float)
            tg.create_dataset("names", data=names_arr, dtype=h5py.string_dtype())
            tg.create_dataset("md_values", data=mds_arr, dtype="float64")


def _save_seismic_refs(f: h5py.File, refs: list[SeismicRef]) -> None:
    grp = f.create_group("seismic_refs")
    for i, ref in enumerate(refs):
        sg = grp.create_group(str(i))
        sg.attrs["path"] = ref.path
        sg.attrs["name"] = ref.name
        sg.attrs["x_field"] = ref.x_field
        sg.attrs["y_field"] = ref.y_field
        sg.attrs["scalar_field"] = ref.scalar_field
        sg.attrs["apply_scalar"] = int(ref.apply_scalar)
        sg.attrs["domain"] = ref.domain
        sg.attrs["depth_units"] = ref.depth_units
        sg.attrs["crs_epsg"] = ref.crs_epsg


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_sections(f: h5py.File) -> list[Section]:
    if "sections" not in f:
        return []
    grp = f["sections"]
    return [
        Section(
            nodes=grp[k]["nodes"][:],
            name=_str(grp[k].attrs.get("name", "")),
            depth_domain=_str(grp[k].attrs.get("depth_domain", "depth")),
            depth_units=_str(grp[k].attrs.get("depth_units", "m")),
            vertical_exaggeration=float(grp[k].attrs.get("vertical_exaggeration", 1.0)),
            crs_epsg=int(grp[k].attrs.get("crs_epsg", 32632)),
        )
        for k in _sorted_keys(grp)
    ]


def _load_surfaces(f: h5py.File) -> list[Surface]:
    if "surfaces" not in f:
        return []
    grp = f["surfaces"]
    result = []
    for k in _sorted_keys(grp):
        sg = grp[k]
        kwargs = dict(
            name=_str(sg.attrs.get("name", "")),
            kind=_str(sg.attrs.get("kind", "horizon")),
            z_units=_str(sg.attrs.get("z_units", "m")),
            crs_epsg=int(sg.attrs.get("crs_epsg", 32632)),
        )
        if bool(sg.attrs.get("is_grid", 0)):
            surf = Surface.from_grid(sg["grid_x"][:], sg["grid_y"][:], sg["grid_z"][:], **kwargs)
        else:
            surf = Surface(sg["x"][:], sg["y"][:], sg["z"][:], **kwargs)
        result.append(surf)
    return result


def _load_horizon_picks_group(
    f: h5py.File, group_name: str, default_color: str = "#1f77b4"
) -> list[HorizonPick]:
    if group_name not in f:
        return []
    grp = f[group_name]
    result = []
    for k in _sorted_keys(grp):
        sg = grp[k]
        distances  = sg["distances"][:]
        depths     = sg["depths"][:]
        name       = _str(sg.attrs.get("name", ""))
        z_units    = _str(sg.attrs.get("z_units", "m"))
        color      = _str(sg.attrs.get("color", default_color))
        line_width = float(sg.attrs.get("line_width", 1.5))
        line_style = _str(sg.attrs.get("line_style", "solid"))
        if "section_names" in sg:
            raw = sg["section_names"][:]
            section_names = [s.decode() if isinstance(s, bytes) else str(s)
                             for s in raw]
        else:
            section_names = None
        contact_type    = _str(sg.attrs.get("contact_type", "conformable"))
        formation_above = _str(sg.attrs.get("formation_above", ""))
        formation_below = _str(sg.attrs.get("formation_below", ""))
        confidence      = float(sg.attrs.get("confidence", 1.0))
        fault_type      = _str(sg.attrs.get("fault_type", "normal"))
        dip_direction   = _str(sg.attrs.get("dip_direction", "right"))
        sense_of_slip   = _str(sg.attrs.get("sense_of_slip", "dip_slip"))
        age_ma          = float(sg.attrs["age_ma"]) if "age_ma" in sg.attrs else None
        displacement    = float(sg.attrs["displacement"]) if "displacement" in sg.attrs else None
        extra = dict(contact_type=contact_type, formation_above=formation_above,
                     formation_below=formation_below, confidence=confidence,
                     fault_type=fault_type, dip_direction=dip_direction,
                     sense_of_slip=sense_of_slip, age_ma=age_ma, displacement=displacement)
        if len(distances) == 0:
            hp = HorizonPick.empty(name=name, z_units=z_units, color=color,
                                   line_width=line_width, line_style=line_style)
            for k, v in extra.items():
                setattr(hp, k, v)
        else:
            hp = HorizonPick(distances=distances, depths=depths, name=name,
                             z_units=z_units, color=color,
                             line_width=line_width, line_style=line_style,
                             section_names=section_names, **extra)
        result.append(hp)
    return result


def _load_horizon_picks(f: h5py.File) -> list[HorizonPick]:
    return _load_horizon_picks_group(f, "horizon_picks")


def _load_wells(f: h5py.File) -> list[Well]:
    if "wells" not in f:
        return []
    grp = f["wells"]
    result = []
    for k in _sorted_keys(grp):
        wg = grp[k]
        dg = wg["deviation"]
        dev = DeviationSurvey(
            md=dg["md"][:],
            inc_deg=dg["inc_deg"][:],
            azi_deg=dg["azi_deg"][:],
            surface_x=float(dg.attrs["surface_x"]),
            surface_y=float(dg.attrs["surface_y"]),
        )
        well = Well(
            name=_str(wg.attrs.get("name", "")),
            x=float(wg.attrs["x"]),
            y=float(wg.attrs["y"]),
            kb=float(wg.attrs.get("kb", 0.0)),
            uwi=_str(wg.attrs.get("uwi", "")),
            deviation=dev,
        )
        # Logs
        for ck in wg.get("logs", {}).keys():
            cg = wg["logs"][ck]
            well.add_log(LogCurve(
                name=_str(cg.attrs["name"]),
                units=_str(cg.attrs.get("units", "")),
                depths=cg["depths"][:],
                values=cg["values"][:],
            ))
        # Formation tops
        tg = wg.get("formation_tops")
        if tg is not None and "names" in tg and "md_values" in tg:
            for name, md in zip(tg["names"][:], tg["md_values"][:]):
                well.add_formation_top(_str(name), float(md))
        result.append(well)
    return result


def _load_seismic_refs(f: h5py.File) -> list[SeismicRef]:
    if "seismic_refs" not in f:
        return []
    grp = f["seismic_refs"]
    return [
        SeismicRef(
            path=_str(grp[k].attrs["path"]),
            name=_str(grp[k].attrs.get("name", "")),
            x_field=int(grp[k].attrs.get("x_field", 181)),
            y_field=int(grp[k].attrs.get("y_field", 185)),
            scalar_field=int(grp[k].attrs.get("scalar_field", 71)),
            apply_scalar=bool(grp[k].attrs.get("apply_scalar", 1)),
            domain=_str(grp[k].attrs.get("domain", "twt")),
            depth_units=_str(grp[k].attrs.get("depth_units", "ms")),
            crs_epsg=int(grp[k].attrs.get("crs_epsg", 32632)),
        )
        for k in _sorted_keys(grp)
    ]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_polygons(f: h5py.File, polygons: list[SectionPolygon]) -> None:
    grp = f.create_group("polygons")
    for i, poly in enumerate(polygons):
        sg = grp.create_group(str(i))
        sg.attrs["name"] = poly.name
        sg.attrs["fill_color"] = poly.fill_color
        sg.attrs["fill_alpha"] = poly.fill_alpha
        sg.attrs["edge_color"] = poly.edge_color
        sg.attrs["edge_width"] = poly.edge_width
        sg.create_dataset("vertices", data=poly._vertices, dtype="float64")


def _load_polygons(f: h5py.File) -> list[SectionPolygon]:
    if "polygons" not in f:
        return []
    grp = f["polygons"]
    return [
        SectionPolygon(
            vertices=grp[k]["vertices"][:],
            name=_str(grp[k].attrs.get("name", "")),
            fill_color=_str(grp[k].attrs.get("fill_color", "#9467bd")),
            fill_alpha=float(grp[k].attrs.get("fill_alpha", 0.6)),
            edge_color=_str(grp[k].attrs.get("edge_color", "#555555")),
            edge_width=float(grp[k].attrs.get("edge_width", 1.0)),
        )
        for k in _sorted_keys(grp)
    ]


def _save_strat_column(f: h5py.File, col: StratigraphicColumn) -> None:
    import json
    grp = f.create_group("strat_column")
    grp.attrs["data"] = json.dumps(col.to_list())


def _load_strat_column(f: h5py.File) -> StratigraphicColumn:
    import json
    if "strat_column" not in f:
        return StratigraphicColumn()
    raw = f["strat_column"].attrs.get("data", "[]")
    try:
        data = json.loads(raw)
        return StratigraphicColumn.from_list(data)
    except Exception:
        return StratigraphicColumn()


def _save_reference_lines(
    f: h5py.File, lines: list[ReferenceLine]
) -> None:
    grp = f.create_group("reference_lines")
    for i, rl in enumerate(lines):
        sg = grp.create_group(str(i))
        sg.attrs["kind"]      = rl.kind
        sg.attrs["value"]     = rl.value
        sg.attrs["name"]      = rl.name
        sg.attrs["visible"]   = int(rl.visible)
        sg.attrs["color"]     = rl.color
        sg.attrs["angle_deg"] = rl.angle_deg
        sg.attrs["anchor_x"]  = rl.anchor_x
        sg.attrs["anchor_y"]  = rl.anchor_y


def _load_reference_lines(f: h5py.File) -> list[ReferenceLine]:
    if "reference_lines" not in f:
        return []
    grp = f["reference_lines"]
    result = []
    for k in _sorted_keys(grp):
        sg = grp[k]
        result.append(ReferenceLine(
            kind=_str(sg.attrs.get("kind", "horizontal")),
            value=float(sg.attrs.get("value", 0.0)),
            name=_str(sg.attrs.get("name", "")),
            visible=bool(int(sg.attrs.get("visible", 1))),
            color=_str(sg.attrs.get("color", "#999999")),
            angle_deg=float(sg.attrs.get("angle_deg", 0.0)),
            anchor_x=float(sg.attrs.get("anchor_x", 0.0)),
            anchor_y=float(sg.attrs.get("anchor_y", 0.0)),
        ))
    return result


def _str(value: Any) -> str:
    """Coerce h5py attribute / dataset values to plain Python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value) if value is not None else ""


def _safe_key(name: str) -> str:
    """Return a valid HDF5 group name derived from *name*."""
    return re.sub(r"[/\x00]", "_", name) or "unnamed"


def _sorted_keys(grp: h5py.Group) -> list[str]:
    """Return group keys sorted numerically (keys are '0', '1', '2', …)."""
    return sorted(grp.keys(), key=int)
