"""Backend export functions — no Qt/GUI required.

All functions accept plain Python / numpy objects and write to disk.
They are safe to call from any thread.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# CSV exports — accept a ProjectDatabase (or any object with get_all_* methods)
# ---------------------------------------------------------------------------

def export_horizons_csv(
    db,
    output_path: Union[str, Path],
    section_name: Optional[str] = None,
) -> int:
    """Export all horizon picks to CSV.

    Computes map (x, y) from section geometry stored in the database.

    Parameters
    ----------
    db:
        ProjectDatabase instance.
    output_path:
        Destination CSV file path.
    section_name:
        If given, only export picks from this section.

    Returns
    -------
    int
        Number of data rows written (excludes header).
    """
    sec_nodes = _load_section_nodes(db)
    rows = []
    for h in db.get_all_horizons():
        for p in h["picks"]:
            if section_name and p["section_name"] != section_name:
                continue
            depth = float(p["depth"])
            x, y = _distance_to_xy(sec_nodes, p["section_name"], float(p["distance_along"]))
            rows.append({
                "horizon":      h["name"],
                "section":      p["section_name"],
                "distance":     round(float(p["distance_along"]), 3),
                "depth":        round(depth, 3),
                "elevation":    round(float(p.get("elevation") or -depth), 3),
                "x":            round(x, 3) if x is not None else "",
                "y":            round(y, 3) if y is not None else "",
                "confidence":   round(float(h.get("confidence") or 1.0), 3),
                "quality":      "picked",
                "color":        h.get("color", ""),
                "contact_type": h.get("contact_type", "conformable"),
            })
    _write_csv(rows, output_path, fieldnames=[
        "horizon", "section", "distance", "depth", "elevation",
        "x", "y", "confidence", "quality", "color", "contact_type",
    ])
    return len(rows)


def export_faults_csv(
    db,
    output_path: Union[str, Path],
    section_name: Optional[str] = None,
) -> int:
    """Export all fault picks to CSV.

    Same column structure as :func:`export_horizons_csv` plus fault-specific
    metadata (fault_type, dip_direction).

    Returns
    -------
    int
        Number of data rows written.
    """
    sec_nodes = _load_section_nodes(db)
    rows = []
    for f in db.get_all_faults():
        for p in f["picks"]:
            if section_name and p["section_name"] != section_name:
                continue
            depth = float(p["depth"])
            x, y = _distance_to_xy(sec_nodes, p["section_name"], float(p["distance_along"]))
            rows.append({
                "fault":         f["name"],
                "section":       p["section_name"],
                "distance":      round(float(p["distance_along"]), 3),
                "depth":         round(depth, 3),
                "elevation":     round(-depth, 3),
                "x":             round(x, 3) if x is not None else "",
                "y":             round(y, 3) if y is not None else "",
                "confidence":    round(float(f.get("confidence") or 1.0), 3),
                "fault_type":    f.get("fault_type", "normal"),
                "dip_direction": f.get("dip_direction", "right"),
                "color":         f.get("color", ""),
            })
    _write_csv(rows, output_path, fieldnames=[
        "fault", "section", "distance", "depth", "elevation",
        "x", "y", "confidence", "fault_type", "dip_direction", "color",
    ])
    return len(rows)


def export_wells_csv(
    db,
    output_path: Union[str, Path],
) -> int:
    """Export all wells plus formation tops to CSV.

    One row per formation top.  Wells with no tops get a single row with
    blank formation/depth_md columns.

    Returns
    -------
    int
        Number of data rows written.
    """
    rows = []
    for w in db.get_all_wells():
        base = {
            "well":         w["name"],
            "uwi":          w.get("uwi") or "",
            "x":            w.get("x", ""),
            "y":            w.get("y", ""),
            "kb_elevation": w.get("kb_elevation", ""),
            "td":           w.get("td", ""),
            "status":       w.get("status", "actual"),
            "purpose":      w.get("purpose", "exploration"),
        }
        tops = w.get("tops", [])
        if tops:
            for t in tops:
                rows.append({**base,
                              "formation": t["formation_name"],
                              "depth_md":  round(float(t["md"]), 3)})
        else:
            rows.append({**base, "formation": "", "depth_md": ""})
    _write_csv(rows, output_path, fieldnames=[
        "well", "uwi", "x", "y", "kb_elevation", "td",
        "status", "purpose", "formation", "depth_md",
    ])
    return len(rows)


# ---------------------------------------------------------------------------
# Section figure export — accepts in-memory Python objects
# ---------------------------------------------------------------------------

def export_section_figure(
    section,
    picks,
    faults,
    polygons,
    seismic_data,
    output_path: Union[str, Path],
    width_inches: float = 16,
    height_inches: float = 8,
    dpi: int = 300,
    colormap: str = "seismic",
    clip_pct: float = 99.0,
) -> None:
    """Render a publication-quality section figure to file.

    Uses the non-interactive Agg backend — no display required.
    Output format is inferred from the file extension (png, pdf, svg, …).

    Parameters
    ----------
    section:
        :class:`~section_tool.core.section.Section` instance.
    picks:
        Iterable of :class:`~section_tool.core.surfaces.HorizonPick`.
    faults:
        Iterable of :class:`~section_tool.core.surfaces.HorizonPick` (fault picks).
    polygons:
        Iterable of :class:`~section_tool.core.polygons.SectionPolygon`.
    seismic_data:
        :class:`~section_tool.io.segy.SeismicDataset` or *None*.
    output_path:
        Destination file.
    width_inches, height_inches:
        Figure dimensions.
    dpi:
        Resolution for raster formats (PNG).  Ignored for SVG/PDF.
    colormap:
        Matplotlib colormap name for seismic amplitude display.
    clip_pct:
        Percentile used to compute the amplitude colour scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    # ── Seismic ──────────────────────────────────────────────────────────
    if seismic_data is not None:
        _render_seismic_layer(ax, section, seismic_data, colormap, clip_pct)

    # ── Polygons ─────────────────────────────────────────────────────────
    for poly in polygons:
        verts = poly.vertices  # (N, 2): distance, depth
        patch = MplPolygon(
            verts, closed=True,
            facecolor=poly.fill_color, alpha=poly.fill_alpha,
            edgecolor=poly.edge_color, linewidth=poly.edge_width,
            zorder=2,
        )
        ax.add_patch(patch)
        if poly.name:
            cx = float(np.mean(verts[:, 0]))
            cy = float(np.mean(verts[:, 1]))
            ax.text(cx, cy, poly.name, ha="center", va="center",
                    fontsize=7, color="black", clip_on=True, zorder=3)

    # ── Horizon picks ─────────────────────────────────────────────────────
    for h in picks:
        dists, depths = h.picks_for_section(section.name)
        if len(dists) == 0:
            continue
        order = np.argsort(dists)
        ax.plot(dists[order], depths[order],
                color=h.color,
                linewidth=getattr(h, "line_width", 1.5),
                linestyle=_mpl_linestyle(getattr(h, "line_style", "solid")),
                label=h.name, zorder=4)
        mid_idx = order[len(order) // 2]
        ax.text(float(dists[mid_idx]), float(depths[mid_idx]), h.name,
                fontsize=7, va="bottom", color=h.color, clip_on=True, zorder=4)

    # ── Fault picks ───────────────────────────────────────────────────────
    for f in faults:
        dists, depths = f.picks_for_section(section.name)
        if len(dists) == 0:
            continue
        order = np.argsort(dists)
        ax.plot(dists[order], depths[order],
                color=f.color,
                linewidth=getattr(f, "line_width", 1.5),
                linestyle=_mpl_linestyle(getattr(f, "line_style", "solid")),
                zorder=5)
        ax.text(float(dists[order[0]]), float(depths[order[0]]), f.name,
                fontsize=7, va="bottom", color=f.color, clip_on=True, zorder=5)

    # ── Axes setup ────────────────────────────────────────────────────────
    total_len = section.total_length()
    ax.set_xlim(0, total_len)
    max_depth = _infer_max_depth(picks, faults, polygons, seismic_data)
    ax.set_ylim(max_depth, 0)

    domain = getattr(section, "depth_domain", "depth")
    units  = getattr(section, "depth_units", "m")
    ax.set_xlabel("Distance along section (m)", fontsize=10)
    ax.set_ylabel(
        f"Two-way time ({units})" if domain == "twt" else f"Depth ({units})",
        fontsize=10,
    )
    ax.set_title(section.name, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_section_nodes(db) -> dict[str, np.ndarray]:
    """Return {section_name: nodes_array} from the database."""
    result = {}
    for s in db.get_all_sections():
        try:
            nodes = np.array(json.loads(s["nodes_json"]), dtype=float)
            if nodes.ndim == 2 and nodes.shape[1] == 2 and len(nodes) >= 2:
                result[s["name"]] = nodes
        except (KeyError, ValueError, TypeError):
            pass
    return result


def _distance_to_xy(
    sec_nodes: dict[str, np.ndarray],
    section_name: str,
    distance: float,
) -> tuple[Optional[float], Optional[float]]:
    """Interpolate map (x, y) from a distance-along value and section nodes."""
    nodes = sec_nodes.get(section_name)
    if nodes is None:
        return None, None
    deltas = np.sqrt(np.sum(np.diff(nodes, axis=0) ** 2, axis=1))
    cumdist = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumdist[-1])
    d = max(0.0, min(float(distance), total))
    seg = max(0, int(np.searchsorted(cumdist, d, side="right")) - 1)
    seg = min(seg, len(nodes) - 2)
    d0, d1 = float(cumdist[seg]), float(cumdist[seg + 1])
    t = (d - d0) / (d1 - d0) if (d1 - d0) > 1e-9 else 0.0
    x = nodes[seg, 0] + t * (nodes[seg + 1, 0] - nodes[seg, 0])
    y = nodes[seg, 1] + t * (nodes[seg + 1, 1] - nodes[seg, 1])
    return float(x), float(y)


def _write_csv(
    rows: list[dict],
    path: Union[str, Path],
    fieldnames: Optional[list[str]] = None,
) -> None:
    keys = fieldnames or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_seismic_layer(ax, section, seismic_data, colormap, clip_pct) -> None:
    """Rasterise seismic traces onto section axes as an imshow layer."""
    try:
        distances, data, _ = seismic_data.traces_sorted_by_section(section)
        t_min, t_max = seismic_data.time_range
        if len(distances) < 2 or data.size == 0:
            return
        # Interpolate onto a regular distance grid
        n_cols = min(512, len(distances))
        x_grid = np.linspace(0, section.total_length(), n_cols)
        n_rows = data.shape[1]
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        for j, xg in enumerate(x_grid):
            idx = np.argmin(np.abs(distances - xg))
            grid[:, j] = data[idx]
        vmax = float(np.percentile(np.abs(grid[np.isfinite(grid)]), clip_pct) or 1.0)
        ax.imshow(
            grid, aspect="auto",
            extent=[0, section.total_length(), t_max, t_min],
            cmap=colormap, vmin=-vmax, vmax=vmax,
            interpolation="bilinear", zorder=1,
        )
    except Exception:
        pass  # never let seismic failure abort the export


def _infer_max_depth(picks, faults, polygons, seismic_data) -> float:
    """Return a reasonable default max depth / TWT for the depth axis."""
    candidates = [3000.0]
    for h in list(picks) + list(faults):
        depths = getattr(h, "_depths", np.array([]))
        if len(depths) > 0:
            candidates.append(float(np.max(depths)) * 1.1)
    for poly in polygons:
        candidates.append(float(np.max(poly.vertices[:, 1])) * 1.1)
    if seismic_data is not None:
        try:
            candidates.append(float(seismic_data.time_range[1]) * 1.1)
        except Exception:
            pass
    return max(candidates)


def _mpl_linestyle(style: str) -> str:
    return {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}.get(style, "-")
