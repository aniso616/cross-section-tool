"""1D decompaction — Athy porosity, solid thickness, column decompaction, burial history.

Implements the mathematical core of 1D backstripping.  No GUI or Qt dependencies.

Reference
---------
Sclater, J.G., & Christie, P.A.F. (1980). Continental stretching: An explanation
of the post-mid-Cretaceous subsidence of the central North Sea basin.
JGR, 85(B7), 3711–3739.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Porosity model
# ---------------------------------------------------------------------------

def porosity_athy(
    z: float | np.ndarray,
    phi0: float,
    c: float,
) -> float | np.ndarray:
    """Athy's law porosity-depth relationship.

    phi(z) = phi0 * exp(-c * z)

    Parameters
    ----------
    z:
        Depth in metres (positive downward).
    phi0:
        Surface (zero-depth) porosity in [0, 1).
    c:
        Compaction coefficient (1/m).

    Returns
    -------
    float | ndarray
        Porosity at depth *z*.
    """
    return phi0 * np.exp(-c * z)


# ---------------------------------------------------------------------------
# Solid (grain) thickness
# ---------------------------------------------------------------------------

def solid_thickness(
    z_top: float,
    z_bottom: float,
    phi0: float,
    c: float,
) -> float:
    """Solid (grain) thickness of a compactible layer.

    T_solid = integral from z_top to z_bottom of (1 - phi(z)) dz

    Analytical solution for Athy porosity::

        T_solid = (z_bottom - z_top)
                  - (phi0/c) * (exp(-c*z_top) - exp(-c*z_bottom))

    For c = 0 (incompressible material, e.g. salt or basement):
    T_solid = z_bottom - z_top.

    Parameters
    ----------
    z_top:
        Depth of layer top (m).
    z_bottom:
        Depth of layer base (m).
    phi0:
        Surface porosity.
    c:
        Compaction coefficient (1/m).

    Returns
    -------
    float
        Solid thickness in metres.
    """
    if c == 0:
        return float(z_bottom - z_top)
    return float(
        (z_bottom - z_top)
        - (phi0 / c) * (np.exp(-c * z_top) - np.exp(-c * z_bottom))
    )


# ---------------------------------------------------------------------------
# Layer decompaction
# ---------------------------------------------------------------------------

def decompact_layer(
    z_top_new: float,
    T_solid: float,
    phi0: float,
    c: float,
    tolerance: float = 0.01,
) -> float:
    """Find the new bottom depth of a layer when its top moves to *z_top_new*.

    The solid thickness is invariant: we solve for z_bottom_new such that
    ``solid_thickness(z_top_new, z_bottom_new, phi0, c) == T_solid``.

    Uses Brent's method.  Falls back to a linear porosity estimate if the
    root-finder bracket cannot be established.

    Parameters
    ----------
    z_top_new:
        New top depth of the layer (m).
    T_solid:
        Solid (grain) thickness — conserved invariant (m).
    phi0:
        Surface porosity.
    c:
        Compaction coefficient (1/m).
    tolerance:
        Convergence tolerance for Brent's method (m).

    Returns
    -------
    float
        New bottom depth (m).
    """
    if c == 0:
        return float(z_top_new + T_solid)

    def residual(z_bottom: float) -> float:
        return solid_thickness(z_top_new, z_bottom, phi0, c) - T_solid

    # Lower bound: z_bottom >= z_top + T_solid (zero-porosity limit)
    z_min = z_top_new + T_solid
    # Upper bound: generous estimate — expand if necessary
    z_max = z_top_new + T_solid / max(1.0 - phi0, 1e-6) * 2.0

    # Ensure the bracket straddles zero
    if residual(z_min) >= 0:
        return float(z_min)
    for _ in range(30):
        if residual(z_max) > 0:
            break
        z_max = z_top_new + (z_max - z_top_new) * 2.0

    try:
        z_bottom_new = brentq(residual, z_min, z_max, xtol=tolerance)
    except ValueError:
        # Fallback: linear estimate using average porosity at mid-depth
        avg_z = z_top_new + T_solid / 2.0
        avg_phi = porosity_athy(avg_z, phi0, c)
        z_bottom_new = z_top_new + T_solid / max(1.0 - avg_phi, 1e-6)

    return float(z_bottom_new)


# ---------------------------------------------------------------------------
# Column decompaction
# ---------------------------------------------------------------------------

def decompact_column(
    layers: list[dict],
    target_time_index: int = 0,
) -> list[dict]:
    """Decompact a stratigraphic column to its state at a given geological time.

    Parameters
    ----------
    layers:
        Ordered top-to-bottom list of layer dicts, each containing:

        ============  ================================================
        Key           Description
        ============  ================================================
        name          display label
        z_top         current top depth (m, positive downward)
        z_bottom      current base depth (m)
        phi0          surface porosity [0, 1)
        c             compaction coefficient (1/m)
        age_top       age of top surface (Ma)
        age_base      age of base (Ma)
        ============  ================================================

    target_time_index:
        * 0 — present day (no change returned, with thickness fields added).
        * k — remove the k youngest (shallowest) layers and decompact the rest
          to the paleo-surface (z = 0).

    Returns
    -------
    list[dict]
        Surviving layers with updated ``z_top``, ``z_bottom``,
        ``decompacted_thickness``, and ``original_thickness``.
        Returns an empty list if all layers are stripped.
    """
    if target_time_index == 0:
        return [
            {**lyr,
             "decompacted_thickness": lyr["z_bottom"] - lyr["z_top"],
             "original_thickness":    lyr["z_bottom"] - lyr["z_top"]}
            for lyr in layers
        ]

    surviving = layers[target_time_index:]
    if not surviving:
        return []

    # Solid thicknesses are conserved during (de)compaction
    solids = [
        solid_thickness(lyr["z_top"], lyr["z_bottom"], lyr["phi0"], lyr["c"])
        for lyr in surviving
    ]

    # Rebuild from paleo-surface downward
    result: list[dict] = []
    current_top = 0.0

    for lyr, T_s in zip(surviving, solids):
        new_bottom = decompact_layer(current_top, T_s, lyr["phi0"], lyr["c"])
        result.append({
            **lyr,
            "z_top":                 current_top,
            "z_bottom":              new_bottom,
            "decompacted_thickness": new_bottom - current_top,
            "original_thickness":    lyr["z_bottom"] - lyr["z_top"],
        })
        current_top = new_bottom

    return result


# ---------------------------------------------------------------------------
# Burial history
# ---------------------------------------------------------------------------

def burial_history(layers: list[dict]) -> np.ndarray:
    """Compute the full burial history for a stratigraphic column.

    Layers are stripped one at a time from youngest to oldest.  At each time
    step the remaining column is decompacted back to the paleo-surface (z = 0).

    Parameters
    ----------
    layers:
        See :func:`decompact_column` for the expected key set.

    Returns
    -------
    np.ndarray, shape (n_steps, n_layers + 1)
        ``depths[step, i]`` = depth of stratigraphic boundary *i* at time
        step *step*.

        * step 0 → present day
        * step k → k youngest layers stripped, rest decompacted
        * boundary 0 → top of shallowest layer (= 0 at step 0)
        * boundary n_layers → base of deepest layer

        Boundaries that have not yet been deposited at a given step retain
        their initialised value of 0.
    """
    n_layers = len(layers)
    n_steps  = n_layers + 1

    depths = np.zeros((n_steps, n_layers + 1))

    for step in range(n_steps):
        decompacted = decompact_column(layers, target_time_index=step)
        for i, lyr in enumerate(decompacted):
            orig_idx = step + i          # offset: we removed `step` layers
            depths[step, orig_idx]     = lyr["z_top"]
            depths[step, orig_idx + 1] = lyr["z_bottom"]

    return depths
