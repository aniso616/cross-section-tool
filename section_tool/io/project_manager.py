"""ProjectManager — folder-based project lifecycle management.

A project is a directory containing:
  project.sqlite   — all interpretation data
  seismic/         — SEG-Y files (copied or referenced)
  wells/           — LAS files (copied or referenced)
  images/          — draped section images
  exports/         — user-generated exports
  cache/           — cached numpy extractions
  autosave/        — periodic database backups
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from section_tool.io.database import ProjectDatabase

_SUBDIRS = ("seismic", "wells", "images", "exports", "cache", "autosave")
_DB_NAME = "project.sqlite"
_AUTOSAVE_NAME = "autosave/project.sqlite.bak"


def _safe_name(s: str) -> str:
    """Filesystem-safe token (alphanumerics, dash, underscore)."""
    return re.sub(r"[^\w\-]", "_", str(s))


def seismic_cache_key(section, seismic_name: str, segy_path: str) -> str:
    """Geometry-aware, project-independent cache key for a seismic extraction.

    The key incorporates the SEG-Y source path *and* the section geometry
    (node coordinates + total length), so two sections that share a name but
    differ in geometry — e.g. a default-named ``"Section 1"`` in two projects —
    produce different keys and never collide on the same cache file.  When a
    section's nodes change, the key changes automatically, so the next lookup
    misses and the seismic is re-extracted for the new geometry.

    The sample range is determined by the SEG-Y, so ``segy_path`` covers it;
    azimuth is a function of the nodes, so the nodes cover it.
    """
    import numpy as np
    nodes = tuple(map(tuple, np.asarray(section.nodes, dtype=float).round(3).tolist()))
    geom = repr((str(segy_path), nodes, round(float(section.total_length()), 2)))
    digest = hashlib.sha1(geom.encode("utf-8")).hexdigest()[:12]
    return f"{_safe_name(section.name)}_{_safe_name(seismic_name)}_{digest}"


class ProjectManager:
    """Manages the project folder, database lifecycle, and file imports."""

    def __init__(self) -> None:
        self.project_path: str | None = None
        self.db: ProjectDatabase | None = None

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def new_project(
        self,
        folder_path: str | Path,
        name: str,
        crs_epsg: int = 32632,
        depth_units: str = "m",
        depth_domain: str = "md",
        default_depth_min: float = 0.0,
        default_depth_max: float = 5000.0,
    ) -> None:
        """Create a new project folder and initialise the database."""
        folder_path = str(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        for sub in _SUBDIRS:
            os.makedirs(os.path.join(folder_path, sub), exist_ok=True)

        db_path = os.path.join(folder_path, _DB_NAME)
        if self.db is not None:
            self.db.close()
        self.db = ProjectDatabase(db_path)
        self.db.set_project_settings(
            name=name,
            crs_epsg=crs_epsg,
            depth_units=depth_units,
            depth_domain=depth_domain,
            default_depth_min=default_depth_min,
            default_depth_max=default_depth_max,
        )
        self.project_path = folder_path

    def open_project(self, folder_path: str | Path) -> None:
        """Open an existing project folder."""
        folder_path = str(folder_path)
        db_path = os.path.join(folder_path, _DB_NAME)
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Not a valid project folder (missing {_DB_NAME}): {folder_path}"
            )
        if self.db is not None:
            self.db.close()
        # Ensure subdirectories exist (project may have been created on another machine)
        for sub in _SUBDIRS:
            os.makedirs(os.path.join(folder_path, sub), exist_ok=True)
        self.db = ProjectDatabase(db_path)
        self.project_path = folder_path

    def save(self) -> None:
        """Commit any pending changes to the database."""
        if self.db:
            self.db.commit()

    def save_as(self, new_folder_path: str | Path) -> None:
        """Copy the entire project folder to a new location."""
        new_folder_path = str(new_folder_path)
        if os.path.exists(new_folder_path):
            shutil.rmtree(new_folder_path)
        shutil.copytree(self.project_path, new_folder_path)
        # Reopen database at new location
        if self.db:
            self.db.close()
        self.db = ProjectDatabase(os.path.join(new_folder_path, _DB_NAME))
        self.project_path = new_folder_path

    def autosave(self) -> None:
        """Copy project.sqlite to autosave/ as a backup."""
        if not self.project_path or not self.db:
            return
        src = os.path.join(self.project_path, _DB_NAME)
        dst = os.path.join(self.project_path, _AUTOSAVE_NAME)
        try:
            self.db.commit()
            shutil.copy2(src, dst)
        except Exception:
            pass

    def autosave_is_newer(self) -> bool:
        """Return True if the autosave backup is newer than the main database."""
        if not self.project_path:
            return False
        src = os.path.join(self.project_path, _DB_NAME)
        dst = os.path.join(self.project_path, _AUTOSAVE_NAME)
        if not os.path.exists(dst):
            return False
        return os.path.getmtime(dst) > os.path.getmtime(src)

    def recover_autosave(self) -> None:
        """Overwrite the main database with the autosave backup."""
        if not self.project_path:
            return
        src = os.path.join(self.project_path, _DB_NAME)
        bak = os.path.join(self.project_path, _AUTOSAVE_NAME)
        if not os.path.exists(bak):
            return
        if self.db:
            self.db.close()
        shutil.copy2(bak, src)
        self.db = ProjectDatabase(src)

    def close(self) -> None:
        if self.db:
            self.db.close()
            self.db = None

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    def import_file(
        self,
        source_path: str | Path,
        file_type: str,
        copy: bool = True,
    ) -> str:
        """Import a file into the project folder.

        Parameters
        ----------
        source_path:
            Absolute path to the source file.
        file_type:
            One of ``'segy'``, ``'las'``, ``'image'``.
        copy:
            If True, copy the file into the project folder.
            If False, store an external reference (absolute path).

        Returns
        -------
        str
            Path to the file as it should be stored on :class:`SeismicRef`
            (absolute for external references, relative for copies).
        """
        source_path = str(source_path)
        subdir_map = {"segy": "seismic", "las": "wells", "image": "images"}
        subdir = subdir_map.get(file_type, "imports")

        if not copy:
            return source_path  # external reference — absolute path

        dest_dir = os.path.join(self.project_path, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(source_path))
        if os.path.abspath(source_path) != os.path.abspath(dest):
            shutil.copy2(source_path, dest)
        return dest

    def resolve_file_path(self, stored_path: str | None) -> str | None:
        """Resolve a stored path (relative or absolute) to an absolute path.

        Returns None if *stored_path* is None or empty.
        """
        if not stored_path:
            return None
        if os.path.isabs(stored_path):
            return stored_path
        return os.path.join(self.project_path, stored_path)

    # ------------------------------------------------------------------
    # Cached seismic extractions
    # ------------------------------------------------------------------

    def seismic_extract_npy_path(self, section, seismic_name: str, segy_path: str) -> str:
        """Project-scoped, geometry-aware path for a seismic extraction (.npy).

        Always resolves under ``<project>/cache/`` — never a shared/global dir —
        and the filename carries a geometry hash (see :func:`seismic_cache_key`)
        so same-named sections in different projects/geometries never collide.
        Raises if no project is open (extraction must target a project).
        """
        if not self.project_path:
            raise RuntimeError("No project open; cannot resolve a seismic cache path.")
        key = seismic_cache_key(section, seismic_name, segy_path)
        return os.path.join(self.project_path, "cache", f"{key}.extract.npy")

    def seismic_cache_path(self, section_name: str, seismic_name: str) -> str:
        """Return the path for a cached .npy seismic extraction."""
        safe_sec = "".join(c if c.isalnum() or c in "-_" else "_"
                           for c in section_name)
        safe_seis = "".join(c if c.isalnum() or c in "-_" else "_"
                            for c in seismic_name)
        return os.path.join(
            self.project_path, "cache",
            f"{safe_sec}__{safe_seis}.extract.npz"
        )

    def save_seismic_extract(
        self,
        section_name: str,
        seismic_name: str,
        distances: "np.ndarray",
        data: "np.ndarray",
        samples: "np.ndarray",
    ) -> str:
        """Save a processed seismic extraction to the cache folder."""
        import numpy as np
        path = self.seismic_cache_path(section_name, seismic_name)
        np.savez_compressed(path, distances=distances, data=data, samples=samples)
        return path

    def load_seismic_extract(
        self, section_name: str, seismic_name: str
    ) -> dict | None:
        """Load a cached seismic extraction, or None if not cached."""
        import numpy as np
        path = self.seismic_cache_path(section_name, seismic_name)
        if not os.path.exists(path):
            return None
        try:
            npz = np.load(path)
            return {
                "distances": npz["distances"],
                "data": npz["data"],
                "samples": npz["samples"],
            }
        except Exception:
            return None

    def invalidate_seismic_cache(
        self, section_name: str | None = None, seismic_name: str | None = None
    ) -> None:
        """Delete cached extractions matching the given names (or all if both None)."""
        cache_dir = os.path.join(self.project_path, "cache")
        if not os.path.isdir(cache_dir):
            return
        for fname in os.listdir(cache_dir):
            if not fname.endswith(".extract.npz"):
                continue
            if section_name and section_name not in fname:
                continue
            if seismic_name and seismic_name not in fname:
                continue
            try:
                os.remove(os.path.join(cache_dir, fname))
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self.project_path is not None and self.db is not None

    @property
    def db_path(self) -> str | None:
        if not self.project_path:
            return None
        return os.path.join(self.project_path, _DB_NAME)
