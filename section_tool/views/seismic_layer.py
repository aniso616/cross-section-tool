"""Fast seismic display layer using pyqtgraph ImageItem.

This widget sits BEHIND the matplotlib canvas in the section view.
It is purely a display layer — all mouse events pass through to
matplotlib above it for pan/zoom/pick/etc.

Usage
-----
1. Place it behind the matplotlib canvas in a QStackedLayout.
2. Call ``set_data()`` when seismic data loads or changes.
3. Call ``sync_view()`` at the end of every render to match the
   matplotlib axes limits.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout

import pyqtgraph as pg

# Suppress pyqtgraph startup message
pg.setConfigOption("antialias", False)
pg.setConfigOption("background", "#0e1014")
pg.setConfigOption("foreground", "#888888")

# Grayscale LUTs — built once
_LUT_GRAY_R = None   # reversed: -clip → white, +clip → black  (default)
_LUT_GRAY   = None   # normal:  -clip → black, +clip → white


def _build_luts() -> None:
    global _LUT_GRAY_R, _LUT_GRAY
    _LUT_GRAY_R = np.empty((256, 3), dtype=np.uint8)
    _LUT_GRAY   = np.empty((256, 3), dtype=np.uint8)
    for i in range(256):
        v_r = 255 - i
        _LUT_GRAY_R[i] = [v_r, v_r, v_r]
        _LUT_GRAY  [i] = [i,   i,   i  ]


_build_luts()


class SeismicLayer(QWidget):
    """pyqtgraph-based seismic image layer.

    Events pass through (``WA_TransparentForMouseEvents``).
    No interactive view changes — the viewbox is driven entirely by
    ``sync_view()`` calls from section_view after matplotlib sets limits.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Events must reach the matplotlib canvas on top
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._gw = pg.GraphicsLayoutWidget(self)
        self._gw.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._gw.setBackground("#0e1014")
        layout.addWidget(self._gw)

        self._plot = self._gw.addPlot()
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setMenuEnabled(False)
        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)   # navigation is in matplotlib
        vb.invertY(True)                         # depth increases downward

        self._img: pg.ImageItem | None = None
        self._cmap_key: str = ""
        self._vmax: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(
        self,
        data: np.ndarray,
        *,
        vmax: float,
        dist_min: float,
        dist_max: float,
        y_top: float,
        y_bot: float,
        cmap_key: str = "gray_r",
    ) -> None:
        """Load seismic image data.

        Parameters
        ----------
        data:
            (n_traces, n_samples) float32 amplitude array, already
            normalised/gain-applied.
        vmax:
            Clip level — values in [-vmax, +vmax] fill the LUT range.
        dist_min, dist_max:
            Section-distance extent (x-axis).
        y_top, y_bot:
            Depth / TWT extent (y-axis; y_bot > y_top).
        cmap_key:
            ``'gray_r'`` (default) or ``'gray'``.
        """
        if self._img is not None:
            self._plot.removeItem(self._img)
            self._img = None

        if data is None or data.size == 0:
            return

        # pyqtgraph ImageItem: axis 0 = x (traces), axis 1 = y (samples).
        # Data arrives as (n_samples, n_traces) — transpose to correct orientation.
        lut = _LUT_GRAY if cmap_key == "gray" else _LUT_GRAY_R
        self._img = pg.ImageItem(image=np.ascontiguousarray(data.T),
                                  levels=(-vmax, vmax), lut=lut)
        # Disable pyqtgraph's own auto-levelling
        self._img.setAutoDownsample(True)

        # Position the image in data space.
        # pyqtgraph ImageItem.setRect(x, y, w, h):
        #   x, y  = top-left corner in data coords (with invertY=True, y_top is top)
        #   w, h  = width, height in data coords
        self._img.setRect(
            dist_min, y_top,
            dist_max - dist_min, y_bot - y_top,
        )
        self._plot.addItem(self._img)

    def clear(self) -> None:
        """Remove seismic image."""
        if self._img is not None:
            self._plot.removeItem(self._img)
            self._img = None

    def sync_view(
        self,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
    ) -> None:
        """Match the pyqtgraph viewbox to the current matplotlib limits.

        Called at the end of every section render after matplotlib has
        applied xlim/ylim.  ``ymin`` is the shallow end (top of section),
        ``ymax`` is the deep end (bottom).
        """
        vb = self._plot.getViewBox()
        # Block signals so this doesn't recurse
        vb.blockSignals(True)
        vb.setXRange(xmin, xmax, padding=0)
        # invertY=True: setYRange(max, min) puts shallow (small) at top
        vb.setYRange(ymin, ymax, padding=0)
        vb.blockSignals(False)
