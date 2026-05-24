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

# LUTs — built once at import time
_LUT_GRAY_R = None   # reversed: -clip → white, +clip → black  (default)
_LUT_GRAY   = None   # normal:  -clip → black, +clip → white
_LUT_SEIS   = None   # diverging: -clip → blue, 0 → white, +clip → red


def _build_luts() -> None:
    global _LUT_GRAY_R, _LUT_GRAY, _LUT_SEIS
    _LUT_GRAY_R = np.empty((256, 3), dtype=np.uint8)
    _LUT_GRAY   = np.empty((256, 3), dtype=np.uint8)
    _LUT_SEIS   = np.empty((256, 3), dtype=np.uint8)
    for i in range(256):
        v_r = 255 - i
        _LUT_GRAY_R[i] = [v_r, v_r, v_r]
        _LUT_GRAY  [i] = [i,   i,   i  ]
        t = i / 255.0   # 0 = -vmax (trough), 1 = +vmax (peak)
        if t < 0.5:
            s = t * 2.0          # 0 → 1  (blue → white)
            _LUT_SEIS[i] = [int(s * 255), int(s * 255), 255]
        else:
            s = (t - 0.5) * 2.0  # 0 → 1  (white → red)
            _LUT_SEIS[i] = [255, int((1.0 - s) * 255), int((1.0 - s) * 255)]


_build_luts()


class SeismicLayer(QWidget):
    """pyqtgraph-based seismic image layer.

    Events pass through (``WA_TransparentForMouseEvents``).
    No interactive view changes — the viewbox is driven entirely by
    ``sync_view()`` calls from section_view after matplotlib sets limits.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Do NOT set WA_TransparentForMouseEvents on self — that flag propagates
        # to all children, so the matplotlib canvas (a child of this widget) would
        # also become invisible to mouse events.  The canvas sits above _gw via
        # raise_(), so it naturally receives events first.  _gw keeps its own
        # WA_TransparentForMouseEvents so pyqtgraph doesn't steal events.
        self.setMinimumSize(0, 0)   # don't let pyqtgraph's default minimum squeeze the splitter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._gw = pg.GraphicsLayoutWidget(self)
        self._gw.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._gw.setMinimumSize(0, 0)
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
        if cmap_key == "gray":
            lut = _LUT_GRAY
        elif cmap_key == "seismic":
            lut = _LUT_SEIS
        else:
            lut = _LUT_GRAY_R
        self._img = pg.ImageItem(image=np.ascontiguousarray(data.T),
                                  levels=(-vmax, vmax), lut=lut)
        # Keep full image detail — let the compositing layer handle scaling.
        # autoDownsample(True) would reduce traces to view-pixel resolution,
        # which on HiDPI is only half the physical pixels and looks blurry.
        self._img.setAutoDownsample(False)

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

    def render_to_pixmap(self, pm: "QPixmap") -> None:
        """Render only the pyqtgraph seismic scene to *pm* (no overlay children).

        Uses scene().render() with an explicit target rect so the scene is
        sampled at the pixmap's full physical-pixel resolution rather than at
        the view's logical resolution.
        """
        from PySide6.QtGui import QPainter
        from PySide6.QtCore import QRectF
        pm.fill(Qt.GlobalColor.black)
        painter = QPainter(pm)
        dpr = pm.devicePixelRatioF()
        # Painter coordinate space is logical pixels (pm.width()/dpr × pm.height()/dpr).
        # Rendering into the full logical rect maps to the full physical pixmap.
        logical_rect = QRectF(0, 0, pm.width() / dpr, pm.height() / dpr)
        scene_rect = self._gw.mapToScene(self._gw.viewport().rect()).boundingRect()
        self._gw.scene().render(painter, logical_rect, scene_rect)
        painter.end()

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
