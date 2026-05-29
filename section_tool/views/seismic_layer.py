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

Two-tier render
---------------
During pan/zoom (interaction) the layer shows whatever data was last
uploaded — no re-upload, no resampling.  200 ms after the last
interaction event (``on_pan_zoom_end``), ``_settle_timer`` fires and
the section_view calls ``_update_seismic_layer`` with full-res data,
which calls ``set_data`` with the full-resolution array.  The
pyqtgraph ImageItem itself does no downsampling (``setAutoDownsample(False)``);
all resampling is left to the GPU/Qt compositing path.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame

import pyqtgraph as pg
from section_tool.style.theme import get_theme

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
        self._gw.setBackground(get_theme().background)
        # Strip the QGraphicsView frame/viewport chrome so the scene fills the
        # widget rect (otherwise a default frame insets the viewbox by px).
        self._gw.setFrameStyle(QFrame.Shape.NoFrame)
        self._gw.setViewportMargins(0, 0, 0, 0)
        self._gw.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gw)

        self._plot = self._gw.addPlot()
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setMenuEnabled(False)
        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)   # navigation is in matplotlib
        vb.invertY(True)                         # depth increases downward

        # Zero every nested margin so the viewbox occupies the same pixel
        # rectangle as the matplotlib axes it sits behind.  pyqtgraph's default
        # GraphicsLayout/PlotItem margins inset the viewbox by ~10px, which makes
        # the seismic map data→pixels at a different scale than matplotlib and
        # drift vertically during pan/zoom.  (A ~1px residual from pyqtgraph's
        # GraphicsView scene mapping is unavoidable but constant and sub-pixel.)
        self._gw.ci.layout.setContentsMargins(0, 0, 0, 0)
        self._gw.ci.layout.setSpacing(0)
        self._plot.setContentsMargins(0, 0, 0, 0)
        vb.setContentsMargins(0, 0, 0, 0)

        # Horizontal inset (fraction of widget width) matching the matplotlib
        # main-axes position when the strat column is shown.  Set by section_view
        # via set_left_inset_frac(); applied as a left layout margin in sync_view
        # (which always runs with the current widget width before each paint).
        self._left_inset_frac: float = 0.0
        self._applied_left_px: int = -1

        self._img: pg.ImageItem | None = None
        self._cmap_key: str = ""
        self._vmax: float = 1.0

        # Two-tier render state: flag + settle timer.
        # While interacting (pan/zoom) the layer shows the existing image unchanged.
        # 200 ms after the last gesture event the timer fires, which triggers a
        # full-res re-upload via the _settle_callback (set by section_view).
        self._interacting: bool = False
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(200)
        self._settle_timer.timeout.connect(self._on_settle)
        # _settle_callback is injected by section_view after construction
        self._settle_callback = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_pan_zoom_start(self) -> None:
        """Call when a pan/zoom gesture begins.

        Marks the layer as interacting so no re-upload is triggered
        mid-gesture.  The settle timer is stopped (not yet needed).
        """
        self._interacting = True
        self._settle_timer.stop()

    def on_pan_zoom_tick(self) -> None:
        """Call on each pan/zoom event tick (e.g. every scroll step).

        Restarts the settle timer so it fires 200 ms after the *last*
        event, not the first.
        """
        self._interacting = True
        if self._settle_callback is not None:
            self._settle_timer.start()

    def on_pan_zoom_end(self) -> None:
        """Call when the gesture is known to be finished (e.g. mouse release).

        Starts the settle timer immediately; it will fire in 200 ms if
        no further ``on_pan_zoom_tick`` calls restart it first.
        """
        self._interacting = True
        if self._settle_callback is not None:
            self._settle_timer.start()

    def _on_settle(self) -> None:
        """Fired by _settle_timer when the view has been still for 200 ms.

        Clears the interacting flag and triggers a full-res re-upload.
        """
        self._interacting = False
        if self._settle_callback is not None:
            self._settle_callback()

    def apply_theme(self, theme) -> None:
        """Update pyqtgraph background to match *theme*."""
        self._gw.setBackground(theme.background)

    def set_left_inset_frac(self, frac: float) -> None:
        """Set the left inset (fraction of widget width) for strat-column alignment.

        The matplotlib main axes is positioned at ``[left, 0, 1-left, 1]`` when
        the strat column is visible.  The seismic viewbox must occupy the same
        horizontal rectangle, so we inset it by the same fraction.  Applied
        lazily in :meth:`sync_view` against the current widget width.
        """
        frac = max(0.0, float(frac))
        if frac != self._left_inset_frac:
            self._left_inset_frac = frac
            self._applied_left_px = -1   # force re-apply on next sync_view

    def _apply_left_inset(self) -> None:
        """Apply the left inset as a pixel layout margin (guarded; no-op if unchanged)."""
        px = round(self._left_inset_frac * self._gw.width())
        if px != self._applied_left_px:
            self._gw.ci.layout.setContentsMargins(px, 0, 0, 0)
            self._applied_left_px = px

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
        from PySide6.QtGui import QPainter, QColor
        from PySide6.QtCore import QRectF
        pm.fill(QColor(get_theme().background))
        painter = QPainter(pm)
        # Antialias the scene->pixmap scaling.  Without SmoothPixmapTransform the
        # ImageItem's drawImage falls back to nearest-neighbour, which aliases the
        # high-frequency reflectors into diagonal moire streaks when the image is
        # downscaled to screen.  This is the actual alias point (not the data
        # resampling upstream), so enabling smooth transform fixes the streaking.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # The pixmap is created at size*dpr physical px with setDevicePixelRatio(dpr),
        # so the QPainter operates in LOGICAL coordinates (dpr-aware) — it still
        # samples at full physical resolution.  The target rect must therefore be
        # the LOGICAL size (pm.width()/dpr), NOT the physical pixel count.  Passing
        # the physical size scaled the scene up by dpr and displaced it toward a
        # corner — invisible at dpr=1 but a gross mis-scale on HiDPI displays.
        dpr = pm.devicePixelRatio() or 1.0
        target_rect = QRectF(0, 0, pm.width() / dpr, pm.height() / dpr)
        scene_rect = self._gw.mapToScene(self._gw.viewport().rect()).boundingRect()
        self._gw.scene().render(painter, target_rect, scene_rect)
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
        # Keep the horizontal inset in sync with the current widget width so the
        # viewbox tracks the matplotlib main-axes rectangle across resizes.
        self._apply_left_inset()
        # Block signals so this doesn't recurse
        vb.blockSignals(True)
        vb.setXRange(xmin, xmax, padding=0)
        # invertY=True: setYRange(max, min) puts shallow (small) at top
        vb.setYRange(ymin, ymax, padding=0)
        vb.blockSignals(False)
