"""Arrow-key navigator for the section canvas.

Navigation keys:
  Arrow keys          — pan  (continuous, accelerates on hold)
  Shift + Up/Down     — zoom in / out
  Scroll wheel        — zoom centred on cursor (handled by matplotlib)
"""
from PySide6.QtCore import QEvent, QObject, QTimer, Qt

_NAV_KEYS = {
    Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right,
}


class WASDNavigator(QObject):
    """
    Continuous pan and zoom from held arrow keys.
    Attach with: canvas.installEventFilter(WASDNavigator(canvas, view_state))

    view_state must expose:
        .x_min, .x_max  (horizontal extent, metres)
        .z_min, .z_max  (vertical extent, metres depth)
        .pan(dx_m, dz_m)
        .zoom(factor, center_x_m, center_z_m)
    """

    PAN_RATE  = 0.015   # fraction of view extent per tick
    ZOOM_RATE = 0.012
    TICK_MS   = 16      # ~60 fps

    def __init__(self, canvas, view_state):
        super().__init__(canvas)
        self.canvas = canvas
        self.view   = view_state
        self._held  = set()
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.Type.KeyPress and not event.isAutoRepeat():
            key = event.key()
            if key in _NAV_KEYS:
                self._held.add(key)
                if not self._timer.isActive():
                    self._timer.start()
                return True   # consume arrow keys so they don't scroll the view
            return False      # all other keys pass through unchanged

        if t == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            key = event.key()
            if key in _NAV_KEYS:
                self._held.discard(key)
                if not self._nav_keys():
                    self._timer.stop()
                return True
            return False

        return False

    def _nav_keys(self):
        return bool(self._held & _NAV_KEYS)

    def _tick(self):
        xext = self.view.x_max - self.view.x_min
        zext = self.view.z_max - self.view.z_min
        shift = Qt.Key.Key_Shift in self._held

        dx, dz, zf = 0.0, 0.0, 1.0

        if Qt.Key.Key_Left  in self._held: dx = -xext * self.PAN_RATE
        if Qt.Key.Key_Right in self._held: dx = +xext * self.PAN_RATE
        if Qt.Key.Key_Up    in self._held:
            if shift: zf *= (1.0 - self.ZOOM_RATE)
            else:     dz  = -zext * self.PAN_RATE
        if Qt.Key.Key_Down  in self._held:
            if shift: zf *= (1.0 + self.ZOOM_RATE)
            else:     dz  = +zext * self.PAN_RATE

        if dx or dz:
            self.view.pan(dx, dz)
        if zf != 1.0:
            cx = (self.view.x_min + self.view.x_max) / 2
            cz = (self.view.z_min + self.view.z_max) / 2
            self.view.zoom(zf, cx, cz)
