from PySide6.QtCore import QEvent, QObject, QTimer, Qt


class WASDNavigator(QObject):
    """
    Continuous pan and zoom from held WASD + Shift keys.
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
            self._held.add(event.key())
            if self._nav_keys() and not self._timer.isActive():
                self._timer.start()
            return False  # don't consume — tool keys still reach canvas

        if t == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self._held.discard(event.key())
            if not self._nav_keys():
                self._timer.stop()
            return False

        return False

    def _nav_keys(self):
        return bool(self._held & {Qt.Key.Key_W, Qt.Key.Key_A,
                                  Qt.Key.Key_S, Qt.Key.Key_D})

    def _tick(self):
        xext = self.view.x_max - self.view.x_min
        zext = self.view.z_max - self.view.z_min
        shift = Qt.Key.Key_Shift in self._held

        dx, dz, zf = 0.0, 0.0, 1.0

        if Qt.Key.Key_A in self._held: dx = -xext * self.PAN_RATE
        if Qt.Key.Key_D in self._held: dx = +xext * self.PAN_RATE
        if Qt.Key.Key_W in self._held:
            if shift: zf *= (1.0 - self.ZOOM_RATE)
            else:     dz  = -zext * self.PAN_RATE
        if Qt.Key.Key_S in self._held:
            if shift: zf *= (1.0 + self.ZOOM_RATE)
            else:     dz  = +zext * self.PAN_RATE

        if dx or dz:
            self.view.pan(dx, dz)
        if zf != 1.0:
            cx = (self.view.x_min + self.view.x_max) / 2
            cz = (self.view.z_min + self.view.z_max) / 2
            self.view.zoom(zf, cx, cz)
