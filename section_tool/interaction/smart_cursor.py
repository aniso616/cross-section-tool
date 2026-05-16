from enum import Enum, auto

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCursor


class CursorContext(Enum):
    DEFAULT          = auto()
    HORIZON_BODY     = auto()
    HORIZON_ENDPOINT = auto()
    FAULT_BODY       = auto()
    FAULT_ENDPOINT   = auto()
    WELL_LOG         = auto()
    EMPTY_DRAW       = auto()
    EMPTY_DEFAULT    = auto()


CURSOR_SHAPES = {
    CursorContext.DEFAULT:          Qt.CursorShape.ArrowCursor,
    CursorContext.HORIZON_BODY:     Qt.CursorShape.OpenHandCursor,
    CursorContext.HORIZON_ENDPOINT: Qt.CursorShape.SizeHorCursor,
    CursorContext.FAULT_BODY:       Qt.CursorShape.OpenHandCursor,
    CursorContext.FAULT_ENDPOINT:   Qt.CursorShape.SizeAllCursor,
    CursorContext.WELL_LOG:         Qt.CursorShape.PointingHandCursor,
    CursorContext.EMPTY_DRAW:       Qt.CursorShape.CrossCursor,
    CursorContext.EMPTY_DEFAULT:    Qt.CursorShape.ArrowCursor,
}


class SmartCursor(QObject):
    """
    Tracks the object under the mouse and updates the cursor shape.

    scene must expose:
        .hit_test(x_m, z_m, radius_m) -> HitResult or None
        HitResult has: .type (str), .object, .subtype (str or None)

    Pass scene=None (or a stub) for stub mode; cursor defaults to arrow.
    """
    context_changed = Signal(object, object)  # CursorContext, hit or None

    HIT_RADIUS_PX = 8

    def __init__(self, canvas, scene=None):
        super().__init__(canvas)
        self.canvas       = canvas
        self.scene        = scene
        self._active_tool = None
        self._last_ctx    = CursorContext.DEFAULT
        self._last_hit    = None

    def set_active_tool(self, tool):
        self._active_tool = tool

    def update(self, canvas_pos, view_state):
        x_m, z_m = view_state.pixel_to_world(canvas_pos)
        radius_m  = view_state.pixels_to_metres(self.HIT_RADIUS_PX)
        hit       = self._hit_test(x_m, z_m, radius_m)
        ctx       = self._resolve(hit)

        if ctx != self._last_ctx or hit != self._last_hit:
            self._last_ctx = ctx
            self._last_hit = hit
            self.canvas.setCursor(QCursor(CURSOR_SHAPES[ctx]))
            self.context_changed.emit(ctx, hit)

    def _hit_test(self, x_m, z_m, radius_m):
        if self.scene is None:
            return None
        try:
            return self.scene.hit_test(x_m, z_m, radius_m)
        except Exception:
            return None

    def _resolve(self, hit):
        if hit is None:
            if self._active_tool in ("horizon", "fault", "pick", "annotation"):
                return CursorContext.EMPTY_DRAW
            return CursorContext.EMPTY_DEFAULT
        t, s = hit.type, hit.subtype
        if t == "horizon":
            return (CursorContext.HORIZON_ENDPOINT if s == "endpoint"
                    else CursorContext.HORIZON_BODY)
        if t == "fault":
            return (CursorContext.FAULT_ENDPOINT if s == "endpoint"
                    else CursorContext.FAULT_BODY)
        if t == "well_log":
            return CursorContext.WELL_LOG
        return CursorContext.DEFAULT

    @property
    def current_hit(self):
        return self._last_hit
