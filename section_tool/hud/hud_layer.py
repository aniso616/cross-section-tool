from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


class HUDLayer(QWidget):
    """Transparent display-only overlay: tool pill, coord readout, depth scale.

    WA_TransparentForMouseEvents is True so mouse/wheel events fall through to
    the canvas below.  The minimap is now a MinimapOverlay child of the section
    view (game-HUD style); this layer no longer carries it.
    The command palette lives outside this widget (sibling of canvas_stack).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.command_palette = None  # wired externally by SectionMainWindow
        self._init_children()

    def _init_children(self):
        from section_tool.hud.depth_scale    import DepthScale
        from section_tool.hud.strat_column   import StratColumn
        from section_tool.hud.coord_readout  import CoordReadout
        from section_tool.hud.tool_indicator import ToolIndicator

        self.depth_scale    = DepthScale(self)
        self.strat_column   = StratColumn(self)
        self.coord_readout  = CoordReadout(self)
        self.tool_indicator = ToolIndicator(self)

    def resizeEvent(self, event):
        self._layout_children()
        super().resizeEvent(event)

    def _layout_children(self):
        w, h    = self.width(), self.height()
        SCALE   = 44   # left depth scale width
        STRAT   = 64   # right strat column width
        COORD_H = 22
        IND_W   = 128
        IND_H   = 28
        M       = 12

        self.depth_scale.setGeometry(0, 0, SCALE, h)
        self.strat_column.setGeometry(w - STRAT, 0, STRAT, h)
        self.coord_readout.setGeometry(
            SCALE + M,
            h - COORD_H - M,
            w - SCALE - STRAT - M * 2,
            COORD_H,
        )
        self.tool_indicator.setGeometry(SCALE + M, M, IND_W, IND_H)

    def reconfigure_for_mode(self, mode):
        from section_tool.modes import Mode
        self.depth_scale.setVisible(mode == Mode.SECTION)
        self.strat_column.setVisible(mode == Mode.SECTION)
