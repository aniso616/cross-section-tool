from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


class HUDLayer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self._init_children()

    def _init_children(self):
        from section_tool.hud.depth_scale     import DepthScale
        from section_tool.hud.strat_column    import StratColumn
        from section_tool.hud.minimap         import Minimap
        from section_tool.hud.coord_readout   import CoordReadout
        from section_tool.hud.tool_indicator  import ToolIndicator
        from section_tool.hud.command_palette import CommandPalette

        self.depth_scale     = DepthScale(self)
        self.strat_column    = StratColumn(self)
        self.minimap         = Minimap(self)
        self.coord_readout   = CoordReadout(self)
        self.tool_indicator  = ToolIndicator(self)
        self.command_palette = CommandPalette(self)
        self.command_palette.hide()

    def resizeEvent(self, event):
        self._layout_children()
        super().resizeEvent(event)

    def _layout_children(self):
        w, h    = self.width(), self.height()
        SCALE   = 44
        STRAT   = 64
        MM_W    = 200
        MM_H    = 160
        COORD_H = 22
        IND_W   = 128
        IND_H   = 28
        M       = 12

        self.depth_scale.setGeometry(0, 0, SCALE, h)
        self.strat_column.setGeometry(w - STRAT, 0, STRAT, h)
        self.minimap.setGeometry(SCALE + M, h - MM_H - M, MM_W, MM_H)
        self.coord_readout.setGeometry(
            SCALE + MM_W + M * 2,
            h - COORD_H - M,
            w - SCALE - STRAT - MM_W - M * 3,
            COORD_H,
        )
        self.tool_indicator.setGeometry(SCALE + M, M, IND_W, IND_H)

    def reconfigure_for_mode(self, mode):
        from section_tool.modes import Mode
        self.depth_scale.setVisible(mode == Mode.SECTION)
        self.strat_column.setVisible(mode == Mode.SECTION)
        labels = {
            Mode.SECTION: "Map",
            Mode.MAP:     "Section",
            Mode.THREE_D: "Section / Map",
        }
        self.minimap.set_label(labels[mode])
