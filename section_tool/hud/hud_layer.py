from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


class HUDLayer(QWidget):
    """Transparent overlay: depth ruler, formation strip, scale bar, map inset,
    tool indicator, nav readout.

    WA_TransparentForMouseEvents passes all mouse/wheel events to the canvas.
    The command palette is a sibling of this widget (child of root), not here.
    """

    def __init__(self, parent, state=None, show_minimap: bool = True):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.command_palette = None   # wired externally
        self._state = state
        self._show_minimap = show_minimap
        self._init_children()

    def _init_children(self):
        from section_tool.hud.depth_ruler     import DepthRuler
        from section_tool.hud.formation_strip import FormationStrip
        from section_tool.hud.scale_bar       import ScaleBar
        from section_tool.hud.nav_readout     import NavReadout
        from section_tool.hud.tool_indicator  import ToolIndicator

        self.depth_ruler      = DepthRuler(self)
        self.formation_strip  = FormationStrip(self)
        self.scale_bar        = ScaleBar(self)
        self.nav_readout      = NavReadout(self)
        self.tool_indicator   = ToolIndicator(self)

        # Map inset only in full-screen game-UI mode; tiled layout uses a full map tile
        if self._show_minimap and self._state:
            from section_tool.hud.map_inset import MapInset
            self.map_inset = MapInset(self, self._state)
        else:
            self.map_inset = None

    def resizeEvent(self, event):
        self._layout_children()
        super().resizeEvent(event)

    def _layout_children(self):
        w, h  = self.width(), self.height()
        DR_W  = 52    # depth ruler
        FS_W  = 60    # formation strip
        MI_W  = 200   # map inset (only in minimap mode)
        MI_H  = 160
        SB_H  = 26    # scale bar
        NR_H  = 20    # nav readout
        M     = 10

        self.depth_ruler.setGeometry(0, 0, DR_W, h - SB_H)
        self.formation_strip.setGeometry(w - FS_W, 0, FS_W, h - SB_H)
        self.scale_bar.setGeometry(DR_W, h - SB_H, w - DR_W - FS_W, SB_H)
        if self.map_inset:
            self.map_inset.setGeometry(
                DR_W + M, h - MI_H - SB_H - M, MI_W, MI_H
            )
            nr_x = DR_W + MI_W + M * 2
            nr_w = w - DR_W - FS_W - MI_W - M * 3
        else:
            nr_x = DR_W + M
            nr_w = w - DR_W - FS_W - M * 2
        self.nav_readout.setGeometry(nr_x, h - NR_H - SB_H - M, nr_w, NR_H)
        self.tool_indicator.setGeometry(DR_W + M, M, 200, 34)

    def reconfigure_for_mode(self, mode):
        from section_tool.modes import Mode
        is_section = mode == Mode.SECTION
        self.depth_ruler.setVisible(is_section)
        self.formation_strip.setVisible(is_section)
        self.scale_bar.setVisible(is_section)
        if self.map_inset:
            self.map_inset.setVisible(True)  # always shown
