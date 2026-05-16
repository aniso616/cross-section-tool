from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class CoordReadout(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("""
            color: rgba(170, 170, 170, 160);
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 11px;
            background: transparent;
            padding: 2px 6px;
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setText("x: —  |  depth: —  |  elev: —")

    def update_coords(self, x_m: float, depth_m: float, elev_m: float):
        self.setText(
            f"x: {x_m:,.0f} E  |  depth: {depth_m:,.0f} m  |  "
            f"elev: {elev_m:+,.0f} m asl"
        )
