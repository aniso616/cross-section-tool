from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class NavReadout(QLabel):
    """Coordinate readout strip — monospace, dim, no border."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("""
            color: rgba(120, 135, 145, 175);
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 9px;
            background: transparent;
            padding: 0px 4px;
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setText("x: —  |  depth: —  |  elev: —")

    def update_coords(self, x_m: float, depth_m: float, elev_m: float):
        self.setText(
            f"x: {x_m:,.0f} m  |  "
            f"depth: {depth_m:,.0f} m  |  "
            f"elev: {elev_m:+,.0f} m asl"
        )
