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
        """Section readout: distance / depth / elevation."""
        self.setText(
            f"x: {x_m:,.0f} m  |  "
            f"depth: {depth_m:,.0f} m  |  "
            f"elev: {elev_m:+,.0f} m asl"
        )

    def update_map_coords(self, e: float, n: float,
                          lon: float | None = None, lat: float | None = None):
        """Map readout: easting / northing, with lat/long when a CRS is known."""
        txt = f"E: {e:,.0f}  |  N: {n:,.0f}"
        if lon is not None and lat is not None:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            txt += f"  |  {abs(lat):.2f}°{ns} {abs(lon):.2f}°{ew}"
        self.setText(txt)
