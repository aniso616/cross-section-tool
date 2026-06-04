"""View3DTile — full-size 3D viewer tile (provisional slot in center splitter).

Wraps the existing live Viewer3D (do NOT build a second PyVista scene); the
viewer's OpenGL context is created lazily when the user clicks "Enable 3D
View", so reparenting the wrapper here is safe — there is no live GL context
to move until after the tile is in place.
"""
from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget


class View3DTile(QWidget):
    """Center-column tile hosting the perspective 3D viewer."""

    def __init__(self, viewer_3d: QWidget, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 180)
        self.setStyleSheet("background: #0e1014;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        viewer_3d.setParent(self)
        layout.addWidget(viewer_3d)
        self.viewer = viewer_3d   # alias
