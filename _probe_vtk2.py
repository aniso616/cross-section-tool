import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
app = QApplication.instance() or QApplication(sys.argv[:1])

import pyvista as pv
import numpy as np
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Build widget
w = QWidget()
layout = QVBoxLayout(w)
layout.setContentsMargins(0,0,0,0)
vtk_widget = QVTKRenderWindowInteractor(w)
layout.addWidget(vtk_widget)

# Attach pyvista plotter to vtk_widget render window
plotter = pv.Plotter(
    window_size=[800, 600],
    off_screen=False,
)
# Use the render window from the VTK widget
render_window = vtk_widget.GetRenderWindow()
plotter.ren_win = render_window

# Add content
sphere = pv.Sphere()
plotter.add_mesh(sphere, color="white", show_edges=True)
plotter.add_axes()

# Can we add a PolyData line?
pts = np.array([[0,0,0],[0,0,-1000],[100,0,-1000]], dtype=float)
lines = np.array([3,0,1,2])
poly = pv.PolyData(pts, lines=lines)
plotter.add_mesh(poly, color="brown", line_width=3)

# Screenshot works for testing
plotter.show(auto_close=False)
img = plotter.screenshot(return_img=True)
print("screenshot shape:", img.shape)

# Check render methods
plotter.reset_camera()
print("reset_camera: OK")
plotter.render()
print("render: OK")

plotter.close()
print("all OK")
