#!/usr/bin/env python
import svtk

# create points in the configuration of an octant with one 2:1 face
#
points = svtk.svtkPoints()
aConvex = svtk.svtkConvexPointSet()
points.InsertPoint(0, 0, 0, 0)
points.InsertPoint(1, 1, 0, 0)
points.InsertPoint(2, 1, 1, 0)
points.InsertPoint(3, 0, 1, 0)
points.InsertPoint(4, 0, 0, 1)
points.InsertPoint(5, 1, 0, 1)
points.InsertPoint(6, 1, 1, 1)
points.InsertPoint(7, 0, 1, 1)
points.InsertPoint(8, 0.5, 0, 0)
points.InsertPoint(9, 1, 0.5, 0)
points.InsertPoint(10, 0.5, 1, 0)
points.InsertPoint(11, 0, 0.5, 0)
points.InsertPoint(12, 0.5, 0.5, 0)
i = 0
while i < 13:
    aConvex.GetPointIds().InsertId(i, i)
    i += 1

aConvexGrid = svtk.svtkUnstructuredGrid()
aConvexGrid.Allocate(1, 1)
aConvexGrid.InsertNextCell(aConvex.GetCellType(), aConvex.GetPointIds())
aConvexGrid.SetPoints(points)

# Display the cell
dsm = svtk.svtkDataSetMapper()
dsm.SetInputData(aConvexGrid)
a = svtk.svtkActor()
a.SetMapper(dsm)
a.GetProperty().SetColor(0, 1, 0)

# Contour and clip the cell with elevation scalars
ele = svtk.svtkElevationFilter()
ele.SetInputData(aConvexGrid)
ele.SetLowPoint(-1, -1, -1)
ele.SetHighPoint(1, 1, 1)
ele.SetScalarRange(-1, 1)

# Clip
#
clip = svtk.svtkClipDataSet()
clip.SetInputConnection(ele.GetOutputPort())
clip.SetValue(0.5)
g = svtk.svtkDataSetSurfaceFilter()
g.SetInputConnection(clip.GetOutputPort())
map = svtk.svtkPolyDataMapper()
map.SetInputConnection(g.GetOutputPort())
map.ScalarVisibilityOff()
clipActor = svtk.svtkActor()
clipActor.SetMapper(map)
clipActor.GetProperty().SetColor(1, 0, 0)
clipActor.AddPosition(2, 0, 0)

# Contour
#
contour = svtk.svtkContourFilter()
contour.SetInputConnection(ele.GetOutputPort())
contour.SetValue(0, 0.5)
g2 = svtk.svtkDataSetSurfaceFilter()
g2.SetInputConnection(contour.GetOutputPort())
map2 = svtk.svtkPolyDataMapper()
map2.SetInputConnection(g2.GetOutputPort())
map2.ScalarVisibilityOff()
contourActor = svtk.svtkActor()
contourActor.SetMapper(map2)
contourActor.GetProperty().SetColor(1, 0, 0)
contourActor.AddPosition(1, 2, 0)

# Create graphics stuff
#
ren1 = svtk.svtkRenderer()
renWin = svtk.svtkRenderWindow()
renWin.AddRenderer(ren1)
iren = svtk.svtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the renderer, set the background and size
#
ren1.AddActor(a)
ren1.AddActor(clipActor)
ren1.AddActor(contourActor)
ren1.SetBackground(1, 1, 1)

renWin.SetSize(250, 150)

aCam = svtk.svtkCamera()
aCam.SetFocalPoint(1.38705, 1.37031, 0.639901)
aCam.SetPosition(1.89458, -5.07106, -4.17439)
aCam.SetViewUp(0.00355726, 0.598843, -0.800858)
aCam.SetClippingRange(4.82121, 12.1805)

ren1.SetActiveCamera(aCam)

renWin.Render()

cam1 = ren1.GetActiveCamera()
cam1.Zoom(1.5)

# render the image
#
renWin.Render()

#iren.Start()
