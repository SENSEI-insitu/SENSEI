#!/usr/bin/env python
import svtk
from svtk.util.misc import svtkGetDataRoot
SVTK_DATA_ROOT = svtkGetDataRoot()

# warp an image with a thin plate spline
# first, create an image to warp
imageGrid = svtk.svtkImageGridSource()
imageGrid.SetGridSpacing(16,16,0)
imageGrid.SetGridOrigin(0,0,0)
imageGrid.SetDataExtent(0,255,0,255,0,0)
imageGrid.SetDataScalarTypeToUnsignedChar()
table = svtk.svtkLookupTable()
table.SetTableRange(0,1)
table.SetValueRange(1.0,0.0)
table.SetSaturationRange(0.0,0.0)
table.SetHueRange(0.0,0.0)
table.SetAlphaRange(0.0,1.0)
table.Build()
alpha = svtk.svtkImageMapToColors()
alpha.SetInputConnection(imageGrid.GetOutputPort())
alpha.SetLookupTable(table)
reader1 = svtk.svtkBMPReader()
reader1.SetFileName("" + str(SVTK_DATA_ROOT) + "/Data/masonry.bmp")
blend = svtk.svtkImageBlend()
blend.AddInputConnection(0,reader1.GetOutputPort())
blend.AddInputConnection(0,alpha.GetOutputPort())
# next, create a ThinPlateSpline transform
p1 = svtk.svtkPoints()
p1.SetNumberOfPoints(8)
p1.SetPoint(0,0,0,0)
p1.SetPoint(1,0,255,0)
p1.SetPoint(2,255,0,0)
p1.SetPoint(3,255,255,0)
p1.SetPoint(4,96,96,0)
p1.SetPoint(5,96,159,0)
p1.SetPoint(6,159,159,0)
p1.SetPoint(7,159,96,0)
p2 = svtk.svtkPoints()
p2.SetNumberOfPoints(8)
p2.SetPoint(0,0,0,0)
p2.SetPoint(1,0,255,0)
p2.SetPoint(2,255,0,0)
p2.SetPoint(3,255,255,0)
p2.SetPoint(4,96,159,0)
p2.SetPoint(5,159,159,0)
p2.SetPoint(6,159,96,0)
p2.SetPoint(7,96,96,0)
transform = svtk.svtkThinPlateSplineTransform()
transform.SetSourceLandmarks(p2)
transform.SetTargetLandmarks(p1)
transform.SetBasisToR2LogR()
# you must invert the transform before passing it to svtkImageReslice
transform.Inverse()
reslice = svtk.svtkImageReslice()
reslice.SetInputConnection(blend.GetOutputPort())
reslice.SetResliceTransform(transform)
reslice.SetInterpolationModeToLinear()
# set the window/level to 255.0/127.5 to view full range
viewer = svtk.svtkImageViewer()
viewer.SetInputConnection(reslice.GetOutputPort())
viewer.SetColorWindow(255.0)
viewer.SetColorLevel(127.5)
viewer.SetZSlice(0)
viewer.Render()
# --- end of script --
