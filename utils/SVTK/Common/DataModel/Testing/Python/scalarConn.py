#!/usr/bin/env python
import svtk
from svtk.util.misc import svtkGetDataRoot
SVTK_DATA_ROOT = svtkGetDataRoot()

# Quadric definition
quadric = svtk.svtkQuadric()
quadric.SetCoefficients([.5,1,.2,0,.1,0,0,.2,0,0])
sample = svtk.svtkSampleFunction()
sample.SetSampleDimensions(30,30,30)
sample.SetImplicitFunction(quadric)
sample.Update()
#sample Print
sample.ComputeNormalsOff()
# Extract cells that contains isosurface of interest
conn = svtk.svtkConnectivityFilter()
conn.SetInputConnection(sample.GetOutputPort())
conn.ScalarConnectivityOn()
conn.SetScalarRange(0.6,0.6)
conn.SetExtractionModeToCellSeededRegions()
conn.AddSeed(105)
# Create a surface
contours = svtk.svtkContourFilter()
contours.SetInputConnection(conn.GetOutputPort())
#  contours SetInputConnection [sample GetOutputPort]
contours.GenerateValues(5,0.0,1.2)
contMapper = svtk.svtkDataSetMapper()
#  contMapper SetInputConnection [contours GetOutputPort]
contMapper.SetInputConnection(conn.GetOutputPort())
contMapper.SetScalarRange(0.0,1.2)
contActor = svtk.svtkActor()
contActor.SetMapper(contMapper)
# Create outline
outline = svtk.svtkOutlineFilter()
outline.SetInputConnection(sample.GetOutputPort())
outlineMapper = svtk.svtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())
outlineActor = svtk.svtkActor()
outlineActor.SetMapper(outlineMapper)
outlineActor.GetProperty().SetColor(0,0,0)
# Graphics
# create a window to render into
ren1 = svtk.svtkRenderer()
renWin = svtk.svtkRenderWindow()
renWin.SetMultiSamples(0)
renWin.AddRenderer(ren1)
# create a renderer
# interactiver renderer catches mouse events (optional)
iren = svtk.svtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
ren1.SetBackground(1,1,1)
ren1.AddActor(contActor)
ren1.AddActor(outlineActor)
ren1.ResetCamera()
ren1.GetActiveCamera().Zoom(1.4)
iren.Initialize()
# --- end of script --
