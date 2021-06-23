#!/usr/bin/env python
import svtk
from svtk.util.misc import svtkGetDataRoot
SVTK_DATA_ROOT = svtkGetDataRoot()

def GetRGBColor(colorName):
    '''
        Return the red, green and blue components for a
        color as doubles.
    '''
    rgb = [0.0, 0.0, 0.0]  # black
    svtk.svtkNamedColors().GetColorRGB(colorName, rgb)
    return rgb


# Create the RenderWindow, Renderer and both Actors
#
ren1 = svtk.svtkRenderer()
renWin = svtk.svtkRenderWindow()
renWin.AddRenderer(ren1)
iren = svtk.svtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
lgt = svtk.svtkLight()
# create pipeline
#
locator = svtk.svtkMergePoints()
locator.SetDivisions(32, 32, 46)
locator.SetNumberOfPointsPerBucket(2)
locator.AutomaticOff()

v16 = svtk.svtkVolume16Reader()
v16.SetDataDimensions(64, 64)
v16.GetOutput().SetOrigin(0.0, 0.0, 0.0)
v16.SetDataByteOrderToLittleEndian()
v16.SetFilePrefix(SVTK_DATA_ROOT + "/Data/headsq/quarter")
v16.SetImageRange(1, 93)
v16.SetDataSpacing(3.2, 3.2, 1.5)

iso = svtk.svtkMarchingCubes()
iso.SetInputConnection(v16.GetOutputPort())
iso.SetValue(0, 1150)
iso.ComputeGradientsOn()
iso.ComputeScalarsOff()
iso.SetLocator(locator)

gradient = svtk.svtkVectorNorm()
gradient.SetInputConnection(iso.GetOutputPort())

isoMapper = svtk.svtkDataSetMapper()
isoMapper.SetInputConnection(gradient.GetOutputPort())
isoMapper.ScalarVisibilityOn()
isoMapper.SetScalarRange(0, 1200)

isoActor = svtk.svtkActor()
isoActor.SetMapper(isoMapper)

isoProp = isoActor.GetProperty()
isoProp.SetColor(GetRGBColor('antique_white'))

outline = svtk.svtkOutlineFilter()
outline.SetInputConnection(v16.GetOutputPort())

outlineMapper = svtk.svtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())

outlineActor = svtk.svtkActor()
outlineActor.SetMapper(outlineMapper)

outlineProp = outlineActor.GetProperty()
# outlineProp.SetColor(0, 0, 0)

# Add the actors to the renderer, set the background and size
#
ren1.AddActor(outlineActor)
ren1.AddActor(isoActor)
ren1.SetBackground(1, 1, 1)
ren1.AddLight(lgt)

renWin.SetSize(250, 250)

ren1.SetBackground(0.1, 0.2, 0.4)
ren1.ResetCamera()

cam1 = ren1.GetActiveCamera()
cam1.Elevation(90)
cam1.SetViewUp(0, 0, -1)
cam1.Zoom(1.5)

lgt.SetPosition(cam1.GetPosition())
lgt.SetFocalPoint(cam1.GetFocalPoint())

ren1.ResetCameraClippingRange()

# render the image
#
renWin.Render()

#iren.Start()
