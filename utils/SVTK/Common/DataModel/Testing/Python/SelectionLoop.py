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

#
# Demonstrate the use of implicit selection loop as well as closest point
# connectivity
#
# create pipeline
#
sphere = svtk.svtkSphereSource()
sphere.SetRadius(1)
sphere.SetPhiResolution(100)
sphere.SetThetaResolution(100)
selectionPoints = svtk.svtkPoints()
selectionPoints.InsertPoint(0, 0.07325, 0.8417, 0.5612)
selectionPoints.InsertPoint(1, 0.07244, 0.6568, 0.7450)
selectionPoints.InsertPoint(2, 0.1727, 0.4597, 0.8850)
selectionPoints.InsertPoint(3, 0.3265, 0.6054, 0.7309)
selectionPoints.InsertPoint(4, 0.5722, 0.5848, 0.5927)
selectionPoints.InsertPoint(5, 0.4305, 0.8138, 0.4189)
loop = svtk.svtkImplicitSelectionLoop()
loop.SetLoop(selectionPoints)
extract = svtk.svtkExtractGeometry()
extract.SetInputConnection(sphere.GetOutputPort())
extract.SetImplicitFunction(loop)
connect = svtk.svtkConnectivityFilter()
connect.SetInputConnection(extract.GetOutputPort())
connect.SetExtractionModeToClosestPointRegion()
connect.SetClosestPoint(selectionPoints.GetPoint(0))
clipMapper = svtk.svtkDataSetMapper()
clipMapper.SetInputConnection(connect.GetOutputPort())
backProp = svtk.svtkProperty()
backProp.SetDiffuseColor(GetRGBColor('tomato'))
clipActor = svtk.svtkActor()
clipActor.SetMapper(clipMapper)
clipActor.GetProperty().SetColor(GetRGBColor('peacock'))
clipActor.SetBackfaceProperty(backProp)

# Create graphics stuff
#
ren1 = svtk.svtkRenderer()
renWin = svtk.svtkRenderWindow()
renWin.AddRenderer(ren1)
iren = svtk.svtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the renderer, set the background and size
#
ren1.AddActor(clipActor)
ren1.SetBackground(1, 1, 1)
ren1.ResetCamera()
ren1.GetActiveCamera().Azimuth(30)
ren1.GetActiveCamera().Elevation(30)
ren1.GetActiveCamera().Dolly(1.2)
ren1.ResetCameraClippingRange()

renWin.SetSize(400, 400)

renWin.Render()

# render the image
#

#iren.Start()
