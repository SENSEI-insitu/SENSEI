#!/usr/bin/env python
import svtk
from svtk.util.misc import svtkGetDataRoot
SVTK_DATA_ROOT = svtkGetDataRoot()

# This example demonstrates adding two implicit models
# to produce an (unexpected!) result
# first we load in the standard svtk packages into tcl
geomObject1 = svtk.svtkCone()
geomObject2 = svtk.svtkSphere()
geomObject2.SetRadius(0.5)
geomObject2.SetCenter(0.5,0,0)
sum = svtk.svtkImplicitSum()
sum.SetNormalizeByWeight(1)
sum.AddFunction(geomObject1,2)
sum.AddFunction(geomObject2,1)
sample = svtk.svtkSampleFunction()
sample.SetImplicitFunction(sum)
sample.SetSampleDimensions(60,60,60)
sample.ComputeNormalsOn()
surface = svtk.svtkContourFilter()
surface.SetInputConnection(sample.GetOutputPort())
surface.SetValue(0,0.0)
mapper = svtk.svtkPolyDataMapper()
mapper.SetInputConnection(surface.GetOutputPort())
mapper.ScalarVisibilityOff()
actor = svtk.svtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetDiffuseColor(0.2,0.4,0.6)
actor.GetProperty().SetSpecular(0.4)
actor.GetProperty().SetDiffuse(0.7)
actor.GetProperty().SetSpecularPower(40)
ren1 = svtk.svtkRenderer()
renWin = svtk.svtkRenderWindow()
renWin.AddRenderer(ren1)
iren = svtk.svtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
# Add the actors to the renderer, set the background and size
#
ren1.AddActor(actor)
ren1.SetBackground(1,1,1)
renWin.SetSize(300,300)
ren1.ResetCamera()
ren1.GetActiveCamera().Azimuth(60)
ren1.GetActiveCamera().Elevation(-10)
ren1.GetActiveCamera().Dolly(1.5)
ren1.ResetCameraClippingRange()
iren.Initialize()
# prevent the tk window from showing up then start the event loop
# render the image
#
renWin.Render()
# --- end of script --
