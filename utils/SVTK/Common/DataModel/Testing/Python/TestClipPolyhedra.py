#!/usr/bin/env python
import svtk
from svtk.util.misc import svtkGetDataRoot
SVTK_DATA_ROOT = svtkGetDataRoot()

# Create the RenderWindow, Renderer
#
ren = svtk.svtkRenderer()
renWin = svtk.svtkRenderWindow()
renWin.AddRenderer( ren )
renWin.SetSize(600,200)

iren = svtk.svtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Read input dataset that has n-faced polyhedra
reader = svtk.svtkExodusIIReader()
reader.SetFileName(str(SVTK_DATA_ROOT) + "/Data/cube-1.exo")
reader.Update()
dataset = reader.GetOutput()

# clip the dataset
clipper = svtk.svtkClipDataSet()
clipper.SetInputData(dataset.GetBlock(0).GetBlock(0))
plane = svtk.svtkPlane()
plane.SetNormal(0.5,0.5,0.5)
plane.SetOrigin(0.5,0.5,0.5)
clipper.SetClipFunction(plane)
clipper.Update()

# get surface representation to render
surfaceFilter = svtk.svtkDataSetSurfaceFilter()
surfaceFilter.SetInputData(clipper.GetOutput())
surfaceFilter.Update()
surface = surfaceFilter.GetOutput()

mapper = svtk.svtkPolyDataMapper()
mapper.SetInputData(surfaceFilter.GetOutput())
mapper.Update()

actor = svtk.svtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetRepresentationToSurface()
actor.GetProperty().EdgeVisibilityOn()

ren.AddActor(actor)

ren.GetActiveCamera().SetPosition(-0.5,0.5,0)
ren.GetActiveCamera().SetFocalPoint(0.5, 0.5, 0.5)
ren.GetActiveCamera().SetViewUp(0.0820, 0.934, -0.348)
ren.ResetCamera()
renWin.Render()
iren.Start()
