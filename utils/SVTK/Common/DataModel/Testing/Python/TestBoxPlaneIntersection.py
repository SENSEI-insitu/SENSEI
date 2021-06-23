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

# Create a cube and cut it with a plane
#
boxL = svtk.svtkCubeSource()
boxL.SetBounds(-2.5,-1.5, -0.5,0.5, -0.5,0.5)

boxC = svtk.svtkCubeSource()
boxC.SetBounds(-0.5,0.5, -0.5,0.5, -0.5,0.5)

boxR = svtk.svtkCubeSource()
boxR.SetBounds(1.5,2.5, -0.5,0.5, -0.5,0.5)

mapperL = svtk.svtkPolyDataMapper()
mapperL.SetInputConnection(boxL.GetOutputPort())

mapperC = svtk.svtkPolyDataMapper()
mapperC.SetInputConnection(boxC.GetOutputPort())

mapperR = svtk.svtkPolyDataMapper()
mapperR.SetInputConnection(boxR.GetOutputPort())

actorL = svtk.svtkActor()
actorL.SetMapper(mapperL)
actorL.GetProperty().SetRepresentationToWireframe()
actorL.GetProperty().SetAmbient(1)

actorC = svtk.svtkActor()
actorC.SetMapper(mapperC)
actorC.GetProperty().SetRepresentationToWireframe()
actorC.GetProperty().SetAmbient(1)

actorR = svtk.svtkActor()
actorR.SetMapper(mapperR)
actorR.GetProperty().SetRepresentationToWireframe()
actorR.GetProperty().SetAmbient(1)

ren.AddActor(actorL)
ren.AddActor(actorC)
ren.AddActor(actorR)

# Now clip boxes
origin = [0,0,0]
xout = [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
bds = [0,0, 0,0, 0,0]
clipBox = svtk.svtkBox()

# Left
normal = [1,1,1]
origin = boxL.GetCenter()
pdL = svtk.svtkPolyData()
polyL = svtk.svtkCellArray()
ptsL = svtk.svtkPoints()
pdL.SetPoints(ptsL)
pdL.SetPolys(polyL)
ptsL.SetDataTypeToDouble()
boxL.GetBounds(bds)
numInts = clipBox.IntersectWithPlane(bds, origin, normal, xout)
print("Num ints: ", numInts)
ptsL.SetNumberOfPoints(numInts)
polyL.InsertNextCell(numInts)
for i in range(0,numInts):
    ptsL.SetPoint(i,xout[3*i],xout[3*i+1],xout[3*i+2])
    polyL.InsertCellPoint(i)
mapperPL = svtk.svtkPolyDataMapper()
mapperPL.SetInputData(pdL)
actorPL = svtk.svtkActor()
actorPL.SetMapper(mapperPL)
ren.AddActor(actorPL)

# Center
normal = [.4,.8,.4]
origin = boxC.GetCenter()
pdC = svtk.svtkPolyData()
polyC = svtk.svtkCellArray()
ptsC = svtk.svtkPoints()
pdC.SetPoints(ptsC)
pdC.SetPolys(polyC)
ptsC.SetDataTypeToDouble()
boxC.GetBounds(bds)
numInts = clipBox.IntersectWithPlane(bds, origin, normal, xout)
print("Num ints: ", numInts)
ptsC.SetNumberOfPoints(numInts)
polyC.InsertNextCell(numInts)
for i in range(0,numInts):
    ptsC.SetPoint(i,xout[3*i],xout[3*i+1],xout[3*i+2])
    polyC.InsertCellPoint(i)
mapperPC = svtk.svtkPolyDataMapper()
mapperPC.SetInputData(pdC)
actorPC = svtk.svtkActor()
actorPC.SetMapper(mapperPC)
ren.AddActor(actorPC)

# Right
normal = [0,0,1]
origin = boxR.GetCenter()
pdR = svtk.svtkPolyData()
polyR = svtk.svtkCellArray()
ptsR = svtk.svtkPoints()
pdR.SetPoints(ptsR)
pdR.SetPolys(polyR)
ptsR.SetDataTypeToDouble()
boxR.GetBounds(bds)
numInts = clipBox.IntersectWithPlane(bds, origin, normal, xout)
print("Num ints: ", numInts)
ptsR.SetNumberOfPoints(numInts)
polyR.InsertNextCell(numInts)
for i in range(0,numInts):
    ptsR.SetPoint(i,xout[3*i],xout[3*i+1],xout[3*i+2])
    polyR.InsertCellPoint(i)
mapperPR = svtk.svtkPolyDataMapper()
mapperPR.SetInputData(pdR)
actorPR = svtk.svtkActor()
actorPR.SetMapper(mapperPR)
ren.AddActor(actorPR)

ren.GetActiveCamera().SetFocalPoint(0,0,0)
ren.GetActiveCamera().SetPosition(0,0.5,1)
ren.ResetCamera()
ren.GetActiveCamera().Zoom(2.5)

renWin.Render()
iren.Start()
