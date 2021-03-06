#!/usr/bin/env python
import svtk
from svtk.util.misc import svtkGetDataRoot
SVTK_DATA_ROOT = svtkGetDataRoot()

# This example demonstrates how to use a matrix in place of a transform
# via svtkMatrixToLinearTransform and svtkMatrixToHomogeneousTransform.
# create a rendering window
renWin = svtk.svtkRenderWindow()
renWin.SetSize(600,300)
# set up first set of polydata
p1 = svtk.svtkPlaneSource()
p1.SetOrigin(0.5,0.508,-0.5)
p1.SetPoint1(-0.5,0.508,-0.5)
p1.SetPoint2(0.5,0.508,0.5)
p1.SetXResolution(5)
p1.SetYResolution(5)
p2 = svtk.svtkPlaneSource()
p2.SetOrigin(-0.508,0.5,-0.5)
p2.SetPoint1(-0.508,-0.5,-0.5)
p2.SetPoint2(-0.508,0.5,0.5)
p2.SetXResolution(5)
p2.SetYResolution(5)
p3 = svtk.svtkPlaneSource()
p3.SetOrigin(-0.5,-0.508,-0.5)
p3.SetPoint1(0.5,-0.508,-0.5)
p3.SetPoint2(-0.5,-0.508,0.5)
p3.SetXResolution(5)
p3.SetYResolution(5)
p4 = svtk.svtkPlaneSource()
p4.SetOrigin(0.508,-0.5,-0.5)
p4.SetPoint1(0.508,0.5,-0.5)
p4.SetPoint2(0.508,-0.5,0.5)
p4.SetXResolution(5)
p4.SetYResolution(5)
p5 = svtk.svtkPlaneSource()
p5.SetOrigin(0.5,0.5,-0.508)
p5.SetPoint1(0.5,-0.5,-0.508)
p5.SetPoint2(-0.5,0.5,-0.508)
p5.SetXResolution(5)
p5.SetYResolution(5)
p6 = svtk.svtkPlaneSource()
p6.SetOrigin(0.5,0.5,0.508)
p6.SetPoint1(-0.5,0.5,0.508)
p6.SetPoint2(0.5,-0.5,0.508)
p6.SetXResolution(5)
p6.SetYResolution(5)
# append together
ap = svtk.svtkAppendPolyData()
ap.AddInputConnection(p1.GetOutputPort())
ap.AddInputConnection(p2.GetOutputPort())
ap.AddInputConnection(p3.GetOutputPort())
ap.AddInputConnection(p4.GetOutputPort())
ap.AddInputConnection(p5.GetOutputPort())
ap.AddInputConnection(p6.GetOutputPort())
#--------------------------
# linear transform matrix
t1 = svtk.svtkMatrixToLinearTransform()
m1 = svtk.svtkMatrix4x4()
t1.SetInput(m1)
m1.SetElement(0,0,1.127631)
m1.SetElement(0,1,0.205212)
m1.SetElement(0,2,-0.355438)
m1.SetElement(1,0,0.000000)
m1.SetElement(1,1,0.692820)
m1.SetElement(1,2,0.400000)
m1.SetElement(2,0,0.200000)
m1.SetElement(2,1,-0.469846)
m1.SetElement(2,2,0.813798)
f11 = svtk.svtkTransformPolyDataFilter()
f11.SetInputConnection(ap.GetOutputPort())
f11.SetTransform(t1)
m11 = svtk.svtkDataSetMapper()
m11.SetInputConnection(f11.GetOutputPort())
a11 = svtk.svtkActor()
a11.SetMapper(m11)
a11.GetProperty().SetColor(1,0,0)
a11.GetProperty().SetRepresentationToWireframe()
ren11 = svtk.svtkRenderer()
ren11.SetViewport(0.0,0.5,0.25,1.0)
ren11.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren11.AddActor(a11)
renWin.AddRenderer(ren11)
# inverse identity transform
f12 = svtk.svtkTransformPolyDataFilter()
f12.SetInputConnection(ap.GetOutputPort())
f12.SetTransform(t1.GetInverse())
m12 = svtk.svtkDataSetMapper()
m12.SetInputConnection(f12.GetOutputPort())
a12 = svtk.svtkActor()
a12.SetMapper(m12)
a12.GetProperty().SetColor(0.9,0.9,0)
a12.GetProperty().SetRepresentationToWireframe()
ren12 = svtk.svtkRenderer()
ren12.SetViewport(0.0,0.0,0.25,0.5)
ren12.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren12.AddActor(a12)
renWin.AddRenderer(ren12)
#--------------------------
# perspective transform matrix
m2 = svtk.svtkMatrix4x4()
m2.SetElement(3,0,-0.11)
m2.SetElement(3,1,0.3)
m2.SetElement(3,2,0.2)
t2 = svtk.svtkMatrixToHomogeneousTransform()
t2.SetInput(m2)
f21 = svtk.svtkTransformPolyDataFilter()
f21.SetInputConnection(ap.GetOutputPort())
f21.SetTransform(t2)
m21 = svtk.svtkDataSetMapper()
m21.SetInputConnection(f21.GetOutputPort())
a21 = svtk.svtkActor()
a21.SetMapper(m21)
a21.GetProperty().SetColor(1,0,0)
a21.GetProperty().SetRepresentationToWireframe()
ren21 = svtk.svtkRenderer()
ren21.SetViewport(0.25,0.5,0.50,1.0)
ren21.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren21.AddActor(a21)
renWin.AddRenderer(ren21)
# inverse linear transform
f22 = svtk.svtkTransformPolyDataFilter()
f22.SetInputConnection(ap.GetOutputPort())
f22.SetTransform(t2.GetInverse())
m22 = svtk.svtkDataSetMapper()
m22.SetInputConnection(f22.GetOutputPort())
a22 = svtk.svtkActor()
a22.SetMapper(m22)
a22.GetProperty().SetColor(0.9,0.9,0)
a22.GetProperty().SetRepresentationToWireframe()
ren22 = svtk.svtkRenderer()
ren22.SetViewport(0.25,0.0,0.50,0.5)
ren22.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren22.AddActor(a22)
renWin.AddRenderer(ren22)
#--------------------------
# linear concatenation - should end up with identity here
t3 = svtk.svtkTransform()
t3.Concatenate(t1)
t3.Concatenate(t1.GetInverse())
f31 = svtk.svtkTransformPolyDataFilter()
f31.SetInputConnection(ap.GetOutputPort())
f31.SetTransform(t3)
m31 = svtk.svtkDataSetMapper()
m31.SetInputConnection(f31.GetOutputPort())
a31 = svtk.svtkActor()
a31.SetMapper(m31)
a31.GetProperty().SetColor(1,0,0)
a31.GetProperty().SetRepresentationToWireframe()
ren31 = svtk.svtkRenderer()
ren31.SetViewport(0.50,0.5,0.75,1.0)
ren31.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren31.AddActor(a31)
renWin.AddRenderer(ren31)
# inverse linear transform
f32 = svtk.svtkTransformPolyDataFilter()
f32.SetInputConnection(ap.GetOutputPort())
f32.SetTransform(t3.GetInverse())
m32 = svtk.svtkDataSetMapper()
m32.SetInputConnection(f32.GetOutputPort())
a32 = svtk.svtkActor()
a32.SetMapper(m32)
a32.GetProperty().SetColor(0.9,0.9,0)
a32.GetProperty().SetRepresentationToWireframe()
ren32 = svtk.svtkRenderer()
ren32.SetViewport(0.5,0.0,0.75,0.5)
ren32.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren32.AddActor(a32)
renWin.AddRenderer(ren32)
#--------------------------
# perspective transform concatenation
t4 = svtk.svtkPerspectiveTransform()
t4.Concatenate(t1)
t4.Concatenate(t2)
t4.Concatenate(t3)
f41 = svtk.svtkTransformPolyDataFilter()
f41.SetInputConnection(ap.GetOutputPort())
f41.SetTransform(t4)
m41 = svtk.svtkDataSetMapper()
m41.SetInputConnection(f41.GetOutputPort())
a41 = svtk.svtkActor()
a41.SetMapper(m41)
a41.GetProperty().SetColor(1,0,0)
a41.GetProperty().SetRepresentationToWireframe()
ren41 = svtk.svtkRenderer()
ren41.SetViewport(0.75,0.5,1.0,1.0)
ren41.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren41.AddActor(a41)
renWin.AddRenderer(ren41)
# inverse of transform concatenation
f42 = svtk.svtkTransformPolyDataFilter()
f42.SetInputConnection(ap.GetOutputPort())
f42.SetTransform(t4.GetInverse())
m42 = svtk.svtkDataSetMapper()
m42.SetInputConnection(f42.GetOutputPort())
a42 = svtk.svtkActor()
a42.SetMapper(m42)
a42.GetProperty().SetColor(0.9,0.9,0)
a42.GetProperty().SetRepresentationToWireframe()
ren42 = svtk.svtkRenderer()
ren42.SetViewport(0.75,0.0,1.0,0.5)
ren42.ResetCamera(-0.5,0.5,-0.5,0.5,-1,1)
ren42.AddActor(a42)
renWin.AddRenderer(ren42)
renWin.SetMultiSamples(0)
renWin.Render()
# --- end of script --
