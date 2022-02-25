#!/usr/bin/env python
# -*- coding: utf-8 -*-

import svtk
import sys

# Test speed of compute bounds in svtkPolyData, svtkPoints, and
# svtkBoundingBox.

# Control model size
res = 500
timer = svtk.svtkTimerLog()

# Uncomment if you want to use as a little interactive program
#if len(sys.argv) >= 2 :
#    res = int(sys.argv[1])
#else:
#    res = 500

# Data source. Note that different types of cells are created
# to exercise the svtkPolyData::GetBounds() properly.
plane = svtk.svtkPlaneSource()
plane.SetResolution(res,res)

edges = svtk.svtkFeatureEdges()
edges.SetInputConnection(plane.GetOutputPort())
#edges.ExtractAllEdgeTypesOff()
edges.BoundaryEdgesOn()
edges.ManifoldEdgesOff()
edges.NonManifoldEdgesOff()
edges.FeatureEdgesOff()

t1 = svtk.svtkTransform()
t1.Translate(-1.0,0,0)
tf1 = svtk.svtkTransformPolyDataFilter()
tf1.SetInputConnection(edges.GetOutputPort())
tf1.SetTransform(t1)

t2 = svtk.svtkTransform()
t2.Translate(1.0,0,0)
tf2 = svtk.svtkTransformPolyDataFilter()
tf2.SetInputConnection(edges.GetOutputPort())
tf2.SetTransform(t2)

append = svtk.svtkAppendPolyData()
append.AddInputConnection(tf1.GetOutputPort())
append.AddInputConnection(plane.GetOutputPort())
append.AddInputConnection(tf2.GetOutputPort())
append.Update()

output = append.GetOutput()
points = output.GetPoints()
box = [0.0,0.0,0.0,0.0,0.0,0.0]

print("Input data:")
print("\tNum Points: {0}".format(output.GetNumberOfPoints()))
print("\tNum Cells: {0}".format(output.GetNumberOfCells()))

# Currently svtkPolyData takes into account cells that are connected to
# points; hence only connected points (i.e., points used by cells) are
# considered.

# Compute bounds on polydata
points.Modified()
timer.StartTimer()
output.GetBounds(box)
timer.StopTimer()
time = timer.GetElapsedTime()
print("svtkPolyData::ComputeBounds():")
print("\tTime: {0}".format(time))
print("\tBounds: {0}".format(box))

assert box[0] == -1.5
assert box[1] ==  1.5
assert box[2] == -0.5
assert box[3] ==  0.5
assert box[4] ==  0.0
assert box[5] ==  0.0

# Uses svtkPoints::ComputeBounds() which uses threaded svtkSMPTools and
# svtkArrayDispatch (see svtkDataArrayPrivate.txx). In other words, cell
# connectivity is not taken into account.
points.Modified()
timer.StartTimer()
points.GetBounds(box)
timer.StopTimer()
time = timer.GetElapsedTime()
print("svtkPoints::ComputeBounds():")
print("\tTime: {0}".format(time))
print("\tBounds: {0}".format(box))

assert box[0] == -1.5
assert box[1] ==  1.5
assert box[2] == -0.5
assert box[3] ==  0.5
assert box[4] ==  0.0
assert box[5] ==  0.0

# Uses svtkBoundingBox with svtkSMPTools. This method takes into account
# an (optional) pointUses array to only consider selected points.
bbox = svtk.svtkBoundingBox()
timer.StartTimer()
bbox.ComputeBounds(points,box)
timer.StopTimer()
time = timer.GetElapsedTime()
print("svtkBoundingBox::ComputeBounds():")
print("\tTime: {0}".format(time))
print("\tBounds: {0}".format(box))

assert box[0] == -1.5
assert box[1] ==  1.5
assert box[2] == -0.5
assert box[3] ==  0.5
assert box[4] ==  0.0
assert box[5] ==  0.0
