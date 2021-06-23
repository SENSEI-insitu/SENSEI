"""this is python equivalent of ./Wrapping/Tcl/svtktesting/backdrop.tcl
This script is used while running python tests translated from Tcl."""

import svtk

basePlane = None
baseMapper = None
base = None

backPlane = None
backMapper = None
back = None

leftPlane = None
leftMapper = None
left = None

def BuildBackdrop (minX, maxX, minY, maxY, minZ, maxZ, thickness):
    global basePlane
    global baseMapper
    global base
    global backPlane
    global backMapper
    global back
    global left
    global leftPlane
    global leftMapper

    if not basePlane:
        basePlane = svtk.svtkCubeSource()
    basePlane.SetCenter( (maxX + minX)/2.0, minY, (maxZ + minZ)/2.0)
    basePlane.SetXLength(maxX-minX)
    basePlane.SetYLength(thickness)
    basePlane.SetZLength(maxZ - minZ)

    if not baseMapper:
        baseMapper = svtk.svtkPolyDataMapper()
    baseMapper.SetInputConnection(basePlane.GetOutputPort())

    if not base:
        base = svtk.svtkActor()
    base.SetMapper(baseMapper)

    if not backPlane:
        backPlane = svtk.svtkCubeSource()
    backPlane.SetCenter( (maxX + minX)/2.0, (maxY + minY)/2.0, minZ)
    backPlane.SetXLength(maxX-minX)
    backPlane.SetYLength(maxY - minY)
    backPlane.SetZLength(thickness)

    if not backMapper:
        backMapper = svtk.svtkPolyDataMapper()
    backMapper.SetInputConnection(backPlane.GetOutputPort())

    if not back:
        back = svtk.svtkActor()
    back.SetMapper(backMapper)

    if not leftPlane:
        leftPlane = svtk.svtkCubeSource()
    leftPlane.SetCenter( minX, (maxY+minY)/2.0, (maxZ+minZ)/2.0)
    leftPlane.SetXLength(thickness)
    leftPlane.SetYLength(maxY-minY)
    leftPlane.SetZLength(maxZ-minZ)

    if not leftMapper:
        leftMapper = svtk.svtkPolyDataMapper()
    leftMapper.SetInputConnection(leftPlane.GetOutputPort())

    if not left:
        left = svtk.svtkActor()
    left.SetMapper(leftMapper)

    return [base, back, left]

