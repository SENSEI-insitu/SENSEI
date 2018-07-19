from mpi4py import *
import sys
import sensei
import vtk, vtk.util.numpy_support as vtknp
import numpy as np

#                     *
#                     * *
#                   * * *
#                   * * * *
#                 * * * * *
#               * * * * * * *
baselineHist = [1,2,4,6,5,3,1]
#               0,1,2,3,4,5,6

data = np.array([0, 1,1, 2,2,2,2, \
  3,3,3,3,3,3, 4,4,4,4,4, \
  5,5,5, 6], dtype=np.float64)

# get mesh callback
def getMesh(meshName, structureOnly):
  sys.stderr.write('===getMesh\n')
  if (meshName == 'image'):
    im = vtk.vtkImageData()
    im.SetDimensions(len(data), 1, 1)
    return im
  raise RuntimeError('failed to get mesh')

# add array callback
def addArray(mesh, meshName, assoc, arrayName):
  sys.stderr.write('===addArray\n')
  if ((meshName == 'image') and (assoc == vtk.vtkDataObject.POINT) \
    and (arrayName == 'data')):
    da = vtknp.numpy_to_vtk(data)
    da.SetName('data')
    mesh.GetPointData().AddArray(da)
    return
  raise RuntimeError('failed to add array')

# number of arrays callback
def getNumArrays(meshName, assoc):
  sys.stderr.write('===getNumArrays\n')
  if (meshName == 'image') and (assoc == vtk.vtkDataObject.POINT):
    return 1
  raise RuntimeError('failed to get the number of arrays')

# array name callback
def getArrayName(meshName, assoc, aid):
  sys.stderr.write('===getArrayName\n')
  if ((meshName == 'image') and (assoc == vtk.vtkDataObject.POINT) \
    and (aid == 0)):
    return 'data'
  raise RuntimeError('failed to get array name')

# release data callback
def releaseData():
  sys.stderr.write('===releaseData\n')


pda = sensei.ProgrammableDataAdaptor.New()
pda.SetGetMeshCallback(getMesh)
pda.SetAddArrayCallback(addArray)
pda.SetGetNumberOfArraysCallback(getNumArrays)
pda.SetGetArrayNameCallback(getArrayName)
pda.SetReleaseDataCallback(releaseData)

result = -1
if pda.GetNumberOfArrays("image", vtk.vtkDataObject.POINT) > 0:

  ha = sensei.Histogram.New()

  ha.Initialize(7, 'image', vtk.vtkDataObject.POINT,
    pda.GetArrayName('image', vtk.vtkDataObject.POINT, 0))

  ha.Execute(pda)

  hmin,hmax,hist = ha.GetHistogram()

  if hist == baselineHist:
    result = 0

  ha.Delete()

pda.ReleaseData()
pda.Delete()

sys.exit(result)
