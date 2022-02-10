from mpi4py import *
import sys
import sensei
import svtk, svtk.numpy_support as svtknp
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

# get num meshes callback
def getNumMeshes():
  sys.stderr.write('===getNumMeshes\n')
  return 1

def getMeshMetadata(i, flags):
  sys.stderr.write('===getMeshMetadata\n')
  sys.stderr.write('flags=%s'%(flags))
  if i == 0:
    md = sensei.MeshMetadata.New()
    md.MeshName = 'image'
    md.MeshType = svtk.SVTK_IMAGE_DATA
    md.BlockType = svtk.SVTK_IMAGE_DATA
    md.NumArrays = 1
    md.ArrayName = ['data']
    md.ArrayCentering = [svtk.svtkDataObject.POINT]
    md.ArrayComponents = [1]
    md.ArrayType = [svtk.SVTK_DOUBLE]
    return md
  raise RunTimeError('no mesh %d'%(i))

# get mesh callback
def getMesh(meshName, structureOnly):
  sys.stderr.write('===getMesh\n')
  if (meshName == 'image'):
    im = svtk.svtkImageData()
    im.SetDimensions(len(data), 1, 1)
    return im
  raise RuntimeError('failed to get mesh')

# add array callback
def addArray(mesh, meshName, assoc, arrayName):
  sys.stderr.write('===addArray\n')
  if ((meshName == 'image') and (assoc == svtk.svtkDataObject.POINT) \
    and (arrayName == 'data')):
    da = svtknp.numpy_to_svtk(data)
    da.SetName('data')
    mesh.GetPointData().AddArray(da)
    return
  raise RuntimeError('failed to add array')

# release data callback
def releaseData():
  sys.stderr.write('===releaseData\n')

m = getMesh('image', False)
addArray(m, 'image', svtk.svtkDataObject.POINT, 'data')
sys.stderr.write('m = %s\n'%str(m))
sys.exit(0)



pda = sensei.ProgrammableDataAdaptor.New()
pda.SetGetNumberOfMeshesCallback(getNumMeshes)
pda.SetGetMeshMetadataCallback(getMeshMetadata)
pda.SetGetMeshCallback(getMesh)
pda.SetAddArrayCallback(addArray)
pda.SetReleaseDataCallback(releaseData)

mmd = pda.GetMeshMetadata(0)
sys.stderr.write('mmd = %s\n'%str(mmd))

meshName = mmd.MeshName
arrayName = mmd.ArrayName[0]
arrayCen = mmd.ArrayCentering[0]

ha = sensei.Histogram.New()
ha.Initialize(7, meshName, arrayCen, arrayName, '')
ha.Execute(pda)

hmin,hmax,hist = ha.GetHistogram()

result = -1
if hist == baselineHist:
  result = 0

ha.Delete()

pda.ReleaseData()
pda.Delete()

sys.exit(result)
